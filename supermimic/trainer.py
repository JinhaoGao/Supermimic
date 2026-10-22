"""SuperMimic training loop (Algorithm A.1 / Appendix D).

Per iteration:
  1. Collect: roll the policy (+ action noise) in the simulator, push to replay.
  2. Fit world model on k-step replay windows (autoregressive).
  3. Improve the policy by BPTT through H imagined steps with a critic
     bootstrap, then fit the critic to GAE targets of the same rollout.

Reference motions live in CPU RAM; the simulator loop gathers the current
frames / goal windows every step and each policy update gathers one
(H+K)-frame window pack per imagined rollout.

Data parallel (torchrun, see distributed.py): every rank runs this loop on its
own envs / replay (seed + rank) and the ranks stay one model (gradient mean,
synced normalizers and reset sampler); rank 0 logs, saves and evaluates.
Without torchrun the code path is exactly the single-GPU one.

NaN guard (every mode): after the gradient all-reduce of each module update the
reduced gradient is checked; if it is non-finite (any rank) every rank drops it
and skips that optimizer step (optimizer state untouched). The watchdog treats
an iteration with a skipped update or a non-finite imagined reward as a
transient and only acts (rollback) after wd_nf_patience consecutive ones;
checkpoints written while non-finite (or with non-finite weights) are never
rollback targets. Simulator states that come back non-finite end their episode
as a failure (the replay keeps the finite pre-step state).
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from collections import deque
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from .distributed import DistCtx
from .env_cfg import make_env_cfg
from .evaluate import Evaluator
from .fns import StepFns
from .heads import HeadsCfg
from .motions import read_name_list, resolve_clip_names
from .policy import Critic, Policy
from .replay import ReplayBuffer
from .sim_interface import ObsHistory, SimInterface
from .world_model import DynamicsModel, WorldModelCfg


@dataclass
class TrainCfg:
  motion_files: list[str] = field(default_factory=list)
  robot: str = "g1"
  num_envs: int = 4096
  device: str = "cuda:0"
  seed: int = 0
  log_dir: str = "logs/supermimic/run"
  cache_dir: str = ""
  num_workers: int = 32
  iterations: int = 320_000
  env_steps_per_iter: float = 1.0
  action_noise_std: float = 0.1
  replay_capacity: int = 1000
  episode_length_s: float = 10.0
  compile: bool = True
  compile_mode: str | None = None
  num_threads: int = 0  # torch intra-op CPU threads (0 = torch default); scripts/train.py also sets OMP_NUM_THREADS
  # Data parallel (torchrun). Every rank: num_envs envs + rollouts_per_update rollouts (global = world_size x).
  world_size: int = 1  # informational (set by scripts/train.py from torchrun)
  base_lr_scale: float = 1.0  # --lr_scale, already folded into policy_lr / critic_lr / wm.lr (record only)
  sampler_sync_interval: int = 0  # iterations between reset-sampler merges across ranks (0 = log_interval)
  param_check_interval: int = 1000  # iterations between cross-rank state hash checks (0 = only at start / end)
  dist_timeout_min: float = 120.0  # process-group timeout; must exceed one evaluation (and a cache build)
  # Imagination / policy.
  H: int = 32
  rollouts_per_update: int = 4096
  gamma: float = 0.99
  lam: float = 0.95
  policy_lr: float = 2e-4
  policy_betas: tuple[float, float] = (0.9, 0.95)
  critic_lr: float = 2e-4
  critic_epochs: int = 5
  critic_minibatches: int = 4
  grad_clip_norm: float = 1.0
  grad_clip_elem: float = 0.1
  lr_decay_iters: int = 250_000
  lr_final_ratio: float = 0.03
  use_critic: bool = True
  critic_huber_delta: float = 1.0  # 2*Huber(delta) == MSE for |err|<delta; linear beyond (outlier targets)
  clip_value_targets: bool = True  # clip GAE targets to [-v_max, v_max], v_max = r_max/(1-gamma)
  # Watchdog: roll back to the last healthy checkpoint when imagination collapses (see _watchdog).
  wd_enabled: bool = True
  wd_ema: float = 0.02  # EMA rate of the imagined reward (normalized)
  wd_drop: float = 0.08  # trigger when EMA < best EMA of the last wd_window iterations - wd_drop
  wd_window: int = 4000
  wd_vloss_ratio: float = 50.0  # or when median critic loss of the last 50 its > ratio * median of the last 2000
  wd_warmup: int = 500  # iterations after start / resume / rollback before the watchdog may trigger
  wd_lr_factor: float = 0.5  # LR multiplier applied at every rollback
  wd_max_rollbacks: int = 6
  wd_force_at: int = -1  # debug: force one rollback at this iteration
  wd_nf_patience: int = 100  # consecutive non-finite iterations (skipped update / NaN imagined reward) before a rollback
  wd_early_ckpt: int = 500  # one early checkpoint (rollback point) at this iteration when none exists yet (0 = off)
  # Logging / eval / checkpoints.
  log_interval: int = 50
  eval_interval: int = 2000
  ckpt_interval: int = 2000
  eval_enabled: bool = True
  eval_clips: int = 0  # 0 = all clips; otherwise a fixed random subset of this size
  eval_clip_names: str = ""  # txt of npz basenames (one per line): fixed eval subset by name; overrides eval_clips
  eval_envs: int = 1024
  wm: WorldModelCfg = field(default_factory=WorldModelCfg)
  heads: HeadsCfg = field(default_factory=HeadsCfg)


def cosine_lambda(decay_iters: int, final_ratio: float):
  def f(it: int) -> float:
    if it >= decay_iters:
      return final_ratio
    return final_ratio + 0.5 * (1.0 - final_ratio) * (1.0 + math.cos(math.pi * it / decay_iters))

  return f


def finite_mean(v) -> float:
  """Mean of the finite entries (= np.mean when all are finite; nan when none is). Logging only."""
  a = np.asarray(v, dtype=np.float64)
  f = np.isfinite(a)
  if f.all():
    return float(np.mean(v))
  return float(a[f].mean()) if f.any() else float("nan")


def clip_grads(params, norm: float, elem: float | None) -> float:
  total = torch.nn.utils.clip_grad_norm_(params, norm).item()
  if elem is not None:
    for p in params:
      if p.grad is not None:
        p.grad.clamp_(-elem, elem)
  return total


class Trainer:
  LIB_KEY = "train"

  def __init__(self, cfg: TrainCfg, dist: DistCtx | None = None):
    self.cfg = cfg
    self.dist = dist if dist is not None else DistCtx(device=cfg.device)
    dctx = self.dist
    self.main = dctx.is_main
    self._print = print if self.main else (lambda *a, **k: None)
    if cfg.num_threads > 0:
      torch.set_num_threads(cfg.num_threads)
    seed = cfg.seed + dctx.rank  # own envs / replay / noise per rank; models are broadcast from rank 0 below
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    self.device = cfg.device
    os.makedirs(cfg.log_dir, exist_ok=True)
    if self.main:
      with open(os.path.join(cfg.log_dir, "train_cfg.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2, default=str)
    self.writer = SummaryWriter(cfg.log_dir) if self.main else None

    from mjlab.envs import ManagerBasedRlEnv

    env_cfg = make_env_cfg(
      cfg.motion_files, cfg.num_envs, mode="train", episode_length_s=cfg.episode_length_s, robot=cfg.robot,
      library_key=self.LIB_KEY, cache_dir=cfg.cache_dir, num_workers=cfg.num_workers,
      ee_termination=cfg.heads.ee_termination, ee_threshold=cfg.heads.ee_term_z,
    )
    env_cfg.seed = seed
    # Rank 0 builds the env first (and the packed motion cache if it is missing); the others wait so that
    # they never build the same cache concurrently, then all map the same cache files (one page-cache copy).
    if dctx.enabled and not self.main:
      dctx.barrier()
    self.env = ManagerBasedRlEnv(cfg=env_cfg, device=self.device)
    if dctx.enabled and self.main:
      dctx.barrier()
    self.sim = SimInterface(self.env, cfg.heads)
    self.heads = self.sim.heads
    self.lib = self.sim.lib
    self.cmd = self.sim.cmd
    self.nj = self.sim.nj
    self.xi_dim = self.sim.xi_dim
    self.E = cfg.num_envs
    hc = cfg.heads
    self.dt = self.env.step_dt
    assert abs(self.dt - hc.dt) < 1e-9, f"env dt {self.dt} != heads dt {hc.dt}"

    self.policy = Policy(self.heads.proprio_dim, hc.hist, self.heads.goal_frame_dim, hc.K, self.nj).to(self.device)
    self.critic = Critic(self.heads.feat_dim, self.heads.goal_frame_dim, hc.K, self.xi_dim).to(self.device)
    self.wm = DynamicsModel(self.nj, self.xi_dim, self.dt, cfg.wm).to(self.device)
    dctx.broadcast_modules_(self.policy, self.critic, self.wm)  # one model: rank 0's initialization everywhere
    self.fns = StepFns(self.heads, self.policy, self.wm, self.critic, compile_=cfg.compile, mode=cfg.compile_mode)
    self.opt_wm = torch.optim.Adam(self.wm.parameters(), lr=cfg.wm.lr)
    self.opt_pi = torch.optim.Adam(self.policy.parameters(), lr=cfg.policy_lr, betas=cfg.policy_betas)
    self.opt_v = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
    lam = cosine_lambda(cfg.lr_decay_iters, cfg.lr_final_ratio)
    self.sched = [torch.optim.lr_scheduler.LambdaLR(o, lam) for o in (self.opt_wm, self.opt_pi, self.opt_v)]
    self._base_lrs0 = [list(sch.base_lrs) for sch in self.sched]
    self.lr_scale = 1.0
    self.buffer = ReplayBuffer(self.E, cfg.replay_capacity, self.heads.state_dim, self.nj, self.xi_dim, self.device)
    self.evaluator: Evaluator | None = None
    self.it = 0
    self.env_steps = 0
    self._step_carry = 0.0
    self._t_start = time.time()
    if cfg.eval_clip_names:
      self.eval_clip_ids = self._eval_ids_from_names(cfg.eval_clip_names)
    elif cfg.eval_clips and cfg.eval_clips < self.lib.num_clips:
      rng = random.Random(cfg.seed)
      self.eval_clip_ids = tuple(sorted(rng.sample(range(self.lib.num_clips), cfg.eval_clips)))
    else:
      self.eval_clip_ids = tuple(range(self.lib.num_clips))

    n_params = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    self._print(f"[Trainer] policy {n_params(self.policy)/1e6:.1f}M  critic {n_params(self.critic)/1e6:.2f}M  wm {n_params(self.wm)/1e6:.1f}M")
    self._print(f"[Trainer] {self.lib.summary()}")
    self._print(f"[Trainer] goal_frame_dim={self.heads.goal_frame_dim} (anchor_pos={hc.goal_anchor_pos}), eval clips {len(self.eval_clip_ids)}")
    self._print(
      "[Trainer] end-effector termination: "
      +(f"on, z error > {hc.ee_term_z} on {list(self.sim.ee_body_names)} (soft temp {hc.ee_term_z_temp})" if hc.ee_termination else "off (paper recipe)")
    )
    if dctx.enabled:
      self._print(
        f"[Trainer] data parallel: {dctx.world} ranks x {self.E} envs / {cfg.rollouts_per_update} rollouts "
        f"(global {dctx.world * self.E} / {dctx.world * cfg.rollouts_per_update}), seeds {cfg.seed}..{cfg.seed + dctx.world - 1}"
      )

    # ---- simulator bookkeeping ----
    self.env.reset()
    E = self.E
    self.state = self.sim.state()
    self.xi = self.sim.xi()
    self.frame = self.cmd.time_steps.clone()
    self.clip_end = self.cmd.clip_end_env.clone()
    self.prev_action = torch.zeros(E, self.nj, device=self.device)
    self.ep_step = torch.zeros(E, dtype=torch.long, device=self.device)
    self.hist = ObsHistory(E, hc.hist, self.heads.proprio_dim, self.device)
    with torch.no_grad():
      self.kin, obs = self.fns.refresh(self.state, self.prev_action, self.sim.bias_from_xi(self.xi, self.nj), True)
    self.hist.reset(torch.arange(E, device=self.device), obs)
    L = self.env.max_episode_length + 2
    self.ep_bins = torch.zeros(E, L, dtype=torch.long, device=self.device)
    self.ep_rews = torch.zeros(E, L, device=self.device)
    self.ep_ret = torch.zeros(E, device=self.device)
    self._gamma_pow = cfg.gamma ** torch.arange(L, device=self.device, dtype=torch.float32)
    self._ar = torch.arange(E, device=self.device)
    self.stats = {"ep_return_norm": [], "ep_len": [], "ep_failed": [], "ep_failed_ee": [], "term_mismatch": [], "sim_nonfinite": []}
    self._ee_on = hc.ee_termination
    # ---- NaN guard ----
    self.guard_skips = {"wm": 0, "pi": 0, "critic": 0}  # skipped (non-finite) updates since start, same on every rank
    self.guard_nf_iters = 0  # iterations with a skipped update or a non-finite imagined reward
    # ---- watchdog ----
    self.wd_reward_ema: float | None = None
    self.wd_hist: deque = deque()
    self.wd_vloss: deque = deque(maxlen=2000)
    self.wd_healthy_ckpt: str | None = None
    self.wd_rollbacks = 0
    self.wd_quiet_until = cfg.wd_warmup
    self.wd_forced = False
    self.wd_nf_streak = 0  # consecutive non-finite iterations

  # ---------------------------------------------------------------------------
  # 1. Data collection
  # ---------------------------------------------------------------------------

  @torch.no_grad()
  def collect_step(self) -> None:
    cfg = self.cfg
    ref_q, ref_qd, ref_ap, ref_aq = self.sim.goal_refs(self.frame, self.clip_end)
    ref_cur = self.sim.current_ref()  # reference at frame f_t: reward target of this step
    goal, a = self.fns.act(self.hist.buf, self.kin, ref_q, ref_qd, ref_ap, ref_aq)
    self.dist.rms_update([(self.policy.obs_norm, self.hist.buf[:, -1]), (self.policy.goal_norm, goal)])
    a_exec = a + cfg.action_noise_std * torch.randn_like(a)

    self.env.step(a_exec)
    next_state = self.sim.state()
    r, term_hard = self.fns.post(next_state, self.kin, a_exec, self.prev_action, *ref_cur)
    terminated = self.env.reset_terminated.bool()
    timeouts = self.env.reset_time_outs.bool()
    # A non-finite simulator state never trips the (NaN-blind) termination tests: end it as a failure and keep the
    # finite pre-step state in the replay instead (no-op, bitwise, when everything is finite).
    bad = ~torch.isfinite(next_state).all(-1)
    tm_frac, n_bad = torch.stack([(term_hard != terminated).float().mean(), bad.float().sum()]).tolist()  # one sync
    self.stats["term_mismatch"].append(tm_frac)
    self.stats["sim_nonfinite"].append(n_bad)
    terminated = terminated | bad
    r = torch.where(bad, torch.zeros_like(r), r)
    next_state = torch.where(bad[:, None], self.state, next_state)
    done = terminated | timeouts
    if self._ee_on:  # end-effector-only failures (the eval site's "body_pos" reason): logging only
      tm = self.env.termination_manager
      ee_fail = tm.get_term("ee_body_pos") & ~(tm.get_term("anchor_pos") | tm.get_term("anchor_ori")) & ~bad

    self.buffer.add(
      state=self.state, next_state=next_state, action=a_exec, prev_action=self.prev_action, xi=self.xi, reward=r,
      frame=self.frame, clip_end=self.clip_end, ep_step=self.ep_step, term=terminated, done=done,
    )
    self.ep_bins[self._ar, self.ep_step] = self.lib.bin_of_frame[self.frame].long()
    self.ep_rews[self._ar, self.ep_step] = r
    self.ep_ret += r
    self.ep_step += 1
    self.env_steps += 1

    if done.any():
      ids = done.nonzero(as_tuple=False).squeeze(-1)
      self._finish_episodes(ids, terminated[ids], ee_fail[ids] if self._ee_on else None)
      self.env.reset(env_ids=ids)
      xi_new = self.sim.xi()
      self.xi[ids] = xi_new[ids]
      self.state = self.sim.state()
      self.prev_action[ids] = 0.0
      self.ep_step[ids] = 0
      self.ep_ret[ids] = 0.0
    else:
      self.state = next_state
    self.prev_action = torch.where(done[:, None], torch.zeros_like(a_exec), a_exec)
    self.frame = self.cmd.time_steps.clone()
    self.clip_end = self.cmd.clip_end_env.clone()
    self.kin, obs = self.fns.refresh(self.state, self.prev_action, self.sim.bias_from_xi(self.xi, self.nj), True)
    self.hist.push(obs)
    if done.any():
      self.hist.reset(ids, obs)

  @torch.no_grad()
  def _finish_episodes(self, ids: torch.Tensor, failed: torch.Tensor, failed_ee: torch.Tensor | None = None) -> None:
    g = self.cfg.gamma
    L = self.ep_step[ids]
    Lmax = int(L.max().item())
    rews = self.ep_rews[ids, :Lmax]
    bins = self.ep_bins[ids, :Lmax]
    t = torch.arange(Lmax, device=self.device)
    valid = t[None, :] < L[:, None]
    r_last = rews.gather(1, (L - 1)[:, None]).squeeze(1)
    boot = torch.where(failed, torch.zeros_like(r_last), r_last / (1.0 - g))
    w = rews * valid * self._gamma_pow[:Lmax][None, :]
    S = torch.flip(torch.cumsum(torch.flip(w, [1]), 1), [1])
    R = S / self._gamma_pow[:Lmax][None, :] + boot[:, None] * (g ** (L[:, None] - t[None, :]).float())
    self.cmd.update_bin_returns(bins[valid], R[valid])
    rmax = self.cfg.heads.reward_max_per_step
    norm_ret = self.ep_ret[ids] / (L.float() * rmax)
    self.stats["ep_return_norm"].extend(norm_ret.tolist())
    self.stats["ep_len"].extend(L.tolist())
    self.stats["ep_failed"].extend(failed.float().tolist())
    if failed_ee is not None:
      self.stats["ep_failed_ee"].extend((failed_ee & failed).float().tolist())

  # ---------------------------------------------------------------------------
  # 2. World model
  # ---------------------------------------------------------------------------

  def update_world_model(self) -> dict:
    cfg = self.cfg.wm
    self.wm.train()
    agg: dict[str, float] = {}
    for _ in range(cfg.batches_per_update):
      batch = self.buffer.sample_windows(cfg.windows_per_batch, cfg.k)
      self.dist.rms_update(self.wm.stats_pairs(batch))
      loss, e1, ek = self.fns.wm_loss(batch["s0"], batch["actions"], batch["next_states"], batch["xi"], batch["mask"])
      self.opt_wm.zero_grad(set_to_none=True)
      loss.backward()
      gn = self._apply_grads("wm", self.opt_wm, list(self.wm.parameters()), cfg.grad_clip, None)
      m = {"wm/loss": loss.item(), "wm/err_step1": e1.item(), f"wm/err_step{cfg.k}": ek.item(), "wm/grad_norm": gn, "wm/valid_frac": batch["mask"].float().mean().item()}
      for k, v in m.items():
        agg[k] = agg.get(k, 0.0) + v / cfg.batches_per_update
    return agg

  # ---------------------------------------------------------------------------
  # 3. Policy + critic through imagination
  # ---------------------------------------------------------------------------

  def update_policy(self) -> dict:
    cfg, heads, H, K = self.cfg, self.heads, self.cfg.H, self.cfg.heads.K
    N = cfg.rollouts_per_update
    g = cfg.gamma
    for p in self.wm.parameters():
      p.requires_grad_(False)
    self.wm.eval()
    st = self.buffer.sample_starts(N, cfg.heads.hist)
    xi = st["xi"]
    bias = self.sim.bias_from_xi(xi, self.nj)
    t_g0 = time.time()
    with torch.no_grad():
      hs, hp = st["hist_states"], st["hist_prev_action"]
      hist_n = hs.shape[1]
      obs_hist = heads.proprio(
        hs.reshape(-1, hs.shape[-1]), hp.reshape(-1, self.nj), bias[:, None, :].expand(N, hist_n, self.nj).reshape(-1, self.nj), True
      ).view(N, hist_n, -1)
      kin = heads.kin(st["state"])
      # Reference window pack for this batch: frames f0 .. f0+H+K-1 of every rollout.
      Lw = H + K
      pack = self.lib.gather_windows(st["frame"], Lw)
      f_loc = torch.zeros(N, dtype=torch.long, device=self.device)
      clip_end_loc = torch.clamp(st["clip_end"] - st["frame"], max=Lw)
    self._t_gather = time.time() - t_g0
    s = st["state"]
    prev_a = st["prev_action"]
    c = torch.ones(N, device=self.device)
    J = torch.zeros(N, device=self.device)
    feats_n, goals_n, rews, dsoft = [], [], [], []
    p = (pack["joint_pos"], pack["joint_vel"], pack["body_pos"], pack["body_quat"], pack["body_lin_vel"], pack["body_ang_vel"])
    for t in range(H):
      s, obs_hist, prev_a, f_loc, kin, r, d, fn, gn = self.fns.imagine_step(s, obs_hist, prev_a, f_loc, clip_end_loc, xi, bias, kin, *p)
      J = J + (g**t) * c * r
      c = c * (1.0 - d)
      feats_n.append(fn.detach())
      goals_n.append(gn.detach())
      rews.append(r.detach())
      dsoft.append(d.detach())
    goal_H = self.fns.goal_from_pack(kin, f_loc, clip_end_loc, p[0], p[1], p[2], p[3])
    fn_H, gn_H = self.fns.critic_inputs(s, goal_H)
    if cfg.use_critic:
      V_H = self.critic(fn_H, gn_H, xi)
      J = J + (g**H) * c * V_H
    loss_pi = -J.mean() / H
    self.opt_pi.zero_grad(set_to_none=True)
    loss_pi.backward()
    gn_pi = self._apply_grads("pi", self.opt_pi, list(self.policy.parameters()), cfg.grad_clip_norm, cfg.grad_clip_elem)
    for p_ in self.wm.parameters():
      p_.requires_grad_(True)

    metrics = {
      "pi/imagined_return": (J.mean() / H).item(),
      "pi/imagined_reward_norm": torch.stack(rews).mean().item() / self.cfg.heads.reward_max_per_step,
      "pi/survival_H": c.mean().item(),
      "pi/grad_norm": gn_pi,
      "pi/loss": loss_pi.item(),
      "time/ref_gather": self._t_gather,
    }
    if cfg.use_critic:
      metrics.update(self._fit_critic(feats_n, goals_n, fn_H.detach(), gn_H.detach(), xi, rews, dsoft))
    return metrics

  def _fit_critic(self, feats_n, goals_n, fn_H, gn_H, xi, rews, dsoft) -> dict:
    cfg, H = self.cfg, self.cfg.H
    g, lam = cfg.gamma, cfg.lam
    N = xi.shape[0]
    with torch.no_grad():
      V = [self.critic(feats_n[t], goals_n[t], xi) for t in range(H)] + [self.critic(fn_H, gn_H, xi)]
      R = [None] * (H + 1)
      R[H] = V[H]
      for t in reversed(range(H)):
        R[t] = rews[t] + g * (1.0 - dsoft[t]) * ((1.0 - lam) * V[t + 1] + lam * R[t + 1])
      targets = torch.stack(R[:H], dim=1)
      if cfg.clip_value_targets:
        v_max = self.cfg.heads.reward_max_per_step / (1.0 - g)
        targets = targets.clamp(-v_max, v_max)
    F = torch.stack(feats_n, dim=1).reshape(N * H, -1)
    G = torch.stack(goals_n, dim=1).reshape(N * H, -1)
    X = xi[:, None, :].expand(N, H, -1).reshape(N * H, -1)
    T = targets.reshape(N * H)
    total = N * H
    mb = total // cfg.critic_minibatches
    losses = []
    for _ in range(cfg.critic_epochs):
      perm = torch.randperm(total, device=self.device)
      for i in range(cfg.critic_minibatches):
        idx = perm[i * mb : (i + 1) * mb]
        v = self.critic(F[idx], G[idx], X[idx])
        loss = 2.0 * torch.nn.functional.huber_loss(v, T[idx], delta=cfg.critic_huber_delta)
        self.opt_v.zero_grad(set_to_none=True)
        loss.backward()
        self._apply_grads("critic", self.opt_v, list(self.critic.parameters()), cfg.grad_clip_norm, None)
        losses.append(loss.item())
    return {"critic/loss": float(np.mean(losses)), "critic/target_mean": T.mean().item(), "critic/value_H_mean": V[H].mean().item()}

  def _apply_grads(self, name: str, opt: torch.optim.Optimizer, params: list, norm: float, elem: float | None) -> float:
    """Gradient all-reduce (data parallel) -> finite check -> clip -> optimizer step; returns the grad norm.

    A non-finite reduced gradient (NaN / inf on any rank, or an overflowing sum) is the same on every rank, so
    every rank skips this update identically: gradients dropped, no optimizer step (Adam moments / step count
    untouched), counted in guard_skips[name]; returns nan."""
    if self.dist.allreduce_grads_(params):
      gn = clip_grads(params, norm, elem)
      opt.step()
      return gn
    opt.zero_grad(set_to_none=True)
    self.guard_skips[name] += 1
    return float("nan")

  def _learned_state_finite(self) -> bool:
    ts = [t for m in (self.policy, self.critic, self.wm) for t in list(m.parameters()) + list(m.buffers()) if t.is_floating_point()]
    return bool(torch.stack([torch.isfinite(t).all() for t in ts]).all())

  # ---------------------------------------------------------------------------
  # Checkpoints / evaluation
  # ---------------------------------------------------------------------------

  def _ckpt_path(self, tag: str) -> str:
    return os.path.join(self.cfg.log_dir, f"model_{tag}.pt")

  def save(self, tag: str) -> str:
    assert self.main, "only rank 0 writes checkpoints"
    path = self._ckpt_path(tag)
    torch.save(
      {
        "it": self.it,
        "env_steps": self.env_steps,
        "policy": self.policy.state_dict(),
        "critic": self.critic.state_dict(),
        "wm": self.wm.state_dict(),
        "opt_wm": self.opt_wm.state_dict(),
        "opt_pi": self.opt_pi.state_dict(),
        "opt_v": self.opt_v.state_dict(),
        "bin_G": self.cmd.bin_G,
        "bin_visits": self.cmd.bin_visits,
        "lr_scale": self.lr_scale,
        "cfg": asdict(self.cfg),
        "dims": dict(proprio=self.heads.proprio_dim, hist=self.cfg.heads.hist, goal_frame=self.heads.goal_frame_dim, K=self.cfg.heads.K, nj=self.nj, xi=self.xi_dim),
      },
      path,
    )
    return path

  def load(self, path: str, rollback: bool = False) -> None:
    ck = torch.load(path, map_location=self.device, weights_only=False)
    self.policy.load_state_dict(ck["policy"])
    self.critic.load_state_dict(ck["critic"])
    self.wm.load_state_dict(ck["wm"])
    self.opt_wm.load_state_dict(ck["opt_wm"])
    self.opt_pi.load_state_dict(ck["opt_pi"])
    self.opt_v.load_state_dict(ck["opt_v"])
    if ck["bin_G"].shape == self.cmd.bin_G.shape:
      self.cmd.bin_G.copy_(ck["bin_G"])
      self.cmd.bin_visits.copy_(ck["bin_visits"])
    self.it = ck["it"]
    self.env_steps = ck["env_steps"]
    for s in self.sched:
      s.last_epoch = self.it
    if not rollback and float(ck.get("lr_scale", 1.0)) != 1.0:
      self._set_lr_scale(float(ck["lr_scale"]))
      self._print(f"[Trainer] lr_scale {self.lr_scale} restored from checkpoint")
    self.wd_healthy_ckpt = path
    self.wd_quiet_until = self.it + self.cfg.wd_warmup
    self._print(f"[Trainer] resumed from {path} at iteration {self.it}")

  # ---------------------------------------------------------------------------
  # Watchdog: G1 full-atlas run 2026-10-22 collapsed at ~41k iterations (imagined reward 0.91 -> 0.4 in
  # ~400 iterations, critic loss x5000, then NaN in the reset sampler). Roll back instead of dying.
  # ---------------------------------------------------------------------------

  def _watchdog(self, m_pi: dict, skipped: int = 0) -> bool:
    """Update watchdog statistics; roll back if training collapsed. Returns True after a rollback.

    skipped: module updates the NaN guard skipped this iteration (identical on every rank)."""
    cfg = self.cfg
    r = float(m_pi.get("pi/imagined_reward_norm", float("nan")))
    vl = float(m_pi.get("critic/loss", float("nan")))
    if self.dist.enabled:  # data parallel: every rank decides on the same rank means -> rolls back together
      g = self.dist.reduce_scalars({"r": r, "vl": vl}, "mean", check_keys=False)
      r, vl = g["r"], g["vl"]
    if skipped or not math.isfinite(r):
      # Transient: the non-finite updates were skipped, the weights are intact. Only a long streak means the
      # weights / normalizers themselves are broken -> roll back (no checkpoint yet: keep skipping, never give up).
      self.guard_nf_iters += 1
      self.wd_nf_streak += 1
      if self.wd_nf_streak <= 3 or self.wd_nf_streak % 100 == 0:
        sk = ", ".join(f"{k} {v}" for k, v in self.guard_skips.items())
        self._print(f"[guard] it {self.it}: non-finite (imagined reward {r:.4g}, {skipped} updates skipped; streak {self.wd_nf_streak}, total skips {sk})", flush=True)
      if not cfg.wd_enabled or self.wd_nf_streak < cfg.wd_nf_patience:
        return False
      self.wd_nf_streak = 0
      return self._rollback(f"non-finite for {cfg.wd_nf_patience} consecutive iterations")
    self.wd_nf_streak = 0
    if not cfg.wd_enabled:
      return False
    if math.isfinite(vl):
      self.wd_vloss.append(vl)
    a = cfg.wd_ema
    self.wd_reward_ema = r if self.wd_reward_ema is None else (1 - a) * self.wd_reward_ema + a * r
    self.wd_hist.append((self.it, self.wd_reward_ema))
    while self.wd_hist and self.wd_hist[0][0] < self.it - cfg.wd_window:
      self.wd_hist.popleft()
    if cfg.wd_force_at >= 0 and self.it >= cfg.wd_force_at and not self.wd_forced:
      self.wd_forced = True
      return self._rollback(f"forced at {self.it} (debug)")
    if self.it < self.wd_quiet_until:
      return False
    best = max(e for _, e in self.wd_hist)
    if self.wd_reward_ema < best - cfg.wd_drop:
      return self._rollback(f"imagined reward EMA {self.wd_reward_ema:.3f} < best {best:.3f} - {cfg.wd_drop}")
    if len(self.wd_vloss) >= 500:
      recent = float(np.median(list(self.wd_vloss)[-50:]))
      base = float(np.median(self.wd_vloss))
      if recent > cfg.wd_vloss_ratio * max(base, 1e-6):
        return self._rollback(f"critic loss median {recent:.3g} > {cfg.wd_vloss_ratio} x {base:.3g}")
    return False

  def _healthy(self) -> bool:
    """May the checkpoint written at this iteration become the rollback target? Never while non-finite: the EMA is
    frozen during a non-finite streak (smoke 10-22: a checkpoint with NaN weights was taken as healthy and
    restored six times in a row)."""
    if self.wd_reward_ema is None or not self.wd_hist or self.wd_nf_streak > 0:
      return False
    if not self._learned_state_finite():
      return False
    best = max(e for _, e in self.wd_hist)
    return self.wd_reward_ema >= best - 0.5 * self.cfg.wd_drop

  def _set_lr_scale(self, scale: float) -> None:
    self.lr_scale = scale
    for sch, b0 in zip(self.sched, self._base_lrs0):
      sch.base_lrs = [b * scale for b in b0]
      for grp, b in zip(sch.optimizer.param_groups, sch.base_lrs):
        grp["lr"] = b * sch.lr_lambdas[0](self.it)

  def _rollback(self, reason: str) -> bool:
    cfg = self.cfg
    path = self.wd_healthy_ckpt
    self._print(f"[watchdog] it {self.it}: {reason}", flush=True)
    if path is None or not os.path.exists(path):
      self._print("[watchdog] no healthy checkpoint yet; continuing", flush=True)
      self.wd_quiet_until = self.it + cfg.wd_warmup
      return False
    if self.wd_rollbacks >= cfg.wd_max_rollbacks:
      raise RuntimeError(f"[watchdog] {self.wd_rollbacks} rollbacks already; giving up ({reason})")
    it_before = self.it
    self.load(path, rollback=True)  # models, normalizers, optimizers, sampler bins, iteration counter
    self._set_lr_scale(self.lr_scale * cfg.wd_lr_factor)
    self.wd_rollbacks += 1
    self.wd_reward_ema = None
    self.wd_hist.clear()
    self.wd_vloss.clear()
    self.wd_nf_streak = 0
    self.wd_quiet_until = self.it + cfg.wd_warmup
    msg = (f"[watchdog] rollback #{self.wd_rollbacks}: it {it_before} -> {self.it} from {path}; "
           f"LR x{cfg.wd_lr_factor} (policy lr now {self.opt_pi.param_groups[0]['lr']:.2e})")
    if self.dist.enabled:
      self._dist_start(f"rollback to it {self.it}")  # every rank loaded the same file; re-sync sampler, check state
    if self.main:
      print(msg, flush=True)
      with open(os.path.join(cfg.log_dir, "watchdog.log"), "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {reason} | {msg}\n")
      self.writer.add_scalar("guard/rollbacks", self.wd_rollbacks, it_before)
    return True

  def evaluate(self) -> dict:
    if self.evaluator is None:
      self.evaluator = Evaluator(
        self.cfg.motion_files, self.cfg.heads, self.device, library_key=self.LIB_KEY, eval_clips=self.eval_clip_ids,
        num_clips_total=self.lib.num_clips, max_envs=self.cfg.eval_envs, compile_=self.cfg.compile, cache_dir=self.cfg.cache_dir,
        robot=self.cfg.robot,
      )
    return self.evaluator.run(self.policy)

  def _eval_ids_from_names(self, path: str) -> tuple[int, ...]:
    """Fixed eval subset by npz basename (independent of the library's clip order / size)."""
    wanted = read_name_list(path)
    ids, missing = resolve_clip_names([c.name for c in self.lib.clips], wanted, label="eval_clip_names", log=self._print)
    if not ids:
      raise ValueError(f"--eval_clip_names {path}: none of the {len(wanted)} names is in the library")
    if not self.main:
      return tuple(ids)
    with open(os.path.join(self.cfg.log_dir, "eval_clip_names.txt"), "w", encoding="utf-8") as f:
      f.write("\n".join(self.lib.clips[i].name for i in ids) + "\n")
    with open(os.path.join(self.cfg.log_dir, "eval_clip_names_missing.txt"), "w", encoding="utf-8") as f:
      f.write("".join(n + "\n" for n in missing))
    self._print(f"[Trainer] eval subset from {path}: {len(ids)} found, {len(missing)} missing (requested {len(wanted)} lines)")
    return tuple(ids)

  # ---------------------------------------------------------------------------
  # Data parallel bookkeeping (no-ops without torchrun)
  # ---------------------------------------------------------------------------

  def _dist_start(self, tag: str = "start") -> None:
    """Start of training / after resume or a watchdog rollback: rank 0's sampler state everywhere
    (bin_weight_hist is not in the checkpoint), fresh merge snapshot, full state check."""
    c = self.cmd
    self.dist.broadcast_tensors_([c.bin_G, c.bin_visits, c.bin_weight_hist])
    self._bins0 = (c.bin_G.clone(), c.bin_visits.clone())
    self._check_consistency(tag)

  def _merge_sampler(self) -> None:
    c = self.cmd
    G0, V0 = self._bins0
    G, V, W = self.dist.merge_sampler_bins(c.bin_G, c.bin_visits, c.bin_weight_hist, G0, V0)
    c.bin_G.copy_(G)
    c.bin_visits.copy_(V)
    c.bin_weight_hist = W
    self._bins0 = (G, V)

  def _state_groups(self) -> dict[str, list]:
    def named(sd: dict) -> list:
      out: list = []
      for k, v in sd.items():
        out += [k, v]
      return out

    def opt(o: torch.optim.Optimizer) -> list:
      sd = o.state_dict()
      out: list = []
      for pid in sorted(sd["state"]):
        for k in sorted(sd["state"][pid]):
          out += [f"{pid}.{k}", sd["state"][pid][k]]
      out.append(repr(sd["param_groups"]))
      return out

    c = self.cmd
    return {
      "policy": named(self.policy.state_dict()),  # incl. obs/goal normalizers
      "critic": named(self.critic.state_dict()),
      "wm": named(self.wm.state_dict()),  # incl. in/tgt/err normalizers
      "opt_wm": opt(self.opt_wm),
      "opt_pi": opt(self.opt_pi),
      "opt_v": opt(self.opt_v),
      "sampler": [c.bin_G, c.bin_visits, c.bin_weight_hist],  # only checked right after a merge
      "counters": [self.it, self.env_steps],
    }

  def _check_consistency(self, tag: str) -> None:
    t0 = time.time()
    d = self.dist.check_consistent(self._state_groups(), tag=tag)
    self._print(f"[dist] {tag}: learned state identical on all {self.dist.world} ranks (policy {d['policy'][:12]}, {time.time() - t0:.1f}s)", flush=True)

  def _dist_log_scalars(self, agg: dict, dt: float, dsteps: int) -> dict:
    """Log scalars of all ranks: means of the per-rank metrics, episode stats pooled over ranks."""
    cfg, dctx, st = self.cfg, self.dist, self.stats
    local = {k: finite_mean(v) for k, v in agg.items()}
    local["sampler/entropy"] = self.cmd.metrics["sampling_entropy"][0].item()
    local["sampler/top1_prob"] = self.cmd.metrics["sampling_top1_prob"][0].item()
    scal = dctx.reduce_scalars(local, "mean")
    sums = dctx.reduce_scalars(
      {
        "ret": float(np.sum(st["ep_return_norm"])), "len": float(np.sum(st["ep_len"])), "fail": float(np.sum(st["ep_failed"])),
        "n": float(len(st["ep_return_norm"])), "tm": float(np.sum(st["term_mismatch"])), "tm_n": float(len(st["term_mismatch"])),
        "steps": float(dsteps * self.E), "fail_ee": float(np.sum(st["ep_failed_ee"])), "nonfin": float(np.sum(st["sim_nonfinite"])),
      },
      "sum",
    )
    mx = dctx.reduce_scalars({k: local.get(k, 0.0) for k in ("time/iter", "time/collect", "time/wm", "time/policy")}, "max")
    scal["time/env_fps"] = sums["steps"] / max(dt, 1e-6)  # all ranks together
    scal["time/iter_per_s"] = cfg.log_interval / max(dt, 1e-6)
    scal["train/env_steps_per_env"] = self.env_steps
    scal["train/env_steps_global"] = float(self.env_steps) * self.E * dctx.world
    scal["train/lr"] = self.opt_pi.param_groups[0]["lr"]
    scal["train/wall_h"] = (time.time() - self._t_start) / 3600
    if sums["n"] > 0:
      scal["train/ep_return_norm"] = sums["ret"] / sums["n"]
      scal["train/ep_len"] = sums["len"] / sums["n"]
      scal["train/ep_failed_frac"] = sums["fail"] / sums["n"]
      if self._ee_on:
        scal["train/ep_failed_ee_frac"] = sums["fail_ee"] / sums["n"]
    scal["guard/sim_nonfinite"] = sums["nonfin"]
    scal["train/term_mismatch"] = sums["tm"] / sums["tm_n"] if sums["tm_n"] else 0.0
    scal["sampler/bins_visited_frac"] = (self.cmd.bin_visits > 0).float().mean().item()  # merged state
    for k, v in mx.items():
      scal[f"dist/{k.split('/')[1]}_max"] = v  # slowest rank (stragglers)
    scal["dist/world_size"] = dctx.world
    return scal

  # ---------------------------------------------------------------------------
  # Loop
  # ---------------------------------------------------------------------------

  def train(self) -> None:
    cfg = self.cfg
    dctx = self.dist
    warmup = max(cfg.wm.k + cfg.heads.hist + 2, 32)
    t_log = time.time()
    steps_at_log = self.env_steps
    agg: dict[str, list[float]] = {}
    sync_every = cfg.sampler_sync_interval or cfg.log_interval
    if dctx.enabled:
      self._dist_start()
    while self.it < cfg.iterations:
      t0 = time.time()
      self._step_carry += cfg.env_steps_per_iter
      n_steps = int(self._step_carry)
      self._step_carry -= n_steps
      for _ in range(max(n_steps, 0)):
        self.collect_step()
      if self.buffer.size < warmup:
        for _ in range(warmup - self.buffer.size):
          self.collect_step()
        continue
      t1 = time.time()
      skips0 = sum(self.guard_skips.values())
      m_wm = self.update_world_model()
      t2 = time.time()
      m_pi = self.update_policy()
      t3 = time.time()
      for s in self.sched:
        s.step()
      self.it += 1
      if self._watchdog(m_pi, sum(self.guard_skips.values()) - skips0):
        agg = {}
        continue
      for k, v in {**m_wm, **m_pi, "time/collect": t1 - t0, "time/wm": t2 - t1, "time/policy": t3 - t2, "time/iter": t3 - t0}.items():
        agg.setdefault(k, []).append(v)

      # Same schedule on every rank (depends only on cfg and the iteration counter).
      ckpt_slot = bool(cfg.ckpt_interval) and self.it % cfg.ckpt_interval == 0
      # One early rollback point (the first regular checkpoint comes only at ckpt_interval).
      ckpt_slot |= cfg.wd_enabled and cfg.wd_early_ckpt > 0 and self.it == cfg.wd_early_ckpt and self.wd_healthy_ckpt is None
      eval_slot = bool(cfg.eval_interval) and self.it % cfg.eval_interval == 0
      check_slot = bool(cfg.param_check_interval) and self.it % cfg.param_check_interval == 0
      if dctx.enabled and (self.it % sync_every == 0 or ckpt_slot or check_slot):
        self._merge_sampler()

      if self.it % cfg.log_interval == 0:
        now = time.time()
        if dctx.enabled:
          scal = self._dist_log_scalars(agg, now - t_log, self.env_steps - steps_at_log)
          fps = scal["time/env_fps"]
        else:
          fps = (self.env_steps - steps_at_log) * self.E / max(now - t_log, 1e-6)
          scal = {k: finite_mean(v) for k, v in agg.items()}
          scal["time/env_fps"] = fps
          scal["time/iter_per_s"] = cfg.log_interval / max(now - t_log, 1e-6)
          scal["train/env_steps_per_env"] = self.env_steps
          scal["train/lr"] = self.opt_pi.param_groups[0]["lr"]
          scal["train/wall_h"] = (now - self._t_start) / 3600
          if self.stats["ep_return_norm"]:
            scal["train/ep_return_norm"] = float(np.mean(self.stats["ep_return_norm"]))
            scal["train/ep_len"] = float(np.mean(self.stats["ep_len"]))
            scal["train/ep_failed_frac"] = float(np.mean(self.stats["ep_failed"]))
          if self.stats["ep_failed_ee"]:
            scal["train/ep_failed_ee_frac"] = float(np.mean(self.stats["ep_failed_ee"]))
          scal["guard/sim_nonfinite"] = float(np.sum(self.stats["sim_nonfinite"]))
          scal["train/term_mismatch"] = float(np.mean(self.stats["term_mismatch"])) if self.stats["term_mismatch"] else 0.0
          scal["sampler/entropy"] = self.cmd.metrics["sampling_entropy"][0].item()
          scal["sampler/top1_prob"] = self.cmd.metrics["sampling_top1_prob"][0].item()
          scal["sampler/bins_visited_frac"] = (self.cmd.bin_visits > 0).float().mean().item()
        if self.wd_reward_ema is not None:
          scal["guard/reward_ema"] = self.wd_reward_ema  # same on every rank (watchdog runs on rank means)
        scal["guard/rollbacks_total"] = self.wd_rollbacks
        for k, v in self.guard_skips.items():
          scal[f"guard/skips_{k}"] = v  # cumulative, same on every rank
        scal["guard/nonfinite_iters"] = self.guard_nf_iters
        if self.main:
          for k, v in scal.items():
            self.writer.add_scalar(k, v, self.it)
          print(
            f"[it {self.it:7d} | {scal['train/wall_h']:5.2f} h] wm {scal.get('wm/loss', 0):.4f} (e1 {scal.get('wm/err_step1', 0):.3f}, e{cfg.wm.k} {scal.get(f'wm/err_step{cfg.wm.k}', 0):.3f}) "
            f"| imag_ret {scal.get('pi/imagined_return', 0):.4f} rew {scal.get('pi/imagined_reward_norm', 0):.3f} surv {scal.get('pi/survival_H', 0):.3f} | "
            f"ep_ret {scal.get('train/ep_return_norm', 0):.3f} len {scal.get('train/ep_len', 0):.0f} fail {scal.get('train/ep_failed_frac', 0):.2f}"
            + (f" (ee {scal.get('train/ep_failed_ee_frac', 0):.2f})" if self._ee_on else "") + " | "
            f"{scal['time/iter']*1000:.0f} ms/it (c {scal['time/collect']*1000:.0f} wm {scal['time/wm']*1000:.0f} pi {scal['time/policy']*1000:.0f} ref {scal.get('time/ref_gather', 0)*1000:.0f}) fps {fps:.0f}"
            + (f" | max/rank {scal['dist/iter_max']*1000:.0f} ms" if dctx.enabled else "")
            + (f" | guard skips {sum(self.guard_skips.values())} nonfinite its {self.guard_nf_iters}" if self.guard_nf_iters else ""),
            flush=True,
          )
        agg = {}
        for k in self.stats:
          self.stats[k] = []
        t_log, steps_at_log = now, self.env_steps

      if dctx.enabled and check_slot:
        self._check_consistency(f"it {self.it}")
      if ckpt_slot:
        p_ck = self.save(str(self.it)) if self.main else self._ckpt_path(str(self.it))  # all ranks track the path
        if self._healthy():
          self.wd_healthy_ckpt = p_ck
      if cfg.eval_enabled and eval_slot and self.main:
        te = time.time()
        try:
          res = self.evaluate()
          for k, v in res["scalars"].items():
            self.writer.add_scalar(f"eval/{k}", v, self.it)
          self.writer.add_scalar("time/eval", time.time() - te, self.it)
          print(f"[eval it {self.it}] return {res['scalars']['return_mean']:.4f} min {res['scalars']['return_min']:.3f} failed {int(res['scalars']['failed'])}/{res['num_clips']} ({time.time()-te:.0f}s)", flush=True)
          with open(os.path.join(cfg.log_dir, "eval.jsonl"), "a") as f:
            f.write(json.dumps({"it": self.it, "env_steps": self.env_steps, "wall_s": time.time() - self._t_start, **res["scalars"], "failed_clips": res["failed_clips"]}) + "\n")
        except Exception as e:  # noqa: BLE001
          import traceback

          traceback.print_exc()
          print(f"[eval] failed: {e!r}", flush=True)
          cfg.eval_enabled = False
      if dctx.enabled and (ckpt_slot or eval_slot):
        dctx.barrier()  # the other ranks wait (CPU barrier) while rank 0 saves / evaluates
    if dctx.enabled:
      self._merge_sampler()
      self._check_consistency("final")
    if self.main:
      self.save("final")
    if dctx.enabled:
      dctx.barrier()

  def run(self) -> None:
    self._t_start = time.time()
    self.train()
