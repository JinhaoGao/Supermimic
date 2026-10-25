"""SuperMimic training loop (Algorithm A.1 / Appendix D).

Per iteration:
  1. Collect: roll the policy (+ action noise) in the simulator, push to replay.
  2. Fit world model on k-step replay windows (autoregressive).
  3. Improve the policy by BPTT through H imagined steps with a critic
     bootstrap, then fit the critic to GAE targets of the same rollout.

Reference motions live in CPU RAM; the simulator loop gathers the current
frames / goal windows every step and each policy update gathers one
(H+K)-frame window pack per imagined rollout.
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

from .env_cfg import make_env_cfg
from .evaluate import Evaluator
from .fns import StepFns
from .heads import HeadsCfg
from .policy import Critic, Policy
from .replay import ReplayBuffer
from .sim_interface import ObsHistory, SimInterface
from .world_model import DynamicsModel, WorldModelCfg


@dataclass
class TrainCfg:
  motion_files: list[str] = field(default_factory=list)
  robot: str = "g1"
  robot_variant: str = ""  # see env_cfg.base_tracking_cfg (g1: eval; e2: nodamp)
  source_weights: tuple = ()  # ((source prefix, weight), ...) prior of the reset sampler's uniform share
  name_weights: tuple = ()  # ((name keyword, weight), ...) max with it (skill quotas, e.g. crawl / getup)
  # Hard-clip termination curriculum: clips matching hard_names (name keywords) get every termination threshold
  # (env + soft head, sigmoid temperatures too) multiplied by a piecewise-linear schedule over absolute
  # iterations ((it, mult), ...), constant outside; e.g. ((104000, 2.0), (108000, 2.0), (114000, 1.0)).
  hard_names: tuple = ()
  hard_term_schedule: tuple = ()
  sampler_clamp_returns: bool = False  # reset sampler bin returns clamped to [0, g_max] (library_command)
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
  # Trigger when EMA < best EMA of the last wd_window iterations - wd_drop. 0.08 (v1) misfired on the G1 full-atlas
  # resume: the return-based sampler drifts to harder bins and the imagined reward slides by ~0.08 without any
  # collapse (evals stayed 0.855-0.874); the real collapse at 41k dropped it by 0.5.
  wd_drop: float = 0.2
  wd_window: int = 4000
  wd_vloss_ratio: float = 50.0  # or when median critic loss of the last 50 its > ratio * median of the last 2000
  # Iterations after start / resume / rollback before the watchdog may trigger. Also the reward baseline starts only
  # after it: a resume begins with an empty replay that fills with fresh episode starts (easy states, imagined reward
  # ~0.90) and then turns over to the sampler's hard mid-episode states (~0.8) -- with a 500-iteration warm-up every
  # round-16 resume (10-24, G1 and E2) "collapsed" against that inflated baseline although the imagined return,
  # survival and WM errors were flat. >= replay turnover (replay_capacity / env_steps_per_iter) + margin.
  wd_warmup: int = 1500
  wd_sustain: int = 200  # the reward drop must hold this many consecutive iterations (single-iteration outliers)
  wd_lr_factor: float = 0.5  # LR multiplier applied at every rollback
  wd_max_rollbacks: int = 6
  wd_force_at: int = -1  # debug: force one rollback at this iteration
  # Logging / eval / checkpoints.
  log_interval: int = 50
  eval_interval: int = 2000
  ckpt_interval: int = 2000
  eval_enabled: bool = True
  eval_clips: int = 0  # 0 = all clips; otherwise a fixed random subset of this size
  eval_envs: int = 1024
  # Fixes from the comparison with open-source reproductions (10-25), all off by default (v2 behaviour):
  input_clip: float = 10.0  # clamp of normalized policy / critic inputs; 10 sigma saturates pitch rate at 5.5 rad/s,
  #                           roll beyond 56 deg and tilt beyond 103 deg on the G1 v2 statistics (flips are blind)
  clip_values: bool = False  # bootstrap V_H and the lambda-return values clamped to +-r_max/(1-gamma) (targets already are)
  critic_weight_survival: bool = False  # critic loss weighted by the imagined survival probability (post-death states)
  imag_truncate: bool = False  # imagined rollouts end at the clip end with a critic bootstrap (the env times out there;
  #                               v2 kept tracking the frozen last frame, whose velocity targets are not zero)
  wm_prefix_mask: bool = False  # WM windows that cross an episode end keep their steps up to it (v2 dropped the whole
  #                                window: the last k-1 transitions before every fall got up to k x less weight)
  clean_prev_action: bool = False  # proprio a_{t-1} = the policy's action, not the executed noisy one (as deployed)
  wm: WorldModelCfg = field(default_factory=WorldModelCfg)
  heads: HeadsCfg = field(default_factory=HeadsCfg)


def cosine_lambda(decay_iters: int, final_ratio: float):
  def f(it: int) -> float:
    if it >= decay_iters:
      return final_ratio
    return final_ratio + 0.5 * (1.0 - final_ratio) * (1.0 + math.cos(math.pi * it / decay_iters))

  return f


def clip_grads(params, norm: float, elem: float | None) -> float:
  total = torch.nn.utils.clip_grad_norm_(params, norm).item()
  if elem is not None:
    for p in params:
      if p.grad is not None:
        p.grad.clamp_(-elem, elem)
  return total


class Trainer:
  LIB_KEY = "train"

  def __init__(self, cfg: TrainCfg):
    self.cfg = cfg
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    self.device = cfg.device
    os.makedirs(cfg.log_dir, exist_ok=True)
    with open(os.path.join(cfg.log_dir, "train_cfg.json"), "w") as f:
      json.dump(asdict(cfg), f, indent=2, default=str)
    self.writer = SummaryWriter(cfg.log_dir)

    from mjlab.envs import ManagerBasedRlEnv

    env_cfg = make_env_cfg(
      cfg.motion_files, cfg.num_envs, mode="train", episode_length_s=cfg.episode_length_s, robot=cfg.robot,
      library_key=self.LIB_KEY, cache_dir=cfg.cache_dir, num_workers=cfg.num_workers,
      ee_term_z=cfg.heads.ee_term_z, robot_variant=cfg.robot_variant, source_weights=cfg.source_weights,
      name_weights=cfg.name_weights, hard_names=cfg.hard_names, clamp_returns=cfg.sampler_clamp_returns,
    )
    env_cfg.seed = cfg.seed
    self.env = ManagerBasedRlEnv(cfg=env_cfg, device=self.device)
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

    self.policy = Policy(self.heads.proprio_dim, hc.hist, self.heads.goal_frame_dim, hc.K, self.nj, input_clip=cfg.input_clip).to(self.device)
    self.critic = Critic(self.heads.feat_dim, self.heads.goal_frame_dim, hc.K, self.xi_dim).to(self.device)
    self.wm = DynamicsModel(self.nj, self.xi_dim, self.dt, cfg.wm).to(self.device)
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
    if cfg.eval_clips and cfg.eval_clips < self.lib.num_clips:
      rng = random.Random(cfg.seed)
      self.eval_clip_ids = tuple(sorted(rng.sample(range(self.lib.num_clips), cfg.eval_clips)))
    else:
      self.eval_clip_ids = tuple(range(self.lib.num_clips))

    n_params = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    print(f"[Trainer] policy {n_params(self.policy)/1e6:.1f}M  critic {n_params(self.critic)/1e6:.2f}M  wm {n_params(self.wm)/1e6:.1f}M")
    print(f"[Trainer] {self.lib.summary()}")
    print(f"[Trainer] goal_frame_dim={self.heads.goal_frame_dim} (anchor_pos={hc.goal_anchor_pos}), eval clips {len(self.eval_clip_ids)}")

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
    self.stats = {"ep_return_norm": [], "ep_len": [], "ep_failed": [], "term_mismatch": [], "hard_ep_failed": [], "hard_ep_len": []}
    # ---- watchdog ----
    self.wd_reward_ema: float | None = None
    self.wd_hist: deque = deque()
    self.wd_vloss: deque = deque(maxlen=2000)
    self.wd_healthy_ckpt: str | None = None
    self.wd_rollbacks = 0
    self.wd_quiet_until = cfg.wd_warmup
    self.wd_forced = False
    self.wd_low = 0

  # ---------------------------------------------------------------------------
  # 1. Data collection
  # ---------------------------------------------------------------------------

  @torch.no_grad()
  def collect_step(self) -> None:
    cfg = self.cfg
    ref_q, ref_qd, ref_ap, ref_aq = self.sim.goal_refs(self.frame, self.clip_end)
    ref_cur = self.sim.current_ref()  # reference at frame f_t: reward target of this step
    goal, a = self.fns.act(self.hist.buf, self.kin, ref_q, ref_qd, ref_ap, ref_aq)
    self.policy.obs_norm.update(self.hist.buf[:, -1])
    self.policy.goal_norm.update(goal)
    a_exec = a + cfg.action_noise_std * torch.randn_like(a)

    self.env.step(a_exec)
    next_state = self.sim.state()
    r, term_hard = self.fns.post(next_state, self.kin, a_exec, self.prev_action, *ref_cur, self.cmd.term_scale_env)
    terminated = self.env.reset_terminated.bool()
    timeouts = self.env.reset_time_outs.bool()
    done = terminated | timeouts
    self.stats["term_mismatch"].append((term_hard != terminated).float().mean().item())

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
      self._finish_episodes(ids, terminated[ids])
      self.env.reset(env_ids=ids)
      xi_new = self.sim.xi()
      self.xi[ids] = xi_new[ids]
      self.state = self.sim.state()
      self.prev_action[ids] = 0.0
      self.ep_step[ids] = 0
      self.ep_ret[ids] = 0.0
    else:
      self.state = next_state
    a_obs = a if cfg.clean_prev_action else a_exec
    self.prev_action = torch.where(done[:, None], torch.zeros_like(a_obs), a_obs)
    self.frame = self.cmd.time_steps.clone()
    self.clip_end = self.cmd.clip_end_env.clone()
    self.kin, obs = self.fns.refresh(self.state, self.prev_action, self.sim.bias_from_xi(self.xi, self.nj), True)
    self.hist.push(obs)
    if done.any():
      self.hist.reset(ids, obs)

  @torch.no_grad()
  def _finish_episodes(self, ids: torch.Tensor, failed: torch.Tensor) -> None:
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
    hard = self.cmd.env_hard[ids]
    if bool(hard.any()):
      self.stats["hard_ep_failed"].extend(failed[hard].float().tolist())
      self.stats["hard_ep_len"].extend(L[hard].tolist())

  # ---------------------------------------------------------------------------
  # 2. World model
  # ---------------------------------------------------------------------------

  def update_world_model(self) -> dict:
    cfg = self.cfg.wm
    self.wm.train()
    agg: dict[str, float] = {}
    for _ in range(cfg.batches_per_update):
      batch = self.buffer.sample_windows(cfg.windows_per_batch, cfg.k, prefix=self.cfg.wm_prefix_mask)
      self.wm.update_stats(batch)
      loss, e1, ek = self.fns.wm_loss(batch["s0"], batch["actions"], batch["next_states"], batch["xi"], batch["step_mask"])
      self.opt_wm.zero_grad(set_to_none=True)
      loss.backward()
      gn = clip_grads(list(self.wm.parameters()), cfg.grad_clip, None)
      self.opt_wm.step()
      m = {"wm/loss": loss.item(), "wm/err_step1": e1.item(), f"wm/err_step{cfg.k}": ek.item(), "wm/grad_norm": gn, "wm/valid_frac": batch["mask"].float().mean().item(), "wm/step_valid_frac": batch["step_mask"].float().mean().item()}
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
      # Termination multiplier of the hard-clip curriculum for the clip each rollout starts in (1 elsewhere).
      term_scale = 1.0 + (self.cmd.hard_mult - 1.0) * self.cmd.clip_hard[self.lib.clip_of_frame(st["frame"])].float()
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
    feats_n, goals_n, rews, dsoft, surv, lives = [], [], [], [], [], []
    v_max = cfg.heads.reward_max_per_step / (1.0 - g)
    # Clip-end truncation (imag_truncate): step t is a transition of the episode only while t < tlim (target frame
    # f0 + t <= clip_end - 2); the state reached at t = tlim is the env's time-out state -> critic bootstrap there.
    tlim = (clip_end_loc - 1) if cfg.imag_truncate else None
    p = (pack["joint_pos"], pack["joint_vel"], pack["body_pos"], pack["body_quat"], pack["body_lin_vel"], pack["body_ang_vel"])
    for t in range(H):
      s, obs_hist, prev_a, f_loc, kin, r, d, fn, gn = self.fns.imagine_step(s, obs_hist, prev_a, f_loc, clip_end_loc, xi, bias, kin, *p, term_scale)
      if tlim is not None:
        if cfg.use_critic:
          v_t = self.critic(fn, gn, xi)  # value of s_t (fn, gn are the critic inputs of the step's start state)
          if cfg.clip_values:
            v_t = v_t.clamp(-v_max, v_max)
          J = J + (g**t) * c * (tlim == t).float() * v_t
        live = (t < tlim).float()
      else:
        live = torch.ones_like(c)
      J = J + (g**t) * c * live * r
      surv.append(c.detach())
      lives.append(live)
      c = c * (1.0 - d)
      feats_n.append(fn.detach())
      goals_n.append(gn.detach())
      rews.append(r.detach())
      dsoft.append(d.detach())
    goal_H = self.fns.goal_from_pack(kin, f_loc, clip_end_loc, p[0], p[1], p[2], p[3])
    fn_H, gn_H = self.fns.critic_inputs(s, goal_H)
    if cfg.use_critic:
      V_H = self.critic(fn_H, gn_H, xi)
      if cfg.clip_values:
        V_H = V_H.clamp(-v_max, v_max)
      if tlim is not None:
        V_H = V_H * (tlim >= H).float()
      J = J + (g**H) * c * V_H
    loss_pi = -J.mean() / H
    self.opt_pi.zero_grad(set_to_none=True)
    loss_pi.backward()
    gn_pi = clip_grads(list(self.policy.parameters()), cfg.grad_clip_norm, cfg.grad_clip_elem)
    self.opt_pi.step()
    for p_ in self.wm.parameters():
      p_.requires_grad_(True)

    rmax = self.cfg.heads.reward_max_per_step
    R_all, L_all = torch.stack(rews), torch.stack(lives)
    metrics = {
      "pi/imagined_return": (J.mean() / H).item(),
      "pi/imagined_reward_norm": ((R_all * L_all).sum() / L_all.sum().clamp(min=1.0)).item() / rmax,
      # watchdog metric: per-step rewards clamped to [-r_max, r_max] (a few diverged rollouts with unbounded joint-limit
      # penalties drove the plain mean to -17 on single iterations: E2 round 16, 10-24)
      "pi/imagined_reward_robust": ((R_all.clamp(-rmax, rmax) * L_all).sum() / L_all.sum().clamp(min=1.0)).item() / rmax,
      "pi/survival_H": c.mean().item(),
      "pi/grad_norm": gn_pi,
      "pi/loss": loss_pi.item(),
      "time/ref_gather": self._t_gather,
    }
    if cfg.use_critic:
      metrics.update(self._fit_critic(feats_n, goals_n, fn_H.detach(), gn_H.detach(), xi, rews, dsoft, tlim,
                                      surv if cfg.critic_weight_survival else None))
    return metrics

  def _fit_critic(self, feats_n, goals_n, fn_H, gn_H, xi, rews, dsoft, tlim=None, surv=None) -> dict:
    cfg, H = self.cfg, self.cfg.H
    g, lam = cfg.gamma, cfg.lam
    N = xi.shape[0]
    v_max = self.cfg.heads.reward_max_per_step / (1.0 - g)
    with torch.no_grad():
      V = [self.critic(feats_n[t], goals_n[t], xi) for t in range(H)] + [self.critic(fn_H, gn_H, xi)]
      if cfg.clip_values:
        V = [v.clamp(-v_max, v_max) for v in V]
      R = [None] * (H + 1)
      R[H] = V[H]
      for t in reversed(range(H)):
        nxt = (1.0 - lam) * V[t + 1] + lam * R[t + 1]
        if tlim is not None:  # the state after step tlim-1 is a time-out: bootstrap, no lambda-mixing beyond it
          nxt = torch.where(tlim == t + 1, V[t + 1], nxt)
        R[t] = rews[t] + g * (1.0 - dsoft[t]) * nxt
      targets = torch.stack(R[:H], dim=1)
      if cfg.clip_value_targets:
        targets = targets.clamp(-v_max, v_max)
      W = torch.ones(N, H, device=self.device)
      if tlim is not None:
        W = W * (torch.arange(H, device=self.device)[None, :] < tlim[:, None]).float()
      if surv is not None:
        W = W * torch.stack(surv, dim=1)
    F = torch.stack(feats_n, dim=1).reshape(N * H, -1)
    G = torch.stack(goals_n, dim=1).reshape(N * H, -1)
    X = xi[:, None, :].expand(N, H, -1).reshape(N * H, -1)
    T = targets.reshape(N * H)
    Wf = W.reshape(N * H)
    total = N * H
    mb = total // cfg.critic_minibatches
    losses = []
    for _ in range(cfg.critic_epochs):
      perm = torch.randperm(total, device=self.device)
      for i in range(cfg.critic_minibatches):
        idx = perm[i * mb : (i + 1) * mb]
        v = self.critic(F[idx], G[idx], X[idx])
        hub = 2.0 * torch.nn.functional.huber_loss(v, T[idx], delta=cfg.critic_huber_delta, reduction="none")
        loss = (hub * Wf[idx]).sum() / Wf[idx].sum().clamp(min=1e-6)
        self.opt_v.zero_grad(set_to_none=True)
        loss.backward()
        clip_grads(list(self.critic.parameters()), cfg.grad_clip_norm, None)
        self.opt_v.step()
        losses.append(loss.item())
    return {"critic/loss": float(np.mean(losses)), "critic/target_mean": T.mean().item(), "critic/value_H_mean": V[H].mean().item()}

  # ---------------------------------------------------------------------------
  # Loop
  # ---------------------------------------------------------------------------

  def save(self, tag: str) -> str:
    path = os.path.join(self.cfg.log_dir, f"model_{tag}.pt")
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
    nb = ck["bin_G"].shape[0]
    old_files = list(ck.get("cfg", {}).get("motion_files", []))
    if ck["bin_G"].shape == self.cmd.bin_G.shape:
      self.cmd.bin_G.copy_(ck["bin_G"])
      self.cmd.bin_visits.copy_(ck["bin_visits"])
    elif nb < self.cmd.bin_G.shape[0] and old_files and old_files == list(self.cfg.motion_files[: len(old_files)]):
      # Library extended by appending clips (scripts/extend_cache.py): the old bins keep their indices.
      self.cmd.bin_G[:nb].copy_(ck["bin_G"])
      self.cmd.bin_visits[:nb].copy_(ck["bin_visits"])
      print(f"[Trainer] sampler statistics of {nb} bins restored into the extended library ({self.cmd.bin_G.shape[0]} bins)")
    if self.cfg.sampler_clamp_returns:
      self.cmd.bin_G.clamp_(0.0, self.cmd.cfg.g_max)
    self.it = ck["it"]
    self.env_steps = ck["env_steps"]
    for s in self.sched:
      s.last_epoch = self.it
    if not rollback and float(ck.get("lr_scale", 1.0)) != 1.0:
      self._set_lr_scale(float(ck["lr_scale"]))
      print(f"[Trainer] lr_scale {self.lr_scale} restored from checkpoint")
    self.wd_healthy_ckpt = path
    self.wd_quiet_until = self.it + self.cfg.wd_warmup
    print(f"[Trainer] resumed from {path} at iteration {self.it}")

  # ---------------------------------------------------------------------------
  # Watchdog: G1 full-atlas run 2026-10-22 collapsed at ~41k iterations (imagined reward 0.91 -> 0.4 in
  # ~400 iterations, critic loss x5000, then NaN in the reset sampler). Roll back instead of dying.
  # ---------------------------------------------------------------------------

  def _watchdog(self, m_pi: dict) -> bool:
    """Update watchdog statistics; roll back if training collapsed. Returns True after a rollback."""
    cfg = self.cfg
    if not cfg.wd_enabled:
      return False
    r = float(m_pi.get("pi/imagined_reward_robust", m_pi.get("pi/imagined_reward_norm", float("nan"))))
    vl = float(m_pi.get("critic/loss", float("nan")))
    if math.isfinite(vl):
      self.wd_vloss.append(vl)
    if not math.isfinite(r):
      return self._rollback("imagined reward is not finite")
    a = cfg.wd_ema
    self.wd_reward_ema = r if self.wd_reward_ema is None else (1 - a) * self.wd_reward_ema + a * r
    if cfg.wd_force_at >= 0 and self.it >= cfg.wd_force_at and not self.wd_forced:
      self.wd_forced = True
      return self._rollback(f"forced at {self.it} (debug)")
    if self.it < self.wd_quiet_until:  # no baseline from the warm-up (replay refill after a start / resume)
      self.wd_low = 0
      return False
    self.wd_hist.append((self.it, self.wd_reward_ema))
    while self.wd_hist and self.wd_hist[0][0] < self.it - cfg.wd_window:
      self.wd_hist.popleft()
    best = max(e for _, e in self.wd_hist)
    self.wd_low = self.wd_low + 1 if self.wd_reward_ema < best - cfg.wd_drop else 0
    if self.wd_low >= cfg.wd_sustain:
      return self._rollback(f"imagined reward EMA {self.wd_reward_ema:.3f} < best {best:.3f} - {cfg.wd_drop} for {self.wd_low} its")
    if len(self.wd_vloss) >= 500:
      recent = float(np.median(list(self.wd_vloss)[-50:]))
      base = float(np.median(self.wd_vloss))
      if recent > cfg.wd_vloss_ratio * max(base, 1e-6):
        return self._rollback(f"critic loss median {recent:.3g} > {cfg.wd_vloss_ratio} x {base:.3g}")
    return False

  def hard_mult_at(self, it: int) -> float:
    pts = sorted((int(a), float(b)) for a, b in self.cfg.hard_term_schedule)
    if not pts:
      return 1.0
    if it <= pts[0][0]:
      return pts[0][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
      if it <= x1:
        return y0 + (y1 - y0) * (it - x0) / max(x1 - x0, 1)
    return pts[-1][1]

  def _healthy(self) -> bool:
    if self.wd_reward_ema is None or not self.wd_hist:
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
    print(f"[watchdog] it {self.it}: {reason}", flush=True)
    if path is None or not os.path.exists(path):
      print("[watchdog] no healthy checkpoint yet; continuing", flush=True)
      self.wd_quiet_until = self.it + cfg.wd_warmup
      return False
    if self.wd_rollbacks >= cfg.wd_max_rollbacks:
      raise RuntimeError(f"[watchdog] {self.wd_rollbacks} rollbacks already; giving up ({reason})")
    it_before = self.it
    self.load(path, rollback=True)  # models, normalizers, optimizers, sampler bins, iteration counter
    # Forget the replay written since the checkpoint (collapsed policy): windows / imagination starts are drawn
    # by age < size, so size 0 restricts them to fresh data; the train loop refills the warm-up first. v1 kept it,
    # and every reload re-measured the imagined reward on the collapsed states.
    self.buffer.size = 0
    self._set_lr_scale(self.lr_scale * cfg.wd_lr_factor)
    self.wd_rollbacks += 1
    self.wd_reward_ema = None
    self.wd_hist.clear()
    self.wd_vloss.clear()
    self.wd_quiet_until = self.it + cfg.wd_warmup
    msg = (f"[watchdog] rollback #{self.wd_rollbacks}: it {it_before} -> {self.it} from {path}; "
           f"LR x{cfg.wd_lr_factor} (policy lr now {self.opt_pi.param_groups[0]['lr']:.2e})")
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
        robot=self.cfg.robot, robot_variant=self.cfg.robot_variant,
      )
    return self.evaluator.run(self.policy)

  def train(self) -> None:
    cfg = self.cfg
    warmup = max(cfg.wm.k + cfg.heads.hist + 2, 32)
    t_log = time.time()
    steps_at_log = self.env_steps
    agg: dict[str, list[float]] = {}
    while self.it < cfg.iterations:
      t0 = time.time()
      if cfg.hard_term_schedule:
        self.cmd.set_hard_mult(self.hard_mult_at(self.it))
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
      m_wm = self.update_world_model()
      t2 = time.time()
      m_pi = self.update_policy()
      t3 = time.time()
      for s in self.sched:
        s.step()
      self.it += 1
      if self._watchdog(m_pi):
        agg = {}
        continue
      for k, v in {**m_wm, **m_pi, "time/collect": t1 - t0, "time/wm": t2 - t1, "time/policy": t3 - t2, "time/iter": t3 - t0}.items():
        agg.setdefault(k, []).append(v)

      if self.it % cfg.log_interval == 0:
        now = time.time()
        fps = (self.env_steps - steps_at_log) * self.E / max(now - t_log, 1e-6)
        scal = {k: float(np.mean(v)) for k, v in agg.items()}
        scal["time/env_fps"] = fps
        scal["time/iter_per_s"] = cfg.log_interval / max(now - t_log, 1e-6)
        scal["train/env_steps_per_env"] = self.env_steps
        scal["train/lr"] = self.opt_pi.param_groups[0]["lr"]
        scal["train/wall_h"] = (now - self._t_start) / 3600
        if self.stats["ep_return_norm"]:
          scal["train/ep_return_norm"] = float(np.mean(self.stats["ep_return_norm"]))
          scal["train/ep_len"] = float(np.mean(self.stats["ep_len"]))
          scal["train/ep_failed_frac"] = float(np.mean(self.stats["ep_failed"]))
        scal["train/term_mismatch"] = float(np.mean(self.stats["term_mismatch"])) if self.stats["term_mismatch"] else 0.0
        hard_txt = ""
        if cfg.hard_names:
          scal["hard/term_mult"] = self.cmd.hard_mult
          if self.stats["hard_ep_failed"]:
            scal["hard/ep_failed_frac"] = float(np.mean(self.stats["hard_ep_failed"]))
            scal["hard/ep_len"] = float(np.mean(self.stats["hard_ep_len"]))
          hard_txt = (f" | hard x{self.cmd.hard_mult:.2f} fail {scal.get('hard/ep_failed_frac', float('nan')):.2f} "
                      f"len {scal.get('hard/ep_len', float('nan')):.0f}")
        scal["sampler/entropy"] = self.cmd.metrics["sampling_entropy"][0].item()
        scal["sampler/top1_prob"] = self.cmd.metrics["sampling_top1_prob"][0].item()
        scal["sampler/bins_visited_frac"] = (self.cmd.bin_visits > 0).float().mean().item()
        if self.wd_reward_ema is not None:
          scal["guard/reward_ema"] = self.wd_reward_ema
        scal["guard/rollbacks_total"] = self.wd_rollbacks
        for k, v in scal.items():
          self.writer.add_scalar(k, v, self.it)
        print(
          f"[it {self.it:7d} | {scal['train/wall_h']:5.2f} h] wm {scal.get('wm/loss', 0):.4f} (e1 {scal.get('wm/err_step1', 0):.3f}, e{cfg.wm.k} {scal.get(f'wm/err_step{cfg.wm.k}', 0):.3f}) "
          f"| imag_ret {scal.get('pi/imagined_return', 0):.4f} rew {scal.get('pi/imagined_reward_norm', 0):.3f} surv {scal.get('pi/survival_H', 0):.3f} | "
          f"ep_ret {scal.get('train/ep_return_norm', 0):.3f} len {scal.get('train/ep_len', 0):.0f} fail {scal.get('train/ep_failed_frac', 0):.2f}{hard_txt} | "
          f"{scal['time/iter']*1000:.0f} ms/it (c {scal['time/collect']*1000:.0f} wm {scal['time/wm']*1000:.0f} pi {scal['time/policy']*1000:.0f} ref {scal.get('time/ref_gather', 0)*1000:.0f}) fps {fps:.0f}",
          flush=True,
        )
        agg = {}
        for k in self.stats:
          self.stats[k] = []
        t_log, steps_at_log = now, self.env_steps

      if cfg.ckpt_interval and self.it % cfg.ckpt_interval == 0:
        p_ck = self.save(str(self.it))
        if self._healthy():
          self.wd_healthy_ckpt = p_ck
      if cfg.eval_enabled and cfg.eval_interval and self.it % cfg.eval_interval == 0:
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
    self.save("final")

  def run(self) -> None:
    self._t_start = time.time()
    self.train()
