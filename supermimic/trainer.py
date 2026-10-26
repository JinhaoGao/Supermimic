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
rollback targets. With wd_reward_floor set, each rank's finite imagined reward
is clamped from below first (one rank's spike must not roll everyone back).
Simulator states that come back non-finite end their episode as a failure (the
replay keeps the finite pre-step state).

master 4314a50 / c1c94be / 5f020a9 (merged): hard-clip termination curriculum
(hard_names / hard_term_schedule; the multiplier is a function of the iteration
only, so every rank applies the same one), resume into a library extended by
appending clips (scripts/extend_cache.py; the old bins keep their sampler
statistics), and the sampler return clamp, which is this branch's switch already.

Opt-in (2026-10-25): wd_signal "imagined_reward_alive" makes the watchdog watch
sum_t c_t r_t / sum_t c_t (the survival weights of the policy objective; logged
as pi/imagined_reward_alive either way) instead of the unweighted mean; early
stop (es_eval_fail_ratio / es_patience) on the in-training eval's failures, with
rank 0's result broadcast so every rank stops at the same iteration; a watchdog
rollback drops the evals of the abandoned branch from its history, and a resume
at an eval slot evaluates the loaded weights first.

Root orientation (2026-10-25): opt-in heads.root_ori_term (simulator term + soft
term in imagination, pelvis full angle incl. yaw, the eval site's root_ori test)
and the always-on FailureWatch (supermimic/watch.py): train/would_fail_ori and
the floor / spin watch groups, counted on the device, summed over ranks at the
log; off == the code before (the counters never reach the learner). Logging
only as well: the same angle in imagination (pi/root_ori_angle_t0 / _tH /
over_frac, with the term on pi/root_ori_hazard_mean / death_share) and the
world model's k-step root drift (wm/root_ang_err_step<k>, wm/root_yaw_err_*).
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from collections import deque
from dataclasses import asdict, dataclass, field, fields

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from .distributed import DistCtx
from .env_cfg import make_env_cfg
from .evaluate import Evaluator
from .fns import StepFns
from .heads import HeadsCfg, window_frames
from .math_utils import quat_angle, quat_apply_inverse, yaw_quat
from .motions import read_name_list, resolve_clip_names
from .policy import Critic, Policy
from .replay import ReplayBuffer
from .sim_interface import ObsHistory, SimInterface
from .watch import ORI_FAIL_THRESHOLD, FailureWatch
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
  hard_term_schedule: tuple = ()  # master 4314a50 (bj27 r13 / r14); off = mjlab's terminations, untouched
  # allmimic sampler pieces (10-26; LibraryMotionCommandCfg.softmax_cap / cluster_k / cluster_alpha): per-bin cap of the
  # softmax share at sampler_cap x uniform, cluster-balanced uniform share; sampler_boot_mean bootstraps time-outs with
  # the episode's mean reward instead of its last reward (one joint-limit penalty step no longer sinks a bin).
  sampler_cap: float = 0.0
  sampler_clusters: int = 0
  sampler_cluster_alpha: float = 0.3
  sampler_cluster_mix: float = 0.5
  sampler_boot_mean: bool = False
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
  # Reset sampler: bin returns clamped to [0, g_max] (LibraryMotionCommandCfg.clamp_returns). One switch for ddp-bj31
  # 5bb8c0d and master 5f020a9 (same name in both branches' TrainCfg, so either checkpoint cfg compares / loads).
  sampler_clamp_returns: bool = False
  compile: bool = True
  compile_mode: str | None = None
  num_threads: int = 0  # torch intra-op CPU threads (0 = torch default); scripts/train.py also sets OMP_NUM_THREADS
  # Data parallel (torchrun). Every rank: num_envs envs + rollouts_per_update rollouts (global = world_size x).
  world_size: int = 1  # informational (set by scripts/train.py from torchrun)
  base_lr_scale: float = 1.0  # --base_lr_scale, already folded into policy_lr / critic_lr / wm.lr (record only)
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
  wd_warmup: int = 500  # watchdog quiet iterations after start / resume / rollback (bj27 single-GPU line: 1500)
  wd_baseline_after_warmup: bool = False  # see _watchdog (master 10-25 behaviour when on)
  wd_sustain: int = 1  # the reward drop must hold this many consecutive iterations (master 10-25 used 200 against
  # single-iteration outliers; 1 = immediate rollback, the data-parallel branch's behaviour, which has wd_reward_floor)
  wd_lr_factor: float = 0.5  # LR multiplier applied at every rollback
  wd_max_rollbacks: int = 6
  wd_force_at: int = -1  # debug: force one rollback at this iteration
  wd_nf_patience: int = 100  # consecutive non-finite iterations (skipped update / NaN imagined reward) before a rollback
  # Lower bound on each rank's imagined reward (normalized) before the rank mean and the EMA; None = off (as before),
  # production -1.0. bj31 g1v2_ddp8_ws86k 2026-10-23: one rank's value came back finite but astronomically negative
  # (rank 0 0.918, rank-mean EMA -2.2 at it 100808 -> rollback; the EMA restarted from such a value read -47392 at the
  # end of the warm-up -> rollback #2): two LR halvings for nothing (weights intact, grad clipping). A real collapse
  # (10-22: 0.91 -> 0.4) stays far above -1; non-finite values keep the non-finite path.
  wd_reward_floor: float | None = None
  wd_early_ckpt: int = 500  # one early checkpoint (rollback point) at this iteration when none exists yet (0 = off)
  # Reward signal of the watchdog EMA (both normalized by reward_max_per_step, both logged every iteration):
  #   "imagined_reward_norm" (default, as before): mean of r_t over every imagined step and rollout, dead ones included;
  #   "imagined_reward_alive": sum_t c_t r_t / sum_t c_t with the survival weights c_t of the policy objective
  #   J = sum_t gamma^t c_t r_t. 10-24 diagnosis: spikes of the first come from world-model divergence after the soft
  #   termination (states the objective weights ~0); r16 (bj27) rolled back 6x on them down to LR 2e-8 and gave up.
  wd_signal: str = "imagined_reward_norm"
  # Early stop on the in-training benchmark (None = off, the loop as before). After every eval: bad if its failed
  # count >= es_eval_fail_ratio x median(failed of all previous evals of this run; needs >= 3, median floored at 1);
  # es_patience bad evals in a row -> stop (earlystop.json, model_final, clean exit on every rank). 10-24: failures of
  # the deterministic eval rise ~28k iterations before the v4 cliff (corr -0.984). Also tracks the eval with the
  # fewest failures (best_eval.json + model_best_eval.pt, rewritten at every new best). A watchdog rollback drops the
  # evals after its target from the history (streak / median of the restored branch only); resumed at an eval slot,
  # the loaded weights are evaluated before the first update (history and best start from them).
  es_eval_fail_ratio: float | None = None
  es_patience: int = 2
  es_median_floor: float = 1.0  # floor of the previous evals' median failed count (see early_stop_verdict)
  # Logging / eval / checkpoints.
  log_interval: int = 50
  eval_interval: int = 2000
  ckpt_interval: int = 2000
  eval_enabled: bool = True
  eval_clips: int = 0  # 0 = all clips; otherwise a fixed random subset of this size
  eval_clip_names: str = ""  # txt of npz basenames (one per line): fixed eval subset by name; overrides eval_clips
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
  # After a start / resume: iterations of world-model-only fitting (e.g. a fresh WM after a representation change), then
  # iterations in which imagination runs and the critic is fitted but the policy is not stepped.
  policy_delay: int = 0
  policy_freeze: int = 0
  wm_delay_lr: float = 1.0  # world-model learning-rate multiplier during policy_delay (a grafted / fresh WM refits faster)
  wm: WorldModelCfg = field(default_factory=WorldModelCfg)
  heads: HeadsCfg = field(default_factory=HeadsCfg)


def remap_columns(w: torch.Tensor, cols: torch.Tensor, sign: torch.Tensor | None = None) -> torch.Tensor:
  """[out, in_old] -> [out, in_new]: new column j = sign[j] * old column cols[j]; zero where cols[j] < 0 (new input)."""
  out = torch.zeros(w.shape[0], len(cols), dtype=w.dtype, device=w.device)
  keep = (cols >= 0).to(w.device)
  src = w[:, cols.to(w.device)[keep]]
  out[:, keep] = src if sign is None else src * sign.to(w)[keep]
  return out


# Heading-frame WM features [z, rot6d(R_local) 6, v_h, w_b, q, qd]: rot6d = (R00, R01, R10, R11, R20, R21) of the
# heading-removed rotation, whose last row equals the full rotation's, so root-frame gravity g_b = -(R20, R21, R22):
# rot6d entries 4, 5 are exactly -g_b[0], -g_b[1] (any orientation). Grafts map them over with the sign flipped.
ROT6D_TO_GRAV = ((4, 0), (5, 1))


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


ROOT_ORI_KEYS = ("root_ori_term", "root_ori_term_temp", "root_ori_body")  # HeadsCfg fields added 2026-10-25
# Opt-in HeadsCfg fields of later features: (switch, keys); written only when switched on (see cfg_dict).
OPT_IN_HEADS_KEYS = (
  (lambda h: h.root_ori_on, ROOT_ORI_KEYS),
  (lambda h: h.goal_ref_root, ("goal_ref_root",)),  # master 052247b (10-25)
  (lambda h: h.critic_errors, ("critic_errors",)),  # master 052247b (10-25)
  (lambda h: h.root_z_on, ("root_z_term", "root_z_term_temp")),  # 10-26
)


def heads_off_keys(h: HeadsCfg) -> tuple[str, ...]:
  """HeadsCfg keys of the opt-in features that are off in h (left out of saved cfgs)."""
  return tuple(k for on, keys in OPT_IN_HEADS_KEYS if not on(h) for k in keys)


def cfg_dict(cfg: "TrainCfg") -> dict:
  """asdict(cfg) for train_cfg.json and the checkpoints. The heads dict leaves out the keys of every opt-in feature
  that is off (root-orientation term, goal_ref_root, critic_errors, root-height term), so a run with them off has
  55e0eaf's keys and older code (55e0eaf / master HeadsCfg(**d), HeadsCfg.from_dict) still loads it; this code's
  from_dict fills the defaults back in. On: written as they are (older code refuses them)."""
  d = asdict(cfg)
  for k in heads_off_keys(cfg.heads):
    d["heads"].pop(k)
  return d


WD_SIGNALS = {"imagined_reward_norm": "imagined reward", "imagined_reward_alive": "imagined alive reward",
              "imagined_reward_robust": "imagined robust reward"}  # -> log label
ES_MIN_PREV = 3  # early stop: previous evals needed before the first verdict
EVAL_MAX_ERRORS = 3  # in-training evaluation is switched off after this many failures in a row


def _check_cfg(cfg: TrainCfg) -> None:
  if cfg.wd_signal not in WD_SIGNALS:
    raise ValueError(f"wd_signal {cfg.wd_signal!r}: one of {sorted(WD_SIGNALS)}")
  if cfg.es_eval_fail_ratio is not None and not (cfg.es_eval_fail_ratio > 0 and cfg.es_patience >= 1):
    raise ValueError(f"early stop needs es_eval_fail_ratio > 0 and es_patience >= 1 (got {cfg.es_eval_fail_ratio}, {cfg.es_patience})")
  if cfg.hard_term_schedule and not cfg.hard_names:
    raise ValueError("hard_term_schedule needs hard_names (the hard set it applies to)")
  f = cfg.wd_reward_floor  # nan = silently off, inf = a rollback every wd_nf_patience its, > 0 masks every collapse
  if f is not None and not (math.isfinite(float(f)) and float(f) <= 0.0):
    raise ValueError(f"wd_reward_floor must be finite and <= 0 (normalized reward; production -1.0), got {f}")
  if not cfg.es_median_floor >= 1.0:
    raise ValueError(f"es_median_floor must be >= 1, got {cfg.es_median_floor}")


def early_stop_verdict(prev_failed: list, failed: float, ratio: float, median_floor: float = 1.0) -> tuple[bool, float | None]:
  """Is an eval with ``failed`` failures bad, given the failed counts of all previous evals of the run? ->
  (bad, threshold). threshold = ratio x max(median(prev), median_floor) once there are ES_MIN_PREV previous evals (the
  floor keeps a median of 0 failures from making every later eval bad; at the default 1 a median of 2 already makes 3
  failures bad -- --es_median_floor 5 for 1024-clip evals with single-digit failures), else (False, None)."""
  if len(prev_failed) < ES_MIN_PREV:
    return False, None
  thr = float(ratio) * max(float(np.median(np.asarray(prev_failed, dtype=np.float64))), float(median_floor))
  return bool(float(failed) >= thr), thr


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
    _check_cfg(cfg)
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
        json.dump(cfg_dict(cfg), f, indent=2, default=str)
    self.writer = SummaryWriter(cfg.log_dir) if self.main else None

    from mjlab.envs import ManagerBasedRlEnv

    env_cfg = make_env_cfg(
      cfg.motion_files, cfg.num_envs, mode="train", episode_length_s=cfg.episode_length_s, robot=cfg.robot,
      library_key=self.LIB_KEY, cache_dir=cfg.cache_dir, num_workers=cfg.num_workers,
      ee_termination=cfg.heads.ee_termination, ee_threshold=cfg.heads.ee_term_z, clamp_returns=cfg.sampler_clamp_returns,
      robot_variant=cfg.robot_variant, source_weights=cfg.source_weights, name_weights=cfg.name_weights,
      hard_names=cfg.hard_names, root_ori_term=cfg.heads.root_ori_term, root_ori_body=cfg.heads.root_ori_body,
      root_z_term=cfg.heads.root_z_term, softmax_cap=cfg.sampler_cap, cluster_k=cfg.sampler_clusters,
      cluster_alpha=cfg.sampler_cluster_alpha, cluster_mix=cfg.sampler_cluster_mix,
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

    self.policy = Policy(self.heads.proprio_dim, hc.hist, self.heads.goal_frame_dim, hc.K, self.nj, input_clip=cfg.input_clip).to(self.device)
    self.wm = DynamicsModel(self.nj, self.xi_dim, self.dt, cfg.wm).to(self.device)
    self.critic = Critic(self.wm.feat_dim, self.heads.goal_frame_dim, hc.K, self.xi_dim, err_dim=self.heads.critic_err_dim).to(self.device)
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
    self._print(f"[Trainer] goal_frame_dim={self.heads.goal_frame_dim} (anchor_pos={hc.goal_anchor_pos}, ref_root={hc.goal_ref_root}), "
                f"critic error features {self.heads.critic_err_dim}, eval clips {len(self.eval_clip_ids)}")
    self._print(
      "[Trainer] end-effector termination: "
      +(f"on, z error > {hc.ee_term_z} on {list(self.sim.ee_body_names)} (soft temp {hc.ee_term_z_temp})" if hc.ee_termination else "off (paper recipe)")
    )
    self._print(
      "[Trainer] root-orientation termination: "
      + (f"on, {self.sim.root_ori_body} full rotation angle (incl. yaw) > {hc.root_ori_term:g} rad (soft temp {hc.root_ori_term_temp:g})"
         if hc.root_ori_on else f"off ({self.sim.root_ori_body} angle logged only: train/would_fail_ori)")
    )
    # Always-on failure logging (no effect on training): pelvis angle vs the eval site's 1.0 rad + floor / spin groups.
    self.watch = FailureWatch([c.name for c in self.lib.clips], self.E, self.device)
    self._print(f"[Trainer] failure watch: {self.watch.summary(self.lib.bin_clip)}"
                + ("" if self.heads.root_ori is not None else f" -- {self.sim.root_ori_body} not tracked: no pelvis-angle logging"))
    try:  # the robot variant actually built (G1 eval: 35.11 kg, mjlab's G1: 33.34 kg)
      mass = f"{float(self.env.sim.mj_model.body_mass.sum()):.2f} kg"
    except Exception:  # noqa: BLE001
      mass = "?"
    self._print(f"[Trainer] robot {cfg.robot} variant {cfg.robot_variant or '(default)'}: model mass {mass}")
    mc = self.cmd.cfg
    if mc.source_weights or mc.name_weights:
      self._print(f"[Trainer] reset sampler prior: source_weights {list(mc.source_weights)} name_weights {list(mc.name_weights)}")
    self._print("[Trainer] reset sampler: bin returns " + (f"clamped to [0, {mc.g_max:g}]" if mc.clamp_returns else "unclamped (paper recipe)")
                + (f", softmax share capped at {mc.softmax_cap:g} x uniform" if mc.softmax_cap > 0 else "")
                + (f", cluster-balanced prior (k={mc.cluster_k}, alpha={mc.cluster_alpha:g}, mix {mc.cluster_mix:g})" if mc.cluster_k > 0 else "")
                + (", time-outs bootstrapped with the mean reward" if cfg.sampler_boot_mean else ""))
    if hc.root_z_on:
      self._print(f"[Trainer] root-height termination: on, {self.sim.root_ori_body} height error > {hc.root_z_term:g} m "
                  f"(soft temp {hc.root_z_term_temp:g})")
    if cfg.hard_names:
      self._print(f"[Trainer] hard-clip termination curriculum: {int(self.cmd.clip_hard.sum())} clips match {list(cfg.hard_names)}, "
                  f"schedule {list(cfg.hard_term_schedule) or 'none (x1)'}")
    if cfg.wd_signal != "imagined_reward_norm":
      self._print(f"[Trainer] watchdog signal: pi/{cfg.wd_signal}")
    if cfg.es_eval_fail_ratio is not None:
      self._print(f"[Trainer] early stop: eval failed >= {cfg.es_eval_fail_ratio:g} x median(previous evals, >= {ES_MIN_PREV}) for "
                  f"{cfg.es_patience} evals in a row" + ("" if cfg.eval_enabled and cfg.eval_interval > 0 else " -- INACTIVE: evaluation is off"))
    self._print(f"[Trainer] library fingerprint {self.lib.fingerprint} ({os.path.basename(self.lib.cache_dir)})")
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
    self.stats = {"ep_return_norm": [], "ep_len": [], "ep_failed": [], "ep_failed_ee": [], "term_mismatch": [], "sim_nonfinite": [],
                  "hard_ep_failed": [], "hard_ep_len": []}
    self._ee_on = hc.ee_termination
    self._ro_on = hc.root_ori_on  # root-orientation term in the env (its failures are not end-effector-only)
    self._hard_on = bool(cfg.hard_names)  # curriculum off: term_scale None everywhere (the code path before it)
    # ---- early stop (in-training benchmark; same state on every rank: rank 0's eval result is broadcast) ----
    self._es_on = cfg.es_eval_fail_ratio is not None and cfg.eval_enabled and cfg.eval_interval > 0
    self.es_hist: list[dict] = []  # one entry per eval of this run: it, failed, return_mean, threshold, bad
    self.es_streak = 0
    self.es_best: dict | None = None  # eval with the fewest failures: it, failed, return_mean, ckpt
    self.es_stop: dict | None = None  # set when the early stop fired (the loop ends after that iteration)
    # ---- NaN guard ----
    self.guard_skips = {"wm": 0, "pi": 0, "critic": 0}  # skipped (non-finite) updates since start, same on every rank
    self.guard_nf_iters = 0  # iterations with a skipped update or a non-finite imagined reward
    # ---- watchdog ----
    self.wd_reward_ema: float | None = None
    self.wd_hist: deque = deque()
    self.wd_vloss: deque = deque(maxlen=2000)
    self.wd_healthy_ckpt: str | None = None
    self._resume_from: tuple[str, str] | None = None  # (path, clip-list relation) of --resume (Trainer.load)
    self.wd_rollbacks = 0
    self.wd_quiet_until = cfg.wd_warmup
    self.wd_forced = False
    self.wd_low = 0
    self.wd_nf_streak = 0  # consecutive non-finite iterations
    self.guard_floored = 0  # per-rank imagined rewards raised to wd_reward_floor since start (sum over ranks)
    self.guard_floor_iters = 0  # iterations with at least one of them (same on every rank)

  # ---------------------------------------------------------------------------
  # 1. Data collection
  # ---------------------------------------------------------------------------

  @torch.no_grad()
  def collect_step(self) -> None:
    cfg = self.cfg
    refs = self.sim.goal_refs(self.frame, self.clip_end)
    ref_cur = self.sim.current_ref()  # reference at frame f_t: reward target of this step
    goal, a = self.fns.act(self.hist.buf, self.kin, *refs)
    self.dist.rms_update([(self.policy.obs_norm, self.hist.buf[:, -1]), (self.policy.goal_norm, goal)])
    a_exec = a + cfg.action_noise_std * torch.randn_like(a)

    self.env.step(a_exec)
    next_state = self.sim.state()
    r, term_hard = self.fns.post(next_state, self.kin, a_exec, self.prev_action, *ref_cur, self.cmd.term_scale_env if self._hard_on else None)
    self.watch.step(self.heads.root_ori_angle_state(next_state, ref_cur[1]))  # logging only: pelvis angle > 1.0 rad
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
      root = tm.get_term("anchor_pos") | tm.get_term("anchor_ori")
      if self._ro_on:  # the eval site tests root_ori before the end effectors too
        root = root | tm.get_term("root_ori")
      ee_fail = tm.get_term("ee_body_pos") & ~root & ~bad

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
    a_obs = a if cfg.clean_prev_action else a_exec
    self.prev_action = torch.where(done[:, None], torch.zeros_like(a_obs), a_obs)
    self.frame = self.cmd.time_steps.clone()
    self.clip_end = self.cmd.clip_end_env.clone()
    self.kin, obs = self.fns.refresh(self.state, self.prev_action, self.sim.bias_from_xi(self.xi, self.nj), True)
    self.hist.push(obs)
    if done.any():
      self.hist.reset(ids, obs)
      self.watch.resets(self.lib.clip_of_frame(self.frame[ids]))  # logging only: the clips the new episodes start in

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
    if self.cfg.sampler_boot_mean:
      r_last = self.ep_ret[ids] / L.float().clamp(min=1.0)
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
    if self._hard_on:  # hard-clip curriculum: episodes of the hard set (the clip they reset into)
      hard = self.cmd.env_hard[ids]
      if bool(hard.any()):
        self.stats["hard_ep_failed"].extend(failed[hard].float().tolist())
        self.stats["hard_ep_len"].extend(L[hard].tolist())
    # Logging only: pelvis-angle flag + watch groups of the ended episodes (self.frame = their last frame, in their clip).
    self.watch.episodes(ids, failed, self.lib.clip_of_frame(self.frame[ids]))

  # ---------------------------------------------------------------------------
  # 2. World model
  # ---------------------------------------------------------------------------

  def update_world_model(self) -> dict:
    cfg = self.cfg.wm
    self.wm.train()
    agg: dict[str, float] = {}
    for _ in range(cfg.batches_per_update):
      if self.cfg.wm_prefix_mask:  # windows crossing an episode end keep the steps up to it (step_mask)
        batch = self.buffer.sample_windows(cfg.windows_per_batch, cfg.k, prefix=True)
      else:
        batch = self.buffer.sample_windows(cfg.windows_per_batch, cfg.k)
      step_mask = batch.get("step_mask", batch["mask"])  # [n, k] or the whole-window mask [n]
      self.dist.rms_update(self.wm.stats_pairs(batch))
      loss, e1, ek = self.fns.wm_loss(batch["s0"], batch["actions"], batch["next_states"], batch["xi"], step_mask)
      self.opt_wm.zero_grad(set_to_none=True)
      loss.backward()
      gn = self._apply_grads("wm", self.opt_wm, list(self.wm.parameters()), cfg.grad_clip, None)
      m = {"wm/loss": loss.item(), "wm/err_step1": e1.item(), f"wm/err_step{cfg.k}": ek.item(), "wm/grad_norm": gn,
           "wm/valid_frac": batch["mask"].float().mean().item()}
      if self.cfg.wm_prefix_mask:
        m["wm/step_valid_frac"] = step_mask.float().mean().item()
      for k, v in m.items():
        agg[k] = agg.get(k, 0.0) + v / cfg.batches_per_update
    # logging only: last batch, updated weights; whole windows (with wm_prefix_mask a window may end at an episode end)
    agg.update(self._wm_root_drift({**batch, "mask": step_mask.all(1)} if step_mask.dim() == 2 else batch))
    return agg

  @torch.no_grad()
  def _wm_root_drift(self, batch: dict) -> dict:
    """Logging only (2026-10-25; no grad, no RNG, nothing written): the root (free-joint body, G1 pelvis) orientation
    error of the world model's open-loop k-step rollout on a replay batch, mean over valid windows -- the full rotation
    angle (the eval site's root_ori, incl. yaw; the --root_ori_term quantity) and its yaw part, after 1 and after k steps.
    The model is yaw-invariant: its root yaw only integrates predicted angular velocity, so imagined root yaw over the
    H-step horizon drifts by more than this k-step value (compare with pi/root_ori_angle_* and the 1.0 rad threshold)."""
    s = batch["s0"]
    acts, nxt, m = batch["actions"], batch["next_states"], batch["mask"]
    k = acts.shape[1]
    n = m.float().sum().clamp(min=1.0)
    out = {}
    for j in range(k):
      s = self.wm.step(s, acts[:, j], batch["xi"])
      if j in (0, k - 1):
        qt, qp = nxt[:, j, 3:7], s[:, 3:7]
        for name, e in (("ang", quat_angle(qt, qp)), ("yaw", quat_angle(yaw_quat(qt), yaw_quat(qp)))):
          out[f"wm/root_{name}_err_step{j + 1}"] = float(torch.where(m, e, torch.zeros_like(e)).sum() / n)
    return out

  # ---------------------------------------------------------------------------
  # 3. Policy + critic through imagination
  # ---------------------------------------------------------------------------

  def update_policy(self, step_policy: bool = True) -> dict:
    """Imagined rollouts -> policy step (unless step_policy is False: policy_freeze) and critic fit."""
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
      # Termination multiplier of the hard-clip curriculum for the clip each rollout starts in (1 elsewhere); None
      # without the curriculum (the code path before it).
      term_scale = None
      if self._hard_on:
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
      if getattr(heads, "critic_err_dim", 0):  # error normalizer statistics from the rollout start states (real replay states)
        self.dist.rms_update([(self.critic.err_norm, heads.critic_errors(kin, pack["body_pos"][:, 0], pack["body_quat"][:, 0]))])
      f_loc = torch.zeros(N, dtype=torch.long, device=self.device)
      clip_end_loc = torch.clamp(st["clip_end"] - st["frame"], max=Lw)
    self._t_gather = time.time() - t_g0
    s = st["state"]
    prev_a = st["prev_action"]
    c = torch.ones(N, device=self.device)
    J = torch.zeros(N, device=self.device)
    feats_n, goals_n, rews, dsoft, alive, lives = [], [], [], [], [], []
    v_max = cfg.heads.reward_max_per_step / (1.0 - g)
    # Clip-end truncation (imag_truncate): step t is a transition of the episode only while t < tlim (target frame
    # f0 + t <= clip_end - 2); the state reached at t = tlim is the env's time-out state -> critic bootstrap there.
    tlim = (clip_end_loc - 1) if cfg.imag_truncate else None
    p = (pack["joint_pos"], pack["joint_vel"], pack["body_pos"], pack["body_quat"], pack["body_lin_vel"], pack["body_ang_vel"])
    ro_angles = [] if heads.root_ori is not None else None  # logging only (pi/root_ori_*; None: body not tracked)
    ar_n = torch.arange(N, device=self.device)
    for t in range(H):
      f_ref = f_loc  # target frame of this step's reward / termination (imagine_step advances f_loc)
      s, obs_hist, prev_a, f_loc, kin, r, d, fn, gn = self.fns.imagine_step(s, obs_hist, prev_a, f_loc, clip_end_loc, xi, bias, kin, *p, term_scale)
      if ro_angles is not None:
        ro_angles.append(heads.root_ori_angle_state(s.detach(), p[3][ar_n, f_ref]))
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
      alive.append(c.detach())  # c_t: survival weight of r_t in J (product of 1 - d_s for s < t; c_0 = 1)
      lives.append(live)
      c = c * (1.0 - d)
      feats_n.append(fn.detach())
      goals_n.append(gn.detach())
      rews.append(r.detach())
      dsoft.append(d.detach())
    goal_H = self.fns.goal_from_pack(kin, f_loc, clip_end_loc, *p)
    if getattr(heads, "critic_err_dim", 0):  # privileged critic error features (critic_errors)
      fn_H, gn_H = self.fns.critic_inputs(s, goal_H, self.fns.critic_err(kin, f_loc, p[2], p[3]))
    else:
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
    gn_pi = 0.0
    if step_policy:  # policy_delay / policy_freeze: the same iterations on every rank
      loss_pi.backward()
      gn_pi = self._apply_grads("pi", self.opt_pi, list(self.policy.parameters()), cfg.grad_clip_norm, cfg.grad_clip_elem)
    for p_ in self.wm.parameters():
      p_.requires_grad_(True)

    rmax = self.cfg.heads.reward_max_per_step
    R_all, L_all = torch.stack(rews), torch.stack(lives)
    metrics = {
      "pi/imagined_return": (J.mean() / H).item(),
      # mean over the imagined steps of the episodes (imag_truncate drops the steps past the clip end)
      "pi/imagined_reward_norm": (R_all.mean() if tlim is None else (R_all * L_all).sum() / L_all.sum().clamp(min=1.0)).item() / rmax,
      # watchdog metric: per-step rewards clamped to [-r_max, r_max] (a few diverged rollouts with unbounded joint-limit
      # penalties drove the plain mean to -17 on single iterations: E2 round 16, 10-24)
      "pi/imagined_reward_robust": ((R_all.clamp(-rmax, rmax) * L_all).sum() / L_all.sum().clamp(min=1.0)).item() / rmax,
      # sum_t c_t r_t / sum_t c_t over all rollouts (c as in J, no discount): the reward the objective actually weighs;
      # the means above also average the steps after a soft death, where the world model diverges (10-24).
      "pi/imagined_reward_alive": self._alive_reward(torch.stack(alive) * L_all, R_all),
      "pi/survival_H": c.mean().item(),
      "pi/grad_norm": gn_pi,
      "pi/loss": loss_pi.item(),
      "time/ref_gather": self._t_gather,
    }
    if ro_angles is not None:
      metrics.update(self._imagined_root_ori(torch.stack(ro_angles), torch.stack(alive), torch.stack(dsoft), term_scale))
    if cfg.use_critic:
      metrics.update(self._fit_critic(feats_n, goals_n, fn_H.detach(), gn_H.detach(), xi, rews, dsoft, tlim,
                                      alive if cfg.critic_weight_survival else None))
    return metrics

  @torch.no_grad()
  def _imagined_root_ori(self, ang: torch.Tensor, c: torch.Tensor, d: torch.Tensor, term_scale: torch.Tensor | None) -> dict:
    """Logging only (2026-10-25): the root-orientation body's full angle vs the reference in imagination, ang [H, N] (the
    quantity of --root_ori_term; logged with the term off too), with survival weights c [H, N] and soft dones d [H, N].
      pi/root_ori_angle_t0 / _tH        mean angle after the first / the last imagined step (world-model yaw drift shows
                                        as tH >> t0 when the real episodes stay below, train/ori_over_step_frac)
      pi/root_ori_over_frac             imagined steps with angle > 1.0 rad (the eval site; real side: train/ori_over_step_frac)
      pi/root_ori_hazard_mean           (term on) mean soft root-orientation term over all imagined steps
      pi/root_ori_death_share           (term on) sum c p_ro / sum c d: the share of the imagined soft deaths that comes from
                                        the root-orientation term (noisy-OR: an upper bound when several terms fire at once)"""
    out = {
      "pi/root_ori_angle_t0": float(ang[0].nanmean()),
      "pi/root_ori_angle_tH": float(ang[-1].nanmean()),
      "pi/root_ori_over_frac": float((ang > ORI_FAIL_THRESHOLD).float().mean()),
    }
    if self.cfg.heads.root_ori_on:
      hz = self.heads.root_ori_hazard(ang, term_scale)  # [H, N] (scale [N] broadcasts)
      out["pi/root_ori_hazard_mean"] = float(hz.nanmean())
      dsum = (c * d).sum()
      out["pi/root_ori_death_share"] = float((c * hz).nansum() / dsum) if float(dsum) > 0 else 0.0
    return out

  def _alive_reward(self, c: torch.Tensor, r: torch.Tensor) -> float:
    """sum c_t r_t / sum c_t over [H, N] survival weights c and rewards r, normalized by reward_max_per_step (the same
    units as pi/imagined_reward_norm, so wd_drop / wd_reward_floor mean the same). c_0 = 1: the denominator >= N."""
    return ((c * r).sum() / c.sum()).item() / self.cfg.heads.reward_max_per_step

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
        "cfg": cfg_dict(self.cfg),
        "dims": dict(proprio=self.heads.proprio_dim, hist=self.cfg.heads.hist, goal_frame=self.heads.goal_frame_dim, K=self.cfg.heads.K, nj=self.nj, xi=self.xi_dim),
      },
      path,
    )
    return path

  def _resume_diffs(self, ck_cfg: dict) -> tuple[list[str], str]:
    """(training settings of the checkpoint's run that differ from this run, clip-list relation). Resume only: a
    rollback reloads this run's own checkpoint, or the resume checkpoint with the relation found here (Trainer.load).
    Keys missing in master / v1 checkpoints take their defaults there (no variant, no prior, no clamp, no curriculum;
    EE via HeadsCfg.from_dict). Clip lists compare by basename (bj27 vs bj31 roots): "same" (or no list in the
    checkpoint = unknown = same), "prefix" (this list = the checkpoint's + clips appended, scripts/extend_cache.py;
    master c1c94be) or "other"."""
    cur = self.cfg

    def w(x) -> tuple:
      return tuple((str(k), float(v)) for k, v in (x or ()))

    def sched(x) -> tuple:
      return tuple((int(a), float(b)) for a, b in (x or ()))

    pairs = [
      ("robot", ck_cfg.get("robot", "g1"), cur.robot),
      ("robot_variant", ck_cfg.get("robot_variant", "") or "", cur.robot_variant or ""),
      ("source_weights", w(ck_cfg.get("source_weights")), w(cur.source_weights)),
      ("name_weights", w(ck_cfg.get("name_weights")), w(cur.name_weights)),
      ("sampler_clamp_returns", bool(ck_cfg.get("sampler_clamp_returns", False)), bool(cur.sampler_clamp_returns)),
      ("hard_names", tuple(str(k) for k in ck_cfg.get("hard_names", ()) or ()), tuple(str(k) for k in cur.hard_names)),
      ("hard_term_schedule", sched(ck_cfg.get("hard_term_schedule")), sched(cur.hard_term_schedule)),
    ]
    if "H" in ck_cfg:
      pairs.append(("H", int(ck_cfg["H"]), int(cur.H)))
    # Master-side settings (10-25 / 10-26) that change the forward pass, the targets or the sampling without being in
    # any state_dict: a resume that forgets --fixes silently drops input_clip from 50 back to 10, etc.
    pairs += [
      ("input_clip", float(ck_cfg.get("input_clip", 10.0)), float(cur.input_clip)),
      ("clip_values", bool(ck_cfg.get("clip_values", False)), bool(cur.clip_values)),
      ("critic_weight_survival", bool(ck_cfg.get("critic_weight_survival", False)), bool(cur.critic_weight_survival)),
      ("imag_truncate", bool(ck_cfg.get("imag_truncate", False)), bool(cur.imag_truncate)),
      ("wm_prefix_mask", bool(ck_cfg.get("wm_prefix_mask", False)), bool(cur.wm_prefix_mask)),
      ("clean_prev_action", bool(ck_cfg.get("clean_prev_action", False)), bool(cur.clean_prev_action)),
      ("action_noise_std", float(ck_cfg.get("action_noise_std", 0.1)), float(cur.action_noise_std)),
      ("env_steps_per_iter", float(ck_cfg.get("env_steps_per_iter", 1.0)), float(cur.env_steps_per_iter)),
      ("wm.body_frame", bool((ck_cfg.get("wm") or {}).get("body_frame", False)), bool(cur.wm.body_frame)),
      ("sampler_cap", float(ck_cfg.get("sampler_cap", 0.0)), float(cur.sampler_cap)),
      ("sampler_clusters", int(ck_cfg.get("sampler_clusters", 0)), int(cur.sampler_clusters)),
      ("sampler_boot_mean", bool(ck_cfg.get("sampler_boot_mean", False)), bool(cur.sampler_boot_mean)),
    ]
    if int(ck_cfg.get("sampler_clusters", 0)) or int(cur.sampler_clusters):
      pairs += [
        ("sampler_cluster_alpha", float(ck_cfg.get("sampler_cluster_alpha", 0.3)), float(cur.sampler_cluster_alpha)),
        ("sampler_cluster_mix", float(ck_cfg.get("sampler_cluster_mix", 0.5)), float(cur.sampler_cluster_mix)),
      ]
    diffs = [f"{k} = {a!r} in the checkpoint, {b!r} here" for k, a, b in pairs if a != b]
    try:
      hk = HeadsCfg.from_dict(ck_cfg["heads"])
      a, b = asdict(hk), asdict(cur.heads)
      if not (hk.ee_termination or cur.heads.ee_termination):  # threshold / temperature unused when off on both
        for d in (a, b):
          for k in ("ee_term_z", "ee_term_z_temp"):
            d.pop(k)
      if not (hk.root_ori_on or cur.heads.root_ori_on):  # root_ori_term 0 on both (checkpoints before it: 0)
        for d in (a, b):
          for k in ("root_ori_term_temp", "root_ori_body"):
            d.pop(k)
      diffs += [f"heads.{k} = {a.get(k)!r} in the checkpoint, {b[k]!r} here" for k in b if a.get(k) != b[k]]
    except Exception as e:  # noqa: BLE001 -- a warning helper must never stop a resume
      diffs.append(f"heads cfg not comparable ({type(e).__name__}: {e})")
    rel = "same"
    ck_files = ck_cfg.get("motion_files")
    if ck_files is not None:
      old = [os.path.basename(f) for f in ck_files]
      new = [os.path.basename(f) for f in cur.motion_files]
      if old == new:
        rel = "same"
      elif 0 < len(old) < len(new) and new[: len(old)] == old:
        rel = "prefix"
        diffs.append(f"motion files: the checkpoint's {len(old)} + {len(new) - len(old)} appended here (extended library)")
      elif len(old) != len(new):
        rel = "other"
        diffs.append(f"motion files: {len(old)} in the checkpoint, {len(new)} here")
      else:
        rel = "other"
        diffs.append(f"motion files: {len(old)} on both, but other basenames / order")
    return diffs, rel

  def load(self, path: str, rollback: bool = False) -> None:
    ck = torch.load(path, map_location=self.device, weights_only=False)
    if rollback:  # this run's own checkpoint ("same"), or the resume checkpoint (until this run wrote a healthy one; r15
      # rolled back to it twice): then the clip-list relation found at the resume (prefix / other), not "same"
      rf = self._resume_from
      rel = rf[1] if rf is not None and rf[0] == path else "same"
    else:  # resume: v2 options / EE / data are silent dimensions -> say what differs (identical on every rank)
      diffs, rel = self._resume_diffs(ck.get("cfg") or {})
      for d in diffs:
        self._print(f"[Trainer] WARNING resume from another setup: {d}")
      self._resume_from = (path, rel)
    # New policy / critic inputs (goal_ref_root, critic_errors): zero columns grafted onto the checkpoint's first layers
    # (same on every rank: a pure function of the checkpoint and the cfg).
    new_goal = self._graft_inputs(ck)
    self.policy.load_state_dict(ck["policy"])
    self.critic.load_state_dict(ck["critic"])
    if bool(ck["cfg"].get("wm", {}).get("body_frame", False)) != self.cfg.wm.body_frame:
      self._graft_wm(ck)
    self.wm.load_state_dict(ck["wm"])
    self.opt_wm.load_state_dict(ck["opt_wm"])
    self.opt_pi.load_state_dict(ck["opt_pi"])
    self.opt_v.load_state_dict(ck["opt_v"])
    self._load_sampler_state(ck, rel, path, rollback)
    self.it = ck["it"]
    self.env_steps = ck["env_steps"]
    for s in self.sched:
      s.last_epoch = self.it
    if not rollback:  # also at 1.0: the optimizer state brought the saved lr, not this run's --lr (resumed line below)
      self._set_lr_scale(float(ck.get("lr_scale", 1.0)))
      if self.lr_scale != 1.0:
        self._print(f"[Trainer] lr_scale {self.lr_scale} restored from checkpoint")
    if new_goal:
      self._calibrate_goal_norm(new_goal)
    self.wd_healthy_ckpt = path
    self.wd_quiet_until = self.it + self.cfg.wd_warmup
    self._print(f"[Trainer] resumed from {path} at iteration {self.it}" + ("" if rollback else
                f" (policy lr {self.opt_pi.param_groups[0]['lr']:.3e} = lr x lr_scale {self.lr_scale:g} x cosine {self.sched[1].lr_lambdas[0](self.it):.4f})"))

  def _load_sampler_state(self, ck: dict, rel: str, path: str, rollback: bool = False) -> None:
    """Reset-sampler statistics (bin_G / bin_visits) of a checkpoint: all bins for the same clip list; for a library
    extended by appending clips (rel "prefix", master c1c94be) into its first bins, which keep their indices (checked:
    the first files of this list must have exactly the checkpoint's bin count here); otherwise not at all (they would
    land on other clips). Then clamped when clamp_returns is on (5bb8c0d / 5f020a9). Every rank runs this on the same
    file and library -> identical state (Trainer._dist_start re-checks). Rollback: bins not taken from the checkpoint
    keep their current statistics (as master's load does)."""
    c = self.cmd
    G, V = ck["bin_G"], ck["bin_visits"]
    nb, NB = G.shape[0], c.bin_G.shape[0]
    restored = False
    if G.shape == c.bin_G.shape and rel == "same":
      c.bin_G.copy_(G)
      c.bin_visits.copy_(V)
      restored = True
    elif rel == "prefix" and nb < NB:
      n_old = len((ck.get("cfg") or {}).get("motion_files") or ())
      have = self.lib.bins_of_first_files(n_old)
      if have == nb:
        c.bin_G[:nb].copy_(G)
        c.bin_visits[:nb].copy_(V)
        restored = True
        self._print(f"[Trainer] sampler statistics of {nb} bins restored into the extended library ({NB} bins; "
                    + (f"the {NB - nb} appended bins keep their current statistics)" if rollback else f"{NB - nb} new bins start unvisited)"))
      else:
        why = f"extended library, but its first {n_old} files have {have} bins here, {nb} in the checkpoint"
    if restored:
      if c.cfg.clamp_returns:  # resume / rollback: the EMA restarts from bounded values (same on every rank)
        n, g_min = c.clamp_bin_returns_()
        self._print(f"[Trainer] clamp_returns: {n} of {c.bin_G.numel()} loaded bin returns clamped to [0, {c.cfg.g_max:g}] (min before {g_min:.3f})")
      return
    if rel != "prefix" or nb >= NB:
      why = f"{nb} bins in the checkpoint, {NB} here" if G.shape != c.bin_G.shape else "same bin count, other clip list"
    self._print(f"[Trainer] WARNING reset-sampler state (bin_G / bin_visits) of {path} not loaded ({why}): the return-based sampler "
                + ("kept the current statistics (rollback)" if rollback else "starts from scratch"))

  # ---------------------------------------------------------------------------
  # Input graft: resume a checkpoint with fewer goal segments / no critic error features (10-25, round 18).
  # ---------------------------------------------------------------------------

  def _graft_inputs(self, ck: dict) -> list[str]:
    """Rewrite ``ck`` in place from its goal / critic input layout to this run's: every input column of the goal encoder
    and of the critic's first layer, their Adam moments and the goal normalizer move to their new positions; new inputs
    get zero weights, so the grafted networks compute exactly what they computed before (the critic only approximately
    when the WM representation changed). Returns the goal segments that are new (their normalizer statistics still need
    calibrating)."""
    known = {f.name for f in fields(HeadsCfg)} | {"ee_term_temp"}  # from_dict maps master's v2 EE keys
    old_h = HeadsCfg.from_dict({k: v for k, v in ck["cfg"]["heads"].items() if k in known})
    if "dims" not in ck or not hasattr(self.heads, "goal_layout"):  # (stand-ins in tests: nothing to graft)
      return []
    if ck["dims"]["goal_frame"] == 2 * self.nj + 9:  # checkpoints from before the goal_anchor_pos flag
      old_h.goal_anchor_pos = True
    old_seg, new_seg = self.heads.goal_layout(old_h), self.heads.goal_layout()
    old_err = 4 * len(self.heads.tracked) + 1 if old_h.critic_errors else 0
    new_err = self.heads.critic_err_dim
    J = self.nj
    wm_layout = lambda body: [("z", 1), ("grav" if body else "rot6d", 3 if body else 6), ("v", 3), ("w", 3), ("q", J), ("qd", J)]  # noqa: E731
    old_feat, new_feat = wm_layout(bool(ck["cfg"].get("wm", {}).get("body_frame", False))), wm_layout(self.cfg.wm.body_frame)
    if old_seg == new_seg and old_err == new_err and old_feat == new_feat:
      return []
    assert sum(n for _, n in old_seg) == ck["dims"]["goal_frame"], (old_seg, ck["dims"])
    missing = [n for n, _ in old_seg if n not in dict(new_seg)]
    if missing or (old_err and old_err != new_err):
      raise ValueError(f"graft can only add inputs; checkpoint has {missing or 'critic error features'} this run lacks")

    def frame_map(seg_from, seg_to):  # new goal-frame column -> old column (-1 = new input)
      off, start = 0, {}
      for n, k in seg_from:
        start[n], off = off, off + k
      return torch.cat([torch.arange(start[n], start[n] + k) if n in start else torch.full((k,), -1) for n, k in seg_to])

    fm = frame_map(old_seg, new_seg)
    Go, Gn, K = ck["dims"]["goal_frame"], len(fm), self.cfg.heads.K
    goal_cols = torch.cat([torch.where(fm >= 0, fm + k * Go, -1) for k in range(K)])  # encoder input map
    # Critic state features (WM features): heading-frame v -> root-frame v keeps its weights (equal when upright);
    # gravity x, y take the rot6d entries they equal (sign flipped, see ROT6D_TO_GRAV), gravity z is new.
    fd, xd = sum(n for _, n in old_feat), self.xi_dim
    feat_cols = frame_map(old_feat, new_feat)
    feat_sign = torch.ones(len(feat_cols))
    if dict(old_feat).get("rot6d") and dict(new_feat).get("grav"):
      for e, g in ROT6D_TO_GRAV:
        feat_cols[1 + g], feat_sign[1 + g] = 1 + e, -1.0
    crit_cols = torch.cat([feat_cols, torch.full((new_err,), -1), torch.where(goal_cols >= 0, goal_cols + fd + old_err, -1),
                           torch.arange(xd) + fd + old_err + K * Go])
    crit_sign = torch.cat([feat_sign, torch.ones(len(crit_cols) - len(feat_cols))])

    def graft(sd, opt_sd, module, key, cols, sign=None):
      sd[key] = remap_columns(sd[key], cols, sign)
      idx = [n for n, _ in module.named_parameters()].index(key)
      st = opt_sd["state"].get(idx)
      if st is not None:
        st["exp_avg"] = remap_columns(st["exp_avg"], cols, sign)
        st["exp_avg_sq"] = remap_columns(st["exp_avg_sq"], cols)

    pol, crit = ck["policy"], ck["critic"]
    graft(pol, ck["opt_pi"], self.policy, "encoder.0.weight", goal_cols)
    graft(crit, ck["opt_v"], self.critic, "net.0.weight", crit_cols, crit_sign)
    keep = fm >= 0
    for b, fill in (("mean", 0.0), ("var", 1.0)):
      v = torch.full((Gn,), fill, dtype=pol[f"goal_norm.{b}"].dtype, device=pol[f"goal_norm.{b}"].device)
      v[keep.to(v.device)] = pol[f"goal_norm.{b}"][fm[keep].to(v.device)]
      pol[f"goal_norm.{b}"] = v
    for k, v in self.critic.state_dict().items():  # new critic error normalizer: fresh statistics
      crit.setdefault(k, v)
    new = [n for n, _ in new_seg if n not in dict(old_seg)]
    self._print(f"[Trainer] graft: goal frame {Go} -> {Gn} (new {new}), critic error features {old_err} -> {new_err}, "
          f"critic state features {[n for n, _ in old_feat]} -> {[n for n, _ in new_feat]}")
    return new

  @torch.no_grad()
  def _graft_wm(self, ck: dict, n: int = 65536) -> None:
    """Heading-frame world model -> root-frame (body_frame) warm start, in place on ck: the input columns of z, v (heading
    -> root frame, equal when upright), w, q, qd, action and xi keep their weights and Adam moments; root-frame gravity
    x, y take over the two rot6d entries they equal (ROT6D_TO_GRAV: weights and normalizer mean negated, same std), so
    the first-order tilt information is kept exactly; gravity z is a new input (zero weights, statistics from n random
    reference root orientations); the other four rot6d entries (second order in the tilt) are dropped. The output layout
    is unchanged (dv / rp are now read in the root frame). Fit it before the policy uses it: policy_delay."""
    if not self.cfg.wm.body_frame:
      raise ValueError("world-model graft only goes from the heading frame to body_frame")
    J, xd = self.nj, self.xi_dim
    na = 13 + 3 * J  # old features + action
    cols = torch.cat([torch.zeros(1, dtype=torch.long), torch.full((3,), -1), torch.arange(7, na), torch.arange(na, na + xd)])
    sign = torch.ones(len(cols))
    for e, gi in ROT6D_TO_GRAV:
      cols[1 + gi], sign[1 + gi] = 1 + e, -1.0
    w = ck["wm"]["net.0.weight"]
    ck["wm"]["net.0.weight"] = remap_columns(w, cols, sign)
    idx = [k for k, _ in self.wm.named_parameters()].index("net.0.weight")
    st = ck["opt_wm"]["state"].get(idx)
    if st is not None:
      st["exp_avg"] = remap_columns(st["exp_avg"], cols, sign)
      st["exp_avg_sq"] = remap_columns(st["exp_avg_sq"], cols)
    gen = torch.Generator(device=self.device).manual_seed(0)  # same frames on every DDP rank (identical statistics)
    fr = torch.randint(0, self.lib.time_step_total, (n,), device=self.device, generator=gen)
    qr = self.lib.gather(fr, fields=["body_quat"])["body_quat"][:, 0]  # reference root orientation
    g = quat_apply_inverse(qr, torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(n, 3))
    m_old, v_old = ck["wm"]["in_norm.mean"], ck["wm"]["in_norm.var"]
    g_mean = torch.stack([-m_old[1 + 4], -m_old[1 + 5], g.mean(0)[2].to(m_old)])
    g_var = torch.stack([v_old[1 + 4], v_old[1 + 5], g.var(0, unbiased=False)[2].to(v_old)])
    ck["wm"]["in_norm.mean"] = torch.cat([m_old[:1], g_mean, m_old[7:]])
    ck["wm"]["in_norm.var"] = torch.cat([v_old[:1], g_var, v_old[7:]])
    self._print(f"[Trainer] world model grafted heading frame -> root frame: input {w.shape[1]} -> {len(cols)}, gravity mean "
          f"{g_mean.cpu().numpy().round(3).tolist()} std {g_var.sqrt().cpu().numpy().round(3).tolist()}; "
          f"policy_delay {self.cfg.policy_delay} (WM lr x{self.cfg.wm_delay_lr}), policy_freeze {self.cfg.policy_freeze}")

  @torch.no_grad()
  def _calibrate_goal_norm(self, segments: list[str], n: int = 65536) -> None:
    """Normalizer statistics of grafted goal segments from n random library frames (the running count of the
    checkpoint is too large for training updates to move them). Only segments that depend on the reference alone."""
    assert set(segments) <= {"ref_root"}, segments
    lib, K = self.lib, self.cfg.heads.K
    gen = torch.Generator(device=self.device).manual_seed(0)  # same frames on every DDP rank (identical statistics)
    fr = torch.randint(0, lib.time_step_total, (n,), device=self.device, generator=gen)
    clip_end = lib.clip_end[lib.clip_of_frame(fr)]
    win = window_frames(fr, clip_end, K).reshape(-1)
    g = lib.gather(win, fields=["joint_pos", "joint_vel", "body_pos", "body_quat", "body_lin_vel", "body_ang_vel"])
    a = self.heads.anchor
    ap, aq = g["body_pos"].view(n, K, -1, 3)[:, :, a], g["body_quat"].view(n, K, -1, 4)[:, :, a]
    kin = {"anchor_pos": ap[:, 0], "anchor_quat": aq[:, 0]}
    goal = self.heads.goal(kin, g["joint_pos"].view(n, K, -1), g["joint_vel"].view(n, K, -1), ap, aq,
                           g["body_lin_vel"].view(n, K, -1, 3)[:, :, a], g["body_ang_vel"].view(n, K, -1, 3)[:, :, a])
    x = goal.reshape(-1, goal.shape[-1])
    off = 0
    gn = self.policy.goal_norm
    for name, k in self.heads.goal_layout():
      if name in segments:
        gn.mean[off : off + k] = x[:, off : off + k].mean(0)
        gn.var[off : off + k] = x[:, off : off + k].var(0, unbiased=False)
        self._print(f"[Trainer] goal_norm {name}: mean {gn.mean[off:off + k].cpu().numpy().round(3).tolist()} "
              f"std {gn.std[off:off + k].cpu().numpy().round(3).tolist()}")
      off += k

  # ---------------------------------------------------------------------------
  # Watchdog: G1 full-atlas run 2026-10-22 collapsed at ~41k iterations (imagined reward 0.91 -> 0.4 in
  # ~400 iterations, critic loss x5000, then NaN in the reset sampler). Roll back instead of dying.
  # ---------------------------------------------------------------------------

  def _watchdog(self, m_pi: dict, skipped: int = 0) -> bool:
    """Update watchdog statistics; roll back if training collapsed. Returns True after a rollback.

    skipped: module updates the NaN guard skipped this iteration (identical on every rank).
    The reward signal is pi/<cfg.wd_signal> (default imagined_reward_norm, as before; imagined_reward_alive: the
    survival-weighted mean of the policy objective); the floor, the rank mean and every test below act on it alike."""
    cfg = self.cfg
    r = float(m_pi.get(f"pi/{cfg.wd_signal}", float("nan")))
    label = WD_SIGNALS[cfg.wd_signal]
    vl = float(m_pi.get("critic/loss", float("nan")))
    floor, raw = cfg.wd_reward_floor, None
    if floor is not None and math.isfinite(r) and r < floor:  # one rank's spike must not drag the mean (see TrainCfg)
      raw, r = r, float(floor)
    n_floored = int(raw is not None)
    if self.dist.enabled:  # data parallel: every rank decides on the same rank means -> rolls back together
      s = {"r": r, "vl": vl}
      if floor is not None:  # same keys on every rank (cfg); off: the reduce of before
        s["floored"] = float(n_floored)
      g = self.dist.reduce_scalars(s, "mean", check_keys=False)
      r, vl = g["r"], g["vl"]
      if floor is not None:
        n_floored = round(g["floored"] * self.dist.world)
    if n_floored:
      self._note_reward_floor(n_floored, raw)
    if skipped or not math.isfinite(r):
      # Transient: the non-finite updates were skipped, the weights are intact. Only a long streak means the
      # weights / normalizers themselves are broken -> roll back (no checkpoint yet: keep skipping, never give up).
      self.guard_nf_iters += 1
      self.wd_nf_streak += 1
      if self.wd_nf_streak <= 3 or self.wd_nf_streak % 100 == 0:
        sk = ", ".join(f"{k} {v}" for k, v in self.guard_skips.items())
        self._print(f"[guard] it {self.it}: non-finite ({label} {r:.4g}, {skipped} updates skipped; streak {self.wd_nf_streak}, total skips {sk})", flush=True)
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
    # wd_baseline_after_warmup (master 10-25): the warm-up (replay refill after a start / resume) adds nothing to the
    # history the drop is measured against; off (default) = the data-parallel branch: history from the first iteration.
    if not cfg.wd_baseline_after_warmup or self.it >= self.wd_quiet_until:
      self.wd_hist.append((self.it, self.wd_reward_ema))
      while self.wd_hist and self.wd_hist[0][0] < self.it - cfg.wd_window:
        self.wd_hist.popleft()
    if cfg.wd_force_at >= 0 and self.it >= cfg.wd_force_at and not self.wd_forced:
      self.wd_forced = True
      return self._rollback(f"forced at {self.it} (debug)")
    if self.it < self.wd_quiet_until:
      self.wd_low = 0
      return False
    best = max(e for _, e in self.wd_hist)
    self.wd_low = getattr(self, "wd_low", 0) + 1 if self.wd_reward_ema < best - cfg.wd_drop else 0
    if self.wd_low >= cfg.wd_sustain:  # (identical on every rank: rank-mean signal; wd_sustain 1 = immediate)
      return self._rollback(f"{label} EMA {self.wd_reward_ema:.3f} < best {best:.3f} - {cfg.wd_drop}"
                            + (f" for {self.wd_low} its" if cfg.wd_sustain > 1 else ""))
    if len(self.wd_vloss) >= 500:
      recent = float(np.median(list(self.wd_vloss)[-50:]))
      base = float(np.median(self.wd_vloss))
      if recent > cfg.wd_vloss_ratio * max(base, 1e-6):
        return self._rollback(f"critic loss median {recent:.3g} > {cfg.wd_vloss_ratio} x {base:.3g}")
    return False

  def _note_reward_floor(self, n: int, raw: float | None) -> None:
    """n ranks (identical on every rank) had their imagined reward floored this iteration; raw: this rank's value."""
    self.guard_floored += n
    self.guard_floor_iters += 1
    k = self.guard_floor_iters
    if k <= 3 or k % 100 == 0:  # same k on every rank -> every rank joins the gather
      raws = self.dist.all_gather_object(raw)
      which = ", ".join(f"rank {i} {v:.4g}" for i, v in enumerate(raws) if v is not None)
      self._print(f"[guard] it {self.it}: {WD_SIGNALS[self.cfg.wd_signal]} floored to {self.cfg.wd_reward_floor:g} on {n}/{self.dist.world} "
                  f"rank(s) (raw: {which}); {k} iterations / {self.guard_floored} rank values floored so far", flush=True)

  def _guard_scalars(self) -> dict:
    """Watchdog / NaN-guard scalars: built from rank-reduced state only, so keys and values match on every rank."""
    scal = {}
    if self.wd_reward_ema is not None:
      scal["guard/reward_ema"] = self.wd_reward_ema  # same on every rank (watchdog runs on rank means)
    scal["guard/rollbacks_total"] = self.wd_rollbacks
    for k, v in self.guard_skips.items():
      scal[f"guard/skips_{k}"] = v  # cumulative, same on every rank
    scal["guard/nonfinite_iters"] = self.guard_nf_iters
    if self.cfg.wd_reward_floor is not None:  # key depends on cfg only
      scal["guard/reward_floored"] = self.guard_floored  # cumulative floored rank values
    return scal

  def hard_mult_at(self, it: int) -> float:
    """Termination multiplier of the hard set at iteration ``it`` (master 4314a50): piecewise linear through
    cfg.hard_term_schedule ((it, mult), ...), constant outside; 1 without a schedule. A pure function of cfg and the
    iteration counter, so every DDP rank (and a resumed / rolled-back run at the same iteration) gets the same value."""
    pts = sorted((int(a), float(b)) for a, b in self.cfg.hard_term_schedule)
    if not pts:
      return 1.0
    if it <= pts[0][0]:
      return pts[0][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
      if it <= x1:
        return y0 + (y1 - y0) * (it - x0) / max(x1 - x0, 1)
    return pts[-1][1]

  def _hard_scalars(self) -> dict:
    """Curriculum scalars (master 4314a50): multiplier and the hard-set episodes since the last log, pooled over ranks
    (data parallel: the sums of _dist_log_scalars) or this process's episodes."""
    scal = {"hard/term_mult": float(self.cmd.hard_mult)}
    if self.dist.enabled:
      fail, ln, n = self._hard_pooled
    else:
      fail, ln, n = float(np.sum(self.stats["hard_ep_failed"])), float(np.sum(self.stats["hard_ep_len"])), float(len(self.stats["hard_ep_failed"]))
    if n > 0:
      scal["hard/ep_failed_frac"] = fail / n
      scal["hard/ep_len"] = ln / n
    return scal

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

  def _set_wm_lr_mult(self, mult: float) -> None:
    """World-model learning rate = schedule x lr_scale x mult (policy_delay boost)."""
    sch = self.sched[0]
    sch.base_lrs = [b * self.lr_scale * mult for b in self._base_lrs0[0]]
    for grp, b in zip(sch.optimizer.param_groups, sch.base_lrs):
      grp["lr"] = b * sch.lr_lambdas[0](self.it)
    self._print(f"[Trainer] it {self.it}: world-model lr x{mult} -> {sch.optimizer.param_groups[0]['lr']:.2e}", flush=True)

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
    # Forget the replay written since the checkpoint (collapsed policy): windows / imagination starts are drawn
    # by age < size, so size 0 restricts them to fresh data; the train loop refills the warm-up first. v1 kept it,
    # and every reload re-measured the imagined reward on the collapsed states.
    self.buffer.size = 0
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
    self._es_rollback()
    return True

  def evaluate(self) -> dict:
    if self.evaluator is None:
      self.evaluator = Evaluator(
        self.cfg.motion_files, self.cfg.heads, self.device, library_key=self.LIB_KEY, eval_clips=self.eval_clip_ids,
        num_clips_total=self.lib.num_clips, max_envs=self.cfg.eval_envs, compile_=self.cfg.compile, cache_dir=self.cfg.cache_dir,
        robot=self.cfg.robot, robot_variant=self.cfg.robot_variant,
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
    groups = {
      "policy": named(self.policy.state_dict()),  # incl. obs/goal normalizers
      "critic": named(self.critic.state_dict()),
      "wm": named(self.wm.state_dict()),  # incl. in/tgt/err normalizers
      "opt_wm": opt(self.opt_wm),
      "opt_pi": opt(self.opt_pi),
      "opt_v": opt(self.opt_v),
      "sampler": [c.bin_G, c.bin_visits, c.bin_weight_hist],  # only checked right after a merge
      # Reset-sampler prior (--source_weights / --name_weights; constant): built per rank from the broadcast file list.
      "sampler_prior": [c.bin_prior if c.bin_prior is not None else "uniform"],
      "counters": [self.it, self.env_steps],
      # The packed library every rank mapped (cache dir + clips.json + bin layout; an extended cache included).
      "library": [self.lib.fingerprint],
    }
    if self.cfg.hard_names:  # curriculum (cfg-dependent key): hard set (from the clip list) + current multiplier
      groups["curriculum"] = [c.clip_hard, float(c.hard_mult)]
    return groups

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
        **({"hard_fail": float(np.sum(st["hard_ep_failed"])), "hard_len": float(np.sum(st["hard_ep_len"])),
            "hard_n": float(len(st["hard_ep_failed"]))} if self._hard_on else {}),
        **self.watch.sums(),  # always-on failure watch (fixed keys): counts of all ranks
      },
      "sum",
    )
    self._hard_pooled = (sums["hard_fail"], sums["hard_len"], sums["hard_n"]) if self._hard_on else None
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
    scal.update(self.watch.scalars(sums))  # pooled over ranks: the same values on every rank
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
    policy_from = self.it + cfg.policy_delay  # WM-only fitting before this iteration
    step_from = policy_from + cfg.policy_freeze  # critic-only (imagination without policy steps) before this one
    self.wd_quiet_until = max(self.wd_quiet_until, step_from + cfg.wd_warmup)
    if self.it < policy_from and cfg.wm_delay_lr != 1.0:
      self._set_wm_lr_mult(cfg.wm_delay_lr)
    t_log = time.time()
    steps_at_log = self.env_steps
    agg: dict[str, list[float]] = {}
    sync_every = cfg.sampler_sync_interval or cfg.log_interval
    if dctx.enabled:
      self._dist_start()
    if self._es_on and self.it > 0 and self.it % cfg.eval_interval == 0:
      # Resumed at an eval slot (warm start): evaluate the loaded weights before the first update, so the early-stop
      # history and the best eval start from them (else the first evals after the resume are the baseline, and
      # model_best_eval.pt only ever competes among those). Depends on cfg / the iteration only -> every rank joins.
      ev = self._run_eval() if cfg.eval_enabled and self.main else None
      if dctx.enabled:
        dctx.barrier()
      self._early_stop_step(ev, False)  # first eval of the history: never a stop
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
      if self.it == policy_from and cfg.policy_delay and cfg.wm_delay_lr != 1.0:
        self._set_wm_lr_mult(1.0)
      skips0 = sum(self.guard_skips.values())
      m_wm = self.update_world_model()
      t2 = time.time()
      if self.it < policy_from:  # policy_delay: world-model-only fitting
        m_pi = {}
      else:
        m_pi = self.update_policy() if self.it >= step_from else self.update_policy(step_policy=False)
      t3 = time.time()
      for s in self.sched:
        s.step()
      self.it += 1
      if m_pi and self._watchdog(m_pi, sum(self.guard_skips.values()) - skips0):  # no policy step yet: no watchdog
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
      # eval slots too: model_best_eval.pt (saved by rank 0 at an eval) must hold the merged sampler statistics
      if dctx.enabled and (self.it % sync_every == 0 or ckpt_slot or check_slot or eval_slot):
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
          scal.update(self.watch.scalars(self.watch.sums()))
        scal.update(self._guard_scalars())
        hard_txt = ""
        if self._hard_on:  # hard-clip curriculum (master 4314a50); keys depend on cfg only
          scal.update(self._hard_scalars())
          hard_txt = (f" | hard x{self.cmd.hard_mult:.2f} fail {scal.get('hard/ep_failed_frac', float('nan')):.2f} "
                      f"len {scal.get('hard/ep_len', float('nan')):.0f}")
        if self.main:
          for k, v in scal.items():
            self.writer.add_scalar(k, v, self.it)
          print(
            f"[it {self.it:7d} | {scal['train/wall_h']:5.2f} h] wm {scal.get('wm/loss', 0):.4f} (e1 {scal.get('wm/err_step1', 0):.3f}, e{cfg.wm.k} {scal.get(f'wm/err_step{cfg.wm.k}', 0):.3f}) "
            f"| imag_ret {scal.get('pi/imagined_return', 0):.4f} rew {scal.get('pi/imagined_reward_norm', 0):.3f} surv {scal.get('pi/survival_H', 0):.3f} | "
            f"ep_ret {scal.get('train/ep_return_norm', 0):.3f} len {scal.get('train/ep_len', 0):.0f} fail {scal.get('train/ep_failed_frac', 0):.2f}"
            + (f" (ee {scal.get('train/ep_failed_ee_frac', 0):.2f})" if self._ee_on else "") + hard_txt + " | "
            f"{scal['time/iter']*1000:.0f} ms/it (c {scal['time/collect']*1000:.0f} wm {scal['time/wm']*1000:.0f} pi {scal['time/policy']*1000:.0f} ref {scal.get('time/ref_gather', 0)*1000:.0f}) fps {fps:.0f}"
            + (f" | max/rank {scal['dist/iter_max']*1000:.0f} ms" if dctx.enabled else "")
            + (f" | guard skips {sum(self.guard_skips.values())} nonfinite its {self.guard_nf_iters}" if self.guard_nf_iters else "")
            + (f" | ori>1 {scal['train/would_fail_ori']:.2f}" if "train/would_fail_ori" in scal else ""),
            flush=True,
          )
        agg = {}
        for k in self.stats:
          self.stats[k] = []
        self.watch.clear()
        t_log, steps_at_log = now, self.env_steps

      if dctx.enabled and check_slot:
        self._check_consistency(f"it {self.it}")
      if ckpt_slot:
        p_ck = self.save(str(self.it)) if self.main else self._ckpt_path(str(self.it))  # all ranks track the path
        if self._healthy():
          self.wd_healthy_ckpt = p_ck
      ev = self._run_eval() if cfg.eval_enabled and eval_slot and self.main else None  # rank 0's summary (early stop)
      if dctx.enabled and (ckpt_slot or eval_slot):
        dctx.barrier()  # the other ranks wait (CPU barrier) while rank 0 saves / evaluates
      # Early stop (opt-in): _es_on and eval_slot depend on cfg / the iteration only -> every rank takes part.
      if self._es_on and eval_slot and self._early_stop_step(ev, ckpt_slot):
        break
    if dctx.enabled:
      self._merge_sampler()
      self._check_consistency("final")
    if self.main:
      self.save("final")
    if dctx.enabled:
      dctx.barrier()

  def _run_eval(self) -> dict | None:
    """Rank 0: one in-training eval (TB eval/*, the [eval] line, eval.jsonl) -> its summary for the early stop, or
    None after an error (eval switched off for the rest of the run)."""
    cfg = self.cfg
    te = time.time()
    try:
      res = self.evaluate()
      for k, v in res["scalars"].items():
        self.writer.add_scalar(f"eval/{k}", v, self.it)
      self.writer.add_scalar("time/eval", time.time() - te, self.it)
      sc = res["scalars"]
      ori = f" (pelvis angle > 1 rad: {int(sc['failed_ori'])}, failed or that: {int(sc['failed_with_ori'])})" if "failed_ori" in sc else ""
      print(f"[eval it {self.it}] return {sc['return_mean']:.4f} min {sc['return_min']:.3f} failed {int(sc['failed'])}/{res['num_clips']}{ori} ({time.time()-te:.0f}s)", flush=True)
      with open(os.path.join(cfg.log_dir, "eval.jsonl"), "a") as f:
        f.write(json.dumps({"it": self.it, "env_steps": self.env_steps, "wall_s": time.time() - self._t_start, **res["scalars"], "failed_clips": res["failed_clips"]}) + "\n")
      self._eval_errors = 0
      return {"failed": float(res["scalars"]["failed"]), "return_mean": float(res["scalars"]["return_mean"]), "num_clips": int(res["num_clips"])}
    except Exception as e:  # noqa: BLE001
      import traceback

      traceback.print_exc()
      # A transient failure (e.g. a CUDA OOM next to another job) must not end evaluation -- and with it the early stop --
      # silently for the rest of the run: off only after EVAL_MAX_ERRORS failures in a row, and then said loudly.
      self._eval_errors = getattr(self, "_eval_errors", 0) + 1
      if self._eval_errors >= EVAL_MAX_ERRORS:
        cfg.eval_enabled = False
        print(f"[eval] failed {self._eval_errors} times in a row ({e!r}): in-training evaluation OFF for the rest of the run"
              + (" -- the early stop is inactive from here" if self._es_on else ""), flush=True)
      else:
        print(f"[eval] failed ({self._eval_errors} in a row, off after {EVAL_MAX_ERRORS}): {e!r}", flush=True)
      return None

  # ---------------------------------------------------------------------------
  # Early stop on the in-training benchmark (opt-in, cfg.es_eval_fail_ratio)
  # ---------------------------------------------------------------------------

  def _es_rollback(self) -> None:
    """After a watchdog rollback to self.it: the evals of the abandoned branch (it > self.it) leave the early-stop
    history -- they would carry their streak into the restored branch (one bad eval there would then stop the run) and
    raise the median of later verdicts; the streak is the one of the last kept eval. A best eval of that branch stays
    the best (model_best_eval.pt holds its weights) but no longer names model_<it>.pt, which the restored branch
    rewrites when it gets there again. Rank-identical inputs -> the same state on every rank; no-op with es off."""
    if not self._es_on:
      return
    dropped = [h["it"] for h in self.es_hist if h["it"] > self.it]
    if not dropped:
      return
    self.es_hist = [h for h in self.es_hist if h["it"] <= self.it]
    self.es_streak = self.es_hist[-1]["streak"] if self.es_hist else 0
    b = self.es_best
    if b is not None and b["it"] > self.it and b["regular_ckpt"] is not None:
      b["regular_ckpt"] = None
      if self.main:
        with open(os.path.join(self.cfg.log_dir, "best_eval.json"), "w") as f:
          json.dump(b, f, indent=1)
    if self.main:
      print(f"[es] rollback to it {self.it}: eval(s) at {dropped} of the abandoned branch left out of the early-stop history; "
            f"streak {self.es_streak}/{self.cfg.es_patience}", flush=True)

  def _early_stop_step(self, ev_local: dict | None, ckpt_saved: bool) -> bool:
    """After an eval slot: rank 0's eval summary (None if it did not evaluate: eval switched off after an error) goes
    to every rank, which all update the same history / best / streak and return the same verdict (True = stop now).
    Every new best eval (fewest failures, ties: higher return) is saved as model_best_eval.pt (its own file: a regular
    model_<it>.pt is overwritten when a watchdog rollback re-reaches <it>); ckpt_saved: a regular checkpoint of the
    same weights was written at this iteration too."""
    cfg, dctx = self.cfg, self.dist
    ev = dctx.broadcast_object(ev_local) if dctx.enabled else ev_local
    if ev is None:
      return False
    failed, ret = float(ev["failed"]), float(ev["return_mean"])
    prev = [h["failed"] for h in self.es_hist]
    bad, thr = early_stop_verdict(prev, failed, cfg.es_eval_fail_ratio, cfg.es_median_floor)
    self.es_streak = self.es_streak + 1 if bad else 0
    self.es_hist.append({"it": self.it, "failed": failed, "return_mean": ret, "threshold": thr, "bad": bad, "streak": self.es_streak})
    b = self.es_best
    if b is None or (failed, -ret) < (b["failed"], -b["return_mean"]):
      if self.main:
        self.save("best_eval")
      if dctx.enabled:
        dctx.barrier()  # the other ranks wait (CPU barrier) while rank 0 saves
      self.es_best = {"it": self.it, "failed": failed, "return_mean": ret, "ckpt": self._ckpt_path("best_eval"),
                      "regular_ckpt": self._ckpt_path(str(self.it)) if ckpt_saved else None}
      if self.main:
        with open(os.path.join(cfg.log_dir, "best_eval.json"), "w") as f:
          json.dump(self.es_best, f, indent=1)
    if self.main:
      self.writer.add_scalar("es/streak", self.es_streak, self.it)
      self.writer.add_scalar("es/best_failed", self.es_best["failed"], self.it)
      if thr is not None:
        self.writer.add_scalar("es/threshold", thr, self.it)
      verdict = "warming up" if thr is None else f"threshold {thr:.1f} -> {'BAD' if bad else 'ok'}, streak {self.es_streak}/{cfg.es_patience}"
      print(f"[es] it {self.it}: eval failed {failed:.0f} ({verdict}); best it {self.es_best['it']} failed {self.es_best['failed']:.0f}", flush=True)
    if self.es_streak < cfg.es_patience:
      return False
    self.es_stop = {
      "iteration": self.it, "reason": f"{self.es_streak} consecutive evals with failed >= {cfg.es_eval_fail_ratio:g} x median of the previous ones",
      "es_eval_fail_ratio": cfg.es_eval_fail_ratio, "es_patience": cfg.es_patience, "best_eval": self.es_best,
      "final_ckpt": self._ckpt_path("final"), "history": self.es_hist,
    }
    if self.main:
      print(f"[earlystop] it {self.it}: eval failed {failed:.0f} >= {thr:.1f} ({cfg.es_eval_fail_ratio:g} x median of {len(prev)} previous evals) "
            f"for {self.es_streak} evals in a row; best eval it {self.es_best['it']} (failed {self.es_best['failed']:.0f}): "
            f"{self.es_best['ckpt']}; saving model_final and stopping", flush=True)
      with open(os.path.join(cfg.log_dir, "earlystop.json"), "w") as f:
        json.dump(self.es_stop, f, indent=1)
    return True

  def run(self) -> None:
    self._t_start = time.time()
    self.train()
