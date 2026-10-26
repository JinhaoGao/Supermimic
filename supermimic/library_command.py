"""Multi-clip motion command with return-based reset sampling (Appendix D.1).

Subclasses mjlab's ``MotionCommand`` so the stock BeyondMimic reward /
termination functions keep working, but:
  * the clips live in CPU RAM (``MotionLibrary``); only the current frame of
    every env is cached on the GPU (refreshed once per step),
  * resets pick a frame by the paper's return-based bin sampling,
  * the time step never wraps / teleports: reaching the clip end is a
    time-out termination handled by ``motion_clip_end``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

from mjlab.managers import CommandTerm
from mjlab.tasks.tracking.mdp.commands import MotionCommand, MotionCommandCfg
from mjlab.tasks.tracking.mdp.rewards import _get_body_indexes
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_from_euler_xyz, quat_mul, sample_uniform

from .math_utils import quat_angle
from .motions import MotionLibrary

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# Libraries shared between envs of one process (train env + eval env).
LIBRARY_REGISTRY: dict[str, MotionLibrary] = {}


class LibraryMotionCommand(MotionCommand):
  cfg: "LibraryMotionCommandCfg"

  def __init__(self, cfg: "LibraryMotionCommandCfg", env: "ManagerBasedRlEnv"):
    CommandTerm.__init__(self, cfg, env)
    self.robot = env.scene[cfg.entity_name]
    self.robot_anchor_body_index = self.robot.body_names.index(cfg.anchor_body_name)
    self.motion_anchor_body_index = cfg.body_names.index(cfg.anchor_body_name)
    self.body_indexes = torch.tensor(
      self.robot.find_bodies(cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
    )
    key = cfg.library_key
    if key and key in LIBRARY_REGISTRY:
      self.library = LIBRARY_REGISTRY[key]
      assert self.library.joint_names == tuple(self.robot.joint_names) and self.library.body_names == tuple(cfg.body_names)
    else:
      self.library = MotionLibrary(
        list(cfg.motion_files), tuple(self.robot.joint_names), cfg.body_names, self.device, cfg.bin_seconds,
        cache_dir=cfg.cache_dir or None, num_workers=cfg.num_workers,
      )
      if key:
        LIBRARY_REGISTRY[key] = self.library
    self.motion = self.library
    print(f"[LibraryMotionCommand] {self.library.summary()}")
    E = self.num_envs
    lib = self.library
    self.time_steps = torch.zeros(E, dtype=torch.long, device=self.device)
    self.clip_end_env = torch.full((E,), int(lib.clip_end[0].item()), dtype=torch.long, device=self.device)
    nb = len(cfg.body_names)
    J = lib.nj
    self._cur = dict(
      joint_pos=torch.zeros(E, J, device=self.device), joint_vel=torch.zeros(E, J, device=self.device),
      body_pos=torch.zeros(E, nb, 3, device=self.device), body_quat=torch.zeros(E, nb, 4, device=self.device),
      body_lin_vel=torch.zeros(E, nb, 3, device=self.device), body_ang_vel=torch.zeros(E, nb, 3, device=self.device),
    )
    self._cur["body_quat"][:, :, 0] = 1.0
    self.body_pos_relative_w = torch.zeros(E, nb, 3, device=self.device)
    self.body_quat_relative_w = torch.zeros(E, nb, 4, device=self.device)
    self.body_quat_relative_w[:, :, 0] = 1.0
    # Return-based sampling state.
    NB = lib.num_bins
    self.bin_count = NB
    self.bin_G = torch.zeros(NB, device=self.device)
    self.bin_visits = torch.zeros(NB, device=self.device)
    self.bin_weight_hist = torch.full((NB,), 1.0 / NB, device=self.device)
    # Source prior for the uniform share of the reset distribution (v2): clip "<robot>__<source>_..." gets the
    # weight of its longest matching source prefix (default 1). The team's own sources are <1% of the bins.
    self.bin_prior: torch.Tensor | None = None
    if cfg.source_weights or cfg.name_weights:
      kws = sorted(((str(k), float(v)) for k, v in cfg.source_weights), key=lambda kv: -len(kv[0]))
      nws = [(str(k).lower(), float(v)) for k, v in cfg.name_weights]
      def src_weight(name: str) -> tuple[str, float]:
        src = name.split("__", 1)[-1]
        for k, v in kws:
          if src == k or src.startswith(k + "_"):
            return k, v
        return "", 1.0
      tags = [src_weight(c.name) for c in lib.clips]
      # Name keywords (case-insensitive substring, e.g. crawl / getup): a skill quota across sources (ground work is
      # spread over sonic / motiondecode / hiphi / g1_all). A clip gets the larger of its source and keyword weights.
      name_w = [max([v for k, v in nws if k in c.name.lower()], default=1.0) for c in lib.clips]
      w_clip = torch.tensor([max(w, nw) for (_, w), nw in zip(tags, name_w)], dtype=torch.float32, device=self.device)
      prior = w_clip[lib.bin_clip]
      self.bin_prior = prior / prior.sum()
      tag_of_bin = [tags[i][0] for i in lib.bin_clip.tolist()]
      for k, v in kws:
        mask = torch.tensor([t == k for t in tag_of_bin], device=self.device)
        print(f"[LibraryMotionCommand] source {k!r} x{v:g}: {mask.float().mean().item() * 100:.2f}% of bins -> "
              f"{self.bin_prior[mask].sum().item() * 100:.2f}% of the uniform share")
      if nws:
        hit = torch.tensor([nw > 1.0 for nw in name_w], device=self.device)[lib.bin_clip]
        print(f"[LibraryMotionCommand] name keywords {nws}: {int(sum(nw > 1.0 for nw in name_w))} clips, "
              f"{hit.float().mean().item() * 100:.2f}% of bins -> {self.bin_prior[hit].sum().item() * 100:.2f}% of the uniform share")
    # Cluster-balanced prior (allmimic, see bin_clusters.py): multiplies the source / keyword prior of the uniform share.
    if cfg.cluster_k > 0:
      from .bin_clusters import bin_cluster_prior

      cw = bin_cluster_prior(lib, cfg.cluster_k, cfg.cluster_alpha, self.device, log=print)
      base = self.bin_prior if self.bin_prior is not None else torch.full((NB,), 1.0 / NB, device=self.device)
      self.bin_prior = base * cw / (base * cw).sum()
    # Hard-clip termination curriculum (v2 r13): clips whose name contains a hard_names keyword get every
    # termination threshold (env terms below + the trainer's soft head) multiplied by hard_mult, which the
    # trainer anneals back to 1; other clips keep 1. Episodes keep the factor of the clip they reset into.
    hn = [str(k).lower() for k in cfg.hard_names]
    self.clip_hard = torch.tensor([any(k in c.name.lower() for k in hn) for c in lib.clips], dtype=torch.bool, device=self.device)
    self.env_hard = torch.zeros(E, dtype=torch.bool, device=self.device)
    self.term_scale_env = torch.ones(E, device=self.device)
    self.hard_mult = 1.0
    if hn:
      hb = self.clip_hard[lib.bin_clip]
      print(f"[LibraryMotionCommand] hard set {hn}: {int(self.clip_hard.sum())} clips, {hb.float().mean().item() * 100:.2f}% of bins")
    # Sequential clip assignment for "start" (evaluation) mode.
    ec = cfg.eval_clips if cfg.eval_clips else tuple(range(lib.num_clips))
    self.eval_clips = torch.tensor(ec, dtype=torch.long, device=self.device)
    self._start_counter = 0
    self.env_clip = torch.zeros(E, dtype=torch.long, device=self.device)  # clip index currently played by each env
    for k in (
      "error_anchor_pos", "error_anchor_rot", "error_anchor_lin_vel", "error_anchor_ang_vel", "error_body_pos",
      "error_body_rot", "error_body_lin_vel", "error_body_ang_vel", "error_joint_pos", "error_joint_vel",
      "sampling_entropy", "sampling_top1_prob", "sampling_top1_bin",
    ):
      self.metrics[k] = torch.zeros(E, device=self.device)
    self._ghost_model = None
    self._ghost_color = np.array(cfg.viz.ghost_color, dtype=np.float32)
    self._pending_forward = False

  # ---- current-frame cache ---------------------------------------------------

  @torch.no_grad()
  def refresh_current(self, env_ids: torch.Tensor | None = None) -> None:
    if env_ids is None:
      g = self.library.gather(self.time_steps)
      for k in self._cur:
        self._cur[k] = g[k]
    elif env_ids.numel() > 0:
      g = self.library.gather(self.time_steps[env_ids])
      for k in self._cur:
        self._cur[k][env_ids] = g[k]

  def current_ref(self) -> dict:
    """Reference fields at the current time step (env-local frame, no origins)."""
    return dict(self._cur)

  # Base-class property overrides (mjlab reward / termination functions read these).
  @property
  def joint_pos(self) -> torch.Tensor:
    return self._cur["joint_pos"]

  @property
  def joint_vel(self) -> torch.Tensor:
    return self._cur["joint_vel"]

  @property
  def body_pos_w(self) -> torch.Tensor:
    return self._cur["body_pos"] + self._env.scene.env_origins[:, None, :]

  @property
  def body_quat_w(self) -> torch.Tensor:
    return self._cur["body_quat"]

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    return self._cur["body_lin_vel"]

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    return self._cur["body_ang_vel"]

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    return self._cur["body_pos"][:, self.motion_anchor_body_index] + self._env.scene.env_origins

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    return self._cur["body_quat"][:, self.motion_anchor_body_index]

  @property
  def anchor_lin_vel_w(self) -> torch.Tensor:
    return self._cur["body_lin_vel"][:, self.motion_anchor_body_index]

  @property
  def anchor_ang_vel_w(self) -> torch.Tensor:
    return self._cur["body_ang_vel"][:, self.motion_anchor_body_index]

  def set_hard_mult(self, m: float) -> None:
    self.hard_mult = float(m)
    self.term_scale_env = 1.0 + (self.hard_mult - 1.0) * self.env_hard.float()

  # ---- return-based reset sampling -----------------------------------------

  @torch.no_grad()
  def update_bin_returns(self, bin_ids: torch.Tensor, returns: torch.Tensor) -> None:
    finite = torch.isfinite(returns)
    if not bool(finite.all()):
      bin_ids, returns = bin_ids[finite], returns[finite]
    if self.cfg.clamp_returns:
      returns = returns.clamp(0.0, self.cfg.g_max)
    if bin_ids.numel() == 0:
      return
    NB = self.library.num_bins
    bin_ids = bin_ids.long()
    s = torch.zeros(NB, device=self.device).index_add_(0, bin_ids, returns)
    c = torch.zeros(NB, device=self.device).index_add_(0, bin_ids, torch.ones_like(returns))
    visited = c > 0
    mean = s[visited] / c[visited]
    a = self.cfg.return_ema
    first = self.bin_visits[visited] == 0
    self.bin_G[visited] = torch.where(first, mean, (1 - a) * self.bin_G[visited] + a * mean)
    self.bin_visits[visited] += c[visited]

  @torch.no_grad()
  def sampling_probabilities(self) -> torch.Tensor:
    c = self.cfg
    NB = self.library.num_bins
    uniform = self.bin_prior if self.bin_prior is not None else torch.full((NB,), 1.0 / NB, device=self.device)
    bin_G = self.bin_G.clamp(0.0, c.g_max) if c.clamp_returns else self.bin_G
    G = torch.where(self.bin_visits > 0, bin_G, torch.full_like(self.bin_G, c.g_max * 0.5))
    soft = torch.softmax(-c.softmax_temp * G / c.g_max, dim=0)
    if c.softmax_cap > 0:
      soft = cap_simplex(soft, c.softmax_cap / NB)
    hist = self.bin_weight_hist / self.bin_weight_hist.sum()
    probs = c.mix_uniform * uniform + c.mix_softmax * soft + c.mix_history * hist
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0) + 1e-12
    probs = probs / probs.sum()
    self.bin_weight_hist = (1 - c.weight_ema) * self.bin_weight_hist + c.weight_ema * soft
    return probs

  @torch.no_grad()
  def sample_frames(self, env_ids: torch.Tensor) -> torch.Tensor:
    n = len(env_ids)
    lib = self.library
    mode = self.cfg.sampling_mode
    if mode == "start":
      k = torch.arange(self._start_counter, self._start_counter + n, device=self.device) % self.eval_clips.numel()
      self._start_counter += n
      clip = self.eval_clips[k]
      self.env_clip[env_ids] = clip
      return lib.clip_start[clip]
    if mode == "uniform":
      f = torch.randint(0, lib.time_step_total, (n,), device=self.device)
    else:
      probs = self.sampling_probabilities()
      b = torch.multinomial(probs, n, replacement=True)
      span = (lib.bin_end[b] - lib.bin_start[b]).float()
      f = lib.bin_start[b] + (torch.rand(n, device=self.device) * span).long()
      H = -(probs * (probs + 1e-12).log()).sum()
      self.metrics["sampling_entropy"][:] = H / np.log(max(lib.num_bins, 2))
      pmax, imax = probs.max(dim=0)
      self.metrics["sampling_top1_prob"][:] = pmax
      self.metrics["sampling_top1_bin"][:] = imax.float() / lib.num_bins
    f = torch.minimum(f, lib.clip_end_of_frame[f].long() - 2)
    return f

  # ---- overrides -----------------------------------------------------------

  def _resample_command(self, env_ids: torch.Tensor):
    frames = self.sample_frames(env_ids)
    self.time_steps[env_ids] = frames
    hard = self.clip_hard[self.library.clip_of_frame(frames)]
    self.env_hard[env_ids] = hard
    self.term_scale_env[env_ids] = 1.0 + (self.hard_mult - 1.0) * hard.float()
    self.clip_end_env[env_ids] = self.library.clip_end_of_frame[frames].long()
    rs = self.library.root_state(frames)
    origins = self._env.scene.env_origins[env_ids]
    root_pos = rs["root_pos"] + origins
    root_ori = rs["root_quat"].clone()
    root_lin_vel = rs["root_lin_vel"].clone()
    root_ang_vel = rs["root_ang_vel"].clone()
    n = len(env_ids)
    keys = ["x", "y", "z", "roll", "pitch", "yaw"]
    pr = torch.tensor([self.cfg.pose_range.get(k, (0.0, 0.0)) for k in keys], device=self.device)
    r1 = sample_uniform(pr[:, 0], pr[:, 1], (n, 6), device=self.device)
    root_pos = root_pos + r1[:, 0:3]
    root_ori = quat_mul(quat_from_euler_xyz(r1[:, 3], r1[:, 4], r1[:, 5]), root_ori)
    vr = torch.tensor([self.cfg.velocity_range.get(k, (0.0, 0.0)) for k in keys], device=self.device)
    r2 = sample_uniform(vr[:, 0], vr[:, 1], (n, 6), device=self.device)
    root_lin_vel = root_lin_vel + r2[:, :3]
    root_ang_vel = root_ang_vel + r2[:, 3:]
    joint_pos = rs["joint_pos"].clone()
    joint_vel = rs["joint_vel"]
    lo, hi = self.cfg.joint_position_range
    if hi > lo:
      joint_pos += sample_uniform(lower=lo, upper=hi, size=joint_pos.shape, device=joint_pos.device)  # type: ignore
    self._write_reference_state_to_sim(env_ids, root_pos, root_ori, root_lin_vel, root_ang_vel, joint_pos, joint_vel)
    self._pending_forward = True

  def _update_command(self, env_ids: torch.Tensor | None = None):
    if env_ids is None:
      self.time_steps += 1
    else:
      self.time_steps[env_ids] += 1
    self.time_steps = torch.minimum(self.time_steps, self.clip_end_env - 1)
    self.refresh_current(env_ids)
    if self._pending_forward:
      self._pending_forward = False
      self._env.sim.forward()
    self.update_relative_body_poses()


@torch.no_grad()
def cap_simplex(p: torch.Tensor, cap: float) -> torch.Tensor:
  """Water-filling cap of a distribution: min(cap, s * p) with s such that the result sums to 1 (the mass above the cap
  goes to the uncapped entries in proportion; needs cap * len(p) >= 1). Exact, via one sort (float64)."""
  if float(p.max()) <= cap * float(p.sum()):
    return p
  q, idx = torch.sort(p.double(), descending=True)
  q = q / q.sum()
  k = torch.arange(q.numel(), device=q.device, dtype=torch.float64)  # entries before position k are capped
  rest = (1.0 - (torch.cumsum(q, 0) - q)).clamp(min=1e-300)  # mass of the entries from k on
  s = (1.0 - k * cap) / rest
  ok = (s * q <= cap) & (k * cap < 1.0)  # first such k: the capped set is exactly the k largest entries
  k0 = int(torch.argmax(ok.to(torch.uint8)).item())
  assert bool(ok[k0]), "cap_simplex: cap * len(p) must be >= 1"
  out = torch.empty_like(q)
  out[:k0] = cap
  out[k0:] = s[k0] * q[k0:]
  res = torch.empty_like(out).scatter_(0, idx, out)
  return (res / res.sum()).to(p.dtype)


def motion_clip_end(env: "ManagerBasedRlEnv", command_name: str) -> torch.Tensor:
  """Time-out when the reference clip has been played to its last frame."""
  cmd = env.command_manager.get_term(command_name)
  return cmd.time_steps >= cmd.clip_end_env - 1  # type: ignore[attr-defined]


# mjlab's anchor / end-effector terminations with the per-env threshold multiplier of the hard-clip curriculum
# (LibraryMotionCommand.term_scale_env, 1 outside the hard set); swapped in by make_env_cfg when it is used.


def scaled_anchor_pos_z(env: "ManagerBasedRlEnv", command_name: str, threshold: float) -> torch.Tensor:
  cmd = env.command_manager.get_term(command_name)
  return (cmd.anchor_pos_w[:, -1] - cmd.robot_anchor_pos_w[:, -1]).abs() > threshold * cmd.term_scale_env  # type: ignore[attr-defined]


def scaled_anchor_ori(env: "ManagerBasedRlEnv", asset_cfg, command_name: str, threshold: float) -> torch.Tensor:
  g = env.scene[asset_cfg.name].data.gravity_vec_w
  cmd = env.command_manager.get_term(command_name)
  dz = quat_apply_inverse(cmd.anchor_quat_w, g)[:, 2] - quat_apply_inverse(cmd.robot_anchor_quat_w, g)[:, 2]  # type: ignore[attr-defined]
  return dz.abs() > threshold * cmd.term_scale_env  # type: ignore[attr-defined]


def scaled_body_pos_z(env: "ManagerBasedRlEnv", command_name: str, threshold: float, body_names=None) -> torch.Tensor:
  cmd = env.command_manager.get_term(command_name)
  idx = _get_body_indexes(cmd, body_names)
  err = (cmd.body_pos_relative_w[:, idx, -1] - cmd.robot_body_pos_w[:, idx, -1]).abs()  # type: ignore[attr-defined]
  return (err > (threshold * cmd.term_scale_env)[:, None]).any(-1)  # type: ignore[attr-defined]


def bad_root_ori_full(env: "ManagerBasedRlEnv", command_name: str, threshold: float, body_name: str) -> torch.Tensor:
  """Root-orientation termination (HeadsCfg.root_ori_term): full rotation angle (yaw included) between the reference and
  the robot's ``body_name`` (the root: G1 pelvis) > threshold x term_scale_env; the team eval site's root_ori test."""
  cmd = env.command_manager.get_term(command_name)
  ref_q = cmd.body_quat_w[:, cmd.cfg.body_names.index(body_name)]  # type: ignore[attr-defined]
  rob_q = cmd.robot.data.body_link_quat_w[:, cmd.robot.body_names.index(body_name)]  # type: ignore[attr-defined]
  return quat_angle(ref_q, rob_q) > threshold * cmd.term_scale_env  # type: ignore[attr-defined]


def bad_root_pos_z(env: "ManagerBasedRlEnv", command_name: str, threshold: float, body_name: str) -> torch.Tensor:
  """Root-height termination (HeadsCfg.root_z_term): |z_ref - z_robot| of ``body_name`` (the root: G1 pelvis) >
  threshold x term_scale_env; the team eval site's root_pos test (root_pos_mode z). Flat ground: env origins have z = 0."""
  cmd = env.command_manager.get_term(command_name)
  ref_z = cmd.body_pos_w[:, cmd.cfg.body_names.index(body_name), 2]  # type: ignore[attr-defined]
  rob_z = cmd.robot.data.body_link_pos_w[:, cmd.robot.body_names.index(body_name), 2]  # type: ignore[attr-defined]
  return (ref_z - rob_z).abs() > threshold * cmd.term_scale_env  # type: ignore[attr-defined]


@dataclass(kw_only=True)
class LibraryMotionCommandCfg(MotionCommandCfg):
  motion_files: tuple[str, ...] = ()
  motion_file: str = ""  # unused; kept for the base dataclass
  bin_seconds: float = 1.0
  mix_uniform: float = 0.5
  mix_softmax: float = 0.4
  mix_history: float = 0.1
  softmax_temp: float = 25.0
  g_max: float = 10.0
  return_ema: float = 0.05
  # Clamp bin returns (EMA input and sampled G) to [0, g_max]. Off = paper: a clip-end bin whose last reward is
  # negative (joint-limit penalty) gets G << 0 through the time-out bootstrap r_last / (1 - gamma) and takes the whole
  # softmax + history share (sampler/top1_prob 0.5 in every G1 run; found by the bj31 session, branch ddp-bj31 5bb8c0d).
  clamp_returns: bool = False
  weight_ema: float = 0.01
  source_weights: tuple = ()  # ((source prefix, weight), ...): prior of the uniform share, see LibraryMotionCommand
  name_weights: tuple = ()  # ((name keyword, weight), ...): max with the source weight (case-insensitive substring)
  hard_names: tuple = ()  # clip-name keywords of the hard set (termination curriculum, case-insensitive substring)
  # allmimic sampler pieces (10-26): cap every bin of the softmax share at softmax_cap x the uniform share 1/NB
  # (allmimic: 50; 0 = off). Without it about half of all resets went to ~1.6k mostly always-failing bins of 5.4M
  # (G1 r25 208000: 90 % of the softmax mass on G <= 1.1). cluster_k > 0: cluster-balanced prior (bin_clusters.py).
  softmax_cap: float = 0.0
  cluster_k: int = 0
  cluster_alpha: float = 0.3
  sampling_mode: Literal["return", "uniform", "start"] = "return"  # type: ignore[assignment]
  cache_dir: str = ""
  num_workers: int = 32
  library_key: str = ""  # share one in-RAM library between envs of a process
  eval_clips: tuple[int, ...] = ()  # "start" mode cycles through these clip indices (default: all)

  def build(self, env: "ManagerBasedRlEnv") -> LibraryMotionCommand:
    return LibraryMotionCommand(self, env)
