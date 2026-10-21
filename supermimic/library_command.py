"""Multi-clip motion command with return-based reset sampling (Appendix D.1).

Subclasses mjlab's ``MotionCommand`` so the stock BeyondMimic reward /
termination functions keep working, but:
  * ``self.motion`` is a ``MotionLibrary`` (all clips on one global timeline),
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
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, sample_uniform

from .motions import MotionLibrary

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


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
    self.library = MotionLibrary(
      list(cfg.motion_files), tuple(self.robot.joint_names), cfg.body_names, self.device, cfg.bin_seconds
    )
    self.motion = self.library  # duck-typed MotionLoader
    print(f"[LibraryMotionCommand] {self.library.summary()}")
    E = self.num_envs
    self.time_steps = torch.zeros(E, dtype=torch.long, device=self.device)
    self.clip_end_env = torch.full((E,), int(self.library.clip_end[0].item()), dtype=torch.long, device=self.device)
    nb = len(cfg.body_names)
    self.body_pos_relative_w = torch.zeros(E, nb, 3, device=self.device)
    self.body_quat_relative_w = torch.zeros(E, nb, 4, device=self.device)
    self.body_quat_relative_w[:, :, 0] = 1.0
    # Return-based sampling state.
    NB = self.library.num_bins
    self.bin_count = NB
    self.bin_G = torch.zeros(NB, device=self.device)
    self.bin_visits = torch.zeros(NB, device=self.device)
    self.bin_weight_hist = torch.full((NB,), 1.0 / NB, device=self.device)
    for k in (
      "error_anchor_pos", "error_anchor_rot", "error_anchor_lin_vel", "error_anchor_ang_vel", "error_body_pos",
      "error_body_rot", "error_body_lin_vel", "error_body_ang_vel", "error_joint_pos", "error_joint_vel",
      "sampling_entropy", "sampling_top1_prob", "sampling_top1_bin",
    ):
      self.metrics[k] = torch.zeros(E, device=self.device)
    self._ghost_model = None
    self._ghost_color = np.array(cfg.viz.ghost_color, dtype=np.float32)
    self._pending_forward = False
    self._last_probs = None

  # ---- return-based reset sampling -----------------------------------------

  @torch.no_grad()
  def update_bin_returns(self, bin_ids: torch.Tensor, returns: torch.Tensor) -> None:
    """EMA-update the per-bin discounted return estimate with new samples."""
    if bin_ids.numel() == 0:
      return
    NB = self.library.num_bins
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
    uniform = torch.full((NB,), 1.0 / NB, device=self.device)
    G = torch.where(self.bin_visits > 0, self.bin_G, torch.full_like(self.bin_G, c.g_max * 0.5))
    soft = torch.softmax(-c.softmax_temp * G / c.g_max, dim=0)
    hist = self.bin_weight_hist / self.bin_weight_hist.sum()
    probs = c.mix_uniform * uniform + c.mix_softmax * soft + c.mix_history * hist
    probs = probs / probs.sum()
    # Track the past softmax weight so hard bins keep some practice once learned.
    self.bin_weight_hist = (1 - c.weight_ema) * self.bin_weight_hist + c.weight_ema * soft
    self._last_probs = probs
    return probs

  @torch.no_grad()
  def sample_frames(self, env_ids: torch.Tensor) -> torch.Tensor:
    n = len(env_ids)
    lib = self.library
    mode = self.cfg.sampling_mode
    if mode == "start":
      clip = env_ids % lib.num_clips
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
    # Need at least one step before the clip ends (frame is incremented after reset).
    f = torch.minimum(f, lib.clip_end_of_frame[f] - 2)
    return f

  # ---- overrides -----------------------------------------------------------

  def _resample_command(self, env_ids: torch.Tensor):
    frames = self.sample_frames(env_ids)
    self.time_steps[env_ids] = frames
    self.clip_end_env[env_ids] = self.library.clip_end_of_frame[frames]

    root_pos = self.body_pos_w[env_ids, 0].clone()
    root_ori = self.body_quat_w[env_ids, 0].clone()
    root_lin_vel = self.body_lin_vel_w[env_ids, 0].clone()
    root_ang_vel = self.body_ang_vel_w[env_ids, 0].clone()
    n = len(env_ids)
    keys = ["x", "y", "z", "roll", "pitch", "yaw"]
    pr = torch.tensor([self.cfg.pose_range.get(k, (0.0, 0.0)) for k in keys], device=self.device)
    rs = sample_uniform(pr[:, 0], pr[:, 1], (n, 6), device=self.device)
    root_pos += rs[:, 0:3]
    root_ori = quat_mul(quat_from_euler_xyz(rs[:, 3], rs[:, 4], rs[:, 5]), root_ori)
    vr = torch.tensor([self.cfg.velocity_range.get(k, (0.0, 0.0)) for k in keys], device=self.device)
    rs = sample_uniform(vr[:, 0], vr[:, 1], (n, 6), device=self.device)
    root_lin_vel += rs[:, :3]
    root_ang_vel += rs[:, 3:]
    joint_pos = self.joint_pos[env_ids].clone()
    joint_vel = self.joint_vel[env_ids]
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
    if self._pending_forward:
      self._pending_forward = False
      self._env.sim.forward()
    self.update_relative_body_poses()


def motion_clip_end(env: "ManagerBasedRlEnv", command_name: str) -> torch.Tensor:
  """Time-out when the reference clip has been played to its last frame."""
  cmd = env.command_manager.get_term(command_name)
  return cmd.time_steps >= cmd.clip_end_env - 1  # type: ignore[attr-defined]


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
  weight_ema: float = 0.01
  sampling_mode: Literal["return", "uniform", "start"] = "return"  # type: ignore[assignment]

  def build(self, env: "ManagerBasedRlEnv") -> LibraryMotionCommand:
    return LibraryMotionCommand(self, env)
