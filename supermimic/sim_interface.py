"""Thin interface between the mjlab env and the SuperMimic learner.

Extracts the minimal state vector and the sim context xi from the simulator,
owns the FK / analytic heads, and keeps the proprioceptive history buffer.
"""

from __future__ import annotations

import torch

from .fk import TorchFK
from .heads import AnalyticHeads, HeadsCfg


class ObsHistory:
  def __init__(self, num_envs: int, hist: int, dim: int, device: str):
    self.buf = torch.zeros(num_envs, hist, dim, device=device)

  def reset(self, env_ids: torch.Tensor, obs: torch.Tensor) -> None:
    """Fill the whole history of ``env_ids`` with their current obs (mjlab semantics)."""
    if env_ids.numel() > 0:
      self.buf[env_ids] = obs[env_ids][:, None, :]

  def push(self, obs: torch.Tensor) -> None:
    self.buf = torch.cat([self.buf[:, 1:], obs[:, None, :]], dim=1)


class SimInterface:
  FRICTION_RANGE = (0.3, 1.2)
  BIAS_RANGE = 0.01
  COM_RANGE = (0.025, 0.05, 0.05)

  def __init__(self, env, heads_cfg: HeadsCfg):
    self.env = env
    self.device = env.device
    self.robot = env.scene["robot"]
    self.cmd = env.command_manager.get_term("motion")
    self.lib = self.cmd.library
    tracked_local = [int(i) for i in self.cmd.body_indexes.tolist()]
    self.fk = TorchFK(env.sim.mj_model, self.robot, self.device, needed_bodies=tracked_local)
    self.heads = AnalyticHeads(
      self.fk,
      self.lib,
      tracked_local,
      int(self.cmd.motion_anchor_body_index),
      self.robot.data.default_joint_pos[0],
      self.robot.data.soft_joint_pos_limits[0],
      heads_cfg,
    )
    self.nj = self.heads.nj
    self.xi_dim = 4 + self.nj
    idx = self.robot.indexing
    foot_local = self.robot.find_geoms(r"^(left|right)_foot[1-7]_collision$")[0]
    self.foot_geom_gid = int(idx.geom_ids[foot_local[0]].item())
    torso_local = self.robot.body_names.index("torso_link")
    self.torso_gid = int(idx.body_ids[torso_local].item())
    self.ipos_nominal = torch.as_tensor(env.sim.mj_model.body_ipos[self.torso_gid], dtype=torch.float32, device=self.device)
    self.env_origins = env.scene.env_origins.to(self.device)
    self.com_scale = torch.tensor(self.COM_RANGE, device=self.device)
    # MuJoCo Warp prints a solver/linesearch overflow warning every step the
    # iteration cap is hit (common with random actions); silence it.
    try:
      env.sim.wp_model.opt.warn_overflow = 0
    except Exception:  # noqa: BLE001
      pass

  # ---- state -----------------------------------------------------------------

  @torch.no_grad()
  def state(self) -> torch.Tensor:
    d = self.robot.data
    pose = d.root_link_pose_w
    vel = d.root_link_vel_w
    p = pose[:, :3] - self.env_origins
    return torch.cat([p, pose[:, 3:7], vel[:, :3], vel[:, 3:6], d.joint_pos, d.joint_vel], dim=-1).clone()

  @torch.no_grad()
  def xi(self) -> torch.Tensor:
    """Scaled sim context [friction(1), encoder bias(J), torso COM offset(3)]."""
    E = self.env.num_envs
    model = self.env.sim.model
    fr = torch.as_tensor(model.geom_friction)
    friction = fr[:, self.foot_geom_gid, 0] if fr.dim() == 3 else fr[self.foot_geom_gid, 0].reshape(1)
    if friction.shape[0] != E:
      friction = friction.expand(E)
    ipos = torch.as_tensor(model.body_ipos)
    com = (ipos[:, self.torso_gid] if ipos.dim() == 3 else ipos[self.torso_gid].reshape(1, 3)) - self.ipos_nominal
    if com.shape[0] != E:
      com = com.expand(E, 3)
    bias = self.robot.data.encoder_bias
    lo, hi = self.FRICTION_RANGE
    f_n = (friction - 0.5 * (lo + hi)) / (0.5 * (hi - lo))
    return torch.cat([f_n[:, None], bias / self.BIAS_RANGE, com / self.com_scale], dim=-1).float().clone()

  @staticmethod
  def bias_from_xi(xi: torch.Tensor, nj: int) -> torch.Tensor:
    return xi[:, 1 : 1 + nj] * SimInterface.BIAS_RANGE
