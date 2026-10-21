"""Analytic observation / goal / reward / termination heads on the minimal state.

State vector layout (float32, D = 13 + 2*J):
  [0:3]   root position (world, env-local frame)
  [3:7]   root quaternion (w, x, y, z)
  [7:10]  root linear velocity (world)
  [10:13] root angular velocity (world)
  [13:13+J]     joint positions
  [13+J:13+2J]  joint velocities

Reference data is passed in explicitly (tensors already gathered from the
motion library), so the same code serves the simulator loop (global frame
indices) and imagined rollouts (per-rollout window packs).

All heads are differentiable w.r.t. the state so that the policy gradient can
flow through them during imagined rollouts (paper §3.2: heads may be analytic).
Reward and termination mirror mjlab's BeyondMimic tracking task (Appendix A of
the paper): self-collision penalty removed, joint-limit weight -2, no
end-effector termination.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .fk import TorchFK
from .math_utils import (
  quat_apply,
  quat_apply_inverse,
  quat_error_magnitude,
  quat_inv,
  quat_mul,
  rot6d_from_quat,
  subtract_frame_transforms,
  yaw_quat,
)


@dataclass
class HeadsCfg:
  K: int = 10  # future reference frames fed to the goal encoder
  hist: int = 10  # proprioceptive history length
  dt: float = 0.02
  goal_anchor_pos: bool = False  # include ref anchor position relative to the robot anchor (needs odometry on hardware)
  # Sensor noise (uniform half-widths), Table A.1.
  noise_joint_pos: float = 0.01
  noise_joint_vel: float = 0.5
  noise_ang_vel: float = 0.2
  noise_gravity: float = 0.05
  # Reward (Table A.2); every term is multiplied by dt.
  w_body_pos: float = 1.0
  w_body_ori: float = 1.0
  w_body_lin_vel: float = 1.0
  w_body_ang_vel: float = 1.0
  w_root_pos: float = 0.5
  w_root_ori: float = 0.5
  w_action_rate: float = -0.1
  w_joint_limit: float = -2.0
  std_body_pos: float = 0.3
  std_body_ori: float = 0.4
  std_body_lin_vel: float = 1.0
  std_body_ang_vel: float = 3.14
  std_root_pos: float = 0.3
  std_root_ori: float = 0.4
  # Termination (Appendix A).
  term_z: float = 0.25
  term_tilt: float = 0.8
  term_z_temp: float = 0.02
  term_tilt_temp: float = 0.05
  extra: dict = field(default_factory=dict)

  @property
  def reward_max_per_step(self) -> float:
    return (
      self.w_body_pos + self.w_body_ori + self.w_body_lin_vel + self.w_body_ang_vel + self.w_root_pos + self.w_root_ori
    ) * self.dt


def window_frames(frame: torch.Tensor, clip_end: torch.Tensor, K: int) -> torch.Tensor:
  """[B] -> [B,K] frame indices frame..frame+K-1 clamped at clip_end-1 (same index space)."""
  return torch.minimum(frame[:, None] + torch.arange(K, device=frame.device)[None, :], clip_end[:, None] - 1)


class AnalyticHeads:
  def __init__(
    self,
    fk: TorchFK,
    tracked_local_ids: list[int],
    anchor_index: int,
    default_joint_pos: torch.Tensor,
    soft_joint_limits: torch.Tensor,
    cfg: HeadsCfg,
  ):
    self.fk = fk
    self.cfg = cfg
    self.device = fk.device
    self.tracked = torch.as_tensor(tracked_local_ids, device=self.device)
    self.anchor = anchor_index  # index within the tracked-body list
    self.anchor_local = int(tracked_local_ids[anchor_index])
    self.nj = default_joint_pos.shape[-1]
    self.q0 = default_joint_pos.to(self.device).reshape(-1)
    self.soft_lim = soft_joint_limits.to(self.device).reshape(self.nj, 2)
    self.gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device)
    self.proprio_dim = 3 * self.nj + 6
    self.goal_frame_dim = 2 * self.nj + 6 + (3 if cfg.goal_anchor_pos else 0)
    self.state_dim = 13 + 2 * self.nj
    self.feat_dim = 13 + 2 * self.nj  # 1 + 6 + 3 + 3 + 2J

  # ---- state helpers -------------------------------------------------------

  def split(self, s: torch.Tensor):
    J = self.nj
    return s[..., 0:3], s[..., 3:7], s[..., 7:10], s[..., 10:13], s[..., 13 : 13 + J], s[..., 13 + J : 13 + 2 * J]

  @staticmethod
  def join(p, qw, v, w, q, qd) -> torch.Tensor:
    return torch.cat([p, qw, v, w, q, qd], dim=-1)

  def kin(self, s: torch.Tensor) -> dict:
    """FK of the tracked bodies. Returns dict of [B,Bt,*] tensors + anchor pose."""
    p, qw, v, w, q, qd = self.split(s)
    pos, quat, lin, ang = self.fk.forward(p, qw, v, w, q, qd)
    out = dict(
      body_pos=pos[:, self.tracked],
      body_quat=quat[:, self.tracked],
      body_lin_vel=lin[:, self.tracked],
      body_ang_vel=ang[:, self.tracked],
    )
    out["anchor_pos"] = out["body_pos"][:, self.anchor]
    out["anchor_quat"] = out["body_quat"][:, self.anchor]
    return out

  def state_features(self, s: torch.Tensor) -> torch.Tensor:
    """Yaw-invariant features: [z, rot6d(local), v_heading, w_body, q, qd]."""
    p, qw, v, w, q, qd = self.split(s)
    yq = yaw_quat(qw)
    rot_local = quat_mul(quat_inv(yq), qw)
    return torch.cat(
      [p[:, 2:3], rot6d_from_quat(rot_local), quat_apply_inverse(yq, v), quat_apply_inverse(qw, w), q, qd], dim=-1
    )

  # ---- observation ---------------------------------------------------------

  def proprio(self, s: torch.Tensor, prev_action: torch.Tensor, bias: torch.Tensor, noise: bool) -> torch.Tensor:
    """[q_biased - q0, qd, base ang vel (pelvis frame), projected gravity, a_{t-1}]."""
    p, qw, v, w, q, qd = self.split(s)
    B = s.shape[0]
    q_rel = q + bias - self.q0
    w_b = quat_apply_inverse(qw, w)
    g_b = quat_apply_inverse(qw, self.gravity.expand(B, 3))
    if noise:
      c = self.cfg
      u = lambda x, h: x + (torch.rand_like(x) * 2.0 - 1.0) * h  # noqa: E731
      q_rel, qd, w_b, g_b = u(q_rel, c.noise_joint_pos), u(qd, c.noise_joint_vel), u(w_b, c.noise_ang_vel), u(g_b, c.noise_gravity)
    return torch.cat([q_rel, qd, w_b, g_b, prev_action], dim=-1)

  def goal(self, kin: dict, ref_q: torch.Tensor, ref_qd: torch.Tensor, ref_ap: torch.Tensor, ref_aq: torch.Tensor) -> torch.Tensor:
    """K reference frames relative to the robot anchor. Inputs [B,K,*] -> [B,K,G]."""
    B, K = ref_q.shape[0], ref_q.shape[1]
    rob_ap = kin["anchor_pos"][:, None, :].expand(B, K, 3).reshape(-1, 3)
    rob_aq = kin["anchor_quat"][:, None, :].expand(B, K, 4).reshape(-1, 4)
    rel_p, rel_q = subtract_frame_transforms(rob_ap, rob_aq, ref_ap.reshape(-1, 3), ref_aq.reshape(-1, 4))
    parts = [ref_q, ref_qd]
    if self.cfg.goal_anchor_pos:
      parts.append(rel_p.view(B, K, 3))
    parts.append(rot6d_from_quat(rel_q).view(B, K, 6))
    return torch.cat(parts, dim=-1)

  # ---- reward --------------------------------------------------------------

  def relative_reference(self, kin_prev: dict, ref_bp: torch.Tensor, ref_bq: torch.Tensor):
    """Reference bodies re-anchored to the robot anchor of the pre-step state
    (mjlab ``update_relative_body_poses``). ref_bp [B,Bt,3], ref_bq [B,Bt,4]."""
    ref_ap = ref_bp[:, self.anchor]
    ref_aq = ref_bq[:, self.anchor]
    rob_ap, rob_aq = kin_prev["anchor_pos"], kin_prev["anchor_quat"]
    Bt = ref_bp.shape[1]
    delta_pos = torch.cat([rob_ap[:, :2], ref_ap[:, 2:3]], dim=-1)
    delta_ori = yaw_quat(quat_mul(rob_aq, quat_inv(ref_aq)))
    dori = delta_ori[:, None, :].expand(-1, Bt, 4).reshape(-1, 4)
    body_quat_rel = quat_mul(dori, ref_bq.reshape(-1, 4)).view(-1, Bt, 4)
    body_pos_rel = delta_pos[:, None, :] + quat_apply(dori, (ref_bp - ref_ap[:, None, :]).reshape(-1, 3)).view(-1, Bt, 3)
    return body_pos_rel, body_quat_rel, ref_ap, ref_aq

  def reward(self, kin_prev: dict, kin_next: dict, s_next: torch.Tensor, action: torch.Tensor, prev_action: torch.Tensor, ref: dict, return_terms: bool = False):
    """ref: dict with body_pos/body_quat/body_lin_vel/body_ang_vel [B,Bt,*] at the target frame."""
    c = self.cfg
    body_pos_rel, body_quat_rel, ref_ap, ref_aq = self.relative_reference(kin_prev, ref["body_pos"], ref["body_quat"])
    bp, bq = kin_next["body_pos"], kin_next["body_quat"]
    Bt = bp.shape[1]
    e_root_pos = ((ref_ap - kin_next["anchor_pos"]) ** 2).sum(-1)
    e_root_ori = quat_error_magnitude(ref_aq, kin_next["anchor_quat"]) ** 2
    e_body_pos = ((body_pos_rel - bp) ** 2).sum(-1).mean(-1)
    e_body_ori = (quat_error_magnitude(body_quat_rel.reshape(-1, 4), bq.reshape(-1, 4)).view(-1, Bt) ** 2).mean(-1)
    e_lin = ((ref["body_lin_vel"] - kin_next["body_lin_vel"]) ** 2).sum(-1).mean(-1)
    e_ang = ((ref["body_ang_vel"] - kin_next["body_ang_vel"]) ** 2).sum(-1).mean(-1)
    _, _, _, _, q, _ = self.split(s_next)
    lim = (-(q - self.soft_lim[:, 0]).clamp(max=0.0) + (q - self.soft_lim[:, 1]).clamp(min=0.0)).sum(-1)
    terms = {
      "root_pos": c.w_root_pos * torch.exp(-e_root_pos / c.std_root_pos**2),
      "root_ori": c.w_root_ori * torch.exp(-e_root_ori / c.std_root_ori**2),
      "body_pos": c.w_body_pos * torch.exp(-e_body_pos / c.std_body_pos**2),
      "body_ori": c.w_body_ori * torch.exp(-e_body_ori / c.std_body_ori**2),
      "body_lin_vel": c.w_body_lin_vel * torch.exp(-e_lin / c.std_body_lin_vel**2),
      "body_ang_vel": c.w_body_ang_vel * torch.exp(-e_ang / c.std_body_ang_vel**2),
      "action_rate": c.w_action_rate * ((action - prev_action) ** 2).sum(-1),
      "joint_limit": c.w_joint_limit * lim,
    }
    r = sum(terms.values()) * c.dt
    if return_terms:
      return r, {k: v * c.dt for k, v in terms.items()}
    return r

  # ---- termination ---------------------------------------------------------

  def termination(self, kin_next: dict, ref_ap: torch.Tensor, ref_aq: torch.Tensor):
    """ref anchor pose [B,3]/[B,4] at the target frame. Returns (hard_bool [B], soft_prob [B])."""
    c = self.cfg
    B = ref_ap.shape[0]
    g = self.gravity.expand(B, 3)
    dz = (ref_ap[:, 2] - kin_next["anchor_pos"][:, 2]).abs()
    tilt = (quat_apply_inverse(ref_aq, g)[:, 2] - quat_apply_inverse(kin_next["anchor_quat"], g)[:, 2]).abs()
    hard = (dz > c.term_z) | (tilt > c.term_tilt)
    pz = torch.sigmoid((dz - c.term_z) / c.term_z_temp)
    pt = torch.sigmoid((tilt - c.term_tilt) / c.term_tilt_temp)
    soft = 1.0 - (1.0 - pz) * (1.0 - pt)
    return hard, soft
