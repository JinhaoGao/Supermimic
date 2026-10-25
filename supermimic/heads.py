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
end-effector termination (optional again via ``HeadsCfg.ee_term_z``: the team's
MuJoCo benchmark terminates on end-effector height, see README).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .fk import TorchFK
from .math_utils import (
  quat_angle,
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
  # + per goal frame the reference anchor's height, body-frame linear / angular velocity and projected gravity (10 values,
  # reference side only, so deployable without odometry; every open-source reproduction feeds the reference root motion)
  goal_ref_root: bool = False
  # Privileged tracking errors as extra critic inputs (training only): per tracked body the position error in the robot
  # anchor frame and the orientation error angle, plus the anchor height error, against goal frame 0.
  critic_errors: bool = False
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
  # End-effector height termination (BeyondMimic's ee_body_pos: |z_ref - z_robot| > ee_term_z on the robot's
  # ee bodies, G1 ankles + wrists, E2 ankles + elbows). The paper drops it (0 = off, the default); the team's
  # MotionAtlas v4 benchmark terminates on it at 0.25 m, so v2 runs train with it (env + soft head).
  ee_term_z: float = 0.0
  ee_term_temp: float = 0.02
  # Root-orientation termination (branch ddp-bj31-v2 ecc551f, 10-25; +1.0 pp on v4 there): fail when the robot root's
  # (G1 pelvis) full rotation angle vs the reference, yaw included, exceeds root_ori_term rad, the eval site's root_ori
  # test at 1.0 (the anchor tilt test sees neither the yaw nor the pelvis / torso decoupling through the waist); soft
  # term sigmoid((angle - A) / root_ori_term_temp) in the noisy-OR. 0 = off.
  root_ori_term: float = 0.0
  root_ori_term_temp: float = 0.05
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
    ee_indices: list[int] | None = None,
    root_ori_index: int | None = None,
  ):
    self.fk = fk
    self.cfg = cfg
    self.device = fk.device
    self.tracked = torch.as_tensor(tracked_local_ids, device=self.device)
    self.ee = torch.as_tensor(list(ee_indices or []), dtype=torch.long, device=self.device)  # within the tracked list
    if cfg.ee_term_z > 0 and self.ee.numel() == 0:
      raise ValueError("ee_term_z > 0 needs the end-effector bodies (env termination 'ee_body_pos')")
    self.root_ori = root_ori_index  # root body (pelvis) within the tracked-body list (root_ori_term)
    if cfg.root_ori_term > 0 and root_ori_index is None:
      raise ValueError("root_ori_term > 0 needs the root body's index in the tracked-body list")
    self.anchor = anchor_index  # index within the tracked-body list
    self.anchor_local = int(tracked_local_ids[anchor_index])
    self.nj = default_joint_pos.shape[-1]
    self.q0 = default_joint_pos.to(self.device).reshape(-1)
    self.soft_lim = soft_joint_limits.to(self.device).reshape(self.nj, 2)
    self.gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device)
    self.proprio_dim = 3 * self.nj + 6
    self.goal_frame_dim = sum(n for _, n in self.goal_layout())
    self.critic_err_dim = 4 * len(tracked_local_ids) + 1 if cfg.critic_errors else 0
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

  def goal_layout(self, cfg: HeadsCfg | None = None) -> list[tuple[str, int]]:
    """Segments of one goal frame (name, size) in order; cfg defaults to this head's (graft: a checkpoint's)."""
    c = cfg or self.cfg
    out = [("q", self.nj), ("qd", self.nj)] + ([("anchor_pos", 3)] if c.goal_anchor_pos else []) + [("rot6d", 6)]
    return out + ([("ref_root", 10)] if c.goal_ref_root else [])

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

  def goal(self, kin: dict, ref_q: torch.Tensor, ref_qd: torch.Tensor, ref_ap: torch.Tensor, ref_aq: torch.Tensor,
           ref_av: torch.Tensor | None = None, ref_aw: torch.Tensor | None = None) -> torch.Tensor:
    """K reference frames relative to the robot anchor. Inputs [B,K,*] -> [B,K,G]. ref_av / ref_aw: reference anchor
    linear / angular velocity (world), needed with goal_ref_root."""
    B, K = ref_q.shape[0], ref_q.shape[1]
    rob_ap = kin["anchor_pos"][:, None, :].expand(B, K, 3).reshape(-1, 3)
    rob_aq = kin["anchor_quat"][:, None, :].expand(B, K, 4).reshape(-1, 4)
    rel_p, rel_q = subtract_frame_transforms(rob_ap, rob_aq, ref_ap.reshape(-1, 3), ref_aq.reshape(-1, 4))
    parts = [ref_q, ref_qd]
    if self.cfg.goal_anchor_pos:
      parts.append(rel_p.view(B, K, 3))
    parts.append(rot6d_from_quat(rel_q).view(B, K, 6))
    if self.cfg.goal_ref_root:
      aq = ref_aq.reshape(-1, 4)
      v_b = quat_apply_inverse(aq, ref_av.reshape(-1, 3)).view(B, K, 3)
      w_b = quat_apply_inverse(aq, ref_aw.reshape(-1, 3)).view(B, K, 3)
      g_b = quat_apply_inverse(aq, self.gravity.expand(B * K, 3)).view(B, K, 3)
      parts += [ref_ap[..., 2:3], v_b, w_b, g_b]
    return torch.cat(parts, dim=-1)

  def critic_errors(self, kin: dict, ref_bp: torch.Tensor, ref_bq: torch.Tensor) -> torch.Tensor:
    """[B, 4*Bt+1] tracking errors of the state (kin) against reference bodies [B,Bt,*] re-anchored as in the reward:
    position error in the robot anchor frame (full rotation: no yaw singularity when inverted), orientation error
    angle, anchor height error."""
    body_pos_rel, body_quat_rel, ref_ap, _ = self.relative_reference(kin, ref_bp, ref_bq)
    B, Bt = ref_bp.shape[0], ref_bp.shape[1]
    aq = kin["anchor_quat"][:, None, :].expand(B, Bt, 4).reshape(-1, 4)
    dp = quat_apply_inverse(aq, (body_pos_rel - kin["body_pos"]).reshape(-1, 3)).view(B, Bt * 3)
    dq = quat_error_magnitude(body_quat_rel.reshape(-1, 4), kin["body_quat"].reshape(-1, 4)).view(B, Bt)
    dz = (ref_ap[:, 2] - kin["anchor_pos"][:, 2])[:, None]
    return torch.cat([dp, dq, dz], dim=-1)

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

  def termination(self, kin_next: dict, ref_bp: torch.Tensor, ref_bq: torch.Tensor, scale: torch.Tensor | None = None):
    """Reference tracked-body poses [B,Bt,3]/[B,Bt,4] at the target frame. Returns (hard_bool [B], soft_prob [B]).

    Anchor height / tilt as in BeyondMimic; with ee_term_z > 0 also the end-effector height error. The
    re-anchored reference used by mjlab's ee_body_pos keeps the reference z, so world z is compared.
    scale [B] (optional): per-sample multiplier of every threshold and sigmoid temperature (hard-clip
    curriculum, see LibraryMotionCommand.term_scale_env); the env terminations use the same factor."""
    c = self.cfg
    ref_ap, ref_aq = ref_bp[:, self.anchor], ref_bq[:, self.anchor]
    B = ref_ap.shape[0]
    g = self.gravity.expand(B, 3)
    k = scale if scale is not None else torch.ones(B, device=ref_bp.device)
    dz = (ref_ap[:, 2] - kin_next["anchor_pos"][:, 2]).abs()
    tilt = (quat_apply_inverse(ref_aq, g)[:, 2] - quat_apply_inverse(kin_next["anchor_quat"], g)[:, 2]).abs()
    hard = (dz > c.term_z * k) | (tilt > c.term_tilt * k)
    pz = torch.sigmoid((dz - c.term_z * k) / (c.term_z_temp * k))
    pt = torch.sigmoid((tilt - c.term_tilt * k) / (c.term_tilt_temp * k))
    alive = (1.0 - pz) * (1.0 - pt)
    if c.ee_term_z > 0:
      ez = (ref_bp[:, self.ee, 2] - kin_next["body_pos"][:, self.ee, 2]).abs()  # [B, n_ee]
      hard = hard | (ez > c.ee_term_z * k[:, None]).any(-1)
      alive = alive * (1.0 - torch.sigmoid((ez - c.ee_term_z * k[:, None]) / (c.ee_term_temp * k[:, None]))).prod(-1)
    if c.root_ori_term > 0:
      ang = quat_angle(ref_bq[:, self.root_ori], kin_next["body_quat"][:, self.root_ori])
      hard = hard | (ang > c.root_ori_term * k)
      alive = alive * (1.0 - torch.sigmoid((ang - c.root_ori_term * k) / (c.root_ori_term_temp * k)))
    return hard, 1.0 - alive
