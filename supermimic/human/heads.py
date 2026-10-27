"""Analytic heads for a human (SMPL) reference: goal, reward and termination without any retargeted robot motion.

Same interface as heads.AnalyticHeads (kin / proprio / goal / reward / termination on the minimal state, all
differentiable through TorchFK), so the world model, replay, policy / critic and the imagined BPTT are unchanged.
The reference is the per-frame human fields of human/motions.py; the robot side is compared through the scale-free
SMPL <-> robot bone map (human/bone_map.py: unit bone directions, each skeleton in its own heading frame or each bone
in its parent's frame) plus root quantities rescaled by k = robot rest pelvis height / human rest pelvis height.

Goal (per frame, K frames): h_feat (138, reference side only) + optionally (cos, sin) of the human heading minus the
robot heading (goal_rel_yaw), the only robot-relative input -- without it heading errors are unobservable.

Reward (x dt, like AnalyticHeads): exp kernels on
  bone_global_torso    torso bones (pelvis / torso up and left) in each skeleton's heading frame
  bone_local_<group>   every bone in its parent's frame (torso / leg / arm)
  root_vel             robot pelvis velocity (robot heading frame) vs k x human root velocity (human heading frame)
  yaw_rate, heading    pelvis yaw rate; heading (hip-line yaw) difference, 2 - 2 cos (smooth at 0)
  height               pelvis height / rest height (each skeleton's own)
  contact / slip       optional (off by default): soft ankle contact from the ankle height vs the human labels
plus action rate and joint limits. Bone errors are squared chords |d_h - d_r|^2 = 2 - 2 cos(angle) (~ angle^2), smooth
at 0 (an atan2 angle has an infinite gradient there, which the BPTT would turn into NaN).

Termination (hard test in the simulator, sigmoid soft term in imagination, noisy-OR like AnalyticHeads): angle of the
pelvis_up / torso_up bones (heading frames) > term_trunk, normalized pelvis height error > term_height.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..fk import TorchFK
from ..heads import AnalyticHeads, HeadsCfg
from . import bone_map as bone
from .motions import GOAL_FEAT_DIM

# Reference fields per step (reward / termination), in the positional order of HumanStepFns._post / the pack tuple.
STEP_FIELDS = ("h_dir", "h_dir_local", "h_root_vel", "h_yaw", "h_yaw_rate", "h_height", "h_rest", "h_contact")
GOAL_FIELDS = ("h_feat", "h_yaw")
PACK_FIELDS = ("h_feat",) + STEP_FIELDS  # imagination window pack (h_yaw is in STEP_FIELDS)
TRUNK_SEGMENTS = ("pelvis_up", "torso_up")
GROUPS = ("torso", "leg", "arm")


@dataclass
class HumanHeadsCfg(HeadsCfg):
  reference: str = "human"
  bone_map: str = "g1"
  ee_termination: bool = False  # robot-reference terms do not exist here (HeadsCfg fields kept for the shared code)
  goal_rel_yaw: bool = True
  # Reward weights / kernel widths (bone std in chord units ~ radians for small angles); start values from RGMT P1.
  w_bone_global_torso: float = 1.0
  w_bone_local_torso: float = 0.5
  w_bone_local_leg: float = 1.5
  w_bone_local_arm: float = 1.0
  std_bone: float = 0.35
  w_root_vel: float = 1.0
  std_root_vel: float = 0.5
  w_yaw_rate: float = 0.5
  std_yaw_rate: float = 1.0
  w_heading: float = 0.5
  std_heading: float = 0.4
  w_height: float = 0.5
  std_height: float = 0.1
  w_contact: float = 0.0
  w_slip: float = 0.0
  contact_height: float = 0.03  # ankle lift (above its standing height) at which the soft contact is 0.5
  contact_temp: float = 0.01
  # Terminations.
  term_trunk: float = 1.0  # rad, pelvis_up / torso_up bone angle (heading frames)
  term_trunk_temp: float = 0.05
  term_height: float = 0.3  # |pelvis height / rest height| error
  term_height_temp: float = 0.02

  def __post_init__(self):
    super().__post_init__()
    if self.ee_termination or self.root_ori_on or self.root_z_on or self.goal_anchor_pos or self.goal_ref_root or self.critic_errors:
      raise ValueError("human reference: the robot-reference options (ee / root_ori / root_z terms, goal_anchor_pos, "
                       "goal_ref_root, critic_errors) do not apply")

  @property
  def reward_max_per_step(self) -> float:
    w = (self.w_bone_global_torso, self.w_bone_local_torso, self.w_bone_local_leg, self.w_bone_local_arm, self.w_root_vel,
         self.w_yaw_rate, self.w_heading, self.w_height, self.w_contact)
    return sum(x for x in w if x > 0) * self.dt


def robot_heading(link_pos: torch.Tensor, bt: bone.BoneMapTensors, eps: float = 0.05) -> torch.Tensor:
  """bone.robot_heading with a larger normalization floor: when the hip line turns vertical (robot on its side) the
  heading is undefined and 1 / |left_xy| would flood the BPTT gradient. Below eps (hip line tilted > ~75 deg on the G1,
  far past the trunk termination) the frame is no longer normalized and stops rotating."""
  left = link_pos[..., bt.r_heading[0], :] - link_pos[..., bt.r_heading[1], :]
  left = torch.cat([left[..., :2], torch.zeros_like(left[..., 2:])], dim=-1)
  left = left / left.norm(dim=-1, keepdim=True).clamp(min=eps)
  up = torch.zeros_like(left)
  up[..., 2] = 1.0
  return torch.stack([torch.cross(left, up, dim=-1), left, up], dim=-1)


def chord2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  """|a - b|^2 of unit vectors = 2 - 2 cos(angle), [...]."""
  return ((a - b) ** 2).sum(-1)


def chord_angle(c2: torch.Tensor) -> torch.Tensor:
  """Angle from a squared chord (finite gradient everywhere: the eps keeps sqrt' finite where c2 -> 0)."""
  return 2.0 * torch.asin((torch.sqrt(c2 + 1e-12) * 0.5).clamp(max=1.0 - 1e-6))


class HumanHeads(AnalyticHeads):
  def __init__(self, fk: TorchFK, link_local_ids: list[int], default_joint_pos: torch.Tensor, soft_joint_limits: torch.Tensor,
               cfg: HumanHeadsCfg, robot_rest_height: float, robot_ankle_height: float):
    self.fk = fk
    self.cfg = cfg
    self.device = fk.device
    self.links = torch.as_tensor(link_local_ids, dtype=torch.long, device=self.device)
    self.bone_map = bone.SMPL_BONE_MAPS[cfg.bone_map]
    self.bt = bone.BoneMapTensors(self.bone_map, self.bone_map.load_calibration(), device=self.device)
    self.rest_h = float(robot_rest_height)
    self.ankle_h = float(robot_ankle_height)
    # AnalyticHeads attributes read by the shared code (proprio / state features / trainer bookkeeping).
    self.anchor = None
    self.tracked = self.links
    self.ee = torch.zeros(0, dtype=torch.long, device=self.device)
    self.root_ori = None  # no robot-reference root orientation: root_ori_angle_state -> None (no pelvis-angle logging)
    self.root_ori_is_root = True
    self.nj = default_joint_pos.shape[-1]
    self.q0 = default_joint_pos.to(self.device).reshape(-1)
    self.soft_lim = soft_joint_limits.to(self.device).reshape(self.nj, 2)
    self.gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device)
    self.proprio_dim = 3 * self.nj + 6
    self.goal_frame_dim = GOAL_FEAT_DIM + (2 if cfg.goal_rel_yaw else 0)
    self.critic_err_dim = 0
    self.state_dim = 13 + 2 * self.nj
    self.feat_dim = 13 + 2 * self.nj
    bt = self.bt
    self.w_global = bt.weights * bt.group_mask("torso")
    self.w_local = {g: bt.weights * bt.group_mask(g, local=True) for g in GROUPS}
    self.trunk = torch.tensor([bt.names.index(s) for s in TRUNK_SEGMENTS], dtype=torch.long, device=self.device)
    self.feet = torch.tensor(bt.r_feet, dtype=torch.long, device=self.device)

  def goal_layout(self, cfg: HeadsCfg | None = None) -> list[tuple[str, int]]:
    c = cfg or self.cfg
    return [("h_feat", GOAL_FEAT_DIM)] + ([("rel_yaw", 2)] if getattr(c, "goal_rel_yaw", False) else [])

  # ---- kinematics ------------------------------------------------------------

  def kin(self, s: torch.Tensor) -> dict:
    """FK of the bone-map links -> link poses, robot heading frame / yaw and bone directions (heading frame, local)."""
    p, qw, v, w, q, qd = self.split(s)
    pos, quat, lin, _ = self.fk.forward(p, qw, v, w, q, qd)
    lp, lq = pos[:, self.links], quat[:, self.links]
    rot = bone.quat_wxyz_to_matrix(lq)
    heading = robot_heading(lp, self.bt)
    r_w = bone.robot_bone_dirs_w(lp, rot, self.bt)
    return dict(
      link_pos=lp, link_lin_vel=lin[:, self.links], heading=heading, yaw=torch.atan2(heading[:, 1, 0], heading[:, 0, 0]),
      dirs=bone.to_frame(r_w, heading), dirs_local=bone.robot_bone_dirs_local(r_w, rot, self.bt),
    )

  # ---- goal --------------------------------------------------------------------

  def goal(self, kin: dict, h_feat: torch.Tensor, h_yaw: torch.Tensor) -> torch.Tensor:
    """h_feat [B,K,138], h_yaw [B,K] -> goal [B,K,G]."""
    if not self.cfg.goal_rel_yaw:
      return h_feat
    d = h_yaw - kin["yaw"][:, None]
    return torch.cat([h_feat, torch.cos(d)[..., None], torch.sin(d)[..., None]], dim=-1)

  # ---- reward ------------------------------------------------------------------

  def _errors(self, kin: dict, s: torch.Tensor, ref: dict) -> dict:
    """Squared tracking errors of state s (kin = its FK) against the reference frame ref."""
    v_h = torch.einsum("bji,bj->bi", kin["heading"], s[:, 7:10])  # pelvis velocity in the robot heading frame
    k = (self.rest_h / ref["h_rest"])[:, None]
    return dict(
      bone_g=chord2(ref["h_dir"], kin["dirs"]), bone_l=chord2(ref["h_dir_local"], kin["dirs_local"]),
      root_vel=((k * ref["h_root_vel"] - v_h) ** 2).sum(-1), yaw_rate=(ref["h_yaw_rate"] - s[:, 12]) ** 2,
      heading=2.0 - 2.0 * torch.cos(ref["h_yaw"] - kin["yaw"]), height=(s[:, 2] / self.rest_h - ref["h_height"]) ** 2,
    )

  def soft_contact(self, kin: dict) -> torch.Tensor:
    """[B,2] soft contact of the (left, right) ankle links from their lift above the standing ankle height."""
    lift = kin["link_pos"][:, self.feet, 2] - self.ankle_h
    return torch.sigmoid((self.cfg.contact_height - lift) / self.cfg.contact_temp)

  def reward(self, kin_prev: dict, kin_next: dict, s_next: torch.Tensor, action: torch.Tensor, prev_action: torch.Tensor,
             ref: dict, return_terms: bool = False):
    c = self.cfg
    e = self._errors(kin_next, s_next, ref)
    sb = torch.exp(-e["bone_g"] / c.std_bone**2)
    sl = torch.exp(-e["bone_l"] / c.std_bone**2)
    wg = self.w_global
    terms = {"bone_global_torso": c.w_bone_global_torso * (sb * wg).sum(-1) / wg.sum()}
    for g, wl in zip(GROUPS, (c.w_bone_local_torso, c.w_bone_local_leg, c.w_bone_local_arm)):
      m = self.w_local[g]
      terms[f"bone_local_{g}"] = wl * (sl * m).sum(-1) / m.sum()
    terms["root_vel"] = c.w_root_vel * torch.exp(-e["root_vel"] / c.std_root_vel**2)
    terms["yaw_rate"] = c.w_yaw_rate * torch.exp(-e["yaw_rate"] / c.std_yaw_rate**2)
    if c.w_heading:
      terms["heading"] = c.w_heading * torch.exp(-e["heading"] / c.std_heading**2)
    terms["height"] = c.w_height * torch.exp(-e["height"] / c.std_height**2)
    if c.w_contact or c.w_slip:
      pc = self.soft_contact(kin_next)
      if c.w_contact:
        terms["contact"] = c.w_contact * (1.0 - (pc - ref["h_contact"]).abs().mean(-1))
      if c.w_slip:
        vxy = (kin_next["link_lin_vel"][:, self.feet, :2] ** 2).sum(-1)
        terms["slip"] = -c.w_slip * (pc * vxy).sum(-1)
    _, _, _, _, q, _ = self.split(s_next)
    lim = (-(q - self.soft_lim[:, 0]).clamp(max=0.0) + (q - self.soft_lim[:, 1]).clamp(min=0.0)).sum(-1)
    terms["action_rate"] = c.w_action_rate * ((action - prev_action) ** 2).sum(-1)
    terms["joint_limit"] = c.w_joint_limit * lim
    r = sum(terms.values()) * c.dt
    if return_terms:
      return r, {k: v * c.dt for k, v in terms.items()}
    return r

  # ---- termination -------------------------------------------------------------

  def termination(self, kin_next: dict, s_next: torch.Tensor, ref: dict):
    """(hard [B] bool, soft [B]) at the target frame; see the module docstring."""
    c = self.cfg
    ang = chord_angle(chord2(ref["h_dir"][:, self.trunk], kin_next["dirs"][:, self.trunk]))  # [B,2]
    dh = (s_next[:, 2] / self.rest_h - ref["h_height"]).abs()
    hard = (ang > c.term_trunk).any(-1) | (dh > c.term_height)
    alive = torch.prod(1.0 - torch.sigmoid((ang - c.term_trunk) / c.term_trunk_temp), dim=-1)
    alive = alive * (1.0 - torch.sigmoid((dh - c.term_height) / c.term_height_temp))
    return hard, 1.0 - alive

  # ---- evaluation metrics (no grad) ----------------------------------------------

  @torch.no_grad()
  def metrics(self, kin: dict, s: torch.Tensor, ref: dict) -> dict[str, torch.Tensor]:
    """Per-env tracking errors [B]: bone angles in degrees (global torso, local torso / leg / arm, weighted means), root
    velocity error (m/s, k-scaled target), yaw-rate error (rad/s), heading error (deg), normalized height error."""
    e = self._errors(kin, s, ref)
    deg_g = torch.rad2deg(chord_angle(e["bone_g"]))
    deg_l = torch.rad2deg(chord_angle(e["bone_l"]))
    out = {"bone_global_torso_deg": (deg_g * self.w_global).sum(-1) / self.w_global.sum()}
    for g in GROUPS:
      m = self.w_local[g]
      out[f"bone_local_{g}_deg"] = (deg_l * m).sum(-1) / m.sum()
    out["root_vel_err"] = e["root_vel"].sqrt()
    out["yaw_rate_err"] = e["yaw_rate"].sqrt()
    out["heading_err_deg"] = torch.rad2deg(chord_angle(e["heading"]))
    out["height_err"] = e["height"].sqrt()
    return out
