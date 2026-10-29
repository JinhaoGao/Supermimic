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
  bone_global_<limb>   optional (w_bone_global_arm / _leg, default 0): limb bones in the heading frame too. The local
                       terms alone leave the twist of a parent link free -- on ab1 (10-27) the policy rotated the upper
                       arms about their own axis so that forearm / hand / palm matched in the twisted shoulder frames
                       (local 3-17 deg) while pointing 45-105 deg away in the world (arms swung out)
  bone_local_<group>   every bone in its parent's frame (torso / leg / arm)
  root_vel             robot pelvis velocity (robot heading frame) vs k x human root velocity (human heading frame)
  yaw_rate, heading    pelvis yaw rate; heading (hip-line yaw) difference, 2 - 2 cos (smooth at 0)
  height               pelvis height / rest height (each skeleton's own)
  contact / slip       optional (off by default): soft ankle contact from the ankle height vs the human labels
plus action rate and joint limits. Bone errors are squared chords |d_h - d_r|^2 = 2 - 2 cos(angle) (~ angle^2), smooth
at 0 (an atan2 angle has an infinite gradient there, which the BPTT would turn into NaN).

Termination (hard test in the simulator, sigmoid soft term in imagination, noisy-OR like AnalyticHeads): angle of the
pelvis_up / torso_up bones (heading frames) > term_trunk, normalized pelvis height error > term_height, optionally
(term_foot_lift > 0) an ankle-lift error > term_foot_lift on either foot.

End-effector world tracking (ee_mode, docs/ee_world_tracking_plan.md; after Workhorse, arXiv 2610.09117): the task links
of the bone map (torso, wrists, feet) as world poses, unscaled, mapped onto the episode by a per-env condition
cond = [alpha, ee_valid, anchor xy (2), drift (dx, dy, dyaw), drift rate (3)] (COND_DIM) that the command samples:
  target = drift o (human task pose - anchor xy): the drift is a rigid yaw about the frame's own torso point plus an xy
  shift, extrapolated to each sample's own time (rate x steps, clamped to the drift box).
  goal       + EE block (task links in the robot's current torso_link frame: position, rot6d; zeroed when ee_valid = 0),
             alpha, ee_valid
  reward     (1 - (1 - ee_style) alpha) x the positive H4 terms + alpha x task kernels: torso xy / rotation (wide and
             narrow), wrist position (wide and narrow) / rotation (world), feet xy / rotation relative to the torso point
             in the heading frame, swing-foot clearance; penalties unchanged. The torso height is not a target: the
             human shoulder line (median 1.30 m on AMASS) is out of the G1's reach (1.08 m standing straight).
  termination + wrist / torso-xy world error > term_ee_* / alpha (off at alpha = 0)
With ee_mode off everything is as before (same goal, reward, termination and compiled graphs).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..fk import TorchFK
from ..heads import AnalyticHeads, HeadsCfg
from . import bone_map as bone
from .motions import GOAL_FEAT_DIM

# Reference fields per step (reward / termination), in the positional order of HumanStepFns._post / the pack tuple.
STEP_FIELDS = ("h_dir", "h_dir_local", "h_root_vel", "h_yaw", "h_yaw_rate", "h_height", "h_rest", "h_contact", "h_foot_lift")
GOAL_FIELDS = ("h_feat", "h_yaw")
PACK_FIELDS = ("h_feat",) + STEP_FIELDS  # imagination window pack (h_yaw is in STEP_FIELDS)
EE_STEP_FIELDS = ("h_task_pos", "h_task_rot6d")  # ee_mode: appended to STEP_FIELDS / GOAL_FIELDS / PACK_FIELDS
COND_DIM = 10  # alpha, ee_valid, anchor x, y, drift dx, dy, dyaw, drift rate x 3 (per step)
TASK_TORSO, TASK_WRISTS, TASK_FEET = 0, (1, 2), (3, 4)  # bone_map.task_links order
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
  w_bone_global_arm: float = 0.0  # 0 = RGMT P1 (limbs local only); see the module docstring
  w_bone_global_leg: float = 0.0
  w_bone_local_torso: float = 0.5
  w_bone_local_leg: float = 1.5
  w_bone_local_arm: float = 1.0
  std_bone: float = 0.35
  std_bone_leg: float = 0.0  # kernel width of the leg terms (global + local); 0 = std_bone. ab2 (10-28): with 0.35 a leg
  #                            40 deg off (lifts) scores exp(-4) ~ 0.02 -- almost no gradient back towards the lift
  # Foot lift: robot ankle-link height above its standing height vs k x the human ankle lift (h_foot_lift), both feet.
  w_foot_lift: float = 0.0
  std_foot_lift: float = 0.1
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
  # Foot-lift termination (0 = off): |robot ankle lift - k x human ankle lift| > term_foot_lift m on either foot -- the
  # human-side analogue of the baseline's end-effector height test (0.25 m). ma1 (10-28): without it the human runs
  # skipped fast kicks (robot foot peak / human peak 0.17-0.26, baseline 1.08) -- a skipped kick only cost the leg
  # terms of those frames, while the baseline's termination costs every future reward.
  term_foot_lift: float = 0.0
  term_foot_lift_temp: float = 0.02
  # End-effector world tracking (module docstring). The task weights below are relative: they are rescaled so that they
  # sum to (1 - ee_style) x the H4 sum (ee_task_scale), so the reward scale is the same for every alpha and every
  # ablation of them (the defaults sum to 7 = 0.7 x 10: scale 1 with the H4 recipe).
  ee_mode: bool = False
  ee_style: float = 0.3  # share of the positive H4 terms kept at alpha = 1
  w_ee_torso_wide: float = 1.0
  std_ee_torso_wide_pos: float = 0.3
  std_ee_torso_wide_rot: float = 0.4
  w_ee_torso: float = 1.0
  std_ee_torso_pos: float = 0.06
  std_ee_torso_rot: float = 0.15
  w_ee_wrist_wide: float = 1.0  # not in Workhorse: a narrow kernel alone has no gradient far from the target (BPTT)
  std_ee_wrist_wide: float = 0.3
  w_ee_wrist: float = 2.0
  std_ee_wrist: float = 0.06
  w_ee_wrist_rot: float = 0.5
  std_ee_wrist_rot: float = 0.15
  w_ee_foot: float = 1.0
  std_ee_foot_pos: float = 0.06
  std_ee_foot_rot: float = 0.1
  w_ee_clear: float = 0.5
  ee_clear_height: float = 0.10  # ankle lift asked for while the human foot swings
  std_ee_clear: float = 0.035
  term_ee_wrist: float = 0.4  # m at alpha = 1 (threshold / alpha); 0 = off
  term_ee_torso: float = 0.5
  term_ee_temp: float = 0.02
  # Condition sampling (human.command): alpha at reset 0 w.p. alpha_p0, 1 w.p. alpha_p1, else U(0, 1); mid-episode
  # switch to a new alpha w.p. alpha_switch_p per episode (linear ramp alpha_ramp_s); alpha = 0 episodes drop the EE
  # block (ee_valid = 0) w.p. ee_dropout. Drift: piecewise-linear random walk in the box (drift_xy, drift_yaw), the rate
  # resampled every U(drift_tmin, drift_tmax) s towards a uniform point of the box; drift_xy = drift_yaw = 0: off.
  alpha_p0: float = 0.25
  alpha_p1: float = 0.25
  alpha_switch_p: float = 0.3
  alpha_ramp_s: float = 0.5
  ee_dropout: float = 0.5
  drift_xy: float = 0.10
  drift_yaw: float = 0.1745
  drift_tmin: float = 0.2
  drift_tmax: float = 1.0
  eval_alpha: float = 1.0  # evaluation episodes: this alpha, ee_valid = 1, no drift
  # Clips the G1 can track in the world (scripts/human/prepare_ee_ik.py ee_list.txt: paths or basenames, one per line):
  # training episodes of the other clips keep alpha = 0 (H4 only). "" = every clip.
  ee_clip_list: str = ""

  def __post_init__(self):
    super().__post_init__()
    if self.ee_termination or self.root_ori_on or self.root_z_on or self.goal_anchor_pos or self.goal_ref_root or self.critic_errors:
      raise ValueError("human reference: the robot-reference options (ee / root_ori / root_z terms, goal_anchor_pos, "
                       "goal_ref_root, critic_errors) do not apply")

  @property
  def h4_weight_sum(self) -> float:
    w = (self.w_bone_global_torso, self.w_bone_global_arm, self.w_bone_global_leg, self.w_bone_local_torso, self.w_bone_local_leg, self.w_bone_local_arm, self.w_root_vel,
         self.w_yaw_rate, self.w_heading, self.w_height, self.w_contact, self.w_foot_lift)
    return sum(x for x in w if x > 0)

  @property
  def ee_weight_sum(self) -> float:
    w = (self.w_ee_torso_wide, self.w_ee_torso, self.w_ee_wrist_wide, self.w_ee_wrist, self.w_ee_wrist_rot, self.w_ee_foot, self.w_ee_clear)
    return sum(x for x in w if x > 0)

  @property
  def ee_task_scale(self) -> float:
    """Multiplier of the w_ee_* weights: their sum becomes (1 - ee_style) x the H4 sum."""
    return (1.0 - self.ee_style) * self.h4_weight_sum / max(self.ee_weight_sum, 1e-9)

  @property
  def reward_max_per_step(self) -> float:
    return self.h4_weight_sum * self.dt  # ee_mode: the same for every alpha (ee_task_scale)


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


def rot6d_to_matrix(r6: torch.Tensor) -> torch.Tensor:
  """[..., 6] (first two columns, math_utils.rot6d_from_quat layout) -> [..., 3, 3] (Gram-Schmidt)."""
  m = r6.unflatten(-1, (3, 2))
  a = m[..., 0] / m[..., 0].norm(dim=-1, keepdim=True).clamp(min=1e-6)
  b = m[..., 1] - (a * m[..., 1]).sum(-1, keepdim=True) * a
  b = b / b.norm(dim=-1, keepdim=True).clamp(min=1e-6)
  return torch.stack([a, b, torch.cross(a, b, dim=-1)], dim=-1)


def rot_z(yaw: torch.Tensor) -> torch.Tensor:
  c, s = torch.cos(yaw), torch.sin(yaw)
  z, o = torch.zeros_like(yaw), torch.ones_like(yaw)
  return torch.stack([c, -s, z, s, c, z, z, z, o], dim=-1).unflatten(-1, (3, 3))


def rot_err2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  """|A - B|_F^2 / 2 = 2 - 2 cos(angle) (~ angle^2), smooth at 0 like chord2; [..., 3, 3] -> [...]."""
  return ((a - b) ** 2).sum((-1, -2)) * 0.5


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
    self.ee_on = bool(cfg.ee_mode)
    self.n_task = len(self.bone_map.task_links)
    self.goal_frame_dim = sum(n for _, n in self.goal_layout())
    self.critic_err_dim = 0
    self.state_dim = 13 + 2 * self.nj
    self.feat_dim = 13 + 2 * self.nj
    bt = self.bt
    self.w_global = bt.weights * bt.group_mask("torso")
    self.w_global_limb = {g: bt.weights * bt.group_mask(g) for g in ("arm", "leg")}
    self.w_local = {g: bt.weights * bt.group_mask(g, local=True) for g in GROUPS}
    self.trunk = torch.tensor([bt.names.index(s) for s in TRUNK_SEGMENTS], dtype=torch.long, device=self.device)
    self.feet = torch.tensor(bt.r_feet, dtype=torch.long, device=self.device)
    self.leg_mask = bt.group_mask("leg")
    self.drift_box = torch.tensor([cfg.drift_xy, cfg.drift_xy, cfg.drift_yaw], device=self.device)

  @property
  def step_fields(self) -> tuple[str, ...]:
    """Positional step reference of the simulator side (current_ref) / reward: STEP_FIELDS (+ EE fields, cond)."""
    return STEP_FIELDS + (EE_STEP_FIELDS + ("cond",) if self.ee_on else ())

  @property
  def goal_fields(self) -> tuple[str, ...]:
    """Library fields of the goal window (cond is per env, appended by the caller)."""
    return GOAL_FIELDS + (EE_STEP_FIELDS if self.ee_on else ())

  @property
  def pack_fields(self) -> tuple[str, ...]:
    """Positional imagination pack (trainer: Trainer._pack_args): library windows, then cond [N,C] of the start."""
    return PACK_FIELDS + (EE_STEP_FIELDS + ("cond",) if self.ee_on else ())

  def goal_layout(self, cfg: HeadsCfg | None = None) -> list[tuple[str, int]]:
    c = cfg or self.cfg
    out = [("h_feat", GOAL_FEAT_DIM)] + ([("rel_yaw", 2)] if getattr(c, "goal_rel_yaw", False) else [])
    if getattr(c, "ee_mode", False):
      out += [("ee", 9 * len(self.bone_map.task_links)), ("alpha", 1), ("ee_valid", 1)]
    return out

  # ---- kinematics ------------------------------------------------------------

  def kin(self, s: torch.Tensor) -> dict:
    """FK of the bone-map links -> link poses, robot heading frame / yaw and bone directions (heading frame, local)."""
    p, qw, v, w, q, qd = self.split(s)
    pos, quat, lin, _ = self.fk.forward(p, qw, v, w, q, qd)
    lp, lq = pos[:, self.links], quat[:, self.links]
    rot = bone.quat_wxyz_to_matrix(lq)
    heading = robot_heading(lp, self.bt)
    r_w = bone.robot_bone_dirs_w(lp, rot, self.bt)
    out = dict(
      link_pos=lp, link_lin_vel=lin[:, self.links], heading=heading, yaw=torch.atan2(heading[:, 1, 0], heading[:, 0, 0]),
      dirs=bone.to_frame(r_w, heading), dirs_local=bone.robot_bone_dirs_local(r_w, rot, self.bt),
    )
    if self.ee_on:  # task link poses (env-local world frame, the state's p)
      out["task_pos"], out["task_rot"] = bone.robot_task_poses(lp, rot, self.bt)
    return out

  # ---- end-effector targets ------------------------------------------------------

  def cond_at(self, cond: torch.Tensor, steps: torch.Tensor) -> torch.Tensor:
    """cond [B,C] with its drift advanced by steps [B] (rate x steps x dt, clamped to the drift box)."""
    d = cond[:, 4:7] + cond[:, 7:10] * (steps.to(cond.dtype) * self.cfg.dt)[:, None]
    d = torch.maximum(torch.minimum(d, self.drift_box), -self.drift_box)
    return torch.cat([cond[:, :4], d, cond[:, 7:]], dim=-1)

  def ee_targets(self, h_task_pos: torch.Tensor, h_task_rot6d: torch.Tensor, cond: torch.Tensor, steps=None):
    """World targets of the task links: h_task_pos [B,(K,)L,3], h_task_rot6d [B,(K,)L,6], cond [B,C] at the first
    sample; steps [K] (optional): each sample's offset in steps, its drift extrapolated to its own time.
    -> positions [B,(K,)L,3], rotations [B,(K,)L,3,3], drift yaw [B,(K,)]."""
    c = self.cfg
    d = cond[:, 4:7]
    if steps is not None:
      d = d[:, None] + cond[:, None, 7:10] * (steps.to(cond.dtype) * c.dt)[None, :, None]
      d = torch.maximum(torch.minimum(d, self.drift_box), -self.drift_box)  # [B,K,3]
      anchor = cond[:, None, 2:4]
    else:
      anchor = cond[:, 2:4]
    R = rot_z(d[..., 2])  # [B,(K,)3,3]
    pivot = h_task_pos[..., TASK_TORSO : TASK_TORSO + 1, :2]  # each frame's own torso point
    rel = h_task_pos[..., :2] - pivot
    xy = torch.einsum("...ij,...lj->...li", R[..., :2, :2], rel) + pivot - anchor[..., None, :] + d[..., None, :2]
    pos = torch.cat([xy, h_task_pos[..., 2:]], dim=-1)
    rot = R[..., None, :, :] @ rot6d_to_matrix(h_task_rot6d)
    return pos, rot, d[..., 2]

  # ---- goal --------------------------------------------------------------------

  def goal(self, kin: dict, h_feat: torch.Tensor, h_yaw: torch.Tensor, h_task_pos=None, h_task_rot6d=None, cond=None) -> torch.Tensor:
    """h_feat [B,K,138], h_yaw [B,K] (ee_mode: h_task_pos [B,K,L,3], h_task_rot6d [B,K,L,6], cond [B,C] at the first
    sample) -> goal [B,K,G]."""
    parts = [h_feat]
    if self.cfg.goal_rel_yaw:
      d = h_yaw - kin["yaw"][:, None]
      parts += [torch.cos(d)[..., None], torch.sin(d)[..., None]]
    if self.ee_on:
      B, K = h_feat.shape[:2]
      steps = torch.arange(K, device=h_feat.device)
      pos, rot, _ = self.ee_targets(h_task_pos, h_task_rot6d, cond, steps)  # [B,K,L,3], [B,K,L,3,3]
      tp, tr = kin["task_pos"][:, TASK_TORSO], kin["task_rot"][:, TASK_TORSO]  # robot torso_link frame [B,3], [B,3,3]
      p_rel = torch.einsum("bji,bklj->bkli", tr, pos - tp[:, None, None, :])
      r_rel = torch.einsum("bji,bkljm->bklim", tr, rot)
      valid = cond[:, 1][:, None, None]
      ee = torch.cat([p_rel, r_rel[..., :, :2].flatten(-2)], dim=-1).flatten(-2) * valid
      parts += [ee, cond[:, None, 0:1].expand(B, K, 1), cond[:, None, 1:2].expand(B, K, 1)]
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)

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

  def ee_errors(self, kin: dict, ref: dict) -> dict:
    """Squared task-link errors of kin against the (drifted) targets of ref (h_task_pos, h_task_rot6d, cond, h_yaw)."""
    pos, rot, dyaw = self.ee_targets(ref["h_task_pos"], ref["h_task_rot6d"], ref["cond"])
    rp, rr = kin["task_pos"], kin["task_rot"]
    w, f, t = list(TASK_WRISTS), list(TASK_FEET), TASK_TORSO
    hh = rot_z(ref["h_yaw"] + dyaw)  # human heading frame (drifted), robot: hip-line heading
    rh = kin["heading"]
    f_h = torch.einsum("bji,blj->bli", hh, pos[:, f] - pos[:, t : t + 1])[..., :2]
    f_r = torch.einsum("bji,blj->bli", rh, rp[:, f] - rp[:, t : t + 1])[..., :2]
    return dict(
      torso_xy=((rp[:, t, :2] - pos[:, t, :2]) ** 2).sum(-1), torso_rot=rot_err2(rr[:, t], rot[:, t]),
      wrist_pos=((rp[:, w] - pos[:, w]) ** 2).sum(-1), wrist_rot=rot_err2(rr[:, w], rot[:, w]),
      foot_xy=((f_r - f_h) ** 2).sum(-1),
      foot_rot=rot_err2(rh.transpose(-1, -2)[:, None] @ rr[:, f], hh.transpose(-1, -2)[:, None] @ rot[:, f]),
    )

  def ee_terms(self, kin: dict, ref: dict) -> dict:
    """Task kernels (unweighted by alpha, x their w_ee_* weight)."""
    c = self.cfg
    e = self.ee_errors(kin, ref)
    k = lambda x, s: torch.exp(-x / s**2)  # noqa: E731
    out = {}
    if c.w_ee_torso_wide:
      out["ee_torso_wide"] = c.w_ee_torso_wide * 0.5 * (k(e["torso_xy"], c.std_ee_torso_wide_pos) + k(e["torso_rot"], c.std_ee_torso_wide_rot))
    if c.w_ee_torso:
      out["ee_torso"] = c.w_ee_torso * 0.5 * (k(e["torso_xy"], c.std_ee_torso_pos) + k(e["torso_rot"], c.std_ee_torso_rot))
    if c.w_ee_wrist_wide:
      out["ee_wrist_wide"] = c.w_ee_wrist_wide * k(e["wrist_pos"], c.std_ee_wrist_wide).mean(-1)
    if c.w_ee_wrist:
      out["ee_wrist"] = c.w_ee_wrist * k(e["wrist_pos"], c.std_ee_wrist).mean(-1)
    if c.w_ee_wrist_rot:
      out["ee_wrist_rot"] = c.w_ee_wrist_rot * k(e["wrist_rot"], c.std_ee_wrist_rot).mean(-1)
    if c.w_ee_foot:
      out["ee_foot"] = c.w_ee_foot * 0.5 * (k(e["foot_xy"], c.std_ee_foot_pos) + k(e["foot_rot"], c.std_ee_foot_rot)).mean(-1)
    if c.w_ee_clear:
      lift = kin["link_pos"][:, self.feet, 2] - self.ankle_h
      short = (c.ee_clear_height - lift).clamp(min=0.0)
      swing = 1.0 - ref["h_contact"]
      out["ee_clear"] = c.w_ee_clear * (1.0 - swing * (1.0 - k(short**2, c.std_ee_clear))).mean(-1)
    return out

  def foot_lift_err(self, kin: dict, ref: dict) -> torch.Tensor:
    """[B,2] robot ankle lift (above the standing ankle height) minus k x the human ankle lift."""
    k = (self.rest_h / ref["h_rest"])[:, None]
    return (kin["link_pos"][:, self.feet, 2] - self.ankle_h) - k * ref["h_foot_lift"]

  def soft_contact(self, kin: dict) -> torch.Tensor:
    """[B,2] soft contact of the (left, right) ankle links from their lift above the standing ankle height."""
    lift = kin["link_pos"][:, self.feet, 2] - self.ankle_h
    return torch.sigmoid((self.cfg.contact_height - lift) / self.cfg.contact_temp)

  def reward(self, kin_prev: dict, kin_next: dict, s_next: torch.Tensor, action: torch.Tensor, prev_action: torch.Tensor,
             ref: dict, return_terms: bool = False):
    c = self.cfg
    e = self._errors(kin_next, s_next, ref)
    std = torch.full((e["bone_g"].shape[-1],), c.std_bone, device=e["bone_g"].device)
    if c.std_bone_leg > 0:
      std = torch.where(self.leg_mask, torch.full_like(std, c.std_bone_leg), std)
    sb = torch.exp(-e["bone_g"] / std**2)
    sl = torch.exp(-e["bone_l"] / std**2)
    wg = self.w_global
    terms = {"bone_global_torso": c.w_bone_global_torso * (sb * wg).sum(-1) / wg.sum()}
    for g, wl in (("arm", c.w_bone_global_arm), ("leg", c.w_bone_global_leg)):
      if wl:
        m = self.w_global_limb[g]
        terms[f"bone_global_{g}"] = wl * (sb * m).sum(-1) / m.sum()
    for g, wl in zip(GROUPS, (c.w_bone_local_torso, c.w_bone_local_leg, c.w_bone_local_arm)):
      m = self.w_local[g]
      terms[f"bone_local_{g}"] = wl * (sl * m).sum(-1) / m.sum()
    terms["root_vel"] = c.w_root_vel * torch.exp(-e["root_vel"] / c.std_root_vel**2)
    terms["yaw_rate"] = c.w_yaw_rate * torch.exp(-e["yaw_rate"] / c.std_yaw_rate**2)
    if c.w_heading:
      terms["heading"] = c.w_heading * torch.exp(-e["heading"] / c.std_heading**2)
    terms["height"] = c.w_height * torch.exp(-e["height"] / c.std_height**2)
    if c.w_foot_lift:
      terms["foot_lift"] = c.w_foot_lift * torch.exp(-(self.foot_lift_err(kin_next, ref) ** 2) / c.std_foot_lift**2).mean(-1)
    if c.w_contact or c.w_slip:
      pc = self.soft_contact(kin_next)
      if c.w_contact:
        terms["contact"] = c.w_contact * (1.0 - (pc - ref["h_contact"]).abs().mean(-1))
      if c.w_slip:
        vxy = (kin_next["link_lin_vel"][:, self.feet, :2] ** 2).sum(-1)
        terms["slip"] = -c.w_slip * (pc * vxy).sum(-1)
    if self.ee_on:  # alpha blend: positive H4 terms x (1 - (1 - ee_style) alpha), task kernels x alpha
      a = ref["cond"][:, 0]
      keep = 1.0 - (1.0 - c.ee_style) * a
      terms = {n: v * keep for n, v in terms.items()}
      ts = c.ee_task_scale
      terms.update({n: v * (a * ts) for n, v in self.ee_terms(kin_next, ref).items()})
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
    if c.term_foot_lift > 0:
      fl = self.foot_lift_err(kin_next, ref).abs()  # [B,2]
      hard = hard | (fl > c.term_foot_lift).any(-1)
      alive = alive * torch.prod(1.0 - torch.sigmoid((fl - c.term_foot_lift) / c.term_foot_lift_temp), dim=-1)
    if self.ee_on and (c.term_ee_wrist > 0 or c.term_ee_torso > 0):
      e = self.ee_errors(kin_next, ref)
      inv_a = 1.0 / ref["cond"][:, 0].clamp(min=1e-3)  # threshold / alpha: off at alpha = 0
      for err2, th in ((e["wrist_pos"].amax(-1), c.term_ee_wrist), (e["torso_xy"], c.term_ee_torso)):
        if th > 0:
          err, thr = torch.sqrt(err2 + 1e-12), th * inv_a
          hard = hard | (err > thr)
          alive = alive * (1.0 - torch.sigmoid((err - thr) / c.term_ee_temp))
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
    for g, m in self.w_global_limb.items():  # limbs in the heading frame (the world-space pose a viewer sees)
      out[f"bone_global_{g}_deg"] = (deg_g * m).sum(-1) / m.sum()
    for g in GROUPS:
      m = self.w_local[g]
      out[f"bone_local_{g}_deg"] = (deg_l * m).sum(-1) / m.sum()
    out["root_vel_err"] = e["root_vel"].sqrt()
    out["yaw_rate_err"] = e["yaw_rate"].sqrt()
    out["heading_err_deg"] = torch.rad2deg(chord_angle(e["heading"]))
    out["height_err"] = e["height"].sqrt()
    out["foot_lift_err"] = self.foot_lift_err(kin, ref).abs().mean(-1)
    if self.ee_on:  # world task-link errors against the (drifted) targets: metres / degrees
      ee = self.ee_errors(kin, ref)
      deg = lambda x: torch.rad2deg(chord_angle(x))  # noqa: E731  (2 - 2 cos is a squared chord)
      out["ee_torso_xy_err"] = ee["torso_xy"].sqrt()
      out["ee_torso_rot_deg"] = deg(ee["torso_rot"])
      out["ee_wrist_err"] = ee["wrist_pos"].sqrt().mean(-1)
      out["ee_wrist_rot_deg"] = deg(ee["wrist_rot"]).mean(-1)
      out["ee_wrist_5cm"] = (ee["wrist_pos"].sqrt() < 0.05).float().mean(-1)
      out["ee_foot_xy_err"] = ee["foot_xy"].sqrt().mean(-1)
      out["alpha"] = ref["cond"][:, 0]
    return out
