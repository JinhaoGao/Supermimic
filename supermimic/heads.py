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
the paper): self-collision penalty removed, joint-limit weight -2. The paper
also drops BeyondMimic's end-effector termination; ``HeadsCfg.ee_termination``
(default on) restores it on both sides: the simulator's ``ee_body_pos`` term
(env_cfg) and a matching soft term in the imagined termination head below.
master's v2 spelling of the same switch (``ee_term_z``, 0 = off, temperature
``ee_term_temp``) is accepted by ``ee_flags`` (CLI) and ``HeadsCfg.from_dict``
(checkpoint cfgs); it is the same criterion, see ``HeadsCfg``.

Root-orientation termination (opt-in, ``HeadsCfg.root_ori_term``, 2026-10-25): the team benchmark fails a clip when
the pelvis' FULL rotation angle (yaw included) against the reference pelvis exceeds 1.0 rad (101 of 129 failures of
r15ws model_148000), while the anchor tilt test above sees only the torso's gravity direction (no yaw, no pelvis /
torso decoupling through the waist). With root_ori_term = A > 0 the simulator terminates at angle > A
(env_cfg ``root_ori``) and the imagined soft termination gets sigmoid((angle - A) / root_ori_term_temp) in its
noisy-OR. The pelvis is the robot's root body (free joint: G1 pelvis, E2 base_link), whose world-model state
quaternion is s[3:7] (= its FK pose); the reference quaternion is the tracked-body quaternion of that body.
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
  # End-effector termination (BeyondMimic ``ee_body_pos``, removed by the paper, restored 2026-10-22):
  # fail when any end-effector body (G1: ankle_roll / wrist_yaw links, E2: ankle_roll / elbow links; names
  # from the robot's tracking env cfg) is more than ee_term_z away from the reference in height (z only).
  # Off = the paper's recipe, bitwise identical to the code before the switch existed.
  # master (v2, e57be5b) spells the same criterion ee_term_z alone (0 = off, the default there) with the
  # temperature ee_term_temp; same bodies, threshold, temperature and noisy-OR soft term. Its checkpoint cfgs
  # load through from_dict, its CLI flag through ee_flags. Default here: ON at 0.25 m (the eval site's test).
  ee_termination: bool = True
  ee_term_z: float = 0.25
  ee_term_z_temp: float = 0.02  # same sigmoid temperature as the anchor-height term
  # Root-orientation termination (2026-10-25, opt-in): fail when the full rotation angle (incl. yaw) between the robot's
  # and the reference's root body exceeds root_ori_term [rad] (the eval site's root_ori test at 1.0); soft term
  # sigmoid((angle - root_ori_term) / root_ori_term_temp). 0 = off = the code before, bitwise. root_ori_body: "" = the
  # robot's root (free-joint) body, G1 pelvis / E2 base_link (env_cfg.ROOT_BODY), else a tracked body name.
  root_ori_term: float = 0.0
  root_ori_term_temp: float = 0.05
  root_ori_body: str = ""
  extra: dict = field(default_factory=dict)

  def __post_init__(self):
    if self.ee_termination and not self.ee_term_z > 0:
      raise ValueError(f"ee_termination needs a threshold ee_term_z > 0 (got {self.ee_term_z}); off: ee_termination=False")
    if not float(self.root_ori_term) >= 0.0:
      raise ValueError(f"root_ori_term must be >= 0 rad (0 = off), got {self.root_ori_term}")
    if self.root_ori_term > 0 and not self.root_ori_term_temp > 0:
      raise ValueError(f"root_ori_term needs a temperature root_ori_term_temp > 0 (got {self.root_ori_term_temp})")

  @property
  def root_ori_on(self) -> bool:
    return float(self.root_ori_term) > 0.0

  @classmethod
  def from_dict(cls, d: dict) -> "HeadsCfg":
    """HeadsCfg from a checkpoint's cfg["heads"] written by any branch:

    * ddp-bj31 (01c4d58 on): ee_termination / ee_term_z / ee_term_z_temp, taken as they are;
    * master v2 (e57be5b on, e.g. bj27 g1v2_scratch): ee_term_z (> 0 = on, 0 = off) / ee_term_temp;
    * before either (v1, paper recipe): no end-effector key -> off.
    When off, a non-positive ee_term_z is replaced by the default threshold (only used if switched on later)."""
    d = dict(d)
    if "ee_term_temp" in d:  # master v2 name of the soft-term temperature
      t = d.pop("ee_term_temp")
      d.setdefault("ee_term_z_temp", t)
    if "ee_termination" not in d:  # master v2 (switch = threshold > 0) or v1 (no key: off)
      d["ee_termination"] = float(d.get("ee_term_z", 0.0)) > 0
    if not d["ee_termination"] and not float(d.get("ee_term_z", 0.0)) > 0:
      d["ee_term_z"] = cls.ee_term_z
    return cls(**d)

  @property
  def reward_max_per_step(self) -> float:
    return (
      self.w_body_pos + self.w_body_ori + self.w_body_lin_vel + self.w_body_ang_vel + self.w_root_pos + self.w_root_ori
    ) * self.dt


def ee_flags(ee_term_z: float | None = None, ee_termination: bool | None = None, ee_threshold: float | None = None,
             default: tuple[bool, float] = (True, 0.25)) -> tuple[bool, float]:
  """(ee_termination, ee_term_z) from the two CLI spellings of the end-effector termination.

  ee_term_z: master's v2 flag (--ee_term_z Z: Z > 0 = on with threshold Z, 0 = off = paper recipe);
  ee_termination / ee_threshold: this branch's --ee_termination | --no_ee_termination and --ee_threshold.
  None = not given; nothing given -> default. Both spellings together must agree, else ValueError."""
  if ee_term_z is not None:
    z = float(ee_term_z)
    if not z >= 0:
      raise ValueError(f"--ee_term_z must be >= 0 (0 = off), got {ee_term_z}")
    on = z > 0
    if ee_termination is not None and bool(ee_termination) != on:
      raise ValueError(f"--ee_term_z {z:g} ({'on' if on else 'off'}) contradicts --{'' if ee_termination else 'no_'}ee_termination")
    if on and ee_threshold is not None and abs(float(ee_threshold) - z) > 1e-12:
      raise ValueError(f"--ee_term_z {z:g} contradicts --ee_threshold {ee_threshold:g}")
    thr = z if on else (float(ee_threshold) if ee_threshold is not None else default[1])
    return on, thr
  on = default[0] if ee_termination is None else bool(ee_termination)
  thr = default[1] if ee_threshold is None else float(ee_threshold)
  if on and not thr > 0:
    raise ValueError(f"--ee_threshold must be > 0, got {thr}")
  return on, thr


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
    root_ori_is_root: bool = True,
  ):
    self.fk = fk
    self.cfg = cfg
    self.device = fk.device
    self.tracked = torch.as_tensor(tracked_local_ids, device=self.device)
    self.anchor = anchor_index  # index within the tracked-body list
    self.anchor_local = int(tracked_local_ids[anchor_index])
    # End-effector bodies as indices into the tracked-body list (order of the tracked list, like mjlab).
    self.ee = torch.as_tensor(list(ee_indices or ()), dtype=torch.long, device=self.device)
    if cfg.ee_termination and self.ee.numel() == 0:
      raise ValueError("HeadsCfg.ee_termination is on but no end-effector bodies were given")
    # Root-orientation body (pelvis) as an index into the tracked-body list (SimInterface.root_ori_body_of); None = not
    # tracked (no root-orientation term, no pelvis-angle logging). root_ori_is_root: it is the free-joint body, whose
    # quaternion is the state's s[3:7] (bitwise equal to its FK quaternion).
    self.root_ori = None if root_ori_index is None else int(root_ori_index)
    self.root_ori_is_root = bool(root_ori_is_root)
    if cfg.root_ori_on and self.root_ori is None:
      raise ValueError("HeadsCfg.root_ori_term is on but no root-orientation body (tracked) was given")
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

  def root_ori_angle(self, kin: dict, ref_rq: torch.Tensor) -> torch.Tensor:
    """Full rotation angle [B] (incl. yaw, the eval site's root_ori error) between the robot's root-orientation body
    (kin, from the state) and its reference quaternion ref_rq [B,4]."""
    return quat_angle(ref_rq, kin["body_quat"][:, self.root_ori])

  @torch.no_grad()
  def root_ori_angle_state(self, s: torch.Tensor, ref_bq: torch.Tensor) -> torch.Tensor | None:
    """Logging (no grad): root_ori_angle of the states s [B,D] against the tracked-body reference quaternions ref_bq
    [B,Bt,4]; the root body reads s[3:7] directly (= its FK quaternion, no FK pass). None when the body is not tracked.
    Non-finite states give NaN (> any threshold is False)."""
    if self.root_ori is None:
      return None
    q = s[:, 3:7] if self.root_ori_is_root else self.kin(s)["body_quat"][:, self.root_ori]
    return quat_angle(ref_bq[:, self.root_ori], q)

  def termination(self, kin_next: dict, ref_ap: torch.Tensor, ref_aq: torch.Tensor, ref_bp: torch.Tensor | None = None,
                  scale: torch.Tensor | None = None, ref_rq: torch.Tensor | None = None):
    """Hard (simulator) and soft (imagination) termination at the target frame.

    ref_ap [B,3] / ref_aq [B,4]: reference anchor pose; ref_bp [B,Bt,3]: reference positions of all tracked
    bodies (needed only with ``ee_termination``); ref_rq [B,4] (keyword, needed only with ``root_ori_term``): reference
    quaternion of the root-orientation body (the pelvis). Returns (hard_bool [B], soft_prob [B]).

    Each criterion i has a hard test x_i > thr_i and a soft probability p_i = sigmoid((x_i - thr_i) / temp_i);
    the soft termination is the noisy-OR 1 - prod_i (1 - p_i) over anchor height, anchor tilt, (optional) every
    end-effector height error and (optional) the root body's full rotation angle (kin_next's quaternion of that body,
    = the state's root quaternion for the pelvis), and the hard one is the OR of the tests (= the soft one as temp -> 0).
    The end-effector height error |ref_z - z| equals mjlab's ``bad_motion_body_pos_z_only``: its re-anchored
    reference (``body_pos_relative_w``) only shifts xy and rotates about z, so its z is the raw reference z.

    scale [B] (optional, master 4314a50): per-sample multiplier of every threshold AND sigmoid temperature (the
    hard-clip termination curriculum, LibraryMotionCommand.term_scale_env; the env terms use the same factor).
    None = the code before the curriculum, bitwise (python factor 1.0, same graph); a tensor of ones gives the
    same numbers. (master's version takes (kin_next, ref_bp, ref_bq, scale) and slices the anchor itself; the
    numbers are the same.)"""
    c = self.cfg
    B = ref_ap.shape[0]
    g = self.gravity.expand(B, 3)
    k = 1.0 if scale is None else scale  # [B]
    dz = (ref_ap[:, 2] - kin_next["anchor_pos"][:, 2]).abs()
    tilt = (quat_apply_inverse(ref_aq, g)[:, 2] - quat_apply_inverse(kin_next["anchor_quat"], g)[:, 2]).abs()
    hard = (dz > c.term_z * k) | (tilt > c.term_tilt * k)
    pz = torch.sigmoid((dz - c.term_z * k) / (c.term_z_temp * k))
    pt = torch.sigmoid((tilt - c.term_tilt * k) / (c.term_tilt_temp * k))
    alive = (1.0 - pz) * (1.0 - pt)
    if c.ee_termination:
      assert ref_bp is not None, "ee_termination needs the reference body positions"
      ke = 1.0 if scale is None else scale[:, None]  # [B, 1] against [B, n_ee]
      dz_ee = (ref_bp[:, self.ee, 2] - kin_next["body_pos"][:, self.ee, 2]).abs()  # [B, n_ee]
      hard = hard | (dz_ee > c.ee_term_z * ke).any(-1)
      p_ee = torch.sigmoid((dz_ee - c.ee_term_z * ke) / (c.ee_term_z_temp * ke))
      alive = alive * torch.prod(1.0 - p_ee, dim=-1)
    if c.root_ori_on:
      assert ref_rq is not None, "root_ori_term needs the reference root-orientation quaternion (ref_rq)"
      ang = self.root_ori_angle(kin_next, ref_rq)  # [B]
      hard = hard | (ang > c.root_ori_term * k)
      p_ro = torch.sigmoid((ang - c.root_ori_term * k) / (c.root_ori_term_temp * k))
      alive = alive * (1.0 - p_ro)
    soft = 1.0 - alive
    return hard, soft
