"""Deploy-side goal of a human-reference SuperMimic policy (--reference human, H4 or ee_mode), numpy only.

No torch / mjlab / mujoco: the robot loop imports this file (or a copy of it) with the meta json and the per-clip
reference npz written by scripts/human/export_deploy_human.py. Each 50 Hz step:

  b = HumanGoalBuilder.from_files("meta.json", "clips/<clip>.npz")
  b.start(frame0, q, imu_quat, odom)                        # align the reference onto the robot (episode start)
  goal = b.step(q, imu_quat, frame, alpha_cmd, odom)        # [1, K, goal_frame] float32 -> ONNX input "goal"

  q         [nj] joint positions in meta["joint_names"] order (the policy order)
  imu_quat  [4] wxyz pelvis orientation (the IMU of the root link; roll / pitch are used, yaw only without odometry)
  odom      None, or (x, y, yaw) of the pelvis from leg odometry (any fixed world frame); base_z (optional kwarg): pelvis
            height above the floor, else estimated from the lower ankle (that foot assumed flat on the floor)
  alpha_cmd ee_mode: requested alpha in [0, 1]; the applied alpha moves towards it at <= alpha_rate per second (training
            ramps an alpha switch over 0.5 s) and is forced to 0 (ee_valid = 0, EE block zeroed) while odom is None

Goal frame (HumanHeads.goal, drift 0), K samples at frames frame..frame+K-1 (clamped at the clip end):
  h_feat 138 | cos, sin of (human heading - robot hip-line heading) 2 | ee_mode: 5 task links (torso, wrists, feet) in
  the robot's torso_link frame, pos 3 + rot6d 6 each, 45 (x ee_valid) | alpha 1 | ee_valid 1
The world-frame quantities live in the "training world": the human world shifted so that the human pelvis xy of the
alignment frame sits at the robot's pelvis, the robot yaw mapped so that its pelvis yaw equals the IK reset pose's
(init_root_quat) at that frame -- what an episode reset at that frame does in training (command._resample_command).
Alignment:
  * start(): both xy and yaw.
  * the yaw source switches (odometry appears / is lost): the yaw offset is re-chosen so the robot's training-world yaw
    does not jump (rel_yaw stays continuous).
  * an EE segment starts (applied alpha leaves 0) with realign_on_activate (default): xy only, so the world targets
    start at the robot instead of wherever H4 tracking has drifted it; off = keep the start anchor (true world tracking).
  * realign(): both, on demand.
"""

from __future__ import annotations

import json

import numpy as np

GOAL_FEAT_DIM = 138


# ---- small rotation helpers ------------------------------------------------------


def quat_to_matrix(q) -> np.ndarray:
  """wxyz [..., 4] -> [..., 3, 3]."""
  q = np.asarray(q, dtype=np.float64)
  q = q / np.linalg.norm(q, axis=-1, keepdims=True)
  w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
  m = np.stack([
    1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
    2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
    2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
  ], axis=-1)  # fmt: skip
  return m.reshape(*q.shape[:-1], 3, 3)


def axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
  x, y, z = axis
  c, s = np.cos(angle), np.sin(angle)
  t = 1.0 - c
  return np.array([[t * x * x + c, t * x * y - s * z, t * x * z + s * y],
                   [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
                   [t * x * z - s * y, t * y * z + s * x, t * z * z + c]])  # fmt: skip


def rot_z(yaw: float) -> np.ndarray:
  c, s = np.cos(yaw), np.sin(yaw)
  return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def yaw_of(R: np.ndarray) -> float:
  return float(np.arctan2(R[1, 0], R[0, 0]))


def wrap(a: float) -> float:
  return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def rot6d_to_matrix(r6: np.ndarray) -> np.ndarray:
  """[..., 6] (first two columns, row-major 3x2) -> [..., 3, 3] (Gram-Schmidt), as human.heads.rot6d_to_matrix."""
  m = r6.reshape(*r6.shape[:-1], 3, 2)
  a = m[..., 0] / np.maximum(np.linalg.norm(m[..., 0], axis=-1, keepdims=True), 1e-6)
  b = m[..., 1] - (a * m[..., 1]).sum(-1, keepdims=True) * a
  b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-6)
  return np.stack([a, b, np.cross(a, b)], axis=-1)


# ---- forward kinematics -------------------------------------------------------------


class RobotKinematics:
  """MuJoCo mj_kinematics of the bodies in meta["fk"]["bodies"] (parent-first, root = the free-joint pelvis): the same
  computation as supermimic.fk.TorchFK (hinge about its anchor, angle q - qpos0, fixed bodies)."""

  def __init__(self, fk: dict, joint_names: list[str]):
    self.names = [b["name"] for b in fk["bodies"]]
    self.parent = [int(b["parent"]) for b in fk["bodies"]]
    self.pos = [np.asarray(b["pos"], dtype=np.float64) for b in fk["bodies"]]
    self.rot = [quat_to_matrix(b["quat"]) for b in fk["bodies"]]
    self.joint = [joint_names.index(b["joint"]) if b["joint"] else -1 for b in fk["bodies"]]
    self.axis = [np.asarray(b["axis"], dtype=np.float64) for b in fk["bodies"]]
    self.jpos = [np.asarray(b["jnt_pos"], dtype=np.float64) for b in fk["bodies"]]
    self.qpos0 = [float(b["qpos0"]) for b in fk["bodies"]]
    assert self.parent[0] == -1 and all(0 <= p < i for i, p in enumerate(self.parent) if i), "parent-first, root first"
    self.index = {n: i for i, n in enumerate(self.names)}

  def forward(self, root_pos: np.ndarray, root_rot: np.ndarray, q: np.ndarray):
    """-> positions [n,3], rotations [n,3,3] (world) of the bodies in meta order."""
    n = len(self.names)
    P, R = np.zeros((n, 3)), np.zeros((n, 3, 3))
    P[0], R[0] = root_pos, root_rot
    for i in range(1, n):
      pp, pr = P[self.parent[i]], R[self.parent[i]]
      r0 = pr @ self.rot[i]
      p0 = pp + pr @ self.pos[i]
      j = self.joint[i]
      if j < 0:
        P[i], R[i] = p0, r0
        continue
      r = r0 @ axis_angle_matrix(self.axis[i], float(q[j]) - self.qpos0[i])
      anchor = p0 + r0 @ self.jpos[i]
      P[i], R[i] = anchor - r @ self.jpos[i], r
    return P, R


# ---- goal builder -------------------------------------------------------------------


def load_clip(path: str) -> dict:
  with np.load(path, allow_pickle=False) as d:
    return {k: np.asarray(d[k]) for k in d.files}


class HumanGoalBuilder:
  def __init__(self, meta: dict, clip: dict, alpha_rate: float = 1.0, realign_on_activate: bool = True):
    self.meta, self.clip = meta, clip
    self.K = int(meta["K"])
    self.dt = float(meta["control_dt"])
    self.layout = [(str(n), int(k)) for n, k in meta["goal_layout"]]
    self.goal_frame = sum(k for _, k in self.layout)
    names = [n for n, _ in self.layout]
    self.rel_yaw = "rel_yaw" in names
    self.ee_mode = "ee" in names
    self.joint_names = list(meta["joint_names"])
    self.kin = RobotKinematics(meta["fk"], self.joint_names)
    ix = self.kin.index
    self.heading = [ix[n] for n in meta["heading_links"]]  # left, right hip links
    self.feet = [ix[n] for n in meta["foot_links"]]
    self.task_points = [[ix[n] for n in t["points"]] for t in meta["task_links"]]
    self.task_frame = [ix[t["frame"]] for t in meta["task_links"]]
    self.ankle_height = float(meta["robot_ankle_height"])
    self.T = int(clip["h_feat"].shape[0])
    self.h_feat = clip["h_feat"].astype(np.float64)
    self.h_yaw = clip["h_yaw"].astype(np.float64)
    if self.ee_mode:
      self.h_task_pos = clip["h_task_pos"].astype(np.float64)
      self.h_task_rot = rot6d_to_matrix(clip["h_task_rot6d"].astype(np.float64))  # [T,L,3,3]
      self.h_root_xy = clip["h_root_xy"].astype(np.float64)
    self.init_yaw = np.array([yaw_of(R) for R in quat_to_matrix(clip["init_root_quat"])])
    self.alpha_rate = float(alpha_rate)
    self.realign_on_activate = bool(realign_on_activate)
    self.alpha = 0.0
    self.ee_valid = 0.0
    self._src = None  # yaw source of the alignment: "imu" | "odom"
    self._yaw_off = 0.0  # training-world yaw = source yaw + _yaw_off
    self._xy_src = np.zeros(2)  # source pelvis xy at the xy alignment ...
    self._xy_train = np.zeros(2)  # ... and where it sits in the training world
    self._anchor = np.zeros(2)  # cond anchor xy (human pelvis xy of the xy alignment frame)

  @classmethod
  def from_files(cls, meta_json: str, clip_npz: str, **kw) -> "HumanGoalBuilder":
    with open(meta_json) as f:
      meta = json.load(f)
    return cls(meta, load_clip(clip_npz), **kw)

  # ---- pure goal (training-world robot pose) ----------------------------------------

  def window(self, frame: int) -> np.ndarray:
    return np.minimum(int(frame) + np.arange(self.K), self.T - 1)

  def robot_poses(self, q, pelvis_pos, pelvis_rot) -> dict:
    """FK -> hip-line yaw, task-link poses, ankle heights of the robot at that root pose."""
    P, R = self.kin.forward(np.asarray(pelvis_pos, dtype=np.float64), np.asarray(pelvis_rot, dtype=np.float64), np.asarray(q, dtype=np.float64))
    left = P[self.heading[0]] - P[self.heading[1]]
    out = {"yaw": float(np.arctan2(-left[0], left[1])), "ankle_z": P[self.feet, 2]}  # forward = left x up
    out["task_pos"] = np.stack([P[ids].mean(0) for ids in self.task_points])
    out["task_rot"] = R[self.task_frame]
    return out

  def goal_world(self, q, pelvis_pos, pelvis_rot, frame: int, alpha: float = 0.0, ee_valid: float = 0.0, anchor_xy=(0.0, 0.0)) -> np.ndarray:
    """[K, goal_frame] goal of the robot at pelvis_pos / pelvis_rot (training world): HumanHeads.goal with
    cond = [alpha, ee_valid, anchor_xy, no drift]."""
    idx = self.window(frame)
    rob = self.robot_poses(q, pelvis_pos, pelvis_rot)
    parts = [self.h_feat[idx]]
    if self.rel_yaw:
      d = self.h_yaw[idx] - rob["yaw"]
      parts += [np.cos(d)[:, None], np.sin(d)[:, None]]
    if self.ee_mode:
      tp, tr = rob["task_pos"][0], rob["task_rot"][0]  # torso_link (task link 0)
      pos = self.h_task_pos[idx].copy()  # [K,L,3]
      pos[..., :2] -= np.asarray(anchor_xy, dtype=np.float64)
      p_rel = np.einsum("ji,klj->kli", tr, pos - tp)
      r_rel = np.einsum("ji,kljm->klim", tr, self.h_task_rot[idx])
      ee = np.concatenate([p_rel, r_rel[..., :, :2].reshape(self.K, -1, 6)], axis=-1).reshape(self.K, -1) * float(ee_valid)
      parts += [ee, np.full((self.K, 1), float(alpha)), np.full((self.K, 1), float(ee_valid))]
    g = np.concatenate(parts, axis=-1)
    assert g.shape == (self.K, self.goal_frame), (g.shape, self.goal_frame)
    return g

  # ---- deploy state ------------------------------------------------------------------

  def _source_pose(self, q, imu_quat, odom, base_z):
    """Pelvis pose in the source frame (odometry, or the IMU's own frame with xy = 0) and the yaw source."""
    R = quat_to_matrix(imu_quat)
    if odom is not None:
      x, y, yaw = (float(v) for v in odom)
      R = rot_z(yaw - yaw_of(R)) @ R  # IMU roll / pitch, odometry yaw
      xy, src = np.array([x, y]), "odom"
    else:
      xy, src = np.zeros(2), "imu"
    if base_z is None:  # lower ankle link at its standing height above the floor
      z = self.ankle_height - float(self.robot_poses(q, np.zeros(3), R)["ankle_z"].min())
    else:
      z = float(base_z)
    return xy, z, R, src

  def _to_train(self, xy, z, R):
    Rz = rot_z(self._yaw_off)
    p = np.concatenate([Rz[:2, :2] @ (xy - self._xy_src) + self._xy_train, [z]])
    return p, Rz @ R

  def _align_xy(self, frame: int, xy_src) -> None:
    self._xy_src = np.asarray(xy_src, dtype=np.float64).copy()
    self._xy_train = np.zeros(2)
    if self.ee_mode:
      self._anchor = self.h_root_xy[min(int(frame), self.T - 1)].copy()

  def realign(self, frame: int, q, imu_quat, odom=None, base_z=None) -> None:
    """Reference aligned onto the robot as at an episode start at `frame` (xy and yaw)."""
    xy, _, R, src = self._source_pose(q, imu_quat, odom, base_z)
    self._src = src
    self._yaw_off = wrap(self.init_yaw[min(int(frame), self.T - 1)] - yaw_of(R))
    self._align_xy(frame, xy)
    self._last_yaw_train, self._last_xy_train = yaw_of(rot_z(self._yaw_off) @ R), np.zeros(2)

  def start(self, frame: int, q, imu_quat, odom=None, base_z=None, alpha: float = 0.0) -> None:
    """Episode start: align at `frame`; the applied alpha starts at `alpha` (default 0: ramp up from H4)."""
    self.realign(frame, q, imu_quat, odom, base_z)
    self.alpha = float(np.clip(alpha, 0.0, 1.0)) if (self.ee_mode and odom is not None) else 0.0
    self.ee_valid = 1.0 if self.alpha > 0 else 0.0

  def step(self, q, imu_quat, frame: int, alpha_cmd: float = 0.0, odom=None, base_z=None) -> np.ndarray:
    """Goal [1, K, goal_frame] float32 of this control step (see the module docstring)."""
    xy, z, R, src = self._source_pose(q, imu_quat, odom, base_z)
    if self._src is None:
      self.realign(frame, q, imu_quat, odom, base_z)
    elif src != self._src:  # yaw source switched: continue from the last training-world pose (no jump)
      yaw_now = yaw_of(rot_z(self._yaw_off) @ quat_to_matrix(imu_quat)) if self._src == "imu" else self._last_yaw_train
      self._src = src
      self._yaw_off = wrap(yaw_now - yaw_of(R))
      self._xy_src, self._xy_train = xy.copy(), self._last_xy_train.copy()
    if self.ee_mode:
      prev = self.alpha
      if odom is None:
        self.alpha = 0.0
      else:
        tgt = float(np.clip(alpha_cmd, 0.0, 1.0))
        dmax = self.alpha_rate * self.dt
        self.alpha = float(np.clip(tgt, prev - dmax, prev + dmax))
      if prev == 0.0 and self.alpha > 0.0 and self.realign_on_activate:  # an EE segment starts
        self._align_xy(frame, xy)
      self.ee_valid = 1.0 if self.alpha > 0.0 else 0.0
    p, Rt = self._to_train(xy, z, R)
    self._last_yaw_train, self._last_xy_train = yaw_of(Rt), p[:2].copy()
    g = self.goal_world(q, p, Rt, frame, self.alpha, self.ee_valid, self._anchor)
    return g[None].astype(np.float32)
