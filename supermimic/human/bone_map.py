"""Scale-free SMPL <-> robot bone correspondence (ported from Noetix-RGMT utils/smpl_bone_map.py, P1).

Every segment is a unit direction on both skeletons.

Human side
 * point segment: unit vector between two SMPL joints (scale-free);
 * frame segment: an axis rigidly attached to an SMPL joint frame (palm normal, foot
  side axis). It covers twist that point segments cannot see.

Robot side
 * geometric segment: unit vector between two link origins (both joint centres exist);
 * calibrated segment: a fixed axis of the carrying link, ``d = R_link @ c``.

Calibration (Noetix-RGMT ``scripts/calibrate_smpl_bone_map.py``; mjlab's G1 has the same link frames) makes the robot T-pose match the
SMPL rest pose for every calibrated/frame axis. Directions are compared

 * globally: each skeleton in its own heading frame (yaw from the hip line, z up);
 * locally: each bone in its parent's frame (adjacent-link relation), using the fixed
  parent-frame correspondence ``Q`` measured in the T-pose.

Only torch is imported so offline scripts can load this file by path.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import torch

# SMPL-X body joints 0..21, then left/right hand points (SMPL-X middle1, joints 28/43).
SMPL_JOINT_NAMES = [
  "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2", "left_ankle",
  "right_ankle", "spine3", "left_foot", "right_foot", "neck", "left_collar", "right_collar", "head",
  "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
  "left_hand", "right_hand",
]  # fmt: skip
SMPLX_HAND_POINT_INDICES = (28, 43)
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
SMPL_HIP_JOINTS = (1, 2)  # left, right: define the human heading frame
SMPL_FOOT_JOINTS = (10, 11)  # left, right: floor reference and contact points
SMPL_PELVIS = 0


@dataclass
class BoneSegment:
  name: str
  group: str  # torso / leg / arm
  robot_link: str  # carrying link (calibrated) or start link (geometric)
  weight: float = 1.0
  human_from: str | None = None  # point segment
  human_to: str | None = None
  human_frame: str | None = None  # frame segment: SMPL joint carrying the axis
  tpose_axis: tuple[float, float, float] | None = None  # that axis in heading coords (fwd, left, up) at rest
  robot_to_link: str | None = None  # set -> geometric robot segment
  human_parent: str | None = None  # set -> local (adjacent-link) term against this SMPL joint frame
  robot_parent: str | None = None  # ... and this robot link frame


@dataclass
class TaskLink:
  """A task-space target: a point and a frame on both skeletons (metric, never scaled).

  Points are the mean of the listed SMPL joints / robot link origins; frames are one SMPL
  joint rotation and one robot link rotation, related by a fixed offset from calibration.
  """

  name: str
  human_points: tuple[str, ...]
  human_frame: str
  robot_points: tuple[str, ...]
  robot_frame: str


@dataclass
class SmplBoneMapCfg:
  robot: str
  pelvis_link: str
  heading_links: tuple[str, str]  # left, right
  foot_links: tuple[str, str]  # left, right
  segments: list[BoneSegment]
  task_links: list[TaskLink] = field(default_factory=list)
  tpose_joint_pos: dict[str, float] = field(default_factory=dict)  # initial guess for the calibration pose
  # Calibration-pose fit: these joints are solved so the robot matches the SMPL rest pose on the
  # geometric segments plus the anatomical axes below (link-frame axis -> segment it must match).
  rest_fit_joints: list[str] = field(default_factory=list)
  rest_fit_axes: dict[str, tuple[str, tuple[float, float, float]]] = field(default_factory=dict)
  # ... and these robot joint axes must lie along the given rest-pose line (heading coords, sign-free).
  rest_fit_joint_axes: dict[str, tuple[float, float, float]] = field(default_factory=dict)
  default_joint_pos: dict[str, float] = field(default_factory=dict)  # IK regularization target
  calib_file: str = ""

  @property
  def segment_names(self) -> list[str]:
    return [s.name for s in self.segments]

  @property
  def robot_links(self) -> list[str]:
    """Every robot link referenced by the map, in a fixed order."""
    names: list[str] = []
    for s in self.segments:
      for n in (s.robot_link, s.robot_to_link, s.robot_parent):
        if n is not None and n not in names:
          names.append(n)
    for n in (self.pelvis_link, *self.heading_links, *self.foot_links):
      if n not in names:
        names.append(n)
    for t in self.task_links:
      for n in (*t.robot_points, t.robot_frame):
        if n not in names:
          names.append(n)
    return names

  def load_calibration(self) -> dict:
    with open(self.calib_file) as f:
      calib = json.load(f)
    if calib["segments"] != self.segment_names or calib["robot_links"] != self.robot_links:
      raise ValueError(f"{self.calib_file} does not match the bone map; re-run calibrate_smpl_bone_map.py")
    return calib


_CALIB_DIR = os.path.dirname(os.path.abspath(__file__))


def _g1_side(side: str) -> list[BoneSegment]:
  s, p = side[0], side
  return [
    # Thigh is calibrated, not geometric: G1's hips are wider than a human's, so a neutral
    # human thigh slants inwards while the robot's is vertical; the T-pose absorbs that.
    BoneSegment(f"thigh_{s}", "leg", f"{p}_hip_yaw_link", 1.0, f"{side}_hip", f"{side}_knee",
          human_parent="pelvis", robot_parent="pelvis"),
    BoneSegment(f"shin_{s}", "leg", f"{p}_knee_link", 1.0, f"{side}_knee", f"{side}_ankle",
          robot_to_link=f"{p}_ankle_roll_link", human_parent=f"{side}_hip", robot_parent=f"{p}_hip_yaw_link"),
    BoneSegment(f"foot_{s}", "leg", f"{p}_ankle_roll_link", 0.5, f"{side}_ankle", f"{side}_foot",
          human_parent=f"{side}_knee", robot_parent=f"{p}_knee_link"),
    BoneSegment(f"foot_side_{s}", "leg", f"{p}_ankle_roll_link", 0.5, human_frame=f"{side}_ankle",
          tpose_axis=(0.0, 1.0, 0.0), human_parent=f"{side}_knee", robot_parent=f"{p}_knee_link"),
    BoneSegment(f"upper_arm_{s}", "arm", f"{p}_shoulder_roll_link", 1.0, f"{side}_shoulder", f"{side}_elbow",
          robot_to_link=f"{p}_elbow_link", human_parent="spine3", robot_parent="torso_link"),
    BoneSegment(f"forearm_{s}", "arm", f"{p}_elbow_link", 1.0, f"{side}_elbow", f"{side}_wrist",
          robot_to_link=f"{p}_wrist_pitch_link", human_parent=f"{side}_shoulder",
          robot_parent=f"{p}_shoulder_yaw_link"),
    BoneSegment(f"hand_{s}", "arm", f"{p}_wrist_yaw_link", 0.5, f"{side}_wrist", f"{side}_hand",
          human_parent=f"{side}_elbow", robot_parent=f"{p}_elbow_link"),
    # Palm normal (palms face down in the T-pose): fixes forearm/upper-arm twist on a straight arm.
    BoneSegment(f"palm_{s}", "arm", f"{p}_wrist_yaw_link", 0.5, human_frame=f"{side}_wrist",
          tpose_axis=(0.0, 0.0, -1.0), human_parent=f"{side}_elbow", robot_parent=f"{p}_elbow_link"),
  ]  # fmt: skip


G1_SMPL_BONE_MAP = SmplBoneMapCfg(
  robot="g1",
  pelvis_link="pelvis",
  heading_links=("left_hip_roll_link", "right_hip_roll_link"),
  foot_links=("left_ankle_roll_link", "right_ankle_roll_link"),
  segments=[
    BoneSegment("pelvis_up", "torso", "pelvis", 1.0, "pelvis", "spine1"),
    BoneSegment("pelvis_left", "torso", "right_hip_roll_link", 1.0, "right_hip", "left_hip",
          robot_to_link="left_hip_roll_link"),
    BoneSegment("torso_up", "torso", "torso_link", 1.0, "spine3", "neck",
          human_parent="pelvis", robot_parent="pelvis"),
    BoneSegment("torso_left", "torso", "right_shoulder_roll_link", 1.0, "right_shoulder", "left_shoulder",
          robot_to_link="left_shoulder_roll_link", human_parent="pelvis", robot_parent="pelvis"),
    *_g1_side("left"),
    *_g1_side("right"),
  ],  # fmt: skip
  task_links=[
    TaskLink("torso", ("left_shoulder", "right_shoulder"), "spine3",
        ("left_shoulder_roll_link", "right_shoulder_roll_link"), "torso_link"),
    TaskLink("wrist_l", ("left_wrist",), "left_wrist", ("left_wrist_pitch_link",), "left_wrist_yaw_link"),
    TaskLink("wrist_r", ("right_wrist",), "right_wrist", ("right_wrist_pitch_link",), "right_wrist_yaw_link"),
    TaskLink("foot_l", ("left_ankle",), "left_ankle", ("left_ankle_roll_link",), "left_ankle_roll_link"),
    TaskLink("foot_r", ("right_ankle",), "right_ankle", ("right_ankle_roll_link",), "right_ankle_roll_link"),
  ],  # fmt: skip
  tpose_joint_pos={
    "left_shoulder_pitch_joint": 0.2006477150487628,
    "left_shoulder_roll_joint": 1.476490678407796,
    "left_shoulder_yaw_joint": 0.8742948427729365,
    "left_elbow_joint": 1.414980653432781,
    "right_shoulder_pitch_joint": 0.2006477150487628,
    "right_shoulder_roll_joint": -1.476490678407796,
    "right_shoulder_yaw_joint": -0.8742948427729365,
    "right_elbow_joint": 1.414980653432781,
  },
  rest_fit_joints=[
    f"{side}_{j}_joint"
    for side in ("left", "right")
    for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll")
  ],
  # G1 palms face the body midline in the zero pose (wrist_yaw_link axis -y left / +y right).
  rest_fit_axes={
    "palm_l": ("left_wrist_yaw_link", (0.0, -1.0, 0.0)),
    "palm_r": ("right_wrist_yaw_link", (0.0, 1.0, 0.0)),
  },
  # SMPL elbows flex about the vertical in the palms-down rest pose (principal axis of AMASS elbow
  # rotations); without this the shoulder yaw of a straight calibration arm is unobservable.
  rest_fit_joint_axes={"left_elbow_joint": (0.0, 0.0, 1.0), "right_elbow_joint": (0.0, 0.0, 1.0)},
  default_joint_pos={
    "left_hip_pitch_joint": -0.312, "right_hip_pitch_joint": -0.312,
    "left_knee_joint": 0.669, "right_knee_joint": 0.669,
    "left_ankle_pitch_joint": -0.363, "right_ankle_pitch_joint": -0.363,
    "left_elbow_joint": 0.6, "right_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2, "right_shoulder_roll_joint": -0.2,
    "left_shoulder_pitch_joint": 0.2, "right_shoulder_pitch_joint": 0.2,
  },  # fmt: skip
  calib_file=os.path.join(_CALIB_DIR, "smpl_bone_calib_g1.json"),
)

SMPL_BONE_MAPS = {"g1": G1_SMPL_BONE_MAP}


# ---------------------------------------------------------------------------
# Index/calibration tensors for fast evaluation
# ---------------------------------------------------------------------------


class BoneMapTensors:
  """Per-segment index arrays and calibration constants on one device.

  Robot quantities are passed as ``[..., L, *]`` arrays over ``bone_map.robot_links``.
  """

  def __init__(self, bone_map: SmplBoneMapCfg, calib: dict | None, device="cpu", dtype=torch.float32):
    segs = bone_map.segments
    links = bone_map.robot_links
    j = SMPL_JOINT_NAMES.index
    t = lambda x, dt=torch.long: torch.tensor(x, dtype=dt, device=device)  # noqa: E731
    self.names = bone_map.segment_names
    self.groups = [s.group for s in segs]
    self.weights = t([s.weight for s in segs], dtype)
    self.is_point = t([s.human_from is not None for s in segs], torch.bool)
    self.geometric = t([s.robot_to_link is not None for s in segs], torch.bool)
    self.has_parent = t([s.human_parent is not None for s in segs], torch.bool)
    self.h_src = t([j(s.human_from) if s.human_from else 0 for s in segs])
    self.h_dst = t([j(s.human_to) if s.human_to else 0 for s in segs])
    self.h_frame = t([j(s.human_frame) if s.human_frame else 0 for s in segs])
    self.h_parent = t([j(s.human_parent) if s.human_parent else 0 for s in segs])
    self.r_from = t([links.index(s.robot_link) for s in segs])
    self.r_to = t([links.index(s.robot_to_link or s.robot_link) for s in segs])
    self.r_parent = t([links.index(s.robot_parent or s.robot_link) for s in segs])
    self.r_pelvis = links.index(bone_map.pelvis_link)
    self.r_heading = (links.index(bone_map.heading_links[0]), links.index(bone_map.heading_links[1]))
    self.r_feet = [links.index(n) for n in bone_map.foot_links]
    n = len(segs)
    zeros = torch.zeros(n, 3, dtype=dtype, device=device)
    self.robot_axes = t(calib["robot_axes"], dtype) if calib else zeros.clone()
    self.human_axes = t(calib["human_axes"], dtype) if calib else zeros.clone()
    self.parent_q = (
      t(calib["parent_q"], dtype) if calib else torch.eye(3, dtype=dtype, device=device).repeat(n, 1, 1)
    )
    # Task links: point = mean of up to two joints/links (index repeated for single points).
    tl = bone_map.task_links
    self.task_names = [x.name for x in tl]
    pair = lambda names, f: [f(names[0]), f(names[-1])]  # noqa: E731
    self.task_h_points = t([pair(x.human_points, j) for x in tl])
    self.task_h_frame = t([j(x.human_frame) for x in tl])
    self.task_r_points = t([pair(x.robot_points, links.index) for x in tl])
    self.task_r_frame = t([links.index(x.robot_frame) for x in tl])
    self.task_rot_offset = (
      t(calib["task_rot_offset"], dtype)
      if calib and "task_rot_offset" in calib
      else torch.eye(3, dtype=dtype, device=device).repeat(len(tl), 1, 1)
    )

  def group_mask(self, group: str | None, local: bool = False) -> torch.Tensor:
    mask = torch.tensor([group is None or g == group for g in self.groups], device=self.weights.device)
    return mask & self.has_parent if local else mask


# ---------------------------------------------------------------------------
# Geometry (torch, arbitrary leading dims)
# ---------------------------------------------------------------------------


def normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
  return v / v.norm(dim=-1, keepdim=True).clamp(min=eps)


def quat_wxyz_to_matrix(q: torch.Tensor) -> torch.Tensor:
  w, x, y, z = q.unbind(-1)
  return torch.stack(
    [
      1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
      2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
      2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ],
    dim=-1,
  ).reshape(*q.shape[:-1], 3, 3)  # fmt: skip


def heading_frame(left_point: torch.Tensor, right_point: torch.Tensor) -> torch.Tensor:
  """Yaw-only frame from a left/right point pair: columns (forward, left, up), [..., 3, 3]."""
  left = (left_point - right_point).clone()
  left[..., 2] = 0.0
  left = normalize(left)
  up = torch.zeros_like(left)
  up[..., 2] = 1.0
  return torch.stack([torch.cross(left, up, dim=-1), left, up], dim=-1)


def to_frame(vec_w: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
  """World vectors [..., N, 3] in frames [..., 3, 3] (columns = axes)."""
  return torch.einsum("...ij,...ni->...nj", frame, vec_w)


def in_frames(vec_w: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
  """Vectors [..., N, 3] each in its own frame [..., N, 3, 3]."""
  return torch.einsum("...nij,...ni->...nj", frames, vec_w)


def human_heading(joints_w: torch.Tensor) -> torch.Tensor:
  return heading_frame(joints_w[..., SMPL_HIP_JOINTS[0], :], joints_w[..., SMPL_HIP_JOINTS[1], :])


def robot_heading(link_pos_w: torch.Tensor, bt: BoneMapTensors) -> torch.Tensor:
  return heading_frame(link_pos_w[..., bt.r_heading[0], :], link_pos_w[..., bt.r_heading[1], :])


def human_bone_dirs_w(joints_w: torch.Tensor, joint_rot_w: torch.Tensor, bt: BoneMapTensors) -> torch.Tensor:
  """Human bones in world [..., S, 3] from joints [..., 24, 3] and joint rotations [..., 22, 3, 3]."""
  point = normalize(joints_w[..., bt.h_dst, :] - joints_w[..., bt.h_src, :])
  frame = torch.einsum("...sij,sj->...si", joint_rot_w[..., bt.h_frame, :, :], bt.human_axes)
  return torch.where(bt.is_point.unsqueeze(-1), point, frame)


def robot_bone_dirs_w(link_pos_w: torch.Tensor, link_rot_w: torch.Tensor, bt: BoneMapTensors) -> torch.Tensor:
  """Robot bones in world [..., S, 3] from map links' positions [..., L, 3] and rotations [..., L, 3, 3]."""
  geo = normalize(link_pos_w[..., bt.r_to, :] - link_pos_w[..., bt.r_from, :])
  cal = torch.einsum("...sij,sj->...si", link_rot_w[..., bt.r_from, :, :], bt.robot_axes)
  return torch.where(bt.geometric.unsqueeze(-1), geo, cal)


def human_bone_dirs_local(dirs_w: torch.Tensor, joint_rot_w: torch.Tensor, bt: BoneMapTensors) -> torch.Tensor:
  """Each human bone in its parent joint frame [..., S, 3]."""
  return in_frames(dirs_w, joint_rot_w[..., bt.h_parent, :, :])


def robot_bone_dirs_local(dirs_w: torch.Tensor, link_rot_w: torch.Tensor, bt: BoneMapTensors) -> torch.Tensor:
  """Each robot bone in its parent link frame, mapped into the human parent-frame convention."""
  local = in_frames(dirs_w, link_rot_w[..., bt.r_parent, :, :])
  return torch.einsum("sij,...sj->...si", bt.parent_q, local)


def human_task_poses(joints_w: torch.Tensor, joint_rot_w: torch.Tensor, bt: BoneMapTensors):
  """Human task targets as robot link poses: positions [..., K, 3], rotations [..., K, 3, 3]."""
  pos = joints_w[..., bt.task_h_points, :].mean(-2)
  rot = joint_rot_w[..., bt.task_h_frame, :, :] @ bt.task_rot_offset
  return pos, rot


def robot_task_poses(link_pos_w: torch.Tensor, link_rot_w: torch.Tensor, bt: BoneMapTensors):
  """Robot task link poses: positions [..., K, 3], rotations [..., K, 3, 3]."""
  return link_pos_w[..., bt.task_r_points, :].mean(-2), link_rot_w[..., bt.task_r_frame, :, :]


def rotation_angle(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  """Angle of the relative rotation between rotation matrices [..., 3, 3] (radians)."""
  cos = ((a.transpose(-1, -2) @ b).diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0
  return torch.acos(cos.clamp(-1.0, 1.0))


def angle_between(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  """Angle between unit vectors (radians), robust near 0 and pi."""
  return torch.atan2(torch.cross(a, b, dim=-1).norm(dim=-1), (a * b).sum(-1))


def calibrate(
  human_tpose_joints_w: torch.Tensor,
  human_tpose_rot_w: torch.Tensor,
  robot_link_pos_w: torch.Tensor,
  robot_link_rot_w: torch.Tensor,
  bone_map: SmplBoneMapCfg,
) -> dict:
  """Robot axes, human frame axes and parent-frame maps from one shared T-pose.

  Human T-pose joints [24, 3] / rotations [22, 3, 3]; robot link positions [L, 3] /
  rotations [L, 3, 3] over ``bone_map.robot_links``.
  """
  dtype = human_tpose_joints_w.dtype
  bt = BoneMapTensors(bone_map, None, dtype=dtype)
  hh = human_heading(human_tpose_joints_w)
  rh = robot_heading(robot_link_pos_w, bt)

  # Human frame axes: the requested T-pose axis (heading coords) expressed in the carrying joint frame.
  tpose_axes = torch.tensor([s.tpose_axis or (0.0, 0.0, 0.0) for s in bone_map.segments], dtype=dtype)
  bt.human_axes = torch.einsum("sji,sj->si", human_tpose_rot_w[bt.h_frame], tpose_axes @ hh.T)

  # Calibrated robot axes: the human T-pose bone, carried into the robot heading frame and link frame.
  target_w = to_frame(human_bone_dirs_w(human_tpose_joints_w, human_tpose_rot_w, bt), hh) @ rh.T
  robot_axes = normalize(torch.einsum("sji,sj->si", robot_link_rot_w[bt.r_from], target_w))

  # Parent-frame map Q (robot-parent coords -> human-parent coords): at the T-pose,
  # Q @ (R_rp^T d) == R_hp^T (heading_map @ d) for every robot-world direction d.
  heading_map = hh @ rh.T
  parent_q = human_tpose_rot_w[bt.h_parent].transpose(-1, -2) @ heading_map @ robot_link_rot_w[bt.r_parent]
  # Task-link orientation offset C: robot link rotation = human joint rotation @ C (shared world).
  task_rot_offset = (
    human_tpose_rot_w[bt.task_h_frame].transpose(-1, -2) @ heading_map @ robot_link_rot_w[bt.task_r_frame]
  )
  return {
    "segments": bone_map.segment_names,
    "robot_links": bone_map.robot_links,
    "robot_axes": robot_axes.tolist(),
    "human_axes": bt.human_axes.tolist(),
    "parent_q": parent_q.tolist(),
    "task_links": bt.task_names,
    "task_rot_offset": task_rot_offset.tolist(),
  }
