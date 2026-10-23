"""Noetix E2 (23 DoF) entity for mjlab.

Actuator gains, torque limits, reflected rotor inertia (armature), ankle joint
friction, default pose and action scale are taken verbatim from the GMT
Isaac Lab config (``general_motion_tracking/assets/e2/e2.py``); the actuator
delay of that config is not reproduced (the paper's recipe has none).

Collision: every link's visual mesh is duplicated as a convex collision mesh
that collides with the terrain only (no self-collision, like
``FULL_COLLISION_WITHOUT_SELF`` in mjlab's G1); the feet get condim 3.
"""

from __future__ import annotations

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

E2_XML: Path = Path(__file__).parent / "e2" / "e2.xml"
assert E2_XML.exists(), E2_XML


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(E2_XML))


# Motors (from GMT e2.py): rotor inertia reflected to the joint = J_rotor * gear^2.
ARMATURE_5025 = 8.4e-6 * 17.0**2
ARMATURE_7216 = 32.4e-6 * 22.0**2
ARMATURE_7225 = 49.2e-6 * 22.0**2
ARMATURE_9016 = 118.1e-6 * 23.625**2
TORQUE_5025, TORQUE_7216, TORQUE_7225, TORQUE_9016 = 50.0, 140.0, 200.0, 240.0


def _act(names, kp, kd, effort, armature):
  return BuiltinPositionActuatorCfg(target_names_expr=tuple(names), stiffness=kp, damping=kd, effort_limit=effort, armature=armature)


E2_ACTUATORS = (
  _act((".*_hip_yaw_joint", ".*_hip_roll_joint"), 140.0, 12.0, TORQUE_7216, ARMATURE_7216),
  _act((".*_hip_pitch_joint",), 240.0, 20.0, TORQUE_9016, ARMATURE_9016),
  _act((".*_knee_joint",), 200.0, 18.0, TORQUE_7225, ARMATURE_7225),
  _act((".*_ankle_pitch_joint",), 50.0, 1.5, TORQUE_5025, ARMATURE_5025),
  _act((".*_ankle_roll_joint",), 50.0, 0.5, TORQUE_5025, ARMATURE_5025),
  _act(("waist_yaw_joint",), 140.0, 10.0, TORQUE_7216, ARMATURE_7216),
  _act(("waist_roll_joint", "waist_pitch_joint"), 70.0, 0.5, TORQUE_5025, ARMATURE_5025),
  _act((".*_shoulder_pitch_joint", ".*_shoulder_roll_joint"), 140.0, 10.0, TORQUE_7216, ARMATURE_7216),
  _act((".*_shoulder_yaw_joint",), 40.0, 5.0, TORQUE_5025, ARMATURE_5025),
  _act((".*_elbow_joint",), 40.0, 7.0, TORQUE_5025, ARMATURE_5025),
)

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.82),
  joint_pos={
    "l_leg_hip_yaw_joint": 0.002,
    "l_leg_hip_roll_joint": 0.034,
    "l_leg_hip_pitch_joint": 0.0,
    "l_leg_knee_joint": 0.18,
    "l_leg_ankle_pitch_joint": -0.2,
    "l_leg_ankle_roll_joint": 0.0,
    "r_leg_hip_yaw_joint": -0.002,
    "r_leg_hip_roll_joint": -0.034,
    "r_leg_hip_pitch_joint": 0.0,
    "r_leg_knee_joint": 0.18,
    "r_leg_ankle_pitch_joint": -0.2,
    "r_leg_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": -0.02,
    "l_arm_shoulder_pitch_joint": 0.0,
    "l_arm_shoulder_roll_joint": 0.1,
    "l_arm_shoulder_yaw_joint": 0.0,
    "l_arm_elbow_joint": 0.0,
    "r_arm_shoulder_pitch_joint": 0.0,
    "r_arm_shoulder_roll_joint": -0.1,
    "r_arm_shoulder_yaw_joint": 0.0,
    "r_arm_elbow_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)

# Convex mesh collision with the terrain only (contype 0, conaffinity 1); feet condim 3.
COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={r"^(l|r)_foot_collision$": 3, ".*_collision": 1},
  priority={r"^(l|r)_foot_collision$": 1, ".*": 0},
  friction={r"^(l|r)_foot_collision$": (0.6,)},
)

# The team's E2 eval model (E2_V1.0_0805_02_eval.xml): every collision mesh collides with everything, incl. the
# robot itself (MuJoCo drops parent-child pairs), condim 3, friction 0.5 (floor 0.5 too). Priority 1 makes the
# robot's 0.5 win against the 1.0 training terrain; the per-episode foot-friction DR still applies on top.
EVAL_CONTACT_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=1,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.5,),
)

E2_ARTICULATION = EntityArticulationInfoCfg(actuators=E2_ACTUATORS, soft_joint_pos_limit_factor=1.0)


def get_e2_robot_cfg() -> EntityCfg:
  return EntityCfg(init_state=HOME_KEYFRAME, collisions=(COLLISION_WITHOUT_SELF,), spec_fn=get_spec, articulation=E2_ARTICULATION)


E2_ACTION_SCALE: dict[str, float] = {}
for a in E2_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  for n in a.target_names_expr:
    E2_ACTION_SCALE[n] = 0.25 * a.effort_limit / a.stiffness  # type: ignore[operator]

# Tracked bodies (E2 analogue of mjlab's 14-body G1 list; E2 has no wrists).
E2_TRACKED_BODIES = (
  "base_link",
  "l_leg_hip_roll_link",
  "l_leg_knee_link",
  "l_leg_ankle_roll_link",
  "r_leg_hip_roll_link",
  "r_leg_knee_link",
  "r_leg_ankle_roll_link",
  "waist_pitch_link",
  "l_arm_shoulder_roll_link",
  "l_arm_elbow_link",
  "r_arm_shoulder_roll_link",
  "r_arm_elbow_link",
)
E2_ANCHOR_BODY = "waist_pitch_link"
E2_FOOT_GEOMS = r"^(l|r)_foot_collision$"
