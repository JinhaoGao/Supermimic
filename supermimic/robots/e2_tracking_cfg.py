"""Noetix E2 flat-ground tracking environment (mirrors mjlab's G1 config)."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

from .e2_constants import E2_ACTION_SCALE, E2_ANCHOR_BODY, E2_FOOT_GEOMS, E2_TRACKED_BODIES, get_e2_robot_cfg


def noetix_e2_flat_tracking_env_cfg() -> ManagerBasedRlEnvCfg:
  cfg = make_tracking_env_cfg()
  cfg.scene.entities = {"robot": get_e2_robot_cfg()}
  cfg.scene.sensors = ()
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = E2_ACTION_SCALE
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.anchor_body_name = E2_ANCHOR_BODY
  motion_cmd.body_names = E2_TRACKED_BODIES
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = E2_FOOT_GEOMS
  cfg.events["base_com"].params["asset_cfg"].body_names = (E2_ANCHOR_BODY,)
  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "l_leg_ankle_roll_link",
    "r_leg_ankle_roll_link",
    "l_arm_elbow_link",
    "r_arm_elbow_link",
  )
  cfg.rewards.pop("self_collisions", None)  # no self-collision sensor for E2
  cfg.viewer.body_name = E2_ANCHOR_BODY
  return cfg
