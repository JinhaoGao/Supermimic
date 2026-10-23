"""mjlab environment configuration for SuperMimic (G1, flat ground).

Starts from mjlab's BeyondMimic G1 tracking task and applies the paper's
changes (Appendix A): no push events, no self-collision penalty, joint-limit
weight -2, no end-effector termination, DR context re-sampled per episode,
multi-clip library command with return-based reset sampling and a clip-end
time-out. The env never auto-resets: the trainer reads terminal states and
calls ``reset(env_ids)`` itself.

v2 options: ``ee_term_z`` keeps BeyondMimic's end-effector height termination (the team's
benchmark terminates on it), ``robot_variant`` swaps the robot model (see base_tracking_cfg).
"""

from __future__ import annotations

import mujoco
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.tracking import mdp

from .library_command import LibraryMotionCommandCfg, motion_clip_end


def _zero_joint_damping(spec_fn):
  def build():
    spec = spec_fn()
    from .robots.g1_eval_model import set_scalar

    for j in spec.joints:
      if j.type == mujoco.mjtJoint.mjJNT_HINGE:
        set_scalar(j, "damping", 0.0)
    return spec

  return build


def base_tracking_cfg(robot: str, variant: str = ""):
  """variant ("+"-joined) -- g1: "" mjlab's G1 | "eval" the team's eval / deploy G1 (robots/g1_eval_model.py);
  e2: "" as trained in v1 | "nodamp" without the passive joint damping of the source MJCF (it carries
  damping = kd on every joint, which doubled the actuator damping in v1; the real motor PD has kd only) |
  "evalcontact" contacts as in the team's E2 eval model (self-collision, condim 3, friction 0.5; v1/v2 E2
  collided with the terrain only and without friction off the feet); g1 "evalcontact": mjlab's G1 collision
  set with condim 3 everywhere (mjlab: frictionless off the feet; eval model and robot: friction 1.0)."""
  parts = {p for p in variant.split("+") if p}
  allowed = {"g1": {"eval", "evalcontact"}, "e2": {"nodamp", "evalcontact"}}.get(robot, set())
  if parts - allowed:
    raise ValueError(f"unknown {robot} variant(s) {sorted(parts - allowed)}")
  if robot == "g1":
    from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg

    cfg = unitree_g1_flat_tracking_env_cfg(has_state_estimation=True, play=False)
    ent = cfg.scene.entities["robot"]
    if "eval" in parts:
      from .robots.g1_eval_model import patch_spec

      ent.spec_fn = patch_spec(ent.spec_fn)
    if "evalcontact" in parts:
      from .robots.g1_eval_model import eval_contact_collision

      ent.collisions = (eval_contact_collision(),)
    return cfg
  if robot == "e2":
    from .robots.e2_tracking_cfg import noetix_e2_flat_tracking_env_cfg

    cfg = noetix_e2_flat_tracking_env_cfg()
    ent = cfg.scene.entities["robot"]
    if "nodamp" in parts:
      ent.spec_fn = _zero_joint_damping(ent.spec_fn)
    if "evalcontact" in parts:
      from .robots.e2_constants import EVAL_CONTACT_COLLISION

      ent.collisions = (EVAL_CONTACT_COLLISION,)
    return cfg
  raise ValueError(f"unknown robot {robot}")


def make_env_cfg(
  motion_files: list[str],
  num_envs: int,
  mode: str = "train",
  episode_length_s: float = 10.0,
  bin_seconds: float = 1.0,
  robot: str = "g1",
  library_key: str = "",
  eval_clips: tuple[int, ...] = (),
  cache_dir: str = "",
  num_workers: int = 32,
  ee_term_z: float = 0.0,
  robot_variant: str = "",
  source_weights: tuple = (),
  name_weights: tuple = (),
  hard_names: tuple = (),
) -> ManagerBasedRlEnvCfg:
  assert mode in ("train", "eval")
  cfg = base_tracking_cfg(robot, robot_variant)
  cfg.scene.num_envs = num_envs
  cfg.scene.sensors = ()  # self-collision sensor not needed
  cfg.auto_reset = False

  base = cfg.commands["motion"]
  cfg.commands["motion"] = LibraryMotionCommandCfg(
    entity_name="robot",
    resampling_time_range=(1.0e9, 1.0e9),
    debug_vis=False,
    pose_range=dict(base.pose_range) if mode == "train" else {},
    velocity_range=dict(base.velocity_range) if mode == "train" else {},
    joint_position_range=base.joint_position_range if mode == "train" else (0.0, 0.0),
    motion_files=tuple(motion_files),
    anchor_body_name=base.anchor_body_name,
    body_names=base.body_names,
    bin_seconds=bin_seconds,
    sampling_mode="return" if mode == "train" else "start",
    cache_dir=cache_dir,
    num_workers=num_workers,
    library_key=library_key,
    eval_clips=tuple(eval_clips),
    source_weights=tuple(tuple(kv) for kv in source_weights) if mode == "train" else (),
    name_weights=tuple(tuple(kv) for kv in name_weights) if mode == "train" else (),
    hard_names=tuple(hard_names) if mode == "train" else (),
  )

  # Minimal observation group (the trainer builds policy inputs itself).
  cfg.observations = {
    "actor": ObservationGroupCfg(
      terms={"joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel)},
      concatenate_terms=True,
      enable_corruption=False,
    )
  }

  # Events: drop pushes; per-episode sim context in training; clean physics in eval.
  cfg.events.pop("push_robot", None)
  if mode == "train":
    for k in ("base_com", "encoder_bias", "foot_friction"):
      cfg.events[k].mode = "reset"
  else:
    cfg.events.pop("base_com", None)
    cfg.events.pop("encoder_bias", None)
    cfg.events["foot_friction"].params["ranges"] = (1.0, 1.0)

  # Rewards.
  cfg.rewards.pop("self_collisions", None)
  cfg.rewards["joint_limit"].weight = -2.0

  # Terminations. ee_body_pos (end-effector height) only when asked for (v2); the paper drops it.
  if ee_term_z > 0:
    cfg.terminations["ee_body_pos"].params["threshold"] = float(ee_term_z)
  else:
    cfg.terminations.pop("ee_body_pos", None)
  if hard_names and mode == "train":  # hard-clip curriculum: per-env threshold multiplier (library_command)
    from .library_command import scaled_anchor_ori, scaled_anchor_pos_z, scaled_body_pos_z

    cfg.terminations["anchor_pos"].func = scaled_anchor_pos_z
    cfg.terminations["anchor_ori"].func = scaled_anchor_ori
    if "ee_body_pos" in cfg.terminations:
      cfg.terminations["ee_body_pos"].func = scaled_body_pos_z
  cfg.terminations["clip_end"] = TerminationTermCfg(
    func=motion_clip_end, params={"command_name": "motion"}, time_out=True
  )
  cfg.episode_length_s = episode_length_s if mode == "train" else 1.0e6
  return cfg
