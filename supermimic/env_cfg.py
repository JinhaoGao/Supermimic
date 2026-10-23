"""mjlab environment configuration for SuperMimic (G1, flat ground).

Starts from mjlab's BeyondMimic G1 tracking task and applies the paper's
changes (Appendix A): no push events, no self-collision penalty, joint-limit
weight -2, no end-effector termination (restored by default since 2026-10-22,
``ee_termination``; see heads.HeadsCfg), DR context re-sampled per episode,
multi-clip library command with return-based reset sampling and a clip-end
time-out. The env never auto-resets: the trainer reads terminal states and
calls ``reset(env_ids)`` itself.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.tracking import mdp

from .library_command import LibraryMotionCommandCfg, motion_clip_end


def base_tracking_cfg(robot: str):
  if robot == "g1":
    from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg

    return unitree_g1_flat_tracking_env_cfg(has_state_estimation=True, play=False)
  if robot == "e2":
    from .robots.e2_tracking_cfg import noetix_e2_flat_tracking_env_cfg

    return noetix_e2_flat_tracking_env_cfg()
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
  ee_termination: bool = True,
  ee_threshold: float = 0.25,
  clamp_returns: bool = False,
) -> ManagerBasedRlEnvCfg:
  """ee_termination: keep BeyondMimic's end-effector height termination (``ee_body_pos``: any end-effector body
  z error > ee_threshold; bodies per robot cfg). Must match ``HeadsCfg.ee_termination`` / ``ee_term_z`` of the
  learner (SimInterface checks). Off = the paper's recipe (term removed).
  clamp_returns: reset sampler bin returns clamped to [0, g_max] (LibraryMotionCommandCfg.clamp_returns)."""
  assert mode in ("train", "eval")
  cfg = base_tracking_cfg(robot)
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
    clamp_returns=clamp_returns,
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

  # Terminations.
  if ee_termination:
    ee = cfg.terminations["ee_body_pos"]
    assert ee.params["body_names"], f"{robot}: ee_body_pos has no end-effector bodies"
    ee.params["threshold"] = float(ee_threshold)
  else:
    cfg.terminations.pop("ee_body_pos", None)
  cfg.terminations["clip_end"] = TerminationTermCfg(
    func=motion_clip_end, params={"command_name": "motion"}, time_out=True
  )
  cfg.episode_length_s = episode_length_s if mode == "train" else 1.0e6
  return cfg
