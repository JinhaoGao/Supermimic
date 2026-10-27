"""Thin interface between the mjlab env and the SuperMimic learner.

Extracts the minimal state vector and the sim context xi from the simulator,
owns the FK / analytic heads, fetches goal references from the in-RAM motion
library, and keeps the proprioceptive history buffer.
"""

from __future__ import annotations

import torch

from .fk import TorchFK
from .heads import AnalyticHeads, HeadsCfg, window_frames


class ObsHistory:
  def __init__(self, num_envs: int, hist: int, dim: int, device: str):
    self.buf = torch.zeros(num_envs, hist, dim, device=device)

  def reset(self, env_ids: torch.Tensor, obs: torch.Tensor) -> None:
    """Fill the whole history of ``env_ids`` with their current obs (mjlab semantics)."""
    if env_ids.numel() > 0:
      self.buf[env_ids] = obs[env_ids][:, None, :]

  def push(self, obs: torch.Tensor) -> None:
    self.buf = torch.cat([self.buf[:, 1:], obs[:, None, :]], dim=1)


class SimInterface:
  FRICTION_RANGE = (0.3, 1.2)
  BIAS_RANGE = 0.01
  COM_RANGE = (0.025, 0.05, 0.05)

  def __init__(self, env, heads_cfg: HeadsCfg):
    self.env = env
    self.device = env.device
    self.robot = env.scene["robot"]
    self.cmd = env.command_manager.get_term("motion")
    self.lib = self.cmd.library
    tracked_local = [int(i) for i in self.cmd.body_indexes.tolist()]
    self.fk = TorchFK(env.sim.mj_model, self.robot, self.device, needed_bodies=tracked_local)
    self.ee_body_names, ee_idx = self.ee_bodies(env, heads_cfg)
    self.root_body_name = str(self.robot.body_names[self.fk.root_local])  # free-joint body (G1 pelvis, E2 base_link)
    self.root_ori_body, ro_idx = self.root_ori_body_of(env, heads_cfg, self.root_body_name)
    # Root-height termination (10-26): same switch / threshold in the env (root_z) and the heads, on the root-orientation
    # body (the eval site's root_pos test uses the same root body as its root_ori test).
    rz_term = env.cfg.terminations.get("root_z")
    if bool(heads_cfg.root_z_on) != (rz_term is not None):
      raise ValueError(f"root-height termination mismatch: HeadsCfg.root_z_term={heads_cfg.root_z_term} but the env "
                       f"{'has' if rz_term is not None else 'has no'} 'root_z' term (make_env_cfg(root_z_term=...))")
    if rz_term is not None:
      if abs(float(rz_term.params["threshold"]) - float(heads_cfg.root_z_term)) > 1e-9:
        raise ValueError(f"root-height threshold mismatch: env {rz_term.params['threshold']} vs HeadsCfg.root_z_term {heads_cfg.root_z_term}")
      if rz_term.params["body_name"] != self.root_ori_body or ro_idx is None:
        raise ValueError(f"root-height body {rz_term.params['body_name']!r} must be the tracked root-orientation body {self.root_ori_body!r}")
    self.heads = AnalyticHeads(
      self.fk,
      tracked_local,
      int(self.cmd.motion_anchor_body_index),
      self.robot.data.default_joint_pos[0],
      self.robot.data.soft_joint_pos_limits[0],
      heads_cfg,
      ee_indices=ee_idx,
      root_ori_index=ro_idx,
      root_ori_is_root=self.root_ori_body == self.root_body_name,
    )
    self._init_context(env)

  def _init_context(self, env) -> None:
    """Sim-context (xi) bookkeeping after the heads exist; shared with human.sim_interface.HumanSimInterface."""
    self.nj = self.heads.nj
    self.xi_dim = 4 + self.nj
    idx = self.robot.indexing
    foot_regex = env.cfg.events["foot_friction"].params["asset_cfg"].geom_names  # per-robot (G1 capsules / E2 mesh)
    foot_local = self.robot.find_geoms(foot_regex)[0]
    self.foot_geom_gid = int(idx.geom_ids[foot_local[0]].item())
    torso_local = self.robot.body_names.index(self.cmd.cfg.anchor_body_name)
    self.torso_gid = int(idx.body_ids[torso_local].item())
    self.ipos_nominal = torch.as_tensor(env.sim.mj_model.body_ipos[self.torso_gid], dtype=torch.float32, device=self.device)
    self.env_origins = env.scene.env_origins.to(self.device)
    self.com_scale = torch.tensor(self.COM_RANGE, device=self.device)
    # MuJoCo Warp prints a solver/linesearch overflow warning every step the
    # iteration cap is hit (common with random actions); silence it. The flag is a
    # compile-time constant of the kernels (wp.static), baked into the CUDA graphs
    # captured at env construction, so the graphs are re-captured after changing it.
    try:
      env.sim.wp_model.opt.warn_overflow = 0
      env.sim.create_graph()
    except Exception:  # noqa: BLE001
      pass

  @staticmethod
  def ee_bodies(env, heads_cfg: HeadsCfg) -> tuple[tuple[str, ...], list[int]]:
    """End-effector bodies of the simulator's ``ee_body_pos`` termination as indices into the tracked-body list
    (mjlab ``_get_body_indexes`` order). The env and the heads must agree: the imagined termination has to see
    the same failures as the simulator, so a mismatch in switch or threshold is an error."""
    terms = env.cfg.terminations
    has = "ee_body_pos" in terms
    if bool(heads_cfg.ee_termination) != has:
      raise ValueError(
        f"end-effector termination mismatch: HeadsCfg.ee_termination={heads_cfg.ee_termination} but the env "
        f"{'has' if has else 'has no'} 'ee_body_pos' term (make_env_cfg(ee_termination=...))"
      )
    if not has:
      return (), []
    params = terms["ee_body_pos"].params
    thr = float(params["threshold"])
    if abs(thr - float(heads_cfg.ee_term_z)) > 1e-9:
      raise ValueError(f"end-effector threshold mismatch: env {thr} vs HeadsCfg.ee_term_z {heads_cfg.ee_term_z}")
    names = tuple(params["body_names"])
    tracked = tuple(env.command_manager.get_term("motion").cfg.body_names)
    missing = [n for n in names if n not in tracked]
    if missing:
      raise ValueError(f"end-effector bodies {missing} are not tracked bodies {tracked}")
    idx = [i for i, n in enumerate(tracked) if n in names]
    return tuple(tracked[i] for i in idx), idx

  @staticmethod
  def root_ori_body_of(env, heads_cfg: HeadsCfg, root_body: str) -> tuple[str, int | None]:
    """Body of the root-orientation test -> (name, index into the tracked-body list or None if it is not tracked).

    The eval site compares the reference's first tracked body with qpos[3:7], i.e. the robot's free-joint root body
    (root_body: G1 pelvis, E2 base_link); HeadsCfg.root_ori_body "" means that body. The env's ``root_ori`` term and the
    heads must agree (switch, threshold, body), and the default body name of env_cfg.ROOT_BODY must be the model's root.
    With the term off the index still serves the always-on pelvis-angle logging (None = not tracked: no logging)."""
    terms = env.cfg.terminations
    has = "root_ori" in terms
    on = bool(heads_cfg.root_ori_on)
    if on != has:
      raise ValueError(
        f"root-orientation termination mismatch: HeadsCfg.root_ori_term={heads_cfg.root_ori_term} but the env "
        f"{'has' if has else 'has no'} 'root_ori' term (make_env_cfg(root_ori_term=...))"
      )
    name = heads_cfg.root_ori_body or root_body
    if has:
      params = terms["root_ori"].params
      if abs(float(params["threshold"]) - float(heads_cfg.root_ori_term)) > 1e-9:
        raise ValueError(f"root-orientation threshold mismatch: env {params['threshold']} vs HeadsCfg.root_ori_term {heads_cfg.root_ori_term}")
      if params["body_name"] != name:
        raise ValueError(f"root-orientation body mismatch: env {params['body_name']!r} vs heads {name!r} (robot root {root_body!r}; "
                         "env_cfg.ROOT_BODY wrong for this model?)")
    tracked = tuple(env.command_manager.get_term("motion").cfg.body_names)
    if name not in tracked:
      if on:
        raise ValueError(f"root-orientation body {name!r} is not a tracked body {tracked}")
      return name, None
    return name, tracked.index(name)

  # ---- state -----------------------------------------------------------------

  @torch.no_grad()
  def state(self) -> torch.Tensor:
    d = self.robot.data
    pose = d.root_link_pose_w
    vel = d.root_link_vel_w
    p = pose[:, :3] - self.env_origins
    return torch.cat([p, pose[:, 3:7], vel[:, :3], vel[:, 3:6], d.joint_pos, d.joint_vel], dim=-1).clone()

  @torch.no_grad()
  def xi(self) -> torch.Tensor:
    """Scaled sim context [friction(1), encoder bias(J), torso COM offset(3)]."""
    E = self.env.num_envs
    model = self.env.sim.model
    fr = torch.as_tensor(model.geom_friction)
    friction = fr[:, self.foot_geom_gid, 0] if fr.dim() == 3 else fr[self.foot_geom_gid, 0].reshape(1)
    if friction.shape[0] != E:
      friction = friction.expand(E)
    ipos = torch.as_tensor(model.body_ipos)
    com = (ipos[:, self.torso_gid] if ipos.dim() == 3 else ipos[self.torso_gid].reshape(1, 3)) - self.ipos_nominal
    if com.shape[0] != E:
      com = com.expand(E, 3)
    bias = self.robot.data.encoder_bias
    lo, hi = self.FRICTION_RANGE
    f_n = (friction - 0.5 * (lo + hi)) / (0.5 * (hi - lo))
    return torch.cat([f_n[:, None], bias / self.BIAS_RANGE, com / self.com_scale], dim=-1).float().clone()

  @staticmethod
  def bias_from_xi(xi: torch.Tensor, nj: int) -> torch.Tensor:
    return xi[:, 1 : 1 + nj] * SimInterface.BIAS_RANGE

  # ---- references ------------------------------------------------------------

  @torch.no_grad()
  def goal_refs(self, frame: torch.Tensor, clip_end: torch.Tensor):
    """K-frame goal references for all envs: (ref_q [E,K,J], ref_qd, ref_ap [E,K,3], ref_aq [E,K,4], ref_av [E,K,3],
    ref_aw [E,K,3]); the anchor velocities (world) only with goal_ref_root, else None."""
    K = self.heads.cfg.K
    E = frame.shape[0]
    fr = window_frames(frame, clip_end, K).reshape(-1)
    vel = self.heads.cfg.goal_ref_root
    g = self.lib.gather(fr, fields=["joint_pos", "joint_vel", "body_pos", "body_quat"] + (["body_lin_vel", "body_ang_vel"] if vel else []))
    a = self.heads.anchor
    return (
      g["joint_pos"].view(E, K, -1),
      g["joint_vel"].view(E, K, -1),
      g["body_pos"].view(E, K, -1, 3)[:, :, a],
      g["body_quat"].view(E, K, -1, 4)[:, :, a],
      g["body_lin_vel"].view(E, K, -1, 3)[:, :, a] if vel else None,
      g["body_ang_vel"].view(E, K, -1, 3)[:, :, a] if vel else None,
    )

  def current_ref(self) -> tuple:
    """Reference body fields at the command's current frame (for the reward of the next step)."""
    c = self.cmd.current_ref()
    return c["body_pos"], c["body_quat"], c["body_lin_vel"], c["body_ang_vel"]
