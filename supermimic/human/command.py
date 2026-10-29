"""Human-reference motion command: the library command's reset sampler on a HumanMotionLibrary.

Keeps LibraryMotionCommand's return-based bin sampling, clip-end time-out, "start" eval mode and sampler state (so the
trainer's bookkeeping, checkpoints and DDP sampler sync are the same), but
  * the per-step cache holds the human reward / termination fields (human.heads.STEP_FIELDS) at the current frame,
  * resets write the bone-direction IK reset state (init_* fields) at the env origin: height from the IK, heading =
    the human heading, root velocity = k x the human root velocity, plus the usual pose / joint / velocity noise,
  * the simulator termination is the heads' own hard test (human_failed), so simulator and imagination agree.
  * ee_mode (attach_heads): a per-env condition ``cond`` (human.heads.COND_DIM: alpha, ee_valid, anchor xy, drift,
    drift rate) sampled at resets and advanced every step (alpha switches, drift random walk); "start" (evaluation) mode
    uses alpha = eval_alpha, ee_valid = 1, no drift.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch

from mjlab.managers import CommandTerm
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, sample_uniform

from ..library_command import LIBRARY_REGISTRY, LibraryMotionCommand, LibraryMotionCommandCfg
from . import bone_map as bone
from .heads import COND_DIM, EE_STEP_FIELDS, STEP_FIELDS
from .motions import HumanMotionLibrary

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

INIT_FIELDS = ("init_joint_pos", "init_joint_vel", "init_root_quat", "init_root_height", "h_root_vel", "h_yaw", "h_yaw_rate", "h_rest")


def foot_geoms(mj_model: mujoco.MjModel, foot_geom_regex: str) -> list[int]:
  rx = re.compile(foot_geom_regex)
  ids = [g for g in range(mj_model.ngeom) if rx.search((mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").split("/")[-1])]
  if not ids:
    raise ValueError(f"no foot collision geom matches {foot_geom_regex!r}")
  return ids


def lowest_point(mj_model: mujoco.MjModel, d: mujoco.MjData, geoms: list[int]) -> float:
  """Lowest world z of the geoms (sphere / capsule / box exact, other types: their centre) after mj_kinematics."""
  low = []
  for g in geoms:
    t, size, z = mj_model.geom_type[g], mj_model.geom_size[g], d.geom_xpos[g, 2]
    R = d.geom_xmat[g].reshape(3, 3)
    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
      low.append(z - size[0])
    elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
      low.append(z - abs(R[2, 2]) * size[1] - size[0])
    elif t == mujoco.mjtGeom.mjGEOM_BOX:
      low.append(z - float(np.abs(R[2, :]) @ size[:3]))
    else:
      low.append(z)
  return float(min(low))


def body_id(mj_model: mujoco.MjModel, name: str) -> int:
  """Body id by name, entity prefixes ("robot/") ignored."""
  return next(i for i in range(mj_model.nbody) if (mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, i) or "").split("/")[-1] == name)


def robot_rest_geometry(mj_model: mujoco.MjModel, pelvis: str, feet: tuple[str, ...], foot_geom_regex: str) -> tuple[float, float]:
  """(pelvis height, ankle-link height) above the lowest sole point with every hinge at 0 (straight legs, feet flat),
  from the foot collision geoms (sphere / capsule / box) of the simulated model."""
  d = mujoco.MjData(mj_model)
  d.qpos[:] = mj_model.qpos0
  for j in range(mj_model.njnt):
    if mj_model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
      d.qpos[mj_model.jnt_qposadr[j]] = 0.0
  mujoco.mj_kinematics(mj_model, d)
  sole = lowest_point(mj_model, d, foot_geoms(mj_model, foot_geom_regex))
  ankle = min(d.xpos[body_id(mj_model, n), 2] for n in feet)
  return float(d.xpos[body_id(mj_model, pelvis), 2] - sole), float(ankle - sole)


class HumanMotionCommand(LibraryMotionCommand):
  cfg: "HumanMotionCommandCfg"

  def __init__(self, cfg: "HumanMotionCommandCfg", env: "ManagerBasedRlEnv"):
    CommandTerm.__init__(self, cfg, env)
    self.robot = env.scene[cfg.entity_name]
    if cfg.cluster_k > 0 or cfg.hard_names:
      raise ValueError("human reference: sampler clusters and the hard-clip curriculum need robot reference features")
    self.bone_map = bone.SMPL_BONE_MAPS[cfg.bone_map]
    key = cfg.library_key
    if key and key in LIBRARY_REGISTRY:
      self.library = LIBRARY_REGISTRY[key]
      assert isinstance(self.library, HumanMotionLibrary) and self.library.joint_names == tuple(self.robot.joint_names)
    else:
      self.library = HumanMotionLibrary(list(cfg.motion_files), cfg.robot_name, tuple(self.robot.joint_names), self.bone_map,
                                        self.device, cfg.bin_seconds, cache_dir=cfg.cache_dir or None)
      if key:
        LIBRARY_REGISTRY[key] = self.library
    self.motion = self.library
    self.rest_height, self.ankle_height = robot_rest_geometry(
      env.sim.mj_model, self.bone_map.pelvis_link, self.bone_map.foot_links, cfg.foot_geom_regex)
    E = self.num_envs
    lib = self.library
    self._cur = {k: torch.zeros(E, *lib.fields[k].shape[1:], device=self.device) for k in STEP_FIELDS}
    self._cur["h_rest"].fill_(1.0)
    self._init_sampler()
    self.heads = None  # HumanHeads, attached by HumanSimInterface (attach_heads; used by the simulator termination term)
    self.ee_on = False
    self.cond = torch.zeros(E, COND_DIM, device=self.device)
    self._term_cache: tuple[int, torch.Tensor] | None = None
    self._pending_forward = False

  def attach_heads(self, heads) -> None:
    """Heads of the simulator termination; ee_mode: EE library fields, their current-frame cache and the condition."""
    self.heads = heads
    if not heads.ee_on:
      return
    self.ee_on = True
    lib = self.library
    lib.ensure_ee(self.bone_map)
    E = self.num_envs
    for k in EE_STEP_FIELDS:
      self._cur[k] = torch.zeros(E, *lib.fields[k].shape[1:], device=self.device)
    self._alpha_tgt = torch.zeros(E, device=self.device)
    self._alpha_rate = torch.zeros(E, device=self.device)  # per step
    self._alpha_switch = torch.full((E,), -1, dtype=torch.long, device=self.device)  # steps until the switch (-1: none)
    self._drift_left = torch.zeros(E, dtype=torch.long, device=self.device)  # steps until the next drift rate
    self.ee_clip_ok = torch.ones(lib.num_clips, dtype=torch.bool, device=self.device)
    if heads.cfg.ee_clip_list:
      import os

      ok = {os.path.basename(ln.strip()) for ln in open(os.path.expanduser(heads.cfg.ee_clip_list)) if ln.strip()}
      self.ee_clip_ok = torch.tensor([c.name in ok for c in lib.clips], dtype=torch.bool, device=self.device)
      print(f"[HumanMotionCommand] ee_clip_list: {int(self.ee_clip_ok.sum())}/{lib.num_clips} clips with alpha > 0")
    self.refresh_current()
    self._sample_cond(torch.arange(E, device=self.device), self.time_steps)

  def _fields(self) -> tuple[str, ...]:
    return STEP_FIELDS + (EE_STEP_FIELDS if self.ee_on else ())

  def current_ref(self) -> dict:
    return {**self._cur, "cond": self.cond} if self.ee_on else dict(self._cur)

  # ---- end-effector condition ----------------------------------------------------

  def _sample_alpha(self, n: int) -> torch.Tensor:
    c = self.heads.cfg
    u = torch.rand(n, device=self.device)
    a = torch.rand(n, device=self.device)
    a = torch.where(u < c.alpha_p0, torch.zeros_like(a), a)
    return torch.where((u >= c.alpha_p0) & (u < c.alpha_p0 + c.alpha_p1), torch.ones_like(a), a)

  def _drift_rate(self, env_ids: torch.Tensor) -> None:
    """New drift rate of env_ids: towards a uniform point of the drift box, reached after U(tmin, tmax) s."""
    c, n, dt = self.heads.cfg, len(env_ids), self._env.step_dt
    box = self.heads.drift_box
    steps = torch.clamp((torch.empty(n, device=self.device).uniform_(c.drift_tmin, c.drift_tmax) / dt).round().long(), min=1)
    target = (torch.rand(n, 3, device=self.device) * 2.0 - 1.0) * box
    self.cond[env_ids, 7:10] = (target - self.cond[env_ids, 4:7]) / (steps.float()[:, None] * dt)
    self._drift_left[env_ids] = steps

  @torch.no_grad()
  def _sample_cond(self, env_ids: torch.Tensor, frames: torch.Tensor) -> None:
    c, n = self.heads.cfg, len(env_ids)
    cond = torch.zeros(n, COND_DIM, device=self.device)
    cond[:, 2:4] = self.library.gather(frames, fields=("h_root_xy",))["h_root_xy"]
    if self.cfg.sampling_mode == "start":  # evaluation: fixed alpha, no drift, no switch
      cond[:, 0] = c.eval_alpha
      cond[:, 1] = 1.0
      self.cond[env_ids] = cond
      self._alpha_tgt[env_ids] = c.eval_alpha
      self._alpha_switch[env_ids] = -1
      self._drift_left[env_ids] = 1 << 40
      return
    a = self._sample_alpha(n) * self.ee_clip_ok[self.library.clip_of_frame(frames)]
    cond[:, 0] = a
    cond[:, 1] = torch.where((a == 0) & (torch.rand(n, device=self.device) < c.ee_dropout), 0.0, 1.0)
    self.cond[env_ids] = cond
    self._alpha_tgt[env_ids] = a
    self._alpha_rate[env_ids] = 0.0
    T = 10.0 / self._env.step_dt  # switch time: uniform over the first 10 s (episode length)
    sw = (torch.rand(n, device=self.device) * T).long() + 1
    self._alpha_switch[env_ids] = torch.where(torch.rand(n, device=self.device) < c.alpha_switch_p, sw, torch.full_like(sw, -1))
    if c.drift_xy > 0 or c.drift_yaw > 0:
      self._drift_rate(env_ids)
    else:
      self._drift_left[env_ids] = 1 << 40

  @torch.no_grad()
  def _advance_cond(self) -> None:
    c, dt = self.heads.cfg, self._env.step_dt
    box = self.heads.drift_box
    d = self.cond[:, 4:7] + self.cond[:, 7:10] * dt
    self.cond[:, 4:7] = torch.maximum(torch.minimum(d, box), -box)
    self._drift_left -= 1
    due = (self._drift_left <= 0).nonzero(as_tuple=False).squeeze(-1)
    if due.numel():
      self._drift_rate(due)
    # alpha: scheduled switch -> linear ramp towards a new alpha (ee_valid back on when it leaves 0)
    self._alpha_switch = torch.where(self._alpha_switch > 0, self._alpha_switch - 1, self._alpha_switch)
    sw = (self._alpha_switch == 0).nonzero(as_tuple=False).squeeze(-1)
    if sw.numel():
      new = self._sample_alpha(sw.numel()) * self.ee_clip_ok[self.library.clip_of_frame(self.time_steps[sw])]
      self._alpha_tgt[sw] = new
      self._alpha_rate[sw] = (new - self.cond[sw, 0]).abs() / max(c.alpha_ramp_s / dt, 1.0)
      self._alpha_switch[sw] = -1
      self.cond[sw, 1] = torch.where(new > 0, torch.ones_like(new), self.cond[sw, 1])
    a, tgt = self.cond[:, 0], self._alpha_tgt
    self.cond[:, 0] = torch.where(a < tgt, torch.minimum(a + self._alpha_rate, tgt), torch.maximum(a - self._alpha_rate, tgt))

  # ---- current-frame cache ---------------------------------------------------

  @torch.no_grad()
  def refresh_current(self, env_ids: torch.Tensor | None = None) -> None:
    if env_ids is None:
      g = self.library.gather(self.time_steps, fields=self._fields())
      for k in self._cur:
        self._cur[k] = g[k]
    elif env_ids.numel() > 0:
      g = self.library.gather(self.time_steps[env_ids], fields=self._fields())
      for k in self._cur:
        self._cur[k][env_ids] = g[k]

  @property
  def command(self) -> torch.Tensor:
    return self._cur["h_dir"].flatten(1)

  def _update_metrics(self):
    pass  # sampling metrics are written by sample_frames; tracking errors are logged by the trainer / evaluator

  # ---- resets ------------------------------------------------------------------

  def _resample_command(self, env_ids: torch.Tensor):
    frames = self.sample_frames(env_ids)
    self.time_steps[env_ids] = frames
    self.clip_end_env[env_ids] = self.library.clip_end_of_frame[frames].long()
    g = self.library.gather(frames, fields=INIT_FIELDS)
    n = len(env_ids)
    keys = ["x", "y", "z", "roll", "pitch", "yaw"]
    pr = torch.tensor([self.cfg.pose_range.get(k, (0.0, 0.0)) for k in keys], device=self.device)
    r1 = sample_uniform(pr[:, 0], pr[:, 1], (n, 6), device=self.device)
    root_pos = self._env.scene.env_origins[env_ids] + r1[:, 0:3]
    root_pos[:, 2] += g["init_root_height"] + self.cfg.reset_height_offset
    root_ori = quat_mul(quat_from_euler_xyz(r1[:, 3], r1[:, 4], r1[:, 5]), g["init_root_quat"])
    # Root velocity: k x the human's, from its heading frame to the world (the robot starts on the human heading).
    k = self.rest_height / g["h_rest"]
    cy, sy = torch.cos(g["h_yaw"]), torch.sin(g["h_yaw"])
    vh = g["h_root_vel"] * k[:, None]
    lin = torch.stack([cy * vh[:, 0] - sy * vh[:, 1], sy * vh[:, 0] + cy * vh[:, 1], vh[:, 2]], dim=-1)
    ang = torch.zeros(n, 3, device=self.device)
    ang[:, 2] = g["h_yaw_rate"]
    vr = torch.tensor([self.cfg.velocity_range.get(k_, (0.0, 0.0)) for k_ in keys], device=self.device)
    r2 = sample_uniform(vr[:, 0], vr[:, 1], (n, 6), device=self.device)
    joint_pos = g["init_joint_pos"].clone()
    lo, hi = self.cfg.joint_position_range
    if hi > lo:
      joint_pos += sample_uniform(lower=lo, upper=hi, size=joint_pos.shape, device=joint_pos.device)  # type: ignore
    self._write_reference_state_to_sim(env_ids, root_pos, root_ori, lin + r2[:, :3], ang + r2[:, 3:], joint_pos, g["init_joint_vel"])
    self._pending_forward = True
    if self.ee_on:
      self._sample_cond(env_ids, frames)

  def _update_command(self, env_ids: torch.Tensor | None = None):
    if env_ids is None:
      self.time_steps += 1
    else:
      self.time_steps[env_ids] += 1
    self.time_steps = torch.minimum(self.time_steps, self.clip_end_env - 1)
    self.refresh_current(env_ids)
    if self.ee_on and env_ids is None:
      self._advance_cond()
    if self._pending_forward:
      self._pending_forward = False
      self._env.sim.forward()

  # ---- simulator termination ---------------------------------------------------

  @torch.no_grad()
  def failed(self) -> torch.Tensor:
    """Heads' hard termination of the current simulator state against the current reference frame (once per step)."""
    step = int(self._env.common_step_counter)
    if self._term_cache is not None and self._term_cache[0] == step:
      return self._term_cache[1]
    if self.heads is None:
      out = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    else:
      d = self.robot.data
      pose, vel = d.root_link_pose_w, d.root_link_vel_w
      s = torch.cat([pose[:, :3] - self._env.scene.env_origins, pose[:, 3:7], vel[:, :3], vel[:, 3:6], d.joint_pos, d.joint_vel], dim=-1)
      kin = self.heads.kin(s)
      out, _ = self.heads.termination(kin, s, self.current_ref())
    self._term_cache = (step, out)
    return out


def human_failed(env: "ManagerBasedRlEnv", command_name: str) -> torch.Tensor:
  return env.command_manager.get_term(command_name).failed()  # type: ignore[attr-defined]


@dataclass(kw_only=True)
class HumanMotionCommandCfg(LibraryMotionCommandCfg):
  robot_name: str = "g1"  # key of the init_<robot>_* reset states in the npz files
  bone_map: str = "g1"
  foot_geom_regex: str = r"^(left|right)_foot[1-7]_collision$"
  reset_height_offset: float = 0.01

  def build(self, env: "ManagerBasedRlEnv") -> HumanMotionCommand:
    return HumanMotionCommand(self, env)
