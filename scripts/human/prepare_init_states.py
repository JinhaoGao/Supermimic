#!/usr/bin/env python3
"""Robot reset states for human (SMPL) motions: per-frame bone-direction IK on the simulated robot model.

Port of Noetix-RGMT scripts/prepare_smpl_init_states.py onto the robot the trainer simulates (mjlab's G1, optionally
the team eval variant, built from supermimic.env_cfg.base_tracking_cfg), so the reset height uses its own foot
collision geoms. For every frame it solves the pelvis roll / pitch and the joint angles so that all bone directions
match the human's (each in its own heading frame -- the reward's global objective), warm-started from the previous
frame; yaw = the human heading; the pelvis height puts the lowest sole on the floor while the human has a foot down,
else follows the normalized human pelvis height. Only resets use the result (never the policy input or the reward).

Adds to each npz (<r> = robot):
  init_<r>_joint_names [J] (model hinge order), init_<r>_joint_pos / _joint_vel [T,J], init_<r>_root_quat [T,4] wxyz,
  init_<r>_root_height [T], init_<r>_ik_err_deg [T] (weighted mean bone-direction error of the solution)

  cd ~/supermimic_ws/mjlab && uv run python ~/Supermimic/scripts/human/prepare_init_states.py \\
      --motion_dir ~/datasets/smpl_human_50hz --robot_variant eval+evalcontact --num_workers 16 [--output_dir <dir>]
"""

from __future__ import annotations

import argparse
import glob
import os
import time

import mujoco
import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R

from supermimic.env_cfg import _regex, base_tracking_cfg
from supermimic.human import bone_map as bone
from supermimic.human.command import body_id, foot_geoms, lowest_point


def _unit(v: np.ndarray) -> np.ndarray:
  return v / np.linalg.norm(v, axis=-1, keepdims=True).clip(1e-8)


def _heading(left: np.ndarray, right: np.ndarray) -> np.ndarray:
  l = left - right  # noqa: E741
  l = _unit(np.array([l[0], l[1], 0.0]))  # noqa: E741
  return np.stack([np.cross(l, [0.0, 0.0, 1.0]), l, [0.0, 0.0, 1.0]], axis=-1)


def build_model(robot: str, variant: str) -> tuple[mujoco.MjModel, str]:
  cfg = base_tracking_cfg(robot, variant)
  model = cfg.scene.entities["robot"].spec_fn().compile()
  return model, _regex(cfg.events["foot_friction"].params["asset_cfg"].geom_names)


class RobotFK:
  """MuJoCo kinematics of the bone map's links for the floating-base robot model."""

  def __init__(self, model: mujoco.MjModel, bone_map, foot_regex: str):
    self.model = model
    self.data = mujoco.MjData(model)
    self.link_ids = [body_id(model, n) for n in bone_map.robot_links]
    hinge = [i for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
    self.joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i).split("/")[-1] for i in hinge]
    self.qpos_adr = np.array([model.jnt_qposadr[i] for i in hinge])
    self.joint_range = np.array([model.jnt_range[i] if model.jnt_limited[i] else (-np.pi, np.pi) for i in hinge])
    self.feet = foot_geoms(model, foot_regex)
    free = [i for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE]
    self.root_adr = int(model.jnt_qposadr[free[0]])

  def joint_vector(self, joints: dict[str, float]) -> np.ndarray:
    q = np.zeros(len(self.joint_names))
    for name, value in joints.items():
      q[self.joint_names.index(name)] = value
    return q

  def set(self, root_rotvec, joints) -> None:
    d = self.data
    a = self.root_adr
    d.qpos[a : a + 3] = 0.0
    d.qpos[a + 3 : a + 7] = R.from_rotvec(root_rotvec).as_quat()[[3, 0, 1, 2]]
    d.qpos[self.qpos_adr] = joints
    mujoco.mj_kinematics(self.model, d)

  def __call__(self, root_rotvec, joints):
    self.set(root_rotvec, joints)
    return self.data.xpos[self.link_ids].copy(), self.data.xmat[self.link_ids].reshape(-1, 3, 3).copy()

  def sole_z(self) -> float:
    return lowest_point(self.model, self.data, self.feet)


class BoneIK:
  def __init__(self, fk: RobotFK, bone_map, calib: dict, reg: float, smooth: float):
    self.fk = fk
    self.bt = bone.BoneMapTensors(bone_map, calib, dtype=torch.float64)
    bt = self.bt
    self.r_from, self.r_to = bt.r_from.numpy(), bt.r_to.numpy()
    self.geometric = bt.geometric.numpy()[:, None]
    self.axes = bt.robot_axes.numpy()
    self.sqrt_w = np.sqrt(bt.weights.numpy())[:, None]
    self.q_default = fk.joint_vector(bone_map.default_joint_pos)
    self.reg, self.smooth = np.sqrt(reg), np.sqrt(smooth)
    lo = np.concatenate([[-0.8, -0.8], fk.joint_range[:, 0]])
    hi = np.concatenate([[0.8, 0.8], fk.joint_range[:, 1]])
    self.bounds = (lo, hi)

  def robot_dirs_heading(self, x: np.ndarray) -> np.ndarray:
    pos, rot = self.fk(R.from_euler("YX", [x[1], x[0]]).as_rotvec(), x[2:])
    geo = _unit(pos[self.r_to] - pos[self.r_from])
    cal = np.einsum("sij,sj->si", rot[self.r_from], self.axes)
    dirs = np.where(self.geometric, geo, cal)
    return dirs @ _heading(pos[self.bt.r_heading[0]], pos[self.bt.r_heading[1]])

  def residual(self, x, target, x_prev):
    res = [(self.sqrt_w * (self.robot_dirs_heading(x) - target)).ravel(), self.reg * (x[2:] - self.q_default)]
    if x_prev is not None:
      res.append(self.smooth * (x - x_prev))
    return np.concatenate(res)

  def solve(self, target, x0, x_prev, max_nfev: int) -> np.ndarray:
    x0 = np.clip(x0, self.bounds[0] + 1e-6, self.bounds[1] - 1e-6)
    return least_squares(self.residual, x0, args=(target, x_prev), bounds=self.bounds, max_nfev=max_nfev).x

  def error_deg(self, x, target) -> float:
    ang = np.degrees(np.arccos(np.clip((self.robot_dirs_heading(x) * target).sum(-1), -1.0, 1.0)))
    w = self.bt.weights.numpy()
    return float((ang * w).sum() / w.sum())


def process(path: str, ik: BoneIK, robot_rest: float, args) -> np.ndarray:
  data = dict(np.load(path, allow_pickle=True))
  bt = ik.bt
  joints = torch.as_tensor(data["smpl_joints_w"], dtype=torch.float64)
  rots = bone.quat_wxyz_to_matrix(torch.as_tensor(data["smpl_global_quat"], dtype=torch.float64))
  h_heading = bone.human_heading(joints)
  targets = bone.to_frame(bone.human_bone_dirs_w(joints, rots, bt), h_heading).numpy()
  yaw = np.arctan2(h_heading[:, 1, 0].numpy(), h_heading[:, 0, 0].numpy())
  num = targets.shape[0]
  xs = np.zeros((num, 2 + len(ik.fk.joint_names)))
  err = np.zeros(num)
  x = np.concatenate([[0.0, 0.0], ik.q_default])
  for t in range(num):
    x = ik.solve(targets[t], x, x if t > 0 else None, args.max_nfev if t > 0 else 4 * args.max_nfev)
    xs[t], err[t] = x, ik.error_deg(x, targets[t])
  rot = R.from_euler("ZYX", np.stack([yaw, xs[:, 1], xs[:, 0]], axis=-1))
  joint_pos = xs[:, 2:]
  human_rest = float(np.asarray(data["human_rest_pelvis_height"]).reshape(-1)[0])
  h_target = data["root_pos_w"][:, 2] / human_rest * robot_rest
  height = np.empty(num)
  for t in range(num):
    ik.fk.set(R.from_euler("YX", [xs[t, 1], xs[t, 0]]).as_rotvec(), joint_pos[t])
    on_floor = -ik.fk.sole_z()
    height[t] = on_floor if data["foot_contact"][t].any() else max(on_floor, h_target[t])
  dt = 1.0 / float(np.asarray(data["fps"]).reshape(-1)[0])
  win = min(9, (num // 2) * 2 - 1)
  joint_vel = savgol_filter(joint_pos, win, 3, deriv=1, delta=dt, axis=0) if win >= 5 else np.gradient(joint_pos, dt, axis=0)
  joint_vel = np.clip(joint_vel, -args.max_joint_vel, args.max_joint_vel)
  r = args.robot
  data[f"init_{r}_joint_names"] = np.array(ik.fk.joint_names)
  data[f"init_{r}_joint_pos"] = joint_pos.astype(np.float32)
  data[f"init_{r}_joint_vel"] = joint_vel.astype(np.float32)
  data[f"init_{r}_root_quat"] = rot.as_quat()[:, [3, 0, 1, 2]].astype(np.float32)
  data[f"init_{r}_root_height"] = height.astype(np.float32)
  data[f"init_{r}_ik_err_deg"] = err.astype(np.float32)
  out = os.path.join(args.output_dir, os.path.basename(path)) if args.output_dir else path
  np.savez(out, **data)
  return err


_IK: dict = {}


def _work(job):
  path, args = job
  if "ik" not in _IK:
    torch.set_num_threads(1)
    model, regex = build_model(args.robot, args.robot_variant)
    bone_map = bone.SMPL_BONE_MAPS[args.robot]
    fk = RobotFK(model, bone_map, regex)
    from supermimic.human.command import robot_rest_geometry

    _IK["rest"] = robot_rest_geometry(model, bone_map.pelvis_link, bone_map.foot_links, regex)[0]
    _IK["ik"] = BoneIK(fk, bone_map, bone_map.load_calibration(), args.reg, args.smooth)
  t0 = time.time()
  try:
    err = process(path, _IK["ik"], _IK["rest"], args)
  except Exception as exc:  # noqa: BLE001
    return f"[skip] {os.path.basename(path)}: {exc!r}"
  return (f"[{os.path.basename(path)[:55]}] T={len(err)} IK bone err mean {err.mean():.1f} deg, "
          f"p95 {np.percentile(err, 95):.1f} deg ({time.time() - t0:.0f}s)")


def main():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--motion_dir", required=True, help="npz files from prepare_smpl_motion.py (updated in place unless --output_dir)")
  p.add_argument("--output_dir", default="")
  p.add_argument("--robot", default="g1", choices=sorted(bone.SMPL_BONE_MAPS))
  p.add_argument("--robot_variant", default="eval+evalcontact", help="the variant the trainer simulates (env_cfg.base_tracking_cfg)")
  p.add_argument("--reg", type=float, default=0.01, help="pull towards the default pose")
  p.add_argument("--smooth", type=float, default=0.1, help="pull towards the previous frame's solution")
  p.add_argument("--max_nfev", type=int, default=20)
  p.add_argument("--max_joint_vel", type=float, default=10.0)
  p.add_argument("--num_workers", type=int, default=1)
  p.add_argument("--skip_existing", action="store_true", help="skip files that already carry init_<robot>_* states")
  args = p.parse_args()
  if args.output_dir:
    os.makedirs(args.output_dir, exist_ok=True)
  files = sorted(glob.glob(os.path.join(os.path.expanduser(args.motion_dir), "*.npz")))
  if args.skip_existing:
    def done(f):
      out = os.path.join(args.output_dir, os.path.basename(f)) if args.output_dir else f
      return os.path.exists(out) and f"init_{args.robot}_ik_err_deg" in np.load(out).files  # npz keys have no .npy
    files = [f for f in files if not done(f)]
  print(f"{len(files)} clips")
  jobs = [(f, args) for f in files]
  if args.num_workers > 1:
    from multiprocessing import get_context

    with get_context("spawn").Pool(args.num_workers) as pool:
      for msg in pool.imap_unordered(_work, jobs, chunksize=1):
        print(msg, flush=True)
  else:
    for job in jobs:
      print(_work(job), flush=True)


if __name__ == "__main__":
  main()
