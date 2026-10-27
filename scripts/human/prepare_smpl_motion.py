#!/usr/bin/env python3
"""Build robot-agnostic SMPL motion NPZs (no scaling, no retargeting) from AMASS SMPL-X.

Each output NPZ holds raw human motion at ``--fps`` in a Z-up world whose floor is the
lowest foot joint (z = 0):

  fps                  (1,)
  smpl_joints_w        (T, 24, 3)  22 body joints + left/right hand points, metres
  smpl_global_quat     (T, 22, 4)  per-joint world rotations, wxyz
  smpl_local_rot6d     (T, 21, 6)  body-joint local rotations (first two matrix columns)
  root_pos_w           (T, 3)      pelvis joint position
  root_quat_w          (T, 4)      SMPL pelvis world rotation, wxyz
  root_lin_vel_w       (T, 3)
  root_ang_vel_w       (T, 3)
  foot_contact         (T, 2)      left/right, float 0/1
  human_rest_pelvis_height  (1,)   subject T-pose pelvis height above the foot joints
  human_rest_torso_height   (1,)   subject T-pose shoulder-midpoint height above the foot joints
  betas, gender, source

Robot reset states are added afterwards by scripts/human/prepare_init_states.py.

Usage:
  ~/miniconda3/envs/noetix_gmt/bin/python scripts/human/prepare_smpl_motion.py \
    --amass_paths ~/datasets/smplkp_full/hs_amass_input --smplx_model_dir ~/models/smplx_v1_1/models \
    --output_dir ~/datasets/smpl_human_50hz [--names clip_names.txt] [--num_workers 16]

Ported from Noetix-RGMT scripts/prepare_smpl_motion_dataset.py (same output). Added: --names (only these clips, by
output name), --num_workers, --skip_existing; flat input directories (dataset_subject_clip_stageii.npz) keep their name.
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from smpl_motion_utils import SmplxBody, bone, global_rotations


def find_clips(paths: list[str]) -> list[str]:
  clips = []
  for path in paths:
    if os.path.isdir(path):
      clips.extend(glob.glob(os.path.join(path, "**", "*_stageii.npz"), recursive=True))
    elif path.endswith(".npz"):
      clips.append(path)
  return sorted(set(clips))


def clip_name(path: str) -> str:
  """AMASS layout is <dataset>/<subject>/<clip>_stageii.npz -> dataset_subject_clip."""
  parts = os.path.abspath(path).split(os.sep)[-3:]
  return "_".join(parts).replace("_stageii.npz", "").replace(".npz", "")


def resample_rotvec(rotvec: np.ndarray, t_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
  """Slerp each joint's axis-angle track: (T, J*3) -> (T', J*3)."""
  n_j = rotvec.shape[1] // 3
  out = np.empty((len(t_dst), n_j, 3), dtype=np.float64)
  for j in range(n_j):
    out[:, j] = Slerp(t_src, R.from_rotvec(rotvec[:, 3 * j : 3 * j + 3]))(t_dst).as_rotvec()
  return out.reshape(len(t_dst), -1)


def smooth_diff(x: np.ndarray, dt: float, window: int) -> np.ndarray:
  win = min(window, (x.shape[0] // 2) * 2 - 1)
  if win >= 5:
    return savgol_filter(x, win, 3, deriv=1, delta=dt, axis=0, mode="interp")
  return np.gradient(x, dt, axis=0)


def angular_velocity(rot: R, dt: float, window: int) -> np.ndarray:
  """World-frame angular velocity of a rotation track."""
  rel = rot[1:] * rot[:-1].inv()
  omega_mid = rel.as_rotvec() / dt
  omega = np.empty((len(rot), 3))
  omega[1:-1] = 0.5 * (omega_mid[1:] + omega_mid[:-1])
  omega[0], omega[-1] = omega_mid[0], omega_mid[-1]
  win = min(window, (len(rot) // 2) * 2 - 1)
  return savgol_filter(omega, win, 3, axis=0, mode="interp") if win >= 5 else omega


def process_clip(path: str, args) -> tuple[dict, dict]:
  src = np.load(path, allow_pickle=True)
  src_fps = float(src["mocap_frame_rate"])
  gender = str(src["gender"]) if args.gender is None else args.gender
  betas = np.asarray(src["betas"], dtype=np.float32)
  body = SmplxBody(args.smplx_model_dir, gender, betas)

  t_src = np.arange(src["trans"].shape[0]) / src_fps
  t_dst = np.arange(0.0, t_src[-1] + 1e-9, 1.0 / args.fps)
  t_dst = np.minimum(t_dst, t_src[-1])  # the +1e-9 end may overshoot t_src[-1] by rounding: Slerp refuses that
  root_orient = resample_rotvec(np.asarray(src["root_orient"], np.float64), t_src, t_dst)
  pose_body = resample_rotvec(np.asarray(src["pose_body"], np.float64), t_src, t_dst)
  pose_hand = resample_rotvec(np.asarray(src["pose_hand"], np.float64), t_src, t_dst) if "pose_hand" in src else None
  trans = np.stack([np.interp(t_dst, t_src, src["trans"][:, i]) for i in range(3)], axis=-1)
  num = len(t_dst)
  dt = 1.0 / args.fps

  joints = body.joints(root_orient, pose_body, trans, pose_hand).astype(np.float64)
  foot_z = joints[:, list(bone.SMPL_FOOT_JOINTS), 2]
  floor = np.percentile(foot_z.min(axis=1), args.floor_percentile)
  joints[..., 2] -= floor

  glob_rot = global_rotations(root_orient, pose_body)
  glob_quat_xyzw = R.from_matrix(glob_rot.reshape(-1, 3, 3)).as_quat().reshape(num, 22, 4)
  root_rot = R.from_matrix(glob_rot[:, 0])
  local = np.concatenate([root_orient[:, None], pose_body.reshape(num, 21, 3)], axis=1)
  local_m = R.from_rotvec(local[:, 1:].reshape(-1, 3)).as_matrix().reshape(num, 21, 3, 3)
  local_rot6d = np.concatenate([local_m[..., :, 0], local_m[..., :, 1]], axis=-1)

  root_pos = joints[:, bone.SMPL_PELVIS]
  root_lin_vel = smooth_diff(root_pos, dt, args.smooth_window)
  root_ang_vel = angular_velocity(root_rot, dt, args.smooth_window)

  # Foot contact: low and slow foot joints.
  foot_pos = joints[:, list(bone.SMPL_FOOT_JOINTS)]
  foot_speed = np.linalg.norm(smooth_diff(foot_pos, dt, args.smooth_window)[..., :2], axis=-1)
  contact = (foot_pos[..., 2] < args.contact_height) & (foot_speed < args.contact_speed)

  tpose = body.tpose_joints()
  out = {
    "fps": np.array([args.fps], dtype=np.float32),
    "smpl_joints_w": joints.astype(np.float32),
    "smpl_global_quat": glob_quat_xyzw[..., [3, 0, 1, 2]].astype(np.float32),
    "smpl_local_rot6d": local_rot6d.astype(np.float32),
    "root_pos_w": root_pos.astype(np.float32),
    "root_quat_w": root_rot.as_quat()[:, [3, 0, 1, 2]].astype(np.float32),
    "root_lin_vel_w": root_lin_vel.astype(np.float32),
    "root_ang_vel_w": root_ang_vel.astype(np.float32),
    "foot_contact": contact.astype(np.float32),
    "human_rest_pelvis_height": np.array([tpose[bone.SMPL_PELVIS, 2]], dtype=np.float32),
    "human_rest_torso_height": np.array([tpose[[16, 17], 2].mean()], dtype=np.float32),
    "betas": betas,
    "gender": np.array(body.gender),
    "source": np.array(os.path.abspath(path)),
  }
  stats = {"frames": num, "src_fps": src_fps, "gender": body.gender, "contact": contact.mean(axis=0)}
  return out, stats


def _work(job):
  path, out_path, args = job
  name = os.path.splitext(os.path.basename(out_path))[0]
  try:
    out, st = process_clip(path, args)
  except Exception as exc:  # keep going on malformed clips
    return f"[skip] {name}: {exc}", False
  np.savez(out_path, **out)
  return (f"[{name}] T={st['frames']} ({st['src_fps']:.0f}->{args.fps:.0f} Hz, {st['gender']}) "
          f"rest_h={out['human_rest_pelvis_height'][0]:.3f} contact L/R={st['contact'][0]:.2f}/{st['contact'][1]:.2f}"), True


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--amass_paths", nargs="+", required=True, help="AMASS *_stageii.npz files or directories.")
  parser.add_argument("--smplx_model_dir", required=True, help="Directory containing smplx/SMPLX_*.npz.")
  parser.add_argument("--output_dir", required=True)
  parser.add_argument("--names", default=None, help="Text file of output clip names (one per line, '_stageii' / '.npz' "
                                                    "suffixes ignored): only these clips.")
  parser.add_argument("--num_workers", type=int, default=1)
  parser.add_argument("--skip_existing", action="store_true")
  parser.add_argument("--fps", type=float, default=50.0)
  parser.add_argument("--gender", default=None, help="Override the gender stored in the AMASS file.")
  parser.add_argument("--smooth_window", type=int, default=9)
  parser.add_argument("--floor_percentile", type=float, default=2.0)
  parser.add_argument("--contact_height", type=float, default=0.05, help="Foot-joint height threshold [m].")
  parser.add_argument("--contact_speed", type=float, default=0.3, help="Foot-joint xy speed threshold [m/s].")
  args = parser.parse_args()

  clips = find_clips(args.amass_paths)
  if not clips:
    raise SystemExit("No *_stageii.npz files found.")
  if args.names:
    strip = lambda n: n.strip().replace(".npz", "").replace("_stageii", "")  # noqa: E731
    wanted = {strip(n) for n in open(args.names) if n.strip()}
    clips = [c for c in clips if clip_name(c) in wanted]
    print(f"{len(clips)} of {len(wanted)} requested clips found")
  os.makedirs(args.output_dir, exist_ok=True)
  jobs = [(c, os.path.join(args.output_dir, f"{clip_name(c)}.npz"), args) for c in clips]
  if args.skip_existing:
    jobs = [j for j in jobs if not os.path.exists(j[1])]
  written = 0
  if args.num_workers > 1:
    from multiprocessing import Pool

    with Pool(args.num_workers) as pool:
      results = pool.imap_unordered(_work, jobs, chunksize=4)
      for msg, ok in results:
        print(msg, flush=True)
        written += ok
  else:
    for job in jobs:
      msg, ok = _work(job)
      print(msg, flush=True)
      written += ok
  print(f"wrote {written}/{len(jobs)} clips to {args.output_dir}")


if __name__ == "__main__":
  main()
