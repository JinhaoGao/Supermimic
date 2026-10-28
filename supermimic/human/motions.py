"""Human-reference motion library: raw SMPL motions (no retargeting) on one global timeline.

Input npz (Noetix-RGMT ``prepare_smpl_motion_dataset.py`` + ``prepare_smpl_init_states.py``, or this repo's
scripts/human/ ports of them), 50 Hz, Z-up, floor at z = 0:
  smpl_joints_w [T,24,3], smpl_global_quat [T,22,4] (wxyz), smpl_local_rot6d [T,21,6], root_pos_w / root_quat_w /
  root_lin_vel_w / root_ang_vel_w, foot_contact [T,2], human_rest_pelvis_height (1,),
  init_<robot>_joint_names [J], init_<robot>_joint_pos / _joint_vel [T,J], init_<robot>_root_quat [T,4] (wxyz),
  init_<robot>_root_height [T], init_<robot>_ik_err_deg [T] (optional).

The cache stores per-frame quantities derived once from that (``derive_fields``), so the goal / reward / termination
heads only gather:
  h_feat [T,138]       policy goal features: gravity in the SMPL pelvis frame 3 | root velocity in the human heading
                       frame / rest pelvis height 3 | pelvis angular velocity (pelvis frame) 3 | pelvis height / rest
                       height 1 | 21 local joint rotations as rot6d 126 | foot contact 2
  h_dir, h_dir_local   [T,S,3] human bone directions in the human heading frame / in each bone's parent joint frame
  h_root_vel [T,3]     root velocity in the human heading frame (metres per second, unscaled)
  h_yaw, h_yaw_rate    [T] human heading (hip line) yaw and pelvis yaw rate
  h_height [T]         pelvis height / rest pelvis height;  h_rest [T]: rest pelvis height of the clip's subject
  h_contact [T,2]      foot contact labels (left, right)
  h_foot_lift [T,2]    ankle height above its standing height (median over the side's contact frames), metres
  init_*               robot reset state from bone-direction IK (joint order of the robot entity); used only by resets
Nothing here depends on the robot except the init_* fields (and the bone map that defines the segments).
"""

from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np
import torch

from ..motions import MotionLibrary
from . import bone_map as bone

GOAL_FEAT_DIM = 138
CACHE_VERSION = "human-v2"  # v2: + h_foot_lift


def field_shapes(num_segments: int, num_joints: int) -> dict[str, tuple[int, ...]]:
  S, J = num_segments, num_joints
  return {
    "h_feat": (GOAL_FEAT_DIM,), "h_dir": (S, 3), "h_dir_local": (S, 3), "h_root_vel": (3,), "h_yaw": (), "h_yaw_rate": (),
    "h_height": (), "h_rest": (), "h_contact": (2,), "h_foot_lift": (2,), "init_joint_pos": (J,), "init_joint_vel": (J,), "init_root_quat": (4,),
    "init_root_height": (), "init_ik_err": (),
  }


@torch.no_grad()
def derive_fields(d, bt: bone.BoneMapTensors, robot: str, joint_names: tuple[str, ...]) -> dict[str, np.ndarray]:
  """Per-frame cache fields (see module docstring) of one human npz (any mapping with its keys)."""
  f64 = lambda k: torch.as_tensor(np.asarray(d[k]), dtype=torch.float64)  # noqa: E731
  joints = f64("smpl_joints_w")
  rot = bone.quat_wxyz_to_matrix(f64("smpl_global_quat"))
  T = joints.shape[0]
  heading = bone.human_heading(joints)  # [T,3,3] columns (forward, left, up)
  h_w = bone.human_bone_dirs_w(joints, rot, bt)
  rest = float(np.asarray(d["human_rest_pelvis_height"]).reshape(-1)[0])
  v = f64("root_lin_vel_w")
  w = f64("root_ang_vel_w")
  r_root = bone.quat_wxyz_to_matrix(f64("root_quat_w"))
  g = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float64)
  gravity_b = torch.einsum("tji,j->ti", r_root, g)
  ang_vel_b = torch.einsum("tji,tj->ti", r_root, w)
  v_h = torch.einsum("tji,tj->ti", heading, v)
  height = f64("root_pos_w")[:, 2]
  contact = f64("foot_contact")
  ankle_z = joints[:, [7, 8], 2]
  down = contact > 0.5
  stance = torch.stack([ankle_z[down[:, i], i].median() if bool(down[:, i].any()) else ankle_z[:, i].min() for i in (0, 1)])
  feat = torch.cat([gravity_b, v_h / rest, ang_vel_b, (height / rest)[:, None], f64("smpl_local_rot6d").reshape(T, -1), contact], -1)
  assert feat.shape[1] == GOAL_FEAT_DIM, feat.shape
  src = [str(n) for n in d[f"init_{robot}_joint_names"]]
  perm = [src.index(n) for n in joint_names]
  ik_err = np.asarray(d[f"init_{robot}_ik_err_deg"]) if f"init_{robot}_ik_err_deg" in d else np.zeros(T)
  out = {
    "h_feat": feat, "h_dir": bone.to_frame(h_w, heading), "h_dir_local": bone.human_bone_dirs_local(h_w, rot, bt),
    "h_root_vel": v_h, "h_yaw": torch.atan2(heading[:, 1, 0], heading[:, 0, 0]), "h_yaw_rate": w[:, 2],
    "h_height": height / rest, "h_rest": torch.full((T,), rest, dtype=torch.float64), "h_contact": contact, "h_foot_lift": ankle_z - stance,
    "init_joint_pos": np.asarray(d[f"init_{robot}_joint_pos"])[:, perm], "init_joint_vel": np.asarray(d[f"init_{robot}_joint_vel"])[:, perm],
    "init_root_quat": np.asarray(d[f"init_{robot}_root_quat"]), "init_root_height": np.asarray(d[f"init_{robot}_root_height"]),
    "init_ik_err": ik_err,
  }
  return {k: np.asarray(v, dtype=np.float32) for k, v in out.items()}


def human_cache_dir(cache_root: str, files, robot: str, joint_names, bone_map: bone.SmplBoneMapCfg, fps: float) -> str:
  h = hashlib.sha1()
  for part in (CACHE_VERSION, robot, *joint_names, *bone_map.segment_names, f"{fps:g}", *files):
    h.update(part.encode() + b"\0")
  return os.path.join(cache_root, f"human_{len(files)}_{h.hexdigest()[:16]}")


def build_human_cache(files, robot, joint_names, bone_map, cache_dir, expected_fps=50.0, log=print) -> None:
  calib = bone_map.load_calibration()
  bt = bone.BoneMapTensors(bone_map, calib, dtype=torch.float64)
  shapes = field_shapes(len(bone_map.segments), len(joint_names))
  t0 = time.time()
  per, clips, bad = {k: [] for k in shapes}, [], []
  start = 0
  for i, path in enumerate(files):
    try:
      with np.load(path, allow_pickle=True) as d:
        fps = float(np.asarray(d["fps"]).reshape(-1)[0])
        if abs(fps - expected_fps) > 1e-4:
          raise ValueError(f"fps {fps}")
        f = derive_fields(d, bt, robot, tuple(joint_names))
    except Exception as e:  # noqa: BLE001 -- a broken clip is excluded, not fatal
      bad.append((path, repr(e)))
      continue
    T = f["h_feat"].shape[0]
    for k in shapes:
      per[k].append(f[k])
    clips.append({"name": os.path.basename(path), "start": start, "end": start + T, "ok": True})
    start += T
    if (i + 1) % 500 == 0:
      log(f"[human cache] {i + 1}/{len(files)} files ({time.time() - t0:.0f}s)")
  if not clips:
    raise RuntimeError(f"no usable human motion among {len(files)} files (first errors: {bad[:3]})")
  if bad:
    log(f"[human cache] WARNING: {len(bad)} files skipped (first: {bad[:3]})")
  os.makedirs(cache_dir, exist_ok=True)
  for k in shapes:
    np.save(os.path.join(cache_dir, f"{k}.npy"), np.concatenate(per[k], axis=0))
  meta = {"version": CACHE_VERSION, "robot": robot, "joint_names": list(joint_names), "segments": bone_map.segment_names,
          "fps": expected_fps, "frames": start, "files": len(files), "failed_files": len(bad), "build_seconds": time.time() - t0}
  with open(os.path.join(cache_dir, "clips.json"), "w") as fh:
    json.dump(clips, fh)
  with open(os.path.join(cache_dir, "meta.json"), "w") as fh:  # written last: marks a complete cache
    json.dump(meta, fh, indent=2)
  log(f"[human cache] {len(clips)} clips, {start} frames in {time.time() - t0:.0f}s -> {cache_dir}")


class HumanMotionLibrary(MotionLibrary):
  """Same timeline / bin / gather interface as MotionLibrary (the reset sampler, replay and windows are shared), with
  the human fields of ``field_shapes``. ``nb`` = 0 (no tracked robot bodies), ``nj`` = robot joints of the reset state."""

  def __init__(self, files, robot: str, joint_names, bone_map: bone.SmplBoneMapCfg, device: str, bin_seconds: float = 1.0,
               expected_fps: float = 50.0, cache_dir: str | None = None, log=print):
    self.device = device
    self.joint_names = tuple(joint_names)
    self.body_names = ()
    self.fps = expected_fps
    self.dt = 1.0 / expected_fps
    files = [os.path.abspath(os.path.expanduser(f)) for f in files]
    cache_root = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "supermimic_motions")
    cdir = human_cache_dir(cache_root, files, robot, self.joint_names, bone_map, expected_fps)
    if not os.path.exists(os.path.join(cdir, "meta.json")):
      build_human_cache(files, robot, self.joint_names, bone_map, cdir, expected_fps, log)
    meta = json.load(open(os.path.join(cdir, "meta.json")))
    with open(os.path.join(cdir, "clips.json"), "rb") as fh:
      clips_raw = fh.read()
    clips_all = json.loads(clips_raw)
    self.fingerprint = hashlib.blake2b(os.path.basename(cdir).encode() + f"|{meta['frames']}|".encode() + clips_raw, digest_size=12).hexdigest()
    self.file_frame_end = [int(c["end"]) for c in clips_all]  # (failed files have no entry: prefix logic is approximate)
    self.cache_dir = cdir
    self.fields = {k: torch.from_numpy(np.load(os.path.join(cdir, f"{k}.npy"))) for k in field_shapes(len(bone_map.segments), len(joint_names))}
    self.time_step_total = int(meta["frames"])
    self._build_tables(clips_all, bin_seconds, expected_fps, device)
    self.nj = len(self.joint_names)
    self.nb = 0
    log(f"[HumanMotionLibrary] {self.summary()}")

  def summary(self) -> str:
    mb = sum(t.numel() * t.element_size() for t in self.fields.values()) / 1e6
    return (f"HumanMotionLibrary: {self.num_clips} clips, {self.time_step_total} frames ({self.total_seconds / 3600:.2f} h), "
            f"{self.num_bins} bins, {self.nj} joints (reset states), {mb:.0f} MB in RAM")

  def root_state(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
    raise NotImplementedError("human references have no robot root state; HumanMotionCommand resets from init_* fields")
