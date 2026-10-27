"""Motion library: many clips concatenated on one global timeline, held in CPU
RAM, with per-frame index tables on the GPU and gather helpers that move only
the frames a consumer needs (current frames for the simulator, K-frame goal
windows, (H+K)-frame windows for imagined rollouts).

Compact storage: joint_pos / body_pos / body_quat in fp32, velocities in fp16
(~734 B per frame for 29 joints and 14 tracked bodies; 1500 h ~ 200 GB).

Input npz format (our GMT / BeyondMimic convention):
  fps, joint_names[J], joint_pos[T,J], joint_vel[T,J], body_names[Bn],
  body_pos_w[T,Bn,3], body_quat_w[T,Bn,4] (wxyz), body_lin_vel_w, body_ang_vel_w.
Joints are remapped by name to the entity's joint order; bodies to the tracked
body list. A cache (one .npy per field + json) is built once per file list.
Older G1 renders have no body_names key; their 37-body arrays are in the fixed
Isaac order G1_ISAAC_BODY_NAMES_37, which is used as the fallback.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import zipfile
from dataclasses import dataclass
from multiprocessing import Pool

import numpy as np
import torch

FIELD_DTYPES = {
  "joint_pos": np.float32,
  "joint_vel": np.float16,
  "body_pos": np.float32,
  "body_quat": np.float32,
  "body_lin_vel": np.float16,
  "body_ang_vel": np.float16,
}
NPZ_KEYS = {
  "joint_pos": "joint_pos",
  "joint_vel": "joint_vel",
  "body_pos": "body_pos_w",
  "body_quat": "body_quat_w",
  "body_lin_vel": "body_lin_vel_w",
  "body_ang_vel": "body_ang_vel_w",
}

# Body order of the body_*_w arrays in GMT / MotionAtlas G1 npz files (Isaac order, 37 bodies).
# Older renders (e.g. bj31 ~/motion_dataset/g1_atlas_full, 09-23) were written without a body_names
# key; they were checked against the MotionAtlas motion50 originals, which carry this exact list:
# same index -> same body (numerically equal up to FK float noise).
G1_ISAAC_BODY_NAMES_37 = (
  "pelvis", "left_hip_pitch_link", "pelvis_contour_link", "right_hip_pitch_link", "waist_yaw_link",
  "left_hip_roll_link", "right_hip_roll_link", "waist_roll_link", "left_hip_yaw_link", "right_hip_yaw_link",
  "torso_link", "left_knee_link", "right_knee_link", "head_link", "left_shoulder_pitch_link", "logo_link",
  "right_shoulder_pitch_link", "left_ankle_pitch_link", "right_ankle_pitch_link", "left_shoulder_roll_link",
  "right_shoulder_roll_link", "left_ankle_roll_link", "right_ankle_roll_link", "left_shoulder_yaw_link",
  "right_shoulder_yaw_link", "LL_FOOT", "LR_FOOT", "left_elbow_link", "right_elbow_link",
  "left_wrist_roll_link", "right_wrist_roll_link", "left_wrist_pitch_link", "right_wrist_pitch_link",
  "left_wrist_yaw_link", "right_wrist_yaw_link", "left_rubber_hand", "right_rubber_hand",
)
# Fallback source body order by body count, for npz files without a body_names key.
FALLBACK_BODY_NAMES: dict[int, tuple[str, ...]] = {37: G1_ISAAC_BODY_NAMES_37}
# Joint order of the same Isaac G1 renders (29 joints). The AMASS retarget set (smplkp_full/g1_amass_retarget_npz,
# 10-22) has neither name key; checked 10-27: mjlab FK of its joint_pos in this order reproduces its body_pos_w in
# G1_ISAAC_BODY_NAMES_37 order to 5e-7 m (the same files' g1_smpl_kp_50hz copies carry this joint_names list).
G1_ISAAC_JOINT_NAMES_29 = (
  "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint", "left_hip_roll_joint", "right_hip_roll_joint",
  "waist_roll_joint", "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint", "left_knee_joint",
  "right_knee_joint", "left_shoulder_pitch_joint", "right_shoulder_pitch_joint", "left_ankle_pitch_joint",
  "right_ankle_pitch_joint", "left_shoulder_roll_joint", "right_shoulder_roll_joint", "left_ankle_roll_joint",
  "right_ankle_roll_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint", "left_elbow_joint", "right_elbow_joint",
  "left_wrist_roll_joint", "right_wrist_roll_joint", "left_wrist_pitch_joint", "right_wrist_pitch_joint",
  "left_wrist_yaw_joint", "right_wrist_yaw_joint",
)
# Fallback source joint order by (joint count, body count): only together with the G1 Isaac body fallback.
FALLBACK_JOINT_NAMES: dict[tuple[int, int], tuple[str, ...]] = {(29, 37): G1_ISAAC_JOINT_NAMES_29}


def npz_body_names(d, num_bodies: int | None = None) -> tuple[list[str], bool]:
  """Source body order of a loaded motion npz -> (names, used_fallback).

  Uses the file's body_names key when present; otherwise the fixed order for its
  body count (FALLBACK_BODY_NAMES). ``num_bodies`` avoids re-reading body_pos_w
  when the caller already has it. Raises KeyError when neither is available."""
  if "body_names" in d.files:
    return [str(x) for x in d["body_names"]], False
  nb = int(num_bodies) if num_bodies is not None else int(d["body_pos_w"].shape[1])
  fb = FALLBACK_BODY_NAMES.get(nb)
  if fb is None:
    raise KeyError(f"no body_names key and no fallback body order for {nb} bodies")
  return list(fb), True


@dataclass
class ClipInfo:
  name: str
  start: int
  end: int  # exclusive

  @property
  def length(self) -> int:
    return self.end - self.start


# ---------------------------------------------------------------------------
# Cache building (multiprocess)
# ---------------------------------------------------------------------------


def _npz_frames(path: str) -> int:
  """Number of frames of an npz without loading the arrays."""
  with zipfile.ZipFile(path) as z:
    with z.open("joint_pos.npy") as f:
      ver = np.lib.format.read_magic(f)
      if ver == (1, 0):
        shape, _, _ = np.lib.format.read_array_header_1_0(f)
      else:
        shape, _, _ = np.lib.format.read_array_header_2_0(f)
  return int(shape[0])


_W: dict = {}


def _worker_init(cache_dir, joint_names, body_names, fields_shape):
  _W["cache_dir"] = cache_dir
  _W["joint_names"] = list(joint_names)
  _W["body_names"] = list(body_names)
  _W["mm"] = {k: np.lib.format.open_memmap(os.path.join(cache_dir, f"{k}.npy"), mode="r+") for k in FIELD_DTYPES}


def _worker_fill(job):
  """Write one file's frames into the cache -> (path, error or None, used body-name fallback)."""
  path, start, T, expected_fps = job
  d = np.load(path)
  fps = float(np.asarray(d["fps"]).reshape(-1)[0])
  if not math.isclose(fps, expected_fps, rel_tol=1e-6):
    return (path, f"fps {fps}", False)
  # Every npz member is decompressed on each access: read each array once.
  joint_pos = d["joint_pos"]
  body_pos = d["body_pos_w"]
  try:
    src_b, fallback = npz_body_names(d, body_pos.shape[1])
  except KeyError as e:
    return (path, str(e), False)
  if "joint_names" in d.files:
    src_j = [str(x) for x in d["joint_names"]]
  elif fallback and (joint_pos.shape[1], body_pos.shape[1]) in FALLBACK_JOINT_NAMES:
    src_j = list(FALLBACK_JOINT_NAMES[(joint_pos.shape[1], body_pos.shape[1])])
  else:
    return (path, "no joint_names key (and no fallback joint order)", fallback)
  try:
    jperm = [src_j.index(n) for n in _W["joint_names"]]
    bperm = [src_b.index(n) for n in _W["body_names"]]
  except ValueError as e:
    return (path, f"name mismatch{' (fallback body order)' if fallback else ''}: {e}", fallback)
  if int(joint_pos.shape[0]) != T:
    return (path, "frame count changed", fallback)
  mm = _W["mm"]
  mm["joint_pos"][start : start + T] = joint_pos[:, jperm]
  mm["joint_vel"][start : start + T] = d["joint_vel"][:, jperm]
  mm["body_pos"][start : start + T] = body_pos[:, bperm]
  mm["body_quat"][start : start + T] = d["body_quat_w"][:, bperm]
  mm["body_lin_vel"][start : start + T] = d["body_lin_vel_w"][:, bperm]
  mm["body_ang_vel"][start : start + T] = d["body_ang_vel_w"][:, bperm]
  return (path, None, fallback)


def build_cache(files, joint_names, body_names, cache_dir, expected_fps=50.0, num_workers=32, log=print):
  os.makedirs(cache_dir, exist_ok=True)
  t0 = time.time()
  with Pool(num_workers) as pool:
    lengths = pool.map(_npz_frames, files, chunksize=64)
  starts = np.cumsum([0] + lengths[:-1]).tolist()
  T = int(sum(lengths))
  J, Bt = len(joint_names), len(body_names)
  shapes = {
    "joint_pos": (T, J), "joint_vel": (T, J), "body_pos": (T, Bt, 3), "body_quat": (T, Bt, 4),
    "body_lin_vel": (T, Bt, 3), "body_ang_vel": (T, Bt, 3),
  }
  log(f"[cache] {len(files)} files, {T} frames ({T / expected_fps / 3600:.1f} h), headers in {time.time() - t0:.0f}s; allocating {cache_dir}")
  for k, shp in shapes.items():
    np.lib.format.open_memmap(os.path.join(cache_dir, f"{k}.npy"), mode="w+", dtype=FIELD_DTYPES[k], shape=shp).flush()
  jobs = [(f, s, n, expected_fps) for f, s, n in zip(files, starts, lengths)]
  bad = []
  n_fallback = 0
  t1 = time.time()
  with Pool(num_workers, initializer=_worker_init, initargs=(cache_dir, joint_names, body_names, shapes)) as pool:
    for i, (path, err, fallback) in enumerate(pool.imap_unordered(_worker_fill, jobs, chunksize=16)):
      if err is not None:
        bad.append((path, err))
      elif fallback:
        n_fallback += 1
      if (i + 1) % 20000 == 0:
        log(f"[cache] {i + 1}/{len(files)} files written ({time.time() - t1:.0f}s)")
  if n_fallback:
    log(f"[cache] {n_fallback}/{len(files)} files have no body_names key -> fixed fallback body order used")
  if bad:
    log(f"[cache] WARNING: {len(bad)} files failed (first: {bad[:3]}); their frames stay zero -> excluded below")
  bad_set = {b[0] for b in bad}
  clips = [{"name": os.path.basename(f), "start": s, "end": s + n, "ok": f not in bad_set} for f, s, n in zip(files, starts, lengths)]
  meta = {
    "joint_names": list(joint_names), "body_names": list(body_names), "fps": expected_fps, "frames": T, "files": len(files),
    "failed_files": len(bad), "body_names_fallback_files": n_fallback, "build_seconds": time.time() - t0,
  }
  with open(os.path.join(cache_dir, "clips.json"), "w") as f:
    json.dump(clips, f)
  with open(os.path.join(cache_dir, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
  log(f"[cache] done in {time.time() - t0:.0f}s")


def library_cache_dir(cache_root: str, files, joint_names, body_names, expected_fps: float) -> str:
  """Cache directory of a file list: lib_<n>_<key>, key = sha1 of the exact ordered paths + TARGET joint / body names
  (entity order, tracked bodies) + fps. The source body order (npz body_names or the fallback) is not part of it: both
  yield the same cache contents. MotionLibrary and extend_cache use this one function."""
  files = list(files)
  key = hashlib.sha1(("\n".join(files) + "|" + ",".join(joint_names) + "|" + ",".join(body_names) + f"|{expected_fps}").encode()).hexdigest()[:10]
  return os.path.join(cache_root, f"lib_{len(files)}_{key}")


def extend_cache(old_dir: str, old_files, extra_files, new_dir: str, num_workers: int = 32, chunk_frames: int = 2_000_000, log=print) -> dict:
  """Cache of old_files + extra_files (in that order) from the cache of old_files without re-reading it (master c1c94be):
  the old arrays are copied row by row (bitwise), only the extra npz files are read, through the same per-file writer as
  build_cache (_worker_fill: joint / body remap by name, body_names fallback for npz files without the key, fps and
  frame-count checks; a failing file keeps zero frames and ok=false, exactly as in a full build). clips.json = the old
  one (same ok flags -> the old clips keep their bins and bin indices) + the extra clips; meta.json is written last
  (MotionLibrary treats it as "cache complete"). Returns the new meta."""
  old_files, extra_files = list(old_files), list(extra_files)
  meta = json.load(open(os.path.join(old_dir, "meta.json")))
  J, B, fps = meta["joint_names"], meta["body_names"], float(meta["fps"])
  assert int(meta.get("files", len(old_files))) == len(old_files), f"{old_dir}: {meta.get('files')} files, list has {len(old_files)}"
  clips = json.load(open(os.path.join(old_dir, "clips.json")))
  assert len(clips) == len(old_files) and [c["name"] for c in clips] == [os.path.basename(f) for f in old_files], \
    f"{old_dir}/clips.json does not match the old list"
  assert not os.path.exists(os.path.join(new_dir, "meta.json")), f"{new_dir} exists"
  t0 = time.time()
  with Pool(num_workers) as pool:
    lengths = pool.map(_npz_frames, extra_files, chunksize=16)
  T_old = int(meta["frames"])
  T = T_old + int(sum(lengths))
  os.makedirs(new_dir, exist_ok=True)
  shapes = {}
  for k in FIELD_DTYPES:
    old = np.load(os.path.join(old_dir, f"{k}.npy"), mmap_mode="r")
    assert old.shape[0] == T_old and old.dtype == FIELD_DTYPES[k], (k, old.shape, old.dtype)
    shapes[k] = (T,) + old.shape[1:]
    new = np.lib.format.open_memmap(os.path.join(new_dir, f"{k}.npy"), mode="w+", dtype=FIELD_DTYPES[k], shape=shapes[k])
    for i in range(0, T_old, chunk_frames):
      j = min(i + chunk_frames, T_old)
      new[i:j] = old[i:j]
    new.flush()
    del new, old
    log(f"[extend] copied {k} ({time.time() - t0:.0f}s)")
  starts = (T_old + np.cumsum([0] + lengths[:-1])).tolist()
  jobs = [(f, int(s), int(n), fps) for f, s, n in zip(extra_files, starts, lengths)]
  bad, n_fallback = [], 0
  with Pool(num_workers, initializer=_worker_init, initargs=(new_dir, J, B, shapes)) as pool:
    for path, err, fallback in pool.imap_unordered(_worker_fill, jobs, chunksize=8):
      if err is not None:
        bad.append((path, err))
      elif fallback:
        n_fallback += 1
  if n_fallback:
    log(f"[extend] {n_fallback}/{len(extra_files)} extra files have no body_names key -> fixed fallback body order used")
  if bad:
    log(f"[extend] WARNING: {len(bad)} extra files failed (first: {bad[:3]}); their frames stay zero -> excluded")
  bad_set = {b[0] for b in bad}
  clips += [{"name": os.path.basename(f), "start": int(s), "end": int(s) + int(n), "ok": f not in bad_set} for f, s, n in zip(extra_files, starts, lengths)]
  with open(os.path.join(new_dir, "clips.json"), "w") as f:
    json.dump(clips, f)
  m2 = {"joint_names": J, "body_names": B, "fps": fps, "frames": T, "files": len(old_files) + len(extra_files),
        "failed_files": int(meta.get("failed_files", 0)) + len(bad),
        "body_names_fallback_files": int(meta.get("body_names_fallback_files", 0)) + n_fallback,
        "build_seconds": time.time() - t0, "extended_from": old_dir, "extra_files": len(extra_files), "extra_failed": len(bad)}
  with open(os.path.join(new_dir, "meta.json"), "w") as f:  # last: MotionLibrary treats meta.json as "cache complete"
    json.dump(m2, f, indent=2)
  log(f"[extend] done in {time.time() - t0:.0f}s: {m2['files']} files, {T} frames -> {new_dir}")
  return m2


# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------


class MotionLibrary:
  def __init__(
    self,
    files: list[str],
    joint_names: tuple[str, ...],
    body_names: tuple[str, ...],
    device: str,
    bin_seconds: float = 1.0,
    expected_fps: float = 50.0,
    cache_dir: str | None = None,
    num_workers: int = 32,
    log=print,
  ):
    self.device = device
    self.joint_names = tuple(joint_names)
    self.body_names = tuple(body_names)
    self.fps = expected_fps
    self.dt = 1.0 / expected_fps
    files = list(files)
    cache_root = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "supermimic_motions")
    cdir = library_cache_dir(cache_root, files, self.joint_names, self.body_names, expected_fps)
    if not os.path.exists(os.path.join(cdir, "meta.json")):
      build_cache(files, self.joint_names, self.body_names, cdir, expected_fps, num_workers, log)
    meta = json.load(open(os.path.join(cdir, "meta.json")))
    with open(os.path.join(cdir, "clips.json"), "rb") as f:
      clips_raw = f.read()
    clips_all = json.loads(clips_raw)
    # Content fingerprint of the library this process mapped (cache dir, frame count, clips.json incl. ok flags):
    # data-parallel ranks compare it in their state check (one library everywhere, an extended cache included).
    self.fingerprint = hashlib.blake2b(os.path.basename(cdir).encode() + f"|{meta['frames']}|".encode() + clips_raw, digest_size=12).hexdigest()
    # End frame of each input file (file order = clips.json order): bins of a prefix of the list (bins_of_first_files).
    self.file_frame_end = [int(c["end"]) for c in clips_all]
    self.cache_dir = cdir
    t0 = time.time()
    self.fields: dict[str, torch.Tensor] = {}
    for k in FIELD_DTYPES:
      arr = np.load(os.path.join(cdir, f"{k}.npy"), mmap_mode="r")
      self.fields[k] = torch.from_numpy(np.ascontiguousarray(arr))  # CPU RAM copy
    self.time_step_total = int(meta["frames"])
    log(f"[MotionLibrary] loaded cache {cdir} into RAM in {time.time() - t0:.0f}s")
    self._build_tables(clips_all, bin_seconds, expected_fps, device)
    self.nj = len(self.joint_names)
    self.nb = len(self.body_names)

  def _build_tables(self, clips_all: list[dict], bin_seconds: float, expected_fps: float, device: str) -> None:
    """Clip list, per-frame clip-end / bin tables and the 1 s bins from clips.json entries (also used by the
    human-reference library, supermimic/human/motions.py)."""
    self.clips = [ClipInfo(c["name"], c["start"], c["end"]) for c in clips_all if c.get("ok", True) and c["end"] > c["start"] + 1]
    self.num_clips = len(self.clips)
    cs = torch.tensor([c.start for c in self.clips], dtype=torch.long)
    ce = torch.tensor([c.end for c in self.clips], dtype=torch.long)
    self.clip_start = cs.to(device)
    self.clip_end = ce.to(device)
    # Per-frame tables (int32 on GPU). Frames of excluded clips map to their own clip end (never sampled).
    clip_end_of_frame = torch.zeros(self.time_step_total, dtype=torch.int32)
    for c in clips_all:
      clip_end_of_frame[c["start"] : c["end"]] = c["end"]
    self.clip_end_of_frame = clip_end_of_frame.to(device)
    bin_len = int(round(bin_seconds * expected_fps))
    starts, ends, bclip = [], [], []
    bin_of_frame = torch.zeros(self.time_step_total, dtype=torch.int32)
    for i, c in enumerate(self.clips):
      for s in range(c.start, c.end, bin_len):
        e = min(s + bin_len, c.end)
        bin_of_frame[s:e] = len(starts)
        starts.append(s)
        ends.append(e)
        bclip.append(i)
    self.bin_start = torch.tensor(starts, dtype=torch.long, device=device)
    self.bin_end = torch.tensor(ends, dtype=torch.long, device=device)
    self.bin_clip = torch.tensor(bclip, dtype=torch.long, device=device)
    self.num_bins = len(starts)
    self.bin_of_frame = bin_of_frame.to(device)

  # ---- stats -----------------------------------------------------------------

  @property
  def total_seconds(self) -> float:
    return sum(c.length for c in self.clips) * self.dt

  def summary(self) -> str:
    gb = sum(t.numel() * t.element_size() for t in self.fields.values()) / 1e9
    return (
      f"MotionLibrary: {self.num_clips} clips, {self.time_step_total} frames ({self.total_seconds / 3600:.2f} h), "
      f"{self.num_bins} bins, {self.nb} tracked bodies, {self.nj} joints, {gb:.1f} GB in RAM"
    )

  def bins_of_first_files(self, n: int) -> int:
    """Number of bins of the clips of the first n files of the list (they come first: bins are laid out in file
    order). A library extended by appending files keeps the old files' bins at the same indices."""
    if n <= 0:
      return 0
    if n > len(self.file_frame_end):
      return -1
    return int((self.bin_start < self.file_frame_end[n - 1]).sum())

  def clip_of_frame(self, frames: torch.Tensor) -> torch.Tensor:
    """Clip index for global frame indices (GPU), via searchsorted on clip ends."""
    return torch.searchsorted(self.clip_end, frames, right=True)

  # ---- gathers ---------------------------------------------------------------

  def _take(self, k: str, idx_cpu: torch.Tensor) -> torch.Tensor:
    t = torch.index_select(self.fields[k], 0, idx_cpu)
    return t.to(self.device, non_blocking=False).float()

  def gather(self, frames: torch.Tensor, fields=None) -> dict[str, torch.Tensor]:
    """Rows for global frame indices ``frames`` (GPU long [N]) -> dict of GPU fp32 tensors [N, ...]."""
    idx = frames.detach().to("cpu", dtype=torch.long)
    keys = fields or list(self.fields)
    return {k: self._take(k, idx) for k in keys}

  def gather_windows(self, start: torch.Tensor, L: int, fields=None) -> dict[str, torch.Tensor]:
    """Windows of L consecutive frames from ``start`` (GPU long [N]) -> dict of GPU fp32 [N, L, ...].
    Frames past the library end are clamped; callers clamp to clip ends themselves."""
    N = start.shape[0]
    idx = (start.detach().to("cpu", dtype=torch.long)[:, None] + torch.arange(L)[None, :]).clamp_(max=self.time_step_total - 1).reshape(-1)
    keys = fields or list(self.fields)
    return {k: self._take(k, idx).view(N, L, *self.fields[k].shape[1:]) for k in keys}

  def root_state(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
    """Reference root (body 0) pose/velocity and joints for RSI resets."""
    g = self.gather(frames)
    return dict(
      root_pos=g["body_pos"][:, 0], root_quat=g["body_quat"][:, 0], root_lin_vel=g["body_lin_vel"][:, 0],
      root_ang_vel=g["body_ang_vel"][:, 0], joint_pos=g["joint_pos"], joint_vel=g["joint_vel"],
    )


# ---------------------------------------------------------------------------
# File selection helpers
# ---------------------------------------------------------------------------


def collect_motion_files(dirs: list[str], blacklist_json: str | None = None, name_exclude: list[str] | None = None, max_motions: int | None = None, log=print) -> list[str]:
  import glob

  files: list[str] = []
  for d in dirs:
    d = os.path.expanduser(d)
    if os.path.isfile(d):
      files.append(d)
    else:
      files.extend(sorted(glob.glob(os.path.join(d, "**", "*.npz"), recursive=True)))
  n0 = len(files)
  if blacklist_json:
    bl = json.load(open(os.path.expanduser(blacklist_json)))
    names = bl["bad_motions"] if isinstance(bl, dict) and "bad_motions" in bl else bl
    bad = set(os.path.basename(str(x)) for x in names)
    files = [f for f in files if os.path.basename(f) not in bad]
    log(f"[motions] blacklist {blacklist_json}: {n0 - len(files)} of {n0} files removed")
  if name_exclude:
    kws = [k.lower() for k in name_exclude]
    n1 = len(files)
    files = [f for f in files if not any(k in os.path.basename(f).lower() for k in kws)]
    log(f"[motions] name filter {kws}: {n1 - len(files)} files removed")
  if max_motions:
    files = files[:max_motions]
  return files


def read_name_list(path: str) -> list[str]:
  """Clip names from a text file, one per line (UTF-8; names may contain spaces).
  Full paths are reduced to their basename; blank lines are skipped."""
  names = []
  with open(os.path.expanduser(path), encoding="utf-8") as f:
    for line in f:
      s = line.rstrip("\r\n")
      if s.strip():
        names.append(os.path.basename(s))
  return names


def resolve_clip_names(clip_names: list[str], wanted: list[str], log=print, label: str = "clip names") -> tuple[list[int], list[str]]:
  """Map basenames to clip indices -> (sorted unique indices, names not found).
  Exact match first, then the whitespace-stripped name. Duplicates are counted once."""
  index: dict[str, int] = {}
  for i, n in enumerate(clip_names):
    index.setdefault(n, i)
  ids: set[int] = set()
  missing: list[str] = []
  seen: set[str] = set()
  for n in wanted:
    if n in seen:
      continue
    seen.add(n)
    i = index.get(n)
    if i is None:
      i = index.get(n.strip())
    if i is None:
      missing.append(n)
    else:
      ids.add(i)
  if len(seen) != len(wanted):
    log(f"[{label}] {len(wanted) - len(seen)} duplicate names ignored")
  if missing:
    log(f"[{label}] WARNING: {len(missing)} of {len(seen)} names not found in the library (first: {missing[:5]})")
  log(f"[{label}] {len(ids)} clips selected by name ({len(missing)} missing)")
  return sorted(ids), missing
