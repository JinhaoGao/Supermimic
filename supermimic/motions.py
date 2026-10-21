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
      shape, _, _ = np.lib.format._read_array_header(f, ver)  # type: ignore[attr-defined]
  return int(shape[0])


_W: dict = {}


def _worker_init(cache_dir, joint_names, body_names, fields_shape):
  _W["cache_dir"] = cache_dir
  _W["joint_names"] = list(joint_names)
  _W["body_names"] = list(body_names)
  _W["mm"] = {k: np.lib.format.open_memmap(os.path.join(cache_dir, f"{k}.npy"), mode="r+") for k in FIELD_DTYPES}


def _worker_fill(job):
  path, start, T, expected_fps = job
  d = np.load(path)
  fps = float(np.asarray(d["fps"]).reshape(-1)[0])
  if not math.isclose(fps, expected_fps, rel_tol=1e-6):
    return (path, f"fps {fps}")
  src_j = [str(x) for x in d["joint_names"]]
  src_b = [str(x) for x in d["body_names"]]
  try:
    jperm = [src_j.index(n) for n in _W["joint_names"]]
    bperm = [src_b.index(n) for n in _W["body_names"]]
  except ValueError as e:
    return (path, f"name mismatch: {e}")
  if int(d["joint_pos"].shape[0]) != T:
    return (path, "frame count changed")
  mm = _W["mm"]
  mm["joint_pos"][start : start + T] = d["joint_pos"][:, jperm]
  mm["joint_vel"][start : start + T] = d["joint_vel"][:, jperm]
  mm["body_pos"][start : start + T] = d["body_pos_w"][:, bperm]
  mm["body_quat"][start : start + T] = d["body_quat_w"][:, bperm]
  mm["body_lin_vel"][start : start + T] = d["body_lin_vel_w"][:, bperm]
  mm["body_ang_vel"][start : start + T] = d["body_ang_vel_w"][:, bperm]
  return None


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
  t1 = time.time()
  with Pool(num_workers, initializer=_worker_init, initargs=(cache_dir, joint_names, body_names, shapes)) as pool:
    for i, res in enumerate(pool.imap_unordered(_worker_fill, jobs, chunksize=16)):
      if res is not None:
        bad.append(res)
      if (i + 1) % 20000 == 0:
        log(f"[cache] {i + 1}/{len(files)} files written ({time.time() - t1:.0f}s)")
  if bad:
    log(f"[cache] WARNING: {len(bad)} files failed (first: {bad[:3]}); their frames stay zero -> excluded below")
  bad_set = {b[0] for b in bad}
  clips = [{"name": os.path.basename(f), "start": s, "end": s + n, "ok": f not in bad_set} for f, s, n in zip(files, starts, lengths)]
  meta = {"joint_names": list(joint_names), "body_names": list(body_names), "fps": expected_fps, "frames": T, "files": len(files), "build_seconds": time.time() - t0}
  with open(os.path.join(cache_dir, "clips.json"), "w") as f:
    json.dump(clips, f)
  with open(os.path.join(cache_dir, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
  log(f"[cache] done in {time.time() - t0:.0f}s")


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
    key = hashlib.sha1(("\n".join(files) + "|" + ",".join(self.joint_names) + "|" + ",".join(self.body_names) + f"|{expected_fps}").encode()).hexdigest()[:10]
    cache_root = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "supermimic_motions")
    cdir = os.path.join(cache_root, f"lib_{len(files)}_{key}")
    if not os.path.exists(os.path.join(cdir, "meta.json")):
      build_cache(files, self.joint_names, self.body_names, cdir, expected_fps, num_workers, log)
    meta = json.load(open(os.path.join(cdir, "meta.json")))
    clips_all = json.load(open(os.path.join(cdir, "clips.json")))
    self.cache_dir = cdir
    t0 = time.time()
    self.fields: dict[str, torch.Tensor] = {}
    for k in FIELD_DTYPES:
      arr = np.load(os.path.join(cdir, f"{k}.npy"), mmap_mode="r")
      self.fields[k] = torch.from_numpy(np.ascontiguousarray(arr))  # CPU RAM copy
    self.time_step_total = int(meta["frames"])
    log(f"[MotionLibrary] loaded cache {cdir} into RAM in {time.time() - t0:.0f}s")
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
    self.nj = len(self.joint_names)
    self.nb = len(self.body_names)

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
    keys = fields or list(FIELD_DTYPES)
    return {k: self._take(k, idx) for k in keys}

  def gather_windows(self, start: torch.Tensor, L: int, fields=None) -> dict[str, torch.Tensor]:
    """Windows of L consecutive frames from ``start`` (GPU long [N]) -> dict of GPU fp32 [N, L, ...].
    Frames past the library end are clamped; callers clamp to clip ends themselves."""
    N = start.shape[0]
    idx = (start.detach().to("cpu", dtype=torch.long)[:, None] + torch.arange(L)[None, :]).clamp_(max=self.time_step_total - 1).reshape(-1)
    keys = fields or list(FIELD_DTYPES)
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
