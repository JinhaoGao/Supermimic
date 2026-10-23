"""Extend a packed motion cache with extra files without re-reading the old ones.

The library cache is keyed by the whole file list, so adding a few hundred clips to a 250k-clip list would rebuild
everything (~1 h of npz reads). This copies the old cache arrays into the cache of old_list + extra (in that order),
reads only the extra files, and writes the combined list for --motion_list.

  uv run python scripts/extend_cache.py --old_list logs/<run>/motion_files.txt --extra extra.txt \
      --cache_dir ~/supermimic/data/cache --out_list ~/supermimic/lists/g1_plus_phc.txt
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from supermimic.motions import FIELD_DTYPES, _npz_frames, _worker_fill, _worker_init


def lib_dir(cache_root: str, files: list[str], joint_names, body_names, fps: float) -> str:
  """Same key as MotionLibrary."""
  key = hashlib.sha1(("\n".join(files) + "|" + ",".join(joint_names) + "|" + ",".join(body_names) + f"|{fps}").encode()).hexdigest()[:10]
  return os.path.join(cache_root, f"lib_{len(files)}_{key}")


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--old_list", required=True)
  p.add_argument("--extra", required=True, help="text file, one npz path per line (already in the old list: skipped)")
  p.add_argument("--cache_dir", required=True)
  p.add_argument("--out_list", required=True)
  p.add_argument("--num_workers", type=int, default=32)
  p.add_argument("--chunk_frames", type=int, default=2_000_000)
  args = p.parse_args()

  root = os.path.expanduser(args.cache_dir)
  old_files = [l.strip() for l in open(os.path.expanduser(args.old_list)) if l.strip()]
  have = set(old_files)
  extra = list(dict.fromkeys(l.strip() for l in open(os.path.expanduser(args.extra)) if l.strip() and l.strip() not in have))
  old_dir = None
  for d in glob.glob(os.path.join(root, f"lib_{len(old_files)}_*")):
    if not os.path.exists(os.path.join(d, "meta.json")):
      continue
    m = json.load(open(os.path.join(d, "meta.json")))
    if lib_dir(root, old_files, m["joint_names"], m["body_names"], m["fps"]) == d:
      old_dir, meta = d, m
  assert old_dir, f"no cache of the old list ({len(old_files)} files) under {root}"
  J, B, fps = meta["joint_names"], meta["body_names"], meta["fps"]
  files = old_files + extra
  new_dir = lib_dir(root, files, J, B, fps)
  assert not os.path.exists(os.path.join(new_dir, "meta.json")), f"{new_dir} exists"
  print(f"[extend] {old_dir} ({len(old_files)} files) + {len(extra)} -> {new_dir}")
  t0 = time.time()
  with Pool(args.num_workers) as pool:
    lengths = pool.map(_npz_frames, extra, chunksize=16)
  T_old = int(meta["frames"])
  T = T_old + int(sum(lengths))
  os.makedirs(new_dir, exist_ok=True)
  shapes = {}
  for k in FIELD_DTYPES:
    old = np.load(os.path.join(old_dir, f"{k}.npy"), mmap_mode="r")
    assert old.shape[0] == T_old and old.dtype == FIELD_DTYPES[k], (k, old.shape, old.dtype)
    shapes[k] = (T,) + old.shape[1:]
    new = np.lib.format.open_memmap(os.path.join(new_dir, f"{k}.npy"), mode="w+", dtype=FIELD_DTYPES[k], shape=shapes[k])
    for i in range(0, T_old, args.chunk_frames):
      j = min(i + args.chunk_frames, T_old)
      new[i:j] = old[i:j]
    new.flush()
    del new, old
    print(f"[extend] copied {k} ({time.time() - t0:.0f}s)")
  starts = (T_old + np.cumsum([0] + lengths[:-1])).tolist()
  jobs = [(f, int(s), int(n), fps) for f, s, n in zip(extra, starts, lengths)]
  with Pool(args.num_workers, initializer=_worker_init, initargs=(new_dir, J, B, shapes)) as pool:
    bad = [r for r in pool.imap_unordered(_worker_fill, jobs, chunksize=8) if r is not None]
  if bad:
    print(f"[extend] WARNING: {len(bad)} files failed (first: {bad[:3]}); their frames stay zero -> excluded")
  bad_set = {b[0] for b in bad}
  clips = json.load(open(os.path.join(old_dir, "clips.json")))
  clips += [{"name": os.path.basename(f), "start": int(s), "end": int(s) + int(n), "ok": f not in bad_set} for f, s, n in zip(extra, starts, lengths)]
  with open(os.path.join(new_dir, "clips.json"), "w") as f:
    json.dump(clips, f)
  m2 = {"joint_names": J, "body_names": B, "fps": fps, "frames": T, "files": len(files), "build_seconds": time.time() - t0,
        "extended_from": old_dir}
  with open(os.path.join(new_dir, "meta.json"), "w") as f:  # last: MotionLibrary treats meta.json as "cache complete"
    json.dump(m2, f, indent=2)
  os.makedirs(os.path.dirname(os.path.abspath(os.path.expanduser(args.out_list))), exist_ok=True)
  with open(os.path.expanduser(args.out_list), "w") as f:
    f.write("\n".join(files))
  print(f"[extend] done in {time.time() - t0:.0f}s: {len(files)} files, {T} frames; list {args.out_list}")


if __name__ == "__main__":
  main()
