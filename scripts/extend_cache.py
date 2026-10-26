"""Extend a packed motion cache with extra files without re-reading the old ones (master c1c94be).

The library cache is keyed by the whole file list, so adding a few hundred clips to a 250k-clip list would rebuild
everything (~1 h of npz reads; bj31 254,518 clips: 27 min). This copies the old cache arrays into the cache of
old_list + extra (in that order), reads only the extra files, and writes the combined list for --motion_list.

  python scripts/extend_cache.py --old_list logs/<run>/motion_files.txt --extra extra.txt \
      --cache_dir ~/supermimic/data/cache --out_list ~/supermimic/lists/g1_plus_phc.txt

Run it ONCE, before the (data-parallel) training; every rank of the run then maps the same new cache directory (the
same key: supermimic.motions.library_cache_dir, which MotionLibrary uses too) and the ranks' state check compares the
library fingerprint. A checkpoint trained on old_list resumes into old_list + extra with its reset-sampler statistics
(Trainer._load_sampler_state: the old clips keep their bins and indices).

Differences from master's script (same output for the same inputs): the per-file writer is this branch's
motions._worker_fill, whose result is (path, error or None, used fallback) -- master tested ``r is not None`` and
would mark every extra file as failed with it -- and it reads npz files without a body_names key (bj31 renders) via
the fixed fallback body order; --out_list may already exist only with exactly this content; the new cache directory
is checked to be the one MotionLibrary looks up for --out_list.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from supermimic.motions import extend_cache, library_cache_dir  # noqa: E402


def read_list(path: str) -> list[str]:
  return [l.strip() for l in open(os.path.expanduser(path)) if l.strip()]


def find_cache(root: str, files: list[str]) -> tuple[str, dict]:
  """The complete cache of exactly this file list under root (any target names / fps) -> (dir, meta)."""
  for d in sorted(glob.glob(os.path.join(root, f"lib_{len(files)}_*"))):
    mp = os.path.join(d, "meta.json")
    if not os.path.exists(mp):
      continue
    m = json.load(open(mp))
    if library_cache_dir(root, files, m["joint_names"], m["body_names"], m["fps"]) == d:
      return d, m
  raise SystemExit(f"[extend] no complete cache of the old list ({len(files)} files) under {root}")


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--old_list", required=True, help="the list the existing cache was built from (exact paths and order)")
  p.add_argument("--extra", required=True, help="text file, one npz path per line (already in the old list: skipped)")
  p.add_argument("--cache_dir", required=True)
  p.add_argument("--out_list", required=True, help="old list + extra, for --motion_list (may exist only with this exact content)")
  p.add_argument("--num_workers", type=int, default=32)
  p.add_argument("--chunk_frames", type=int, default=2_000_000)
  p.add_argument("--dry_run", action="store_true", help="only resolve the old / new cache directories and the list")
  args = p.parse_args()

  root = os.path.expanduser(args.cache_dir)
  old_files = read_list(args.old_list)
  have = set(old_files)
  extra = list(dict.fromkeys(f for f in read_list(args.extra) if f not in have))
  missing = [f for f in extra if not os.path.exists(f)]
  if missing:
    raise SystemExit(f"[extend] {len(missing)} of {len(extra)} extra files do not exist (first: {missing[0]})")
  old_dir, meta = find_cache(root, old_files)
  files = old_files + extra
  new_dir = library_cache_dir(root, files, meta["joint_names"], meta["body_names"], meta["fps"])
  out = os.path.expanduser(args.out_list)
  text = "\n".join(files)
  if os.path.exists(out):
    if read_list(out) != files:
      raise SystemExit(f"[extend] {out} exists with another content; pick another --out_list")
    print(f"[extend] {out} exists with exactly this list")
  print(f"[extend] {old_dir} ({len(old_files)} files) + {len(extra)} -> {new_dir}")
  if args.dry_run:
    return
  if os.path.exists(os.path.join(new_dir, "meta.json")):
    print(f"[extend] {new_dir} is already complete; nothing to do")
  else:
    extend_cache(old_dir, old_files, extra, new_dir, args.num_workers, args.chunk_frames)
  if not os.path.exists(out):
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
      f.write(text)
  # The directory MotionLibrary will look up for --motion_list out (same function, same key).
  assert library_cache_dir(root, read_list(out), meta["joint_names"], meta["body_names"], meta["fps"]) == new_dir
  print(f"[extend] list {out} ({len(files)} files) -> cache {new_dir}")


if __name__ == "__main__":
  main()
