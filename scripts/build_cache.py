"""Build the packed motion cache for a file list ahead of training (same key as
the trainer will compute, so the run starts instantly).

  python scripts/build_cache.py --robot g1 --motion_dirs <dir ...> --blacklist <json> --cache_dir ~/supermimic/data/cache --num_workers 64

Needs mjlab only to read the entity's joint order / tracked body list.
"""

from __future__ import annotations

import argparse
import os
import time


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--motion_dirs", type=str, nargs="+", required=True)
  p.add_argument("--blacklist", type=str, default=None)
  p.add_argument("--name_exclude", type=str, nargs="*", default=None)
  p.add_argument("--max_motions", type=int, default=None)
  p.add_argument("--cache_dir", type=str, required=True)
  p.add_argument("--robot", type=str, default="g1", choices=["g1", "e2"])
  p.add_argument("--num_workers", type=int, default=64)
  p.add_argument("--list_out", type=str, default=None, help="also write the resolved file list here")
  args = p.parse_args()

  from supermimic.env_cfg import base_tracking_cfg
  from supermimic.motions import MotionLibrary, collect_motion_files

  files = collect_motion_files(args.motion_dirs, args.blacklist, args.name_exclude, args.max_motions)
  print(f"[build_cache] {len(files)} files")
  if args.list_out:
    with open(args.list_out, "w") as f:
      f.write("\n".join(files))
  cfg = base_tracking_cfg(args.robot)
  body_names = tuple(cfg.commands["motion"].body_names)
  # Entity joint order without compiling a full env.
  from mjlab.entity import Entity

  entity = Entity(cfg.scene.entities["robot"])
  joint_names = tuple(entity.joint_names)
  print(f"[build_cache] joints {len(joint_names)}, tracked bodies {body_names}")
  t0 = time.time()
  lib = MotionLibrary(files, joint_names, body_names, device="cpu", cache_dir=os.path.expanduser(args.cache_dir), num_workers=args.num_workers)
  print(f"[build_cache] {lib.summary()}  ({time.time() - t0:.0f}s)  cache: {lib.cache_dir}")


if __name__ == "__main__":
  main()
