"""Evaluate a saved checkpoint on the benchmark protocol (Appendix F.1).

  uv run python scripts/eval_ckpt.py --ckpt logs/.../model_20000.pt                 # all clips of the run
  uv run python scripts/eval_ckpt.py --ckpt ... --eval_clips 1024 --eval_envs 1024   # same fixed subset as the trainer
  uv run python scripts/eval_ckpt.py --ckpt ... --eval_clip_names eval_1024.txt      # subset fixed by npz basename
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random

import torch


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--ckpt", type=str, required=True)
  p.add_argument("--motion_list", type=str, default=None, help="text file with npz paths (default: the run's motion_files)")
  p.add_argument("--device", type=str, default="cuda:0")
  p.add_argument("--no_compile", action="store_true")
  p.add_argument("--eval_clips", type=int, default=None, help="random subset size (seeded like the trainer); default: the run's setting")
  p.add_argument("--eval_clip_names", type=str, default=None, help="txt of npz basenames; default: the run's setting (if any)")
  p.add_argument("--eval_envs", type=int, default=1024)
  p.add_argument("--out", type=str, default=None, help="write per-clip json here")
  p.add_argument("--ee_termination", choices=["ckpt", "on", "off"], default="ckpt",
                 help="count end-effector height failures (eval-site protocol); ckpt = as trained (off for checkpoints older than the switch)")
  p.add_argument("--ee_threshold", type=float, default=None, help="end-effector z threshold (default: the checkpoint's, else 0.25)")
  p.add_argument("--ee_term_z", type=float, default=None,
                 help="master's v2 spelling: override the end-effector height termination (m, 0 = off); default: the run's")
  p.add_argument("--robot_variant", type=str, default=None, help="override the robot model variant; default: the run's")
  args = p.parse_args()

  from supermimic.evaluate import Evaluator
  from supermimic.heads import HeadsCfg, ee_flags
  from supermimic.motions import read_name_list, resolve_clip_names
  from supermimic.policy import Policy

  ck = torch.load(args.ckpt, map_location=args.device)
  cfg = ck["cfg"]
  files = [l.strip() for l in open(args.motion_list) if l.strip()] if args.motion_list else cfg["motion_files"]
  d = ck["dims"]
  hk = {k: v for k, v in cfg["heads"].items() if k != "extra"}
  hk["goal_anchor_pos"] = d["goal_frame"] == 2 * d["nj"] + 9
  # Any branch's checkpoint (ddp-bj31 ee_termination / master v2 ee_term_z / v1 without: off); then the overrides,
  # e.g. score a v1 checkpoint under the eval site's end-effector termination (--ee_termination on / --ee_term_z 0.25).
  heads_cfg = HeadsCfg.from_dict(hk)
  try:
    ee_on, ee_thr = ee_flags(
      args.ee_term_z, None if args.ee_termination == "ckpt" else args.ee_termination == "on", args.ee_threshold,
      default=(heads_cfg.ee_termination, heads_cfg.ee_term_z),
    )
  except ValueError as e:
    p.error(str(e))
  heads_cfg = dataclasses.replace(heads_cfg, ee_termination=ee_on, ee_term_z=ee_thr)
  print(f"[eval_ckpt] end-effector termination: " + (f"on, z error > {ee_thr:g}" if ee_on else "off"))
  policy = Policy(d["proprio"], d["hist"], d["goal_frame"], d["K"], d["nj"]).to(args.device)
  policy.load_state_dict(ck["policy"])
  names_txt = args.eval_clip_names
  if names_txt is None and args.eval_clips is None:
    names_txt = cfg.get("eval_clip_names") or None
  n_sub = args.eval_clips if args.eval_clips is not None else int(cfg.get("eval_clips", 0))
  eval_clips: tuple[int, ...] = ()
  if names_txt:
    # Indices into the file list; checked against the library's clip list below (clips that failed
    # the cache build or have < 2 frames are dropped from the library and would shift the indices).
    eval_clips = tuple(resolve_clip_names([os.path.basename(f) for f in files], read_name_list(names_txt), label="eval_clip_names")[0])
    assert eval_clips, f"no name of {names_txt} is in the file list"
  elif n_sub and n_sub < len(files):
    eval_clips = tuple(sorted(random.Random(int(cfg.get("seed", 0))).sample(range(len(files)), n_sub)))
  ev = Evaluator(
    files, heads_cfg, args.device, eval_clips=eval_clips, num_clips_total=len(files), max_envs=args.eval_envs,
    compile_=not args.no_compile, cache_dir=cfg.get("cache_dir", ""), robot=cfg.get("robot", "g1"),
    robot_variant=args.robot_variant if args.robot_variant is not None else cfg.get("robot_variant", ""),
  )
  if names_txt:
    got = [ev.lib.clips[i].name for i in ev.eval_clips]
    want = [os.path.basename(files[i]) for i in eval_clips]
    assert got == want, "library clip list differs from the file list (failed/short clips); eval subset would be wrong"
  res = ev.run(policy)
  print(json.dumps({"it": ck["it"], **res["scalars"]}, indent=2))
  print("failed clips (first 30):", res["failed_clips"][:30])
  worst = sorted(res["per_clip"].items(), key=lambda kv: kv[1])[:10]
  print("lowest 10 clips:")
  for n, v in worst:
    print(f"  {v:.3f}  {n}")
  if args.out:
    with open(args.out, "w") as f:
      json.dump({"ckpt": args.ckpt, "it": ck["it"], **res}, f, indent=2)


if __name__ == "__main__":
  main()
