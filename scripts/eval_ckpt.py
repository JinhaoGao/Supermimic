"""Evaluate a saved checkpoint on the benchmark protocol (Appendix F.1).

  uv run python scripts/eval_ckpt.py --ckpt logs/.../model_20000.pt                 # all clips of the run
  uv run python scripts/eval_ckpt.py --ckpt ... --eval_clips 1024 --eval_envs 1024   # same fixed subset as the trainer
"""

from __future__ import annotations

import argparse
import json
import random

import torch


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--ckpt", type=str, required=True)
  p.add_argument("--motion_list", type=str, default=None, help="text file with npz paths (default: the run's motion_files)")
  p.add_argument("--device", type=str, default="cuda:0")
  p.add_argument("--no_compile", action="store_true")
  p.add_argument("--eval_clips", type=int, default=None, help="random subset size (seeded like the trainer); default: the run's setting")
  p.add_argument("--eval_envs", type=int, default=1024)
  p.add_argument("--out", type=str, default=None, help="write per-clip json here")
  p.add_argument("--ee_term_z", type=float, default=None, help="override the end-effector height termination (m); default: the run's")
  p.add_argument("--robot_variant", type=str, default=None, help="override the robot model variant; default: the run's")
  args = p.parse_args()

  from supermimic.evaluate import Evaluator
  from supermimic.heads import HeadsCfg
  from supermimic.policy import Policy

  ck = torch.load(args.ckpt, map_location=args.device)
  cfg = ck["cfg"]
  files = [l.strip() for l in open(args.motion_list) if l.strip()] if args.motion_list else cfg["motion_files"]
  d = ck["dims"]
  hk = {k: v for k, v in cfg["heads"].items() if k != "extra"}
  hk["goal_anchor_pos"] = d["goal_frame"] == 2 * d["nj"] + 9
  if args.ee_term_z is not None:  # e.g. score a v1 checkpoint under the v2 (end-effector) termination
    hk["ee_term_z"] = args.ee_term_z
  heads_cfg = HeadsCfg(**hk)
  policy = Policy(d["proprio"], d["hist"], d["goal_frame"], d["K"], d["nj"]).to(args.device)
  policy.load_state_dict(ck["policy"])
  n_sub = args.eval_clips if args.eval_clips is not None else int(cfg.get("eval_clips", 0))
  eval_clips: tuple[int, ...] = ()
  if n_sub and n_sub < len(files):
    eval_clips = tuple(sorted(random.Random(int(cfg.get("seed", 0))).sample(range(len(files)), n_sub)))
  ev = Evaluator(
    files, heads_cfg, args.device, eval_clips=eval_clips, num_clips_total=len(files), max_envs=args.eval_envs,
    compile_=not args.no_compile, cache_dir=cfg.get("cache_dir", ""), robot=cfg.get("robot", "g1"),
    robot_variant=args.robot_variant if args.robot_variant is not None else cfg.get("robot_variant", ""),
  )
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
