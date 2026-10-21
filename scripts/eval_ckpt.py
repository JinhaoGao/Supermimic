"""Evaluate a saved checkpoint on the benchmark protocol (Appendix F.1).

  uv run python scripts/eval_ckpt.py --ckpt logs/.../model_20000.pt --motion_dir ~/supermimic/data/lafan1
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import torch


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--ckpt", type=str, required=True)
  p.add_argument("--motion_dir", type=str, default=None, help="defaults to the motion files stored in the checkpoint")
  p.add_argument("--device", type=str, default="cuda:0")
  p.add_argument("--no_compile", action="store_true")
  p.add_argument("--out", type=str, default=None, help="write per-clip json here")
  args = p.parse_args()

  from supermimic.evaluate import Evaluator
  from supermimic.heads import HeadsCfg
  from supermimic.policy import Policy

  ck = torch.load(args.ckpt, map_location=args.device)
  if args.motion_dir:
    files = sorted(glob.glob(os.path.join(os.path.expanduser(args.motion_dir), "*.npz")))
  else:
    files = ck["cfg"]["motion_files"]
  d = ck["dims"]
  heads_cfg = HeadsCfg(**{k: v for k, v in ck["cfg"]["heads"].items() if k != "extra"})
  policy = Policy(d["proprio"], d["hist"], d["goal_frame"], d["K"], d["nj"]).to(args.device)
  policy.load_state_dict(ck["policy"])
  ev = Evaluator(files, heads_cfg, args.device, compile_=not args.no_compile)
  res = ev.run(policy)
  print(json.dumps(res["scalars"], indent=2))
  print("failed clips:", res["failed_clips"])
  worst = sorted(res["per_clip"].items(), key=lambda kv: kv[1])[:10]
  print("lowest 10 clips:")
  for n, v in worst:
    print(f"  {v:.3f}  {n}")
  if args.out:
    with open(args.out, "w") as f:
      json.dump({"ckpt": args.ckpt, "it": ck["it"], **res}, f, indent=2)


if __name__ == "__main__":
  main()
