"""Train SuperMimic on a directory / list of G1 motion npz files.

Example (LAFAN1, one GPU):
  python scripts/train.py --motion_dir ~/supermimic/data/lafan1 --run_name lafan1_h32
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from datetime import datetime

import torch


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--motion_dir", type=str, default=None)
  p.add_argument("--motion_glob", type=str, default="*.npz")
  p.add_argument("--motion_list", type=str, default=None, help="text file with one npz path per line")
  p.add_argument("--max_motions", type=int, default=None)
  p.add_argument("--run_name", type=str, default="run")
  p.add_argument("--log_root", type=str, default="logs/supermimic")
  p.add_argument("--device", type=str, default="cuda:0")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--num_envs", type=int, default=4096)
  p.add_argument("--iterations", type=int, default=320_000)
  p.add_argument("--env_steps_per_iter", type=float, default=1.0)
  p.add_argument("--H", type=int, default=32)
  p.add_argument("--rollouts", type=int, default=4096)
  p.add_argument("--no_critic", action="store_true")
  p.add_argument("--wm_batches", type=int, default=10)
  p.add_argument("--wm_windows", type=int, default=2048)
  p.add_argument("--wm_k", type=int, default=8)
  p.add_argument("--lr", type=float, default=2e-4)
  p.add_argument("--lr_decay_iters", type=int, default=250_000)
  p.add_argument("--log_interval", type=int, default=50)
  p.add_argument("--eval_interval", type=int, default=2000)
  p.add_argument("--ckpt_interval", type=int, default=2000)
  p.add_argument("--no_eval", action="store_true")
  p.add_argument("--resume", type=str, default=None)
  p.add_argument("--no_compile", action="store_true")
  p.add_argument("--compile_mode", type=str, default=None, help="e.g. reduce-overhead or max-autotune")
  args = p.parse_args()

  from supermimic.trainer import TrainCfg, Trainer
  from supermimic.world_model import WorldModelCfg

  if args.motion_list:
    files = [l.strip() for l in open(args.motion_list) if l.strip()]
  else:
    assert args.motion_dir, "--motion_dir or --motion_list required"
    files = sorted(glob.glob(os.path.join(os.path.expanduser(args.motion_dir), args.motion_glob)))
  if args.max_motions:
    files = files[: args.max_motions]
  assert files, "no motion files found"
  print(f"[train] {len(files)} motion files")

  log_dir = os.path.join(args.log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + args.run_name)
  cfg = TrainCfg(
    motion_files=files,
    num_envs=args.num_envs,
    device=args.device,
    seed=args.seed,
    log_dir=log_dir,
    iterations=args.iterations,
    env_steps_per_iter=args.env_steps_per_iter,
    H=args.H,
    rollouts_per_update=args.rollouts,
    use_critic=not args.no_critic,
    policy_lr=args.lr,
    critic_lr=args.lr,
    lr_decay_iters=args.lr_decay_iters,
    log_interval=args.log_interval,
    eval_interval=args.eval_interval,
    ckpt_interval=args.ckpt_interval,
    eval_enabled=not args.no_eval,
    compile=not args.no_compile,
    compile_mode=args.compile_mode,
    wm=WorldModelCfg(k=args.wm_k, lr=args.lr, batches_per_update=args.wm_batches, windows_per_batch=args.wm_windows),
  )
  print(f"[train] log_dir = {log_dir}")
  os.makedirs(log_dir, exist_ok=True)
  with open(os.path.join(args.log_root, "last_run.txt"), "a") as f:
    f.write(log_dir + "\n")
  trainer = Trainer(cfg)
  if args.resume:
    trainer.load(args.resume)
  trainer.run()


if __name__ == "__main__":
  sys.exit(main())
