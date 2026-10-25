"""Train SuperMimic on a set of G1 / E2 motion npz files.

Examples:
  # LAFAN1 (40 clips)
  python scripts/train.py --motion_dirs ~/supermimic/data/lafan1 --run_name lafan1
  # G1 atlas minus the GMT blacklist, eval on a fixed random subset of 1024 clips
  python scripts/train.py --motion_dirs /DATA/disk0/gmt_data/ma50/data/in/motion50/g1/g1 \
      --blacklist ~/g1_resume/atlas_full_blacklist_motion50_1010.json \
      --cache_dir ~/supermimic/data/cache --eval_clips 1024 --run_name g1_atlas
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--motion_dirs", type=str, nargs="*", default=None, help="directories (recursive) or npz files")
  p.add_argument("--motion_list", type=str, default=None,
                 help="text file, one npz path per line (e.g. a run's motion_files.txt); replaces --motion_dirs / --blacklist")
  p.add_argument("--blacklist", type=str, default=None, help="json with a 'bad_motions' list of npz basenames")
  p.add_argument("--name_exclude", type=str, nargs="*", default=None, help="drop clips whose name contains any keyword")
  p.add_argument("--max_motions", type=int, default=None)
  p.add_argument("--cache_dir", type=str, default="", help="where the packed motion cache lives (default ~/.cache/supermimic_motions)")
  p.add_argument("--num_workers", type=int, default=32)
  p.add_argument("--robot", type=str, default="g1", choices=["g1", "e2"])
  p.add_argument("--robot_variant", type=str, default="", help="g1: eval (team eval / deploy model); e2: nodamp (no passive joint damping)")
  p.add_argument("--source_weights", type=str, default="",
                 help="reset-sampler prior per data source, e.g. noetix_kungfu=6,sportsmeeting=10 (clip '<robot>__<source>_...')")
  p.add_argument("--name_weights", type=str, default="",
                 help="reset-sampler weight per clip-name keyword (case-insensitive substring), e.g. crawl=4,getup=4")
  p.add_argument("--hard_names", type=str, default="",
                 help="clip-name keywords (comma-separated, case-insensitive) of the hard set for --hard_term_schedule")
  p.add_argument("--hard_term_schedule", type=str, default="",
                 help="termination multiplier of the hard set, piecewise linear over absolute iterations, e.g. 104000:2,108000:2,114000:1")
  p.add_argument("--sampler_clamp_returns", action="store_true", help="clamp reset-sampler bin returns to [0, g_max]")
  p.add_argument("--ee_term_z", type=float, default=0.0,
                 help="end-effector height termination in m (env + soft head); 0 = off as in the paper, 0.25 = BeyondMimic / team benchmark")
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
  p.add_argument("--goal_anchor_pos", action="store_true", help="include ref anchor position in the goal (needs odometry on hardware)")
  p.add_argument("--wm_batches", type=int, default=10)
  p.add_argument("--wm_windows", type=int, default=2048)
  p.add_argument("--wm_k", type=int, default=8)
  p.add_argument("--lr", type=float, default=2e-4)
  p.add_argument("--lr_decay_iters", type=int, default=250_000)
  p.add_argument("--log_interval", type=int, default=50)
  p.add_argument("--eval_interval", type=int, default=2000)
  p.add_argument("--eval_clips", type=int, default=0, help="0 = all clips, else a fixed random subset")
  p.add_argument("--eval_envs", type=int, default=1024)
  p.add_argument("--ckpt_interval", type=int, default=2000)
  p.add_argument("--no_eval", action="store_true")
  p.add_argument("--resume", type=str, default=None)
  p.add_argument("--no_watchdog", action="store_true", help="disable the collapse watchdog (auto-rollback + LR halving)")
  p.add_argument("--wd_drop", type=float, default=0.2)
  p.add_argument("--wd_force_at", type=int, default=-1, help="debug: force one rollback at this iteration")
  p.add_argument("--wd_warmup", type=int, default=1500, help="watchdog quiet iterations after start / resume / rollback")
  # 10-25 fixes from the open-source comparison (all off by default)
  p.add_argument("--input_clip", type=float, default=10.0, help="clamp of normalized policy / critic inputs (v2: 10)")
  p.add_argument("--clip_values", action="store_true", help="clamp bootstrap / lambda-return values to +-r_max/(1-gamma)")
  p.add_argument("--critic_weight_survival", action="store_true", help="critic loss weighted by imagined survival")
  p.add_argument("--imag_truncate", action="store_true", help="imagined rollouts end at the clip end with a critic bootstrap")
  p.add_argument("--wm_prefix_mask", action="store_true", help="WM windows crossing an episode end keep the steps up to it")
  p.add_argument("--clean_prev_action", action="store_true", help="proprio a_{t-1} = policy action (not the noisy executed one)")
  p.add_argument("--fixes", action="store_true", help="all six 10-25 fixes above, input_clip 50")
  p.add_argument("--no_compile", action="store_true")
  p.add_argument("--compile_mode", type=str, default=None, help="e.g. reduce-overhead or max-autotune")
  args = p.parse_args()

  from supermimic.heads import HeadsCfg
  from supermimic.motions import collect_motion_files
  from supermimic.trainer import TrainCfg, Trainer
  from supermimic.world_model import WorldModelCfg

  if args.motion_list:
    files = [l.strip() for l in open(os.path.expanduser(args.motion_list)) if l.strip()]
  else:
    assert args.motion_dirs, "--motion_dirs or --motion_list is required"
    files = collect_motion_files(args.motion_dirs, args.blacklist, args.name_exclude, args.max_motions)
  assert files, "no motion files found"
  print(f"[train] {len(files)} motion files")

  log_dir = os.path.join(args.log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + args.run_name)
  cfg = TrainCfg(
    motion_files=files,
    robot=args.robot,
    robot_variant=args.robot_variant,
    source_weights=tuple((kv.split("=")[0], float(kv.split("=")[1])) for kv in args.source_weights.split(",") if kv),
    name_weights=tuple((kv.split("=")[0], float(kv.split("=")[1])) for kv in args.name_weights.split(",") if kv),
    hard_names=tuple(k for k in args.hard_names.split(",") if k),
    sampler_clamp_returns=args.sampler_clamp_returns,
    hard_term_schedule=tuple((int(kv.split(":")[0]), float(kv.split(":")[1])) for kv in args.hard_term_schedule.split(",") if kv),
    num_envs=args.num_envs,
    device=args.device,
    seed=args.seed,
    log_dir=log_dir,
    cache_dir=os.path.expanduser(args.cache_dir) if args.cache_dir else "",
    num_workers=args.num_workers,
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
    eval_clips=args.eval_clips,
    eval_envs=args.eval_envs,
    ckpt_interval=args.ckpt_interval,
    eval_enabled=not args.no_eval,
    compile=not args.no_compile,
    wd_enabled=not args.no_watchdog,
    wd_drop=args.wd_drop,
    wd_force_at=args.wd_force_at,
    wd_warmup=args.wd_warmup,
    input_clip=50.0 if args.fixes and args.input_clip == 10.0 else args.input_clip,
    clip_values=args.clip_values or args.fixes,
    critic_weight_survival=args.critic_weight_survival or args.fixes,
    imag_truncate=args.imag_truncate or args.fixes,
    wm_prefix_mask=args.wm_prefix_mask or args.fixes,
    clean_prev_action=args.clean_prev_action or args.fixes,
    compile_mode=args.compile_mode,
    wm=WorldModelCfg(k=args.wm_k, lr=args.lr, batches_per_update=args.wm_batches, windows_per_batch=args.wm_windows),
    heads=HeadsCfg(goal_anchor_pos=args.goal_anchor_pos, ee_term_z=args.ee_term_z),
  )
  print(f"[train] log_dir = {log_dir}")
  os.makedirs(log_dir, exist_ok=True)
  with open(os.path.join(args.log_root, "last_run.txt"), "a") as f:
    f.write(log_dir + "\n")
  with open(os.path.join(log_dir, "motion_files.txt"), "w") as f:
    f.write("\n".join(files))
  trainer = Trainer(cfg)
  if args.resume:
    trainer.load(args.resume)
  trainer.run()


if __name__ == "__main__":
  sys.exit(main())
