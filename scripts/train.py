"""Train SuperMimic on a set of G1 / E2 motion npz files.

Examples:
  # LAFAN1 (40 clips)
  python scripts/train.py --motion_dirs ~/supermimic/data/lafan1 --run_name lafan1
  # G1 atlas minus the GMT blacklist, eval on a fixed random subset of 1024 clips
  python scripts/train.py --motion_dirs /DATA/disk0/gmt_data/ma50/data/in/motion50/g1/g1 \
      --blacklist ~/g1_resume/atlas_full_blacklist_motion50_1010.json \
      --cache_dir ~/supermimic/data/cache --eval_clips 1024 --run_name g1_atlas
  # Same, but the eval subset fixed by npz basename (comparable across file lists) and 24 CPU threads
  python scripts/train.py ... --eval_clip_names eval_1024.txt --num_threads 24
  # One model on 8 GPUs of one machine (data parallel; every rank keeps the per-GPU config above,
  # so the global batch is 8x: 32768 envs / 32768 imagined rollouts). See scripts/launch_bj31_ddp.sh.
  torchrun --standalone --nproc_per_node 8 scripts/train.py ... --num_threads 24
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime


def _die_with_torchrun_agent() -> None:
  """torchrun starts every rank in its own session (torch elastic start_new_session=True): a `kill -9` on the
  launcher's process group (runs/<run>.pgid) reaches only the agent and would leave the ranks running as orphans
  that keep training and hold the GPUs. Ask the kernel to SIGKILL this rank when its parent (the agent) dies."""
  if not sys.platform.startswith("linux") or "TORCHELASTIC_RUN_ID" not in os.environ:
    return
  import ctypes
  import signal

  ppid = os.getppid()
  libc = ctypes.CDLL(None, use_errno=True)
  if libc.prctl(1, int(signal.SIGKILL), 0, 0, 0) != 0:  # 1 = PR_SET_PDEATHSIG
    print(f"[train] prctl(PR_SET_PDEATHSIG) failed (errno {ctypes.get_errno()}); stop with scripts/stop_bj31_ddp.sh", flush=True)
    return
  if ppid == 1 or os.getppid() != ppid:  # the agent died before prctl took effect
    os.kill(os.getpid(), signal.SIGKILL)


def main():
  _die_with_torchrun_agent()
  p = argparse.ArgumentParser()
  p.add_argument("--motion_dirs", type=str, nargs="+", required=True, help="directories (recursive) or npz files")
  p.add_argument("--blacklist", type=str, default=None, help="json with a 'bad_motions' list of npz basenames")
  p.add_argument("--name_exclude", type=str, nargs="*", default=None, help="drop clips whose name contains any keyword")
  p.add_argument("--max_motions", type=int, default=None)
  p.add_argument("--cache_dir", type=str, default="", help="where the packed motion cache lives (default ~/.cache/supermimic_motions)")
  p.add_argument("--num_workers", type=int, default=32)
  p.add_argument("--robot", type=str, default="g1", choices=["g1", "e2"])
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
  # End-effector termination (BeyondMimic ee_body_pos; the paper removed it). On by default since 2026-10-22: without
  # it the policy swings feet / hands far off the reference and fails the eval site's end-effector test.
  p.add_argument("--ee_termination", dest="ee_termination", action="store_true", default=True,
                 help="end-effector height termination in the simulator + matching soft term in imagination (default on)")
  p.add_argument("--no_ee_termination", dest="ee_termination", action="store_false", help="paper recipe: no end-effector termination")
  p.add_argument("--ee_threshold", type=float, default=0.25, help="end-effector z error threshold [m] (eval site: 0.25)")
  p.add_argument("--wm_batches", type=int, default=10)
  p.add_argument("--wm_windows", type=int, default=2048)
  p.add_argument("--wm_k", type=int, default=8)
  p.add_argument("--lr", type=float, default=2e-4)
  p.add_argument("--lr_decay_iters", type=int, default=250_000)
  p.add_argument("--log_interval", type=int, default=50)
  p.add_argument("--eval_interval", type=int, default=2000)
  p.add_argument("--eval_clips", type=int, default=0, help="0 = all clips, else a fixed random subset")
  p.add_argument("--eval_clip_names", type=str, default="", help="txt with npz basenames (one per line): fixed eval subset by name; overrides --eval_clips")
  p.add_argument("--eval_envs", type=int, default=1024)
  p.add_argument("--ckpt_interval", type=int, default=2000)
  p.add_argument("--no_eval", action="store_true")
  p.add_argument("--resume", type=str, default=None)
  p.add_argument("--no_watchdog", action="store_true", help="disable the collapse watchdog (auto-rollback + LR halving)")
  p.add_argument("--wd_drop", type=float, default=0.08)
  p.add_argument("--wd_force_at", type=int, default=-1, help="debug: force one rollback at this iteration")
  p.add_argument("--wd_nf_patience", type=int, default=100, help="consecutive non-finite iterations (skipped updates) before a rollback")
  p.add_argument("--wd_early_ckpt", type=int, default=500, help="early checkpoint / rollback point at this iteration if none exists (0 = off)")
  p.add_argument("--no_compile", action="store_true")
  p.add_argument("--compile_mode", type=str, default=None, help="e.g. reduce-overhead or max-autotune")
  p.add_argument("--num_threads", type=int, default=0, help="CPU threads: OMP/MKL/OpenBLAS env + torch.set_num_threads (0 = leave defaults)")
  # Data parallel (only active under torchrun with more than one process).
  p.add_argument("--lr_scale", type=float, default=1.0, help="multiply policy / critic / world-model LR (e.g. for the larger DDP batch); default 1 = paper LR")
  p.add_argument("--dist_timeout_min", type=float, default=120.0, help="process-group timeout; must exceed one evaluation on rank 0 (and a cache build)")
  p.add_argument("--sampler_sync_interval", type=int, default=0, help="iterations between reset-sampler merges across ranks (0 = --log_interval)")
  p.add_argument("--param_check_interval", type=int, default=1000, help="iterations between cross-rank learned-state hash checks (0 = start/end only)")
  args = p.parse_args()

  if args.num_threads > 0:
    # Before torch / numpy spin up their pools (they are imported below). Pool workers inherit the env.
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
      os.environ[k] = str(args.num_threads)
    import torch

    torch.set_num_threads(args.num_threads)
    print(f"[train] num_threads={args.num_threads} (torch.get_num_threads()={torch.get_num_threads()})")
  if args.eval_clip_names:
    args.eval_clip_names = os.path.abspath(os.path.expanduser(args.eval_clip_names))
    assert os.path.isfile(args.eval_clip_names), f"--eval_clip_names {args.eval_clip_names} not found"

  from supermimic.distributed import DistCtx

  # torchrun -> one process per GPU (device = cuda:LOCAL_RANK); plain python -> disabled, single-GPU code path.
  dctx = DistCtx.from_env(args.device, args.dist_timeout_min)
  args.device = dctx.device

  from supermimic.heads import HeadsCfg
  from supermimic.motions import collect_motion_files
  from supermimic.trainer import TrainCfg, Trainer
  from supermimic.world_model import WorldModelCfg

  # Rank 0 resolves the file list and the run directory; the others take them verbatim (the cache key
  # is the exact ordered path list, and every rank must log into the same directory).
  if dctx.is_main:
    files = collect_motion_files(args.motion_dirs, args.blacklist, args.name_exclude, args.max_motions)
    assert files, "no motion files found"
    print(f"[train] {len(files)} motion files")
    log_dir = os.path.join(args.log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + args.run_name)
  else:
    files, log_dir = None, None
  files, log_dir = dctx.broadcast_object((files, log_dir))
  lr = args.lr * args.lr_scale
  cfg = TrainCfg(
    motion_files=files,
    robot=args.robot,
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
    policy_lr=lr,
    critic_lr=lr,
    lr_decay_iters=args.lr_decay_iters,
    log_interval=args.log_interval,
    eval_interval=args.eval_interval,
    eval_clips=args.eval_clips,
    eval_clip_names=args.eval_clip_names,
    num_threads=args.num_threads,
    eval_envs=args.eval_envs,
    ckpt_interval=args.ckpt_interval,
    eval_enabled=not args.no_eval,
    compile=not args.no_compile,
    wd_enabled=not args.no_watchdog,
    wd_drop=args.wd_drop,
    wd_force_at=args.wd_force_at,
    wd_nf_patience=args.wd_nf_patience,
    wd_early_ckpt=args.wd_early_ckpt,
    compile_mode=args.compile_mode,
    world_size=dctx.world,
    base_lr_scale=args.lr_scale,
    sampler_sync_interval=args.sampler_sync_interval,
    param_check_interval=args.param_check_interval,
    dist_timeout_min=args.dist_timeout_min,
    wm=WorldModelCfg(k=args.wm_k, lr=lr, batches_per_update=args.wm_batches, windows_per_batch=args.wm_windows),
    heads=HeadsCfg(goal_anchor_pos=args.goal_anchor_pos, ee_termination=args.ee_termination, ee_term_z=args.ee_threshold),
  )
  os.makedirs(log_dir, exist_ok=True)
  if dctx.is_main:
    print(f"[train] log_dir = {log_dir}")
    if args.lr_scale != 1.0:
      print(f"[train] lr_scale {args.lr_scale}: policy / critic / world-model lr = {lr:g}")
    with open(os.path.join(args.log_root, "last_run.txt"), "a") as f:
      f.write(log_dir + "\n")
    with open(os.path.join(log_dir, "motion_files.txt"), "w") as f:
      f.write("\n".join(files))
  trainer = Trainer(cfg, dist=dctx)
  if args.resume:
    trainer.load(args.resume)  # every rank loads the same checkpoint (checked for equality at start)
  trainer.run()
  # Clean exit only; on an exception the process just dies and torchrun tears the other ranks down
  # (a destroy_process_group with peers still inside a collective could block).
  dctx.close()


if __name__ == "__main__":
  sys.exit(main())
