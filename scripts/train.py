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
  # master's v2 recipe (bj27 g1v2_scratch): team eval G1 model + contacts, source prior, EE 0.25 (see
  # scripts/launch_bj31_v2.example.sh for the 8-GPU version)
  python scripts/train.py --motion_list <motion_files.txt> --robot g1 --robot_variant eval+evalcontact --ee_term_z 0.25 \
      --source_weights noetix_kungfu=10,noetix_kungfu_dance_show=15,sportsmeeting=10,lafan1=15,g1_all=10,mixamo_cn=2,virtual_point=3
  # Root-orientation termination (pelvis full angle incl. yaw > 1.0 rad, the eval site's root_ori test) on top
  # (A/B on bj31: scripts/launch_bj31_oriterm.example.sh)
  python scripts/train.py ... --root_ori_term 1.0 --root_ori_term_temp 0.05
  # Warm start with early stop on the in-training eval + watchdog on the survival-weighted imagined reward (8-GPU
  # version: scripts/launch_bj31_r15ws.example.sh)
  python scripts/train.py ... --resume model_136000.pt --eval_interval 2000 --es_eval_fail_ratio 1.5 --es_patience 2 \
      --wd_signal imagined_reward_alive --wd_reward_floor -1.0
"""

from __future__ import annotations

import argparse
import math
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


def motion_files_from_args(args) -> list[str]:
  """--motion_list (one npz path per line, e.g. a run's motion_files.txt, used verbatim) or --motion_dirs filtered by
  --blacklist / --name_exclude / --max_motions."""
  if args.motion_list:
    files = [l.strip() for l in open(os.path.expanduser(args.motion_list)) if l.strip()]
    # Any missing file is fatal (here, on rank 0, before the broadcast): the list is used verbatim and is the cache
    # key, nothing is dropped. Otherwise build_cache dies on the first missing header (rank 0 only, the others hang
    # in the barrier until torchrun tears them down), or, with a cache of this exact list, trains on the stale clips.
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
      msg = f"--motion_list {args.motion_list}: {len(missing)} of {len(files)} files do not exist here (first: {missing[0]})"
      raise FileNotFoundError(msg + (" -- a list written on another machine?" if len(missing) == len(files) else
                                     " -- fix the list (it is used verbatim, nothing is dropped)"))
  else:
    from supermimic.motions import collect_motion_files

    files = collect_motion_files(args.motion_dirs, args.blacklist, args.name_exclude, args.max_motions)
  assert files, "no motion files found"
  return files


def human_heads_cfg(args):
  """HumanHeadsCfg from --human_cfg key=value pairs (types from the dataclass defaults; bools: 0/1/true/false)."""
  from dataclasses import fields

  from supermimic.human.heads import HumanHeadsCfg

  types = {f.name: type(f.default) for f in fields(HumanHeadsCfg)}
  kw = {}
  for kv in args.human_cfg.split(","):
    if not kv.strip():
      continue
    k, sep, v = kv.partition("=")
    k = k.strip()
    if not sep or k not in types:
      raise ValueError(f"--human_cfg {kv!r}: key=value with a HumanHeadsCfg field ({sorted(types)})")
    t = types[k]
    kw[k] = v.strip().lower() in ("1", "true", "yes") if t is bool else t(v)
  return HumanHeadsCfg(**kw)


def parse_hard_schedule(spec: str) -> tuple:
  """'it:mult,it:mult,...' -> ((it, mult), ...) (master 4314a50's format; empty -> ())."""
  out = []
  for kv in spec.split(","):
    if not kv.strip():
      continue
    a, sep, b = kv.partition(":")
    if not sep:
      raise ValueError(f"{kv!r} is not it:mult")
    m = float(b)
    if not m > 0:
      raise ValueError(f"multiplier {m} must be > 0")
    out.append((int(a), m))
  return tuple(out)


def main():
  _die_with_torchrun_agent()
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
  p.add_argument("--reference", type=str, default="robot", choices=["robot", "human"],
                 help="robot: retargeted robot npz + BeyondMimic rewards (default); human: raw SMPL npz (supermimic/human), "
                      "goal = human motion features, bone-direction rewards, no retargeting")
  p.add_argument("--human_cfg", type=str, default="",
                 help="reference human: HumanHeadsCfg overrides key=value,... (e.g. w_heading=0,goal_rel_yaw=0,term_trunk=1.2)")
  p.add_argument("--robot_variant", type=str, default="", help="g1: eval (team eval / deploy model); e2: nodamp (no passive joint damping)")
  p.add_argument("--source_weights", type=str, default="",
                 help="reset-sampler prior per data source, e.g. noetix_kungfu=6,sportsmeeting=10 (clip '<robot>__<source>_...')")
  p.add_argument("--name_weights", type=str, default="",
                 help="reset-sampler weight per clip-name keyword (case-insensitive substring), e.g. crawl=4,getup=4")
  # Hard-clip termination curriculum (master 4314a50; bj27 r13 / r14 used it, r15 did not): every termination threshold
  # (env + soft head) of the clips matching --hard_names is multiplied by the schedule's value at the iteration.
  p.add_argument("--hard_names", type=str, default="",
                 help="clip-name keywords (comma-separated, case-insensitive) of the hard set for --hard_term_schedule")
  p.add_argument("--hard_term_schedule", type=str, default="",
                 help="termination multiplier of the hard set, piecewise linear over absolute iterations, e.g. 104000:2,108000:2,114000:1")
  p.add_argument("--sampler_cap", type=float, default=0.0,
                 help="cap every bin's softmax share at this x the uniform share 1/num_bins (allmimic: 50); 0 = off")
  p.add_argument("--sampler_clusters", type=int, default=0,
                 help="k-means clusters of 12 kinematic bin features; the uniform share becomes cluster-balanced (allmimic: 256)")
  p.add_argument("--sampler_cluster_alpha", type=float, default=0.3, help="cluster share ~ n_bins^alpha")
  p.add_argument("--sampler_cluster_mix", type=float, default=0.5,
                 help="uniform-share prior = (1 - mix) x source / keyword prior + mix x cluster prior")
  p.add_argument("--sampler_boot_mean", action="store_true",
                 help="sampler returns: bootstrap time-outs with the episode's mean reward instead of its last reward")
  p.add_argument("--ee_term_z", type=float, default=None,
                 help="master's v2 spelling of the end-effector height termination [m] (env + soft head): 0 = off as in "
                      "the paper, 0.25 = BeyondMimic / team benchmark; same as --ee_termination --ee_threshold Z / --no_ee_termination")
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
  p.add_argument("--goal_ref_root", action="store_true",
                 help="goal frames + reference anchor height / body-frame lin & ang vel / projected gravity (no odometry needed)")
  p.add_argument("--critic_errors", action="store_true", help="privileged tracking-error features for the critic")
  # End-effector termination (BeyondMimic ee_body_pos; the paper removed it). On by default since 2026-10-22: without
  # it the policy swings feet / hands far off the reference and fails the eval site's end-effector test.
  # NOTE master's default is OFF (--ee_term_z 0.0); here no flag = ON at 0.25 m. --ee_term_z (master) and these two
  # spellings may be mixed only when they agree (heads.ee_flags).
  p.add_argument("--ee_termination", dest="ee_termination", action="store_true", default=None,
                 help="end-effector height termination in the simulator + matching soft term in imagination (default on)")
  p.add_argument("--no_ee_termination", dest="ee_termination", action="store_false", help="paper recipe: no end-effector termination")
  p.add_argument("--ee_threshold", type=float, default=None, help="end-effector z error threshold [m] (default 0.25 = eval site)")
  # Root-orientation termination (2026-10-25, off by default): 101 of 129 eval-site failures of r15ws model_148000 were its
  # root_ori test (pelvis full rotation angle incl. yaw > 1.0 rad), which no training termination saw.
  p.add_argument("--root_ori_term", type=float, default=0.0,
                 help="terminate when the pelvis' (robot root's) full rotation angle vs the reference, yaw included, exceeds "
                      "this [rad] in the simulator, + matching soft term in imagination; 1.0 = the eval site; 0 = off (default)")
  p.add_argument("--root_ori_term_temp", type=float, default=0.05,
                 help="sigmoid temperature [rad] of the imagined root-orientation term: a per-step hazard, 0.5 at the "
                      "threshold for ONE step; over --H 48 imagined steps a constant angle survives 50%% at threshold - 4.23 x "
                      "temp (0.79 rad at 1.0 / 0.05, 0.92 at 0.02), i.e. imagination already counts ~0.8-0.98 rad as dying")
  p.add_argument("--root_ori_body", type=str, default="",
                 help="tracked body of the root-orientation test (default: the robot's root body, G1 pelvis / E2 base_link)")
  p.add_argument("--root_z_term", type=float, default=0.0,
                 help="terminate when the root-orientation body's (G1 pelvis) height error vs the reference exceeds this [m] "
                      "(env + soft term in imagination); 0.25 = the eval site's root_pos test (mode z); 0 = off")
  p.add_argument("--root_z_term_temp", type=float, default=0.02, help="sigmoid temperature of the imagined root_z term")
  p.add_argument("--wm_body_frame", action="store_true",
                 help="WM features / targets in the root frame (no ZYX heading, singular at pitch +-90 deg); resuming a "
                      "heading-frame checkpoint starts a fresh WM -> use --policy_delay")
  p.add_argument("--wm_layernorm", action="store_true",
                 help="LayerNorm after every hidden layer of the world model (resuming a checkpoint without it grafts "
                      "identity LayerNorms -> use --policy_delay)")
  p.add_argument("--wm_input_clip", type=float, default=0.0, help="clamp of the world model's normalized inputs (0 = off)")
  p.add_argument("--wm_min_std", type=float, default=1e-4, help="floor of the world model's input normalizer std")
  p.add_argument("--wm_acro_weight", type=float, default=1.0,
                 help="weight of acrobatic transitions in the world-model loss (start state spinning > --wm_acro_ang_vel "
                      "rad/s or root tilted past 60 deg); 1 = off")
  p.add_argument("--wm_acro_ang_vel", type=float, default=4.0, help="angular speed [rad/s] of an acrobatic WM transition")
  p.add_argument("--policy_delay", type=int, default=0, help="iterations of world-model-only fitting after start / resume")
  p.add_argument("--policy_freeze", type=int, default=0, help="then iterations of critic fitting without policy steps")
  p.add_argument("--wm_delay_lr", type=float, default=1.0, help="world-model lr multiplier during --policy_delay")
  p.add_argument("--resume_lr_scale", type=float, default=None,
                 help="override the run-time learning-rate scale restored from --resume (watchdog halvings accumulate along a "
                      "resume chain: the bj27 G1 v2 chain carried 0.0625 from round 15 on); not --base_lr_scale, which multiplies "
                      "the base learning rate")
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
  p.add_argument("--wd_drop", type=float, default=0.2)
  p.add_argument("--wd_force_at", type=int, default=-1, help="debug: force one rollback at this iteration")
  p.add_argument("--wd_warmup", type=int, default=500,
                 help="watchdog quiet iterations after start / resume / rollback (the bj27 single-GPU line used 1500)")
  p.add_argument("--wd_baseline_after_warmup", action="store_true",
                 help="the watchdog's reward history starts after the warm-up (no baseline from the replay refill after a "
                      "resume; the bj27 single-GPU line's behaviour)")
  # 10-25 fixes from the open-source comparison (all off by default)
  p.add_argument("--input_clip", type=float, default=10.0, help="clamp of normalized policy / critic inputs (v2: 10)")
  p.add_argument("--clip_values", action="store_true", help="clamp bootstrap / lambda-return values to +-r_max/(1-gamma)")
  p.add_argument("--critic_weight_survival", action="store_true", help="critic loss weighted by imagined survival")
  p.add_argument("--imag_truncate", action="store_true", help="imagined rollouts end at the clip end with a critic bootstrap")
  p.add_argument("--wm_prefix_mask", action="store_true", help="WM windows crossing an episode end keep the steps up to it")
  p.add_argument("--clean_prev_action", action="store_true", help="proprio a_{t-1} = policy action (not the noisy executed one)")
  p.add_argument("--fixes", action="store_true", help="all six 10-25 fixes above, input_clip 50")
  p.add_argument("--wd_nf_patience", type=int, default=100, help="consecutive non-finite iterations (skipped updates) before a rollback")
  p.add_argument("--wd_early_ckpt", type=int, default=500, help="early checkpoint / rollback point at this iteration if none exists (0 = off)")
  p.add_argument("--wd_reward_floor", type=float, default=None,
                 help="floor each rank's imagined reward (normalized) at this value before the rank mean / watchdog EMA "
                      "(production -1.0; default off = as before). A finite but astronomically negative value on one rank caused two "
                      "needless rollbacks on bj31 (2026-10-23)")
  p.add_argument("--wd_sustain", type=int, default=1,
                 help="the watchdog's reward drop must hold this many consecutive iterations (default 1 = immediate rollback; "
                      "the bj27 single-GPU line used 200 with --wd_signal imagined_reward_robust)")
  p.add_argument("--wd_signal", type=str, default="imagined_reward_norm",
                 choices=["imagined_reward_norm", "imagined_reward_alive", "imagined_reward_robust"],
                 help="watchdog reward signal: imagined_reward_norm (default, as before: mean reward over every imagined step, "
                      "dead ones included) or imagined_reward_alive (sum_t c_t r_t / sum_t c_t with the policy objective's "
                      "survival weights c_t: blind to post-death world-model divergence); --wd_reward_floor applies to either")
  # Early stop on the in-training benchmark (needs evaluation: --eval_interval > 0, no --no_eval). Off by default.
  p.add_argument("--es_eval_fail_ratio", type=float, default=None,
                 help="stop when an in-training eval fails >= R x median(failed of all previous evals of this run, >= 3 of "
                      "them) for --es_patience consecutive evals (writes <log_dir>/earlystop.json, saves model_final); "
                      "also keeps the checkpoint with the fewest eval failures (model_best_eval.pt, best_eval.json). A watchdog "
                      "rollback drops the evals of the abandoned branch; a --resume at an eval slot evaluates the loaded "
                      "weights first. Default off")
  p.add_argument("--es_patience", type=int, default=2, help="consecutive bad evals before the early stop (default 2)")
  p.add_argument("--es_median_floor", type=float, default=1.0,
                 help="floor of the previous evals' median failed count in the early-stop threshold (default 1; with "
                      "single-digit failures on a 1024-clip eval set 5, else a median of 2 makes 3 failures bad)")
  p.add_argument("--no_compile", action="store_true")
  p.add_argument("--compile_mode", type=str, default=None, help="e.g. reduce-overhead or max-autotune")
  p.add_argument("--num_threads", type=int, default=0, help="CPU threads: OMP/MKL/OpenBLAS env + torch.set_num_threads (0 = leave defaults)")
  # Data parallel (only active under torchrun with more than one process).
  p.add_argument("--base_lr_scale", type=float, default=1.0,
                 help="multiply the policy / critic / world-model base LR (e.g. for the larger DDP batch); default 1 = paper LR")
  # Removed name (10-26 consolidation): on ddp-bj31-v2 it was the base multiplier above, on master 7afc772 the run-time override
  # now called --resume_lr_scale. Old launch scripts would silently get the other meaning, so it is an error.
  p.add_argument("--lr_scale", type=float, default=None, help=argparse.SUPPRESS)
  p.add_argument("--dist_timeout_min", type=float, default=120.0, help="process-group timeout; must exceed one evaluation on rank 0 (and a cache build)")
  p.add_argument("--sampler_sync_interval", type=int, default=0, help="iterations between reset-sampler merges across ranks (0 = --log_interval)")
  p.add_argument("--param_check_interval", type=int, default=1000, help="iterations between cross-rank learned-state hash checks (0 = start/end only)")
  # Reset sampler: penalty-dominated clip-end bins reach G << 0 (r_last / (1 - gamma) bootstrap) and one of them takes
  # the whole softmax share (g1_full_ee_ddp8, 2026-10-23). Off by default = paper / runs before the switch.
  p.add_argument("--sampler_clamp_returns", action="store_true",
                 help="clamp reset-sampler bin returns to [0, g_max] (EMA input, sampled G, bin_G loaded on resume / rollback)")
  args = p.parse_args()
  if args.lr_scale is not None:
    p.error("--lr_scale no longer exists (10-26): use --base_lr_scale to multiply the base learning rate (data-parallel "
            "branch meaning) or --resume_lr_scale to override the run-time scale restored by --resume (master 7afc772 meaning)")
  if args.motion_list and (args.motion_dirs or args.blacklist or args.name_exclude or args.max_motions):
    p.error("--motion_list replaces --motion_dirs / --blacklist / --name_exclude / --max_motions (give one or the other)")
  if not args.motion_list and not args.motion_dirs:
    p.error("--motion_dirs or --motion_list is required")
  hard_names = tuple(k for k in args.hard_names.split(",") if k)
  try:
    hard_schedule = parse_hard_schedule(args.hard_term_schedule)
  except ValueError as e:
    p.error(f"--hard_term_schedule {args.hard_term_schedule!r}: {e}")
  if hard_schedule and not hard_names:
    p.error("--hard_term_schedule needs --hard_names (the hard set it applies to)")
  if not (0.0 <= args.root_ori_term < math.pi):
    p.error(f"--root_ori_term must be in [0, pi) rad (0 = off; the angle never exceeds pi), got {args.root_ori_term}")
  if args.root_ori_term > 0 and not args.root_ori_term_temp > 0:
    p.error(f"--root_ori_term_temp must be > 0, got {args.root_ori_term_temp}")
  if args.es_eval_fail_ratio is not None:
    if not args.es_eval_fail_ratio > 0 or args.es_patience < 1:
      p.error("--es_eval_fail_ratio must be > 0 and --es_patience >= 1")
    if args.no_eval or args.eval_interval <= 0:
      p.error("--es_eval_fail_ratio needs the in-training evaluation (drop --no_eval, --eval_interval > 0)")

  if args.num_threads > 0:
    # Before torch / numpy spin up their pools (they are imported below). Pool workers inherit the env.
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
      os.environ[k] = str(args.num_threads)
    import torch

    torch.set_num_threads(args.num_threads)
    print(f"[train] num_threads={args.num_threads} (torch.get_num_threads()={torch.get_num_threads()})")
  from supermimic.heads import ee_flags  # imports torch: after the thread env above

  try:
    ee_on, ee_thr = ee_flags(args.ee_term_z, args.ee_termination, args.ee_threshold)
  except ValueError as e:
    p.error(str(e))
  if args.eval_clip_names:
    args.eval_clip_names = os.path.abspath(os.path.expanduser(args.eval_clip_names))
    assert os.path.isfile(args.eval_clip_names), f"--eval_clip_names {args.eval_clip_names} not found"

  from supermimic.distributed import DistCtx

  # torchrun -> one process per GPU (device = cuda:LOCAL_RANK); plain python -> disabled, single-GPU code path.
  dctx = DistCtx.from_env(args.device, args.dist_timeout_min)
  args.device = dctx.device

  from supermimic.heads import HeadsCfg
  from supermimic.trainer import TrainCfg, Trainer
  from supermimic.world_model import WorldModelCfg

  # Rank 0 resolves the file list and the run directory; the others take them verbatim (the cache key
  # is the exact ordered path list, and every rank must log into the same directory). The reset sampler's
  # source / name prior is built per rank from this same list (checked equal across ranks at start).
  if dctx.is_main:
    files = motion_files_from_args(args)
    print(f"[train] {len(files)} motion files")
    log_dir = os.path.join(args.log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + args.run_name)
  else:
    files, log_dir = None, None
  files, log_dir = dctx.broadcast_object((files, log_dir))
  lr = args.lr * args.base_lr_scale
  cfg = TrainCfg(
    motion_files=files,
    robot=args.robot,
    robot_variant=args.robot_variant,
    source_weights=tuple((kv.split("=")[0], float(kv.split("=")[1])) for kv in args.source_weights.split(",") if kv),
    name_weights=tuple((kv.split("=")[0], float(kv.split("=")[1])) for kv in args.name_weights.split(",") if kv),
    hard_names=hard_names,
    hard_term_schedule=hard_schedule,
    sampler_cap=args.sampler_cap,
    sampler_clusters=args.sampler_clusters,
    sampler_cluster_alpha=args.sampler_cluster_alpha,
    sampler_cluster_mix=args.sampler_cluster_mix,
    sampler_boot_mean=args.sampler_boot_mean,
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
    wd_warmup=args.wd_warmup,
    input_clip=50.0 if args.fixes and args.input_clip == 10.0 else args.input_clip,
    clip_values=args.clip_values or args.fixes,
    critic_weight_survival=args.critic_weight_survival or args.fixes,
    imag_truncate=args.imag_truncate or args.fixes,
    wm_prefix_mask=args.wm_prefix_mask or args.fixes,
    clean_prev_action=args.clean_prev_action or args.fixes,
    policy_delay=args.policy_delay,
    policy_freeze=args.policy_freeze,
    wm_delay_lr=args.wm_delay_lr,
    wd_nf_patience=args.wd_nf_patience,
    wd_early_ckpt=args.wd_early_ckpt,
    wd_reward_floor=args.wd_reward_floor,
    wd_signal=args.wd_signal,
    wd_sustain=args.wd_sustain,
    wd_baseline_after_warmup=args.wd_baseline_after_warmup,
    es_eval_fail_ratio=args.es_eval_fail_ratio,
    es_patience=args.es_patience,
    es_median_floor=args.es_median_floor,
    compile_mode=args.compile_mode,
    world_size=dctx.world,
    base_lr_scale=args.base_lr_scale,
    sampler_sync_interval=args.sampler_sync_interval,
    sampler_clamp_returns=args.sampler_clamp_returns,
    param_check_interval=args.param_check_interval,
    dist_timeout_min=args.dist_timeout_min,
    wm=WorldModelCfg(k=args.wm_k, lr=lr, batches_per_update=args.wm_batches, windows_per_batch=args.wm_windows,
                     body_frame=args.wm_body_frame, layernorm=args.wm_layernorm, input_clip=args.wm_input_clip,
                     min_std=args.wm_min_std, acro_weight=args.wm_acro_weight, acro_ang_vel=args.wm_acro_ang_vel),
    reference=args.reference,
    heads=human_heads_cfg(args) if args.reference == "human" else HeadsCfg(goal_anchor_pos=args.goal_anchor_pos, goal_ref_root=args.goal_ref_root, critic_errors=args.critic_errors,
                   ee_termination=ee_on, ee_term_z=ee_thr, root_ori_term=args.root_ori_term,
                   root_ori_term_temp=args.root_ori_term_temp, root_ori_body=args.root_ori_body,
                   root_z_term=args.root_z_term, root_z_term_temp=args.root_z_term_temp),
  )
  os.makedirs(log_dir, exist_ok=True)
  if dctx.is_main:
    print(f"[train] log_dir = {log_dir}")
    if args.base_lr_scale != 1.0:
      print(f"[train] base_lr_scale {args.base_lr_scale}: policy / critic / world-model lr = {lr:g}")
    with open(os.path.join(args.log_root, "last_run.txt"), "a") as f:
      f.write(log_dir + "\n")
    with open(os.path.join(log_dir, "motion_files.txt"), "w") as f:
      f.write("\n".join(files))
  trainer = Trainer(cfg, dist=dctx)
  if args.resume:
    trainer.load(args.resume)  # every rank loads the same checkpoint (checked for equality at start)
  if args.resume_lr_scale is not None:
    trainer._set_lr_scale(args.resume_lr_scale)
    if dctx.is_main:
      print(f"[train] run-time lr_scale set to {args.resume_lr_scale} (policy lr {trainer.opt_pi.param_groups[0]['lr']:.2e})")
  trainer.run()
  # Clean exit only; on an exception the process just dies and torchrun tears the other ranks down
  # (a destroy_process_group with peers still inside a collective could block).
  dctx.close()


if __name__ == "__main__":
  sys.exit(main())
