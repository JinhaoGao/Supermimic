#!/usr/bin/env bash
# EXAMPLE -- not run by anyone yet. bj31, 8 GPUs, torchrun data parallel: warm start from bj27's best SuperMimic G1
# checkpoint, g1v2_clamp_r15 model_136000 (eval site run 35806e9d, 89.85% MotionAtlas v4; parent r14 model_128000
# 89.06%), with r15's recipe per rank, plus the 10-24 fixes:
#   * in-training eval every 2000 its on the fixed 1024-name subset (r15: --no_eval) + early stop when its failures
#     reach 1.5 x the median of the earlier evals twice in a row (--es_eval_fail_ratio 1.5 --es_patience 2); the
#     checkpoint with the fewest eval failures is kept as model_best_eval.pt (best_eval.json; earlystop.json at a stop);
#   * watchdog on the survival-weighted imagined reward (--wd_signal imagined_reward_alive) with the per-rank floor
#     (--wd_reward_floor -1.0): bj27 r16 rolled back 6x on the unweighted monitor (LR 2e-8, gave up);
#   * effective LR at the start 2.5e-5 = r15's own start (5e-5 x cosine(128000) = 2.48e-5) instead of the 1.4e-6 the
#     checkpoint carries (lr_scale 0.0625 after r15's four rollbacks), see LR below.
# Needs branch ddp-bj31-v2 (this commit or later) deployed on bj31 (REPO below, with a COMMIT file) and free GPUs.
#
#   bash scripts/launch_bj31_r15ws.example.sh prep            # once: checks + extended cache (~11 min on 64 workers)
#   bash scripts/launch_bj31_r15ws.example.sh launch [run]    # default run name g1v2_r15ws_ddp8
#   bash scripts/stop_bj31_ddp.sh <run>
#
# r15 (bj27, from its session log 10-23 20:39; code = master 5f020a9, every py md5-identical; checkpoint cfg checked):
#   train.py --run_name g1v2_clamp_r15 --robot g1 --robot_variant eval+evalcontact --motion_list .../g1_plus_phc.txt
#     --resume .../g1v2_phc_r14/model_128000.pt --iterations 136000 --lr 5e-5 --H 48 --ee_term_z 0.25 --eval_clips 1024
#     --no_eval --num_workers 64 --source_weights <SOURCES> --name_weights <NAMES> --sampler_clamp_returns
#   defaults: 4096 envs, 4096 rollouts, wm 10 x 2048 windows k 8, cosine over 250k -> 3%, seed 0, wd_drop 0.2, watchdog
#   on (rolled back 4x: lr_scale 0.0625), EE 0.25 m (temp 0.02), no hard-clip curriculum (r13 / r14 had one, it was back
#   at x1 from 126000), single GPU.
# Same here per rank (tests/test_merge_master_1025.py::test_r15ws_example_reproduces_r15 parses the ARGS block below
# into TrainCfg and checks it against r15's checkpoint cfg: no difference reported on resume). Differences from r15:
#   * 8 ranks, seeds 0..7, global batch 8x (32768 envs / 32768 imagined rollouts), LR not scaled for it -- as every
#     bj31 line so far;
#   * the eval / early stop / watchdog signal / floor / LR above.
#
# DATA. r15's clip set = 254,518 g1__ (atlas minus blacklist, = $OLD_LIST, cache lib_254518_926db30c44 exists) + 490
#   g1_phc__ kungfu retargets in bj27's phc_extra.txt order ($EXTRA): 255,008 files, 265,826,690 frames, 5,428,418 bins
#   = the checkpoint's bin_G. $LIST (prepared 10-25) = $OLD_LIST + $EXTRA, md5 a038555b..., basename sequence md5
#   fbd2ddf1... = the checkpoint's motion_files (bj27 roots): the reset sampler's statistics resume as they are.
#   bj31's phc npz files have no body_names key (body arrays equal to bj27's up to 1e-6, joints bitwise): the cache
#   reads them through the fixed fallback order, as the 254,518 before. extend_cache copies lib_254518 (~182 GB more
#   on /data2, 2.9 TB free) instead of a ~27 min rebuild on rank 0.
#
# LR. Trainer.load restores lr_scale 0.0625 from the checkpoint and the cosine continues at 136000, so the optimizer LR
#   is --lr x 0.0625 x cosine(136000) = --lr x 0.0625 x 0.448172 -> --lr 8.9252e-4 gives 2.5000e-5 (policy, critic and
#   world model alike; the log prints "policy lr 2.500e-05 = lr x lr_scale 0.0625 x cosine 0.4482" at resume). A
#   watchdog rollback halves it from there. At 250000 it reaches 8.9252e-4 x 0.0625 x 0.03 = 1.7e-6.
set -euo pipefail
STEP="${1:?prep | launch [run_name]}"
RUN="${2:-g1v2_r15ws_ddp8}"
REPO="${REPO:-/home/ubuntu/supermimic/supermimic_repro_v2}"  # ddp-bj31-v2 checkout / rsync
VENV="${VENV:-/home/ubuntu/supermimic/mjlab/.venv}"
CACHE=/data2/supermimic/cache
OLD_LIST=/data2/supermimic/lists/g1v2_scratch_254518_bj31.txt
EXTRA=/data2/supermimic/prep_1025/phc_extra_490_bj31.txt
LIST=/data2/supermimic/prep_1025/g1v2_r15_255008_bj31.txt
NEW_CACHE="$CACHE/lib_255008_3b84d5f141"
EVAL_NAMES=/data2/supermimic/lists/eval_1024_g1atlasv1.txt  # the 10-24 diagnosis' eval set (all 1024 in $LIST)
# bj27 /home/ubuntu/supermimic/logs/2026-10-23_20-39-52_g1v2_clamp_r15/model_136000.pt, copied to bj31 from the
# workstation (bj27 is read-only for us; scp only reads it):
#   scp -3 tencent_bj27:/home/ubuntu/supermimic/logs/2026-10-23_20-39-52_g1v2_clamp_r15/model_136000.pt \
#       tencent_bj31:/data2/supermimic/warmstart_g1v2_clamp_r15_model_136000.pt
CKPT=/data2/supermimic/warmstart_g1v2_clamp_r15_model_136000.pt
CKPT_MD5=aa46502442505f0536fc6204ab08cd13  # 303,347,240 bytes
SOURCES=noetix_kungfu=10,noetix_kungfu_dance_show=15,sportsmeeting=10,lafan1=15,g1_all=10,mixamo_cn=2,virtual_point=3
NAMES=tengkong=200,kongfan=200,shoufan=200,erqijiao=200,danpaijiao=200,xuanfeng=200,waibailian=100,lihetui=100,saotang=100,ceti=100,houti=100,zhengti=100
ITERS="${ITERS:-250000}"  # the cosine's end; the early stop is expected to end the run before

export PYTHONPATH="$REPO"
check_list() {
  echo "a038555bfba5a0220426e3ef3ca6ebbe  $LIST" | md5sum -c --quiet -
  [ "$(wc -l < "$EXTRA")" -eq 490 ] && [ "$(wc -l < "$OLD_LIST")" -eq 254518 ]
}
check_cache() {  # complete, 255,008 clips (all ok), 5,428,418 bins of 1 s = the checkpoint's bin_G
  "$VENV/bin/python" - "$NEW_CACHE" <<'EOF'
import json, math, sys
d = sys.argv[1]
meta = json.load(open(f"{d}/meta.json"))
clips = json.load(open(f"{d}/clips.json"))
ok = [c for c in clips if c.get("ok", True) and c["end"] > c["start"] + 1]
bins = sum(math.ceil((c["end"] - c["start"]) / 50) for c in ok)
print(f"[r15ws] {d}: {len(clips)} clips ({len(ok)} usable), {meta['frames']} frames, {bins} bins")
assert len(clips) == 255008 and len(ok) == 255008 and meta["frames"] == 265826690 and bins == 5428418
EOF
}

case "$STEP" in
  prep)
    check_list
    # The cache key of $LIST (supermimic.motions.library_cache_dir) must be the directory checked below.
    "$VENV/bin/python" "$REPO/scripts/extend_cache.py" --old_list "$OLD_LIST" --extra "$EXTRA" --cache_dir "$CACHE" \
      --out_list "$LIST" --dry_run
    nice -n 5 "$VENV/bin/python" "$REPO/scripts/extend_cache.py" --old_list "$OLD_LIST" --extra "$EXTRA" --cache_dir "$CACHE" \
      --out_list "$LIST" --num_workers 64
    check_cache
    if [ -f "$CKPT" ]; then echo "$CKPT_MD5  $CKPT" | md5sum -c -; else echo "[r15ws] $CKPT missing: copy it (see CKPT above)"; fi
    exit 0
    ;;
  launch) ;;
  *) echo "STEP must be prep or launch" >&2; exit 1 ;;
esac

if [ -z "${FORCE:-}" ] && [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; then
  echo "[r15ws] GPUs are in use; stop that run first or set FORCE=1" >&2
  exit 1
fi
check_list
check_cache
echo "$CKPT_MD5  $CKPT" | md5sum -c --quiet -

# >>> ARGS (train.py arguments; parsed by tests/test_merge_master_1025.py -- keep one flag group per line)
ARGS=(
  --motion_list "$LIST"
  --robot g1 --robot_variant eval+evalcontact
  --ee_term_z 0.25
  --source_weights "$SOURCES"
  --name_weights "$NAMES"
  --sampler_clamp_returns
  --num_envs 4096 --rollouts 4096 --H 48 --wm_batches 10 --wm_windows 2048 --wm_k 8
  --lr 8.9252e-4 --lr_decay_iters 250000 --iterations "$ITERS" --seed 0
  --wd_drop 0.2 --wd_reward_floor -1.0 --wd_signal imagined_reward_alive
  --num_workers 64
  --eval_clip_names "$EVAL_NAMES" --eval_interval 2000 --eval_envs 1024
  --es_eval_fail_ratio 1.5 --es_patience 2
  --ckpt_interval 2000 --log_interval 50 --param_check_interval 1000
  --resume "$CKPT"
)
# <<< ARGS

# The exact command (launch_bj31_ddp.sh adds: torchrun --standalone --nproc_per_node 8 ... scripts/train.py
#   --log_root /data2/supermimic/logs --cache_dir /data2/supermimic/cache --run_name $RUN --num_threads 24).
# After the start, the log must show: no "WARNING resume from another setup" line; "clamp_returns: 0 of 5428418 loaded
# bin returns clamped" (r15 stored them clamped); "resumed from ... at iteration 136000 (policy lr 2.500e-05 = lr x
# lr_scale 0.0625 x cosine 0.4482)"; "watchdog signal: pi/imagined_reward_alive"; "early stop: eval failed >= 1.5 x
# median"; "[dist] start: learned state identical on all 8 ranks". First eval at 138000, first early-stop verdict at
# 144000 (needs 3 earlier evals); "[es] it ..." after every eval, "[earlystop] ..." + earlystop.json when it fires.
NPROC=8 THREADS=24 bash "$REPO/scripts/launch_bj31_ddp.sh" "$RUN" "${ARGS[@]}"
