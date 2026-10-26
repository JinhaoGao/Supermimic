#!/usr/bin/env bash
# EXAMPLE -- not run by anyone yet. bj31, 8 GPUs, torchrun data parallel: master's v2 G1 recipe as the bj27 run
# g1v2_scratch used it (single GPU, code 75788d7, 87.01% MotionAtlas v4 at model_86000), per rank, on its exact clip
# list (DATASET=v2list, the default), plus the differences listed below. DATASET=dir is NOT that recipe (see DATA).
# Needs branch ddp-bj31-v2 deployed on bj31 (default REPO below, with a COMMIT file like supermimic_repro_clamp) and
# free GPUs: the run g1_full_ee_ddp8_r52 holds all 8 now.
#
#   bash scripts/launch_bj31_v2.example.sh [run_name]          # DATASET=v2list (default) | dir, see DATA below
#   bash scripts/stop_bj31_ddp.sh <run_name>
#
# g1v2_scratch (bj27, from ps; everything not listed is a default):
#   train.py --run_name g1v2_scratch --ee_term_z 0.25 --eval_clips 1024 --no_eval --num_workers 64
#     --source_weights noetix_kungfu=10,noetix_kungfu_dance_show=15,sportsmeeting=10,lafan1=15,g1_all=10,mixamo_cn=2,virtual_point=3
#     --robot g1 --robot_variant eval+evalcontact --motion_list .../2026-10-22_11-09-31_g1_atlas_v1r/motion_files.txt
#   defaults: 4096 envs, 4096 rollouts, H 32, LR 2e-4 (policy / critic / WM) cosine over 250k -> 3%, 320k its,
#   wd_drop 0.2, seed 0, ckpt every 2000; EE term 0.25 m on ankle_roll + wrist_yaw, soft temp 0.02.
#   (It was rolled back twice by the watchdog, at 67496 and 74620; model_86000 trained at 0.25x LR.)
#
# Same here, per rank. Differences from g1v2_scratch:
#   * --sampler_clamp_returns (reset-sampler bin returns in [0, g_max]; the bj31 fix for the softmax collapse);
#   * 8 ranks, seeds 0..7, global batch 8x (32768 envs / 32768 imagined rollouts), LR not scaled -- as g1_full_ee_ddp8;
#   * in-training benchmark on rank 0 every 4000 its on the fixed 1024-name subset (g1v2_scratch: --no_eval, scored on
#     the eval site); it does not change the training.
# Differences from bj31's g1_full_ee_ddp8_r52 (84.75% peak; commit 5bb8c0d, same EE 0.25 / clamp / batch / LR / seeds):
#   * the v2 options: --robot_variant eval+evalcontact (team eval G1, 35.11 kg, condim 3; r52: mjlab's 33.34 kg G1) and
#     --source_weights (r52: uniform share without a prior);
#   * the data: v2list = 254,518 clips (r52: the dir set, 365,855), unless DATASET=dir;
#   * the watchdog: wd_drop 0.2 and the replay emptied on a rollback (master e57be5b); r52 runs 5bb8c0d's defaults,
#     wd_drop 0.08 and the replay kept.
#   So neither comparison isolates one change: vs g1v2_scratch it is the data-parallel scale-up (+ clamp), vs r52 it is
#   v2 options + data + watchdog.
#
# DATA. g1v2_scratch trained on 254,518 clips (G1 MotionAtlas g1/g1 minus the GMT blacklist; its motion_files.txt =
#   bj27 ~/supermimic/lists/g1_atlas_minus_blacklist.txt). bj31 has no bj31-path copy of that list. It has the same
#   list with bj27 paths (/data2/supermimic/lists/bj27_g1_atlas_v1_motion_files.txt, md5 1980719c..., identical to
#   g1v2_scratch's), and all 254,518 basenames exist in $DATA (checked 2026-10-23).
#     DATASET=v2list (default) exact g1v2 clip set and order: writes $V2LIST (bj27 list re-rooted at $DATA by basename)
#                    and passes --motion_list (new cache, ~145 GB on /data2 (3.1 TB free), ~25 min on rank 0 before the
#                    first iteration). A g1v2_scratch checkpoint resumes into it with the sampler state (same basenames).
#     DATASET=dir    --motion_dirs $DATA --blacklist $BLACKLIST = 365,855 clips (the r52 set, cache exists): the 254,518
#                    + 110,847 g1_umr__* (sonic, weight 1) + 490 g1_phc__noetix_kungfu_* retargets (x10: kungfu 905 ->
#                    1,395 clips). Not g1v2's recipe: "v2 options on the r52 data", run name gets _r52data. Uniform
#                    share per source (master's reset_prior on bj31's lib_365855 clips.json, 1 s bins; v2list -> dir):
#                    noetix_kungfu 2.40 -> 3.32%, dance_show 1.97 -> 1.71%, sportsmeeting 1.97 -> 1.71%, lafan1
#                    1.82 -> 1.58%, g1_all 2.03 -> 1.76%, mixamo_cn 3.30 -> 2.87%, virtual_point 1.45 -> 1.26%
#                    (every source but kungfu -13%); bins 5.42M -> 6.27M (+15.7%).
#   The eval names ($EVAL_NAMES) are all in both sets.
set -euo pipefail
REPO="${REPO:-/home/ubuntu/supermimic/supermimic_repro_v2}"  # ddp-bj31-v2 checkout / rsync
DATA=/home/ubuntu/motion_dataset/g1_atlas_full
BLACKLIST=/data2/supermimic/lists/atlas_full_blacklist_motion50_1010.json
EVAL_NAMES=/data2/supermimic/lists/eval_1024_g1atlasv1.txt
BJ27_LIST=/data2/supermimic/lists/bj27_g1_atlas_v1_motion_files.txt
V2LIST=/data2/supermimic/lists/g1v2_scratch_254518_bj31.txt
SOURCES=noetix_kungfu=10,noetix_kungfu_dance_show=15,sportsmeeting=10,lafan1=15,g1_all=10,mixamo_cn=2,virtual_point=3
DATASET="${DATASET:-v2list}"
if [ "$DATASET" = dir ]; then RUN="${1:-g1v2_ddp8_clamp_r52data}"; else RUN="${1:-g1v2_ddp8_clamp}"; fi

if [ -z "${FORCE:-}" ] && [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; then
  echo "[v2] GPUs are in use (g1_full_ee_ddp8_r52?); stop that run first or set FORCE=1" >&2
  exit 1
fi

case "$DATASET" in
  dir)
    DATA_ARGS=(--motion_dirs "$DATA" --blacklist "$BLACKLIST")
    ;;
  v2list)
    if [ ! -s "$V2LIST" ]; then
      echo "1980719c5268e75649b902ec049d79d4  $BJ27_LIST" | md5sum -c --quiet -
      awk -F/ -v d="$DATA" 'NF { print d "/" $NF }' "$BJ27_LIST" > "$V2LIST.tmp"
      n=$(wc -l < "$V2LIST.tmp")
      miss=$(while IFS= read -r f; do [ -e "$f" ] || echo x; done < "$V2LIST.tmp" | wc -l)
      if [ "$n" -ne 254518 ] || [ "$miss" -ne 0 ]; then
        echo "[v2] $V2LIST.tmp: $n lines, $miss missing (want 254518 / 0)" >&2
        exit 1
      fi
      mv "$V2LIST.tmp" "$V2LIST"
    fi
    DATA_ARGS=(--motion_list "$V2LIST")
    ;;
  *)
    echo "[v2] DATASET must be dir or v2list" >&2
    exit 1
    ;;
esac

# The exact command (launch_bj31_ddp.sh adds: torchrun --standalone --nproc_per_node 8 ... scripts/train.py
#   --log_root /data2/supermimic/logs --cache_dir /data2/supermimic/cache --run_name $RUN --num_threads 24).
NPROC=8 THREADS=24 bash "$REPO/scripts/launch_bj31_ddp.sh" "$RUN" \
  "${DATA_ARGS[@]}" \
  --robot g1 --robot_variant eval+evalcontact \
  --ee_term_z 0.25 \
  --source_weights "$SOURCES" \
  --num_envs 4096 --rollouts 4096 --H 32 --wm_batches 10 --wm_windows 2048 --wm_k 8 \
  --lr 2e-4 --lr_decay_iters 250000 --iterations 320000 --seed 0 \
  --wd_drop 0.2 \
  --num_workers 64 \
  --eval_clip_names "$EVAL_NAMES" --eval_interval 4000 --eval_envs 1024 \
  --ckpt_interval 2000 --log_interval 50 --param_check_interval 1000 \
  --sampler_clamp_returns
