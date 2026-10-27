#!/usr/bin/env bash
# A/B on one machine, one GPU per arm, same clips (pair_with_retarget.py output), same recipe:
#   robot: baseline SuperMimic, retargeted G1 clips, BeyondMimic rewards (--reference robot == main, bitwise)
#   human: raw SMPL clips, human goal, bone-direction rewards (--reference human)
#
#   PAIRS=~/datasets/paired_amass ROOT=~/supermimic_runs MJLAB=~/supermimic_ws/mjlab bash scripts/human/launch_pair.sh <tag> [extra train.py args]
#
# Env: PAIRS (dir with robot_list.txt / human_list.txt), ROOT (logs + caches), MJLAB (the mjlab uv project), GPU_ROBOT /
# GPU_HUMAN (default 0 / 1), ITERS (default 40000), HUMAN_CFG (--human_cfg overrides, default none).
# Each arm runs in the background with nohup; console logs in $ROOT/<tag>_{robot,human}.out, runs in $ROOT/logs.
# Stop: kill the pids printed below (also in $ROOT/<tag>_{robot,human}.pid).
set -euo pipefail
TAG="${1:?usage: launch_pair.sh <tag> [extra train.py args]}"
shift
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PAIRS="${PAIRS:?set PAIRS to the pair_with_retarget.py output dir}"
ROOT="${ROOT:-$HOME/supermimic_runs}"
MJLAB="${MJLAB:-$HOME/supermimic_ws/mjlab}"
GPU_ROBOT="${GPU_ROBOT:-0}"
GPU_HUMAN="${GPU_HUMAN:-1}"
ITERS="${ITERS:-40000}"
mkdir -p "$ROOT/logs" "$ROOT/cache"

# One recipe for both arms (single-GPU line of the README, without the MotionAtlas source / keyword priors).
COMMON=(--robot g1 --robot_variant eval+evalcontact --H 48 --fixes --sampler_clamp_returns --num_workers 16
  --iterations "$ITERS" --eval_interval 2000 --eval_clips 512 --eval_envs 512 --ckpt_interval 2000 --log_interval 50
  --wd_sustain 200 --wd_warmup 1500 --wd_baseline_after_warmup --wd_signal imagined_reward_robust
  --log_root "$ROOT/logs" --cache_dir "$ROOT/cache")

launch() {  # arm gpu list args...
  local arm=$1 gpu=$2 list=$3
  shift 3
  cd "$MJLAB"
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH="$REPO" nohup uv run python "$REPO/scripts/train.py" --motion_list "$list" \
    --run_name "${TAG}_$arm" "${COMMON[@]}" "$@" > "$ROOT/${TAG}_$arm.out" 2>&1 &
  echo $! > "$ROOT/${TAG}_$arm.pid"
  echo "[launch_pair] $arm on GPU $gpu: pid $! -> $ROOT/${TAG}_$arm.out"
}

launch robot "$GPU_ROBOT" "$PAIRS/robot_list.txt" --reference robot --ee_term_z 0.25 "$@"
launch human "$GPU_HUMAN" "$PAIRS/human_list.txt" --reference human ${HUMAN_CFG:+--human_cfg "$HUMAN_CFG"} "$@"
echo "[launch_pair] compare with: uv run python $REPO/scripts/human/eval_paired.py --ckpt <run>/model_N.pt --pairs $PAIRS/pairs.tsv --subset $PAIRS/ambig_top.txt"
