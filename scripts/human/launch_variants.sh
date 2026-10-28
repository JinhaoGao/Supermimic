#!/usr/bin/env bash
# Baseline + three human-reference reward variants on one machine, one GPU each, same clips and recipe.
#
#   PAIRS=<pair_with_retarget.py out dir> ROOT=<runs dir> MJLAB=<mjlab uv project> GPUS="0 1 4 5" \
#     bash scripts/human/launch_variants.sh <tag> [extra train.py args for every arm]
#
# Arms (GPUS in this order):
#   robot   original SuperMimic on the retargeted clips (robot reference)
#   human1  control: the ab2 reward (limb global terms on)                     -- isolates the variants below
#   human2  legs: leg kernel 0.35 -> 0.6, global leg weight 0.5 -> 1.5, foot-lift term 1.0
#   human3  legs + motion: human2 and root velocity 1 -> 2, heading 0.5 -> 1, yaw rate 0.5 -> 1
# Override any arm's --human_cfg with HUMAN1_CFG / HUMAN2_CFG / HUMAN3_CFG. Logs: $ROOT/<tag>_<arm>.out, pids .pid.
# ARMS="human2 human3" starts only those (each keeps its GPU slot). The human arms share one library cache: start one
# first when the cache does not exist yet (concurrent builds race), e.g. ARMS="robot human1", then the others.
set -euo pipefail
TAG="${1:?usage: launch_variants.sh <tag> [extra train.py args]}"
shift
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PAIRS="${PAIRS:?set PAIRS}"
ROOT="${ROOT:-$HOME/supermimic_runs}"
MJLAB="${MJLAB:-$HOME/supermimic_ws/mjlab}"
read -r -a G <<< "${GPUS:-0 1 2 3}"
ITERS="${ITERS:-40000}"
H1="${HUMAN1_CFG:-w_bone_global_arm=1.0,w_bone_global_leg=0.5}"
H2="${HUMAN2_CFG:-w_bone_global_arm=1.0,w_bone_global_leg=1.5,std_bone_leg=0.6,w_foot_lift=1.0}"
H3="${HUMAN3_CFG:-w_bone_global_arm=1.0,w_bone_global_leg=1.5,std_bone_leg=0.6,w_foot_lift=1.0,w_root_vel=2.0,w_heading=1.0,w_yaw_rate=1.0}"
mkdir -p "$ROOT/logs" "$ROOT/cache"
COMMON=(--robot g1 --robot_variant eval+evalcontact --H 48 --fixes --sampler_clamp_returns --sampler_cap 50 --sampler_boot_mean
  --num_workers 16 --iterations "$ITERS" --eval_interval 2000 --eval_clips 512 --eval_envs 512 --ckpt_interval 500
  --log_interval 50 --wd_sustain 200 --wd_warmup 1500 --wd_baseline_after_warmup --wd_signal imagined_reward_robust
  --log_root "$ROOT/logs" --cache_dir "$ROOT/cache")

ARMS="${ARMS:-robot human1 human2 human3}"
launch() {  # arm gpu args...
  local arm=$1 gpu=$2
  shift 2
  [[ " $ARMS " == *" $arm "* ]] || return 0
  cd "$MJLAB"
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH="$REPO" setsid nohup uv run python "$REPO/scripts/train.py" --run_name "${TAG}_$arm" \
    "${COMMON[@]}" "$@" > "$ROOT/${TAG}_$arm.out" 2>&1 < /dev/null &
  echo $! > "$ROOT/${TAG}_$arm.pid"
  echo "[launch_variants] $arm on GPU $gpu: pid $! -> $ROOT/${TAG}_$arm.out"
}

launch robot "${G[0]}" --motion_list "$PAIRS/robot_list.txt" --reference robot --ee_term_z 0.25 "$@"
launch human1 "${G[1]}" --motion_list "$PAIRS/human_list.txt" --reference human --human_cfg "$H1" "$@"
launch human2 "${G[2]}" --motion_list "$PAIRS/human_list.txt" --reference human --human_cfg "$H2" "$@"
launch human3 "${G[3]}" --motion_list "$PAIRS/human_list.txt" --reference human --human_cfg "$H3" "$@"
