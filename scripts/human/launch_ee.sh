#!/usr/bin/env bash
# End-effector world tracking (HumanHeadsCfg.ee_mode, docs/ee_world_tracking_plan.md): eight ablation arms, one GPU each.
#
#   PAIRS=<pair_with_retarget.py out dir> EE=<prepare_ee_ik.py out dir> ROOT=<runs dir> MJLAB=<mjlab uv project> \
#     WARM=<H4 checkpoint> GPUS="0 1 2 3 4 5 6 7" bash scripts/human/launch_ee.sh <tag> [extra train.py args for every arm]
#
# Every arm: the H4 recipe (launch_variants.sh human4) + ee_mode, trained on $PAIRS/human_list.txt with alpha > 0 only on
# the clips of $EE/ee_list.txt (the others stay H4), evaluated at alpha = 1 on 512 clips of ee_list (eval_ee512.txt).
#   e1  defaults (wide + narrow kernels, drift 10 cm / 10 deg, EE termination 0.4 / 0.5 m, ee_style 0.3), from scratch
#   e2  e1 warm-started from the H4 checkpoint $WARM (goal graft: new inputs zero -> starts as H4), ITERS more iterations
#   e3  looser narrow kernels (0.1 m, 0.25 rad)            e4  no wide wrist kernel (narrow only, as Workhorse)
#   e5  ee_style 0.5 (more H4 at alpha = 1)                e6  no drift augmentation
#   e7  no EE termination                                  e8  trained on the ee_list clips only
# ARMS="e1 e3" starts only those (each keeps its GPU slot). The arms share one library cache whose EE fields are built on
# first use: start one arm first when they do not exist yet (ARMS=e1), then the others.
set -euo pipefail
TAG="${1:?usage: launch_ee.sh <tag> [extra train.py args]}"
shift
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PAIRS="${PAIRS:?set PAIRS}"
EE="${EE:?set EE (prepare_ee_ik.py --out_dir)}"
ROOT="${ROOT:-$HOME/supermimic_runs}"
MJLAB="${MJLAB:-$HOME/supermimic_ws/mjlab}"
read -r -a G <<< "${GPUS:-0 1 2 3 4 5 6 7}"
ITERS="${ITERS:-40000}"
H4="w_bone_global_arm=1.0,w_bone_global_leg=1.5,std_bone_leg=0.6,w_foot_lift=1.0,term_foot_lift=0.25"
BASE="$H4,ee_mode=1,ee_clip_list=$EE/ee_list.txt"
[ -f "$EE/eval_ee512.txt" ] || python3 - "$EE" <<'EOF'
import os, random, sys
names = [os.path.basename(l.strip()) for l in open(os.path.join(sys.argv[1], "ee_list.txt")) if l.strip()]
pick = sorted(random.Random(0).sample(names, min(512, len(names))))
open(os.path.join(sys.argv[1], "eval_ee512.txt"), "w").write("\n".join(pick) + "\n")
EOF
mkdir -p "$ROOT/logs" "$ROOT/cache"
COMMON=(--robot g1 --robot_variant eval+evalcontact --H 48 --fixes --sampler_clamp_returns --sampler_cap 50 --sampler_boot_mean
  --num_workers 16 --eval_interval 2000 --eval_clip_names "$EE/eval_ee512.txt" --eval_envs 512 --ckpt_interval 500
  --log_interval 50 --wd_sustain 200 --wd_warmup 1500 --wd_baseline_after_warmup --wd_signal imagined_reward_robust
  --log_root "$ROOT/logs" --cache_dir "$ROOT/cache" --reference human)

ARMS="${ARMS:-e1 e2 e3 e4 e5 e6 e7 e8}"
launch() {  # arm gpu args...
  local arm=$1 gpu=$2
  shift 2
  [[ " $ARMS " == *" $arm "* ]] || return 0
  cd "$MJLAB"
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH="$REPO" setsid nohup uv run python "$REPO/scripts/train.py" --run_name "${TAG}_$arm" \
    "${COMMON[@]}" "$@" > "$ROOT/${TAG}_$arm.out" 2>&1 < /dev/null &
  echo $! > "$ROOT/${TAG}_$arm.pid"
  echo "[launch_ee] $arm on GPU $gpu: pid $! -> $ROOT/${TAG}_$arm.out"
}

L=(--motion_list "$PAIRS/human_list.txt")
launch e1 "${G[0]}" "${L[@]}" --iterations "$ITERS" --human_cfg "$BASE" "$@"
if [[ " $ARMS " == *" e2 "* ]]; then
  WARM="${WARM:?set WARM (H4 checkpoint) for e2}"
  WARM_IT=$(cd "$MJLAB" && uv run python -c "import torch,sys; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['it'])" "$WARM" 2>/dev/null | tail -1)
  launch e2 "${G[1]}" "${L[@]}" --iterations $((WARM_IT + ITERS)) --resume "$WARM" --human_cfg "$BASE" "$@"
fi
launch e3 "${G[2]}" "${L[@]}" --iterations "$ITERS" --human_cfg "$BASE,std_ee_wrist=0.1,std_ee_torso_pos=0.1,std_ee_foot_pos=0.1,std_ee_wrist_rot=0.25,std_ee_torso_rot=0.25" "$@"
launch e4 "${G[3]}" "${L[@]}" --iterations "$ITERS" --human_cfg "$BASE,w_ee_wrist_wide=0" "$@"
launch e5 "${G[4]}" "${L[@]}" --iterations "$ITERS" --human_cfg "$BASE,ee_style=0.5" "$@"
launch e6 "${G[5]}" "${L[@]}" --iterations "$ITERS" --human_cfg "$BASE,drift_xy=0,drift_yaw=0" "$@"
launch e7 "${G[6]}" "${L[@]}" --iterations "$ITERS" --human_cfg "$BASE,term_ee_wrist=0,term_ee_torso=0" "$@"
launch e8 "${G[7]}" --motion_list "$EE/ee_list.txt" --iterations "$ITERS" --human_cfg "$H4,ee_mode=1" "$@"
