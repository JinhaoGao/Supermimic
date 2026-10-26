#!/usr/bin/env bash
# Export a SuperMimic checkpoint on bj27, relay the eval ONNX to bj31 through this machine, upload it to the eval
# site and start a MotionAtlas v4 official run (64 workers, videos of failed / hardest 5+5).
#   bash eval_site/submit_v4.sh <run_name> <iteration> "<site policy name>" [training|actuator]
#   SKIP_EXPORT=1: the ONNX is already on bj31 (only upload + run)
#   REPRO=<dir>: export with ~/supermimic/<dir> on bj27 (PYTHONPATH; default supermimic_repro = the editable install)
#   OFFLINE=1: no site upload; run the private offline v4 evaluation on bj31 instead (evaluator_r18/eval_r18.sh,
#              identical protocol, nothing on the leaderboard)
# The observation profile follows the ONNX goal_layout: q,qd,rot6d -> supermimic_g1_v1 / e2_supermimic_v1,
# q,qd,rot6d,ref_root -> supermimic_g1_v2 / e2_supermimic_v2 (site profiles added 10-26). With SKIP_EXPORT set GOAL_LAYOUT.
# run_name = the bj27 run (logs/*_<run_name>), damping: actuator (kd only, default; G1 eval XML has its own passive
# damping) or training (kd + passive damping of the training model, v1 E2).
set -euo pipefail
RUN=${1:?run}; IT=${2:?iteration}; NAME=${3:?policy name}; DAMP=${4:-actuator}; REPRO=${REPRO:-supermimic_repro}
G1B=pelvis,left_hip_roll_link,left_knee_link,left_ankle_roll_link,right_hip_roll_link,right_knee_link,right_ankle_roll_link,torso_link,left_shoulder_roll_link,left_elbow_link,left_wrist_yaw_link,right_shoulder_roll_link,right_elbow_link,right_wrist_yaw_link
E2B=base_link,waist_pitch_link,l_leg_hip_pitch_link,r_leg_hip_pitch_link,l_leg_hip_yaw_link,r_leg_hip_yaw_link,l_leg_knee_link,r_leg_knee_link,l_arm_shoulder_roll_link,r_arm_shoulder_roll_link,l_leg_ankle_roll_link,r_leg_ankle_roll_link,l_arm_elbow_link,r_arm_elbow_link
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20"
retry() { for i in 1 2 3 4; do "$@" && return 0; sleep 5; done; return 1; }
OUT=supermimic_${RUN}_${IT}.onnx
if [ -z "${SKIP_EXPORT:-}" ]; then
EXP=$(retry $SSH tencent_bj27 "bash -s -- $(printf '%q ' "$RUN" "$IT" "$DAMP" "$G1B" "$E2B" "$OUT" "$REPRO")" <<'EOS'
set -euo pipefail
RUN=$1; IT=$2; DAMP=$3; G1B=$4; E2B=$5; OUT=$6; REPRO=$7
[ "$REPRO" = supermimic_repro ] || export PYTHONPATH="$HOME/supermimic/$REPRO"
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY || true
export PATH="$HOME/.local/bin:$PATH"
D=$(ls -d "$HOME"/supermimic/logs/*_"$RUN" | tail -1)
CK="$D/model_$IT.pt"; [ -f "$CK" ] || { echo "no $CK" >&2; exit 1; }
ROBOT=$(python3 -c "import json;print(json.load(open('$D/train_cfg.json'))['robot'])")
M=$(head -1 "$D/motion_files.txt")
B="$HOME/supermimic/bundle_eval_${RUN}_$IT"
cd "$HOME/supermimic/mjlab"
CUDA_VISIBLE_DEVICES= nice -n 5 uv run python ../$REPRO/scripts/export_bundle.py --ckpt "$CK" --motion "$M" --out "$B" --device cpu 2>&1 | grep "^\[export\]"
BODIES=$G1B; [ "$ROBOT" = e2 ] && BODIES=$E2B
mkdir -p "$HOME/supermimic/eval_onnx"
.venv/bin/python ../$REPRO/scripts/export_eval_onnx.py --bundle "$B" --out "$HOME/supermimic/eval_onnx/$OUT" --metric_bodies "$BODIES" --damping "$DAMP" | tail -2
echo "ROBOT=$ROBOT"
.venv/bin/python -c "import onnx, sys; m = {p.key: p.value for p in onnx.load(sys.argv[1], load_external_data=False).metadata_props}; print('GOAL_LAYOUT=' + m.get('goal_layout', 'q,qd,rot6d'))" "$HOME/supermimic/eval_onnx/$OUT"
EOS
)
echo "$EXP"
ROBOT=$(echo "$EXP" | sed -n "s/^ROBOT=//p" | tail -1)
GOAL_LAYOUT=$(echo "$EXP" | sed -n "s/^GOAL_LAYOUT=//p" | tail -1)
[ -n "$ROBOT" ] && [ -n "$GOAL_LAYOUT" ] || { echo "[submit] export failed" >&2; exit 1; }
echo "[submit] relaying $OUT (robot $ROBOT) bj27 -> bj31"
retry bash -c "$SSH tencent_bj27 'cat ~/supermimic/eval_onnx/$OUT' < /dev/null | $SSH tencent_bj31 'mkdir -p ~/work/supermimic_eval_onnx && cat > ~/work/supermimic_eval_onnx/$OUT'"
else
ROBOT=${ROBOT:?set ROBOT=g1|e2 with SKIP_EXPORT}
GOAL_LAYOUT=${GOAL_LAYOUT:-q,qd,rot6d}
fi
if [ -n "${OFFLINE:-}" ]; then
  retry $SSH tencent_bj31 "cd ~/work/supermimic_eval_onnx && mkdir -p evaluator_r18/runs && (setsid bash evaluator_r18/eval_r18.sh $OUT evaluator_r18/runs/${RUN}_$IT 48 > evaluator_r18/runs/${RUN}_$IT.out 2>&1 < /dev/null &)"
  echo "[submit] offline evaluation started: bj31 ~/work/supermimic_eval_onnx/evaluator_r18/runs/${RUN}_$IT ($NAME)"
  exit 0
fi
case "$GOAL_LAYOUT" in q,qd,rot6d) V=v1 ;; q,qd,rot6d,ref_root) V=v2 ;; *) echo "[submit] no site profile for goal_layout $GOAL_LAYOUT" >&2; exit 1 ;; esac
if [ "$ROBOT" = e2 ]; then OBS=e2_supermimic_$V; DS=e2_motionatlas_v4; BM=noetix_e2_motion_official_v1; else OBS=supermimic_g1_$V; DS=g1_motionatlas_v4; BM=kimodo_g1_motion_official_v3; fi
echo "[submit] goal_layout $GOAL_LAYOUT -> observation profile $OBS"
retry $SSH tencent_bj31 "bash -s -- $(printf '%q ' "$OUT" "$NAME" "$OBS" "$DS" "$BM")" <<'EOS'
set -euo pipefail
OUT=$1; NAME=$2; OBS=$3; DS=$4; BM=$5
cd ~/work/supermimic_eval_onnx
PID=$(curl -s -m 200 -F "file=@$OUT" -F "name=$NAME" -F "observation_spec_id=$OBS" -F "robot_variant=standard" http://127.0.0.1:8010/api/v1/policies/upload | python3 -c "
import json,sys; d=json.load(sys.stdin); md=d.get('metadata') or {}
assert d.get('status') == 'compatible', d
print(d['id'])")
RID=$(curl -s -m 60 -H "Content-Type: application/json" -d "{\"policy_id\": \"$PID\", \"dataset_id\": \"$DS\", \"benchmark_id\": \"$BM\", \"mode\": \"official\", \"overrides\": {\"workers\": 64, \"num_envs\": 1, \"video_mode\": \"failed_and_hardest\", \"video_failed_motions\": 5, \"video_hard_motions\": 5}}" http://127.0.0.1:8010/api/v1/runs | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])")
echo "[submit] policy $PID run $RID ($NAME)"
EOS
