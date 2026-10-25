#!/usr/bin/env bash
# Installed on bj31 as ~/work/supermimic_eval_onnx/evaluator_r18/eval_r18.sh next to evaluator_r18/scripts (a copy of the
# site's vendor/Noetix_GMT/scripts with this repo's eval_supermimic.py and its sha256 pin) and supermimic_g1_v2r18.yaml.
# Private offline MotionAtlas v4 evaluation with the site's official protocol (benchmark kimodo_g1_motion_official_v3,
# dataset g1_motionatlas_v4: same model / motions / thresholds as a site run) but with the round-18 evaluator copy
# (evaluator_r18/scripts: eval_supermimic.py also builds goal_layout "q,qd,rot6d,ref_root" goals). Does not touch the site.
#   bash eval_r18.sh <onnx> <out_dir> [workers=48] [motion_list=the 3134 v4 motions]
set -euo pipefail
ONNX=$(readlink -f "$1"); OUT=$(readlink -m "$2"); W=${3:-48}
ML=$(readlink -f "${4:-/home/ubuntu/work/NOetix_GMT_eval/storage/runs/f749f4aaf9ac4b7e8772d514453e2120/motions.txt}")
E=/home/ubuntu/work/supermimic_eval_onnx/evaluator_r18
P=/home/ubuntu/miniconda3/envs/noetix_eval/bin/python
LAYOUT=$($P -c "import onnx,sys;m=onnx.load(sys.argv[1]);print({p.key:p.value for p in m.metadata_props}.get('goal_layout','q,qd,rot6d'))" "$ONNX")
if [ "$LAYOUT" = "q,qd,rot6d,ref_root" ]; then OBS=$E/supermimic_g1_v2r18.yaml; else OBS=/home/ubuntu/work/NOetix_GMT_eval/config/observations/supermimic_g1_v1.yaml; fi
mkdir -p "$OUT"
echo "[eval_r18] $ONNX goal_layout=$LAYOUT obs=$OBS motions=$(wc -l < $ML) workers=$W" | tee "$OUT/cmd.txt"
cd /home/ubuntu/work/NOetix_GMT_eval
nice -n 5 $P $E/scripts/evaluate_mujoco.py --policy-type supermimic --obs-config "$OBS" --motion-path "$ML" \
  --model /home/ubuntu/work/unitree_mujoco_noetix_g1_gmt/unitree_robots/g1/scene_29dof.xml --motion-joint-order policy \
  --eval-unit motion --bin-duration 1.0 --sim-dt 0.001 --decimation 20 --device cpu --workers "$W" --num-envs 1 \
  --root-pos-threshold 0.25 --root-height-threshold 0.25 --root-ori-threshold 1.0 --body-pos-threshold 0.25 \
  --root-pos-mode z --body-pos-mode z \
  --termination-bodies left_ankle_roll_link,right_ankle_roll_link,left_wrist_yaw_link,right_wrist_yaw_link \
  --video-mode none --report-dir "$OUT" --report-prefix evaluation --policy "$ONNX" > "$OUT/evaluation.log" 2>&1
python3 -c "import json,sys;d=json.load(open(sys.argv[1]));print('[eval_r18] success', d['success_rate'], 'progress', d['progress_rate'], 'n', d['num_segments'])" "$OUT/evaluation_summary.json" | tee -a "$OUT/cmd.txt"
