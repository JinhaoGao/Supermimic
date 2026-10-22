#!/bin/bash
# Background contact diagnosis (nice 10, 16 workers per variant, sequential). Run from the NOetix_GMT_eval root.
set -u
P=~/miniconda3/envs/noetix_eval/bin/python
D=/tmp/claude_diag
mkdir -p $D
$P ~/work/supermimic_eval_onnx/diag_contacts.py 2> $D/models.log
yamlget() { python3 -c "import yaml;print(yaml.safe_load(open('$1'))['model_path'])"; }
find $PWD/data/e2_motionatlas_v4 -name "*.npz" | sort | awk 'NR%36==1' | head -96 > $D/e2_list.txt
find $PWD/data/g1_motionatlas_v4 -name "*.npz" | sort | awk 'NR%32==1' | head -96 > $D/g1_list.txt
for robot in e2 g1; do
  if [ $robot = e2 ]; then OBS=config/observations/e2_supermimic_v1.yaml; POL=~/work/supermimic_eval_onnx/supermimic_e2_atlas_v1r_68000.onnx; TB=l_leg_ankle_roll_link,r_leg_ankle_roll_link,l_arm_elbow_link,r_arm_elbow_link; OFF=$(yamlget config/robots/e2_23dof_v1.yaml);
  else OBS=config/observations/supermimic_g1_v1.yaml; POL=~/work/supermimic_eval_onnx/supermimic_g1_atlas_v1_32000.onnx; TB=left_ankle_roll_link,right_ankle_roll_link,left_wrist_yaw_link,right_wrist_yaw_link; OFF=$(yamlget config/robots/g1_29dof_v1.yaml); fi
  for v in official noself fric1 noself_fric1; do
    MODEL=$OFF; [ $v != official ] && MODEL=$D/models/${robot}_$v.xml
    nice -n 10 timeout 3000 $P vendor/Noetix_GMT/scripts/evaluate_mujoco.py --policy-type supermimic --obs-config $OBS --motion-path $D/${robot}_list.txt --model "$MODEL" --motion-joint-order policy --eval-unit motion --bin-duration 1.0 --sim-dt 0.001 --decimation 20 --device cpu --workers 16 --num-envs 1 --root-pos-threshold 0.25 --root-height-threshold 0.25 --root-ori-threshold 1.0 --body-pos-threshold 0.25 --root-pos-mode z --body-pos-mode z --termination-bodies $TB --video-mode none --report-dir $D/${robot}_$v --report-prefix evaluation --policy $POL > $D/${robot}_$v.log 2>&1
    echo "$(date +%T) $robot $v: $(grep -E 'Success rate' $D/${robot}_$v.log)" >> $D/progress.txt
  done
done
echo DONE >> $D/progress.txt
