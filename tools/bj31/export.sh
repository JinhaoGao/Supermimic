#!/usr/bin/env bash
# CPU-only export of one g1_full_ee_ddp8 checkpoint for the eval site (bj31). Read-only on the training dir.
#   [RUN_DIR=<log dir> TAG=<name> REPO=<code dir> XS=<export script dir>] bash /data2/supermimic/eval_export/scripts/export.sh <iteration>
#   v2 runs (robot_variant in the ckpt): REPO=~/supermimic/supermimic_repro_v2 XS=\$REPO/scripts (master's export_bundle builds the variant)
set -euo pipefail
IT=${1:?iteration}
RUN_DIR=${RUN_DIR:-/data2/supermimic/logs/2026-10-22_17-06-52_g1_full_ee_ddp8}
TAG=${TAG:-g1_full_ee_ddp8}
E=/data2/supermimic/eval_export
CK=$RUN_DIR/model_$IT.pt
[ -f "$CK" ] || { echo "no $CK" >&2; exit 1; }
# torch.save writes in place: wait until the file is >= 90 s old and its size is stable
for i in $(seq 1 60); do
  s1=$(stat -c %s "$CK"); age=$(( $(date +%s) - $(stat -c %Y "$CK") )); sleep 5; s2=$(stat -c %s "$CK")
  [ "$s1" = "$s2" ] && [ "$age" -ge 90 ] && break
done
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY LD_PRELOAD || true
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
REPO=${REPO:-/home/ubuntu/supermimic/supermimic_repro}; XS=${XS:-$E/scripts}
export PYTHONPATH=$REPO PYTHONUNBUFFERED=1
export WARP_CACHE_PATH=$E/warp_cache TORCHINDUCTOR_CACHE_DIR=$E/inductor_cache TRITON_CACHE_DIR=$E/triton_cache
export SM_EXPORT_CACHE=$E/motion_cache MPLCONFIGDIR=$E/mpl
mkdir -p $E/bundles $E/onnx $E/motion_cache $E/warp_cache
PY=/home/ubuntu/supermimic/mjlab/.venv/bin/python
G1B=pelvis,left_hip_roll_link,left_knee_link,left_ankle_roll_link,right_hip_roll_link,right_knee_link,right_ankle_roll_link,torso_link,left_shoulder_roll_link,left_elbow_link,left_wrist_yaw_link,right_shoulder_roll_link,right_elbow_link,right_wrist_yaw_link
M=$(head -1 $RUN_DIR/motion_files.txt)
B=$E/bundles/${TAG}_$IT
OUT=$E/onnx/supermimic_${TAG}_$IT.onnx
echo "[exp] $(date '+%F %T') ckpt $CK ($(stat -c %s "$CK") B) motion $M"
cd $E
nice -n 19 ionice -c 3 $PY $XS/export_bundle.py --ckpt "$CK" --motion "$M" --out "$B" --device cpu 2>&1 | grep -v -e "iterations limit" -e warn_overflow
nice -n 19 $PY $XS/export_eval_onnx.py --bundle "$B" --out "$OUT" --metric_bodies "$G1B" --damping actuator
nice -n 19 $PY -c "import json;m=json.load(open('$B/meta.json'));print('[exp] passive damping max',max(m['joint_passive_damping']),'robot',m['robot'],'variant',repr(m.get('robot_variant')),'iteration',m['iteration'])"
nice -n 19 $PY $E/scripts/check_onnx_vs_torch.py "$CK" "$OUT" 200
sha256sum "$OUT"
echo "[exp] done $(date '+%F %T') -> $OUT"
