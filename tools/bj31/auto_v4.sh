#!/usr/bin/env bash
# Auto v4: every checkpoint of RUN whose iteration is a multiple of STEP gets exported (CPU, export.sh) and sent
# to the eval site (submit.sh, MotionAtlas v4 official). One at a time; submitted iterations are listed in $DONE.
#   RUN=<run name> NAME="<policy name prefix>" [STEP=4000 FROM=<first it>] nohup bash auto_v4.sh > auto_v4_<run>.log 2>&1 &
# Stops by itself when the training process is gone and no checkpoint is left to submit.
set -uo pipefail
RUN=${RUN:?run name}; NAME=${NAME:?policy name prefix}; STEP=${STEP:-4000}; FROM=${FROM:-0}
E=/data2/supermimic/eval_export
START=$(date +%s); GRACE=${GRACE:-1800}  # 训练还没起来时(启动错开/建缓存)不退出
DONE=$E/auto_v4_${RUN}.done
touch "$DONE"
while true; do
  RUN_DIR=$(ls -d /data2/supermimic/logs/*_"$RUN" 2>/dev/null | tail -1)
  todo=""
  if [ -n "$RUN_DIR" ]; then
    for ck in "$RUN_DIR"/model_*.pt; do
      [ -e "$ck" ] || continue
      it=$(basename "$ck" .pt); it=${it#model_}
      case "$it" in ""|*[!0-9]*) continue;; esac  # model_best_eval.pt / model_final.pt
      [ $((it % STEP)) -eq 0 ] && [ "$it" -ge "$FROM" ] && ! grep -qx "$it" "$DONE" && todo="$todo $it"
    done
  fi
  it=$(echo $todo | tr ' ' '\n' | sort -n | head -1)
  if [ -n "$it" ]; then
    echo "[auto_v4] $(date '+%F %T') $RUN model_$it"
    if RUN_DIR="$RUN_DIR" TAG="$RUN" bash $E/scripts/export.sh "$it" \
      && bash $E/scripts/submit.sh "$E/onnx/supermimic_${RUN}_$it.onnx" "$NAME model_$it (bj31 $RUN $(date +%m-%d))"; then
      echo "$it" >> "$DONE"
    else
      echo "[auto_v4] $(date '+%F %T') model_$it FAILED (retry in 10 min)"; sleep 600
    fi
    continue
  fi
  if ! pgrep -f "[s]cripts/train.py .*--run_name $RUN( |\$)" > /dev/null; then
    if [ -z "$RUN_DIR" ] && [ $(( $(date +%s) - START )) -lt "$GRACE" ]; then sleep 60; continue; fi
    echo "[auto_v4] $(date '+%F %T') training gone, nothing left -> exit"; exit 0
  fi
  sleep 300
done
