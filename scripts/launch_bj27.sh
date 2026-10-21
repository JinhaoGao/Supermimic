#!/usr/bin/env bash
# Launch a SuperMimic run on bj27 (GPU index via $GPU, default 4) under nohup.
# Usage: bash scripts/launch_bj27.sh <run_name> [extra train.py args...]
set -euo pipefail
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY || true
export PATH="$HOME/.local/bin:$PATH"
GPU="${GPU:-4}"
RUN="${1:?run name}"
shift || true
cd "$HOME/supermimic/mjlab"
mkdir -p "$HOME/supermimic/runs"
LOG="$HOME/supermimic/runs/${RUN}.log"
echo "[launch] GPU=$GPU run=$RUN log=$LOG"
CUDA_VISIBLE_DEVICES="$GPU" nohup uv run python ../supermimic_repro/scripts/train.py \
  --motion_dir "$HOME/supermimic/data/lafan1" \
  --log_root "$HOME/supermimic/logs" \
  --run_name "$RUN" "$@" > "$LOG" 2>&1 &
echo "[launch] pid $!"
