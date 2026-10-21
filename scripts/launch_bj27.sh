#!/usr/bin/env bash
# Launch a SuperMimic run on bj27 (GPU index via $GPU, default 4) under nohup.
# Usage: bash scripts/launch_bj27.sh <run_name> [extra train.py args...]
set -euo pipefail
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY || true
export PATH="$HOME/.local/bin:$PATH"
GPU="${GPU:-4}"
RUN="${1:?run name}"
shift || true
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HOME/supermimic/mjlab"
mkdir -p "$HOME/supermimic/runs"
LOG="$HOME/supermimic/runs/${RUN}.log"
echo "[launch] GPU=$GPU run=$RUN log=$LOG"
# MuJoCo Warp solver-overflow warnings are filtered out of the log (they spam every step).
CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 nohup uv run python "$REPO_DIR/scripts/train.py" \
  --log_root "$HOME/supermimic/logs" --cache_dir "$HOME/supermimic/data/cache" \
  --run_name "$RUN" "$@" 2>&1 | grep --line-buffered -v "iterations limit reached\|warn_overflow" > "$LOG" &
echo "[launch] started (log: $LOG)"
