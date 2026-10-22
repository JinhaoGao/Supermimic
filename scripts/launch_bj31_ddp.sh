#!/usr/bin/env bash
# Launch ONE SuperMimic model on NPROC GPUs of one machine (torchrun data parallel), detached (setsid nohup).
#
# Usage: bash scripts/launch_bj31_ddp.sh <run_name> [extra train.py args...]
#   env: NPROC   GPUs / ranks (default 8)
#        THREADS CPU threads per rank (OMP/MKL + torch.set_num_threads, default 24)
#        ROOT    run data root (default /data2/supermimic): logs/ (TB, ckpt), cache/, runs/<run>.log
#
# Every rank keeps the per-GPU paper config (4096 envs, 4096 rollouts, LR 2e-4) -> global 8x batch.
# Uses the venv's own torch (cu130) and its bundled NCCL; LD_PRELOAD is cleared on purpose (the machine's
# nccl_2277 is a cu12 build). NCCL_DEBUG=VERSION puts the NCCL version into the log.
# Console log = rank 0 only (runs/<run>.log); every rank's full stdout/stderr: runs/<run>_ranks/.
# Stop: the whole run is one process group (setsid); its numeric id is in runs/<run>.pgid:
#   PG=$(cat /data2/supermimic/runs/<run>.pgid); kill -9 -- -"$PG"   (agent + all ranks + log filter)
set -euo pipefail
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY LD_PRELOAD PYTHONPATH || true
RUN="${1:?run name}"
shift || true
NPROC="${NPROC:-8}"
THREADS="${THREADS:-24}"
ROOT="${ROOT:-/data2/supermimic}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
# Import the supermimic package next to this script, whatever the venv's editable install points to
# (on bj27 / bj31 both are /home/ubuntu/supermimic/supermimic_repro).
export PYTHONPATH="$REPO_DIR"
VENV="${VENV:-/home/ubuntu/supermimic/mjlab/.venv}"
LOG="$ROOT/runs/${RUN}.log"
RANK_LOGS="$ROOT/runs/${RUN}_ranks"
mkdir -p "$ROOT/runs" "$ROOT/logs" "$ROOT/cache" "$RANK_LOGS"
if [ -e "$LOG" ]; then
  echo "[launch] $LOG exists; pick another run name" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1 NCCL_DEBUG=VERSION
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS"

CMD=(
  "$VENV/bin/torchrun" --standalone --nproc_per_node "$NPROC"
  --log-dir "$RANK_LOGS" --redirects 3 --tee 3 --local-ranks-filter 0
  "$REPO_DIR/scripts/train.py"
  --log_root "$ROOT/logs" --cache_dir "$ROOT/cache" --run_name "$RUN" --num_threads "$THREADS"
  "$@"
)
printf -v CMDQ '%q ' "${CMD[@]}"
printf -v LOGQ '%q' "$LOG"
{
  echo "[launch] $(date '+%F %T') host $(hostname) commit $(cat "$REPO_DIR/COMMIT" 2>/dev/null || echo '?')"
  echo "[launch] $CMDQ"
} > "$LOG"
# MuJoCo Warp solver-overflow warnings are filtered out of the console log (SimInterface also silences them).
cd "$REPO_DIR"
setsid nohup bash -c "$CMDQ 2>&1 | grep --line-buffered -v -e 'iterations limit reached' -e 'warn_overflow' >> $LOGQ" < /dev/null > /dev/null 2>&1 &
PG=$!
echo "$PG" > "$ROOT/runs/${RUN}.pgid"
echo "[launch] started run=$RUN nproc=$NPROC threads/rank=$THREADS (process group $PG -> $ROOT/runs/${RUN}.pgid)"
echo "[launch] log: $LOG   per-rank: $RANK_LOGS"
