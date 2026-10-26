#!/usr/bin/env bash
# Launch ONE SuperMimic model on NPROC GPUs of one machine (torchrun data parallel), detached (setsid nohup).
#
# Usage: bash scripts/launch_bj31_ddp.sh <run_name> [extra train.py args...]
#   env: NPROC   GPUs / ranks (default 8; with CUDA_VISIBLE_DEVICES set: the number of GPUs listed there)
#        THREADS CPU threads per rank (OMP/MKL + torch.set_num_threads, default 24)
#        ROOT    run data root (default /data2/supermimic): logs/ (TB, ckpt), cache/, runs/<run>.log
#        CUDA_VISIBLE_DEVICES  GPU subset of this job, e.g. 4,5,6,7 (ranks use cuda:0..NPROC-1 of it); unset = all
#        PORT    rendezvous port of this job (aliases RDZV_PORT, MASTER_PORT): set -> torchrun --nnodes 1
#                --rdzv-backend c10d --rdzv-endpoint localhost:$PORT --rdzv-id <run> --rdzv-conf is_host=1 instead of
#                --standalone. Every job on one host needs its OWN PORT (and its own GPUs) to run several at once, e.g.
#                two 4-GPU jobs:
#                  CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29521 NPROC=4 bash scripts/launch_bj31_ddp.sh runA ...
#                  CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29522 NPROC=4 bash scripts/launch_bj31_ddp.sh runB ...
#                is_host=1: this agent must host the rendezvous store on PORT; if the port is taken the job exits at
#                once (RendezvousConnectionError / EADDRINUSE). Without it torch's c10d backend silently falls back
#                to joining the store already on that port as a client (c10d_rendezvous_backend._create_tcp_store),
#                which couples the two jobs: stopping one kills the other. (--standalone binds localhost:0, an
#                OS-chosen port.) Neither set = the command as before (--standalone, 8 ranks).
#        DRY_RUN=1  print the torchrun command and exit (nothing started, no log written)
#
# Every rank keeps the per-GPU paper config (4096 envs, 4096 rollouts, LR 2e-4) -> global 8x batch.
# Uses the venv's own torch (cu130) and its bundled NCCL; LD_PRELOAD is cleared on purpose (the machine's
# nccl_2277 is a cu12 build). NCCL_DEBUG=VERSION puts the NCCL version into the log.
# Console log = rank 0 only (runs/<run>.log); every rank's full stdout/stderr: runs/<run>_ranks/.
# Stop: bash scripts/stop_bj31_ddp.sh <run>   (kill -9 every rank + children, then the launcher's process group)
#   The launcher (bash + torchrun agent + log filter) is one process group, numeric id in runs/<run>.pgid, but
#   torchrun starts every rank in its OWN session/process group (torch elastic start_new_session=True), so
#   `kill -9 -- -"$PG"` alone reaches only the agent. The ranks set PR_SET_PDEATHSIG=SIGKILL (scripts/train.py),
#   so they still die with the agent; the stop script does not rely on that.
set -euo pipefail
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY LD_PRELOAD PYTHONPATH || true
RUN="${1:?run name}"
shift || true
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then  # GPU subset: one rank per listed GPU unless NPROC says fewer
  IFS=',' read -r -a GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
  NPROC="${NPROC:-${#GPU_LIST[@]}}"
  if [ "$NPROC" -gt "${#GPU_LIST[@]}" ]; then
    echo "[launch] NPROC=$NPROC > ${#GPU_LIST[@]} GPUs in CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
    exit 1
  fi
  export CUDA_VISIBLE_DEVICES
else
  NPROC="${NPROC:-8}"
fi
PORT="${PORT:-${RDZV_PORT:-${MASTER_PORT:-}}}"
if [ -n "$PORT" ]; then
  if ! [[ "$PORT" =~ ^[0-9]+$ ]] || [ "$PORT" -lt 1024 ] || [ "$PORT" -gt 65535 ]; then
    echo "[launch] PORT=$PORT is not a port number in 1024..65535" >&2
    exit 1
  fi
  RDZV=(--nnodes 1 --rdzv-backend c10d --rdzv-endpoint "localhost:$PORT" --rdzv-id "$RUN" --rdzv-conf is_host=1)
else
  RDZV=(--standalone)
fi
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
  "$VENV/bin/torchrun" "${RDZV[@]}" --nproc_per_node "$NPROC"
  --log-dir "$RANK_LOGS" --redirects 3 --tee 3 --local-ranks-filter 0
  "$REPO_DIR/scripts/train.py"
  --log_root "$ROOT/logs" --cache_dir "$ROOT/cache" --run_name "$RUN" --num_threads "$THREADS"
  "$@"
)
printf -v CMDQ '%q ' "${CMD[@]}"
printf -v LOGQ '%q' "$LOG"
if [ -n "${DRY_RUN:-}" ]; then
  echo "[launch] dry run: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>} NPROC=$NPROC PORT=${PORT:-<standalone>}"
  echo "[launch] $CMDQ"
  exit 0
fi
{
  echo "[launch] $(date '+%F %T') host $(hostname) commit $(cat "$REPO_DIR/COMMIT" 2>/dev/null || echo '?')"
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] || [ -n "$PORT" ]; then
    echo "[launch] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all>} rendezvous $([ -n "$PORT" ] && echo "localhost:$PORT" || echo standalone)"
  fi
  echo "[launch] $CMDQ"
} > "$LOG"
# MuJoCo Warp solver-overflow warnings are filtered out of the console log (SimInterface also silences them).
cd "$REPO_DIR"
setsid nohup bash -c "$CMDQ 2>&1 | grep --line-buffered -v -e 'iterations limit reached' -e 'warn_overflow' >> $LOGQ" < /dev/null > /dev/null 2>&1 &
PG=$!
echo "$PG" > "$ROOT/runs/${RUN}.pgid"
echo "[launch] started run=$RUN nproc=$NPROC threads/rank=$THREADS${CUDA_VISIBLE_DEVICES:+ gpus=$CUDA_VISIBLE_DEVICES}${PORT:+ port=$PORT} (process group $PG -> $ROOT/runs/${RUN}.pgid)"
echo "[launch] log: $LOG   per-rank: $RANK_LOGS"
