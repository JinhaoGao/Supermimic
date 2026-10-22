#!/usr/bin/env bash
# Stop a run started by launch_bj31_ddp.sh: kill -9 every rank (and their children), then the torchrun agent's
# process group. torchrun starts each rank in its own session, so killing the agent's group alone would leave
# the ranks running as orphans.
# Usage: bash scripts/stop_bj31_ddp.sh <run_name>        env: ROOT (default /data2/supermimic)
set -euo pipefail
RUN="${1:?run name}"
ROOT="${ROOT:-/data2/supermimic}"
PG="$(tr -cd '0-9' < "$ROOT/runs/${RUN}.pgid")"
[ -n "$PG" ] || { echo "[stop] no numeric process group in $ROOT/runs/${RUN}.pgid" >&2; exit 1; }

descendants() {  # numeric PIDs of all descendants of $1
  local c
  for c in $(pgrep -P "$1" || true); do
    echo "$c"
    descendants "$c"
  done
}

PIDS=""
for p in $(pgrep -g "$PG" || true); do
  PIDS="$PIDS $(descendants "$p")"
done
PIDS="$(echo $PIDS | tr ' ' '\n' | grep -E '^[0-9]+$' | sort -un | tr '\n' ' ' || true)"
echo "[stop] run $RUN: process group $PG, rank/child pids: ${PIDS:-none}"
[ -n "${PIDS// /}" ] && kill -9 $PIDS 2>/dev/null || true
kill -9 -- "-$PG" 2>/dev/null || true
sleep 1
LEFT=""
for p in $(pgrep -g "$PG" || true) $PIDS; do
  if [ -d "/proc/$p" ]; then LEFT="$LEFT $p"; fi
done
if [ -n "${LEFT// /}" ]; then
  echo "[stop] still alive: $LEFT" >&2
  exit 1
fi
echo "[stop] done"
