#!/usr/bin/env bash
# Interactive sim2sim viewer on the local machine (CPU MuJoCo + MuJoCo viewer).
#   bash scripts/view_local.sh [clip substring] [seconds] [slowmo] [extra args...]
#   e.g. bash scripts/view_local.sh jumps1_subject1 20 1.5
# Green translucent robot = reference motion; solid robot = policy in CPU MuJoCo.
# Env: ~/supermimic/mjlab (uv venv), ckpt ~/supermimic/ckpt/model_16000.pt, clips ~/supermimic/data/lafan1.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
CLIP="${1:-dance1_subject1}"
SECONDS_="${2:-15}"
SLOWMO="${3:-1.0}"
shift 3 2>/dev/null || shift $#
CKPT="${CKPT:-$HOME/supermimic/ckpt/model_16000.pt}"
DATA="${DATA:-$HOME/supermimic/data/lafan1}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HOME/supermimic/mjlab"
exec uv run python "$REPO_DIR/scripts/sim2sim_mujoco.py" --ckpt "$CKPT" --motion_dir "$DATA" \
  --clips "$CLIP" --seconds "$SECONDS_" --slowmo "$SLOWMO" --device cpu --viewer "$@"
