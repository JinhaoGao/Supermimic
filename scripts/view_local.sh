#!/usr/bin/env bash
# Interactive sim2sim on the local machine: plain MuJoCo + onnxruntime policy.
#   bash scripts/view_local.sh [clip substring] [seconds] [slowmo] [extra play_mujoco.py args...]
#   e.g. bash scripts/view_local.sh jumps1_subject1 20 1.5
#        bash scripts/view_local.sh dance1_subject1 30 1 --start_s 40
# Solid robot = policy (ONNX) in CPU MuJoCo, green translucent robot = reference motion.
# Bundle (scene.mjb + policy.onnx + meta.json) from scripts/export_bundle.py.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
CLIP="${1:-dance1_subject1}"
SECONDS_="${2:-15}"
SLOWMO="${3:-1.0}"
if [ $# -ge 3 ]; then shift 3; else shift $#; fi
BUNDLE="${BUNDLE:-$HOME/supermimic/bundle_16000}"
DATA="${DATA:-$HOME/supermimic/data/lafan1}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MOTION=$(ls "$DATA"/*"$CLIP"*.npz | head -1)
echo "[view] bundle=$BUNDLE motion=$MOTION"
cd "$HOME/supermimic/mjlab"
exec uv run python "$REPO_DIR/scripts/play_mujoco.py" --bundle "$BUNDLE" --motion "$MOTION" \
  --seconds "$SECONDS_" --slowmo "$SLOWMO" --viewer "$@"
