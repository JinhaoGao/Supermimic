#!/usr/bin/env bash
# Pull a SuperMimic G1 checkpoint from the training server into the local unitree_mujoco deploy
# (State_SuperMimic, see deploy/robots/g1_29dof/SUPERMIMIC_DEPLOY_NOTES.md).
#
#   bash scripts/deploy_from_server.sh <run_name> [iteration|latest] [clip.npz ...]
#   e.g. bash scripts/deploy_from_server.sh g1_atlas_v1 latest
#        bash scripts/deploy_from_server.sh g1_atlas_v1 24000 ~/supermimic/data/lafan1/g1__lafan1_walk1_subject1.npz
#
# Steps: export_bundle.py on the server (CPU, mjlab venv) -> scp meta.json + policy.onnx ->
#        export_deploy.py locally (named ONNX under exported/, deploy.yaml, optional reference clips).
# Env: SERVER (tencent_bj27), DEPLOY_DIR (~/Workspace/unitree_mujoco/deploy/robots/g1_29dof), PY (python with numpy).
set -euo pipefail
RUN="${1:?run name, e.g. g1_atlas_v1}"
IT="${2:-latest}"
if [ $# -ge 2 ]; then shift 2; else shift $#; fi
SERVER="${SERVER:-tencent_bj27}"
DEPLOY_DIR="${DEPLOY_DIR:-$HOME/Workspace/unitree_mujoco/deploy/robots/g1_29dof}"
PY="${PY:-python3}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
BUNDLE="$HOME/supermimic/bundle_deploy_${RUN}"

echo "[deploy] exporting $RUN @ $IT on $SERVER ..."
ssh -o BatchMode=yes "$SERVER" bash -s -- "$RUN" "$IT" <<'EOS'
set -euo pipefail
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY || true
export PATH="$HOME/.local/bin:$PATH"
RUN=$1; IT=$2
D=$(ls -d "$HOME"/supermimic/logs/*_"$RUN" | tail -1)
if [ "$IT" = latest ]; then
  CK=$(ls "$D"/model_[0-9]*.pt | awk -F'model_|\\.pt' '{print $(NF-1), $0}' | sort -n | tail -1 | cut -d' ' -f2)
else
  CK="$D/model_$IT.pt"
fi
[ -f "$CK" ] || { echo "no checkpoint $CK" >&2; exit 1; }
echo "[deploy] checkpoint $CK"
cd "$HOME/supermimic/mjlab"
CUDA_VISIBLE_DEVICES= uv run python ../supermimic_repro/scripts/export_bundle.py --ckpt "$CK" \
  --motion "$HOME/supermimic/data/lafan1/g1__lafan1_dance1_subject1.npz" --out "$HOME/supermimic/bundle_deploy_$RUN" --device cpu 2>&1 \
  | grep "^\[export\]"
EOS
mkdir -p "$BUNDLE"
scp -q "$SERVER:supermimic/bundle_deploy_${RUN}/meta.json" "$SERVER:supermimic/bundle_deploy_${RUN}/policy.onnx" "$BUNDLE/"
MOTIONS=()
if [ $# -gt 0 ]; then MOTIONS=(--motions "$@"); fi
"$PY" "$REPO/scripts/export_deploy.py" --bundle "$BUNDLE" --deploy_dir "$DEPLOY_DIR" "${MOTIONS[@]}"
echo "[deploy] done. Rebuild is not needed (policy/config only); restart g1_ctrl after editing policy_file in config.yaml."
