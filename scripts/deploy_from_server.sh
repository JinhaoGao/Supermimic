#!/usr/bin/env bash
# Pull a SuperMimic G1 checkpoint from the training server into the local unitree_mujoco deploy
# (State_SuperMimic, see deploy/robots/g1_29dof/SUPERMIMIC_DEPLOY_NOTES.md).
#
#   bash scripts/deploy_from_server.sh <run_name> [best|latest|<iteration>]     (default: best)
#   e.g. bash scripts/deploy_from_server.sh g1_atlas_v1r best
#   best   = checkpoint with the highest training-time benchmark return (eval.jsonl), the safe default
#   latest = newest checkpoint file; refused if its benchmark return is far below the best (collapsed run)
#
# Steps: export_bundle.py on the server (CPU, mjlab venv) -> scp meta.json + policy.onnx ->
#        export_deploy.py locally (named ONNX under exported/, deploy.yaml). Reference clips need no conversion.
# Env: SERVER (tencent_bj27), DEPLOY_DIR (~/Workspace/unitree_mujoco/deploy/robots/g1_29dof), PY (python with numpy),
#      REPRO (code dir under ~/supermimic on the server, default supermimic_repro; must know the checkpoint's heads, e.g.
#      goal_ref_root runs: a checkout with chain body_pos in export_bundle.py).
set -euo pipefail
RUN="${1:?run name, e.g. g1_atlas_v1}"
IT="${2:-best}"
if [ $# -ge 2 ]; then shift 2; else shift $#; fi
SERVER="${SERVER:-tencent_bj27}"
DEPLOY_DIR="${DEPLOY_DIR:-$HOME/Workspace/unitree_mujoco/deploy/robots/g1_29dof}"
PY="${PY:-python3}"
REPRO="${REPRO:-supermimic_repro}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
BUNDLE="$HOME/supermimic/bundle_deploy_${RUN}"

echo "[deploy] exporting $RUN @ $IT on $SERVER ..."
ssh -o BatchMode=yes "$SERVER" bash -s -- "$RUN" "$IT" "$REPRO" <<'EOS'
set -euo pipefail
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY || true
export PATH="$HOME/.local/bin:$PATH"
RUN=$1; IT=$2; REPRO=$3
D=$(ls -d "$HOME"/supermimic/logs/*_"$RUN" | tail -1)
CK=$(python3 - "$D" "$IT" <<'PY'
import glob, json, os, re, sys
d, sel = sys.argv[1], sys.argv[2]
cks = {int(re.findall(r"model_(\d+)\.pt", p)[0]): p for p in glob.glob(f"{d}/model_[0-9]*.pt")}
ev = {}
if os.path.exists(f"{d}/eval.jsonl"):
    for line in open(f"{d}/eval.jsonl"):
        r = json.loads(line); ev[r["it"]] = r
best = max((it for it in ev if it in cks), key=lambda it: (ev[it]["return_mean"], -ev[it]["failed"]), default=None)
if sel == "best":
    it = best if best is not None else max(cks)
elif sel == "latest":
    it = max(cks)
    if it in ev and best is not None and ev[it]["return_mean"] < ev[best]["return_mean"] - 0.1:
        sys.exit(f"[deploy] latest checkpoint {it} benchmark return {ev[it]['return_mean']:.3f} is far below best {best} "
                 f"({ev[best]['return_mean']:.3f}) - collapsed run? use 'best' or an explicit iteration")
else:
    it = int(sel)
    if it not in cks: sys.exit(f"[deploy] no checkpoint model_{it}.pt in {d}")
info = f"return {ev[it]['return_mean']:.3f}, failed {int(ev[it]['failed'])}/{ev[it].get('clips_done', '?')}" if it in ev else "not evaluated"
print(f"[deploy] {sel} -> iteration {it} ({info}; best evaluated: {best})", file=sys.stderr)
print(cks[it])
PY
)
[ -n "$CK" ] && [ -f "$CK" ] || { echo "[deploy] no checkpoint selected" >&2; exit 1; }
echo "[deploy] checkpoint $CK"
cd "$HOME/supermimic/mjlab"
PYTHONPATH="$HOME/supermimic/$REPRO" CUDA_VISIBLE_DEVICES= uv run python "../$REPRO/scripts/export_bundle.py" --ckpt "$CK" \
  --motion "$HOME/supermimic/data/lafan1/g1__lafan1_dance1_subject1.npz" --out "$HOME/supermimic/bundle_deploy_$RUN" --device cpu 2>&1 \
  | grep "^\[export\]"
EOS
mkdir -p "$BUNDLE"
scp -q "$SERVER:supermimic/bundle_deploy_${RUN}/meta.json" "$SERVER:supermimic/bundle_deploy_${RUN}/policy.onnx" "$BUNDLE/"
"$PY" "$REPO/scripts/export_deploy.py" --bundle "$BUNDLE" --deploy_dir "$DEPLOY_DIR"
echo "[deploy] done. Rebuild is not needed (policy/config only); restart g1_ctrl after editing policy_file in config.yaml."
