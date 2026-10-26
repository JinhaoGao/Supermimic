#!/usr/bin/env bash
# Upload one exported ONNX to the eval site and start a MotionAtlas v4 official run (same settings as run d7027de5).
# (Copy of bj31 /data2/supermimic/eval_export/scripts/submit.sh, the bj31 8-GPU line's; the repo-wide tool is
# eval_site/submit_v4.sh.)
#   bash submit.sh <onnx> "<policy name>"
set -euo pipefail
OUT=${1:?onnx}; NAME=${2:?name}
# The site profile follows the ONNX goal layout (10-26): q,qd,rot6d -> supermimic_g1_v1, + ref_root -> supermimic_g1_v2
# (a v1 profile for a ref_root ONNX is "incompatible" and auto_v4 would retry it forever). Reads the metadata with a
# python that has onnx (bj31: the mjlab venv; the system python3 has none).
PY=${PY:-/home/ubuntu/supermimic/mjlab/.venv/bin/python}
LAYOUT=$("$PY" -c "import onnx, sys; m = {p.key: p.value for p in onnx.load(sys.argv[1], load_external_data=False).metadata_props}; print(m.get('goal_layout', 'q,qd,rot6d'))" "$OUT")
case "$LAYOUT" in
  q,qd,rot6d) OBS=supermimic_g1_v1 ;;
  q,qd,rot6d,ref_root) OBS=supermimic_g1_v2 ;;
  *) echo "[submit] no site profile for goal_layout $LAYOUT" >&2; exit 1 ;;
esac
DS=g1_motionatlas_v4; BM=kimodo_g1_motion_official_v3
RESP=$(curl -s -m 200 -F "file=@$OUT" -F "name=$NAME" -F "observation_spec_id=$OBS" -F "robot_variant=standard" http://127.0.0.1:8010/api/v1/policies/upload)
echo "[submit] upload response: $(echo "$RESP" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(json.dumps({k: d.get(k) for k in ("id","name","status","observation_spec_id","robot_variant","policy_type")}, ensure_ascii=False))')"
PID=$(echo "$RESP" | python3 -c "
import json,sys; d=json.load(sys.stdin)
assert d.get('status') == 'compatible', d
print(d['id'])")
RID=$(curl -s -m 60 -H "Content-Type: application/json" -d "{\"policy_id\": \"$PID\", \"dataset_id\": \"$DS\", \"benchmark_id\": \"$BM\", \"mode\": \"official\", \"overrides\": {\"workers\": 64, \"num_envs\": 1, \"video_mode\": \"failed_and_hardest\", \"video_failed_motions\": 5, \"video_hard_motions\": 5}}" http://127.0.0.1:8010/api/v1/runs | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])")
echo "[submit] policy $PID run $RID ($NAME)"
