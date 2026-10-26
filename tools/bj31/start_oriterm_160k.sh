#!/usr/bin/env bash
# 10-25 用户批准:停主线 g1v2_r15ws_ddp8,从 model_160000 起 T(root_ori 1.0)/C 两臂 4+4 卡。
set -euo pipefail
export REPO=/home/ubuntu/supermimic/supermimic_repro_oriterm
export SRC_CKPT=/data2/supermimic/logs/2026-10-25_12-31-57_g1v2_r15ws_ddp8/model_160000.pt
export CKPT=/data2/supermimic/warmstart_r15ws_model_160000.pt
export CKPT_MD5=c0818a376c9ac801722bda7d539df61c
export ITERS=172000 RUN_T=g1v2_ro10_160k_T4 RUN_C=g1v2_ro0_160k_C4
echo "[start] $(date '+%F %T') stopping live run"
cd /home/ubuntu/supermimic/supermimic_repro_r15ws && { bash scripts/stop_bj31_ddp.sh g1v2_r15ws_ddp8 || true; } | tail -2
for i in $(seq 1 30); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 2; done
echo "[start] gpu procs: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"
cd "$REPO"
bash scripts/launch_bj31_oriterm.example.sh prep
bash scripts/launch_bj31_oriterm.example.sh launch
echo "[start] $(date '+%F %T') done"
