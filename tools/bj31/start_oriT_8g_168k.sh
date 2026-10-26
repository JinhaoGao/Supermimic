#!/usr/bin/env bash
# 10-26 抗摔 T 臂通过判据 → 按批准方案把 T 最好点(168000,v4 91.80%)放回 8 卡主线续训。参数与 T 臂完全相同,只改 rank 数/resume/run 名/iterations。
set -euo pipefail
REPO=/home/ubuntu/supermimic/supermimic_repro_oriterm
SRC=/data2/supermimic/logs/2026-10-25_22-22-06_g1v2_ro10_160k_T4/model_168000.pt
CKPT=/data2/supermimic/warmstart_oriT_model_168000.pt
MD5=311b86bdaedc7fb1ff2741b92457856a
RUN=g1v2_ro10_8g_168k
[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && { echo "[start] GPUs busy" >&2; exit 1; }
echo "$MD5  $SRC" | md5sum -c --quiet -
[ -f "$CKPT" ] || { cp "$SRC" "$CKPT.part"; mv "$CKPT.part" "$CKPT"; }
echo "$MD5  $CKPT" | md5sum -c --quiet -
cd "$REPO"
NPROC=8 THREADS=24 bash scripts/launch_bj31_ddp.sh "$RUN" \
  --motion_list /data2/supermimic/prep_1025/g1v2_r15_255008_bj31.txt --robot g1 --robot_variant eval+evalcontact --ee_term_z 0.25 \
  --source_weights noetix_kungfu=10,noetix_kungfu_dance_show=15,sportsmeeting=10,lafan1=15,g1_all=10,mixamo_cn=2,virtual_point=3 \
  --name_weights tengkong=200,kongfan=200,shoufan=200,erqijiao=200,danpaijiao=200,xuanfeng=200,waibailian=100,lihetui=100,saotang=100,ceti=100,houti=100,zhengti=100 \
  --sampler_clamp_returns --num_envs 4096 --rollouts 4096 --H 48 --wm_batches 10 --wm_windows 2048 --wm_k 8 \
  --lr 8.9252e-4 --lr_decay_iters 250000 --iterations 200000 --seed 0 --wd_drop 0.2 --wd_reward_floor -1.0 --wd_signal imagined_reward_alive \
  --num_workers 64 --eval_clip_names /data2/supermimic/lists/eval_1024_g1atlasv1.txt --eval_interval 2000 --eval_envs 1024 \
  --es_eval_fail_ratio 1.5 --es_patience 2 --ckpt_interval 2000 --log_interval 50 --param_check_interval 1000 \
  --resume "$CKPT" --root_ori_term 1.0 --root_ori_term_temp 0.05
cd /data2/supermimic/eval_export
RUN=$RUN NAME='SuperMimic G1 抗摔8卡主线(骨盆朝向终止1.0,从T臂168000)' STEP=2000 FROM=170000 REPO=$REPO XS=$REPO/scripts \
  setsid nohup bash scripts/auto_v4.sh > auto_v4_$RUN.log 2>&1 < /dev/null &
echo "[start] $(date '+%F %T') launched $RUN + auto_v4"
