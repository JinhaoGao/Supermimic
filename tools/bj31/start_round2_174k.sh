#!/usr/bin/env bash
# 10-26 用户批准:第二轮 T/C,从 8 卡主线 model_174000(v4 91.83%)起 4+4 卡到 186000。
# T2 = root_ori 0.8;C2 = root_ori 1.0(= 主线配方)。其余参数与主线完全相同。
set -uo pipefail
REPO=/home/ubuntu/supermimic/supermimic_repro_oriterm
SRC=$(ls -d /data2/supermimic/logs/*_g1v2_ro10_8g_168k)/model_174000.pt
CKPT=/data2/supermimic/warmstart_ori8g_model_174000.pt
echo "[r2] $(date '+%F %T') stopping main line"
cd $REPO && { bash scripts/stop_bj31_ddp.sh g1v2_ro10_8g_168k || true; } | tail -1
for i in $(seq 1 30); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 2; done
echo "[r2] gpu procs: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"
M=$(md5sum "$SRC" | cut -d' ' -f1); [ -f "$CKPT" ] || { cp "$SRC" "$CKPT.part"; mv "$CKPT.part" "$CKPT"; }
[ "$(md5sum "$CKPT" | cut -d' ' -f1)" = "$M" ] || { echo "[r2] md5 mismatch" >&2; exit 1; }
echo "[r2] ckpt $CKPT md5 $M"
ARGS=(--motion_list /data2/supermimic/prep_1025/g1v2_r15_255008_bj31.txt --robot g1 --robot_variant eval+evalcontact --ee_term_z 0.25
  --source_weights noetix_kungfu=10,noetix_kungfu_dance_show=15,sportsmeeting=10,lafan1=15,g1_all=10,mixamo_cn=2,virtual_point=3
  --name_weights tengkong=200,kongfan=200,shoufan=200,erqijiao=200,danpaijiao=200,xuanfeng=200,waibailian=100,lihetui=100,saotang=100,ceti=100,houti=100,zhengti=100
  --sampler_clamp_returns --num_envs 4096 --rollouts 4096 --H 48 --wm_batches 10 --wm_windows 2048 --wm_k 8
  --lr 8.9252e-4 --lr_decay_iters 250000 --iterations 186000 --seed 0 --wd_drop 0.2 --wd_reward_floor -1.0 --wd_signal imagined_reward_alive
  --num_workers 64 --eval_clip_names /data2/supermimic/lists/eval_1024_g1atlasv1.txt --eval_interval 2000 --eval_envs 1024
  --es_eval_fail_ratio 1.5 --es_patience 2 --ckpt_interval 2000 --log_interval 50 --param_check_interval 1000 --resume "$CKPT" --root_ori_term_temp 0.05)
launch() {  # run gpus port A
  echo "[r2] $(date '+%F %T') launching $1 on GPUs $2 port $3 root_ori_term $4"
  CUDA_VISIBLE_DEVICES="$2" PORT="$3" NPROC=4 THREADS=24 bash $REPO/scripts/launch_bj31_ddp.sh "$1" "${ARGS[@]}" --root_ori_term "$4"
  cd /data2/supermimic/eval_export && RUN="$1" NAME="$5" STEP=2000 FROM=176000 REPO=$REPO XS=$REPO/scripts \
    setsid nohup bash scripts/auto_v4.sh > auto_v4_$1.log 2>&1 < /dev/null &
  cd $REPO
}
launch g1v2_ro08_174k_T4 0,1,2,3 29521 0.8 'SuperMimic G1 第二轮T2(骨盆朝向终止0.8,4卡,从174000)'
echo "[r2] waiting 330s"; sleep 330
launch g1v2_ro10_174k_C4 4,5,6,7 29522 1.0 'SuperMimic G1 第二轮C2(骨盆朝向终止1.0对照,4卡,从174000)'
echo "[r2] $(date '+%F %T') done"
