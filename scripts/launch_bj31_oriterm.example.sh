#!/usr/bin/env bash
# EXAMPLE -- not run by anyone yet. bj31: A/B of the root-orientation termination (--root_ori_term, 2026-10-25), two
# 4-GPU jobs on the same host at the same time, both warm-started from the live r15ws run's model_148000:
#   arm T  g1v2_ro10_148k_T4  GPUs 0-3, rendezvous port 29511: --root_ori_term 1.0 --root_ori_term_temp 0.05
#   arm C  g1v2_ro0_148k_C4   GPUs 4-7, rendezvous port 29512: control (same code, start, seeds; no root-orientation term)
# Everything else is exactly the live run's ARGS (g1v2_r15ws_ddp8, scripts/launch_bj31_r15ws.example.sh, launched 10-25
# 12:31 at 55e0eaf) except --resume (the copied 148000) and --iterations 160000 (12k iterations per arm).
# Why: 101 of the 129 eval-site failures of model_148000 are its root_ori test (pelvis full rotation angle incl. yaw >
# 1.0 rad); training terminates on torso tilt / torso + end-effector heights only and never sees pelvis yaw or the
# pelvis / torso decoupling through the waist.
#
# NOT like the live run in two ways that follow from the design:
#   * 4 ranks per arm instead of 8: the GLOBAL BATCH HALVES (4 x 4096 = 16384 envs / 16384 imagined rollouts per
#     update instead of 32768); both arms alike, so the A/B is fair, but neither arm continues the live run 1:1;
#   * the learning rate is NOT rescaled for that: --lr 8.9252e-4 (the live run's) -> Trainer.load restores the
#     checkpoint's lr_scale 0.0625 (no watchdog rollback in the live run so far) and the cosine continues at 148000:
#     8.9252e-4 x 0.0625 x cosine(148000) 0.376766 = 2.10169e-5 = the optimizer LR stored in model_148000.pt
#     (2.101693e-05, read 10-25), i.e. both arms start at exactly the live run's LR at 148000.
#
# Needs: this branch (ddp-bj31-v2 at the commit that added this file or later) deployed on bj31 at $REPO with a COMMIT
# file, and GPUs 0-7 free (the live run g1v2_r15ws_ddp8 holds all 8 now: stop it first, or wait until it ends).
#
#   bash scripts/launch_bj31_oriterm.example.sh prep          # once: copy + md5-check the checkpoint, check list / cache
#   bash scripts/launch_bj31_oriterm.example.sh launch        # arm T, wait STAGGER_S (default 330 s), arm C
#   bash scripts/launch_bj31_oriterm.example.sh launch_T      # or one arm at a time
#   bash scripts/launch_bj31_oriterm.example.sh launch_C
#   bash scripts/stop_bj31_ddp.sh g1v2_ro10_148k_T4 ; bash scripts/stop_bj31_ddp.sh g1v2_ro0_148k_C4
#
# Expected in each log (runs/<run>.log): "resumed from /data2/supermimic/warmstart_r15ws_model_148000.pt at iteration
# 148000 (policy lr 2.102e-05 = lr x lr_scale 0.0625 x cosine 0.3768)"; "clamp_returns: 0 of 5428418 loaded bin returns
# clamped"; "[dist] start: learned state identical on all 4 ranks"; "data parallel: 4 ranks x 4096 envs ... (global
# 16384 / 16384)"; "failure watch: floor [...]: N clips ..., spin [...]"; 148000 is an eval slot, so "[eval it 148000]"
# + "[es] it 148000 ... (warming up)" before the first iteration line.
#   arm T only: ONE "WARNING resume from another setup: heads.root_ori_term = 0.0 in the checkpoint, 1.0 here" and
#     "root-orientation termination: on, pelvis full rotation angle (incl. yaw) > 1 rad (soft temp 0.05)";
#   arm C: no WARNING line, "root-orientation termination: off (pelvis angle logged only: ...)".
# Compare (TB / eval.jsonl): eval/failed_with_ori (clips failed by the training terminations or with pelvis angle
# > 1 rad: the comparable count; arm T's eval/failed already contains its root_ori terminations, arm C's does not),
# eval/failed_ori, train/would_fail_ori, train/watch_{floor,spin}_{fail,would_fail_ori,reset_share}; then the v4 site.
# Imagination vs simulator (both arms): pi/root_ori_angle_{t0,tH}, pi/root_ori_over_frac vs train/ori_over_step_frac,
# and the world model's root drift wm/root_{ang,yaw}_err_step{1,8} (its yaw is integrated, not observed); arm T also
# pi/root_ori_hazard_mean / pi/root_ori_death_share (the share of the imagined soft deaths from this term).
# Temperature 0.05 (the tilt term's): the soft term is a per-step hazard, so over --H 48 imagined steps a rollout
# holding a constant pelvis angle survives 50 % at 1.0 - 4.23 x 0.05 = 0.79 rad (10 % at 0.85, 0.2 % at 0.90): arm
# T's imagination treats ~0.8-0.98 rad as dying while its simulator and the eval site (> 1.0) do not. Deliberate here
# (the spec's arm, same pattern as the tilt term); --root_ori_term_temp 0.02 would put that point at 0.92 rad.
# The early stop (--es_eval_fail_ratio 1.5 x the median of the earlier evals) judges each arm against its own start
# eval at 148000; arm T's start count includes the root_ori failures of the old weights.
set -euo pipefail
STEP="${1:?prep | launch | launch_T | launch_C}"
REPO="${REPO:-/home/ubuntu/supermimic/supermimic_repro_oriterm}"  # ddp-bj31-v2 checkout / rsync with this file
VENV="${VENV:-/home/ubuntu/supermimic/mjlab/.venv}"
SRC_CKPT=/data2/supermimic/logs/2026-10-25_12-31-57_g1v2_r15ws_ddp8/model_148000.pt
CKPT=/data2/supermimic/warmstart_r15ws_model_148000.pt
CKPT_MD5=cb3cbd99b12144017fe4649f1290b990  # 291,457,320 bytes; md5 of $SRC_CKPT read 2026-10-25 (it 148000, lr_scale 0.0625)
LIST=/data2/supermimic/prep_1025/g1v2_r15_255008_bj31.txt
NEW_CACHE=/data2/supermimic/cache/lib_255008_3b84d5f141
EVAL_NAMES=/data2/supermimic/lists/eval_1024_g1atlasv1.txt
SOURCES=noetix_kungfu=10,noetix_kungfu_dance_show=15,sportsmeeting=10,lafan1=15,g1_all=10,mixamo_cn=2,virtual_point=3
NAMES=tengkong=200,kongfan=200,shoufan=200,erqijiao=200,danpaijiao=200,xuanfeng=200,waibailian=100,lihetui=100,saotang=100,ceti=100,houti=100,zhengti=100
ITERS="${ITERS:-160000}"
LR="${LR:-8.9252e-4}"  # x lr_scale 0.0625 (restored) x cosine(148000) 0.376766 = 2.10169e-5 = the live run's LR at 148000
STAGGER_S="${STAGGER_S:-330}"  # >= 300: the second arm starts after the first one's env build / cache mapping / compile
RUN_T="${RUN_T:-g1v2_ro10_148k_T4}"
RUN_C="${RUN_C:-g1v2_ro0_148k_C4}"

check_list() {
  echo "a038555bfba5a0220426e3ef3ca6ebbe  $LIST" | md5sum -c --quiet -
}
check_cache() {  # complete, 255,008 clips (all ok), 5,428,418 bins of 1 s = the checkpoint's bin_G
  "$VENV/bin/python" - "$NEW_CACHE" <<'EOF'
import json, math, sys
d = sys.argv[1]
meta = json.load(open(f"{d}/meta.json"))
clips = json.load(open(f"{d}/clips.json"))
ok = [c for c in clips if c.get("ok", True) and c["end"] > c["start"] + 1]
bins = sum(math.ceil((c["end"] - c["start"]) / 50) for c in ok)
print(f"[oriterm] {d}: {len(clips)} clips ({len(ok)} usable), {meta['frames']} frames, {bins} bins")
assert len(clips) == 255008 and len(ok) == 255008 and meta["frames"] == 265826690 and bins == 5428418
EOF
}
check_ckpt() {
  echo "$CKPT_MD5  $CKPT" | md5sum -c --quiet -
}
gpus_busy() {  # GPUs of the comma list $1 that run a compute process
  local g out=""
  for g in ${1//,/ }; do
    if [ -n "$(nvidia-smi -i "$g" --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; then out="$out $g"; fi
  done
  echo "$out"
}

# >>> ARGS (train.py arguments of BOTH arms = the live run's ARGS except --iterations / --resume; parsed by
# tests/test_root_ori_term.py -- keep one flag group per line)
ARGS=(
  --motion_list "$LIST"
  --robot g1 --robot_variant eval+evalcontact
  --ee_term_z 0.25
  --source_weights "$SOURCES"
  --name_weights "$NAMES"
  --sampler_clamp_returns
  --num_envs 4096 --rollouts 4096 --H 48 --wm_batches 10 --wm_windows 2048 --wm_k 8
  --lr "$LR" --lr_decay_iters 250000 --iterations "$ITERS" --seed 0
  --wd_drop 0.2 --wd_reward_floor -1.0 --wd_signal imagined_reward_alive
  --num_workers 64
  --eval_clip_names "$EVAL_NAMES" --eval_interval 2000 --eval_envs 1024
  --es_eval_fail_ratio 1.5 --es_patience 2
  --ckpt_interval 2000 --log_interval 50 --param_check_interval 1000
  --resume "$CKPT"
)
ARGS_T=(--root_ori_term 1.0 --root_ori_term_temp 0.05)
ARGS_C=()
# <<< ARGS

launch_arm() {  # <run> <gpus> <port> [extra train.py args...]
  local run="$1" gpus="$2" port="$3"
  shift 3
  local busy
  busy="$(gpus_busy "$gpus")"
  if [ -z "${FORCE:-}" ] && [ -n "${busy// /}" ]; then
    echo "[oriterm] $run: GPU(s)$busy of $gpus are in use; stop that run first or set FORCE=1" >&2
    exit 1
  fi
  check_ckpt
  echo "[oriterm] $(date '+%F %T') launching $run on GPUs $gpus, rendezvous port $port"
  CUDA_VISIBLE_DEVICES="$gpus" PORT="$port" NPROC=4 THREADS="${THREADS:-24}" \
    bash "$REPO/scripts/launch_bj31_ddp.sh" "$run" "${ARGS[@]}" "$@"
}

export PYTHONPATH="$REPO"
case "$STEP" in
  prep)
    check_list
    check_cache
    if [ ! -f "$CKPT" ]; then
      echo "$CKPT_MD5  $SRC_CKPT" | md5sum -c --quiet -  # the live run rewrites model_148000.pt only after a rollback to < 148000
      cp "$SRC_CKPT" "$CKPT.part"
      mv "$CKPT.part" "$CKPT"
    fi
    check_ckpt && echo "[oriterm] $CKPT ok ($CKPT_MD5)"
    exit 0
    ;;
  launch|launch_T|launch_C) ;;
  *) echo "STEP must be prep, launch, launch_T or launch_C" >&2; exit 1 ;;
esac
[ "$STAGGER_S" -ge 300 ] || { echo "[oriterm] STAGGER_S=$STAGGER_S: keep the arms >= 300 s apart" >&2; exit 1; }
check_list
check_cache
check_ckpt
if [ "$STEP" = launch ] && [ -z "${FORCE:-}" ] && [ -n "$(gpus_busy 0,1,2,3,4,5,6,7 | tr -d ' ')" ]; then
  echo "[oriterm] GPUs$(gpus_busy 0,1,2,3,4,5,6,7) are in use; both arms need 0-7 (stop the live run first or set FORCE=1)" >&2
  exit 1
fi
if [ "$STEP" != launch_C ]; then
  launch_arm "$RUN_T" 0,1,2,3 29511 "${ARGS_T[@]}"
fi
if [ "$STEP" = launch ]; then
  echo "[oriterm] waiting ${STAGGER_S}s before arm C"
  sleep "$STAGGER_S"
fi
if [ "$STEP" != launch_T ]; then
  launch_arm "$RUN_C" 4,5,6,7 29512 ${ARGS_C[@]+"${ARGS_C[@]}"}
fi
