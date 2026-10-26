#!/usr/bin/env bash
# 2-rank gloo CPU torchrun e2e on the merged master (scratch clone), lafan1 8 clips.
cd /tmp/claude-1000/-home-user-Workspace-Noetix-GMT/7380f59e-8ed0-40ea-9060-69c5f5a5a1ef/scratchpad/handoff_tmp/mt
export TMPDIR=/tmp/claude-1000/-home-user-Workspace-Noetix-GMT/7380f59e-8ed0-40ea-9060-69c5f5a5a1ef/scratchpad/handoff_tmp/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=/tmp/claude-1000/-home-user-Workspace-Noetix-GMT/7380f59e-8ed0-40ea-9060-69c5f5a5a1ef/scratchpad/handoff_tmp/mt OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset LD_PRELOAD
TR=/home/user/supermimic/mjlab/.venv/bin/torchrun
COMMON="--motion_dirs /home/user/supermimic/data/lafan1 --device cpu --num_envs 16 --rollouts 16 --wm_windows 64 --wm_batches 2 --no_compile --cache_dir /tmp/claude-1000/-home-user-Workspace-Noetix-GMT/7380f59e-8ed0-40ea-9060-69c5f5a5a1ef/scratchpad/handoff_tmp/ddp/cache --log_root /tmp/claude-1000/-home-user-Workspace-Noetix-GMT/7380f59e-8ed0-40ea-9060-69c5f5a5a1ef/scratchpad/handoff_tmp/ddp/logs --log_interval 2 --param_check_interval 2 --ckpt_interval 4 --eval_interval 4 --eval_clips 3 --eval_envs 3 --num_threads 4"
# A: bj31 production-style switches (EE default on, root_ori 1.0, clamp, alive signal + floor, early stop) + forced rollback at 10
$TR --standalone --nproc_per_node 2 scripts/train.py $COMMON --run_name A --iterations 12 --root_ori_term 1.0 --sampler_clamp_returns --wd_signal imagined_reward_alive --wd_reward_floor -1 --es_eval_fail_ratio 1.5 --wd_force_at 10 > /tmp/claude-1000/-home-user-Workspace-Noetix-GMT/7380f59e-8ed0-40ea-9060-69c5f5a5a1ef/scratchpad/handoff_tmp/ddp/A.log 2>&1
echo "A rc=$?"
# B: master-side features under DDP (critic errors, goal_ref_root, fixes, sampler cap/clusters/boot_mean, root_z, robust watchdog)
$TR --standalone --nproc_per_node 2 scripts/train.py $COMMON --run_name B --iterations 8 --critic_errors --goal_ref_root --fixes --sampler_cap 50 --sampler_clusters 4 --sampler_boot_mean --root_z_term 0.25 --wd_signal imagined_reward_robust --wd_sustain 200 --wd_warmup 1500 --wd_baseline_after_warmup > /tmp/claude-1000/-home-user-Workspace-Noetix-GMT/7380f59e-8ed0-40ea-9060-69c5f5a5a1ef/scratchpad/handoff_tmp/ddp/B.log 2>&1
echo "B rc=$?"
# C: resume A's model_8 with new goal / critic inputs (graft + goal-normalizer calibration under DDP) and --resume_lr_scale
CK=$(ls -d /tmp/claude-1000/-home-user-Workspace-Noetix-GMT/7380f59e-8ed0-40ea-9060-69c5f5a5a1ef/scratchpad/handoff_tmp/ddp/logs/*_A | head -1)/model_8.pt
$TR --standalone --nproc_per_node 2 scripts/train.py $COMMON --run_name C --iterations 12 --root_ori_term 1.0 --sampler_clamp_returns --goal_ref_root --critic_errors --resume $CK --resume_lr_scale 0.25 > /tmp/claude-1000/-home-user-Workspace-Noetix-GMT/7380f59e-8ed0-40ea-9060-69c5f5a5a1ef/scratchpad/handoff_tmp/ddp/C.log 2>&1
echo "C rc=$?"
