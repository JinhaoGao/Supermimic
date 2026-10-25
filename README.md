# SuperMimic reproduction (mjlab / MuJoCo Warp)

Re-implementation of *SuperMimic: Scaling Humanoid Whole-Body Control with One
Differentiable World Model* (Liao et al., arXiv:2610.28480, CoRL 2026) on top of
[mjlab](https://github.com/mujocolab/mjlab) (commit `8bb2bc9`, 2026-10-18).
The paper's code is not public (project page "coming soon"); everything here is
built from the paper text plus mjlab's BeyondMimic tracking task, which the paper's
Appendix A matches term by term.

## Status (2026-10-21)

LAFAN1 G1 subset (40 clips, 2.45 h, same as the paper's controlled comparison),
one RTX PRO 5000 Blackwell (bj27 GPU 4), paper's full configuration
(4096 envs, 4096 imagined rollouts, H=32, 10×2048 replay windows), 484 ms/iteration:

| wall clock | iteration | benchmark return | worst clip | failed / 40 |
| --- | --- | --- | --- | --- |
| 0.33 h | 2000 | 0.59 | 0.03 | 17 |
| 0.63 h | 4000 | 0.70 | 0.03 | 9 |
| 0.95 h | 6000 | 0.83 | 0.32 | 4 |
| 1.6 h | 10000 | 0.87 | 0.42 | 2 |
| 2.44 h | 16000 | 0.92 | 0.87 | 0 |

Paper reference points on the same data: SuperMimic reaches 0.82 after 1.2 h on one
H100; PPO-Sim reaches 0.72 after 18 h on eight GPUs. Benchmark protocol = Appendix F
(each clip from frame 0, friction 1.0, no DR / noise, failure = training termination).

sim2sim (checkpoint 16000 exported to ONNX, run by onnxruntime in plain CPU MuJoCo via
`scripts/play_mujoco.py`, 40 s per clip, no falls): walk 0.98, run 0.95, dance1 0.94,
dance2 0.94, sprint 0.92, jumps 0.89, fight 0.87, fall-and-get-up 0.86 normalized return.

## Deploy / playback chain

```bash
# 1. once, where mjlab is installed (CPU is fine): compiled scene + ONNX policy + metadata
uv run python scripts/export_bundle.py --ckpt model_16000.pt --motion <any clip.npz> --out ~/supermimic/bundle_16000
# 2. anywhere with mujoco + torch + onnxruntime: interactive viewer (floor + green reference ghost) or mp4
python scripts/play_mujoco.py --bundle ~/supermimic/bundle_16000 --motion <clip.npz> --seconds 20 --viewer
python scripts/play_mujoco.py --bundle ~/supermimic/bundle_16000 --motion <clip.npz> --video out.mp4
bash scripts/view_local.sh jumps1_subject1 20 1.5        # local shortcut (clip, seconds, slow-motion factor)
```

`policy.onnx` interface: inputs `obs_hist [1,10,93]` (oldest first; per frame joint_pos−default,
joint_vel, pelvis angular velocity, projected gravity, previous action) and `goal [1,10,67]`
(next 10 reference frames; per frame ref joint pos/vel, ref torso position and 6D rotation in the
current robot torso frame); output `action [1,29]`, joint target = `action_offset + action_scale * action`.
Normalization is inside the graph; joint/action order, scales and the layout text are stored as ONNX
metadata. `--backend torch` runs the PyTorch weights instead (matches ONNX to 2e-6).

## Full-atlas runs (2026-10-21 night, bj27)

Deployable variant: the goal no longer contains the reference anchor position (needs odometry),
only ref joint pos/vel + anchor orientation relative to the robot (`--goal_anchor_pos` to re-enable).
The motion library lives in CPU RAM (page cache of a packed `.npy` cache); the GPU holds only the
current frames / goal windows of the simulator and one (H+K)-frame window pack per policy update.

| run | data | clips | hours | GPU | first eval (1024-clip subset) |
| --- | --- | --- | --- | --- | --- |
| `g1_atlas_v1` | G1 MotionAtlas `g1/g1` minus GMT blacklist | 254,518 | 1474 | bj27:5 | it 4000 (0.77 h): return 0.835, 50 failed |
| `e2_atlas_v1` | E2 MotionAtlas (4 retargets) minus GMT blacklist | 372,026 | 1744 | bj27:6 | pending |

E2 is not shipped by mjlab: `supermimic/robots/` holds an entity built from the GMT sim2sim MJCF
(mesh visuals + convex mesh collision against the terrain only, no self-collision), with PD gains,
torque limits, armature, ankle friction, default pose and action scale copied from the GMT Isaac
config (`assets/e2/e2.py`); 12 tracked bodies, anchor `waist_pitch_link`.

## One model on several GPUs (torchrun data parallel)

The paper is a single-GPU method; `scripts/train.py` started by `torchrun` trains **one** model on N GPUs of
one machine. Every rank keeps the per-GPU config (4096 envs, own replay, 4096 imagined rollouts; seed + rank),
so the global batch is N× (8 GPUs: 32768 envs / 32768 rollouts per update); LR is not scaled (`--lr_scale`
to change it). The ranks stay one model (`supermimic/distributed.py`):

* rank 0's initial parameters / buffers are broadcast (so rank 0 starts from the single-GPU seed-`--seed` model);
* after every backward (WM ×10, policy ×1, critic ×20 per iteration) the gradients are flattened, all-reduced and
  averaged, then clipped / stepped exactly as before (no DDP wrapper: the step functions are compiled free functions);
* RunningMeanStd normalizers (policy obs / goal, WM in / tgt / err) are updated with the batch moments of all ranks
  (all-gather, combined in rank order) -> bitwise identical on every rank;
* the return-based reset sampler (bin_G / bin_visits / bin_weight_hist) is merged every `--sampler_sync_interval`
  iterations (default = `--log_interval`): visit-weighted mean of what each rank learned since the last merge;
* rank 0 alone writes TensorBoard (metrics averaged over ranks, episode stats pooled, `time/env_fps` = all ranks,
  `dist/*_max` = slowest rank), checkpoints and runs the benchmark; the others wait in a CPU (gloo) barrier
  (`--dist_timeout_min`, default 120, must exceed one evaluation and a cache build);
* every `--param_check_interval` iterations (default 1000, plus start / end) all learned state (models, normalizers,
  Adam moments, sampler, counters) is hashed on every rank; any difference to rank 0 stops the run;
* `--resume` loads the checkpoint on every rank (equality checked at start);
* the collapse watchdog decides on the rank means of its signals, so every rank rolls back to the same
  checkpoint at the same iteration (LR halved everywhere; state re-checked after the rollback).

Without torchrun (or with one process) the code path is the original single-GPU one (bitwise identical results).

```bash
# bj31: 8 GPUs, data / logs / cache on /data2 (prebuild the cache with scripts/build_cache.py first)
bash scripts/launch_bj31_ddp.sh g1_full_ddp8 --motion_dirs ~/motion_dataset/g1_atlas_full \
    --blacklist /data2/supermimic/lists/atlas_full_blacklist_motion50_1010.json \
    --eval_clip_names /data2/supermimic/lists/eval_1024.txt --eval_interval 4000
bash scripts/stop_bj31_ddp.sh g1_full_ddp8       # kill -9 all ranks + the torchrun agent
# two 4-GPU jobs on one host: own GPU subset + own rendezvous port each (default = --standalone on all 8 GPUs)
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29511 bash scripts/launch_bj31_ddp.sh runA ...
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29512 bash scripts/launch_bj31_ddp.sh runB ...
```

## Deploy on the G1 (unitree_mujoco `State_SuperMimic`)

The deployable G1 policy (goal without anchor position) runs as FSM mode `SuperMimic` in
`~/Workspace/unitree_mujoco/deploy/robots/g1_29dof` (Velocity → RB + left), offline from any standard GMT /
MotionAtlas npz (no conversion; joints by `joint_names` or Isaac BFS order, torso orientation from `body_names`
or pelvis ∘ waist FK) or online from the GMR-CPP Redis stream. Details and test results: `SUPERMIMIC_DEPLOY_NOTES.md` there.

```bash
bash scripts/deploy_from_server.sh g1_atlas_v1r best   # best (default) | latest | <iteration>; latest is refused if it collapsed
# then set SuperMimic.policy_file in config.yaml to the printed name; no rebuild needed
```

Isolated headless closed loop (original `scene_29dof.xml`, iteration 32000): gangnam 32 s, MotionAtlas jumps 20 s,
online gangnam 29 s, no falls, joint mean abs error 0.12 / 0.10 / 0.11 rad; controller inputs/outputs match an
independent Python recomputation to ≤ 5e-8.

## Team MuJoCo benchmark (NOetix_GMT_eval on bj31, MotionAtlas v4 official)

The eval site runs SuperMimic exports as `policy_type: supermimic` (profiles `supermimic_g1_v1`, `e2_supermimic_v1`;
evaluator plugin and site patch in `eval_site/`). Export: `export_bundle.py` (now writes `joint_stiffness`,
`joint_damping` = actuator kd + passive joint damping, `joint_armature`) then `export_eval_onnx.py` (metric `body_names`:
the benchmark's usual 14 bodies). The evaluator inputs match an independent recomputation to 2e-7 (`eval_site/golden_check.py`).

| policy | v4 success | progress | MPJPE rad | MPKPE mm | rank | failures by end-effector height |
| --- | --- | --- | --- | --- | --- | --- |
| G1 `g1_atlas_v1` model_32000 | 67.80% | 79.71% | 0.0847 | 15.27 | 144 / 168 | 913 of 1009 |
| G1 leader (GMT v2 cluster 112500) | 91.42% | 94.88% | 0.0853 | 11.46 | 1 | |
| E2 `e2_atlas_v1r` model_68000 | 64.31% | 77.50% | 0.0809 | 14.80 | 226 / 272 | 1094 of 1242 |
| E2 leader (GMT e2hdc8 C5 185500) | 90.95% | 94.82% | 0.0802 | 13.34 | 1 | |

Joint tracking is on par with the best GMT policies, but the official protocol terminates when an ankle / wrist (G1) or
ankle / elbow (E2) is more than 0.25 m off in height, and SuperMimic's recipe (paper) has no end-effector termination,
so large leg raises and arm swings are tracked loosely (reproduced in the training model too, `eval_site/trace_train_sim.py`).
Turning that check off on a 96-clip subset: G1 69.8% -> 83.3%, E2 55.2% -> 78.1%. The gap concentrates on sources that are
<1% of the training clips (noetix_kungfu 12% vs 64%, sportsmeeting 10% vs 61%, kungfu_dance_show 31% vs 99%, long lafan1
clips 15% vs 63%; G1, ours vs leader). Contact settings (self-collision on, friction 0.5 in the eval models) cost E2 ~5
points and G1 none. Model mismatches: E2 was trained with passive damping = kd on every joint (twice the actuator damping;
the eval export reproduces it), G1 on mjlab's 33.3 kg model while the eval / deploy model is the original 35.1 kg (torso +1.8 kg).

## Training stability (watchdog)

The first G1 full-atlas run collapsed at ~41k iterations (imagined reward 0.91 → 0.4 within ~400 iterations,
critic loss ×5000, then NaN in the reset sampler and a CUDA assert). The trainer now uses a Huber critic loss with
clipped value targets, drops non-finite returns in the sampler, and runs a watchdog: if the imagined-reward EMA
falls `--wd_drop` below its recent best (or the critic-loss median jumps 50×) it reloads the last healthy checkpoint,
empties the replay buffer (the collapsed policy's data; refilled by the warm-up) and halves the learning rates
(`--no_watchdog`, `--wd_force_at` for tests). `--wd_drop` is 0.2 since master e57be5b (0.08 misfired when the
return-based sampler drifted to harder bins). G1 was resumed from 40000 with LR 1e-4 (`g1_atlas_v1r`), E2 from 86000
(`e2_atlas_v1w`).

NaN guard (2026-10-22, after the 8-GPU smoke went non-finite at iteration 118 and then rolled back six times to a
checkpoint that already held NaN weights): every module update checks the gradient after the all-reduce; a
non-finite one (any rank) is dropped on every rank and the optimizer step skipped (Adam state untouched,
`guard/skips_*`). An iteration with a skipped update or a non-finite imagined reward is a transient for the watchdog;
only `--wd_nf_patience` (100) consecutive ones trigger a rollback, and without a checkpoint the run keeps skipping
instead of giving up. Checkpoints written during a non-finite streak or with non-finite weights never become the
rollback target; one early rollback point is written at `--wd_early_ckpt` (500). Simulator states that come back
non-finite end their episode as a failure (`guard/sim_nonfinite`).

Reward floor (2026-10-23, bj31 `g1v2_ddp8_ws86k`): one rank's imagined reward came back finite but astronomically
negative (rank 0 read 0.918 while the rank-mean EMA hit −2.2 at it 100808; after that rollback the EMA read −47392 at
the end of the warm-up → rollback #2). Weights were fine, but each rollback halves the LR and the sixth ends the run.
`--wd_reward_floor F` clamps each rank's finite value to ≥ F before the rank mean and the EMA (production −1.0;
default off = the behaviour above, bitwise). A real collapse (0.91 → 0.4) is far above −1 and still rolls back;
non-finite values keep the non-finite path. Rank 0 logs floored iterations with the raw values (first 3, then every
100), `guard/reward_floored` counts the floored rank values.

Watchdog signal (2026-10-25): `pi/imagined_reward_norm` averages r_t over every imagined step, also after a soft death,
where the world model diverges; the policy objective weighs those steps ~0 (J = Σ γᵗ cₜ rₜ). bj27 r16 rolled back six
times on such spikes (LR down to 2e-8) and gave up. `pi/imagined_reward_alive` = Σ cₜ rₜ / Σ cₜ (same cₜ as J, same
units) is logged every iteration; `--wd_signal imagined_reward_alive` makes the watchdog (floor, rank mean, EMA, drop
test) use it. Default `imagined_reward_norm` = the behaviour above.

Early stop (2026-10-25, opt-in): the 8-GPU lines degrade after a peak; failures of the deterministic in-training eval
correlate −0.984 with v4 and rise ~28k iterations before the cliff. With `--es_eval_fail_ratio R` (`--es_patience P`,
default 2) an eval is bad when its failed count ≥ R × median(failed of all earlier evals of the run; ≥ 3 of them,
median floored at 1); P bad evals in a row → `[earlystop]` on rank 0, `<log_dir>/earlystop.json` (iteration, history,
best-eval checkpoint), `model_final.pt`, clean exit on every rank (rank 0's eval result is broadcast, every rank takes
the same decision). The eval with the fewest failures is tracked in `best_eval.json` and saved as `model_best_eval.pt`
(rewritten at every new best only; a regular `model_<it>.pt` can be overwritten after a watchdog rollback). Needs the
in-training eval (`--eval_interval`, no `--no_eval`).

## End-effector termination

The paper drops BeyondMimic's end-effector termination; without it the G1 / E2 full-atlas policies swing feet and
hands far off the reference and fail the eval site's end-effector test (913 of 1009 G1 v4 failures). It is back on
by default (`--no_ee_termination` = paper recipe, bitwise identical to the code before the switch;
`--ee_threshold`, default 0.25 m like the eval site): the simulator terminates when any end-effector body (G1:
ankle_roll + wrist_yaw links, E2: ankle_roll + elbow links) is more than the threshold off the reference in height,
the imagined termination head gets the matching soft term (sigmoid, temperature 0.02, noisy-OR with the anchor
terms), and the in-training benchmark counts it as a failure (`failed_ee` = end-effector-only failures).

master (v2, e57be5b) added the same criterion independently as `--ee_term_z Z` (0 = off, the default there). The
merge (branch `ddp-bj31-v2`) keeps one implementation (this one) and accepts both spellings: `--ee_term_z Z` =
`--ee_termination --ee_threshold Z` for Z > 0 and `--no_ee_termination` for Z = 0; mixing them is an error unless they
agree. Bodies (the env term's), threshold, temperature (0.02) and soft term are identical in both versions (bitwise,
`tests/test_v2_merge.py`) and match the eval site (`--termination-bodies` ankle_roll + wrist_yaw / elbow,
`--body-pos-threshold 0.25 --body-pos-mode z`). Only the default differs: no flag = **on** here, off on master.
Checkpoint cfgs of every branch load through `HeadsCfg.from_dict` (`eval_ckpt.py`: as trained, or `--ee_termination
on|off` / `--ee_term_z`).

## v2 options from master (robot variants, reset-sampler prior) under data parallel

* `--robot_variant` (`+`-joined): G1 `eval` = the team's eval / deploy model (35.11 kg, waist / torso inertials,
  passive damping 0.05, friction 0.1 / 0.2; `supermimic/robots/g1_eval_model.py`), G1 / E2 `evalcontact` (condim 3 on
  every collision geom; E2 also self-collision, friction 0.5), E2 `nodamp`. Pure functions of the cfg: every rank builds
  the same model (the FK of the heads reads it from the compiled model).
* `--source_weights src=w,...` / `--name_weights keyword=w,...`: prior of the uniform share of the reset sampler
  (`library_command.reset_prior`; clip `<robot>__<source>_...`, longest prefix wins; a clip gets max(source, keyword)
  weight). Built per rank from the broadcast file list; the start-up / periodic state check hashes it with the rest of
  the learned state (`sampler_prior`), so a rank with another prior stops the run. The merge of the per-rank samplers is
  unchanged (the prior is constant and shared; it never enters bin_G / bin_visits); with `--sampler_clamp_returns` the
  clamp acts on the softmax's returns only, the prior on the uniform share only.
* `--motion_list` (a run's `motion_files.txt`) replaces `--motion_dirs` / `--blacklist`; rank 0 reads it and
  broadcasts it like the directory scan. It is used verbatim (it is the cache key): any listed file missing on this
  machine fails fast on rank 0, before the broadcast (a list from another machine: all missing); nothing is dropped.
* `scripts/launch_bj31_v2.example.sh`: the bj27 `g1v2_scratch` recipe (87.01% v4 at 86000) per rank on bj31's 8 GPUs,
  on g1v2_scratch's exact clip list (`DATASET=v2list`, default) or on the r52 set (`DATASET=dir`: same v2 options,
  different data and source shares, not a replication).
* Resume (`--resume`) prints a WARNING for every training setting of the checkpoint's run that differs from this run
  (robot / variant, source / name weights, heads incl. EE termination, sampler clamp, curriculum, H, clip list) and skips
  the reset sampler's state, with a WARNING, when the clip list differs (bin count, or basenames / order), except for an
  extended list (below). The log line of the resume shows the resulting policy LR (= --lr × checkpoint lr_scale × cosine).

## master 5f020a9 (curriculum, extend_cache, clamp) under data parallel

* `--sampler_clamp_returns`: master 5f020a9 is the same fix as this branch's 5bb8c0d (same flag, same TrainCfg /
  command-cfg names): one code path, checkpoints of either branch compare / resume (`tests/test_merge_master_1025.py`).
* `--hard_names k1,k2 --hard_term_schedule it:mult,...` (master 4314a50; bj27 r13 / r14): termination thresholds and
  soft temperatures of the matching clips × a piecewise-linear multiplier over absolute iterations. The multiplier is a
  function of cfg and iteration only (same on every rank, after a rollback / resume too); the hard set is checked across
  ranks (`curriculum` state group). Off (no `--hard_names`) = the code before, bitwise.
* `scripts/extend_cache.py --old_list --extra --cache_dir --out_list` (master c1c94be): run once before training; the cache
  of old + extra is the old cache's rows + the extra files (== a full rebuild, bitwise). Every rank maps the same
  library (`library` fingerprint in the state check). A checkpoint of the old list resumes into old + extra with its
  sampler statistics in the first bins (only if the bin count of the old files matches).
* `scripts/launch_bj31_r15ws.example.sh prep|launch`: bj27's best G1 checkpoint (g1v2_clamp_r15 model_136000, 89.85% v4)
  warm-started on bj31's 8 GPUs with r15's recipe per rank + in-training eval / early stop / alive watchdog signal /
  floor, effective LR 2.5e-5 (`--lr 8.9252e-4` against the checkpoint's lr_scale 0.0625 and cosine(136000) 0.448).

## Layout

| file | role |
| --- | --- |
| `supermimic/env_cfg.py` | mjlab env: G1 tracking task + paper's changes (no push, no self-collision, joint-limit −2, EE termination switchable (default on), per-episode DR, no auto-reset) |
| `supermimic/library_command.py` | multi-clip motion command, return-based reset sampling (App. D.1), clip-end time-out |
| `supermimic/motions.py` | loads our GMT npz clips (joint/body remap by name) onto one GPU timeline |
| `supermimic/fk.py` | batched differentiable level-wise FK (MuJoCo semantics) with link velocities |
| `supermimic/heads.py` | analytic observation / goal / reward / termination heads on the minimal state |
| `supermimic/world_model.py` | dynamics MLP 3×2048, yaw-invariant, semi-implicit + residual, k-step autoregressive loss |
| `supermimic/policy.py` | goal encoder (K=10 future frames → z∈R³², tanh) + decoder (10-frame proprio history) + critic |
| `supermimic/replay.py` | per-env ring buffer; window / imagination-start sampling |
| `supermimic/fns.py` | per-step functions wrapped in `torch.compile` (9× faster than eager) |
| `supermimic/trainer.py` | Algorithm A.1: collect → fit WM → BPTT policy update (H=32) + GAE critic |
| `supermimic/evaluate.py` | App. F benchmark: each clip from frame 0, friction 1.0, no DR/noise |
| `scripts/setup_env.sh` | clone mjlab at the pinned commit, uv venv, register this package |
| `scripts/check_fk.py` | validates FK / reward / termination against the simulator |
| `scripts/train.py` | training CLI |
| `scripts/eval_ckpt.py` | offline benchmark evaluation of a checkpoint |
| `scripts/sim2sim_mujoco.py` | checkpoint → CPU MuJoCo rollout + side-by-side video vs reference |
| `scripts/launch_bj27.sh` | nohup launcher used on the server |
| `supermimic/distributed.py` | torchrun data parallel: gradient / normalizer / sampler sync, cross-rank state check |
| `scripts/launch_bj31_ddp.sh`, `scripts/stop_bj31_ddp.sh` | detached 8-GPU torchrun launcher / stopper (bj31) |
| `tests/test_ddp_sync.py` | multi-process (CPU / gloo) checks of the sync helpers |
| `tests/test_v2_merge.py` | master's v2 options on this branch: one EE termination (== master's, bitwise), flag / checkpoint mapping, prior (== master's) under DDP, CLI wiring |
| `supermimic/robots/g1_eval_model.py` | G1 `eval` / `evalcontact` robot variants (team eval / deploy model) |
| `scripts/export_eval_onnx.py`, `eval_site/` | export for and analysis of the team's MuJoCo benchmark |
| `scripts/launch_bj31_v2.example.sh` | example (not run): v2 recipe on bj31, 8 GPUs + `--sampler_clamp_returns` |
| `scripts/extend_cache.py`, `scripts/wm_diag.py` | append clips to a packed cache (master c1c94be) / world-model diagnostic (master 4314a50) |
| `scripts/launch_bj31_r15ws.example.sh` | example (not run): r15 warm start on bj31, 8 GPUs + early stop / alive watchdog signal |
| `tests/test_merge_master_1025.py`, `tests/test_early_stop_alive.py` | master 5f020a9 merge (clamp / curriculum / extend under DDP); early stop + alive signal (real loop, 3 gloo ranks) |
| `supermimic/watch.py` | always-on failure logging: `train/would_fail_ori` (pelvis angle > 1 rad, the eval site's root_ori), floor / spin watch groups; `eval/failed_ori` in the evaluator |
| `scripts/launch_bj31_oriterm.example.sh` | example (not run): A/B of `--root_ori_term 1.0` vs control from r15ws model_148000, two 4-GPU jobs on bj31 |
| `tests/test_root_ori_term.py` | root-orientation termination (off == 55e0eaf bitwise, on: threshold / soft / gradients / env term), watch (closed form, 3 gloo ranks), launcher ports / GPU subsets |

## Setup and run

```bash
bash scripts/setup_env.sh ~/supermimic          # mjlab + venv + this package
cd ~/supermimic/mjlab                             # uv project that owns the venv
CUDA_VISIBLE_DEVICES=0 uv run python ~/supermimic/supermimic/scripts/check_fk.py --motion_dir <npz dir>
CUDA_VISIBLE_DEVICES=0 uv run python ~/supermimic/supermimic/scripts/train.py --motion_dir <npz dir> --run_name lafan1
CUDA_VISIBLE_DEVICES=0 uv run python ~/supermimic/supermimic/scripts/eval_ckpt.py --ckpt logs/.../model_16000.pt
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 uv run python ~/supermimic/supermimic/scripts/sim2sim_mujoco.py \
    --ckpt logs/.../model_16000.pt --motion_dir <npz dir> --clips dance1_subject1 --seconds 12 --out_dir videos
```

Motion data: npz with `fps, joint_names, joint_pos, joint_vel, body_names, body_pos_w,
body_quat_w (wxyz), body_lin_vel_w, body_ang_vel_w` at 50 Hz (our GMT / BeyondMimic
format). The 40 LAFAN1 clips used above live on bj27 under
`/DATA/disk0/gmt_data/ma50/data/in/motion50/g1/g1/lafan1/` (copy in `~/supermimic/data/lafan1`).

Server notes (bj27): no GitHub access (clone locally, rsync), dead proxy env vars must be
unset, pypi.org CDN is unreachable so `setup_env.sh` switches mjlab's index to the Tencent
mirror. TensorBoard for the runs: `ssh -N -L 6107:127.0.0.1:6107 tencent_bj27`.

## Deviations / choices the paper leaves open

* State for the world model = root pose + root twist + joint pos/vel (71 dims);
  reward, observation and termination heads are analytic through torch FK.
* Goal per future frame = ref joint pos/vel + ref anchor (torso) pose relative to
  the robot anchor (BeyondMimic's `motion_anchor_pos_b/ori_b`), K=10 frames.
* Termination in imagination is a soft sigmoid version of the hard thresholds
  (noisy-OR over anchor height, anchor tilt and, by default, end-effector heights; opt-in `--root_ori_term A`:
  the pelvis' full rotation angle incl. yaw > A rad, the team benchmark's root_ori test at A = 1.0, which the anchor
  tilt test cannot see -- 101 of 129 benchmark failures of r15ws model_148000).
* "Update-to-data ratio 0.4" is ambiguous; `--env_steps_per_iter` controls how
  many simulator steps are collected per (WM + policy) update, default 1.
* Returns-to-go of time-outs are bootstrapped with r_last/(1−γ) for the sampler.
* `torch.compile` default mode; `reduce-overhead` (CUDA graphs) fails on output aliasing.
