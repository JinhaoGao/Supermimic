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
falls 0.08 below its recent best (or the critic-loss median jumps 50×) it reloads the last healthy checkpoint and
halves the learning rates (`--no_watchdog`, `--wd_drop`, `--wd_force_at` for tests). G1 was resumed from 40000
with LR 1e-4 (`g1_atlas_v1r`), E2 from 86000 (`e2_atlas_v1w`).

## Layout

| file | role |
| --- | --- |
| `supermimic/env_cfg.py` | mjlab env: G1 tracking task + paper's changes (no push, no self-collision, joint-limit −2, no EE termination, per-episode DR, no auto-reset) |
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
* Termination in imagination is a soft sigmoid version of the hard thresholds.
* "Update-to-data ratio 0.4" is ambiguous; `--env_steps_per_iter` controls how
  many simulator steps are collected per (WM + policy) update, default 1.
* Returns-to-go of time-outs are bootstrapped with r_last/(1−γ) for the sampler.
* `torch.compile` default mode; `reduce-overhead` (CUDA graphs) fails on output aliasing.
