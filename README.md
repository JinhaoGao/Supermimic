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

sim2sim (checkpoint 16000 in CPU MuJoCo, `scripts/sim2sim_mujoco.py`): dance 0.97,
sprint 0.93, jumps 0.90, fall-and-get-up 0.88, fight 0.86 normalized return, no falls.

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
