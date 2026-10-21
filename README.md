# SuperMimic reproduction (mjlab / MuJoCo Warp)

Re-implementation of *SuperMimic: Scaling Humanoid Whole-Body Control with One
Differentiable World Model* (Liao et al., arXiv:2610.28480, CoRL 2026) on top of
[mjlab](https://github.com/mujocolab/mjlab) (commit `8bb2bc9`, 2026-10-18).
The paper's code is not public (project page "coming soon"); everything here is
built from the paper text plus mjlab's BeyondMimic tracking task, which the paper's
Appendix A matches term by term.

## Layout

| file | role |
| --- | --- |
| `supermimic/env_cfg.py` | mjlab env: G1 tracking task + paper's changes (no push, no self-collision, joint-limit −2, no EE termination, per-episode DR, no auto-reset) |
| `supermimic/library_command.py` | multi-clip motion command, return-based reset sampling (App. D.1), clip-end time-out |
| `supermimic/motions.py` | loads our GMT npz clips (joint/body remap by name) onto one GPU timeline |
| `supermimic/fk.py` | batched differentiable FK (MuJoCo semantics) with link velocities |
| `supermimic/heads.py` | analytic observation / goal / reward / termination heads on the minimal state |
| `supermimic/world_model.py` | dynamics MLP 3×2048, yaw-invariant, semi-implicit + residual, k-step autoregressive loss |
| `supermimic/policy.py` | goal encoder (K=10 future frames → z∈R³², tanh) + decoder (10-frame proprio history) + critic |
| `supermimic/replay.py` | per-env ring buffer; window / imagination-start sampling |
| `supermimic/trainer.py` | Algorithm A.1: collect → fit WM → BPTT policy update (H=32) + GAE critic |
| `supermimic/evaluate.py` | App. F benchmark: each clip from frame 0, friction 1.0, no DR/noise |
| `scripts/check_fk.py` | validates FK / reward / termination against the simulator |
| `scripts/train.py` | CLI |

## Run

```bash
cd ~/supermimic/mjlab   # the uv project that owns the venv
uv pip install -e ../supermimic_repro
CUDA_VISIBLE_DEVICES=4 uv run python ../supermimic_repro/scripts/check_fk.py --motion_dir ~/supermimic/data/lafan1
CUDA_VISIBLE_DEVICES=4 uv run python ../supermimic_repro/scripts/train.py --motion_dir ~/supermimic/data/lafan1 --run_name lafan1
```

## Deviations / choices the paper leaves open

* State for the world model = root pose + root twist + joint pos/vel (71 dims);
  reward, observation and termination heads are analytic through torch FK.
* Goal per future frame = ref joint pos/vel + ref anchor (torso) pose relative to
  the robot anchor (BeyondMimic's `motion_anchor_pos_b/ori_b`), K=10 frames.
* Termination in imagination is a soft sigmoid version of the hard thresholds.
* "Update-to-data ratio 0.4" is ambiguous; `--env_steps_per_iter` controls how
  many simulator steps are collected per (WM + policy) update, default 1.
* Returns-to-go of time-outs are bootstrapped with r_last/(1−γ) for the sampler.
