"""SuperMimic tracker inputs for the MuJoCo evaluator (policy_type: supermimic).

SuperMimic = Liao et al., arXiv 2610.28480 (differentiable world model + BPTT); our reproduction is the
~/Workspace/supermimic repo (mjlab), ONNX from its scripts/export_bundle.py + scripts/export_eval_onnx.py.
SuperMimicRunner in evaluate_mujoco.py loads this file on first use, pinned by sha256 like
eval_anchor_ori_b.py (the web replay cache key hashes evaluate_mujoco.py only).

ONNX inputs, raw values (normalization is inside the graph), joints in metadata joint_names order:
  obs_hist [1, H, 3*nj+6]   oldest -> newest; per frame
      joint_pos - default_joint_pos (nj) | joint_vel (nj) | root angular velocity in the root frame (3)
      | projected gravity in the root frame (3) | previous action (nj)
  goal     [1, K, 2*nj+6]   reference frames t .. t+K-1, clamped at the clip end; per frame
      ref joint_pos (nj) | ref joint_vel (nj) | rot6d(R_robot_anchor^T R_ref_anchor) (6)
      rot6d = first two columns of the rotation matrix, row-major (matrix_3x2_row_major);
      anchor = metadata anchor_body_name (G1 torso_link, E2 waist_pitch_link).
  output action [1, nj]: joint target = default_joint_pos + action_scale * action (absolute PD).
Reset as in training: the history holds the first frame H times, previous action 0.
Frame convention: goal row 0 is the frame the state after this control step is scored against
(the evaluator calls act(t) -> step -> compute_metrics(t)); in training the first goal row is
likewise the frame the reward compares the next state with. Root angular velocity and gravity come
from the IMU sensors (as for the other runners), the robot anchor orientation from xquat.
Exports whose goal carries the reference anchor position (2*nj+9 values, needs odometry) are refused.
Metadata goal_layout "q,qd,rot6d,ref_root" (round 18, 10-25; absent = "q,qd,rot6d"): each goal frame
additionally ends with the reference anchor's world height (1), linear velocity (3) and angular velocity
(3) in the reference anchor frame and its projected gravity (3) -> 2*nj+16 values, reference side only.
"""

from __future__ import annotations

import numpy as np

HIST_INPUT = "obs_hist"
GOAL_INPUT = "goal"
GRAVITY = np.array([0.0, 0.0, -1.0])


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """wxyz quaternion(s) [..., 4] -> rotation matrices [..., 3, 3]."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )


def _frames_and_dim(item) -> tuple[int, int] | None:
    shape = list(item.shape)
    if len(shape) != 3 or not all(isinstance(d, int) and d > 0 for d in shape[1:]):
        return None
    return int(shape[1]), int(shape[2])


def configure(runner) -> None:
    """Check the export against the evaluator setup; cache sizes and indices on the runner."""
    cfg, md, nj = runner.cfg, runner.metadata, runner.num_action
    inputs = {item.name: item for item in runner.session.get_inputs()}
    if set(inputs) != {HIST_INPUT, GOAL_INPUT}:
        raise RuntimeError(f"SuperMimic ONNX needs inputs {HIST_INPUT!r} and {GOAL_INPUT!r}, got {sorted(inputs)}")
    hist, goal = _frames_and_dim(inputs[HIST_INPUT]), _frames_and_dim(inputs[GOAL_INPUT])
    if hist is None or goal is None:
        raise RuntimeError("SuperMimic ONNX inputs must be static [1, frames, dim]")
    (H, P), (K, G) = hist, goal
    if P != 3 * nj + 6:
        raise RuntimeError(f"obs_hist frame has {P} values, expected 3*{nj}+6")
    layout = [str(x) for x in (md.get("goal_layout") or ["q", "qd", "rot6d"])]
    if G == 2 * nj + 9 or "anchor_pos" in layout:
        raise RuntimeError("goal carries the reference anchor position (needs odometry); export a deployable policy")
    if layout not in (["q", "qd", "rot6d"], ["q", "qd", "rot6d", "ref_root"]):
        raise RuntimeError(f"unsupported goal_layout {layout}")
    ref_root = layout[-1] == "ref_root"
    if G != 2 * nj + 6 + (10 if ref_root else 0):
        raise RuntimeError(f"goal frame has {G} values, expected 2*{nj}+6{' + 10' if ref_root else ''} (goal_layout {layout})")
    if cfg.obs_expected_input_dim is not None and H * P + K * G != cfg.obs_expected_input_dim:
        raise RuntimeError(f"profile expects input {cfg.obs_expected_input_dim}, ONNX has {H * P + K * G}")
    if cfg.rotation_6d_layout != "matrix_3x2_row_major":
        raise RuntimeError(f"SuperMimic rot6d is matrix_3x2_row_major, profile says {cfg.rotation_6d_layout}")
    if cfg.action_mode != "absolute":
        raise RuntimeError("SuperMimic actions are absolute joint targets around the default pose")
    if [str(x) for x in md.get("action_joint_names", runner.joint_names)] != list(runner.joint_names):
        raise RuntimeError("action_joint_names must equal joint_names")
    if "action_offset" in md:
        offset = np.asarray(md["action_offset"], dtype=np.float64)
        default = runner.default_joint_pos_policy
        if offset.shape != default.shape or np.abs(offset - default).max() > 1e-6:
            raise RuntimeError("action_offset differs from default_joint_pos; absolute PD targets would be shifted")
    if "control_dt" in md and abs(float(md["control_dt"][0]) - cfg.control_dt) > 1e-9:
        raise RuntimeError(f"policy control_dt {md['control_dt'][0]} != evaluator sim_dt*decimation {cfg.control_dt}")
    if runner.anchor_body_name not in runner.motion.body_names:
        raise RuntimeError(f"motion {runner.motion.name!r} has no body {runner.anchor_body_name!r} (needs body_names)")
    outputs = [item.name for item in runner.session.get_outputs()]
    runner._sm = {
        "H": H,
        "K": K,
        "ref_anchor": runner.motion.body_names.index(runner.anchor_body_name),
        "ref_root": ref_root,
        "output": "action" if "action" in outputs else outputs[0],
        "hist": None,
    }


def reset(runner) -> None:
    runner._sm["hist"] = None


def proprio_frame(runner) -> np.ndarray:
    q, dq = runner.current_joint_policy_order()
    quat = runner.sensor_data(runner.quat_sensor_name)
    gyro = runner.sensor_data(runner.gyro_sensor_name)
    gravity = quat_to_matrix(quat).T @ GRAVITY
    return np.concatenate([q - runner.default_joint_pos_policy, dq, gyro, gravity, runner.action.astype(np.float64)])


def goal_window(runner, timestep: int) -> np.ndarray:
    motion, K = runner.motion, runner._sm["K"]
    idx = np.clip(timestep + np.arange(K), 0, motion.length - 1)
    ref_q = motion.joint_pos[idx][:, runner.motion_to_policy_idx].astype(np.float64)
    ref_dq = motion.joint_vel[idx][:, runner.motion_to_policy_idx].astype(np.float64)
    robot_rot = quat_to_matrix(runner.data.xquat[runner.anchor_body_id])
    ref_rot = quat_to_matrix(motion.body_quat_w[idx, runner._sm["ref_anchor"]])
    rel = np.einsum("ji,tjk->tik", robot_rot, ref_rot)  # R_robot^T R_ref, [K, 3, 3]
    parts = [ref_q, ref_dq, rel[:, :, :2].reshape(K, 6)]
    if runner._sm["ref_root"]:
        a = runner._sm["ref_anchor"]
        to_ref = lambda v: np.einsum("tji,tj->ti", ref_rot, v)  # noqa: E731  R_ref^T v
        parts += [motion.body_pos_w[idx, a, 2:3].astype(np.float64), to_ref(motion.body_lin_vel_w[idx, a].astype(np.float64)),
                  to_ref(motion.body_ang_vel_w[idx, a].astype(np.float64)), to_ref(np.broadcast_to(GRAVITY, (K, 3)))]
    return np.concatenate(parts, axis=1)


def act(runner, timestep: int) -> np.ndarray:
    state = runner._sm
    frame = proprio_frame(runner).astype(np.float32)
    if state["hist"] is None:
        state["hist"] = np.repeat(frame[None], state["H"], axis=0)
    else:
        state["hist"] = np.concatenate([state["hist"][1:], frame[None]], axis=0)
    goal = goal_window(runner, timestep).astype(np.float32)
    out = runner.session.run([state["output"]], {HIST_INPUT: state["hist"][None], GOAL_INPUT: goal[None]})[0]
    action = np.asarray(out, dtype=np.float32).reshape(-1)
    return np.clip(action, -runner.cfg.clip_actions, runner.cfg.clip_actions).astype(np.float32)
