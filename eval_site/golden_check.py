"""Golden check of the evaluator's SuperMimic runner (run from the NOetix_GMT_eval root, noetix_eval python).

Builds the runner exactly like a run does (build_argparser -> build_rollout_cfg -> MotionData ->
create_runner), steps N control ticks with the evaluator's own loop, captures the ONNX feeds and
recomputes them independently from raw MuJoCo data (qpos/qvel/sensordata/xquat) and the raw npz.
  python golden_check.py <policy.onnx> <obs_profile.yaml> <robot_model.xml> <motion.npz> [start_frame] [ticks]
"""
import importlib.util
import json
import sys

import mujoco
import numpy as np
import onnxruntime as ort

policy, obs_cfg, model_xml, motion_path = sys.argv[1:5]
t0 = int(sys.argv[5]) if len(sys.argv) > 5 else 0
ticks = int(sys.argv[6]) if len(sys.argv) > 6 else 50

spec = importlib.util.spec_from_file_location("evm", "vendor/Noetix_GMT/scripts/evaluate_mujoco.py")
ev = importlib.util.module_from_spec(spec)
sys.modules["evm"] = ev
spec.loader.exec_module(ev)
args = ev.build_argparser().parse_args([
    "--policy-type", "supermimic", "--obs-config", obs_cfg, "--policy", policy, "--motion-path", motion_path,
    "--model", model_xml, "--motion-joint-order", "policy", "--sim-dt", "0.001", "--decimation", "20",
])
cfg = ev.build_rollout_cfg(args)
motion = ev.MotionData(motion_path, ev.discover_tracked_body_names(cfg))
runner = ev.create_runner(cfg, motion, ev.create_runner_resources(cfg))


class Spy:
    def __init__(self, sess):
        self.sess, self.feeds = sess, []

    def run(self, outs, feeds):
        self.feeds.append({k: np.array(v) for k, v in feeds.items()})
        return self.sess.run(outs, feeds)

    def __getattr__(self, name):
        return getattr(self.sess, name)


runner.session = Spy(runner.session)

# ---- independent side: raw ONNX metadata, raw npz, raw MuJoCo arrays ----------------------
meta = ort.InferenceSession(policy, providers=["CPUExecutionProvider"]).get_modelmeta().custom_metadata_map
names = json.loads(meta["joint_names"])
q0 = np.array(json.loads(meta["default_joint_pos"]))
anchor = meta["anchor_body_name"]
nj = len(names)
m, d = runner.model, runner.data
hinge = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
qadr = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names])
vadr = np.array([m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names])
aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, anchor)
raw = np.load(motion_path, allow_pickle=True)
rj = [str(x) for x in raw["joint_names"]]
col = [rj.index(n) for n in names]
rb = [str(x) for x in raw["body_names"]]
ref_q, ref_qd = raw["joint_pos"][:, col].astype(np.float64), raw["joint_vel"][:, col].astype(np.float64)
ref_aq = raw["body_quat_w"][:, rb.index(anchor)].astype(np.float64)
T = ref_q.shape[0]


def sensor(name):
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, name)
    a, n = m.sensor_adr[sid], m.sensor_dim[sid]
    return d.sensordata[a:a + n].astype(np.float64).copy()


def mat(q):
    out = np.zeros(9)
    mujoco.mju_quat2Mat(out, np.asarray(q, dtype=np.float64) / np.linalg.norm(q))
    return out.reshape(3, 3)


def expected(timestep, prev_action):
    quat = sensor(runner.quat_sensor_name)
    gyro = sensor(runner.gyro_sensor_name)
    g = np.zeros(3)
    mujoco.mju_rotVecQuat(g, np.array([0.0, 0.0, -1.0]), quat * np.array([1, -1, -1, -1]) / np.linalg.norm(quat))
    frame = np.concatenate([d.qpos[qadr] - q0, d.qvel[vadr], gyro, g, prev_action])
    rows = []
    Rr = mat(d.xquat[aid])
    for k in range(10):
        f = min(timestep + k, T - 1)
        rel = Rr.T @ mat(ref_aq[f])
        rows.append(np.concatenate([ref_q[f], ref_qd[f], rel[:, :2].reshape(6)]))
    return frame, np.stack(rows)


runner.reset(t0)
hist_exp = None
worst = {"obs_newest": 0.0, "obs_hist": 0.0, "goal": 0.0}
prev = np.zeros(nj)
fell = None
for k in range(ticks):
    ts = t0 + k
    runner.prev_action = runner.action.copy()
    frame_exp, goal_exp = expected(ts, prev)
    hist_exp = np.repeat(frame_exp[None], 10, 0) if hist_exp is None else np.concatenate([hist_exp[1:], frame_exp[None]])
    action = runner.act(ts)
    feed = runner.session.feeds[-1]
    worst["obs_newest"] = max(worst["obs_newest"], np.abs(feed["obs_hist"][0, -1] - frame_exp).max())
    worst["obs_hist"] = max(worst["obs_hist"], np.abs(feed["obs_hist"][0] - hist_exp).max())
    worst["goal"] = max(worst["goal"], np.abs(feed["goal"][0] - goal_exp).max())
    runner.action[:] = action
    runner.step_sim(action)
    metrics = runner.compute_metrics(ts)
    reason = runner.termination_reason(metrics)
    prev = action.astype(np.float64)
    if reason and fell is None:
        fell = (k, reason)
        break
print(f"[golden] {motion_path.split('/')[-1]} start {t0}, {k + 1} ticks, termination {fell}")
for key, v in worst.items():
    print(f"[golden] {key:10s} max |diff| {v:.2e} {'OK' if v < 1e-4 else 'FAIL'}")
print(f"[golden] last mpjpe_rad {metrics.mpjpe_rad:.4f} mpkpe_mm {metrics.mpkpe_mm:.2f}")
