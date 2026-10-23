"""Trace termination-body height errors of a SuperMimic policy on one motion in the eval site's model.
  python ee_trace.py <policy.onnx> <obs.yaml> <model.xml> <motion.npz> <tb1,tb2,...>"""
import importlib.util
import sys

import numpy as np

policy, obs_cfg, model_xml, motion_path, tbs = sys.argv[1:6]
spec = importlib.util.spec_from_file_location("evm", "vendor/Noetix_GMT/scripts/evaluate_mujoco.py")
ev = importlib.util.module_from_spec(spec)
sys.modules["evm"] = ev
spec.loader.exec_module(ev)
args = ev.build_argparser().parse_args([
    "--policy-type", "supermimic", "--obs-config", obs_cfg, "--policy", policy, "--motion-path", motion_path,
    "--model", model_xml, "--motion-joint-order", "policy", "--sim-dt", "0.001", "--decimation", "20",
    "--body-pos-threshold", "10", "--termination-bodies", tbs,
])
cfg = ev.build_rollout_cfg(args)
motion = ev.MotionData(motion_path, ev.discover_tracked_body_names(cfg))
runner = ev.create_runner(cfg, motion, ev.create_runner_resources(cfg))
names = [n for n, _, _ in runner.tracked_body_pairs]
tb = [names.index(n) for n in tbs.split(",")]
runner.reset(0)
first = {}
worst = np.zeros(len(tb))
jerr = []
for t in range(motion.length):
    runner.prev_action = runner.action.copy()
    a = runner.act(t)
    runner.action[:] = a
    runner.step_sim(a)
    m = runner.compute_metrics(t)
    pos, _ = runner.current_body_arrays()
    ref, _ = runner.reference_body_arrays(t)
    dz = pos[tb, 2] - ref[tb, 2]
    worst = np.maximum(worst, np.abs(dz))
    q, _ = runner.current_joint_policy_order()
    rq = motion.joint_pos[t, runner.motion_to_policy_idx]
    jerr.append(np.abs(q - rq))
    for i, b in enumerate(tbs.split(",")):
        if abs(dz[i]) > 0.25 and b not in first:
            first[b] = (t, round(float(dz[i]), 3))
    r = runner.termination_reason(m)
    if r:
        print("terminated", r, "at", t)
        break
jerr = np.array(jerr)
print(f"{motion.name}: T={motion.length}, first |dz|>0.25 per body: {first}")
print("max |dz| per termination body:", dict(zip(tbs.split(","), np.round(worst, 3))))
worst_j = np.argsort(-jerr.mean(0))[:6]
print("joints with largest mean |q-ref| (rad):", [(runner.joint_names[j], round(float(jerr[:, j].mean()), 3), round(float(jerr[:, j].max()), 3)) for j in worst_j])
