"""Phase 1 of the eval-model D1 complement, run with the eval site's own python (noetix_eval env): the eval site's
deterministic rollout (its runner / MJCF / explicit PD / ONNX; the loop of diag_1025/taskB/scripts/trace_ori.py), dumping
qpos / qvel after every control step plus the site's root_ori error and termination reason, up to CONT frames after the
listed failure frame. Phase 2 (d1_evalmodel.py, mjlab venv) evaluates the new code on these states.
  python -I trace_states.py <list.tsv> <out_dir> <chunk_index> <num_chunks>"""
import importlib.util
import json
import os
import sys
import time

import numpy as np

EV = "/home/ubuntu/work/NOetix_GMT_eval/vendor/Noetix_GMT/scripts/evaluate_mujoco.py"
OBS = "/home/ubuntu/work/NOetix_GMT_eval/config/observations/supermimic_g1_v1.yaml"
XML = "/home/ubuntu/work/unitree_mujoco_noetix_g1_gmt/unitree_robots/g1/scene_29dof.xml"
ONNX = "/home/ubuntu/work/NOetix_GMT_eval/storage/models/f801f1e6873e78224281d1382363684af7f33c314d14e4e242c630aa73d2f1ec/model.onnx"
TB = "left_ankle_roll_link,right_ankle_roll_link,left_wrist_yaw_link,right_wrist_yaw_link"
CONT = int(os.environ.get("CONT", "300"))
sys.path.insert(0, os.path.dirname(EV))
spec = importlib.util.spec_from_file_location("evm", EV)
ev = importlib.util.module_from_spec(spec)
sys.modules["evm"] = ev
spec.loader.exec_module(ev)

lst, out, ci, nc = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
lines = open(lst).read().splitlines()
hdr = lines[0].split("\t")
rows = [dict(zip(hdr, x.split("\t"))) for x in lines[1:]]
os.makedirs(out, exist_ok=True)
for k in range(ci, len(rows), nc):
  r = rows[k]
  t0 = time.time()
  mp = r["path"]
  eval_fail = int(r["eval_fail_frame"])
  args = ev.build_argparser().parse_args([
    "--policy-type", "supermimic", "--obs-config", OBS, "--policy", ONNX, "--motion-path", mp, "--model", XML,
    "--motion-joint-order", "policy", "--sim-dt", "0.001", "--decimation", "20", "--device", "cpu",
    "--root-pos-threshold", "0.25", "--root-height-threshold", "0.25", "--root-ori-threshold", "1.0",
    "--body-pos-threshold", "0.25", "--root-pos-mode", "z", "--body-pos-mode", "z", "--termination-bodies", TB])
  cfg = ev.build_rollout_cfg(args)
  motion = ev.MotionData(mp, ev.discover_tracked_body_names(cfg))
  runner = ev.create_runner(cfg, motion)
  T = motion.length
  last = min(T - 1, eval_fail + CONT)
  runner.reset(0)
  qp, qv, ro, reasons = [], [], [], []
  fail_step, fail_reason = None, None
  for step in range(last + 1):
    runner.prev_action = runner.action.copy()
    a = runner.act(step)
    runner.action[:] = a
    runner.step_sim(a)
    m = runner.compute_metrics(step)
    reason = runner.termination_reason(m)
    qp.append(runner.data.qpos.copy())
    qv.append(runner.data.qvel.copy())
    ro.append(float(m.root_ori_error))
    reasons.append(reason or "")
    if reason and fail_step is None:
      fail_step, fail_reason = step, reason
  name = os.path.basename(mp)
  np.savez(os.path.join(out, name), qpos=np.array(qp), qvel=np.array(qv), root_ori_error=np.array(ro), reason=np.array(reasons),
           root_body_id=int(runner.root_body_id))
  rec = dict(motion=name, eval_fail=eval_fail, fail_step=fail_step, fail_reason=fail_reason, steps=len(qp), sec=round(time.time() - t0, 1))
  with open(os.path.join(out, f"chunk_{ci}.jsonl"), "a") as f:
    f.write(json.dumps(rec) + "\n")
  print(rec, flush=True)
