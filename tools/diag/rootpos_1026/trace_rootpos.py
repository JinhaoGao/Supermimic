"""Read-only: re-run the eval site's deterministic rollout (its runner / MJCF / explicit PD / ONNX, same args as
diag_1025/oriterm/scripts/trace_states.py) on the root_pos-failing segments and record, per control step, the robot and
reference z of pelvis (eval root_pos: |dz| > 0.25) and torso_link (training anchor_pos: |dz| > 0.25), plus torso
gravity-z tilt diff (training anchor_ori > 0.8) and EE z (training/eval > 0.25). Answers: does a training criterion fire
before the eval site's pelvis-height failure?
  ONNX=... python -I trace_rootpos.py <list.tsv> <out.jsonl> <chunk_index> <num_chunks>"""
import importlib.util, json, os, sys, time
import numpy as np

EV = "/home/ubuntu/work/NOetix_GMT_eval/vendor/Noetix_GMT/scripts/evaluate_mujoco.py"
OBS = "/home/ubuntu/work/NOetix_GMT_eval/config/observations/supermimic_g1_v1.yaml"
XML = "/home/ubuntu/work/unitree_mujoco_noetix_g1_gmt/unitree_robots/g1/scene_29dof.xml"
ONNX = os.environ["ONNX"]
TB = "left_ankle_roll_link,right_ankle_roll_link,left_wrist_yaw_link,right_wrist_yaw_link"
CONT = int(os.environ.get("CONT", "50"))
sys.path.insert(0, os.path.dirname(EV))
spec = importlib.util.spec_from_file_location("evm", EV)
ev = importlib.util.module_from_spec(spec)
sys.modules["evm"] = ev
spec.loader.exec_module(ev)
import mujoco  # noqa: E402

def quat_rot_inv(q, v):  # wxyz
  w, x, y, z = q
  R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
  return R.T @ v

lst, out, ci, nc = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
lines = open(lst).read().splitlines()
hdr = lines[0].split("\t")
rows = [dict(zip(hdr, x.split("\t"))) for x in lines[1:]]
G = np.array([0.0, 0.0, -1.0])
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
  m_ = runner.model
  bid = {n: mujoco.mj_name2id(m_, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("pelvis", "torso_link") + tuple(TB.split(","))}
  mi = {n: motion.body_names.index(n) for n in bid}
  T = motion.length
  last = min(T - 1, eval_fail + CONT)
  runner.reset(0)
  first = dict(eval=None, pelvis_z=None, torso_z=None, torso_tilt=None, ee_z=None, pelvis_ori=None)
  trace = []
  for step in range(last + 1):
    runner.prev_action = runner.action.copy()
    a = runner.act(step)
    runner.action[:] = a
    runner.step_sim(a)
    mtr = runner.compute_metrics(step)
    reason = runner.termination_reason(mtr)
    f = motion.frame(step)
    d = runner.data
    dz_p = d.xpos[bid["pelvis"]][2] - motion.body_pos_w[f, mi["pelvis"]][2]
    dz_t = d.xpos[bid["torso_link"]][2] - motion.body_pos_w[f, mi["torso_link"]][2]
    gz_r = quat_rot_inv(d.xquat[bid["torso_link"]], G)[2]
    gz_m = quat_rot_inv(motion.body_quat_w[f, mi["torso_link"]].astype(np.float64), G)[2]
    ee = max(abs(d.xpos[bid[n]][2] - motion.body_pos_w[f, mi[n]][2]) for n in TB.split(","))
    ro = float(mtr.root_ori_error)
    for key, hit in (("eval", bool(reason)), ("pelvis_z", abs(dz_p) > 0.25), ("torso_z", abs(dz_t) > 0.25),
                     ("torso_tilt", abs(gz_r - gz_m) > 0.8), ("ee_z", ee > 0.25), ("pelvis_ori", ro > 1.0)):
      if hit and first[key] is None:
        first[key] = step
    if first["eval"] is not None and step == first["eval"]:
      at = dict(dz_pelvis=round(float(dz_p), 3), dz_torso=round(float(dz_t), 3), tilt=round(float(gz_r - gz_m), 3), ee=round(float(ee), 3),
                ref_pelvis_z=round(float(motion.body_pos_w[f, mi["pelvis"]][2]), 3), ref_torso_z=round(float(motion.body_pos_w[f, mi["torso_link"]][2]), 3),
                reason=reason)
  rec = dict(motion=os.path.basename(mp), src=r.get("src"), leader=r.get("leader"), eval_fail_csv=eval_fail, first=first,
             at_eval=at if first["eval"] is not None else None, sec=round(time.time() - t0, 1))
  with open(out, "a") as fo:
    fo.write(json.dumps(rec) + "\n")
  print(rec, flush=True)
