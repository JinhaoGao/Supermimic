"""At the eval failure step: waist joint tracking error and reference waist angle vs the model's joint range (deterministic).
  python trace_waist.py <policy.onnx> <out.jsonl> <motion.npz> ..."""
import importlib.util, json, sys, os
import numpy as np, mujoco
EV = "/home/ubuntu/work/NOetix_GMT_eval/vendor/Noetix_GMT/scripts/evaluate_mujoco.py"
OBS = "/home/ubuntu/work/NOetix_GMT_eval/config/observations/supermimic_g1_v1.yaml"
XML = "/home/ubuntu/work/unitree_mujoco_noetix_g1_gmt/unitree_robots/g1/scene_29dof.xml"
TB = "left_ankle_roll_link,right_ankle_roll_link,left_wrist_yaw_link,right_wrist_yaw_link"
sys.path.insert(0, os.path.dirname(EV))
spec = importlib.util.spec_from_file_location("evm", EV)
ev = importlib.util.module_from_spec(spec); sys.modules["evm"] = ev; spec.loader.exec_module(ev)
policy, out = sys.argv[1], sys.argv[2]
with open(out, "a") as fo:
    for mp in sys.argv[3:]:
        args = ev.build_argparser().parse_args([
            "--policy-type", "supermimic", "--obs-config", OBS, "--policy", policy, "--motion-path", mp, "--model", XML,
            "--motion-joint-order", "policy", "--sim-dt", "0.001", "--decimation", "20", "--device", "cpu",
            "--root-pos-threshold", "0.25", "--root-height-threshold", "0.25", "--root-ori-threshold", "1.0",
            "--body-pos-threshold", "0.25", "--root-pos-mode", "z", "--body-pos-mode", "z", "--termination-bodies", TB])
        cfg = ev.build_rollout_cfg(args)
        motion = ev.MotionData(mp, ev.discover_tracked_body_names(cfg))
        runner = ev.create_runner(cfg, motion)
        jn = list(runner.joint_names)
        wi = [i for i, n in enumerate(jn) if "waist" in n]
        rng = []
        for i in wi:
            jid = mujoco.mj_name2id(runner.model, mujoco.mjtObj.mjOBJ_JOINT, jn[i])
            rng.append([float(x) for x in runner.model.jnt_range[jid]])
        runner.reset(0)
        werr_hist = []
        rec = None
        for step in range(motion.length):
            runner.prev_action = runner.action.copy()
            a = runner.act(step); runner.action[:] = a; runner.step_sim(a)
            m = runner.compute_metrics(step)
            q, _ = runner.current_joint_policy_order()
            rq = motion.joint_pos[motion.frame(step), runner.motion_to_policy_idx]
            werr_hist.append(np.abs(q[wi] - rq[wi]))
            r = runner.termination_reason(m)
            if r:
                W = np.array(werr_hist)
                rec = dict(motion=os.path.basename(mp), reason=r, step=step, waist=[jn[i] for i in wi], range=rng,
                           q=[round(float(x), 3) for x in q[wi]], ref_q=[round(float(x), 3) for x in rq[wi]],
                           err=[round(float(x), 3) for x in np.abs(q[wi] - rq[wi])],
                           err_max_last1s=[round(float(x), 3) for x in W[-50:].max(0)],
                           ref_out_of_range=[bool(rq[wi][k] < rng[k][0] - 1e-3 or rq[wi][k] > rng[k][1] + 1e-3) for k in range(len(wi))],
                           ref_out_of_range_any_frame=[bool(((motion.joint_pos[:, runner.motion_to_policy_idx][:, wi[k]] < rng[k][0] - 1e-3) | (motion.joint_pos[:, runner.motion_to_policy_idx][:, wi[k]] > rng[k][1] + 1e-3)).any()) for k in range(len(wi))],
                           mean_abs_joint_err=round(float(np.mean(np.abs(q - rq))), 3))
                break
        if rec is None:
            rec = dict(motion=os.path.basename(mp), reason="ok")
        fo.write(json.dumps(rec) + "\n"); fo.flush()
