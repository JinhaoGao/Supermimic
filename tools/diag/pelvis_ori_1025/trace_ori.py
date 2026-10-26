"""Per-step decomposition of the eval-site termination criteria vs SuperMimic training criteria, on the eval site's
MuJoCo model and evaluator (deterministic, same loop as evaluate_segment). Read-only use of the eval-site code.
  python trace_ori.py <policy.onnx> <out.jsonl> <motion.npz> [more.npz ...]
Per motion: first step each criterion fires; values at the eval failure step; continues after the eval failure
(up to +CONT s) to see whether the robot actually falls afterwards."""
import importlib.util, json, sys, os, time
import numpy as np

EV = "/home/ubuntu/work/NOetix_GMT_eval/vendor/Noetix_GMT/scripts/evaluate_mujoco.py"
OBS = "/home/ubuntu/work/NOetix_GMT_eval/config/observations/supermimic_g1_v1.yaml"
XML = "/home/ubuntu/work/unitree_mujoco_noetix_g1_gmt/unitree_robots/g1/scene_29dof.xml"
TB = "left_ankle_roll_link,right_ankle_roll_link,left_wrist_yaw_link,right_wrist_yaw_link"
CONT = float(os.environ.get("CONT", "6.0"))
NOISE = float(os.environ.get("NOISE", "0.0"))
sys.path.insert(0, os.path.dirname(EV))
spec = importlib.util.spec_from_file_location("evm", EV)
ev = importlib.util.module_from_spec(spec); sys.modules["evm"] = ev; spec.loader.exec_module(ev)

policy, out = sys.argv[1], sys.argv[2]
G = np.array([0.0, 0.0, -1.0])


def tilt_twist(q_ref, q_rob):
    dq = ev.quat_multiply(q_ref, ev.quat_inverse(q_rob))
    dq = dq / np.linalg.norm(dq)
    full = float(ev.quat_error_angle(q_ref, q_rob))
    gr, gb = ev.quat_apply_inverse(q_ref, G), ev.quat_apply_inverse(q_rob, G)
    swing = float(np.arccos(np.clip(np.dot(gr, gb), -1, 1)))  # = angle between body-frame gravity vectors
    tw = np.array([dq[0], dq[3]]); n = np.linalg.norm(tw)
    twist = float(2 * np.arctan2(abs(tw[1]), abs(tw[0]))) if n > 1e-9 else float("nan")  # rotation about world z
    gzd = float(abs(gr[2] - gb[2]))
    return full, swing, twist, gzd


rng = np.random.default_rng(0)
with open(out, "a") as fo:
    for mp in sys.argv[3:]:
        t0 = time.time()
        args = ev.build_argparser().parse_args([
            "--policy-type", "supermimic", "--obs-config", OBS, "--policy", policy, "--motion-path", mp, "--model", XML,
            "--motion-joint-order", "policy", "--sim-dt", "0.001", "--decimation", "20", "--device", "cpu",
            "--root-pos-threshold", "0.25", "--root-height-threshold", "0.25", "--root-ori-threshold", "1.0",
            "--body-pos-threshold", "0.25", "--root-pos-mode", "z", "--body-pos-mode", "z", "--termination-bodies", TB])
        cfg = ev.build_rollout_cfg(args)
        motion = ev.MotionData(mp, ev.discover_tracked_body_names(cfg))
        runner = ev.create_runner(cfg, motion)
        names = [n for n, _, _ in runner.tracked_body_pairs]
        ti = names.index("torso_link") if "torso_link" in names else None
        runner.reset(0)
        first = {}
        at_fail = None
        fail_step = None
        rows = []
        T = motion.length
        for step in range(T):
            runner.prev_action = runner.action.copy()
            a = runner.act(step)
            if NOISE > 0:
                a = a + NOISE * rng.standard_normal(a.shape)
            runner.action[:] = a
            runner.step_sim(a)
            m = runner.compute_metrics(step)
            t = motion.frame(step)
            rp, rq, _, _ = runner.current_root_state()
            full, swing, twist, gzd = tilt_twist(motion.root_quat(t), rq)
            bp, bq = runner.current_body_arrays(); rbp, rbq = runner.reference_body_arrays(step)
            if ti is not None:
                tz = float(abs(bp[ti, 2] - rbp[ti, 2]))
                _, tsw, ttw, tgzd = tilt_twist(rbq[ti], bq[ti])
            else:
                tz = tsw = ttw = tgzd = float("nan")
            ref_h = float(motion.root_pos(t)[2])
            crit = {
                "eval_root_pos": m.root_height_error > 0.25,
                "eval_root_ori": m.root_ori_error > 1.0,
                "eval_ee": m.termination_body_z_error_max > 0.25,
                "pelvis_swing>1.0": swing > 1.0,
                "pelvis_twist>1.0": twist > 1.0,
                "train_torso_z>0.25": tz > 0.25,
                "train_torso_gz>0.8": tgzd > 0.8,
                "leader_ori_full>0.447": full > 0.4472,
                "leader_pos_z>0.15": m.root_height_error > (0.75 if ref_h < 0.5 else 0.15),
                "leader_ee>0.15": m.termination_body_z_error_max > 0.15,
                "fell_pelvis_z>0.5": m.root_height_error > 0.5,
            }
            for k, v in crit.items():
                if v and k not in first:
                    first[k] = step
            rows.append((full, swing, twist, gzd, tz, tgzd, m.root_height_error, m.termination_body_z_error_max, ref_h))
            reason = runner.termination_reason(m)
            if reason and fail_step is None:
                fail_step = step
                at_fail = dict(reason=reason, full=full, swing=swing, twist=twist, gzd_pelvis=gzd, torso_z=tz, torso_gzd=tgzd,
                               torso_swing=tsw, torso_twist=ttw, root_h=m.root_height_error, ee=m.termination_body_z_error_max,
                               ref_h=ref_h)
            if fail_step is not None and step - fail_step >= CONT / 0.02:
                break
        R = np.array(rows)
        pre = R[: (fail_step + 1) if fail_step is not None else len(R)]
        rec = dict(motion=os.path.basename(mp), T=T, fail_step=fail_step, at_fail=at_fail, first=first,
                   pre_max=dict(full=float(pre[:, 0].max()), swing=float(pre[:, 1].max()), twist=float(np.nanmax(pre[:, 2])),
                                gzd=float(pre[:, 3].max()), torso_z=float(np.nanmax(pre[:, 4])), torso_gzd=float(np.nanmax(pre[:, 5]))),
                   post=dict(steps=int(len(R) - (fail_step + 1)) if fail_step is not None else 0,
                             max_root_h=float(R[fail_step:, 6].max()) if fail_step is not None else None,
                             max_ee=float(R[fail_step:, 7].max()) if fail_step is not None else None,
                             min_full=float(R[fail_step:, 0].min()) if fail_step is not None else None,
                             end_full=float(R[-1, 0])),
                   twist_track=[round(float(x), 3) for x in R[:: 25, 2]],
                   noise=NOISE, sec=round(time.time() - t0, 1))
        fo.write(json.dumps(rec) + "\n"); fo.flush()
        print(rec["motion"], fail_step, at_fail and at_fail["reason"], first.get("train_torso_gz>0.8"), rec["sec"], flush=True)
