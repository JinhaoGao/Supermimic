"""CPU MuJoCo closed-loop probe of exported bundles (bundle scene = training-sim robot variant, 4x5 ms substeps).
Conditions: clean (eval-like, deterministic) / act (training-like action noise 0.1, prev_a = executed action) / obs (training obs noise).
Logs failure (anchor z / tilt / end-effector z, training thresholds), action-rate and high-frequency content.
  python sm_roll.py <out.jsonl> <nproc> <max_s>   (task list from tasks.json)"""
import os, sys, json, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
sys.path.insert(0, "/home/ubuntu/supermimic/supermimic_repro_v2wd")
sys.path.insert(0, "/home/ubuntu/supermimic/supermimic_repro_v2wd/scripts")
import multiprocessing as mp

EE = ["left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link", "right_wrist_yaw_link"]
_P = {}

def get_player(bundle):
    import torch
    torch.set_num_threads(1)
    from play_mujoco import MujocoPlayer
    if bundle not in _P:
        _P[bundle] = MujocoPlayer(bundle, backend="onnx")
    return _P[bundle]

def hf_frac(x, f_cut=10.0, dt=0.02):
    # fraction of (mean-removed) power above f_cut, averaged over channels; x [T, C]
    if x.shape[0] < 64: return float("nan")
    x = x - x.mean(0, keepdims=True)
    P = np.abs(np.fft.rfft(x, axis=0)) ** 2
    f = np.fft.rfftfreq(x.shape[0], dt)
    tot = P.sum(0) + 1e-12
    return float(np.mean(P[f >= f_cut].sum(0) / tot))

def run(task):
    import torch, mujoco
    from play_mujoco import Reference, GRAVITY
    from supermimic.math_utils import quat_apply_inverse
    bundle, clip, cond, seed, max_s = task["bundle"], task["clip"], task["cond"], task["seed"], task["max_s"]
    pl = get_player(bundle)
    ref = Reference(clip, pl.meta)
    if abs(ref.fps - 1.0 / pl.dt) > 1e-6:
        return dict(task, err=f"fps {ref.fps}")
    rng = np.random.default_rng(seed)
    h = pl.h
    model, data = pl.model, pl.data
    ee_ids = [pl.meta["body_names"].index(n) for n in EE]
    ee_bids = [pl.body_ids[i] for i in ee_ids]
    f_last = min(ref.T - 1, int(max_s / pl.dt))
    pl.set_reference_pose(data, ref, 0, with_velocity=True)
    nj = pl.nj
    def obs(prev_a):
        o = pl.proprio(prev_a)
        if cond == "obs":
            u = lambda n, w: torch.as_tensor((rng.random(n) * 2 - 1) * w, dtype=torch.float32)
            o = o.clone()
            o[0:nj] += u(nj, h["noise_joint_pos"]); o[nj:2 * nj] += u(nj, h["noise_joint_vel"])
            o[2 * nj:2 * nj + 3] += u(3, h["noise_ang_vel"]); o[2 * nj + 3:2 * nj + 6] += u(3, h["noise_gravity"])
        return o
    prev_a = torch.zeros(nj)
    hist = obs(prev_a)[None].repeat(pl.hist, 1)
    frame = 1
    A, QD, why, fail_t = [], [], None, None
    t0 = time.time()
    for t in range(1, f_last + 1):
        goal = pl.goal(ref, frame)
        a = pl.act(hist, goal)
        a_exec = a + torch.as_tensor(rng.normal(0, 0.1, nj), dtype=torch.float32) if cond == "act" else a
        data.ctrl[pl.act_ids] = pl.offset + pl.scale * a_exec.numpy().astype(np.float64)
        for _ in range(pl.decimation):
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        A.append(a.numpy().copy()); QD.append(data.qvel[pl.v_adr].copy())
        # termination at the target frame (training thresholds)
        ap = torch.as_tensor(data.xpos[pl.anchor_bid].copy(), dtype=torch.float32)
        aq = torch.as_tensor(data.xquat[pl.anchor_bid].copy(), dtype=torch.float32)
        ref_ap, ref_aq = ref.bp[frame, ref.anchor], ref.bq[frame, ref.anchor]
        dz = float((ref_ap[2] - ap[2]).abs())
        tilt = float((quat_apply_inverse(ref_aq, GRAVITY)[2] - quat_apply_inverse(aq, GRAVITY)[2]).abs())
        dz_ee = max(abs(float(ref.bp[frame, i, 2]) - float(data.xpos[b][2])) for i, b in zip(ee_ids, ee_bids))
        if not np.isfinite(data.qpos).all():
            why = "nan"
        elif dz > h["term_z"]:
            why = "anchor_z"
        elif tilt > h["term_tilt"]:
            why = "tilt"
        elif dz_ee > h["ee_term_z"]:
            why = "ee"
        if why:
            fail_t = t * pl.dt
            break
        prev_a = a_exec
        frame = min(frame + 1, ref.T - 1)
        hist = torch.cat([hist[1:], obs(prev_a)[None]], dim=0)
    A = np.array(A); QD = np.array(QD)
    dA = np.diff(A, axis=0) if len(A) > 1 else np.zeros((1, nj))
    legs, waist, arms = slice(0, 12), slice(12, 15), slice(15, 29)
    out = dict(task, steps=len(A), T=int(f_last), fail=why, fail_t=fail_t, wall=time.time() - t0,
               da_rms_legs=float(np.sqrt((dA[:, legs] ** 2).mean())), da_rms_waist=float(np.sqrt((dA[:, waist] ** 2).mean())), da_rms_arms=float(np.sqrt((dA[:, arms] ** 2).mean())),
               hf_a_legs=hf_frac(A[:, legs]), hf_a_waist=hf_frac(A[:, waist]), hf_a_arms=hf_frac(A[:, arms]),
               hf_qd_legs=hf_frac(QD[:, legs]), hf_qd_waist=hf_frac(QD[:, waist]), hf_qd_arms=hf_frac(QD[:, arms]),
               a_abs=float(np.abs(A).mean()))
    return out

if __name__ == "__main__":
    out, nproc, max_s = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
    tasks = json.load(open(os.path.join(os.path.dirname(out), "tasks.json")))
    for t in tasks: t["max_s"] = max_s
    # group tasks by bundle so each worker reuses players (chunking)
    tasks.sort(key=lambda t: (t["bundle"], t["cond"]))
    t0 = time.time()
    with mp.get_context("fork").Pool(nproc, maxtasksperchild=200) as pool, open(out, "w") as f:
        for i, r in enumerate(pool.imap_unordered(run, tasks, chunksize=4)):
            f.write(json.dumps(r) + "\n"); f.flush()
            if i % 200 == 0: print(i, len(tasks), f"{time.time() - t0:.0f}s", flush=True)
    print("done", time.time() - t0)
