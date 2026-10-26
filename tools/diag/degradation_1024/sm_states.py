"""Record real policy inputs (obs history, goal) along clean CPU-MuJoCo rollouts of one bundle (every 5th step, first 20 s)."""
import os, sys, json, numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, "/home/ubuntu/supermimic/supermimic_repro_v2wd"); sys.path.insert(0, "/home/ubuntu/supermimic/supermimic_repro_v2wd/scripts")
import multiprocessing as mp
def run(args):
    bundle, clip = args
    import torch, mujoco
    torch.set_num_threads(1)
    from play_mujoco import MujocoPlayer, Reference
    import onnxruntime as ort
    pl = MujocoPlayer(bundle, backend="torch")
    ref = Reference(clip, pl.meta)
    data, model = pl.data, pl.model
    pl.set_reference_pose(data, ref, 0, with_velocity=True)
    prev_a = torch.zeros(pl.nj); hist = pl.proprio(prev_a)[None].repeat(pl.hist, 1); frame = 1
    H, G = [], []
    for t in range(1, min(ref.T - 1, 1000) + 1):
        goal = pl.goal(ref, frame)
        if t % 5 == 0: H.append(hist.numpy().copy()); G.append(goal.numpy().copy())
        a = pl.act(hist, goal)
        data.ctrl[pl.act_ids] = pl.offset + pl.scale * a.numpy().astype(np.float64)
        for _ in range(pl.decimation): mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        if abs(float(ref.bp[frame, ref.anchor, 2]) - float(data.xpos[pl.anchor_bid][2])) > 0.25: break
        prev_a = a; frame = min(frame + 1, ref.T - 1)
        hist = torch.cat([hist[1:], pl.proprio(prev_a)[None]], 0)
    return np.array(H, dtype=np.float32), np.array(G, dtype=np.float32)
if __name__ == "__main__":
    tag, bundle = sys.argv[1], sys.argv[2]
    tasks = json.load(open("/data2/supermimic/diag_1024/taskC/roll/tasks.json"))
    clips = sorted({t["clip"] for t in tasks if t["grp"] == "R"})
    with mp.get_context("fork").Pool(16) as pool:
        res = pool.map(run, [(bundle, c) for c in clips])
    H = np.concatenate([r[0] for r in res if len(r[0])]); G = np.concatenate([r[1] for r in res if len(r[1])])
    np.savez_compressed(f"/data2/supermimic/diag_1024/taskC/states_{tag}.npz", H=H, G=G)
    print(tag, H.shape, G.shape)
