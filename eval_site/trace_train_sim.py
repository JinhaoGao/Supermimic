"""Same clip in the TRAINING model (bundle scene.mjb, implicitfast 5 ms, mjlab contacts): end-effector height error.
  python trace_train_sim.py <bundle> <clip.npz> <tb1,tb2,...>   (mjlab venv; uses scripts/play_mujoco.py)"""
import os
import sys

import mujoco
import numpy as np
import torch

sys.path.insert(0, os.path.expanduser("~/supermimic/supermimic_repro/scripts"))
from play_mujoco import MujocoPlayer, Reference  # noqa: E402

bundle, npz, tbs = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
p = MujocoPlayer(bundle, backend="onnx")
ref = Reference(npz, p.meta)
model, data = p.model, p.data
bidx = [p.meta["body_names"].index(b) for b in tbs]
bids = [p.body_ids[i] for i in bidx]
p.set_reference_pose(data, ref, 0, with_velocity=True)
prev_a = torch.zeros(p.nj)
hist = p.proprio(prev_a)[None].repeat(p.hist, 1)
frame = 1
anchor_prev = p.body_state(p.anchor_bid)[:2]
first, worst, fell_at = {}, np.zeros(len(tbs)), None
jerr = []
with torch.no_grad():
    for t in range(1, ref.T):
        goal = p.goal(ref, frame)
        a = p.act(hist, goal)
        data.ctrl[p.act_ids] = p.offset + p.scale * a.numpy().astype(np.float64)
        for _ in range(p.decimation):
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        _, fell = p.reward_and_term(ref, frame, anchor_prev, a, prev_a)
        dz = data.xpos[bids, 2] - ref.bp[frame, bidx, 2].numpy()
        worst = np.maximum(worst, np.abs(dz))
        jerr.append(np.abs(data.qpos[p.q_adr] - ref.q[frame].numpy()))
        for i, b in enumerate(tbs):
            if abs(dz[i]) > 0.25 and b not in first:
                first[b] = (frame, round(float(dz[i]), 3))
        if fell and fell_at is None:
            fell_at = frame
        hist = torch.cat([hist[1:], p.proprio(a)[None]])
        prev_a = a
        frame = min(frame + 1, ref.T - 1)
        anchor_prev = p.body_state(p.anchor_bid)[:2]
jerr = np.array(jerr)
names = p.meta["joint_names"]
print(f"[train-sim] {os.path.basename(npz)}: T={ref.T}, training-termination fell at frame {fell_at}; first |dz|>0.25: {first}")
print("[train-sim] max |dz|:", dict(zip(tbs, np.round(worst, 3))))
top = np.argsort(-jerr.mean(0))[:6]
print("[train-sim] worst joints mean/max |q-ref|:", [(names[j], round(float(jerr[:, j].mean()), 3), round(float(jerr[:, j].max()), 3)) for j in top])
