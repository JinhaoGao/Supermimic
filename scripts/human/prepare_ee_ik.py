#!/usr/bin/env python3
"""Can the G1 reach the human's end-effector world poses unscaled? Per-clip task-space IK -> report + filtered list.

For end-effector world tracking (HumanHeadsCfg.ee_mode, docs/ee_world_tracking_plan.md §6): every `--stride`-th frame of
each clip is solved for the floating base (position + orientation) and the 29 joints so that the robot's task links
(bone map task_links) hit the human's, unscaled, in the human world frame:
  wrists       position + rotation (calibrated offset)            -- the task
  torso        xy + rotation (no height: the human shoulder line is out of the G1's reach)
  feet         world xy; ankle height = its standing height while the human foot is down, + the human ankle lift
               while it swings; rotation while down
plus a pull towards the bone-direction IK reset pose (init_g1_*, prepare_init_states.py: also the initial guess) and
towards the neighbouring solved frames (smoothness), joints clamped to their limits. Batched Adam on TorchFK (GPU if
available), many clips per batch.

Report (TSV, one row per clip): residual percentiles of the solution (wrist / torso xy / foot position in metres, wrist
rotation in degrees), the joint speed of the solution, the human pelvis speed; and pass/fail by the thresholds.
The list holds the passing clips (same paths as --motion_list), ready for train.py --motion_list.

  cd ~/supermimic_ws/mjlab && uv run python ~/Supermimic/scripts/human/prepare_ee_ik.py \\
      --motion_list ~/datasets/paired_ma/human_list.txt --out_dir ~/datasets/paired_ma/ee [--device cuda:0]
"""

from __future__ import annotations

import argparse
import os
import time
import types

import mujoco
import numpy as np
import torch

from supermimic.fk import TorchFK
from supermimic.human import bone_map as bone
from supermimic.human.command import robot_rest_geometry
from supermimic.human.heads import TASK_FEET, TASK_TORSO, TASK_WRISTS, rot6d_to_matrix, rot_err2
from supermimic.human.motions import derive_ee_fields
from supermimic.math_utils import quat_from_rotvec, quat_mul

FOOT_REGEX = r"^(left|right)_foot[1-7]_collision$"


def robot_model():
  from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec

  model = get_spec().compile()
  bodies = list(range(1, model.nbody))
  hinge = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
  ent = types.SimpleNamespace(
    indexing=types.SimpleNamespace(body_ids=torch.tensor(bodies), joint_ids=torch.tensor(hinge)),
    body_names=[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in bodies],
  )
  names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in hinge]
  rng = np.array([model.jnt_range[j] for j in hinge])
  return model, ent, names, rng


def load_clip(path: str, bt, joint_names, stride: int) -> dict:
  with np.load(path, allow_pickle=True) as d:
    ee = derive_ee_fields(d, bt)
    joints = np.asarray(d["smpl_joints_w"])
    contact = np.asarray(d["foot_contact"]) > 0.5
    src = [str(n) for n in d["init_g1_joint_names"]]
    perm = [src.index(n) for n in joint_names]
    q0 = np.asarray(d["init_g1_joint_pos"])[:, perm]
    quat0 = np.asarray(d["init_g1_root_quat"])
    z0 = np.asarray(d["init_g1_root_height"])
    root = np.asarray(d["root_pos_w"])
    v = np.asarray(d["root_lin_vel_w"])
  ankle_z = joints[:, [7, 8], 2]
  stance = np.array([np.median(ankle_z[contact[:, i], i]) if contact[:, i].any() else ankle_z[:, i].min() for i in (0, 1)])
  idx = np.arange(0, joints.shape[0], stride)
  return dict(
    pos=ee["h_task_pos"][idx], rot6d=ee["h_task_rot6d"][idx], contact=contact[idx], lift=(ankle_z - stance)[idx],
    q0=q0[idx], quat0=quat0[idx], root=np.concatenate([root[idx, :2], z0[idx, None]], -1),
    pelvis_speed=np.linalg.norm(v[:, :2], axis=-1),
  )


def solve_batch(clips: list[dict], fk: TorchFK, links, bt, rng, ankle_h: float, args, device) -> list[dict]:
  t = lambda k, dt=torch.float32: torch.as_tensor(np.concatenate([c[k] for c in clips]), dtype=dt, device=device)  # noqa: E731
  pos, rot = t("pos"), rot6d_to_matrix(t("rot6d"))
  contact, lift = t("contact", torch.bool), t("lift")
  q_ref, quat0, root0 = t("q0"), t("quat0"), t("root")
  N = pos.shape[0]
  lens = [c["pos"].shape[0] for c in clips]
  # neighbour pairs inside each clip (smoothness)
  first = torch.zeros(N, dtype=torch.bool, device=device)
  first[torch.as_tensor(np.cumsum([0] + lens[:-1]), device=device)] = True
  nb = ~first[1:]
  lo, hi = (torch.as_tensor(rng[:, i], dtype=torch.float32, device=device) for i in (0, 1))
  q = q_ref.clone().clamp(lo, hi).requires_grad_(True)
  p = root0.clone().requires_grad_(True)
  dr = torch.zeros(N, 3, device=device, requires_grad=True)
  opt = torch.optim.Adam([q, p, dr], lr=args.lr)
  w, f, tt = list(TASK_WRISTS), list(TASK_FEET), TASK_TORSO
  zeros3 = torch.zeros(N, 3, device=device)
  zq = torch.zeros_like(q)

  def forward():
    qw = quat_mul(quat0, quat_from_rotvec(dr))
    lp, lq, _, _ = fk.forward(p, qw, zeros3, zeros3, q, zq)
    lp, lq = lp[:, links], lq[:, links]
    return bone.robot_task_poses(lp, bone.quat_wxyz_to_matrix(lq), bt)

  foot_z = torch.where(contact, torch.full_like(lift, ankle_h), ankle_h + lift.clamp(min=0.0))
  for it in range(args.iters):
    rp, rr = forward()
    e_w = ((rp[:, w] - pos[:, w]) ** 2).sum(-1) / 0.02**2 + rot_err2(rr[:, w], rot[:, w]) / 0.2**2
    e_t = ((rp[:, tt, :2] - pos[:, tt, :2]) ** 2).sum(-1) / 0.03**2 + rot_err2(rr[:, tt], rot[:, tt]) / 0.2**2
    e_f = ((rp[:, f, :2] - pos[:, f, :2]) ** 2).sum(-1) / 0.03**2 + (rp[:, f, 2] - foot_z) ** 2 / 0.02**2
    e_f = e_f + contact.float() * rot_err2(rr[:, f], rot[:, f]) / 0.2**2
    e_reg = ((q - q_ref) ** 2).sum(-1) / 0.5**2
    x = torch.cat([q, p, dr], -1)
    e_smooth = ((x[1:] - x[:-1]) ** 2).sum(-1)[nb].sum() / 0.1**2 if N > 1 else 0.0
    loss = (e_w.sum(-1) + e_t + e_f.sum(-1) + args.reg * e_reg).sum() + args.smooth * e_smooth
    opt.zero_grad(set_to_none=True)
    (loss / N).backward()
    opt.step()
    with torch.no_grad():
      q.clamp_(lo, hi)
  with torch.no_grad():
    rp, rr = forward()
    wrist = (rp[:, w] - pos[:, w]).norm(dim=-1).amax(-1)
    wrist_rot = torch.rad2deg(2 * torch.asin((rot_err2(rr[:, w], rot[:, w]).sqrt() / 2).clamp(max=1.0))).amax(-1)
    torso = (rp[:, tt, :2] - pos[:, tt, :2]).norm(dim=-1)
    foot = torch.cat([rp[:, f, :2] - pos[:, f, :2], (rp[:, f, 2] - foot_z)[..., None]], -1).norm(dim=-1).amax(-1)
    qd = torch.zeros(N, device=device)
    qd[1:] = ((q[1:] - q[:-1]).abs().amax(-1) / (args.stride / 50.0)) * nb
  out, s = [], 0
  for c, n in zip(clips, lens):
    sl = slice(s, s + n)
    pc = lambda x, k: float(np.percentile(x[sl].cpu().numpy(), k))  # noqa: E731
    out.append(dict(
      frames=n, wrist_p50=pc(wrist, 50), wrist_p95=pc(wrist, 95), wrist_max=float(wrist[sl].max()), wrist_rot_p95=pc(wrist_rot, 95),
      wrist_5cm=float((wrist[sl] < 0.05).float().mean()), torso_p95=pc(torso, 95), foot_p95=pc(foot, 95), qd_p99=pc(qd, 99),
      pelvis_speed_p95=float(np.percentile(c["pelvis_speed"], 95)),
    ))
    s += n
  return out


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--motion_list", required=True, help="human npz paths, one per line (prepare_init_states.py output)")
  ap.add_argument("--out_dir", required=True, help="writes ee_report.tsv and ee_list.txt")
  ap.add_argument("--stride", type=int, default=5, help="solve every stride-th frame (50 Hz clips: 5 = 10 Hz)")
  ap.add_argument("--iters", type=int, default=400)
  ap.add_argument("--lr", type=float, default=0.02)
  ap.add_argument("--reg", type=float, default=0.1, help="pull towards the bone-direction IK pose")
  ap.add_argument("--smooth", type=float, default=0.05, help="pull towards the neighbouring frames' solutions")
  ap.add_argument("--batch_frames", type=int, default=40000)
  ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
  ap.add_argument("--max_wrist_p95", type=float, default=0.05, help="m")
  ap.add_argument("--max_torso_p95", type=float, default=0.05, help="m (xy)")
  ap.add_argument("--max_foot_p95", type=float, default=0.05, help="m")
  ap.add_argument("--max_qd_p99", type=float, default=20.0, help="rad/s, joint speed of the solution")
  ap.add_argument("--max_pelvis_speed_p95", type=float, default=2.0, help="m/s, human pelvis horizontal speed")
  args = ap.parse_args()
  os.makedirs(args.out_dir, exist_ok=True)
  files = [ln.strip() for ln in open(os.path.expanduser(args.motion_list)) if ln.strip()]
  device = torch.device(args.device)

  model, ent, joint_names, rng = robot_model()
  bm = bone.SMPL_BONE_MAPS["g1"]
  bt = bone.BoneMapTensors(bm, bm.load_calibration(), device=device)
  links = torch.as_tensor([ent.body_names.index(n) for n in bm.robot_links], device=device)
  fk = TorchFK(model, ent, str(device))
  _, ankle_h = robot_rest_geometry(model, bm.pelvis_link, bm.foot_links, FOOT_REGEX)
  bt_cpu = bone.BoneMapTensors(bm, bm.load_calibration(), dtype=torch.float64)

  rows, batch, names, t0 = [], [], [], time.time()

  def flush():
    if batch:
      for nm, r in zip(names, solve_batch(batch, fk, links, bt, rng, ankle_h, args, device)):
        rows.append((nm, r))
      batch.clear()
      names.clear()
      print(f"[ee_ik] {len(rows)}/{len(files)} clips ({time.time() - t0:.0f}s)", flush=True)

  for path in files:
    try:
      c = load_clip(path, bt_cpu, joint_names, args.stride)
    except Exception as exc:  # noqa: BLE001
      print(f"[skip] {os.path.basename(path)}: {exc!r}")
      continue
    batch.append(c)
    names.append(path)
    if sum(x["pos"].shape[0] for x in batch) >= args.batch_frames:
      flush()
  flush()

  keys = list(rows[0][1].keys())
  limits = dict(wrist_p95=args.max_wrist_p95, torso_p95=args.max_torso_p95, foot_p95=args.max_foot_p95, qd_p99=args.max_qd_p99,
                pelvis_speed_p95=args.max_pelvis_speed_p95)
  keep, why_n = [], {k: 0 for k in limits}
  with open(os.path.join(args.out_dir, "ee_report.tsv"), "w") as fh:
    fh.write("\t".join(["file", *keys, "pass", "why"]) + "\n")
    for path, r in rows:
      why = [k for k, lim in limits.items() if r[k] > lim]
      for k in why:
        why_n[k] += 1
      if not why:
        keep.append(path)
      fh.write("\t".join([path, *(f"{r[k]:.4g}" for k in keys), str(int(not why)), ",".join(why)]) + "\n")
  with open(os.path.join(args.out_dir, "ee_list.txt"), "w") as fh:
    fh.write("\n".join(keep) + "\n")
  med = {k: float(np.median([r[k] for _, r in rows])) for k in keys}
  print(f"[ee_ik] {len(keep)}/{len(rows)} clips pass {limits}; failing per criterion {why_n}")
  print("[ee_ik] medians: " + ", ".join(f"{k} {v:.3g}" for k, v in med.items()))
  print(f"[ee_ik] -> {args.out_dir}/ee_report.tsv, ee_list.txt ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
  main()
