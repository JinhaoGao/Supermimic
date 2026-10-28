#!/usr/bin/env python3
"""Score the retargeted reference itself against the human motion (no policy): what the baseline is trained to track.

For every pair of pairs.tsv the retargeted robot clip is posed kinematically (root = body 0 pose, joints from
joint_pos, mjlab G1 FK) and compared frame by frame (human frame = robot frame + lag) with HumanHeads.metrics: bone
angles (global torso / arm / leg, local torso / leg / arm), heading and normalized pelvis height errors (velocity terms
are skipped: the reference is posed, not simulated). Summaries over the time-aligned pairs, plus --subset lists.

  cd ~/supermimic_ws/mjlab && uv run python ~/Supermimic/scripts/human/score_retarget.py \\
      --pairs ~/datasets/paired_amass/pairs.tsv --subset ~/datasets/paired_amass/ambig_top.txt --out retarget_vs_human.json
"""

from __future__ import annotations

import argparse
import json
import os
import types

import mujoco
import numpy as np
import torch


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--pairs", required=True)
  ap.add_argument("--subset", action="append", default=[])
  ap.add_argument("--out", default=None)
  ap.add_argument("--device", default="cpu")
  args = ap.parse_args()

  from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec

  from supermimic.fk import TorchFK
  from supermimic.human import bone_map as bone
  from supermimic.human.command import robot_rest_geometry
  from supermimic.human.heads import STEP_FIELDS, HumanHeads, HumanHeadsCfg
  from supermimic.human.motions import derive_fields
  from supermimic.motions import FALLBACK_JOINT_NAMES

  model = get_spec().compile()
  bodies = list(range(1, model.nbody))
  hinge = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
  ent = types.SimpleNamespace(indexing=types.SimpleNamespace(body_ids=torch.tensor(bodies), joint_ids=torch.tensor(hinge)),
                              body_names=[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in bodies])
  joint_names = tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in hinge)
  bm = bone.SMPL_BONE_MAPS["g1"]
  links = [ent.body_names.index(n) for n in bm.robot_links]
  fk = TorchFK(model, ent, args.device, needed_bodies=links)
  rest, ankle = robot_rest_geometry(model, bm.pelvis_link, bm.foot_links, r"^(left|right)_foot[1-7]_collision$")
  J = len(joint_names)
  heads = HumanHeads(fk, links, torch.zeros(J), torch.tensor([[-9.0, 9.0]] * J), HumanHeadsCfg(), rest, ankle)
  bt64 = bone.BoneMapTensors(bm, bm.load_calibration(), dtype=torch.float64)

  with open(os.path.expanduser(args.pairs)) as f:
    head = f.readline().rstrip("\n").split("\t")
    pairs = [dict(zip(head, l.rstrip("\n").split("\t"))) for l in f if l.strip()]
  per = {}
  for i, pr in enumerate(pairs):
    r = np.load(pr["robot"])
    src = [str(x) for x in r["joint_names"]] if "joint_names" in r.files else list(FALLBACK_JOINT_NAMES[(r["joint_pos"].shape[1], r["body_pos_w"].shape[1])])
    q = r["joint_pos"][:, [src.index(n) for n in joint_names]]
    T = q.shape[0]
    s = torch.zeros(T, 13 + 2 * J)
    s[:, 0:3] = torch.as_tensor(r["body_pos_w"][:, 0])
    s[:, 3:7] = torch.as_tensor(r["body_quat_w"][:, 0])
    s[:, 13 : 13 + J] = torch.as_tensor(q)
    with np.load(pr["human"], allow_pickle=True) as h:
      hf = derive_fields(h, bt64, "g1", joint_names)
    lag = int(pr.get("lag", 0) or 0)
    idx = np.clip(np.arange(T) + lag, 0, hf["h_feat"].shape[0] - 1)
    ref = {k: torch.as_tensor(hf[k][idx]) for k in STEP_FIELDS}
    m = heads.metrics(heads.kin(s), s, ref)
    per[pr["name"]] = {k: float(v.mean()) for k, v in m.items() if k not in ("root_vel_err", "yaw_rate_err")}
    if (i + 1) % 200 == 0:
      print(f"{i + 1}/{len(pairs)}", flush=True)

  aligned = {p["name"] for p in pairs if p.get("aligned", "1") == "1"}

  def summary(names):
    rows = [per[n] for n in names if n in per]
    return {"clips": len(rows), **{k: float(np.mean([r[k] for r in rows])) for k in rows[0]}} if rows else {"clips": 0}

  report = {"all": summary([n for n in per if n in aligned])}
  for sub in args.subset:
    report[os.path.basename(sub)] = summary([l.strip().replace(".npz", "") for l in open(os.path.expanduser(sub)) if l.strip() and l.strip().replace(".npz", "") in aligned])
  print(json.dumps(report, indent=2))
  if args.out:
    json.dump({**report, "per_clip": per}, open(args.out, "w"), indent=2)


if __name__ == "__main__":
  main()
