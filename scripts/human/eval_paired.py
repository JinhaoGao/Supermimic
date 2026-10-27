#!/usr/bin/env python3
"""Score a checkpoint of either reference against the HUMAN motion, on the paired clips (pair_with_retarget.py).

Both runs are evaluated with the same protocol (Appendix F.1: every clip once from frame 0, no DR / noise,
deterministic policy) on their own reference -- the baseline on the retargeted robot clips with its own terminations,
the human-reference run on the human clips -- and every live step is scored against the human frame at the same time
with the human heads' metrics (supermimic/human/heads.py HumanHeads.metrics): bone-direction angles (global torso,
local torso / leg / arm), k-scaled root velocity, yaw rate, heading and normalized pelvis height errors. Failure =
the run's own training termination (the two runs' criteria differ; --human_fail additionally counts, for the baseline,
the steps where the human termination test would have fired -> fail_human). Summaries use the time-aligned pairs.

  cd ~/supermimic_ws/mjlab && uv run python ~/Supermimic/scripts/human/eval_paired.py --ckpt <model.pt> \\
      --pairs ~/datasets/paired_amass/pairs.tsv [--subset ~/datasets/paired_amass/flag_kp_vs_rt.txt] --out res.json
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch


def read_pairs(path: str) -> list[dict]:
  rows = []
  with open(os.path.expanduser(path)) as f:
    head = f.readline().rstrip("\n").split("\t")
    for line in f:
      if line.strip():
        rows.append(dict(zip(head, line.rstrip("\n").split("\t"))))
  return rows


def main():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--ckpt", required=True)
  p.add_argument("--pairs", required=True, help="pairs.tsv of pair_with_retarget.py")
  p.add_argument("--subset", action="append", default=[], help="name list (human basenames); summarized separately (repeatable)")
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--eval_envs", type=int, default=1024)
  p.add_argument("--no_compile", action="store_true")
  p.add_argument("--robot_variant", default=None, help="default: the run's")
  p.add_argument("--cache_dir", default=None, help="default: the run's")
  p.add_argument("--out", default=None)
  args = p.parse_args()

  from supermimic.evaluate import Evaluator
  from supermimic.fk import TorchFK
  from supermimic.heads import HeadsCfg
  from supermimic.human import bone_map as bone
  from supermimic.human.command import robot_rest_geometry
  from supermimic.human.heads import STEP_FIELDS, HumanHeads, HumanHeadsCfg
  from supermimic.human.motions import HumanMotionLibrary
  from supermimic.policy import Policy

  ck = torch.load(args.ckpt, map_location=args.device, weights_only=False)
  cfg, d = ck["cfg"], ck["dims"]
  reference = cfg.get("reference", "robot")
  hk = {k: v for k, v in cfg["heads"].items() if k != "extra"}
  if reference == "robot":
    hk["goal_anchor_pos"] = d["goal_frame"] == 2 * d["nj"] + 9
  heads_cfg = HeadsCfg.from_dict(hk)
  kw = {"input_clip": float(cfg["input_clip"])} if "input_clip" in cfg else {}
  policy = Policy(d["proprio"], d["hist"], d["goal_frame"], d["K"], d["nj"], **kw).to(args.device)
  policy.load_state_dict(ck["policy"])

  pairs = read_pairs(args.pairs)
  files = [r["human" if reference == "human" else "robot"] for r in pairs]
  cache_dir = args.cache_dir if args.cache_dir is not None else cfg.get("cache_dir", "")
  ev = Evaluator(files, heads_cfg, args.device, max_envs=args.eval_envs, compile_=not args.no_compile, cache_dir=cache_dir,
                 robot=cfg.get("robot", "g1"), robot_variant=args.robot_variant if args.robot_variant is not None else cfg.get("robot_variant", ""),
                 reference=reference)
  # library clip name (as stored) -> pair name
  to_pair = {os.path.basename(r["human" if reference == "human" else "robot"]): r["name"] for r in pairs}

  if reference == "robot":  # human-side metrics of the baseline: own FK over the bone-map links, human library, same frames
    hcfg = HumanHeadsCfg()
    sim, cmd = ev.sim, ev.cmd
    robot = sim.robot
    bm = bone.SMPL_BONE_MAPS[hcfg.bone_map]
    hlib = HumanMotionLibrary([r["human"] for r in pairs], cfg.get("robot", "g1"), tuple(robot.joint_names), bm, args.device,
                              cache_dir=cache_dir or None)
    links = [robot.body_names.index(n) for n in bm.robot_links]
    fk = TorchFK(ev.env.sim.mj_model, robot, args.device, needed_bodies=links)
    regex = ev.env.cfg.events["foot_friction"].params["asset_cfg"].geom_names
    rest_h, ankle_h = robot_rest_geometry(ev.env.sim.mj_model, bm.pelvis_link, bm.foot_links, regex if isinstance(regex, str) else "|".join(regex))
    hh = HumanHeads(fk, links, robot.data.default_joint_pos[0], robot.data.soft_joint_pos_limits[0], hcfg, rest_h, ankle_h)
    hname = {c.name: i for i, c in enumerate(hlib.clips)}
    pair_h = {r["name"]: hname[os.path.basename(r["human"])] for r in pairs if os.path.basename(r["human"]) in hname}
    lag = {r["name"]: int(r.get("lag", 0) or 0) for r in pairs}  # human frame = robot frame + lag
    rlib = ev.lib
    h_of_r = torch.tensor([pair_h.get(to_pair.get(c.name, ""), -1) for c in rlib.clips], device=args.device)
    lag_of_r = torch.tensor([lag.get(to_pair.get(c.name, ""), 0) for c in rlib.clips], device=args.device)
    if bool((h_of_r < 0).any()):
      print(f"[eval_paired] WARNING {int((h_of_r < 0).sum())} robot clips without a human clip (scored against clip 0)")
    h_of_r = h_of_r.clamp(min=0)

    def metric_fn(next_state, frame, env_clip):
      hc = h_of_r[env_clip]
      off = (frame - rlib.clip_start[env_clip] + lag_of_r[env_clip]).clamp(min=0)
      hf = torch.minimum(hlib.clip_start[hc] + off, hlib.clip_end[hc] - 1)
      ref = hlib.gather(hf, fields=STEP_FIELDS)
      kin = hh.kin(next_state)
      m = hh.metrics(kin, next_state, ref)
      m["fail_human"] = hh.termination(kin, next_state, ref)[0].float()  # per-step: clip mean = share of steps
      return m

    ev.metric_fn = metric_fn

  res = ev.run(policy)
  pcm = res["per_clip_metrics"]
  per = {to_pair.get(n, n): {"failed": res["per_clip_failed"][n], "return": res["per_clip"][n], **{k: v[n] for k, v in pcm.items()}}
         for n in res["per_clip"]}

  def summary(names) -> dict:
    rows = [per[n] for n in names if n in per]
    if not rows:
      return {"clips": 0}
    keys = [k for k in rows[0] if k != "failed"]
    out = {"clips": len(rows), "failed": int(sum(r["failed"] for r in rows)), "failed_frac": float(np.mean([r["failed"] for r in rows]))}
    out.update({k: float(np.mean([r[k] for r in rows])) for k in keys})
    return out

  # Scores only on time-aligned pairs (pairs.tsv "aligned"; older files without the column: all); unaligned separately.
  aligned = {r["name"] for r in pairs if r.get("aligned", "1") == "1"}
  report = {"ckpt": args.ckpt, "it": ck["it"], "reference": reference, "all": summary([n for n in per if n in aligned]),
            "unaligned (frames may be offset)": summary([n for n in per if n not in aligned])}
  for sub in args.subset:
    names = [l.strip().replace(".npz", "") for l in open(os.path.expanduser(sub)) if l.strip()]
    report[os.path.basename(sub)] = summary([n for n in names if n in aligned])
  print(json.dumps(report, indent=2))
  if args.out:
    with open(args.out, "w") as f:
      json.dump({**report, "per_clip": per}, f, indent=2)


if __name__ == "__main__":
  main()
