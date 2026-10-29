#!/usr/bin/env python3
"""World end-effector errors of human-reference checkpoints on one clip set: ee_mode policies (at given alphas) vs H4.

Every checkpoint is evaluated like scripts/eval_ckpt.py (clips from their first frame, no noise / DR, deterministic
policy), and every run gets the same end-effector metrics, whatever the policy was trained for (an H4 policy has no EE
heads): the task links of the robot against the human's, unscaled, the human world mapped onto the episode by the
clip's first-frame pelvis xy (the reset puts the robot pelvis there, on the human heading), no drift --
  wrist_err (m, mean of both), wrist_5cm (fraction of wrist samples within 5 cm), wrist_rot_deg, torso_xy_err (m),
  torso_rot_deg, foot_xy_err (m, feet relative to the torso point in the heading frame), and the wrist / torso-xy errors
  after the first 5 s of a clip (*_t5: world drift accumulates there for a policy without world feedback).
Besides: failures, the H4 bone / root metrics of the heads (comparable across all runs), normalized return.

  cd ~/supermimic_ws/mjlab && uv run python ~/Supermimic/scripts/human/eval_ee_compare.py \\
      --ckpt H4=<h4 ckpt> --ckpt e1=<ee ckpt> ... --alphas 1,0 --eval_clip_names <names.txt> --out cmp.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os

import numpy as np
import torch

T5 = 250  # frames (5 s at 50 Hz)


def ee_metric_fn(ev, heads_ee):
  """Evaluator.metric_fn: EE errors of next_state against the human task links of the target frames."""
  from supermimic.human.heads import COND_DIM

  lib = ev.lib
  lib.ensure_ee(ev.cmd.bone_map)

  def fn(next_state, frame, env_clip):
    g = lib.gather(frame, fields=("h_task_pos", "h_task_rot6d", "h_yaw", "h_contact"))
    cond = torch.zeros(frame.shape[0], COND_DIM, device=frame.device)
    cond[:, 0], cond[:, 1] = 1.0, 1.0
    start = lib.clip_start.to(frame.device)[env_clip]
    cond[:, 2:4] = lib.gather(start, fields=("h_root_xy",))["h_root_xy"]
    e = heads_ee.ee_errors(heads_ee.kin(next_state), {**g, "cond": cond})
    late = ((frame - start) >= T5).float()
    wrist = e["wrist_pos"].sqrt()
    deg = lambda x: torch.rad2deg(2 * torch.asin((x.clamp(min=0).sqrt() / 2).clamp(max=1.0)))  # noqa: E731
    return {
      "cmp_wrist_err": wrist.mean(-1), "cmp_wrist_5cm": (wrist < 0.05).float().mean(-1), "cmp_wrist_rot_deg": deg(e["wrist_rot"]).mean(-1),
      "cmp_torso_xy_err": e["torso_xy"].sqrt(), "cmp_torso_rot_deg": deg(e["torso_rot"]), "cmp_foot_xy_err": e["foot_xy"].sqrt().mean(-1),
      "cmp_late": late, "cmp_wrist_err_t5": wrist.mean(-1) * late, "cmp_torso_xy_err_t5": e["torso_xy"].sqrt() * late,
    }

  return fn


def run_one(path: str, alpha: float | None, names_txt: str, args) -> dict:
  from supermimic.evaluate import Evaluator
  from supermimic.heads import HeadsCfg
  from supermimic.human.heads import HumanHeads
  from supermimic.motions import read_name_list, resolve_clip_names
  from supermimic.policy import Policy

  ck = torch.load(path, map_location=args.device, weights_only=False)
  cfg, d = ck["cfg"], ck["dims"]
  assert cfg.get("reference") == "human", f"{path}: not a human-reference checkpoint"
  files = cfg["motion_files"]
  heads_cfg = HeadsCfg.from_dict({k: v for k, v in cfg["heads"].items() if k != "extra"})
  if alpha is not None:
    heads_cfg = dataclasses.replace(heads_cfg, eval_alpha=alpha)
  kw = {"input_clip": float(cfg["input_clip"])} if "input_clip" in cfg else {}
  policy = Policy(d["proprio"], d["hist"], d["goal_frame"], d["K"], d["nj"], **kw).to(args.device)
  policy.load_state_dict(ck["policy"])
  eval_clips = tuple(resolve_clip_names([os.path.basename(f) for f in files], read_name_list(names_txt), label="eval_clip_names")[0])
  ev = Evaluator(files, heads_cfg, args.device, eval_clips=eval_clips, num_clips_total=len(files), max_envs=args.eval_envs,
                 compile_=not args.no_compile, cache_dir=cfg.get("cache_dir", ""), robot=cfg.get("robot", "g1"),
                 robot_variant=cfg.get("robot_variant", ""), reference="human")
  h = ev.heads
  heads_ee = HumanHeads(h.fk, h.links.tolist(), h.q0, h.soft_lim, dataclasses.replace(heads_cfg, ee_mode=True), h.rest_h, h.ankle_h)
  ev.metric_fn = ee_metric_fn(ev, heads_ee)
  res = ev.run(policy)
  sc = res["scalars"]
  pcm = res["per_clip_metrics"]
  out = {k: sc[k] for k in ("failed", "return_mean") if k in sc}
  out.update({k: v for k, v in sc.items() if k.startswith(("bone_", "root_vel", "heading", "height", "foot_lift"))})
  out.update({k[4:]: v for k, v in sc.items() if k.startswith("cmp_") and not k.endswith(("_t5", "_late"))})
  late = np.array(list(pcm["cmp_late"].values()))
  m = late > 0
  for k in ("cmp_wrist_err_t5", "cmp_torso_xy_err_t5"):
    v = np.array(list(pcm[k].values()))
    out[k[4:]] = float((v[m] / late[m]).mean()) if m.any() else float("nan")
  out["wrist_err_clip_p90"] = float(np.percentile(list(pcm["cmp_wrist_err"].values()), 90))
  out["it"] = ck["it"]
  out["clips"] = res["num_clips"]
  out["clips_t5"] = int(m.sum())
  del ev
  torch.cuda.empty_cache()
  return out


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--ckpt", action="append", required=True, help="label=path (repeat)")
  ap.add_argument("--alphas", default="1,0", help="eval_alpha values for ee_mode checkpoints")
  ap.add_argument("--eval_clip_names", required=True, help="npz basenames (one per line), e.g. prepare_ee_ik's eval_ee512.txt")
  ap.add_argument("--eval_envs", type=int, default=512)
  ap.add_argument("--device", default="cuda:0")
  ap.add_argument("--no_compile", action="store_true")
  ap.add_argument("--out", required=True, help="json: {label[@alpha]: metrics}")
  args = ap.parse_args()
  results = json.load(open(args.out)) if os.path.exists(args.out) else {}
  for spec in args.ckpt:
    label, path = spec.split("=", 1)
    ck_cfg = torch.load(path, map_location="cpu", weights_only=False)["cfg"]["heads"]
    alphas = [float(a) for a in args.alphas.split(",")] if ck_cfg.get("ee_mode") else [None]
    for a in alphas:
      key = label if a is None else f"{label}@a{a:g}"
      results[key] = run_one(path, a, args.eval_clip_names, args)
      print(f"[eval_ee_compare] {key}: " + ", ".join(f"{k} {v:.4g}" for k, v in results[key].items() if isinstance(v, float)), flush=True)
      with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)


if __name__ == "__main__":
  main()
