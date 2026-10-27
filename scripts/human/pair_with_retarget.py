#!/usr/bin/env python3
"""Pair human (SMPL) npz files with retargeted robot npz files of the same clips -> matched motion lists.

The human-reference run and the baseline (robot reference, retargeted motion) train on the same clips; the paired
evaluation (eval_paired.py) scores both against the human motion frame by frame. Names: human "<clip>.npz" (from
prepare_smpl_motion.py) <-> retargeted "<clip>_stageii.npz" (or "<clip>.npz").

Checks per pair (failures are dropped and listed): both files load, the human file carries the robot reset states
(init_<robot>_*), the frame counts differ by at most --max_len_diff. The time offset is the lag (searched over
+-max_lag frames) at which the horizontal pelvis speeds correlate best; a clip is "aligned" when that correlation is
>= --min_corr or it barely moves horizontally (lag 0). Unaligned clips (mostly low-motion gestures / stances, where the
speed is noise) stay in the training lists with lag 0 / aligned 0; eval_paired.py scores aligned clips only. The retarget pipeline (30 Hz retarget -> 50 Hz) shifts about half
of the AMASS clips by 1-4 frames against the direct 50 Hz SMPL; training does not care (each run uses its own clips),
eval_paired.py compares human frame = robot frame + lag. Optional --quality_report (smplkp_full/quality_report.csv) drops clips with any of --drop_flags.

Writes to --out_dir: human_list.txt, robot_list.txt (same order), pairs.tsv (name, paths, lengths, lag, corr, flags),
dropped.tsv, and one name list per quality flag found (e.g. flag_kp_vs_rt.txt, human basenames) for subset analysis.

  python scripts/human/pair_with_retarget.py --human_dir ~/datasets/smpl_human_50hz \\
      --retarget_dir ~/datasets/smplkp_full/g1_amass_retarget_npz --quality_report ~/datasets/smplkp_full/quality_report.csv \\
      --out_dir ~/datasets/paired_amass
"""

from __future__ import annotations

import argparse
import csv
import glob
import os

import numpy as np


def hspeed(pos: np.ndarray, fps: float) -> np.ndarray:
  return np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=-1) * fps


def best_lag(a: np.ndarray, b: np.ndarray, max_lag: int) -> tuple[int, float]:
  best = (0, -2.0)
  for lag in range(-max_lag, max_lag + 1):
    x, y = (a[lag:], b) if lag >= 0 else (a, b[-lag:])
    n = min(len(x), len(y))
    if n < 10 or x[:n].std() < 1e-6 or y[:n].std() < 1e-6:
      continue
    c = float(np.corrcoef(x[:n], y[:n])[0, 1])
    if c > best[1]:
      best = (lag, c)
  return best


def main():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--human_dir", required=True)
  p.add_argument("--retarget_dir", required=True)
  p.add_argument("--out_dir", required=True)
  p.add_argument("--robot", default="g1")
  p.add_argument("--quality_report", default="")
  p.add_argument("--drop_flags", default="horizontal,elevated,low_pelvis",
                 help="quality flags to drop (kp_vs_rt is kept on purpose: the retarget-ambiguity subset)")
  p.add_argument("--ambig_quantile", type=float, default=0.9,
                 help="with --quality_report: clips whose kp_vs_retarget_err_m is at or above this quantile of the paired "
                      "clips -> ambig_top.txt (the retarget-ambiguity subset for eval_paired.py --subset)")
  p.add_argument("--max_len_diff", type=int, default=2)
  p.add_argument("--max_lag", type=int, default=6)
  p.add_argument("--min_corr", type=float, default=0.8)
  p.add_argument("--still_speed", type=float, default=0.05, help="m/s: clips slower than this on both sides skip the lag test")
  args = p.parse_args()

  flags: dict[str, list[str]] = {}
  kp_err: dict[str, float] = {}
  if args.quality_report:
    with open(os.path.expanduser(args.quality_report)) as f:
      for row in csv.DictReader(f):
        flags[row["name"]] = [x for x in row["flags"].split("|") if x]
        if row.get("kp_vs_retarget_err_m"):
          kp_err[row["name"]] = float(row["kp_vs_retarget_err_m"])
  drop = {x for x in args.drop_flags.split(",") if x}
  rt = {os.path.basename(f).replace("_stageii.npz", ".npz"): f for f in glob.glob(os.path.join(os.path.expanduser(args.retarget_dir), "*.npz"))}
  pairs, dropped = [], []
  for hp in sorted(glob.glob(os.path.join(os.path.expanduser(args.human_dir), "*.npz"))):
    base = os.path.basename(hp)
    name = base[:-4]
    fl = flags.get(name, [])
    if drop & set(fl):
      dropped.append((name, "quality:" + "|".join(sorted(drop & set(fl)))))
      continue
    if base not in rt:
      dropped.append((name, "no retarget file"))
      continue
    try:
      h = np.load(hp, allow_pickle=True)
      r = np.load(rt[base], allow_pickle=True)
      if f"init_{args.robot}_joint_pos" not in h.files:
        dropped.append((name, "no reset states (run prepare_init_states.py)"))
        continue
      th, tr = h["smpl_joints_w"].shape[0], r["joint_pos"].shape[0]
      fps = float(np.asarray(h["fps"]).reshape(-1)[0])
      if abs(float(np.asarray(r["fps"]).reshape(-1)[0]) - fps) > 1e-3:
        dropped.append((name, "fps differs"))
        continue
      if abs(th - tr) > args.max_len_diff:
        dropped.append((name, f"length {th} vs {tr}"))
        continue
      sh = hspeed(h["root_pos_w"], fps)
      sr = hspeed(r["body_pos_w"][:, 0], fps)  # body 0 = pelvis (Isaac and mjlab orders)
      if max(sh.mean(), sr.mean()) < args.still_speed:
        lag, corr = 0, float("nan")
      else:
        lag, corr = best_lag(sh, sr, args.max_lag)
    except Exception as e:  # noqa: BLE001
      dropped.append((name, f"load error {e!r}"))
      continue
    aligned = bool(np.isnan(corr) or corr >= args.min_corr)
    pairs.append((name, os.path.abspath(hp), os.path.abspath(rt[base]), th, tr, lag if aligned else 0, corr, "|".join(fl), int(aligned)))

  os.makedirs(args.out_dir, exist_ok=True)
  with open(os.path.join(args.out_dir, "human_list.txt"), "w") as f:
    f.write("".join(x[1] + "\n" for x in pairs))
  with open(os.path.join(args.out_dir, "robot_list.txt"), "w") as f:
    f.write("".join(x[2] + "\n" for x in pairs))
  with open(os.path.join(args.out_dir, "pairs.tsv"), "w") as f:
    f.write("name\thuman\trobot\tT_human\tT_robot\tlag\tcorr\tflags\taligned\n")
    f.write("".join("\t".join(map(str, x)) + "\n" for x in pairs))
  with open(os.path.join(args.out_dir, "dropped.tsv"), "w") as f:
    f.write("".join(f"{n}\t{why}\n" for n, why in dropped))
  for fl in sorted({x for pr in pairs for x in pr[7].split("|") if x}):
    with open(os.path.join(args.out_dir, f"flag_{fl}.txt"), "w") as f:
      f.write("".join(pr[0] + ".npz\n" for pr in pairs if fl in pr[7].split("|")))
  errs = [(kp_err[x[0]], x[0]) for x in pairs if x[0] in kp_err]
  if errs:
    thr = float(np.quantile([e for e, _ in errs], args.ambig_quantile))
    top = [n for e, n in errs if e >= thr]
    with open(os.path.join(args.out_dir, "ambig_top.txt"), "w") as f:
      f.write("".join(n + ".npz\n" for n in top))
    print(f"ambiguity subset: {len(top)} clips with kp_vs_retarget_err_m >= {thr:.3f} m -> ambig_top.txt")
  secs = sum(x[3] for x in pairs) / 50.0
  print(f"{len(pairs)} pairs ({secs / 3600:.2f} h; {sum(x[8] for x in pairs)} time-aligned for eval_paired), {len(dropped)} dropped -> {args.out_dir}")
  why: dict[str, int] = {}
  for _, w in dropped:
    k = w.split(":")[0] if w.startswith(("quality", "time offset")) else w.split(" ")[0]
    why[k] = why.get(k, 0) + 1
  print("dropped by reason:", why)


if __name__ == "__main__":
  main()
