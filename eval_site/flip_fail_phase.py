"""Where do v4 flip clips fail? Reference torso tilt at the failure frame vs the clip's inverted phase."""
import csv
import glob
import os
import re
import sys

import numpy as np

root = "/home/ubuntu/work/NOetix_GMT_eval/data/g1_motionatlas_v4"
runs = sys.argv[1:]
paths = {os.path.basename(p)[:-4]: p for p in glob.glob(root + "/**/*.npz", recursive=True)}


def is_flip(n):
  src = os.path.relpath(paths[n], root).split(os.sep)[0]
  return src == "noetix_kungfu" and re.search(r"fan|翻", n.lower())


def tilt(q):  # wxyz -> angle between body z axis and world z (deg)
  w, x, y, z = q.T
  return np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1)))


for run in runs:
  rows = [r for r in csv.DictReader(open(f"{run}/evaluation_per_segment.csv")) if r["motion_name"] in paths and is_flip(r["motion_name"])]
  print(f"== {run.split('/')[-1]}: flip clips {len(rows)}, success {sum(r['success'] == 'True' for r in rows)}")
  phase = {"before inverted": 0, "inverted (>90)": 0, "after inverted": 0, "no inversion": 0}
  for r in sorted(rows, key=lambda r: (r["success"], r["motion_name"])):
    d = np.load(paths[r["motion_name"]], allow_pickle=True)
    names = [str(b) for b in d["body_names"]] if "body_names" in d.files else []
    a = names.index("torso_link") if "torso_link" in names else 0
    fps = float(np.asarray(d["fps"]).reshape(-1)[0]) if "fps" in d.files else 50.0
    tl = tilt(d["body_quat_w"][:, a])
    T = len(tl)
    inv = np.where(tl > 90)[0]
    i0, i1 = (inv[0], inv[-1]) if len(inv) else (-1, -1)
    fr = int(round(int(r["failure_step"]) * fps / 50.0)) if r["success"] == "False" else -1
    tag = ""
    if fr >= 0:
      tag = "no inversion" if i0 < 0 else ("before inverted" if fr < i0 else ("inverted (>90)" if fr <= i1 else "after inverted"))
      phase[tag] += 1
    print(f"  {r['success']:5s} {r['termination_reason']:13s} fail {fr:4d}/{T:4d} tilt@fail {tl[min(max(fr, 0), T - 1)]:6.1f}  "
          f"inverted frames {i0:4d}-{i1:4d}  max tilt {tl.max():6.1f}  {tag:16s} {r['motion_name'][:38]}")
  print("  failure phase:", phase)
