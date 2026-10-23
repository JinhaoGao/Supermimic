"""Per retarget layer (second path component, e.g. unitree_g1 / unitree_g1_phc) success of site runs.
  python by_layer.py <dataset_dir> label=run_id ..."""
import collections
import csv
import os
import sys

root = sys.argv[1]
pairs = [a.split("=", 1) for a in sys.argv[2:]]
layer, src = {}, {}
for dp, _, fs in os.walk(root):
    for f in fs:
        if f.endswith(".npz"):
            parts = os.path.relpath(os.path.join(dp, f), root).split(os.sep)
            src[f[:-4]] = parts[0]
            layer[f[:-4]] = parts[1] if len(parts) > 2 else "-"
tabs = []
for label, run in pairs:
    t = collections.defaultdict(lambda: [0, 0])
    for r in csv.DictReader(open(f"/home/ubuntu/work/NOetix_GMT_eval/storage/runs/{run}/evaluation_per_segment.csv")):
        n = r["motion_name"]
        for key in (layer.get(n, "?"), f"{src.get(n, '?')}/{layer.get(n, '?')}"):
            t[key][0] += 1
            t[key][1] += r["success"] == "True"
    tabs.append(t)
keys = sorted(tabs[0], key=lambda k: ("/" in k, k))
print(f"{'layer':42s} {'n':>5s} " + " ".join(f"{l:>9s}" for l, _ in pairs))
for k in keys:
    print(f"{k:42s} {tabs[0][k][0]:5d} " + " ".join(f"{100 * t[k][1] / max(t[k][0], 1):9.1f}" for t in tabs))
