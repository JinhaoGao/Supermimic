"""Per-source success (%) of several site runs side by side.  python by_source_multi.py <dataset_dir> label=run_id ..."""
import collections
import csv
import os
import sys

root = sys.argv[1]
pairs = [a.split("=", 1) for a in sys.argv[2:]]
src = {}
for dp, _, fs in os.walk(root):
    for f in fs:
        if f.endswith(".npz"):
            src[f[:-4]] = os.path.relpath(os.path.join(dp, f), root).split(os.sep)[0]
tables = []
for label, run in pairs:
    t = collections.defaultdict(lambda: [0, 0])
    for r in csv.DictReader(open(f"/home/ubuntu/work/NOetix_GMT_eval/storage/runs/{run}/evaluation_per_segment.csv")):
        s = src.get(r["motion_name"], "?")
        t[s][0] += 1
        t[s][1] += r["success"] == "True"
        t["ALL"][0] += 1
        t["ALL"][1] += r["success"] == "True"
    tables.append(t)
keys = sorted(tables[0], key=lambda k: (k != "ALL", -tables[0][k][0]))
print(f"{'source':26s} {'n':>5s} " + " ".join(f"{l:>10s}" for l, _ in pairs))
for k in keys:
    print(f"{k:26s} {tables[0][k][0]:5d} " + " ".join(f"{100 * t[k][1] / max(t[k][0], 1):10.1f}" for t in tables))
