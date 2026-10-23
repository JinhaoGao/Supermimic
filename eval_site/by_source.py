"""Per-source success of two site runs on a v4 dataset.  python by_source.py <dataset_dir> <run_a> <run_b>"""
import collections
import csv
import os
import sys

root, ra, rb = sys.argv[1:4]
src = {}
for dp, _, fs in os.walk(root):
    for f in fs:
        if f.endswith(".npz"):
            rel = os.path.relpath(os.path.join(dp, f), root).split(os.sep)
            src[f[:-4]] = rel[0]
def load(run):
    rows = list(csv.DictReader(open(f"/home/ubuntu/work/NOetix_GMT_eval/storage/runs/{run}/evaluation_per_segment.csv")))
    out = collections.defaultdict(lambda: [0, 0, 0])
    for r in rows:
        s = src.get(r["motion_name"], "?")
        out[s][0] += 1
        out[s][1] += r["success"] == "True"
        out[s][2] += r["termination_reason"] == "body_pos"
    return out
a, b = load(ra), load(rb)
print(f"{'source':28s} {'n':>5s} {'ours%':>7s} {'EE-term':>8s} {'leader%':>8s}")
for s in sorted(a, key=lambda s: -a[s][0]):
    n, sa, ee = a[s]
    nb, sb, _ = b.get(s, [0, 0, 0])
    print(f"{s:28s} {n:5d} {100 * sa / n:7.1f} {ee:8d} {100 * sb / max(nb, 1):8.1f}")
