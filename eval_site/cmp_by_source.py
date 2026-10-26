"""Per-source failure counts of two report dirs (or site run ids) + motions won / lost by the second."""
import collections
import csv
import glob
import os
import sys

root = "/home/ubuntu/work/NOetix_GMT_eval/data/g1_motionatlas_v4"
src = {}
for p in glob.glob(root + "/**/*.npz", recursive=True):
  rel = os.path.relpath(p, root).split(os.sep)
  src[os.path.basename(p)[:-4]] = rel[0] + "/" + rel[1]


def load(d):
  d = d if os.path.isdir(d) else f"/home/ubuntu/work/NOetix_GMT_eval/storage/runs/{d}"
  return {r["motion_name"]: r for r in csv.DictReader(open(f"{d}/evaluation_per_segment.csv"))}


a, b = load(sys.argv[1]), load(sys.argv[2])
for lab, x in (("A", a), ("B", b)):
  print(lab, "termination", dict(collections.Counter(r["termination_reason"] for r in x.values())))
fa = collections.Counter(src.get(m, "?") for m, r in a.items() if r["success"] != "True")
fb = collections.Counter(src.get(m, "?") for m, r in b.items() if r["success"] != "True")
print(f"{'source':40s} {'A fail':>7s} {'B fail':>7s}")
for s in sorted(set(fa) | set(fb), key=lambda s: -(fa[s] + fb[s]))[:12]:
  print(f"{s:40s} {fa[s]:7d} {fb[s]:7d}")
win = [m for m in a if a[m]["success"] != "True" and b[m]["success"] == "True"]
lose = [m for m in a if a[m]["success"] == "True" and b[m]["success"] != "True"]
print("B wins", len(win), "loses", len(lose))
kw = ("ground", "getup", "get_up", "lie", "prone", "fall", "break", "floor", "kneel", "crawl")
print("B wins (ground-ish):", [m[:48] for m in win if any(k in m.lower() for k in kw)])
print("B loses (ground-ish):", [m[:48] for m in lose if any(k in m.lower() for k in kw)])
