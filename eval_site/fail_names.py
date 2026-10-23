"""Failed motions of a run within one v4 source, with the leader's outcome and our failure point.
  python fail_names.py <dataset_dir> <source> <our_run> <leader_run>"""
import collections
import csv
import os
import re
import sys

root, source, ours, lead = sys.argv[1:5]
names = set()
for dp, _, fs in os.walk(os.path.join(root, source)):
    names.update(f[:-4] for f in fs if f.endswith(".npz"))
def load(run):
    return {r["motion_name"]: r for r in csv.DictReader(open(f"/home/ubuntu/work/NOetix_GMT_eval/storage/runs/{run}/evaluation_per_segment.csv")) if r["motion_name"] in names}
a, b = load(ours), load(lead)
fails = [n for n, r in a.items() if r["success"] != "True"]
both = sum(1 for n in fails if b.get(n, {}).get("success") != "True")
print(f"{source}: {len(a)} motions, ours failed {len(fails)} (leader also failed {both}); leader failed {sum(1 for r in b.values() if r['success'] != 'True')}")
kind = collections.Counter()
for n in fails:
    key = re.split(r"[_\-0-9]", n.upper())[0][:24]
    kind[key] += 1
print("failed by name prefix:", kind.most_common(15))
print("reasons:", collections.Counter(a[n]["termination_reason"] for n in fails))
for n in sorted(fails)[:40]:
    print(f"  {a[n]['termination_reason']:9s} at {float(a[n]['length_ratio']):.2f} | leader {b.get(n, {}).get('success', '?'):5s} | {n[:90]}")
