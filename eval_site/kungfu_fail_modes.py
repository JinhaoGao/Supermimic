"""How kungfu sub-groups fail: termination reason, failure point, and the leader's outcome on the same clips."""
import collections
import csv
import os
import re
import sys

root, ours, lead = sys.argv[1:4]
rows = {}
for dp, _, fs in os.walk(os.path.join(root, "noetix_kungfu")):
    for f in fs:
        if f.endswith(".npz"):
            rows[f[:-4]] = os.path.basename(dp)
def load(run):
    return {r["motion_name"]: r for r in csv.DictReader(open(f"/home/ubuntu/work/NOetix_GMT_eval/storage/runs/{run}/evaluation_per_segment.csv")) if r["motion_name"] in rows}
a, b = load(ours), load(lead)
groups = [("腾空", r"tengkong|腾空|xuanfeng"), ("翻腾", r"fan|翻"), ("踢腿", r"tui|jiao|ceti|踢|腿|lian")]
for g, pat in groups:
    names = [n for n in a if re.search(pat, n.lower())]
    fails = [n for n in names if a[n]["success"] != "True"]
    reasons = collections.Counter(a[n]["termination_reason"] for n in fails)
    lr = sorted(float(a[n]["length_ratio"]) for n in fails)
    lead_ok = sum(1 for n in fails if b.get(n, {}).get("success") == "True")
    print(f"{g}: {len(names)} clips, ours failed {len(fails)} (leader ok on {lead_ok} of them); reasons {dict(reasons)}; "
          f"failure point median {lr[len(lr)//2] if lr else float('nan'):.2f} of the clip")
    for n in sorted(fails)[:6]:
        print(f"    {a[n]['termination_reason']:9s} at {float(a[n]['length_ratio']):.2f} | leader {b.get(n, {}).get('success', '?'):5s} {b.get(n, {}).get('termination_reason', ''):15s} | {n[:60]}")
