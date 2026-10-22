import collections
import csv
import sys

for f in sys.argv[1:]:
    rows = list(csv.DictReader(open(f)))
    reasons = collections.Counter(r["termination_reason"] for r in rows)
    fails = [r for r in rows if r["success"] != "True"]
    early = sum(1 for r in fails if r.get("failure_step") not in (None, "", "nan") and float(r["failure_step"]) < 50)
    lr = sorted(float(r["length_ratio"]) for r in fails)
    print(f"{f.split('/')[-2]:22s} n={len(rows)} success={sum(r['success'] == 'True' for r in rows)} reasons={dict(reasons)} "
          f"fails<1s={early} median length_ratio of fails={lr[len(lr) // 2] if lr else float('nan'):.2f}")
