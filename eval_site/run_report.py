"""Leaderboard position + failure breakdown of a site run.  python run_report.py <run_id> <dataset_id> <benchmark_id>"""
import collections
import csv
import json
import sys
import urllib.request

run_id, ds, bench = sys.argv[1:4]
base = "http://127.0.0.1:8010/api/v1"
lb = json.load(urllib.request.urlopen(f"{base}/leaderboard?mode=official&dataset_id={ds}&benchmark_id={bench}"))
mine = [r for r in lb if r["run_id"] == run_id]
print(f"leaderboard rows {len(lb)}")
for r in lb[:3] + mine:
    print(f"  #{r['rank']:>3} {r['policy_name'][:70]:70s} succ {r['success_rate']:.4f} prog {r['progress_rate']:.4f} "
          f"mpjpe {r['mpjpe_rad']:.4f} mpkpe {r['mpkpe_mm']:.2f}")
names = {"SONIC": None}
for r in lb:
    if "SONIC" in r["policy_name"].upper() and "OFFICIAL" in r["policy_name"].upper():
        print(f"  ref #{r['rank']} {r['policy_name'][:60]} succ {r['success_rate']:.4f} mpjpe {r['mpjpe_rad']:.4f} mpkpe {r['mpkpe_mm']:.2f}")
        break
rows = list(csv.DictReader(open(f"/home/ubuntu/work/NOetix_GMT_eval/storage/runs/{run_id}/evaluation_per_segment.csv")))
print("termination reasons:", dict(collections.Counter(r["termination_reason"] for r in rows)))
cat = collections.defaultdict(lambda: [0, 0])
for r in rows:
    c = r.get("category") or "?"
    cat[c][0] += 1
    cat[c][1] += r["success"] == "True"
print("per category (n, success%):", {k: (v[0], round(100 * v[1] / v[0], 1)) for k, v in sorted(cat.items(), key=lambda kv: -kv[1][0])})
