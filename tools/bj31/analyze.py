"""Headline + failure breakdown of our v4 run vs the other SuperMimic G1 runs and the G1 leader (read-only).
  python3 analyze.py <our_run_id> [out.json]"""
import collections
import csv
import json
import os
import sys
import urllib.request

ours = sys.argv[1]
out_json = sys.argv[2] if len(sys.argv) > 2 else None
base = "http://127.0.0.1:8010/api/v1"
ST = "/home/ubuntu/work/NOetix_GMT_eval/storage/runs"
DS, BM = "g1_motionatlas_v4", "kimodo_g1_motion_official_v3"
DATA = "/home/ubuntu/work/NOetix_GMT_eval/data/g1_motionatlas_v4"
LEADER = "130b4a731f3244ff877baee835fe256b"

lb = json.load(urllib.request.urlopen(f"{base}/leaderboard?mode=official&dataset_id={DS}&benchmark_id={BM}"))
by_run = {r["run_id"]: r for r in lb}
runs = json.load(urllib.request.urlopen(f"{base}/runs?limit=500"))
runs = runs if isinstance(runs, list) else next(v for v in runs.values() if isinstance(v, list))
sm = [r for r in runs if "SuperMimic G1" in (r.get("policy_name") or "") and r.get("dataset_id") == DS and r.get("mode") == "official" and r.get("status") == "completed"]
sm.sort(key=lambda r: r["created_at"])
ids = [r["id"] for r in sm if r["id"] != ours] + [LEADER, ours]
names = {r["id"]: r["policy_name"] for r in sm}
names[LEADER] = by_run[LEADER]["policy_name"]

# motion -> source (first directory under the dataset root); detect name collisions
src, coll = {}, 0
for dp, _, fs in os.walk(DATA):
    for f in fs:
        if f.endswith(".npz"):
            s = os.path.relpath(os.path.join(dp, f), DATA).split(os.sep)[0]
            if f[:-4] in src and src[f[:-4]] != s:
                coll += 1
            src[f[:-4]] = s

FALL = {"root_ori", "root_pos", "root_height", "root_gravity"}
def load(rid):
    rows = list(csv.DictReader(open(f"{ST}/{rid}/evaluation_per_segment.csv")))
    reasons = collections.Counter(r["termination_reason"] for r in rows)
    per = collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        s = src.get(r["motion_name"], "?")
        for k in (s, "ALL"):
            per[k]["n"] += 1
            per[k]["ok"] += r["success"] == "True"
            if r["success"] != "True":
                t = r["termination_reason"]
                per[k]["ee" if t == "body_pos" else ("fall" if t in FALL else "other:" + t)] += 1
    return rows, reasons, per

res = {}
for rid in ids:
    rows, reasons, per = load(rid)
    row = by_run.get(rid, {})
    res[rid] = dict(name=names.get(rid, row.get("policy_name")), rank=row.get("rank"), success=row.get("success_rate"),
                    progress=row.get("progress_rate"), mpjpe=row.get("mpjpe_rad"), mpkpe=row.get("mpkpe_mm"),
                    mpjve=row.get("mpjve_rad_s"), n=len(rows), reasons=dict(reasons), per={k: dict(v) for k, v in per.items()})
print(f"leaderboard rows {len(lb)}; motion-name source collisions {coll}")
print(f"{'run':10s} {'rank':>5s} {'succ%':>7s} {'prog%':>7s} {'mpjpe':>7s} {'mpkpe':>6s} {'fails':>5s} {'EE':>4s} {'fall':>4s}  name")
for rid in ids:
    r = res[rid]
    nf = r["n"] - r["reasons"].get("motion_complete", 0)
    ee = r["reasons"].get("body_pos", 0)
    fall = sum(v for k, v in r["reasons"].items() if k in FALL)
    print(f"{rid[:8]:10s} {r['rank']:>5} {100 * r['success']:7.2f} {100 * r['progress']:7.2f} {r['mpjpe']:7.4f} {r['mpkpe']:6.2f} {nf:5d} {ee:4d} {fall:4d}  {r['name']}")
print("our termination reasons:", res[ours]["reasons"])
keys = sorted(res[ours]["per"], key=lambda k: (k != "ALL", -res[ours]["per"][k]["n"]))
cols = [rid for rid in ids if rid != ours]
print(f"\nper source success% (ours = {ours[:8]}; EE / fall = our failure counts)")
print(f"{'source':26s} {'n':>5s} {'ours':>6s} {'EE':>4s} {'fall':>4s} " + " ".join(f"{c[:8]:>8s}" for c in cols))
for k in keys:
    p = res[ours]["per"][k]
    print(f"{k:26s} {p['n']:5d} {100 * p['ok'] / p['n']:6.1f} {p.get('ee', 0):4d} {p.get('fall', 0):4d} "
          + " ".join(f"{100 * res[c]['per'].get(k, {}).get('ok', 0) / max(res[c]['per'].get(k, {}).get('n', 0), 1):8.1f}" for c in cols))
if out_json:
    json.dump(res, open(out_json, "w"), ensure_ascii=False, indent=1)
    print("wrote", out_json)
