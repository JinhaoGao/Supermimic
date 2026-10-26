import json, urllib.request
u = "http://127.0.0.1:8010/api/v1/leaderboard?mode=official&dataset_id=g1_motionatlas_v4&benchmark_id=kimodo_g1_motion_official_v3"
d = json.load(urllib.request.urlopen(u, timeout=60))
rows = d if isinstance(d, list) else d.get("rows") or d.get("items") or d.get("data") or d
if isinstance(rows, dict):
    print(list(rows.keys())[:20]); rows = next(v for v in rows.values() if isinstance(v, list))
print(len(rows)); print(json.dumps(rows[0], ensure_ascii=False)[:1500])
for r in rows:
    pn = r.get("policy_name", "")
    if "SuperMimic" in pn or r.get("run_id","").startswith("130b4a73"):
        print(r.get("run_id"), r.get("success_rate"), pn)
