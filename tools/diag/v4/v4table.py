"""Read-only: list all official v4 runs whose policy name contains 'SuperMimic G1' (plus leader) with success and termination counts."""
import collections, csv, json, os, urllib.request
base = "http://127.0.0.1:8010/api/v1"
ST = "/home/ubuntu/work/NOetix_GMT_eval/storage/runs"
DS, BM = "g1_motionatlas_v4", "kimodo_g1_motion_official_v3"
LEADER = "130b4a731f3244ff877baee835fe256b"
lb = json.load(urllib.request.urlopen(f"{base}/leaderboard?mode=official&dataset_id={DS}&benchmark_id={BM}"))
by_run = {r["run_id"]: r for r in lb}
runs = json.load(urllib.request.urlopen(f"{base}/runs?limit=500"))
runs = runs if isinstance(runs, list) else next(v for v in runs.values() if isinstance(v, list))
sm = [r for r in runs if ("SuperMimic G1" in (r.get("policy_name") or "") or r["id"] == LEADER) and r.get("dataset_id") == DS and r.get("mode") == "official"]
sm.sort(key=lambda r: r["created_at"])
print("leaderboard rows", len(lb), "sm runs", len(sm))
for r in sm:
    rid = r["id"]
    row = by_run.get(rid, {})
    p = f"{ST}/{rid}/evaluation_per_segment.csv"
    reasons = collections.Counter()
    n = 0
    if os.path.exists(p):
        for x in csv.DictReader(open(p)):
            n += 1
            if x["success"] != "True":
                reasons[x["termination_reason"]] += 1
    print("%s|%s|%s|rank=%s|succ=%s|prog=%s|mpjpe=%s|n=%d|fail=%d|%s|%s" % (
        rid[:8], r["created_at"][:16], r.get("status"), row.get("rank"), row.get("success_rate"), row.get("progress_rate"),
        row.get("mpjpe_rad"), n, sum(reasons.values()), dict(reasons), (r.get("policy_name") or "")[:140]))
