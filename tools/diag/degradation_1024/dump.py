import json, urllib.request, csv, os, re, gzip
u = "http://127.0.0.1:8010/api/v1/leaderboard?mode=official&dataset_id=g1_motionatlas_v4&benchmark_id=kimodo_g1_motion_official_v3"
rows = json.load(urllib.request.urlopen(u, timeout=60))
ST = "/home/ubuntu/work/NOetix_GMT_eval/storage/runs"
DATA = "/home/ubuntu/work/NOetix_GMT_eval/data/g1_motionatlas_v4/"
sel = []
for r in rows:
    pn = r["policy_name"]
    m = re.search(r"model_(\d+)", pn)
    tag = None
    if r["run_id"].startswith("130b4a73"): tag = "L"
    elif "g1v2_ddp8_ws86k_b" in pn: tag = "b"
    elif "g1v2_ddp8_ws86k" in pn: tag = "a"
    elif "完整配方从零" in pn: tag = "s"
    elif "g1_full_ee_ddp8_r52" in pn: tag = "r52"
    elif "g1_full_ee_ddp8" in pn: tag = "e"
    if tag: sel.append((tag, int(m.group(1)) if m else 0, r["run_id"], r["success_rate"]))
sel.sort()
os.makedirs("/data2/supermimic/diag_1024", exist_ok=True)
json.dump(sel, open("/data2/supermimic/diag_1024/runs.json", "w"))
cols = ["motion_name","start_frame","end_frame","success","length_ratio","termination_reason","failure_step","completed_steps","mpjpe_l_mm","mpjpe_g_mm"]
with gzip.open("/data2/supermimic/diag_1024/seg.tsv.gz", "wt") as w:
    w.write("\t".join(["tag","it","run"] + cols + ["relpath"]) + "\n")
    for tag, it, rid, sr in sel:
        p = f"{ST}/{rid}/evaluation_per_segment.csv"
        if not os.path.exists(p): print("missing", tag, it, rid); continue
        rr = list(csv.DictReader(open(p)))
        if tag == "L" or it in (88000, 212000): print(tag, it, len(rr), list(rr[0].keys()))
        for r in rr:
            mp = r.get("motion_path", ""); rel = mp[len(DATA):] if mp.startswith(DATA) else mp
            w.write("\t".join([tag, str(it), rid] + [r.get(c, "") for c in cols] + [rel]) + "\n")
for s in sel: print(*s)
