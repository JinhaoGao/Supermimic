import csv, statistics, os
ST = "/home/ubuntu/work/NOetix_GMT_eval/storage/runs"
RUNS = [("L112.5k", "130b4a731f3244ff877baee835fe256b"), ("bj27_136k_clamp", "35806e9d"), ("d8_92k", "12842997"),
        ("r15ws_138k", "3c7491a3"), ("r15ws_140k", "5625ecbc"), ("r15ws_142k", "084e2ca9"), ("r15ws_144k", "e99fba57"),
        ("r15ws_146k", "c366ee27"), ("r15ws_148k", "ba74bb15"), ("r15ws_150k", "3a9f95e7")]
dirs = os.listdir(ST)
def full(p):
    if len(p) == 32: return p
    m = [d for d in dirs if d.startswith(p)]
    return m[0]
METS = ["body_local_lin_acc_error", "root_ang_acc_error", "joint_vel_mae", "mpjpe_g_mm"]
data = {}
for tag, rid in RUNS:
    rid = full(rid)
    rows = list(csv.DictReader(open(f"{ST}/{rid}/evaluation_per_segment.csv")))
    data[tag] = {(r["motion_path"], r["start_frame"]): r for r in rows}
keys = set.intersection(*[set(v) for v in data.values()])
ok = [k for k in keys if all(data[t][k]["success"] in ("True", "true", "1") for t in data)]
print("fixed all-success set", len(ok), "of", len(keys))
for tag, _ in RUNS:
    out = []
    for m in METS:
        try:
            out.append(round(statistics.median(float(data[tag][k][m]) for k in ok), 4))
        except Exception as e:
            out.append(str(e)[:20])
    sr = sum(data[tag][k]["success"] in ("True", "true", "1") for k in keys) / len(keys)
    print(f"{tag:18s} succ={sr:.4f}", dict(zip(METS, out)))
