import csv, statistics, os
ST = "/home/ubuntu/work/NOetix_GMT_eval/storage/runs"
RUNS = [("e40k","9787ca6c"),("e48k","f9bb9870"),("e52k","15d424a6"),("e60k","7a794a5e"),("r52_56k","39d602aa"),("r52_60k","98eaa10e"),("bj27_136k_ctl","2b02b12a"),("bj27_136k_clamp","35806e9d")]
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
