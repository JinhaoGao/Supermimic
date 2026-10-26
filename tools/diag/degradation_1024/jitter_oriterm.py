import csv, statistics, os
ST = "/home/ubuntu/work/NOetix_GMT_eval/storage/runs"
RUNS = [("L112.5k", "130b4a731f3244ff877baee835fe256b"), ("main_160k", "c8e6ab46"),
        ("T_162k", "732d181c"), ("T_164k", "12d8284f"), ("T_166k", "0f342404"), ("T_168k", "0d5abcd3"), ("T_170k", "06323db8"),
        ("C_162k", "3aff745b"), ("C_164k", "e32c1753"), ("C_166k", "52d9f0ac"), ("C_168k", "f268494e"), ("C_170k", "abe53ca2"), ("C_172k", "a2059982")]
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
