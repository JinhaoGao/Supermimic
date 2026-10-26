import csv, statistics, os
ST = "/home/ubuntu/work/NOetix_GMT_eval/storage/runs"
RUNS = [("L112.5k", "130b4a731f3244ff877baee835fe256b"),
        ("s86k", "e4926201141241b2bb51ee1e36d33b1e"), ("s102k", "2526cf72701d4c95833779e3b9358d87"),
        ("bj27_104k_ctl", "03c36e81"), ("bj27_H48_88k", "7e981d85"), ("bj27_H48_96k", "bc7ee970"),
        ("bj27_116k_exp", "f49e0de0"), ("bj27_122k_phc", "27e1d54e"), ("bj27_128k_cyc", "ce12b7ed"),
        ("bj27_128k_phc", "668c7d97"), ("bj27_136k_ctl", "2b02b12a"), ("bj27_136k_clamp", "35806e9d"),
        ("d8_88k", "5c02c005"), ("d8_92k", "12842997"), ("d8_116k", "9b628e05"), ("d8_140k", "81789454"),
        ("d8_164k", "7a2944b5"), ("d8_168k", "78de9b66"), ("d8_188k", "698edbc5"), ("d8_212k", "7943ef8e")]
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
