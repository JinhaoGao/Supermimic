"""Dump the union of failing segments across 4 v4 runs (read-only). python3 dump_union.py out.tsv"""
import csv, os, sys
ST = "/home/ubuntu/work/NOetix_GMT_eval/storage/runs"
P = ["b43fdd83", "621caa35", "669a45a8", "130b4a73"]
def full(p): return next(d for d in sorted(os.listdir(ST)) if d.startswith(p))
KEEP = ["segment_id", "motion_name", "motion_path", "success", "termination_reason", "failure_step", "target_length",
        "root_ori_error_max", "root_height_error_max", "start_frame", "end_frame"]
R = {}
for p in P:
    R[p] = {r["segment_id"]: {k: r[k] for k in KEEP} for r in csv.DictReader(open(f"{ST}/{full(p)}/evaluation_per_segment.csv"))}
segs = list(R[P[0]])
assert all(set(R[p]) == set(segs) for p in P), "segment sets differ"
with open(sys.argv[1], "w") as f:
    f.write("seg\tsrc\tname\t" + "\t".join(P) + "\n")
    for s in segs:
        if all(R[p][s]["success"] == "True" for p in P):
            continue
        r0 = R[P[0]][s]
        src = r0["motion_path"].split("/g1_motionatlas_v4/")[1].split("/")[0]
        st = ["ok" if R[p][s]["success"] == "True" else R[p][s]["termination_reason"] + "@" + R[p][s]["failure_step"] + "/" + R[p][s]["target_length"] for p in P]
        f.write(f"{s}\t{src}\t{r0['motion_name']}\t" + "\t".join(st) + "\n")
print("n segs", len(segs), {p: full(p) for p in P})
