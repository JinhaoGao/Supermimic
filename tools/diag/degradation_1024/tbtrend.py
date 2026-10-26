import json, numpy as np, sys
C = "/tmp/claude-1000/-home-user-Workspace-Noetix-GMT/7380f59e-8ed0-40ea-9060-69c5f5a5a1ef/scratchpad/supermimic/diag_1024/taskC"
a = json.load(open(f"{C}/tb/a/scalars.json")); b = json.load(open(f"{C}/tb/b/scalars.json"))
# merge: for a, keep steps < 104000 (b starts from 104000); a also has rollbacks -> keep the last value written for each step
def series(tag):
    pts = {}
    for st, w, v in a.get(tag, []):
        if st < 104000: pts[st] = v
    for st, w, v in b.get(tag, []):
        pts[st] = v
    s = np.array(sorted(pts)); return s, np.array([pts[x] for x in s])
groups = [sys.argv[1].split(",")] if len(sys.argv) > 1 else None
tags = sys.argv[1].split(",")
BIN = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
edges = list(range(86000, 216000, BIN))
print("bin_start " + " ".join(f"{t.split('/',1)[1][:14]:>14s}" for t in tags))
for e in edges:
    row = []
    for t in tags:
        s, v = series(t)
        m = (s >= e) & (s < e + BIN)
        if not m.any(): row.append(f"{'':>14s}"); continue
        x = v[m]; med = np.median(x)
        row.append(f"{med:14.5g}")
    print(f"{e:9d} " + " ".join(row))
