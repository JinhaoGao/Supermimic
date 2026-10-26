import pandas as pd, numpy as np
C = "/tmp/claude-1000/-home-user-Workspace-Noetix-GMT/7380f59e-8ed0-40ea-9060-69c5f5a5a1ef/scratchpad/supermimic/diag_1024/taskC"
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
df = pd.read_pickle(f"{C}/data/seg.pkl")
df["segid"] = df.relpath + ":" + df.start_frame.astype(str)
df["len"] = df.end_frame - df.start_frame
def get(k): return df[df.key == k].set_index("segid")
A, B = get("b116000"), get("b212000")
A = A.loc[B.index]
print("segment length (frames) pct:", np.percentile(A.len, [5, 25, 50, 75, 95, 99]))
lost = A.ok & ~B.ok; gained = ~A.ok & B.ok
print("lost", lost.sum(), "gained", gained.sum())
L = B[lost]
print("lost by reason:", L.termination_reason.value_counts().to_dict())
fs = L.failure_step.astype(float)
print("failure_step of lost (steps) pct 10/25/50/75/90:", np.percentile(fs, [10, 25, 50, 75, 90]))
print("failure_step/len of lost pct:", np.percentile(fs / L.len, [10, 25, 50, 75, 90]).round(3))
print("frac lost failing after 500 steps (10 s):", (fs > 500).mean().round(3))
# length bins: success rate by segment length at several its
bins = [0, 250, 500, 1000, 2000, 5000, 100000]
df["lenbin"] = pd.cut(df.len, bins)
keys = ["L112500", "a092000", "b116000", "b140000", "b164000", "b168000", "b188000", "b212000"]
t = df[df.key.isin(keys)].pivot_table(index="lenbin", columns="key", values="ok", aggfunc="mean", observed=True)[keys] * 100
t["n"] = df[df.key == "b116000"].groupby("lenbin", observed=True).size()
print(t.round(1))
# among fails: failure step distribution per key
for k in keys:
    d = df[(df.key == k) & ~df.ok]
    fs = d.failure_step.astype(float)
    print(k, "fails", len(d), "fail<50", int((fs < 50).sum()), "fail 50-500", int(((fs >= 50) & (fs < 500)).sum()), "fail>=500", int((fs >= 500).sum()), "median ratio", round(float((fs / d.len).median()), 3))
# tracking error on segments successful in both
both = A.ok & B.ok
for col in ["mpjpe_l_mm", "mpjpe_g_mm"]:
    a, b = A.loc[both, col].astype(float), B.loc[both, col].astype(float)
    print(col, "both-success n", both.sum(), "median 116k", round(a.median(), 2), "212k", round(b.median(), 2), "mean ratio", round(float((b / a).mean()), 3), "frac worse", round(float((b > a).mean()), 3))
# trajectory of mpjpe medians on always-successful segments
ok_all = df[df.key.str[0].isin(["a", "b"])].groupby("segid").ok.all()
ids = ok_all[ok_all].index
print("always-success segs:", len(ids))
m = df[df.segid.isin(ids) & df.key.str[0].isin(["a", "b", "L"])].groupby("key")[["mpjpe_l_mm", "mpjpe_g_mm"]].mean()
print(m.round(2).T.to_string())
