import pandas as pd, numpy as np
C = "/tmp/claude-1000/-home-user-Workspace-Noetix-GMT/7380f59e-8ed0-40ea-9060-69c5f5a5a1ef/scratchpad/supermimic/diag_1024/taskC"
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200); pd.set_option("display.max_columns", 50)
df = pd.read_pickle(f"{C}/data/seg.pkl")
df["segid"] = df.relpath + ":" + df.start_frame.astype(str)
df["len"] = df.end_frame - df.start_frame
keys = ["L112500", "s064000", "s086000", "s102000", "a088000", "a092000", "a096000", "a100000", "a104000", "b108000", "b116000", "b124000", "b132000", "b140000", "b148000", "b156000", "b164000", "b168000", "b176000", "b184000", "b192000", "b200000", "b212000"]
sub = df[df.key.isin(keys)]
ok_all = sub.groupby("segid").ok.all()
ids = ok_all[ok_all].index
d = sub[sub.segid.isin(ids)].copy()
print("always-success over keys:", len(ids))
d["lenbin"] = pd.cut(d.len, [0, 300, 500, 1000, 3000, 100000])
for col in ["mpjpe_g_mm", "mpjpe_l_mm"]:
    t = d.pivot_table(index="key", columns="lenbin", values=col, aggfunc="median", observed=True).loc[keys]
    t["ALLmed"] = d.groupby("key")[col].median().loc[keys]
    t["ALLmean"] = d.groupby("key")[col].mean().loc[keys]
    print(col); print(t.round(1).to_string())
print(d[d.key == "b116000"].groupby("lenbin", observed=True).size())
# per source median mpjpe_g
t = d.pivot_table(index="key", columns="src", values="mpjpe_g_mm", aggfunc="median").loc[keys]
print("mpjpe_g median by source (always-success segs)"); print(t.round(0).to_string())
