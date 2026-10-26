import pandas as pd, numpy as np
C = "/tmp/claude-1000/-home-user-Workspace-Noetix-GMT/7380f59e-8ed0-40ea-9060-69c5f5a5a1ef/scratchpad/supermimic/diag_1024/taskC"
pd.set_option("display.width", 320); pd.set_option("display.max_columns", 80); pd.set_option("display.max_rows", 500)
df = pd.read_pickle(f"{C}/data/seg.pkl")
keys = ["L112500", "s086000", "s102000"] + sorted(k for k in df.key.unique() if k[0] in "ab")
d = df[df.key.isin(keys)]
srcs = d[d.key=="L112500"].groupby("src").size().sort_values(ascending=False).index.tolist()
t = d.pivot_table(index="key", columns="src", values="ok", aggfunc="mean").loc[keys, srcs] * 100
t["ALL"] = d.groupby("key").ok.mean().loc[keys] * 100
n = d[d.key=="L112500"].groupby("src").size()
print("n per src:", dict(n))
print(t.round(1).to_string())
# failure types
fail = d[~d.ok]
ft = fail.groupby(["key", "termination_reason"]).size().unstack(fill_value=0).loc[keys]
ft["fails"] = ft.sum(1)
print(ft.to_string())
# failure counts per source split EE vs fall
for kind, cond in [("EE", fail.termination_reason == "body_pos"), ("FALL", fail.termination_reason != "body_pos")]:
    x = fail[cond].groupby(["key", "src"]).size().unstack(fill_value=0).reindex(index=keys, columns=srcs, fill_value=0)
    x["tot"] = x.sum(1)
    print(f"\n#{kind} failures per source"); print(x.to_string())
