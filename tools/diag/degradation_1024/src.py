import pandas as pd, numpy as np, sys
C = "/tmp/claude-1000/-home-user-Workspace-Noetix-GMT/7380f59e-8ed0-40ea-9060-69c5f5a5a1ef/scratchpad/supermimic/diag_1024/taskC"
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 80); pd.set_option("display.max_rows", 500)
df = pd.read_csv(f"{C}/data/seg.tsv.gz", sep="\t", dtype={"run": str})
df["ok"] = df.success.astype(str) == "True"
df["src"] = df.relpath.str.split("/").str[0]
df["sub"] = df.relpath.str.split("/").str[1]
df["key"] = df.tag + df.it.astype(str).str.zfill(6)
df.to_pickle(f"{C}/data/seg.pkl")
print(df.termination_reason.value_counts())
print(df.groupby("key").size().head())
print(df[df.key=="L112500"].groupby("src").size())
