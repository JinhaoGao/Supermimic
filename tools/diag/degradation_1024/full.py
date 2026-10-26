import pandas as pd, numpy as np
C = "/tmp/claude-1000/-home-user-Workspace-Noetix-GMT/7380f59e-8ed0-40ea-9060-69c5f5a5a1ef/scratchpad/supermimic/diag_1024/taskC"
pd.set_option("display.width", 300); pd.set_option("display.max_rows", 300); pd.set_option("display.max_columns", 60)
df = pd.read_csv(f"{C}/data/seg_full.tsv.gz", sep="\t", low_memory=False)
df["key"] = df.tag + df.it.astype(str).str.zfill(6)
df["ok"] = df.success.astype(str) == "True"
df["segid"] = df.motion_name + ":" + df.start_frame.astype(str)
df.to_pickle(f"{C}/data/seg_full.pkl")
print(df.shape)
