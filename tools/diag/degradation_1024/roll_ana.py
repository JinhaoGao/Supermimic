import json, sys, pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
rows = [json.loads(l) for l in open(sys.argv[1])]
df = pd.DataFrame(rows)
df["failed"] = df.fail.notna()
df["src"] = df.name.str.split("__").str[1].str.split("_").str[0]
df["dur"] = df["T"] * 0.02
print("tasks", len(df), "errors", df.get("err", pd.Series(dtype=str)).notna().sum())
g = df.groupby(["grp", "cond", "it"])
t = g.agg(n=("failed", "size"), fail=("failed", "mean"), da_legs=("da_rms_legs", "median"), da_waist=("da_rms_waist", "median"), da_arms=("da_rms_arms", "median"),
          hf_a_legs=("hf_a_legs", "median"), hf_a_waist=("hf_a_waist", "median"), hf_qd_legs=("hf_qd_legs", "median"), hf_qd_waist=("hf_qd_waist", "median"), a_abs=("a_abs", "median"))
print(t.round(4).to_string())
# fail reasons
print(df[df.failed].groupby(["grp", "cond", "it"]).fail.value_counts().unstack(fill_value=0).to_string())
# paired comparison per clip: clean vs act at same it
for grp in ("F212", "R"):
    d = df[df.grp == grp]
    p = d.pivot_table(index="name", columns=["cond", "it"], values="failed", aggfunc="first")
    print(grp, "fail counts:", {k: int(v) for k, v in p.sum().items()})
# early-window jitter (first 10 s, survived): use clips that survived >= 10 s in all its? (metrics are whole-run medians)
