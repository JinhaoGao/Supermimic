import pandas as pd, numpy as np, warnings, json
warnings.filterwarnings("ignore")
C = "/tmp/claude-1000/-home-user-Workspace-Noetix-GMT/7380f59e-8ed0-40ea-9060-69c5f5a5a1ef/scratchpad/supermimic/diag_1024/taskC"
pd.set_option("display.width", 320); pd.set_option("display.max_rows", 300); pd.set_option("display.max_columns", 60)
df = pd.read_pickle(f"{C}/data/seg_full.pkl")
ab = sorted(k for k in df.key.unique() if k[0] in "ab")
keys = ["L112500", "s086000", "s102000"] + ab
sub = df[df.key.isin(keys)]
ok_all = sub.groupby("segid").ok.all()
ids = ok_all[ok_all].index
d = sub[sub.segid.isin(ids)]
print("fixed set", len(ids))
succ = df.groupby("key").ok.mean()
m = d.groupby("key")[["body_local_lin_acc_error", "root_ang_acc_error", "root_lin_vel_error", "mpjae_rad_s2", "joint_vel_mae", "mpjpe_l_mm", "root_xy_error", "joint_pos_mae"]].median()
m["success"] = succ
m = m.loc[keys]
print(m.round(4).to_string())
x = m.loc[ab]
print("Pearson r with success over a/b checkpoints:")
print(x.corr()["success"].round(3).to_string())
# per-joint acc / vel / pos error growth
jcols = [c for c in d.columns if c.startswith("joint_acc_error__")]
vcols = [c for c in d.columns if c.startswith("joint_vel_error__")]
pcols = [c for c in d.columns if c.startswith("joint_pos_error__")]
def growth(cols, k0, k1):
    a = d[d.key == k0][cols].median(); b = d[d.key == k1][cols].median()
    return pd.DataFrame({k0: a, k1: b, "ratio": b / a})
for cols, nm in ((jcols, "acc"), (vcols, "vel"), (pcols, "pos")):
    g = growth(cols, "a088000", "b212000")
    g2 = growth(cols, "b116000", "b212000")
    g["ratio116_212"] = g2["ratio"]
    g["L"] = d[d.key == "L112500"][cols].median()
    g.index = [i.split("__")[1] for i in g.index]
    print(f"\njoint_{nm}_error median (fixed set)"); print(g.sort_values("ratio", ascending=False).round(3).to_string())
