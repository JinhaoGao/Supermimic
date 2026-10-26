import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
C = "/tmp/claude-1000/-home-user-Workspace-Noetix-GMT/7380f59e-8ed0-40ea-9060-69c5f5a5a1ef/scratchpad/supermimic/diag_1024/taskC"
pd.set_option("display.width", 320); pd.set_option("display.max_rows", 300); pd.set_option("display.max_columns", 60)
df = pd.read_pickle(f"{C}/data/seg_full.pkl")
keys = ["L112500", "s086000", "s102000", "a088000", "a092000", "a096000", "a100000", "a104000", "b108000", "b116000", "b124000", "b132000", "b140000", "b148000", "b156000", "b164000", "b168000", "b176000", "b184000", "b192000", "b200000", "b212000"]
sub = df[df.key.isin(keys)]
ok_all = sub.groupby("segid").ok.all()
ids = ok_all[ok_all].index
d = sub[sub.segid.isin(ids)]
print("fixed always-success set:", len(ids))
cols = ["mpjpe_l_mm", "mpjpe_g_mm", "root_xy_error", "root_height_error", "root_ori_error", "root_lin_vel_error", "root_ang_vel_error", "root_lin_acc_error", "root_ang_acc_error", "joint_pos_mae", "joint_vel_mae", "mpjve_rad_s", "mpjae_rad_s2", "body_local_lin_acc_error", "body_local_ang_acc_error", "body_z_error", "mpjpe_l_feet_mm", "mpjpe_l_upper_mm", "mpjpe_g_feet_mm", "s2r_global_root_xy_m", "s2r_body_pos_mm", "s2r_wrist_pos_mm", "root_gravity_z_error"]
cols = [c for c in cols if c in d.columns]
t = d.groupby("key")[cols].median().loc[keys]
print(t.T.round(4).to_string())
