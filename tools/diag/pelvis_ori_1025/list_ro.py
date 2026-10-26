import pandas as pd, numpy as np, sys, re
L = sys.argv[1]
d = pd.read_csv(f"{L}/data/seg_s148.tsv", sep="\t")
ld = pd.read_csv(f"{L}/data/seg_leader.tsv", sep="\t").set_index("segment_id")
f = d[d.termination_reason.isin(["root_ori","root_pos"])].copy()
f["src"] = f.motion_path.str.extract(r"/([^/]+)/[^/]+$")[0]
f["T_s"] = f.target_length.astype(float)*0.02
f["t_fail"] = f.failure_step.astype(float)*0.02
f["L"] = [ld.termination_reason.get(s, "?") for s in f.segment_id]
f["Lori"] = [ld.root_ori_error_max.get(s, np.nan) for s in f.segment_id]
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300); pd.set_option("display.max_colwidth", 46)
print(f.category.value_counts().head(20))
print(f.src.value_counts().head(20))
print(f[["motion_name","category","termination_reason","t_fail","T_s","length_ratio","root_gravity_z_error_max","root_ori_error_max","root_height_error_max","L","Lori"]].sort_values(["termination_reason","category"]).to_string())
