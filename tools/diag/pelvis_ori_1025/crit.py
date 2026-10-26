import pandas as pd, numpy as np, sys
L = sys.argv[1]
runs = ["r15_136","s138","s144","s148","s152","leader"]
D = {r: pd.read_csv(f"{L}/data/seg_{r}.tsv", sep="\t") for r in runs}
for r, d in D.items():
    d["success"] = d["success"].astype(str).str.lower().eq("true")
    vc = d.termination_reason.value_counts().to_dict()
    print(r, "succ %.2f" % (100*d.success.mean()), vc)
print()
# For root_ori failures: gravity-z error max (training-tilt-like, pelvis) and root_ori max
for r in ["s148","leader"]:
    d = D[r]
    f = d[d.termination_reason=="root_ori"]
    g = f.root_gravity_z_error_max.astype(float)
    print(r, "root_ori fails", len(f), "| grav_z_max >0.8 (training tilt fires, pelvis):", int((g>0.8).sum()), " <0.8:", int((g<=0.8).sum()),
          " median gz %.2f" % g.median())
    print("   quantiles gz:", np.round(g.quantile([.1,.25,.5,.75,.9]).values,2))
    p = d[d.termination_reason=="root_pos"]
    print(r, "root_pos fails", len(p), "| grav_z_max >0.8:", int((p.root_gravity_z_error_max.astype(float)>0.8).sum()), " root_ori_max>1:", int((p.root_ori_error_max.astype(float)>1).sum()))
# successful segments: how close do they get?
for r in ["s148","leader"]:
    d = D[r]; s = d[d.success]
    print(r, "SUCCESS segs: root_ori_max quantiles", np.round(s.root_ori_error_max.astype(float).quantile([.5,.9,.99]).values,3),
          " root_h_max q", np.round(s.root_height_error_max.astype(float).quantile([.5,.9,.99]).values,3),
          " ee_z_max q", np.round(s.termination_body_z_error_max_max.astype(float).quantile([.5,.9,.99]).values,3),
          " gz_max q", np.round(s.root_gravity_z_error_max.astype(float).quantile([.5,.9,.99]).values,3))
# common successful segments: margin comparison
a, b = D["s148"].set_index("segment_id"), D["leader"].set_index("segment_id")
both = a.index[(a.success) & (b.reindex(a.index).success.fillna(False).astype(bool))]
for k in ["root_ori_error_max","root_height_error_max","termination_body_z_error_max_max","root_gravity_z_error_max","root_xy_error_max"]:
    x, y = a.loc[both, k].astype(float), b.loc[both, k].astype(float)
    print(f"both-success n={len(both)} {k}: s148 median {x.median():.3f} p90 {x.quantile(.9):.3f} | leader median {y.median():.3f} p90 {y.quantile(.9):.3f} | frac s148>leader {(x>y).mean():.2f}")
