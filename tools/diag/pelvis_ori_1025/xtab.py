import pandas as pd, numpy as np, sys
L = sys.argv[1]
a = pd.read_csv(f"{L}/data/seg_s148.tsv", sep="\t").set_index("segment_id")
b = pd.read_csv(f"{L}/data/seg_leader.tsv", sep="\t").set_index("segment_id")
b = b.reindex(a.index)
x = pd.crosstab(a.termination_reason.rename("s148"), b.termination_reason.rename("leader"), margins=True)
print(x)
# among segments where we fail root_ori: compare failure time vs leader failure time
m = a.termination_reason.eq("root_ori")
lf = b.termination_reason.ne("motion_complete")
print("our root_ori fails:", m.sum(), "leader passes:", (m & ~lf).sum(), "leader fails too:", (m & lf).sum())
both = m & lf
dt = (a.failure_step.astype(float) - b.failure_step.astype(float))[both]*0.02
print("  where both fail: our fail time minus leader fail time (s): median %.2f, frac ours earlier %.2f" % (dt.median(), (dt<0).mean()))
# net
print("ours fail", (a.termination_reason!="motion_complete").sum(), "leader fail", (b.termination_reason!="motion_complete").sum())
print("ours fail & leader ok", ((a.termination_reason!="motion_complete") & (b.termination_reason=="motion_complete")).sum(),
      "| ours ok & leader fail", ((a.termination_reason=="motion_complete") & (b.termination_reason!="motion_complete")).sum())
src = a.motion_path.str.extract(r"/([^/]+)/[^/]+$")[0]
for lay in ["unitree_g1","unitree_g1_phc","unitree_g1_umr"]:
    s = src.eq(lay)
    print(lay, "n", s.sum(), "ours ok %.1f" % (100*(a.termination_reason[s]=="motion_complete").mean()), "leader ok %.1f" % (100*(b.termination_reason[s]=="motion_complete").mean()))
