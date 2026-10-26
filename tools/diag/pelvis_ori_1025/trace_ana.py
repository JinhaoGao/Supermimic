import json, sys, pandas as pd, numpy as np, re
L = sys.argv[1]
out = pd.read_csv(f"{L}/data/v4_outcomes.tsv", sep="\t").set_index("motion")
def load(tag):
    R = [json.loads(l) for l in open(f"{L}/data/trace_{tag}.jsonl")]
    rows = []
    for r in R:
        m = r["motion"].replace(".npz", "")
        af = r["at_fail"] or {}
        fs = r["fail_step"]
        f = r["first"]
        def rel(k):
            return (f[k] - fs) * 0.02 if (k in f and fs is not None) else np.nan
        rows.append(dict(motion=m, csv_reason=out.ours.get(m), leader=out.leader.get(m), reason=af.get("reason", "ok"), fail_s=None if fs is None else fs * 0.02,
                         full=af.get("full"), swing=af.get("swing"), twist=af.get("twist"), gz_pelvis=af.get("gzd_pelvis"), torso_gz=af.get("torso_gzd"),
                         torso_sw=af.get("torso_swing"), torso_tw=af.get("torso_twist"), torso_z=af.get("torso_z"), root_h=af.get("root_h"), ee=af.get("ee"), ref_h=af.get("ref_h"),
                         dt_train_gz=rel("train_torso_gz>0.8"), dt_train_z=rel("train_torso_z>0.25"), dt_ee=rel("eval_ee"), dt_leader_ori=rel("leader_ori_full>0.447"),
                         dt_leader_pos=rel("leader_pos_z>0.15"), dt_leader_ee=rel("leader_ee>0.15"), dt_fell=rel("fell_pelvis_z>0.5"),
                         post_s=r["post"]["steps"] * 0.02, post_max_h=r["post"]["max_root_h"], post_max_ee=r["post"]["max_ee"], post_end_full=r["post"]["end_full"],
                         pre_max_torso_gz=r["pre_max"]["torso_gzd"]))
    return pd.DataFrame(rows)
d = load("det")
print("reproduced reason == csv:", (d.reason == d.csv_reason).mean(), "n", len(d))
o = d[d.reason == "root_ori"].copy()
print(f"\n== root_ori failures (det, n={len(o)}) at the failure step ==")
o["dom"] = np.where(o.twist > o.swing, "twist(yaw about world z)", "swing(tilt)")
print(o.dom.value_counts())
print("pelvis swing at fail: quantiles", np.round(o.swing.quantile([.1, .25, .5, .75, .9]).values, 2))
print("pelvis twist at fail: quantiles", np.round(o.twist.quantile([.1, .25, .5, .75, .9]).values, 2))
print("torso (training anchor) full-angle parts at fail: swing med %.2f twist med %.2f" % (o.torso_sw.median(), o.torso_tw.median()))
print("training tilt (torso gz diff) at fail: >0.8:", int((o.torso_gz > 0.8).sum()), " median %.2f" % o.torso_gz.median(), "| ever >0.8 before fail:", int((o.pre_max_torso_gz > 0.8).sum()))
print("training torso z>0.25 at/before fail:", int((o.dt_train_z <= 0).sum()), "| training EE z>0.25 at/before fail:", int((o.dt_ee <= 0).sum()))
any_train = (o.dt_train_gz <= 0) | (o.dt_train_z <= 0) | (o.dt_ee <= 0)
print("ANY training criterion fired at/before eval root_ori fail:", int(any_train.sum()), "of", len(o))
print("leader-style ori (full>25.6deg) fired before:", int((o.dt_leader_ori <= 0).sum()), " median lead time %.2f s" % (-o.dt_leader_ori).median())
# after failure (continued 6 s)
fell = (o.post_max_h > 0.25) | (o.post_max_ee > 0.25)
print(f"after the root_ori trip (continue up to 6 s): later pelvis z>0.25 or EE>0.25: {int(fell.sum())}; pelvis z>0.5 (true fall): {int((o.dt_fell > 0).sum())}; "
      f"neither (recovers/continues tracking): {int((~fell).sum())}; end full-angle median {o.post_end_full.median():.2f}")
# leader-pass subset
lp = o[o.leader == "motion_complete"]
print(f"\n-- subset leader passes (n={len(lp)}): twist-dominant {int((lp.twist > lp.swing).sum())}, training-crit fired {int(((lp.dt_train_gz <= 0) | (lp.dt_train_z <= 0) | (lp.dt_ee <= 0)).sum())}, later fell(z/EE>0.25) {int(((lp.post_max_h > 0.25) | (lp.post_max_ee > 0.25)).sum())}")
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 40); pd.set_option("display.max_rows", 200)
print(lp[["motion", "fail_s", "full", "swing", "twist", "torso_sw", "torso_tw", "torso_gz", "ref_h", "post_max_h", "post_max_ee", "post_end_full"]].round(2).to_string())
p = d[d.reason == "root_pos"]
print(f"\n== root_pos failures (n={len(p)}): torso z>0.25 at/before: {int((p.dt_train_z <= 0).sum())}, ref_h<0.5: {int((p.ref_h < 0.5).sum())}, later true fall z>0.5: {int((p.dt_fell > 0).sum())}, fail at t<0.5s: {int((p.fail_s < 0.5).sum())}")
print(p[["motion", "leader", "fail_s", "root_h", "torso_z", "ref_h", "swing", "twist", "post_max_h"]].round(2).to_string())
n = load("n01")
print("\n== with action noise sigma=0.1 (training-like) on the same 129 clips ==")
print(n.reason.value_counts().to_dict())
m = d.merge(n[["motion", "reason", "fail_s"]], on="motion", suffixes=("", "_n"))
print(pd.crosstab(m.reason, m.reason_n))
print("of the det root_ori fails: with noise -> ", m[m.reason == "root_ori"].reason_n.value_counts().to_dict())
