"""D1 analysis: does the new pelvis full-angle criterion fire first?
  python3 d1_analyze.py <d1_x.jsonl> [<out.tsv>]
fire_first = new fired at a frame <= the eval-site failure frame AND (no existing training criterion fired, or it fired
strictly later). PASS if >= 60 % of the 101."""
import json
import sys

import numpy as np
import pandas as pd

R = [json.loads(line) for line in open(sys.argv[1])]
rows = []
for r in R:
  f = r["first"]
  new, old, anc, eeo, site, env, tms = (f[k] for k in ("new", "old", "anchor", "ee_only", "site", "env", "tm_site"))
  ef = r["eval_fail"]
  rows.append(dict(
    motion=r["motion"].replace(".npz", ""), cluster=r["cluster"], src=r["src"], T=r["T"], eval_fail=ef, last=r["last_frame"],
    new=new, old=old, anchor=anc, ee_only=eeo, site=site, env=env, tm_site=tms, tm_site_reason=r["tm_site_reason"],
    new_by_eval=new is not None and new <= ef,
    new_before_old=new is not None and (old is None or new < old),
    tie_old=new is not None and old is not None and new == old,
    old_first=old is not None and (new is None or old < new),
    fire_first=new is not None and new <= ef and (old is None or new < old),
    fire_first_or_tie=new is not None and new <= ef and (old is None or new <= old),
    new_any=new is not None, old_any=old is not None,
    lead_vs_eval=(ef - new) if new is not None else np.nan, lead_vs_old=(old - new) if (new is not None and old is not None) else np.nan,
    ang_at_eval_fail=r["at"].get("eval_fail", {}).get("ang", np.nan), max_ang_pre=r["max_ang_pre_eval_fail"],
    soft_on_at_new=r["at"].get("new", {}).get("soft_on", np.nan), soft_off_at_new=r["at"].get("new", {}).get("soft_off", np.nan),
    site_eq_new=site == new, env_eq_new=env == new, n_hard_mismatch=r["n_hard_mismatch"], n_env_mismatch=r["n_env_mismatch"],
    max_dang_watch=r["max_dang_watch"], max_dang_site=r["max_dang_site"], fk_q=r["fk_quat_err_max"], fk_p=r["fk_pos_err_max"],
  ))
d = pd.DataFrame(rows).sort_values(["cluster", "motion"])
n = len(d)
k = int(d.fire_first.sum())
print(f"== {sys.argv[1]}: n={n} (proto {R[0]['proto']})")
print(f"fire_first (new <= eval-site fail frame and strictly before every existing training criterion): {k}/{n} = {100 * k / n:.1f}%  -> "
      f"{'PASS' if k >= 0.6 * n else 'FAIL'} (>= 60%)")
print(f"  incl. ties with old: {int(d.fire_first_or_tie.sum())}; new fired at all in the window: {int(d.new_any.sum())}; "
      f"new <= eval fail: {int(d.new_by_eval.sum())}; new before old (any time): {int(d.new_before_old.sum())}; tie: {int(d.tie_old.sum())}; "
      f"old first: {int(d.old_first.sum())}; old fired at all: {int(d.old_any.sum())}; neither: {int((~d.new_any & ~d.old_any).sum())}")
print("  outcome of the TRAINING-model replay under the eval site's criteria (first failure):",
      d.tm_site_reason.fillna("none(completes / no failure in window)").value_counts().to_dict())
tm_ro = d[d.tm_site_reason == "root_ori"]
print(f"  among clips that fail by root_ori on the training model too (n={len(tm_ro)}): fire_first {int(tm_ro.fire_first.sum())}, "
      f"new<=eval {int(tm_ro.new_by_eval.sum())}, new before old {int(tm_ro.new_before_old.sum())}")
nn = d[d.new_any]
print(f"  where new fires: lead vs eval-site fail frame (frames, + = earlier): median {nn.lead_vs_eval.median():.0f}, "
      f"q10/q90 {nn.lead_vs_eval.quantile(.1):.0f}/{nn.lead_vs_eval.quantile(.9):.0f}; lead vs old (when both) median {nn.lead_vs_old.median():.0f}")
print(f"consistency: site angle fires at the new frame {int(d.site_eq_new.sum())}/{n}, env term == new {int(d.env_eq_new.sum())}/{n}, "
      f"steps with env mismatch {int(d.n_env_mismatch.sum())}, heads hard != old|new {int(d.n_hard_mismatch.sum())}, "
      f"max |angle - watch angle| {d.max_dang_watch.max():.2e}, max |angle - eval-site angle| {d.max_dang_site.max():.2e}, "
      f"FK vs MuJoCo max quat {d.fk_q.max():.2e} rad / pos {d.fk_p.max():.2e} m; soft_on at the new frame min {nn.soft_on_at_new.min():.3f}")
print("\nby cluster (n, fire_first, new_any, old_first, training-model outcome root_ori/none):")
g = d.groupby("cluster").agg(n=("motion", "size"), fire_first=("fire_first", "sum"), new_any=("new_any", "sum"), old_first=("old_first", "sum"),
                             tm_ro=("tm_site_reason", lambda s: int((s == "root_ori").sum())), tm_none=("tm_site_reason", lambda s: int(s.isna().sum())))
print(g.sort_values("n", ascending=False).to_string())
if len(sys.argv) > 2:
  d.to_csv(sys.argv[2], sep="\t", index=False)
