import json, sys, collections
R = [json.loads(l) for l in open(sys.argv[1])]
print("n", len(R))
keys = ["torso_z", "torso_tilt", "ee_z", "pelvis_ori"]
agree = sum(r["first"]["eval"] is not None for r in R)
print("eval failure reproduced (any reason):", agree, " reasons:", collections.Counter((r["at_eval"] or {}).get("reason") for r in R))
rp = [r for r in R if r["at_eval"] and r["at_eval"]["reason"] == "root_pos"]
print("reproduced as root_pos:", len(rp))
def lead(r, k):
  f = r["first"][k]; e = r["first"]["eval"]
  return None if f is None else e - f
any_train_first = 0; torso_by = collections.Counter(); late = []
for r in rp:
  L = {k: lead(r, k) for k in ["torso_z", "torso_tilt", "ee_z"]}
  ok = [v for v in L.values() if v is not None and v >= 0]
  any_train_first += bool(ok)
  t = L["torso_z"]
  torso_by["never" if t is None else ("<=eval" if t >= 0 else f"late")] += 1
  if t is None or t < 0:
    late.append((r["motion"][:50], r["src"], r["leader"], L, r["at_eval"]))
print("some training criterion (torso z / torso tilt / EE z) fires at or before the eval root_pos frame:", any_train_first, "/", len(rp))
print("torso_z relative to eval frame:", dict(torso_by))
leads = sorted(lead(r, "torso_z") for r in rp if lead(r, "torso_z") is not None)
print("torso_z lead (frames, + = earlier) sorted:", leads)
print("\ncases where torso z did NOT fire by the eval frame:")
for x in late: print("  ", x)
print("\nleader outcome on these segments:", collections.Counter(r["leader"] for r in rp))
print("dz at eval (pelvis, torso):", [(r["at_eval"]["dz_pelvis"], r["at_eval"]["dz_torso"]) for r in rp])
