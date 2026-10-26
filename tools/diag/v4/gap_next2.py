"""Follow-up to gap_next.py (read-only): flaky/stable counts per line, transfer estimate of bj27 feature effects onto our line,
jump/aerial + GROUND_* trend over more checkpoints, overlap of feature-fixed sets with our failures, why ours fails jumps."""
import collections, os, re, sys
import numpy as np
sys.argv = [sys.argv[0]]
src_code = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gap_next.py")).read()
# reuse definitions only (everything before the headline section)
exec(src_code.split("# ---------------------------------------------------------------- headline")[0])

names = list(load(SINGLE["O"]).keys())
meta = {n: dict(src=src(load(SINGLE["O"])[n]), cl=cluster(n), g=gcat(n)) for n in names}
prob = {g: {n: float(np.mean([load(d)[n]["ok"] for d in ds])) for n in names} for g, ds in GROUPS.items()}

print("== stability per line (segments): always pass / flaky / always fail")
for g in ("ours_line", "bj27_floor", "bj27_ro", "bj27_roroot"):
    p = prob[g]; a = sum(v == 1 for v in p.values()); z = sum(v == 0 for v in p.values())
    print(f"   {g:12s} ({len(GROUPS[g])} ckpts) always {a}  flaky {len(names)-a-z}  never {z}  expected fails {sum(1-v for v in p.values()):.1f}")
fl = [n for n in names if 0 < prob["ours_line"][n] < 1]
print("   ours flaky by cluster", dict(collections.Counter(meta[n]["cl"] for n in fl).most_common()),
      " expected fails within flaky %.1f" % sum(1 - prob["ours_line"][n] for n in fl))
nev = [n for n in names if prob["ours_line"][n] == 0]
print("   ours never-pass by cluster", dict(collections.Counter(meta[n]["cl"] for n in nev).most_common()))
nev_lpass = [n for n in nev if load(SINGLE["L"])[n]["ok"]]
nev_bpass = [n for n in nev if prob["bj27_floor"][n] > 0]
print(f"   ours never-pass but leader passes {len(nev_lpass)} {dict(collections.Counter(meta[n]['cl'] for n in nev_lpass))}; bj27 floor line ever passes {len(nev_bpass)} {dict(collections.Counter(meta[n]['cl'] for n in nev_bpass))}")

def transfer(off, on, label):
    """bj27 matched effect d_n = p_on - p_off; transfer to ours: gains scale by our remaining failure share, losses by our pass share"""
    tot, bycl, raw = 0.0, collections.Counter(), 0.0
    for n in names:
        d = prob[on][n] - prob[off][n]; raw += d
        po, pa = prob[off][n], prob["ours_line"][n]
        if d > 0: t = d * min(1.0, (1 - pa) / max(1 - po, 1e-9))
        elif d < 0: t = d * min(1.0, pa / max(po, 1e-9))
        else: t = 0.0
        tot += t; bycl[meta[n]["cl"]] += t
    print(f"   {label}: bj27 raw {raw:+.2f} segs -> transferred to ours {tot:+.2f} segs ({100*tot/len(names):+.2f} pp); by cluster "
          + ", ".join(f"{k} {v:+.2f}" for k, v in sorted(bycl.items(), key=lambda x: -abs(x[1])) if abs(v) >= 0.2))
print("\n== transfer estimate of bj27 matched feature effects onto ours_line")
transfer("bj27_ro", "bj27_roroot", "root motion (164-176k x4)")
transfer("bj27_roroot_r23", "bj27_floor_r23", "ground exposure (188/192k x2)")

print("\n== overlap: segments a feature robustly fixed on bj27 (p_on>=2/3, p_off<=1/3) vs our line")
for off, on, label in (("bj27_ro", "bj27_roroot", "root motion"), ("bj27_roroot_r23", "bj27_floor_r23", "ground exposure")):
    S = [n for n in names if prob[on][n] >= 2 / 3 and prob[off][n] <= 1 / 3]
    print(f"   {label}: {len(S)} fixed; our line p: " + ", ".join(f"{re.sub(r'__[0-9a-f]+$', '', n)[:34]}={prob['ours_line'][n]:.2f}" for n in S))
    print(f"      expected our failures on that set {sum(1-prob['ours_line'][n] for n in S):.1f} / {len(S)}")

# more checkpoints for trends (site runs + offline), cluster success
EXTRA = [("ours r15ws 152k", site("5c5901e2")), ("ours r15ws 160k", site("c8e6ab46")), ("ours ro0 C 162k", site("3aff745b")),
         ("ours ro0 C 168k", site("f268494e")), ("ours 8g 186k", site("813a7e01")), ("ours 8g 188k", site("be621069")),
         ("ours T2 0.8 178k", site("69da320f")), ("ours T2 0.8 182k", site("583af17a")),
         ("bj27 root_r18 152k", off("g1v2_root_r18_152000")), ("bj27 roroot_r22 184k", off("g1v2_roroot_r22_184000")),
         ("bj27 roroot08 192k", off("g1v2_roroot08_r23_192000")), ("bj27 floor08 200k", off("g1v2_floor08_r24_200000")),
         ("bj27 avg3 208k", off("g1v2_avg3_r25_208000"))]
print("\n== cluster success over more checkpoints (pass counts)")
cls = ["getup/lie/ground", "breaking/floor", "flip/acro", "jump/aerial", "sit/chair", "other"]
print(f"   {'ckpt':22s} {'ALL%':>6s} " + " ".join(f"{c[:10]:>10s}" for c in cls) + "  GR_GETUP GR_CRAWL  EE r_ori r_pos")
for lab, d in EXTRA + [(k, SINGLE[k]) for k in SINGLE]:
    R = load(d)
    c = collections.Counter(reason(r) for r in R.values() if not r["ok"])
    row = [sum(R[n]["ok"] for n in names if meta[n]["cl"] == cl) for cl in cls]
    gg = [sum(R[n]["ok"] for n in names if meta[n]["g"] == g) for g in ("GROUND_GETUP", "GROUND_CRAWL")]
    print(f"   {lab:22s} {100*sum(r['ok'] for r in R.values())/len(R):6.2f} " + " ".join(f"{v:10d}" for v in row)
          + f"  {gg[0]:8d} {gg[1]:8d} {c['EE']:4d} {c['root_ori']:5d} {c['root_pos']:5d}")
print("   cluster sizes", {cl: sum(meta[n]["cl"] == cl for n in names) for cl in cls})

# why ours fails jump/aerial: reasons + ref phase over our line (13 ckpts) for segments with p_ours < p_bj27 or < p_leader
print("\n== jump/aerial where ours_line < leader or bj27 line: our failure reasons and phase (summed over 13 ckpts)")
J = [n for n in names if meta[n]["cl"] == "jump/aerial" and (prob["ours_line"][n] < prob["leader"][n] - 0.3 or prob["ours_line"][n] < prob["bj27_floor"][n] - 0.3)]
rc, ph = collections.Counter(), collections.Counter()
for n in J:
    for d in GROUPS["ours_line"]:
        r = load(d)[n]
        if not r["ok"]:
            rc[reason(r)] += 1; ph[phase_at(r, r["failure_step"])] += 1
print(f"   {len(J)} segs; reasons {dict(rc)}; phase {dict(ph)}")
for n in J:
    print(f"     {meta[n]['src'][:13]:13s} ours={prob['ours_line'][n]:.2f} bj27={prob['bj27_floor'][n]:.2f} L={prob['leader'][n]:.0f} ro={prob['bj27_ro'][n]:.2f} roroot={prob['bj27_roroot'][n]:.2f} {re.sub(r'__[0-9a-f]+$', '', n)[:60]}")
