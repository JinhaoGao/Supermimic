import json, sys, numpy as np
CDIR = "/data2/supermimic/cache/lib_254518_926db30c44"
TEMP, GMAX, BL = 25.0, 10.0, 50
clips_all = json.load(open(f"{CDIR}/clips.json"))
clips = [c for c in clips_all if c.get("ok", True) and c["end"] > c["start"] + 1]
b_clip, b_idx, b_n = [], [], []
for i, c in enumerate(clips):
    nb = (c["end"] - c["start"] + BL - 1) // BL
    b_clip += [i] * nb; b_idx += list(range(nb)); b_n += [nb] * nb
b_clip, b_idx, b_n = np.array(b_clip), np.array(b_idx), np.array(b_n)
NB = len(b_clip)
src = np.array([c["name"].split("__", 1)[-1].split("_")[0] for c in clips])
print(f"clips {len(clips)} bins {NB}")
# source prior (as reset_prior): weights by longest source prefix
SW = {"noetix_kungfu": 10, "noetix_kungfu_dance_show": 15, "sportsmeeting": 10, "lafan1": 15, "g1_all": 10, "mixamo_cn": 2, "virtual_point": 3}
kws = sorted(SW.items(), key=lambda kv: -len(kv[0]))
def sw(name):
    s = name.split("__", 1)[-1]
    for k, v in kws:
        if s == k or s.startswith(k + "_"): return k, v
    return s.split("_")[0], 1.0
tags = [sw(c["name"]) for c in clips]
wclip = np.array([t[1] for t in tags]); sclip = np.array([t[0] for t in tags])
prior = wclip[b_clip]; prior = prior / prior.sum()
prev = None
for it in sys.argv[1:]:
    d = np.load(f"/data2/supermimic/diag_1024/taskC/ck/ck_{it}.npz")
    G = d["bin_G"].astype(np.float64); V = d["bin_visits"].astype(np.float64)
    assert G.shape[0] == NB
    vis = V > 0
    Ge = np.where(vis, G, GMAX * 0.5)
    z = -TEMP * Ge / GMAX
    soft = np.exp(z - z.max()); soft /= soft.sum()
    order = np.argsort(-soft)
    cum = np.cumsum(soft[order])
    print(f"\n=== {it}: visited {vis.mean():.4f}  G(visited) pct 0/0.1/1/50/99/100 = " + " ".join(f"{np.percentile(G[vis], q):.4g}" for q in (0, 0.1, 1, 50, 99, 100))
          + f"  #G<0 {int((G[vis] < 0).sum())}  softmax top1/top10/top100 {cum[0]:.4f}/{cum[9]:.4f}/{cum[99]:.4f}")
    for r in order[:5]:
        c = clips[b_clip[r]]
        print(f"   bin {r:8d} soft {soft[r]:.4f} G {G[r]:10.3f} visits {V[r]:10.0f} bin {b_idx[r]}/{b_n[r]-1}{' LAST' if b_idx[r]==b_n[r]-1 else ''} frames {c['end']-c['start']} {c['name'][:80]}")
    # visit share by source since previous checkpoint
    if prev is not None:
        dV = V - prev
        tot = dV.sum()
        bysrc = {}
        for s in np.unique(sclip):
            m = sclip[b_clip] == s
            bysrc[s] = dV[m].sum() / tot
        top = sorted(bysrc.items(), key=lambda x: -x[1])[:14]
        trap = order[0]
        print(f"   visits since prev ckpt: total {tot:.4g}; trap bin share {dV[trap] / tot:.4f}; by source " + ", ".join(f"{k} {v:.3f}" for k, v in top))
        # prior share per source for comparison
    prev = V
    # G by source (mean over visited bins), to see what the sampler thinks is hard
    gs = {}
    for s in ("gmt", "sonic", "noetix_kungfu", "noetix_kungfu_dance_show", "sportsmeeting", "lafan1", "hiphi", "g1_all", "mixamo_cn", "phuma", "motiondecode", "virtual_point"):
        m = (sclip[b_clip] == s) & vis
        if m.any(): gs[s] = (np.median(G[m]), np.percentile(G[m], 5), m.sum())
    print("   G median/p5 by source: " + ", ".join(f"{k} {v[0]:.2f}/{v[1]:.2f}" for k, v in gs.items()))
