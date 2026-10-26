"""What would the softmax share of the reset sampler look like if bin returns were clamped to [0, g_max]?
Read-only, CPU, bj31. Uses the bin_G / bin_visits saved in a g1_full_ee_ddp8 checkpoint.
  python sampler_sim_clamp.py <it>"""
import collections, json, re, sys
import numpy as np
import torch

torch.set_num_threads(4)
import glob as _g, os as _o; RUN = _o.environ.get("RUN_DIR") or sorted(_g.glob("/data2/supermimic/logs/*_g1_full_ee_ddp8_r52"))[-1]
CDIR = "/data2/supermimic/cache/lib_365855_b608e70f1d"
TEMP, GMAX, BL = 25.0, 10.0, 50
it = sys.argv[1]

clips_all = json.load(open(f"{CDIR}/clips.json"))
clips = [c for c in clips_all if c.get("ok", True) and c["end"] > c["start"] + 1]
b_clip, b_idx, b_n = [], [], []
for i, c in enumerate(clips):
    nb = (c["end"] - c["start"] + BL - 1) // BL
    b_clip += [i] * nb; b_idx += list(range(nb)); b_n += [nb] * nb
b_clip, b_idx, b_n = np.array(b_clip), np.array(b_idx), np.array(b_n)
last = b_idx == b_n - 1


def source(name):
    # "<robot>__<source>_..." ; group by layer + first source token(s)
    layer, rest = name.split("__", 1) if "__" in name else ("?", name)
    tok = re.split(r"[_\-]", rest)
    key = tok[0] if tok[0] not in ("g1", "all", "noetix") else "_".join(tok[:3])
    return f"{layer}:{key}"


src_of_clip = np.array([source(c["name"]) for c in clips])
src_of_bin = src_of_clip[b_clip]

ck = torch.load(f"{RUN}/model_{it}.pt", map_location="cpu", weights_only=False)
G = ck["bin_G"].double().numpy(); V = ck["bin_visits"].double().numpy()


def soft_of(Gx):
    Ge = np.where(V > 0, Gx, GMAX * 0.5)
    z = -TEMP * Ge / GMAX
    s = np.exp(z - z.max()); return s / s.sum()


for label, Gx in (("clamped (as trained)", np.clip(G, 0.0, GMAX)), ("clamped-dup", np.clip(G, 0.0, GMAX)))[:1]:
    s = soft_of(Gx)
    o = np.argsort(-s); cum = np.cumsum(s[o])
    neff = float(np.exp(-(s[s > 0] * np.log(s[s > 0])).sum()))
    print(f"\n=== model_{it} {label}: top1 {s[o[0]]:.4f}  top100 {cum[99]:.3f} top1k {cum[999]:.3f} top10k {cum[9999]:.3f} top100k {cum[99999]:.3f}"
          f"  eff.bins {neff:.0f}  last-bin share {s[last].sum():.3f}  unvisited share {s[V == 0].sum():.3f}")
    if True:
        agg = collections.Counter(); uni = collections.Counter()
        for k, m in zip(src_of_bin, s):
            agg[k] += m
        for k in src_of_bin:
            uni[k] += 1.0 / len(src_of_bin)
        print("  softmax share by source (top 25)        soft%   uniform%")
        for k, m in agg.most_common(25):
            print(f"  {k[:40]:40s} {100 * m:7.2f} {100 * uni[k]:9.3f}")
        print("  top 15 bins:")
        for r in o[:15]:
            c = clips[b_clip[r]]
            print(f"   G {G[r]:8.3f} visits {V[r]:9.0f} bin {b_idx[r]}/{b_n[r] - 1}{' LAST' if last[r] else ''}  {c['name'][:95]}")
