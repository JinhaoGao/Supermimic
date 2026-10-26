"""Read-only probe of the reset sampler state saved in g1_full_ee_ddp8 checkpoints (CPU, bj31).
Rebuilds the bin table exactly like MotionLibrary (ok clips with end > start + 1, 50-frame bins), recomputes the
softmax part of sampling_probabilities() (temp 25, g_max 10, unvisited G = 5) and reports which bins hold the mass.
  python sampler_probe.py <it> [<it> ...]"""
import json, sys
import numpy as np
import torch

torch.set_num_threads(4)
import os as _o; RUN = _o.environ["RUN_DIR"]
CDIR = _o.environ["CDIR"]
TEMP, GMAX, BL = 25.0, 10.0, 50

clips_all = json.load(open(f"{CDIR}/clips.json"))
clips = [c for c in clips_all if c.get("ok", True) and c["end"] > c["start"] + 1]
b_clip, b_idx, b_n = [], [], []
for i, c in enumerate(clips):
    nb = (c["end"] - c["start"] + BL - 1) // BL
    b_clip += [i] * nb; b_idx += list(range(nb)); b_n += [nb] * nb
b_clip, b_idx, b_n = np.array(b_clip), np.array(b_idx), np.array(b_n)
NB = len(b_clip)
print(f"clips {len(clips)} bins {NB}")

for it in sys.argv[1:]:
    ck = torch.load(f"{RUN}/model_{it}.pt", map_location="cpu", weights_only=False)
    G = ck["bin_G"].double().numpy(); V = ck["bin_visits"].double().numpy()
    assert G.shape[0] == NB, (G.shape, NB)
    Ge = np.where(V > 0, G, GMAX * 0.5)
    z = -TEMP * Ge / GMAX
    soft = np.exp(z - z.max()); soft /= soft.sum()
    order = np.argsort(-soft)
    vis = V > 0
    print(f"\n=== model_{it}: visited {vis.mean():.4f}  G(visited) pct 0/1/5/50/95/99/100 = "
          + " ".join(f"{np.percentile(G[vis], q):.3g}" for q in (0, 1, 5, 50, 95, 99, 100))
          + f"  #G<0 {int((G[vis] < 0).sum())}  #G<2 {int((G[vis] < 2).sum())}")
    cum = np.cumsum(soft[order])
    print("  softmax mass top1/top10/top100/top1000 = " + " ".join(f"{cum[k - 1]:.4f}" for k in (1, 10, 100, 1000))
          + f"  -> overall reset prob top1 ~ {0.5 / NB + 0.4 * soft[order[0]]:.3f} (+ history 0.1)")
    for r in order[:12]:
        c = clips[b_clip[r]]
        print(f"  bin {r:8d} soft {soft[r]:.4f} G {G[r]:9.3f} visits {V[r]:9.0f}  bin {b_idx[r]}/{b_n[r] - 1}"
              f"{' LAST' if b_idx[r] == b_n[r] - 1 else ''}  frames {c['end'] - c['start']}  {c['name'][:90]}")
    # are low-G bins mostly clip-final bins?
    low = np.where(vis & (G < np.percentile(G[vis], 0.1)))[0]
    print(f"  lowest 0.1% bins ({len(low)}): last-bin share {np.mean(b_idx[low] == b_n[low] - 1):.3f} (all bins: {np.mean(b_idx == b_n - 1):.3f})")
