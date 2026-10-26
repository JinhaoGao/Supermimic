"""Where does the reset sampler put its mass? Offline, from a checkpoint's bin_G / bin_visits and the library index.

  python sampler_analysis.py <model.pt> <cache dir lib_*>

Rebuilds the bins exactly like MotionLibrary (OK clips, 50-frame bins), the source / keyword prior exactly like
LibraryMotionCommand, and the mixture 0.5 prior + 0.4 softmax(-25 G / 10) + 0.1 history (history ~ softmax here).
Reports the softmax concentration, which sources / keywords / clip positions hold it, and what a per-bin cap at
c x uniform would do (water-filling).
"""

import json
import os
import sys
from collections import Counter

import numpy as np
import torch


def cap_probs(p: torch.Tensor, cap: float, iters: int = 50) -> torch.Tensor:
  """Clamp each entry at `cap`, give the excess to the uncapped entries in proportion, repeat."""
  p = p.clone()
  for _ in range(iters):
    over = p > cap
    if not bool(over.any()):
      break
    excess = (p[over] - cap).sum()
    p[over] = cap
    free = ~over & (p < cap)
    p[free] += excess * p[free] / p[free].sum()
  return p / p.sum()


def main():
  ck_path, cdir = sys.argv[1], sys.argv[2]
  ck = torch.load(ck_path, map_location="cpu", weights_only=False)
  cfg = ck["cfg"]
  G, visits = ck["bin_G"].float().cpu(), ck["bin_visits"].float().cpu()
  clips_all = json.load(open(os.path.join(cdir, "clips.json")))
  clips = [c for c in clips_all if c.get("ok", True) and c["end"] > c["start"] + 1]
  bclip, bpos, blast = [], [], []
  for i, c in enumerate(clips):
    n = (c["end"] - c["start"] + 49) // 50
    bclip += [i] * n
    bpos += list(range(n))
    blast += [False] * (n - 1) + [True]
  bclip = torch.tensor(bclip)
  blast = torch.tensor(blast)
  NB = len(bclip)
  assert NB == G.numel(), (NB, G.numel())
  names = [c["name"] for c in clips]
  sw = cfg.get("source_weights") or ()
  nw = cfg.get("name_weights") or ()
  print(f"bins {NB}, clips {len(clips)}, visited {(visits > 0).float().mean():.3f}; source_weights {len(sw)}, name_weights {len(nw)}")
  kws = sorted(((str(k), float(v)) for k, v in sw), key=lambda kv: -len(kv[0]))
  nws = [(str(k).lower(), float(v)) for k, v in nw]

  def src_of(name):
    src = name.split("__", 1)[-1]
    for k, _ in kws:
      if src == k or src.startswith(k + "_"):
        return k
    return src.split("_")[0]

  def src_w(name):
    src = name.split("__", 1)[-1]
    for k, v in kws:
      if src == k or src.startswith(k + "_"):
        return v
    return 1.0

  w_clip = torch.tensor([max(src_w(n), max([v for k, v in nws if k in n.lower()], default=1.0)) for n in names])
  prior = w_clip[bclip]
  prior = prior / prior.sum()
  gmax, temp = 10.0, 25.0
  Gc = G.clamp(0.0, gmax)
  Gs = torch.where(visits > 0, Gc, torch.full_like(Gc, gmax * 0.5))
  soft = torch.softmax(-temp * Gs / gmax, dim=0)
  probs = 0.5 * prior + 0.4 * soft + 0.1 * soft
  u = 1.0 / NB
  print(f"uniform 1/NB = {u:.2e}; probs top1 {probs.max():.2e} ({probs.max() / u:.0f}x uniform)")
  srt, idx = torch.sort(soft, descending=True)
  cs = torch.cumsum(srt, 0)
  for q in (0.5, 0.8, 0.9, 0.99):
    k = int(torch.searchsorted(cs, torch.tensor(q)).item()) + 1
    print(f"softmax: {q:.0%} of its mass in the top {k} bins ({k / NB:.2%} of bins), G there <= {Gs[idx[k - 1]]:.2f}")
  for thr in (0.5, 1.0, 2.0, 4.0):
    m = Gs < thr
    print(f"bins with G < {thr}: {m.float().mean():.2%} of bins, {soft[m].sum():.1%} of the softmax mass")
  print(f"last bin of a clip: {blast.float().mean():.2%} of bins, {soft[blast].sum():.1%} of the softmax mass, "
        f"{prior[blast].sum():.1%} of the prior")
  # mass by source and keyword
  src_bin = [src_of(names[i]) for i in bclip.tolist()]
  by_src_soft, by_src_prior, by_src_bins = Counter(), Counter(), Counter()
  sv, pv = soft.tolist(), prior.tolist()
  for s, a, b in zip(src_bin, sv, pv):
    by_src_soft[s] += a
    by_src_prior[s] += b
    by_src_bins[s] += 1
  print("source: bins% | prior% | softmax% (top 15 by softmax)")
  for s, v in by_src_soft.most_common(15):
    print(f"  {s:28s} {by_src_bins[s] / NB:7.2%} | {by_src_prior[s]:7.2%} | {v:7.2%}")
  # clips that hold the softmax mass: how many distinct clips in the top 90 %
  k90 = int(torch.searchsorted(cs, torch.tensor(0.9)).item()) + 1
  top_clips = Counter(bclip[idx[:k90]].tolist())
  print(f"top-90% softmax bins come from {len(top_clips)} clips; the 15 heaviest:")
  clip_soft = torch.zeros(len(clips)).index_add_(0, bclip, soft)
  cl_sorted = torch.argsort(clip_soft, descending=True)[:15]
  clip_G = torch.zeros(len(clips)).index_add_(0, bclip, Gs) / torch.bincount(bclip, minlength=len(clips)).float()
  for ci in cl_sorted.tolist():
    print(f"  {clip_soft[ci]:6.2%}  meanG {clip_G[ci]:5.2f}  len {(clips[ci]['end'] - clips[ci]['start']) / 50:6.1f}s  {names[ci][:90]}")
  # cap
  for c in (50.0, 200.0):
    sc = cap_probs(soft, c * u)
    kk = int(torch.searchsorted(torch.cumsum(torch.sort(sc, descending=True)[0], 0), torch.tensor(0.9)).item()) + 1
    print(f"cap {c:.0f}x uniform: softmax top1 {sc.max() / u:.0f}x, 90% of mass over {kk} bins ({kk / NB:.2%}); "
          f"G<1 bins hold {sc[Gs < 1].sum():.1%}, last-bin {sc[blast].sum():.1%}")


if __name__ == "__main__":
  main()
