"""Build (and cache) the bin clusters of a training library, and show how the uniform-share prior moves by source.

  PYTHONPATH=<repo> python cluster_build.py <model.pt of a run on that library> <cache dir lib_*> [K] [alpha]
"""

import json
import os
import sys
import time
import types
from collections import Counter

import numpy as np
import torch

from supermimic.bin_clusters import bin_cluster_prior


def load_lib(cdir: str):
  """Library-like view of a cache dir (no rebuild risk): mmap fields, OK clips, 50-frame bins (as MotionLibrary)."""
  meta = json.load(open(os.path.join(cdir, "meta.json")))
  clips_all = json.load(open(os.path.join(cdir, "clips.json")))
  clips = [types.SimpleNamespace(name=c["name"], start=c["start"], end=c["end"]) for c in clips_all
           if c.get("ok", True) and c["end"] > c["start"] + 1]
  fields = {k: torch.from_numpy(np.load(os.path.join(cdir, f"{k}.npy"), mmap_mode="r")) for k in ("body_pos", "body_quat", "body_ang_vel", "body_lin_vel")}
  starts, ends, bclip = [], [], []
  for i, c in enumerate(clips):
    for a in range(c.start, c.end, 50):
      starts.append(a)
      ends.append(min(a + 50, c.end))
      bclip.append(i)
  return types.SimpleNamespace(cache_dir=cdir, body_names=tuple(meta["body_names"]), time_step_total=int(meta["frames"]),
                               fields=fields, clips=clips, num_clips=len(clips), num_bins=len(starts),
                               bin_start=torch.tensor(starts), bin_end=torch.tensor(ends), bin_clip=torch.tensor(bclip))


def main():
  ck_path, cdir = sys.argv[1], sys.argv[2]
  K = int(sys.argv[3]) if len(sys.argv) > 3 else 256
  alpha = float(sys.argv[4]) if len(sys.argv) > 4 else 0.3
  ck = torch.load(ck_path, map_location="cpu", weights_only=False)
  cfg = ck["cfg"]
  t0 = time.time()
  lib = load_lib(cdir)
  assert lib.num_bins == ck["bin_G"].numel(), (lib.num_bins, ck["bin_G"].numel())
  print(f"library {lib.cache_dir}: {lib.num_bins} bins, bodies {lib.body_names} ({time.time() - t0:.0f}s)")
  t0 = time.time()
  w = bin_cluster_prior(lib, K, alpha, "cpu")
  print(f"cluster prior in {time.time() - t0:.0f}s")
  sw = cfg.get("source_weights") or ()
  nw = cfg.get("name_weights") or ()
  kws = sorted(((str(k), float(v)) for k, v in sw), key=lambda kv: -len(kv[0]))
  nws = [(str(k).lower(), float(v)) for k, v in nw]

  def src_of(name):
    src = name.split("__", 1)[-1]
    for k, _ in kws:
      if src == k or src.startswith(k + "_"):
        return k
    return src.split("_")[0]

  def wt(name):
    src = name.split("__", 1)[-1]
    sv = next((v for k, v in kws if src == k or src.startswith(k + "_")), 1.0)
    return max(sv, max([v for k, v in nws if k in name.lower()], default=1.0))

  bclip = lib.bin_clip.cpu()
  wc = torch.tensor([wt(c.name) for c in lib.clips])[bclip]
  base = wc / wc.sum()
  both = base * w
  both = both / both.sum()
  src = [src_of(lib.clips[i].name) for i in bclip.tolist()]
  agg = {k: Counter() for k in ("bins", "base", "clus", "both")}
  for s, b, c, x in zip(src, base.tolist(), w.tolist(), both.tolist()):
    agg["bins"][s] += 1
    agg["base"][s] += b
    agg["clus"][s] += c
    agg["both"][s] += x
  NB = lib.num_bins
  print("source: bins% | source/keyword prior% | cluster prior% | both%")
  for s, _ in agg["both"].most_common(16):
    print(f"  {s:26s} {agg['bins'][s] / NB:7.2%} | {agg['base'][s]:7.2%} | {agg['clus'][s]:7.2%} | {agg['both'][s]:7.2%}")
  flips = torch.tensor([any(k in lib.clips[i].name.lower() for k in ("kongfan", "shoufan", "houfan", "qianfan", "flip", "cartwheel", "robot_acro", "空翻"))
                        for i in range(lib.num_clips)])[bclip]
  ground = torch.tensor([any(k in lib.clips[i].name.lower() for k in ("getup", "get_up", "lying", "prone", "crawl", "fallandgetup", "ground_recovery"))
                         for i in range(lib.num_clips)])[bclip]
  for nm, m in (("flips", flips), ("ground / getup", ground)):
    print(f"{nm}: {m.float().mean():.3%} of bins; prior {base[m].sum():.2%} | clusters {w[m].sum():.2%} | both {both[m].sum():.2%}")


if __name__ == "__main__":
  main()
