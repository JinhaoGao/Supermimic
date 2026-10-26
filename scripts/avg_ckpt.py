"""Average the policy weights (incl. input normalizer buffers) of several checkpoints of one training chain.

  python scripts/avg_ckpt.py --ckpts a/model_200000.pt b/model_204000.pt c/model_208000.pt --out_run g1v2_avg3_r25

Writes <log_root>/<date>_<out_run>/model_<it of the last ckpt>.pt: every floating-point tensor of ck["policy"] is the
mean over the checkpoints, everything else (critic, world model, optimizer, cfg, integer buffers) comes from the last
one, so the result exports / evaluates like any checkpoint (export_bundle.py, submit_v4.sh). train_cfg.json and
motion_files.txt are copied from the last checkpoint's run dir. Only meant for checkpoints of one chain
(same architecture, consecutive resumes), where the weights stay in one basin.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil

import torch


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--ckpts", nargs="+", required=True)
  p.add_argument("--out_run", required=True)
  p.add_argument("--log_root", default=os.path.expanduser("~/supermimic/logs"))
  args = p.parse_args()
  cks = [torch.load(c, map_location="cpu", weights_only=False) for c in args.ckpts]
  last = cks[-1]
  keys = list(last["policy"].keys())
  for c, ck in zip(args.ckpts, cks):
    assert list(ck["policy"].keys()) == keys, f"{c}: policy keys differ"
    assert ck["dims"] == last["dims"], f"{c}: dims differ"
  avg, n_avg = {}, 0
  for k in keys:
    t = last["policy"][k]
    if torch.is_floating_point(t):
      avg[k] = torch.stack([ck["policy"][k].double() for ck in cks]).mean(0).to(t.dtype)
      n_avg += 1
    else:
      avg[k] = t.clone()
  out = dict(last)
  out["policy"] = avg
  out["avg_of"] = [os.path.abspath(c) for c in args.ckpts]
  src_dir = os.path.dirname(os.path.abspath(args.ckpts[-1]))
  d = os.path.join(args.log_root, f"{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}_{args.out_run}")
  os.makedirs(d)
  path = os.path.join(d, f"model_{last['it']}.pt")
  torch.save(out, path)
  for f in ("train_cfg.json", "motion_files.txt"):
    if os.path.exists(os.path.join(src_dir, f)):
      shutil.copy(os.path.join(src_dir, f), d)
  with open(os.path.join(d, "avg_of.json"), "w") as f:
    json.dump({"ckpts": out["avg_of"], "its": [int(ck["it"]) for ck in cks]}, f, indent=1)
  drift = max(float((cks[0]["policy"][k].double() - last["policy"][k].double()).abs().max()) for k in keys
              if torch.is_floating_point(last["policy"][k]))
  print(f"[avg_ckpt] {path}: {n_avg}/{len(keys)} policy tensors averaged over its {[int(ck['it']) for ck in cks]} "
        f"(max |first - last| {drift:.3e})")


if __name__ == "__main__":
  main()
