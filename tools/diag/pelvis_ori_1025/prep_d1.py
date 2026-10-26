"""D1 / D3 prep (single process): heads cfg of the bundle == the checkpoint's; MotionLibrary cache over the D1 list (the
trainer's cache builder); the NEW code's setup on the training model; the train-mode reset perturbation (for the report).
  python prep_d1.py <list.tsv> <out.json>"""

from __future__ import annotations

import dataclasses
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

import mujoco  # noqa: E402
import torch  # noqa: E402

from supermimic import env_cfg as ec  # noqa: E402
from supermimic.heads import HeadsCfg  # noqa: E402
from supermimic.motions import MotionLibrary  # noqa: E402

CKPT = "/data2/supermimic/logs/2026-10-25_12-31-57_g1v2_r15ws_ddp8/model_148000.pt"


def main():
  lst, out = sys.argv[1], sys.argv[2]
  files = [line.split("\t")[0] for line in open(lst).read().splitlines()[1:]]
  meta = common.load_meta()
  res = dict(n_files=len(files), bundle=common.BUNDLE, repo=common.REPO, commit=open(os.path.join(common.REPO, "COMMIT")).read().strip())
  ck = torch.load(CKPT, map_location="cpu", weights_only=False, mmap=True)  # read-only
  h_ck = dataclasses.asdict(HeadsCfg.from_dict(ck["cfg"]["heads"]))
  h_meta = dataclasses.asdict(HeadsCfg.from_dict(dict(meta["heads"])))
  res["heads_ckpt_eq_bundle"] = h_ck == h_meta
  res["heads_diff"] = {k: (h_ck[k], h_meta.get(k)) for k in h_ck if h_ck[k] != h_meta.get(k)}
  res["ckpt_it"] = int(ck["it"])
  res["ckpt_robot"] = (ck["cfg"].get("robot"), ck["cfg"].get("robot_variant"))
  del ck
  lib = MotionLibrary(files, tuple(meta["joint_names"]), tuple(meta["body_names"]), "cpu", cache_dir=common.lib_cache_dir(), num_workers=8)
  res["lib"] = dict(cache_dir=lib.cache_dir, clips=lib.num_clips, frames=lib.time_step_total, summary=lib.summary())
  model = mujoco.MjModel.from_binary_path(os.path.join(common.BUNDLE, "scene.mjb"))
  S = common.Setup(model, meta)
  res["setup"] = S.describe()
  tr = ec.make_env_cfg(["x.npz"], 1, mode="train", robot=meta["robot"], robot_variant=meta["robot_variant"]).commands["motion"]
  res["train_reset_perturbation"] = dict(pose_range=dict(tr.pose_range), velocity_range=dict(tr.velocity_range), joint_position_range=tr.joint_position_range)
  json.dump(res, open(out, "w"), indent=1, default=str)
  print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
  main()
