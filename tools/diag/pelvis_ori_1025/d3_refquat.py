"""D3: the reference pelvis quaternion the NEW code passes to the root-orientation test equals the npz pelvis body
quaternion and the eval site's reference (angle error < 1e-3 rad), on every frame of 20 sampled clips per set:

  A. 20 of the D1 list (eval-site npz, MotionLibrary cache of D1): the simulator side (fns._post: StepFns._ref_root of
     lib.gather(...)["body_quat"]), the imagination side (fns._imagine_step: _ref_root of the pack from lib.gather_windows)
     and the env term's body index (bad_root_ori_full: cmd.cfg.body_names.index(body_name) on cmd.body_quat_w) vs the npz's
     body_quat_w[:, body_names.index("pelvis")] and vs the eval site's MotionData.root_quat(t) (its first tracked body);
  B. 20 random clips of the LIVE run's library (cache lib_255008_3b84d5f141, memory-mapped read-only, through
     MotionLibrary.gather) vs their training npz (no body_names key: 37-body fallback order) read by the eval site's own
     MotionData (independent fallback) and by supermimic.motions.npz_body_names.
  python d3_refquat.py <d1_list.tsv> <out_prefix>"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from d1_replay import site_tracked_names  # noqa: E402
from supermimic.math_utils import quat_angle  # noqa: E402
from supermimic.motions import MotionLibrary, npz_body_names  # noqa: E402

LIVE_CACHE = "/data2/supermimic/cache/lib_255008_3b84d5f141"
LIVE_LIST = "/data2/supermimic/prep_1025/g1v2_r15_255008_bj31.txt"
TOL = 1e-3


def errs(ev, q_lib: np.ndarray, q_other: np.ndarray) -> dict:
  a_site = ev.quat_error_angle(q_other.astype(np.float64), q_lib.astype(np.float64))  # eval site's formula (normalizes)
  a_new = quat_angle(torch.tensor(q_other, dtype=torch.float32), torch.tensor(q_lib, dtype=torch.float32)).numpy()  # new code's
  return dict(max_angle_site_fn=float(np.max(a_site)), max_angle_new_fn=float(np.max(a_new)), max_abs_comp=float(np.max(np.abs(q_lib - q_other))),
              max_abs_comp_signfree=float(np.max(np.minimum(np.abs(q_lib - q_other), np.abs(q_lib + q_other)).max(-1))))


def main():
  lst, outp = sys.argv[1], sys.argv[2]
  rows = [dict(zip(["path"], line.split("\t")[:1])) for line in open(lst).read().splitlines()[1:]]
  files = [r["path"] for r in rows]
  meta = common.load_meta()
  model = mujoco.MjModel.from_binary_path(os.path.join(common.BUNDLE, "scene.mjb"))
  S = common.Setup(model, meta)
  ev = common.eval_site_module()
  site_names = site_tracked_names(ev, files[0])
  env_idx = S.tracked.index(S.env_term_params["body_name"])
  assert env_idx == S.ro_idx == S.h_on.root_ori, (env_idx, S.ro_idx, S.h_on.root_ori)
  out = dict(setup=dict(root_ori_body=S.ro_name, root_ori_index=S.ro_idx, env_body=S.env_term_params["body_name"], site_tracked0=site_names[0]), A=[], B=[])
  # ---- A: eval-site files through the D1 library ------------------------------------------------------------------
  lib = MotionLibrary(files, tuple(meta["joint_names"]), tuple(meta["body_names"]), "cpu", cache_dir=common.lib_cache_dir(), log=lambda *a: None)
  clip_of = {c.name: c for c in lib.clips}
  rng = np.random.default_rng(0)
  pick = sorted(rng.choice(len(files), 20, replace=False).tolist())
  for i in pick:
    p = files[i]
    c = clip_of[os.path.basename(p)]
    fr = torch.arange(c.start, c.end)
    bq = lib.gather(fr, fields=["body_quat"])["body_quat"]  # [T,Bt,4] (simulator side: current frames)
    q_post = S.f_on._ref_root(bq).numpy()
    pack = lib.gather_windows(torch.tensor([c.start]), c.length, fields=["body_quat"])["body_quat"]  # [1,T,Bt,4] (imagination pack)
    ar = torch.zeros(c.length, dtype=torch.long)
    q_imag = S.f_on._ref_root(pack[ar, torch.arange(c.length)]).numpy()
    q_env = bq[:, env_idx].numpy()
    d = np.load(p, allow_pickle=True)
    names = [str(x) for x in d["body_names"]]
    q_npz = np.asarray(d["body_quat_w"])[:, names.index("pelvis")]
    mot = ev.MotionData(p, site_names)
    q_site = np.stack([mot.root_quat(t) for t in range(mot.length)])
    rec = dict(clip=os.path.basename(p), T=c.length, site_anchor=mot.anchor_body, site_T=mot.length,
               post_eq_imag=bool(np.array_equal(q_post, q_imag)), post_eq_env=bool(np.array_equal(q_post, q_env)), post_eq_npz_bitwise=bool(np.array_equal(q_post, q_npz)),
               vs_npz=errs(ev, q_post, q_npz), vs_site=errs(ev, q_post, q_site))
    rec["pass"] = bool(rec["post_eq_imag"] and rec["post_eq_env"] and mot.anchor_body == "pelvis" and mot.length == c.length
                       and max(rec["vs_npz"]["max_angle_site_fn"], rec["vs_site"]["max_angle_site_fn"], rec["vs_site"]["max_angle_new_fn"]) < TOL)
    out["A"].append(rec)
    print("A", rec["clip"], rec["pass"], rec["vs_npz"]["max_angle_site_fn"], rec["vs_site"]["max_angle_site_fn"], flush=True)
  # ---- B: the live run's library (read-only memory map) -----------------------------------------------------------
  lm = json.load(open(os.path.join(LIVE_CACHE, "meta.json")))
  assert tuple(lm["body_names"]) == S.tracked and tuple(lm["joint_names"]) == tuple(meta["joint_names"])
  clips = json.load(open(os.path.join(LIVE_CACHE, "clips.json")))
  paths = open(LIVE_LIST).read().splitlines()
  assert len(clips) == len(paths) == lm["files"]
  live = MotionLibrary.__new__(MotionLibrary)  # gather() on the mapped cache, no 190 GB RAM copy
  live.device = "cpu"
  live.time_step_total = int(lm["frames"])
  arr = np.load(os.path.join(LIVE_CACHE, "body_quat.npy"), mmap_mode="r")
  import warnings

  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    live.fields = {"body_quat": torch.from_numpy(arr)}
  rng = np.random.default_rng(1)
  pickB = sorted(rng.choice(len(clips), 20, replace=False).tolist())
  for i in pickB:
    c, p = clips[i], paths[i]
    assert c["name"] == os.path.basename(p) and c["ok"], (c, p)
    fr = torch.arange(c["start"], c["end"])
    q_live = S.f_on._ref_root(live.gather(fr, fields=["body_quat"])["body_quat"]).numpy()
    d = np.load(p, allow_pickle=True)
    nb = int(d["body_quat_w"].shape[1])
    sm_names, fb = npz_body_names(d, nb)
    q_npz = np.asarray(d["body_quat_w"])[:, sm_names.index("pelvis")]
    mot = ev.MotionData(p, site_names)
    q_site = np.stack([mot.root_quat(t) for t in range(mot.length)])
    rec = dict(clip=c["name"], T=c["end"] - c["start"], npz_bodies=nb, supermimic_fallback=fb, site_body_mapping=mot.body_mapping_source,
               site_anchor=mot.anchor_body, site_anchor_idx=int(mot.anchor_idx), site_T=mot.length,
               live_eq_npz_bitwise=bool(np.array_equal(q_live, q_npz)), vs_npz=errs(ev, q_live, q_npz), vs_site=errs(ev, q_live, q_site))
    rec["pass"] = bool(mot.anchor_body == "pelvis" and mot.length == rec["T"]
                       and max(rec["vs_npz"]["max_angle_site_fn"], rec["vs_site"]["max_angle_site_fn"], rec["vs_site"]["max_angle_new_fn"]) < TOL)
    out["B"].append(rec)
    print("B", rec["clip"], rec["pass"], rec["vs_npz"]["max_angle_site_fn"], rec["vs_site"]["max_angle_site_fn"], flush=True)
  for k in ("A", "B"):
    R = out[k]
    out[f"summary_{k}"] = dict(n=len(R), passed=sum(r["pass"] for r in R), max_angle_vs_npz=max(r["vs_npz"]["max_angle_site_fn"] for r in R),
                               max_angle_vs_site=max(r["vs_site"]["max_angle_site_fn"] for r in R),
                               max_angle_vs_site_newfn=max(r["vs_site"]["max_angle_new_fn"] for r in R),
                               bitwise_npz=sum(r.get("post_eq_npz_bitwise", r.get("live_eq_npz_bitwise", False)) for r in R),
                               frames=sum(r["T"] for r in R))
  json.dump(out, open(outp + ".json", "w"), indent=1)
  print(json.dumps({k: v for k, v in out.items() if k.startswith(("summary", "setup"))}, indent=1))


if __name__ == "__main__":
  main()
