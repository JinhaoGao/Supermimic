"""Checks for the npz body-name fallback and name-based clip selection (numpy + torch, no mjlab / GPU).

  python tests/test_motions_names.py      # or: pytest tests/test_motions_names.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from supermimic.motions import (  # noqa: E402
  G1_ISAAC_BODY_NAMES_37,
  MotionLibrary,
  npz_body_names,
  read_name_list,
  resolve_clip_names,
)

TRACKED = (
  "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link", "right_hip_roll_link", "right_knee_link",
  "right_ankle_roll_link", "torso_link", "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
  "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
)
SRC_JOINTS = tuple(f"j{i}_joint" for i in range(29))
DST_JOINTS = tuple(reversed(SRC_JOINTS))  # entity order differs from the file order


def _clip(rng, T, nb=37, with_names=True):
  d = dict(
    fps=np.array([50.0]), joint_names=np.array(SRC_JOINTS),
    joint_pos=rng.standard_normal((T, 29)).astype(np.float32), joint_vel=rng.standard_normal((T, 29)).astype(np.float32),
    body_pos_w=rng.standard_normal((T, nb, 3)).astype(np.float32), body_quat_w=rng.standard_normal((T, nb, 4)).astype(np.float32),
    body_lin_vel_w=rng.standard_normal((T, nb, 3)).astype(np.float32), body_ang_vel_w=rng.standard_normal((T, nb, 3)).astype(np.float32),
  )
  if with_names:
    d["body_names"] = np.array(G1_ISAAC_BODY_NAMES_37[:nb])
  return d


def _lib(files, cache):
  return MotionLibrary(files, DST_JOINTS, TRACKED, device="cpu", cache_dir=cache, num_workers=2, log=lambda *a: None)


def test_fallback_cache_is_bitwise_identical():
  names = ["g1__a.npz", "g1__b c (空格).npz", "g1_umr__d.npz"]
  with tempfile.TemporaryDirectory() as tmp:
    for sub in ("with", "without"):
      os.makedirs(os.path.join(tmp, sub))
    for i, n in enumerate(names):
      d = _clip(np.random.default_rng(i), T=40 + 7 * i)
      np.savez(os.path.join(tmp, "with", n), **d)
      np.savez(os.path.join(tmp, "without", n), **{k: v for k, v in d.items() if k != "body_names"})
    fa = [os.path.join(tmp, "with", n) for n in names]
    fb = [os.path.join(tmp, "without", n) for n in names]
    a, b = _lib(fa, os.path.join(tmp, "ca")), _lib(fb, os.path.join(tmp, "cb"))
    for k in a.fields:
      assert a.fields[k].numpy().tobytes() == b.fields[k].numpy().tobytes(), k
    assert [(c.name, c.start, c.end) for c in a.clips] == [(c.name, c.start, c.end) for c in b.clips]
    meta_b = json.load(open(os.path.join(b.cache_dir, "meta.json")))
    assert meta_b["body_names_fallback_files"] == len(names) and meta_b["failed_files"] == 0
    # The remap really picks the tracked bodies of the Isaac order.
    d = np.load(fa[0])
    idx = [G1_ISAAC_BODY_NAMES_37.index(n) for n in TRACKED]
    np.testing.assert_array_equal(b.fields["body_pos"].numpy()[: a.clips[0].end], d["body_pos_w"][:, idx])


def test_unknown_body_count_fails_cleanly():
  with tempfile.TemporaryDirectory() as tmp:
    good = os.path.join(tmp, "good.npz")
    bad = os.path.join(tmp, "bad30.npz")
    np.savez(good, **{k: v for k, v in _clip(np.random.default_rng(1), 30).items() if k != "body_names"})
    np.savez(bad, **_clip(np.random.default_rng(2), 30, nb=30, with_names=False))
    lib = _lib([good, bad], os.path.join(tmp, "c"))
    assert [c.name for c in lib.clips] == ["good.npz"]  # the 30-body file is excluded, not a crash
    meta = json.load(open(os.path.join(lib.cache_dir, "meta.json")))
    assert meta["failed_files"] == 1 and meta["body_names_fallback_files"] == 1
    try:
      npz_body_names(np.load(bad))
      raise AssertionError("expected KeyError")
    except KeyError:
      pass
    names, fb = npz_body_names(np.load(good))
    assert fb and tuple(names) == G1_ISAAC_BODY_NAMES_37


def test_resolve_clip_names():
  lib_names = ["a.npz", "b c.npz", "中文 舞.npz", "d.npz"]
  with tempfile.TemporaryDirectory() as tmp:
    p = os.path.join(tmp, "names.txt")
    with open(p, "w", encoding="utf-8") as f:
      f.write("d.npz\n/some/dir/b c.npz\n\n中文 舞.npz\r\nmissing.npz\nd.npz\n a.npz \n")
    wanted = read_name_list(p)
    assert wanted == ["d.npz", "b c.npz", "中文 舞.npz", "missing.npz", "d.npz", " a.npz "]
    ids, missing = resolve_clip_names(lib_names, wanted, log=lambda *a: None)
    assert ids == [0, 1, 2, 3] and missing == ["missing.npz"]


if __name__ == "__main__":
  for name, fn in list(globals().items()):
    if name.startswith("test_") and callable(fn):
      fn()
      print(f"ok  {name}")
