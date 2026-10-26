"""Unit tests for the r28 pieces (CPU): cap_simplex, bin features + k-means + cluster prior, root_z_term head.

  PYTHONPATH=. python tests/test_sampler_cap_clusters_rootz.py
"""

import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # the repo root

import numpy as np
import torch


def iterative_cap(p, cap, iters=200):
  p = p.double().clone() / p.sum()
  for _ in range(iters):
    over = p > cap
    if not bool(over.any()):
      break
    ex = (p[over] - cap).sum()
    p[over] = cap
    free = ~over & (p < cap)
    p[free] += ex * p[free] / p[free].sum()
  return p / p.sum()


def test_cap_simplex():
  from supermimic.math_utils import cap_simplex

  g = torch.Generator().manual_seed(0)
  for n, temp in ((1000, 25.0), (50_000, 25.0), (200_000, 40.0)):
    G = torch.rand(n, generator=g) * 10.0
    G[: n // 500] = torch.rand(n // 500, generator=g) * 0.3  # a few very hard bins
    soft = torch.softmax(-temp * G / 10.0, 0)
    for c in (5.0, 50.0, 200.0):
      cap = c / n
      out = cap_simplex(soft, cap)
      ref = iterative_cap(soft, cap)
      assert abs(float(out.sum()) - 1.0) < 1e-5, out.sum()
      assert float(out.max()) <= cap * (1 + 1e-4), (out.max(), cap)
      assert float((out.double() - ref).abs().max()) < 1e-7, (n, c, float((out.double() - ref).abs().max()))
      # order preserved, uncapped entries scaled by one factor
      unc = out < cap * (1 - 1e-4)
      r = out[unc].double() / soft[unc].double()
      assert float(r.max() / r.min()) < 1 + 1e-4
  p = torch.full((100,), 0.01)
  assert torch.equal(cap_simplex(p, 0.05), p)  # nothing above the cap: unchanged
  print("cap_simplex ok")


def fake_lib(tmp, NB=600, L=50):
  """A library-like object: bins of L frames, 14 tracked G1 bodies; three motion types."""
  names = ("pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link", "right_hip_roll_link", "right_knee_link",
           "right_ankle_roll_link", "torso_link", "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
           "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link")
  T = NB * L
  g = torch.Generator().manual_seed(1)
  pos = torch.zeros(T, 14, 3)
  quat = torch.zeros(T, 14, 4)
  quat[..., 0] = 1.0
  ang = torch.zeros(T, 14, 3, dtype=torch.float16)
  lin = torch.zeros(T, 14, 3, dtype=torch.float16)
  kind = torch.zeros(NB, dtype=torch.long)
  for b in range(NB):
    s = slice(b * L, (b + 1) * L)
    k = 0 if b < NB * 0.9 else (1 if b < NB * 0.97 else 2)  # 90 % standing, 7 % lying, 3 % flipping
    kind[b] = k
    if k == 0:
      pos[s, :, 2] = torch.linspace(0.0, 1.3, 14)  # upright: ankles low ... head high
      pos[s, 0, 2] = 0.78
      pos[s, 3, 2] = pos[s, 6, 2] = 0.05
      pos[s, 10, 2] = pos[s, 13, 2] = 0.8
    elif k == 1:
      pos[s, :, 2] = 0.1
      quat[s, 0] = torch.tensor([0.7071, 0.0, 0.7071, 0.0])  # pitched 90 deg
    else:
      pos[s, :, 2] = 1.2
      pos[s, 0, 2] = 1.0
      th = torch.linspace(0, 2 * np.pi, L)
      quat[s, 0] = torch.stack([torch.cos(th / 2), torch.zeros(L), torch.sin(th / 2), torch.zeros(L)], -1)
      ang[s, 0, 1] = 9.0
      lin[s, 0, 2] = torch.linspace(3.0, -3.0, L).half()
    pos[s] += 0.01 * torch.randn(L, 14, 3, generator=g)
  lib = types.SimpleNamespace(
    cache_dir=tmp, num_bins=NB, time_step_total=T, body_names=names,
    bin_start=torch.arange(NB) * L, bin_end=torch.arange(NB) * L + L,
    fields={"body_pos": pos, "body_quat": quat, "body_ang_vel": ang, "body_lin_vel": lin},
  )
  return lib, kind


def test_clusters():
  from supermimic.bin_clusters import bin_cluster_prior, bin_features

  with tempfile.TemporaryDirectory() as tmp:
    lib, kind = fake_lib(tmp)
    f = bin_features(lib, chunk=128, log=lambda *_: None)
    assert f.shape == (lib.num_bins, 12) and np.isfinite(f).all()
    # features separate the three kinds: airborne fraction high only for flips, torso-down only for lying
    assert f[kind.numpy() == 2, 8].mean() > 0.9 and f[kind.numpy() == 0, 8].mean() < 0.1
    assert f[kind.numpy() == 1, 11].mean() > 0.9 and f[kind.numpy() == 0, 11].mean() < 0.1
    assert os.path.exists(os.path.join(tmp, "bin_feat12_v1.npy"))
    w1 = bin_cluster_prior(lib, 3, 0.3, "cpu", log=lambda *_: None)
    w2 = bin_cluster_prior(lib, 3, 0.3, "cpu", log=lambda *_: None)  # cached labels
    assert torch.allclose(w1, w2)
    assert abs(float(w1.sum()) - 1.0) < 1e-5
    share = torch.zeros(3).index_add_(0, kind, w1)
    n = torch.bincount(kind).double()
    expect = n.pow(0.3) / n.pow(0.3).sum()
    assert torch.allclose(share.double(), expect, atol=2e-3), (share, expect)  # k-means recovered the three kinds
    print(f"clusters ok: shares {share.tolist()} (bins {n.tolist()})")


def test_root_z_head():
  from supermimic.heads import AnalyticHeads, HeadsCfg

  cfg = HeadsCfg(root_z_term=0.25, ee_termination=False)
  h = AnalyticHeads.__new__(AnalyticHeads)  # only termination(): set the attributes it reads
  h.cfg, h.anchor, h.root_ori, h.root_ori_is_root, h.ee = cfg, 1, 0, True, torch.zeros(0, dtype=torch.long)
  h.gravity = torch.tensor([0.0, 0.0, -1.0])
  B = 4
  ref_bp = torch.zeros(B, 2, 3)
  ref_bq = torch.zeros(B, 2, 4)
  ref_bq[..., 0] = 1.0
  ref_bp[:, 0, 2] = 0.78
  ref_bp[:, 1, 2] = 1.05
  kin = {"anchor_pos": ref_bp[:, 1].clone(), "anchor_quat": ref_bq[:, 1].clone(), "body_pos": ref_bp.clone(), "body_quat": ref_bq.clone()}
  kin["body_pos"][:, 0, 2] = torch.tensor([0.78, 0.60, 0.50, 0.40])  # pelvis height errors 0, 0.18, 0.28, 0.38
  hard, p = h.termination(kin, ref_bp[:, 1], ref_bq[:, 1], ref_bp)
  assert hard.tolist() == [False, False, True, True], hard
  assert float(p[0]) < 1e-3 and float(p[1]) < 0.05 and float(p[2]) > 0.75 and float(p[3]) > 0.99, p
  print("root_z head ok")


if __name__ == "__main__":
  test_cap_simplex()
  test_clusters()
  test_root_z_head()
  print("ALL OK")
