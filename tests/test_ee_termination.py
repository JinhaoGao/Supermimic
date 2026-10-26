"""End-effector termination of the analytic heads (torch only, no mjlab / GPU).

  python tests/test_ee_termination.py      # or: pytest tests/test_ee_termination.py

Checks:
  * switch off -> hard / soft termination and their gradients bitwise equal to the implementation before the
    switch existed (copied verbatim below), eager and torch.compile'd;
  * switch on -> hard test flips exactly at the threshold (both directions: foot / hand above or below the
    reference), soft term = sigmoid((dz - thr) / temp) noisy-OR'ed with the anchor terms, 0.5 at the threshold;
  * gradients: autograd == closed form, only the z of end-effector bodies gets a gradient, sign pushes the
    body back toward the reference height, peak at the threshold;
  * the end-effector z error equals mjlab's bad_motion_body_pos_z_only on the re-anchored reference;
  * misuse is caught (switch on without end-effector bodies / reference body positions).
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from supermimic.heads import AnalyticHeads, HeadsCfg  # noqa: E402
from supermimic.math_utils import quat_apply_inverse, quat_from_angle_axis, quat_mul  # noqa: E402

# G1 tracked bodies (mjlab tracking cfg) and its ee_body_pos bodies (ankle_roll / wrist_yaw links).
G1_TRACKED = (
  "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link", "right_hip_roll_link", "right_knee_link",
  "right_ankle_roll_link", "torso_link", "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
  "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
)
G1_EE = ("left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link", "right_wrist_yaw_link")
EE_IDX = [i for i, n in enumerate(G1_TRACKED) if n in G1_EE]  # [3, 6, 10, 13]
ANCHOR = G1_TRACKED.index("torso_link")
BT = len(G1_TRACKED)
NJ = 29


class _FK:  # termination() only needs the device
  device = "cpu"


def make_heads(ee: bool, **kw) -> AnalyticHeads:
  cfg = HeadsCfg(ee_termination=ee, **kw)
  return AnalyticHeads(_FK(), list(range(BT)), ANCHOR, torch.zeros(NJ), torch.zeros(NJ, 2), cfg, ee_indices=EE_IDX if ee else None)


def termination_before_switch(self: AnalyticHeads, kin_next: dict, ref_ap: torch.Tensor, ref_aq: torch.Tensor):
  """heads.AnalyticHeads.termination at b9415fc (before the end-effector switch), verbatim."""
  c = self.cfg
  B = ref_ap.shape[0]
  g = self.gravity.expand(B, 3)
  dz = (ref_ap[:, 2] - kin_next["anchor_pos"][:, 2]).abs()
  tilt = (quat_apply_inverse(ref_aq, g)[:, 2] - quat_apply_inverse(kin_next["anchor_quat"], g)[:, 2]).abs()
  hard = (dz > c.term_z) | (tilt > c.term_tilt)
  pz = torch.sigmoid((dz - c.term_z) / c.term_z_temp)
  pt = torch.sigmoid((tilt - c.term_tilt) / c.term_tilt_temp)
  soft = 1.0 - (1.0 - pz) * (1.0 - pt)
  return hard, soft


def _unit(q: torch.Tensor) -> torch.Tensor:
  return q / q.norm(dim=-1, keepdim=True)


def random_batch(B: int, seed: int, spread: float = 0.3):
  """Reference + robot tracked-body states; errors spread across the thresholds (some envs terminate)."""
  g = torch.Generator().manual_seed(seed)
  ref_bp = torch.randn(B, BT, 3, generator=g) * 0.3 + torch.tensor([0.0, 0.0, 0.8])
  ref_bq = _unit(torch.randn(B, BT, 4, generator=g))
  bp = ref_bp + torch.randn(B, BT, 3, generator=g) * spread
  axis = _unit(torch.randn(B, BT, 3, generator=g))
  bq = quat_mul(quat_from_angle_axis(torch.rand(B, BT, generator=g) * 1.5, axis), ref_bq)
  return ref_bp, ref_bq, bp, bq


def kin_of(bp: torch.Tensor, bq: torch.Tensor) -> dict:
  return dict(body_pos=bp, body_quat=bq, anchor_pos=bp[:, ANCHOR], anchor_quat=bq[:, ANCHOR])


# ---------------------------------------------------------------------------


def test_off_is_bitwise_identical_to_before():
  heads = make_heads(False)
  for seed in range(3):
    ref_bp, ref_bq, bp0, bq0 = random_batch(4096, seed)
    outs = []
    for fn in ("new", "new_with_ref_bp", "old"):
      bp = bp0.clone().requires_grad_(True)
      bq = bq0.clone().requires_grad_(True)
      kin = kin_of(bp, bq)
      if fn == "old":
        hard, soft = termination_before_switch(heads, kin, ref_bp[:, ANCHOR], ref_bq[:, ANCHOR])
      else:
        extra = (ref_bp,) if fn == "new_with_ref_bp" else ()
        hard, soft = heads.termination(kin, ref_bp[:, ANCHOR], ref_bq[:, ANCHOR], *extra)
      gp, gq = torch.autograd.grad((soft * torch.linspace(0.5, 1.5, soft.shape[0])).sum(), (bp, bq))
      outs.append((hard, soft.detach(), gp, gq))
    assert 0 < int(outs[2][0].sum()) < 4096, "batch should contain both terminating and surviving envs"
    for o in outs[:2]:
      for a, b in zip(o, outs[2]):
        assert torch.equal(a, b), "switch off must reproduce the old termination bitwise"


def test_off_compiled_is_bitwise_identical_to_before():
  try:
    heads = make_heads(False)
    new = torch.compile(lambda k, ap, aq, bp: heads.termination(k, ap, aq, bp), dynamic=False)
    old = torch.compile(lambda k, ap, aq: termination_before_switch(heads, k, ap, aq), dynamic=False)
    ref_bp, ref_bq, bp, bq = random_batch(1024, 7)
    a = new(kin_of(bp, bq), ref_bp[:, ANCHOR], ref_bq[:, ANCHOR], ref_bp)
    b = old(kin_of(bp, bq), ref_bp[:, ANCHOR], ref_bq[:, ANCHOR])
  except Exception as e:  # noqa: BLE001  (no C++ toolchain for inductor on this machine)
    print(f"  (compiled check skipped: {type(e).__name__}: {str(e)[:120]})")
    return
  assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def _aligned_batch(dz_ee: torch.Tensor, body: int = EE_IDX[0]):
  """Robot exactly on the reference except body ``body`` offset in z by dz_ee[b] (anchor terms ~ 0)."""
  B = dz_ee.shape[0]
  ref_bp, ref_bq, _, _ = random_batch(B, 11)
  bp = ref_bp.clone()
  bp[:, body, 2] = bp[:, body, 2] + dz_ee
  return ref_bp, ref_bq, bp, ref_bq.clone()


def test_threshold_both_sides():
  thr, temp = 0.25, 0.02
  heads = make_heads(True, ee_term_z=thr, ee_term_z_temp=temp)
  eps = 1e-4
  for body in EE_IDX:
    for sign in (+1.0, -1.0):  # end-effector above (kick / swing) or below the reference
      d = sign * torch.tensor([0.0, 0.1, thr - eps, thr + eps, 0.4])
      ref_bp, ref_bq, bp, bq = _aligned_batch(d, body)
      hard, soft = heads.termination(kin_of(bp, bq), ref_bp[:, ANCHOR], ref_bq[:, ANCHOR], ref_bp)
      assert hard.tolist() == [False, False, False, True, True], (body, sign, hard.tolist())
      # anchor terms are ~sigmoid(-12.5) / sigmoid(-16); the end-effector term dominates
      p = torch.sigmoid((d.abs() - thr) / temp)
      base = 1.0 - (1.0 - torch.sigmoid(torch.tensor(-thr / 0.02))) * (1.0 - torch.sigmoid(torch.tensor(-0.8 / 0.05)))
      expect = 1.0 - (1.0 - base) * (1.0 - p) * (1.0 - torch.sigmoid(torch.tensor(-thr / temp))) ** 3
      assert torch.allclose(soft, expect, rtol=1e-5, atol=1e-6), (soft.tolist(), expect.tolist())
      assert abs(float(soft[2]) - 0.5) < 0.01 and abs(float(soft[3]) - 0.5) < 0.01  # 0.5 at the threshold
      assert float(soft[0]) < 1e-4 and float(soft[4]) > 0.999
  # A non-end-effector body (knee) far off in z: no end-effector termination.
  ref_bp, ref_bq, bp, bq = _aligned_batch(torch.tensor([0.5, -0.5]), body=G1_TRACKED.index("left_knee_link"))
  hard, soft = heads.termination(kin_of(bp, bq), ref_bp[:, ANCHOR], ref_bq[:, ANCHOR], ref_bp)
  assert not hard.any() and float(soft.max()) < 1e-4
  # Horizontal end-effector error does not count (z only, like mjlab / the eval site).
  ref_bp, ref_bq, bp, bq = _aligned_batch(torch.zeros(2))
  bp[:, EE_IDX[2], 0] += 0.6
  hard, soft = heads.termination(kin_of(bp, bq), ref_bp[:, ANCHOR], ref_bq[:, ANCHOR], ref_bp)
  assert not hard.any() and float(soft.max()) < 1e-4


def test_noisy_or_combination():
  """Several criteria near threshold at once: soft = 1 - prod(1 - p_i), hard = OR."""
  heads = make_heads(True)
  ref_bp, ref_bq, bp, bq = _aligned_batch(torch.zeros(3))
  dz = torch.tensor([0.24, 0.25, 0.27])
  for k, body in enumerate(EE_IDX[:3]):
    bp[:, body, 2] += dz[k]
  hard, soft = heads.termination(kin_of(bp, bq), ref_bp[:, ANCHOR], ref_bq[:, ANCHOR], ref_bp)
  assert hard.all()
  p = torch.sigmoid((dz - 0.25) / 0.02)
  expect = 1.0 - torch.prod(1.0 - p)
  assert torch.allclose(soft, expect.expand(3), atol=1e-5), (soft, expect)


def test_gradients():
  thr, temp = 0.25, 0.02
  heads = make_heads(True, ee_term_z=thr, ee_term_z_temp=temp)
  d = torch.tensor([0.0, 0.15, 0.24, 0.25, 0.26, 0.35, -0.24, -0.25, -0.30])
  ref_bp, ref_bq, bp0, bq = _aligned_batch(d)
  bp = bp0.clone().requires_grad_(True)
  _, soft = heads.termination(kin_of(bp, bq), ref_bp[:, ANCHOR], ref_bq[:, ANCHOR], ref_bp)
  (gp,) = torch.autograd.grad(soft.sum(), bp)
  e = EE_IDX[0]
  # closed form: soft = 1 - A * prod_j (1 - p_j)  ->  d soft / d z_e = A * prod_{j != e} (1 - p_j) * p_e (1 - p_e) / temp * sign(z_e - ref_z_e)
  with torch.no_grad():
    dz_all = (ref_bp[:, EE_IDX, 2] - bp0[:, EE_IDX, 2]).abs()
    p_all = torch.sigmoid((dz_all - thr) / temp)
    a_z = (ref_bp[:, ANCHOR, 2] - bp0[:, ANCHOR, 2]).abs()
    A = (1.0 - torch.sigmoid((a_z - 0.25) / 0.02)) * (1.0 - torch.sigmoid(torch.tensor(-0.8 / 0.05)))
    others = torch.prod(1.0 - p_all[:, 1:], dim=-1)
    pe = p_all[:, 0]
    expect = A * others * pe * (1.0 - pe) / temp * torch.sign(d)
  got = gp[:, e, 2]
  assert torch.allclose(got, expect, rtol=1e-4, atol=1e-7), (got.tolist(), expect.tolist())
  # gradient pushes the body back toward the reference height (descent on soft reduces |dz|)
  nz = d != 0
  assert torch.all(torch.sign(got[nz]) == torch.sign(d[nz]))
  # peak at the threshold: 1/(4 temp) = 12.5 per unit survival
  assert abs(float(got[3]) - A[3] * others[3] * 0.25 / temp) < 1e-3
  assert float(got[3].abs()) > float(got[2].abs()) > float(got[1].abs()) > float(got[0].abs())
  # only z of end-effector bodies (the anchor's z also matters through its own term, ~0 here)
  mask = torch.zeros(BT, 3, dtype=torch.bool)
  mask[EE_IDX, 2] = True
  mask[ANCHOR, 2] = True
  assert float(gp[:, ~mask].abs().max()) == 0.0


def test_matches_mjlab_relative_reference():
  """mjlab's bad_motion_body_pos_z_only compares the robot with body_pos_relative_w (reference re-anchored to the
  robot anchor: xy shift + yaw rotation); its z equals the raw reference z, which the head uses."""
  heads = make_heads(True)
  ref_bp, ref_bq, bp, bq = random_batch(2048, 3, spread=0.2)
  kin_prev = kin_of(bp + 0.05, bq)  # any pre-step robot anchor
  rel_p, _, _, _ = heads.relative_reference(kin_prev, ref_bp, ref_bq)
  assert torch.allclose(rel_p[..., 2], ref_bp[..., 2], atol=2e-6)
  dz_mjlab = (rel_p[:, EE_IDX, 2] - bp[:, EE_IDX, 2]).abs()
  hard_mjlab = (dz_mjlab > 0.25).any(-1)
  hard_anchor, _ = make_heads(False).termination(kin_of(bp, bq), ref_bp[:, ANCHOR], ref_bq[:, ANCHOR])
  hard, _ = heads.termination(kin_of(bp, bq), ref_bp[:, ANCHOR], ref_bq[:, ANCHOR], ref_bp)
  near = ((dz_mjlab - 0.25).abs() < 1e-5).any(-1)  # float ties at the threshold may round either way
  assert torch.equal((hard_anchor | hard_mjlab)[~near], hard[~near])
  assert int(hard_mjlab.sum()) > 100 and int((hard & ~hard_anchor).sum()) > 0


def test_misuse():
  try:
    AnalyticHeads(_FK(), list(range(BT)), ANCHOR, torch.zeros(NJ), torch.zeros(NJ, 2), HeadsCfg(ee_termination=True))
  except ValueError:
    pass
  else:
    raise AssertionError("ee_termination without end-effector bodies must fail")
  heads = make_heads(True)
  ref_bp, ref_bq, bp, bq = random_batch(4, 0)
  try:
    heads.termination(kin_of(bp, bq), ref_bp[:, ANCHOR], ref_bq[:, ANCHOR])
  except AssertionError:
    pass
  else:
    raise AssertionError("ee_termination without reference body positions must fail")
  assert HeadsCfg().ee_termination and HeadsCfg().ee_term_z == 0.25  # default on, eval-site threshold


if __name__ == "__main__":
  for name, fn in list(globals().items()):
    if name.startswith("test_") and callable(fn):
      fn()
      print(f"ok  {name}")
