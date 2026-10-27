"""Human (SMPL) reference heads (supermimic/human), CPU, on mjlab's G1 model.

1. Bone map / calibration on mjlab's G1: at the bone-direction IK reset state of a human clip frame, the robot bone
   directions computed by HumanHeads.kin (TorchFK) match the human ones (cache field h_dir) about as well as the IK
   reports (init_ik_err), i.e. the RGMT calibration transfers to mjlab's link frames.
2. Gradients of reward and soft termination w.r.t. the state are finite at an exact match (zero bone error), at the
   reset state, with the robot on its side (vertical hip line) and upside down.
3. chord_angle == angle_between; reward_max_per_step = positive weights x dt; goal layout 138 + 2.
4. Trainer._pack_args keeps the robot pack order (robot reference unchanged).

Needs mjlab (G1 spec) and the pilot clips (~/datasets/smpl_motion_pilot_50hz, RGMT format with init_g1_* states);
skips otherwise.  Run: uv run python tests/test_human_heads.py
"""

from __future__ import annotations

import glob
import math
import os
import sys
import tempfile
import types

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
PILOT = os.path.expanduser("~/datasets/smpl_motion_pilot_50hz")


def g1_fk():
  import mujoco
  from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec

  from supermimic.fk import TorchFK

  model = get_spec().compile()
  bodies = list(range(1, model.nbody))
  hinge = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
  ent = types.SimpleNamespace(
    indexing=types.SimpleNamespace(body_ids=torch.tensor(bodies), joint_ids=torch.tensor(hinge)),
    body_names=[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in bodies],
  )
  joint_names = tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in hinge)
  return model, ent, joint_names, TorchFK(model, ent, "cpu")


def make_heads(model, ent, fk, joint_names):
  from supermimic.human import bone_map as bone
  from supermimic.human.command import robot_rest_geometry
  from supermimic.human.heads import HumanHeads, HumanHeadsCfg

  bm = bone.SMPL_BONE_MAPS["g1"]
  links = [ent.body_names.index(n) for n in bm.robot_links]
  rest, ankle = robot_rest_geometry(model, bm.pelvis_link, bm.foot_links, r"^(left|right)_foot[1-7]_collision$")
  J = len(joint_names)
  lim = torch.tensor([[-3.0, 3.0]] * J)
  return HumanHeads(fk, links, torch.zeros(J), lim, HumanHeadsCfg(), rest, ankle), rest, ankle


def main():
  try:
    import mjlab  # noqa: F401
  except ImportError:
    print("SKIP: mjlab not installed")
    return
  files = sorted(glob.glob(os.path.join(PILOT, "*.npz")))
  if not files:
    print(f"SKIP: no pilot clips in {PILOT}")
    return
  from supermimic.human import bone_map as bone
  from supermimic.human.heads import STEP_FIELDS, chord2, chord_angle
  from supermimic.human.motions import GOAL_FEAT_DIM, HumanMotionLibrary

  model, ent, joint_names, fk = g1_fk()
  heads, rest, ankle = make_heads(model, ent, fk, joint_names)
  print(f"G1 rest pelvis height {rest:.4f} m, ankle height {ankle:.4f} m")
  assert 0.75 < rest < 0.82 and 0.02 < ankle < 0.05

  with tempfile.TemporaryDirectory() as tmp:
    lib = HumanMotionLibrary(files[:3], "g1", joint_names, bone.SMPL_BONE_MAPS["g1"], "cpu", cache_dir=tmp)
    assert lib.fields["h_feat"].shape[1] == GOAL_FEAT_DIM
    frames = torch.arange(0, lib.time_step_total, 7)
    g = lib.gather(frames)

  # 1. robot bone directions at the IK reset state vs the human's
  B = frames.numel()
  yaw_q = g["init_root_quat"]
  s = torch.cat([torch.zeros(B, 2), g["init_root_height"][:, None], yaw_q, torch.zeros(B, 6), g["init_joint_pos"], torch.zeros(B, len(joint_names))], -1)
  kin = heads.kin(s)
  ang = torch.rad2deg(chord_angle(chord2(g["h_dir"], kin["dirs"])))
  w = heads.bt.weights
  err = (ang * w).sum(-1) / w.sum()
  print(f"IK reset state: bone error {err.mean():.2f} deg (IK reported {g['init_ik_err'].mean():.2f} deg), p95 {err.quantile(0.95):.2f}")
  assert float(err.mean()) < float(g["init_ik_err"].mean()) + 2.0, "calibration does not transfer to mjlab's G1"
  hd = torch.rad2deg(torch.atan2(torch.sin(g["h_yaw"] - kin["yaw"]), torch.cos(g["h_yaw"] - kin["yaw"]))).abs()
  print(f"  heading error {hd.mean():.2f} deg, height error {(s[:, 2] / rest - g['h_height']).abs().mean():.3f} (normalized)")
  assert float(hd.mean()) < 5.0

  # 2. finite gradients
  ref = {k: g[k] for k in STEP_FIELDS}
  cases = {"reset": s.clone()}
  exact = dict(ref)
  exact["h_dir"], exact["h_dir_local"] = kin["dirs"].detach(), kin["dirs_local"].detach()
  side, flip = s.clone(), s.clone()
  side[:, 3:7] = torch.tensor([math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0])  # 90 deg roll: hip line vertical
  flip[:, 3:7] = torch.tensor([0.0, 1.0, 0.0, 0.0])  # upside down
  cases.update(side=side, flip=flip)
  for name, st in list(cases.items()) + [("exact", s.clone())]:
    st = st.clone().requires_grad_(True)
    rf = exact if name == "exact" else ref
    k2 = heads.kin(st)
    a = torch.zeros(B, len(joint_names))
    r = heads.reward(k2, k2, st, a, a, rf)
    _, d = heads.termination(k2, st, rf)
    gr, = torch.autograd.grad((r.sum() + d.sum()), st)
    ok = bool(torch.isfinite(gr).all())
    print(f"  grad {name:6s}: finite={ok} max|g|={gr.abs().max():.3g} reward/max={float(r.mean()) / heads.cfg.reward_max_per_step:.3f} soft-term={float(d.mean()):.3f}")
    assert ok, name

  # 3. small consistency checks
  u = bone.normalize(torch.randn(1000, 3))
  v = bone.normalize(torch.randn(1000, 3))
  assert torch.allclose(chord_angle(chord2(u, v)), bone.angle_between(u, v), atol=2e-3)
  c = heads.cfg
  assert abs(c.reward_max_per_step - (1.0 + 0.5 + 1.5 + 1.0 + 1.0 + 0.5 + 0.5 + 0.5) * c.dt) < 1e-12
  assert heads.goal_frame_dim == GOAL_FEAT_DIM + 2 and sum(n for _, n in heads.goal_layout()) == heads.goal_frame_dim

  # 4. robot pack order unchanged
  from supermimic.trainer import Trainer

  t = object.__new__(Trainer)
  pack = {k: torch.full((1,), float(i)) for i, k in enumerate(Trainer.ROBOT_PACK)}
  assert [float(x) for x in t._pack_args(pack)] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0] and t._pack_fields() is None
  print("test_human_heads: OK")


if __name__ == "__main__":
  main()
