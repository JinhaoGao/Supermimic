"""End-effector world tracking of the human reference (HumanHeadsCfg.ee_mode, docs/ee_world_tracking_plan.md), CPU.

1. alpha = 0: reward and termination of the ee_mode heads equal the H4 heads' (same H4 recipe) bitwise in value.
2. Exact task match (targets = the robot's own task-link poses, heading = the robot's): every task kernel scores its full
   weight and the torso entry of the EE goal block is the identity pose.
3. Drift: targets of cond_at(cond, n) == the n-th sample of the goal-window extrapolation; drift stays in its box.
4. Goal layout 138 + 2 + 45 + 2; the EE block is zero when ee_valid = 0; reward_max_per_step is the same for alpha 0 / 1.
5. Finite gradients of reward + soft termination (alpha = 1) at the reset state, on the side, upside down.
6. HumanStepFns.goal_from_pack with the cond pack entry == heads.goal on the gathered windows.

Needs mjlab and the pilot clips (like test_human_heads.py); skips otherwise.  Run: uv run python tests/test_human_ee.py
"""

from __future__ import annotations

import glob
import math
import os
import sys
import tempfile

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tests"))
PILOT = os.path.expanduser("~/datasets/smpl_motion_pilot_50hz")

H4 = dict(w_bone_global_arm=1.0, w_bone_global_leg=1.5, std_bone_leg=0.6, w_foot_lift=1.0, term_foot_lift=0.25)


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
  from test_human_heads import g1_fk

  from supermimic.human import bone_map as bone
  from supermimic.human.command import robot_rest_geometry
  from supermimic.human.fns import HumanStepFns
  from supermimic.human.heads import COND_DIM, EE_STEP_FIELDS, STEP_FIELDS, HumanHeads, HumanHeadsCfg
  from supermimic.human.motions import EE_FIELDS, GOAL_FEAT_DIM, HumanMotionLibrary

  model, ent, joint_names, fk = g1_fk()
  bm = bone.SMPL_BONE_MAPS["g1"]
  links = [ent.body_names.index(n) for n in bm.robot_links]
  rest, ankle = robot_rest_geometry(model, bm.pelvis_link, bm.foot_links, r"^(left|right)_foot[1-7]_collision$")
  J = len(joint_names)
  lim = torch.tensor([[-3.0, 3.0]] * J)
  h4 = HumanHeads(fk, links, torch.zeros(J), lim, HumanHeadsCfg(**H4), rest, ankle)
  ee = HumanHeads(fk, links, torch.zeros(J), lim, HumanHeadsCfg(**H4, ee_mode=True), rest, ankle)

  with tempfile.TemporaryDirectory() as tmp:
    lib = HumanMotionLibrary(files[:3], "g1", joint_names, bm, "cpu", cache_dir=tmp)
    lib.ensure_ee(bm)
    assert all(k in lib.fields for k in EE_FIELDS) and os.path.exists(os.path.join(lib.cache_dir, "meta_ee.json"))
    frames = torch.arange(0, lib.time_step_total - 20, 7)
    g = lib.gather(frames)
  B = frames.numel()
  s = torch.cat([torch.zeros(B, 2), g["init_root_height"][:, None], g["init_root_quat"], torch.zeros(B, 6), g["init_joint_pos"], torch.zeros(B, J)], -1)
  ref = {k: g[k] for k in STEP_FIELDS + EE_STEP_FIELDS}
  cond = torch.zeros(B, COND_DIM)
  cond[:, 1] = 1.0
  cond[:, 2:4] = g["h_root_xy"]  # each frame its own episode start: robot pelvis at the env origin
  a0 = torch.zeros(B, J)

  # 1. alpha = 0 == H4
  ref0 = dict(ref, cond=cond.clone())
  k_h4, k_ee = h4.kin(s), ee.kin(s)
  r_h4, r_ee = h4.reward(k_h4, k_h4, s, a0, a0, ref), ee.reward(k_ee, k_ee, s, a0, a0, ref0)
  (hh, dh), (he, de) = h4.termination(k_h4, s, ref), ee.termination(k_ee, s, ref0)
  assert torch.allclose(r_h4, r_ee, atol=1e-7) and torch.equal(hh, he) and torch.allclose(dh, de, atol=1e-7), "alpha = 0 is not H4"
  print(f"1. alpha = 0: reward {float(r_ee.mean()) / ee.cfg.reward_max_per_step:.3f} of max, identical to H4")

  # 2. exact task match at alpha = 1
  c1 = cond.clone()
  c1[:, 0] = 1.0
  c1[:, 2:4] = 0.0
  tr = k_ee["task_rot"]
  exact = dict(ref, cond=c1, h_task_pos=k_ee["task_pos"].clone(), h_task_rot6d=tr[..., :, :2].flatten(-2).clone(), h_yaw=k_ee["yaw"].clone())
  _, terms = ee.reward(k_ee, k_ee, s, a0, a0, exact, return_terms=True)
  c = ee.cfg
  for name, w in (("ee_torso_wide", c.w_ee_torso_wide), ("ee_torso", c.w_ee_torso), ("ee_wrist_wide", c.w_ee_wrist_wide),
                  ("ee_wrist", c.w_ee_wrist), ("ee_wrist_rot", c.w_ee_wrist_rot), ("ee_foot", c.w_ee_foot)):
    assert torch.allclose(terms[name] / c.dt, torch.full((B,), w), atol=1e-4), (name, terms[name][:4] / c.dt)
  e = ee.ee_errors(k_ee, exact)
  assert all(float(v.max()) < 1e-8 for v in e.values()), {k: float(v.max()) for k, v in e.items()}
  win = {k: v[:, None].expand(B, c.K, *v.shape[1:]) for k, v in exact.items() if k in ("h_task_pos", "h_task_rot6d")}
  goal = ee.goal(k_ee, g["h_feat"][:, None].expand(B, c.K, -1), g["h_yaw"][:, None].expand(B, c.K), win["h_task_pos"], win["h_task_rot6d"], c1)
  torso = goal[:, 0, GOAL_FEAT_DIM + 2 : GOAL_FEAT_DIM + 2 + 9]
  assert torch.allclose(torso, torch.tensor([0.0, 0, 0, 1, 0, 0, 1, 0, 0]).expand(B, 9), atol=1e-5), torso[0]
  hard, _ = ee.termination(k_ee, s, exact)
  print(f"2. exact task match: task kernels at full weight, torso goal entry = identity, terminated {int(hard.sum())}/{B}")

  # 3. drift extrapolation
  cd = c1.clone()
  cd[:, 4:7] = torch.tensor([0.03, -0.02, 0.05])
  cd[:, 7:10] = torch.tensor([0.2, 0.1, -0.3])  # per second
  steps = torch.arange(c.K)
  pw, rw, _ = ee.ee_targets(win["h_task_pos"], win["h_task_rot6d"], cd, steps)
  for n in (0, 3, c.K - 1):
    pn, rn, _ = ee.ee_targets(exact["h_task_pos"], exact["h_task_rot6d"], ee.cond_at(cd, torch.full((B,), n)))
    assert torch.allclose(pn, pw[:, n], atol=1e-5) and torch.allclose(rn, rw[:, n], atol=1e-5), n
  far = ee.cond_at(cd, torch.full((B,), 10_000))
  assert bool((far[:, 4:7].abs() <= ee.drift_box + 1e-6).all())
  print("3. drift: per-sample extrapolation == cond_at, clamped to the box")

  # 4. layout, ee_valid, reward scale
  assert ee.goal_frame_dim == GOAL_FEAT_DIM + 2 + 45 + 2 == sum(n for _, n in ee.goal_layout())
  c0 = c1.clone()
  c0[:, 1] = 0.0
  g0 = ee.goal(k_ee, g["h_feat"][:, None].expand(B, c.K, -1), g["h_yaw"][:, None].expand(B, c.K), win["h_task_pos"], win["h_task_rot6d"], c0)
  assert float(g0[..., GOAL_FEAT_DIM + 2 : GOAL_FEAT_DIM + 47].abs().max()) == 0.0
  assert abs(c.h4_weight_sum - 10.0) < 1e-9 and abs(c.ee_task_scale - 1.0) < 1e-9
  c5 = HumanHeadsCfg(**H4, ee_mode=True, ee_style=0.5, w_ee_wrist_wide=0.0)
  assert abs(c5.ee_style * c5.h4_weight_sum + c5.ee_task_scale * c5.ee_weight_sum - c5.h4_weight_sum) < 1e-9
  assert abs(c.reward_max_per_step - h4.cfg.reward_max_per_step) < 1e-12
  print(f"4. goal frame {ee.goal_frame_dim}, ee_valid = 0 zeroes the EE block, reward max {c.reward_max_per_step:.3f} (= H4)")

  # 5. finite gradients at alpha = 1
  side, flip = s.clone(), s.clone()
  side[:, 3:7] = torch.tensor([math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0])
  flip[:, 3:7] = torch.tensor([0.0, 1.0, 0.0, 0.0])
  ref1 = dict(ref, cond=torch.cat([torch.ones(B, 1), cond[:, 1:]], -1))
  for name, st in (("reset", s), ("side", side), ("flip", flip)):
    st = st.clone().requires_grad_(True)
    k2 = ee.kin(st)
    r = ee.reward(k2, k2, st, a0, a0, ref1)
    _, d = ee.termination(k2, st, ref1)
    gr, = torch.autograd.grad(r.sum() + d.sum(), st)
    assert bool(torch.isfinite(gr).all()), name
    m = ee.metrics(k2, st.detach(), ref1)
    print(f"5. grad {name:5s}: finite, reward/max {float(r.mean()) / c.reward_max_per_step:.3f}, soft-term {float(d.mean()):.3f}, "
          f"wrist err {float(m['ee_wrist_err'].mean()):.3f} m, torso xy {float(m['ee_torso_xy_err'].mean()):.3f} m")

  # 6. imagination pack path
  fns = HumanStepFns(ee, policy=None, compile_=False)
  L = c.K + 4
  with tempfile.TemporaryDirectory():
    pk = lib.gather_windows(frames, L, fields=[k for k in ee.pack_fields if k != "cond"])
  pk["cond"] = cd
  p = tuple(pk[k] for k in ee.pack_fields)
  f_loc = torch.full((B,), 2, dtype=torch.long)
  clip_end_loc = torch.full((B,), L, dtype=torch.long)
  gp = fns.goal_from_pack(k_ee, f_loc, clip_end_loc, *p)
  sl = slice(2, 2 + c.K)
  gd = ee.goal(k_ee, pk["h_feat"][:, sl], pk["h_yaw"][:, sl], pk["h_task_pos"][:, sl], pk["h_task_rot6d"][:, sl], ee.cond_at(cd, f_loc))
  assert torch.allclose(gp, gd)
  print("6. goal_from_pack (cond advanced by the local frame) == heads.goal")
  print("test_human_ee: OK")


if __name__ == "__main__":
  main()
