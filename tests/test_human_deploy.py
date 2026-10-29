"""Deploy path of human-reference policies (supermimic/human/deploy_goal.py, deploy_export.py), CPU.

1. Per-clip reference npz (clip_reference) == the training cache (HumanMotionLibrary + ensure_ee) on those frames.
2. numpy FK (RobotKinematics on the exported fk table) == TorchFK on the bone-map links.
3. HumanGoalBuilder.goal_world == HumanHeads.goal (drift 0) for H4 and ee_mode, random root poses / joints / alpha /
   ee_valid / anchors, frames up to the clip end (window clamp).
4. Stateful builder: robot in an arbitrary odometry frame (rotated + shifted, IMU yaw off by another angle), start() at
   f0 from the IK reset pose -> each step == HumanHeads.goal with the episode's anchor h_root_xy[f0]; alpha rate limit;
   odom None -> alpha 0, ee_valid 0, EE block 0 and the H4 part continuous; activation realign puts the robot pelvis on
   the reference pelvis; pelvis height from the ankles == the true height on the IK pose (within 2 cm).
5. ONNX (export_policy) == torch policy, on random inputs and on goals from the builder. SUPERMIMIC_CKPT=<model.pt>
   uses that checkpoint's policy (its goal_frame must match), else a random policy with non-trivial normalizers.

Needs mjlab and the pilot clips (like test_human_ee.py); skips otherwise.  Run: uv run python tests/test_human_deploy.py
"""

from __future__ import annotations

import glob
import math
import os
import sys
import tempfile

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tests"))
PILOT = os.path.expanduser("~/datasets/smpl_motion_pilot_50hz")

H4 = dict(w_bone_global_arm=1.0, w_bone_global_leg=1.5, std_bone_leg=0.6, w_foot_lift=1.0, term_foot_lift=0.25)


def quat_yaw(yaw: float) -> np.ndarray:
  return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def qmul(a, b):
  w1, x1, y1, z1 = a
  w2, x2, y2, z2 = b
  return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                   w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])  # fmt: skip


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
  from supermimic.human.deploy_export import clip_reference, export_policy, robot_goal_meta
  from supermimic.human.deploy_goal import HumanGoalBuilder, RobotKinematics, quat_to_matrix
  from supermimic.human.heads import COND_DIM, HumanHeads, HumanHeadsCfg
  from supermimic.human.motions import HumanMotionLibrary
  from supermimic.policy import Policy

  model, ent, joint_names, fk = g1_fk()
  bm = bone.SMPL_BONE_MAPS["g1"]
  links = [ent.body_names.index(n) for n in bm.robot_links]
  rest, ankle = robot_rest_geometry(model, bm.pelvis_link, bm.foot_links, r"^(left|right)_foot[1-7]_collision$")
  J = len(joint_names)
  lim = torch.tensor([[-3.0, 3.0]] * J)
  heads = {"h4": HumanHeads(fk, links, torch.zeros(J), lim, HumanHeadsCfg(**H4), rest, ankle),
           "ee": HumanHeads(fk, links, torch.zeros(J), lim, HumanHeadsCfg(**H4, ee_mode=True), rest, ankle)}
  rmeta = robot_goal_meta(model, ent.indexing.body_ids.tolist(), ent.indexing.joint_ids.tolist(), ent.body_names, joint_names, bm, rest, ankle)

  def meta_of(h: HumanHeads) -> dict:
    return {**rmeta, "K": h.cfg.K, "control_dt": h.cfg.dt, "goal_layout": h.goal_layout(), "joint_names": list(joint_names)}

  use = files[:3]
  clips = [clip_reference(f, bm, "g1", joint_names) for f in use]

  # 1. clip npz == training cache
  with tempfile.TemporaryDirectory() as tmp:
    lib = HumanMotionLibrary(use, "g1", joint_names, bm, "cpu", cache_dir=tmp)
    lib.ensure_ee(bm)
    for c, clip in zip(lib.clips, clips):
      g = lib.gather(torch.arange(c.start, c.end))
      for k in ("h_feat", "h_yaw", "h_task_pos", "h_task_rot6d", "h_root_xy", "init_root_quat", "init_joint_pos"):
        assert np.array_equal(g[k].numpy(), clip[k]), (c.name, k)
  print(f"1. clip reference npz == training cache ({len(clips)} clips, {sum(len(c['h_feat']) for c in clips)} frames)")

  # 2. numpy FK == TorchFK
  rk = RobotKinematics(rmeta["fk"], list(joint_names))
  rng = np.random.default_rng(0)
  err = 0.0
  for _ in range(20):
    q = rng.uniform(-1.0, 1.0, J)
    rq = rng.normal(size=4)
    rq /= np.linalg.norm(rq)
    p = rng.normal(size=3)
    P, R = rk.forward(p, quat_to_matrix(rq), q)
    pt, qt, _, _ = fk.forward(torch.tensor(p[None], dtype=torch.float32), torch.tensor(rq[None], dtype=torch.float32),
                              torch.zeros(1, 3), torch.zeros(1, 3), torch.tensor(q[None], dtype=torch.float32), torch.zeros(1, J))
    for i, n in enumerate(rk.names):
      li = ent.body_names.index(n)
      err = max(err, float(np.abs(P[i] - pt[0, li].numpy()).max()), float(np.abs(R[i] - bone.quat_wxyz_to_matrix(qt[0, li]).numpy()).max()))
  assert err < 1e-5, err
  print(f"2. numpy FK == TorchFK on {len(rk.names)} bodies (max err {err:.1e})")

  def torch_goal(h: HumanHeads, clip: dict, frame: int, s: np.ndarray, cond: np.ndarray) -> np.ndarray:
    T = len(clip["h_feat"])
    idx = np.minimum(frame + np.arange(h.cfg.K), T - 1)
    t = lambda k: torch.tensor(clip[k][idx][None])  # noqa: E731
    st = torch.tensor(s[None], dtype=torch.float32)
    kin = h.kin(st)
    args = [t("h_feat"), t("h_yaw")]
    if h.ee_on:
      args += [t("h_task_pos"), t("h_task_rot6d"), torch.tensor(cond[None], dtype=torch.float32)]
    return h.goal(kin, *args)[0].numpy()

  def state(pos, quat, q) -> np.ndarray:
    return np.concatenate([pos, quat, np.zeros(6), q, np.zeros(J)])

  # 3. pure goal == HumanHeads.goal
  for name, h in heads.items():
    worst = 0.0
    for clip in clips:
      b = HumanGoalBuilder(meta_of(h), clip)
      T = len(clip["h_feat"])
      for f in list(rng.integers(0, T, 12)) + [T - 3, T - 1]:
        f = int(f)
        tilt = rng.normal(size=3) * 0.2
        rq = qmul(quat_yaw(rng.uniform(-math.pi, math.pi)), np.concatenate([[1.0], tilt]) / np.linalg.norm(np.concatenate([[1.0], tilt])))
        pos = np.array([*rng.normal(size=2), 0.75 + 0.05 * rng.normal()])
        q = clip["init_joint_pos"][f] + rng.normal(size=J) * 0.2
        alpha, valid = float(rng.uniform()), float(rng.integers(0, 2))
        anchor = clip["h_root_xy"][int(rng.integers(0, T))] + rng.normal(size=2) * 0.3
        cond = np.zeros(COND_DIM)
        cond[:4] = [alpha, valid, *anchor]
        gt = torch_goal(h, clip, f, state(pos, rq, q), cond)
        gn = b.goal_world(q, pos, quat_to_matrix(rq), f, alpha, valid, anchor)
        worst = max(worst, float(np.abs(gt - gn).max()))
    assert worst < 2e-4, (name, worst)
    print(f"3. {name}: goal_world == HumanHeads.goal (goal frame {h.goal_frame_dim}, max err {worst:.1e})")

  # 4. stateful builder in an odometry frame
  h = heads["ee"]
  clip = clips[0]
  T = len(clip["h_feat"])
  f0 = 5
  th, t_off, imu_off = 1.1, np.array([3.0, -2.0]), -0.7  # odometry frame = training world rotated by th, shifted by t_off
  Rth = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])

  def robot_at(f: int, xy_train: np.ndarray):
    """IK reset pose of frame f at xy_train (training world) -> (state, q, imu quat, odom)."""
    q = clip["init_joint_pos"][f].astype(np.float64)
    rq = clip["init_root_quat"][f].astype(np.float64)
    z = float(clip["init_root_height"][f]) + 0.01
    s = state(np.array([*xy_train, z]), rq, q)
    rq_odom = qmul(quat_yaw(th), rq)
    yaw_odom = math.atan2(*(lambda R: (R[1, 0], R[0, 0]))(quat_to_matrix(rq_odom)))
    xy_odom = Rth @ xy_train + t_off
    return s, q, qmul(quat_yaw(imu_off), rq_odom), (xy_odom[0], xy_odom[1], yaw_odom), z

  b = HumanGoalBuilder(meta_of(h), clip, alpha_rate=1.0, realign_on_activate=False)
  s, q, imu, odom, z = robot_at(f0, np.zeros(2))
  b.start(f0, q, imu, odom, base_z=z)
  est_z = b._source_pose(q, imu, odom, None)[1]
  worst, a_prev, alphas = 0.0, 0.0, []
  for n in range(40):
    f = f0 + n
    xy = np.array([0.01 * n, -0.005 * n])
    s, q, imu, odom, z = robot_at(f, xy)
    g = b.step(q, imu, f, alpha_cmd=1.0, odom=odom, base_z=z)[0]
    alphas.append(b.alpha)
    assert b.alpha - a_prev <= 1.0 * h.cfg.dt + 1e-9 and b.ee_valid == (1.0 if b.alpha > 0 else 0.0)
    a_prev = b.alpha
    cond = np.zeros(COND_DIM)
    cond[:4] = [b.alpha, b.ee_valid, *clip["h_root_xy"][f0]]
    worst = max(worst, float(np.abs(torch_goal(h, clip, f, s, cond) - g).max()))
  assert worst < 2e-4, worst
  assert abs(alphas[-1] - 40 * h.cfg.dt) < 1e-9, alphas[-1]
  # odometry lost: alpha 0, ee_valid 0, EE block zero, the H4 part (rel_yaw from the IMU) continuous
  f = f0 + 40
  s, q, imu, odom, z = robot_at(f, np.array([0.4, -0.2]))
  g_odom = b.step(q, imu, f, alpha_cmd=1.0, odom=odom, base_z=z)[0]
  g_none = b.step(q, imu, f, alpha_cmd=1.0, odom=None, base_z=z)[0]
  G = 138 + 2
  assert b.alpha == 0.0 and b.ee_valid == 0.0 and float(np.abs(g_none[:, G:]).max()) == 0.0
  assert float(np.abs(g_none[:, :G] - g_odom[:, :G]).max()) < 1e-5
  cond = np.zeros(COND_DIM)
  assert float(np.abs(torch_goal(h, clip, f, s, cond) - g_none).max()) < 2e-4
  # odometry back with realign_on_activate: the EE segment starts with the robot pelvis on the reference pelvis
  b.realign_on_activate = True
  f += 1
  s, q, imu, odom, z = robot_at(f, np.array([0.9, 0.3]))  # robot far from where it was
  g = b.step(q, imu, f, alpha_cmd=1.0, odom=odom, base_z=z)[0]
  assert b.ee_valid == 1.0 and abs(b.alpha - h.cfg.dt) < 1e-9
  torso_xy = g[0, G : G + 2]  # reference torso point in the robot torso frame: near the robot (same frame's IK pose)
  assert float(np.linalg.norm(torso_xy)) < 0.15, torso_xy
  assert abs(est_z - (float(clip["init_root_height"][f0]) + 0.01)) < 0.02, est_z
  print(f"4. builder in an odometry frame == HumanHeads.goal over 40 steps (max err {worst:.1e}); alpha ramp {alphas[-1]:.2f} "
        f"after 0.8 s; odom None -> H4 part continuous, EE block 0; activation realign torso offset {np.linalg.norm(torso_xy):.3f} m; "
        f"ankle height estimate {est_z:.3f} vs {float(clip['init_root_height'][f0]) + 0.01:.3f} m")

  # 5. ONNX vs torch
  ck_path = os.environ.get("SUPERMIMIC_CKPT", "")
  if ck_path:
    ck = torch.load(os.path.expanduser(ck_path), map_location="cpu", weights_only=False)
    h = heads["ee"] if ck["cfg"]["heads"].get("ee_mode") else heads["h4"]
  else:
    h = heads["ee"]
    pol = Policy(h.proprio_dim, h.cfg.hist, h.goal_frame_dim, h.cfg.K, J)
    torch.manual_seed(0)
    for nm, dim in (("obs_norm", h.proprio_dim), ("goal_norm", h.goal_frame_dim)):
      getattr(pol, nm).mean.copy_(torch.randn(dim) * 0.3)
      getattr(pol, nm).var.copy_(torch.rand(dim) + 0.05)
    ck = {"dims": dict(proprio=h.proprio_dim, hist=h.cfg.hist, goal_frame=h.goal_frame_dim, K=h.cfg.K, nj=J), "cfg": {"input_clip": 10.0},
          "policy": pol.state_dict()}
  assert ck["dims"]["goal_frame"] == h.goal_frame_dim
  with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "policy.onnx")
    err_rand = export_policy(ck, path, {"test": "1"})
    import onnxruntime as ort

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    d = ck["dims"]
    pol = Policy(d["proprio"], d["hist"], d["goal_frame"], d["K"], d["nj"], input_clip=float(ck["cfg"].get("input_clip", 10.0))).eval()
    pol.load_state_dict(ck["policy"])
    b = HumanGoalBuilder(meta_of(h), clip)
    s, q, imu, odom, z = robot_at(f0, np.zeros(2))
    b.start(f0, q, imu, odom, base_z=z)
    err_goal = 0.0
    for n in range(10):
      s, q, imu, odom, z = robot_at(f0 + n, np.zeros(2))
      g = b.step(q, imu, f0 + n, alpha_cmd=0.5, odom=odom, base_z=z)
      o = (torch.randn(1, d["hist"], d["proprio"]) * pol.obs_norm.std + pol.obs_norm.mean).numpy()
      with torch.no_grad():
        ref = pol(torch.tensor(o), torch.tensor(g)).numpy()
      out = sess.run(None, {"obs_hist": o, "goal": g})[0]
      err_goal = max(err_goal, float(np.abs(out - ref).max()))
  assert err_rand < 1e-4 and err_goal < 1e-4, (err_rand, err_goal)
  print(f"5. ONNX == torch ({'checkpoint ' + os.path.basename(ck_path) if ck_path else 'random policy'}): max err {err_rand:.1e} "
        f"random inputs, {err_goal:.1e} builder goals")
  print("test_human_deploy: OK")


if __name__ == "__main__":
  main()
