"""Play a SuperMimic policy in plain MuJoCo (no mjlab, no Warp): interactive
viewer with floor and a translucent reference ghost, or an mp4.

  python scripts/play_mujoco.py --bundle ~/supermimic/bundle_16000 \
      --motion ~/supermimic/data/lafan1/g1__lafan1_jumps1_subject1.npz --seconds 20 --viewer
  python scripts/play_mujoco.py --bundle ... --motion ... --video out.mp4

Needs: mujoco, torch, numpy and the ``supermimic`` package (pure torch parts only).
The bundle comes from scripts/export_bundle.py. Physics: 4 x 5 ms MuJoCo substeps per
50 Hz control step, built-in position actuators with the training PD gains.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import mujoco
import numpy as np
import torch

from supermimic.math_utils import (
  quat_apply,
  quat_apply_inverse,
  quat_error_magnitude,
  quat_inv,
  quat_mul,
  rot6d_from_quat,
  subtract_frame_transforms,
  yaw_quat,
)
from supermimic.policy import Policy

GRAVITY = torch.tensor([0.0, 0.0, -1.0])


def name2id(model, objtype, name):
  for cand in (f"robot/{name}", name):
    i = mujoco.mj_name2id(model, objtype, cand)
    if i >= 0:
      return i
  raise KeyError(name)


class Reference:
  """One clip remapped to the bundle's joint / body order."""

  def __init__(self, npz_path, meta):
    d = np.load(npz_path)
    self.fps = float(np.asarray(d["fps"]).reshape(-1)[0])
    src_j = [str(x) for x in d["joint_names"]]
    src_b = [str(x) for x in d["body_names"]]
    jperm = [src_j.index(n) for n in meta["joint_names"]]
    bperm = [src_b.index(n) for n in meta["body_names"]]
    self.q = torch.as_tensor(d["joint_pos"][:, jperm], dtype=torch.float32)
    self.qd = torch.as_tensor(d["joint_vel"][:, jperm], dtype=torch.float32)
    self.bp = torch.as_tensor(d["body_pos_w"][:, bperm], dtype=torch.float32)
    self.bq = torch.as_tensor(d["body_quat_w"][:, bperm], dtype=torch.float32)
    self.blv = torch.as_tensor(d["body_lin_vel_w"][:, bperm], dtype=torch.float32)
    self.bav = torch.as_tensor(d["body_ang_vel_w"][:, bperm], dtype=torch.float32)
    self.anchor = meta["body_names"].index(meta["anchor_body_name"])
    self.root = meta["body_names"].index(meta["root_body_name"])
    self.T = self.q.shape[0]


class MujocoPlayer:
  def __init__(self, bundle: str, device: str = "cpu", backend: str = "onnx"):
    self.meta = json.load(open(os.path.join(bundle, "meta.json")))
    m = self.meta
    self.model = mujoco.MjModel.from_binary_path(os.path.join(bundle, "scene.mjb"))
    self.data = mujoco.MjData(self.model)
    self.data_ref = mujoco.MjData(self.model)
    self.device = device
    model = self.model
    self.dt = float(m["control_dt"])
    self.decimation = int(m["decimation"])
    assert abs(model.opt.timestep * self.decimation - self.dt) < 1e-9
    # Joint addressing in the bundle's (entity) joint order.
    jids = [name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in m["joint_names"]]
    self.q_adr = np.array([model.jnt_qposadr[j] for j in jids])
    self.v_adr = np.array([model.jnt_dofadr[j] for j in jids])
    self.root_bid = name2id(model, mujoco.mjtObj.mjOBJ_BODY, m["root_body_name"])
    rj = int(model.body_jntadr[self.root_bid])
    assert model.jnt_type[rj] == mujoco.mjtJoint.mjJNT_FREE
    self.fq = int(model.jnt_qposadr[rj])
    self.fv = int(model.jnt_dofadr[rj])
    self.anchor_bid = name2id(model, mujoco.mjtObj.mjOBJ_BODY, m["anchor_body_name"])
    self.body_ids = [name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in m["body_names"]]
    # Action i -> actuator driving joint action_joint_names[i].
    act_of_joint = {int(model.actuator_trnid[a, 0]): a for a in range(model.nu)}
    self.act_ids = np.array([act_of_joint[name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in m["action_joint_names"]])
    self.scale = np.asarray(m["action_scale"], dtype=np.float64)
    self.offset = np.asarray(m["action_offset"], dtype=np.float64)
    self.nj = len(m["joint_names"])
    self.q0 = torch.tensor(m["default_joint_pos"])
    self.soft_lim = torch.tensor(m["soft_joint_limits"])
    h = m["heads"]
    self.K, self.hist = int(h["K"]), int(h["hist"])
    self.h = h
    d = m["dims"]
    onnx_path = os.path.join(bundle, "policy.onnx")
    self.sess = None
    if backend == "onnx" and os.path.exists(onnx_path):
      import onnxruntime as ort

      self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
      self.backend = "onnx"
      print(f"[play] policy backend: onnxruntime ({onnx_path})")
    else:
      self.policy = Policy(d["proprio"], d["hist"], d["goal_frame"], d["K"], d["nj"]).to(device).eval()
      self.policy.load_state_dict(torch.load(os.path.join(bundle, "policy.pt"), map_location=device))
      self.backend = "torch"
      print(f"[play] policy backend: torch ({os.path.join(bundle, 'policy.pt')})")
    self._vel6 = np.zeros(6)

  def act(self, hist: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    if self.sess is not None:
      out = self.sess.run(None, {"obs_hist": hist[None].numpy().astype(np.float32), "goal": goal[None].numpy().astype(np.float32)})[0]
      return torch.as_tensor(out[0])
    with torch.no_grad():
      return self.policy(hist[None], goal[None])[0]

  # ---- simulator readouts ---------------------------------------------------

  def body_state(self, bid):
    mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_XBODY, bid, self._vel6, 0)
    pos = torch.as_tensor(self.data.xpos[bid].copy(), dtype=torch.float32)
    quat = torch.as_tensor(self.data.xquat[bid].copy(), dtype=torch.float32)
    lin = torch.as_tensor(self._vel6[3:6].copy(), dtype=torch.float32)
    ang = torch.as_tensor(self._vel6[0:3].copy(), dtype=torch.float32)
    return pos, quat, lin, ang

  def proprio(self, prev_a):
    q = torch.as_tensor(self.data.qpos[self.q_adr], dtype=torch.float32)
    qd = torch.as_tensor(self.data.qvel[self.v_adr], dtype=torch.float32)
    _, rq, _, rw = self.body_state(self.root_bid)
    w_b = quat_apply_inverse(rq, rw)
    g_b = quat_apply_inverse(rq, GRAVITY)
    return torch.cat([q - self.q0, qd, w_b, g_b, prev_a])

  def goal(self, ref: Reference, frame: int):
    fr = torch.clamp(torch.arange(self.K) + frame, max=ref.T - 1)
    ap, aq, _, _ = self.body_state(self.anchor_bid)
    rel_p, rel_q = subtract_frame_transforms(ap.expand(self.K, 3), aq.expand(self.K, 4), ref.bp[fr, ref.anchor], ref.bq[fr, ref.anchor])
    return torch.cat([ref.q[fr], ref.qd[fr], rel_p, rot6d_from_quat(rel_q)], dim=-1)  # [K, 67]

  def set_reference_pose(self, dd: mujoco.MjData, ref: Reference, frame: int, with_velocity: bool = True):
    dd.qpos[self.fq : self.fq + 3] = ref.bp[frame, ref.root].numpy()
    dd.qpos[self.fq + 3 : self.fq + 7] = ref.bq[frame, ref.root].numpy()
    dd.qpos[self.q_adr] = ref.q[frame].numpy()
    if with_velocity:
      dd.qvel[self.fv : self.fv + 3] = ref.blv[frame, ref.root].numpy()
      R = np.zeros(9)
      mujoco.mju_quat2Mat(R, ref.bq[frame, ref.root].numpy().astype(np.float64))
      dd.qvel[self.fv + 3 : self.fv + 6] = R.reshape(3, 3).T @ ref.bav[frame, ref.root].numpy()
      dd.qvel[self.v_adr] = ref.qd[frame].numpy()
    else:
      dd.qvel[:] = 0.0
    mujoco.mj_forward(self.model, dd)

  # ---- reward / termination (same formulas as training, for logging) -------

  def reward_and_term(self, ref: Reference, frame: int, anchor_prev, a, prev_a):
    h = self.h
    Bt = len(self.body_ids)
    bp, bq, blv, bav = [], [], [], []
    for bid in self.body_ids:
      p, q, l, w = self.body_state(bid)
      bp.append(p); bq.append(q); blv.append(l); bav.append(w)
    bp, bq, blv, bav = torch.stack(bp), torch.stack(bq), torch.stack(blv), torch.stack(bav)
    ref_bp, ref_bq = ref.bp[frame], ref.bq[frame]
    ref_ap, ref_aq = ref_bp[ref.anchor], ref_bq[ref.anchor]
    rob_ap_prev, rob_aq_prev = anchor_prev
    delta_pos = torch.cat([rob_ap_prev[:2], ref_ap[2:3]])
    delta_ori = yaw_quat(quat_mul(rob_aq_prev, quat_inv(ref_aq)))
    body_pos_rel = delta_pos + quat_apply(delta_ori.expand(Bt, 4), ref_bp - ref_ap)
    body_quat_rel = quat_mul(delta_ori.expand(Bt, 4), ref_bq)
    ap, aq = bp[ref.anchor], bq[ref.anchor]
    e_root_pos = ((ref_ap - ap) ** 2).sum()
    e_root_ori = quat_error_magnitude(ref_aq[None], aq[None])[0] ** 2
    e_body_pos = ((body_pos_rel - bp) ** 2).sum(-1).mean()
    e_body_ori = (quat_error_magnitude(body_quat_rel, bq) ** 2).mean()
    e_lin = ((ref.blv[frame] - blv) ** 2).sum(-1).mean()
    e_ang = ((ref.bav[frame] - bav) ** 2).sum(-1).mean()
    q = torch.as_tensor(self.data.qpos[self.q_adr], dtype=torch.float32)
    lim = (-(q - self.soft_lim[:, 0]).clamp(max=0.0) + (q - self.soft_lim[:, 1]).clamp(min=0.0)).sum()
    r = (
      h["w_root_pos"] * torch.exp(-e_root_pos / h["std_root_pos"] ** 2)
      + h["w_root_ori"] * torch.exp(-e_root_ori / h["std_root_ori"] ** 2)
      + h["w_body_pos"] * torch.exp(-e_body_pos / h["std_body_pos"] ** 2)
      + h["w_body_ori"] * torch.exp(-e_body_ori / h["std_body_ori"] ** 2)
      + h["w_body_lin_vel"] * torch.exp(-e_lin / h["std_body_lin_vel"] ** 2)
      + h["w_body_ang_vel"] * torch.exp(-e_ang / h["std_body_ang_vel"] ** 2)
      + h["w_action_rate"] * ((a - prev_a) ** 2).sum()
      + h["w_joint_limit"] * lim
    ) * h["dt"]
    dz = (ref_ap[2] - ap[2]).abs()
    tilt = (quat_apply_inverse(ref_aq, GRAVITY)[2] - quat_apply_inverse(aq, GRAVITY)[2]).abs()
    fell = bool(dz > h["term_z"] or tilt > h["term_tilt"])
    return float(r), fell

  @property
  def reward_max(self):
    h = self.h
    return (h["w_root_pos"] + h["w_root_ori"] + h["w_body_pos"] + h["w_body_ori"] + h["w_body_lin_vel"] + h["w_body_ang_vel"]) * h["dt"]

  # ---- rollout -------------------------------------------------------------

  @torch.no_grad()
  def play(self, ref: Reference, start_s=0.0, seconds=15.0, viewer=False, video=None, slowmo=1.0, width=960, height=600, fps_out=25, stop_on_fall=True):
    model, data = self.model, self.data
    f0 = int(start_s / self.dt)
    f_last = min(ref.T - 1, f0 + int(seconds / self.dt))
    self.set_reference_pose(data, ref, f0, with_velocity=True)
    prev_a = torch.zeros(self.nj)
    hist = self.proprio(prev_a)[None].repeat(self.hist, 1)
    frame = f0 + 1  # mjlab semantics: the first target is the frame after the reset frame
    anchor_prev = self.body_state(self.anchor_bid)[:2]

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = self.root_bid
    cam.distance, cam.azimuth, cam.elevation = 3.2, 135.0, -15.0
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[0] = opt.geomgroup[1] = opt.geomgroup[2] = 1  # floor + visual meshes
    ghost_opt = mujoco.MjvOption()
    ghost_opt.geomgroup[:] = 0
    ghost_opt.geomgroup[2] = 1
    pert = mujoco.MjvPerturb()
    v = None
    renderer = None
    frames = []
    if viewer:
      import importlib

      mjv = importlib.import_module("mujoco.viewer")
      v = mjv.launch_passive(model, data, show_left_ui=False, show_right_ui=False)
      v.cam.type, v.cam.trackbodyid = cam.type, cam.trackbodyid
      v.cam.distance, v.cam.azimuth, v.cam.elevation = cam.distance, cam.azimuth, cam.elevation
      v.opt.geomgroup[:] = opt.geomgroup
      t_wall0 = time.time()
    if video:
      os.environ.setdefault("MUJOCO_GL", "egl")
      renderer = mujoco.Renderer(model, height, width)
      render_every = max(1, int(round(1.0 / (self.dt * fps_out))))

    sum_r, steps, fell_at = 0.0, 0, None
    for t in range(f0 + 1, f_last + 1):
      goal = self.goal(ref, frame)
      a = self.act(hist, goal)
      data.ctrl[self.act_ids] = self.offset + self.scale * a.numpy().astype(np.float64)
      for _ in range(self.decimation):
        mujoco.mj_step(model, data)
      mujoco.mj_forward(model, data)
      r, fell = self.reward_and_term(ref, frame, anchor_prev, a, prev_a)
      sum_r += r
      steps += 1
      if fell and fell_at is None:
        fell_at = t * self.dt
        print(f"[play] fell at t={fell_at:.2f}s (frame {t})", flush=True)
      # draw
      if v is not None or (renderer is not None and (t - f0) % render_every == 0):
        self.set_reference_pose(self.data_ref, ref, t, with_velocity=False)
      if renderer is not None and (t - f0) % render_every == 0:
        renderer.update_scene(data, camera=cam, scene_option=opt)
        scn = renderer.scene
        n0 = scn.ngeom
        mujoco.mjv_addGeoms(model, self.data_ref, ghost_opt, pert, mujoco.mjtCatBit.mjCAT_DYNAMIC, scn)
        for gi in range(n0, scn.ngeom):
          scn.geoms[gi].rgba[:] = (0.3, 0.9, 0.4, 0.35)
        frames.append(renderer.render().copy())
      if v is not None:
        if not v.is_running():
          break
        scn = v.user_scn
        scn.ngeom = 0
        mujoco.mjv_addGeoms(model, self.data_ref, ghost_opt, pert, mujoco.mjtCatBit.mjCAT_DYNAMIC, scn)
        for gi in range(scn.ngeom):
          scn.geoms[gi].rgba[:] = (0.3, 0.9, 0.4, 0.35)
        v.sync()
        lag = t_wall0 + (t - f0) * self.dt * slowmo - time.time()
        if lag > 0:
          time.sleep(lag)
      if fell and stop_on_fall:
        break
      prev_a = a
      frame = min(frame + 1, ref.T - 1)
      anchor_prev = self.body_state(self.anchor_bid)[:2]
      hist = torch.cat([hist[1:], self.proprio(prev_a)[None]], dim=0)
    if v is not None:
      v.close()
    if frames:
      import mediapy

      mediapy.write_video(video, frames, fps=fps_out)
    return dict(steps=steps, seconds=steps * self.dt, fell_at=fell_at, norm_return=sum_r / max(steps * self.reward_max, 1e-9), video=video)


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--bundle", type=str, required=True)
  p.add_argument("--motion", type=str, required=True, help="reference npz")
  p.add_argument("--start_s", type=float, default=0.0)
  p.add_argument("--seconds", type=float, default=15.0)
  p.add_argument("--viewer", action="store_true")
  p.add_argument("--video", type=str, default=None, help="mp4 path (offscreen, floor + ghost in one view)")
  p.add_argument("--slowmo", type=float, default=1.0)
  p.add_argument("--keep_going", action="store_true", help="do not stop at the first fall")
  p.add_argument("--width", type=int, default=960)
  p.add_argument("--height", type=int, default=600)
  p.add_argument("--backend", type=str, default="onnx", choices=["onnx", "torch"], help="policy inference backend (default: the exported ONNX)")
  args = p.parse_args()
  torch.set_num_threads(4)
  player = MujocoPlayer(os.path.expanduser(args.bundle), backend=args.backend)
  ref = Reference(os.path.expanduser(args.motion), player.meta)
  assert abs(ref.fps - 1.0 / player.dt) < 1e-6, f"reference fps {ref.fps} != control rate {1/player.dt}"
  res = player.play(ref, args.start_s, args.seconds, viewer=args.viewer, video=args.video, slowmo=args.slowmo, width=args.width, height=args.height, stop_on_fall=not args.keep_going)
  print("[play]", json.dumps(res))


if __name__ == "__main__":
  main()
