"""Sim2sim check: run a SuperMimic checkpoint in CPU MuJoCo (not MuJoCo Warp)
and render a side-by-side video (left: policy-driven robot, right: reference
motion played kinematically).

  CUDA_VISIBLE_DEVICES=5 uv run python ../supermimic_repro/scripts/sim2sim_mujoco.py \
      --ckpt ~/supermimic/logs/<run>/model_16000.pt --motion_dir ~/supermimic/data/lafan1 \
      --clips dance1_subject1 jumps1_subject1 --seconds 12 --out_dir ~/supermimic/videos

The compiled model (actuators, PD gains, joint limits, floor) is taken from the
mjlab env so the only thing that changes is the physics backend.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import mujoco
import numpy as np
import torch


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--ckpt", type=str, required=True)
  p.add_argument("--motion_dir", type=str, required=True)
  p.add_argument("--clips", type=str, nargs="+", default=["dance1_subject1"], help="substrings of clip names")
  p.add_argument("--start_s", type=float, default=0.0)
  p.add_argument("--seconds", type=float, default=12.0)
  p.add_argument("--out_dir", type=str, default="videos")
  p.add_argument("--device", type=str, default="cuda:0")
  p.add_argument("--width", type=int, default=640)
  p.add_argument("--height", type=int, default=480)
  p.add_argument("--fps_out", type=int, default=25)
  p.add_argument("--no_video", action="store_true")
  args = p.parse_args()
  os.environ.setdefault("MUJOCO_GL", "egl")

  import mediapy

  from mjlab.envs import ManagerBasedRlEnv

  from supermimic.env_cfg import make_env_cfg
  from supermimic.heads import HeadsCfg
  from supermimic.policy import Policy
  from supermimic.sim_interface import ObsHistory, SimInterface

  ck = torch.load(args.ckpt, map_location=args.device)
  d = ck["dims"]
  heads_cfg = HeadsCfg(**{k: v for k, v in ck["cfg"]["heads"].items() if k != "extra"})
  policy = Policy(d["proprio"], d["hist"], d["goal_frame"], d["K"], d["nj"]).to(args.device).eval()
  policy.load_state_dict(ck["policy"])

  all_files = sorted(glob.glob(os.path.join(os.path.expanduser(args.motion_dir), "*.npz")))
  files = [f for f in all_files if any(c in os.path.basename(f) for c in args.clips)]
  assert files, f"no clips match {args.clips}"
  os.makedirs(args.out_dir, exist_ok=True)
  results = {}
  for mf in files:
    name = os.path.basename(mf).replace(".npz", "")
    res = run_clip(mf, name, policy, heads_cfg, args, ManagerBasedRlEnv, make_env_cfg, SimInterface, ObsHistory, mediapy)
    results[name] = res
    print(f"[sim2sim] {name}: {json.dumps(res)}", flush=True)
  with open(os.path.join(args.out_dir, "sim2sim_results.json"), "w") as f:
    json.dump(results, f, indent=2)


def run_clip(mf, name, policy, heads_cfg, args, ManagerBasedRlEnv, make_env_cfg, SimInterface, ObsHistory, mediapy):
  dev = args.device
  # mjlab env only to get the compiled model + heads/FK; physics runs in CPU MuJoCo.
  env = ManagerBasedRlEnv(cfg=make_env_cfg([mf], 1, mode="eval"), device=dev)
  sim = SimInterface(env, heads_cfg)
  heads, lib, robot = sim.heads, sim.lib, sim.robot
  model: mujoco.MjModel = env.sim.mj_model
  data = mujoco.MjData(model)
  data_ref = mujoco.MjData(model)
  idx = robot.indexing
  q_adr = idx.joint_q_adr.cpu().numpy()
  v_adr = idx.joint_v_adr.cpu().numpy()
  fq = idx.free_joint_q_adr.cpu().numpy()
  fv = idx.free_joint_v_adr.cpu().numpy()
  root_bid = int(idx.root_body_id)
  nj = sim.nj
  # Action -> actuator mapping (policy action index i controls joint target_names[i]).
  term = env.action_manager.get_term("joint_pos")
  tnames = list(term.target_names)
  scale = term._scale[0].detach().cpu().numpy() if torch.is_tensor(term._scale) else np.full(nj, float(term._scale))
  offset = term._offset[0].detach().cpu().numpy() if torch.is_tensor(term._offset) else np.full(nj, float(term._offset))
  jname2act = {}
  for a in range(model.nu):
    jid = int(model.actuator_trnid[a, 0])
    jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
    jname2act[jname.split("/")[-1]] = a  # scene joints are namespaced "robot/<joint>"
  act_ids = np.array([jname2act[n.split("/")[-1]] for n in tnames])
  assert len(set(act_ids.tolist())) == nj, "action -> actuator mapping is not one-to-one"
  dt = env.step_dt
  decimation = env.cfg.decimation
  assert abs(model.opt.timestep * decimation - dt) < 1e-9

  # Reference clip.
  f0 = int(args.start_s / dt)
  clip_end = int(lib.clip_end[0].item())
  f_last = min(clip_end - 1, f0 + int(args.seconds / dt))
  jp = lib.joint_pos.cpu().numpy()
  jv = lib.joint_vel.cpu().numpy()
  bp = lib.body_pos_w.cpu().numpy()
  bq = lib.body_quat_w.cpu().numpy()
  blv = lib.body_lin_vel_w.cpu().numpy()
  bav = lib.body_ang_vel_w.cpu().numpy()

  def set_ref(dd, f):
    dd.qpos[fq[0:3]] = bp[f, 0]
    dd.qpos[fq[3:7]] = bq[f, 0]
    dd.qpos[q_adr] = jp[f]
    dd.qvel[fv[0:3]] = blv[f, 0]
    # free joint angular velocity is in the body frame
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, bq[f, 0])
    dd.qvel[fv[3:6]] = R.reshape(3, 3).T @ bav[f, 0]
    dd.qvel[v_adr] = jv[f]
    mujoco.mj_forward(model, dd)

  set_ref(data, f0)
  vel6 = np.zeros(6)

  def state_vec():
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, root_bid, vel6, 0)  # [ang, lin] world
    s = np.concatenate([data.xpos[root_bid], data.xquat[root_bid], vel6[3:6], vel6[0:3], data.qpos[q_adr], data.qvel[v_adr]])
    return torch.as_tensor(s, dtype=torch.float32, device=dev)[None]

  # Rendering setup.
  frames = []
  if not args.no_video:
    renderer = mujoco.Renderer(model, args.height, args.width)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[2] = 1  # visual meshes only
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = root_bid
    cam.distance, cam.azimuth, cam.elevation = 3.0, 135.0, -12.0
    render_every = max(1, int(round(1.0 / (dt * args.fps_out))))

  # Policy state.
  prev_a = torch.zeros(1, nj, device=dev)
  bias = torch.zeros(1, nj, device=dev)
  frame = torch.tensor([f0 + 1], device=dev)  # mjlab increments the target after reset
  clip_end_t = torch.tensor([clip_end], device=dev)
  state = state_vec()
  with torch.no_grad():
    kin = heads.kin(state)
    hist = ObsHistory(1, heads_cfg.hist, heads.proprio_dim, dev)
    hist.reset(torch.zeros(1, dtype=torch.long, device=dev), heads.proprio(state, prev_a, bias, False))
  sum_r, steps, failed = 0.0, 0, False
  for t in range(f0 + 1, f_last + 1):
    with torch.no_grad():
      goal = heads.goal(kin, frame, clip_end_t)
      a = policy(hist.buf, goal)
    target = offset + scale * a[0].cpu().numpy()
    data.ctrl[act_ids] = target
    for _ in range(decimation):
      mujoco.mj_step(model, data)
    next_state = state_vec()
    with torch.no_grad():
      kin_next = heads.kin(next_state)
      r = heads.reward(kin, kin_next, next_state, a, prev_a, frame)
      hard, _ = heads.termination(kin_next, frame)
    sum_r += float(r)
    steps += 1
    if bool(hard):
      failed = True
      print(f"[sim2sim] {name}: terminated at t={t * dt:.2f}s (frame {t})", flush=True)
    if not args.no_video and (t - f0) % render_every == 0:
      set_ref(data_ref, t)
      renderer.update_scene(data, camera=cam, scene_option=opt)
      img_sim = renderer.render().copy()
      renderer.update_scene(data_ref, camera=cam, scene_option=opt)
      img_ref = renderer.render().copy()
      frames.append(np.concatenate([img_sim, img_ref], axis=1))
    if failed:
      break
    prev_a = a
    frame = torch.minimum(frame + 1, clip_end_t - 1)
    state, kin = next_state, kin_next
    with torch.no_grad():
      hist.push(heads.proprio(state, prev_a, bias, False))
  rmax = heads_cfg.reward_max_per_step
  res = dict(steps=steps, seconds=steps * dt, failed=failed, norm_return=sum_r / max(steps * rmax, 1e-9))
  if frames:
    out = os.path.join(args.out_dir, f"{name}.mp4")
    mediapy.write_video(out, frames, fps=args.fps_out)
    res["video"] = out
  env.close()
  return res


if __name__ == "__main__":
  main()
