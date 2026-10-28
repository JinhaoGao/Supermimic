#!/usr/bin/env python3
"""Play a checkpoint of either reference on one clip, with the human (SMPL) skeleton drawn next to the robot.

Runs the policy exactly as the evaluator does (mjlab env, MuJoCo Warp physics, eval mode: clip from frame 0, no DR /
noise, deterministic policy) and renders the robot with MuJoCo; the human motion of the same clip (pairs.tsv) is drawn
as an orange skeleton at its real size, its pelvis placed beside the robot's start (--side) and moving with the human's
own trajectory. Robot-reference checkpoints play the retargeted clip, human-reference ones the human clip.

  cd ~/supermimic_ws/mjlab
  uv run python ~/Supermimic/scripts/human/play.py --ckpt <model.pt> --pairs ~/datasets/paired_amass/pairs.tsv \\
      --clip CMU_06_06_08 --video out.mp4            # mp4 (default)
  uv run python ~/Supermimic/scripts/human/play.py ... --viewer   # interactive MuJoCo viewer (real time, needs a display)
  --list [substring]: list clip names of pairs.tsv (with --subset: only those, e.g. ambig_top.txt)
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import mujoco
import numpy as np
import torch

SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]  # 22 body + 2 hand points


def read_pairs(path):
  with open(os.path.expanduser(path)) as f:
    head = f.readline().rstrip("\n").split("\t")
    return [dict(zip(head, line.rstrip("\n").split("\t"))) for line in f if line.strip()]


def add_sphere(scn, pos, r, rgba):
  if scn.ngeom >= scn.maxgeom:
    return
  g = scn.geoms[scn.ngeom]
  mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([r, 0, 0]), np.asarray(pos, np.float64), np.eye(3).ravel(), np.asarray(rgba, np.float32))
  scn.ngeom += 1


def add_bone(scn, a, b, r, rgba):
  if scn.ngeom >= scn.maxgeom:
    return
  g = scn.geoms[scn.ngeom]
  mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.asarray(rgba, np.float32))
  mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, r, np.asarray(a, np.float64), np.asarray(b, np.float64))
  scn.ngeom += 1


def draw_human(scn, joints, rgba=(1.0, 0.55, 0.1, 0.9)):
  for j, p in enumerate(SMPL_PARENTS):
    add_sphere(scn, joints[j], 0.03, rgba)
    if p >= 0:
      add_bone(scn, joints[p], joints[j], 0.018, rgba)


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--ckpt", required=False)
  ap.add_argument("--pairs", required=True, help="pairs.tsv of pair_with_retarget.py")
  ap.add_argument("--clip", default=None, help="clip name (exact, or a unique substring)")
  ap.add_argument("--list", nargs="?", const="", default=None, help="list clip names containing this substring and exit")
  ap.add_argument("--subset", default=None, help="with --list: only names in this list (e.g. ambig_top.txt)")
  ap.add_argument("--video", default=None, help="mp4 path (default: <clip>_<reference>.mp4 next to the checkpoint)")
  ap.add_argument("--viewer", action="store_true", help="interactive viewer instead of a video")
  ap.add_argument("--side", type=float, default=1.0, help="human skeleton offset to the robot's left [m] (0 = overlaid)")
  ap.add_argument("--no_human", action="store_true")
  ap.add_argument("--seconds", type=float, default=0.0, help="stop after this many seconds (0 = whole clip)")
  ap.add_argument("--width", type=int, default=1280)
  ap.add_argument("--height", type=int, default=720)
  ap.add_argument("--cache_dir", default=os.path.expanduser("~/.cache/supermimic_play"))
  ap.add_argument("--device", default="cuda:0")
  args = ap.parse_args()

  pairs = read_pairs(args.pairs)
  if args.list is not None:
    keep = None
    if args.subset:
      keep = {l.strip().replace(".npz", "") for l in open(os.path.expanduser(args.subset)) if l.strip()}
    for r in pairs:
      if args.list.lower() in r["name"].lower() and (keep is None or r["name"] in keep):
        print(f"{r['name']}\t{int(r['T_human']) / 50.0:.1f} s\t{r.get('flags', '')}")
    return
  if not args.ckpt or not args.clip:
    ap.error("--ckpt and --clip are required (or --list)")
  hits = [r for r in pairs if r["name"] == args.clip] or [r for r in pairs if args.clip in r["name"]]
  if len(hits) != 1:
    sys.exit(f"--clip {args.clip!r}: {len(hits)} matches ({[r['name'] for r in hits[:8]]}); use --list")
  pair = hits[0]

  from supermimic.evaluate import Evaluator
  from supermimic.heads import HeadsCfg
  from supermimic.policy import Policy
  from supermimic.sim_interface import ObsHistory

  ck = torch.load(args.ckpt, map_location=args.device, weights_only=False)
  cfg, d = ck["cfg"], ck["dims"]
  reference = cfg.get("reference", "robot")
  hk = {k: v for k, v in cfg["heads"].items() if k != "extra"}
  if reference == "robot":
    hk["goal_anchor_pos"] = d["goal_frame"] == 2 * d["nj"] + 9
  heads_cfg = HeadsCfg.from_dict(hk)
  kw = {"input_clip": float(cfg["input_clip"])} if "input_clip" in cfg else {}
  policy = Policy(d["proprio"], d["hist"], d["goal_frame"], d["K"], d["nj"], **kw).to(args.device)
  policy.load_state_dict(ck["policy"])
  policy.eval()
  motion = pair["human" if reference == "human" else "robot"]
  print(f"[play] {reference}-reference checkpoint (it {ck['it']}) on {pair['name']} ({motion})")
  ev = Evaluator([motion], heads_cfg, args.device, max_envs=1, compile_=False, cache_dir=args.cache_dir,
                 robot=cfg.get("robot", "g1"), robot_variant=cfg.get("robot_variant", ""), reference=reference)
  env, sim, heads, cmd = ev.env, ev.sim, ev.heads, ev.cmd
  fns = ev._get_fns(policy)

  # Human skeleton of the same clip (human frame = robot frame + lag for the retargeted clip).
  hum = np.load(pair["human"], allow_pickle=True)
  hj = np.asarray(hum["smpl_joints_w"], np.float64)
  lag = int(pair.get("lag", 0) or 0) if reference == "robot" else 0

  cmd._start_counter = 0
  env.reset()
  model = env.sim.mj_model
  data = mujoco.MjData(model)
  state = sim.state()
  xi = sim.xi()
  bias = sim.bias_from_xi(xi, sim.nj)
  prev = torch.zeros(1, sim.nj, device=args.device)
  kin, obs = fns.refresh(state, prev, bias, False)
  hist = ObsHistory(1, heads_cfg.hist, heads.proprio_dim, args.device)
  hist.reset(torch.arange(1, device=args.device), obs)
  f0 = int(cmd.time_steps[0])
  clip_len = int(cmd.clip_end_env[0]) - f0
  steps = clip_len - 1 if args.seconds <= 0 else min(clip_len - 1, int(args.seconds * 50))

  # Place the human: its start pelvis beside the robot's start pelvis (robot heading's left), floor on the ground.
  q0 = env.sim.data.qpos[0].cpu().numpy()
  rq = q0[3:7]
  yaw = np.arctan2(2 * (rq[0] * rq[3] + rq[1] * rq[2]), 1 - 2 * (rq[2] ** 2 + rq[3] ** 2))
  left = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
  origin = env.scene.env_origins[0].cpu().numpy()
  shift = np.array([q0[0], q0[1], 0.0]) - np.array([hj[lag, 0, 0], hj[lag, 0, 1], 0.0]) + args.side * left
  shift[2] = origin[2]

  def human_at(t):
    return hj[min(max(t + lag, 0), len(hj) - 1)] + shift

  cam = mujoco.MjvCamera()
  cam.type = mujoco.mjtCamera.mjCAMERA_FREE
  cam.distance, cam.azimuth, cam.elevation = 4.5, float(np.degrees(yaw)) + 150.0, -15.0  # front-left of the start heading
  frames, failed_at = [], None
  renderer = viewer = None
  if args.viewer:
    from mujoco import viewer as mj_viewer

    viewer = mj_viewer.launch_passive(model, data)
  else:
    renderer = mujoco.Renderer(model, args.height, args.width)

  with torch.no_grad():
    for t in range(steps):
      t_wall = time.time()
      refs = sim.goal_refs(cmd.time_steps.clone(), cmd.clip_end_env.clone())
      _, a = fns.act(hist.buf, kin, *refs)
      env.step(a)
      ns = sim.state()
      if failed_at is None and bool(env.reset_terminated[0]):
        failed_at = t
        print(f"[play] training termination at step {t} ({t / 50.0:.2f} s); playing on")
      if bool(env.reset_terminated[0]):  # playback only: keep simulating past the termination (no reset)
        env._manual_reset_pending[0] = False
      data.qpos[:] = env.sim.data.qpos[0].cpu().numpy()
      data.qvel[:] = env.sim.data.qvel[0].cpu().numpy()
      mujoco.mj_forward(model, data)
      pelvis = data.qpos[:3].copy()
      if viewer is not None:
        if not viewer.is_running():
          break
        with viewer.lock():
          viewer.user_scn.ngeom = 0
          if not args.no_human:
            draw_human(viewer.user_scn, human_at(t + 1))
        viewer.sync()
        time.sleep(max(0.0, 0.02 - (time.time() - t_wall)))
      else:
        cam.lookat[:] = pelvis + 0.5 * args.side * left
        renderer.update_scene(data, cam)
        if not args.no_human:
          draw_human(renderer.scene, human_at(t + 1))
        frames.append(renderer.render().copy())
      prev = a
      kin, obs = fns.refresh(ns, prev, bias, False)
      hist.push(obs)
  print(f"[play] {t + 1} steps ({(t + 1) / 50.0:.1f} s), " + ("no termination" if failed_at is None else f"terminated at {failed_at / 50.0:.2f} s"))
  if renderer is not None:
    import imageio

    out = args.video or os.path.join(os.path.dirname(os.path.abspath(args.ckpt)), f"{pair['name']}_{reference}.mp4")
    imageio.mimwrite(out, frames, fps=50, quality=8)
    print(f"[play] wrote {out}")


if __name__ == "__main__":
  main()
