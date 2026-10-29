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

End-effector view (--align ee, human-reference checkpoints, H4 or ee_mode): the human drawn exactly where the reward
of world end-effector tracking puts it -- its world motion, unscaled and unrotated, shifted so that the clip's
first-frame pelvis xy lands on the env origin (where the reset puts the robot pelvis, on the human heading) -- and at
the frame the reward of each step uses. On top: the five task targets (green: torso point, wrists, ankles; the torso
height is not a target), the robot's task links (blue) and a red line from each robot wrist to its target. --alpha sets
an ee_mode policy's alpha (default 1). The wrist / torso-xy errors are written into the video and printed per second.
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
  ap.add_argument("--align", choices=["side", "ee"], default="side",
                  help="side: human beside the robot (--side); ee: where the end-effector reward puts it (module docstring)")
  ap.add_argument("--alpha", type=float, default=1.0, help="--align ee with an ee_mode checkpoint: its alpha")
  ap.add_argument("--label", default="", help="text in the video's top-left corner (default: checkpoint run name)")
  args = ap.parse_args()
  if args.align == "ee":
    args.side = 0.0

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
  if args.align == "ee":
    if reference != "human":
      ap.error("--align ee needs a human-reference checkpoint")
    if getattr(heads_cfg, "ee_mode", False):
      import dataclasses

      heads_cfg = dataclasses.replace(heads_cfg, eval_alpha=args.alpha)
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

  ee = None
  if args.align == "ee":  # the reward's mapping: human world - first-frame pelvis xy + env origin; targets / robot task links
    import dataclasses

    from supermimic.human.heads import COND_DIM, HumanHeads

    lib = ev.lib
    lib.ensure_ee(cmd.bone_map)
    hd = heads
    heads_ee = hd if hd.ee_on else HumanHeads(hd.fk, hd.links.tolist(), hd.q0, hd.soft_lim, dataclasses.replace(heads_cfg, ee_mode=True),
                                              hd.rest_h, hd.ankle_h)
    start = int(lib.clip_start[int(cmd.env_clip[0])])
    anchor = lib.gather(torch.tensor([start], device=args.device), fields=("h_root_xy",))["h_root_xy"][0].cpu().numpy()
    shift = np.array([origin[0] - anchor[0], origin[1] - anchor[1], origin[2]])
    cond = torch.zeros(1, COND_DIM, device=args.device)
    cond[:, :2] = 1.0
    cond[:, 2:4] = torch.as_tensor(anchor, device=args.device)
    ee = dict(heads=heads_ee, lib=lib, start=start, cond=cond, origin=origin, log=[])
    print(f"[play] align ee: clip frames from {start}, anchor (first-frame pelvis xy) {anchor.round(3)}, "
          + (f"ee_mode policy at alpha {args.alpha:g}" if hd.ee_on else "H4 policy (no end-effector input)"))

  def ee_overlay(scn, ns, f):
    """Targets (green), robot task links (blue), wrist error lines (red) of state ns against target frame f."""
    h = ee["heads"]
    g = ee["lib"].gather(torch.tensor([f], device=args.device), fields=("h_task_pos", "h_task_rot6d", "h_yaw", "h_contact"))
    kin_e = h.kin(ns)
    e = h.ee_errors(kin_e, {**g, "cond": ee["cond"]})
    tgt = h.ee_targets(g["h_task_pos"], g["h_task_rot6d"], ee["cond"])[0][0].cpu().numpy() + ee["origin"]
    rob = kin_e["task_pos"][0].cpu().numpy() + ee["origin"]
    for i in range(tgt.shape[0]):
      add_sphere(scn, tgt[i], 0.045 if i else 0.035, (0.1, 0.9, 0.2, 0.8))
      add_sphere(scn, rob[i], 0.035, (0.15, 0.4, 1.0, 0.9))
    for i in (1, 2):
      add_bone(scn, rob[i], tgt[i], 0.008, (1.0, 0.1, 0.1, 1.0))
    w = e["wrist_pos"].sqrt()[0].cpu().numpy()
    ee["log"].append((w[0], w[1], float(e["torso_xy"].sqrt()[0])))
    return w, float(e["torso_xy"].sqrt()[0])

  def annotate(img, lines):
    from PIL import Image, ImageDraw

    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    try:
      from PIL import ImageFont

      font = ImageFont.load_default(size=max(14, args.height // 32))
    except TypeError:  # Pillow < 10.1
      font = None
    for k, txt in enumerate(lines):
      dr.text((12, 10 + int(1.3 * max(14, args.height // 32)) * k), txt, font=font, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    return np.asarray(im)

  label = args.label or os.path.basename(os.path.dirname(os.path.abspath(args.ckpt)))

  cam = mujoco.MjvCamera()
  cam.type = mujoco.mjtCamera.mjCAMERA_FREE
  cam.distance, cam.azimuth, cam.elevation = 4.5, float(np.degrees(yaw)) + 150.0, -15.0  # front-left of the start heading
  if args.align == "ee":
    cam.distance, cam.elevation = 2.8, -12.0
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
      f_tgt = int(cmd.time_steps[0])  # the frame this step's reward compares the next state with
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
            draw_human(viewer.user_scn, hj[f_tgt - ee["start"]] + shift if ee else human_at(t + 1), (1.0, 0.55, 0.1, 0.5) if ee else (1.0, 0.55, 0.1, 0.9))
          if ee:
            ee_overlay(viewer.user_scn, ns, f_tgt)
        viewer.sync()
        time.sleep(max(0.0, 0.02 - (time.time() - t_wall)))
      else:
        cam.lookat[:] = pelvis + 0.5 * args.side * left
        renderer.update_scene(data, cam)
        if not args.no_human:
          draw_human(renderer.scene, hj[f_tgt - ee["start"]] + shift if ee else human_at(t + 1), (1.0, 0.55, 0.1, 0.5) if ee else (1.0, 0.55, 0.1, 0.9))
        if ee:
          w, txy = ee_overlay(renderer.scene, ns, f_tgt)
        img = renderer.render().copy()
        if not ee:
          img = annotate(img, [label, f"t {t / 50.0:5.2f} s" + ("   TERMINATED (training test) at %.2f s" % (failed_at / 50.0) if failed_at is not None else ""),
                               "orange: human (SMPL, real size)"])
        if ee:
          img = annotate(img, [
            label, f"t {t / 50.0:5.2f} s" + ("   TERMINATED (training test) at %.2f s" % (failed_at / 50.0) if failed_at is not None else ""),
            f"wrist error L {w[0] * 100:5.1f} cm  R {w[1] * 100:5.1f} cm   torso xy {txy * 100:5.1f} cm",
            "orange: human (reward alignment)  green: targets  blue: robot task links  red: wrist error"])
        frames.append(img)
      prev = a
      kin, obs = fns.refresh(ns, prev, bias, False)
      hist.push(obs)
  print(f"[play] {t + 1} steps ({(t + 1) / 50.0:.1f} s), " + ("no termination" if failed_at is None else f"terminated at {failed_at / 50.0:.2f} s"))
  if ee and ee["log"]:
    L = np.array(ee["log"])
    for s0 in range(0, len(L), 50):
      seg = L[s0 : s0 + 50]
      print(f"[play]   {s0 / 50.0:5.1f} s: wrist L {seg[:, 0].mean() * 100:5.1f} R {seg[:, 1].mean() * 100:5.1f} cm, torso xy {seg[:, 2].mean() * 100:5.1f} cm")
    print(f"[play] mean wrist error {L[:, :2].mean() * 100:.1f} cm (within 5 cm: {(L[:, :2] < 0.05).mean() * 100:.0f}%), torso xy {L[:, 2].mean() * 100:.1f} cm")
  if renderer is not None:
    import imageio

    out = args.video or os.path.join(os.path.dirname(os.path.abspath(args.ckpt)), f"{pair['name']}_{reference}.mp4")
    imageio.mimwrite(out, frames, fps=50, quality=8)
    print(f"[play] wrote {out}")


if __name__ == "__main__":
  main()
