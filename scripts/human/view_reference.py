#!/usr/bin/env python3
"""Kinematic playback of the robot reference motions of a clip (no policy, no physics), beside the SMPL human.

Panels (left to right), each with the human skeleton (orange, real size) to the robot's left:
  retarget   the retargeted robot clip of pairs.tsv (smplkp_full holosoma retarget: the ab1 baseline's reference)
  ik         the bone-direction IK of the human clip (init_g1_* states; scripts/human/prepare_init_states.py): root
             height / orientation and joints from the IK, horizontal root path = k x the human pelvis path
The G1 is mjlab's model (team eval variant by default) on a floor; the per-frame bone errors against the human
(global torso / arm / leg, HumanHeads.metrics) are printed per panel.

  cd ~/supermimic_ws/mjlab && uv run python ~/Supermimic/scripts/human/view_reference.py \\
      --pairs ~/datasets/paired_amass/pairs.tsv --clip CMU_94_94_07 --video ref_CMU_94_94_07.mp4 [--panels retarget,ik]
"""

from __future__ import annotations

import argparse
import os
import types

import mujoco
import numpy as np
import torch

from play import draw_human, read_pairs  # same directory


def build_model(variant: str):
  from supermimic.env_cfg import base_tracking_cfg

  spec = base_tracking_cfg("g1", variant).scene.entities["robot"].spec_fn()
  spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[50, 50, 0.1], rgba=[0.55, 0.65, 0.78, 1])
  spec.worldbody.add_light(pos=[0, 0, 6], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8])
  spec.visual.global_.offwidth, spec.visual.global_.offheight = 1920, 1080
  spec.visual.headlight.ambient, spec.visual.headlight.diffuse = [0.45, 0.45, 0.45], [0.6, 0.6, 0.6]
  return spec.compile()


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--pairs", required=True)
  ap.add_argument("--clip", required=True)
  ap.add_argument("--panels", default="retarget,ik")
  ap.add_argument("--video", default=None)
  ap.add_argument("--variant", default="eval+evalcontact")
  ap.add_argument("--side", type=float, default=1.0)
  ap.add_argument("--seconds", type=float, default=0.0)
  ap.add_argument("--width", type=int, default=960)
  ap.add_argument("--height", type=int, default=720)
  args = ap.parse_args()

  from supermimic.fk import TorchFK
  from supermimic.human import bone_map as bone
  from supermimic.human.command import robot_rest_geometry
  from supermimic.human.heads import STEP_FIELDS, HumanHeads, HumanHeadsCfg
  from supermimic.human.motions import derive_fields
  from supermimic.motions import FALLBACK_JOINT_NAMES

  pairs = read_pairs(args.pairs)
  hits = [r for r in pairs if r["name"] == args.clip] or [r for r in pairs if args.clip in r["name"]]
  assert len(hits) == 1, [r["name"] for r in hits[:8]]
  pr = hits[0]
  model = build_model(args.variant)
  data = mujoco.MjData(model)
  hinge = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
  jnames = tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in hinge)
  qadr = np.array([model.jnt_qposadr[j] for j in hinge])
  free = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE][0]
  ra = int(model.jnt_qposadr[free])

  # Human (skeleton + derived reference fields for the error printout).
  bm = bone.SMPL_BONE_MAPS["g1"]
  h = dict(np.load(pr["human"], allow_pickle=True))
  hj = np.asarray(h["smpl_joints_w"], np.float64)
  hf = derive_fields(h, bone.BoneMapTensors(bm, bm.load_calibration(), dtype=torch.float64), "g1", jnames)
  regex = r"^(left|right)_foot[1-7]_collision$"
  rest, ankle = robot_rest_geometry(model, bm.pelvis_link, bm.foot_links, regex)
  k = rest / float(np.asarray(h["human_rest_pelvis_height"]).reshape(-1)[0])

  # Reference robot trajectories: (root pos [T,3], root quat wxyz [T,4], joints [T,J] in model order, human frame offset)
  refs = {}
  r = np.load(pr["robot"])
  src = [str(x) for x in r["joint_names"]] if "joint_names" in r.files else list(FALLBACK_JOINT_NAMES[(r["joint_pos"].shape[1], r["body_pos_w"].shape[1])])
  refs["retarget"] = (r["body_pos_w"][:, 0].astype(np.float64), r["body_quat_w"][:, 0].astype(np.float64),
                      r["joint_pos"][:, [src.index(n) for n in jnames]].astype(np.float64), int(pr.get("lag", 0) or 0))
  isrc = [str(x) for x in h["init_g1_joint_names"]]
  xy = (hj[:, 0, :2] - hj[0, 0, :2]) * k
  refs["ik"] = (np.concatenate([xy, h["init_g1_root_height"][:, None]], 1), h["init_g1_root_quat"].astype(np.float64),
                h["init_g1_joint_pos"][:, [isrc.index(n) for n in jnames]].astype(np.float64), 0)

  # Bone errors of each reference against the human (TorchFK on this model).
  bodies = list(range(1, model.nbody))
  ent = types.SimpleNamespace(indexing=types.SimpleNamespace(body_ids=torch.tensor(bodies), joint_ids=torch.tensor(hinge)),
                              body_names=[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in bodies])
  links = [ent.body_names.index(n) for n in bm.robot_links]
  J = len(jnames)
  heads = HumanHeads(TorchFK(model, ent, "cpu", needed_bodies=links), links, torch.zeros(J), torch.tensor([[-9.0, 9.0]] * J),
                     HumanHeadsCfg(), rest, ankle)
  panels = [p for p in args.panels.split(",") if p]
  for name in panels:
    pos, quat, q, lag = refs[name]
    T = len(q)
    s = torch.zeros(T, 13 + 2 * J)
    s[:, :3], s[:, 3:7], s[:, 13 : 13 + J] = torch.as_tensor(pos), torch.as_tensor(quat), torch.as_tensor(q)
    idx = np.clip(np.arange(T) + lag, 0, len(hj) - 1)
    m = heads.metrics(heads.kin(s.float()), s.float(), {kk: torch.as_tensor(hf[kk][idx]) for kk in STEP_FIELDS})
    print(f"[{name:8s}] vs human: global torso {m['bone_global_torso_deg'].mean():.1f} arm {m['bone_global_arm_deg'].mean():.1f} "
          f"leg {m['bone_global_leg_deg'].mean():.1f} deg | heading {m['heading_err_deg'].mean():.1f} deg")

  T = min(len(refs[p][2]) for p in panels)
  if args.seconds > 0:
    T = min(T, int(args.seconds * 50))
  renderer = mujoco.Renderer(model, args.height, args.width)
  cam = mujoco.MjvCamera()
  cam.type = mujoco.mjtCamera.mjCAMERA_FREE
  frames = []
  setup = {}
  for name in panels:
    pos, quat, q, lag = refs[name]
    w, x, y, z = quat[0]
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    left = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    shift = np.array([pos[0, 0], pos[0, 1], 0.0]) - np.array([hj[lag, 0, 0], hj[lag, 0, 1], 0.0]) + args.side * left
    setup[name] = (yaw, left, shift)
  for t in range(T):
    row = []
    for name in panels:
      pos, quat, q, lag = refs[name]
      yaw, left, shift = setup[name]
      data.qpos[ra : ra + 3], data.qpos[ra + 3 : ra + 7], data.qpos[qadr] = pos[t], quat[t], q[t]
      mujoco.mj_kinematics(model, data)
      cam.lookat[:] = pos[t] + 0.5 * args.side * left
      cam.distance, cam.azimuth, cam.elevation = 4.5, float(np.degrees(yaw)) + 150.0, -15.0
      renderer.update_scene(data, cam)
      draw_human(renderer.scene, hj[min(t + lag, len(hj) - 1)] + shift)
      row.append(renderer.render().copy())
    frames.append(np.concatenate(row, axis=1))
  import imageio

  out = args.video or f"ref_{pr['name']}.mp4"
  imageio.mimwrite(out, frames, fps=50, quality=7, macro_block_size=1)
  print(f"wrote {out} ({T} frames, panels {panels})")


if __name__ == "__main__":
  main()
