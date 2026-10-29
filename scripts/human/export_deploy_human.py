"""Deploy export of a human-reference SuperMimic policy (--reference human: H4, or ee_mode with a runtime alpha).

  cd ~/supermimic_ws/mjlab && PYTHONPATH=<repo> uv run python <repo>/scripts/human/export_deploy_human.py \
      --ckpt ~/supermimic_runs/logs/<run>/model_final.pt --motions ~/datasets/smpl_motion_pilot_50hz --out ~/deploy/<run>

Writes under --out:
  policy.onnx     obs_hist [1,hist,proprio] (oldest first; per frame q - default, dq, pelvis gyro, pelvis projected
                  gravity, previous action), goal [1,K,goal_frame] -> action [1,nj] (joint target = offset + scale x
                  action); input normalization inside the graph; metadata props (goal_layout, joint order, ...)
  meta.json       joint order, default pose, action scale / offset, PD gains (+ G1 SDK index map), K / hist / dims,
                  goal_layout, normalizer stats, heads cfg, and what deploy_goal.HumanGoalBuilder needs: the FK table
                  of the bone-map links, heading / foot / task links, robot ankle height
  clips/<name>.npz  per-clip reference side (deploy_export.CLIP_FIELDS: h_feat, h_yaw, h_task_pos, h_task_rot6d,
                  h_root_xy, IK reset pose init_*; fps): everything of the goal that does not depend on the robot
  deploy_goal.py  copy of supermimic/human/deploy_goal.py (numpy only) for the robot side
Usage on the robot: docs/ee_world_tracking_plan.md §5.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import tempfile

import mujoco
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)


def motion_files(args) -> list[str]:
  out = []
  for m in args.motions or []:
    m = os.path.expanduser(m)
    out += sorted(glob.glob(os.path.join(m, "**", "*.npz"), recursive=True)) if os.path.isdir(m) else [m]
  if args.motion_list:
    out += [os.path.expanduser(ln.strip()) for ln in open(os.path.expanduser(args.motion_list)) if ln.strip()]
  return out


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--ckpt", type=str, required=True)
  p.add_argument("--out", type=str, required=True)
  p.add_argument("--motions", type=str, nargs="*", default=None, help="human npz files or directories: reference clips to export")
  p.add_argument("--motion_list", type=str, default=None, help="txt with one human npz path per line")
  p.add_argument("--device", type=str, default="cpu")
  args = p.parse_args()

  from mjlab.envs import ManagerBasedRlEnv

  from supermimic.env_cfg import make_env_cfg
  from supermimic.heads import HeadsCfg
  from supermimic.human import bone_map as bone
  from supermimic.human.deploy_export import clip_reference, export_policy, normalizer_stats, robot_goal_meta
  from supermimic.human.motions import GOAL_FEAT_DIM

  ck = torch.load(os.path.expanduser(args.ckpt), map_location="cpu", weights_only=False)
  cfg = ck["cfg"]
  assert cfg.get("reference") == "human" and cfg["heads"].get("reference") == "human", "not a --reference human checkpoint"
  hc = HeadsCfg.from_dict({k: v for k, v in cfg["heads"].items() if k != "extra"})
  d = ck["dims"]
  files = motion_files(args)
  assert files, "no reference clips (--motions / --motion_list)"
  robot_name, variant = cfg.get("robot", "g1"), cfg.get("robot_variant", "")
  bm = bone.SMPL_BONE_MAPS[hc.bone_map]

  with tempfile.TemporaryDirectory() as tmp:  # one-env eval scene: the training robot model, action term, rest geometry
    env = ManagerBasedRlEnv(cfg=make_env_cfg([files[0]], 1, mode="eval", robot=robot_name, robot_variant=variant, cache_dir=tmp,
                                             ee_termination=False, reference="human", bone_map=hc.bone_map), device=args.device)
    robot = env.scene["robot"]
    cmd = env.command_manager.get_term("motion")
    term = env.action_manager.get_term("joint_pos")
    mj = env.sim.mj_model
    joint_names = [str(n) for n in robot.joint_names]
    body_names = [str(n) for n in robot.body_names]
    scale = term._scale[0] if torch.is_tensor(term._scale) else torch.full((len(term.target_names),), float(term._scale))
    offset = term._offset[0] if torch.is_tensor(term._offset) else torch.full((len(term.target_names),), float(term._offset))
    pd_kp, pd_kd = [], []
    for n in joint_names:
      jid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, f"robot/{n}")
      acts = [a for a in range(mj.nu) if int(mj.actuator_trnid[a, 0]) == jid]
      assert len(acts) == 1, f"joint {n}: {len(acts)} actuators"
      pd_kp.append(float(mj.actuator_gainprm[acts[0], 0]))
      pd_kd.append(float(-mj.actuator_biasprm[acts[0], 2]))
    rmeta = robot_goal_meta(mj, robot.indexing.body_ids.tolist(), robot.indexing.joint_ids.tolist(), body_names, joint_names, bm,
                            cmd.rest_height, cmd.ankle_height)
    meta = {
      "ckpt": os.path.abspath(os.path.expanduser(args.ckpt)),
      "iteration": ck["it"],
      "reference": "human",
      "robot": robot_name,
      "robot_variant": variant,
      "control_dt": float(env.step_dt),
      "joint_names": joint_names,
      "action_joint_names": [n.split("/")[-1] for n in term.target_names],
      "default_joint_pos": robot.data.default_joint_pos[0].cpu().tolist(),
      "action_scale": scale.cpu().tolist(),
      "action_offset": offset.cpu().tolist(),
      "pd_kp": pd_kp,
      "pd_kd": pd_kd,
    }
    env.close()
  assert meta["action_joint_names"] == joint_names, "action order must equal joint order"
  if robot_name == "g1":
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    from export_deploy import G1_SDK_JOINTS

    meta["sdk_joint_ids_map"] = [G1_SDK_JOINTS.index(n) for n in joint_names]  # policy index -> Unitree SDK motor index

  layout = [("h_feat", GOAL_FEAT_DIM)] + ([("rel_yaw", 2)] if hc.goal_rel_yaw else [])
  if hc.ee_mode:
    layout += [("ee", 9 * len(bm.task_links)), ("alpha", 1), ("ee_valid", 1)]
  assert sum(n for _, n in layout) == d["goal_frame"], (layout, d)
  nj = d["nj"]
  meta.update({
    "dims": d, "K": d["K"], "hist": d["hist"], "proprio": d["proprio"], "goal_frame": d["goal_frame"], "goal_layout": layout,
    "ee_mode": bool(hc.ee_mode), "input_clip": float(cfg.get("input_clip", 10.0)), "alpha_rate": 1.0,
    "obs_layout": f"[1,{d['hist']},{d['proprio']}] oldest->newest; per frame: joint_pos - default_joint_pos ({nj}), joint_vel ({nj}), "
                  f"pelvis angular velocity in the pelvis frame (3), gravity in the pelvis frame (3), previous action ({nj})",
    "goal_doc": "per future frame t..t+K-1 (clamped at the clip end): h_feat (138, clip npz) | cos, sin(h_yaw - robot hip-line yaw) "
                + ("| 5 task links (torso, wrist_l, wrist_r, foot_l, foot_r) in the robot torso_link frame: pos 3 + rot6d 6 (x ee_valid) "
                   "| alpha | ee_valid" if hc.ee_mode else "") + "; built by deploy_goal.HumanGoalBuilder",
    "normalizers": normalizer_stats(ck["policy"]),
    "heads": {k: v for k, v in cfg["heads"].items() if k != "extra"},
    **rmeta,
  })

  out = os.path.expanduser(args.out)
  os.makedirs(os.path.join(out, "clips"), exist_ok=True)
  props = {"reference": "human", "iteration": str(ck["it"]), "control_dt": str(meta["control_dt"]), "joint_names": json.dumps(joint_names),
           "default_joint_pos": json.dumps(meta["default_joint_pos"]), "action_scale": json.dumps(meta["action_scale"]),
           "action_offset": json.dumps(meta["action_offset"]), "goal_layout": json.dumps(layout), "ee_mode": str(int(hc.ee_mode)),
           "layout": json.dumps({"obs_hist": meta["obs_layout"], "goal": meta["goal_doc"],
                                 "action": f"[1,{nj}] joint target = action_offset + action_scale * action"})}
  err = export_policy(ck, os.path.join(out, "policy.onnx"), props)
  print(f"[export_deploy_human] policy.onnx: onnx vs torch max abs diff {err:.2e}")

  bt = bone.BoneMapTensors(bm, bm.load_calibration(), dtype=torch.float64)
  clips, bad = [], []
  for f in files:
    try:
      c = clip_reference(f, bm, robot_name, joint_names, bt)
    except Exception as e:  # noqa: BLE001 -- a clip without the robot's IK reset fields is skipped, not fatal
      bad.append((f, repr(e)))
      continue
    name = os.path.splitext(os.path.basename(f))[0]
    np.savez(os.path.join(out, "clips", f"{name}.npz"), **c)
    clips.append({"name": name, "frames": int(c["h_feat"].shape[0]), "fps": float(c["fps"])})
  if bad:
    print(f"[export_deploy_human] WARNING: {len(bad)} clips skipped (first: {bad[:2]})")
  meta["clips"] = clips
  with open(os.path.join(out, "meta.json"), "w") as fh:
    json.dump(meta, fh, indent=1)
  shutil.copy(os.path.join(REPO, "supermimic", "human", "deploy_goal.py"), os.path.join(out, "deploy_goal.py"))
  print(f"[export_deploy_human] {out}: policy.onnx, meta.json, deploy_goal.py, {len(clips)} clips "
        f"(goal {d['K']}x{d['goal_frame']}, {'ee_mode' if hc.ee_mode else 'H4'}, iteration {ck['it']})")


if __name__ == "__main__":
  main()
