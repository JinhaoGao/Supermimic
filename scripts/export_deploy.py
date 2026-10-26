"""Export a SuperMimic bundle to the unitree_mujoco G1 deploy layout (State_SuperMimic; goal_layout ref_root: State_SuperMimicV2).

  python scripts/export_deploy.py --bundle ~/supermimic/bundle_g1_atlas_latest \
      --deploy_dir ~/Workspace/unitree_mujoco/deploy/robots/g1_29dof \


Writes under <deploy_dir>/config/policy/gmt/supermimic/:
  exported/supermimic_g1_<it>.onnx  (copied; metadata already inside; --onnx_name to override)
  params/deploy.yaml                (joint_ids_map, default pose, kp/kd in SDK order, action scale/offset,
                                     obs groups obs_hist / goal, root->torso chain, Isaac BFS joint permutation)
Reference clips need no conversion: State_SuperMimic reads standard GMT / MotionAtlas npz directly.
Policies whose goal carries the reference root motion (goal_layout q,qd,rot6d,ref_root, 2nj+16 per frame) go to
config/policy/gmt/supermimic_v2/ for the separate FSM state SuperMimicV2 (observations supermimic_v2_*, chain with
body_pos for the reference anchor height); the plain export (State_SuperMimic) is unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import numpy as np

# Unitree G1 29-DoF SDK motor order (lowstate motor_state index).
G1_SDK_JOINTS = [
  "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
  "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
  "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
  "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
# Isaac Lab BFS order: what GMR-CPP publishes on the online Redis stream (and our atlas npz joint_names).
G1_ISAAC_BFS_JOINTS = [
  "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint", "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
  "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint", "left_knee_joint", "right_knee_joint", "left_shoulder_pitch_joint",
  "right_shoulder_pitch_joint", "left_ankle_pitch_joint", "right_ankle_pitch_joint", "left_shoulder_roll_joint", "right_shoulder_roll_joint",
  "left_ankle_roll_joint", "right_ankle_roll_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint", "left_elbow_joint", "right_elbow_joint",
  "left_wrist_roll_joint", "right_wrist_roll_joint", "left_wrist_pitch_joint", "right_wrist_pitch_joint", "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]


def fmt_list(xs, nd=6):
  return "[" + ", ".join(f"{float(x):.{nd}f}" for x in xs) + "]"


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--bundle", type=str, required=True)
  p.add_argument("--deploy_dir", type=str, required=True, help=".../deploy/robots/g1_29dof")
  p.add_argument("--name", type=str, default=None, help="config/policy/gmt/<name>/ (default supermimic, ref_root goals supermimic_v2)")
  p.add_argument("--onnx_name", type=str, default=None, help="file name under exported/ (default supermimic_<robot>_<iteration>.onnx; "
                 "the deploy repo's .gitignore keeps *.onnx out of git except exported/policy.onnx)")
  args = p.parse_args()
  bundle = os.path.expanduser(args.bundle)
  meta = json.load(open(os.path.join(bundle, "meta.json")))
  assert meta.get("robot", "g1") == "g1", "this exporter targets the G1 deploy (SDK order table is G1's)"
  jn = meta["joint_names"]
  nj = len(jn)
  d = meta["dims"]
  ref_root = bool(meta.get("heads", {}).get("goal_ref_root", False))
  assert d["goal_frame"] == 2 * nj + 6 + (10 if ref_root else 0), \
    f"deployable policy needs goal without anchor position (goal_frame={d['goal_frame']}, ref_root={ref_root})"
  name = args.name or ("supermimic_v2" if ref_root else "supermimic")
  obs_prefix = "supermimic_v2" if ref_root else "supermimic"
  assert meta["action_joint_names"] == jn, "action order must equal joint order"
  sdk_index = {n: i for i, n in enumerate(G1_SDK_JOINTS)}
  joint_ids_map = [sdk_index[n] for n in jn]
  kp_sdk = [0.0] * len(G1_SDK_JOINTS)
  kd_sdk = [0.0] * len(G1_SDK_JOINTS)
  for i, n in enumerate(jn):
    kp_sdk[sdk_index[n]] = meta["pd_kp"][i]
    kd_sdk[sdk_index[n]] = meta["pd_kd"][i]
  isaac_perm = [G1_ISAAC_BFS_JOINTS.index(n) for n in jn]  # policy index -> Isaac BFS index (online stream, npz without joint_names)
  chain = meta["anchor_chain"]
  if ref_root:
    assert all("body_pos" in l for l in chain), "ref_root goals need the chain body_pos (re-export the bundle with export_bundle.py)"

  root = os.path.join(os.path.expanduser(args.deploy_dir), "config", "policy", "gmt", name)
  os.makedirs(os.path.join(root, "exported"), exist_ok=True)
  os.makedirs(os.path.join(root, "params"), exist_ok=True)
  onnx_name = args.onnx_name or f"{'supermimic_v2' if ref_root else 'supermimic'}_{meta.get('robot', 'g1')}_{meta['iteration']}.onnx"
  shutil.copy(os.path.join(bundle, "policy.onnx"), os.path.join(root, "exported", onnx_name))

  y = []
  y.append(f"# SuperMimic deploy config, generated by supermimic/scripts/export_deploy.py from {bundle} (iteration {meta['iteration']}).")
  y.append("# Policy joint order = mjlab/MuJoCo order (== Unitree SDK order for G1). stiffness/damping are SDK order; other vectors policy order.")
  y.append(f"# ONNX: obs_hist [1,{d['hist']},{d['proprio']}] (oldest->newest; per frame q-default, dq, pelvis gyro, projected gravity, prev action),")
  y.append(f"#       goal [1,{d['K']},{d['goal_frame']}] (per future frame: ref q, ref dq, rot6d(R_robot_anchor^T R_ref_anchor)"
           + (", ref anchor height, ref anchor lin / ang vel in the ref anchor frame, ref anchor projected gravity" if ref_root else "")
           + f"), action [1,{nj}].")
  y.append(f"joint_ids_map: {joint_ids_map}")
  y.append(f"joint_names: [{', '.join(jn)}]")
  y.append(f"step_dt: {meta['control_dt']}")
  y.append(f"stiffness: {fmt_list(kp_sdk)}")
  y.append(f"damping: {fmt_list(kd_sdk)}")
  y.append(f"default_joint_pos: {fmt_list(meta['default_joint_pos'])}")
  y.append("commands: {}")
  y.append("")
  y.append("actions:")
  y.append("  JointPositionAction:")
  y.append("    clip_actions: null               # training does not clip the network output")
  y.append("    clip: null                       # PD targets not clamped (training does not)")
  y.append("    joint_names: [.*]")
  y.append(f"    scale: {fmt_list(meta['action_scale'])}")
  y.append(f"    offset: {fmt_list(meta['action_offset'])}")
  y.append("    joint_ids: null")
  y.append("")
  y.append("observations:")
  y.append("  scale_first: false")
  y.append("  obs_hist:")
  y.append(f"    {obs_prefix + '_obs_hist:':33s}# {d['hist']}x{d['proprio']} frame-major, oldest first")
  y.append("      params: {}")
  y.append("      clip: null")
  y.append("      scale: null")
  y.append("      history_length: 1")
  y.append("  goal:")
  y.append(f"    {obs_prefix + '_goal:':33s}# {d['K']}x{d['goal_frame']} future reference frames")
  y.append("      params: {}")
  y.append("      clip: null")
  y.append("      scale: null")
  y.append("      history_length: 1")
  y.append("")
  y.append("use_gym_history: false")
  y.append("")
  y.append("supermimic:")
  y.append(f"  K: {d['K']}")
  y.append(f"  hist: {d['hist']}")
  y.append(f"  anchor_body_name: {meta['anchor_body_name']}")
  y.append(f"  root_body_name: {meta['root_body_name']}")
  if ref_root:
    y.append("  goal_layout: q,qd,rot6d,ref_root      # + ref anchor height (1), lin vel (3), ang vel (3), projected gravity (3), ref anchor frame")
  y.append(f"  isaac_bfs_joint_perm: {isaac_perm}   # policy index -> Isaac BFS index (online stream; npz without joint_names)")
  y.append("  anchor_chain:                    # root -> anchor: body_quat (wxyz) offsets and hinge joints")
  for link in chain:
    pos = f"body_pos: {fmt_list(link['body_pos'])}, " if ref_root else ""
    y.append(f"    - {{body: {link['body']}, {pos}body_quat: {fmt_list(link['body_quat'])}, joint: {link['joint'] if link['joint'] else 'null'}, axis: {fmt_list(link['axis'], 4)}, qpos0: {link['qpos0']:.6f}}}")
  with open(os.path.join(root, "params", "deploy.yaml"), "w") as f:
    f.write("\n".join(y) + "\n")

  print(f"[export_deploy] wrote {root}/exported/{onnx_name} and params/deploy.yaml (joint_ids_map={joint_ids_map})")
  print(f"[export_deploy] set `policy_file: {onnx_name}` in the {'SuperMimicV2' if ref_root else 'SuperMimic'} block of config.yaml")


if __name__ == "__main__":
  main()
