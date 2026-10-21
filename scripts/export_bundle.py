"""Export a self-contained playback bundle from a SuperMimic checkpoint.

Needs mjlab once (to compile the exact scene model used in training and read the
action mapping); afterwards ``scripts/play_mujoco.py`` only needs mujoco + torch.

  uv run python scripts/export_bundle.py --ckpt ~/supermimic/ckpt/model_16000.pt \
      --motion ~/supermimic/data/lafan1/g1__lafan1_dance1_subject1.npz --out ~/supermimic/bundle_16000

Bundle contents: scene.mjb (compiled MuJoCo model incl. floor, actuators, PD gains),
meta.json (joint order, action->joint mapping + scale/offset, tracked bodies, heads
config, network dims), policy.pt (policy state dict incl. input normalizers).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess

import mujoco
import torch


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--ckpt", type=str, required=True)
  p.add_argument("--motion", type=str, required=True, help="any one motion npz (only needed to build the env)")
  p.add_argument("--out", type=str, required=True)
  p.add_argument("--device", type=str, default="cpu")
  args = p.parse_args()

  from mjlab.envs import ManagerBasedRlEnv

  from supermimic.env_cfg import make_env_cfg

  ck = torch.load(args.ckpt, map_location="cpu")
  env = ManagerBasedRlEnv(cfg=make_env_cfg([args.motion], 1, mode="eval"), device=args.device)
  robot = env.scene["robot"]
  cmd = env.command_manager.get_term("motion")
  term = env.action_manager.get_term("joint_pos")
  os.makedirs(args.out, exist_ok=True)
  mujoco.mj_saveModel(env.sim.mj_model, os.path.join(args.out, "scene.mjb"), None)
  try:
    with open(os.path.join(args.out, "scene.xml"), "w") as f:
      f.write(env.scene.spec.to_xml())
  except Exception as e:  # noqa: BLE001
    print(f"[export] scene.xml not written ({e!r}); scene.mjb is what play_mujoco.py uses")
  scale = term._scale[0] if torch.is_tensor(term._scale) else torch.full((len(term.target_names),), float(term._scale))
  offset = term._offset[0] if torch.is_tensor(term._offset) else torch.full((len(term.target_names),), float(term._offset))
  try:
    mjlab_commit = subprocess.check_output(["git", "-C", os.path.dirname(__import__("mjlab").__file__), "rev-parse", "HEAD"], text=True).strip()
  except Exception:  # noqa: BLE001
    mjlab_commit = "unknown"
  meta = {
    "ckpt": os.path.abspath(args.ckpt),
    "iteration": ck["it"],
    "mjlab_commit": mjlab_commit,
    "control_dt": env.step_dt,
    "decimation": env.cfg.decimation,
    "joint_names": list(robot.joint_names),
    "default_joint_pos": robot.data.default_joint_pos[0].cpu().tolist(),
    "soft_joint_limits": robot.data.soft_joint_pos_limits[0].cpu().tolist(),
    "action_joint_names": [n.split("/")[-1] for n in term.target_names],
    "action_scale": scale.cpu().tolist(),
    "action_offset": offset.cpu().tolist(),
    "body_names": list(cmd.cfg.body_names),
    "anchor_body_name": cmd.cfg.anchor_body_name,
    "root_body_name": robot.body_names[0],
    "heads": {k: v for k, v in ck["cfg"]["heads"].items() if k != "extra"},
    "dims": ck["dims"],
  }
  with open(os.path.join(args.out, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
  torch.save(ck["policy"], os.path.join(args.out, "policy.pt"))
  env.close()
  export_onnx(ck, meta, os.path.join(args.out, "policy.onnx"))
  print(f"[export] wrote {args.out}: scene.mjb, meta.json, policy.pt, policy.onnx (iteration {ck['it']})")


def export_onnx(ck: dict, meta: dict, path: str) -> None:
  """Policy -> ONNX. Inputs: obs_hist [1, hist, proprio] (oldest first), goal [1, K, 67].
  Output: action [1, nj] (normalized joint setpoint; target = offset + scale * action).
  Input normalization is part of the graph. Layout documented in the ONNX metadata."""
  import onnx

  from supermimic.policy import Policy

  d = meta["dims"]
  policy = Policy(d["proprio"], d["hist"], d["goal_frame"], d["K"], d["nj"]).eval()
  policy.load_state_dict(ck["policy"])
  obs = torch.zeros(1, d["hist"], d["proprio"])
  goal = torch.zeros(1, d["K"], d["goal_frame"])
  torch.onnx.export(
    policy, (obs, goal), path, export_params=True, opset_version=18, input_names=["obs_hist", "goal"],
    output_names=["action"], dynamic_axes={}, dynamo=False,
  )
  nj = d["nj"]
  layout = {
    "obs_hist": f"[1,{d['hist']},{d['proprio']}] oldest->newest; per frame: joint_pos-default({nj}), joint_vel({nj}), "
    f"base_ang_vel_pelvis(3), projected_gravity_pelvis(3), prev_action({nj}); joints in joint_names order",
    "goal": f"[1,{d['K']},{d['goal_frame']}] future reference frames t..t+K-1 (clamped at clip end); per frame: "
    f"ref_joint_pos({nj}), ref_joint_vel({nj}), ref_anchor_pos_in_robot_anchor_frame(3), ref_anchor_rot6d_in_robot_anchor_frame(6)",
    "action": f"[1,{nj}] in action_joint_names order; joint target = action_offset + action_scale * action",
  }
  model = onnx.load(path)
  props = {
    "iteration": str(meta["iteration"]),
    "control_dt": str(meta["control_dt"]),
    "joint_names": json.dumps(meta["joint_names"]),
    "default_joint_pos": json.dumps(meta["default_joint_pos"]),
    "action_joint_names": json.dumps(meta["action_joint_names"]),
    "action_scale": json.dumps(meta["action_scale"]),
    "action_offset": json.dumps(meta["action_offset"]),
    "body_names": json.dumps(meta["body_names"]),
    "anchor_body_name": meta["anchor_body_name"],
    "layout": json.dumps(layout),
  }
  for k, v in props.items():
    e = model.metadata_props.add()
    e.key, e.value = k, v
  onnx.save(model, path)
  # Verify against torch.
  try:
    import onnxruntime as ort

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    torch.manual_seed(0)
    o = torch.randn(1, d["hist"], d["proprio"]) * 0.5
    g = torch.randn(1, d["K"], d["goal_frame"]) * 0.5
    with torch.no_grad():
      ref = policy(o, g).numpy()
    out = sess.run(None, {"obs_hist": o.numpy(), "goal": g.numpy()})[0]
    print(f"[export] onnx vs torch max abs diff: {abs(out - ref).max():.2e}")
  except Exception as e:  # noqa: BLE001
    print(f"[export] onnxruntime check skipped: {e!r}")


if __name__ == "__main__":
  main()
