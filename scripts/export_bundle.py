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
  print(f"[export] wrote {args.out}: scene.mjb, meta.json, policy.pt (iteration {ck['it']})")


if __name__ == "__main__":
  main()
