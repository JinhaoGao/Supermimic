"""Export side of the human-reference deploy path (scripts/human/export_deploy_human.py, tests/test_human_deploy.py).

  fk_table          kinematic tree of the bone-map links (+ ancestors) for deploy_goal.RobotKinematics
  robot_goal_meta   the robot / bone-map part of the meta json that deploy_goal.HumanGoalBuilder reads
  clip_reference    per-clip reference npz: every goal input that does not depend on the robot
  export_policy     ONNX of the policy (input normalization inside) + metadata, checked against torch
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch

from . import bone_map as bone
from .motions import derive_ee_fields, derive_fields

CLIP_FIELDS = ("h_feat", "h_yaw", "h_task_pos", "h_task_rot6d", "h_root_xy", "init_joint_pos", "init_root_quat", "init_root_height")


def fk_table(mj_model: mujoco.MjModel, body_ids, joint_ids, body_names, joint_names, needed: list[str]) -> dict:
  """Bodies `needed` (entity names) and their ancestors, parent-first from the free-joint root: name, parent (index in
  this list, -1 for the root), body pos / quat (wxyz) in the parent frame, hinge (policy joint name or None), axis,
  joint anchor, qpos0 -- what supermimic.fk.TorchFK reads from the model."""
  body_ids = [int(b) for b in body_ids]
  g2l = {g: i for i, g in enumerate(body_ids)}
  j2name = {int(g): n for g, n in zip(joint_ids, joint_names)}
  keep: set[int] = set()
  for n in needed:
    i = list(body_names).index(n)
    while i >= 0 and i not in keep:
      keep.add(i)
      i = g2l.get(int(mj_model.body_parentid[body_ids[i]]), -1)
  order = sorted(keep)  # entity body order is parent-first
  pos_in = {i: k for k, i in enumerate(order)}
  out = []
  for i in order:
    g = body_ids[i]
    jadr, jnum = int(mj_model.body_jntadr[g]), int(mj_model.body_jntnum[g])
    parent = pos_in.get(g2l.get(int(mj_model.body_parentid[g]), -1), -1)
    b = {"name": body_names[i], "parent": parent, "pos": mj_model.body_pos[g].tolist(), "quat": mj_model.body_quat[g].tolist(),
         "joint": None, "axis": [0.0, 0.0, 1.0], "jnt_pos": [0.0, 0.0, 0.0], "qpos0": 0.0}
    if jnum:
      assert jnum == 1, body_names[i]
      jt = int(mj_model.jnt_type[jadr])
      if jt == mujoco.mjtJoint.mjJNT_FREE:
        assert parent == -1, "free joint off the root"
      else:
        assert jt == mujoco.mjtJoint.mjJNT_HINGE, (body_names[i], jt)
        b.update(joint=j2name[jadr], axis=mj_model.jnt_axis[jadr].tolist(), jnt_pos=mj_model.jnt_pos[jadr].tolist(),
                 qpos0=float(mj_model.qpos0[int(mj_model.jnt_qposadr[jadr])]))
    out.append(b)
  assert out[0]["parent"] == -1 and sum(b["parent"] == -1 for b in out) == 1, "one root"
  return {"bodies": out}


def robot_goal_meta(mj_model, body_ids, joint_ids, body_names, joint_names, bone_map: bone.SmplBoneMapCfg, rest_height: float,
                    ankle_height: float) -> dict:
  needed = bone_map.robot_links
  return {
    "fk": fk_table(mj_model, body_ids, joint_ids, body_names, joint_names, needed),
    "heading_links": list(bone_map.heading_links),
    "foot_links": list(bone_map.foot_links),
    "task_links": [{"name": t.name, "points": list(t.robot_points), "frame": t.robot_frame} for t in bone_map.task_links],
    "robot_rest_height": float(rest_height),
    "robot_ankle_height": float(ankle_height),
  }


def clip_reference(path: str, bone_map: bone.SmplBoneMapCfg, robot: str, joint_names, bt=None) -> dict[str, np.ndarray]:
  """CLIP_FIELDS of one human npz (the same derive_fields / derive_ee_fields as the training cache), float32."""
  bt = bt or bone.BoneMapTensors(bone_map, bone_map.load_calibration(), dtype=torch.float64)
  with np.load(path, allow_pickle=True) as d:
    f = derive_fields(d, bt, robot, tuple(joint_names))
    f.update(derive_ee_fields(d, bt))
    fps = float(np.asarray(d["fps"]).reshape(-1)[0])
  out = {k: f[k] for k in CLIP_FIELDS}
  out["fps"] = np.float32(fps)
  return out


def export_policy(ck: dict, path: str, props: dict[str, str]) -> float:
  """Policy of checkpoint ck -> ONNX (obs_hist [1,hist,proprio], goal [1,K,goal_frame] -> action [1,nj]; input
  normalization and clamp in the graph), metadata props added. Returns the max |onnx - torch| on random inputs
  (nan when onnxruntime is missing)."""
  import onnx

  from ..policy import Policy

  d = ck["dims"]
  policy = Policy(d["proprio"], d["hist"], d["goal_frame"], d["K"], d["nj"], input_clip=float(ck["cfg"].get("input_clip", 10.0))).eval()
  policy.load_state_dict(ck["policy"])
  obs = torch.zeros(1, d["hist"], d["proprio"])
  goal = torch.zeros(1, d["K"], d["goal_frame"])
  torch.onnx.export(policy, (obs, goal), path, export_params=True, opset_version=18, input_names=["obs_hist", "goal"],
                    output_names=["action"], dynamic_axes={}, dynamo=False)
  model = onnx.load(path)
  for k, v in props.items():
    e = model.metadata_props.add()
    e.key, e.value = k, v
  onnx.save(model, path)
  try:
    import onnxruntime as ort
  except ImportError:
    return float("nan")
  sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
  g = torch.Generator().manual_seed(0)
  o = policy.obs_norm.mean + torch.randn(4, d["hist"], d["proprio"], generator=g) * policy.obs_norm.std
  gl = policy.goal_norm.mean + torch.randn(4, d["K"], d["goal_frame"], generator=g) * policy.goal_norm.std
  err = 0.0
  for i in range(4):
    with torch.no_grad():
      ref = policy(o[i : i + 1], gl[i : i + 1]).numpy()
    out = sess.run(None, {"obs_hist": o[i : i + 1].numpy(), "goal": gl[i : i + 1].numpy()})[0]
    err = max(err, float(np.abs(out - ref).max()))
  return err


def normalizer_stats(policy_state: dict) -> dict:
  """Running mean / std of the policy inputs (already inside the ONNX graph; for logging / other runtimes)."""
  out = {}
  for name in ("obs_norm", "goal_norm"):
    var = policy_state[f"{name}.var"].float()
    out[name] = {"mean": policy_state[f"{name}.mean"].float().tolist(), "std": torch.sqrt(var).clamp(min=1e-4).tolist()}
  return out

