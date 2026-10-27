"""Helpers of the offline SMPL scripts (ported from Noetix-RGMT scripts/smpl_motion_utils.py): SMPL-X forward
kinematics and the bone map, imported by file path (supermimic/human/bone_map.py, torch only), so the scripts run in
any env with torch / numpy / scipy (+ smplx for prepare_smpl_motion.py, + mujoco for prepare_init_states.py)."""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_bone_map_module():
  path = os.path.join(REPO_ROOT, "supermimic", "human", "bone_map.py")
  spec = importlib.util.spec_from_file_location("smpl_bone_map", path)
  module = importlib.util.module_from_spec(spec)
  sys.modules["smpl_bone_map"] = module  # dataclasses look the module up while the class is created
  spec.loader.exec_module(module)
  return module


bone = load_bone_map_module()

# Y-up SMPL canonical frame -> Z-up world (AMASS stageii data is already Z-up).
Y_UP_TO_Z_UP = R.from_euler("x", 90, degrees=True)


class SmplxBody:
  """Thin batched wrapper around an SMPL-X model for one subject (gender + betas)."""

  def __init__(self, model_dir: str, gender: str = "neutral", betas: np.ndarray | None = None, device: str = "cpu"):
    gender = gender if gender in ("male", "female", "neutral") else "neutral"
    import smplx

    num_betas = 16 if betas is None else int(len(betas))
    self.model = smplx.create(
      model_dir,
      model_type="smplx",
      gender=gender,
      num_betas=num_betas,
      use_pca=False,
      flat_hand_mean=True,
    ).to(device)
    self.gender = gender
    self.device = device
    self.betas = np.zeros(num_betas, dtype=np.float32) if betas is None else np.asarray(betas, dtype=np.float32)

  @torch.no_grad()
  def joints(
    self,
    root_orient: np.ndarray,
    pose_body: np.ndarray,
    trans: np.ndarray,
    pose_hand: np.ndarray | None = None,
    batch_size: int = 512,
  ) -> np.ndarray:
    """World joints (T, 24, 3): 22 body joints + left/right hand points."""
    n = root_orient.shape[0]
    out = []
    for start in range(0, n, batch_size):
      sl = slice(start, min(n, start + batch_size))
      b = sl.stop - sl.start
      t = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float32), device=self.device)  # noqa: E731
      hands = np.zeros((b, 90), np.float32) if pose_hand is None else pose_hand[sl]
      res = self.model(
        betas=t(np.repeat(self.betas[None], b, axis=0)),
        global_orient=t(root_orient[sl]),
        body_pose=t(pose_body[sl]),
        transl=t(trans[sl]),
        left_hand_pose=t(hands[:, :45]),
        right_hand_pose=t(hands[:, 45:]),
        jaw_pose=torch.zeros(b, 3, device=self.device),
        leye_pose=torch.zeros(b, 3, device=self.device),
        reye_pose=torch.zeros(b, 3, device=self.device),
        expression=torch.zeros(b, self.model.num_expression_coeffs, device=self.device),
      )
      j = res.joints.cpu().numpy()
      out.append(np.concatenate([j[:, :22], j[:, list(bone.SMPLX_HAND_POINT_INDICES)]], axis=1))
    return np.concatenate(out, axis=0)

  def tpose_joints(self) -> np.ndarray:
    """Z-up T-pose joints (24, 3) with the lowest foot joint on z = 0."""
    root = Y_UP_TO_Z_UP.as_rotvec()[None].astype(np.float32)
    j = self.joints(root, np.zeros((1, 63), np.float32), np.zeros((1, 3), np.float32))[0]
    j[:, 2] -= j[list(bone.SMPL_FOOT_JOINTS), 2].min()
    return j


def global_rotations(root_orient: np.ndarray, pose_body: np.ndarray) -> np.ndarray:
  """Per-joint world rotations (T, 22) as scipy matrices array (T, 22, 3, 3)."""
  n = root_orient.shape[0]
  local = np.concatenate([root_orient[:, None], pose_body.reshape(n, 21, 3)], axis=1)
  local_m = R.from_rotvec(local.reshape(-1, 3)).as_matrix().reshape(n, 22, 3, 3)
  glob = np.empty_like(local_m)
  for j, p in enumerate(bone.SMPL_PARENTS):
    glob[:, j] = local_m[:, j] if p < 0 else glob[:, p] @ local_m[:, j]
  return glob
