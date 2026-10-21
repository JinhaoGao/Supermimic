"""Batched, differentiable forward kinematics for an mjlab entity.

Reproduces MuJoCo's mj_kinematics for a floating-base tree of hinge joints
(free joint on the root body, at most one hinge per body, fixed bodies allowed)
and propagates link velocities so that the output matches
``EntityData.body_link_pose_w`` / ``EntityData.body_link_vel_w``.

The world model predicts a minimal state (root pose/velocity + joint
pos/vel); this module turns it into tracked-body poses and velocities for the
analytic reward / termination / goal heads.
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch

from .math_utils import quat_apply, quat_from_angle_axis, quat_mul


class TorchFK:
  def __init__(self, mj_model: mujoco.MjModel, entity, device: str, needed_bodies=None):
    """
    Args:
      mj_model: compiled scene model (``env.sim.mj_model``).
      entity: mjlab Entity (``env.scene["robot"]``).
      device: torch device.
      needed_bodies: iterable of local body indices (entity order) whose poses are
        required; their ancestors are added automatically. None = all bodies.
    """
    idx = entity.indexing
    body_ids = idx.body_ids.cpu().numpy().astype(np.int64)  # local -> global
    joint_ids = idx.joint_ids.cpu().numpy().astype(np.int64)  # local joint -> global
    gjoint2local = {int(g): i for i, g in enumerate(joint_ids)}
    gbody2local = {int(g): i for i, g in enumerate(body_ids)}
    n = len(body_ids)
    self.n = n
    self.device = device
    self.body_names = tuple(entity.body_names)

    parent = np.full(n, -1, dtype=np.int64)
    has_hinge = np.zeros(n, dtype=bool)
    joint_local = np.full(n, -1, dtype=np.int64)
    jnt_axis = np.zeros((n, 3), dtype=np.float32)
    jnt_pos = np.zeros((n, 3), dtype=np.float32)
    qpos0 = np.zeros(n, dtype=np.float32)
    body_pos = np.zeros((n, 3), dtype=np.float32)
    body_quat = np.zeros((n, 4), dtype=np.float32)
    self.root_local = None
    for i, g in enumerate(body_ids):
      pg = int(mj_model.body_parentid[g])
      parent[i] = gbody2local.get(pg, -1)
      body_pos[i] = mj_model.body_pos[g]
      body_quat[i] = mj_model.body_quat[g]
      jadr, jnum = int(mj_model.body_jntadr[g]), int(mj_model.body_jntnum[g])
      if jnum == 0:
        continue
      if jnum != 1:
        raise NotImplementedError(f"body {entity.body_names[i]} has {jnum} joints")
      jtype = int(mj_model.jnt_type[jadr])
      if jtype == mujoco.mjtJoint.mjJNT_FREE:
        assert parent[i] == -1, "free joint must be on the root body"
        self.root_local = i
        continue
      if jtype != mujoco.mjtJoint.mjJNT_HINGE:
        raise NotImplementedError(f"joint type {jtype} not supported")
      has_hinge[i] = True
      joint_local[i] = gjoint2local[jadr]
      jnt_axis[i] = mj_model.jnt_axis[jadr]
      jnt_pos[i] = mj_model.jnt_pos[jadr]
      qpos0[i] = mj_model.qpos0[int(mj_model.jnt_qposadr[jadr])]
    assert self.root_local is not None, "entity has no free joint"
    for i in range(n):
      if i != self.root_local:
        assert parent[i] >= 0 and parent[i] < i, "bodies must be parent-first ordered"

    # Which bodies to evaluate.
    if needed_bodies is None:
      needed = set(range(n))
    else:
      needed = set()
      stack = list(int(b) for b in needed_bodies)
      while stack:
        b = stack.pop()
        if b in needed:
          continue
        needed.add(b)
        if parent[b] >= 0:
          stack.append(int(parent[b]))
      needed.add(self.root_local)
    self.order = [i for i in range(n) if i in needed]

    self.parent = parent
    self.has_hinge = has_hinge
    self.joint_local = joint_local
    t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)  # noqa: E731
    self.jnt_axis = t(jnt_axis)
    self.jnt_pos = t(jnt_pos)
    self.qpos0 = t(qpos0)
    self.body_pos = t(body_pos)
    self.body_quat = t(body_quat)

  @torch.no_grad()
  def _zeros(self, B, k):
    return torch.zeros(B, k, device=self.device)

  def forward(
    self,
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    q: torch.Tensor,
    qd: torch.Tensor,
  ):
    """Returns pos [B,n,3], quat [B,n,4], lin_vel [B,n,3], ang_vel [B,n,3].

    Bodies not in ``self.order`` are filled with the root values (never read).
    """
    B = root_pos.shape[0]
    n = self.n
    pos = [None] * n
    quat = [None] * n
    lin = [None] * n
    ang = [None] * n
    r = self.root_local
    pos[r], quat[r], lin[r], ang[r] = root_pos, root_quat, root_lin_vel, root_ang_vel
    for i in self.order:
      if i == r:
        continue
      p = int(self.parent[i])
      xq0 = quat_mul(quat[p], self.body_quat[i].expand(B, 4))
      xp0 = pos[p] + quat_apply(quat[p], self.body_pos[i].expand(B, 3))
      if self.has_hinge[i]:
        j = int(self.joint_local[i])
        axis_l = self.jnt_axis[i].expand(B, 3)
        axis_w = quat_apply(xq0, axis_l)
        anchor = xp0 + quat_apply(xq0, self.jnt_pos[i].expand(B, 3))
        qloc = quat_from_angle_axis(q[:, j] - self.qpos0[i], axis_l)
        xq = quat_mul(xq0, qloc)
        xp = anchor - quat_apply(xq, self.jnt_pos[i].expand(B, 3))
        wj = axis_w * qd[:, j : j + 1]
        w = ang[p] + wj
        v = lin[p] + torch.cross(ang[p], xp - pos[p], dim=-1) + torch.cross(wj, xp - anchor, dim=-1)
      else:
        xq, xp = xq0, xp0
        w = ang[p]
        v = lin[p] + torch.cross(ang[p], xp - pos[p], dim=-1)
      pos[i], quat[i], lin[i], ang[i] = xp, xq, v, w
    for i in range(n):
      if pos[i] is None:
        pos[i], quat[i], lin[i], ang[i] = root_pos, root_quat, root_lin_vel, root_ang_vel
    return (
      torch.stack(pos, dim=1),
      torch.stack(quat, dim=1),
      torch.stack(lin, dim=1),
      torch.stack(ang, dim=1),
    )
