"""Batched, differentiable, level-wise forward kinematics for an mjlab entity.

Reproduces MuJoCo's mj_kinematics for a floating-base tree of hinge joints
(free joint on the root body, at most one hinge per body, fixed bodies allowed)
and propagates link velocities so that the output matches
``EntityData.body_link_pose_w`` / ``EntityData.body_link_vel_w``.

Bodies at the same tree depth are processed together (one batched op chain
per level) so the whole FK is ~10 fused op groups instead of a per-body loop,
which matters because policy learning backpropagates through it 32 times per
imagined rollout. Fixed bodies are treated as hinges with angle = rate = 0.
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch

from .math_utils import quat_apply, quat_from_angle_axis, quat_mul


class TorchFK:
  def __init__(self, mj_model: mujoco.MjModel, entity, device: str, needed_bodies=None):
    idx = entity.indexing
    body_ids = idx.body_ids.cpu().numpy().astype(np.int64)  # local -> global
    joint_ids = idx.joint_ids.cpu().numpy().astype(np.int64)
    gjoint2local = {int(g): i for i, g in enumerate(joint_ids)}
    gbody2local = {int(g): i for i, g in enumerate(body_ids)}
    n = len(body_ids)
    self.n = n
    self.device = device
    self.body_names = tuple(entity.body_names)

    parent = np.full(n, -1, dtype=np.int64)
    has_hinge = np.zeros(n, dtype=bool)
    joint_local = np.zeros(n, dtype=np.int64)
    jnt_axis = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (n, 1))
    jnt_pos = np.zeros((n, 3), dtype=np.float32)
    qpos0 = np.zeros(n, dtype=np.float32)
    body_pos = np.zeros((n, 3), dtype=np.float32)
    body_quat = np.zeros((n, 4), dtype=np.float32)
    root = None
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
        root = i
        continue
      if jtype != mujoco.mjtJoint.mjJNT_HINGE:
        raise NotImplementedError(f"joint type {jtype} not supported")
      has_hinge[i] = True
      joint_local[i] = gjoint2local[jadr]
      jnt_axis[i] = mj_model.jnt_axis[jadr]
      jnt_pos[i] = mj_model.jnt_pos[jadr]
      qpos0[i] = mj_model.qpos0[int(mj_model.jnt_qposadr[jadr])]
    assert root is not None, "entity has no free joint"
    self.root_local = root
    for i in range(n):
      if i != root:
        assert 0 <= parent[i] < i, "bodies must be parent-first ordered"

    # Needed bodies (+ ancestors).
    if needed_bodies is None:
      needed = set(range(n))
    else:
      needed = set()
      stack = [int(b) for b in needed_bodies]
      while stack:
        b = stack.pop()
        if b in needed:
          continue
        needed.add(b)
        if parent[b] >= 0:
          stack.append(int(parent[b]))
    needed.add(root)

    # Depth levels.
    depth = np.zeros(n, dtype=np.int64)
    for i in range(n):
      depth[i] = 0 if parent[i] < 0 else depth[parent[i]] + 1
    max_depth = int(depth.max())
    done_order = [root]
    done_index = {root: 0}
    self.levels = []
    t = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt, device=device)  # noqa: E731
    for d in range(1, max_depth + 1):
      bodies = [i for i in range(n) if depth[i] == d and i in needed]
      if not bodies:
        continue
      lvl = dict(
        parent_done=t([done_index[int(parent[i])] for i in bodies], torch.long),
        body_pos=t(body_pos[bodies]),
        body_quat=t(body_quat[bodies]),
        jnt_axis=t(jnt_axis[bodies]),
        jnt_pos=t(jnt_pos[bodies]),
        qpos0=t(qpos0[bodies]),
        joint_local=t(joint_local[bodies], torch.long),
        hinge=t(has_hinge[bodies].astype(np.float32)),
      )
      for i in bodies:
        done_index[i] = len(done_order)
        done_order.append(i)
      self.levels.append(lvl)
    self.n_done = len(done_order)
    # Map local body id -> position in done order (un-needed bodies -> root).
    self.local_to_done = t([done_index.get(i, 0) for i in range(n)], torch.long)
    self.parent = parent

  def forward(
    self,
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    q: torch.Tensor,
    qd: torch.Tensor,
  ):
    """Returns pos [B,n,3], quat [B,n,4], lin_vel [B,n,3], ang_vel [B,n,3] in
    entity-local body order (bodies that were not requested hold root values)."""
    B = root_pos.shape[0]
    P = root_pos[:, None]
    Q = root_quat[:, None]
    V = root_lin_vel[:, None]
    W = root_ang_vel[:, None]
    for lvl in self.levels:
      pd = lvl["parent_done"]
      L = pd.shape[0]
      pp, pq, pv, pw = P[:, pd], Q[:, pd], V[:, pd], W[:, pd]
      bq = lvl["body_quat"].expand(B, L, 4)
      bp = lvl["body_pos"].expand(B, L, 3)
      axis = lvl["jnt_axis"].expand(B, L, 3)
      jpos = lvl["jnt_pos"].expand(B, L, 3)
      xq0 = quat_mul(pq, bq)
      xp0 = pp + quat_apply(pq, bp)
      ang = (q[:, lvl["joint_local"]] - lvl["qpos0"]) * lvl["hinge"]
      rate = qd[:, lvl["joint_local"]] * lvl["hinge"]
      qloc = quat_from_angle_axis(ang, axis)
      xq = quat_mul(xq0, qloc)
      axis_w = quat_apply(xq0, axis)
      anchor = xp0 + quat_apply(xq0, jpos)
      xp = anchor - quat_apply(xq, jpos)
      wj = axis_w * rate[..., None]
      w = pw + wj
      v = pv + torch.cross(pw, xp - pp, dim=-1) + torch.cross(wj, xp - anchor, dim=-1)
      P = torch.cat([P, xp], dim=1)
      Q = torch.cat([Q, xq], dim=1)
      V = torch.cat([V, v], dim=1)
      W = torch.cat([W, w], dim=1)
    m = self.local_to_done
    return P[:, m], Q[:, m], V[:, m], W[:, m]
