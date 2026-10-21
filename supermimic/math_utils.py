"""Differentiable quaternion / frame helpers (torch, wxyz convention).

Where possible we reuse mjlab's torch implementations so that the analytic
reward and termination heads match the simulator-side managers bit for bit.
"""

from __future__ import annotations

import torch

from mjlab.utils.lab_api.math import (  # noqa: F401  (re-exported)
  axis_angle_from_quat,
  matrix_from_quat,
  quat_apply,
  quat_apply_inverse,
  quat_conjugate,
  quat_error_magnitude,
  quat_from_angle_axis,
  quat_inv,
  quat_mul,
  yaw_quat,
)


def quat_from_rotvec(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
  """Exponential map so(3) -> unit quaternion (w, x, y, z), Taylor-safe at 0."""
  theta2 = (v * v).sum(-1, keepdim=True)
  theta = torch.sqrt(theta2 + eps)
  half = 0.5 * theta
  # sin(half)/theta with series fallback near zero.
  small = theta2 < 1e-6
  k = torch.where(small, 0.5 - theta2 / 48.0, torch.sin(half) / theta)
  w = torch.where(small, 1.0 - theta2 / 8.0, torch.cos(half))
  return torch.cat([w, k * v], dim=-1)


def quat_normalize(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
  return q / (q.norm(dim=-1, keepdim=True) + eps)


def rot6d_from_quat(q: torch.Tensor) -> torch.Tensor:
  """First two columns of the rotation matrix, flattened to 6."""
  m = matrix_from_quat(q)
  return m[..., :, :2].reshape(*q.shape[:-1], 6)


def quat_log(q: torch.Tensor) -> torch.Tensor:
  """Rotation vector of a unit quaternion (wraps mjlab's axis_angle_from_quat)."""
  return axis_angle_from_quat(q)


def expand_quat(q: torch.Tensor, n: int) -> torch.Tensor:
  """[B,4] -> [B,n,4] (view, no copy)."""
  return q[:, None, :].expand(q.shape[0], n, 4)


def quat_apply_batched(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  """quat_apply for q [B,4] and v [B,N,3]."""
  B, N = v.shape[0], v.shape[1]
  return quat_apply(q[:, None, :].expand(B, N, 4).reshape(-1, 4), v.reshape(-1, 3)).view(
    B, N, 3
  )


def quat_mul_batched(q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
  """quat_mul for q [B,4] broadcast against r [B,N,4]."""
  B, N = r.shape[0], r.shape[1]
  return quat_mul(q[:, None, :].expand(B, N, 4).reshape(-1, 4), r.reshape(-1, 4)).view(
    B, N, 4
  )
