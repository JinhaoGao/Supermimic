"""Differentiable quaternion / frame helpers (plain torch, wxyz convention).

Plain-torch re-implementations of the mjlab ``lab_api.math`` functions we need,
so that ``torch.compile`` can trace through them (mjlab's versions are
TorchScript). Semantics match mjlab (verified by scripts/check_fk.py).
"""

from __future__ import annotations

import torch


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
  w1, x1, y1, z1 = q1.unbind(-1)
  w2, x2, y2, z2 = q2.unbind(-1)
  return torch.stack(
    [
      w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
      w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
      w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
      w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ],
    dim=-1,
  )


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
  return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


quat_inv = quat_conjugate  # unit quaternions


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  """Rotate v by q (both (..., 4)/(..., 3), broadcastable)."""
  qv = q[..., 1:]
  w = q[..., :1]
  t = 2.0 * torch.cross(qv, v, dim=-1)
  return v + w * t + torch.cross(qv, t, dim=-1)


def quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  return quat_apply(quat_conjugate(q), v)


def quat_normalize(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
  return q / (q.norm(dim=-1, keepdim=True) + eps)


def yaw_quat(q: torch.Tensor) -> torch.Tensor:
  w, x, y, z = q.unbind(-1)
  yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
  half = 0.5 * yaw
  zeros = torch.zeros_like(half)
  return torch.stack([torch.cos(half), zeros, zeros, torch.sin(half)], dim=-1)


def axis_angle_from_quat(q: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
  """Rotation vector of a unit quaternion (mjlab semantics, NaN-safe gradients)."""
  q = q * (1.0 - 2.0 * (q[..., :1] < 0.0).to(q.dtype))
  mag = torch.linalg.norm(q[..., 1:], dim=-1)
  half = torch.atan2(mag, q[..., 0])
  angle = 2.0 * half
  big = angle.abs() > eps
  angle_safe = torch.where(big, angle, torch.ones_like(angle))
  ratio = torch.where(big, torch.sin(half) / angle_safe, 0.5 - angle * angle / 48.0)
  return q[..., 1:] / ratio[..., None]


def quat_error_magnitude(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
  return torch.linalg.norm(axis_angle_from_quat(quat_mul(q1, quat_conjugate(q2))), dim=-1)


def quat_angle(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
  """Full rotation angle in [0, pi] between q1 and q2 (yaw included): 2 atan2(|v|, |w|) of q1 q2^*.

  The team eval site's ``quat_error_angle`` (evaluate_mujoco.py: 2 arccos(|w|) of q_ref q_robot^-1, both normalized)
  in a form that needs no normalization (atan2 is scale-invariant: the world model's / the motion files' quaternions
  are unit only up to float error) and has finite gradients everywhere (0 at angle 0, where arccos' is infinite)."""
  dq = quat_mul(q1, quat_conjugate(q2))
  return 2.0 * torch.atan2(torch.linalg.vector_norm(dq[..., 1:], dim=-1), dq[..., 0].abs())


def matrix_from_quat(q: torch.Tensor) -> torch.Tensor:
  w, x, y, z = q.unbind(-1)
  two_s = 2.0 / (q * q).sum(-1)
  o = torch.stack(
    [
      1 - two_s * (y * y + z * z), two_s * (x * y - z * w), two_s * (x * z + y * w),
      two_s * (x * y + z * w), 1 - two_s * (x * x + z * z), two_s * (y * z - x * w),
      two_s * (x * z - y * w), two_s * (y * z + x * w), 1 - two_s * (x * x + y * y),
    ],
    dim=-1,
  )
  return o.reshape(q.shape[:-1] + (3, 3))


def rot6d_from_quat(q: torch.Tensor) -> torch.Tensor:
  """First two columns of the rotation matrix, flattened to 6."""
  m = matrix_from_quat(q)
  return m[..., :, :2].reshape(*q.shape[:-1], 6)


def quat_from_angle_axis(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
  """angle (...,), unit axis (..., 3) -> (w, x, y, z)."""
  half = 0.5 * angle
  return torch.cat([torch.cos(half)[..., None], torch.sin(half)[..., None] * axis], dim=-1)


def quat_from_rotvec(v: torch.Tensor) -> torch.Tensor:
  """Exponential map so(3) -> unit quaternion, Taylor-safe at 0."""
  theta2 = (v * v).sum(-1, keepdim=True)
  small = theta2 < 1e-8
  theta = torch.sqrt(torch.where(small, torch.ones_like(theta2), theta2))
  half = 0.5 * theta
  k = torch.where(small, 0.5 - theta2 / 48.0, torch.sin(half) / theta)
  w = torch.where(small, 1.0 - theta2 / 8.0, torch.cos(half))
  return torch.cat([w, k * v], dim=-1)


def subtract_frame_transforms(t01: torch.Tensor, q01: torch.Tensor, t02: torch.Tensor, q02: torch.Tensor):
  """Pose of frame 2 expressed in frame 1 (both given in frame 0)."""
  q10 = quat_conjugate(q01)
  q12 = quat_mul(q10, q02)
  t12 = quat_apply(q10, t02 - t01)
  return t12, q12


def quat_log(q: torch.Tensor) -> torch.Tensor:
  return axis_angle_from_quat(q)
