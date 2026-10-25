"""Learned differentiable dynamics model f_omega(s, a, xi) -> s'.

Yaw-invariant parameterisation: inputs are expressed in the heading (yaw-only)
frame of the root, outputs are velocity increments plus small residual
corrections on top of a semi-implicit Euler integration, so that the initial
model (zero last layer) is already a sensible integrator.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .math_utils import (
  axis_angle_from_quat,
  quat_apply,
  quat_apply_inverse,
  quat_from_rotvec,
  quat_inv,
  quat_mul,
  quat_normalize,
  rot6d_from_quat,
  yaw_quat,
)


class RunningMeanStd(nn.Module):
  def __init__(self, dim: int, eps: float = 1e-6, min_std: float = 1e-4):
    super().__init__()
    self.register_buffer("mean", torch.zeros(dim))
    self.register_buffer("var", torch.ones(dim))
    self.register_buffer("count", torch.tensor(eps))
    self.min_std = min_std

  @torch.no_grad()
  def update(self, x: torch.Tensor):
    x = x.reshape(-1, x.shape[-1]).float()
    n = x.shape[0]
    if n == 0:
      return
    bm, bv = x.mean(0), x.var(0, unbiased=False)
    tot = self.count + n
    delta = bm - self.mean
    self.mean += delta * n / tot
    self.var = (self.var * self.count + bv * n + delta**2 * self.count * n / tot) / tot
    self.count = tot

  @property
  def std(self) -> torch.Tensor:
    return torch.sqrt(self.var).clamp(min=self.min_std)

  def normalize(self, x: torch.Tensor) -> torch.Tensor:
    return (x - self.mean) / self.std

  def denormalize(self, x: torch.Tensor) -> torch.Tensor:
    return x * self.std + self.mean


def mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int, act=nn.SiLU, zero_last: bool = False) -> nn.Sequential:
  layers: list[nn.Module] = []
  d = in_dim
  for h in hidden:
    layers += [nn.Linear(d, h), act()]
    d = h
  last = nn.Linear(d, out_dim)
  if zero_last:
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)
  layers.append(last)
  return nn.Sequential(*layers)


@dataclass
class WorldModelCfg:
  hidden: tuple[int, ...] = (2048, 2048, 2048)
  k: int = 8  # replay window length for the autoregressive fit
  lr: float = 2e-4
  grad_clip: float = 1.0
  batches_per_update: int = 10
  windows_per_batch: int = 2048


class DynamicsModel(nn.Module):
  def __init__(self, nj: int, xi_dim: int, dt: float, cfg: WorldModelCfg):
    super().__init__()
    self.nj, self.xi_dim, self.dt, self.cfg = nj, xi_dim, dt, cfg
    self.feat_dim = 13 + 2 * nj
    self.out_dim = 12 + 2 * nj  # dv(3) dw(3) dqd(J) rp(3) rr(3) rq(J)
    self.in_norm = RunningMeanStd(self.feat_dim + nj)  # features + action
    self.tgt_norm = RunningMeanStd(self.out_dim)
    self.err_norm = RunningMeanStd(12 + 2 * nj)  # dp drot dv dw dq dqd (1-step diffs)
    self.net = mlp(self.feat_dim + nj + xi_dim, cfg.hidden, self.out_dim, zero_last=True)

  # ---- state algebra -------------------------------------------------------

  def split(self, s):
    J = self.nj
    return s[..., 0:3], s[..., 3:7], s[..., 7:10], s[..., 10:13], s[..., 13 : 13 + J], s[..., 13 + J : 13 + 2 * J]

  def features(self, s: torch.Tensor) -> torch.Tensor:
    p, qw, v, w, q, qd = self.split(s)
    yq = yaw_quat(qw)
    rot_local = quat_mul(quat_inv(yq), qw)
    return torch.cat(
      [p[:, 2:3], rot6d_from_quat(rot_local), quat_apply_inverse(yq, v), quat_apply_inverse(qw, w), q, qd], dim=-1
    )

  def targets(self, s: torch.Tensor, s_next: torch.Tensor) -> torch.Tensor:
    """Raw regression targets for (s -> s_next)."""
    p, qw, v, w, q, qd = self.split(s)
    p2, qw2, v2, w2, q2, qd2 = self.split(s_next)
    yq = yaw_quat(qw)
    dt = self.dt
    dv_h = quat_apply_inverse(yq, v2 - v)
    dw_b = quat_apply_inverse(qw, w2 - w)
    dqd = qd2 - qd
    rp_h = quat_apply_inverse(yq, p2 - p - v2 * dt)
    rot_b = axis_angle_from_quat(quat_mul(quat_inv(qw), qw2))
    rr_b = rot_b - quat_apply_inverse(qw, w2) * dt
    rq = q2 - q - qd2 * dt
    return torch.cat([dv_h, dw_b, dqd, rp_h, rr_b, rq], dim=-1)

  def diffs(self, s: torch.Tensor, s_next: torch.Tensor) -> torch.Tensor:
    """Raw 1-step differences used to scale the prediction error."""
    p, qw, v, w, q, qd = self.split(s)
    p2, qw2, v2, w2, q2, qd2 = self.split(s_next)
    drot = axis_angle_from_quat(quat_mul(quat_inv(qw), qw2))
    return torch.cat([p2 - p, drot, v2 - v, w2 - w, q2 - q, qd2 - qd], dim=-1)

  def compose(self, s: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    J = self.nj
    p, qw, v, w, q, qd = self.split(s)
    yq = yaw_quat(qw)
    dv_h, dw_b, dqd = out[:, 0:3], out[:, 3:6], out[:, 6 : 6 + J]
    rp_h, rr_b, rq = out[:, 6 + J : 9 + J], out[:, 9 + J : 12 + J], out[:, 12 + J : 12 + 2 * J]
    dt = self.dt
    v2 = v + quat_apply(yq, dv_h)
    w2 = w + quat_apply(qw, dw_b)
    qd2 = qd + dqd
    p2 = p + v2 * dt + quat_apply(yq, rp_h)
    rot_b = quat_apply_inverse(qw, w2) * dt + rr_b
    qw2 = quat_normalize(quat_mul(qw, quat_from_rotvec(rot_b)))
    q2 = q + qd2 * dt + rq
    return torch.cat([p2, qw2, v2, w2, q2, qd2], dim=-1)

  # ---- forward -------------------------------------------------------------

  def step(self, s: torch.Tensor, a: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
    x = self.in_norm.normalize(torch.cat([self.features(s), a], dim=-1))
    out_n = self.net(torch.cat([x, xi], dim=-1))
    return self.compose(s, self.tgt_norm.denormalize(out_n))

  def state_error(self, s_pred: torch.Tensor, s_true: torch.Tensor) -> torch.Tensor:
    """Per-sample normalized squared error (mean over components)."""
    d = self.diffs(s_pred, s_true)
    return ((d / self.err_norm.std) ** 2).mean(-1)

  @torch.no_grad()
  def update_stats(self, batch: dict) -> None:
    s0, actions, next_states = batch["s0"], batch["actions"], batch["next_states"]
    m = batch["step_mask"] if "step_mask" in batch else batch["mask"][:, None].expand(-1, actions.shape[1])
    prev = torch.cat([s0[:, None], next_states[:, :-1]], dim=1)[m]  # [M, D] valid (s_j, s_j+1) pairs
    nxt = next_states[m]
    self.in_norm.update(torch.cat([self.features(prev), actions[m]], -1))
    self.tgt_norm.update(self.targets(prev, nxt))
    self.err_norm.update(self.diffs(prev, nxt))

  def window_loss(self, s0: torch.Tensor, actions: torch.Tensor, next_states: torch.Tensor, xi: torch.Tensor, mask: torch.Tensor):
    """Autoregressive k-step loss. Returns (loss, err_step1, err_stepk) tensors."""
    k = actions.shape[1]
    s = s0
    errs = []
    for j in range(k):
      s = self.step(s, actions[:, j], xi)
      errs.append(self.state_error(s, next_states[:, j]))
    errs = torch.stack(errs, dim=1)  # [n,k]
    m = (mask if mask.dim() == 2 else mask[:, None].expand(-1, k)).float()  # [n,k] step mask
    loss = (errs * m).sum() / m.sum().clamp(min=1.0)
    e1 = (errs[:, 0] * m[:, 0]).sum() / m[:, 0].sum().clamp(min=1.0)
    ek = (errs[:, -1] * m[:, -1]).sum() / m[:, -1].sum().clamp(min=1.0)
    return loss, e1.detach(), ek.detach()
