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
    self.update_moments(n, bm, bv)

  @torch.no_grad()
  def update_moments(self, n: int, bm: torch.Tensor, bv: torch.Tensor):
    """Merge a batch given by its size, mean and (biased) variance (DistCtx.rms_update combines ranks first)."""
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
  # Root-frame representation (10-25, round 19): features [z, gravity, v, w in the root frame, q, qd] and velocity /
  # position targets in the root frame. The default removes the ZYX-Euler heading instead, which is singular at
  # pitch +-90 deg: in a back flip the heading jumps by pi as the torso passes vertical, so features and targets jump.
  body_frame: bool = False
  # Flight-phase robustness (open-source comparison 10-25, allmimic networks.py: 3x2048 SiLU + LayerNorm): during flips the
  # normalized WM inputs reach 10-20 sigma unclamped. layernorm: LayerNorm after every hidden Linear (separate modules, the
  # net.* keys stay as they are, so heading / root-frame checkpoints graft in: Trainer._graft_wm_ln); input_clip > 0
  # clamps the normalized inputs; min_std is the floor of the input normalizer's std (review: >= 1e-2).
  layernorm: bool = False
  input_clip: float = 0.0
  min_std: float = 1e-4
  # Acrobatic-step weighting of the k-step loss (10-26; the flips fail in flight, where the WM error is ~40x the flat
  # one): a transition whose (true) start state spins faster than acro_ang_vel rad/s or tilts the root's up axis below
  # acro_up_z (0.5 = 60 deg) weighs acro_weight in the (weighted-mean) loss. 1 = off = the plain mean, bitwise.
  acro_weight: float = 1.0
  acro_ang_vel: float = 4.0
  acro_up_z: float = 0.5


class DynamicsModel(nn.Module):
  def __init__(self, nj: int, xi_dim: int, dt: float, cfg: WorldModelCfg):
    super().__init__()
    self.nj, self.xi_dim, self.dt, self.cfg = nj, xi_dim, dt, cfg
    self.feat_dim = (10 if cfg.body_frame else 13) + 2 * nj
    self.out_dim = 12 + 2 * nj  # dv(3) dw(3) dqd(J) rp(3) rr(3) rq(J)
    self.in_norm = RunningMeanStd(self.feat_dim + nj, min_std=cfg.min_std)  # features + action
    self.tgt_norm = RunningMeanStd(self.out_dim)
    self.err_norm = RunningMeanStd(12 + 2 * nj)  # dp drot dv dw dq dqd (1-step diffs)
    self.net = mlp(self.feat_dim + nj + xi_dim, cfg.hidden, self.out_dim, zero_last=True)
    if cfg.layernorm:  # registered after net: its parameters come after net's in parameters() (optimizer graft)
      self.ln = nn.ModuleList([nn.LayerNorm(h) for h in cfg.hidden])

  # ---- state algebra -------------------------------------------------------

  def split(self, s):
    J = self.nj
    return s[..., 0:3], s[..., 3:7], s[..., 7:10], s[..., 10:13], s[..., 13 : 13 + J], s[..., 13 + J : 13 + 2 * J]

  def _frame(self, qw: torch.Tensor) -> torch.Tensor:
    """Rotation the translational velocity / position deltas are expressed in: root (body_frame) or ZYX heading."""
    return qw if self.cfg.body_frame else yaw_quat(qw)

  def features(self, s: torch.Tensor) -> torch.Tensor:
    p, qw, v, w, q, qd = self.split(s)
    if self.cfg.body_frame:
      g = torch.zeros_like(v)
      g[:, 2] = -1.0
      return torch.cat([p[:, 2:3], quat_apply_inverse(qw, g), quat_apply_inverse(qw, v), quat_apply_inverse(qw, w), q, qd], dim=-1)
    yq = yaw_quat(qw)
    rot_local = quat_mul(quat_inv(yq), qw)
    return torch.cat(
      [p[:, 2:3], rot6d_from_quat(rot_local), quat_apply_inverse(yq, v), quat_apply_inverse(qw, w), q, qd], dim=-1
    )

  def targets(self, s: torch.Tensor, s_next: torch.Tensor) -> torch.Tensor:
    """Raw regression targets for (s -> s_next)."""
    p, qw, v, w, q, qd = self.split(s)
    p2, qw2, v2, w2, q2, qd2 = self.split(s_next)
    yq = self._frame(qw)
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
    yq = self._frame(qw)
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
    if self.cfg.input_clip > 0:
      x = x.clamp(-self.cfg.input_clip, self.cfg.input_clip)
    out_n = self._run_net(torch.cat([x, xi], dim=-1))
    return self.compose(s, self.tgt_norm.denormalize(out_n))

  def _run_net(self, h: torch.Tensor) -> torch.Tensor:
    if not self.cfg.layernorm:
      return self.net(h)
    mods, k = list(self.net), 0
    for m in mods[:-1]:  # Linear, act, Linear, act, ...: LayerNorm between each hidden Linear and its activation
      h = m(h)
      if isinstance(m, nn.Linear):
        h = self.ln[k](h)
        k += 1
    return mods[-1](h)

  def state_error(self, s_pred: torch.Tensor, s_true: torch.Tensor) -> torch.Tensor:
    """Per-sample normalized squared error (mean over components)."""
    d = self.diffs(s_pred, s_true)
    return ((d / self.err_norm.std) ** 2).mean(-1)

  @torch.no_grad()
  def stats_pairs(self, batch: dict) -> list:
    """[(normalizer, samples)] for one replay batch (the inputs do not depend on the normalizers). Only valid
    (s_j, s_j+1) pairs count: the per-step mask of wm_prefix_mask when present, else the whole-window mask."""
    s0, actions, next_states = batch["s0"], batch["actions"], batch["next_states"]
    m = batch["step_mask"] if "step_mask" in batch else batch["mask"][:, None].expand(-1, actions.shape[1])
    prev = torch.cat([s0[:, None], next_states[:, :-1]], dim=1)[m]  # [M, D]
    nxt = next_states[m]
    return [
      (self.in_norm, torch.cat([self.features(prev), actions[m]], -1)),
      (self.tgt_norm, self.targets(prev, nxt)),
      (self.err_norm, self.diffs(prev, nxt)),
    ]

  @torch.no_grad()
  def update_stats(self, batch: dict) -> None:
    for rms, x in self.stats_pairs(batch):
      rms.update(x)

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
    c = self.cfg
    if c.acro_weight != 1.0:
      prev = torch.cat([s0[:, None], next_states[:, :-1]], dim=1)  # true start state of every transition [n,k,D]
      qw = prev[..., 3:7]
      up_z = 1.0 - 2.0 * (qw[..., 1] ** 2 + qw[..., 2] ** 2)  # root up axis . world z
      acro = (prev[..., 10:13].norm(dim=-1) > c.acro_ang_vel) | (up_z < c.acro_up_z)
      wm_ = m * (1.0 + (c.acro_weight - 1.0) * acro.float())
      loss = (errs * wm_).sum() / wm_.sum().clamp(min=1.0)
    else:
      loss = (errs * m).sum() / m.sum().clamp(min=1.0)
    e1 = (errs[:, 0] * m[:, 0]).sum() / m[:, 0].sum().clamp(min=1.0)
    ek = (errs[:, -1] * m[:, -1]).sum() / m[:, -1].sum().clamp(min=1.0)
    return loss, e1.detach(), ek.detach()
