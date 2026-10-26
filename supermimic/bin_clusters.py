"""Kinematic clusters of the reset-sampler bins (allmimic tools/bin_clusters.py, read 10-25): 12 features per bin,
k-means, and a cluster-balanced prior (cluster share proportional to n^alpha, split evenly over its bins), so rare motion
types (flips, ground work, handstands) are not drowned by the hours of walking / standing in large libraries.

Features (frames sampled evenly inside the bin, root = tracked body 0):
  root height mean / min, root up-axis z (R22) mean / min, |root-frame angular velocity| x / y / z means,
  std of the root vertical velocity, fractions of frames airborne / hand-down / knee-down / torso-down.
Contacts are proxies from the tracked bodies' world heights (link origins, so the thresholds are loose).
Features and cluster ids are cached next to the library cache (deterministic for a given library and K).
"""

from __future__ import annotations

import os

import numpy as np
import torch

FEAT_VERSION = 1
SAMPLES_PER_BIN = 10
FOOT_Z, HAND_Z, KNEE_Z, TORSO_Z = 0.10, 0.12, 0.12, 0.30


def _body_groups(body_names: tuple[str, ...]) -> dict[str, list[int]]:
  low = [n.lower() for n in body_names]

  def find(*keys):
    return [i for i, n in enumerate(low) if any(k in n for k in keys)]

  g = {"foot": find("ankle"), "hand": find("wrist") or find("elbow"), "knee": find("knee"), "torso": find("torso") or find("waist")}
  for k, v in g.items():
    if not v:
      raise ValueError(f"bin features: no tracked body for {k!r} in {body_names}")
  return g


def _rotate_inv(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  """R(q)^T v for wxyz quaternions."""
  w, u = q[..., :1], -q[..., 1:]
  t = 2.0 * torch.cross(u, v, dim=-1)
  return v + w * t + torch.cross(u, t, dim=-1)


@torch.no_grad()
def bin_features(lib, chunk: int = 200_000, log=print) -> np.ndarray:
  """[num_bins, 12] float32 (cached in the library cache dir)."""
  path = os.path.join(lib.cache_dir, f"bin_feat12_v{FEAT_VERSION}.npy")
  if os.path.exists(path):
    f = np.load(path)
    if f.shape[0] == lib.num_bins:
      return f
  groups = _body_groups(lib.body_names)
  start = lib.bin_start.cpu()
  end = lib.bin_end.cpu()
  NB = start.numel()
  out = np.zeros((NB, 12), dtype=np.float32)
  j = (torch.arange(SAMPLES_PER_BIN, dtype=torch.float64) + 0.5) / SAMPLES_PER_BIN
  for b0 in range(0, NB, chunk):
    b1 = min(NB, b0 + chunk)
    s, e = start[b0:b1].double(), end[b0:b1].double()
    idx = (s[:, None] + (e - s)[:, None] * j[None, :]).long().clamp(max=lib.time_step_total - 1).reshape(-1)
    pos = torch.index_select(lib.fields["body_pos"], 0, idx).float()  # [n*S, Bt, 3]
    quat = torch.index_select(lib.fields["body_quat"], 0, idx)[:, 0].float()  # root
    angw = torch.index_select(lib.fields["body_ang_vel"], 0, idx)[:, 0].float()
    linw = torch.index_select(lib.fields["body_lin_vel"], 0, idx)[:, 0].float()
    n, S = b1 - b0, SAMPLES_PER_BIN
    z = pos[..., 2]
    rz = z[:, 0].view(n, S)
    qn = quat / quat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    r22 = (1.0 - 2.0 * (qn[:, 1] ** 2 + qn[:, 2] ** 2)).view(n, S)
    wb = _rotate_inv(qn, angw).abs().view(n, S, 3)
    vz = linw[:, 2].view(n, S)
    foot = z[:, groups["foot"]].min(-1).values
    hand = z[:, groups["hand"]].min(-1).values
    knee = z[:, groups["knee"]].min(-1).values
    torso = z[:, groups["torso"]].min(-1).values
    others = torch.stack([hand, knee, torso], -1).min(-1).values
    airborne = ((foot > FOOT_Z) & (others > HAND_Z)).float().view(n, S)
    f = torch.stack([
      rz.mean(1), rz.min(1).values, r22.mean(1), r22.min(1).values,
      wb[..., 0].mean(1), wb[..., 1].mean(1), wb[..., 2].mean(1), vz.std(1, unbiased=False),
      airborne.mean(1), (hand < HAND_Z).float().view(n, S).mean(1), (knee < KNEE_Z).float().view(n, S).mean(1),
      (torso < TORSO_Z).float().view(n, S).mean(1),
    ], -1)
    out[b0:b1] = torch.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0).numpy()
    if (b0 // chunk) % 10 == 0:
      log(f"[bin_clusters] features {b1}/{NB} bins")
  tmp = f"{path}.tmp{os.getpid()}.npy"
  np.save(tmp, out)
  os.replace(tmp, path)
  return out


@torch.no_grad()
def kmeans(x: torch.Tensor, k: int, iters: int = 30, seed: int = 0, init_sample: int = 200_000, chunk: int = 1 << 20) -> torch.Tensor:
  """Plain Lloyd k-means (k-means++ init on a subsample). x [N, D] on the compute device -> labels [N] (long)."""
  g = torch.Generator(device="cpu").manual_seed(seed)
  N = x.shape[0]
  sub = x[torch.randperm(N, generator=g)[: min(N, init_sample)].to(x.device)]
  cent = sub[torch.randint(sub.shape[0], (1,), generator=g).to(x.device)]
  d2 = ((sub - cent) ** 2).sum(-1)
  for _ in range(1, k):
    p = (d2 / d2.sum().clamp(min=1e-12)).cpu().double()
    nxt = sub[torch.multinomial(p, 1, generator=g).to(x.device)]
    cent = torch.cat([cent, nxt], 0)
    d2 = torch.minimum(d2, ((sub - nxt) ** 2).sum(-1))
  labels = torch.zeros(N, dtype=torch.long, device=x.device)
  for _ in range(iters):
    for a in range(0, N, chunk):
      labels[a : a + chunk] = torch.cdist(x[a : a + chunk], cent).argmin(1)
    cnt = torch.bincount(labels, minlength=k).float()
    s = torch.zeros_like(cent).index_add_(0, labels, x)
    empty = cnt == 0
    cent = torch.where(empty[:, None], cent, s / cnt.clamp(min=1.0)[:, None])
  return labels


@torch.no_grad()
def bin_cluster_prior(lib, k: int, alpha: float, device: str, seed: int = 0, log=print) -> torch.Tensor:
  """Per-bin weights [num_bins] (sum 1): cluster c gets n_c^alpha / sum_c' n_c'^alpha of the mass, split evenly."""
  path = os.path.join(lib.cache_dir, f"bin_cluster_k{k}_s{seed}_v{FEAT_VERSION}.npy")
  labels = None
  if os.path.exists(path):
    lab = np.load(path)
    if lab.shape[0] == lib.num_bins:
      labels = torch.from_numpy(lab.astype(np.int64)).to(device)
  if labels is None:
    f = torch.from_numpy(bin_features(lib, log=log)).to(device)
    f = (f - f.mean(0)) / f.std(0).clamp(min=1e-6)
    labels = kmeans(f, k, seed=seed)
    tmp = f"{path}.tmp{os.getpid()}.npy"
    np.save(tmp, labels.cpu().numpy().astype(np.int32))
    os.replace(tmp, path)
  n = torch.bincount(labels, minlength=k).double()
  share = torch.where(n > 0, n.pow(alpha), torch.zeros_like(n))
  share = share / share.sum()
  w = (share / n.clamp(min=1.0))[labels]
  w = (w / w.sum()).float()
  log(f"[bin_clusters] k={k} alpha={alpha}: cluster sizes min {int(n[n > 0].min())} / median {int(n[n > 0].median())} / "
      f"max {int(n.max())}; per-bin weight max/min {float(w.max() / w[w > 0].min()):.0f}x")
  return w
