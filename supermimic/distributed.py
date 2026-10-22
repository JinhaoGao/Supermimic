"""Single-node data parallelism for SuperMimic (torchrun, one model on N GPUs).

Every rank runs the full single-GPU algorithm on its own simulator envs and its
own replay buffer (seed + rank); the ranks stay one model by keeping every
piece of learned state bitwise identical:

  * parameters / buffers: broadcast from rank 0 once at start;
  * gradients: after each backward the gradients of the module being updated
    are flattened, all-reduced (sum) and divided by the world size, then the
    original clipping / optimizer step runs on identical inputs on every rank
    (no DistributedDataParallel wrapper: the step functions are torch.compiled
    free functions, not module forwards);
  * RunningMeanStd normalizers: every rank all-gathers the per-rank batch
    moments (n, mean, var), combines them in rank order (Chan's parallel
    formula, float64) and applies the same update;
  * reset sampler (bin_G / bin_visits / bin_weight_hist): merged periodically
    (visit-weighted mean of what each rank learned since the last merge);
  * only rank 0 writes TensorBoard / checkpoints / runs the benchmark; the
    other ranks wait in a CPU (gloo) barrier.

A periodic self-check hashes every piece of learned state on every rank and
stops the run if any rank differs from rank 0.

When the process is not started by torchrun (WORLD_SIZE unset or 1) the
context is disabled and every helper falls back to the exact single-GPU code
path (bitwise identical to the original trainer).
"""

from __future__ import annotations

import hashlib
import os
import zlib
from datetime import timedelta
from typing import Iterable

import torch
import torch.distributed as dist


class DistCtx:
  def __init__(
    self,
    enabled: bool = False,
    rank: int = 0,
    world: int = 1,
    local_rank: int = 0,
    device: str = "cuda:0",
    backend: str = "",
    timeout_min: float = 0.0,
    cpu_group=None,
  ):
    self.enabled = enabled
    self.rank = rank
    self.world = world
    self.local_rank = local_rank
    self.device = device
    self.backend = backend
    self.timeout_min = timeout_min
    self.cpu_group = cpu_group  # gloo group for barriers / python objects (None = default group)
    self._keys_crc: int | None = None

  @property
  def is_main(self) -> bool:
    return self.rank == 0

  # ---- setup / teardown ------------------------------------------------------

  @staticmethod
  def from_env(device: str, timeout_min: float = 120.0) -> "DistCtx":
    """torchrun -> initialized context (device cuda:LOCAL_RANK, NCCL + a gloo group);
    plain ``python`` (or a world of one) -> disabled context on ``device``."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world <= 1:
      return DistCtx(device=device)
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    timeout = timedelta(minutes=timeout_min)
    if device.startswith("cuda"):
      dev = f"cuda:{local_rank}"
      torch.cuda.set_device(local_rank)
      torch.zeros(1, device=dev)  # bind the CUDA context before Warp picks its default device
      dist.init_process_group("nccl", timeout=timeout, device_id=torch.device(dev))
      cpu_group = dist.new_group(backend="gloo", timeout=timeout)
      backend = "nccl"
    else:  # CPU (tests): gloo for everything
      dev = "cpu"
      dist.init_process_group("gloo", timeout=timeout)
      cpu_group = None
      backend = "gloo"
    ctx = DistCtx(True, rank, world, local_rank, dev, backend, timeout_min, cpu_group)
    if ctx.is_main:
      nccl = ""
      if backend == "nccl":
        try:
          nccl = " nccl " + ".".join(str(v) for v in torch.cuda.nccl.version())
        except Exception:  # noqa: BLE001
          pass
      print(f"[dist] world {world} backend {backend}{nccl} timeout {timeout_min:g} min (torch {torch.__version__})", flush=True)
    print(f"[dist] rank {rank}/{world} local_rank {local_rank} device {dev} pid {os.getpid()}", flush=True)
    return ctx

  def close(self) -> None:
    if self.enabled and dist.is_initialized():
      dist.destroy_process_group()

  # ---- control-plane collectives (CPU / gloo) -------------------------------

  def barrier(self) -> None:
    """CPU barrier (gloo): long waits (rank 0 evaluating / building the cache) do not hold a GPU collective."""
    if self.enabled:
      dist.barrier(group=self.cpu_group)

  def broadcast_object(self, obj, src: int = 0):
    if not self.enabled:
      return obj
    box = [obj if self.rank == src else None]
    dist.broadcast_object_list(box, src=src, group=self.cpu_group)
    return box[0]

  def all_gather_object(self, obj) -> list:
    if not self.enabled:
      return [obj]
    out = [None] * self.world
    dist.all_gather_object(out, obj, group=self.cpu_group)
    return out

  # ---- learned state ---------------------------------------------------------

  @torch.no_grad()
  def broadcast_tensors_(self, tensors: Iterable[torch.Tensor], src: int = 0) -> None:
    if not self.enabled:
      return
    for t in tensors:
      dist.broadcast(t.data, src=src)

  def broadcast_modules_(self, *modules: torch.nn.Module) -> None:
    """Parameters and buffers of every module from rank ``src`` (start of training)."""
    for m in modules:
      self.broadcast_tensors_(list(m.parameters()) + list(m.buffers()))

  @torch.no_grad()
  def allreduce_grads_(self, params: Iterable[torch.nn.Parameter]) -> None:
    """In-place gradient mean over ranks (flattened into one buffer -> one collective).

    Parameters without a gradient are skipped; every rank runs the same code path,
    so the set is the same on all ranks (a mismatch would surface in check_consistent)."""
    if not self.enabled:
      return
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
      return
    dt = grads[0].dtype
    assert all(g.dtype == dt for g in grads), "mixed gradient dtypes"
    flat = torch.cat([g.reshape(-1) for g in grads])
    dist.all_reduce(flat)
    flat.div_(self.world)
    off = 0
    for g in grads:
      n = g.numel()
      g.copy_(flat[off : off + n].view_as(g))
      off += n

  @torch.no_grad()
  def rms_update(self, pairs) -> None:
    """Update RunningMeanStd normalizers ``[(rms, x), ...]`` with the batch of all ranks.

    Disabled: ``rms.update(x)`` in order (the original code path). Enabled: one all_gather of
    every rank's (n, mean, var) per normalizer, combined in rank order in float64 on every rank
    (identical inputs, identical ops -> identical result), then ``rms.update_moments``."""
    if not self.enabled:
      for rms, x in pairs:
        rms.update(x)
      return
    parts, dims = [], []
    for rms, x in pairs:
      D = rms.mean.shape[0]
      x = x.reshape(-1, D).float()
      n = x.shape[0]
      if n > 0:
        bm, bv = x.mean(0), x.var(0, unbiased=False)
      else:
        bm = bv = torch.zeros(D, device=x.device)
      parts.append(torch.cat([torch.full((1,), float(n), device=x.device, dtype=torch.float64), bm.double(), bv.double()]))
      dims.append(D)
    vec = torch.cat(parts)
    out = [torch.empty_like(vec) for _ in range(self.world)]
    dist.all_gather(out, vec)
    G = torch.stack(out)  # [W, L]
    ns = G[:, [sum(1 + 2 * d for d in dims[:i]) for i in range(len(dims))]].sum(0).tolist()  # global n per normalizer
    off = 0
    for (rms, _), D, n_tot in zip(pairs, dims, ns):
      n_r = G[:, off]
      bm_r = G[:, off + 1 : off + 1 + D]
      bv_r = G[:, off + 1 + D : off + 1 + 2 * D]
      off += 1 + 2 * D
      if n_tot <= 0:
        continue
      m = (n_r[:, None] * bm_r).sum(0) / n_tot
      v = (n_r[:, None] * (bv_r + (bm_r - m) ** 2)).sum(0) / n_tot
      rms.update_moments(int(round(n_tot)), m.float(), v.float())

  @torch.no_grad()
  def merge_sampler_bins(self, G: torch.Tensor, V: torch.Tensor, W: torch.Tensor, G0: torch.Tensor, V0: torch.Tensor):
    """Merge the per-rank return-based reset samplers.

    G / V / W: this rank's bin_G (return EMA), bin_visits and bin_weight_hist; G0 / V0: the merged
    state after the previous merge (identical on all ranks). Each rank learned dV = V - V0 visits
    since then; merged bin_G = visit-weighted mean of the ranks' bin_G over the bins anyone visited
    (unvisited bins keep G0), bin_visits = V0 + sum dV, bin_weight_hist = rank mean.
    Returns the merged (G, V, W), identical on every rank."""
    dV = V - V0
    buf = torch.stack([torch.where(dV > 0, dV * G, torch.zeros_like(G)), dV, W])
    dist.all_reduce(buf)
    sV = buf[1]
    seen = sV > 0
    G_new = torch.where(seen, buf[0] / torch.where(seen, sV, torch.ones_like(sV)), G0)
    return G_new, V0 + sV, buf[2] / self.world

  # ---- logging ---------------------------------------------------------------

  @torch.no_grad()
  def reduce_scalars(self, scal: dict[str, float], op: str = "mean", check_keys: bool = True) -> dict[str, float]:
    """Element-wise mean / sum / max over ranks of a dict of python floats (same keys on every rank;
    check_keys verifies that first with a small CPU all_gather, skip it for fixed per-iteration keys)."""
    if not self.enabled:
      return dict(scal)
    keys = sorted(scal)
    if check_keys:
      crc = zlib.crc32("\n".join(keys).encode())
      crcs = self.all_gather_object(crc)
      if any(c != crcs[0] for c in crcs):
        raise RuntimeError(f"[dist] rank {self.rank}: scalar keys differ between ranks ({keys})")
    t = torch.tensor([float(scal[k]) for k in keys], dtype=torch.float64, device=self.device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX if op == "max" else dist.ReduceOp.SUM)
    if op == "mean":
      t /= self.world
    return dict(zip(keys, t.tolist()))

  # ---- consistency self-check ------------------------------------------------

  @staticmethod
  def digest(items: Iterable) -> str:
    """Exact content hash of tensors / python scalars (dtype, shape and every byte)."""
    h = hashlib.blake2b(digest_size=16)
    for x in items:
      if torch.is_tensor(x):
        t = x.detach().reshape(-1).contiguous().cpu()
        h.update(f"{t.dtype}{tuple(x.shape)}".encode())
        h.update(t.view(torch.uint8).numpy().tobytes())
      else:
        h.update(repr(x).encode())
    return h.hexdigest()

  def check_consistent(self, groups: dict[str, list], tag: str = "") -> dict[str, str]:
    """Hash every group on every rank and compare with rank 0; raise on any difference.

    groups: name -> list of tensors / python scalars (same structure on every rank)."""
    mine = {k: self.digest(v) for k, v in groups.items()}
    if not self.enabled:
      return mine
    allr = self.all_gather_object(mine)
    bad = [(r, k) for r in range(1, self.world) for k in mine if allr[r].get(k) != allr[0].get(k)]
    if bad:
      by_group: dict[str, list[int]] = {}
      for r, k in bad:
        by_group.setdefault(k, []).append(r)
      msg = f"[dist] state diverged across ranks {tag}: " + "; ".join(f"{k} differs on ranks {v}" for k, v in by_group.items())
      if self.is_main:
        print(msg, flush=True)
      raise RuntimeError(msg)
    return mine
