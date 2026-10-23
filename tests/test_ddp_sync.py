"""Multi-process checks of the data-parallel sync helpers (CPU / gloo, no GPU, no simulator).

  python tests/test_ddp_sync.py        # or: pytest tests/test_ddp_sync.py

Spawns 2 and 3 processes and checks, on every rank:
  * gradient mean (== mean of the per-rank gradients, bitwise equal across ranks);
  * RunningMeanStd sync (== one normalizer fed the concatenated batches, incl. an empty rank;
    bitwise equal across ranks; disabled context == the original rms.update);
  * reset-sampler merge with the real LibraryMotionCommand.update_bin_returns (needs mjlab; skipped otherwise);
  * learned-state hash check passes when equal and stops every rank on a 1-ulp difference;
  * parameter broadcast, scalar reduction (and the key-mismatch guard);
  * NaN guard: a NaN / inf gradient on any one rank (or an overflowing sum) gives the same "not finite"
    verdict on every rank; with the trainer's update helper every rank then skips the step identically
    (parameters and Adam state untouched, still bitwise equal) and the next finite step applies everywhere.
"""

from __future__ import annotations

import os
import socket
import sys
import tempfile
import types

import torch
import torch.multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from supermimic.distributed import DistCtx  # noqa: E402
from supermimic.world_model import RunningMeanStd, mlp  # noqa: E402


def _free_port() -> int:
  with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    return s.getsockname()[1]


def _rank_data(seed: int, n: int, d: int) -> torch.Tensor:
  g = torch.Generator().manual_seed(seed)
  return torch.randn(n, d, generator=g) * (1.0 + seed) + 3.0 * seed


def _check_same_everywhere(ctx: DistCtx, name: str, tensors) -> None:
  ctx.check_consistent({name: list(tensors)}, tag=name)


# ---------------------------------------------------------------------------
# Per-rank test bodies
# ---------------------------------------------------------------------------


def _t_broadcast(ctx: DistCtx):
  torch.manual_seed(100 + ctx.rank)  # different init on every rank
  m = mlp(7, (16, 16), 3)
  rms = RunningMeanStd(7)
  rms.update(_rank_data(ctx.rank, 5, 7))
  m.add_module("rms", rms)
  ctx.broadcast_modules_(m)
  _check_same_everywhere(ctx, "broadcast", list(m.parameters()) + list(m.buffers()))
  return m


def _t_grads(ctx: DistCtx):
  W = ctx.world
  torch.manual_seed(0)
  m = mlp(5, (8,), 2)  # same init everywhere (same seed)
  ref_grads = []
  for r in range(W):  # reference: every rank's gradient computed locally, then averaged
    m.zero_grad(set_to_none=True)
    x = _rank_data(10 + r, 4 + r, 5)
    m(x).pow(2).mean().backward()
    ref_grads.append([p.grad.clone() for p in m.parameters()])
  m.zero_grad(set_to_none=True)
  x = _rank_data(10 + ctx.rank, 4 + ctx.rank, 5)
  m(x).pow(2).mean().backward()
  own = [p.grad.clone() for p in m.parameters()]
  ctx.allreduce_grads_(list(m.parameters()))
  for p, g_own, *g_all in zip(m.parameters(), own, *ref_grads):
    mean = torch.stack(g_all).sum(0) / W
    assert torch.allclose(p.grad, mean, rtol=1e-6, atol=1e-7), "gradient != mean of the rank gradients"
    if W == 2:  # two summands: the order cannot matter -> bitwise
      assert torch.equal(p.grad, (g_all[0] + g_all[1]) / 2)
  assert not all(torch.equal(a, b) for a, b in zip(own, ref_grads[(ctx.rank + 1) % W])), "ranks had identical data"
  _check_same_everywhere(ctx, "grads", [p.grad for p in m.parameters()])
  # a parameter without gradient (on every rank) stays without gradient
  m2 = mlp(5, (8,), 2)
  m2[0].weight.grad = torch.ones_like(m2[0].weight)
  ctx.allreduce_grads_(list(m2.parameters()))
  assert m2[0].bias.grad is None and torch.equal(m2[0].weight.grad, torch.ones_like(m2[0].weight))


def _t_rms(ctx: DistCtx):
  W, D = ctx.world, 6
  synced = RunningMeanStd(D)
  ref = RunningMeanStd(D).double()
  ref.count = ref.count.double()
  for step in range(4):
    sizes = [(0 if (step == 1 and r == 1) else 3 + 5 * r + step) for r in range(W)]  # rank 1 empty at step 1
    xs = [_rank_data(1000 * step + r, sizes[r], D) for r in range(W)]
    ctx.rms_update([(synced, xs[ctx.rank])])
    ref.update(torch.cat(xs).double())
    assert abs(float(synced.count) - float(ref.count)) < 1e-6, (float(synced.count), float(ref.count))
    assert torch.allclose(synced.mean.double(), ref.mean, rtol=1e-5, atol=1e-5)
    assert torch.allclose(synced.var.double(), ref.var, rtol=1e-4, atol=1e-5)
  _check_same_everywhere(ctx, "rms", list(synced.buffers()))
  # several normalizers in one call (the world model updates three at once)
  a, b = RunningMeanStd(3), RunningMeanStd(2)
  ctx.rms_update([(a, _rank_data(7 + ctx.rank, 9, 3)), (b, _rank_data(70 + ctx.rank, 4 + ctx.rank, 2))])
  ra, rb = RunningMeanStd(3), RunningMeanStd(2)
  ra.update(torch.cat([_rank_data(7 + r, 9, 3) for r in range(W)]))
  rb.update(torch.cat([_rank_data(70 + r, 4 + r, 2) for r in range(W)]))
  assert torch.allclose(a.mean, ra.mean, rtol=1e-5, atol=1e-5) and torch.allclose(b.var, rb.var, rtol=1e-4, atol=1e-5)
  _check_same_everywhere(ctx, "rms2", list(a.buffers()) + list(b.buffers()))
  # disabled context == the original update, bitwise
  off = DistCtx(device="cpu")
  u, v = RunningMeanStd(D), RunningMeanStd(D)
  for s in range(3):
    x = _rank_data(50 + s, 11, D)
    off.rms_update([(u, x)])
    v.update(x)
  assert all(torch.equal(p, q) for p, q in zip(u.buffers(), v.buffers()))


def _t_sampler(ctx: DistCtx):
  try:
    from supermimic.library_command import LibraryMotionCommand
  except Exception as e:  # noqa: BLE001
    if ctx.is_main:
      print(f"  (sampler merge skipped: {e!r})")
    return
  NB = 40
  cmd = types.SimpleNamespace(
    library=types.SimpleNamespace(num_bins=NB), cfg=types.SimpleNamespace(return_ema=0.05, clamp_returns=False), device="cpu",
    bin_G=torch.zeros(NB), bin_visits=torch.zeros(NB), bin_weight_hist=torch.full((NB,), 1.0 / NB),
  )
  upd = LibraryMotionCommand.update_bin_returns
  G0, V0 = cmd.bin_G.clone(), cmd.bin_visits.clone()
  for rnd in range(3):
    g = torch.Generator().manual_seed(1000 * rnd + ctx.rank)
    # rank r visits bins in [r*10, r*10+15) plus a few shared ones; bins >= 35 are never visited
    bins = torch.cat([torch.randint(ctx.rank * 10, ctx.rank * 10 + 15, (30,), generator=g), torch.tensor([0, 1, 2])])
    rets = torch.rand(bins.numel(), generator=g) * 10
    upd(cmd, bins, rets)
    cmd.bin_weight_hist = cmd.bin_weight_hist * (1.0 + 0.1 * ctx.rank)
    own_G, own_V, own_W = cmd.bin_G.clone(), cmd.bin_visits.clone(), cmd.bin_weight_hist.clone()
    G, V, Wh = ctx.merge_sampler_bins(cmd.bin_G, cmd.bin_visits, cmd.bin_weight_hist, G0, V0)
    allv = torch.stack(ctx.all_gather_object(own_V - V0))
    allG = torch.stack(ctx.all_gather_object(own_G))
    allW = torch.stack(ctx.all_gather_object(own_W))
    assert torch.equal(V, V0 + allv.sum(0))
    visited = allv.sum(0) > 0
    assert torch.equal(G[~visited], G0[~visited]), "unvisited bins must keep the merged return"
    exp = (allv * allG).sum(0)[visited] / allv.sum(0)[visited]
    assert torch.allclose(G[visited], exp, rtol=1e-5, atol=1e-5)
    only = (allv > 0).sum(0) == 1  # visited by exactly one rank -> that rank's estimate
    who = (allv > 0).float().argmax(0)
    assert torch.allclose(G[only], allG[who[only], torch.nonzero(only).squeeze(-1)], rtol=1e-6)
    assert torch.allclose(Wh, allW.mean(0))
    _check_same_everywhere(ctx, f"sampler{rnd}", [G, V, Wh])
    cmd.bin_G.copy_(G)
    cmd.bin_visits.copy_(V)
    cmd.bin_weight_hist = Wh
    G0, V0 = G, V


def _grads_with(ctx: DistCtx, m: torch.nn.Module, poison: dict[int, float] | None = None, scale: float = 1.0):
  """Fresh per-rank gradients of m; poison {rank: value} writes value into one gradient element on that rank."""
  m.zero_grad(set_to_none=True)
  (m(_rank_data(20 + ctx.rank, 6, 5)).pow(2).mean() * scale).backward()
  if poison and ctx.rank in poison:
    next(iter(m.parameters())).grad.view(-1)[3] = poison[ctx.rank]


def _t_nan_guard(ctx: DistCtx):
  W = ctx.world
  torch.manual_seed(0)
  m = mlp(5, (8,), 2)
  cases = {
    "finite": (None, 1.0, True),
    "nan_on_last_rank": ({W - 1: float("nan")}, 1.0, False),
    "inf_on_rank0": ({0: float("inf")}, 1.0, False),
    "inf_minus_inf": ({0: float("inf"), 1: float("-inf")}, 1.0, False),
    "overflowing_sum": ({r: 3.0e38 for r in range(W)}, 1.0, False),  # every rank finite, the sum is not
  }
  for name, (poison, scale, want) in cases.items():
    _grads_with(ctx, m, poison, scale)
    ok = ctx.allreduce_grads_(list(m.parameters()))
    verdicts = ctx.all_gather_object(ok)
    assert all(v == want for v in verdicts), f"{name}: verdicts {verdicts}, expected {want} on every rank"
  # single-process (disabled) context checks the local gradient
  off = DistCtx(device="cpu")
  m.zero_grad(set_to_none=True)
  m(torch.ones(2, 5)).sum().backward()
  assert off.allreduce_grads_(list(m.parameters())) is True
  next(iter(m.parameters())).grad.view(-1)[0] = float("nan")
  assert off.allreduce_grads_(list(m.parameters())) is False
  m.zero_grad(set_to_none=True)
  assert off.allreduce_grads_(list(m.parameters())) is True  # no gradients at all


def _t_trainer_skip(ctx: DistCtx):
  """Trainer._apply_grads on a stand-in trainer: rank-local NaN -> every rank skips, state stays identical."""
  try:
    from supermimic.trainer import Trainer
  except Exception as e:  # noqa: BLE001
    if ctx.is_main:
      print(f"  (trainer skip test skipped: {e!r})")
    return
  torch.manual_seed(0)
  m = mlp(5, (8,), 2)
  opt = torch.optim.Adam(m.parameters(), lr=1e-2)
  fake = types.SimpleNamespace(dist=ctx, guard_skips={"pi": 0})
  step = lambda: Trainer._apply_grads(fake, "pi", opt, list(m.parameters()), 1.0, 0.1)  # noqa: E731
  _grads_with(ctx, m)
  gn = step()  # finite: applied
  assert gn == gn and len(opt.state) > 0
  before = [p.detach().clone() for p in m.parameters()]
  adam_before = {k: {kk: (vv.clone() if torch.is_tensor(vv) else vv) for kk, vv in v.items()} for k, v in opt.state.items()}
  for bad_rank in range(W := ctx.world):
    _grads_with(ctx, m, {bad_rank: float("nan")})
    gn = step()
    assert gn != gn, "skipped update returns nan"
    assert all(p.grad is None for p in m.parameters()), "skipped update drops the gradients"
    assert all(torch.equal(a, p) for a, p in zip(before, m.parameters())), "skipped update changed the weights"
    for k, v in opt.state.items():
      for kk, vv in v.items():
        assert torch.equal(torch.as_tensor(vv), torch.as_tensor(adam_before[k][kk])), f"Adam state {kk} changed"
  assert fake.guard_skips["pi"] == W
  _check_same_everywhere(ctx, "after_skips", list(m.parameters()) + [t for v in opt.state.values() for t in v.values()])
  _grads_with(ctx, m)
  step()  # finite again: applied on every rank, still one model
  assert not all(torch.equal(a, p) for a, p in zip(before, m.parameters()))
  _check_same_everywhere(ctx, "after_resume", list(m.parameters()) + [t for v in opt.state.values() for t in v.values()])


def _t_check(ctx: DistCtx):
  t = torch.linspace(-1, 1, 101)
  ctx.check_consistent({"a": [t, 3, "x"], "b": [t.double()]}, tag="equal")
  bad = t.clone()
  if ctx.rank == ctx.world - 1:
    bad[17] = torch.nextafter(bad[17], torch.tensor(2.0))  # one ulp
  try:
    ctx.check_consistent({"a": [t], "b": [bad]}, tag="perturbed")
  except RuntimeError as e:
    assert "b differs on ranks" in str(e) and "a differs" not in str(e), str(e)
  else:
    raise AssertionError("1-ulp difference not detected")


def _t_scalars(ctx: DistCtx):
  r = ctx.rank
  m = ctx.reduce_scalars({"x": float(r), "y": 2.0 * r + 1}, "mean")
  W = ctx.world
  assert abs(m["x"] - (W - 1) / 2) < 1e-12 and abs(m["y"] - W) < 1e-12
  assert ctx.reduce_scalars({"x": float(r)}, "sum")["x"] == W * (W - 1) / 2
  assert ctx.reduce_scalars({"x": float(r)}, "max")["x"] == W - 1
  try:
    ctx.reduce_scalars({"x": 1.0} if r == 0 else {"z": 1.0})
  except RuntimeError:
    pass
  else:
    raise AssertionError("key mismatch not detected")


def _worker(rank: int, world: int, port: int, outdir: str):
  os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world))
  torch.set_num_threads(1)
  ctx = DistCtx.from_env("cpu", timeout_min=2.0)
  assert ctx.enabled and ctx.world == world and ctx.device == "cpu"
  done = []
  for fn in (_t_broadcast, _t_grads, _t_rms, _t_sampler, _t_check, _t_scalars, _t_nan_guard, _t_trainer_skip):
    fn(ctx)
    ctx.barrier()
    done.append(fn.__name__)
  ctx.close()
  with open(os.path.join(outdir, f"rank{rank}.ok"), "w") as f:
    f.write("\n".join(done))


def _run(world: int):
  with tempfile.TemporaryDirectory() as tmp:
    mp.spawn(_worker, args=(world, _free_port(), tmp), nprocs=world, join=True)
    for r in range(world):
      assert os.path.exists(os.path.join(tmp, f"rank{r}.ok")), f"rank {r} did not finish"


def test_sync_world2():
  _run(2)


def test_sync_world3():
  _run(3)


def test_disabled_context_is_noop():
  ctx = DistCtx(device="cpu")
  assert not ctx.enabled and ctx.is_main and ctx.world == 1
  m = mlp(3, (4,), 1)
  m(torch.ones(2, 3)).sum().backward()
  g = [p.grad.clone() for p in m.parameters()]
  ctx.allreduce_grads_(list(m.parameters()))
  ctx.broadcast_modules_(m)
  ctx.barrier()
  assert all(torch.equal(a, p.grad) for a, p in zip(g, m.parameters()))
  assert ctx.broadcast_object(("a", 1)) == ("a", 1)
  assert ctx.reduce_scalars({"x": 2.0}) == {"x": 2.0}
  assert set(ctx.check_consistent({"p": list(m.parameters())})) == {"p"}


if __name__ == "__main__":
  for name, fn in list(globals().items()):
    if name.startswith("test_") and callable(fn):
      fn()
      print(f"ok  {name}")
