"""Reset-sampler return clamp (LibraryMotionCommandCfg.clamp_returns / --sampler_clamp_returns); CPU, torch only.

  python tests/test_sampler_clamp.py      # or: pytest tests/test_sampler_clamp.py

When mjlab / tensorboard are not installed they are stubbed via sys.modules, so the REAL LibraryMotionCommand,
Trainer and DistCtx code runs here (with mjlab installed the real package is used).

Checks:
  * switch off -> update_bin_returns (return EMA) and sampling_probabilities bitwise equal to bc9d507 (copied
    verbatim below) on random data with negative, > g_max and non-finite returns;
  * switch on -> the g1_full_ee_ddp8 collapse (one clip-end bin at G = -31.4 owns the whole softmax share,
    sampler/top1_prob 0.5000) is gone: softmax share bounded and spread over the low-return bins; returns outside
    [0, g_max] are clamped before the EMA; non-finite ones are still dropped;
  * checkpoint load (real Trainer.load on resume and via Trainer._rollback): a -31 state comes back clamped and the
    clamp is reported once per load; switch off loads it unchanged;
  * data parallel (2 and 3 gloo processes; real Trainer.load / _dist_start / _merge_sampler, i.e. the real
    DistCtx.merge_sampler_bins): every rank loads + clamps the same checkpoint and ends identical, and the
    visit-weighted merge of clamped per-rank states stays in [0, g_max] (incl. states sitting exactly at 0 / g_max
    with millions of prior visits);
  * wiring: --sampler_clamp_returns -> TrainCfg (scripts/train.py main) -> train_cfg.json / make_env_cfg kwargs
    (Trainer.__init__) -> LibraryMotionCommandCfg.clamp_returns (make_env_cfg).
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import socket
import sys
import tempfile
import types
from collections import deque
from unittest import mock

import torch
import torch.multiprocessing as mp

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)


class _Stub(types.ModuleType):
  """Stand-in module: every attribute is a class that accepts (and stores) any constructor arguments."""

  def __getattr__(self, name):
    if name.startswith("__"):
      raise AttributeError(name)
    cls = type(name, (), {"__init__": lambda obj, *a, **k: obj.__dict__.update(k)})
    setattr(self, name, cls)
    return cls


def _install_stubs() -> list[str]:
  stubbed = []
  try:
    import mjlab  # noqa: F401
  except ImportError:
    for m in (
      "mjlab", "mjlab.envs", "mjlab.managers", "mjlab.managers.observation_manager", "mjlab.managers.termination_manager",
      "mjlab.tasks", "mjlab.tasks.tracking", "mjlab.tasks.tracking.mdp", "mjlab.tasks.tracking.mdp.commands",
      "mjlab.utils", "mjlab.utils.lab_api", "mjlab.utils.lab_api.math",
    ):
      sys.modules[m] = _Stub(m)
      parent, _, child = m.rpartition(".")
      if parent:
        setattr(sys.modules[parent], child, sys.modules[m])  # `from mjlab.tasks.tracking import mdp` -> the module

    @dataclasses.dataclass(kw_only=True)
    class MotionCommandCfg:  # the base-dataclass fields make_env_cfg passes
      entity_name: str = ""
      resampling_time_range: tuple = (0.0, 0.0)
      debug_vis: bool = False
      pose_range: dict = dataclasses.field(default_factory=dict)
      velocity_range: dict = dataclasses.field(default_factory=dict)
      joint_position_range: tuple = (0.0, 0.0)
      anchor_body_name: str = ""
      body_names: tuple = ()
      motion_file: str = ""

    sys.modules["mjlab.tasks.tracking.mdp.commands"].MotionCommandCfg = MotionCommandCfg
    stubbed.append("mjlab")
  try:
    import torch.utils.tensorboard  # noqa: F401
  except ImportError:
    tb = types.ModuleType("torch.utils.tensorboard")
    tb.SummaryWriter = lambda *a, **k: types.SimpleNamespace(add_scalar=lambda *a, **k: None)
    sys.modules["torch.utils.tensorboard"] = tb
    stubbed.append("tensorboard")
  return stubbed


STUBBED = _install_stubs()

import supermimic.env_cfg as env_cfg_mod  # noqa: E402
import supermimic.motions as motions_mod  # noqa: E402
import supermimic.trainer as trainer_mod  # noqa: E402
from supermimic.distributed import DistCtx  # noqa: E402
from supermimic.library_command import LibraryMotionCommand, LibraryMotionCommandCfg  # noqa: E402
from supermimic.trainer import TrainCfg, Trainer, cosine_lambda  # noqa: E402
from supermimic.world_model import mlp  # noqa: E402

_KEYS = ("mix_uniform", "mix_softmax", "mix_history", "softmax_temp", "g_max", "return_ema", "weight_ema", "clamp_returns")
DEFAULTS = {f.name: f.default for f in dataclasses.fields(LibraryMotionCommandCfg) if f.name in _KEYS}
G_MAX = DEFAULTS["g_max"]


# ---------------------------------------------------------------------------
# The sampler before the switch (bc9d507), verbatim
# ---------------------------------------------------------------------------


@torch.no_grad()
def update_bin_returns_before(self, bin_ids: torch.Tensor, returns: torch.Tensor) -> None:
  """library_command.LibraryMotionCommand.update_bin_returns at bc9d507 (before clamp_returns), verbatim."""
  finite = torch.isfinite(returns)
  if not bool(finite.all()):
    bin_ids, returns = bin_ids[finite], returns[finite]
  if bin_ids.numel() == 0:
    return
  NB = self.library.num_bins
  bin_ids = bin_ids.long()
  s = torch.zeros(NB, device=self.device).index_add_(0, bin_ids, returns)
  c = torch.zeros(NB, device=self.device).index_add_(0, bin_ids, torch.ones_like(returns))
  visited = c > 0
  mean = s[visited] / c[visited]
  a = self.cfg.return_ema
  first = self.bin_visits[visited] == 0
  self.bin_G[visited] = torch.where(first, mean, (1 - a) * self.bin_G[visited] + a * mean)
  self.bin_visits[visited] += c[visited]


@torch.no_grad()
def sampling_probabilities_before(self) -> torch.Tensor:
  """library_command.LibraryMotionCommand.sampling_probabilities at bc9d507 (before clamp_returns), verbatim."""
  c = self.cfg
  NB = self.library.num_bins
  uniform = torch.full((NB,), 1.0 / NB, device=self.device)
  G = torch.where(self.bin_visits > 0, self.bin_G, torch.full_like(self.bin_G, c.g_max * 0.5))
  soft = torch.softmax(-c.softmax_temp * G / c.g_max, dim=0)
  hist = self.bin_weight_hist / self.bin_weight_hist.sum()
  probs = c.mix_uniform * uniform + c.mix_softmax * soft + c.mix_history * hist
  probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0) + 1e-12
  probs = probs / probs.sum()
  self.bin_weight_hist = (1 - c.weight_ema) * self.bin_weight_hist + c.weight_ema * soft
  return probs


# ---------------------------------------------------------------------------
# Stand-ins (real methods bound to plain namespaces: no simulator, no library)
# ---------------------------------------------------------------------------


def make_cmd(NB: int, clamp: bool, G: torch.Tensor | None = None, V: torch.Tensor | None = None, prior: torch.Tensor | None = None, **cfg_kw):
  cmd = types.SimpleNamespace(
    library=types.SimpleNamespace(num_bins=NB), cfg=types.SimpleNamespace(**{**DEFAULTS, "clamp_returns": clamp, **cfg_kw}),
    device="cpu", bin_G=torch.zeros(NB) if G is None else G.clone(), bin_visits=torch.zeros(NB) if V is None else V.clone(),
    bin_weight_hist=torch.full((NB,), 1.0 / NB),
    bin_prior=prior,  # --source_weights / --name_weights prior of the uniform share (None = uniform, as before)
  )
  for name in ("update_bin_returns", "sampling_probabilities", "clamp_bin_returns_"):
    setattr(cmd, name, types.MethodType(getattr(LibraryMotionCommand, name), cmd))
  return cmd


def make_trainer(log_dir: str, NB: int, clamp: bool, dist: DistCtx | None = None) -> Trainer:
  """Trainer without simulator: models / optimizers / schedulers / sampler state, real load / rollback / sync code."""
  t = object.__new__(Trainer)
  t.cfg = TrainCfg(log_dir=log_dir, sampler_clamp_returns=clamp, wd_warmup=10)
  t.dist = dist if dist is not None else DistCtx(device="cpu")
  t.main = t.dist.is_main
  t.device = "cpu"
  t.prints = []
  t._print = (lambda *a, **k: t.prints.append(" ".join(str(x) for x in a))) if t.main else (lambda *a, **k: None)
  torch.manual_seed(0)
  t.policy, t.critic, t.wm = mlp(3, (4,), 2), mlp(3, (4,), 1), mlp(3, (4,), 3)
  t.opt_wm = torch.optim.Adam(t.wm.parameters(), lr=1e-3)
  t.opt_pi = torch.optim.Adam(t.policy.parameters(), lr=1e-3)
  t.opt_v = torch.optim.Adam(t.critic.parameters(), lr=1e-3)
  lam = cosine_lambda(t.cfg.lr_decay_iters, t.cfg.lr_final_ratio)
  t.sched = [torch.optim.lr_scheduler.LambdaLR(o, lam) for o in (t.opt_wm, t.opt_pi, t.opt_v)]
  t._base_lrs0 = [list(s.base_lrs) for s in t.sched]
  t.lr_scale = 1.0
  t.it, t.env_steps = 0, 0
  t.cmd = make_cmd(NB, clamp)
  t.lib = types.SimpleNamespace(fingerprint=f"stand-in library, {NB} bins")  # the state check's "library" group
  t.writer = types.SimpleNamespace(add_scalar=lambda *a, **k: None)
  t.buffer = types.SimpleNamespace(size=123)  # replay stand-in (the watchdog rollback empties it)
  t.wd_healthy_ckpt, t.wd_quiet_until, t.wd_rollbacks, t.wd_reward_ema, t.wd_nf_streak = None, 0, 0, None, 0
  t.wd_hist, t.wd_vloss = deque(), deque(maxlen=2000)
  return t


def collapse_state(NB: int, seed: int = 0):
  """60k state of g1_full_ee_ddp8 in miniature: G ~ N(8.7, 1) clipped to [0, 9.86], one clip-end bin at -31.4 with
    3.68M visits, 300 more bins in [-12, 0]."""
  g = torch.Generator().manual_seed(seed)
  G = (torch.randn(NB, generator=g) + 8.7).clamp(0.0, 9.86)
  idx = torch.randperm(NB, generator=g)[:301]
  hot, neg = int(idx[0]), idx[1:]
  G[neg] = -12.0 * torch.rand(neg.numel(), generator=g)
  G[hot] = -31.4
  V = torch.randint(1, 200, (NB,), generator=g).float()
  V[hot] = 3.68e6
  return G, V, hot, neg


def write_ckpt(path: str, NB: int):
  """Checkpoint as Trainer.save writes it, with the collapsed sampler state (+ a few bins above g_max)."""
  t = make_trainer(os.path.dirname(path), NB, False)
  G, V, hot, _ = collapse_state(NB)
  G[torch.nonzero(G >= 0).squeeze(-1)[:5]] = torch.tensor([10.0001, 10.5, 11.0, 12.0, 1e3])
  torch.save(
    {
      "it": 60000, "env_steps": 6_000_000, "policy": t.policy.state_dict(), "critic": t.critic.state_dict(),
      "wm": t.wm.state_dict(), "opt_wm": t.opt_wm.state_dict(), "opt_pi": t.opt_pi.state_dict(), "opt_v": t.opt_v.state_dict(),
      "bin_G": G, "bin_visits": V, "lr_scale": 1.0, "cfg": dataclasses.asdict(t.cfg),
    },
    path,
  )
  return G, V, hot


def clamp_msgs(t: Trainer) -> list[str]:
  return [s for s in t.prints if "clamp_returns:" in s]


# ---------------------------------------------------------------------------
# (a) switch off == before, bitwise
# ---------------------------------------------------------------------------


def test_defaults_off():
  assert DEFAULTS["clamp_returns"] is False and G_MAX == 10.0
  assert TrainCfg().sampler_clamp_returns is False


def test_off_is_bitwise_identical_to_before():
  NB = 5000
  g = torch.Generator().manual_seed(0)
  new, old = make_cmd(NB, False), make_cmd(NB, False)
  for rnd in range(30):
    bins = torch.cat([torch.randint(0, NB, (4000,), generator=g), torch.randint(0, 50, (500,), generator=g)])  # + hot bins
    rets = torch.randn(bins.numel(), generator=g) * 15.0 + 5.0  # ~37% < 0, ~37% > g_max
    if rnd % 5 == 1:
      rets[::97] = float("nan")
      rets[3], rets[5] = float("inf"), float("-inf")
    if rnd == 7:
      rets[:] = float("nan")  # nothing finite: no update
    new.update_bin_returns(bins, rets)
    update_bin_returns_before(old, bins, rets)
    assert torch.equal(new.bin_G, old.bin_G) and torch.equal(new.bin_visits, old.bin_visits)
    for _ in range(3):
      p_new, p_old = new.sampling_probabilities(), sampling_probabilities_before(old)
      assert torch.equal(p_new, p_old) and torch.equal(new.bin_weight_hist, old.bin_weight_hist)
  assert float(old.bin_G.min()) < -5.0 and float(old.bin_G.max()) > 15.0, "data must leave [0, g_max] on both sides"
  # the collapsed state itself (part b) too
  G, V, _, _ = collapse_state(20_000)
  a, b = make_cmd(20_000, False, G, V), make_cmd(20_000, False, G, V)
  for _ in range(5):
    assert torch.equal(a.sampling_probabilities(), sampling_probabilities_before(b))


# ---------------------------------------------------------------------------
# (b) switch on: collapse gone, returns clamped
# ---------------------------------------------------------------------------


def test_on_bounds_the_collapse():
  NB = 200_000
  G, V, hot, neg = collapse_state(NB)
  out = {}
  for name, clamp in (("old (off)", False), ("new (on) ", True)):
    soft = make_cmd(NB, clamp, G, V, mix_uniform=0.0, mix_softmax=1.0, mix_history=0.0).sampling_probabilities()  # softmax term alone
    cmd = make_cmd(NB, clamp, G, V)
    for _ in range(1000):  # bin_weight_hist (EMA 0.01 of the softmax term) converges: the logged steady state
      probs = cmd.sampling_probabilities()
    draws = torch.multinomial(probs, 200_000, replacement=True, generator=torch.Generator().manual_seed(1))
    o = out[name] = dict(
      soft_top1=float(soft.max()), soft_top1_is_hot=int(soft.argmax()) == hot, soft_eff_bins=float(1.0 / (soft**2).sum()),
      soft_on_low_bins=float(soft[neg].sum() + soft[hot]), top1_prob=float(probs.max()),
      top1_draw_share=float(torch.bincount(draws, minlength=NB).max()) / draws.numel(), soft_hot=float(soft[hot]),
    )
    print(f"  {name}: " + ", ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in o.items()))
  old, new = out["old (off)"], out["new (on) "]
  # old: one bin owns the softmax share; top1_prob -> 0.4 + 0.1 = 0.5 (the logged 0.5000)
  assert old["soft_top1"] > 0.999 and old["soft_top1_is_hot"] and old["top1_prob"] > 0.499 and old["top1_draw_share"] > 0.45
  # new: the 301 bins at G <= 0 all sit at 0 (equal weight), nothing owns the share
  assert new["soft_top1"] < 1.01 / 301 and new["soft_eff_bins"] > 290 and new["soft_on_low_bins"] > 0.99
  assert new["top1_prob"] < 0.002 and new["top1_draw_share"] < 0.005
  Gc = G.clamp(0.0, G_MAX)
  # the sampling clamp alone == a clamped state (the load clamp only makes the stored EMA match what is sampled)
  assert torch.equal(make_cmd(NB, True, G, V).sampling_probabilities(), make_cmd(NB, True, Gc, V).sampling_probabilities())
  # nothing outside [0, g_max] -> the switch changes nothing
  assert torch.equal(make_cmd(NB, False, Gc, V).sampling_probabilities(), make_cmd(NB, True, Gc, V).sampling_probabilities())


def test_on_clamps_returns():
  on = make_cmd(8, True)
  bins = torch.tensor([0, 1, 2, 3, 4, 5, 6, 6, 7])
  rets = torch.tensor([-31.4, -5.0, 0.0, 5.0, 10.0, 12.0, float("nan"), 1e3, float("-inf")])
  on.update_bin_returns(bins, rets)
  assert on.bin_G.tolist() == [0.0, 0.0, 0.0, 5.0, 10.0, 10.0, 10.0, 0.0], on.bin_G.tolist()
  assert on.bin_visits.tolist() == [1, 1, 1, 1, 1, 1, 1, 0], "non-finite returns are still dropped"
  # EMA of visited bins: -100 counts as 0, 1e3 as g_max
  a = DEFAULTS["return_ema"]
  on.update_bin_returns(torch.tensor([3, 3, 4, 0]), torch.tensor([-100.0, -50.0, 1e3, 4.0]))
  assert torch.allclose(on.bin_G[[3, 4, 0]], torch.tensor([(1 - a) * 5.0, 10.0, a * 4.0]))
  assert on.bin_visits.tolist() == [2, 1, 1, 3, 2, 1, 1, 0]
  # in-range returns: on == off, bitwise
  g = torch.Generator().manual_seed(3)
  x, y = make_cmd(1000, True), make_cmd(1000, False)
  for _ in range(10):
    bins = torch.randint(0, 1000, (3000,), generator=g)
    rets = torch.rand(3000, generator=g) * G_MAX
    x.update_bin_returns(bins, rets)
    y.update_bin_returns(bins, rets)
  assert torch.equal(x.bin_G, y.bin_G) and torch.equal(x.bin_visits, y.bin_visits)
  # any returns: the clamped EMA never leaves [0, g_max]
  for _ in range(20):
    bins = torch.randint(0, 1000, (3000,), generator=g)
    x.update_bin_returns(bins, torch.randn(3000, generator=g) * 30.0 + 5.0)
    assert float(x.bin_G.min()) >= 0.0 and float(x.bin_G.max()) <= G_MAX


# ---------------------------------------------------------------------------
# (c) checkpoint load: resume and watchdog rollback
# ---------------------------------------------------------------------------


def test_load_clamps_checkpoint_state():
  NB = 4096
  with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "model_60000.pt")
    G, V, hot = write_ckpt(path, NB)
    n_out = int(((G < 0) | (G > G_MAX)).sum())
    assert n_out == 306 and float(G.min()) < -31.0
    # off: loaded verbatim (the old behaviour), no clamp report
    off = make_trainer(tmp, NB, False)
    Trainer.load(off, path)
    assert torch.equal(off.cmd.bin_G, G) and torch.equal(off.cmd.bin_visits, V) and not clamp_msgs(off)
    # on, resume
    on = make_trainer(tmp, NB, True)
    Trainer.load(on, path)
    assert torch.equal(on.cmd.bin_G, G.clamp(0.0, G_MAX)) and torch.equal(on.cmd.bin_visits, V)
    assert on.it == 60000 and float(on.cmd.bin_G[hot]) == 0.0
    (msg,) = clamp_msgs(on)
    print(f"  resume:   {msg}")
    assert f"{n_out} of {NB}" in msg and "min before -31.400" in msg
    # sampling from the loaded state: the hot bin no longer owns the softmax share
    soft = make_cmd(NB, True, on.cmd.bin_G, V, mix_uniform=0.0, mix_softmax=1.0, mix_history=0.0).sampling_probabilities()
    assert float(soft.max()) < 0.01
    # watchdog rollback (real _rollback -> load(rollback=True)): drifted state thrown away, reloaded, clamped again
    on.cmd.bin_G.fill_(-50.0)
    on.it, on.wd_healthy_ckpt = 61000, path
    assert Trainer._rollback(on, "test") is True
    assert on.it == 60000 and on.wd_rollbacks == 1 and torch.equal(on.cmd.bin_G, G.clamp(0.0, G_MAX))
    assert on.buffer.size == 0, "rollback forgets the replay of the collapsed policy (master e57be5b)"
    msgs = clamp_msgs(on)
    print(f"  rollback: {msgs[-1]}")
    assert len(msgs) == 2 and msgs[1] == msgs[0]
    # another library (bin count differs): bins not loaded, nothing clamped / reported
    other = make_trainer(tmp, 100, True)
    Trainer.load(other, path)
    assert float(other.cmd.bin_G.abs().sum()) == 0.0 and not clamp_msgs(other)


# ---------------------------------------------------------------------------
# (d) data parallel: load + clamp + _dist_start + merges (real DistCtx.merge_sampler_bins)
# ---------------------------------------------------------------------------


def _free_port() -> int:
  with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    return s.getsockname()[1]


def _in_range(x: torch.Tensor) -> bool:
  return float(x.min()) >= 0.0 and float(x.max()) <= G_MAX


def _worker(rank: int, world: int, port: int, tmp: str):
  os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world))
  torch.set_num_threads(1)
  ctx = DistCtx.from_env("cpu", timeout_min=2.0)
  NB = 4096
  path = os.path.join(tmp, "ck.pt")
  ck_G = torch.load(path, weights_only=False)["bin_G"]
  hot = int(ck_G.argmin())
  t = make_trainer(tmp, NB, True, dist=ctx)
  t.cmd.bin_weight_hist = t.cmd.bin_weight_hist * (1.0 + 0.1 * rank)  # not in the checkpoint: only _dist_start equalizes it
  Trainer.load(t, path)  # what scripts/train.py does on every rank before run()
  assert torch.equal(t.cmd.bin_G, ck_G.clamp(0.0, G_MAX))
  assert len(clamp_msgs(t)) == (1 if rank == 0 else 0), "clamp reported once, by rank 0"
  Trainer._dist_start(t, "after load")  # rank 0's sampler everywhere + full learned-state check (raises on a difference)
  assert torch.equal(t.cmd.bin_weight_hist, torch.full((NB,), 1.0 / NB))

  # collection rounds with raw returns (collapse values included) -> clamped EMA per rank -> merge
  for rnd in range(4):
    g = torch.Generator().manual_seed(100 * rnd + rank)
    own = torch.randint(rank * 1000, rank * 1000 + 1500, (3000,), generator=g)
    bins = torch.cat([own[own != hot], torch.tensor([0, 1, 2, hot, hot])])  # the hot bin only gets the collapse returns
    rets = torch.randn(bins.numel(), generator=g) * 20.0 + 5.0
    rets[-2:] = torch.tensor([-31.4, -18.9])
    t.cmd.update_bin_returns(bins, rets)
    G0, V0 = t._bins0
    own_G, own_V = t.cmd.bin_G.clone(), t.cmd.bin_visits.clone()
    assert _in_range(own_G)
    Trainer._merge_sampler(t)
    G, V = t.cmd.bin_G, t.cmd.bin_visits
    assert _in_range(G) and float(G[hot]) == 0.0
    allV = torch.stack(ctx.all_gather_object(own_V - V0))
    allG = torch.stack(ctx.all_gather_object(own_G))
    seen = allV.sum(0) > 0
    assert torch.equal(G[~seen], G0[~seen])
    assert torch.allclose(G[seen], (allV * allG).sum(0)[seen] / allV.sum(0)[seen], rtol=1e-5, atol=1e-5)
    ctx.check_consistent({"sampler": [G, V, t.cmd.bin_weight_hist]}, tag=f"merge {rnd}")

  # extreme clamped per-rank states: exactly 0 / g_max / anything between, up to 10M prior visits, <= 5000 new ones
  g0 = torch.Generator().manual_seed(7)  # same G0 / V0 on every rank (the previous merge)
  V0 = torch.randint(0, 10_000_000, (NB,), generator=g0).float()
  G0 = torch.rand(NB, generator=g0) * G_MAX
  gr = torch.Generator().manual_seed(1000 + rank)
  kind = torch.randint(0, 3, (NB,), generator=gr)
  Gr = torch.where(kind == 0, torch.zeros(NB), torch.where(kind == 1, torch.full((NB,), G_MAX), torch.rand(NB, generator=gr) * G_MAX))
  dV = torch.randint(0, 5000, (NB,), generator=gr).float() * (torch.rand(NB, generator=gr) < 0.8)
  t.cmd.bin_G.copy_(Gr)
  t.cmd.bin_visits.copy_(V0 + dV)
  t._bins0 = (G0.clone(), V0.clone())
  Trainer._merge_sampler(t)
  G = t.cmd.bin_G
  assert _in_range(G)
  alldV = torch.stack(ctx.all_gather_object(dV))
  allG = torch.stack(ctx.all_gather_object(Gr))
  vis = alldV > 0
  at_max = vis.any(0) & ((allG == G_MAX) | ~vis).all(0)
  at_zero = vis.any(0) & ((allG == 0.0) | ~vis).all(0)
  assert int(at_max.sum()) > 0 and int(at_zero.sum()) > 0
  assert bool((G[at_max] == G_MAX).all()) and bool((G[at_zero] == 0.0).all()), "no ulp excursion at the bounds"
  ctx.check_consistent({"sampler": [G, t.cmd.bin_visits, t.cmd.bin_weight_hist]}, tag="merge extreme")
  ctx.barrier()
  ctx.close()
  with open(os.path.join(tmp, f"rank{rank}.ok"), "w") as f:
    f.write("ok")


def _run(world: int):
  with tempfile.TemporaryDirectory() as tmp:
    write_ckpt(os.path.join(tmp, "ck.pt"), 4096)
    mp.spawn(_worker, args=(world, _free_port(), tmp), nprocs=world, join=True)
    for r in range(world):
      assert os.path.exists(os.path.join(tmp, f"rank{r}.ok")), f"rank {r} did not finish"


def test_ddp_load_and_merge_world2():
  _run(2)


def test_ddp_load_and_merge_world3():
  _run(3)


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


class _Probe(Exception):
  pass


def _fake_base_cfg(robot: str, variant: str = ""):
  ns = types.SimpleNamespace
  return ns(
    scene=ns(num_envs=0, sensors=("self_collision",)), auto_reset=True,
    commands={"motion": ns(pose_range={"x": (-0.05, 0.05)}, velocity_range={"x": (-0.5, 0.5)}, joint_position_range=(-0.1, 0.1),
                           anchor_body_name="torso_link", body_names=("pelvis", "torso_link"))},
    observations={}, events={"push_robot": None, "base_com": ns(mode="startup"), "encoder_bias": ns(mode="startup"),
                             "foot_friction": ns(mode="startup", params={"ranges": (0.3, 1.6)})},
    rewards={"joint_limit": ns(weight=-10.0), "self_collisions": None},
    terminations={"ee_body_pos": ns(params={"body_names": ("left_ankle_roll_link",), "threshold": 0.0})}, episode_length_s=0.0,
  )


def test_flag_wiring():
  with tempfile.TemporaryDirectory() as tmp:
    # scripts/train.py: --sampler_clamp_returns -> TrainCfg
    spec = importlib.util.spec_from_file_location("_train_script", os.path.join(ROOT, "scripts", "train.py"))
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    seen: list[TrainCfg] = []

    def fake_trainer(cfg, dist=None):
      seen.append(cfg)
      raise _Probe

    with mock.patch.object(trainer_mod, "Trainer", fake_trainer), mock.patch.object(motions_mod, "collect_motion_files", lambda *a, **k: ["/x/a.npz"]):
      for extra in ([], ["--sampler_clamp_returns"]):
        argv = ["train.py", "--motion_dirs", "/x", "--log_root", tmp, "--device", "cpu", "--run_name", "wiring"] + extra
        with mock.patch.object(sys, "argv", argv):
          try:
            script.main()
          except _Probe:
            pass
    assert [c.sampler_clamp_returns for c in seen] == [False, True]
    # Trainer.__init__: TrainCfg -> train_cfg.json and make_env_cfg(clamp_returns=...)
    calls: list[dict] = []

    def fake_make_env_cfg(*a, **k):
      calls.append(k)
      raise _Probe

    for v in (False, True):
      log_dir = os.path.join(tmp, f"init_{v}")
      with mock.patch.object(trainer_mod, "make_env_cfg", fake_make_env_cfg):
        try:
          Trainer(TrainCfg(log_dir=log_dir, device="cpu", sampler_clamp_returns=v))
        except _Probe:
          pass
      with open(os.path.join(log_dir, "train_cfg.json")) as f:
        assert json.load(f)["sampler_clamp_returns"] is v
    assert [k["clamp_returns"] for k in calls] == [False, True]
  # make_env_cfg -> LibraryMotionCommandCfg.clamp_returns (train and eval; default off)
  with mock.patch.object(env_cfg_mod, "base_tracking_cfg", _fake_base_cfg):
    for mode in ("train", "eval"):
      assert env_cfg_mod.make_env_cfg(["a.npz"], 4, mode=mode).commands["motion"].clamp_returns is False
      for v in (False, True):
        c = env_cfg_mod.make_env_cfg(["a.npz"], 4, mode=mode, clamp_returns=v).commands["motion"]
        assert isinstance(c, LibraryMotionCommandCfg) and c.clamp_returns is v and c.g_max == G_MAX


if __name__ == "__main__":
  if STUBBED:
    print(f"  (stubbed via sys.modules: {', '.join(STUBBED)})")
  for name, fn in list(globals().items()):
    if name.startswith("test_") and callable(fn):
      fn()
      print(f"ok  {name}")
