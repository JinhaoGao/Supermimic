"""Watchdog reward floor (TrainCfg.wd_reward_floor / --wd_reward_floor); CPU, torch only.

  python tests/test_watchdog_floor.py      # or: pytest tests/test_watchdog_floor.py

bj31 g1v2_ddp8_ws86k, 2026-10-23 17:50 / 17:56: one rank's imagined reward came back finite but astronomically
negative (rank 0 logged 0.918 while the rank-mean EMA hit -2.2 at it 100808 -> rollback #1; the EMA restarted after it
read -47392 at the end of the 500-iteration warm-up -> rollback #2 at 100500). Weights were fine, but every rollback
halves the LR and the sixth one kills the run. With the floor each rank clamps its finite r before the rank mean / EMA.

mjlab / tensorboard are stubbed like in test_sampler_clamp.py when missing; the REAL Trainer._watchdog / _rollback /
_guard_scalars and DistCtx run here on stand-in trainers (no simulator). Checks:
  (a) floor None -> decisions, EMA (bitwise), window, counters and log lines equal to the watchdog before the floor
      (df670e6, copied verbatim below) on a recorded sequence with the incident spikes, NaN / skipped streaks, a critic
      loss blow-up, a collapse and the give-up RuntimeError; same TB guard scalars;
  (b) floor -1: a single -1e9 spike, and a spike as the first EMA value right after a rollback (+ a second one in the
      warm-up, the incident's rollback #2), trigger nothing (they do without the floor); log rate-limited, TB count;
  (c) a genuine collapse (0.92 -> 0.4 over 400 iterations, 10-22) still rolls back, at the same iteration as before;
  (d) data parallel, 3 gloo processes, only one rank sees -1e9: every rank takes the same decisions with the same EMA,
      the reduced mean equals the closed form (sum of floored values / 3), guard scalars equal on every rank, logged
      once by rank 0 with the raw value; floor None reproduces the incident and equals the old code bitwise; NaN on
      one rank still takes the non-finite path everywhere; a collapse still rolls back on every rank together;
  (e) --wd_reward_floor -> TrainCfg (scripts/train.py main) -> train_cfg.json (Trainer.__init__), default None.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import tempfile
import types
from collections import deque
from unittest import mock

import numpy as np
import torch
import torch.multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)

import test_sampler_clamp as tsc  # noqa: E402  (installs the mjlab / tensorboard stubs when needed)

import supermimic.motions as motions_mod  # noqa: E402
import supermimic.trainer as trainer_mod  # noqa: E402
from supermimic.distributed import DistCtx  # noqa: E402
from supermimic.trainer import TrainCfg, Trainer  # noqa: E402
from supermimic.world_model import mlp  # noqa: E402

CKPT_IT = 500  # iteration of the stand-in's healthy checkpoint (a rollback resets the counter to it)


# ---------------------------------------------------------------------------
# The watchdog before the floor (df670e6), verbatim
# ---------------------------------------------------------------------------


def _watchdog_before(self, m_pi: dict, skipped: int = 0) -> bool:
  """trainer.Trainer._watchdog at df670e6 (before wd_reward_floor), verbatim."""
  cfg = self.cfg
  r = float(m_pi.get("pi/imagined_reward_norm", float("nan")))
  vl = float(m_pi.get("critic/loss", float("nan")))
  if self.dist.enabled:  # data parallel: every rank decides on the same rank means -> rolls back together
    g = self.dist.reduce_scalars({"r": r, "vl": vl}, "mean", check_keys=False)
    r, vl = g["r"], g["vl"]
  if skipped or not math.isfinite(r):
    # Transient: the non-finite updates were skipped, the weights are intact. Only a long streak means the
    # weights / normalizers themselves are broken -> roll back (no checkpoint yet: keep skipping, never give up).
    self.guard_nf_iters += 1
    self.wd_nf_streak += 1
    if self.wd_nf_streak <= 3 or self.wd_nf_streak % 100 == 0:
      sk = ", ".join(f"{k} {v}" for k, v in self.guard_skips.items())
      self._print(f"[guard] it {self.it}: non-finite (imagined reward {r:.4g}, {skipped} updates skipped; streak {self.wd_nf_streak}, total skips {sk})", flush=True)
    if not cfg.wd_enabled or self.wd_nf_streak < cfg.wd_nf_patience:
      return False
    self.wd_nf_streak = 0
    return self._rollback(f"non-finite for {cfg.wd_nf_patience} consecutive iterations")
  self.wd_nf_streak = 0
  if not cfg.wd_enabled:
    return False
  if math.isfinite(vl):
    self.wd_vloss.append(vl)
  a = cfg.wd_ema
  self.wd_reward_ema = r if self.wd_reward_ema is None else (1 - a) * self.wd_reward_ema + a * r
  self.wd_hist.append((self.it, self.wd_reward_ema))
  while self.wd_hist and self.wd_hist[0][0] < self.it - cfg.wd_window:
    self.wd_hist.popleft()
  if cfg.wd_force_at >= 0 and self.it >= cfg.wd_force_at and not self.wd_forced:
    self.wd_forced = True
    return self._rollback(f"forced at {self.it} (debug)")
  if self.it < self.wd_quiet_until:
    return False
  best = max(e for _, e in self.wd_hist)
  if self.wd_reward_ema < best - cfg.wd_drop:
    return self._rollback(f"imagined reward EMA {self.wd_reward_ema:.3f} < best {best:.3f} - {cfg.wd_drop}")
  if len(self.wd_vloss) >= 500:
    recent = float(np.median(list(self.wd_vloss)[-50:]))
    base = float(np.median(self.wd_vloss))
    if recent > cfg.wd_vloss_ratio * max(base, 1e-6):
      return self._rollback(f"critic loss median {recent:.3g} > {cfg.wd_vloss_ratio} x {base:.3g}")
  return False


def _guard_scalars_before(self) -> dict:
  """The guard block of Trainer.train at df670e6 (inline there), verbatim."""
  scal = {}
  if self.wd_reward_ema is not None:
    scal["guard/reward_ema"] = self.wd_reward_ema  # same on every rank (watchdog runs on rank means)
  scal["guard/rollbacks_total"] = self.wd_rollbacks
  for k, v in self.guard_skips.items():
    scal[f"guard/skips_{k}"] = v  # cumulative, same on every rank
  scal["guard/nonfinite_iters"] = self.guard_nf_iters
  return scal


# ---------------------------------------------------------------------------
# Stand-in trainer (real watchdog / rollback code, production watchdog defaults)
# ---------------------------------------------------------------------------


def stand_in(floor: float | None, dist: DistCtx | None = None, ckpt: str | None = None, before: bool = False, **cfg_kw) -> Trainer:
  t = object.__new__(Trainer)
  t.cfg = TrainCfg(log_dir=tempfile.mkdtemp(prefix="wd_floor_"), wd_reward_floor=floor, **cfg_kw)
  t.dist = dist if dist is not None else DistCtx(device="cpu")
  t.main = t.dist.is_main
  t.prints = []
  t._print = (lambda *a, **k: t.prints.append(" ".join(str(x) for x in a))) if t.main else (lambda *a, **k: None)
  t.it = 0
  torch.manual_seed(0)
  t.policy, t.critic, t.wm = mlp(3, (4,), 2), mlp(3, (4,), 1), mlp(3, (4,), 3)
  t.opt_pi = torch.optim.Adam(t.policy.parameters(), lr=1e-3)
  # Trainer.__init__ (NaN guard + watchdog state)
  t.guard_skips = {"wm": 0, "pi": 0, "critic": 0}
  t.guard_nf_iters = 0
  t.wd_reward_ema = None
  t.wd_hist = deque()
  t.wd_vloss = deque(maxlen=2000)
  t.wd_rollbacks = 0
  t.wd_quiet_until = t.cfg.wd_warmup
  t.wd_forced = False
  t.wd_nf_streak = 0
  t._es_on = False  # early stop off (Trainer._rollback -> _es_rollback)
  if before:  # the old code had no floor counters
    t._watchdog = types.MethodType(_watchdog_before, t)
  else:
    t.guard_floored = 0
    t.guard_floor_iters = 0
  if ckpt is None:
    ckpt = os.path.join(t.cfg.log_dir, f"model_{CKPT_IT}.pt")
    open(ckpt, "w").close()
  t.wd_healthy_ckpt = ckpt
  t.loads = []

  def load(path, rollback=False):  # Trainer.load stand-in: the iteration counter goes back to the checkpoint's
    t.loads.append((path, rollback))
    t.it = CKPT_IT

  t.load = load
  t.lr_scale = 1.0
  t._set_lr_scale = lambda s: setattr(t, "lr_scale", s)
  t._dist_start = lambda tag="": None
  t.writer = types.SimpleNamespace(add_scalar=lambda *a, **k: None)
  t.buffer = types.SimpleNamespace(size=77)
  return t


def drive(t: Trainer, seq) -> list:
  """Feed (imagined reward, critic loss, skipped updates) per iteration as Trainer.train does; trace every decision."""
  out = []
  for r, vl, sk in seq:
    t.it += 1
    try:
      d = t._watchdog({"pi/imagined_reward_norm": r, "critic/loss": vl}, sk)
    except RuntimeError as e:
      out.append(("raise", str(e)))
      break
    e = t.wd_reward_ema
    out.append((t.it, d, None if e is None else e.hex(), len(t.wd_hist), t.wd_hist[-1][1].hex() if t.wd_hist else None,
                t.wd_quiet_until, t.wd_rollbacks, t.wd_nf_streak, t.guard_nf_iters, len(t.wd_vloss), t.lr_scale))
  return out


def reasons(t: Trainer) -> list[str]:
  return [s for s in t.prints if s.startswith("[watchdog] it ")]


def floor_msgs(t: Trainer) -> list[str]:
  return [s for s in t.prints if "imagined reward floored" in s]


def normal(n: int, g: np.random.Generator, mu: float = 0.918, sd: float = 0.004, vl: float = 0.05) -> list:
  return [(float(mu + sd * g.standard_normal()), float(vl * (1.0 + 0.2 * g.standard_normal())), 0) for _ in range(n)]


def drift(n: int, g: np.random.Generator, r0: float = 0.92, r1: float = 0.4) -> list:
  """10-22 collapse: imagined reward 0.91 -> 0.4 in ~400 iterations."""
  return [(float(r0 + (r1 - r0) * i / (n - 1) + 0.004 * g.standard_normal()), 0.05, 0) for i in range(n)]


def recorded_sequence() -> list:
  """The 10-23 incident shape plus every other watchdog path, in the order a run could meet them."""
  g = np.random.default_rng(1023)
  nan = float("nan")
  s = normal(808, g)
  s += [(-155.0, 0.05, 0)]  # rank-mean spike: EMA -2.2 as at incident it 100808 -> rollback #1
  s += [(-5.0e6, 0.05, 0)] + normal(300, g) + [(-1.0e9, 0.05, 0)] + normal(400, g)  # first EMA value a spike, another
  # in the warm-up -> EMA still ~ -1e5 at its end -> rollback #2 (incident: -47392 at 100500)
  s += normal(5, g) + [(nan, 0.05, 0)] * 3 + [(0.91, 0.05, 2)] * 2 + normal(5, g)  # transient non-finite / skipped
  s += [(float("-inf"), 0.05, 0)] + [(nan, nan, 1)] * 105 + normal(700, g)  # long streak -> rollback #3
  s += [(0.918, 5e2, 0)] * 60 + normal(600, g)  # critic loss x10^4 -> rollback #4
  s += drift(400, g) + [(0.4, 0.05, 0)] * 50 + normal(800, g)  # collapse -> rollback #5
  s += [(-3.0e3, 0.05, 0)] + normal(700, g)  # spike -> rollback #6
  s += [(-2.0e3, 0.05, 0)] + normal(10, g)  # spike -> 6 rollbacks already: RuntimeError (run killed)
  return s


# ---------------------------------------------------------------------------
# (a) floor None == before, bitwise
# ---------------------------------------------------------------------------


def test_default_is_none():
  assert TrainCfg().wd_reward_floor is None


def test_off_is_bitwise_identical_to_before():
  seq = recorded_sequence()
  old, new = stand_in(None, before=True), stand_in(None)
  tr_old, tr_new = drive(old, seq), drive(new, seq)
  assert tr_new == tr_old and new.prints == old.prints and new.lr_scale == old.lr_scale
  assert [x[1] for x in new.loads] == [x[1] for x in old.loads] == [True] * 6
  assert reasons(new)[0].startswith("[watchdog] it 809: imagined reward EMA -2.200 < best")
  assert new.guard_floored == 0 and new.guard_floor_iters == 0 and not floor_msgs(new)
  # the recorded sequence reaches every path: 6 rollbacks of 4 kinds, then the give-up
  kinds = [r.split(": ", 1)[1].split(" ")[0] for r in reasons(new)]
  assert kinds == ["imagined", "imagined", "non-finite", "critic", "imagined", "imagined", "imagined"], kinds
  assert tr_new[-1][0] == "raise" and new.wd_rollbacks == 6
  print(f"  {len(tr_new)} iterations, {new.wd_rollbacks} rollbacks + give-up: decisions / EMA / log identical")
  # TB guard scalars: same dict (keys, order, values) as the inline block of before
  for t in (stand_in(None), new):
    assert list(Trainer._guard_scalars(t).items()) == list(_guard_scalars_before(t).items())
  assert "guard/reward_floored" not in Trainer._guard_scalars(new)


def test_on_recorded_sequence():
  """Same recording with the floor: the four spike rollbacks (and the give-up) are gone, the real ones stay."""
  t = stand_in(-1.0)
  tr = drive(t, recorded_sequence())
  kinds = [r.split(": ", 1)[1].split(" ")[0] for r in reasons(t)]
  assert kinds == ["non-finite", "critic", "imagined"], reasons(t)
  assert tr[-1][0] != "raise" and t.wd_rollbacks == 3
  assert t.guard_floored == 5 and t.guard_nf_iters == 3 + 2 + 1 + 105, "-inf / NaN: non-finite path, never floored"
  print(f"  with floor -1: {t.wd_rollbacks} rollbacks ({', '.join(kinds)}), {t.guard_floored} values floored")


# ---------------------------------------------------------------------------
# (b) floor -1: spikes do not roll back
# ---------------------------------------------------------------------------


def test_single_spike():
  g = np.random.default_rng(1)
  seq = normal(700, g) + [(-1e9, 0.05, 0)] + normal(1500, g)
  off, on = stand_in(None), stand_in(-1.0)
  tr_off, tr_on = drive(off, seq), drive(on, seq)
  # without the floor: the incident, rollback right at the spike
  assert tr_off[700][1] is True and off.wd_rollbacks == 1 and "imagined reward EMA" in reasons(off)[0]
  # with it: nothing; identical up to the spike, then one bounded EMA dip
  assert tr_on[:700] == tr_off[:700]
  assert on.wd_rollbacks == 0 and not any(x[1] for x in tr_on) and not on.loads
  a = on.cfg.wd_ema
  e0, e1 = float.fromhex(tr_on[699][2]), float.fromhex(tr_on[700][2])
  assert e1 == (1 - a) * e0 + a * -1.0 and e0 - e1 < 0.04
  assert on.guard_floored == 1 and on.guard_floor_iters == 1
  (msg,) = floor_msgs(on)
  print(f"  {msg}")
  assert msg.startswith("[guard] it 701: imagined reward floored to -1 on 1/1 rank(s) (raw: rank 0 -1e+09)")
  assert Trainer._guard_scalars(on)["guard/reward_floored"] == 1


def test_spike_after_rollback():
  """Rollback #1 (forced here), then the first EMA value is a spike and another one comes inside the warm-up."""
  g = np.random.default_rng(2)
  seq = normal(600, g) + [(-5e6, 0.05, 0)] + normal(300, g) + [(-1e9, 0.05, 0)] + normal(1500, g)
  off, on = stand_in(None, wd_force_at=600), stand_in(-1.0, wd_force_at=600)
  tr_off, tr_on = drive(off, seq), drive(on, seq)
  for t in (off, on):
    assert "forced at 600" in reasons(t)[0] and t.loads[0] == (t.wd_healthy_ckpt, True)
  assert tr_on[:600] == tr_off[:600] and tr_on[599][:2] == (CKPT_IT, True)
  # first value after the rollback (it CKPT_IT + 1): becomes the EMA as is
  assert tr_off[600][0] == CKPT_IT + 1 and float.fromhex(tr_off[600][2]) == -5e6
  assert tr_on[600][0] == CKPT_IT + 1 and float.fromhex(tr_on[600][2]) == -1.0
  # without the floor: rollback #2 exactly at the end of the warm-up (incident: 100500, EMA -47392)
  rb2 = [x for x in tr_off if x[1]][1]
  assert off.wd_rollbacks >= 2 and rb2[0] == CKPT_IT and "imagined reward EMA" in reasons(off)[1]
  print(f"  no floor: {reasons(off)[1]}")
  # with the floor: only the forced one
  assert on.wd_rollbacks == 1 and sum(x[1] for x in tr_on) == 1 and on.guard_floored == 2
  assert abs(on.wd_reward_ema - 0.918) < 0.01


def test_log_rate_limited():
  g = np.random.default_rng(3)
  seq = []
  for _ in range(250):
    seq += normal(19, g) + [(-1e6 * (1 + g.random()), 0.05, 0)]
  t = stand_in(-1.0)
  tr = drive(t, seq)
  assert t.wd_rollbacks == 0 and not any(x[1] for x in tr)
  assert t.guard_floored == 250 and t.guard_floor_iters == 250
  msgs = floor_msgs(t)
  assert [m.split("; ")[1].split(" ")[0] for m in msgs] == ["1", "2", "3", "100", "200"], msgs
  assert Trainer._guard_scalars(t)["guard/reward_floored"] == 250


# ---------------------------------------------------------------------------
# (c) a genuine collapse still rolls back
# ---------------------------------------------------------------------------


def test_collapse_still_rolls_back():
  g = np.random.default_rng(4)
  seq = normal(2000, g) + drift(400, g) + [(0.4, 0.05, 0)] * 600
  off, on = stand_in(None), stand_in(-1.0)
  tr_off, tr_on = drive(off, seq), drive(on, seq)
  assert tr_on == tr_off and on.guard_floored == 0, "values above the floor: same trajectory, bitwise"
  i = next(k for k, x in enumerate(tr_on) if x[1])
  assert 2000 < i < 2400 and "imagined reward EMA" in reasons(on)[0]  # (the recording keeps collapsing afterwards)
  print(f"  collapse rolled back at drift step {i + 1 - 2000} of 400 ({reasons(on)[0]})")
  # collapse with spikes mixed in: still caught, no later than without them
  seq2 = list(seq)
  for k in range(2050, 2400, 70):
    seq2[k] = (-1e9, 0.05, 0)
  on2 = stand_in(-1.0)
  tr2 = drive(on2, seq2)
  i2 = next(k for k, x in enumerate(tr2) if x[1])
  assert i2 <= i and on2.guard_floored >= 1


# ---------------------------------------------------------------------------
# (d) data parallel: 3 gloo processes
# ---------------------------------------------------------------------------

SPIKE_AT = 701  # 1-based step of the spike


def _rank_seq(rk: int, spike_rank: int, n: int = 2200) -> list:
  """Per-rank recording (any rank can rebuild every rank's: closed form)."""
  s = normal(n, np.random.default_rng(100 + rk))
  if rk == spike_rank:
    s[SPIKE_AT - 1] = (-1e9, s[SPIKE_AT - 1][1], 0)
  return s


def _ddp_worker(rank: int, world: int, port: int, tmp: str):
  os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world))
  torch.set_num_threads(1)
  ctx = DistCtx.from_env("cpu", timeout_min=2.0)
  ck = os.path.join(tmp, f"model_{CKPT_IT}.pt")
  spike_rank = world - 1
  seqs = [_rank_seq(k, spike_rank) for k in range(world)]
  spy = []
  reduce0 = ctx.reduce_scalars

  def spy_reduce(s, op="mean", check_keys=True):
    out = reduce0(s, op, check_keys)
    spy.append((dict(s), dict(out)))
    return out

  ctx.reduce_scalars = spy_reduce

  def same_everywhere(x, what):
    xs = ctx.all_gather_object(x)
    assert all(v == xs[0] for v in xs), f"{what} differs between ranks"

  # ---- floor -1, only the last rank sees -1e9 ----
  t = stand_in(-1.0, dist=ctx, ckpt=ck)
  tr = drive(t, seqs[rank])
  same_everywhere(tr, "watchdog trace (decisions / EMA / counters)")
  assert t.wd_rollbacks == 0 and not any(x[1] for x in tr) and not t.loads
  inp, out = spy[SPIKE_AT - 1]
  assert sorted(inp) == ["floored", "r", "vl"]
  if rank == spike_rank:
    assert inp["r"] == -1.0 and inp["floored"] == 1.0
  else:
    assert inp["r"] == seqs[rank][SPIKE_AT - 1][0] and inp["floored"] == 0.0
  closed = sum(max(seqs[k][SPIKE_AT - 1][0], -1.0) for k in range(world)) / world
  assert abs(out["r"] - closed) <= 1e-12, (out["r"], closed)
  assert round(out["floored"] * world) == 1
  a = t.cfg.wd_ema
  e0, e1 = float.fromhex(tr[SPIKE_AT - 2][2]), float.fromhex(tr[SPIKE_AT - 1][2])
  assert e1 == (1 - a) * e0 + a * out["r"], "EMA takes the floored rank mean"
  assert t.guard_floored == 1 and t.guard_floor_iters == 1
  gs = Trainer._guard_scalars(t)
  same_everywhere(list(gs.items()), "guard scalars")
  assert gs["guard/reward_floored"] == 1
  if rank == 0:
    (msg,) = floor_msgs(t)
    print(f"  rank 0: {msg}")
    assert f"it {SPIKE_AT}: imagined reward floored to -1 on 1/{world} rank(s) (raw: rank {spike_rank} -1e+09)" in msg
    assert abs(t.wd_reward_ema - 0.918) < 0.01
  else:
    assert t.prints == []

  # ---- floor None: the incident (every rank rolls back at the spike), bitwise the old code ----
  spy.clear()
  off = stand_in(None, dist=ctx, ckpt=ck)
  tr_off = drive(off, seqs[rank])
  assert all(sorted(i) == ["r", "vl"] for i, _ in spy), "off: the reduce of before"
  old = stand_in(None, dist=ctx, ckpt=ck, before=True)
  tr_old = drive(old, seqs[rank])
  assert tr_off == tr_old and off.prints == old.prints
  assert tr_off[SPIKE_AT - 1][1] is True and off.wd_rollbacks >= 1
  same_everywhere(tr_off, "trace without the floor")

  # ---- floor -1, NaN on rank 0 while the last rank spikes: non-finite path everywhere, unchanged ----
  s = list(seqs[rank][:800])
  s[SPIKE_AT - 1] = (float("nan"), 0.05, 0) if rank == 0 else s[SPIKE_AT - 1]
  t2 = stand_in(-1.0, dist=ctx, ckpt=ck)
  tr2 = drive(t2, s)
  same_everywhere(tr2, "trace with a NaN rank")
  x = tr2[SPIKE_AT - 1]
  assert x[1] is False and x[7] == 1 and x[8] == 1, "one non-finite iteration (streak 1), no rollback"
  assert tr2[SPIKE_AT - 2][2] == x[2], "EMA frozen on the non-finite iteration"
  assert t2.guard_floored == 1 and t2.wd_rollbacks == 0

  # ---- floor -1, a real collapse on every rank: rolls back together ----
  g = np.random.default_rng(200 + rank)
  t3 = stand_in(-1.0, dist=ctx, ckpt=ck)
  tr3 = drive(t3, normal(2000, g) + drift(400, g) + [(0.4, 0.05, 0)] * 300)
  same_everywhere(tr3, "trace of the collapse")
  i = next(k for k, x in enumerate(tr3) if x[1])
  assert 2000 < i < 2400 and t3.wd_rollbacks >= 1 and t3.guard_floored == 0

  ctx.barrier()
  ctx.close()
  with open(os.path.join(tmp, f"rank{rank}.ok"), "w") as f:
    f.write("ok")


def _run_ddp(world: int):
  with tempfile.TemporaryDirectory() as tmp:
    open(os.path.join(tmp, f"model_{CKPT_IT}.pt"), "w").close()
    mp.spawn(_ddp_worker, args=(world, tsc._free_port(), tmp), nprocs=world, join=True)
    for r in range(world):
      assert os.path.exists(os.path.join(tmp, f"rank{r}.ok")), f"rank {r} did not finish"


def test_ddp_world3():
  _run_ddp(3)


# ---------------------------------------------------------------------------
# (e) wiring
# ---------------------------------------------------------------------------


def test_cli_flag_wiring():
  with tempfile.TemporaryDirectory() as tmp:
    spec = importlib.util.spec_from_file_location("_train_script", os.path.join(ROOT, "scripts", "train.py"))
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    seen: list[TrainCfg] = []

    def fake_trainer(cfg, dist=None):
      seen.append(cfg)
      raise tsc._Probe

    with mock.patch.object(trainer_mod, "Trainer", fake_trainer), mock.patch.object(motions_mod, "collect_motion_files", lambda *a, **k: ["/x/a.npz"]):
      for extra in ([], ["--wd_reward_floor", "-1.0"], ["--wd_reward_floor=-1"], ["--wd_reward_floor", "-0.5"]):
        argv = ["train.py", "--motion_dirs", "/x", "--log_root", tmp, "--device", "cpu", "--run_name", "floor"] + extra
        with mock.patch.object(sys, "argv", argv):
          try:
            script.main()
          except tsc._Probe:
            pass
    assert [c.wd_reward_floor for c in seen] == [None, -1.0, -1.0, -0.5]
    assert all(c.wd_drop == 0.2 and c.wd_enabled for c in seen)

    def fake_make_env_cfg(*a, **k):
      raise tsc._Probe

    for v in (None, -1.0):
      log_dir = os.path.join(tmp, f"init_{v}")
      with mock.patch.object(trainer_mod, "make_env_cfg", fake_make_env_cfg):
        try:
          Trainer(TrainCfg(log_dir=log_dir, device="cpu", wd_reward_floor=v))
        except tsc._Probe:
          pass
      with open(os.path.join(log_dir, "train_cfg.json")) as f:
        got = json.load(f)["wd_reward_floor"]
      assert got == v and type(got) is type(v), (got, v)


if __name__ == "__main__":
  if tsc.STUBBED:
    print(f"  (stubbed via sys.modules: {', '.join(tsc.STUBBED)})")
  for name, fn in list(globals().items()):
    if name.startswith("test_") and callable(fn):
      fn()
      print(f"ok  {name}")
