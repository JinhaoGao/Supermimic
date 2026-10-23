"""Watchdog behaviour around non-finite iterations (single process, CPU; needs the trainer's imports = mjlab venv).

  python tests/test_nan_guard.py      # or: pytest tests/test_nan_guard.py

Reproduces the 8-GPU smoke of 2026-10-22 (imagined reward NaN from it 118; model_200 saved with NaN weights was
taken as healthy and restored six times -> "giving up") on a stand-in trainer:
  * a non-finite iteration is a transient: no rollback before wd_nf_patience consecutive ones;
  * without a healthy checkpoint a long non-finite streak never raises (keeps skipping);
  * a checkpoint written during a streak, or with non-finite weights, is never marked healthy;
  * with a healthy checkpoint the streak ends in one rollback (counted), and finite iterations reset the streak.
"""

from __future__ import annotations

import os
import sys
import types
from collections import deque

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from supermimic.distributed import DistCtx  # noqa: E402
from supermimic.trainer import Trainer, TrainCfg  # noqa: E402
from supermimic.world_model import mlp  # noqa: E402


def stand_in(patience: int = 5, healthy_ckpt: str | None = None) -> Trainer:
  import tempfile

  t = object.__new__(Trainer)
  t.cfg = TrainCfg(wd_nf_patience=patience, wd_warmup=10, log_dir=tempfile.mkdtemp(prefix="nan_guard_"))
  t.dist = DistCtx(device="cpu")
  t.main = True
  t._print = lambda *a, **k: None
  t.it = 0
  t.policy, t.critic, t.wm = mlp(3, (4,), 2), mlp(3, (4,), 1), mlp(3, (4,), 3)
  t.guard_skips = {"wm": 0, "pi": 0, "critic": 0}
  t.guard_nf_iters = 0
  t.wd_reward_ema = None
  t.wd_hist = deque()
  t.wd_vloss = deque(maxlen=2000)
  t.wd_healthy_ckpt = healthy_ckpt
  t.wd_rollbacks = 0
  t.wd_quiet_until = t.cfg.wd_warmup
  t.wd_forced = False
  t.wd_nf_streak = 0
  t.loads = []
  t.load = lambda path, rollback=False: t.loads.append((path, rollback))  # instance attr shadows the method
  t._set_lr_scale = lambda s: None
  t.lr_scale = 1.0
  t.writer = types.SimpleNamespace(add_scalar=lambda *a, **k: None)
  t.buffer = types.SimpleNamespace(size=77)  # replay stand-in (a rollback empties it, master e57be5b)
  t.opt_pi = torch.optim.Adam(t.policy.parameters(), lr=1e-3)
  return t


def run(t: Trainer, rewards, skipped=0):
  out = []
  for r in rewards:
    t.it += 1
    out.append(t._watchdog({"pi/imagined_reward_norm": r, "critic/loss": 0.1}, skipped))
  return out


def test_transient_nonfinite_is_skipped():
  t = stand_in(patience=5)
  run(t, [0.3 + 0.001 * i for i in range(30)])
  assert t._healthy()
  rb = run(t, [float("nan")] * 4)
  assert rb == [False] * 4 and t.wd_nf_streak == 4 and t.guard_nf_iters == 4
  assert not t._healthy(), "no healthy checkpoint while the iteration is non-finite (EMA is frozen)"
  run(t, [0.33])
  assert t.wd_nf_streak == 0 and t._healthy()
  # skipped updates with a finite reward count too
  run(t, [0.33], skipped=2)
  assert t.wd_nf_streak == 1 and not t._healthy()


def test_no_checkpoint_never_gives_up():
  """The smoke failure mode without a rollback target: keep skipping, never raise."""
  t = stand_in(patience=5)
  run(t, [0.3] * 20)
  rb = run(t, [float("nan")] * 200)
  assert not any(rb) and t.wd_rollbacks == 0 and not t.loads
  assert t.guard_nf_iters == 200


def test_nonfinite_weights_are_not_healthy():
  t = stand_in()
  run(t, [0.3] * 20)
  assert t._healthy()
  with torch.no_grad():
    next(t.wm.parameters()).view(-1)[0] = float("nan")
  assert not t._healthy(), "a checkpoint with NaN weights must never become the rollback target"


def test_patience_then_rollback(tmp_path=None):
  import tempfile

  with tempfile.NamedTemporaryFile(suffix=".pt") as f:
    t = stand_in(patience=5, healthy_ckpt=f.name)
    run(t, [0.3] * 20)
    rb = run(t, [float("nan")] * 4)
    assert not any(rb) and not t.loads
    rb = run(t, [float("nan")])
    assert rb == [True] and t.loads == [(f.name, True)] and t.wd_rollbacks == 1 and t.wd_nf_streak == 0
    assert t.buffer.size == 0
    # after the rollback a new streak needs the full patience again
    assert not any(run(t, [float("nan")] * 4)) and t.wd_rollbacks == 1


def test_finite_path_unchanged():
  """Finite iterations: the EMA / drop logic is the one from before the guard."""
  t = stand_in()
  run(t, [0.5] * 50)
  assert abs(t.wd_reward_ema - 0.5) < 1e-12 and t.wd_nf_streak == 0 and t.guard_nf_iters == 0
  assert run(t, [0.5]) == [False]


if __name__ == "__main__":
  for name, fn in list(globals().items()):
    if name.startswith("test_") and callable(fn):
      fn()
      print(f"ok  {name}")
