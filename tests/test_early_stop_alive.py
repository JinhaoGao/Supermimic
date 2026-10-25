"""Early stop on the in-training eval (--es_eval_fail_ratio / --es_patience) and the survival-weighted watchdog signal
(--wd_signal imagined_reward_alive); CPU, torch only.

  python tests/test_early_stop_alive.py      # or: pytest tests/test_early_stop_alive.py

10-24 diagnosis: the 8-GPU lines degrade after a peak; failures of the deterministic in-training eval (1024 fixed clips)
correlate -0.984 with v4 and rise ~28k iterations before the cliff -> stop there and keep the best-eval checkpoint.
Imagined-reward spikes come from world-model divergence after the soft termination and only reach the UNWEIGHTED
monitor pi/imagined_reward_norm (the objective is J = sum_t gamma^t c_t r_t) -> the watchdog can watch
sum_t c_t r_t / sum_t c_t instead (bj27 r16: six rollbacks on the unweighted one, LR down to 2e-8, gave up).

mjlab / tensorboard are stubbed like in test_sampler_clamp.py when missing; the REAL Trainer.train loop (incl.
_watchdog / _rollback / load / save / _dist_start / _merge_sampler / _check_consistency / _dist_log_scalars),
Trainer.update_policy / _fit_critic / _apply_grads and DistCtx run on stand-in trainers (simulator, world model,
evaluator and policy rollouts replaced by scripted stubs; a fake clock makes every logged time deterministic).
Checks:
  (a) early_stop_verdict: closed form (>= 3 previous evals, median, ratio, median floored at 1);
  (b) the loop, one process: es off == the loop before (ca9266e, copied verbatim below) -- stdout, TB scalars, saves,
      evals, collects, eval.jsonl -- with a forced watchdog rollback in between and the curriculum off / on;
      es on: stops at the closed-form iteration with patience 2, writes earlystop.json (iteration, history, best-eval
      checkpoint) and best_eval.json, saves model_final and leaves the loop; the best eval has its own checkpoint
      (model_best_eval.pt, rewritten at every new best only: a watchdog rollback that re-reaches and overwrites
      model_<it>.pt leaves it alone, and best_eval.json then no longer names model_<it>.pt); a rollback drops the
      evals of the abandoned branch from the history (no streak carried over, no inflated median); resumed at an eval
      slot, the loaded weights are evaluated before the first update and start the history / best; a failing evaluator
      just ends the verdicts; es on but never triggered = es off apart from its own lines / files;
  (c) the loop on 3 gloo processes: only rank 0 evaluates; every rank takes the same verdicts and leaves the loop at the
      same iteration (same history / best / stop record, incl. the rollback's dropped evals and the start eval of a
      resume), only rank 0 writes; the curriculum multiplier is the same on every rank at every iteration (incl. after a
      rollback); es off == the loop before on every rank;
  (d) pi/imagined_reward_alive: closed form sum c_t r_t / sum c_t on a toy imagined rollout (post-death rewards of 1e3
      do not reach it, they do reach imagined_reward_norm); update_policy is otherwise the code before, bitwise
      (parameters, metrics, RNG); default watchdog signal == the watchdog before with the new key present; with the
      alive signal it is a drop-in (same decisions on the same values), the floor applies, and the r16-like norm
      spikes no longer roll back; data parallel: the rank mean / floor act on the alive values;
  (e) wiring: --es_eval_fail_ratio / --es_patience / --wd_signal -> TrainCfg -> train_cfg.json; misuse rejected.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import json
import math
import os
import sys
import tempfile
import time
import types
from unittest import mock

import numpy as np
import torch
import torch.multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import test_sampler_clamp as tsc  # noqa: E402  (installs the mjlab / tensorboard stubs when needed)
import test_watchdog_floor as twf  # noqa: E402
from test_merge_master_1025 import _parse, hard_mult_at_master, tsc_load_train  # noqa: E402

import supermimic.trainer as trainer_mod  # noqa: E402
from supermimic.distributed import DistCtx  # noqa: E402
from supermimic.heads import HeadsCfg  # noqa: E402
from supermimic.library_command import LibraryMotionCommand, hard_clip_mask  # noqa: E402
from supermimic.trainer import TrainCfg, Trainer, early_stop_verdict, finite_mean  # noqa: E402

THIS = sys.modules[__name__]
FAILS = [100, 90, 95, 92, 200, 130, 150, 160, 300, 180]  # failed count of the k-th eval (it 4, 8, ...; see test_es_stops)


# ---------------------------------------------------------------------------
# The code before (ca9266e), verbatim
# ---------------------------------------------------------------------------

def update_policy_before(self) -> dict:
  """trainer.Trainer.update_policy at ca9266e (before the alive metric), verbatim."""
  cfg, heads, H, K = self.cfg, self.heads, self.cfg.H, self.cfg.heads.K
  N = cfg.rollouts_per_update
  g = cfg.gamma
  for p in self.wm.parameters():
    p.requires_grad_(False)
  self.wm.eval()
  st = self.buffer.sample_starts(N, cfg.heads.hist)
  xi = st["xi"]
  bias = self.sim.bias_from_xi(xi, self.nj)
  t_g0 = time.time()
  with torch.no_grad():
    hs, hp = st["hist_states"], st["hist_prev_action"]
    hist_n = hs.shape[1]
    obs_hist = heads.proprio(
      hs.reshape(-1, hs.shape[-1]), hp.reshape(-1, self.nj), bias[:, None, :].expand(N, hist_n, self.nj).reshape(-1, self.nj), True
    ).view(N, hist_n, -1)
    kin = heads.kin(st["state"])
    # Reference window pack for this batch: frames f0 .. f0+H+K-1 of every rollout.
    Lw = H + K
    pack = self.lib.gather_windows(st["frame"], Lw)
    f_loc = torch.zeros(N, dtype=torch.long, device=self.device)
    clip_end_loc = torch.clamp(st["clip_end"] - st["frame"], max=Lw)
  self._t_gather = time.time() - t_g0
  s = st["state"]
  prev_a = st["prev_action"]
  c = torch.ones(N, device=self.device)
  J = torch.zeros(N, device=self.device)
  feats_n, goals_n, rews, dsoft = [], [], [], []
  p = (pack["joint_pos"], pack["joint_vel"], pack["body_pos"], pack["body_quat"], pack["body_lin_vel"], pack["body_ang_vel"])
  for t in range(H):
    s, obs_hist, prev_a, f_loc, kin, r, d, fn, gn = self.fns.imagine_step(s, obs_hist, prev_a, f_loc, clip_end_loc, xi, bias, kin, *p)
    J = J + (g**t) * c * r
    c = c * (1.0 - d)
    feats_n.append(fn.detach())
    goals_n.append(gn.detach())
    rews.append(r.detach())
    dsoft.append(d.detach())
  goal_H = self.fns.goal_from_pack(kin, f_loc, clip_end_loc, p[0], p[1], p[2], p[3])
  fn_H, gn_H = self.fns.critic_inputs(s, goal_H)
  if cfg.use_critic:
    V_H = self.critic(fn_H, gn_H, xi)
    J = J + (g**H) * c * V_H
  loss_pi = -J.mean() / H
  self.opt_pi.zero_grad(set_to_none=True)
  loss_pi.backward()
  gn_pi = self._apply_grads("pi", self.opt_pi, list(self.policy.parameters()), cfg.grad_clip_norm, cfg.grad_clip_elem)
  for p_ in self.wm.parameters():
    p_.requires_grad_(True)

  metrics = {
    "pi/imagined_return": (J.mean() / H).item(),
    "pi/imagined_reward_norm": torch.stack(rews).mean().item() / self.cfg.heads.reward_max_per_step,
    "pi/survival_H": c.mean().item(),
    "pi/grad_norm": gn_pi,
    "pi/loss": loss_pi.item(),
    "time/ref_gather": self._t_gather,
  }
  if cfg.use_critic:
    metrics.update(self._fit_critic(feats_n, goals_n, fn_H.detach(), gn_H.detach(), xi, rews, dsoft))
  return metrics


def train_before(self) -> None:
  """trainer.Trainer.train at ca9266e (before the merge / early stop), verbatim."""
  cfg = self.cfg
  dctx = self.dist
  warmup = max(cfg.wm.k + cfg.heads.hist + 2, 32)
  t_log = time.time()
  steps_at_log = self.env_steps
  agg: dict[str, list[float]] = {}
  sync_every = cfg.sampler_sync_interval or cfg.log_interval
  if dctx.enabled:
    self._dist_start()
  while self.it < cfg.iterations:
    t0 = time.time()
    self._step_carry += cfg.env_steps_per_iter
    n_steps = int(self._step_carry)
    self._step_carry -= n_steps
    for _ in range(max(n_steps, 0)):
      self.collect_step()
    if self.buffer.size < warmup:
      for _ in range(warmup - self.buffer.size):
        self.collect_step()
      continue
    t1 = time.time()
    skips0 = sum(self.guard_skips.values())
    m_wm = self.update_world_model()
    t2 = time.time()
    m_pi = self.update_policy()
    t3 = time.time()
    for s in self.sched:
      s.step()
    self.it += 1
    if self._watchdog(m_pi, sum(self.guard_skips.values()) - skips0):
      agg = {}
      continue
    for k, v in {**m_wm, **m_pi, "time/collect": t1 - t0, "time/wm": t2 - t1, "time/policy": t3 - t2, "time/iter": t3 - t0}.items():
      agg.setdefault(k, []).append(v)

    # Same schedule on every rank (depends only on cfg and the iteration counter).
    ckpt_slot = bool(cfg.ckpt_interval) and self.it % cfg.ckpt_interval == 0
    # One early rollback point (the first regular checkpoint comes only at ckpt_interval).
    ckpt_slot |= cfg.wd_enabled and cfg.wd_early_ckpt > 0 and self.it == cfg.wd_early_ckpt and self.wd_healthy_ckpt is None
    eval_slot = bool(cfg.eval_interval) and self.it % cfg.eval_interval == 0
    check_slot = bool(cfg.param_check_interval) and self.it % cfg.param_check_interval == 0
    if dctx.enabled and (self.it % sync_every == 0 or ckpt_slot or check_slot):
      self._merge_sampler()

    if self.it % cfg.log_interval == 0:
      now = time.time()
      if dctx.enabled:
        scal = self._dist_log_scalars(agg, now - t_log, self.env_steps - steps_at_log)
        fps = scal["time/env_fps"]
      else:
        fps = (self.env_steps - steps_at_log) * self.E / max(now - t_log, 1e-6)
        scal = {k: finite_mean(v) for k, v in agg.items()}
        scal["time/env_fps"] = fps
        scal["time/iter_per_s"] = cfg.log_interval / max(now - t_log, 1e-6)
        scal["train/env_steps_per_env"] = self.env_steps
        scal["train/lr"] = self.opt_pi.param_groups[0]["lr"]
        scal["train/wall_h"] = (now - self._t_start) / 3600
        if self.stats["ep_return_norm"]:
          scal["train/ep_return_norm"] = float(np.mean(self.stats["ep_return_norm"]))
          scal["train/ep_len"] = float(np.mean(self.stats["ep_len"]))
          scal["train/ep_failed_frac"] = float(np.mean(self.stats["ep_failed"]))
        if self.stats["ep_failed_ee"]:
          scal["train/ep_failed_ee_frac"] = float(np.mean(self.stats["ep_failed_ee"]))
        scal["guard/sim_nonfinite"] = float(np.sum(self.stats["sim_nonfinite"]))
        scal["train/term_mismatch"] = float(np.mean(self.stats["term_mismatch"])) if self.stats["term_mismatch"] else 0.0
        scal["sampler/entropy"] = self.cmd.metrics["sampling_entropy"][0].item()
        scal["sampler/top1_prob"] = self.cmd.metrics["sampling_top1_prob"][0].item()
        scal["sampler/bins_visited_frac"] = (self.cmd.bin_visits > 0).float().mean().item()
      scal.update(self._guard_scalars())
      if self.main:
        for k, v in scal.items():
          self.writer.add_scalar(k, v, self.it)
        print(
          f"[it {self.it:7d} | {scal['train/wall_h']:5.2f} h] wm {scal.get('wm/loss', 0):.4f} (e1 {scal.get('wm/err_step1', 0):.3f}, e{cfg.wm.k} {scal.get(f'wm/err_step{cfg.wm.k}', 0):.3f}) "
          f"| imag_ret {scal.get('pi/imagined_return', 0):.4f} rew {scal.get('pi/imagined_reward_norm', 0):.3f} surv {scal.get('pi/survival_H', 0):.3f} | "
          f"ep_ret {scal.get('train/ep_return_norm', 0):.3f} len {scal.get('train/ep_len', 0):.0f} fail {scal.get('train/ep_failed_frac', 0):.2f}"
          + (f" (ee {scal.get('train/ep_failed_ee_frac', 0):.2f})" if self._ee_on else "") + " | "
          f"{scal['time/iter']*1000:.0f} ms/it (c {scal['time/collect']*1000:.0f} wm {scal['time/wm']*1000:.0f} pi {scal['time/policy']*1000:.0f} ref {scal.get('time/ref_gather', 0)*1000:.0f}) fps {fps:.0f}"
          + (f" | max/rank {scal['dist/iter_max']*1000:.0f} ms" if dctx.enabled else "")
          + (f" | guard skips {sum(self.guard_skips.values())} nonfinite its {self.guard_nf_iters}" if self.guard_nf_iters else ""),
          flush=True,
        )
      agg = {}
      for k in self.stats:
        self.stats[k] = []
      t_log, steps_at_log = now, self.env_steps

    if dctx.enabled and check_slot:
      self._check_consistency(f"it {self.it}")
    if ckpt_slot:
      p_ck = self.save(str(self.it)) if self.main else self._ckpt_path(str(self.it))  # all ranks track the path
      if self._healthy():
        self.wd_healthy_ckpt = p_ck
    if cfg.eval_enabled and eval_slot and self.main:
      te = time.time()
      try:
        res = self.evaluate()
        for k, v in res["scalars"].items():
          self.writer.add_scalar(f"eval/{k}", v, self.it)
        self.writer.add_scalar("time/eval", time.time() - te, self.it)
        print(f"[eval it {self.it}] return {res['scalars']['return_mean']:.4f} min {res['scalars']['return_min']:.3f} failed {int(res['scalars']['failed'])}/{res['num_clips']} ({time.time()-te:.0f}s)", flush=True)
        with open(os.path.join(cfg.log_dir, "eval.jsonl"), "a") as f:
          f.write(json.dumps({"it": self.it, "env_steps": self.env_steps, "wall_s": time.time() - self._t_start, **res["scalars"], "failed_clips": res["failed_clips"]}) + "\n")
      except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print(f"[eval] failed: {e!r}", flush=True)
        cfg.eval_enabled = False
    if dctx.enabled and (ckpt_slot or eval_slot):
      dctx.barrier()  # the other ranks wait (CPU barrier) while rank 0 saves / evaluates
  if dctx.enabled:
    self._merge_sampler()
    self._check_consistency("final")
  if self.main:
    self.save("final")
  if dctx.enabled:
    dctx.barrier()


# ---------------------------------------------------------------------------
# (a) verdict
# ---------------------------------------------------------------------------


def test_verdict_closed_form():
  assert early_stop_verdict([], 1e9, 1.5) == (False, None)
  assert early_stop_verdict([100, 90], 1e9, 1.5) == (False, None), "needs 3 previous evals"
  assert early_stop_verdict([100, 90, 95], 142.4, 1.5) == (False, 142.5)
  assert early_stop_verdict([100, 90, 95], 142.5, 1.5) == (True, 142.5), ">= is bad"
  assert early_stop_verdict([100, 90, 95, 92], 140.25, 1.5) == (True, 140.25), "even count: mean of the middle two"
  assert early_stop_verdict([0, 0, 0], 1, 1.5) == (False, 1.5) and early_stop_verdict([0, 0, 0], 2, 1.5)[0], "median floored at 1"
  assert early_stop_verdict([10, 1000, 12, 11], 18, 1.5) == (True, 17.25), "median 11.5: one outlier does not move it"
  for _ in range(200):
    g = np.random.default_rng(_)
    prev = list(g.integers(0, 400, g.integers(3, 30)))
    f, r = float(g.integers(0, 800)), float(g.uniform(1.0, 3.0))
    thr = r * max(float(np.median(prev)), 1.0)
    assert early_stop_verdict(prev, f, r) == (f >= thr, thr)


# ---------------------------------------------------------------------------
# Stand-in for the loop
# ---------------------------------------------------------------------------


class Clock:
  """Deterministic time.time() for the trainer module (every call advances 0.01 s)."""

  def __init__(self):
    self.t = 1_000_000.0

  def time(self) -> float:
    self.t += 0.01
    return self.t


def loop_trainer(log_dir: str, dist: DistCtx | None = None, fails=FAILS, eval_raises_at: int | None = None, **cfg_kw) -> Trainer:
  """Trainer for the real train loop: real watchdog / rollback / load / save / sync / checks, scripted stubs for the
  simulator (collect_step), the world model / policy updates and the evaluator."""
  NB, E = 64, 8
  t = tsc.make_trainer(log_dir, NB, False, dist=dist)
  kw = dict(log_dir=log_dir, iterations=44, log_interval=5, eval_interval=4, ckpt_interval=8, param_check_interval=10,
            sampler_sync_interval=3, wd_warmup=6, wd_early_ckpt=0, eval_enabled=True)
  kw.update(cfg_kw)
  t.cfg = dataclasses.replace(t.cfg, **kw)
  trainer_mod._check_cfg(t.cfg)
  c = t.cmd
  c.metrics = {"sampling_entropy": torch.tensor([0.93]), "sampling_top1_prob": torch.tensor([1e-4])}
  names = [f"g1__src_{i}_{'kongfan' if i % 4 == 0 else 'walk'}.npz" for i in range(16)]
  c.clip_hard = hard_clip_mask(names, t.cfg.hard_names)
  c.env_hard = c.clip_hard[torch.arange(E) % 16]
  c.hard_mult, c.term_scale_env = 1.0, torch.ones(E)
  c.set_hard_mult = types.MethodType(LibraryMotionCommand.set_hard_mult, c)
  t.heads = types.SimpleNamespace(proprio_dim=3, goal_frame_dim=2)  # real Trainer.save
  t.nj, t.xi_dim, t.E = 2, 1, E
  t.env_steps, t._step_carry, t._t_start = 0, 0.0, 0.0
  t.buffer = types.SimpleNamespace(size=1000)
  t.stats = {"ep_return_norm": [], "ep_len": [], "ep_failed": [], "ep_failed_ee": [], "term_mismatch": [], "sim_nonfinite": [],
             "hard_ep_failed": [], "hard_ep_len": []}
  t._ee_on, t._hard_on = False, bool(t.cfg.hard_names)
  t._es_on = t.cfg.es_eval_fail_ratio is not None and t.cfg.eval_enabled and t.cfg.eval_interval > 0
  t.es_hist, t.es_streak, t.es_best, t.es_stop = [], 0, None, None
  t.guard_skips, t.guard_nf_iters = {"wm": 0, "pi": 0, "critic": 0}, 0
  t.wd_forced, t.guard_floored, t.guard_floor_iters = False, 0, 0
  t.wd_quiet_until = t.cfg.wd_warmup
  t.trace = {"collect": [], "evals": [], "saves": [], "tb": []}
  t.writer = types.SimpleNamespace(add_scalar=lambda k, v, s: t.trace["tb"].append((k, float(v), int(s))))

  def collect_step():
    t.trace["collect"].append((t.it, t.cmd.hard_mult))
    t.env_steps += 1
    t.buffer.size = min(t.buffer.size + 1, 1000)
    t.stats["ep_return_norm"].append(0.8)
    t.stats["ep_len"].append(100)
    t.stats["ep_failed"].append(0.0)
    t.stats["term_mismatch"].append(0.0)
    t.stats["sim_nonfinite"].append(0.0)

  n_eval = [0]

  def evaluate():
    assert t.main, "only rank 0 evaluates"
    k = n_eval[0]
    n_eval[0] += 1
    t.trace["evals"].append(t.it)
    if eval_raises_at is not None and k >= eval_raises_at:
      raise RuntimeError("evaluator died (test)")
    f = fails[k % len(fails)]
    return {"scalars": {"return_mean": 0.9 - f / 1e4, "return_min": 0.1, "failed": float(f)}, "num_clips": 1024, "failed_clips": []}

  save0 = t.save

  def save(tag: str) -> str:
    t.trace["saves"].append((t.it, tag))
    return save0(tag)  # real Trainer.save (rollback target)

  t.collect_step, t.evaluate, t.save = collect_step, evaluate, save
  t.update_world_model = lambda: {"wm/loss": 0.01, "wm/err_step1": 0.02, "wm/err_step8": 0.05}
  t.update_policy = lambda: {"pi/imagined_return": 0.5, "pi/imagined_reward_norm": 0.9, "pi/imagined_reward_alive": 0.93,
                             "pi/survival_H": 0.9, "pi/grad_norm": 0.3, "pi/loss": -0.5, "time/ref_gather": 0.0, "critic/loss": 0.05}
  return t


def run_loop(t: Trainer, before: bool = False) -> dict:
  """Run the real loop (or the one before) under the fake clock; capture stdout and every trace."""
  clock = Clock()
  ns = types.SimpleNamespace(time=clock.time, strftime=lambda fmt: "<date>")
  out = io.StringIO()
  with mock.patch.object(trainer_mod, "time", ns), mock.patch.object(THIS, "time", ns), contextlib.redirect_stdout(out):
    if before:
      train_before(t)
    else:
      Trainer.train(t)
  files = {}
  for n in sorted(os.listdir(t.cfg.log_dir)):
    p = os.path.join(t.cfg.log_dir, n)
    if n.endswith((".jsonl", ".json", ".log")):
      files[n] = open(p).read()
  return {"stdout": out.getvalue(), "prints": list(t.prints), "it": t.it, "files": files, **{k: list(v) for k, v in t.trace.items()},
          "rollbacks": t.wd_rollbacks, "lr_scale": t.lr_scale}


# ---------------------------------------------------------------------------
# (b) the loop, one process
# ---------------------------------------------------------------------------


def test_es_off_is_the_loop_before():
  for cur in ({}, {"hard_names": ("kongfan",), "hard_term_schedule": ((0, 2.0), (12, 2.0), (24, 1.0))}):
    for force in (-1, 13):
      with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        new = run_loop(loop_trainer(a, wd_force_at=force, **cur))
        old = run_loop(loop_trainer(b, wd_force_at=force, **cur), before=True)
        new_s = new["stdout"].replace(a, "<dir>")
        old_s = old["stdout"].replace(b, "<dir>")
        if cur:  # the loop before has no curriculum: its multiplier stays 1 and its log line has no hard part
          assert [m for _, m in old["collect"]] == [1.0] * len(old["collect"])
          assert [m for _, m in new["collect"]] == [hard_mult_at_master(cur["hard_term_schedule"], i) for i, _ in new["collect"]]
          assert {2.0, 1.0} <= {m for _, m in new["collect"]} and any(1.0 < m < 2.0 for _, m in new["collect"])
          new_s = "\n".join(l.split(" | hard x")[0] + " | " + l.split(" | hard x")[1].split(" | ", 1)[1] if " | hard x" in l else l
                            for l in new_s.split("\n"))
          new_tb = [x for x in new["tb"] if not x[0].startswith("hard/")]
          assert len(new_tb) < len(new["tb"])
        else:
          new_tb = new["tb"]
        assert new_s == old_s, "stdout"
        assert new_tb == old["tb"], "TB scalars"
        for k in ("saves", "evals", "it", "rollbacks", "lr_scale"):
          assert new[k] == old[k], k
        assert [i for i, _ in new["collect"]] == [i for i, _ in old["collect"]]
        assert {k: v.replace(a, "<dir>") for k, v in new["files"].items()} == {k: v.replace(b, "<dir>") for k, v in old["files"].items()}
        assert "earlystop.json" not in new["files"] and "best_eval.json" not in new["files"]
        assert new["it"] == 44 and new["saves"][-1] == (44, "final") and len(new["evals"]) == (11 if force < 0 else 12)
        assert new["rollbacks"] == (1 if force >= 0 else 0)
  print(f"  es off == ca9266e loop: {len(new['tb'])} TB scalars, {len(new['saves'])} saves, forced rollback {new['rollbacks']}")


def test_es_stops():
  with tempfile.TemporaryDirectory() as d:
    t = loop_trainer(d, es_eval_fail_ratio=1.5, es_patience=2)
    r = run_loop(t)
    assert r["it"] == 32 and t.es_stop is not None and t.es_stop["iteration"] == 32
    assert r["evals"] == [4, 8, 12, 16, 20, 24, 28, 32]
    hist = t.es_stop["history"]
    assert [h["it"] for h in hist] == r["evals"] and [h["failed"] for h in hist] == FAILS[:8]
    assert [h["bad"] for h in hist] == [False, False, False, False, True, False, True, True]
    assert [h["threshold"] for h in hist] == [None, None, None, 142.5, 140.25, 142.5, 146.25, 150.0]
    assert [h["streak"] for h in hist] == [0, 0, 0, 0, 1, 0, 1, 2]
    best = os.path.join(d, "model_best_eval.pt")
    assert t.es_best == {"it": 8, "failed": 90.0, "return_mean": 0.9 - 90 / 1e4, "ckpt": best, "regular_ckpt": os.path.join(d, "model_8.pt")}
    assert torch.load(best, weights_only=False)["it"] == 8
    es = json.loads(r["files"]["earlystop.json"])
    assert es["iteration"] == 32 and es["best_eval"]["ckpt"] == best and es["final_ckpt"] == os.path.join(d, "model_final.pt")
    assert es["history"] == json.loads(json.dumps(hist)) and es["es_eval_fail_ratio"] == 1.5 and es["es_patience"] == 2
    assert json.loads(r["files"]["best_eval.json"])["it"] == 8
    # new bests at 4 (off the checkpoint grid of 8) and 8; final saved after the stop; nothing after 32
    assert r["saves"] == [(4, "best_eval"), (8, "8"), (8, "best_eval"), (16, "16"), (24, "24"), (32, "32"), (32, "final")]
    assert json.loads(r["files"]["best_eval.json"]) == json.loads(json.dumps(t.es_best))
    assert max(i for i, _ in r["collect"]) == 31, "no iteration after the stop"
    lines = [l for l in r["stdout"].split("\n") if l.startswith("[earlystop]")]
    assert len(lines) == 1 and lines[0].startswith("[earlystop] it 32: eval failed 160 >= 150.0 (1.5 x median of 7 previous evals)")
    assert f"best eval it 8 (failed 90): {best}" in lines[0]
    es_lines = [l for l in r["stdout"].split("\n") if l.startswith("[es] ")]
    assert len(es_lines) == 8 and "warming up" in es_lines[0] and "BAD, streak 2/2" in es_lines[-1]
    assert sum(1 for k, _, _ in r["tb"] if k == "es/streak") == 8
    print(f"  {lines[0]}")
  # patience 3: the ok eval at 24 resets the streak, 28 / 32 / 36 bad (300 >= 1.5 x 115) -> stops at 36
  with tempfile.TemporaryDirectory() as d:
    t = loop_trainer(d, es_eval_fail_ratio=1.5, es_patience=3)
    r = run_loop(t)
    assert r["it"] == 36 and [h["bad"] for h in t.es_hist][-3:] == [True, True, True]
  # ratio 3: never triggers -> runs to the end; same loop as es off apart from its own lines / files / best saves
  with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
    on, off = loop_trainer(a, es_eval_fail_ratio=3.0), loop_trainer(b)
    r_on, r_off = run_loop(on), run_loop(off)
    assert r_on["it"] == r_off["it"] == 44 and on.es_stop is None and "earlystop.json" not in r_on["files"]
    strip = lambda s, d: "\n".join(l for l in s.replace(d, "<dir>").split("\n") if not l.startswith("[es] "))  # noqa: E731
    assert strip(r_on["stdout"], a) == strip(r_off["stdout"], b)
    assert [x for x in r_on["tb"] if not x[0].startswith("es/")] == r_off["tb"]
    assert [s for s in r_on["saves"] if s[1] != "best_eval"] == r_off["saves"], "only the best-eval saves"
    assert r_on["evals"] == r_off["evals"] and json.loads(r_on["files"]["best_eval.json"])["it"] == 8


def test_best_eval_checkpoint_survives_a_rollback():
  """Best eval at 16 (model_16.pt + model_best_eval.pt); a rollback to 8 re-reaches 16 and overwrites model_16.pt with
  other weights; model_best_eval.pt still holds the evaluated ones (it is written at a new best only), and the best
  eval no longer names model_16.pt (best_eval.json rewritten at the rollback). The evals at 12 / 16 of the abandoned
  branch leave the history: the restored branch's 12 is the 3rd eval (warming up), 16 / 20 (300 >= 1.5 x median 100 /
  200) stop it at 20 -- with them kept, it stopped at 16."""
  with tempfile.TemporaryDirectory() as d:
    t = loop_trainer(d, fails=[100, 100, 100, 50, 300, 300, 300], es_eval_fail_ratio=1.5, es_patience=2, wd_force_at=18)
    t._healthy = lambda: t.it == 8  # rollback target 8 (not 16)
    stamp = []
    save0 = t.save

    def save(tag):
      with torch.no_grad():  # different weights at every save: tells the files apart
        next(t.policy.parameters()).add_(1.0)
      stamp.append((t.it, tag, float(next(t.policy.parameters()).detach().flatten()[0])))
      return save0(tag)

    t.save = save
    r = run_loop(t)
    assert r["rollbacks"] == 1 and r["evals"] == [4, 8, 12, 16, 12, 16, 20] and r["it"] == 20
    assert [x[:2] for x in stamp] == [(4, "best_eval"), (8, "8"), (16, "16"), (16, "best_eval"), (16, "16"), (20, "final")]
    assert [(h["it"], h["failed"], h["bad"], h["streak"]) for h in t.es_hist] == [
      (4, 100, False, 0), (8, 100, False, 0), (12, 300, False, 0), (16, 300, True, 1), (20, 300, True, 2)]
    (rb,) = [l for l in r["stdout"].split("\n") if l.startswith("[es] rollback")]
    assert rb == "[es] rollback to it 8: eval(s) at [12, 16] of the abandoned branch left out of the early-stop history; streak 0/2", rb
    first16 = next(w for it, tag, w in stamp if tag == "best_eval" and it == 16)
    assert t.es_best["it"] == 16 and t.es_best["failed"] == 50 and t.es_best["regular_ckpt"] is None
    assert json.loads(r["files"]["best_eval.json"]) == json.loads(json.dumps(t.es_best)) and t.es_stop["best_eval"] == t.es_best
    w_best = float(next(iter(torch.load(t.es_best["ckpt"], weights_only=False)["policy"].values())).flatten()[0])
    w_16 = float(next(iter(torch.load(os.path.join(d, "model_16.pt"), weights_only=False)["policy"].values())).flatten()[0])
    assert w_best == first16 and w_16 != first16, "model_16.pt was overwritten, model_best_eval.pt was not"


FAILS_RB = [100, 90, 95, 92, 200, 150, 160]  # eval k; the 5th (it 20) is on the branch the rollback at 21 abandons


def test_es_rollback_drops_the_abandoned_branch():
  """Reviewer 10-25 (a)/(b): bad eval at 20 (streak 1), the watchdog rolls back at 21 to 16; the restored branch's 20
  (150) is bad once. Kept, the abandoned 200 made it streak 2 = a stop after ONE bad eval on the restored branch, and
  raised the median (threshold 142.5 instead of 140.25). Dropped: streak 1 there, the stop needs 24 too."""
  with tempfile.TemporaryDirectory() as d:
    t = loop_trainer(d, fails=FAILS_RB, es_eval_fail_ratio=1.5, es_patience=2, wd_force_at=21)
    r = run_loop(t)
    assert r["rollbacks"] == 1 and r["evals"] == [4, 8, 12, 16, 20, 20, 24] and r["it"] == 24 == t.es_stop["iteration"]
    hist = [(h["it"], h["failed"], h["threshold"], h["bad"], h["streak"]) for h in t.es_hist]
    assert hist == [(4, 100, None, False, 0), (8, 90, None, False, 0), (12, 95, None, False, 0), (16, 92, 142.5, False, 0),
                    (20, 150, 140.25, True, 1), (24, 160, 142.5, True, 2)], hist
    assert json.loads(r["files"]["earlystop.json"])["history"] == json.loads(json.dumps(t.es_hist))
    assert "[es] rollback to it 16: eval(s) at [20] of the abandoned branch left out of the early-stop history; streak 0/2" in r["stdout"]
    assert t.es_best["it"] == 8 and t.es_best["regular_ckpt"] == os.path.join(d, "model_8.pt"), "best before the rollback point: untouched"
    with tempfile.TemporaryDirectory() as d2, mock.patch.object(Trainer, "_es_rollback", lambda self: None):  # the code before
      t2 = loop_trainer(d2, fails=FAILS_RB, es_eval_fail_ratio=1.5, es_patience=2, wd_force_at=21)
      r2 = run_loop(t2)
    assert r2["it"] == 20 and [h["it"] for h in t2.es_hist] == [4, 8, 12, 16, 20, 20] and t2.es_hist[-1]["threshold"] == 142.5
    print(f"  rollback 21 -> 16: stop at {r['it']} (abandoned eval kept, the code before: stop at {r2['it']})")
  # the eval at the rollback point itself (same weights as the checkpoint) stays: rollback at 17 -> 16 drops nothing
  with tempfile.TemporaryDirectory() as d:
    t = loop_trainer(d, fails=FAILS_RB, es_eval_fail_ratio=1.5, es_patience=2, wd_force_at=17)
    r = run_loop(t)
    assert [h["it"] for h in t.es_hist][:4] == [4, 8, 12, 16] and r["evals"][:5] == [4, 8, 12, 16, 20]
    assert "[es] rollback" not in r["stdout"]


def test_resume_at_an_eval_slot_evaluates_the_start():
  """--resume at an eval slot (r15ws: 136000): the loaded weights are evaluated before the first update and are the
  first entry of the history and the first best (model_best_eval.pt = the start); off the slot / es off: no extra eval."""
  with tempfile.TemporaryDirectory() as d:
    t = loop_trainer(d, es_eval_fail_ratio=1.5, es_patience=2)
    t.it = 8  # Trainer.load of a model_8.pt
    r = run_loop(t)
    assert r["evals"][:3] == [8, 12, 16] and r["collect"][0][0] == 8 and r["saves"][0] == (8, "best_eval")
    assert r["stdout"].split("\n")[0].startswith("[eval it 8] return 0.8900") and r["stdout"].split("\n")[1].startswith("[es] it 8: eval failed 100 (warming up)")
    assert [h["it"] for h in t.es_hist] == list(range(8, 37, 4)) and [h["failed"] for h in t.es_hist] == FAILS[:8]
    assert r["it"] == 36 and t.es_best["it"] == 12, "the FAILS sequence shifted by one slot: stops 4 later"
    assert json.loads(r["files"]["eval.jsonl"].split("\n")[0])["it"] == 8
  for kw, it0, first in (({"es_eval_fail_ratio": 1.5}, 9, 12), ({}, 8, 12)):
    with tempfile.TemporaryDirectory() as d:
      t = loop_trainer(d, **kw)
      t.it = it0
      r = run_loop(t)
      assert r["evals"][0] == first and not any(tag == "best_eval" and it == it0 for it, tag in r["saves"]), (kw, it0)


def test_es_survives_a_failing_evaluator():
  with tempfile.TemporaryDirectory() as d:
    t = loop_trainer(d, es_eval_fail_ratio=1.5, eval_raises_at=2)
    with mock.patch("traceback.print_exc"):
      r = run_loop(t)
    assert r["it"] == 44 and t.es_stop is None and len(t.es_hist) == 2 and t.cfg.eval_enabled is False
    assert r["evals"] == [4, 8, 12] and "[eval] failed: RuntimeError('evaluator died (test)')" in r["stdout"]


# ---------------------------------------------------------------------------
# (c) the loop on 3 gloo processes
# ---------------------------------------------------------------------------


def _loop_worker(rank: int, world: int, port: int, tmp: str):
  os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world))
  torch.set_num_threads(1)
  ctx = DistCtx.from_env("cpu", timeout_min=2.0)

  def same(x, what):
    xs = ctx.all_gather_object(x)
    assert all(v == xs[0] for v in xs), f"{what} differs between ranks"

  cur = {"hard_names": ("kongfan",), "hard_term_schedule": ((0, 2.0), (12, 2.0), (24, 1.0))}
  # ---- es on + curriculum + a forced rollback: same verdicts / stop / multipliers everywhere ----
  d = os.path.join(tmp, "es")  # one log dir for every rank (train.py broadcasts it)
  os.makedirs(d, exist_ok=True)
  ctx.barrier()
  t = loop_trainer(d, dist=ctx, fails=FAILS_RB, es_eval_fail_ratio=1.5, es_patience=2, wd_force_at=21, **cur)
  r = run_loop(t)
  same((r["it"], t.es_hist, t.es_best, t.es_stop, t.es_streak), "early-stop state")
  same(r["collect"], "collect trace (iteration, curriculum multiplier)")
  # the forced rollback at 21 (-> 16) drops the bad eval at 20 on every rank: the restored 20 is streak 1, 24 stops
  assert r["rollbacks"] == 1 and r["it"] == 24, r["it"]
  assert [m for _, m in r["collect"]] == [hard_mult_at_master(cur["hard_term_schedule"], i) for i, _ in r["collect"]]
  assert [h["it"] for h in t.es_hist] == [4, 8, 12, 16, 20, 24] and [h["bad"] for h in t.es_hist] == [False] * 4 + [True, True]
  assert t.es_best["it"] == 8 and t.es_stop["iteration"] == 24 and t.es_best["ckpt"] == os.path.join(d, "model_best_eval.pt")
  if rank == 0:
    assert r["evals"] == [4, 8, 12, 16, 20, 20, 24]
    assert r["saves"][-1] == (24, "final") and os.path.exists(os.path.join(d, "earlystop.json"))
    print(f"  rank 0: {[l for l in r['stdout'].split(chr(10)) if l.startswith('[earlystop]')][0]}")
  else:
    assert r["evals"] == [] and r["saves"] == [] and r["stdout"] == ""
  ctx.barrier()
  # ---- resumed at an eval slot: rank 0 evaluates the start, every rank joins the verdict ----
  d = os.path.join(tmp, "start")
  os.makedirs(d, exist_ok=True)
  ctx.barrier()
  t = loop_trainer(d, dist=ctx, es_eval_fail_ratio=1.5, es_patience=2)
  t.it = 8
  r = run_loop(t)
  same((r["it"], t.es_hist, t.es_best, t.es_stop), "early-stop state after a resume")
  assert t.es_hist[0]["it"] == 8 and r["it"] == 36 and r["evals"] == ([h["it"] for h in t.es_hist] if rank == 0 else [])
  ctx.barrier()
  # ---- es off == the loop before, on every rank ----
  da, db = os.path.join(tmp, "new"), os.path.join(tmp, "old")
  for x in (da, db):
    os.makedirs(x, exist_ok=True)
  ctx.barrier()
  new = run_loop(loop_trainer(da, dist=ctx, wd_force_at=13))
  old = run_loop(loop_trainer(db, dist=ctx, wd_force_at=13), before=True)
  assert new["stdout"].replace(da, "<dir>") == old["stdout"].replace(db, "<dir>")
  for k in ("tb", "saves", "evals", "it", "rollbacks", "collect", "lr_scale"):
    assert new[k] == old[k], k
  same((new["it"], new["collect"]), "es-off trace")
  ctx.barrier()
  # ---- the watchdog's rank mean / floor act on the alive signal ----
  spy = []
  reduce0 = ctx.reduce_scalars

  def spy_reduce(s, op="mean", check_keys=True):
    out = reduce0(s, op, check_keys)
    spy.append((dict(s), dict(out)))
    return out

  ctx.reduce_scalars = spy_reduce
  w = twf.stand_in(-1.0, dist=ctx, ckpt=os.path.join(tmp, "model_500.pt"), wd_signal="imagined_reward_alive")
  alive = [0.9 + 0.01 * rank] * 30
  alive[20] = -1e9 if rank == 1 else alive[20]
  for k in range(30):
    w.it += 1
    assert w._watchdog({"pi/imagined_reward_norm": -1e6 * (k % 2), "pi/imagined_reward_alive": alive[k], "critic/loss": 0.05}, 0) is False
  assert len(spy) == 30, "one rank reduce per iteration"
  inp, out = spy[20]
  assert inp["r"] == (-1.0 if rank == 1 else alive[20]) and inp["floored"] == float(rank == 1), "this rank's ALIVE value, floored"
  closed = sum(-1.0 if k == 1 else 0.9 + 0.01 * k for k in range(world)) / world
  assert abs(out["r"] - closed) < 1e-12 and w.guard_floored == 1
  same(w.wd_reward_ema, "alive EMA")
  ctx.reduce_scalars = reduce0
  ctx.barrier()
  ctx.close()
  open(os.path.join(tmp, f"rank{rank}.ok"), "w").close()


def test_ddp_loop_world3():
  with tempfile.TemporaryDirectory() as tmp:
    open(os.path.join(tmp, "model_500.pt"), "w").close()
    mp.spawn(_loop_worker, args=(3, tsc._free_port(), tmp), nprocs=3, join=True)
    assert all(os.path.exists(os.path.join(tmp, f"rank{r}.ok")) for r in range(3))


# ---------------------------------------------------------------------------
# (d) pi/imagined_reward_alive
# ---------------------------------------------------------------------------

N_TOY, H_TOY = 3, 4
D_TAB = torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5]])  # [N, H]: never / at t=1 / leaky
R_TAB = torch.tensor([[0.08, 0.08, 0.08, 0.08], [0.09, 0.05, 1e3, 1e3], [0.07, 0.06, 0.05, 0.04]])  # rollout 1: diverged after death


def policy_trainer(seed: int = 0) -> Trainer:
  """Trainer for update_policy: toy policy / critic / world model, scripted imagination (rewards depend on the
  policy through the toy state, so the gradient is real)."""
  t = object.__new__(Trainer)
  t.cfg = TrainCfg(H=H_TOY, rollouts_per_update=N_TOY, critic_epochs=2, critic_minibatches=2, log_dir=tempfile.mkdtemp(prefix="alive_"))
  t.dist = DistCtx(device="cpu")
  t.device = "cpu"
  t.nj = 1
  t._hard_on = False
  t.guard_skips = {"wm": 0, "pi": 0, "critic": 0}
  torch.manual_seed(seed)
  t.policy = torch.nn.Linear(2, 1)
  t.critic_net = torch.nn.Linear(2, 1)
  t.critic = lambda fn, gn, xi: t.critic_net(fn).squeeze(-1)
  t.critic.parameters = t.critic_net.parameters
  t.wm = torch.nn.Linear(1, 1)
  t.opt_pi = torch.optim.Adam(t.policy.parameters(), lr=1e-2)
  t.opt_v = torch.optim.Adam(t.critic_net.parameters(), lr=1e-2)
  t.heads = types.SimpleNamespace(proprio=lambda s, pa, b, noise: s, kin=lambda s: {"s": s})
  t.sim = types.SimpleNamespace(bias_from_xi=lambda xi, nj: torch.zeros(xi.shape[0], nj))
  g = torch.Generator().manual_seed(seed + 1)
  st = dict(xi=torch.zeros(N_TOY, 1), hist_states=torch.randn(N_TOY, 10, 2, generator=g), hist_prev_action=torch.zeros(N_TOY, 10, 1),
            state=torch.randn(N_TOY, 2, generator=g), frame=torch.zeros(N_TOY, dtype=torch.long), clip_end=torch.full((N_TOY,), 100),
            prev_action=torch.zeros(N_TOY, 1))
  t.buffer = types.SimpleNamespace(sample_starts=lambda n, hist: st)
  t.lib = types.SimpleNamespace(gather_windows=lambda frame, L: {k: torch.zeros(N_TOY, L, 1) for k in
                                ("joint_pos", "joint_vel", "body_pos", "body_quat", "body_lin_vel", "body_ang_vel")})
  t.rec = {"r": [], "args": []}
  step = [0]

  def imagine_step(s, obs_hist, prev_a, f_loc, clip_end_loc, xi, bias, kin, *rest):
    k = step[0] % H_TOY
    step[0] += 1
    t.rec["args"].append(len(rest))
    a = t.policy(s)
    s_next = s + 0.1 * a
    r = R_TAB[:, k] + 0.01 * (s_next**2).sum(-1)
    d = D_TAB[:, k] + 0.0 * a.squeeze(-1)
    t.rec["r"].append(r.detach().clone())
    return s_next, obs_hist, a, f_loc, kin, r, d, s_next, torch.zeros(N_TOY, 1)

  t.fns = types.SimpleNamespace(imagine_step=imagine_step, goal_from_pack=lambda kin, f, ce, *p: torch.zeros(N_TOY, 1),
                                critic_inputs=lambda s, goal: (s, goal))
  return t


def test_alive_closed_form_and_default_identical():
  new, old = policy_trainer(), policy_trainer()
  torch.manual_seed(7)
  m_new = Trainer.update_policy(new)
  rng_new = torch.get_rng_state()
  torch.manual_seed(7)
  m_old = update_policy_before(old)
  rng_old = torch.get_rng_state()
  # closed form on the recorded rewards: c_t = prod_{s<t} (1 - d_s)
  R = torch.stack(new.rec["r"], 1).double()  # [N, H]
  C = torch.ones(N_TOY, H_TOY, dtype=torch.float64)
  for k in range(1, H_TOY):
    C[:, k] = C[:, k - 1] * (1.0 - D_TAB[:, k - 1].double())
  assert C.tolist() == [[1, 1, 1, 1], [1, 1, 0, 0], [1, 0.5, 0.25, 0.125]]
  rmax = HeadsCfg().reward_max_per_step
  closed = float((C * R).sum() / C.sum()) / rmax
  assert math.isclose(m_new["pi/imagined_reward_alive"], closed, rel_tol=1e-6), (m_new["pi/imagined_reward_alive"], closed)
  assert m_new["pi/imagined_reward_alive"] < 1.0 < 1e3 < m_new["pi/imagined_reward_norm"], "post-death 1e3 only in the unweighted mean"
  assert math.isclose(m_new["pi/imagined_reward_norm"], float(R.mean()) / rmax, rel_tol=1e-6)
  # update_policy otherwise = the code before, bitwise
  assert new.rec["args"] == [7] * H_TOY and old.rec["args"] == [6] * H_TOY, "term_scale passed (None) only by the new code"
  for a, b in zip(list(new.policy.parameters()) + list(new.critic_net.parameters()), list(old.policy.parameters()) + list(old.critic_net.parameters())):
    assert torch.equal(a, b)
  assert set(m_new) - set(m_old) == {"pi/imagined_reward_alive"} and all(m_new[k] == m_old[k] for k in m_old if k != "time/ref_gather")
  assert torch.equal(rng_new, rng_old) and all(p.requires_grad for p in new.wm.parameters())
  print(f"  alive {m_new['pi/imagined_reward_alive']:.4f} (closed form {closed:.4f}) vs unweighted {m_new['pi/imagined_reward_norm']:.1f}")


def drive2(t: Trainer, seq, alive) -> list:
  """twf.drive with both reward keys in m_pi."""
  out = []
  for (r, vl, sk), a in zip(seq, alive):
    t.it += 1
    try:
      d = t._watchdog({"pi/imagined_reward_norm": r, "pi/imagined_reward_alive": a, "critic/loss": vl}, sk)
    except RuntimeError as e:
      out.append(("raise", str(e)))
      break
    e = t.wd_reward_ema
    out.append((t.it, d, None if e is None else e.hex(), len(t.wd_hist), t.wd_hist[-1][1].hex() if t.wd_hist else None,
                t.wd_quiet_until, t.wd_rollbacks, t.wd_nf_streak, t.guard_nf_iters, len(t.wd_vloss), t.lr_scale))
  return out


def test_watchdog_signal():
  seq = twf.recorded_sequence()
  g = np.random.default_rng(9)
  noise = [float(0.93 + 0.01 * g.standard_normal()) for _ in seq]
  # default signal, alive key present: == the watchdog before (decisions, EMA, log)
  new, old = twf.stand_in(None), twf.stand_in(None, before=True)
  assert drive2(new, seq, noise) == twf.drive(old, seq) and new.prints == old.prints and new.wd_rollbacks == 6
  # alive signal on the same values: a drop-in (same decisions / EMA), label in the log only
  alive = twf.stand_in(None, wd_signal="imagined_reward_alive")
  tr = drive2(alive, [(float(x), vl, sk) for (_, vl, sk), x in zip(seq, noise)], [r for r, _, _ in seq])
  ref = twf.drive(twf.stand_in(None, before=True), seq)
  assert tr[:-1] == ref[:-1] and tr[-1][0] == ref[-1][0] == "raise"
  assert tr[-1][1].replace("imagined alive reward", "imagined reward") == ref[-1][1]
  assert [p.replace("imagined alive reward", "imagined reward") for p in alive.prints] == old.prints
  assert any("imagined alive reward EMA" in p for p in alive.prints)
  # the floor applies to the alive signal
  fl = twf.stand_in(-1.0, wd_signal="imagined_reward_alive")
  base = twf.normal(1500, np.random.default_rng(3))
  a_seq = [r for r, _, _ in base]
  a_seq[700] = -1e9
  tr = drive2(fl, [(-1e9, vl, sk) for _, vl, sk in base], a_seq)
  assert fl.wd_rollbacks == 0 and fl.guard_floored == 1 and not any(x[1] for x in tr)
  (msg,) = [p for p in fl.prints if "floored" in p]
  assert msg.startswith("[guard] it 701: imagined alive reward floored to -1 on 1/1 rank(s)"), msg
  # r16-like: the unweighted mean dips with post-death divergence, the survival-weighted one does not
  g = np.random.default_rng(16)
  base = twf.normal(3000, g)
  norm = [r - (40.0 if (k > 600 and k % 37 == 0) else 0.0) for k, (r, _, _) in enumerate(base)]
  alive_v = [float(0.95 + 0.003 * g.standard_normal()) for _ in base]
  seq16 = [(n, vl, sk) for n, (_, vl, sk) in zip(norm, base)]
  d_norm, d_alive = twf.stand_in(None), twf.stand_in(None, wd_signal="imagined_reward_alive")  # r16: no floor
  drive2(d_norm, seq16, alive_v)
  drive2(d_alive, seq16, alive_v)
  assert d_norm.wd_rollbacks >= 1 and d_alive.wd_rollbacks == 0, (d_norm.wd_rollbacks, d_alive.wd_rollbacks)
  print(f"  r16-like spikes: {d_norm.wd_rollbacks} rollbacks on imagined_reward_norm, {d_alive.wd_rollbacks} on imagined_reward_alive")
  # a real collapse still rolls back on the alive signal
  col = twf.stand_in(-1.0, wd_signal="imagined_reward_alive")
  s2 = twf.normal(2000, g) + twf.drift(400, g) + [(0.4, 0.05, 0)] * 300
  drive2(col, [(0.9, vl, sk) for _, vl, sk in s2], [r for r, _, _ in s2])
  assert col.wd_rollbacks >= 1


# ---------------------------------------------------------------------------
# (e) wiring
# ---------------------------------------------------------------------------


def test_cli_and_cfg():
  assert (TrainCfg().es_eval_fail_ratio, TrainCfg().es_patience, TrainCfg().wd_signal) == (None, 2, "imagined_reward_norm")
  script = tsc_load_train()
  with tempfile.TemporaryDirectory() as tmp:
    f = os.path.join(tmp, "g1__a.npz")
    open(f, "w").close()
    lst = os.path.join(tmp, "l.txt")
    with open(lst, "w") as fh:
      fh.write(f + "\n")
    base = ["--motion_list", lst]
    c = _parse(script, tmp, base)
    assert (c.es_eval_fail_ratio, c.es_patience, c.wd_signal) == (None, 2, "imagined_reward_norm")
    c = _parse(script, tmp, base + ["--es_eval_fail_ratio", "1.5", "--es_patience", "3", "--wd_signal", "imagined_reward_alive",
                                    "--wd_reward_floor", "-1.0", "--eval_interval", "2000"])
    assert (c.es_eval_fail_ratio, c.es_patience, c.wd_signal, c.wd_reward_floor, c.eval_interval) == (1.5, 3, "imagined_reward_alive", -1.0, 2000)
    for bad in (["--es_eval_fail_ratio", "1.5", "--no_eval"], ["--es_eval_fail_ratio", "1.5", "--eval_interval", "0"],
                ["--es_eval_fail_ratio", "0"], ["--es_eval_fail_ratio", "1.5", "--es_patience", "0"], ["--wd_signal", "survival"]):
      try:
        _parse(script, tmp, base + bad, quiet=True)
      except SystemExit as e:
        assert e.code == 2
      else:
        raise AssertionError(f"must be rejected: {bad}")

    def fake_make_env_cfg(*a, **k):
      raise tsc._Probe

    for kw in ({}, {"es_eval_fail_ratio": 1.5, "es_patience": 2, "wd_signal": "imagined_reward_alive"}):
      log_dir = os.path.join(tmp, f"init_{len(kw)}")
      with mock.patch.object(trainer_mod, "make_env_cfg", fake_make_env_cfg):
        try:
          Trainer(TrainCfg(log_dir=log_dir, device="cpu", **kw))
        except tsc._Probe:
          pass
      got = json.load(open(os.path.join(log_dir, "train_cfg.json")))
      assert (got["es_eval_fail_ratio"], got["es_patience"], got["wd_signal"]) == (kw.get("es_eval_fail_ratio"), kw.get("es_patience", 2),
                                                                                kw.get("wd_signal", "imagined_reward_norm"))
    for bad in ({"wd_signal": "x"}, {"es_eval_fail_ratio": -1.0}, {"es_eval_fail_ratio": 1.5, "es_patience": 0}, {"hard_term_schedule": ((1, 2.0),)}):
      try:
        Trainer(TrainCfg(log_dir=os.path.join(tmp, "bad"), device="cpu", **bad))
      except ValueError:
        pass
      else:
        raise AssertionError(f"must be rejected: {bad}")


if __name__ == "__main__":
  if tsc.STUBBED:
    print(f"  (stubbed via sys.modules: {', '.join(tsc.STUBBED)})")
  for name, fn in list(globals().items()):
    if name.startswith("test_") and callable(fn):
      fn()
      print(f"ok  {name}")
