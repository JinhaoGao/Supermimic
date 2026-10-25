"""master 5f020a9 merged into ddp-bj31-v2 (decf018): one clamp, the hard-clip curriculum and extend_cache under DDP.

  python tests/test_merge_master_1025.py      # or: pytest tests/test_merge_master_1025.py

CPU, torch + numpy only; mjlab / tensorboard are stubbed like in test_sampler_clamp.py when missing, so the REAL
heads, sampler, Trainer.load / _dist_start / _check_consistency, make_env_cfg, MotionLibrary, extend_cache and CLI
code runs here.

Checks:
  (a) sampler return clamp, one code path for both branches: update_bin_returns / sampling_probabilities bitwise equal
      to master 5f020a9 (copied verbatim below) on/off, with and without a source prior; a checkpoint written by
      master (its TrainCfg / heads spelling: sampler_clamp_returns, hard_names, hard_term_schedule, ee_term_z +
      ee_term_temp) and one written here resume into the r15 recipe silently with the sampler state restored + clamped
      exactly as master's load does; a differing switch is reported; the r15 command line parses to the r15 recipe;
  (b) hard-clip curriculum: heads.termination(scale) == master 4314a50 (verbatim) bitwise -- hard, soft, gradients --
      with and without end effectors, scale=None == ones == the code before (bitwise, also compiled); hard_mult_at ==
      master on the r13 / r14 schedules; the env terms == mjlab's originals times the factor; make_env_cfg swaps them
      only with hard_names; hard_clip_mask == master's inline mask;
      data parallel (3 gloo processes): the multiplier is the same on every rank at every iteration (incl. after a
      rollback / resume to an earlier iteration), the curriculum state group passes the state check, and a rank with
      another hard set is caught;
  (c) extend_cache: extended cache == full rebuild of old + extra bitwise (arrays, clips.json, cache dir name, library
      fingerprint), incl. extra files without body_names and a failing one; MotionLibrary finds it (no rebuild); the
      script accepts / rejects an existing --out_list; resume from a checkpoint of the old library keeps its sampler
      statistics in the first bins (other roots: by basename), a mismatching bin count is refused; data parallel
      (2 gloo processes): every rank maps the same extended library (fingerprint in the state check), loads +
      broadcasts the same sampler state, and a rank with another library is caught.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import sys
import tempfile
import types
from unittest import mock

import numpy as np
import torch
import torch.multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)

import test_sampler_clamp as tsc  # noqa: E402  (installs the mjlab / tensorboard stubs when needed)
from test_ee_termination import ANCHOR, BT, EE_IDX, G1_TRACKED, NJ, kin_of, make_heads, random_batch  # noqa: E402
from test_motions_names import DST_JOINTS, TRACKED, _clip  # noqa: E402
from test_v2_merge import G1V2_SOURCES, _weights, master_heads_dict  # noqa: E402

import supermimic.env_cfg as env_cfg_mod  # noqa: E402
import supermimic.library_command as lc_mod  # noqa: E402
import supermimic.motions as motions_mod  # noqa: E402
import supermimic.trainer as trainer_mod  # noqa: E402
from supermimic.distributed import DistCtx  # noqa: E402
from supermimic.heads import HeadsCfg  # noqa: E402
from supermimic.library_command import LibraryMotionCommand, LibraryMotionCommandCfg, hard_clip_mask  # noqa: E402
from supermimic.math_utils import quat_apply_inverse  # noqa: E402
from supermimic.motions import MotionLibrary, extend_cache, library_cache_dir  # noqa: E402
from supermimic.trainer import TrainCfg, Trainer  # noqa: E402

R15_NAMES = "tengkong=200,kongfan=200,shoufan=200,erqijiao=200,danpaijiao=200,xuanfeng=200,waibailian=100,lihetui=100,saotang=100,ceti=100,houti=100,zhengti=100"
R13_SCHEDULE = ((104000, 1.6), (108000, 1.6), (114000, 1.0))
R14_SCHEDULE = ((116000, 1.6), (120000, 1.6), (126000, 1.0))


# ---------------------------------------------------------------------------
# master, verbatim
# ---------------------------------------------------------------------------


@torch.no_grad()
def update_bin_returns_master(self, bin_ids: torch.Tensor, returns: torch.Tensor) -> None:
  """library_command.LibraryMotionCommand.update_bin_returns at master 5f020a9, verbatim."""
  finite = torch.isfinite(returns)
  if not bool(finite.all()):
    bin_ids, returns = bin_ids[finite], returns[finite]
  if self.cfg.clamp_returns:
    returns = returns.clamp(0.0, self.cfg.g_max)
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
def sampling_probabilities_master(self) -> torch.Tensor:
  """library_command.LibraryMotionCommand.sampling_probabilities at master 5f020a9, verbatim."""
  c = self.cfg
  NB = self.library.num_bins
  uniform = self.bin_prior if self.bin_prior is not None else torch.full((NB,), 1.0 / NB, device=self.device)
  bin_G = self.bin_G.clamp(0.0, c.g_max) if c.clamp_returns else self.bin_G
  G = torch.where(self.bin_visits > 0, bin_G, torch.full_like(self.bin_G, c.g_max * 0.5))
  soft = torch.softmax(-c.softmax_temp * G / c.g_max, dim=0)
  hist = self.bin_weight_hist / self.bin_weight_hist.sum()
  probs = c.mix_uniform * uniform + c.mix_softmax * soft + c.mix_history * hist
  probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0) + 1e-12
  probs = probs / probs.sum()
  self.bin_weight_hist = (1 - c.weight_ema) * self.bin_weight_hist + c.weight_ema * soft
  return probs


def master_load_sampler(cmd, ck: dict, cur_files: list, clamp: bool) -> None:
  """The reset-sampler part of trainer.Trainer.load at master 5f020a9, verbatim (self.cmd -> cmd, self.cfg -> args)."""
  nb = ck["bin_G"].shape[0]
  old_files = list(ck.get("cfg", {}).get("motion_files", []))
  if ck["bin_G"].shape == cmd.bin_G.shape:
    cmd.bin_G.copy_(ck["bin_G"])
    cmd.bin_visits.copy_(ck["bin_visits"])
  elif nb < cmd.bin_G.shape[0] and old_files and old_files == list(cur_files[: len(old_files)]):
    # Library extended by appending clips (scripts/extend_cache.py): the old bins keep their indices.
    cmd.bin_G[:nb].copy_(ck["bin_G"])
    cmd.bin_visits[:nb].copy_(ck["bin_visits"])
  if clamp:
    cmd.bin_G.clamp_(0.0, cmd.cfg.g_max)


def termination_master(self, kin_next: dict, ref_bp: torch.Tensor, ref_bq: torch.Tensor, scale: torch.Tensor | None = None):
  """heads.AnalyticHeads.termination at master 4314a50, verbatim (cfg fields ee_term_z / ee_term_temp)."""
  c = self.cfg
  ref_ap, ref_aq = ref_bp[:, self.anchor], ref_bq[:, self.anchor]
  B = ref_ap.shape[0]
  g = self.gravity.expand(B, 3)
  k = scale if scale is not None else torch.ones(B, device=ref_bp.device)
  dz = (ref_ap[:, 2] - kin_next["anchor_pos"][:, 2]).abs()
  tilt = (quat_apply_inverse(ref_aq, g)[:, 2] - quat_apply_inverse(kin_next["anchor_quat"], g)[:, 2]).abs()
  hard = (dz > c.term_z * k) | (tilt > c.term_tilt * k)
  pz = torch.sigmoid((dz - c.term_z * k) / (c.term_z_temp * k))
  pt = torch.sigmoid((tilt - c.term_tilt * k) / (c.term_tilt_temp * k))
  alive = (1.0 - pz) * (1.0 - pt)
  if c.ee_term_z > 0:
    ez = (ref_bp[:, self.ee, 2] - kin_next["body_pos"][:, self.ee, 2]).abs()  # [B, n_ee]
    hard = hard | (ez > c.ee_term_z * k[:, None]).any(-1)
    alive = alive * (1.0 - torch.sigmoid((ez - c.ee_term_z * k[:, None]) / (c.ee_term_temp * k[:, None]))).prod(-1)
  return hard, 1.0 - alive


def termination_before(self, kin_next: dict, ref_ap: torch.Tensor, ref_aq: torch.Tensor, ref_bp: torch.Tensor | None = None):
  """heads.AnalyticHeads.termination at ca9266e (before the merge), verbatim."""
  c = self.cfg
  B = ref_ap.shape[0]
  g = self.gravity.expand(B, 3)
  dz = (ref_ap[:, 2] - kin_next["anchor_pos"][:, 2]).abs()
  tilt = (quat_apply_inverse(ref_aq, g)[:, 2] - quat_apply_inverse(kin_next["anchor_quat"], g)[:, 2]).abs()
  hard = (dz > c.term_z) | (tilt > c.term_tilt)
  pz = torch.sigmoid((dz - c.term_z) / c.term_z_temp)
  pt = torch.sigmoid((tilt - c.term_tilt) / c.term_tilt_temp)
  alive = (1.0 - pz) * (1.0 - pt)
  if c.ee_termination:
    assert ref_bp is not None, "ee_termination needs the reference body positions"
    dz_ee = (ref_bp[:, self.ee, 2] - kin_next["body_pos"][:, self.ee, 2]).abs()  # [B, n_ee]
    hard = hard | (dz_ee > c.ee_term_z).any(-1)
    p_ee = torch.sigmoid((dz_ee - c.ee_term_z) / c.ee_term_z_temp)
    alive = alive * torch.prod(1.0 - p_ee, dim=-1)
  soft = 1.0 - alive
  return hard, soft


def hard_mult_at_master(schedule, it: int) -> float:
  """trainer.Trainer.hard_mult_at at master 4314a50, verbatim (self.cfg.hard_term_schedule -> schedule)."""
  pts = sorted((int(a), float(b)) for a, b in schedule)
  if not pts:
    return 1.0
  if it <= pts[0][0]:
    return pts[0][1]
  for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
    if it <= x1:
      return y0 + (y1 - y0) * (it - x0) / max(x1 - x0, 1)
  return pts[-1][1]


def clip_hard_master(names, hard_names) -> torch.Tensor:
  """LibraryMotionCommand.__init__ hard set at master 4314a50, verbatim (lib.clips -> names)."""
  hn = [str(k).lower() for k in hard_names]
  return torch.tensor([any(k in c.lower() for k in hn) for c in names], dtype=torch.bool)


# mjlab 2026-10 (bj31 ~/supermimic/mjlab/src/mjlab/tasks/tracking/mdp/terminations.py / rewards.py), verbatim.
def _get_body_indexes(command, body_names):
  return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def bad_anchor_pos_z_only(env, command_name: str, threshold: float) -> torch.Tensor:
  command = env.command_manager.get_term(command_name)
  return torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]) > threshold


def bad_anchor_ori(env, asset_cfg, command_name: str, threshold: float) -> torch.Tensor:
  asset = env.scene[asset_cfg.name]
  command = env.command_manager.get_term(command_name)
  motion_projected_gravity_b = quat_apply_inverse(command.anchor_quat_w, asset.data.gravity_vec_w)
  robot_projected_gravity_b = quat_apply_inverse(command.robot_anchor_quat_w, asset.data.gravity_vec_w)
  return (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs() > threshold


def bad_motion_body_pos_z_only(env, command_name: str, threshold: float, body_names=None) -> torch.Tensor:
  command = env.command_manager.get_term(command_name)
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.abs(command.body_pos_relative_w[:, body_indexes, -1] - command.robot_body_pos_w[:, body_indexes, -1])
  return torch.any(error > threshold, dim=-1)


# ---------------------------------------------------------------------------
# (a) one sampler clamp
# ---------------------------------------------------------------------------


def test_clamp_equals_master_bitwise():
  NB = 6000
  g = torch.Generator().manual_seed(5)
  prior = torch.rand(NB, generator=g) + 0.1
  prior = prior / prior.sum()
  for clamp in (False, True):
    for pr in (None, prior):
      ours, master = tsc.make_cmd(NB, clamp, prior=pr), tsc.make_cmd(NB, clamp, prior=pr)
      for rnd in range(20):
        bins = torch.cat([torch.randint(0, NB, (5000,), generator=g), torch.randint(0, 40, (300,), generator=g)])
        rets = torch.randn(bins.numel(), generator=g) * 15.0 + 5.0
        if rnd % 4 == 1:
          rets[::53] = float("nan")
          rets[7] = float("-inf")
        ours.update_bin_returns(bins, rets)
        update_bin_returns_master(master, bins, rets)
        assert torch.equal(ours.bin_G, master.bin_G) and torch.equal(ours.bin_visits, master.bin_visits)
        for _ in range(2):
          assert torch.equal(ours.sampling_probabilities(), sampling_probabilities_master(master))
          assert torch.equal(ours.bin_weight_hist, master.bin_weight_hist)
      if clamp:
        assert float(ours.bin_G.min()) >= 0.0 and float(ours.bin_G.max()) <= tsc.G_MAX
      else:
        assert float(ours.bin_G.min()) < 0.0


def _master_train_cfg(files, **over) -> dict:
  """cfg of a checkpoint written by master 5f020a9 (bj27 r15): master's TrainCfg keys, master heads spelling."""
  d = dataclasses.asdict(TrainCfg())
  for k in ("world_size", "base_lr_scale", "sampler_sync_interval", "param_check_interval", "dist_timeout_min",
            "wd_reward_floor", "eval_clip_names", "num_threads", "wd_nf_patience", "wd_early_ckpt", "wd_signal",
            "es_eval_fail_ratio", "es_patience"):
    d.pop(k, None)  # this branch only
  d.update(motion_files=list(files), heads=master_heads_dict(ee_term_z=0.25), robot="g1", robot_variant="eval+evalcontact",
           source_weights=_weights(G1V2_SOURCES), name_weights=_weights(R15_NAMES), hard_names=(), hard_term_schedule=(),
           sampler_clamp_returns=True, H=48, policy_lr=5e-5, critic_lr=5e-5)
  d.update(over)
  return d


def _r15_here(t: Trainer, files, **over) -> None:
  """This run's cfg: the r15 recipe as scripts/launch_bj31_r15ws.example.sh passes it."""
  kw = dict(motion_files=list(files), robot="g1", robot_variant="eval+evalcontact", source_weights=_weights(G1V2_SOURCES),
            name_weights=_weights(R15_NAMES), sampler_clamp_returns=True, H=48, heads=HeadsCfg(ee_termination=True, ee_term_z=0.25))
  t.cfg = dataclasses.replace(t.cfg, **{**kw, **over})


def _save_ck(path: str, t: Trainer, G, V, cfg: dict, it: int = 136000, lr_scale: float = 0.0625) -> None:
  torch.save({"it": it, "env_steps": 136512, "policy": t.policy.state_dict(), "critic": t.critic.state_dict(), "wm": t.wm.state_dict(),
              "opt_wm": t.opt_wm.state_dict(), "opt_pi": t.opt_pi.state_dict(), "opt_v": t.opt_v.state_dict(),
              "bin_G": G, "bin_visits": V, "lr_scale": lr_scale, "cfg": cfg}, path)


def test_checkpoint_of_either_branch_resumes_into_r15():
  NB = 4096
  names = [f"g1__noetix_kungfu_{i}_kongfan.npz" if i % 3 == 0 else f"g1__lafan1_{i}.npz" for i in range(8)]
  bj27 = [f"/home/ubuntu/motion_dataset/g1_atlas_full/{n}" for n in names]
  bj31 = [f"/data2/x/{n}" for n in names]
  G, V, hot, _ = tsc.collapse_state(NB)  # bins far below 0 (r14 ckpt min -11.7; the collapse state goes to -31.4)
  with tempfile.TemporaryDirectory() as tmp:
    base = tsc.make_trainer(tmp, NB, False)
    ours_cfg = dataclasses.asdict(dataclasses.replace(
      base.cfg, motion_files=bj27, robot_variant="eval+evalcontact", source_weights=_weights(G1V2_SOURCES),
      name_weights=_weights(R15_NAMES), sampler_clamp_returns=True, H=48, heads=HeadsCfg(ee_term_z=0.25)))
    for label, cfg in (("master (bj27 r15)", _master_train_cfg(bj27)), ("ddp-bj31-v2", ours_cfg)):
      path = os.path.join(tmp, "ck.pt")
      _save_ck(path, base, G, V, cfg)
      t = tsc.make_trainer(tmp, NB, True)
      _r15_here(t, bj31)
      Trainer.load(t, path)
      warn = [s for s in t.prints if "WARNING" in s]
      assert warn == [], (label, warn)
      ref = tsc.make_cmd(NB, True)
      master_load_sampler(ref, torch.load(path, weights_only=False), bj27, clamp=True)
      assert torch.equal(t.cmd.bin_G, ref.bin_G) and torch.equal(t.cmd.bin_visits, ref.bin_visits), label
      assert float(t.cmd.bin_G[hot]) == 0.0 and t.it == 136000 and t.lr_scale == 0.0625
      (msg,) = [s for s in t.prints if "clamp_returns:" in s]
      (res,) = [s for s in t.prints if "resumed from" in s]
      lam = trainer_mod.cosine_lambda(250_000, 0.03)(136000)
      assert f"x cosine {lam:.4f}" in res and "lr_scale 0.0625" in res, res
      print(f"  {label}: silent resume, {msg.split('] ')[1]}")
    # the switch differs: reported, and the loaded state is left as it was (off here)
    path = os.path.join(tmp, "ck.pt")
    _save_ck(path, base, G, V, _master_train_cfg(bj27))
    t = tsc.make_trainer(tmp, NB, False)
    _r15_here(t, bj31, sampler_clamp_returns=False)
    Trainer.load(t, path)
    warn = [s for s in t.prints if "WARNING" in s]
    assert len(warn) == 1 and "sampler_clamp_returns = True in the checkpoint, False here" in warn[0], warn
    assert torch.equal(t.cmd.bin_G, G)
    # r14 (curriculum on) / H 32 / heads spelled by master: every difference reported
    _save_ck(path, base, G, V, _master_train_cfg(bj27, hard_names=("kongfan",), hard_term_schedule=R14_SCHEDULE, H=32))
    t = tsc.make_trainer(tmp, NB, True)
    _r15_here(t, bj31)
    Trainer.load(t, path)
    warn = [s for s in t.prints if "WARNING" in s]
    assert len(warn) == 3 and "hard_names = ('kongfan',)" in warn[0] and "hard_term_schedule" in warn[1] and "H = 32" in warn[2], warn


def test_r15_command_line():
  """bj27 g1v2_clamp_r15 (scout 10-25, from its session log; paths aside) -> TrainCfg."""
  script = tsc_load_train()
  with tempfile.TemporaryDirectory() as tmp:
    lst = os.path.join(tmp, "g1_plus_phc.txt")
    files = [os.path.join(tmp, f"g1__lafan1_{i}.npz") for i in range(3)]
    for f in files:
      open(f, "w").close()
    with open(lst, "w") as f:
      f.write("\n".join(files) + "\n")
    r15 = ["--run_name", "g1v2_clamp_r15", "--robot", "g1", "--robot_variant", "eval+evalcontact", "--motion_list", lst,
           "--resume", "x.pt", "--iterations", "136000", "--lr", "5e-5", "--H", "48", "--ee_term_z", "0.25", "--eval_clips", "1024",
           "--no_eval", "--num_workers", "64", "--source_weights", G1V2_SOURCES, "--name_weights", R15_NAMES, "--sampler_clamp_returns"]
    c = _parse(script, tmp, r15)
    assert c.H == 48 and c.policy_lr == c.critic_lr == c.wm.lr == 5e-5 and c.iterations == 136000
    assert c.robot_variant == "eval+evalcontact" and c.sampler_clamp_returns is True and c.eval_enabled is False
    assert c.source_weights == _weights(G1V2_SOURCES) and c.name_weights == _weights(R15_NAMES)
    assert (c.heads.ee_termination, c.heads.ee_term_z, c.heads.ee_term_z_temp) == (True, 0.25, 0.02)
    assert c.hard_names == () and c.hard_term_schedule == () and c.num_workers == 64 and c.motion_files == files
    # the r13 curriculum flags (master format)
    c = _parse(script, tmp, r15 + ["--hard_names", "kongfan,xuanfeng", "--hard_term_schedule", "104000:1.6,108000:1.6,114000:1"])
    assert c.hard_names == ("kongfan", "xuanfeng") and c.hard_term_schedule == R13_SCHEDULE
    for bad in (["--hard_term_schedule", "104000:1.6"], ["--hard_names", "x", "--hard_term_schedule", "104000"],
                ["--hard_names", "x", "--hard_term_schedule", "104000:0"]):
      try:
        _parse(script, tmp, r15 + bad, quiet=True)
      except SystemExit as e:
        assert e.code == 2
      else:
        raise AssertionError(f"must be rejected: {bad}")


def tsc_load_train():
  spec = importlib.util.spec_from_file_location("_train_script_1025", os.path.join(ROOT, "scripts", "train.py"))
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


def _parse(script, tmp: str, extra: list, quiet: bool = False) -> TrainCfg:
  seen = []

  def fake_trainer(cfg, dist=None):
    seen.append(cfg)
    raise tsc._Probe

  argv = ["train.py", "--log_root", tmp, "--device", "cpu"] + list(extra)
  with mock.patch.object(trainer_mod, "Trainer", fake_trainer), mock.patch.object(sys, "argv", argv), \
      mock.patch("sys.stderr", new=open(os.devnull, "w") if quiet else sys.stderr), \
      mock.patch("sys.stdout", new=open(os.devnull, "w")):
    try:
      script.main()
    except tsc._Probe:
      return seen[-1]
  raise AssertionError("Trainer not reached")


# ---------------------------------------------------------------------------
# (b) hard-clip termination curriculum
# ---------------------------------------------------------------------------


def _master_self(heads, ee_on: bool, thr: float, temp: float):
  cfg = types.SimpleNamespace(**{**dataclasses.asdict(heads.cfg), "ee_term_z": thr if ee_on else 0.0, "ee_term_temp": temp})
  return types.SimpleNamespace(cfg=cfg, anchor=heads.anchor, gravity=heads.gravity, ee=torch.as_tensor(EE_IDX if ee_on else [], dtype=torch.long))


def test_scaled_termination_equals_master_bitwise():
  for ee_on in (True, False):
    heads = make_heads(ee_on)
    m = _master_self(heads, ee_on, 0.25, 0.02)
    for seed in range(4):
      ref_bp, ref_bq, bp0, bq0 = random_batch(4096, seed, spread=0.45)
      g = torch.Generator().manual_seed(100 + seed)
      scales = {
        "curriculum": 1.0 + 0.6 * (torch.rand(4096, generator=g) < 0.3).float(),  # r13 x1.6 on the hard set
        "random": 0.5 + 2.0 * torch.rand(4096, generator=g),
        "ones": torch.ones(4096),
      }
      for name, k in scales.items():
        outs = []
        for which in ("ours", "master"):
          bp, bq = bp0.clone().requires_grad_(True), bq0.clone().requires_grad_(True)
          kin = kin_of(bp, bq)
          if which == "ours":
            hard, soft = heads.termination(kin, ref_bp[:, ANCHOR], ref_bq[:, ANCHOR], ref_bp, k)
          else:
            hard, soft = termination_master(m, kin, ref_bp, ref_bq, k)
          gp, gq = torch.autograd.grad((soft * torch.linspace(0.5, 1.5, 4096)).sum(), (bp, bq))
          outs.append((hard, soft.detach(), gp, gq))
        assert 0 < int(outs[0][0].sum()) < 4096
        for a, b in zip(*outs):
          assert torch.equal(a, b), f"ee={ee_on} {name}: differs from master 4314a50"
      # the factor really moves every criterion: x2 terminates strictly fewer, never more
      h1, _ = heads.termination(kin_of(bp0, bq0), ref_bp[:, ANCHOR], ref_bq[:, ANCHOR], ref_bp)
      h2, _ = heads.termination(kin_of(bp0, bq0), ref_bp[:, ANCHOR], ref_bq[:, ANCHOR], ref_bp, torch.full((4096,), 2.0))
      assert int(h2.sum()) < int(h1.sum()) and not bool((h2 & ~h1).any())


def test_scale_none_is_the_code_before():
  for ee_on in (True, False):
    heads = make_heads(ee_on)
    ref_bp, ref_bq, bp0, bq0 = random_batch(2048, 7)
    outs = []
    for fn in ("none", "ones", "before"):
      bp, bq = bp0.clone().requires_grad_(True), bq0.clone().requires_grad_(True)
      args = (kin_of(bp, bq), ref_bp[:, ANCHOR], ref_bq[:, ANCHOR], ref_bp)
      if fn == "none":
        hard, soft = heads.termination(*args)
      elif fn == "ones":
        hard, soft = heads.termination(*args, torch.ones(2048))
      else:
        hard, soft = termination_before(heads, *args)
      gp, gq = torch.autograd.grad(soft.sum(), (bp, bq))
      outs.append((hard, soft.detach(), gp, gq))
    for o in outs[1:]:
      for a, b in zip(outs[0], o):
        assert torch.equal(a, b)
    # compiled, as StepFns runs it
    new = torch.compile(lambda k, ap, aq, bp: heads.termination(k, ap, aq, bp, None), dynamic=False)
    old = torch.compile(lambda k, ap, aq, bp: termination_before(heads, k, ap, aq, bp), dynamic=False)
    a = new(kin_of(bp0, bq0), ref_bp[:, ANCHOR], ref_bq[:, ANCHOR], ref_bp)
    b = old(kin_of(bp0, bq0), ref_bp[:, ANCHOR], ref_bq[:, ANCHOR], ref_bp)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_hard_mult_and_mask_equal_master():
  t = object.__new__(Trainer)
  for sched in ((), R13_SCHEDULE, R14_SCHEDULE, ((10, 2.0),), ((0, 3.0), (5, 1.0), (5, 2.0), (9, 1.0))):
    t.cfg = TrainCfg(hard_names=("x",) if sched else (), hard_term_schedule=sched)
    for it in list(range(0, 130_000, 997)) + [104000, 108000, 111000, 114000, 116000, 126000, 5, 6, 9]:
      assert Trainer.hard_mult_at(t, it) == hard_mult_at_master(sched, it), (sched, it)
  assert abs(Trainer.hard_mult_at(types.SimpleNamespace(cfg=TrainCfg(hard_names=("x",), hard_term_schedule=R13_SCHEDULE)), 111000) - 1.3) < 1e-12
  names = ["g1_phc__noetix_kungfu_x_cekongfan-M", "g1__lafan1_walk", "g1__KongFan_y", "g1__xuanfengjiao", "g1__a"]
  for hn in ((), ("kongfan",), ("KONGFAN", "xuanfeng"), ("zz",)):
    assert torch.equal(hard_clip_mask(names, hn), clip_hard_master(names, hn))
  assert hard_clip_mask(names, ("kongfan", "xuanfeng")).tolist() == [True, False, True, True, False]


def test_env_terms_equal_mjlab_times_factor():
  if "mjlab" in tsc.STUBBED:  # stub: mjlab's math + _get_body_indexes as on bj31 (real mjlab: used as is)
    rew = types.ModuleType("mjlab.tasks.tracking.mdp.rewards")
    rew._get_body_indexes = _get_body_indexes
    sys.modules["mjlab.tasks.tracking.mdp.rewards"] = rew
    patch = mock.patch.object(lc_mod, "quat_apply_inverse", quat_apply_inverse)
  else:
    patch = mock.patch.object(lc_mod, "quat_apply_inverse", lc_mod.quat_apply_inverse)
  E, Bt = 4096, BT
  ref_bp, ref_bq, bp, bq = random_batch(E, 11, spread=0.45)
  g = torch.Generator().manual_seed(3)
  env_hard = torch.rand(E, generator=g) < 0.4
  cmd = types.SimpleNamespace(
    anchor_pos_w=ref_bp[:, ANCHOR], robot_anchor_pos_w=bp[:, ANCHOR], anchor_quat_w=ref_bq[:, ANCHOR], robot_anchor_quat_w=bq[:, ANCHOR],
    body_pos_relative_w=ref_bp + torch.tensor([0.3, -0.2, 0.0]), robot_body_pos_w=bp, cfg=types.SimpleNamespace(body_names=G1_TRACKED),
    env_hard=env_hard, hard_mult=1.0, term_scale_env=torch.ones(E),
  )
  cmd.set_hard_mult = types.MethodType(LibraryMotionCommand.set_hard_mult, cmd)
  grav = torch.tensor([0.0, 0.0, -1.0]).expand(E, 3)
  env = types.SimpleNamespace(command_manager=types.SimpleNamespace(get_term=lambda n: cmd),
                              scene={"robot": types.SimpleNamespace(data=types.SimpleNamespace(gravity_vec_w=grav))})
  asset = types.SimpleNamespace(name="robot")
  ee = ("left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link", "right_wrist_yaw_link")
  with patch:
    for m in (1.0, 2.0, 0.5, 1.5):
      cmd.set_hard_mult(m)
      assert torch.equal(cmd.term_scale_env, 1.0 + (m - 1.0) * env_hard.float())
      pairs = (
        (lc_mod.scaled_anchor_pos_z(env, "motion", 0.25), lambda thr: bad_anchor_pos_z_only(env, "motion", thr), 0.25),
        (lc_mod.scaled_anchor_ori(env, asset, "motion", 0.8), lambda thr: bad_anchor_ori(env, asset, "motion", thr), 0.8),
        (lc_mod.scaled_body_pos_z(env, "motion", 0.25, ee), lambda thr: bad_motion_body_pos_z_only(env, "motion", thr, ee), 0.25),
      )
      for got, orig, thr in pairs:
        if m == 1.0:
          assert torch.equal(got, orig(thr)), "factor 1 == mjlab's term"
        exp = torch.where(env_hard, orig(thr * m), orig(thr))  # per env: the threshold of its own factor
        assert torch.equal(got, exp)
        assert 0 < int(got.sum()) < E


def _fake_base_full(robot: str, variant: str = ""):
  c = tsc._fake_base_cfg(robot, variant)
  ns = types.SimpleNamespace
  c.terminations = {
    "anchor_pos": ns(func="mjlab.bad_anchor_pos_z_only", params={"threshold": 0.25}),
    "anchor_ori": ns(func="mjlab.bad_anchor_ori", params={"threshold": 0.8}),
    "ee_body_pos": ns(func="mjlab.bad_motion_body_pos_z_only", params={"body_names": ("left_ankle_roll_link",), "threshold": 0.0}),
  }
  return c


def test_make_env_cfg_swaps_terms_only_with_hard_names():
  with mock.patch.object(env_cfg_mod, "base_tracking_cfg", _fake_base_full):
    for mode in ("train", "eval"):
      for ee in (True, False):
        for hn in ((), ("kongfan",)):
          e = env_cfg_mod.make_env_cfg(["a.npz"], 4, mode=mode, ee_termination=ee, hard_names=hn, clamp_returns=True)
          c = e.commands["motion"]
          assert isinstance(c, LibraryMotionCommandCfg) and c.clamp_returns is True
          on = bool(hn) and mode == "train"
          assert c.hard_names == (hn if mode == "train" else ())
          t = e.terminations
          assert (t["anchor_pos"].func is lc_mod.scaled_anchor_pos_z) == on and (t["anchor_ori"].func is lc_mod.scaled_anchor_ori) == on
          if not on:
            assert t["anchor_pos"].func == "mjlab.bad_anchor_pos_z_only" and t["anchor_ori"].func == "mjlab.bad_anchor_ori"
          if ee:
            assert (t["ee_body_pos"].func is lc_mod.scaled_body_pos_z) == on and t["ee_body_pos"].params["threshold"] == 0.25
          else:
            assert "ee_body_pos" not in t


def _cur_worker(rank: int, world: int, port: int, tmp: str):
  os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world))
  torch.set_num_threads(1)
  ctx = DistCtx.from_env("cpu", timeout_min=2.0)
  NB, E = 512, 64
  names = ctx.broadcast_object([f"g1__src_{i}_{'kongfan' if i % 5 == 0 else 'walk'}.npz" for i in range(40)] if rank == 0 else None)
  bin_clip = torch.arange(NB) * len(names) // NB  # bins in clip order
  t = tsc.make_trainer(tmp, NB, True, dist=ctx)
  t.cfg = dataclasses.replace(t.cfg, hard_names=("kongfan",), hard_term_schedule=R13_SCHEDULE)
  c = t.cmd
  c.clip_hard = hard_clip_mask(names, t.cfg.hard_names)
  g = torch.Generator().manual_seed(rank)
  c.env_hard = c.clip_hard[bin_clip[torch.randint(0, NB, (E,), generator=g)]]  # each rank: its own envs' clips
  c.hard_mult, c.term_scale_env = 1.0, torch.ones(E)
  c.set_hard_mult = types.MethodType(LibraryMotionCommand.set_hard_mult, c)
  Trainer._dist_start(t, "start")  # incl. the curriculum group
  # the loop's iterations, with a rollback (back to 106000) and a resume (to 100000) in between
  its = list(range(103000, 109000, 250)) + list(range(106000, 115000, 250)) + list(range(100000, 116000, 1000))
  mults = []
  for it in its:
    t.it = it
    c.set_hard_mult(Trainer.hard_mult_at(t, it))  # as Trainer.train does at the top of every iteration
    mults.append(c.hard_mult)
    assert torch.equal(c.term_scale_env, 1.0 + (c.hard_mult - 1.0) * c.env_hard.float())
    if it % 2000 == 0:
      Trainer._check_consistency(t, f"it {it}")  # curriculum group: same hard set + multiplier everywhere
  assert mults == [hard_mult_at_master(R13_SCHEDULE, it) for it in its]
  allm = ctx.all_gather_object(mults)
  assert all(m == allm[0] for m in allm), "multiplier differs between ranks"
  env_hard_all = ctx.all_gather_object(c.env_hard.tolist())
  assert any(e != env_hard_all[0] for e in env_hard_all), "per-env state is per rank (own envs)"
  assert {1.6, 1.0} <= set(mults) and any(1.0 < m < 1.6 for m in mults)
  # a rank whose hard set differs (another --hard_names) is caught by the state check, on every rank
  if rank == world - 1:
    c.clip_hard = hard_clip_mask(names, ("walk",))
  try:
    Trainer._check_consistency(t, "bad hard set")
  except RuntimeError as e:
    assert "curriculum differs" in str(e), str(e)
  else:
    raise AssertionError("another hard set must be caught")
  ctx.barrier()
  ctx.close()
  open(os.path.join(tmp, f"rank{rank}.ok"), "w").close()


def test_ddp_curriculum_world3():
  with tempfile.TemporaryDirectory() as tmp:
    mp.spawn(_cur_worker, args=(3, tsc._free_port(), tmp), nprocs=3, join=True)
    assert all(os.path.exists(os.path.join(tmp, f"rank{r}.ok")) for r in range(3))


# ---------------------------------------------------------------------------
# (c) extend_cache
# ---------------------------------------------------------------------------


def _write_clips(d: str, specs) -> list[str]:
  """specs: (name, T, seed, with body_names, fps) -> npz paths."""
  os.makedirs(d, exist_ok=True)
  out = []
  for name, T, seed, names, fps in specs:
    c = _clip(np.random.default_rng(seed), T, with_names=names)
    c["fps"] = np.array([fps])
    np.savez(os.path.join(d, name), **c)
    out.append(os.path.join(d, name))
  return out


OLD = [("g1__a.npz", 120, 0, True, 50.0), ("g1__b c.npz", 75, 1, False, 50.0), ("g1__sonic_c.npz", 51, 2, True, 50.0)]
EXTRA = [("g1_phc__noetix_kungfu_x.npz", 140, 3, False, 50.0), ("g1_phc__bad_fps.npz", 60, 4, True, 30.0),
         ("g1_phc__noetix_kungfu_y.npz", 33, 5, True, 50.0)]


def _lib(files, cache):
  return MotionLibrary(files, DST_JOINTS, TRACKED, device="cpu", cache_dir=cache, num_workers=2, log=lambda *a: None)


def _extended(tmp: str):
  data = os.path.join(tmp, "data")
  old, extra = _write_clips(data, OLD), _write_clips(data, EXTRA)
  root = os.path.join(tmp, "cache")
  old_lib = _lib(old, root)
  new_dir = library_cache_dir(root, old + extra, DST_JOINTS, TRACKED, 50.0)
  extend_cache(old_lib.cache_dir, old, extra, new_dir, num_workers=2, log=lambda *a: None)
  return old, extra, root, old_lib, new_dir


def test_extend_equals_full_rebuild():
  with tempfile.TemporaryDirectory() as tmp:
    old, extra, root, old_lib, new_dir = _extended(tmp)
    full = _lib(old + extra, os.path.join(tmp, "cache_full"))  # build_cache of the whole list
    assert os.path.basename(full.cache_dir) == os.path.basename(new_dir)
    for k in motions_mod.FIELD_DTYPES:
      a, b = np.load(os.path.join(full.cache_dir, f"{k}.npy")), np.load(os.path.join(new_dir, f"{k}.npy"))
      assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), k
    ca, cb = json.load(open(os.path.join(full.cache_dir, "clips.json"))), json.load(open(os.path.join(new_dir, "clips.json")))
    assert ca == cb and [c["ok"] for c in cb] == [True, True, True, True, False, True], "the 30 fps file fails in both"
    meta = json.load(open(os.path.join(new_dir, "meta.json")))
    assert meta["files"] == 6 and meta["extra_failed"] == 1 and meta["body_names_fallback_files"] == 2
    with mock.patch.object(motions_mod, "build_cache", side_effect=AssertionError("must not rebuild")):
      ext = _lib(old + extra, root)  # MotionLibrary looks up exactly the extended cache
    assert ext.cache_dir == new_dir and ext.fingerprint == full.fingerprint != old_lib.fingerprint
    assert ext.num_bins == full.num_bins and ext.bins_of_first_files(3) == old_lib.num_bins
    assert torch.equal(ext.bin_start[: old_lib.num_bins], old_lib.bin_start) and torch.equal(ext.bin_clip[: old_lib.num_bins], old_lib.bin_clip)
    assert ext.bins_of_first_files(6) == ext.num_bins and ext.bins_of_first_files(0) == 0 and ext.bins_of_first_files(7) == -1
    # this branch's per-file writer reports success as (path, None, fallback): master's `r is not None` would fail all
    motions_mod._worker_init(new_dir, DST_JOINTS, TRACKED, {k: np.load(os.path.join(new_dir, f"{k}.npy"), mmap_mode="r").shape for k in motions_mod.FIELD_DTYPES})
    r = motions_mod._worker_fill((extra[0], 120 + 75 + 51, 140, 50.0))
    assert r == (extra[0], None, True)


def test_extend_script():
  script_spec = importlib.util.spec_from_file_location("_extend_script", os.path.join(ROOT, "scripts", "extend_cache.py"))
  script = importlib.util.module_from_spec(script_spec)
  script_spec.loader.exec_module(script)
  with tempfile.TemporaryDirectory() as tmp:
    data = os.path.join(tmp, "data")
    old, extra = _write_clips(data, OLD), _write_clips(data, EXTRA)
    root = os.path.join(tmp, "cache")
    _lib(old, root)
    lists = {}
    for n, fs in (("old", old), ("extra", extra + old[:1]), ("out", old + extra), ("wrong", extra + old)):
      lists[n] = os.path.join(tmp, f"{n}.txt")
      with open(lists[n], "w") as f:
        f.write("\n".join(fs) + "\n")

    def run(out: str, *extra_args):
      argv = ["extend_cache.py", "--old_list", lists["old"], "--extra", lists["extra"], "--cache_dir", root, "--out_list", out,
              "--num_workers", "2", *extra_args]
      with mock.patch.object(sys, "argv", argv), mock.patch("sys.stdout", new=open(os.devnull, "w")):
        script.main()

    try:
      run(lists["wrong"])
    except SystemExit as e:
      assert "another content" in str(e)
    else:
      raise AssertionError("an --out_list with another content must be refused")
    new_dir = library_cache_dir(root, old + extra, DST_JOINTS, TRACKED, 50.0)
    run(lists["out"], "--dry_run")
    assert not os.path.exists(new_dir)
    run(lists["out"])  # exists with exactly this content (as the prepared bj31 list): accepted, cache built
    assert os.path.exists(os.path.join(new_dir, "meta.json"))
    run(lists["out"])  # idempotent
    fresh = os.path.join(tmp, "fresh", "list.txt")
    run(fresh)
    assert [l.strip() for l in open(fresh) if l.strip()] == old + extra


def test_resume_into_extended_library_keeps_sampler_stats():
  with tempfile.TemporaryDirectory() as tmp:
    old, extra, root, old_lib, new_dir = _extended(tmp)
    ext = _lib(old + extra, root)
    nb, NB = old_lib.num_bins, ext.num_bins
    g = torch.Generator().manual_seed(1)
    G = torch.randn(nb, generator=g) * 6.0 + 3.0  # some < 0 and > g_max
    V = torch.randint(1, 50, (nb,), generator=g).float()
    with tempfile.TemporaryDirectory() as ck_dir:
      base = tsc.make_trainer(ck_dir, nb, False)
      bj27 = [f"/bj27/{os.path.basename(f)}" for f in old]  # the old library's run, on the other machine
      path = os.path.join(ck_dir, "model_128000.pt")
      _save_ck(path, base, G, V, _master_train_cfg(bj27, sampler_clamp_returns=False, hard_names=("kongfan",), hard_term_schedule=R14_SCHEDULE), it=128000, lr_scale=1.0)
      for clamp in (False, True):
        t = tsc.make_trainer(ck_dir, NB, clamp)
        t.lib = ext
        t.cfg = dataclasses.replace(t.cfg, motion_files=old + extra)
        Trainer.load(t, path)
        exp_G = G.clamp(0.0, tsc.G_MAX) if clamp else G
        assert torch.equal(t.cmd.bin_G[:nb], exp_G) and torch.equal(t.cmd.bin_visits[:nb], V)
        assert float(t.cmd.bin_G[nb:].abs().sum()) == 0.0 and float(t.cmd.bin_visits[nb:].sum()) == 0.0, "new bins: unvisited"
        (msg,) = [s for s in t.prints if "restored into the extended library" in s]
        assert f"{nb} bins" in msg and f"({NB} bins" in msg
        assert any("3 + 3 appended here" in s for s in t.prints)
        # same as master's load with the same paths (master compares full paths; here basenames: bj27 vs bj31 roots)
        ref = tsc.make_cmd(NB, clamp)
        ck = torch.load(path, weights_only=False)
        ck["cfg"]["motion_files"] = old
        master_load_sampler(ref, ck, old + extra, clamp)
        assert torch.equal(t.cmd.bin_G, ref.bin_G) and torch.equal(t.cmd.bin_visits, ref.bin_visits)
      # refused: the old library had another bin layout (e.g. a clip failed there) / another order / not a prefix
      for files, Gx, why in ((bj27, torch.zeros(nb + 1), "have"), (bj27[::-1], G, "not loaded"), (bj27[1:], G, "not loaded")):
        _save_ck(path, base, Gx, torch.ones(Gx.shape[0]), _master_train_cfg(files), it=128000, lr_scale=1.0)
        t = tsc.make_trainer(ck_dir, NB, True)
        t.lib = ext
        t.cfg = dataclasses.replace(t.cfg, motion_files=old + extra)
        Trainer.load(t, path)
        assert float(t.cmd.bin_visits.sum()) == 0.0
        (w,) = [s for s in t.prints if "not loaded" in s]
        assert why in w, w


def test_rollback_to_the_resume_checkpoint_keeps_its_clip_relation():
  """Reviewer 10-25: until this run writes a healthy checkpoint, the watchdog rolls back to the --resume one (r15: twice
  to r14's model_128000). That reload must use the relation the resume found, not "same": an extended library gets the
  checkpoint's first bins back (before: none restored, collapse-time statistics kept, "starts from scratch" logged); a
  checkpoint of the same clips in another order loads nothing (before: its bins were copied onto the other clips).
  Bins not taken from the checkpoint keep their current statistics (master's load). A rollback to this run's own
  checkpoint (another path) is "same" as before."""
  with tempfile.TemporaryDirectory() as tmp:
    old, extra, root, old_lib, new_dir = _extended(tmp)
    ext = _lib(old + extra, root)
    nb, NB = old_lib.num_bins, ext.num_bins
    G, V = torch.linspace(1, 9, nb), torch.full((nb,), 7.0)
    with tempfile.TemporaryDirectory() as ck_dir:
      base = tsc.make_trainer(ck_dir, nb, False)
      path = os.path.join(ck_dir, "model_128000.pt")
      _save_ck(path, base, G, V, _master_train_cfg([f"/bj27/{os.path.basename(f)}" for f in old], sampler_clamp_returns=True), it=128000, lr_scale=1.0)
      t = tsc.make_trainer(ck_dir, NB, True)
      t.lib = ext
      t.cfg = dataclasses.replace(t.cfg, motion_files=old + extra)
      Trainer.load(t, path)
      assert t._resume_from == (path, "prefix") and torch.equal(t.cmd.bin_G[:nb], G) and t.wd_healthy_ckpt == path
      t.cmd.bin_G.fill_(3.0)  # training moved the statistics
      t.cmd.bin_visits.fill_(99.0)
      t.prints.clear()
      t.it = 129769
      assert Trainer._rollback(t, "test") is True and t.it == 128000
      assert torch.equal(t.cmd.bin_G[:nb], G) and torch.equal(t.cmd.bin_visits[:nb], V), "first nb bins = the checkpoint"
      assert float(t.cmd.bin_G[nb:].sub(3.0).abs().max()) == 0.0 and float(t.cmd.bin_visits[nb:].sub(99.0).abs().max()) == 0.0
      (msg,) = [s_ for s_ in t.prints if "restored into the extended library" in s_]
      assert f"the {NB - nb} appended bins keep their current statistics" in msg, msg
      assert not any("WARNING" in s_ for s_ in t.prints), "a rollback reports no settings diff"
      # rollback to this run's own checkpoint: "same" (its bin_G covers every bin)
      own = os.path.join(ck_dir, "model_130000.pt")
      _save_ck(own, t, torch.full((NB,), 4.0), torch.full((NB,), 5.0), dataclasses.asdict(t.cfg), it=130000, lr_scale=0.5)
      t.wd_healthy_ckpt, t.it = own, 130338
      assert Trainer._rollback(t, "test") is True and t.it == 130000
      assert float(t.cmd.bin_G.sub(4.0).abs().max()) == 0.0 and float(t.cmd.bin_visits.sub(5.0).abs().max()) == 0.0
  # same clips, other order (same bin count): refused at the resume, and at the rollback
  NB = 64
  files = [f"/x/g1__lafan1_{i}.npz" for i in range(4)]
  with tempfile.TemporaryDirectory() as d:
    base = tsc.make_trainer(d, NB, False)
    G = torch.linspace(1, 9, NB)
    path = os.path.join(d, "ck.pt")
    _save_ck(path, base, G, torch.full((NB,), 7.0), _master_train_cfg(files[::-1]), it=1000, lr_scale=1.0)
    t = tsc.make_trainer(d, NB, False)
    _r15_here(t, files)
    Trainer.load(t, path)
    assert t._resume_from == (path, "other") and float(t.cmd.bin_visits.sum()) == 0.0
    t.cmd.bin_G.fill_(-3.0)
    t.prints.clear()
    t.it = 1500
    assert Trainer._rollback(t, "test") is True and t.it == 1000
    assert float(t.cmd.bin_G.add(3.0).abs().max()) == 0.0, "not copied onto the other clips"
    (w,) = [s_ for s_ in t.prints if "not loaded" in s_]
    assert "same bin count, other clip list" in w and "kept the current statistics (rollback)" in w, w


def _ext_worker(rank: int, world: int, port: int, tmp: str, root: str, files: list, ck: str):
  os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world))
  torch.set_num_threads(1)
  ctx = DistCtx.from_env("cpu", timeout_min=2.0)
  with mock.patch.object(motions_mod, "build_cache", side_effect=AssertionError("must not rebuild")):
    lib = _lib(files, root)
  t = tsc.make_trainer(tmp, lib.num_bins, True, dist=ctx)
  t.lib = lib
  t.cfg = dataclasses.replace(t.cfg, motion_files=files)
  t.cmd.bin_weight_hist = t.cmd.bin_weight_hist * (1.0 + rank)
  Trainer.load(t, ck)
  Trainer._dist_start(t, "after load")  # library group + sampler state identical everywhere (raises otherwise)
  fps = ctx.all_gather_object(lib.fingerprint)
  assert all(f == fps[0] for f in fps)
  Gs = ctx.all_gather_object(t.cmd.bin_G.tolist())
  assert all(x == Gs[0] for x in Gs)
  if rank == 1:  # a rank that mapped another library is caught
    t.lib = types.SimpleNamespace(fingerprint="another library")
  try:
    Trainer._check_consistency(t, "bad library")
  except RuntimeError as e:
    assert "library differs" in str(e), str(e)
  else:
    raise AssertionError("another library must be caught")
  ctx.barrier()
  ctx.close()
  open(os.path.join(tmp, f"rank{rank}.ok"), "w").close()


def test_ddp_extended_library_world2():
  with tempfile.TemporaryDirectory() as tmp:
    old, extra, root, old_lib, new_dir = _extended(tmp)
    nb = old_lib.num_bins
    base = tsc.make_trainer(tmp, nb, False)
    ck = os.path.join(tmp, "model_128000.pt")
    _save_ck(ck, base, torch.linspace(-5, 15, nb), torch.ones(nb), _master_train_cfg(old), it=128000, lr_scale=1.0)
    mp.spawn(_ext_worker, args=(2, tsc._free_port(), tmp, root, old + extra, ck), nprocs=2, join=True)
    assert all(os.path.exists(os.path.join(tmp, f"rank{r}.ok")) for r in range(2))


# ---------------------------------------------------------------------------
# (d) scripts/launch_bj31_r15ws.example.sh reproduces r15
# ---------------------------------------------------------------------------

# cfg of bj27 g1v2_clamp_r15 model_136000.pt (torch.load on bj27's CPU, 10-25; motion_files left out: 255,008 paths whose
# basename sequence md5 fbd2ddf1879f61b71b42f072144ea4af equals the one of bj31's g1v2_r15_255008_bj31.txt). It also
# carried: it 136000, lr_scale 0.0625, bin_G 5,428,418 bins in [0, 9.86], dims proprio 93 / goal 64 / nj 29 / xi 33.
R15_CKPT_CFG = json.loads(
  '{"robot": "g1", "robot_variant": "eval+evalcontact", "source_weights": [["noetix_kungfu", 10.0], '
  '["noetix_kungfu_dance_show", 15.0], ["sportsmeeting", 10.0], ["lafan1", 15.0], ["g1_all", 10.0], ["mixamo_cn", 2.0], '
  '["virtual_point", 3.0]], "name_weights": [["tengkong", 200.0], ["kongfan", 200.0], ["shoufan", 200.0], ["erqijiao", '
  '200.0], ["danpaijiao", 200.0], ["xuanfeng", 200.0], ["waibailian", 100.0], ["lihetui", 100.0], ["saotang", 100.0], '
  '["ceti", 100.0], ["houti", 100.0], ["zhengti", 100.0]], "hard_names": [], "hard_term_schedule": [], '
  '"sampler_clamp_returns": true, "num_envs": 4096, "device": "cuda:0", "seed": 0, "log_dir": '
  '"/home/ubuntu/supermimic/logs/2026-10-23_20-39-52_g1v2_clamp_r15", "cache_dir": '
  '"/home/ubuntu/supermimic/data/cache", "num_workers": 64, "iterations": 136000, "env_steps_per_iter": 1.0, '
  '"action_noise_std": 0.1, "replay_capacity": 1000, "episode_length_s": 10.0, "compile": true, "compile_mode": null, '
  '"H": 48, "rollouts_per_update": 4096, "gamma": 0.99, "lam": 0.95, "policy_lr": 5e-05, "policy_betas": [0.9, 0.95], '
  '"critic_lr": 5e-05, "critic_epochs": 5, "critic_minibatches": 4, "grad_clip_norm": 1.0, "grad_clip_elem": 0.1, '
  '"lr_decay_iters": 250000, "lr_final_ratio": 0.03, "use_critic": true, "critic_huber_delta": 1.0, '
  '"clip_value_targets": true, "wd_enabled": true, "wd_ema": 0.02, "wd_drop": 0.2, "wd_window": 4000, "wd_vloss_ratio": '
  '50.0, "wd_warmup": 500, "wd_lr_factor": 0.5, "wd_max_rollbacks": 6, "wd_force_at": -1, "log_interval": 50, '
  '"eval_interval": 2000, "ckpt_interval": 2000, "eval_enabled": false, "eval_clips": 1024, "eval_envs": 1024, "wm": '
  '{"hidden": [2048, 2048, 2048], "k": 8, "lr": 5e-05, "grad_clip": 1.0, "batches_per_update": 10, "windows_per_batch": '
  '2048}, "heads": {"K": 10, "hist": 10, "dt": 0.02, "goal_anchor_pos": false, "noise_joint_pos": 0.01, '
  '"noise_joint_vel": 0.5, "noise_ang_vel": 0.2, "noise_gravity": 0.05, "w_body_pos": 1.0, "w_body_ori": 1.0, '
  '"w_body_lin_vel": 1.0, "w_body_ang_vel": 1.0, "w_root_pos": 0.5, "w_root_ori": 0.5, "w_action_rate": -0.1, '
  '"w_joint_limit": -2.0, "std_body_pos": 0.3, "std_body_ori": 0.4, "std_body_lin_vel": 1.0, "std_body_ang_vel": 3.14, '
  '"std_root_pos": 0.3, "std_root_ori": 0.4, "term_z": 0.25, "term_tilt": 0.8, "term_z_temp": 0.02, "term_tilt_temp": '
  '0.05, "ee_term_z": 0.25, "ee_term_temp": 0.02, "extra": {}}}'
)


def _example_args(tmp: str) -> tuple[list[str], dict]:
  """The ARGS block of scripts/launch_bj31_r15ws.example.sh with its variables (paths -> temp files)."""
  import re
  import shlex

  text = open(os.path.join(ROOT, "scripts", "launch_bj31_r15ws.example.sh"), encoding="utf-8").read()
  body = text.split("# >>> ARGS", 1)[1].split("# <<< ARGS", 1)[0]
  body = body.split("ARGS=(", 1)[1].rsplit(")", 1)[0]
  vars_ = dict(re.findall(r'^([A-Z_]+)=([^\s"$][^\s]*)$', text, flags=re.M))  # plain NAME=value lines
  vars_["ITERS"] = re.search(r'ITERS="\$\{ITERS:-(\d+)\}"', text).group(1)
  vars_["LR"] = re.search(r'^LR="\$\{LR:-([0-9.e+-]+)\}"', text, flags=re.M).group(1)
  files = [os.path.join(tmp, f"g1__lafan1_{i}.npz") for i in range(3)]
  for f in files:
    open(f, "w").close()
  vars_["LIST"] = os.path.join(tmp, "list.txt")
  with open(vars_["LIST"], "w") as f:
    f.write("\n".join(files) + "\n")
  vars_["EVAL_NAMES"] = os.path.join(tmp, "eval_names.txt")
  with open(vars_["EVAL_NAMES"], "w") as f:
    f.write(os.path.basename(files[0]) + "\n")
  for k, v in vars_.items():
    body = body.replace(f'"${k}"', shlex.quote(v)).replace(f"${k}", v)
  assert "$" not in body, body
  return shlex.split(body.replace("\\\n", " ")), vars_


def test_r15ws_example_reproduces_r15():
  script = tsc_load_train()
  with tempfile.TemporaryDirectory() as tmp:
    args, v = _example_args(tmp)
    c = _parse(script, tmp, args)
  assert v["SOURCES"] == G1V2_SOURCES and v["NAMES"] == R15_NAMES
  # resume guard against r15's own checkpoint cfg: nothing differs (robot / variant / source + name weights / clamp /
  # curriculum / H / every heads field incl. the EE termination in master's spelling)
  t = object.__new__(Trainer)
  t.cfg = c
  diffs, rel = Trainer._resume_diffs(t, R15_CKPT_CFG)
  assert diffs == [] and rel == "same", diffs
  r15 = R15_CKPT_CFG
  same = ("num_envs", "rollouts_per_update", "gamma", "lam", "critic_epochs", "critic_minibatches", "grad_clip_norm", "grad_clip_elem",
          "lr_decay_iters", "lr_final_ratio", "use_critic", "critic_huber_delta", "clip_value_targets", "wd_enabled", "wd_ema", "wd_drop",
          "wd_window", "wd_vloss_ratio", "wd_warmup", "wd_lr_factor", "wd_max_rollbacks", "action_noise_std", "replay_capacity",
          "episode_length_s", "env_steps_per_iter", "seed", "num_workers", "ckpt_interval", "log_interval", "eval_envs")
  for k in same:
    assert getattr(c, k) == r15[k], (k, getattr(c, k), r15[k])
  assert tuple(c.policy_betas) == tuple(r15["policy_betas"])
  for k in ("k", "batches_per_update", "windows_per_batch", "grad_clip"):
    assert getattr(c.wm, k) == r15["wm"][k], k
  assert tuple(c.wm.hidden) == tuple(r15["wm"]["hidden"])
  # the deliberate differences
  lam = trainer_mod.cosine_lambda(250_000, 0.03)(136000)
  assert c.policy_lr == c.critic_lr == c.wm.lr == 8.9252e-4
  assert abs(c.policy_lr * 0.0625 * lam - 2.5e-5) < 1e-9, c.policy_lr * 0.0625 * lam
  assert (c.eval_enabled, c.eval_interval, c.es_eval_fail_ratio, c.es_patience) == (True, 2000, 1.5, 2) and c.eval_clip_names
  assert (c.wd_signal, c.wd_reward_floor, c.iterations) == ("imagined_reward_alive", -1.0, 250000)
  # resume into this cfg: the optimizer LR after Trainer.load (lr_scale 0.0625 restored, cosine at 136000) = 2.5e-5
  with tempfile.TemporaryDirectory() as tmp:
    t = tsc.make_trainer(tmp, 64, True)
    t.opt_wm, t.opt_pi, t.opt_v = (torch.optim.Adam(m.parameters(), lr=c.policy_lr) for m in (t.wm, t.policy, t.critic))
    t.sched = [torch.optim.lr_scheduler.LambdaLR(o, trainer_mod.cosine_lambda(c.lr_decay_iters, c.lr_final_ratio)) for o in (t.opt_wm, t.opt_pi, t.opt_v)]
    t._base_lrs0 = [list(s.base_lrs) for s in t.sched]
    ck = os.path.join(tmp, "model_136000.pt")
    src = tsc.make_trainer(tmp, 64, True)
    for o in (src.opt_wm, src.opt_pi, src.opt_v):  # the checkpoint's optimizers: initial_lr 2e-4 from the scratch run
      for gr in o.param_groups:
        gr["initial_lr"], gr["lr"] = 2e-4, 1.40e-6
    _save_ck(ck, src, torch.rand(64) * 9.86, torch.ones(64), dict(R15_CKPT_CFG), it=136000, lr_scale=0.0625)
    Trainer.load(t, ck)
    for o in (t.opt_wm, t.opt_pi, t.opt_v):
      assert abs(o.param_groups[0]["lr"] - 2.5e-5) < 1e-9, o.param_groups[0]["lr"]  # 2.50001e-5 (--lr rounded to 5 digits)
    for s_ in t.sched:
      s_.step()
    lam1 = trainer_mod.cosine_lambda(250_000, 0.03)(136001)
    assert abs(t.opt_pi.param_groups[0]["lr"] - c.policy_lr * 0.0625 * lam1) < 1e-15
    (res,) = [x for x in t.prints if "resumed from" in x]
    assert "policy lr 2.500e-05 = lr x lr_scale 0.0625 x cosine 0.4482" in res, res
    print(f"  {res.split('] ', 1)[1]}")


if __name__ == "__main__":
  if tsc.STUBBED:
    print(f"  (stubbed via sys.modules: {', '.join(tsc.STUBBED)})")
  for name, fn in list(globals().items()):
    if name.startswith("test_") and callable(fn):
      fn()
      print(f"ok  {name}")
