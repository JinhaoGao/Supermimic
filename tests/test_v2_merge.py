"""master's v2 options merged into the data-parallel branch (ddp-bj31-v2); CPU, torch only.

  python tests/test_v2_merge.py      # or: pytest tests/test_v2_merge.py

mjlab / tensorboard are stubbed like in test_sampler_clamp.py when missing, so the REAL heads, sampler, Trainer,
DistCtx, make_env_cfg and CLI code runs here.

Checks:
  * one end-effector termination: this branch's engine == master's e57be5b implementation (copied verbatim below)
    bitwise -- hard, soft and gradients -- with master's end-effector index order (= the env term's body order, which
    is also the tracked order for G1 and E2), and on == off == before when the switch is off;
  * HeadsCfg.from_dict loads checkpoint cfgs of all three branches (ddp-bj31 / master v2 / v1); ee_flags maps master's
    --ee_term_z onto --ee_termination / --ee_threshold and rejects contradictions;
  * reset_prior == master's inline source / name prior (276c942 + b131dc0, verbatim) bitwise; with clamp_returns the
    prior only shapes the uniform share and the clamp only the softmax's G;
  * data parallel (2 and 3 gloo processes, real Trainer._dist_start / _merge_sampler): every rank builds the same
    prior from the broadcast clip list (checked by the start-up state check, which also catches a rank whose prior
    differs); after per-rank return updates + merge the sampling probabilities are bitwise equal on every rank;
  * wiring: the bj27 g1v2_scratch command line -> TrainCfg / HeadsCfg (scripts/train.py), conflicting flags rejected,
    --motion_list read verbatim (any missing file fails before the Trainer); Trainer.__init__ -> make_env_cfg kwargs; make_env_cfg -> base_tracking_cfg(robot,
    variant), source / name weights (train mode only), ee_body_pos threshold / removal; eval_ckpt.py on master-format,
    v1 and ddp-bj31 checkpoints (EE switch / threshold / robot variant as trained, overrides);
  * resume guard: Trainer.load warns about every differing setting of the checkpoint's run and loads the reset
    sampler's state only for the same clip list.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import os
import sys
import tempfile
import types
from unittest import mock

import torch
import torch.multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import test_sampler_clamp as tsc  # noqa: E402  (installs the mjlab / tensorboard stubs when needed)
from test_ee_termination import ANCHOR, BT, EE_IDX, G1_EE, G1_TRACKED, NJ, _FK, kin_of, random_batch  # noqa: E402

import supermimic.env_cfg as env_cfg_mod  # noqa: E402
import supermimic.motions as motions_mod  # noqa: E402
import supermimic.trainer as trainer_mod  # noqa: E402
from supermimic.distributed import DistCtx  # noqa: E402
from supermimic.heads import AnalyticHeads, HeadsCfg, ee_flags  # noqa: E402
from supermimic.library_command import LibraryMotionCommandCfg, reset_prior  # noqa: E402
from supermimic.math_utils import quat_apply_inverse  # noqa: E402
from supermimic.trainer import TrainCfg, Trainer  # noqa: E402

ROOT = os.path.join(HERE, "..")
E2_TRACKED = (
  "base_link", "l_leg_hip_roll_link", "l_leg_knee_link", "l_leg_ankle_roll_link", "r_leg_hip_roll_link", "r_leg_knee_link",
  "r_leg_ankle_roll_link", "waist_pitch_link", "l_arm_shoulder_roll_link", "l_arm_elbow_link", "r_arm_shoulder_roll_link",
  "r_arm_elbow_link",
)
E2_EE = ("l_leg_ankle_roll_link", "r_leg_ankle_roll_link", "l_arm_elbow_link", "r_arm_elbow_link")  # env term order
G1V2_SOURCES = "noetix_kungfu=10,noetix_kungfu_dance_show=15,sportsmeeting=10,lafan1=15,g1_all=10,mixamo_cn=2,virtual_point=3"


# ---------------------------------------------------------------------------
# master, verbatim
# ---------------------------------------------------------------------------


def termination_master(self, kin_next: dict, ref_bp: torch.Tensor, ref_bq: torch.Tensor):
  """heads.AnalyticHeads.termination at master e57be5b..18f528f, verbatim (cfg fields ee_term_z / ee_term_temp)."""
  c = self.cfg
  ref_ap, ref_aq = ref_bp[:, self.anchor], ref_bq[:, self.anchor]
  B = ref_ap.shape[0]
  g = self.gravity.expand(B, 3)
  dz = (ref_ap[:, 2] - kin_next["anchor_pos"][:, 2]).abs()
  tilt = (quat_apply_inverse(ref_aq, g)[:, 2] - quat_apply_inverse(kin_next["anchor_quat"], g)[:, 2]).abs()
  hard = (dz > c.term_z) | (tilt > c.term_tilt)
  pz = torch.sigmoid((dz - c.term_z) / c.term_z_temp)
  pt = torch.sigmoid((tilt - c.term_tilt) / c.term_tilt_temp)
  alive = (1.0 - pz) * (1.0 - pt)
  if c.ee_term_z > 0:
    ez = (ref_bp[:, self.ee, 2] - kin_next["body_pos"][:, self.ee, 2]).abs()  # [B, n_ee]
    hard = hard | (ez > c.ee_term_z).any(-1)
    alive = alive * (1.0 - torch.sigmoid((ez - c.ee_term_z) / c.ee_term_temp)).prod(-1)
  return hard, 1.0 - alive


def prior_master(clip_names, bin_clip, source_weights, name_weights, device="cpu"):
  """LibraryMotionCommand.__init__ source / name prior at master b131dc0, verbatim (prints dropped)."""
  kws = sorted(((str(k), float(v)) for k, v in source_weights), key=lambda kv: -len(kv[0]))
  nws = [(str(k).lower(), float(v)) for k, v in name_weights]
  def src_weight(name: str) -> tuple[str, float]:
    src = name.split("__", 1)[-1]
    for k, v in kws:
      if src == k or src.startswith(k + "_"):
        return k, v
    return "", 1.0
  clips = [types.SimpleNamespace(name=n) for n in clip_names]
  tags = [src_weight(c.name) for c in clips]
  name_w = [max([v for k, v in nws if k in c.name.lower()], default=1.0) for c in clips]
  w_clip = torch.tensor([max(w, nw) for (_, w), nw in zip(tags, name_w)], dtype=torch.float32, device=device)
  prior = w_clip[bin_clip]
  return prior / prior.sum()


def master_heads_dict(**over) -> dict:
  """cfg["heads"] as master (v2) writes it: no ee_termination, ee_term_temp instead of ee_term_z_temp."""
  d = dataclasses.asdict(HeadsCfg())
  d.pop("ee_termination")
  d["ee_term_temp"] = d.pop("ee_term_z_temp")
  d.update(over)
  return d


def v1_heads_dict() -> dict:
  d = dataclasses.asdict(HeadsCfg())
  for k in ("ee_termination", "ee_term_z", "ee_term_z_temp"):
    d.pop(k)
  return d


# ---------------------------------------------------------------------------
# (a) one end-effector termination
# ---------------------------------------------------------------------------


def _heads(cfg: HeadsCfg, tracked, ee_names) -> AnalyticHeads:
  ee = [i for i, n in enumerate(tracked) if n in ee_names] if cfg.ee_termination else None  # SimInterface.ee_bodies
  anchor = tracked.index("torso_link") if "torso_link" in tracked else tracked.index("waist_pitch_link")
  return AnalyticHeads(_FK(), list(range(len(tracked))), anchor, torch.zeros(NJ), torch.zeros(NJ, 2), cfg, ee_indices=ee)


def _master_self(heads: AnalyticHeads, tracked, ee_names, ee_term_z: float, temp: float):
  cfg = types.SimpleNamespace(**{**dataclasses.asdict(heads.cfg), "ee_term_z": ee_term_z, "ee_term_temp": temp})
  ee = [tracked.index(n) for n in ee_names] if ee_term_z > 0 else []  # master: the term's body_names order
  return types.SimpleNamespace(cfg=cfg, anchor=heads.anchor, gravity=heads.gravity, ee=torch.as_tensor(ee, dtype=torch.long))


def test_ee_engine_equals_master_bitwise():
  assert [G1_TRACKED.index(n) for n in G1_EE] == EE_IDX, "G1: env term order == tracked order"
  e2_idx = [i for i, n in enumerate(E2_TRACKED) if n in E2_EE]
  assert [E2_TRACKED.index(n) for n in E2_EE] == e2_idx == [3, 6, 9, 11], "E2: env term order == tracked order"
  for tracked, ee_names in ((G1_TRACKED, G1_EE), (E2_TRACKED, E2_EE)):
    nb = len(tracked)
    for on, thr, temp in ((True, 0.25, 0.02), (True, 0.1, 0.05), (False, 0.25, 0.02)):
      heads = _heads(HeadsCfg(ee_termination=on, ee_term_z=thr, ee_term_z_temp=temp), tracked, ee_names)
      m = _master_self(heads, tracked, ee_names, thr if on else 0.0, temp)
      for seed in range(3):
        ref_bp, ref_bq, bp0, bq0 = random_batch(2048, seed)
        ref_bp, ref_bq, bp0, bq0 = ref_bp[:, :nb], ref_bq[:, :nb], bp0[:, :nb], bq0[:, :nb]
        outs = []
        for which in ("ours", "master"):
          bp, bq = bp0.clone().requires_grad_(True), bq0.clone().requires_grad_(True)
          k = dict(body_pos=bp, body_quat=bq, anchor_pos=bp[:, heads.anchor], anchor_quat=bq[:, heads.anchor])
          if which == "ours":
            hard, soft = heads.termination(k, ref_bp[:, heads.anchor], ref_bq[:, heads.anchor], ref_bp)
          else:
            hard, soft = termination_master(m, k, ref_bp, ref_bq)
          gp, gq = torch.autograd.grad((soft * torch.linspace(0.5, 1.5, soft.shape[0])).sum(), (bp, bq))
          outs.append((hard, soft.detach(), gp, gq))
        if on:
          assert 0 < int(outs[0][0].sum()) < 2048
        for a, b in zip(*outs):
          assert torch.equal(a, b), f"{len(tracked)} bodies, on={on} thr={thr}: differs from master"


def test_heads_cfg_from_any_checkpoint():
  ours = dataclasses.asdict(HeadsCfg())
  c = HeadsCfg.from_dict(ours)
  assert (c.ee_termination, c.ee_term_z, c.ee_term_z_temp) == (True, 0.25, 0.02) and c == HeadsCfg()
  c = HeadsCfg.from_dict(master_heads_dict(ee_term_z=0.25, ee_term_temp=0.03))  # bj27 g1v2_scratch style
  assert (c.ee_termination, c.ee_term_z, c.ee_term_z_temp) == (True, 0.25, 0.03)
  c = HeadsCfg.from_dict(master_heads_dict(ee_term_z=0.0))  # master v2 code, run without --ee_term_z
  assert (c.ee_termination, c.ee_term_z) == (False, 0.25)
  c = HeadsCfg.from_dict(v1_heads_dict())  # before both switches: paper recipe
  assert (c.ee_termination, c.ee_term_z, c.ee_term_z_temp) == (False, 0.25, 0.02)
  c = HeadsCfg.from_dict({**ours, "ee_termination": False})
  assert (c.ee_termination, c.ee_term_z) == (False, 0.25)
  for bad in ({**ours, "ee_term_z": 0.0}, {**ours, "no_such_key": 1}):
    try:
      HeadsCfg.from_dict(bad)
    except (ValueError, TypeError):
      continue
    raise AssertionError(f"must fail: {bad}")
  try:
    HeadsCfg(ee_termination=True, ee_term_z=0.0)
  except ValueError:
    pass
  else:
    raise AssertionError("ee_termination with threshold 0 must fail")


def test_ee_flags():
  ok = {
    (None, None, None): (True, 0.25),  # no flag: on at the eval site's 0.25 (master: off!)
    (0.25, None, None): (True, 0.25),  # g1v2_scratch
    (0.3, None, None): (True, 0.3),
    (0.0, None, None): (False, 0.25),  # master's paper recipe
    (None, False, None): (False, 0.25),
    (None, True, 0.3): (True, 0.3),
    (None, None, 0.3): (True, 0.3),
    (0.25, True, 0.25): (True, 0.25),
    (0.0, False, 0.4): (False, 0.4),
  }
  for args, want in ok.items():
    assert ee_flags(*args) == want, (args, ee_flags(*args), want)
  assert ee_flags(None, None, None, default=(False, 0.25)) == (False, 0.25)  # eval_ckpt: the checkpoint's setting
  for args in ((0.25, False, None), (0.0, True, None), (0.25, None, 0.3), (-0.1, None, None), (None, True, 0.0)):
    try:
      ee_flags(*args)
    except ValueError:
      continue
    raise AssertionError(f"must fail: {args}")


# ---------------------------------------------------------------------------
# (b) reset sampler: source / name prior (+ clamp)
# ---------------------------------------------------------------------------

SOURCES = ("noetix_kungfu", "noetix_kungfu_dance_show", "sportsmeeting", "lafan1", "g1_all", "mixamo_cn", "virtual_point",
           "sonic", "motiondecode", "hiphi")


def synthetic_library(n_clips: int = 3000, seed: int = 0):
  g = torch.Generator().manual_seed(seed)
  names = []
  for i in range(n_clips):
    src = SOURCES[int(torch.randint(0, len(SOURCES), (1,), generator=g))]
    tag = ("_Crawl" if i % 17 == 0 else "") + ("_getup" if i % 23 == 0 else "")
    names.append(f"g1__{src}_{i:05d}{tag}.npz")
  bins_per_clip = torch.randint(1, 12, (n_clips,), generator=g)
  bin_clip = torch.repeat_interleave(torch.arange(n_clips), bins_per_clip)
  return names, bin_clip


def _weights(s: str):
  return tuple((kv.split("=")[0], float(kv.split("=")[1])) for kv in s.split(",") if kv)


def test_prior_equals_master_bitwise():
  names, bin_clip = synthetic_library()
  logs = []
  for sw, nw in ((_weights(G1V2_SOURCES), ()), ((), _weights("crawl=4,getup=4")), (_weights(G1V2_SOURCES), _weights("crawl=40")),
                 (_weights("noetix_kungfu=0.5"), ())):
    ours = reset_prior(names, bin_clip, sw, nw, "cpu", log=logs.append)
    assert torch.equal(ours, prior_master(names, bin_clip, sw, nw)), (sw, nw)
    assert abs(float(ours.sum()) - 1.0) < 1e-5 and ours.dtype == torch.float32
  assert reset_prior(names, bin_clip, (), (), "cpu") is None
  assert any("noetix_kungfu_dance_show" in m for m in logs), "per-source share report"
  # longest prefix wins: dance_show clips get 15, plain kungfu clips 10
  p = reset_prior(names, bin_clip, _weights(G1V2_SOURCES), (), "cpu", log=lambda *a: None)
  w = {n: float(p[bin_clip == i][0]) for i, n in enumerate(names)}
  base = min(w.values())
  for n, v in w.items():
    want = 15.0 if "_noetix_kungfu_dance_show_" in n else 10.0 if "_noetix_kungfu_" in n else 1.0 if "_sonic_" in n else None
    if want is not None:
      assert abs(v / base - want) < 1e-4, (n, v / base)


def test_prior_and_clamp_compose():
  names, bin_clip = synthetic_library(800)
  NB = bin_clip.numel()
  prior = reset_prior(names, bin_clip, _weights(G1V2_SOURCES), _weights("crawl=4"), "cpu", log=lambda *a: None)
  g = torch.Generator().manual_seed(3)
  G = torch.randn(NB, generator=g) * 20.0 + 5.0  # far outside [0, g_max] both ways
  V = (torch.rand(NB, generator=g) < 0.7).float() * 50.0
  c = tsc.DEFAULTS
  for clamp in (False, True):
    for pr in (None, prior):
      cmd = tsc.make_cmd(NB, clamp, G, V, prior=pr)
      hist0 = cmd.bin_weight_hist.clone()
      got = cmd.sampling_probabilities()
      Gs = G.clamp(0.0, c["g_max"]) if clamp else G
      Gs = torch.where(V > 0, Gs, torch.full_like(G, c["g_max"] * 0.5))
      soft = torch.softmax(-c["softmax_temp"] * Gs / c["g_max"], dim=0)
      uni = pr if pr is not None else torch.full((NB,), 1.0 / NB)
      want = c["mix_uniform"] * uni + c["mix_softmax"] * soft + c["mix_history"] * (hist0 / hist0.sum())
      want = torch.nan_to_num(want, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0) + 1e-12
      want = want / want.sum()
      assert torch.equal(got, want), (clamp, pr is None)
  # uniform share only: probabilities follow the prior exactly (clamp irrelevant)
  only_u = tsc.make_cmd(NB, True, G, V, prior=prior, mix_uniform=1.0, mix_softmax=0.0, mix_history=0.0).sampling_probabilities()
  assert torch.allclose(only_u, prior, rtol=1e-5, atol=1e-11)


def _ddp_worker(rank: int, world: int, port: int, tmp: str):
  os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world))
  torch.set_num_threads(1)
  ctx = DistCtx.from_env("cpu", timeout_min=2.0)
  # rank 0 resolves the clip list (scripts/train.py), everyone builds the library + prior from it
  names = synthetic_library(1500, seed=5)[0] if ctx.is_main else None
  names = ctx.broadcast_object(names)
  bin_clip = torch.repeat_interleave(torch.arange(len(names)), torch.tensor([1 + (len(n) * 7 + i) % 9 for i, n in enumerate(names)]))
  NB = bin_clip.numel()
  t = tsc.make_trainer(tmp, NB, True, dist=ctx)
  t.cmd.bin_prior = reset_prior(names, bin_clip, _weights(G1V2_SOURCES), _weights("crawl=4,getup=4"), "cpu", log=lambda *a: None)
  Trainer._dist_start(t, "start")  # broadcast sampler + full learned-state check, incl. the prior
  for rnd in range(3):
    g = torch.Generator().manual_seed(1000 * rnd + rank)
    bins = torch.randint(0, NB, (5000,), generator=g)
    rets = torch.randn(bins.numel(), generator=g) * 20.0 + 5.0
    t.cmd.update_bin_returns(bins, rets)
    t.cmd.sampling_probabilities()  # per-rank hist drift, as sample_frames does between merges
    Trainer._merge_sampler(t)
    p = t.cmd.sampling_probabilities()
    ctx.check_consistent({"probs": [p], "sampler": [t.cmd.bin_G, t.cmd.bin_visits, t.cmd.bin_weight_hist]}, tag=f"round {rnd}")
  # a rank with another prior (e.g. a different clip list) is caught by the state check
  if rank == world - 1:
    t.cmd.bin_prior = reset_prior(names, bin_clip, _weights("lafan1=3"), (), "cpu", log=lambda *a: None)
  try:
    Trainer._dist_start(t, "bad prior")
  except RuntimeError as e:
    assert "sampler_prior" in str(e), str(e)
  else:
    raise AssertionError("a differing prior must be caught")
  ctx.barrier()
  ctx.close()
  with open(os.path.join(tmp, f"rank{rank}.ok"), "w") as f:
    f.write("ok")


def _run_ddp(world: int):
  with tempfile.TemporaryDirectory() as tmp:
    mp.spawn(_ddp_worker, args=(world, tsc._free_port(), tmp), nprocs=world, join=True)
    for r in range(world):
      assert os.path.exists(os.path.join(tmp, f"rank{r}.ok")), f"rank {r} did not finish"


def test_ddp_prior_world2():
  _run_ddp(2)


def test_ddp_prior_world3():
  _run_ddp(3)


# ---------------------------------------------------------------------------
# (c) wiring
# ---------------------------------------------------------------------------


def _load_script(name: str):
  spec = importlib.util.spec_from_file_location(f"_{name}_script", os.path.join(ROOT, "scripts", f"{name}.py"))
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


def test_train_cli_v2_flags():
  script = _load_script("train")
  with tempfile.TemporaryDirectory() as tmp:
    clips = []
    for i, src in enumerate(("noetix_kungfu", "lafan1", "sonic")):
      clips.append(os.path.join(tmp, f"g1__{src}_{i}.npz"))
      open(clips[-1], "w").close()
    lst = os.path.join(tmp, "motion_files.txt")
    with open(lst, "w") as f:
      f.write("\n".join(clips) + "\n\n")
    seen: list[TrainCfg] = []

    def fake_trainer(cfg, dist=None):
      seen.append(cfg)
      raise tsc._Probe

    def run(extra, files=("--motion_list", lst)):
      argv = ["train.py", *files, "--log_root", tmp, "--device", "cpu", "--run_name", "v2"] + list(extra)
      with mock.patch.object(sys, "argv", argv):
        try:
          script.main()
        except tsc._Probe:
          return seen[-1]
      raise AssertionError("Trainer not reached")

    with mock.patch.object(trainer_mod, "Trainer", fake_trainer), mock.patch.object(motions_mod, "collect_motion_files", lambda *a, **k: ["/x/a.npz"]):
      # the bj27 g1v2_scratch command (paths aside)
      g1v2 = ["--ee_term_z", "0.25", "--eval_clips", "1024", "--no_eval", "--num_workers", "64", "--source_weights", G1V2_SOURCES,
              "--robot", "g1", "--robot_variant", "eval+evalcontact"]
      c = run(g1v2)
      assert c.motion_files == clips and c.robot == "g1" and c.robot_variant == "eval+evalcontact"
      assert c.source_weights == _weights(G1V2_SOURCES) and c.name_weights == ()
      assert (c.heads.ee_termination, c.heads.ee_term_z, c.heads.ee_term_z_temp) == (True, 0.25, 0.02)
      assert c.wd_drop == 0.2 and c.policy_lr == c.critic_lr == c.wm.lr == 2e-4 and c.lr_decay_iters == 250_000 and c.H == 32
      assert c.eval_enabled is False and c.eval_clips == 1024 and c.num_workers == 64 and c.sampler_clamp_returns is False
      c = run(g1v2 + ["--sampler_clamp_returns", "--name_weights", "crawl=4,getup=4"])
      assert c.sampler_clamp_returns is True and c.name_weights == (("crawl", 4.0), ("getup", 4.0))
      for extra, want in (([], (True, 0.25)), (["--ee_term_z", "0"], (False, 0.25)), (["--no_ee_termination"], (False, 0.25)),
                          (["--ee_threshold", "0.3"], (True, 0.3)), (["--ee_termination", "--ee_term_z", "0.25"], (True, 0.25))):
        c = run(extra, files=("--motion_dirs", "/x"))
        assert (c.heads.ee_termination, c.heads.ee_term_z) == want, (extra, c.heads)
        assert c.motion_files == ["/x/a.npz"]
      n_ok = len(seen)
      for extra, files in (
        (["--ee_term_z", "0.25", "--no_ee_termination"], ("--motion_dirs", "/x")),
        (["--ee_term_z", "0.25", "--ee_threshold", "0.3"], ("--motion_dirs", "/x")),
        ([], ("--motion_list", lst, "--motion_dirs", "/x")),
        (["--blacklist", "b.json"], ("--motion_list", lst)),
        ([], ()),
      ):
        argv = ["train.py", *files, "--log_root", tmp, "--device", "cpu"] + extra
        with mock.patch.object(sys, "argv", argv), mock.patch("sys.stderr", new=open(os.devnull, "w")):
          try:
            script.main()
          except SystemExit as e:
            assert e.code == 2, (extra, files)
            continue
        raise AssertionError(f"must be rejected: {files} {extra}")
      assert len(seen) == n_ok
      # a list written on another machine
      other = os.path.join(tmp, "other.txt")
      with open(other, "w") as f:
        f.write("/nonexistent/g1__lafan1_0.npz\n")
      try:
        run([], files=("--motion_list", other))
      except FileNotFoundError as e:
        assert "another machine" in str(e)
      else:
        raise AssertionError("a list of missing files must fail")
      # partly missing: fails too, before the Trainer (build_cache would die on rank 0 only; nothing is dropped)
      partial = os.path.join(tmp, "partial.txt")
      with open(partial, "w") as f:
        f.write("\n".join(clips[:2] + ["/nonexistent/g1__lafan1_9.npz"] + clips[2:]) + "\n")
      n_before = len(seen)
      try:
        run([], files=("--motion_list", partial))
      except FileNotFoundError as e:
        assert "1 of 4 files" in str(e) and "/nonexistent/g1__lafan1_9.npz" in str(e) and "verbatim" in str(e), str(e)
      else:
        raise AssertionError("a partly missing list must fail")
      assert len(seen) == n_before, "Trainer must not be reached"


def test_trainer_and_env_cfg_wiring():
  calls: list[dict] = []

  def fake_make_env_cfg(*a, **k):
    calls.append(k)
    raise tsc._Probe

  with tempfile.TemporaryDirectory() as tmp:
    cfg = TrainCfg(log_dir=tmp, device="cpu", robot_variant="eval+evalcontact", source_weights=_weights(G1V2_SOURCES),
                   name_weights=(("crawl", 4.0),), sampler_clamp_returns=True, heads=HeadsCfg(ee_term_z=0.3))
    with mock.patch.object(trainer_mod, "make_env_cfg", fake_make_env_cfg):
      try:
        Trainer(cfg)
      except tsc._Probe:
        pass
  (k,) = calls
  assert k["robot_variant"] == "eval+evalcontact" and k["source_weights"] == _weights(G1V2_SOURCES)
  assert k["name_weights"] == (("crawl", 4.0),) and k["clamp_returns"] is True
  assert k["ee_termination"] is True and k["ee_threshold"] == 0.3 and "ee_term_z" not in k
  # make_env_cfg -> base_tracking_cfg(robot, variant), command cfg, ee_body_pos
  variants = []

  def fake_base(robot, variant=""):
    variants.append((robot, variant))
    return tsc._fake_base_cfg(robot)

  with mock.patch.object(env_cfg_mod, "base_tracking_cfg", fake_base):
    for mode in ("train", "eval"):
      e = env_cfg_mod.make_env_cfg(["a.npz"], 4, mode=mode, robot="g1", robot_variant="eval+evalcontact",
                                   source_weights=_weights(G1V2_SOURCES), name_weights=[["crawl", 4]], ee_threshold=0.25)
      m = e.commands["motion"]
      assert isinstance(m, LibraryMotionCommandCfg)
      if mode == "train":
        assert m.source_weights == _weights(G1V2_SOURCES) and m.name_weights == (("crawl", 4),)
      else:
        assert m.source_weights == () and m.name_weights == ()  # benchmark: clips in order, no reset sampler
      assert e.terminations["ee_body_pos"].params["threshold"] == 0.25
    e = env_cfg_mod.make_env_cfg(["a.npz"], 4, ee_termination=False)
    assert "ee_body_pos" not in e.terminations and e.commands["motion"].source_weights == ()
  assert variants == [("g1", "eval+evalcontact"), ("g1", "eval+evalcontact"), ("g1", "")]
  for robot, bad in (("g1", "nodamp"), ("e2", "eval"), ("g1", "eval+foo")):
    try:
      env_cfg_mod.base_tracking_cfg(robot, bad)  # validated before anything is imported
    except ValueError:
      continue
    raise AssertionError(f"{robot} variant {bad!r} must be rejected")


def test_eval_ckpt_any_checkpoint():
  script = _load_script("eval_ckpt")
  import supermimic.evaluate as evaluate_mod
  import supermimic.policy as policy_mod

  class _Policy:
    def __init__(self, *a):
      pass

    def to(self, *a):
      return self

    def load_state_dict(self, sd):
      pass

  got = []

  class _Evaluator:
    def __init__(self, files, heads_cfg, device, **kw):
      got.append((heads_cfg, kw["robot_variant"]))
      raise tsc._Probe

  dims = dict(proprio=3 * NJ + 6, hist=10, goal_frame=2 * NJ + 6, K=10, nj=NJ, xi=4 + NJ)
  with tempfile.TemporaryDirectory() as tmp, mock.patch.object(policy_mod, "Policy", _Policy), \
      mock.patch.object(evaluate_mod, "Evaluator", _Evaluator):
    def run(heads: dict, extra=(), variant="eval+evalcontact"):
      path = os.path.join(tmp, "ck.pt")
      torch.save({"it": 86000, "policy": {}, "dims": dims, "cfg": {"heads": heads, "motion_files": ["/x/a.npz"], "robot": "g1",
                  "robot_variant": variant, "eval_clips": 0, "seed": 0}}, path)
      with mock.patch.object(sys, "argv", ["eval_ckpt.py", "--ckpt", path, "--device", "cpu", *extra]):
        try:
          script.main()
        except tsc._Probe:
          h, v = got[-1]
          return h.ee_termination, h.ee_term_z, h.ee_term_z_temp, v
      raise AssertionError("Evaluator not reached")

    master = master_heads_dict(ee_term_z=0.25)
    assert run(master) == (True, 0.25, 0.02, "eval+evalcontact")  # bj27 g1v2_scratch model_86000
    assert run(master, ["--ee_term_z", "0"]) == (False, 0.25, 0.02, "eval+evalcontact")
    assert run(master, ["--robot_variant", ""]) == (True, 0.25, 0.02, "")
    assert run(master_heads_dict(ee_term_z=0.0), variant="") == (False, 0.25, 0.02, "")
    assert run(v1_heads_dict(), variant="") == (False, 0.25, 0.02, "")  # v1: as trained = off
    assert run(v1_heads_dict(), ["--ee_termination", "on"], variant="") == (True, 0.25, 0.02, "")
    assert run(v1_heads_dict(), ["--ee_term_z", "0.3"], variant="") == (True, 0.3, 0.02, "")
    assert run(dataclasses.asdict(HeadsCfg()), variant="") == (True, 0.25, 0.02, "")  # ddp-bj31 r52
    assert run(dataclasses.asdict(HeadsCfg()), ["--ee_threshold", "0.2"], variant="") == (True, 0.2, 0.02, "")


def test_resume_guard():
  """--resume from another setup (real Trainer.load): every differing training setting is reported, for master v2 / v1 /
  ddp-bj31 checkpoint cfgs; the reset sampler's state is loaded only for the same clip list (by basename: bj27 vs bj31
  roots load), with a warning otherwise; a watchdog rollback (own checkpoint) checks nothing."""
  NB = 64
  names = [f"g1__lafan1_{i}.npz" for i in range(4)]
  G = torch.arange(NB, dtype=torch.float32) / NB
  with tempfile.TemporaryDirectory() as tmp:
    def ckpt(heads: dict, files=None, nb=NB, **over) -> str:
      t = tsc.make_trainer(tmp, nb, False)
      cfg = dataclasses.asdict(t.cfg)
      cfg.update(motion_files=files if files is not None else [f"/bj27/{n}" for n in names], heads=heads, **over)
      path = os.path.join(tmp, f"ck{len(os.listdir(tmp))}.pt")
      torch.save({"it": 86000, "env_steps": 1, "policy": t.policy.state_dict(), "critic": t.critic.state_dict(), "wm": t.wm.state_dict(),
                  "opt_wm": t.opt_wm.state_dict(), "opt_pi": t.opt_pi.state_dict(), "opt_v": t.opt_v.state_dict(),
                  "bin_G": G[:nb] if nb <= NB else torch.zeros(nb), "bin_visits": torch.ones(nb), "lr_scale": 1.0, "cfg": cfg}, path)
      return path

    def load(path, rollback=False, **over):  # this run: bj31 roots
      t = tsc.make_trainer(tmp, NB, False)
      t.cfg = dataclasses.replace(t.cfg, motion_files=[f"/bj31/{n}" for n in names], **over)
      Trainer.load(t, path, rollback=rollback)
      return t, [s for s in t.prints if "WARNING" in s]

    v2 = dict(robot_variant="eval+evalcontact", source_weights=_weights(G1V2_SOURCES))
    g1v2 = ckpt(master_heads_dict(ee_term_z=0.25), **v2)  # bj27 g1v2_scratch model_86000, master format
    t, w = load(g1v2, **v2)  # same recipe, other roots: silent, sampler state loaded
    assert w == [] and torch.equal(t.cmd.bin_G, G) and t.it == 86000, w
    t, w = load(g1v2)  # v2 flags forgotten
    assert len(w) == 2 and "robot_variant = 'eval+evalcontact' in the checkpoint, '' here" in w[0] and "source_weights" in w[1], w
    assert torch.equal(t.cmd.bin_G, G)
    print("  " + "\n  ".join(w))
    _, w = load(ckpt(master_heads_dict(ee_term_z=0.0), **v2), **v2)  # master default: EE off -> this branch's default on
    assert w == ["[Trainer] WARNING resume from another setup: heads.ee_termination = False in the checkpoint, True here"], w
    _, w = load(ckpt(v1_heads_dict()), **v2)  # v1: no EE keys = off
    assert len(w) == 3 and "heads.ee_termination = False" in w[-1], w
    _, w = load(ckpt(v1_heads_dict()), heads=HeadsCfg(ee_termination=False, ee_term_z=0.3))  # off on both: threshold moot
    assert w == [], w
    _, w = load(ckpt(dataclasses.asdict(HeadsCfg())), heads=HeadsCfg(ee_term_z=0.3), sampler_clamp_returns=True)
    assert len(w) == 2 and "sampler_clamp_returns = False" in w[0] and "heads.ee_term_z = 0.25 in the checkpoint, 0.3 here" in w[1], w
    # other clip list: sampler state never lands on other clips
    for files, nb, why in (([f"/bj27/x{n}" for n in names], NB, "same bin count, other clip list"),
                           ([f"/bj27/{n}" for n in names[::-1]], NB, "same bin count, other clip list"),
                           ([f"/bj27/{n}" for n in names[:3]], NB, "same bin count, other clip list"),
                           (None, 100, "100 bins in the checkpoint, 64 here")):
      t, w = load(ckpt(dataclasses.asdict(HeadsCfg()), files=files, nb=nb))
      assert float(t.cmd.bin_G.abs().sum()) == 0.0 and why in w[-1] and "not loaded" in w[-1], (files, w)
      assert len(w) == (1 if files is None else 2), w  # + the motion-files line
    print("  " + "\n  ".join(w))
    # rollback: own checkpoint, settings / clip list not compared
    t, w = load(g1v2, rollback=True)
    assert w == [] and torch.equal(t.cmd.bin_G, G)


if __name__ == "__main__":
  if tsc.STUBBED:
    print(f"  (stubbed via sys.modules: {', '.join(tsc.STUBBED)})")
  for name, fn in list(globals().items()):
    if name.startswith("test_") and callable(fn):
      fn()
      print(f"ok  {name}")
