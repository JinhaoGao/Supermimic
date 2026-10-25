"""Root-orientation termination (--root_ori_term / --root_ori_term_temp, opt-in) and the always-on failure watch
(train/would_fail_ori, train/watch_floor_* / watch_spin_*, eval/failed_ori); CPU, torch + mujoco (for a real TorchFK).

  python tests/test_root_ori_term.py      # or: pytest tests/test_root_ori_term.py

10-25 diagnosis: 101 of 129 eval-site failures of r15ws model_148000 are root_ori -- the pelvis' full rotation angle
(yaw included) against the reference > 1.0 rad (evaluate_mujoco.py quat_error_angle) -- and no training termination
sees it (torso gravity tilt, torso / end-effector heights). mjlab / tensorboard are stubbed when missing (as in
test_sampler_clamp.py). Checks:
  (a) switch off: heads.termination (hard, soft and their gradients; eager and torch.compile'd; curriculum scale None /
      tensor; with or without ref_rq) and StepFns._post / _imagine_step (all outputs + policy gradients, real TorchFK on a
      small MJCF, small world model / policy) bitwise equal to 55e0eaf (copied verbatim below);
  (b) switch on: the hard test flips exactly at A (random axes, pure world yaw, pure body yaw, both directions, scaled
      thresholds), the soft term is sigmoid((angle - A) / T) noisy-OR'ed with the anchor / end-effector terms, 0.5 at A;
      gradients = closed form, descend toward the reference, finite and 0 at angle 0, peak at A; the real FK's root
      quaternion is the state's s[3:7] bitwise; pelvis yaw through the waist (torso on the reference): the old terms see
      nothing, the new one fires; the env term (library_command.bad_root_ori_full) == the heads' hard test bitwise (curriculum
      factor included); quat_angle == the eval site's quat_error_angle (copied verbatim below);
  (c) Trainer.collect_step on a scripted simulator: off == 55e0eaf bitwise (replay, sampler returns, stats, normalizers,
      RNG, history) while the watch counts equal the closed form; on: env term and heads agree (term_mismatch 0) and the
      episodes end at the first pelvis angle > A; Evaluator.run: off == 55e0eaf + failed_ori / failed_with_ori closed form;
  (d) the watch under data parallel (3 gloo ranks, real collect_step + _dist_log_scalars): the same scalars on every rank
      = the pooled closed form; the single-process loop (real Trainer.train) logs them and clears them per log window;
  (e) wiring: CLI -> TrainCfg -> train_cfg.json / make_env_cfg kwargs -> env term (train / eval, curriculum, robots);
      SimInterface consistency checks; resume diffs; scripts/launch_bj31_ddp.sh (default command == 55e0eaf's, PORT /
      CUDA_VISIBLE_DEVICES / errors) and scripts/launch_bj31_oriterm.example.sh (ARGS = the live run's except --iterations /
      --resume, restored LR = the live run's at 148000, 4 ranks per arm).
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import io
import json
import math
import os
import re
import shlex
import subprocess
import sys
import tempfile
import types
from unittest import mock

import numpy as np
import torch
import torch.multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import test_sampler_clamp as tsc  # noqa: E402  (installs the mjlab / tensorboard stubs when needed)
import test_early_stop_alive as tes  # noqa: E402
import test_ee_termination as tee  # noqa: E402
import test_merge_master_1025 as tmm  # noqa: E402

import mujoco  # noqa: E402

import supermimic.env_cfg as env_cfg_mod  # noqa: E402
import supermimic.trainer as trainer_mod  # noqa: E402
from supermimic.distributed import DistCtx  # noqa: E402
from supermimic.evaluate import Evaluator  # noqa: E402
from supermimic.fk import TorchFK  # noqa: E402
from supermimic.fns import StepFns  # noqa: E402
from supermimic.heads import AnalyticHeads, HeadsCfg, window_frames  # noqa: E402
from supermimic.library_command import LibraryMotionCommand, bad_root_ori_full  # noqa: E402
from supermimic.math_utils import quat_angle, quat_apply_inverse, quat_from_angle_axis, quat_mul  # noqa: E402
from supermimic.motions import MotionLibrary  # noqa: E402
from supermimic.policy import Policy  # noqa: E402
from supermimic.replay import ReplayBuffer  # noqa: E402
from supermimic.sim_interface import ObsHistory, SimInterface  # noqa: E402
from supermimic.trainer import TrainCfg, Trainer  # noqa: E402
from supermimic.watch import ORI_FAIL_THRESHOLD, WATCH_GROUPS, FailureWatch  # noqa: E402
from supermimic.world_model import DynamicsModel, WorldModelCfg  # noqa: E402

THIS = sys.modules[__name__]
ANCHOR, BT, EE_IDX, NJ = tee.ANCHOR, tee.BT, tee.EE_IDX, tee.NJ


# ---------------------------------------------------------------------------
# 55e0eaf, verbatim
# ---------------------------------------------------------------------------


def termination_55e0eaf(self, kin_next: dict, ref_ap: torch.Tensor, ref_aq: torch.Tensor, ref_bp: torch.Tensor | None = None,
                        scale: torch.Tensor | None = None):
  """heads.AnalyticHeads.termination at 55e0eaf (before the root-orientation term), verbatim (docstring dropped)."""
  c = self.cfg
  B = ref_ap.shape[0]
  g = self.gravity.expand(B, 3)
  k = 1.0 if scale is None else scale  # [B]
  dz = (ref_ap[:, 2] - kin_next["anchor_pos"][:, 2]).abs()
  tilt = (quat_apply_inverse(ref_aq, g)[:, 2] - quat_apply_inverse(kin_next["anchor_quat"], g)[:, 2]).abs()
  hard = (dz > c.term_z * k) | (tilt > c.term_tilt * k)
  pz = torch.sigmoid((dz - c.term_z * k) / (c.term_z_temp * k))
  pt = torch.sigmoid((tilt - c.term_tilt * k) / (c.term_tilt_temp * k))
  alive = (1.0 - pz) * (1.0 - pt)
  if c.ee_termination:
    assert ref_bp is not None, "ee_termination needs the reference body positions"
    ke = 1.0 if scale is None else scale[:, None]  # [B, 1] against [B, n_ee]
    dz_ee = (ref_bp[:, self.ee, 2] - kin_next["body_pos"][:, self.ee, 2]).abs()  # [B, n_ee]
    hard = hard | (dz_ee > c.ee_term_z * ke).any(-1)
    p_ee = torch.sigmoid((dz_ee - c.ee_term_z * ke) / (c.ee_term_z_temp * ke))
    alive = alive * torch.prod(1.0 - p_ee, dim=-1)
  soft = 1.0 - alive
  return hard, soft


def post_55e0eaf(self, next_state, kin, a_exec, prev_action, ref_bp, ref_bq, ref_blv, ref_bav, term_scale=None):
  """fns.StepFns._post at 55e0eaf, verbatim."""
  kin_next = self.heads.kin(next_state)
  ref = dict(body_pos=ref_bp, body_quat=ref_bq, body_lin_vel=ref_blv, body_ang_vel=ref_bav)
  r = self.heads.reward(kin, kin_next, next_state, a_exec, prev_action, ref)
  a_idx = self.heads.anchor
  hard, _ = self.heads.termination(kin_next, ref_bp[:, a_idx], ref_bq[:, a_idx], ref_bp, term_scale)
  return r, hard


def imagine_step_55e0eaf(self, s, obs_hist, prev_a, f_loc, clip_end_loc, xi, bias, kin, p_q, p_qd, p_bp, p_bq, p_blv, p_bav, term_scale=None):
  """fns.StepFns._imagine_step at 55e0eaf, verbatim."""
  heads = self.heads
  N = s.shape[0]
  ar = torch.arange(N, device=s.device)
  goal = self.goal_from_pack(kin, f_loc, clip_end_loc, p_q, p_qd, p_bp, p_bq)
  fn, gn = self.critic_inputs(s, goal)
  a = self.policy(obs_hist, goal)
  s_next = self.wm.step(s, a, xi)
  kin_next = heads.kin(s_next)
  ref = dict(body_pos=p_bp[ar, f_loc], body_quat=p_bq[ar, f_loc], body_lin_vel=p_blv[ar, f_loc], body_ang_vel=p_bav[ar, f_loc])
  r = heads.reward(kin, kin_next, s_next, a, prev_a, ref)
  a_idx = heads.anchor
  _, d = heads.termination(kin_next, ref["body_pos"][:, a_idx], ref["body_quat"][:, a_idx], ref["body_pos"], term_scale)
  f_next = torch.minimum(f_loc + 1, clip_end_loc - 1)
  obs_next = heads.proprio(s_next, a, bias, True)
  obs_hist_next = torch.cat([obs_hist[:, 1:], obs_next[:, None]], dim=1)
  return s_next, obs_hist_next, a, f_next, kin_next, r, d, fn, gn


@torch.no_grad()
def collect_step_55e0eaf(self) -> None:
  """trainer.Trainer.collect_step at 55e0eaf, verbatim."""
  cfg = self.cfg
  ref_q, ref_qd, ref_ap, ref_aq = self.sim.goal_refs(self.frame, self.clip_end)
  ref_cur = self.sim.current_ref()  # reference at frame f_t: reward target of this step
  goal, a = self.fns.act(self.hist.buf, self.kin, ref_q, ref_qd, ref_ap, ref_aq)
  self.dist.rms_update([(self.policy.obs_norm, self.hist.buf[:, -1]), (self.policy.goal_norm, goal)])
  a_exec = a + cfg.action_noise_std * torch.randn_like(a)

  self.env.step(a_exec)
  next_state = self.sim.state()
  r, term_hard = self.fns.post(next_state, self.kin, a_exec, self.prev_action, *ref_cur, self.cmd.term_scale_env if self._hard_on else None)
  terminated = self.env.reset_terminated.bool()
  timeouts = self.env.reset_time_outs.bool()
  # A non-finite simulator state never trips the (NaN-blind) termination tests: end it as a failure and keep the
  # finite pre-step state in the replay instead (no-op, bitwise, when everything is finite).
  bad = ~torch.isfinite(next_state).all(-1)
  tm_frac, n_bad = torch.stack([(term_hard != terminated).float().mean(), bad.float().sum()]).tolist()  # one sync
  self.stats["term_mismatch"].append(tm_frac)
  self.stats["sim_nonfinite"].append(n_bad)
  terminated = terminated | bad
  r = torch.where(bad, torch.zeros_like(r), r)
  next_state = torch.where(bad[:, None], self.state, next_state)
  done = terminated | timeouts
  if self._ee_on:  # end-effector-only failures (the eval site's "body_pos" reason): logging only
    tm = self.env.termination_manager
    ee_fail = tm.get_term("ee_body_pos") & ~(tm.get_term("anchor_pos") | tm.get_term("anchor_ori")) & ~bad

  self.buffer.add(
    state=self.state, next_state=next_state, action=a_exec, prev_action=self.prev_action, xi=self.xi, reward=r,
    frame=self.frame, clip_end=self.clip_end, ep_step=self.ep_step, term=terminated, done=done,
  )
  self.ep_bins[self._ar, self.ep_step] = self.lib.bin_of_frame[self.frame].long()
  self.ep_rews[self._ar, self.ep_step] = r
  self.ep_ret += r
  self.ep_step += 1
  self.env_steps += 1

  if done.any():
    ids = done.nonzero(as_tuple=False).squeeze(-1)
    self._finish_episodes(ids, terminated[ids], ee_fail[ids] if self._ee_on else None)
    self.env.reset(env_ids=ids)
    xi_new = self.sim.xi()
    self.xi[ids] = xi_new[ids]
    self.state = self.sim.state()
    self.prev_action[ids] = 0.0
    self.ep_step[ids] = 0
    self.ep_ret[ids] = 0.0
  else:
    self.state = next_state
  self.prev_action = torch.where(done[:, None], torch.zeros_like(a_exec), a_exec)
  self.frame = self.cmd.time_steps.clone()
  self.clip_end = self.cmd.clip_end_env.clone()
  self.kin, obs = self.fns.refresh(self.state, self.prev_action, self.sim.bias_from_xi(self.xi, self.nj), True)
  self.hist.push(obs)
  if done.any():
    self.hist.reset(ids, obs)


@torch.no_grad()
def finish_episodes_55e0eaf(self, ids: torch.Tensor, failed: torch.Tensor, failed_ee: torch.Tensor | None = None) -> None:
  """trainer.Trainer._finish_episodes at 55e0eaf, verbatim."""
  g = self.cfg.gamma
  L = self.ep_step[ids]
  Lmax = int(L.max().item())
  rews = self.ep_rews[ids, :Lmax]
  bins = self.ep_bins[ids, :Lmax]
  t = torch.arange(Lmax, device=self.device)
  valid = t[None, :] < L[:, None]
  r_last = rews.gather(1, (L - 1)[:, None]).squeeze(1)
  boot = torch.where(failed, torch.zeros_like(r_last), r_last / (1.0 - g))
  w = rews * valid * self._gamma_pow[:Lmax][None, :]
  S = torch.flip(torch.cumsum(torch.flip(w, [1]), 1), [1])
  R = S / self._gamma_pow[:Lmax][None, :] + boot[:, None] * (g ** (L[:, None] - t[None, :]).float())
  self.cmd.update_bin_returns(bins[valid], R[valid])
  rmax = self.cfg.heads.reward_max_per_step
  norm_ret = self.ep_ret[ids] / (L.float() * rmax)
  self.stats["ep_return_norm"].extend(norm_ret.tolist())
  self.stats["ep_len"].extend(L.tolist())
  self.stats["ep_failed"].extend(failed.float().tolist())
  if failed_ee is not None:
    self.stats["ep_failed_ee"].extend((failed_ee & failed).float().tolist())
  if self._hard_on:  # hard-clip curriculum: episodes of the hard set (the clip they reset into)
    hard = self.cmd.env_hard[ids]
    if bool(hard.any()):
      self.stats["hard_ep_failed"].extend(failed[hard].float().tolist())
      self.stats["hard_ep_len"].extend(L[hard].tolist())


@torch.no_grad()
def eval_run_55e0eaf(self, policy, max_steps: int | None = None) -> dict:
  """evaluate.Evaluator.run at 55e0eaf, verbatim."""
  env, heads, sim, cmd, lib = self.env, self.heads, self.sim, self.cmd, self.lib
  fns = self._get_fns(policy)
  E = self.num_envs
  cmd._start_counter = 0
  env.reset()
  state = sim.state()
  xi = sim.xi()
  bias = sim.bias_from_xi(xi, sim.nj)
  frame = cmd.time_steps.clone()
  clip_end = cmd.clip_end_env.clone()
  prev_action = torch.zeros(E, sim.nj, device=self.device)
  kin, obs = fns.refresh(state, prev_action, bias, False)
  hist = ObsHistory(E, self.cfg.hist, heads.proprio_dim, self.device)
  hist.reset(torch.arange(E, device=self.device), obs)
  # Per-clip results, indexed by position in eval_clips.
  n = self.num_clips
  sum_r = torch.zeros(n, device=self.device)
  done_clip = torch.zeros(n, dtype=torch.bool, device=self.device)
  failed = torch.zeros(n, dtype=torch.bool, device=self.device)
  failed_ee = torch.zeros(n, dtype=torch.bool, device=self.device)  # first failure was end-effector only
  tm = env.termination_manager
  ee_on = "ee_body_pos" in tm.active_terms
  # Map clip index -> position in eval list.
  pos_of_clip = torch.full((lib.num_clips,), -1, dtype=torch.long, device=self.device)
  pos_of_clip[self.clip_ids] = torch.arange(n, device=self.device)
  env_pos = pos_of_clip[cmd.env_clip]  # current eval position of each env
  env_active = torch.ones(E, dtype=torch.bool, device=self.device)
  assigned = min(E, n)  # clips handed out so far (the first E at reset)
  clip_len = (lib.clip_end[self.clip_ids] - lib.clip_start[self.clip_ids]).float()
  limit = max_steps or int(clip_len.sum().item() / E * 2) + int(clip_len.max().item()) + 10
  policy.eval()
  for _ in range(limit):
    ref_q, ref_qd, ref_ap, ref_aq = sim.goal_refs(frame, clip_end)
    ref_cur = sim.current_ref()
    _, a = fns.act(hist.buf, kin, ref_q, ref_qd, ref_ap, ref_aq)
    env.step(a)
    next_state = sim.state()
    r, _ = fns.post(next_state, kin, a, prev_action, *ref_cur)
    terminated = env.reset_terminated.bool()
    timeouts = env.reset_time_outs.bool()
    act = env_active
    sum_r.index_add_(0, env_pos[act], r[act])
    fail_now = terminated & act
    if ee_on:
      ee_only = tm.get_term("ee_body_pos") & ~(tm.get_term("anchor_pos") | tm.get_term("anchor_ori"))
      failed_ee[env_pos[fail_now & ee_only]] = True
    failed[env_pos[fail_now]] = True
    done = terminated | timeouts
    fin = done & act
    done_clip[env_pos[fin]] = True
    if bool(done_clip.all()):
      break
    if done.any():
      ids = done.nonzero(as_tuple=False).squeeze(-1)
      env.reset(env_ids=ids)  # "start" mode hands out the next clips in sequence
      new_pos = pos_of_clip[cmd.env_clip[ids]]
      # Envs that received a clip beyond the list (wrap-around) become inactive.
      order = torch.arange(assigned, assigned + ids.numel(), device=self.device)
      env_active[ids] = order < n
      assigned += ids.numel()
      env_pos[ids] = new_pos
      state = sim.state()
      prev_action = torch.where(done[:, None], torch.zeros_like(a), a)
    else:
      state = next_state
      prev_action = a
    frame = cmd.time_steps.clone()
    clip_end = cmd.clip_end_env.clone()
    kin, obs = fns.refresh(state, prev_action, bias, False)
    hist.push(obs)
    if done.any():
      hist.reset(ids, obs)
  policy.train()
  rmax = self.cfg.reward_max_per_step
  norm_ret = sum_r / ((clip_len - 1.0) * rmax)
  names = [lib.clips[i].name for i in self.eval_clips]
  per_clip = {names[i]: float(norm_ret[i]) for i in range(n)}
  return {
    "num_clips": n,
    "scalars": {
      "return_mean": float(norm_ret.mean()),
      "return_min": float(norm_ret.min()),
      "failed": float(failed.sum()),
      "failed_frac": float(failed.float().mean()),
      "failed_ee": float(failed_ee.sum()),
      "failed_fall": float((failed & ~failed_ee).sum()),
      "clips_done": float(done_clip.sum()),
    },
    "per_clip": per_clip,
    "failed_clips": [names[i] for i in range(n) if bool(failed[i])][:200],
  }


# The eval site (bj31 /home/ubuntu/work/NOetix_GMT_eval/vendor/Noetix_GMT/scripts/evaluate_mujoco.py), verbatim.


def site_quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.clip(norm, 1.0e-8, None)


def site_quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    w1, x1, y1, z1 = np.moveaxis(q1, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(q2, -1, 0)
    out = np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )
    return out


def site_quat_inverse(q: np.ndarray) -> np.ndarray:
    q = site_quat_normalize(q)
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def site_quat_error_angle(q_ref: np.ndarray, q_actual: np.ndarray) -> np.ndarray:
    dq = site_quat_multiply(q_ref, site_quat_inverse(q_actual))
    w = np.clip(np.abs(dq[..., 0]), -1.0, 1.0)
    return 2.0 * np.arccos(w)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit(q: torch.Tensor) -> torch.Tensor:
  return q / q.norm(dim=-1, keepdim=True)


def yaw_q(angle: torch.Tensor) -> torch.Tensor:
  a = torch.as_tensor(angle, dtype=torch.float32)
  z = torch.zeros_like(a)
  return torch.stack([torch.cos(0.5 * a), z, z, torch.sin(0.5 * a)], dim=-1)


def g1_heads(ee: bool, ro: float, **kw) -> AnalyticHeads:
  """G1-sized heads on random tracked-body kinematics (test_ee_termination's layout); pelvis = tracked body 0."""
  cfg = HeadsCfg(ee_termination=ee, root_ori_term=ro, **kw)
  return AnalyticHeads(tee._FK(), list(range(BT)), ANCHOR, torch.zeros(NJ), torch.zeros(NJ, 2), cfg,
                       ee_indices=EE_IDX if ee else None, root_ori_index=0)


def old_heads(heads: AnalyticHeads) -> AnalyticHeads:
  """The same heads with 55e0eaf's termination."""
  h = copy.copy(heads)
  h.termination = types.MethodType(termination_55e0eaf, h)
  return h


# A small floating-base model for the real TorchFK: pelvis (free joint) -> leg; pelvis -> waist yaw -> torso -> arm.
MJCF = """
<mujoco>
  <worldbody>
    <body name="pelvis" pos="0 0 0.8">
      <freejoint name="root"/>
      <geom type="box" size="0.1 0.12 0.06" mass="5"/>
      <body name="left_hip_roll_link" pos="0 0.1 -0.08">
        <joint name="left_hip_roll" type="hinge" axis="1 0 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.04" mass="2"/>
        <body name="left_knee_link" pos="0 0 -0.32">
          <joint name="left_knee" type="hinge" axis="0 1 0"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.035" mass="1.5"/>
          <body name="left_ankle_roll_link" pos="0.02 0 -0.32">
            <joint name="left_ankle_roll" type="hinge" axis="1 0 0"/>
            <geom type="box" size="0.08 0.03 0.02" mass="0.5"/>
          </body>
        </body>
      </body>
      <body name="waist_yaw_link" pos="0 0 0.08">
        <joint name="waist_yaw" type="hinge" axis="0 0 1"/>
        <geom type="sphere" size="0.03" mass="0.5"/>
        <body name="torso_link" pos="0 0 0.05">
          <joint name="waist_pitch" type="hinge" axis="0 1 0"/>
          <geom type="box" size="0.08 0.12 0.15" pos="0 0 0.15" mass="8"/>
          <body name="left_wrist_yaw_link" pos="0.05 0.25 0.1">
            <joint name="left_shoulder" type="hinge" axis="0 1 0"/>
            <geom type="capsule" fromto="0 0 0 0 0 -0.4" size="0.03" mass="1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""
TRACKED = ("pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link", "torso_link", "left_wrist_yaw_link")
R_ANCHOR = TRACKED.index("torso_link")
R_EE = [TRACKED.index("left_ankle_roll_link"), TRACKED.index("left_wrist_yaw_link")]
WAIST_YAW = 3  # joint index (entity order: left_hip_roll, left_knee, left_ankle_roll, waist_yaw, waist_pitch, left_shoulder)


def real_fk():
  m = mujoco.MjModel.from_xml_string(MJCF)
  names = tuple(m.body(i).name for i in range(1, m.nbody))
  hinge = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
  ent = types.SimpleNamespace(indexing=types.SimpleNamespace(body_ids=torch.arange(1, m.nbody), joint_ids=torch.tensor(hinge)), body_names=names)
  tracked_local = [names.index(n) for n in TRACKED]
  return TorchFK(m, ent, "cpu", needed_bodies=tracked_local), names, tracked_local, len(hinge)


FK, BODY_NAMES, TRACKED_LOCAL, J = real_fk()
XI = 4 + J


def real_heads(cfg: HeadsCfg) -> AnalyticHeads:
  return AnalyticHeads(FK, TRACKED_LOCAL, R_ANCHOR, torch.zeros(J), torch.tensor([[-2.5, 2.5]] * J), cfg,
                       ee_indices=R_EE if cfg.ee_termination else None, root_ori_index=0, root_ori_is_root=True)


def small_policy(heads: AnalyticHeads, seed: int = 0) -> Policy:
  torch.manual_seed(seed)
  return Policy(heads.proprio_dim, heads.cfg.hist, heads.goal_frame_dim, heads.cfg.K, J, enc_hidden=(16,), dec_hidden=(16,))


def random_states(n: int, seed: int, yaw_spread: float = 3.0) -> torch.Tensor:
  g = torch.Generator().manual_seed(seed)
  p = torch.randn(n, 3, generator=g) * 0.1 + torch.tensor([0.0, 0.0, 0.8])
  tilt = quat_from_angle_axis(torch.rand(n, generator=g) * 0.3, _unit(torch.randn(n, 3, generator=g)))
  q = quat_mul(yaw_q((torch.rand(n, generator=g) * 2 - 1) * yaw_spread), tilt)
  v, w = torch.randn(n, 3, generator=g) * 0.3, torch.randn(n, 3, generator=g) * 0.5
  jq, jqd = torch.randn(n, J, generator=g) * 0.3, torch.randn(n, J, generator=g)
  return torch.cat([p, q, v, w, jq, jqd], -1)


# ---------------------------------------------------------------------------
# (a) switch off == 55e0eaf, bitwise
# ---------------------------------------------------------------------------


def test_off_termination_is_55e0eaf_bitwise():
  for ee in (False, True):
    heads = g1_heads(ee, 0.0)
    for seed, use_scale in ((0, False), (1, True), (2, False)):
      ref_bp, ref_bq, bp0, bq0 = tee.random_batch(4096, seed)
      scale = (0.5 + torch.rand(4096, generator=torch.Generator().manual_seed(seed))) if use_scale else None
      outs = []
      for fn in ("new", "new_ref_rq", "old"):
        bp, bq = bp0.clone().requires_grad_(True), bq0.clone().requires_grad_(True)
        kin = tee.kin_of(bp, bq)
        args = (kin, ref_bp[:, ANCHOR], ref_bq[:, ANCHOR], ref_bp, scale)
        if fn == "old":
          hard, soft = termination_55e0eaf(heads, *args)
        elif fn == "new":
          hard, soft = heads.termination(*args)
        else:  # the reference pelvis passed although the term is off: ignored
          hard, soft = heads.termination(*args, ref_rq=ref_bq[:, 0])
        gp, gq = torch.autograd.grad((soft * torch.linspace(0.5, 1.5, soft.shape[0])).sum(), (bp, bq))
        outs.append((hard, soft.detach(), gp, gq))
      assert 0 < int(outs[2][0].sum()) < 4096
      for o in outs[:2]:
        for a, b in zip(o, outs[2]):
          assert torch.equal(a, b), "switch off must reproduce 55e0eaf's termination bitwise"


def test_off_termination_compiled_is_55e0eaf_bitwise():
  try:
    heads = g1_heads(True, 0.0)
    new = torch.compile(lambda k, ap, aq, bp, rq: heads.termination(k, ap, aq, bp, ref_rq=rq), dynamic=False)
    old = torch.compile(lambda k, ap, aq, bp: termination_55e0eaf(heads, k, ap, aq, bp), dynamic=False)
    ref_bp, ref_bq, bp, bq = tee.random_batch(1024, 7)
    a = new(tee.kin_of(bp, bq), ref_bp[:, ANCHOR], ref_bq[:, ANCHOR], ref_bp, ref_bq[:, 0])
    b = old(tee.kin_of(bp, bq), ref_bp[:, ANCHOR], ref_bq[:, ANCHOR], ref_bp)
  except Exception as e:  # noqa: BLE001  (no C++ toolchain for inductor)
    print(f"  (compiled check skipped: {type(e).__name__}: {str(e)[:120]})")
    return
  assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def _wm(seed: int = 3) -> DynamicsModel:
  torch.manual_seed(seed)
  wm = DynamicsModel(J, XI, 0.02, WorldModelCfg(hidden=(16,)))
  with torch.no_grad():
    for p in wm.net.parameters():  # zero_last=True would make the model a pure integrator
      p.add_(torch.randn_like(p) * 0.05)
  return wm


def _pack(N: int, L: int, seed: int):
  """Reference window pack [N, L, ...] from the real FK of reference states."""
  s = random_states(N * L, seed, yaw_spread=1.0)
  pos, quat, lin, ang = FK.forward(s[:, 0:3], s[:, 3:7], s[:, 7:10], s[:, 10:13], s[:, 13 : 13 + J], s[:, 13 + J :])
  tr = torch.as_tensor(TRACKED_LOCAL)
  v = lambda x: x[:, tr].reshape(N, L, len(TRACKED), -1)  # noqa: E731
  return (s[:, 13 : 13 + J].reshape(N, L, J), s[:, 13 + J :].reshape(N, L, J), v(pos), v(quat), v(lin), v(ang))


def test_off_step_fns_are_55e0eaf_bitwise():
  for ee in (True, False):
    cfg = HeadsCfg(ee_termination=ee)
    heads = real_heads(cfg)
    policy, wm = small_policy(heads), _wm()
    new = StepFns(heads, policy, wm, None, compile_=False)
    old = StepFns(old_heads(heads), policy, wm, None, compile_=False)
    N, K, H = 64, cfg.K, 6
    p = _pack(N, H + K, 11)
    s0 = random_states(N, 12, yaw_spread=1.0)
    g = torch.Generator().manual_seed(13)
    obs_hist = torch.randn(N, cfg.hist, heads.proprio_dim, generator=g)
    prev_a = torch.randn(N, J, generator=g) * 0.1
    xi = torch.randn(N, XI, generator=g) * 0.3
    bias = SimInterface.bias_from_xi(xi, J)
    f_loc = torch.zeros(N, dtype=torch.long)
    clip_end_loc = torch.randint(3, H + K, (N,), generator=g)
    for scale in (None, 0.7 + torch.rand(N, generator=g)):
      res = []
      for fns, fn in ((new, StepFns._imagine_step), (old, imagine_step_55e0eaf)):
        policy.zero_grad(set_to_none=True)
        torch.manual_seed(5)
        s, oh, pa, fl, kin = s0, obs_hist, prev_a, f_loc, heads.kin(s0)
        outs = []
        tot = 0.0
        for _ in range(3):  # a short imagined rollout (BPTT through the world model and the heads)
          s, oh, pa, fl, kin, r, d, fn_, gn_ = fn(fns, s, oh, pa, fl, clip_end_loc, xi, bias, kin, *p, scale)
          outs += [s, oh, pa, fl, r, d, fn_, gn_] + [kin[k] for k in sorted(kin)]
          tot = tot + (r * (1.0 - d)).sum()
        tot.backward()
        outs += [q.grad.clone() for q in policy.parameters()]
        res.append([o.detach() for o in outs])
      for a, b in zip(*res):
        assert torch.equal(a, b), "imagine_step off must be 55e0eaf bitwise"
    # simulator side
    ns = random_states(N, 14, yaw_spread=1.0)
    kin = heads.kin(random_states(N, 15))
    a_exec, pa = torch.randn(N, J, generator=g), torch.randn(N, J, generator=g)
    ref = [x[:, 0] for x in p[2:]]
    for scale in (None, 0.5 + torch.rand(N, generator=g)):
      r1, h1 = new._post(ns, kin, a_exec, pa, *ref, scale)
      r2, h2 = post_55e0eaf(old, ns, kin, a_exec, pa, *ref, scale)
      assert torch.equal(r1, r2) and torch.equal(h1, h2)


# ---------------------------------------------------------------------------
# (b) switch on
# ---------------------------------------------------------------------------


def _pelvis_batch(rot: torch.Tensor, side: str = "world", seed: int = 21):
  """Robot exactly on the reference except the pelvis (tracked body 0) rotated by rot [B,4] (world or body frame)."""
  B = rot.shape[0]
  ref_bp, ref_bq, _, _ = tee.random_batch(B, seed)
  bq = ref_bq.clone()
  bq[:, 0] = quat_mul(rot, ref_bq[:, 0]) if side == "world" else quat_mul(ref_bq[:, 0], rot)
  return ref_bp, ref_bq, ref_bp.clone(), bq


def _base_alive(heads: AnalyticHeads) -> float:
  """Survival factor of the anchor (+ end-effector) terms at zero error."""
  c = heads.cfg
  a = (1.0 - torch.sigmoid(torch.tensor(-c.term_z / c.term_z_temp))) * (1.0 - torch.sigmoid(torch.tensor(-c.term_tilt / c.term_tilt_temp)))
  if c.ee_termination:
    a = a * (1.0 - torch.sigmoid(torch.tensor(-c.ee_term_z / c.ee_term_z_temp))) ** len(EE_IDX)
  return float(a)


def test_on_flips_exactly_at_threshold():
  A, T = 1.0, 0.05
  for ee in (False, True):
    heads = g1_heads(ee, A, root_ori_term_temp=T)
    ang = torch.tensor([0.0, 0.3, A - 1e-3, A - 1e-4, A + 1e-4, A + 1e-3, 1.7, 3.1])
    g = torch.Generator().manual_seed(4)
    axes = [torch.tensor([0.0, 0.0, 1.0]), torch.tensor([1.0, 0.0, 0.0])] + [_unit(torch.randn(3, generator=g)) for _ in range(3)]
    for ai, axis in enumerate(axes):
      for sign in (1.0, -1.0):
        for side in ("world", "body"):
          rot = quat_from_angle_axis(sign * ang, axis.expand(len(ang), 3))
          ref_bp, ref_bq, bp, bq = _pelvis_batch(rot, side)
          kin = tee.kin_of(bp, bq)
          hard, soft = heads.termination(kin, ref_bp[:, ANCHOR], ref_bq[:, ANCHOR], ref_bp, ref_rq=ref_bq[:, 0])
          assert hard.tolist() == [False] * 4 + [True] * 4, (ai, sign, side, hard.tolist())
          got = heads.root_ori_angle(kin, ref_bq[:, 0])
          assert torch.allclose(got, ang, atol=3e-6), (got - ang).abs().max()
          expect = 1.0 - _base_alive(heads) * (1.0 - torch.sigmoid((got - A) / T))
          assert torch.allclose(soft, expect, atol=1e-6), (soft, expect)
          assert abs(float(soft[3]) - 0.5) < 2e-3 and abs(float(soft[4]) - 0.5) < 2e-3, "0.5 at the threshold"
          assert float(soft[-1]) > 0.999999
          if ai == 0 and side == "world":  # pure world yaw: the pelvis' gravity direction is untouched
            g_ = heads.gravity.expand(len(ang), 3)
            tilt = (quat_apply_inverse(ref_bq[:, 0], g_)[:, 2] - quat_apply_inverse(bq[:, 0], g_)[:, 2]).abs()
            assert float(tilt.max()) < 1e-6, "yaw is invisible to a gravity-tilt test"
  # dense sweep: the hard test IS angle > A, no tolerance band
  heads = g1_heads(False, A)
  ang = torch.linspace(A - 0.01, A + 0.01, 20001)
  ref_bp, ref_bq, bp, bq = _pelvis_batch(quat_from_angle_axis(ang, torch.tensor([0.0, 0.0, 1.0]).expand(len(ang), 3)), "world", 5)
  kin = tee.kin_of(bp, bq)
  hard, _ = heads.termination(kin, ref_bp[:, ANCHOR], ref_bq[:, ANCHOR], ref_bp, ref_rq=ref_bq[:, 0])
  got = heads.root_ori_angle(kin, ref_bq[:, 0])
  assert torch.equal(hard, got > A)
  assert torch.equal(hard[(ang - A).abs() > 1e-5], (ang > A)[(ang - A).abs() > 1e-5])
  # curriculum factor: threshold and temperature scaled per sample
  for kf in (0.5, 2.0):
    a2 = torch.tensor([kf * A - 1e-3, kf * A + 1e-3])
    ref_bp, ref_bq, bp, bq = _pelvis_batch(quat_from_angle_axis(a2, torch.tensor([0.0, 0.0, 1.0]).expand(2, 3)), "world", 6)
    hard, soft = heads.termination(tee.kin_of(bp, bq), ref_bp[:, ANCHOR], ref_bq[:, ANCHOR], ref_bp, torch.full((2,), kf),
                                   ref_rq=ref_bq[:, 0])
    assert hard.tolist() == [False, True] and abs(float(soft.mean()) - 0.5) < 0.02


def test_on_gradients():
  A, T = 1.0, 0.05
  heads = g1_heads(False, A, root_ori_term_temp=T)
  psi0 = torch.tensor([0.0, 0.5, 0.9, 0.97, 1.0, 1.03, 1.1, 1.5, -0.9, -1.0, -1.1])
  B = psi0.shape[0]
  ref_bp, ref_bq, bp, _ = _pelvis_batch(yaw_q(torch.zeros(B)), "body", 9)
  psi = psi0.clone().requires_grad_(True)
  bq = ref_bq.clone()
  bq[:, 0] = quat_mul(ref_bq[:, 0], yaw_q(psi))  # pure body-frame yaw (pelvis yaw through the waist)
  _, soft = heads.termination(tee.kin_of(bp, bq), ref_bp[:, ANCHOR], ref_bq[:, ANCHOR], ref_bp, ref_rq=ref_bq[:, 0])
  (gpsi,) = torch.autograd.grad(soft.sum(), psi)
  with torch.no_grad():
    p = torch.sigmoid((psi0.abs() - A) / T)
    expect = _base_alive(heads) * p * (1.0 - p) / T * torch.sign(psi0)
  assert torch.isfinite(gpsi).all()
  assert torch.allclose(gpsi, expect, rtol=2e-3, atol=1e-6), (gpsi.tolist(), expect.tolist())
  assert abs(float(gpsi[0])) < 1e-6, "no push at the reference"
  nz = psi0 != 0
  assert torch.all(torch.sign(gpsi[nz]) == torch.sign(psi0[nz])), "descent on the soft termination turns the pelvis back"
  i = int((psi0 - A).abs().argmin())
  assert float(gpsi[i]) > float(gpsi[i - 1]) and float(gpsi[i]) > float(gpsi[i + 1]), "peak at the threshold"
  assert abs(float(gpsi[i]) - _base_alive(heads) * 0.25 / T) < 1e-2
  # w.r.t. the quaternion itself: finite at angle 0 (arccos would be infinite there), only the pelvis gets a gradient
  bq2 = ref_bq.clone().requires_grad_(True)
  bp2 = bp.clone().requires_grad_(True)
  _, soft = heads.termination(tee.kin_of(bp2, bq2), ref_bp[:, ANCHOR], ref_bq[:, ANCHOR], ref_bp, ref_rq=ref_bq[:, 0])
  gp, gq = torch.autograd.grad(soft.sum(), (bp2, bq2))
  assert torch.isfinite(gq).all() and torch.isfinite(gp).all()
  # through the real FK + state: the pelvis gradient reaches s[3:7], finite everywhere
  rh = real_heads(HeadsCfg(root_ori_term=A, ee_termination=False))
  s = random_states(256, 31).requires_grad_(True)
  kin = rh.kin(s)
  ref = rh.kin(random_states(256, 32))
  _, soft = rh.termination(kin, ref["anchor_pos"], ref["anchor_quat"], ref["body_pos"], ref_rq=ref["body_quat"][:, 0])
  (gs,) = torch.autograd.grad(soft.sum(), s)
  assert torch.isfinite(gs).all() and float(gs[:, 3:7].abs().sum()) > 0


def test_real_fk_root_is_state_and_waist_yaw_case():
  heads = real_heads(HeadsCfg(root_ori_term=1.0))
  s = random_states(512, 41)
  kin = heads.kin(s)
  assert torch.equal(kin["body_quat"][:, 0], s[:, 3:7]), "FK root quaternion == state quaternion, bitwise"
  # root_ori_angle_state reads s[3:7] directly: equal to the FK path
  ref_bq = heads.kin(random_states(512, 42))["body_quat"]
  assert torch.equal(heads.root_ori_angle_state(s, ref_bq), heads.root_ori_angle(kin, ref_bq[:, 0]))
  h_nr = real_heads(HeadsCfg())
  h_nr.root_ori_is_root = False  # any other tracked body would go through the FK
  assert torch.equal(h_nr.root_ori_angle_state(s, ref_bq), heads.root_ori_angle(kin, ref_bq[:, 0]))
  # Pelvis yawed 1.2 rad, waist yaw -1.2: the torso (anchor) is where the reference has it, the feet only move in xy.
  n = 8
  ref = random_states(n, 43, yaw_spread=3.0)
  ref[:, 3:7] = yaw_q((torch.rand(n) * 2 - 1) * 3.0)  # upright reference pelvis (any heading)
  ref[:, 13 + WAIST_YAW] = 0.0
  rob = ref.clone()
  psi = torch.linspace(-1.5, 1.5, n)
  rob[:, 3:7] = quat_mul(yaw_q(psi), ref[:, 3:7])
  rob[:, 13 + WAIST_YAW] = -psi
  kr, kb = heads.kin(ref), heads.kin(rob)
  assert torch.allclose(kb["anchor_quat"].abs(), kr["anchor_quat"].abs(), atol=1e-6)  # torso on the reference
  args = (kr["anchor_pos"], kr["anchor_quat"], kr["body_pos"])
  hard_old, soft_old = termination_55e0eaf(heads, kb, *args)
  hard_new, soft_new = heads.termination(kb, *args, ref_rq=kr["body_quat"][:, 0])
  assert not hard_old.any() and float(soft_old.max()) < 1e-4, "55e0eaf's terms see no failure"
  assert hard_new.tolist() == (psi.abs() > 1.0).tolist(), "the root-orientation term fires on the pelvis yaw"
  assert torch.allclose(heads.root_ori_angle(kb, kr["body_quat"][:, 0]), psi.abs(), atol=2e-6)


def test_env_term_equals_heads_hard_test():
  E = 4096
  ref_bp, ref_bq, bp, bq = tee.random_batch(E, 51)
  # spread the pelvis error around the threshold
  bq[:, 0] = quat_mul(quat_from_angle_axis(torch.rand(E, generator=torch.Generator().manual_seed(52)) * 2.0,
                                           _unit(torch.randn(E, 3, generator=torch.Generator().manual_seed(53)))), ref_bq[:, 0])
  g = torch.Generator().manual_seed(54)
  env_hard = torch.rand(E, generator=g) < 0.4
  robot_names = ("pelvis", "extra_link") + tuple(tee.G1_TRACKED[1:])  # robot body order differs from the tracked list
  link_quat = torch.cat([bq[:, :1], _unit(torch.randn(E, 1, 4, generator=g)), bq[:, 1:]], 1)
  cmd = types.SimpleNamespace(body_quat_w=ref_bq, cfg=types.SimpleNamespace(body_names=tee.G1_TRACKED), env_hard=env_hard,
                              hard_mult=1.0, term_scale_env=torch.ones(E),
                              robot=types.SimpleNamespace(body_names=robot_names, data=types.SimpleNamespace(body_link_quat_w=link_quat)))
  cmd.set_hard_mult = types.MethodType(LibraryMotionCommand.set_hard_mult, cmd)
  env = types.SimpleNamespace(command_manager=types.SimpleNamespace(get_term=lambda n: cmd))
  for A in (1.0, 0.4472):
    heads = g1_heads(False, A)
    for m in (1.0, 2.0, 0.5):
      cmd.set_hard_mult(m)
      got = bad_root_ori_full(env, "motion", A, "pelvis")
      scale = None if m == 1.0 else cmd.term_scale_env  # the trainer passes None without the curriculum
      kin = tee.kin_of(bp, bq)
      hard, _ = heads.termination(kin, ref_bp[:, ANCHOR], ref_bq[:, ANCHOR], None, scale, ref_rq=ref_bq[:, 0])
      hard_rest, _ = g1_heads(False, 0.0).termination(kin, ref_bp[:, ANCHOR], ref_bq[:, ANCHOR], None, scale)
      ro_heads = heads.root_ori_angle(kin, ref_bq[:, 0]) > (A if scale is None else A * scale)
      assert torch.equal(hard, hard_rest | ro_heads)
      assert torch.equal(got, ro_heads), "env term == the heads' root-orientation test, bitwise"
      assert 0 < int(got.sum()) < E and bool((got & ~hard_rest).any())
  # another tracked body by name (robot-configurable): the robot's own body order is used
  got = bad_root_ori_full(env, "motion", 1.0, "torso_link")
  assert torch.equal(got, quat_angle(ref_bq[:, ANCHOR], link_quat[:, robot_names.index("torso_link")]) > cmd.term_scale_env)


def test_quat_angle_is_the_eval_site_angle():
  g = torch.Generator().manual_seed(61)
  q1 = _unit(torch.randn(20000, 4, generator=g, dtype=torch.float64))
  q2 = _unit(torch.randn(20000, 4, generator=g, dtype=torch.float64))
  q2[:5000] = quat_mul(quat_from_angle_axis(torch.rand(5000, generator=g, dtype=torch.float64) * 1e-3,
                                            _unit(torch.randn(5000, 3, generator=g, dtype=torch.float64))), q1[:5000])  # near 0
  q2[5000:6000] = -q2[5000:6000]  # double cover
  ours = quat_angle(q1, q2).numpy()
  site = site_quat_error_angle(q1.numpy(), q2.numpy())
  assert np.abs(ours - site).max() < 2e-7, np.abs(ours - site).max()  # arccos near 1: ~sqrt(eps) on the site's side
  # scale-invariant (unnormalized world-model / file quaternions), float32 as trained
  ours32 = quat_angle((q1 * 1.3).float(), (q2 * 0.7).float()).double().numpy()
  assert np.abs(ours32 - site).max() < 2e-5
  far = np.abs(site - 1.0) > 1e-4
  assert ((ours32 > 1.0) == (site > 1.0))[far].all()


def test_misuse_and_cfg():
  assert HeadsCfg().root_ori_term == 0.0 and not HeadsCfg().root_ori_on and HeadsCfg().root_ori_term_temp == 0.05
  for bad in (dict(root_ori_term=-0.1), dict(root_ori_term=1.0, root_ori_term_temp=0.0), dict(root_ori_term=float("nan"))):
    try:
      HeadsCfg(**bad)
    except ValueError:
      continue
    raise AssertionError(f"{bad} must be rejected")
  try:
    AnalyticHeads(tee._FK(), list(range(BT)), ANCHOR, torch.zeros(NJ), torch.zeros(NJ, 2), HeadsCfg(root_ori_term=1.0), ee_indices=EE_IDX)
  except ValueError:
    pass
  else:
    raise AssertionError("root_ori_term without a root-orientation body must fail")
  heads = g1_heads(True, 1.0)
  ref_bp, ref_bq, bp, bq = tee.random_batch(4, 0)
  try:
    heads.termination(tee.kin_of(bp, bq), ref_bp[:, ANCHOR], ref_bq[:, ANCHOR], ref_bp)
  except AssertionError:
    pass
  else:
    raise AssertionError("root_ori_term without ref_rq must fail")
  # checkpoint cfgs of every branch load (no root keys -> off), this branch's round-trips
  d = dataclasses.asdict(HeadsCfg(root_ori_term=1.0, root_ori_term_temp=0.04))
  assert HeadsCfg.from_dict(d) == HeadsCfg(root_ori_term=1.0, root_ori_term_temp=0.04)
  old = {k: v for k, v in d.items() if not k.startswith("root_ori")}
  assert not HeadsCfg.from_dict(old).root_ori_on


# ---------------------------------------------------------------------------
# (c) collect_step / evaluator on a scripted simulator
# ---------------------------------------------------------------------------

CLIP_NAMES = ("g1__a_floor_getup_0.npz", "g1__b_walk_1.npz", "g1__c_spin360_2.npz", "g1__d_zhuanti_3.npz", "g1__e_crawl_4.npz",
              "g1__f_run_5.npz")
CLIP_LEN = 30
DRIFT = (0.0, 0.07, 0.11, -0.13, 0.3, -0.17)  # pelvis yaw drift per env [rad / step]: crosses 1.0 rad at known steps
FAIL_AT = (7, 1000, 12, 1000, 5, 1000)  # scripted (non-orientation) failure at this episode step


class FakeRobot:
  body_names = BODY_NAMES

  def __init__(self, world):
    self.world = world

  @property
  def data(self):
    s = self.world.robot_state
    _, quat, _, _ = FK.forward(s[:, 0:3], s[:, 3:7], s[:, 7:10], s[:, 10:13], s[:, 13 : 13 + J], s[:, 13 + J :])
    return types.SimpleNamespace(body_link_quat_w=quat)


class FakeCmd:
  """LibraryMotionCommand stand-in: frame bookkeeping + the real update_bin_returns."""

  def __init__(self, world, E: int, NB: int, C: int):
    base = tsc.make_cmd(NB, False)
    self.world, self.library, self.cfg, self.device = world, base.library, base.cfg, "cpu"
    self.cfg.body_names = TRACKED
    self.bin_G, self.bin_visits, self.bin_weight_hist = base.bin_G, base.bin_visits, base.bin_weight_hist
    self.update_bin_returns = types.MethodType(LibraryMotionCommand.update_bin_returns, self)
    self.time_steps = torch.zeros(E, dtype=torch.long)
    self.clip_end_env = torch.zeros(E, dtype=torch.long)
    self.env_clip = torch.zeros(E, dtype=torch.long)
    self.term_scale_env, self.env_hard, self.hard_mult = torch.ones(E), torch.zeros(E, dtype=torch.bool), 1.0
    self.clip_hard = torch.zeros(C, dtype=torch.bool)
    self._start_counter = 0
    self.robot = FakeRobot(world)
    self.metrics = {"sampling_entropy": torch.tensor([0.9]), "sampling_top1_prob": torch.tensor([0.01])}

  @property
  def body_quat_w(self):
    return self.world.ref["body_quat"][self.time_steps]


class FakeWorld:
  """Scripted simulator (env + SimInterface + command + library in one) for Trainer.collect_step / Evaluator.run.

  The robot follows the reference of its current frame exactly, except a pelvis yaw of drift[e] x (episode step) about
  the world z (pure yaw: invisible to every 55e0eaf term) and joints offset by 0.02 tanh(action). Terminations:
  scripted failures at fail_at[e] (as 'anchor_pos'), the real bad_root_ori_full when A > 0, time-out at the clip end.
  Every step is logged with its closed-form pelvis angle (float64) for the watch's closed form."""

  def __init__(self, heads: AnalyticHeads, E: int, A: float = 0.0, fail_at=FAIL_AT, drift=DRIFT, mode: str = "train",
               eval_clips=None):
    self.heads, self.E, self.A, self.mode, self.nj = heads, E, float(A), mode, J
    C = len(CLIP_NAMES)
    T = C * CLIP_LEN
    starts = torch.arange(C) * CLIP_LEN
    lib = types.SimpleNamespace(clip_start=starts, clip_end=starts + CLIP_LEN, num_clips=C, fingerprint="fake library",
                                clips=[types.SimpleNamespace(name=n, start=int(s), end=int(s) + CLIP_LEN) for n, s in zip(CLIP_NAMES, starts)])
    lib.clip_of_frame = types.MethodType(MotionLibrary.clip_of_frame, lib)
    lib.bin_of_frame = (torch.arange(T) // 10).to(torch.int32)
    lib.num_bins = T // 10
    lib.bin_clip = torch.arange(lib.num_bins) // (CLIP_LEN // 10)
    self.lib = lib
    f = torch.arange(T, dtype=torch.float32)
    rp = torch.stack([0.01 * f, torch.zeros_like(f), torch.full_like(f, 0.8)], -1)
    jp = 0.1 * torch.sin(f[:, None] / 7.0 + torch.arange(J)[None, :])
    self.ref_state = torch.cat([rp, yaw_q(0.02 * f), torch.zeros(T, 6), jp, torch.zeros(T, J)], -1)
    s = self.ref_state
    pos, quat, lin, ang = FK.forward(s[:, 0:3], s[:, 3:7], s[:, 7:10], s[:, 10:13], s[:, 13 : 13 + J], s[:, 13 + J :])
    tr = heads.tracked
    self.ref = dict(joint_pos=jp, joint_vel=torch.zeros(T, J), body_pos=pos[:, tr], body_quat=quat[:, tr],
                    body_lin_vel=lin[:, tr], body_ang_vel=ang[:, tr])
    self.drift = torch.tensor(drift[:E], dtype=torch.float32)
    self.fail_at = None if fail_at is None else torch.tensor(fail_at[:E])
    self.cmd = FakeCmd(self, E, lib.num_bins, C)
    self.command_manager = types.SimpleNamespace(get_term=lambda n: self.cmd)
    self.eval_clips = list(eval_clips if eval_clips is not None else range(C))
    self.k = torch.zeros(E, dtype=torch.long)
    self.order = torch.zeros(E, dtype=torch.long)
    self.robot_state = self.ref_state[:E].clone()
    self.next_clip = torch.arange(E) % C
    self.n_resets = 0
    self.log, self.reset_log = [], []
    z = torch.zeros(E, dtype=torch.bool)
    self.terms = {"anchor_pos": z, "anchor_ori": z, "ee_body_pos": z}
    self.reset_terminated, self.reset_time_outs = z, z
    self.termination_manager = self
    self.active_terms = ["anchor_pos", "anchor_ori", "ee_body_pos"] + (["root_ori"] if self.A > 0 else []) + ["clip_end"]
    self.max_episode_length = 10_000

  def get_term(self, name):
    return self.terms[name]

  # ---- SimInterface ----
  def state(self):
    return self.robot_state.clone()

  def xi(self):
    return torch.zeros(self.E, XI)

  bias_from_xi = staticmethod(SimInterface.bias_from_xi)

  def goal_refs(self, frame, clip_end):
    K, E, a = self.heads.cfg.K, frame.shape[0], self.heads.anchor
    fr = window_frames(frame, clip_end, K).reshape(-1)
    r = self.ref
    return (r["joint_pos"][fr].view(E, K, -1), r["joint_vel"][fr].view(E, K, -1), r["body_pos"][fr].view(E, K, -1, 3)[:, :, a],
            r["body_quat"][fr].view(E, K, -1, 4)[:, :, a])

  def current_ref(self):
    ts = self.cmd.time_steps
    return tuple(self.ref[k][ts] for k in ("body_pos", "body_quat", "body_lin_vel", "body_ang_vel"))

  # ---- env ----
  def step(self, a):
    cmd, E = self.cmd, self.E
    self.k = self.k + 1
    f = cmd.time_steps.clone()
    s = self.ref_state[f].clone()
    s[:, 3:7] = quat_mul(yaw_q(self.drift * self.k.float()), s[:, 3:7])
    s[:, 13 : 13 + J] += 0.02 * torch.tanh(a)
    self.robot_state = s
    z = torch.zeros(E, dtype=torch.bool)
    scripted = (self.k >= self.fail_at) if self.fail_at is not None else z
    self.terms = {"anchor_pos": scripted, "anchor_ori": z.clone(), "ee_body_pos": z.clone()}
    term = scripted
    if self.A > 0:
      self.terms["root_ori"] = bad_root_ori_full(self, "motion", self.A, "pelvis")  # the real env term
      term = term | self.terms["root_ori"]
    self.reset_terminated = term
    self.reset_time_outs = f >= cmd.clip_end_env - 1
    th = (self.drift.double() * self.k.double()).abs() % (2 * math.pi)
    clip = cmd.env_clip.clone() if self.mode == "eval" else self.lib.clip_of_frame(f)
    self.log.append(dict(angle=torch.minimum(th, 2 * math.pi - th), clip=clip, term=term.clone(), timeout=self.reset_time_outs.clone(),
                         order=self.order.clone(), k=self.k.clone()))
    cmd.time_steps = torch.minimum(f + 1, cmd.clip_end_env - 1)

  def reset(self, env_ids=None):
    cmd, C = self.cmd, self.lib.num_clips
    ids = torch.arange(self.E) if env_ids is None else env_ids
    for e in ids.tolist():
      if self.mode == "eval":  # "start" mode: clips in sequence from their first frame
        clip = self.eval_clips[cmd._start_counter % len(self.eval_clips)]
        self.order[e] = cmd._start_counter
        cmd._start_counter += 1
        cmd.env_clip[e] = clip
        start = int(self.lib.clip_start[clip])
      else:
        clip = int(self.next_clip[e])
        self.next_clip[e] = (clip + 1 + e) % C
        start = int(self.lib.clip_start[clip]) + self.n_resets % 5
      self.n_resets += 1
      cmd.time_steps[e] = start
      cmd.clip_end_env[e] = int(self.lib.clip_end[clip])
      self.k[e] = 0
      self.robot_state[e] = self.ref_state[start]
      if env_ids is not None:
        self.reset_log.append(clip)


def collect_trainer(world: FakeWorld, heads: AnalyticHeads, policy: Policy, cfg: TrainCfg, dist: DistCtx | None = None,
                    before: bool = False) -> Trainer:
  """Trainer with Trainer.__init__'s simulator bookkeeping on the scripted world (real collect_step / _finish_episodes /
  ReplayBuffer / ObsHistory / StepFns / heads / policy; before=True: the 55e0eaf versions)."""
  t = object.__new__(Trainer)
  t.cfg = cfg
  t.dist = dist if dist is not None else DistCtx(device="cpu")
  t.main = t.dist.is_main
  t.device = "cpu"
  t.env = t.sim = world
  t.cmd, t.lib, t.heads, t.policy = world.cmd, world.lib, heads, policy
  if before:
    t.fns = StepFns(old_heads(heads), policy, compile_=False)
    t.fns.post = types.MethodType(post_55e0eaf, t.fns)
    t.collect_step = types.MethodType(collect_step_55e0eaf, t)
    t._finish_episodes = types.MethodType(finish_episodes_55e0eaf, t)
  else:
    t.fns = StepFns(heads, policy, compile_=False)
    t.watch = FailureWatch([c.name for c in world.lib.clips], world.E, "cpu")
  t.nj, t.xi_dim, t.E = J, XI, world.E
  t.buffer = ReplayBuffer(world.E, 64, heads.state_dim, J, XI, "cpu")
  t.env_steps = 0
  E = world.E
  world.reset()
  t.state, t.xi = world.state(), world.xi()
  t.frame, t.clip_end = world.cmd.time_steps.clone(), world.cmd.clip_end_env.clone()
  t.prev_action = torch.zeros(E, J)
  t.ep_step = torch.zeros(E, dtype=torch.long)
  t.hist = ObsHistory(E, cfg.heads.hist, heads.proprio_dim, "cpu")
  with torch.no_grad():
    t.kin, obs = t.fns.refresh(t.state, t.prev_action, world.bias_from_xi(t.xi, J), True)
  t.hist.reset(torch.arange(E), obs)
  L = 64
  t.ep_bins, t.ep_rews, t.ep_ret = torch.zeros(E, L, dtype=torch.long), torch.zeros(E, L), torch.zeros(E)
  t._gamma_pow = cfg.gamma ** torch.arange(L, dtype=torch.float32)
  t._ar = torch.arange(E)
  t.stats = {"ep_return_norm": [], "ep_len": [], "ep_failed": [], "ep_failed_ee": [], "term_mismatch": [], "sim_nonfinite": [],
             "hard_ep_failed": [], "hard_ep_len": []}
  t._ee_on, t._hard_on = cfg.heads.ee_termination, False
  t._ro_on = cfg.heads.root_ori_on
  return t


def _groups_of(name: str) -> list[bool]:
  return [any(k in name.lower() for k in kws) for _, kws in WATCH_GROUPS]


def closed_form_sums(world: FakeWorld, thr: float = ORI_FAIL_THRESHOLD) -> dict:
  """The watch's counts from the world's own log (python, float64 angles)."""
  E = world.E
  over = [False] * E
  s = {"watch/steps": 0.0, "watch/steps_over": 0.0, "watch/episodes": 0.0, "watch/episodes_over": 0.0, "watch/resets": 0.0}
  for g, _ in WATCH_GROUPS:
    s.update({f"watch/{g}_resets": 0.0, f"watch/{g}_episodes": 0.0, f"watch/{g}_failed": 0.0, f"watch/{g}_over": 0.0})
  for rec in world.log:
    for e in range(E):
      a = float(rec["angle"][e])
      assert abs(a - thr) > 1e-3, "scripted angle too close to the threshold for an exact count"
      s["watch/steps"] += 1
      s["watch/steps_over"] += a > thr
      over[e] = over[e] or a > thr
      if bool(rec["term"][e]) or bool(rec["timeout"][e]):
        s["watch/episodes"] += 1
        s["watch/episodes_over"] += over[e]
        for (g, _), m in zip(WATCH_GROUPS, _groups_of(CLIP_NAMES[int(rec["clip"][e])])):
          if m:
            s[f"watch/{g}_episodes"] += 1
            s[f"watch/{g}_failed"] += bool(rec["term"][e])
            s[f"watch/{g}_over"] += over[e]
        over[e] = False
  for clip in world.reset_log:
    s["watch/resets"] += 1
    for (g, _), m in zip(WATCH_GROUPS, _groups_of(CLIP_NAMES[clip])):
      s[f"watch/{g}_resets"] += m
  return s


def _cfg(**heads_kw) -> TrainCfg:
  return TrainCfg(log_dir=tempfile.mkdtemp(prefix="rootori_"), heads=HeadsCfg(**heads_kw))


def _snapshot(t: Trainer) -> list:
  b = t.buffer
  out = [getattr(b, k).clone() for k in ("state", "next_state", "action", "prev_action", "xi", "reward", "frame", "clip_end", "ep_step", "term", "done")]
  out += [t.cmd.bin_G.clone(), t.cmd.bin_visits.clone(), t.ep_bins.clone(), t.ep_rews.clone(), t.ep_ret.clone(), t.hist.buf.clone(),
          t.state.clone(), t.prev_action.clone(), t.frame.clone(), t.ep_step.clone()]
  out += [v.clone() for v in t.policy.state_dict().values()]  # incl. the obs / goal normalizers
  out += [t.kin[k].clone() for k in sorted(t.kin)]
  return out


def test_collect_off_is_55e0eaf_and_watch_closed_form():
  E, steps = 6, 110
  cfg = _cfg()
  runs = []
  for before in (False, True):
    heads = real_heads(cfg.heads)
    world = FakeWorld(heads, E)
    t = collect_trainer(world, heads, small_policy(heads), cfg, before=before)
    torch.manual_seed(77)
    for _ in range(steps):
      t.collect_step()
    runs.append((t, world, _snapshot(t), torch.get_rng_state()))
  (tn, wn, sn, rn), (to, wo, so, ro) = runs
  for a, b in zip(sn, so):
    assert torch.equal(a, b), "logging must not change what training sees"
  assert torch.equal(rn, ro), "no extra random numbers"
  assert tn.stats == to.stats and tn.env_steps == to.env_steps
  assert len(tn.stats["ep_len"]) > 20 and 0 < sum(tn.stats["ep_failed"]) < len(tn.stats["ep_failed"])
  got, want = tn.watch.sums(), closed_form_sums(wn)
  assert got == want, (got, want)
  assert want["watch/episodes_over"] > 0 and want["watch/floor_episodes"] > 0 and want["watch/spin_episodes"] > 0
  assert want["watch/floor_failed"] > 0 and want["watch/spin_over"] > 0 and want["watch/floor_resets"] > 0
  sc = tn.watch.scalars(got)
  assert sc["train/would_fail_ori"] == want["watch/episodes_over"] / want["watch/episodes"]
  assert sc["train/watch_spin_reset_share"] == want["watch/spin_resets"] / want["watch/resets"]
  print(f"  {steps} steps x {E} envs: {int(want['watch/episodes'])} episodes, would_fail_ori {sc['train/would_fail_ori']:.3f}, "
        f"floor fail {sc['train/watch_floor_fail']:.2f} / spin ori {sc['train/watch_spin_would_fail_ori']:.2f}")
  tn.watch.clear()
  z = tn.watch.sums()
  assert all(v == 0.0 for v in z.values()) and tn.watch.scalars(z) == {}


def test_collect_on_env_and_heads_agree():
  E, steps, A = 6, 110, 1.0
  cfg = _cfg(root_ori_term=A)
  heads = real_heads(cfg.heads)
  world = FakeWorld(heads, E, A=A, fail_at=None)
  t = collect_trainer(world, heads, small_policy(heads), cfg)
  torch.manual_seed(78)
  for _ in range(steps):
    t.collect_step()
  assert t.stats["term_mismatch"] == [0.0] * steps, "the simulator's root_ori term and the heads' hard test disagree"
  # every failure is the first step with angle > A; would_fail_ori counts exactly those episodes
  s = closed_form_sums(world)
  assert t.watch.sums() == s
  n_fail = sum(t.stats["ep_failed"])
  assert n_fail > 0 and s["watch/episodes_over"] == n_fail and s["watch/steps_over"] == n_fail
  first = {}
  for i, rec in enumerate(world.log):
    for e in range(E):
      if bool(rec["term"][e]):
        assert float(rec["angle"][e]) > A
        first.setdefault(e, int(rec["k"][e]))
  for e, k in first.items():  # first failure of env e at the first step its drift exceeds A
    assert k == math.floor(A / abs(DRIFT[e])) + 1, (e, k)
  assert t.stats["ep_failed_ee"] == [0.0] * len(t.stats["ep_failed_ee"])


def _eval_standin(world: FakeWorld, heads: AnalyticHeads, n: int) -> Evaluator:
  ev = object.__new__(Evaluator)
  ev.env = ev.sim = world
  ev.heads, ev.cfg, ev.device, ev.lib, ev.cmd = heads, heads.cfg, "cpu", world.lib, world.cmd
  ev.compile_, ev._fns, ev._fns_policy = False, None, None
  ev.eval_clips = tuple(range(n))
  ev.num_clips, ev.num_envs = n, world.E
  ev.clip_ids = torch.tensor(ev.eval_clips)
  return ev


def test_evaluator_failed_ori():
  E, n = 4, len(CLIP_NAMES)  # fewer envs than clips: reassignment and inactive (wrap-around) envs
  res = {}
  for before in (False, True):
    heads = real_heads(HeadsCfg())
    world = FakeWorld(heads, E, mode="eval", fail_at=(9, 1000, 1000, 6))
    ev = _eval_standin(world, heads, n)
    policy = small_policy(heads)
    if before:
      ev._fns = StepFns(old_heads(heads), policy, compile_=False)
      ev._fns.post = types.MethodType(post_55e0eaf, ev._fns)
      ev._fns_policy = policy
      res[before] = (eval_run_55e0eaf(ev, policy), world)
    else:
      res[before] = (Evaluator.run(ev, policy), world)
  (new, world), (old, _) = res[False], res[True]
  sc = dict(new["scalars"])
  ori, with_ori = sc.pop("failed_ori"), sc.pop("failed_with_ori")
  assert sc == old["scalars"] and new["per_clip"] == old["per_clip"] and new["failed_clips"] == old["failed_clips"]
  over = set()
  for rec in world.log:
    for e in range(E):
      if int(rec["order"][e]) < n and float(rec["angle"][e]) > ORI_FAIL_THRESHOLD:
        over.add(int(rec["clip"][e]))
  failed = {CLIP_NAMES.index(x) for x in old["failed_clips"]}
  assert ori == len(over) and with_ori == len(over | failed), (ori, over, failed)
  assert 0 < len(over) < n and failed - over and over - failed
  print(f"  eval: failed {int(sc['failed'])}, pelvis angle > 1 rad {int(ori)}, failed or that {int(with_ori)} of {n}")


# ---------------------------------------------------------------------------
# (d) the watch under data parallel / in the loop
# ---------------------------------------------------------------------------


def _watch_worker(rank: int, world_size: int, port: int, tmp: str):
  os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world_size))
  torch.set_num_threads(1)
  ctx = DistCtx.from_env("cpu", timeout_min=2.0)
  E = 6
  roll = lambda x: tuple(x[rank:] + x[:rank])  # noqa: E731  (each rank its own envs / clip mix)
  cfg = _cfg()
  heads = real_heads(cfg.heads)
  world = FakeWorld(heads, E, drift=roll(DRIFT), fail_at=roll(FAIL_AT))
  t = collect_trainer(world, heads, small_policy(heads), cfg, dist=ctx)
  t.opt_pi = torch.optim.Adam(t.policy.parameters(), lr=1e-3)
  t._t_start = 0.0
  torch.manual_seed(100 + rank)
  for _ in range(60):
    t.collect_step()
  local = closed_form_sums(world)
  assert t.watch.sums() == local
  scal = Trainer._dist_log_scalars(t, {}, 1.0, 60)
  # wall_h: each rank's clock; bins_visited_frac: the sampler state, merged across ranks before every log in the real loop
  # (not in this stand-in). Every other scalar, the watch's included, must be the same everywhere.
  local_keys = ("train/wall_h", "sampler/bins_visited_frac")
  everyone = ctx.all_gather_object({k: v for k, v in scal.items() if k not in local_keys})
  diff = sorted({k for s in everyone for k in s if s.get(k) != everyone[0].get(k)})
  assert not diff, f"log scalars differ between ranks: {diff}"
  pooled = {k: sum(d[k] for d in ctx.all_gather_object(local)) for k in local}
  want = FailureWatch(CLIP_NAMES, E, "cpu").scalars(pooled)
  assert {k: v for k, v in scal.items() if k in want} == want and set(want) <= set(scal)
  assert want["train/would_fail_ori"] == pooled["watch/episodes_over"] / pooled["watch/episodes"]
  locals_ = ctx.all_gather_object(local)
  assert len({json.dumps(x, sort_keys=True) for x in locals_}) == world_size, "the ranks' own counts should differ"
  if rank == 0:
    with open(os.path.join(tmp, "scal.json"), "w") as f:
      json.dump({k: scal[k] for k in sorted(want)}, f)
  ctx.barrier()
  ctx.close()
  open(os.path.join(tmp, f"rank{rank}.ok"), "w").close()


def test_watch_ddp_world3():
  with tempfile.TemporaryDirectory() as tmp:
    mp.spawn(_watch_worker, args=(3, tsc._free_port(), tmp), nprocs=3, join=True)
    assert all(os.path.exists(os.path.join(tmp, f"rank{r}.ok")) for r in range(3))
    sc = json.load(open(os.path.join(tmp, "scal.json")))
    print(f"  3 ranks pooled: would_fail_ori {sc['train/would_fail_ori']:.3f}, floor fail {sc['train/watch_floor_fail']:.3f}, "
          f"spin reset share {sc['train/watch_spin_reset_share']:.3f}")


def test_single_process_loop_logs_and_clears():
  with tempfile.TemporaryDirectory() as d:
    t = tes.loop_trainer(d, eval_enabled=False)
    E = t.E
    t.watch = FailureWatch(["g1__x_floor.npz", "g1__y_spin.npz", "g1__z_walk.npz"], E, "cpu")
    base = t.collect_step
    events = []

    def collect_step():
      base()
      k = t.env_steps
      t.watch.step(torch.tensor([1.5] + [0.2] * (E - 1)))  # env 0 always over 1 rad
      if k % 2 == 0:  # two episodes end (env 0 failed, env 1 timed out), two new ones start (floor, walk)
        t.watch.episodes(torch.tensor([0, 1]), torch.tensor([True, False]), torch.tensor([0, 1]))
        t.watch.resets(torch.tensor([0, 2]))
        events.append(k)

    t.collect_step = collect_step
    r = tes.run_loop(t)
  tb = {}
  for k, v, it in r["tb"]:
    tb.setdefault(it, {})[k] = v
  log_its = sorted(it for it, d_ in tb.items() if "train/env_steps_per_env" in d_)
  assert len(log_its) >= 5
  prev = 0
  for it in log_its:
    s = tb[it]
    steps_now = int(s["train/env_steps_per_env"])
    n_ev = sum(1 for k in events if prev < k <= steps_now)
    prev = steps_now
    assert s["train/ori_over_step_frac"] == 1.0 / E
    if n_ev:
      assert s["train/would_fail_ori"] == 0.5 and s["train/watch_floor_fail"] == 1.0 and s["train/watch_floor_would_fail_ori"] == 1.0
      assert s["train/watch_spin_fail"] == 0.0 and s["train/watch_spin_would_fail_ori"] == 0.0
      assert s["train/watch_floor_reset_share"] == 0.5 and s["train/watch_spin_reset_share"] == 0.0
      assert s["train/watch_floor_episodes"] == n_ev and s["train/watch_spin_episodes"] == n_ev, "counts per log window (cleared)"
  lines = [l for l in r["stdout"].split("\n") if l.startswith("[it ")]
  assert lines and all(l.endswith("| ori>1 0.50") for l in lines), lines[:2]


# ---------------------------------------------------------------------------
# (e) wiring
# ---------------------------------------------------------------------------


def _files(tmp: str, n: int = 3) -> str:
  files = [os.path.join(tmp, f"g1__lafan1_{i}.npz") for i in range(n)]
  for f in files:
    open(f, "w").close()
  lst = os.path.join(tmp, "list_ro.txt")
  with open(lst, "w") as f:
    f.write("\n".join(files) + "\n")
  return lst


def test_cli_cfg_and_env_wiring():
  script = tmm.tsc_load_train()
  with tempfile.TemporaryDirectory() as tmp:
    lst = _files(tmp)
    c0 = tmm._parse(script, tmp, ["--motion_list", lst])
    assert (c0.heads.root_ori_term, c0.heads.root_ori_term_temp, c0.heads.root_ori_body) == (0.0, 0.05, "")
    c1 = tmm._parse(script, tmp, ["--motion_list", lst, "--root_ori_term", "1.0", "--root_ori_term_temp", "0.04", "--root_ori_body", "pelvis"])
    assert (c1.heads.root_ori_term, c1.heads.root_ori_term_temp, c1.heads.root_ori_body) == (1.0, 0.04, "pelvis")
    assert dataclasses.replace(c1.heads, root_ori_term=0.0, root_ori_term_temp=0.05, root_ori_body="") == c0.heads
    for bad in (["--root_ori_term", "-0.5"], ["--root_ori_term", "3.2"], ["--root_ori_term", "1.0", "--root_ori_term_temp", "0"]):
      try:
        tmm._parse(script, tmp, ["--motion_list", lst] + bad, quiet=True)
      except SystemExit as e:
        assert e.code == 2
      else:
        raise AssertionError(f"must be rejected: {bad}")
    # Trainer.__init__: train_cfg.json + make_env_cfg kwargs
    calls = []

    def fake_make_env_cfg(*a, **k):
      calls.append(k)
      raise tsc._Probe

    for h in (HeadsCfg(), HeadsCfg(root_ori_term=1.0, root_ori_term_temp=0.04)):
      log_dir = os.path.join(tmp, f"init_{h.root_ori_term}")
      with mock.patch.object(trainer_mod, "make_env_cfg", fake_make_env_cfg):
        try:
          Trainer(TrainCfg(log_dir=log_dir, device="cpu", heads=h))
        except tsc._Probe:
          pass
      hj = json.load(open(os.path.join(log_dir, "train_cfg.json")))["heads"]
      assert (hj["root_ori_term"], hj["root_ori_term_temp"], hj["root_ori_body"]) == (h.root_ori_term, h.root_ori_term_temp, "")
    assert [(k["root_ori_term"], k["root_ori_body"]) for k in calls] == [(0.0, ""), (1.0, "")]
  # make_env_cfg -> env term (train / eval, curriculum, robot default body / override)
  with mock.patch.object(env_cfg_mod, "base_tracking_cfg", tmm._fake_base_full):
    for mode in ("train", "eval"):
      for hn in ((), ("kongfan",)):
        e = env_cfg_mod.make_env_cfg(["a.npz"], 4, mode=mode, hard_names=hn)
        assert "root_ori" not in e.terminations
        for robot, body, want in (("g1", "", "pelvis"), ("e2", "", "base_link"), ("g1", "torso_link", "torso_link")):
          e = env_cfg_mod.make_env_cfg(["a.npz"], 4, mode=mode, robot=robot, hard_names=hn, root_ori_term=1.0, root_ori_body=body)
          term = e.terminations["root_ori"]
          assert term.func is bad_root_ori_full and term.params == {"command_name": "motion", "threshold": 1.0, "body_name": want}
          assert not getattr(term, "time_out", False)
          assert list(e.terminations)[-1] == "clip_end"


def _fake_env(terms: dict, tracked=TRACKED):
  cmd = types.SimpleNamespace(cfg=types.SimpleNamespace(body_names=tracked))
  return types.SimpleNamespace(cfg=types.SimpleNamespace(terminations=terms), command_manager=types.SimpleNamespace(get_term=lambda n: cmd))


def test_sim_interface_checks():
  ns = types.SimpleNamespace
  term = lambda thr, body: ns(params={"command_name": "motion", "threshold": thr, "body_name": body})  # noqa: E731
  f = SimInterface.root_ori_body_of
  assert f(_fake_env({}), HeadsCfg(), "pelvis") == ("pelvis", 0)
  assert f(_fake_env({}), HeadsCfg(), "base_link") == ("base_link", None)  # not tracked, off: no logging, no error
  assert f(_fake_env({"root_ori": term(1.0, "pelvis")}), HeadsCfg(root_ori_term=1.0), "pelvis") == ("pelvis", 0)
  assert f(_fake_env({"root_ori": term(0.5, "torso_link")}), HeadsCfg(root_ori_term=0.5, root_ori_body="torso_link"), "pelvis") == ("torso_link", R_ANCHOR)
  for env, h, root in (
    (_fake_env({}), HeadsCfg(root_ori_term=1.0), "pelvis"),  # heads on, env off
    (_fake_env({"root_ori": term(1.0, "pelvis")}), HeadsCfg(), "pelvis"),  # env on, heads off
    (_fake_env({"root_ori": term(0.9, "pelvis")}), HeadsCfg(root_ori_term=1.0), "pelvis"),  # threshold
    (_fake_env({"root_ori": term(1.0, "pelvis")}), HeadsCfg(root_ori_term=1.0), "base_link"),  # ROOT_BODY wrong for the model
    (_fake_env({"root_ori": term(1.0, "base_link")}), HeadsCfg(root_ori_term=1.0), "base_link"),  # not tracked
  ):
    try:
      f(env, h, root)
    except ValueError:
      continue
    raise AssertionError(f"must be rejected: {h} / {env.cfg.terminations} / root {root}")
  assert env_cfg_mod.ROOT_BODY == {"g1": "pelvis", "e2": "base_link"}


def test_resume_diffs():
  script = tmm.tsc_load_train()
  with tempfile.TemporaryDirectory() as tmp:
    args, _ = tmm._example_args(tmp)
    live = tmm._parse(script, tmp, args)
  ck_cfg = dataclasses.asdict(live)
  ck_cfg["heads"] = {k: v for k, v in ck_cfg["heads"].items() if not k.startswith("root_ori")}  # written by 55e0eaf
  t = object.__new__(Trainer)
  t.cfg = live
  assert Trainer._resume_diffs(t, ck_cfg) == ([], "same")
  t.cfg = dataclasses.replace(live, heads=dataclasses.replace(live.heads, root_ori_term=1.0, root_ori_term_temp=0.05))
  assert Trainer._resume_diffs(t, ck_cfg) == (["heads.root_ori_term = 0.0 in the checkpoint, 1.0 here"], "same")
  t.cfg = dataclasses.replace(live, heads=dataclasses.replace(live.heads, root_ori_term_temp=0.07))  # unused while off
  assert Trainer._resume_diffs(t, ck_cfg) == ([], "same")


# ---- launch tooling -----------------------------------------------------------------


def _launch(script: str, root: str, run: str, env: dict, args=("--foo", "1")) -> subprocess.CompletedProcess:
  e = {k: v for k, v in os.environ.items() if k not in ("CUDA_VISIBLE_DEVICES", "PORT", "RDZV_PORT", "MASTER_PORT", "NPROC", "DRY_RUN")}
  e.update(ROOT=root, VENV="/nonexistent_venv_for_test", **env)
  return subprocess.run(["bash", script, run, *args], env=e, capture_output=True, text=True, timeout=60)


def _cmd_of(out: str) -> list[str]:
  (line,) = [l for l in out.splitlines() if l.startswith("[launch] /nonexistent_venv_for_test/bin/torchrun")]
  return shlex.split(line[len("[launch] "):])


def test_launcher_ports_and_gpu_subsets():
  new = os.path.join(ROOT, "scripts", "launch_bj31_ddp.sh")
  with tempfile.TemporaryDirectory() as tmp:
    norm = lambda c, repo, root: [x.replace(repo, "<REPO>").replace(root, "<ROOT>") for x in c]  # noqa: E731
    # 55e0eaf's launcher (copied into a scripts/ dir of the same layout): its default command, from the log it writes.
    # VENV does not exist, so its detached torchrun fails at once; nothing runs.
    old_repo = os.path.join(tmp, "repo")
    os.makedirs(os.path.join(old_repo, "scripts"))
    old = os.path.join(old_repo, "scripts", "launch_bj31_ddp.sh")
    with open(old, "w") as f:
      f.write(subprocess.run(["git", "-C", ROOT, "show", "55e0eaf:scripts/launch_bj31_ddp.sh"], capture_output=True, text=True, check=True).stdout)
    r_old = os.path.join(tmp, "old_root")
    p = _launch(old, r_old, "runX", {})
    assert p.returncode == 0, p.stderr
    cmd_old = norm(_cmd_of(open(os.path.join(r_old, "runs", "runX.log")).read().splitlines()[1]), old_repo, r_old)
    assert cmd_old[1:4] == ["--standalone", "--nproc_per_node", "8"]

    def dry(run: str, env: dict) -> list[str]:
      r = os.path.join(tmp, f"dry_{run}")
      p = _launch(new, r, run, {**env, "DRY_RUN": "1"})
      assert p.returncode == 0, p.stderr
      assert not os.path.exists(os.path.join(r, "runs", f"{run}.log")), "a dry run writes no log"
      return norm(_cmd_of(p.stdout), ROOT, r), p.stdout

    # default (neither PORT nor CUDA_VISIBLE_DEVICES): the 55e0eaf command, dry and real (log header too)
    assert dry("runX", {})[0] == cmd_old
    r_new = os.path.join(tmp, "new_root")
    p = _launch(new, r_new, "runX", {})
    assert p.returncode == 0, p.stderr
    lines = open(os.path.join(r_new, "runs", "runX.log")).read().splitlines()
    assert lines[0].startswith("[launch] ") and " host " in lines[0]
    assert norm(_cmd_of(lines[1]), ROOT, r_new) == cmd_old, "default header and command as before"
    # two concurrent 4-GPU jobs: own GPUs, own rendezvous port (PORT or its aliases)
    base = dry("runA", {})[0]
    for gpus, port, alias in (("0,1,2,3", "29511", "PORT"), ("4,5,6,7", "29512", "MASTER_PORT"), ("4,5,6,7", "29513", "RDZV_PORT")):
      c, out = dry("runA", {"CUDA_VISIBLE_DEVICES": gpus, alias: port})
      want = [base[0], "--nnodes", "1", "--rdzv-backend", "c10d", "--rdzv-endpoint", f"localhost:{port}", "--rdzv-id", "runA",
              "--nproc_per_node", "4"] + base[4:]
      assert c == want, (c, want)
      assert f"CUDA_VISIBLE_DEVICES={gpus} NPROC=4 PORT={port}" in out
    # GPU subset without a port: standalone, NPROC = the number of GPUs listed (or fewer when NPROC says so)
    assert dry("runB", {"CUDA_VISIBLE_DEVICES": "2,3"})[0][1:4] == ["--standalone", "--nproc_per_node", "2"]
    assert dry("runB", {"CUDA_VISIBLE_DEVICES": "2,3,4", "NPROC": "2"})[0][1:4] == ["--standalone", "--nproc_per_node", "2"]
    assert dry("runB", {"PORT": "29600"})[0][1:12] == ["--nnodes", "1", "--rdzv-backend", "c10d", "--rdzv-endpoint", "localhost:29600",
                                                      "--rdzv-id", "runB", "--nproc_per_node", "8", "--log-dir"]
    for env in ({"CUDA_VISIBLE_DEVICES": "0,1", "NPROC": "4"}, {"PORT": "29x"}, {"PORT": "80"}):
      p = _launch(new, os.path.join(tmp, "bad"), "runC", {**env, "DRY_RUN": "1"})
      assert p.returncode == 1 and p.stderr.strip(), (env, p.stdout)


LIVE_CMD_TAIL = (  # /data2/supermimic/runs/g1v2_r15ws_ddp8.log line 2 (launched 2026-10-25 12:31:49, commit 55e0eaf), after train.py
  "--log_root /data2/supermimic/logs --cache_dir /data2/supermimic/cache --run_name g1v2_r15ws_ddp8 --num_threads 24 "
  "--motion_list /data2/supermimic/prep_1025/g1v2_r15_255008_bj31.txt --robot g1 --robot_variant eval+evalcontact --ee_term_z 0.25 "
  "--source_weights noetix_kungfu=10\\,noetix_kungfu_dance_show=15\\,sportsmeeting=10\\,lafan1=15\\,g1_all=10\\,mixamo_cn=2\\,virtual_point=3 "
  "--name_weights tengkong=200\\,kongfan=200\\,shoufan=200\\,erqijiao=200\\,danpaijiao=200\\,xuanfeng=200\\,waibailian=100\\,lihetui=100\\,"
  "saotang=100\\,ceti=100\\,houti=100\\,zhengti=100 --sampler_clamp_returns --num_envs 4096 --rollouts 4096 --H 48 --wm_batches 10 "
  "--wm_windows 2048 --wm_k 8 --lr 8.9252e-4 --lr_decay_iters 250000 --iterations 250000 --seed 0 --wd_drop 0.2 --wd_reward_floor -1.0 "
  "--wd_signal imagined_reward_alive --num_workers 64 --eval_clip_names /data2/supermimic/lists/eval_1024_g1atlasv1.txt "
  "--eval_interval 2000 --eval_envs 1024 --es_eval_fail_ratio 1.5 --es_patience 2 --ckpt_interval 2000 --log_interval 50 "
  "--param_check_interval 1000 --resume /data2/supermimic/warmstart_g1v2_clamp_r15_model_136000.pt"
)
LIVE_LR_148000 = 2.101693e-05  # opt_pi / opt_wm / opt_v param_groups[0]["lr"] in model_148000.pt (read 2026-10-25)


def _oriterm_args(tmp: str, live_vars: dict) -> tuple[list[str], list[str], list[str], dict, str]:
  """(ARGS, ARGS_T, ARGS_C, variables, script text) of scripts/launch_bj31_oriterm.example.sh; LIST / EVAL_NAMES -> the
  temp files of tmm._example_args (the same ones the live ARGS are parsed with)."""
  text = open(os.path.join(ROOT, "scripts", "launch_bj31_oriterm.example.sh"), encoding="utf-8").read()
  block = text.split("# >>> ARGS", 1)[1].split("# <<< ARGS", 1)[0]
  vars_ = dict(re.findall(r'^([A-Z0-9_]+)=([^\s"$(][^\s]*)(?:\s+#.*)?$', text, flags=re.M))
  for k in ("ITERS", "LR", "STAGGER_S"):
    vars_[k] = re.search(rf'^{k}="\$\{{{k}:-([0-9.e+-]+)\}}"', text, flags=re.M).group(1)
  real = dict(vars_)
  vars_["LIST"], vars_["EVAL_NAMES"] = live_vars["LIST"], live_vars["EVAL_NAMES"]

  def arr(name: str) -> list[str]:
    body = re.search(rf"^{name}=\((.*?)\)$", block, flags=re.M | re.S).group(1)
    for k, v in vars_.items():
      body = body.replace(f'"${k}"', shlex.quote(v)).replace(f"${k}", v)
    assert "$" not in body, body
    return shlex.split(body.replace("\\\n", " "))

  return arr("ARGS"), arr("ARGS_T"), arr("ARGS_C"), real, text


def test_oriterm_example():
  script = tmm.tsc_load_train()
  with tempfile.TemporaryDirectory() as tmp:
    live_args, lv = tmm._example_args(tmp)  # the r15ws example's ARGS = the live run's (checked against its log below)
    args, args_t, args_c, v, text = _oriterm_args(tmp, lv)
    cL = tmm._parse(script, tmp, live_args)
    cC = tmm._parse(script, tmp, args + args_c)
    cT = tmm._parse(script, tmp, args + args_t)
  # the live run's command line == the r15ws example's ARGS with its real paths (+ the launcher's own flags)
  tail = shlex.split(LIVE_CMD_TAIL)
  assert tail[:8] == ["--log_root", "/data2/supermimic/logs", "--cache_dir", "/data2/supermimic/cache", "--run_name", "g1v2_r15ws_ddp8",
                      "--num_threads", "24"]
  r15ws = open(os.path.join(ROOT, "scripts", "launch_bj31_r15ws.example.sh"), encoding="utf-8").read()
  rv = dict(re.findall(r'^([A-Z0-9_]+)=([^\s"$(][^\s]*)(?:\s+#.*)?$', r15ws, flags=re.M))
  sub = [x.replace(lv["LIST"], rv["LIST"]).replace(lv["EVAL_NAMES"], rv["EVAL_NAMES"]) for x in live_args]
  assert tail[8:] == sub, [(a, b) for a, b in zip(tail[8:], sub) if a != b]
  # same data / recipe as the live run: the oriterm ARGS name the live files, and parse to the live cfg but for --iterations
  assert (v["LIST"], v["EVAL_NAMES"], v["SOURCES"], v["NAMES"]) == (rv["LIST"], rv["EVAL_NAMES"], rv["SOURCES"], rv["NAMES"])
  assert v["CKPT"] == "/data2/supermimic/warmstart_r15ws_model_148000.pt" and args[-2:] == ["--resume", v["CKPT"]]
  assert v["SRC_CKPT"] == "/data2/supermimic/logs/2026-10-25_12-31-57_g1v2_r15ws_ddp8/model_148000.pt"
  assert re.fullmatch(r"[0-9a-f]{32}", v["CKPT_MD5"]) and int(v["STAGGER_S"]) >= 300
  dl, dc, dt = (dataclasses.asdict(c) for c in (cL, cC, cT))
  for d in (dl, dc, dt):
    d.pop("log_dir")
  diff_c = {k for k in dl if dl[k] != dc[k]}
  assert diff_c == {"iterations"} and dc["iterations"] == 160000 and dl["iterations"] == 250000, diff_c
  diff_t = {k for k in dc if dc[k] != dt[k]}
  assert diff_t == {"heads"} and {k for k in dc["heads"] if dc["heads"][k] != dt["heads"][k]} == {"root_ori_term"}
  assert (cT.heads.root_ori_term, cT.heads.root_ori_term_temp, cC.heads.root_ori_term) == (1.0, 0.05, 0.0)
  # GPUs / ports / ranks of the two arms
  assert 'launch_arm "$RUN_T" 0,1,2,3 29511 "${ARGS_T[@]}"' in text and 'launch_arm "$RUN_C" 4,5,6,7 29512' in text
  assert 'RUN_T="${RUN_T:-g1v2_ro10_148k_T4}"' in text and 'RUN_C="${RUN_C:-g1v2_ro0_148k_C4}"' in text
  assert 'NPROC=4' in text and 'sleep "$STAGGER_S"' in text
  # the restored LR: resume model_148000 (lr_scale 0.0625) with --lr 8.9252e-4 -> the live run's LR at 148000
  lam = trainer_mod.cosine_lambda(250_000, 0.03)
  assert cC.policy_lr == cC.critic_lr == cC.wm.lr == 8.9252e-4
  assert abs(cC.policy_lr * 0.0625 * lam(148000) - LIVE_LR_148000) < 5e-12
  with tempfile.TemporaryDirectory() as tmp:
    t = tsc.make_trainer(tmp, 64, True)
    t.cfg = cC
    t.opt_wm, t.opt_pi, t.opt_v = (torch.optim.Adam(m.parameters(), lr=cC.policy_lr) for m in (t.wm, t.policy, t.critic))
    t.sched = [torch.optim.lr_scheduler.LambdaLR(o, lam) for o in (t.opt_wm, t.opt_pi, t.opt_v)]
    t._base_lrs0 = [list(s.base_lrs) for s in t.sched]
    src = tsc.make_trainer(tmp, 64, True)
    for o in (src.opt_wm, src.opt_pi, src.opt_v):
      for gr in o.param_groups:
        gr["initial_lr"], gr["lr"] = 2e-4, LIVE_LR_148000
    ck_cfg = dataclasses.asdict(cL)
    ck_cfg["heads"] = {k: x for k, x in ck_cfg["heads"].items() if not k.startswith("root_ori")}
    ck = os.path.join(tmp, "model_148000.pt")
    tmm._save_ck(ck, src, torch.rand(64) * 9.0, torch.ones(64), ck_cfg, it=148000, lr_scale=0.0625)
    for c, warn in ((cC, []), (cT, ["heads.root_ori_term = 0.0 in the checkpoint, 1.0 here"])):
      t.cfg, t.prints, t.lr_scale = c, [], 1.0
      t._resume_from = None
      Trainer.load(t, ck)
      for o in (t.opt_wm, t.opt_pi, t.opt_v):
        assert abs(o.param_groups[0]["lr"] - LIVE_LR_148000) < 1e-11, o.param_groups[0]["lr"]
      assert [x.split("setup: ", 1)[1] for x in t.prints if "WARNING resume" in x] == warn
      (res,) = [x for x in t.prints if "resumed from" in x]
      assert "policy lr 2.102e-05 = lr x lr_scale 0.0625 x cosine 0.3768" in res, res
  print(f"  arms: live ARGS + --iterations 160000 (+ --root_ori_term 1.0 on T); restored LR {LIVE_LR_148000:.6e}")
  # the script itself: bash syntax, and it refuses busy GPUs / a short stagger before launching anything
  assert subprocess.run(["bash", "-n", os.path.join(ROOT, "scripts", "launch_bj31_oriterm.example.sh")]).returncode == 0


if __name__ == "__main__":
  if tsc.STUBBED:
    print(f"  (stubbed via sys.modules: {', '.join(tsc.STUBBED)})")
  for name, fn in list(globals().items()):
    if name.startswith("test_") and callable(fn):
      fn()
      print(f"ok  {name}")
