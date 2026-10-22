"""Pure per-step functions shared by the trainer and the evaluator, optionally
wrapped in ``torch.compile`` (the FK + heads + MLP chain is launch-bound in
eager mode; inductor fuses it into a handful of kernels).

Reference data is passed explicitly:
  * simulator side: tensors gathered for the current frames / goal windows,
  * imagination side: a window "pack" [N, L, ...] per rollout and per-rollout
    local frame indices into it.
"""

from __future__ import annotations

import torch

from .heads import AnalyticHeads, window_frames


class StepFns:
  def __init__(self, heads: AnalyticHeads, policy, wm=None, critic=None, compile_: bool = True, mode: str | None = None):
    self.heads, self.policy, self.wm, self.critic = heads, policy, wm, critic
    kw = dict(dynamic=False)
    if mode:
      kw["mode"] = mode
    wrap = (lambda f: torch.compile(f, **kw)) if compile_ else (lambda f: f)
    self.refresh = wrap(self._refresh)
    self.act = wrap(self._act)
    self.post = wrap(self._post)
    self.imagine_step = wrap(self._imagine_step)
    self.wm_loss = wrap(self._wm_loss)

  # ---- simulator side (no grad) -------------------------------------------

  def _refresh(self, state, prev_action, bias, noise: bool):
    kin = self.heads.kin(state)
    obs = self.heads.proprio(state, prev_action, bias, noise)
    return kin, obs

  def _act(self, obs_hist, kin, ref_q, ref_qd, ref_ap, ref_aq):
    goal = self.heads.goal(kin, ref_q, ref_qd, ref_ap, ref_aq)
    a = self.policy(obs_hist, goal)
    return goal, a

  def _post(self, next_state, kin, a_exec, prev_action, ref_bp, ref_bq, ref_blv, ref_bav):
    kin_next = self.heads.kin(next_state)
    ref = dict(body_pos=ref_bp, body_quat=ref_bq, body_lin_vel=ref_blv, body_ang_vel=ref_bav)
    r = self.heads.reward(kin, kin_next, next_state, a_exec, prev_action, ref)
    a_idx = self.heads.anchor
    hard, _ = self.heads.termination(kin_next, ref_bp[:, a_idx], ref_bq[:, a_idx], ref_bp)
    return r, hard

  # ---- imagination (grad) --------------------------------------------------

  def critic_inputs(self, s, goal):
    feat = self.wm.features(s)
    fd = self.heads.feat_dim
    feat_n = ((feat - self.wm.in_norm.mean[:fd]) / self.wm.in_norm.std[:fd]).clamp(-10.0, 10.0)
    goal_n = self.policy.goal_norm.normalize(goal).clamp(-10.0, 10.0)
    return feat_n, goal_n

  def goal_from_pack(self, kin, f_loc, clip_end_loc, p_q, p_qd, p_bp, p_bq):
    N = f_loc.shape[0]
    ar = torch.arange(N, device=f_loc.device)[:, None]
    idx = window_frames(f_loc, clip_end_loc, self.heads.cfg.K)  # [N,K]
    a_idx = self.heads.anchor
    return self.heads.goal(kin, p_q[ar, idx], p_qd[ar, idx], p_bp[ar, idx, a_idx], p_bq[ar, idx, a_idx])

  def _imagine_step(self, s, obs_hist, prev_a, f_loc, clip_end_loc, xi, bias, kin, p_q, p_qd, p_bp, p_bq, p_blv, p_bav):
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
    _, d = heads.termination(kin_next, ref["body_pos"][:, a_idx], ref["body_quat"][:, a_idx], ref["body_pos"])
    f_next = torch.minimum(f_loc + 1, clip_end_loc - 1)
    obs_next = heads.proprio(s_next, a, bias, True)
    obs_hist_next = torch.cat([obs_hist[:, 1:], obs_next[:, None]], dim=1)
    return s_next, obs_hist_next, a, f_next, kin_next, r, d, fn, gn

  def _wm_loss(self, s0, actions, next_states, xi, mask):
    return self.wm.window_loss(s0, actions, next_states, xi, mask)
