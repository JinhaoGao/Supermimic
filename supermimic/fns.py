"""Pure per-step functions shared by the trainer and the evaluator, optionally
wrapped in ``torch.compile`` (the FK + heads + MLP chain is launch-bound in
eager mode; inductor fuses it into a handful of kernels).

Reference data is passed explicitly:
  * simulator side: tensors gathered for the current frames / goal windows,
  * imagination side: a window "pack" [N, L, ...] per rollout and per-rollout
    local frame indices into it.

term_scale (post / imagine_step, optional): per-sample termination multiplier of the hard-clip curriculum
(heads.termination ``scale``); the trainer passes None when the curriculum is off (the code path before it).
The root-orientation termination (heads.termination ``ref_rq``, HeadsCfg.root_ori_term) takes the reference pelvis
quaternion from the same target-frame tracked-body quaternions (simulator: ref_bq; imagination: the pack's body_quat).
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

  def _ref_root(self, ref_bq):
    """Reference quaternion [B,4] of the root-orientation body (pelvis) from the tracked-body quaternions [B,Bt,4] of
    the target frame, for heads.termination's root_ori_term; None when that term is off (the graph before it)."""
    h = self.heads
    return ref_bq[:, h.root_ori] if h.cfg.root_ori_on else None

  # ---- simulator side (no grad) -------------------------------------------

  def _refresh(self, state, prev_action, bias, noise: bool):
    kin = self.heads.kin(state)
    obs = self.heads.proprio(state, prev_action, bias, noise)
    return kin, obs

  def _act(self, obs_hist, kin, ref_q, ref_qd, ref_ap, ref_aq, ref_av=None, ref_aw=None):
    goal = self.heads.goal(kin, ref_q, ref_qd, ref_ap, ref_aq, ref_av, ref_aw)
    a = self.policy(obs_hist, goal)
    return goal, a

  def _post(self, next_state, kin, a_exec, prev_action, ref_bp, ref_bq, ref_blv, ref_bav, term_scale=None):
    kin_next = self.heads.kin(next_state)
    ref = dict(body_pos=ref_bp, body_quat=ref_bq, body_lin_vel=ref_blv, body_ang_vel=ref_bav)
    r = self.heads.reward(kin, kin_next, next_state, a_exec, prev_action, ref)
    a_idx = self.heads.anchor
    hard, _ = self.heads.termination(kin_next, ref_bp[:, a_idx], ref_bq[:, a_idx], ref_bp, term_scale, ref_rq=self._ref_root(ref_bq))
    return r, hard

  # ---- imagination (grad) --------------------------------------------------

  def critic_inputs(self, s, goal, err=None):
    """Normalized critic inputs; err (critic_errors) is appended to the state features."""
    feat = self.wm.features(s)
    fd = self.wm.feat_dim
    c = self.policy.input_clip
    feat_n = ((feat - self.wm.in_norm.mean[:fd]) / self.wm.in_norm.std[:fd]).clamp(-c, c)
    if err is not None:
      feat_n = torch.cat([feat_n, self.critic.err_norm.normalize(err).clamp(-c, c)], dim=-1)
    goal_n = self.policy.goal_norm.normalize(goal).clamp(-c, c)
    return feat_n, goal_n

  def critic_err(self, kin, f_loc, p_bp, p_bq):
    """Critic error features against the reference at the rollouts' local frames (None without critic_errors)."""
    if not self.heads.critic_err_dim:
      return None
    ar = torch.arange(f_loc.shape[0], device=f_loc.device)
    return self.heads.critic_errors(kin, p_bp[ar, f_loc], p_bq[ar, f_loc])

  def goal_from_pack(self, kin, f_loc, clip_end_loc, p_q, p_qd, p_bp, p_bq, p_blv=None, p_bav=None):
    N = f_loc.shape[0]
    ar = torch.arange(N, device=f_loc.device)[:, None]
    idx = window_frames(f_loc, clip_end_loc, self.heads.cfg.K)  # [N,K]
    a_idx = self.heads.anchor
    vel = self.heads.cfg.goal_ref_root
    return self.heads.goal(kin, p_q[ar, idx], p_qd[ar, idx], p_bp[ar, idx, a_idx], p_bq[ar, idx, a_idx],
                           p_blv[ar, idx, a_idx] if vel else None, p_bav[ar, idx, a_idx] if vel else None)

  def _imagine_step(self, s, obs_hist, prev_a, f_loc, clip_end_loc, xi, bias, kin, p_q, p_qd, p_bp, p_bq, p_blv, p_bav, term_scale=None):
    heads = self.heads
    N = s.shape[0]
    ar = torch.arange(N, device=s.device)
    goal = self.goal_from_pack(kin, f_loc, clip_end_loc, p_q, p_qd, p_bp, p_bq, p_blv, p_bav)
    fn, gn = self.critic_inputs(s, goal, self.critic_err(kin, f_loc, p_bp, p_bq))
    a = self.policy(obs_hist, goal)
    s_next = self.wm.step(s, a, xi)
    kin_next = heads.kin(s_next)
    ref = dict(body_pos=p_bp[ar, f_loc], body_quat=p_bq[ar, f_loc], body_lin_vel=p_blv[ar, f_loc], body_ang_vel=p_bav[ar, f_loc])
    r = heads.reward(kin, kin_next, s_next, a, prev_a, ref)
    a_idx = heads.anchor
    _, d = heads.termination(kin_next, ref["body_pos"][:, a_idx], ref["body_quat"][:, a_idx], ref["body_pos"], term_scale,
                             ref_rq=self._ref_root(ref["body_quat"]))
    f_next = torch.minimum(f_loc + 1, clip_end_loc - 1)
    obs_next = heads.proprio(s_next, a, bias, True)
    obs_hist_next = torch.cat([obs_hist[:, 1:], obs_next[:, None]], dim=1)
    return s_next, obs_hist_next, a, f_next, kin_next, r, d, fn, gn

  def _wm_loss(self, s0, actions, next_states, xi, mask):
    return self.wm.window_loss(s0, actions, next_states, xi, mask)
