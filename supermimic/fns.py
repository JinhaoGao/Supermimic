"""Pure per-step functions shared by the trainer and the evaluator, optionally
wrapped in ``torch.compile`` (the FK + heads + MLP chain is launch-bound in
eager mode; inductor fuses it into a handful of kernels).
"""

from __future__ import annotations

import torch

from .heads import AnalyticHeads


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

  def _act(self, obs_hist, kin, frame, clip_end):
    goal = self.heads.goal(kin, frame, clip_end)
    a = self.policy(obs_hist, goal)
    return goal, a

  def _post(self, next_state, kin, a_exec, prev_action, frame):
    kin_next = self.heads.kin(next_state)
    r = self.heads.reward(kin, kin_next, next_state, a_exec, prev_action, frame)
    hard, _ = self.heads.termination(kin_next, frame)
    return r, hard

  # ---- imagination (grad) --------------------------------------------------

  def critic_inputs(self, s, goal):
    feat = self.wm.features(s)
    fd = self.heads.feat_dim
    feat_n = ((feat - self.wm.in_norm.mean[:fd]) / self.wm.in_norm.std[:fd]).clamp(-10.0, 10.0)
    goal_n = self.policy.goal_norm.normalize(goal).clamp(-10.0, 10.0)
    return feat_n, goal_n

  def _imagine_step(self, s, obs_hist, prev_a, frame, clip_end, xi, bias, kin):
    heads = self.heads
    goal = heads.goal(kin, frame, clip_end)
    fn, gn = self.critic_inputs(s, goal)
    a = self.policy(obs_hist, goal)
    s_next = self.wm.step(s, a, xi)
    kin_next = heads.kin(s_next)
    r = heads.reward(kin, kin_next, s_next, a, prev_a, frame)
    _, d = heads.termination(kin_next, frame)
    frame_next = torch.minimum(frame + 1, clip_end - 1)
    obs_next = heads.proprio(s_next, a, bias, True)
    obs_hist_next = torch.cat([obs_hist[:, 1:], obs_next[:, None]], dim=1)
    return s_next, obs_hist_next, a, frame_next, kin_next, r, d, fn, gn

  def _wm_loss(self, s0, actions, next_states, xi, mask):
    return self.wm.window_loss(s0, actions, next_states, xi, mask)
