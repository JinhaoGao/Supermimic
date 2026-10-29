"""StepFns for a human reference: same compiled per-step functions, human reference tensors in the positional slots.

Positional reference arguments (so the trainer / evaluator call sites stay as they are):
  act:   *goal_refs   = (h_feat [B,K,138], h_yaw [B,K]) (+ ee_mode: h_task_pos [B,K,L,3], h_task_rot6d [B,K,L,6], cond [B,C])
  post:  *current_ref = heads.step_fields at the target frame, then term_scale (must be None: no curriculum here)
  pack:  *p           = heads.pack_fields: PACK_FIELDS windows [N,L,...] (+ ee_mode: EE windows, cond [N,C] of the
                        rollout start; its drift is advanced by the local frame index, i.e. the steps since the start)
Trailing positional arguments past the fields (term_scale) are ignored.
"""

from __future__ import annotations

import torch

from ..fns import StepFns
from ..heads import window_frames
from .heads import EE_STEP_FIELDS, STEP_FIELDS


class HumanStepFns(StepFns):
  def _act(self, obs_hist, kin, *goal_refs):
    goal = self.heads.goal(kin, *goal_refs)
    return goal, self.policy(obs_hist, goal)

  def _post(self, next_state, kin, a_exec, prev_action, *ref_args):
    fields = self.heads.step_fields
    ref = dict(zip(fields, ref_args[: len(fields)]))
    kin_next = self.heads.kin(next_state)
    r = self.heads.reward(kin, kin_next, next_state, a_exec, prev_action, ref)
    hard, _ = self.heads.termination(kin_next, next_state, ref)
    return r, hard

  def critic_err(self, kin, f_loc, *_):
    return None

  def _pack(self, p) -> dict:
    fields = self.heads.pack_fields
    return dict(zip(fields, p[: len(fields)]))

  def goal_from_pack(self, kin, f_loc, clip_end_loc, *p):
    P = self._pack(p)
    heads = self.heads
    ar = torch.arange(f_loc.shape[0], device=f_loc.device)[:, None]
    idx = window_frames(f_loc, clip_end_loc, heads.cfg.K)
    g = [P["h_feat"][ar, idx], P["h_yaw"][ar, idx]]
    if heads.ee_on:
      g += [P["h_task_pos"][ar, idx], P["h_task_rot6d"][ar, idx], heads.cond_at(P["cond"], f_loc)]
    return heads.goal(kin, *g)

  def _imagine_step(self, s, obs_hist, prev_a, f_loc, clip_end_loc, xi, bias, kin, *p):
    heads = self.heads
    P = self._pack(p)
    ar = torch.arange(s.shape[0], device=s.device)
    goal = self.goal_from_pack(kin, f_loc, clip_end_loc, *p)
    fn, gn = self.critic_inputs(s, goal)
    a = self.policy(obs_hist, goal)
    s_next = self.wm.step(s, a, xi)
    kin_next = heads.kin(s_next)
    ref = {k: P[k][ar, f_loc] for k in STEP_FIELDS + (EE_STEP_FIELDS if heads.ee_on else ())}
    if heads.ee_on:
      ref["cond"] = heads.cond_at(P["cond"], f_loc)
    r = heads.reward(kin, kin_next, s_next, a, prev_a, ref)
    _, d = heads.termination(kin_next, s_next, ref)
    f_next = torch.minimum(f_loc + 1, clip_end_loc - 1)
    obs_next = heads.proprio(s_next, a, bias, True)
    obs_hist_next = torch.cat([obs_hist[:, 1:], obs_next[:, None]], dim=1)
    return s_next, obs_hist_next, a, f_next, kin_next, r, d, fn, gn
