"""StepFns for a human reference: same compiled per-step functions, human reference tensors in the positional slots.

Positional reference arguments (so the trainer / evaluator call sites stay as they are):
  act:   *goal_refs   = (h_feat [B,K,138], h_yaw [B,K])
  post:  *current_ref = STEP_FIELDS at the target frame, then term_scale (must be None: no curriculum here)
  pack:  *p           = PACK_FIELDS windows [N,L,...] (trainer: Trainer._pack_args)
"""

from __future__ import annotations

import torch

from ..fns import StepFns
from ..heads import window_frames
from .heads import STEP_FIELDS


class HumanStepFns(StepFns):
  def _act(self, obs_hist, kin, h_feat, h_yaw):
    goal = self.heads.goal(kin, h_feat, h_yaw)
    return goal, self.policy(obs_hist, goal)

  def _post(self, next_state, kin, a_exec, prev_action, h_dir, h_dir_local, h_root_vel, h_yaw, h_yaw_rate, h_height, h_rest,
            h_contact, h_foot_lift, term_scale=None):
    ref = dict(zip(STEP_FIELDS, (h_dir, h_dir_local, h_root_vel, h_yaw, h_yaw_rate, h_height, h_rest, h_contact, h_foot_lift)))
    kin_next = self.heads.kin(next_state)
    r = self.heads.reward(kin, kin_next, next_state, a_exec, prev_action, ref)
    hard, _ = self.heads.termination(kin_next, next_state, ref)
    return r, hard

  def critic_err(self, kin, f_loc, *_):
    return None

  def goal_from_pack(self, kin, f_loc, clip_end_loc, p_feat, p_dir, p_dir_local, p_root_vel, p_yaw, *_):
    ar = torch.arange(f_loc.shape[0], device=f_loc.device)[:, None]
    idx = window_frames(f_loc, clip_end_loc, self.heads.cfg.K)
    return self.heads.goal(kin, p_feat[ar, idx], p_yaw[ar, idx])

  def _imagine_step(self, s, obs_hist, prev_a, f_loc, clip_end_loc, xi, bias, kin, p_feat, p_dir, p_dir_local, p_root_vel, p_yaw,
                    p_yaw_rate, p_height, p_rest, p_contact, p_foot_lift, term_scale=None):
    heads = self.heads
    ar = torch.arange(s.shape[0], device=s.device)
    goal = self.goal_from_pack(kin, f_loc, clip_end_loc, p_feat, p_dir, p_dir_local, p_root_vel, p_yaw)
    fn, gn = self.critic_inputs(s, goal)
    a = self.policy(obs_hist, goal)
    s_next = self.wm.step(s, a, xi)
    kin_next = heads.kin(s_next)
    ref = dict(zip(STEP_FIELDS, (p[ar, f_loc] for p in (p_dir, p_dir_local, p_root_vel, p_yaw, p_yaw_rate, p_height, p_rest, p_contact, p_foot_lift))))
    r = heads.reward(kin, kin_next, s_next, a, prev_a, ref)
    _, d = heads.termination(kin_next, s_next, ref)
    f_next = torch.minimum(f_loc + 1, clip_end_loc - 1)
    obs_next = heads.proprio(s_next, a, bias, True)
    obs_hist_next = torch.cat([obs_hist[:, 1:], obs_next[:, None]], dim=1)
    return s_next, obs_hist_next, a, f_next, kin_next, r, d, fn, gn
