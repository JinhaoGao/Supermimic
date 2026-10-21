"""Benchmark evaluation (Appendix F.1): every clip once from its first frame,
friction 1.0, no DR / noise, deterministic policy. Reports the normalized
return (undiscounted reward sum / max attainable) and the number of failed
clips (training termination triggered before the clip end).
"""

from __future__ import annotations

import torch

from .env_cfg import make_env_cfg
from .heads import HeadsCfg
from .sim_interface import ObsHistory, SimInterface


class Evaluator:
  def __init__(self, motion_files: list[str], heads_cfg: HeadsCfg, device: str, num_envs: int | None = None):
    from mjlab.envs import ManagerBasedRlEnv

    self.num_clips = len(motion_files)
    self.num_envs = num_envs or self.num_clips
    env_cfg = make_env_cfg(motion_files, self.num_envs, mode="eval")
    self.env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    self.sim = SimInterface(self.env, heads_cfg)
    self.heads = self.sim.heads
    self.cfg = heads_cfg
    self.device = device
    self.lib = self.sim.lib
    self.cmd = self.sim.cmd

  @torch.no_grad()
  def run(self, policy, max_steps: int | None = None) -> dict:
    env, heads, sim, cmd = self.env, self.heads, self.sim, self.cmd
    E = self.num_envs
    env.reset()
    state = sim.state()
    xi = sim.xi()
    bias = sim.bias_from_xi(xi, sim.nj)
    frame = cmd.time_steps.clone()
    clip_end = cmd.clip_end_env.clone()
    clip = self.lib.clip_of_frame[frame]
    prev_action = torch.zeros(E, sim.nj, device=self.device)
    kin = heads.kin(state)
    hist = ObsHistory(E, self.cfg.hist, heads.proprio_dim, self.device)
    hist.reset(torch.arange(E, device=self.device), heads.proprio(state, prev_action, bias, noise=False))
    sum_r = torch.zeros(E, device=self.device)
    steps_done = torch.zeros(E, device=self.device)
    finished = torch.zeros(E, dtype=torch.bool, device=self.device)
    failed = torch.zeros(E, dtype=torch.bool, device=self.device)
    clip_len = (self.lib.clip_end[clip] - self.lib.clip_start[clip]).float()
    limit = max_steps or int(clip_len.max().item()) + 5
    policy.eval()
    for _ in range(limit):
      goal = heads.goal(kin, frame, clip_end)
      a = policy(hist.buf, goal)
      env.step(a)
      next_state = sim.state()
      kin_next = heads.kin(next_state)
      r = heads.reward(kin, kin_next, next_state, a, prev_action, frame)
      terminated = env.reset_terminated.bool()
      timeouts = env.reset_time_outs.bool()
      active = ~finished
      sum_r += r * active
      steps_done += active.float()
      failed |= terminated & active
      finished |= (terminated | timeouts)
      done = terminated | timeouts
      if finished.all():
        break
      if done.any():
        ids = done.nonzero(as_tuple=False).squeeze(-1)
        env.reset(env_ids=ids)
        state = sim.state()
        prev_action = torch.where(done[:, None], torch.zeros_like(a), a)
      else:
        state = next_state
        prev_action = a
      frame = cmd.time_steps.clone()
      clip_end = cmd.clip_end_env.clone()
      kin = heads.kin(state)
      obs = heads.proprio(state, prev_action, bias, noise=False)
      hist.push(obs)
      if done.any():
        hist.reset(ids, obs)
    policy.train()
    rmax = self.cfg.reward_max_per_step
    norm_ret = sum_r / ((clip_len - 1.0) * rmax)
    n = self.num_clips
    per_clip = {self.lib.clips[i].name: float(norm_ret[i]) for i in range(n)}
    return {
      "num_clips": n,
      "scalars": {
        "return_mean": float(norm_ret[:n].mean()),
        "return_min": float(norm_ret[:n].min()),
        "failed": float(failed[:n].sum()),
        "steps_mean": float(steps_done[:n].mean()),
      },
      "per_clip": per_clip,
      "failed_clips": [self.lib.clips[i].name for i in range(n) if bool(failed[i])],
    }
