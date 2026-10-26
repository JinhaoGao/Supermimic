"""Benchmark evaluation (Appendix F.1): clips played once from their first
frame, friction 1.0, no DR / noise, deterministic policy. Reports the
normalized return (undiscounted reward sum / max attainable) and the number
of failed clips (training termination triggered before the clip end).

Envs cycle through the evaluation clip list (all clips, or a fixed random
subset for large libraries); the in-RAM library is shared with the training
env through the registry key.
"""

from __future__ import annotations

import torch

from .env_cfg import make_env_cfg
from .fns import StepFns
from .heads import HeadsCfg
from .sim_interface import ObsHistory, SimInterface


class Evaluator:
  def __init__(
    self,
    motion_files: list[str],
    heads_cfg: HeadsCfg,
    device: str,
    library_key: str = "",
    eval_clips: tuple[int, ...] = (),
    num_clips_total: int | None = None,
    max_envs: int = 1024,
    compile_: bool = True,
    cache_dir: str = "",
    robot: str = "g1",
    robot_variant: str = "",
  ):
    from mjlab.envs import ManagerBasedRlEnv

    n_total = num_clips_total or len(motion_files)
    self.eval_clips = tuple(eval_clips) if eval_clips else tuple(range(n_total))
    self.num_clips = len(self.eval_clips)
    self.num_envs = min(self.num_clips, max_envs)
    env_cfg = make_env_cfg(
      motion_files, self.num_envs, mode="eval", robot=robot, library_key=library_key, eval_clips=self.eval_clips, cache_dir=cache_dir,
      ee_term_z=heads_cfg.ee_term_z, robot_variant=robot_variant, root_ori_term=heads_cfg.root_ori_term,
      root_z_term=heads_cfg.root_z_term,
    )
    self.env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    self.sim = SimInterface(self.env, heads_cfg)
    self.heads = self.sim.heads
    self.cfg = heads_cfg
    self.device = device
    self.lib = self.sim.lib
    self.cmd = self.sim.cmd
    self.compile_ = compile_
    self._fns: StepFns | None = None
    self._fns_policy = None
    self.clip_ids = torch.tensor(self.eval_clips, dtype=torch.long, device=device)

  def _get_fns(self, policy) -> StepFns:
    if self._fns is None or self._fns_policy is not policy:
      self._fns = StepFns(self.heads, policy, compile_=self.compile_)
      self._fns_policy = policy
    return self._fns

  @torch.no_grad()
  def run(self, policy, max_steps: int | None = None) -> dict:
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
      refs = sim.goal_refs(frame, clip_end)
      ref_cur = sim.current_ref()
      _, a = fns.act(hist.buf, kin, *refs)
      env.step(a)
      next_state = sim.state()
      r, _ = fns.post(next_state, kin, a, prev_action, *ref_cur)
      terminated = env.reset_terminated.bool()
      timeouts = env.reset_time_outs.bool()
      act = env_active
      sum_r.index_add_(0, env_pos[act], r[act])
      fail_now = terminated & act
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
        "clips_done": float(done_clip.sum()),
      },
      "per_clip": per_clip,
      "failed_clips": [names[i] for i in range(n) if bool(failed[i])][:200],
    }
