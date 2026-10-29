"""Benchmark evaluation (Appendix F.1): clips played once from their first
frame, friction 1.0, no DR / noise, deterministic policy. Reports the
normalized return (undiscounted reward sum / max attainable) and the number
of failed clips (training termination triggered before the clip end). With
``HeadsCfg.ee_termination`` the end-effector height test counts as a failure
too (the bj31 eval site's G1 / E2 official protocol: termination bodies
ankle_roll + wrist_yaw / elbow links, z error > 0.25); ``failed_ee`` counts
clips whose first failure was end-effector only (the site's "body_pos"
reason, checked after the root tests), ``failed_fall`` the rest.

Always (2026-10-25, logging only): ``failed_ori`` counts clips whose pelvis (root body) full rotation angle against the
reference exceeded 1.0 rad at some step of their episode -- the eval site's root_ori test, which the training
terminations do not contain unless ``HeadsCfg.root_ori_term`` is on (then the env terminates there and those clips are
in ``failed`` as well); ``failed_with_ori`` = failed or failed_ori, the count comparable across both settings.

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
from .watch import ORI_FAIL_THRESHOLD


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
    reference: str = "robot",
  ):
    from mjlab.envs import ManagerBasedRlEnv

    n_total = num_clips_total or len(motion_files)
    self.eval_clips = tuple(eval_clips) if eval_clips else tuple(range(n_total))
    self.num_clips = len(self.eval_clips)
    self.num_envs = min(self.num_clips, max_envs)
    env_cfg = make_env_cfg(
      motion_files, self.num_envs, mode="eval", robot=robot, library_key=library_key, eval_clips=self.eval_clips, cache_dir=cache_dir,
      ee_termination=heads_cfg.ee_termination, ee_threshold=heads_cfg.ee_term_z, robot_variant=robot_variant,
      root_ori_term=heads_cfg.root_ori_term, root_ori_body=heads_cfg.root_ori_body, root_z_term=heads_cfg.root_z_term,
      reference=reference, bone_map=getattr(heads_cfg, "bone_map", "g1"),
    )
    self.env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    self.human = reference == "human"
    if self.human:
      from .human.sim_interface import HumanSimInterface

      self.sim = HumanSimInterface(self.env, heads_cfg)
    else:
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
    # Optional extra per-step metrics (next_state [E,D], target frames [E], clip indices [E]) -> {name: [E]}, averaged per
    # clip over its live steps like the human-reference errors.
    self.metric_fn = None

  def _get_fns(self, policy) -> StepFns:
    if self._fns is None or self._fns_policy is not policy:
      cls = StepFns
      if self.human:
        from .human.fns import HumanStepFns as cls  # noqa: N813
      self._fns = cls(self.heads, policy, compile_=self.compile_)
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
    failed_ee = torch.zeros(n, dtype=torch.bool, device=self.device)  # first failure was end-effector only
    ori_over = torch.zeros(n, dtype=torch.bool, device=self.device)  # pelvis angle > 1.0 rad at some step (eval site)
    tm = env.termination_manager
    ee_on = "ee_body_pos" in tm.active_terms
    ro_on = "root_ori" in tm.active_terms
    # Map clip index -> position in eval list.
    pos_of_clip = torch.full((lib.num_clips,), -1, dtype=torch.long, device=self.device)
    pos_of_clip[self.clip_ids] = torch.arange(n, device=self.device)
    env_pos = pos_of_clip[cmd.env_clip]  # current eval position of each env
    env_active = torch.ones(E, dtype=torch.bool, device=self.device)
    assigned = min(E, n)  # clips handed out so far (the first E at reset)
    clip_len = (lib.clip_end[self.clip_ids] - lib.clip_start[self.clip_ids]).float()
    limit = max_steps or int(clip_len.sum().item() / E * 2) + int(clip_len.max().item()) + 10
    m_sum: dict[str, torch.Tensor] = {}  # human reference: per-clip sums of the tracking errors (heads.metrics)
    m_cnt = torch.zeros(n, device=self.device)
    policy.eval()
    for _ in range(limit):
      refs = sim.goal_refs(frame, clip_end)
      ref_cur = sim.current_ref()
      _, a = fns.act(hist.buf, kin, *refs)
      env.step(a)
      next_state = sim.state()
      r, _ = fns.post(next_state, kin, a, prev_action, *ref_cur)
      errs = None
      if self.human:  # errors of the live (not yet terminated) steps against the target frame
        errs = heads.metrics(heads.kin(next_state), next_state, dict(zip(heads.step_fields, ref_cur)))
      elif self.metric_fn is not None:  # e.g. human-side errors of a robot-reference policy (scripts/human/eval_paired.py)
        errs = self.metric_fn(next_state, frame, cmd.env_clip)
      if errs is not None:
        live = env_active & ~env.reset_terminated.bool()
        for k, v in errs.items():
          m_sum.setdefault(k, torch.zeros(n, device=self.device)).index_add_(0, env_pos[live], v[live])
        m_cnt.index_add_(0, env_pos[live], torch.ones_like(r)[live])
      ang = heads.root_ori_angle_state(next_state, ref_cur[1])
      terminated = env.reset_terminated.bool()
      timeouts = env.reset_time_outs.bool()
      act = env_active
      sum_r.index_add_(0, env_pos[act], r[act])
      if ang is not None:
        ori_over[env_pos[(ang > ORI_FAIL_THRESHOLD) & act]] = True
      fail_now = terminated & act
      if ee_on:
        root = tm.get_term("anchor_pos") | tm.get_term("anchor_ori")
        if ro_on:  # the eval site tests root_ori before the end effectors
          root = root | tm.get_term("root_ori")
        ee_only = tm.get_term("ee_body_pos") & ~root
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
        **({"failed_ori": float(ori_over.sum()), "failed_with_ori": float((failed | ori_over).sum())}
           if heads.root_ori is not None else {}),
        # human reference: clip-mean tracking errors, averaged over the clips (each clip's live steps)
        **{k: float((v / m_cnt.clamp(min=1.0)).mean()) for k, v in m_sum.items()},
      },
      "per_clip": per_clip,
      "per_clip_failed": {names[i]: bool(failed[i]) for i in range(n)},
      "per_clip_metrics": {k: {names[i]: float(v[i] / m_cnt[i].clamp(min=1.0)) for i in range(n)} for k, v in m_sum.items()},
      "failed_clips": [names[i] for i in range(n) if bool(failed[i])][:200],
    }
