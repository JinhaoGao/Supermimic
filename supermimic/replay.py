"""Per-environment circular replay buffer on the GPU.

Slot layout per env: time-ordered ring of capacity C. ``add`` writes one step
for all envs. Windows and imagination starts are sampled with validity masks
that reject (a) unwritten slots, (b) windows that cross the write pointer and
(c) windows that cross an episode boundary inside the window.
"""

from __future__ import annotations

import torch


class ReplayBuffer:
  def __init__(self, num_envs: int, capacity: int, state_dim: int, action_dim: int, xi_dim: int, device: str):
    E, C = num_envs, capacity
    self.E, self.C, self.device = E, C, device
    z = lambda *shape, dtype=torch.float32: torch.zeros(*shape, dtype=dtype, device=device)  # noqa: E731
    self.state = z(E, C, state_dim)
    self.next_state = z(E, C, state_dim)
    self.action = z(E, C, action_dim)
    self.prev_action = z(E, C, action_dim)
    self.xi = z(E, C, xi_dim)
    self.reward = z(E, C)
    self.frame = z(E, C, dtype=torch.long)  # reference frame index used for obs/reward
    self.clip_end = z(E, C, dtype=torch.long)
    self.ep_step = z(E, C, dtype=torch.long)  # steps since episode start (0 = first obs)
    self.term = z(E, C, dtype=torch.bool)  # failure termination after this step
    self.done = z(E, C, dtype=torch.bool)  # term or time-out after this step
    self.ptr = 0
    self.size = 0
    self._env_ar = torch.arange(E, device=device)

  def add(self, **kw):
    p = self.ptr
    for k, v in kw.items():
      getattr(self, k)[:, p] = v
    self.ptr = (p + 1) % self.C
    self.size = min(self.size + 1, self.C)

  def _age(self, slot: torch.Tensor) -> torch.Tensor:
    """0 = newest written slot."""
    return (self.ptr - 1 - slot) % self.C

  def sample_windows(self, n: int, k: int):
    """Sample n windows of k consecutive transitions.

    Returns dict with s0 [n,D], actions [n,k,A], next_states [n,k,D], xi [n,X],
    frame [n,k], prev_action [n,A], term [n,k], done [n,k] and mask [n] (bool).
    """
    env = torch.randint(0, self.E, (n,), device=self.device)
    # Start slots must be at least k-1 steps older than the newest slot.
    age = torch.randint(k - 1, max(self.size, k), (n,), device=self.device)
    start = (self.ptr - 1 - age) % self.C
    offs = torch.arange(k, device=self.device)
    slots = (start[:, None] + offs[None, :]) % self.C  # [n,k]
    env_k = env[:, None].expand(n, k)
    done = self.done[env_k, slots]
    # No episode end inside the window except possibly at the last step.
    mask = ~done[:, :-1].any(dim=1) if k > 1 else torch.ones(n, dtype=torch.bool, device=self.device)
    mask &= age < self.size
    return dict(
      s0=self.state[env, start],
      actions=self.action[env_k, slots],
      next_states=self.next_state[env_k, slots],
      xi=self.xi[env, start],
      frame=self.frame[env_k, slots],
      prev_action=self.prev_action[env, start],
      term=self.term[env_k, slots],
      done=done,
      reward=self.reward[env_k, slots],
      mask=mask,
    )

  def sample_starts(self, n: int, hist: int):
    """Sample n imagination start states with their observation history slots.

    Returns dict with state [n,D], xi, prev_action, frame, clip_end, and
    hist_states [n,hist,D] / hist_prev_action [n,hist,A] / hist_xi where index
    hist-1 is the start itself and earlier indices are earlier steps of the same
    episode (clamped at the episode start, repeating the first step).
    """
    env = torch.randint(0, self.E, (n,), device=self.device)
    lo = min(hist - 1, max(self.size - 1, 0))
    age = torch.randint(lo, max(self.size, lo + 1), (n,), device=self.device)
    slot = (self.ptr - 1 - age) % self.C
    ep = self.ep_step[env, slot]
    back = torch.arange(hist - 1, -1, -1, device=self.device)  # hist-1 ... 0
    back = torch.minimum(back[None, :], ep[:, None])  # clamp at episode start
    hslots = (slot[:, None] - back) % self.C  # [n,hist]
    env_h = env[:, None].expand(n, hist)
    return dict(
      state=self.state[env, slot],
      xi=self.xi[env, slot],
      prev_action=self.prev_action[env, slot],
      frame=self.frame[env, slot],
      clip_end=self.clip_end[env, slot],
      hist_states=self.state[env_h, hslots],
      hist_prev_action=self.prev_action[env_h, hslots],
    )
