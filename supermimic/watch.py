"""Always-on failure logging of the simulator episodes (2026-10-25; no effect on training).

10-25 diagnosis (r15ws model_148000 on the team benchmark): 101 of 129 failures are the eval site's root_ori test --
the pelvis' full rotation angle (yaw included) against the reference > 1.0 rad -- which no training termination sees
(anchor = torso gravity tilt, torso / end-effector heights). Whether a run is drifting into that failure mode is not
visible in the training logs, so the trainer always counts, per log window:

  train/would_fail_ori         finished episodes whose pelvis angle exceeded 1.0 rad at any step (the episodes the eval
                               site would have failed by root_ori; with --root_ori_term 1.0 they are terminated there)
  train/ori_over_step_frac     simulator steps with pelvis angle > 1.0 rad
  train/watch_<g>_fail         failure rate of the finished episodes in clips of group g (name keywords, WATCH_GROUPS)
  train/watch_<g>_would_fail_ori   the same for the pelvis-angle flag
  train/watch_<g>_reset_share  share of the resets (new episodes) that went into clips of group g
  train/watch_<g>_episodes     finished episodes of group g (all ranks)

Groups (name keywords, case-insensitive substrings): floor work and spins, two kinds of clips with root_ori failures. The
spec's bare "lie" and "turn" are replaced by tokens that do not match ordinary clips (scan of the 255,008-clip r15ws list
on bj31, 10-25): "lie" hit 1,598 clips incl. fitness "Plie" squats and "willie" (a bare "lying" would add "Flying"); "turn" 11,342 (sonic
turn_walk / turn_jog, motiondecode direction transitions, valve turning) of the spin group's 11,647 (~97 % walking turns).
Now: floor 6,208 clips (sonic crawl / kneeling, motiondecode ground recovery, kimodo robot_lie_getup, ...), spin 446
(kung-fu zhuanti / xuanzi / xuanfeng / pingzhuan, pirouettes, ballet chain turns, ...). Of the 101 root_ori failures of
model_148000 (eval-site names) 31 fall in floor, 11 in spin, 59 in neither (kung-fu flips / sweeps, street dance) -- the
groups track these two kinds, not the failure list. Counts live on the device (no host sync per step) and are read once per log
interval; under data parallel the trainer sums them over ranks (``sums`` has the same keys on every rank) before
``scalars`` turns them into rates, so every rank computes the same values and rank 0 logs them. Nothing here draws
random numbers or feeds back into the learner.
"""

from __future__ import annotations

import torch

ORI_FAIL_THRESHOLD = 1.0  # rad: the eval site's root_ori threshold (evaluate_mujoco.py, --root-ori-threshold 1.0)
WATCH_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
  ("floor", ("floor", "ground_", "getup", "get_up", "_lie", "-lie", "_lying", "crawl", "kneel", "prone", "倒地", "躺", "跪", "匍匐")),
  ("spin", ("spin", "zhuanti", "xuanzi", "xuanfeng", "pingzhuan", "pirouette", "chain_turn", "转体", "端腿转")),
)


def name_mask(clip_names, keywords, device=None) -> torch.Tensor:
  """bool [num_clips]: clip name contains any keyword (case-insensitive substring; library_command.hard_clip_mask)."""
  kw = [str(k).lower() for k in keywords]
  return torch.tensor([any(k in str(n).lower() for k in kw) for n in clip_names], dtype=torch.bool, device=device)


class FailureWatch:
  def __init__(self, clip_names, num_envs: int, device, groups=WATCH_GROUPS, threshold: float = ORI_FAIL_THRESHOLD):
    self.groups = tuple(str(g) for g, _ in groups)
    self.keywords = tuple(tuple(k) for _, k in groups)
    self.threshold = float(threshold)
    self.device = device
    n_clips = len(clip_names)
    G = len(self.groups)
    if G:
      self.masks = torch.stack([name_mask(clip_names, kws, device) for kws in self.keywords])  # [G, num_clips]
    else:
      self.masks = torch.zeros(0, n_clips, dtype=torch.bool, device=device)
    self.over = torch.zeros(num_envs, dtype=torch.bool, device=device)  # per env: angle > threshold in this episode
    self.clear()

  def clear(self) -> None:
    """Start a new log window (the per-env episode flags carry over: they belong to running episodes)."""
    self.n_steps = 0
    self.n_episodes = 0
    self.n_resets = 0
    self.cnt = torch.zeros(2, dtype=torch.float64, device=self.device)  # steps over, episodes over
    self.grp = torch.zeros(len(self.groups), 4, dtype=torch.float64, device=self.device)  # resets, episodes, failed, over

  # ---- per step / episode / reset (device ops only) ----------------------------

  @torch.no_grad()
  def step(self, angle: torch.Tensor | None) -> None:
    """angle [E]: root-orientation error of every env at this step (heads.root_ori_angle_state; None: not tracked)."""
    if angle is None:
      return
    over = angle > self.threshold  # NaN (non-finite state) -> False
    self.over |= over
    self.n_steps += int(over.numel())
    self.cnt[0] += over.sum()

  @torch.no_grad()
  def episodes(self, ids: torch.Tensor, failed: torch.Tensor, clip: torch.Tensor) -> None:
    """Episodes of envs ids ended (failed: terminated, bool [n]; clip: their clip indices [n]); resets their flags."""
    ov = self.over[ids]
    self.over[ids] = False
    self.n_episodes += int(ids.numel())
    self.cnt[1] += ov.sum()
    if self.groups:
      gm = self.masks[:, clip]  # [G, n]
      self.grp[:, 1] += gm.sum(1)
      self.grp[:, 2] += (gm & failed.bool()[None, :]).sum(1)
      self.grp[:, 3] += (gm & ov[None, :]).sum(1)

  @torch.no_grad()
  def resets(self, clip: torch.Tensor) -> None:
    """New episodes started in clips ``clip`` [n]."""
    self.n_resets += int(clip.numel())
    if self.groups:
      self.grp[:, 0] += self.masks[:, clip].sum(1)

  # ---- log ------------------------------------------------------------------------

  def sums(self) -> dict[str, float]:
    """Raw counts of this window (fixed keys: summed over ranks under data parallel). One host sync."""
    c = torch.cat([self.cnt, self.grp.reshape(-1)]).tolist()
    out = {"watch/steps": float(self.n_steps), "watch/steps_over": c[0], "watch/episodes": float(self.n_episodes),
           "watch/episodes_over": c[1], "watch/resets": float(self.n_resets)}
    for i, g in enumerate(self.groups):
      r, e, f, o = c[2 + 4 * i : 6 + 4 * i]
      out.update({f"watch/{g}_resets": r, f"watch/{g}_episodes": e, f"watch/{g}_failed": f, f"watch/{g}_over": o})
    return out

  def scalars(self, s: dict[str, float]) -> dict[str, float]:
    """TB scalars from (rank-summed) counts; a rate only when its denominator is > 0."""
    out: dict[str, float] = {}
    if s["watch/episodes"] > 0:
      out["train/would_fail_ori"] = s["watch/episodes_over"] / s["watch/episodes"]
    if s["watch/steps"] > 0:
      out["train/ori_over_step_frac"] = s["watch/steps_over"] / s["watch/steps"]
    for g in self.groups:
      e = s[f"watch/{g}_episodes"]
      if s["watch/episodes"] > 0:
        out[f"train/watch_{g}_episodes"] = e
      if e > 0:
        out[f"train/watch_{g}_fail"] = s[f"watch/{g}_failed"] / e
        out[f"train/watch_{g}_would_fail_ori"] = s[f"watch/{g}_over"] / e
      if s["watch/resets"] > 0:
        out[f"train/watch_{g}_reset_share"] = s[f"watch/{g}_resets"] / s["watch/resets"]
    return out

  def summary(self, bin_clip: torch.Tensor | None = None) -> str:
    """One line for the start of the run: clips (and bins) per group."""
    parts = []
    for i, g in enumerate(self.groups):
      n = int(self.masks[i].sum())
      b = f", {self.masks[i][bin_clip].float().mean().item() * 100:.2f}% of bins" if bin_clip is not None and bin_clip.numel() else ""
      parts.append(f"{g} {list(self.keywords[i])}: {n} clips{b}")
    return "; ".join(parts) + f"; pelvis angle > {self.threshold:g} rad"
