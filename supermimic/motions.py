"""Motion library: many clips concatenated on one global timeline (GPU).

Input npz format (our GMT / BeyondMimic convention):
  fps, joint_names[J], joint_pos[T,J], joint_vel[T,J], body_names[Bn],
  body_pos_w[T,Bn,3], body_quat_w[T,Bn,4] (wxyz), body_lin_vel_w, body_ang_vel_w.
Joints are remapped by name to the entity's joint order; bodies are remapped by
name to the tracked-body list (``cfg.body_names`` of the tracking task).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ClipInfo:
  name: str
  start: int
  end: int  # exclusive

  @property
  def length(self) -> int:
    return self.end - self.start


class MotionLibrary:
  def __init__(
    self,
    files: list[str],
    joint_names: tuple[str, ...],
    body_names: tuple[str, ...],
    device: str,
    bin_seconds: float = 1.0,
    expected_fps: float = 50.0,
  ):
    self.device = device
    self.joint_names = tuple(joint_names)
    self.body_names = tuple(body_names)
    jp, jv, bp, bq, blv, bav = [], [], [], [], [], []
    self.clips: list[ClipInfo] = []
    cursor = 0
    for f in files:
      d = np.load(f)
      fps = float(np.asarray(d["fps"]).reshape(-1)[0])
      if not math.isclose(fps, expected_fps, rel_tol=1e-6):
        raise ValueError(f"{f}: fps {fps} != {expected_fps}")
      src_j = [str(x) for x in d["joint_names"]]
      src_b = [str(x) for x in d["body_names"]]
      jperm = [src_j.index(n) for n in self.joint_names]
      bperm = [src_b.index(n) for n in self.body_names]
      T = int(d["joint_pos"].shape[0])
      jp.append(torch.as_tensor(d["joint_pos"][:, jperm], dtype=torch.float32))
      jv.append(torch.as_tensor(d["joint_vel"][:, jperm], dtype=torch.float32))
      bp.append(torch.as_tensor(d["body_pos_w"][:, bperm], dtype=torch.float32))
      bq.append(torch.as_tensor(d["body_quat_w"][:, bperm], dtype=torch.float32))
      blv.append(torch.as_tensor(d["body_lin_vel_w"][:, bperm], dtype=torch.float32))
      bav.append(torch.as_tensor(d["body_ang_vel_w"][:, bperm], dtype=torch.float32))
      self.clips.append(ClipInfo(os.path.basename(f), cursor, cursor + T))
      cursor += T
    self.fps = expected_fps
    self.dt = 1.0 / expected_fps
    self.joint_pos = torch.cat(jp).to(device)
    self.joint_vel = torch.cat(jv).to(device)
    self.body_pos_w = torch.cat(bp).to(device)
    self.body_quat_w = torch.cat(bq).to(device)
    self.body_lin_vel_w = torch.cat(blv).to(device)
    self.body_ang_vel_w = torch.cat(bav).to(device)
    self.time_step_total = int(cursor)
    self.num_clips = len(self.clips)
    self.clip_start = torch.tensor([c.start for c in self.clips], device=device)
    self.clip_end = torch.tensor([c.end for c in self.clips], device=device)
    clip_of_frame = torch.zeros(self.time_step_total, dtype=torch.long)
    for i, c in enumerate(self.clips):
      clip_of_frame[c.start : c.end] = i
    self.clip_of_frame = clip_of_frame.to(device)
    self.clip_end_of_frame = self.clip_end[self.clip_of_frame]

    # Bins of ``bin_seconds`` per clip (never crossing clip boundaries).
    bin_len = int(round(bin_seconds * expected_fps))
    starts, ends, bclip = [], [], []
    for i, c in enumerate(self.clips):
      for s in range(c.start, c.end, bin_len):
        starts.append(s)
        ends.append(min(s + bin_len, c.end))
        bclip.append(i)
    self.bin_start = torch.tensor(starts, device=device)
    self.bin_end = torch.tensor(ends, device=device)
    self.bin_clip = torch.tensor(bclip, device=device)
    self.num_bins = len(starts)
    bin_of_frame = torch.zeros(self.time_step_total, dtype=torch.long)
    for b, (s, e) in enumerate(zip(starts, ends)):
      bin_of_frame[s:e] = b
    self.bin_of_frame = bin_of_frame.to(device)

  @property
  def total_seconds(self) -> float:
    return self.time_step_total * self.dt

  def summary(self) -> str:
    return (
      f"MotionLibrary: {self.num_clips} clips, {self.time_step_total} frames "
      f"({self.total_seconds / 3600:.3f} h), {self.num_bins} bins, "
      f"{len(self.body_names)} tracked bodies, {len(self.joint_names)} joints"
    )
