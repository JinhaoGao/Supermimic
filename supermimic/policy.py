"""Policy (goal encoder + shared decoder) and critic networks (Table A.4)."""

from __future__ import annotations

import torch
import torch.nn as nn

from .world_model import RunningMeanStd, mlp


class Policy(nn.Module):
  def __init__(
    self,
    proprio_dim: int,
    hist: int,
    goal_frame_dim: int,
    K: int,
    action_dim: int,
    z_dim: int = 32,
    enc_hidden: tuple[int, ...] = (2048, 1024, 512, 256),
    dec_hidden: tuple[int, ...] = (2048, 1024, 512, 256),
    input_clip: float = 10.0,
  ):
    super().__init__()
    self.proprio_dim, self.hist, self.goal_frame_dim, self.K, self.action_dim = proprio_dim, hist, goal_frame_dim, K, action_dim
    self.input_clip = float(input_clip)  # clamp of the normalized inputs (TrainCfg.input_clip; baked into exports)
    self.obs_norm = RunningMeanStd(proprio_dim)
    self.goal_norm = RunningMeanStd(goal_frame_dim)
    self.encoder = mlp(K * goal_frame_dim, enc_hidden, z_dim, act=nn.ELU)
    self.decoder = mlp(z_dim + hist * proprio_dim, dec_hidden, action_dim, act=nn.ELU)

  def normalize_inputs(self, obs_hist: torch.Tensor, goal: torch.Tensor):
    c = self.input_clip
    on = self.obs_norm.normalize(obs_hist).clamp(-c, c)
    gn = self.goal_norm.normalize(goal).clamp(-c, c)
    return on, gn

  def forward(self, obs_hist: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    """obs_hist [B,hist,P] (oldest first), goal [B,K,G] -> action [B,A]."""
    B = obs_hist.shape[0]
    on, gn = self.normalize_inputs(obs_hist, goal)
    z = torch.tanh(self.encoder(gn.reshape(B, -1)))
    return self.decoder(torch.cat([z, on.reshape(B, -1)], dim=-1))


class Critic(nn.Module):
  def __init__(self, feat_dim: int, goal_frame_dim: int, K: int, xi_dim: int, hidden: tuple[int, ...] = (256, 128)):
    super().__init__()
    self.net = mlp(feat_dim + K * goal_frame_dim + xi_dim, hidden, 1, act=nn.ELU)

  def forward(self, feat_n: torch.Tensor, goal_n: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
    B = feat_n.shape[0]
    return self.net(torch.cat([feat_n, goal_n.reshape(B, -1), xi], dim=-1)).squeeze(-1)
