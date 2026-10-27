"""SimInterface for a human reference: HumanHeads on the bone-map links, human goal / step references."""

from __future__ import annotations

import torch

from ..fk import TorchFK
from ..heads import window_frames
from ..sim_interface import SimInterface
from .heads import GOAL_FIELDS, STEP_FIELDS, HumanHeads, HumanHeadsCfg


class HumanSimInterface(SimInterface):
  def __init__(self, env, heads_cfg: HumanHeadsCfg):
    self.env = env
    self.device = env.device
    self.robot = env.scene["robot"]
    self.cmd = env.command_manager.get_term("motion")
    self.lib = self.cmd.library
    names = self.cmd.bone_map.robot_links
    link_local = [self.robot.body_names.index(n) for n in names]
    self.fk = TorchFK(env.sim.mj_model, self.robot, self.device, needed_bodies=link_local)
    self.ee_body_names = ()
    self.root_body_name = str(self.robot.body_names[self.fk.root_local])
    self.root_ori_body = self.root_body_name
    self.heads = HumanHeads(self.fk, link_local, self.robot.data.default_joint_pos[0], self.robot.data.soft_joint_pos_limits[0],
                            heads_cfg, self.cmd.rest_height, self.cmd.ankle_height)
    self.cmd.heads = self.heads  # the simulator termination term uses the same hard test
    self._init_context(env)

  @torch.no_grad()
  def goal_refs(self, frame: torch.Tensor, clip_end: torch.Tensor):
    """(h_feat [E,K,138], h_yaw [E,K]) over the goal window."""
    K = self.heads.cfg.K
    E = frame.shape[0]
    g = self.lib.gather(window_frames(frame, clip_end, K).reshape(-1), fields=GOAL_FIELDS)
    return tuple(g[k].view(E, K, *g[k].shape[1:]) for k in GOAL_FIELDS)

  def current_ref(self) -> tuple:
    """Step reference fields (human.heads.STEP_FIELDS order) at the command's current frame."""
    c = self.cmd.current_ref()
    return tuple(c[k] for k in STEP_FIELDS)
