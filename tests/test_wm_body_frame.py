"""Round 19 unit tests (CPU): root-frame world-model algebra, feature continuity through pitch 90 deg, critic graft."""
import copy
import math
import types
from dataclasses import asdict

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # the repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_sampler_clamp  # noqa: E402,F401  (stubs mjlab / tensorboard via sys.modules when they are missing)
import torch

from supermimic.heads import AnalyticHeads, HeadsCfg
from supermimic.math_utils import quat_mul
from supermimic.policy import Critic, Policy
from supermimic.trainer import Trainer
from supermimic.world_model import DynamicsModel, WorldModelCfg

torch.manual_seed(0)
J, XD, dt = 29, 33, 0.02


def quat_axis(axis, ang):
  a = torch.tensor(axis, dtype=torch.float32)
  a = a / a.norm()
  return torch.cat([torch.cos(ang / 2)[:, None], torch.sin(ang / 2)[:, None] * a[None]], -1)


def rand_states(n):
  q = torch.randn(n, 4)
  q = q / q.norm(dim=-1, keepdim=True)
  return torch.cat([torch.randn(n, 3), q, torch.randn(n, 3), torch.randn(n, 3) * 3, torch.randn(n, J), torch.randn(n, J)], -1)


for body in (False, True):
  wm = DynamicsModel(J, XD, dt, WorldModelCfg(body_frame=body, hidden=(32,)))
  s, s2 = rand_states(512), rand_states(512)
  # make s2 a plausible successor: small rotation increment
  s2[:, 3:7] = quat_mul(s[:, 3:7], quat_axis([0.3, -0.5, 0.8], torch.full((512,), 0.05)))
  rec = wm.compose(s, wm.targets(s, s2))
  err = (rec - s2).abs()
  qerr = torch.minimum((rec[:, 3:7] - s2[:, 3:7]).abs().amax(-1), (rec[:, 3:7] + s2[:, 3:7]).abs().amax(-1)).max()
  print(f"body_frame={body}: feat_dim {wm.feat_dim}, compose(targets) reconstruction max err pos/vel/joints "
        f"{torch.cat([err[:, :3], err[:, 7:]], -1).max():.2e}, quat {qerr:.2e}")
  # back flip: pure pitch through 90 deg with forward velocity
  th = torch.linspace(1.4, 1.75, 8)
  st = torch.zeros(8, 13 + 2 * J)
  st[:, 2] = 1.0
  st[:, 3:7] = quat_axis([0, 1, 0], th)
  st[:, 7] = 1.0  # world vx
  st[:, 11] = -6.0  # pitch rate
  f = wm.features(st)
  jump = (f[1:, :10] - f[:-1, :10]).abs().amax(-1)
  print(f"  pitch {[round(math.degrees(x), 1) for x in th.tolist()]}: max feature jump per 2.9 deg step "
        f"{[round(x, 2) for x in jump.tolist()]}")

# critic graft across the WM representation change
NJ, P, HIST, K, BT = J, 3 * J + 6, 10, 10, 14
hc = HeadsCfg()


class HeadStub:
  goal_layout = AnalyticHeads.goal_layout

  def __init__(self, cfg):
    self.cfg, self.nj, self.tracked = cfg, NJ, torch.arange(BT)
    self.goal_frame_dim = sum(n for _, n in self.goal_layout())
    self.critic_err_dim = 4 * BT + 1 if cfg.critic_errors else 0


h = HeadStub(hc)
po, co = Policy(P, HIST, h.goal_frame_dim, K, NJ), Critic(13 + 2 * J, h.goal_frame_dim, K, XD)
opo, oco = torch.optim.Adam(po.parameters()), torch.optim.Adam(co.parameters())
(co(torch.randn(4, 13 + 2 * J), torch.randn(4, K, h.goal_frame_dim), torch.randn(4, XD)).square().mean()).backward()
oco.step()
(po(torch.randn(4, HIST, P), torch.randn(4, K, h.goal_frame_dim)).square().mean()).backward()
opo.step()
ck = copy.deepcopy({"cfg": {"heads": asdict(hc), "wm": asdict(WorldModelCfg())}, "dims": {"goal_frame": h.goal_frame_dim},
                    "policy": po.state_dict(), "critic": co.state_dict(), "opt_pi": opo.state_dict(), "opt_v": oco.state_dict()})
pn, cn = Policy(P, HIST, h.goal_frame_dim, K, NJ), Critic(10 + 2 * J, h.goal_frame_dim, K, XD)
me = types.SimpleNamespace(heads=h, nj=NJ, xi_dim=XD, cfg=types.SimpleNamespace(heads=hc, wm=WorldModelCfg(body_frame=True)), policy=pn, critic=cn,
                           _print=print)
Trainer._graft_inputs(me, ck)
pn.load_state_dict(ck["policy"])
cn.load_state_dict(ck["critic"])
torch.optim.Adam(cn.parameters()).load_state_dict(ck["opt_v"])
W_old, W_new = co.net[0].weight, cn.net[0].weight
same_cols = [(0, 0)] + [(7 + i, 4 + i) for i in range(6 + 2 * J)]  # z ; v, w, q, qd
ok = all(torch.equal(W_old[:, o], W_new[:, n]) for o, n in same_cols) and bool((W_new[:, 3] == 0).all())
ok &= torch.equal(W_new[:, 1], -W_old[:, 5]) and torch.equal(W_new[:, 2], -W_old[:, 6])  # gravity x, y = -rot6d[4], -rot6d[5]
ok &= torch.equal(W_old[:, 13 + 2 * J:], W_new[:, 10 + 2 * J:])
print("critic graft across WM representations:", "OK" if ok else "MISMATCH", tuple(W_old.shape), "->", tuple(W_new.shape))
assert ok
