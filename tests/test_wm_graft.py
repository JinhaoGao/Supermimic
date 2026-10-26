"""Round 19: heading-frame -> root-frame world-model graft (CPU). Upright (pure-yaw) states must predict the same."""
import copy
import types
from dataclasses import asdict

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # the repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_sampler_clamp  # noqa: E402,F401  (stubs mjlab / tensorboard via sys.modules when they are missing)
import torch

from supermimic.trainer import Trainer
from supermimic.world_model import DynamicsModel, WorldModelCfg

torch.manual_seed(0)
J, XD, dt = 29, 33, 0.02
old = DynamicsModel(J, XD, dt, WorldModelCfg(hidden=(256, 256)))
for p in old.net.parameters():
  torch.nn.init.normal_(p, std=0.05)
with torch.no_grad():  # trained-like statistics: upright mean of the rot6d features, spread elsewhere
  old.in_norm.mean.normal_(0, 0.3)
  old.in_norm.var.uniform_(0.5, 2.0)
  old.in_norm.mean[1:7] = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
  old.in_norm.var[1:7] = 0.01
  old.tgt_norm.var.uniform_(0.5, 2.0)
opt = torch.optim.Adam(old.parameters())
s = torch.cat([torch.randn(8, 3), torch.tensor([[1.0, 0, 0, 0]]).expand(8, 4), torch.randn(8, 6 + 2 * J)], -1)
old.step(s, torch.randn(8, J), torch.randn(8, XD)).square().mean().backward()
opt.step()
ck = copy.deepcopy({"cfg": {"wm": asdict(WorldModelCfg())}, "wm": old.state_dict(), "opt_wm": opt.state_dict()})
new = DynamicsModel(J, XD, dt, WorldModelCfg(hidden=(256, 256), body_frame=True))


def gather(fr, fields=None):
  q = torch.randn(len(fr), 14, 4) * torch.tensor([4.0, 0.3, 0.3, 1.0])  # mostly upright
  return {"body_quat": q / q.norm(dim=-1, keepdim=True)}


me = types.SimpleNamespace(cfg=types.SimpleNamespace(wm=WorldModelCfg(body_frame=True), policy_delay=500, policy_freeze=200, wm_delay_lr=4.0),
                           nj=J, xi_dim=XD, wm=new, device="cpu", lib=types.SimpleNamespace(time_step_total=1000, gather=gather), _print=print)
Trainer._graft_wm(me, ck)
new.load_state_dict(ck["wm"])
torch.optim.Adam(new.parameters()).load_state_dict(ck["opt_wm"])
def states(n, tilt_deg):
  yaw = torch.rand(n) * 6.28
  qy = torch.stack([torch.cos(yaw / 2), torch.zeros(n), torch.zeros(n), torch.sin(yaw / 2)], -1)
  ax = torch.randn(n, 3)
  ax[:, 2] = 0
  ax = ax / ax.norm(dim=-1, keepdim=True)
  ang = torch.deg2rad(torch.full((n,), float(tilt_deg)))
  qt = torch.cat([torch.cos(ang / 2)[:, None], torch.sin(ang / 2)[:, None] * ax], -1)
  from supermimic.math_utils import quat_mul
  return torch.cat([torch.randn(n, 3), quat_mul(qy, qt), torch.randn(n, 6), torch.randn(n, 2 * J)], -1)


n = 256
a, xi = torch.randn(n, J), torch.randn(n, XD)
naive = copy.deepcopy(new)
with torch.no_grad():
  naive.net[0].weight[:, 1:4] = 0  # the first graft version: gravity columns zero, rot6d information dropped
for tilt in (0, 3, 10, 30):
  s = states(n, tilt)
  with torch.no_grad():
    so, sn, sv = old.step(s, a, xi), new.step(s, a, xi), naive.step(s, a, xi)
  step = (so - s).abs().mean()
  print(f"tilt {tilt:2d} deg: mean |old - new| / mean step {((so - sn).abs().mean() / step):.3f}   "
        f"(without the rot6d -> gravity mapping {((so - sv).abs().mean() / step):.3f})")
  if tilt == 0:
    assert (so - sn).abs().max() < 1e-4
print("WM graft OK")
