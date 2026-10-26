"""World-model LayerNorm option (WorldModelCfg.layernorm / input_clip / min_std, 10-26) and the graft of a checkpoint
without it (Trainer._graft_wm_ln): net.* weights kept, LayerNorms at identity init, the WM optimizer state loads with
the new parameters appended. CPU, torch only.

  python tests/test_wm_layernorm.py
"""

import copy
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # the repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_sampler_clamp  # noqa: E402,F401  (stubs mjlab / tensorboard via sys.modules when they are missing)

import torch  # noqa: E402

from supermimic.trainer import Trainer  # noqa: E402
from supermimic.world_model import DynamicsModel, WorldModelCfg  # noqa: E402

J, XD, DT = 5, 9, 0.02


def batch(n=64, seed=0):
  g = torch.Generator().manual_seed(seed)
  s = torch.randn(n, 13 + 2 * J, generator=g)
  s[:, 3:7] = s[:, 3:7] / s[:, 3:7].norm(dim=-1, keepdim=True)
  return s, torch.randn(n, J, generator=g), torch.randn(n, XD, generator=g)


def test_off_is_the_plain_mlp():
  torch.manual_seed(0)
  a = DynamicsModel(J, XD, DT, WorldModelCfg(hidden=(32, 32)))
  b = DynamicsModel(J, XD, DT, WorldModelCfg(hidden=(32, 32), layernorm=False, input_clip=0.0))
  b.load_state_dict(a.state_dict())
  s, act, xi = batch()
  assert torch.equal(a.step(s, act, xi), b.step(s, act, xi))
  assert not any(k.startswith("ln.") for k in a.state_dict())


def test_layernorm_and_clip_act():
  torch.manual_seed(0)
  m = DynamicsModel(J, XD, DT, WorldModelCfg(hidden=(32, 32), layernorm=True, input_clip=3.0, min_std=1e-2))
  assert [k for k in m.state_dict() if k.startswith("ln.")] == ["ln.0.weight", "ln.0.bias", "ln.1.weight", "ln.1.bias"]
  assert m.in_norm.min_std == 1e-2
  for p in m.net[-1].parameters():  # zero-initialized last layer would hide the hidden path: give it weights
    torch.nn.init.normal_(p, std=0.1)
  s, act, xi = batch()
  out = m.step(s, act, xi)
  assert torch.isfinite(out).all()
  # huge inputs: the clamp + LayerNorm keep the prediction bounded
  big = m.step(s * 1e3 + 1.0, act * 1e3, xi)
  assert torch.isfinite(big).all()


def test_graft_from_a_checkpoint_without_layernorm():
  torch.manual_seed(1)
  old = DynamicsModel(J, XD, DT, WorldModelCfg(hidden=(32, 32)))
  for p in old.parameters():
    torch.nn.init.normal_(p, std=0.1)
  opt_old = torch.optim.Adam(old.parameters(), lr=1e-3)
  s, act, xi = batch()
  (old.step(s, act, xi) ** 2).mean().backward()
  opt_old.step()
  ck = {"wm": copy.deepcopy(old.state_dict()), "opt_wm": copy.deepcopy(opt_old.state_dict()), "cfg": {"wm": {"layernorm": False}}}
  new = DynamicsModel(J, XD, DT, WorldModelCfg(hidden=(32, 32), layernorm=True))
  me = types.SimpleNamespace(wm=new, cfg=types.SimpleNamespace(policy_delay=2000, wm_delay_lr=6.0), _print=print)
  Trainer._graft_wm_ln(me, ck)
  new.load_state_dict(ck["wm"])
  opt_new = torch.optim.Adam(new.parameters(), lr=1e-3)
  opt_new.load_state_dict(ck["opt_wm"])
  for k, v in old.state_dict().items():
    assert torch.equal(new.state_dict()[k], v), k
  assert torch.equal(new.ln[0].weight, torch.ones(32)) and torch.equal(new.ln[0].bias, torch.zeros(32))
  st_old, st_new = opt_old.state_dict()["state"], opt_new.state_dict()["state"]
  for i in st_old:  # the net parameters keep their Adam moments
    assert torch.equal(st_old[i]["exp_avg"], st_new[i]["exp_avg"]), i
  (new.step(s, act, xi) ** 2).mean().backward()
  opt_new.step()  # the LayerNorm parameters get their state lazily
  assert len(opt_new.state_dict()["state"]) == len(list(new.parameters()))


if __name__ == "__main__":
  for name, fn in list(globals().items()):
    if name.startswith("test_") and callable(fn):
      fn()
      print(f"ok  {name}")
