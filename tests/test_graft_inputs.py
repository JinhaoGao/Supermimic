"""Unit test of Trainer._graft_inputs (round 18): grafted networks must compute what the old ones computed."""
import types
from dataclasses import asdict

import torch

from supermimic.world_model import WorldModelCfg

from supermimic.heads import AnalyticHeads, HeadsCfg
from supermimic.policy import Critic, Policy
from supermimic.trainer import Trainer

torch.manual_seed(0)
NJ, P, HIST, K, FD, XD, BT = 29, 3 * 29 + 6, 10, 10, 13 + 2 * 29, 33, 14


class HeadStub:
  goal_layout = AnalyticHeads.goal_layout

  def __init__(self, cfg):
    self.cfg, self.nj, self.tracked, self.feat_dim = cfg, NJ, torch.arange(BT), FD
    self.goal_frame_dim = sum(n for _, n in self.goal_layout())
    self.critic_err_dim = 4 * BT + 1 if cfg.critic_errors else 0


def build(cfg):
  h = HeadStub(cfg)
  pol = Policy(P, HIST, h.goal_frame_dim, K, NJ)
  crit = Critic(FD, h.goal_frame_dim, K, XD, err_dim=h.critic_err_dim)
  return h, pol, crit


for new_cfg in (HeadsCfg(goal_ref_root=True), HeadsCfg(critic_errors=True), HeadsCfg(goal_ref_root=True, critic_errors=True)):
  old_cfg = HeadsCfg()
  ho, po, co = build(old_cfg)
  with torch.no_grad():  # non-trivial normalizer statistics
    po.goal_norm.mean.normal_()
    po.goal_norm.var.uniform_(0.5, 2.0)
    po.goal_norm.count.fill_(1e9)
  opo, oco = torch.optim.Adam(po.parameters()), torch.optim.Adam(co.parameters())
  for opt, loss in ((opo, po(torch.randn(8, HIST, P), torch.randn(8, K, ho.goal_frame_dim)).square().mean()),
                    (oco, co(torch.randn(8, FD), torch.randn(8, K, ho.goal_frame_dim), torch.randn(8, XD)).square().mean())):
    opt.zero_grad()
    loss.backward()
    opt.step()
  ck = {"cfg": {"heads": asdict(old_cfg)}, "dims": {"goal_frame": ho.goal_frame_dim}, "policy": po.state_dict(), "critic": co.state_dict(),
        "opt_pi": opo.state_dict(), "opt_v": oco.state_dict()}
  ck = {k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items()}) for k, v in ck.items()}
  import copy
  ck = copy.deepcopy(ck)
  hn, pn, cn = build(new_cfg)
  me = types.SimpleNamespace(heads=hn, nj=NJ, xi_dim=XD, cfg=types.SimpleNamespace(heads=new_cfg, wm=WorldModelCfg()), policy=pn, critic=cn,
                             _print=print)
  new = Trainer._graft_inputs(me, ck)
  pn.load_state_dict(ck["policy"])
  cn.load_state_dict(ck["critic"])
  opn, ocn = torch.optim.Adam(pn.parameters()), torch.optim.Adam(cn.parameters())
  opn.load_state_dict(ck["opt_pi"])
  ocn.load_state_dict(ck["opt_v"])
  # inputs: old goal frames embedded in new frames, random values in the new columns
  B = 16
  obs, go = torch.randn(B, HIST, P), torch.randn(B, K, ho.goal_frame_dim)
  Gn = hn.goal_frame_dim
  gnew = torch.randn(B, K, Gn) * 5
  gnew[..., : ho.goal_frame_dim] = go  # ref_root is appended at the end of each frame
  with torch.no_grad():
    d_pol = (pn(obs, gnew) - po(obs, go)).abs().max().item()
    f = torch.randn(B, FD)
    fe = torch.cat([f, torch.randn(B, hn.critic_err_dim) * 5], -1)
    gno, gnn = po.goal_norm.normalize(go), pn.goal_norm.normalize(gnew)
    x = torch.randn(B, XD)
    d_crit = (cn(fe, gnn, x) - co(f, gno, x)).abs().max().item()
  st = opn.state_dict()["state"][0]["exp_avg"]
  print(f"{new_cfg.goal_ref_root=} {new_cfg.critic_errors=}: new segments {new}, policy diff {d_pol:.2e}, critic diff {d_crit:.2e}, "
        f"enc moment shape {tuple(st.shape)}, new-col moments zero {bool((st.view(st.shape[0], K, Gn)[..., ho.goal_frame_dim:] == 0).all())}")
  assert d_pol < 1e-5 and d_crit < 1e-5
print("graft OK")
