"""Shared setup of the D1 / D3 checks of ddp-bj31-v2 ecc551f (root-orientation termination), bj31, CPU only.

Builds the NEW code's objects the way the trainer does: HeadsCfg.from_dict of the run's heads cfg (bundle meta =
model_148000.pt cfg["heads"]), make_env_cfg terms (G1 eval+evalcontact, root_ori_term on / off), SimInterface.ee_bodies /
root_ori_body_of on those cfgs, TorchFK on the training model (bundle scene.mjb, compiled by mjlab at export), AnalyticHeads
and StepFns._ref_root. Nothing here re-implements the criterion: callers use AnalyticHeads.termination / root_ori_angle /
root_ori_angle_state and the env term function (library_command.bad_root_ori_full) as wired by make_env_cfg.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import sys
import types

REPO = os.environ.get("REPO", "/home/ubuntu/supermimic/supermimic_repro_oriterm_test")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from supermimic import env_cfg as ec  # noqa: E402
from supermimic.fk import TorchFK  # noqa: E402
from supermimic.fns import StepFns  # noqa: E402
from supermimic.heads import AnalyticHeads, HeadsCfg  # noqa: E402
from supermimic.sim_interface import SimInterface  # noqa: E402

BUNDLE = os.environ.get("BUNDLE", "/data2/supermimic/eval_export/bundles/g1v2_r15ws_ddp8_148000")
EV = "/home/ubuntu/work/NOetix_GMT_eval/vendor/Noetix_GMT/scripts/evaluate_mujoco.py"


def eval_site_module():
  """The eval site's evaluate_mujoco.py (read-only import, as diag_1025/taskB/scripts/trace_ori.py)."""
  if "evm" in sys.modules:
    return sys.modules["evm"]
  sys.path.insert(0, os.path.dirname(EV))
  spec = importlib.util.spec_from_file_location("evm", EV)
  ev = importlib.util.module_from_spec(spec)
  sys.modules["evm"] = ev
  spec.loader.exec_module(ev)
  return ev


def stub_env(envc):
  """What SimInterface's static checks read from an env: cfg.terminations and the motion command's cfg."""
  mc = envc.commands["motion"]
  return types.SimpleNamespace(cfg=envc, command_manager=types.SimpleNamespace(get_term=lambda name: types.SimpleNamespace(cfg=mc)))


class Setup:
  def __init__(self, model: mujoco.MjModel, meta: dict, A: float = 1.0, T: float = 0.05, entity: tuple | None = None):
    """entity: (body global ids, body names, hinge joint global ids) of the robot in ``model``; None = the bundle's mjlab
    model (bodies / joints prefixed "robot/", joint order = the run's entity order, checked)."""
    self.meta = meta
    self.cfg_off = HeadsCfg.from_dict(dict(meta["heads"]))  # the run's heads (root_ori_* absent -> off)
    assert not self.cfg_off.root_ori_on
    self.cfg_on = dataclasses.replace(self.cfg_off, root_ori_term=A, root_ori_term_temp=T)  # = CLI --root_ori_term A
    self.cfg_anchor = dataclasses.replace(self.cfg_off, ee_termination=False)  # anchor height / tilt only (attribution)
    robot, variant = meta["robot"], meta["robot_variant"]
    kw = dict(mode="eval", robot=robot, robot_variant=variant)
    ee = dict(ee_termination=self.cfg_off.ee_termination, ee_threshold=self.cfg_off.ee_term_z)
    self.envc_on = ec.make_env_cfg(["x.npz"], 1, root_ori_term=A, **kw, **ee)
    self.envc_off = ec.make_env_cfg(["x.npz"], 1, **kw, **ee)
    self.envc_anchor = ec.make_env_cfg(["x.npz"], 1, ee_termination=False, **kw)
    mc = self.envc_on.commands["motion"]
    assert tuple(mc.body_names) == tuple(meta["body_names"]), (mc.body_names, meta["body_names"])
    assert mc.anchor_body_name == meta["anchor_body_name"]
    self.term_names_on = list(self.envc_on.terminations)
    # mjlab entity of the training model (bundle scene.mjb): robot bodies / hinge joints in model order.
    if entity is None:
      bodies = [b for b in range(model.nbody) if model.body(b).name.startswith("robot/")]
      names = tuple(model.body(b).name.split("/", 1)[1] for b in bodies)
      hinges = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and model.joint(j).name.startswith("robot/")]
      assert [model.joint(j).name.split("/", 1)[1] for j in hinges] == list(meta["joint_names"])
    else:
      bodies, names, hinges = list(entity[0]), tuple(entity[1]), list(entity[2])
    self.hinges = hinges
    ent = types.SimpleNamespace(indexing=types.SimpleNamespace(body_ids=torch.tensor(bodies), joint_ids=torch.tensor(hinges)), body_names=names)
    self.tracked = tuple(meta["body_names"])
    self.tracked_local = [names.index(n) for n in self.tracked]
    self.tracked_gid = [bodies[i] for i in self.tracked_local]
    self.fk = TorchFK(model, ent, "cpu", needed_bodies=self.tracked_local)
    self.root_name = names[self.fk.root_local]  # = SimInterface.root_body_name
    self.entity_body_names = list(names)
    self.entity_body_gids = bodies
    anchor = self.tracked.index(meta["anchor_body_name"])
    q0 = torch.tensor(meta["default_joint_pos"], dtype=torch.float32)
    lim = torch.tensor(meta["soft_joint_limits"], dtype=torch.float32)

    def mk(cfg, envc):
      se = stub_env(envc)
      _, ee_idx = SimInterface.ee_bodies(se, cfg)
      ro_name, ro_idx = SimInterface.root_ori_body_of(se, cfg, self.root_name)
      h = AnalyticHeads(self.fk, self.tracked_local, anchor, q0, lim, cfg, ee_indices=ee_idx, root_ori_index=ro_idx,
                        root_ori_is_root=ro_name == self.root_name)
      return h, ro_name, ro_idx, ee_idx

    self.h_on, self.ro_name, self.ro_idx, self.ee_idx = mk(self.cfg_on, self.envc_on)
    self.h_off, n2, i2, _ = mk(self.cfg_off, self.envc_off)
    self.h_anchor, _, _, _ = mk(self.cfg_anchor, self.envc_anchor)
    assert (n2, i2) == (self.ro_name, self.ro_idx)
    self.f_on = StepFns(self.h_on, None, compile_=False)
    self.f_off = StepFns(self.h_off, None, compile_=False)
    self.f_anchor = StepFns(self.h_anchor, None, compile_=False)
    t = self.envc_on.terminations["root_ori"]
    self.env_term_func, self.env_term_params = t.func, dict(t.params)

  def describe(self) -> dict:
    return dict(
      root_body=self.root_name, root_ori_body=self.ro_name, root_ori_index=self.ro_idx, ee=[self.tracked[i] for i in self.ee_idx],
      anchor=self.tracked[self.h_on.anchor], terms_on=self.term_names_on, env_term_params=self.env_term_params,
      cfg_on={k: v for k, v in dataclasses.asdict(self.cfg_on).items() if k.startswith(("term_", "ee_", "root_ori"))},
    )

  def env_term(self, ref_bq: torch.Tensor, body_link_quat_w: torch.Tensor) -> torch.Tensor:
    """The simulator-side term exactly as mjlab's termination manager calls it (func(env, **params)), on a stand-in env
    holding the command's reference quaternions (cmd.body_quat_w, tracked order) and the robot's body_link_quat_w
    (entity order, = MuJoCo xquat)."""
    cmd = types.SimpleNamespace(
      body_quat_w=ref_bq, cfg=types.SimpleNamespace(body_names=list(self.tracked)), term_scale_env=torch.ones(ref_bq.shape[0]),
      robot=types.SimpleNamespace(data=types.SimpleNamespace(body_link_quat_w=body_link_quat_w), body_names=self.entity_body_names),
    )
    env = types.SimpleNamespace(command_manager=types.SimpleNamespace(get_term=lambda name: cmd))
    return self.env_term_func(env, **self.env_term_params)


def entity_of_tree(model: mujoco.MjModel, root_body: str) -> tuple:
  """(bodies, names, hinge joints) of the kinematic tree under ``root_body`` of a plain MJCF (e.g. the eval site's)."""
  rid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_body)
  bodies = [b for b in range(model.nbody) if int(model.body_rootid[b]) == rid]
  hinges = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and int(model.jnt_bodyid[j]) in bodies]
  return bodies, [model.body(b).name for b in bodies], hinges


def load_meta(bundle: str = BUNDLE) -> dict:
  return json.load(open(os.path.join(bundle, "meta.json")))


def lib_cache_dir() -> str:
  return os.environ.get("D1_CACHE", "/data2/supermimic/diag_1025/oriterm/cache_d1")
