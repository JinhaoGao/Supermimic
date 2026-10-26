"""D1 complement on the EVAL model, phase 2 (mjlab venv): the eval site's own deterministic rollouts (dumped by
trace_states.py with the site's python: its runner, MJCF, explicit PD, ONNX of model_148000 as the site ran it; the loop
of diag_1025/taskB/scripts/trace_ori.py) with the NEW code's criteria on their states (kinematics of the dumped qpos/qvel
on the site's MJCF, TorchFK built on its G1 tree, heads / StepFns / MotionLibrary as in d1_replay.py). Shows whether, on
the very trajectories the site scored, the new criterion fires at the site's failure frame and before every existing
training criterion.
  python d1_evalmodel.py <list.tsv> <dump_dir> <out.jsonl> <chunk_index> <num_chunks>"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from d1_replay import SITE_XML, read_list  # noqa: E402
from supermimic.math_utils import quat_angle  # noqa: E402
from supermimic.motions import MotionLibrary  # noqa: E402

torch.set_num_threads(1)
CONT = int(os.environ.get("CONT", "300"))


def main():
  lst, dump, out, ci, nc = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
  rows = read_list(lst)
  files = [r["path"] for r in rows]
  meta = common.load_meta()
  lib = MotionLibrary(files, tuple(meta["joint_names"]), tuple(meta["body_names"]), "cpu", cache_dir=common.lib_cache_dir(), log=lambda *a: None)
  clip_of = {c.name: c for c in lib.clips}
  model = mujoco.MjModel.from_xml_path(SITE_XML)
  d2 = mujoco.MjData(model)
  rb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
  assert int(model.body_jntnum[rb]) == 1 and model.jnt_type[int(model.body_jntadr[rb])] == mujoco.mjtJoint.mjJNT_FREE
  ent = common.entity_of_tree(model, "pelvis")
  S = common.Setup(model, meta, entity=ent)
  qadr = np.array([model.jnt_qposadr[j] for j in S.hinges])
  vadr = np.array([model.jnt_dofadr[j] for j in S.hinges])
  jorder_same = [model.joint(j).name for j in S.hinges] == list(meta["joint_names"])
  vel6 = np.zeros(6)
  with open(out, "a") as fo:
    for k in range(ci, len(rows), nc):
      r = rows[k]
      t0 = time.time()
      mp = r["path"]
      name = os.path.basename(mp)
      eval_fail = int(r["eval_fail_frame"])
      z = np.load(os.path.join(dump, name))
      assert int(z["root_body_id"]) == rb
      QP, QV, RO, RS = z["qpos"], z["qvel"], z["root_ori_error"], z["reason"]
      clip = clip_of[name]
      T = clip.length
      first = dict(new=None, old=None, anchor=None, ee_only=None, site=None, env=None, tm_site=None)
      site_reason_first = None
      n_hard_mismatch = n_env_mismatch = 0
      max_dang_watch = max_dang_site = 0.0
      fk_q_err = fk_p_err = 0.0
      at = {}
      for step in range(len(QP)):
        d2.qpos[:] = QP[step]
        d2.qvel[:] = QV[step]
        mujoco.mj_forward(model, d2)
        mujoco.mj_objectVelocity(model, d2, mujoco.mjtObj.mjOBJ_XBODY, rb, vel6, 0)
        s = torch.tensor(np.concatenate([d2.xpos[rb], d2.xquat[rb], vel6[3:6], vel6[0:3], d2.qpos[qadr], d2.qvel[vadr]]), dtype=torch.float32)[None]
        xq_all = torch.tensor(d2.xquat[S.entity_body_gids], dtype=torch.float32)[None]
        g = lib.gather(torch.tensor([clip.start + min(step, T - 1)]), fields=["body_pos", "body_quat"])
        ref_bp, ref_bq = g["body_pos"], g["body_quat"]
        with torch.no_grad():
          kin = S.h_on.kin(s)
          ai = S.h_on.anchor
          rq_on = S.f_on._ref_root(ref_bq)
          hard_on, soft_on = S.h_on.termination(kin, ref_bp[:, ai], ref_bq[:, ai], ref_bp, None, ref_rq=rq_on)
          hard_off, soft_off = S.h_off.termination(kin, ref_bp[:, ai], ref_bq[:, ai], ref_bp, None, ref_rq=S.f_off._ref_root(ref_bq))
          hard_an, _ = S.h_anchor.termination(kin, ref_bp[:, ai], ref_bq[:, ai], ref_bp, None, ref_rq=S.f_anchor._ref_root(ref_bq))
          ang = S.h_on.root_ori_angle(kin, rq_on)
          ang_w = S.h_off.root_ori_angle_state(s, ref_bq)
          env_fire = bool(S.env_term(ref_bq, xq_all)[0])
          fk_q_err = max(fk_q_err, float(quat_angle(kin["body_quat"][0], torch.tensor(d2.xquat[S.tracked_gid], dtype=torch.float32)).max()))
          fk_p_err = max(fk_p_err, float((kin["body_pos"][0] - torch.tensor(d2.xpos[S.tracked_gid], dtype=torch.float32)).abs().max()))
        new = bool(ang[0] > S.cfg_on.root_ori_term)
        old, anc = bool(hard_off[0]), bool(hard_an[0])
        n_hard_mismatch += int(bool(hard_on[0]) != (old or new))
        n_env_mismatch += int(env_fire != new)
        max_dang_watch = max(max_dang_watch, abs(float(ang[0]) - float(ang_w[0])))
        max_dang_site = max(max_dang_site, abs(float(ang[0]) - float(RO[step])))
        reason = str(RS[step]) or None
        if step == eval_fail:
          at["eval_fail"] = dict(ang=float(ang[0]), site_root_ori=float(RO[step]), soft_on=float(soft_on[0]), soft_off=float(soft_off[0]), old=old, reason=reason)
        for key, v in (("new", new), ("old", old), ("anchor", anc), ("ee_only", old and not anc), ("site", RO[step] > 1.0), ("env", env_fire),
                       ("tm_site", reason is not None)):
          if v and first[key] is None:
            first[key] = step
            if key == "new":
              at["new"] = dict(ang=float(ang[0]), soft_on=float(soft_on[0]), soft_off=float(soft_off[0]), old=old)
            if key == "tm_site":
              site_reason_first = reason
      rec = dict(motion=name, path=mp, cluster=r.get("cluster", ""), src=r.get("src", ""), T=T, proto="eval_model",
                 eval_fail=eval_fail, last_frame=len(QP) - 1, steps=len(QP), first=first, tm_site_reason=site_reason_first, at=at,
                 max_ang_pre_eval_fail=float("nan"), n_hard_mismatch=n_hard_mismatch, n_env_mismatch=n_env_mismatch,
                 max_dang_watch=max_dang_watch, max_dang_site=max_dang_site, fk_quat_err_max=fk_q_err, fk_pos_err_max=fk_p_err,
                 joint_order_same_as_training=jorder_same, sec=round(time.time() - t0, 1))
      fo.write(json.dumps(rec) + "\n")
      fo.flush()
      print(rec["motion"], "eval_fail", eval_fail, "first", first, site_reason_first, rec["sec"], "s", flush=True)


if __name__ == "__main__":
  main()
