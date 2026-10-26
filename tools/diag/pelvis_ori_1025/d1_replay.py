"""D1: deterministic CPU MuJoCo replay of the eval site's root_ori failures of r15ws model_148000 on the TRAINING robot
model (bundle g1v2_r15ws_ddp8_148000: scene.mjb compiled by mjlab, G1 eval+evalcontact, policy.onnx), with the NEW code's
criteria evaluated every control step:

  new  = AnalyticHeads.root_ori_angle(kin_next, StepFns._ref_root(ref_bq)) > root_ori_term (1.0), i.e. the root_ori part
         of heads.termination(..., ref_rq=...) with HeadsCfg(root_ori_term=1.0); also checked: heads.termination's hard
         output == old | new, root_ori_angle_state (watch / logging path, s[3:7]) and the env term bad_root_ori_full
         (make_env_cfg's root_ori term, called as the termination manager does) agree with it;
  old  = heads.termination's hard output with the run's HeadsCfg (root_ori off): anchor (torso) z > 0.25, anchor tilt
         (gravity-z diff) > 0.8, end-effector z > 0.25 = every existing training criterion; anchor-only for attribution.
  site = the eval site's own quat_error_angle(MotionData.root_quat(t), qpos[3:7]) > 1.0 on this trajectory (reference).

Reference frames for the criteria come from supermimic.motions.MotionLibrary (the trainer's loader: cache over the eval-site
npz files of the list, tracked bodies of the run); the policy goal / reset use play_mujoco.Reference (the repo's player).
Protocol (PROTO=eval, default): the eval site's: reset to frame 0 (root pose + velocities, joints), step s targets frame s
(goal window s..s+K-1) and is scored against frame s, history = first proprio frame H times, prev action 0.
PROTO=mjlab: the training env's (first target = the frame after the reset frame).
Runs until CONT frames after the eval-site failure frame (or the clip end); stops early once new and old have both fired.

  python d1_replay.py <list.tsv> <out.jsonl> <chunk_index> <num_chunks>
list.tsv columns: path, eval_fail_frame, cluster, src (header line).
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402  (sets sys.path to the repo under test)

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402

from play_mujoco import MujocoPlayer, Reference  # noqa: E402  (repo scripts/)
from supermimic.motions import MotionLibrary  # noqa: E402
from supermimic.math_utils import quat_angle  # noqa: E402

torch.set_num_threads(1)
CONT = int(os.environ.get("CONT", "300"))  # frames after the eval failure frame (6 s)
PROTO = os.environ.get("PROTO", "eval")
TB = ("left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link", "right_wrist_yaw_link")  # eval site's
SITE_ONNX = "/home/ubuntu/work/NOetix_GMT_eval/storage/models/f801f1e6873e78224281d1382363684af7f33c314d14e4e242c630aa73d2f1ec/model.onnx"
SITE_OBS = "/home/ubuntu/work/NOetix_GMT_eval/config/observations/supermimic_g1_v1.yaml"
SITE_XML = "/home/ubuntu/work/unitree_mujoco_noetix_g1_gmt/unitree_robots/g1/scene_29dof.xml"


def read_list(path):
  rows = []
  with open(path) as f:
    hdr = f.readline().rstrip("\n").split("\t")
    for line in f:
      rows.append(dict(zip(hdr, line.rstrip("\n").split("\t"))))
  return rows


def site_tracked_names(ev, any_motion):
  args = ev.build_argparser().parse_args([
    "--policy-type", "supermimic", "--obs-config", SITE_OBS, "--policy", SITE_ONNX, "--motion-path", any_motion, "--model", SITE_XML,
    "--motion-joint-order", "policy", "--sim-dt", "0.001", "--decimation", "20", "--device", "cpu",
    "--root-pos-threshold", "0.25", "--root-height-threshold", "0.25", "--root-ori-threshold", "1.0",
    "--body-pos-threshold", "0.25", "--root-pos-mode", "z", "--body-pos-mode", "z", "--termination-bodies", ",".join(TB)])
  return ev.discover_tracked_body_names(ev.build_rollout_cfg(args))


def first_or_none(x):
  return None if x is None else int(x)


def main():
  lst, out, ci, nc = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
  rows = read_list(lst)
  files = [r["path"] for r in rows]
  meta = common.load_meta()
  player = MujocoPlayer(common.BUNDLE, backend="onnx")
  so = ort.SessionOptions()
  so.intra_op_num_threads = 1
  so.inter_op_num_threads = 1
  player.sess = ort.InferenceSession(os.path.join(common.BUNDLE, "policy.onnx"), sess_options=so, providers=["CPUExecutionProvider"])
  model, data = player.model, player.data
  S = common.Setup(model, meta)
  A = float(S.cfg_on.root_ori_term)
  lib = MotionLibrary(files, tuple(meta["joint_names"]), tuple(meta["body_names"]), "cpu", cache_dir=common.lib_cache_dir(), num_workers=8,
                      log=lambda *a: None)
  clip_of = {c.name: c for c in lib.clips}
  assert len(clip_of) == len(lib.clips) == len(files), "clip names not unique / clips dropped"
  ev = common.eval_site_module()
  site_names = site_tracked_names(ev, files[0])
  vel6 = np.zeros(6)
  tb_gid = [player.body_ids[meta["body_names"].index(n)] if n in meta["body_names"] else mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"robot/{n}") for n in TB]
  with open(out, "a") as fo:
    for k in range(ci, len(rows), nc):
      r = rows[k]
      t0 = time.time()
      path = r["path"]
      eval_fail = int(r["eval_fail_frame"])
      ref = Reference(path, meta)
      assert abs(ref.fps - 1.0 / player.dt) < 1e-6
      clip = clip_of[os.path.basename(path)]
      assert clip.length == ref.T
      mot = ev.MotionData(path, site_names)
      T = ref.T
      off = 1 if PROTO == "mjlab" else 0
      last_frame = min(T - 1, eval_fail + CONT)
      player.set_reference_pose(data, ref, 0, with_velocity=True)
      prev_a = torch.zeros(player.nj)
      hist = player.proprio(prev_a)[None].repeat(player.hist, 1)
      first = dict(new=None, old=None, anchor=None, ee_only=None, site=None, env=None, tm_site=None)
      tm_site_reason = None
      n_hard_mismatch = n_env_mismatch = 0
      max_dang_watch = max_dang_site = 0.0
      fk_q_err = fk_p_err = 0.0
      at = {}
      max_ang_pre = 0.0
      steps = 0
      for step in range(0, T):
        frame = min(step + off, T - 1)
        if frame > last_frame:
          break
        goal = player.goal(ref, frame)
        a = player.act(hist, goal)
        data.ctrl[player.act_ids] = player.offset + player.scale * a.numpy().astype(np.float64)
        for _ in range(player.decimation):
          mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        steps += 1
        # state as SimInterface.state(): root link pose / world velocities, joints (entity order)
        rb = player.root_bid
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, rb, vel6, 0)
        s = torch.tensor(np.concatenate([data.xpos[rb], data.xquat[rb], vel6[3:6], vel6[0:3], data.qpos[player.q_adr], data.qvel[player.v_adr]]),
                         dtype=torch.float32)[None]
        xq_all = torch.tensor(data.xquat[S.entity_body_gids], dtype=torch.float32)[None]
        g = lib.gather(torch.tensor([clip.start + frame]), fields=["body_pos", "body_quat"])
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
          fk_q_err = max(fk_q_err, float(quat_angle(kin["body_quat"][0], torch.tensor(data.xquat[S.tracked_gid], dtype=torch.float32)).max()))
          fk_p_err = max(fk_p_err, float((kin["body_pos"][0] - torch.tensor(data.xpos[S.tracked_gid], dtype=torch.float32)).abs().max()))
        new = bool(ang[0] > A)
        old, anc = bool(hard_off[0]), bool(hard_an[0])
        if bool(hard_on[0]) != (old or new):
          n_hard_mismatch += 1
        if env_fire != new:
          n_env_mismatch += 1
        ang_site = float(ev.quat_error_angle(mot.root_quat(frame), ev.quat_normalize(data.qpos[3:7].astype(np.float64).copy())))
        max_dang_watch = max(max_dang_watch, abs(float(ang[0]) - float(ang_w[0])))
        max_dang_site = max(max_dang_site, abs(float(ang[0]) - ang_site))
        # eval-site criteria on this (training-model) trajectory, the site's order: root z, root ori, EE z
        rz = abs(float(data.qpos[2]) - float(mot.root_pos(frame)[2]))
        eez = max(abs(float(data.xpos[b][2]) - float(mot.body_pos_w[mot.frame(frame), mot.body_index(n)][2])) for b, n in zip(tb_gid, TB))
        site_reason = "root_pos" if rz > 0.25 else ("root_ori" if ang_site > 1.0 else ("body_pos" if eez > 0.25 else None))
        if frame <= eval_fail:
          max_ang_pre = max(max_ang_pre, float(ang[0]))
        if frame == eval_fail:
          at["eval_fail"] = dict(ang=float(ang[0]), soft_on=float(soft_on[0]), soft_off=float(soft_off[0]), old=old)
        for key, v in (("new", new), ("old", old), ("anchor", anc), ("ee_only", old and not anc), ("site", ang_site > 1.0), ("env", env_fire),
                       ("tm_site", site_reason is not None)):
          if v and first[key] is None:
            first[key] = frame
            if key == "new":
              at["new"] = dict(ang=float(ang[0]), soft_on=float(soft_on[0]), soft_off=float(soft_off[0]), old=old)
            if key == "tm_site":
              tm_site_reason = site_reason
        if first["new"] is not None and first["old"] is not None and frame >= eval_fail:
          break
        prev_a = a
        hist = torch.cat([hist[1:], player.proprio(prev_a)[None]], dim=0)
      rec = dict(motion=os.path.basename(path), path=path, cluster=r.get("cluster", ""), src=r.get("src", ""), T=T, proto=PROTO,
                 eval_fail=eval_fail, last_frame=last_frame, steps=steps, first={k: first_or_none(v) for k, v in first.items()},
                 tm_site_reason=tm_site_reason, at=at, max_ang_pre_eval_fail=max_ang_pre,
                 n_hard_mismatch=n_hard_mismatch, n_env_mismatch=n_env_mismatch, max_dang_watch=max_dang_watch, max_dang_site=max_dang_site,
                 fk_quat_err_max=fk_q_err, fk_pos_err_max=fk_p_err, sec=round(time.time() - t0, 1))
      fo.write(json.dumps(rec) + "\n")
      fo.flush()
      print(rec["motion"], "eval_fail", eval_fail, "first", rec["first"], rec["sec"], "s", flush=True)
  if ci == 0:
    json.dump(S.describe() | dict(site_tracked=list(site_names)), open(out + ".setup.json", "w"), indent=1, default=str)


if __name__ == "__main__":
  main()
