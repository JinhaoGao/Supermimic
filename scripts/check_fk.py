"""Validate the analytic heads against the simulator.

Checks, on random rollouts of a small env:
  * torch FK body poses / velocities vs ``EntityData.body_link_*``,
  * analytic reward terms vs mjlab's reward manager (expect a tiny gap from
    mjlab's one-substep staleness of derived quantities at reward time),
  * analytic termination vs the termination manager,
  * world-model algebra round trip (targets -> compose == next state).
"""

from __future__ import annotations

import argparse

import torch


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--motion_dirs", type=str, nargs="+", required=True)
  p.add_argument("--robot", type=str, default="g1")
  p.add_argument("--num_envs", type=int, default=64)
  p.add_argument("--steps", type=int, default=30)
  p.add_argument("--device", type=str, default="cuda:0")
  p.add_argument("--cache_dir", type=str, default="")
  args = p.parse_args()

  from mjlab.envs import ManagerBasedRlEnv

  from supermimic.env_cfg import make_env_cfg
  from supermimic.heads import HeadsCfg
  from supermimic.motions import collect_motion_files
  from supermimic.sim_interface import SimInterface
  from supermimic.world_model import DynamicsModel, WorldModelCfg

  files = collect_motion_files(args.motion_dirs, max_motions=8)
  env = ManagerBasedRlEnv(cfg=make_env_cfg(files, args.num_envs, mode="train", robot=args.robot, cache_dir=args.cache_dir), device=args.device)
  sim = SimInterface(env, HeadsCfg())
  heads, cmd, robot = sim.heads, sim.cmd, sim.robot
  print("entity bodies:", robot.body_names)
  print("entity joints:", robot.joint_names)
  print("tracked local ids:", cmd.body_indexes.tolist(), "anchor local:", heads.anchor_local)
  print("env_origins[:4]:", sim.env_origins[:4].tolist())
  print("xi[0]:", sim.xi()[0].tolist())

  env.reset()
  wm = DynamicsModel(sim.nj, sim.xi_dim, env.step_dt, WorldModelCfg()).to(args.device)
  worst = {}
  state = sim.state()
  kin = heads.kin(state)
  prev_a = torch.zeros(args.num_envs, sim.nj, device=args.device)
  for t in range(args.steps):
    pos_sim = robot.data.body_link_pose_w[:, cmd.body_indexes]
    vel_sim = robot.data.body_link_vel_w[:, cmd.body_indexes]
    e_pos = (kin["body_pos"] + sim.env_origins[:, None, :] - pos_sim[..., :3]).abs().max().item()
    qd = (kin["body_quat"] * pos_sim[..., 3:7]).sum(-1).abs()
    e_quat = (1.0 - qd).max().item()
    e_lin = (kin["body_lin_vel"] - vel_sim[..., :3]).abs().max().item()
    e_ang = (kin["body_ang_vel"] - vel_sim[..., 3:6]).abs().max().item()
    for k, v in dict(fk_pos=e_pos, fk_quat=e_quat, fk_linvel=e_lin, fk_angvel=e_ang).items():
      worst[k] = max(worst.get(k, 0.0), v)

    ref_cur = sim.current_ref()
    ref = dict(body_pos=ref_cur[0], body_quat=ref_cur[1], body_lin_vel=ref_cur[2], body_ang_vel=ref_cur[3])
    a = 0.5 * torch.randn(args.num_envs, sim.nj, device=args.device)
    env.step(a)
    next_state = sim.state()
    kin_next = heads.kin(next_state)
    r, terms = heads.reward(kin, kin_next, next_state, a, prev_a, ref, return_terms=True)
    hard, soft = heads.termination(kin_next, ref["body_pos"], ref["body_quat"])
    rm = env.reward_manager
    name_map = {
      "root_pos": "motion_global_root_pos", "root_ori": "motion_global_root_ori", "body_pos": "motion_body_pos",
      "body_ori": "motion_body_ori", "body_lin_vel": "motion_body_lin_vel", "body_ang_vel": "motion_body_ang_vel",
      "action_rate": "action_rate_l2", "joint_limit": "joint_limit",
    }
    for k, mk in name_map.items():
      try:
        sim_term = rm._step_reward[:, rm.active_terms.index(mk)] * env.step_dt
      except Exception:  # noqa: BLE001
        sim_term = None
      if sim_term is not None:
        worst[f"rew_{k}"] = max(worst.get(f"rew_{k}", 0.0), (terms[k] - sim_term).abs().max().item())
    worst["rew_total"] = max(worst.get("rew_total", 0.0), (r - env.reward_buf).abs().max().item())
    term_sim = env.reset_terminated.bool()
    worst["term_mismatch"] = max(worst.get("term_mismatch", 0.0), (hard != term_sim).float().mean().item())
    tg = wm.targets(state, next_state)
    rt = wm.compose(state, tg)
    worst["wm_roundtrip"] = max(worst.get("wm_roundtrip", 0.0), (rt - next_state).abs().max().item())

    done = env.reset_buf.bool()
    if done.any():
      ids = done.nonzero(as_tuple=False).squeeze(-1)
      env.reset(env_ids=ids)
      state = sim.state()
      prev_a = torch.where(done[:, None], torch.zeros_like(a), a)
    else:
      state = next_state
      prev_a = a
    kin = heads.kin(state)
  # Goal refs sanity.
  ref_q, ref_qd, ref_ap, ref_aq = sim.goal_refs(cmd.time_steps.clone(), cmd.clip_end_env.clone())
  g = heads.goal(kin, ref_q, ref_qd, ref_ap, ref_aq)
  print("goal shape:", tuple(g.shape), "finite:", bool(torch.isfinite(g).all()))
  print("\nmax abs errors over rollout:")
  for k, v in worst.items():
    print(f"  {k:16s} {v:.3e}")


if __name__ == "__main__":
  main()
