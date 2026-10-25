"""World-model / imagination diagnostic on groups of clips.

Rolls a checkpoint's policy (deterministic by default, training DR + RSI) in the simulator on the clips of
the given groups, then reports per group and per phase (reference airborne or not):
  * real failure rate (terminations per second of play) and where they fall relative to airborne phases,
  * world-model open-loop k-step prediction error on the real transitions (normalized as in training),
  * imagination bias: the H-step return / survival the world model predicts for the policy from real start
    states minus what the simulator gave from the same states (same policy, same DR context).

  uv run python scripts/wm_diag.py --ckpt logs/.../model_96000.pt \
      --group aerial=aerial.txt --group kicks=kicks.txt --group normal=normal.txt --out diag.json

(ddp-bj31-v2: the checkpoint's heads cfg goes through HeadsCfg.from_dict, so master-format and this branch's
checkpoints both load; imagination runs without the hard-clip curriculum factor, as the policy is evaluated.)
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch


def group_lists(specs):
  files, owner = [], {}
  for spec in specs:
    name, path = spec.split("=", 1)
    for line in open(os.path.expanduser(path)):
      f = line.strip()
      if f and os.path.basename(f) not in owner:
        owner[os.path.basename(f)] = name
        files.append(f)
  return files, owner


@torch.no_grad()
def imagine(tr, env, slot, H):
  """Closed-loop imagination of the policy from buffer slots (as Trainer.update_policy, without gradients)."""
  B, heads, nj, K = tr.buffer, tr.heads, tr.nj, tr.cfg.heads.K
  hist = tr.cfg.heads.hist
  N = env.shape[0]
  ep = B.ep_step[env, slot]
  back = torch.minimum(torch.arange(hist - 1, -1, -1, device=env.device)[None, :], ep[:, None])
  hslots = slot[:, None] - back
  env_h = env[:, None].expand(N, hist)
  xi = B.xi[env, slot]
  bias = tr.sim.bias_from_xi(xi, nj)
  hs, hp = B.state[env_h, hslots], B.prev_action[env_h, hslots]
  obs_hist = heads.proprio(
    hs.reshape(-1, hs.shape[-1]), hp.reshape(-1, nj), bias[:, None, :].expand(N, hist, nj).reshape(-1, nj), True
  ).view(N, hist, -1)
  s = B.state[env, slot]
  prev_a = B.prev_action[env, slot]
  frame, clip_end = B.frame[env, slot], B.clip_end[env, slot]
  kin = heads.kin(s)
  Lw = H + K
  pack = tr.lib.gather_windows(frame, Lw)
  f_loc = torch.zeros(N, dtype=torch.long, device=env.device)
  clip_end_loc = torch.clamp(clip_end - frame, max=Lw)
  p = (pack["joint_pos"], pack["joint_vel"], pack["body_pos"], pack["body_quat"], pack["body_lin_vel"], pack["body_ang_vel"])
  g = tr.cfg.gamma
  c = torch.ones(N, device=env.device)
  J = torch.zeros(N, device=env.device)
  for t in range(H):
    s, obs_hist, prev_a, f_loc, kin, r, d, _, _ = tr.fns.imagine_step(s, obs_hist, prev_a, f_loc, clip_end_loc, xi, bias, kin, *p)
    J = J + (g**t) * c * r
    c = c * (1.0 - d)
  return J, c


@torch.no_grad()
def wm_errors(tr, env, t0, k):
  """Open-loop k-step error of the world model on real windows: total per step [N,k] and per component at step k."""
  B, wm, J = tr.buffer, tr.wm, tr.nj
  N = env.shape[0]
  slots = t0[:, None] + torch.arange(k, device=env.device)[None, :]
  env_k = env[:, None].expand(N, k)
  s = B.state[env, t0]
  xi = B.xi[env, t0]
  acts, nxt = B.action[env_k, slots], B.next_state[env_k, slots]
  std = wm.err_norm.std
  tot = []
  for j in range(k):
    s = wm.step(s, acts[:, j], xi)
    d = (wm.diffs(s, nxt[:, j]) / std) ** 2
    tot.append(d.mean(-1))
  comps = {"root_pos": d[:, 0:3], "root_rot": d[:, 3:6], "root_vel": d[:, 6:9], "root_angvel": d[:, 9:12],
           "joint_pos": d[:, 12 : 12 + J], "joint_vel": d[:, 12 + J : 12 + 2 * J]}
  return torch.stack(tot, 1), {kk: v.mean(-1) for kk, v in comps.items()}


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--ckpt", required=True)
  p.add_argument("--group", action="append", required=True, help="name=list.txt (npz paths); a clip belongs to its first group")
  p.add_argument("--num_envs", type=int, default=2048)
  p.add_argument("--steps", type=int, default=400, help="simulator steps per env (50 Hz)")
  p.add_argument("--H", type=int, default=None, help="imagination horizon (default: the run's)")
  p.add_argument("--k", type=int, default=8, help="world-model window")
  p.add_argument("--start_stride", type=int, default=4, help="imagination starts every n-th step")
  p.add_argument("--air_dz", type=float, default=0.10, help="reference airborne: lower foot this far above its standing height (m)")
  p.add_argument("--action_noise", type=float, default=0.0, help="0 = deterministic policy (as imagined)")
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--log_dir", default="/tmp/wm_diag")
  p.add_argument("--out", default=None)
  args = p.parse_args()

  from supermimic.heads import HeadsCfg
  from supermimic.trainer import TrainCfg, Trainer
  from supermimic.world_model import WorldModelCfg

  ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
  c = ck["cfg"]
  files, owner = group_lists(args.group)
  gnames = list(dict.fromkeys(s.split("=", 1)[0] for s in args.group))
  heads_cfg = HeadsCfg.from_dict({kk: v for kk, v in c["heads"].items() if kk != "extra"})  # any branch's cfg (EE spellings)
  wk = dict(c["wm"])
  wk["hidden"] = tuple(wk["hidden"])
  H = args.H or int(c["H"])
  T = args.steps
  cfg = TrainCfg(
    motion_files=files, robot=c["robot"], robot_variant=c.get("robot_variant", ""), num_envs=args.num_envs,
    device=args.device, seed=1, log_dir=args.log_dir, cache_dir=c.get("cache_dir", ""), num_workers=32,
    iterations=0, action_noise_std=args.action_noise, replay_capacity=T + 8,
    episode_length_s=float(c.get("episode_length_s", 10.0)), compile=False, H=H, gamma=float(c["gamma"]),
    eval_enabled=False, wd_enabled=False, wm=WorldModelCfg(**wk), heads=heads_cfg,
  )
  tr = Trainer(cfg)
  tr.load(args.ckpt)
  for m in (tr.policy.obs_norm, tr.policy.goal_norm):
    m.update = lambda x: None  # keep the checkpoint's normalizers
  tr.policy.eval()
  tr.wm.eval()
  lib, dev, E = tr.lib, tr.device, cfg.num_envs
  clip_group = torch.tensor([gnames.index(owner.get(ci.name, gnames[-1])) for ci in lib.clips], device=dev)

  t0 = time.time()
  for _ in range(T):
    tr.collect_step()
  print(f"[diag] {T} steps x {E} envs in {time.time() - t0:.0f}s")

  B = tr.buffer
  frame = B.frame[:, :T]
  term, done = B.term[:, :T], B.done[:, :T]
  timeout = done & ~term
  grp = clip_group[lib.clip_of_frame(frame.reshape(-1))].view(E, T)
  # Reference flight phase per library frame: every tracked body (link origins; the lowest is an ankle) above
  # the standing ankle height + air_dz. "Near" = within +-0.5 s of a flight frame (dilated along the frame axis).
  bz = lib.fields["body_pos"][:, :, 2].min(-1).values
  stand = torch.quantile(bz[torch.randperm(bz.numel())[:200000]], 0.05).item()
  air_f = bz > stand + args.air_dz
  near_f = torch.nn.functional.max_pool1d(air_f.float()[None, None], 51, stride=1, padding=25)[0, 0] > 0
  air, airk = air_f.to(dev)[frame], near_f.to(dev)[frame]
  print(f"[diag] lowest-body standing z {stand:.3f} m, flight > {stand + args.air_dz:.3f} m; "
        f"{air_f.float().mean().item() * 100:.2f}% of library frames in flight")

  # ---- 1. real failures --------------------------------------------------------------------------
  res: dict = {"ckpt": args.ckpt, "it": ck["it"], "steps": T, "envs": E, "H": H, "k": args.k, "groups": {}}
  for gi, gname in enumerate(gnames):
    m = grp == gi
    n = int(m.sum())
    if n == 0:
      continue
    nt = int((term & m).sum())
    nn_, nf = int((m & airk).sum()), int((m & ~airk).sum())
    res["groups"][gname] = {
      "steps": n, "air_frac": float((air & m).sum() / n), "near_air_step_frac": nn_ / n,
      "fail_per_s": nt / n * 50.0,
      "fail_per_s_near_air": float((term & m & airk).sum()) / max(nn_, 1) * 50.0,
      "fail_per_s_far": float((term & m & ~airk).sum()) / max(nf, 1) * 50.0,
      "fail_near_air_frac": float((term & m & airk).sum() / max(nt, 1)),
    }

  # ---- 2. world-model k-step error ---------------------------------------------------------------
  k = args.k
  ar_t = torch.arange(T - k + 1, device=dev)
  # Valid window: no episode end inside except possibly at the last step.
  dcum = torch.cat([torch.zeros(E, 1, device=dev), done.float().cumsum(1)], 1)
  bad = (dcum[:, ar_t + k - 1] - dcum[:, ar_t]) > 0
  air_w = (torch.cat([torch.zeros(E, 1, device=dev), air.float().cumsum(1)], 1)[:, ar_t + k] -
           torch.cat([torch.zeros(E, 1, device=dev), air.float().cumsum(1)], 1)[:, ar_t]) > 0
  ev, tv = (~bad).nonzero(as_tuple=True)
  e1s, eks, comp_acc = [], [], {}
  for i in range(0, ev.numel(), 16384):
    tot, comps = wm_errors(tr, ev[i : i + 16384], tv[i : i + 16384], k)
    e1s.append(tot[:, 0])
    eks.append(tot[:, -1])
    for kk, v in comps.items():
      comp_acc.setdefault(kk, []).append(v)
  e1, ek = torch.cat(e1s), torch.cat(eks)
  comp = {kk: torch.cat(v) for kk, v in comp_acc.items()}
  wg, wa = grp[ev, tv], air_w[ev, tv]
  for gi, gname in enumerate(gnames):
    for tag, mm in (("all", wg == gi), ("air", (wg == gi) & wa), ("ground", (wg == gi) & ~wa)):
      if int(mm.sum()) == 0 or gname not in res["groups"]:
        continue
      res["groups"][gname][f"wm_{tag}"] = {
        "n": int(mm.sum()), "e1": float(e1[mm].mean()), f"e{k}": float(ek[mm].mean()),
        f"e{k}_p90": float(torch.quantile(ek[mm][:100000], 0.9)),
        **{f"e{k}_{kk}": float(v[mm].mean()) for kk, v in comp.items()},
      }

  # ---- 3. imagination bias ------------------------------------------------------------------------
  g = cfg.gamma
  ts = torch.arange(0, T - H + 1, args.start_stride, device=dev)
  tcum = torch.cat([torch.zeros(E, 1, device=dev), timeout.float().cumsum(1)], 1)
  ok = (tcum[:, ts + H] - tcum[:, ts]) == 0  # no clip-end / episode time-out within H (imagination would run on)
  se, st_idx = ok.nonzero(as_tuple=True)
  st = ts[st_idx]
  # Real H-step return: rewards up to and including the first failure.
  R = B.reward[:, :T]
  idx = st[:, None] + torch.arange(H, device=dev)[None, :]
  r_w = R[se[:, None], idx]
  f_w = term[se[:, None], idx].float()
  alive_before = torch.cat([torch.ones(len(se), 1, device=dev), 1.0 - f_w.cumsum(1).clamp(max=1.0)[:, :-1]], 1)
  J_real = (r_w * alive_before * (g ** torch.arange(H, device=dev, dtype=torch.float32))[None, :]).sum(1)
  surv_real = 1.0 - f_w.max(1).values
  air_s = (torch.cat([torch.zeros(E, 1, device=dev), air.float().cumsum(1)], 1)[se, st + H] -
           torch.cat([torch.zeros(E, 1, device=dev), air.float().cumsum(1)], 1)[se, st]) > 0
  Ji, ci = [], []
  for i in range(0, se.numel(), 8192):
    J_, c_ = imagine(tr, se[i : i + 8192], st[i : i + 8192], H)
    Ji.append(J_)
    ci.append(c_)
  J_im, surv_im = torch.cat(Ji), torch.cat(ci)
  sg = grp[se, st]
  rmax = cfg.heads.reward_max_per_step
  for gi, gname in enumerate(gnames):
    for tag, mm in (("all", sg == gi), ("air", (sg == gi) & air_s), ("ground", (sg == gi) & ~air_s)):
      if int(mm.sum()) == 0 or gname not in res["groups"]:
        continue
      fail_m = mm & (surv_real < 0.5)
      res["groups"][gname][f"imag_{tag}"] = {
        "n": int(mm.sum()), "surv_real": float(surv_real[mm].mean()), "surv_imag": float(surv_im[mm].mean()),
        "ret_real": float(J_real[mm].mean() / (H * rmax)), "ret_imag": float(J_im[mm].mean() / (H * rmax)),
        # On starts the simulator failed from: how much survival the world model still predicts.
        "surv_imag_when_real_failed": float(surv_im[fail_m].mean()) if int(fail_m.sum()) else None,
        "n_real_failed": int(fail_m.sum()),
      }
  res["wall_s"] = time.time() - t0
  print(json.dumps(res, indent=1))
  if args.out:
    with open(args.out, "w") as f:
      json.dump(res, f, indent=1)


if __name__ == "__main__":
  main()
