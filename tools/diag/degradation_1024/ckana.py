import json, numpy as np
C = "/tmp/claude-1000/-home-user-Workspace-Noetix-GMT/7380f59e-8ed0-40ea-9060-69c5f5a5a1ef/scratchpad/supermimic/diag_1024/taskC"
S = json.load(open(f"{C}/ck/scan.json"))
tags = sorted(S, key=int)
n0 = np.load(f"{C}/ck/ck_86000.npz")
print("normalizers: count, max |dmean|/std0, max/min std ratio vs 86000, median std ratio")
for nm in ["policy.obs_norm", "policy.goal_norm", "wm.in_norm", "wm.tgt_norm", "wm.err_norm"]:
    print(nm)
    m0, v0 = n0[nm + ".mean"], n0[nm + ".var"]; s0 = np.sqrt(v0)
    for t in tags:
        n = np.load(f"{C}/ck/ck_{t}.npz")
        m, v, c = n[nm + ".mean"], n[nm + ".var"], float(n[nm + ".count"])
        s = np.sqrt(v)
        dm = np.abs(m - m0) / np.maximum(s0, 1e-4)
        r = s / np.maximum(s0, 1e-6)
        if t in ("86000", "88000", "92000", "104000", "116000", "140000", "160000", "168000", "188000", "212000"):
            print(f"  {t:>6s} count {c:.4g}  max|dmean|/s0 {dm.max():.4f} (dim {dm.argmax()})  std ratio min {r.min():.4f} max {r.max():.4f} med {np.median(r):.5f}  min std {s.min():.3g}")
# weight norms
print("\npolicy weight norms (rel. change vs 86000)")
ks = list(S["86000"]["policy_wnorm"].keys())
w0 = S["86000"]["policy_wnorm"]
print("tag    " + " ".join(k.replace("weight", "W").replace("bias", "b")[-12:] for k in ks if "weight" in k))
for t in tags:
    w = S[t]["policy_wnorm"]
    print(f"{t:>6s} " + " ".join(f"{w[k] / w0[k] - 1:+12.4f}" for k in ks if "weight" in k))
print("\ncritic/wm weight norms rel change")
for t in tags:
    c, c0 = S[t]["critic_wnorm"], S["86000"]["critic_wnorm"]; wm, wm0 = S[t]["wm_wnorm"], S["86000"]["wm_wnorm"]
    print(f"{t:>6s} critic " + " ".join(f"{c[k] / c0[k] - 1:+.4f}" for k in c if "weight" in k) + " | wm " + " ".join(f"{wm[k] / wm0[k] - 1:+.4f}" for k in wm if "weight" in k))
print("\nAdam (policy): step, lr, mean sqrt(v) of each weight (x1e-4), exp_avg norm of last layer")
for t in tags:
    o = S[t]["opt_pi"]; names = o["names"]
    sv = [f"{x * 1e4:.3f}" for x, nmm in zip(o["sqrt_v_mean"], names) if "weight" in nmm]
    print(f"{t:>6s} step {o['step']:.0f} lr {o['lr']:.3g} sqrt_v " + " ".join(sv) + f"  |m| last {o['exp_avg_norm'][-2]:.3g}")
print("\nprobe (fixed synthetic inputs)")
for t in tags:
    p = S[t]["probe"]
    print(f"{t:>6s} |a| {p['a_abs_mean']:.4f} std_b {p['a_std_batch_mean']:.4f} |mean a| {p['a_mean_abs']:.4f} |z| {p['z_abs_mean']:.4f} jvp_obs_last {p['jvp_obs_last']:.4f} jvp_obs_all {p['jvp_obs_all']:.4f} jvp_goal {p['jvp_goal']:.4f} rms_diff_vs_86k {p['a_rms_diff_vs_86000']:.4f}")
