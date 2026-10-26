"""CPU-only scan of checkpoints: normalizers, weight norms, Adam moments, sampler arrays, synthetic policy probe."""
import os, sys, json, numpy as np, torch
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.set_num_threads(8)
sys.path.insert(0, "/home/ubuntu/supermimic/supermimic_repro_v2wd")
from supermimic.policy import Policy
OUT = "/data2/supermimic/diag_1024/taskC/ck"
os.makedirs(OUT, exist_ok=True)
A = "/data2/supermimic/logs/2026-10-23_13-57-03_g1v2_ddp8_ws86k"
B = "/data2/supermimic/logs/2026-10-23_19-29-25_g1v2_ddp8_ws86k_b"
paths = [("86000", "/data2/supermimic/warmstart_g1v2_scratch_model_86000.pt")]
paths += [(str(i), f"{A}/model_{i}.pt") for i in (88000, 92000, 96000, 100000, 104000)]
paths += [(str(i), f"{B}/model_{i}.pt") for i in range(108000, 214000, 4000)]
paths += [(str(i), f"{B}/model_{i}.pt") for i in (162000, 166000, 186000)]
SAVE_BINS = {"86000", "92000", "104000", "116000", "140000", "160000", "164000", "168000", "188000", "212000"}
gen = torch.Generator().manual_seed(0)
probe = None
ref = None
res = {}
for tag, p in paths:
    if not os.path.exists(p):
        print("missing", p); continue
    ck = torch.load(p, map_location="cpu", weights_only=False)
    d = ck["dims"]
    pol = ck["policy"]
    out = {"it": ck["it"], "lr_scale": ck.get("lr_scale")}
    arr = {}
    for m in ("policy", "wm"):
        for k, v in ck[m].items():
            if "norm" in k:
                arr[f"{m}.{k}"] = v.numpy()
    for m in ("policy", "critic", "wm"):
        out[m + "_wnorm"] = {k: float(v.float().norm()) for k, v in ck[m].items() if "norm" not in k}
    # Adam state
    for o, m in (("opt_pi", "policy"), ("opt_v", "critic"), ("opt_wm", "wm")):
        sd = ck[o]
        names = [k for k in ck[m].keys() if "norm" not in k]
        st = sd["state"]
        ea = [float(st[i]["exp_avg"].norm()) for i in sorted(st)]
        es = [float(st[i]["exp_avg_sq"].sqrt().mean()) for i in sorted(st)]
        out[o] = {"step": float(st[sorted(st)[0]]["step"]), "lr": sd["param_groups"][0]["lr"], "exp_avg_norm": ea, "sqrt_v_mean": es,
                  "names": names if len(names) == len(ea) else None}
    if tag in SAVE_BINS:
        arr["bin_G"] = ck["bin_G"].numpy(); arr["bin_visits"] = ck["bin_visits"].numpy()
    np.savez_compressed(f"{OUT}/ck_{tag}.npz", **arr)
    # synthetic policy probe: fixed raw inputs drawn from the 86000 normalizer, pushed through each checkpoint's own normalizer
    net = Policy(d["proprio"], d["hist"], d["goal_frame"], d["K"], d["nj"])
    net.load_state_dict(pol); net.eval()
    if probe is None:
        om, osd = pol["obs_norm.mean"], pol["obs_norm.var"].sqrt()
        gm, gsd = pol["goal_norm.mean"], pol["goal_norm.var"].sqrt()
        Nb = 4096
        probe = (om + osd * torch.randn(Nb, d["hist"], d["proprio"], generator=gen), gm + gsd * torch.randn(Nb, d["K"], d["goal_frame"], generator=gen))
    with torch.no_grad():
        a = net(*probe)
        on, gn = net.normalize_inputs(*probe)
        z = torch.tanh(net.encoder(gn.reshape(gn.shape[0], -1)))
    out["probe"] = {"a_abs_mean": float(a.abs().mean()), "a_std_batch_mean": float(a.std(0).mean()), "a_mean_abs": float(a.mean(0).abs().mean()),
                    "z_abs_mean": float(z.abs().mean()), "z_sat_frac": float((z.abs() > 0.99).float().mean()),
                    "a_std_per_joint": a.std(0).tolist(), "a_mean_per_joint": a.mean(0).tolist()}
    # sensitivity: d a / d (normalized obs of the latest frame) and d a / d (normalized goal), via autograd on 512 samples
    x_o = on[:512].clone().requires_grad_(True); x_g = gn[:512].clone().requires_grad_(True)
    zz = torch.tanh(net.encoder(x_g.reshape(512, -1)))
    aa = net.decoder(torch.cat([zz, x_o.reshape(512, -1)], -1))
    gsum_o, gsum_g = 0.0, 0.0
    v = torch.randn(512, aa.shape[1], generator=gen)
    go, gg = torch.autograd.grad((aa * v).sum(), [x_o, x_g])
    out["probe"]["jvp_obs_last"] = float(go[:, -1].norm(dim=-1).mean()); out["probe"]["jvp_obs_all"] = float(go.reshape(512, -1).norm(dim=-1).mean())
    out["probe"]["jvp_goal"] = float(gg.reshape(512, -1).norm(dim=-1).mean())
    if ref is None:
        ref = a
    out["probe"]["a_rms_diff_vs_86000"] = float((a - ref).pow(2).mean().sqrt())
    np.save(f"{OUT}/probe_a_{tag}.npy", a.numpy().astype(np.float32))
    res[tag] = out
    print(tag, ck["it"], json.dumps(out["probe"])[:300], flush=True)
    del ck
json.dump(res, open(f"{OUT}/scan.json", "w"))
print("done")
