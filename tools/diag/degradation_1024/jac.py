import os, sys, json, numpy as np, torch
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.set_num_threads(8)
sys.path.insert(0, "/home/ubuntu/supermimic/supermimic_repro_v2wd")
from supermimic.policy import Policy
E = "/data2/supermimic/eval_export/bundles"
B = {"88000": f"{E}/g1v2_ddp8_ws86k_88000", "104000": f"{E}/g1v2_ddp8_ws86k_104000"}
for it in (116000, 140000, 164000, 168000, 188000, 212000):
    B[str(it)] = f"{E}/g1v2_ddp8_ws86k_b_{it}"
gen = torch.Generator().manual_seed(1)
X = None
res = {}
for it, b in B.items():
    m = json.load(open(f"{b}/meta.json")); d = m["dims"]
    net = Policy(d["proprio"], d["hist"], d["goal_frame"], d["K"], d["nj"]); net.load_state_dict(torch.load(f"{b}/policy.pt", map_location="cpu")); net.eval()
    om, osd = net.obs_norm.mean, net.obs_norm.std; gm, gsd = net.goal_norm.mean, net.goal_norm.std
    if X is None:
        N = 256
        X = (om + osd * 0.5 * torch.randn(N, d["hist"], d["proprio"], generator=gen), gm + gsd * 0.5 * torch.randn(N, d["K"], d["goal_frame"], generator=gen))
        X0 = (om.expand(1, d["hist"], -1).clone(), gm.expand(1, d["K"], -1).clone())
    nj = d["nj"]
    out = {}
    for lab, (oh, g) in (("mean", X0), ("rand", X)):
        oh = oh.clone().requires_grad_(True)
        Js_a, Js_qd, Js_q = [], [], []
        for i in range(nj):
            a = net(oh, g)
            gr = torch.autograd.grad(a[:, i].sum(), oh)[0]  # [N, hist, P] raw-unit gradient
            Js_a.append(gr[:, -1, 64:93]); Js_qd.append(gr[:, -1, 29:58]); Js_q.append(gr[:, -1, 0:29])
        Ja = torch.stack(Js_a, 1)  # [N, out, in] d a_t / d a_{t-1}
        Jqd = torch.stack(Js_qd, 1); Jq = torch.stack(Js_q, 1)
        ev = torch.linalg.eigvals(Ja).abs().max(-1).values
        out[lab] = dict(prevA_specrad=float(ev.mean()), prevA_diag_mean=float(torch.diagonal(Ja, dim1=1, dim2=2).mean()),
                        prevA_diag_legs=torch.diagonal(Ja, dim1=1, dim2=2)[:, :12].mean(0).tolist(),
                        qd_diag_legs=torch.diagonal(Jqd, dim1=1, dim2=2)[:, :15].mean(0).tolist(),
                        q_diag_legs=torch.diagonal(Jq, dim1=1, dim2=2)[:, :15].mean(0).tolist(),
                        Jqd_fro=float(Jqd.norm(dim=(1, 2)).mean()), Jq_fro=float(Jq.norm(dim=(1, 2)).mean()), Ja_fro=float(Ja.norm(dim=(1, 2)).mean()))
    res[it] = out
    r = out["rand"]
    print(it, f"rand: specrad(dA/dA_prev) {r['prevA_specrad']:.3f} diag {r['prevA_diag_mean']:+.3f} |J_qd| {r['Jqd_fro']:.3f} |J_q| {r['Jq_fro']:.3f} |J_a| {r['Ja_fro']:.3f} | mean-input specrad {out['mean']['prevA_specrad']:.3f}", flush=True)
json.dump(res, open("/data2/supermimic/diag_1024/taskC/jac.json", "w"))
