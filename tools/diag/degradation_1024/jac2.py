import os, sys, json, numpy as np, torch
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.set_num_threads(16)
sys.path.insert(0, "/home/ubuntu/supermimic/supermimic_repro_v2wd")
from supermimic.policy import Policy
E = "/data2/supermimic/eval_export/bundles"
B = {"88000": f"{E}/g1v2_ddp8_ws86k_88000", "104000": f"{E}/g1v2_ddp8_ws86k_104000"}
for it in (116000, 140000, 164000, 168000, 188000, 212000):
    B[str(it)] = f"{E}/g1v2_ddp8_ws86k_b_{it}"
S = {k: np.load(f"/data2/supermimic/diag_1024/taskC/states_{k}.npz") for k in ("116000", "212000")}
rng = np.random.default_rng(0)
idx = {k: rng.choice(len(v["H"]), min(1500, len(v["H"])), replace=False) for k, v in S.items()}
res = {}
for it, b in B.items():
    m = json.load(open(f"{b}/meta.json")); d = m["dims"]; nj = d["nj"]
    net = Policy(d["proprio"], d["hist"], d["goal_frame"], d["K"], d["nj"]); net.load_state_dict(torch.load(f"{b}/policy.pt", map_location="cpu")); net.eval()
    out = {}
    for sk, s in S.items():
        oh = torch.as_tensor(s["H"][idx[sk]]).requires_grad_(True); g = torch.as_tensor(s["G"][idx[sk]])
        rows = []
        a = net(oh, g)
        for i in range(nj):
            gr = torch.autograd.grad(a[:, i].sum(), oh, retain_graph=True)[0]
            rows.append(gr[:, -1])  # [N, P] grads wrt latest frame
        Jl = torch.stack(rows, 1)  # [N, nj, P]
        Ja = Jl[:, :, 64:93]; Jqd = Jl[:, :, 29:58]; Jq = Jl[:, :, 0:29]; Jw = Jl[:, :, 58:61]
        ev = torch.linalg.eigvals(Ja)
        k = ev.abs().argmax(-1)
        lead = ev[torch.arange(len(ev)), k]
        out[sk] = dict(Ja_fro=float(Ja.norm(dim=(1, 2)).mean()), specrad=float(ev.abs().max(-1).values.mean()), lead_neg_real_frac=float((lead.real < 0).float().mean()),
                       Ja_diag=float(torch.diagonal(Ja, dim1=1, dim2=2).mean()), Jqd_fro=float(Jqd.norm(dim=(1, 2)).mean()), Jq_fro=float(Jq.norm(dim=(1, 2)).mean()), Jw_fro=float(Jw.norm(dim=(1, 2)).mean()),
                       Jqd_diag_legs=float(torch.diagonal(Jqd, dim1=1, dim2=2)[:, :15].mean()), Jq_diag_legs=float(torch.diagonal(Jq, dim1=1, dim2=2)[:, :15].mean()),
                       Ja_legs_fro=float(Ja[:, :15, :15].norm(dim=(1, 2)).mean()), Jqd_legs_fro=float(Jqd[:, :15, :15].norm(dim=(1, 2)).mean()))
        a_n = a.detach()
        out[sk]["a_abs"] = float(a_n.abs().mean())
    res[it] = out
    print(it, " | ".join(f"{sk}-states: |dA/dAprev| {o['Ja_fro']:.3f} (legs {o['Ja_legs_fro']:.3f}) specrad {o['specrad']:.3f} neg-real-lead {o['lead_neg_real_frac']:.2f} diag {o['Ja_diag']:+.3f} |dA/dqd| {o['Jqd_fro']:.3f} (legs {o['Jqd_legs_fro']:.3f}) qd-diag-legs {o['Jqd_diag_legs']:+.4f} |dA/dq| {o['Jq_fro']:.2f} q-diag-legs {o['Jq_diag_legs']:+.3f} |dA/dw| {o['Jw_fro']:.3f}" for sk, o in out.items()), flush=True)
json.dump(res, open("/data2/supermimic/diag_1024/taskC/jac2.json", "w"))
