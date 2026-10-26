# CPU-only probe: does the learned WM extrapolate explosively once its (unclamped) normalized inputs leave the data range?
import sys, json, math
sys.path.insert(0, "/home/ubuntu/supermimic/supermimic_repro_v2wd")
import torch
torch.set_num_threads(16)
from supermimic.world_model import DynamicsModel, WorldModelCfg
from supermimic.math_utils import quat_apply
RUN = "/data2/supermimic/logs/2026-10-23_19-29-25_g1v2_ddp8_ws86k_b"
J, XI = 29, 33
torch.manual_seed(0)
for it in sys.argv[1:]:
    ck = torch.load(f"{RUN}/model_{it}.pt", map_location="cpu", weights_only=False)
    wcfg = ck["cfg"]["wm"]; cfg = WorldModelCfg(hidden=tuple(wcfg["hidden"]), k=wcfg["k"])
    wm = DynamicsModel(J, XI, 0.02, cfg); wm.load_state_dict(ck["wm"]); wm.eval()
    m, sd = wm.in_norm.mean, wm.in_norm.std
    FD = wm.feat_dim
    z, v_h, w_b, q, qd = m[0], m[7:10], m[10:13], m[13:13+J], m[13+J:13+2*J]
    a0 = m[FD:]; a_sd = sd[FD:]; qd_sd = sd[13+J:13+2*J]; q_sd = sd[13:13+J]
    s0 = torch.cat([torch.tensor([0.0, 0.0]), z[None], torch.tensor([1.0, 0, 0, 0]), v_h, w_b, q, qd])
    print(f"=== model_{it}: in_norm std  q {q_sd.mean():.3f}  qd {qd_sd.mean():.3f}  a {a_sd.mean():.3f}; tgt_norm std dqd {wm.tgt_norm.std[6:6+J].mean():.3f}")
    B = 256
    with torch.no_grad():
        for mode in ("qd", "act"):
            for k in (0.0, 1.0, 3.0, 10.0, 30.0, 100.0):
                s = s0[None].repeat(B, 1)
                a = a0[None].repeat(B, 1)
                sign = torch.randn(B, J).sign()
                if mode == "qd":
                    s[:, 13+J:] += k * qd_sd * sign
                else:
                    a = a + k * a_sd * sign
                xi = torch.zeros(B, XI)
                xmax, sq, mqd, zz = [], [], [], []
                for t in range(32):
                    x = wm.in_norm.normalize(torch.cat([wm.features(s), a], -1))
                    xmax.append(x.abs().amax(-1).median().item())
                    s = wm.step(s, a, xi)
                    sq.append(s[:, 13:13+J].abs().sum(-1)); mqd.append(s[:, 13+J:].abs().amax(-1)); zz.append(s[:, 2])
                sq = torch.stack(sq); mqd = torch.stack(mqd); zz = torch.stack(zz)
                fin = torch.isfinite(sq[-1])
                def q50(x): x = x[torch.isfinite(x)]; return float(x.median()) if x.numel() else float("nan")
                def qmax(x): x = x[torch.isfinite(x)]; return float(x.max()) if x.numel() else float("nan")
                print(f"  {mode} +{k:5.0f} sd | median sum|q| t=1 {q50(sq[0]):9.3g} t=8 {q50(sq[7]):9.3g} t=16 {q50(sq[15]):9.3g} t=32 {q50(sq[31]):9.3g} (max {qmax(sq[31]):9.3g}) "
                      f"| max|qd| t32 med {q50(mqd[31]):9.3g} | root z t32 med {q50(zz[31]):8.3g} | median max|x_norm| t0 {xmax[0]:6.1f} t31 {xmax[-1]:9.3g} | finite {fin.float().mean():.2f}")
