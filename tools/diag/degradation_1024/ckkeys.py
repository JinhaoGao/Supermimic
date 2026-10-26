import torch
torch.set_num_threads(4)
ck = torch.load("/data2/supermimic/logs/2026-10-23_19-29-25_g1v2_ddp8_ws86k_b/model_212000.pt", map_location="cpu", weights_only=False)
print(ck.keys()); print(ck["dims"], ck["it"], ck["env_steps"], ck["lr_scale"])
for m in ("policy", "critic", "wm"):
    for k, v in ck[m].items():
        print(m, k, tuple(v.shape), v.dtype, (float(v.float().norm()) if v.numel() > 1 else float(v)))
print({k: v for k, v in ck["cfg"].items() if k not in ("motion_files",)})
