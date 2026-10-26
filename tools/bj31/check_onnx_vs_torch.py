"""Eval ONNX (stamped) vs the PyTorch policy rebuilt straight from the training checkpoint (CPU).
  python check_onnx_vs_torch.py <ckpt.pt> <eval.onnx> [n_batches]
Inputs: normalizer mean + std * N(0,1) (realistic scale) and plain N(0, 0.5); batch 1 (the ONNX graph is batch 1).
Also checks finiteness of weights / normalizers and that the stamped metadata matches the checkpoint dims."""
import json
import sys

import numpy as np
import onnxruntime as ort
import torch

from supermimic.policy import Policy

torch.set_num_threads(2)
ck_path, onnx_path = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 200
ck = torch.load(ck_path, map_location="cpu", weights_only=False)
d = ck["dims"]
pol = Policy(d["proprio"], d["hist"], d["goal_frame"], d["K"], d["nj"]).eval()
pol.load_state_dict(ck["policy"])
bad = [k for k, v in ck["policy"].items() if torch.is_tensor(v) and v.is_floating_point() and not torch.isfinite(v).all()]
print(f"[check] ckpt it {ck['it']} dims {d}; non-finite policy tensors: {bad or 'none'}")
print(f"[check] obs_norm count {float(pol.obs_norm.count):.3g} std range [{pol.obs_norm.var.sqrt().min():.3g}, {pol.obs_norm.var.sqrt().max():.3g}]; "
      f"goal_norm std range [{pol.goal_norm.var.sqrt().min():.3g}, {pol.goal_norm.var.sqrt().max():.3g}]")
sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"], sess_options=(lambda o: (setattr(o, "intra_op_num_threads", 2), o)[1])(ort.SessionOptions()))
meta = sess.get_modelmeta().custom_metadata_map
assert int(meta["iteration"]) == ck["it"], (meta["iteration"], ck["it"])
assert len(json.loads(meta["joint_names"])) == d["nj"]
shapes = {i.name: i.shape for i in sess.get_inputs()}
print(f"[check] onnx inputs {shapes}; metadata iteration {meta['iteration']}, body_names {len(json.loads(meta['body_names']))}, damping mode {meta.get('eval_damping_mode')}")
g = torch.Generator().manual_seed(0)
om, osd = pol.obs_norm.mean, pol.obs_norm.var.clamp_min(1e-8).sqrt()
gm, gsd = pol.goal_norm.mean, pol.goal_norm.var.clamp_min(1e-8).sqrt()
worst = {"realistic": 0.0, "randn0.5": 0.0}
amax = 0.0
for i in range(n):
    if i % 2 == 0:
        kind = "realistic"
        o = om + osd * torch.randn(1, d["hist"], d["proprio"], generator=g)
        gl = gm + gsd * torch.randn(1, d["K"], d["goal_frame"], generator=g)
    else:
        kind = "randn0.5"
        o = 0.5 * torch.randn(1, d["hist"], d["proprio"], generator=g)
        gl = 0.5 * torch.randn(1, d["K"], d["goal_frame"], generator=g)
    with torch.no_grad():
        ref = pol(o, gl).numpy()
    out = sess.run(None, {"obs_hist": o.numpy().astype(np.float32), "goal": gl.numpy().astype(np.float32)})[0]
    assert np.isfinite(out).all()
    worst[kind] = max(worst[kind], float(np.abs(out - ref).max()))
    amax = max(amax, float(np.abs(ref).max()))
print(f"[check] {n} batches: max |onnx - torch| realistic {worst['realistic']:.2e}, randn0.5 {worst['randn0.5']:.2e} (max |action| {amax:.2f}) "
      + ("OK" if max(worst.values()) < 1e-4 else "FAIL"))
