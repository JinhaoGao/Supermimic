import sys, torch
a = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
b = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
def walk(x, y, path, out):
  if isinstance(x, dict):
    for k in x:
      if k == "cfg": continue
      if k not in y: out.append(f"{path}/{k}: missing in b"); continue
      walk(x[k], y[k], f"{path}/{k}", out)
  elif isinstance(x, (list, tuple)):
    for i, (u, v) in enumerate(zip(x, y)): walk(u, v, f"{path}[{i}]", out)
  elif torch.is_tensor(x):
    if x.shape != y.shape or x.dtype != y.dtype or not torch.equal(x.reshape(-1).view(torch.uint8) if x.dtype != torch.bool else x, y.reshape(-1).view(torch.uint8) if y.dtype != torch.bool else y):
      d = (x.double() - y.double()).abs().max().item() if x.shape == y.shape else float("nan")
      out.append(f"{path}: DIFF maxabs={d}")
  else:
    if x != y: out.append(f"{path}: {x!r} != {y!r}")
out = []
walk(a, b, "", out)
n = sum(1 for _ in [0])
print("IDENTICAL (all tensors bitwise equal, cfg ignored)" if not out else "\n".join(out[:50]))
