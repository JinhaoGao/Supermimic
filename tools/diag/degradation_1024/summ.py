import json
d = json.load(open("/home/ubuntu/work/NOetix_GMT_eval/storage/runs/7943ef8e9d054cea82f84e3217fb9bb6/evaluation_summary.json"))
def show(x, pre="", depth=0):
    if depth > 4: return
    if isinstance(x, dict):
        for k in list(x)[:25]:
            v = x[k]
            desc = type(v).__name__ + (f"[{len(v)}]" if hasattr(v, "__len__") and not isinstance(v, str) else (f"={v!r}"[:80]))
            print("  " * depth + f"{pre}{k}: {desc}")
            if isinstance(v, (dict, list)): show(v, "", depth + 1)
        if len(x) > 25: print("  " * depth + f"... {len(x)} keys")
    elif isinstance(x, list) and x:
        print("  " * depth + f"[0] {type(x[0]).__name__}")
        show(x[0], "", depth + 1)
show(d)
