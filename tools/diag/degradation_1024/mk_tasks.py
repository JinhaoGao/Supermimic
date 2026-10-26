import json, os, random
B = "/data2/supermimic/logs/2026-10-23_19-29-25_g1v2_ddp8_ws86k_b"
names = [l.strip() for l in open(f"{B}/eval_clip_names.txt") if l.strip()]
paths = {os.path.basename(l.strip()): l.strip() for l in open(f"{B}/motion_files.txt") if l.strip()}
ev = [json.loads(l) for l in open(f"{B}/eval.jsonl")]
f212 = next(e for e in ev if e["it"] == 212000)["failed_clips"]
rnd = random.Random(0)
others = [n for n in names if n not in f212]
setB = rnd.sample(others, 90)
clips = [("F212", n) for n in f212] + [("R", n) for n in setB]
E = "/data2/supermimic/eval_export/bundles"
bund = {"88000": f"{E}/g1v2_ddp8_ws86k_88000", "116000": f"{E}/g1v2_ddp8_ws86k_b_116000", "164000": f"{E}/g1v2_ddp8_ws86k_b_164000",
        "168000": f"{E}/g1v2_ddp8_ws86k_b_168000", "188000": f"{E}/g1v2_ddp8_ws86k_b_188000", "212000": f"{E}/g1v2_ddp8_ws86k_b_212000"}
tasks = []
for it, b in bund.items():
    conds = ["clean", "act"] + (["obs"] if it in ("116000", "212000") else [])
    for c in conds:
        for k, (grp, n) in enumerate(clips):
            tasks.append(dict(it=it, bundle=b, clip=paths[n], name=n, grp=grp, cond=c, seed=k))
os.makedirs("/data2/supermimic/diag_1024/taskC/roll", exist_ok=True)
json.dump(tasks, open("/data2/supermimic/diag_1024/taskC/roll/tasks.json", "w"))
print(len(f212), len(setB), len(tasks), all(os.path.exists(t["clip"]) for t in tasks))
