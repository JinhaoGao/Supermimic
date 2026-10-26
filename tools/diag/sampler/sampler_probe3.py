"""Read-only reset-sampler probe for the r15ws run (CPU). Rebuilds bins like MotionLibrary, the source/name prior like
reset_prior, and the clamped softmax like sampling_probabilities (temp 25, g_max 10, unvisited 5, mix .5/.4/.1 with the
history share approximated by the current softmax). Reports reset mass + exposure per cluster.
  RUN_DIR=... CDIR=... python sampler_probe3.py <v4_outcomes.tsv> <it> [<it0 for exposure delta>]"""
import json, os, re, sys
import numpy as np, torch
torch.set_num_threads(4)
RUN, CDIR = os.environ["RUN_DIR"], os.environ["CDIR"]
TEMP, GMAX, BL = 25.0, 10.0, 50
cfg = json.load(open(f"{RUN}/train_cfg.json"))
clips_all = json.load(open(f"{CDIR}/clips.json"))
clips = [c for c in clips_all if c.get("ok", True) and c["end"] > c["start"] + 1]
names = [c["name"] for c in clips]
bclip = []
for i, c in enumerate(clips):
    bclip += [i] * len(range(c["start"], c["end"], BL))
bclip = np.array(bclip); NB = len(bclip)
# prior (reset_prior)
kws = sorted(((k, float(v)) for k, v in cfg["source_weights"]), key=lambda kv: -len(kv[0]))
nws = [(k.lower(), float(v)) for k, v in cfg["name_weights"]]
def sw(n):
    s = n.split("__", 1)[-1]
    for k, v in kws:
        if s == k or s.startswith(k + "_"): return v
    return 1.0
wclip = np.array([max(sw(n), max([v for k, v in nws if k in n.lower()], default=1.0)) for n in names])
prior = wclip[bclip]; prior = prior / prior.sum()
low = [n.lower() for n in names]
def has(*ks): return np.array([any(k in n for k in ks) for n in low])
src = np.array([n.split("__", 1)[-1].split("_")[0] for n in names])
v4 = [l.rstrip("\n").split("\t") for l in open(sys.argv[1])][1:]
def stems(pred):
    return [s.lower() for m, s, o, ld in v4 if pred(o, ld)]
def match(st):
    st = [s for s in st if len(s) >= 6]
    return np.array([any(s in n for s in st) for n in low])
clusters = {
    "ALL": np.ones(len(names), bool),
    "name_weights hit": np.array([any(k in n for k, _ in nws) for n in low]),
    "aerial (tengkong/erqijiao/xuanfeng/xuanzi/tiaogao/zhongtiao/zhupao/waibailian/kongfan/shoufan)": has("tengkong","erqijiao","xuanfeng","xuanzi","tiaogao","zhongtiao","zhupaotiaoyue","waibailian","kongfan","shoufan"),
    "getup (getup/get_up/getting_up/fallandgetup/ground_recovery/起身)": has("getup","get_up","getting_up","fallandgetup","ground_recovery","起身"),
    "lie/prone/supine (lie/lying/prone/supine/躺/俯卧)": has("lying","lie_","_lie","prone","supine","躺","俯卧"),
    "crawl/kneel (crawl/kneel/爬/跪)": has("crawl","kneel","爬","跪"),
    "breaking/street floor (breaking/街舞/dna_/headspin/windmill)": has("breaking","街舞","dna_","headspin","windmill"),
    "flip/acro (flip/acro/空翻/翻滚/手翻)": has("flip","acro","空翻","翻滚","手翻"),
    "GROUND_ (gmt_eval team getup/crawl)": has("ground_getup","ground_crawl"),
    "spins/turns (zhuanti/xuanfeng/xuanzi/turn/spin/pirouette/mohak/转)": has("zhuanti","xuanfeng","xuanzi","turn","spin","pirouette","mohak","转"),
    "side-lying getup analogues (side/侧躺/侧身 & getup-ish)": has("侧躺","侧身","side_lying","lie_side","getup_side","side_getup"),
    "sitting/crouch top-softmax family (sit/crouch/kneel sonic)": has("sitting","sit_","crouch","kneeling"),
    "v4 fail(ori/pos) @148k, stem match": match(stems(lambda o, ld: o in ("root_ori", "root_pos"))),
    "v4 fail(ori/pos) @148k & leader ok, stem match": match(stems(lambda o, ld: o in ("root_ori", "root_pos") and ld == "motion_complete")),
    "v4 fail EE @148k & leader ok, stem match": match(stems(lambda o, ld: o == "body_pos" and ld == "motion_complete")),
}
for s in ["noetix_kungfu", "sportsmeeting", "lafan1", "g1_all", "mixamo_cn", "sonic", "motiondecode", "hiphi"]:
    clusters[f"source {s}"] = np.array([n.split("__", 1)[-1].startswith(s) for n in names])
def load(it):
    ck = torch.load(f"{RUN}/model_{it}.pt", map_location="cpu", weights_only=False)
    return ck["bin_G"].double().numpy(), ck["bin_visits"].double().numpy()
it = sys.argv[2]
G, V = load(it)
assert G.shape[0] == NB, (G.shape, NB)
Gc = np.clip(G, 0, GMAX); Ge = np.where(V > 0, Gc, GMAX * 0.5)
z = -TEMP * Ge / GMAX; soft = np.exp(z - z.max()); soft /= soft.sum()
probs = 0.5 * prior + 0.5 * soft
uni = np.full(NB, 1.0 / NB)
dV = None
if len(sys.argv) > 3:
    G0, V0 = load(sys.argv[3]); dV = V - V0
print(f"clips {len(names)} bins {NB}  model_{it}  visited {np.mean(V > 0):.4f}  softmax top1 {soft.max():.4f}  eff.bins(soft) {1 / np.sum(soft ** 2):.0f}  eff.bins(probs) {1 / np.sum(probs ** 2):.0f}")
print(f"{'cluster':<96} {'clips':>6} {'bins%':>7} {'prior%':>7} {'soft%':>7} {'reset%':>7} {'xUnif':>6} {'G mean':>6} {'G<3%':>6} {'expo%':>7}")
for k, m in clusters.items():
    bm = m[bclip]
    ex = (dV[bm].sum() / dV.sum() * 100) if dV is not None else float("nan")
    vis = bm & (V > 0)
    print(f"{k[:96]:<96} {int(m.sum()):6d} {bm.mean() * 100:7.3f} {prior[bm].sum() * 100:7.3f} {soft[bm].sum() * 100:7.3f} {probs[bm].sum() * 100:7.3f} "
          f"{probs[bm].sum() / max(bm.mean(), 1e-12):6.1f} {Gc[vis].mean() if vis.any() else float('nan'):6.2f} {np.mean(Gc[vis] < 3) * 100 if vis.any() else float('nan'):6.1f} {ex:7.3f}")
# top softmax bins
order = np.argsort(-soft)[:15]
print("top softmax bins:")
for r in order:
    i = bclip[r]
    print(f"  soft {soft[r]:.5f} G {Gc[r]:.2f} V {V[r]:.0f}  {names[i][:100]}")
# per-clip lowest G among v4 fail-matched
