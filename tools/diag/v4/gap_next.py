"""v4 gap analysis: ours (bj31 pelvis-1.0 line) vs bj27 floor10 (ground exposure + root motion) vs GMT leader. Read-only.
Run with the noetix_eval python (numpy):  python -I gap_next.py out.json
Single checkpoints: O=ours 174000, C=ours C2 182000, B=bj27 floor10_r25 208000 (site), L=GMT cluster leader.
Multi-checkpoint groups (pass probability per segment) to separate systematic gaps from +-0.3-0.5 pp single-ckpt noise.
"""
import collections, csv, json, os, re, sys
import urllib.request
import numpy as np

ST = "/home/ubuntu/work/NOetix_GMT_eval/storage/runs"
OFF = "/home/ubuntu/work/supermimic_eval_onnx/evaluator_r18/runs"
DATA = "/home/ubuntu/work/NOetix_GMT_eval/data/g1_motionatlas_v4"
API = "http://127.0.0.1:8010/api/v1/leaderboard?mode=official&dataset_id=g1_motionatlas_v4&benchmark_id=kimodo_g1_motion_official_v3"
_site = sorted(os.listdir(ST))
def site(p): return os.path.join(ST, next(d for d in _site if d.startswith(p)))
def off(n): return os.path.join(OFF, n)

SINGLE = {"O": site("b43fdd83"), "C": site("621caa35"), "B": site("669a45a8"), "L": site("130b4a73")}
GROUPS = {
    # ours: pelvis 1.0 line, T arm 168k + 8-GPU main 170-182k + C2 arm 176-184k (13 ckpts)
    "ours_line": [site(p) for p in ["0d5abcd3", "3ea11545", "c630b5f4", "b43fdd83", "f01baa81", "65238b80", "b08012a4", "b52f2fd4",
                                    "28318ef4", "4ff629c3", "deda35d3", "621caa35", "4d89cfef"]],
    # bj27 floor10 line (ground exposure + root motion + pelvis 1.0 + fixes), offline == site (checked for 208000)
    "bj27_floor": [off(n) for n in ["g1v2_floor10_r24_196000", "g1v2_floor10_r24_200000", "g1v2_floor10_r25_204000",
                                     "g1v2_floor10_r25_208000", "g1v2_floor10_r26_212000", "g1v2_floor10_r26_216000"]],
    # bj27 root-motion ablation on top of pelvis 1.0 (same start r19 ctl 160000, lr 0.25), matched iterations 164/168/172/176k
    "bj27_ro": [site(p) for p in ["ec1408f9", "1a25cd7d", "f5c0b5d5", "7885d191"]],
    "bj27_roroot": [off(n) for n in ["g1v2_roroot_r20_164000", "g1v2_roroot_r20_168000", "g1v2_roroot_r21_172000", "g1v2_roroot_r21_176000"]],
    # bj27 ground-exposure ablation (same start roroot_r22 184000), matched 188/192k
    "bj27_roroot_r23": [off(n) for n in ["g1v2_roroot_r23_188000", "g1v2_roroot_r23_192000"]],
    "bj27_floor_r23": [off(n) for n in ["g1v2_rorootfloor_r23_188000", "g1v2_rorootfloor_r23_192000"]],
    "leader": [site("130b4a73")],
}

COLS = ["segment_id", "motion_name", "motion_path", "success", "termination_reason", "failure_step", "target_length"]
_cache = {}
def load(d):
    if d in _cache: return _cache[d]
    with open(os.path.join(d, "evaluation_per_segment.csv")) as f:
        rd = csv.reader(f); h = next(rd); ix = [h.index(c) for c in COLS]
        out = {}
        for row in rd:
            r = {c: row[i] for c, i in zip(COLS, ix)}
            r["ok"] = r["success"] == "True"
            out[r["motion_name"]] = r   # motion_name is unique (1 segment per motion file); segment_id order may differ offline
    assert len(out) == 3134, (d, len(out))
    _cache[d] = out
    return out

def src(r): return r["motion_path"].split("/g1_motionatlas_v4/")[1].split("/")[0]

K_GET = ["getup", "get_up", "getting_up", "stand_up", "standup", "fallandgetup", "ground_recovery", "prone", "supine", "起身", "躺", "仰卧", "趴"]
R_GET = re.compile(r"(?<![a-z])(lie|lying)")
K_FLOOR = ["breaking", "windmill", "headspin", "freeze", "flare", "hspu", "handstand", "倒立", "撑地", "撕叉", "单手撑", "floor_support",
           "floor_rolling", "街舞", "jiewu", "gaonandong", "gaonandudong"]
K_FLIP = ["flip", "kongfan", "shoufan", "cekong", "空翻", "手翻", "滚翻", "翻滚", "过肩翻", "摆腿翻", "acro", "handspring", "cartwheel",
          "somersault", "weibosite", "韦伯斯特", "liyudating", "鲤鱼打挺", "houqiao", "qianqiao", "jiqiaozuhe", "技巧", "jianzi"]
R_FLIP = re.compile(r"(?<![a-z])roll")
K_CRAWL = ["crawl", "匍匐", "爬行", "kneel", "跪"]
K_JUMP = ["jump", "tiao", "跳", "tengkong", "xuanfeng", "xaunfeng", "xuanzi", "erqijiao", "danpaijiao", "feishen", "vault", "leap"]
R_JUMP = re.compile(r"(?<![a-z])hop")
R_SIT = re.compile(r"((?<![a-z])sit|chair|vehicle|sofa|bench|stool|坐|zuozi)")
def cluster(name):
    n = name.lower()
    if name.startswith("GROUND_"): return "getup/lie/ground"
    if any(k in n for k in K_GET) or R_GET.search(n): return "getup/lie/ground"
    if any(k in n for k in K_FLOOR): return "breaking/floor"
    if any(k in n for k in K_FLIP) or R_FLIP.search(n): return "flip/acro"
    if any(k in n for k in K_CRAWL): return "crawl/kneel"
    if any(k in n for k in K_JUMP) or R_JUMP.search(n): return "jump/aerial"
    if R_SIT.search(n): return "sit/chair"
    return "other"
CLS = ["getup/lie/ground", "crawl/kneel", "breaking/floor", "flip/acro", "jump/aerial", "sit/chair", "other"]
def gcat(name):
    m = re.match(r"(GROUND_[A-Z]+)_", name)
    return m.group(1) if m else None

def reason(r):
    t = r["termination_reason"]
    return {"body_pos": "EE"}.get(t, t)

# ---------------------------------------------------------------- reference-motion phase features (per motion, lazily)
_feat = {}
def feats(r):
    nm = r["motion_name"]
    if nm in _feat: return _feat[nm]
    d = np.load(r["motion_path"])
    pos = d["body_pos_w"]; q = d["body_quat_w"][:, 0]           # pelvis quat (w, x, y, z)
    pz = pos[:, 0, 2]
    upz = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)               # pelvis z-axis . world z
    minz = pos[:, :, 2].min(axis=1)                               # lowest body origin
    fl = minz > 0.15
    runs, cur = [], 0
    for v in fl:
        cur = cur + 1 if v else 0; runs.append(cur)
    f = dict(T=len(pz), low_frac=float((pz < 0.45).mean()), min_pz=float(pz.min()), inverted=bool((upz < 0).any()),
             max_flight=int(max(runs) if runs else 0), stand_minz=float(np.median(minz)))
    f["_pz"], f["_upz"], f["_fl"] = pz, upz, fl
    _feat[nm] = f
    return f
def phase_at(r, step):
    f = feats(r); T = f["T"]; s = max(0, min(int(step), T - 1)); a, b = max(0, s - 25), min(T, s + 6)
    if (f["_upz"][a:b] < 0.2).any(): return "inverted(>78deg)"
    if f["_fl"][a:b].any(): return "flight"
    if (f["_pz"][a:b] < 0.45).any(): return "low_pelvis(<0.45)"
    return "upright"
def mtype(r):
    f = feats(r)
    if f["inverted"]: return "ref_inverted"
    if f["max_flight"] >= 8: return "ref_flight>=0.16s"
    if f["low_frac"] > 0.05: return "ref_low>5%"
    return "ref_upright"

out = {}
def P(*a):
    print(*a)

# ---------------------------------------------------------------- headline
lb = {r["run_id"]: r for r in json.load(urllib.request.urlopen(API))}
names = list(load(SINGLE["O"]).keys())
for k, d in list(SINGLE.items()) + [(g, x) for g, xs in GROUPS.items() for x in xs]:
    assert set(load(d)) == set(names), d
P("== headline (n=%d)" % len(names))
P(f"{'run':4s} {'id':10s} {'rank':>4s} {'succ%':>6s} {'pass':>5s} {'fail':>4s} {'EE':>4s} {'r_ori':>5s} {'r_pos':>5s} {'oth':>4s} {'prog%':>6s} {'MPJPE':>7s} {'MPKPE':>6s} {'mpjpe_l':>7s}")
out["headline"] = {}
for k, d in SINGLE.items():
    R = load(d); rid = os.path.basename(d); m = lb.get(rid, {})
    c = collections.Counter(reason(r) for r in R.values() if not r["ok"])
    npass = sum(r["ok"] for r in R.values())
    oth = sum(v for kk, v in c.items() if kk not in ("EE", "root_ori", "root_pos"))
    h = dict(id=rid, rank=m.get("rank"), success=100 * npass / len(R), n_pass=npass, fails=dict(c), progress=m.get("progress_rate"),
             mpjpe_rad=m.get("mpjpe_rad"), mpkpe_mm=m.get("mpkpe_mm"), mpjpe_l_mm=m.get("mpjpe_l_mm"))
    out["headline"][k] = h
    P(f"{k:4s} {rid[:8]:10s} {str(h['rank']):>4s} {h['success']:6.2f} {npass:5d} {len(R)-npass:4d} {c['EE']:4d} {c['root_ori']:5d} {c['root_pos']:5d} {oth:4d} "
      f"{100*(h['progress'] or 0):6.2f} {h['mpjpe_rad'] or 0:7.4f} {h['mpkpe_mm'] or 0:6.2f} {h['mpjpe_l_mm'] or 0:7.2f}")
P("other termination reasons:", {k: {kk: v for kk, v in h["fails"].items() if kk not in ("EE", "root_ori", "root_pos")} for k, h in out["headline"].items()})

# ---------------------------------------------------------------- cluster / source sizes
meta = {n: dict(src=src(load(SINGLE["O"])[n]), cl=cluster(n), g=gcat(n)) for n in names}
P("\n== cluster sizes:", dict(collections.Counter(m["cl"] for m in meta.values())))
for cl in CLS:
    ex = [n for n in names if meta[n]["cl"] == cl]
    P(f"   {cl}: e.g.", "; ".join(sorted({re.sub(r'__[0-9a-f]+$', '', n)[:28] for n in ex})[:14]))

# ---------------------------------------------------------------- pairwise single checkpoints
def pair(a, b, verbose=True):
    A, B = load(SINGLE[a]), load(SINGLE[b])
    af = [n for n in names if not A[n]["ok"] and B[n]["ok"]]
    bf = [n for n in names if A[n]["ok"] and not B[n]["ok"]]
    both = [n for n in names if not A[n]["ok"] and not B[n]["ok"]]
    res = dict(a_fail_b_pass=len(af), b_fail_a_pass=len(bf), both_fail=len(both), net_b_minus_a=len(af) - len(bf))
    P(f"\n== {a} vs {b}: {a} fail & {b} pass {len(af)} | {b} fail & {a} pass {len(bf)} | both fail {len(both)} | net({b}-{a}) {len(af)-len(bf):+d}")
    if verbose:
        for tag, S, X in ((f"{a}-fail/{b}-pass", af, A), (f"{b}-fail/{a}-pass", bf, B)):
            cs = collections.Counter(meta[n]["src"] for n in S)
            cc = collections.Counter(meta[n]["cl"] for n in S)
            cr = collections.Counter(reason(X[n]) for n in S)
            cx = collections.Counter((meta[n]["cl"], reason(X[n])) for n in S)
            cg = collections.Counter(meta[n]["g"] for n in S if meta[n]["g"])
            cp = collections.Counter(phase_at(X[n], X[n]["failure_step"]) for n in S)
            ct = collections.Counter(mtype(X[n]) for n in S)
            P(f"  [{tag}] by source {dict(cs.most_common())}")
            P(f"  [{tag}] by cluster {dict(cc.most_common())}")
            P(f"  [{tag}] by reason {dict(cr.most_common())}  gmt GROUND_* {dict(cg)}")
            P(f"  [{tag}] cluster x reason {dict(cx.most_common())}")
            P(f"  [{tag}] ref phase at failure (-0.5s..+0.1s) {dict(cp.most_common())} | whole-ref type {dict(ct.most_common())}")
            res[tag] = dict(source=dict(cs), cluster=dict(cc), reason=dict(cr), cluster_reason={f"{k[0]}|{k[1]}": v for k, v in cx.items()},
                            ground=dict(cg), phase=dict(cp), reftype=dict(ct),
                            names=sorted(f"{meta[n]['src']}|{meta[n]['cl']}|{reason(X[n])}|{n}" for n in S))
        # net by cluster
        net = {cl: sum(1 for n in af if meta[n]["cl"] == cl) - sum(1 for n in bf if meta[n]["cl"] == cl) for cl in CLS}
        nets = collections.Counter()
        for n in af: nets[meta[n]["src"]] += 1
        for n in bf: nets[meta[n]["src"]] -= 1
        P(f"  net({b} better) by cluster {net}")
        P(f"  net({b} better) by source {dict(nets.most_common())}")
        res["net_cluster"], res["net_source"] = net, dict(nets)
    return res
out["pairs"] = {}
for a, b, v in [("O", "B", True), ("C", "B", True), ("O", "C", True), ("O", "L", True), ("B", "L", True), ("C", "L", False)]:
    out["pairs"][f"{a}_{b}"] = pair(a, b, v)

# ---------------------------------------------------------------- union ceilings
P("\n== pass-union ceilings")
out["union"] = {}
for combo in ["O", "C", "B", "L", "OC", "OB", "CB", "OL", "BL", "OBL", "OCB", "OCBL"]:
    k = sum(any(load(SINGLE[x])[n]["ok"] for x in combo) for n in names)
    out["union"][combo] = 100 * k / len(names)
    P(f"   {'+'.join(combo):8s} {k:5d}  {100*k/len(names):6.2f}%")
oracle_line = {g: sum(any(load(d)[n]["ok"] for d in GROUPS[g]) for n in names) for g in ("ours_line", "bj27_floor")}
both_lines = sum(any(load(d)[n]["ok"] for g in ("ours_line", "bj27_floor") for d in GROUPS[g]) for n in names)
P(f"   any-ckpt ceilings: ours_line(13) {100*oracle_line['ours_line']/len(names):.2f}%  bj27_floor(6) {100*oracle_line['bj27_floor']/len(names):.2f}%  both lines {100*both_lines/len(names):.2f}%")
out["union"]["any_ours_line"] = 100 * oracle_line["ours_line"] / len(names)
out["union"]["any_bj27_floor"] = 100 * oracle_line["bj27_floor"] / len(names)
out["union"]["any_both_lines"] = 100 * both_lines / len(names)

# ---------------------------------------------------------------- multi-checkpoint pass probabilities
prob = {g: {n: float(np.mean([load(d)[n]["ok"] for d in ds])) for n in names} for g, ds in GROUPS.items()}
P("\n== group means: " + ", ".join(f"{g}({len(GROUPS[g])}) {100*np.mean(list(prob[g].values())):.2f}% "
                                  f"[{', '.join(f'{100*np.mean([load(d)[n][chr(111)+chr(107)] for n in names]):.2f}' for d in GROUPS[g])}]" for g in GROUPS))
def expdiff(ga, gb, label):
    """expected pass difference gb-ga (segments) by cluster/source, + robust sets"""
    d = {n: prob[gb][n] - prob[ga][n] for n in names}
    bycl = {cl: round(sum(v for n, v in d.items() if meta[n]["cl"] == cl), 2) for cl in CLS}
    bysrc = collections.Counter()
    for n, v in d.items(): bysrc[meta[n]["src"]] += v
    byg = collections.Counter()
    for n, v in d.items():
        if meta[n]["g"]: byg[meta[n]["g"]] += v
    rob_b = [n for n in names if prob[gb][n] >= 2 / 3 and prob[ga][n] <= 1 / 3]
    rob_a = [n for n in names if prob[ga][n] >= 2 / 3 and prob[gb][n] <= 1 / 3]
    P(f"\n== expected pass diff {label} ({gb} - {ga}) total {sum(d.values()):+.2f} segs = {100*sum(d.values())/len(names):+.2f} pp")
    P(f"   by cluster {bycl}")
    P(f"   by source {{{', '.join(f'{k}: {v:+.2f}' for k, v in sorted(bysrc.items(), key=lambda x: -abs(x[1])))}}}")
    P(f"   gmt GROUND_* {{{', '.join(f'{k}: {v:+.2f}' for k, v in byg.items())}}}")
    def desc(S):
        return dict(collections.Counter(meta[n]["cl"] for n in S).most_common())
    P(f"   robust {gb}-better (p_b>=2/3, p_a<=1/3): {len(rob_b)} {desc(rob_b)}")
    P(f"   robust {ga}-better: {len(rob_a)} {desc(rob_a)}")
    for tag, S in ((f"robust_{gb}_better", rob_b), (f"robust_{ga}_better", rob_a)):
        for n in sorted(S, key=lambda n: (meta[n]["cl"], n)):
            P(f"      {tag[:22]:22s} {meta[n]['src'][:14]:14s} {meta[n]['cl'][:16]:16s} p_a={prob[ga][n]:.2f} p_b={prob[gb][n]:.2f} {re.sub(r'__[0-9a-f]+$', '', n)[:70]}")
    return dict(total=sum(d.values()), cluster=bycl, source=dict(bysrc), ground=dict(byg),
                robust_b=[f"{meta[n]['src']}|{meta[n]['cl']}|{n}" for n in rob_b], robust_a=[f"{meta[n]['src']}|{meta[n]['cl']}|{n}" for n in rob_a])
out["exp"] = {
    "bj27_floor_vs_ours_line": expdiff("ours_line", "bj27_floor", "bj27 floor10 line vs ours line"),
    "root_motion_on_pelvis": expdiff("bj27_ro", "bj27_roroot", "root motion (bj27, matched 164-176k)"),
    "ground_exposure": expdiff("bj27_roroot_r23", "bj27_floor_r23", "ground exposure (bj27, matched 188/192k)"),
    "leader_vs_ours_line": expdiff("ours_line", "leader", "leader vs ours line"),
}

# ---------------------------------------------------------------- GROUND_* per run / per group
P("\n== gmt_eval GROUND_* success (pass/n)")
gc = sorted({m["g"] for m in meta.values() if m["g"]})
out["ground"] = {}
for k, d in SINGLE.items():
    R = load(d); row = {}
    for g in gc:
        S = [n for n in names if meta[n]["g"] == g]; row[g] = (sum(R[n]["ok"] for n in S), len(S))
    out["ground"][k] = row
    P(f"   {k:4s} " + "  ".join(f"{g} {p}/{n}" for g, (p, n) in row.items()))
for g_ in GROUPS:
    row = {}
    for g in gc:
        S = [n for n in names if meta[n]["g"] == g]; row[g] = (round(sum(prob[g_][n] for n in S), 2), len(S))
    out["ground"]["mean_" + g_] = row
    P(f"   mean {g_:16s} " + "  ".join(f"{g} {p:.1f}/{n}" for g, (p, n) in row.items()))
# GROUND_GETUP by start posture
def posture(n):
    for p in ["SIDE_LEFT", "SIDE_RIGHT", "SUPINE", "PRONE", "SIT_RECLINE", "LOW_KNEEL", "DOUBLE_KNEEL", "ONE_KNEEL"]:
        if n.startswith("GROUND_GETUP_" + p): return p
    return "other"
P("   GROUND_GETUP by start posture (pass/n):")
for p in ["SIDE_LEFT", "SIDE_RIGHT", "SUPINE", "PRONE", "SIT_RECLINE", "LOW_KNEEL", "DOUBLE_KNEEL", "ONE_KNEEL", "other"]:
    S = [n for n in names if meta[n]["g"] == "GROUND_GETUP" and posture(n) == p]
    if not S: continue
    P(f"     {p:13s} n={len(S):2d}  " + "  ".join(f"{k} {sum(load(SINGLE[k])[n]['ok'] for n in S)}" for k in SINGLE)
      + "  | line means " + "  ".join(f"{g} {sum(prob[g][n] for n in S):.1f}" for g in ("ours_line", "bj27_floor", "bj27_ro", "bj27_roroot", "bj27_roroot_r23", "bj27_floor_r23")))

# ---------------------------------------------------------------- per cluster success table
P("\n== per-cluster success% (single ckpts | line means)")
out["cluster_table"] = {}
for cl in CLS + ["ALL"]:
    S = [n for n in names if cl == "ALL" or meta[n]["cl"] == cl]
    row = {k: 100 * sum(load(d)[n]["ok"] for n in S) / len(S) for k, d in SINGLE.items()}
    row.update({g: 100 * sum(prob[g][n] for n in S) / len(S) for g in GROUPS})
    out["cluster_table"][cl] = dict(n=len(S), **row)
    P(f"   {cl:17s} n={len(S):4d} " + " ".join(f"{k}={v:5.1f}" for k, v in row.items()))

if len(sys.argv) > 1:
    json.dump(out, open(sys.argv[1], "w"), ensure_ascii=False, indent=1, default=str)
    P("wrote", sys.argv[1])
