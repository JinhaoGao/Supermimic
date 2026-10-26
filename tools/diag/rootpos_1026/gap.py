"""Segments where our run fails and the leader passes (and vice versa), by source / reason / name cluster (read-only)."""
import collections, csv, os, re, sys
ST = "/home/ubuntu/work/NOetix_GMT_eval/storage/runs"
LEADER = "130b4a731f3244ff877baee835fe256b"
def full(p): return next(d for d in os.listdir(ST) if d.startswith(p))
def load(rid): return {r["segment_id"]: r for r in csv.DictReader(open(f"{ST}/{full(rid)}/evaluation_per_segment.csv"))}
L = load(LEADER)
runs = sys.argv[1:]
O = {p: load(p) for p in runs}
def src(r): return r["motion_path"].split("/g1_motionatlas_v4/")[1].split("/")[0]
def clus(n):
  n = n.lower()
  for k, pats in [("getup/lie/ground", ["getup", "get_up", "ground_", "lie", "lying", "crawl", "匍匐", "躺", "起身"]),
                  ("breaking/floor", ["breaking", "windmill", "headspin", "freeze", "撑地", "flare"]),
                  ("flip/acro", ["flip", "fan", "acro", "handspring", "cartwheel", "kongfan", "shoufan", "侧手翻", "后空翻"]),
                  ("jump/aerial", ["jump", "tiao", "tengkong", "xuanfeng", "erqijiao", "zhuanti", "跳"]),
                  ("sit/chair/vehicle", ["sit", "chair", "vehicle", "坐", "zuozi"])]:
    if any(p in n for p in pats): return k
  return "other"
seg = list(L)
for p in runs:
  o = O[p]
  ours_fail_lead_pass = [s for s in seg if o[s]["success"] != "True" and L[s]["success"] == "True"]
  lead_fail_ours_pass = [s for s in seg if o[s]["success"] == "True" and L[s]["success"] != "True"]
  both_fail = [s for s in seg if o[s]["success"] != "True" and L[s]["success"] != "True"]
  print(f"== {p}: ours fail & leader pass {len(ours_fail_lead_pass)} | leader fail & ours pass {len(lead_fail_ours_pass)} | both fail {len(both_fail)}")
  c = collections.Counter((o[s]["termination_reason"], clus(o[s]["motion_name"])) for s in ours_fail_lead_pass)
  for k, v in c.most_common(): print(f"   ours-only fail {k[0]:10s} {k[1]:18s} {v}")
  cs = collections.Counter(src(o[s]) for s in ours_fail_lead_pass)
  print("   by source:", dict(cs.most_common()))
  cl = collections.Counter((L[s]["termination_reason"], clus(L[s]["motion_name"])) for s in lead_fail_ours_pass)
  print("   leader-only fail:", dict(cl.most_common(8)))
