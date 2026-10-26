"""Whitelist / high-dynamic breakdown of site runs on a v4 dataset.
  python whitelist_breakdown.py <dataset_dir> label=run_id|report_dir ...  (report_dir: e.g. an eval_r18.sh offline run)
Groups: the GMT hard-set whitelist (noetix_kungfu, noetix_kungfu_dance_show, sportsmeeting) split by move type
from the clip names, plus g1_all (robot_acro / balance / lie_getup ...)."""
import collections
import csv
import os
import re
import sys

root = sys.argv[1]
pairs = [a.split("=", 1) for a in sys.argv[2:]]
src, layer = {}, {}
for dp, _, fs in os.walk(root):
    for f in fs:
        if f.endswith(".npz"):
            parts = os.path.relpath(os.path.join(dp, f), root).split(os.sep)
            src[f[:-4]], layer[f[:-4]] = parts[0], parts[1]

RULES = {
    "noetix_kungfu": [("翻腾(手翻/空翻)", r"fan|翻"), ("腾空(腾空踢/转体)", r"tengkong|腾空|xuanfeng|旋风"),
                      ("地面(鲤鱼打挺/扑步)", r"liyudating|pufu|鲤鱼|扑"), ("踢腿/腿法", r"tui|jiao|ceti|踢|腿|lian"), ("其他功夫", r".")],
    "noetix_kungfu_dance_show": [("太极", r"式|太极"), ("桑巴", r"桑巴|sangba"), ("民族舞", r"民族|维吾尔|维族|蒙古|藏|傣|朝鲜"),
                                 ("现代/啦啦操", r"现代|啦啦|街舞"), ("其他舞蹈", r".")],
    "sportsmeeting": [("breaking", r"breaking|地板"), ("popping/街舞", r"popping|jiewu|街舞"), ("桑巴", r"sangba|桑巴"), ("其他运动会", r".")],
    "g1_all": [("robot_acro 特技", r"robot_acro|flip|空翻"), ("robot_balance 平衡", r"robot_balance"), ("robot_lie_getup 躺地起身", r"lie_getup|getup"),
               ("其他 g1_all", r".")],
}


def group_of(name):
    s = src.get(name)
    if s not in RULES:
        return None
    low = name.lower()
    for label, pat in RULES[s]:
        if re.search(pat, low):
            sub = label + ("(phc)" if layer.get(name, "").endswith("_phc") else "")
            return s, sub
    return None


tabs = []
for label, run in pairs:
    t = collections.defaultdict(lambda: [0, 0])
    run_dir = run if os.path.isdir(run) else f"/home/ubuntu/work/NOetix_GMT_eval/storage/runs/{run}"  # site run id or report dir
    for r in csv.DictReader(open(f"{run_dir}/evaluation_per_segment.csv")):
        g = group_of(r["motion_name"])
        if g is None:
            continue
        ok = r["success"] == "True"
        for key in (g[0], g[0] + " / " + g[1]):
            t[key][0] += 1
            t[key][1] += ok
        if g[0] in ("noetix_kungfu", "noetix_kungfu_dance_show", "sportsmeeting"):
            t["白名单合计"][0] += 1
            t["白名单合计"][1] += ok
    tabs.append(t)
order = ["白名单合计"]
for s in RULES:
    order.append(s)
    order += sorted([k for k in tabs[0] if k.startswith(s + " / ")], key=lambda k: -tabs[0][k][0])
print(f"{'组':44s} {'n':>4s} " + " ".join(f"{l:>9s}" for l, _ in pairs))
for k in order:
    if k in tabs[0]:
        print(f"{k:44s} {tabs[0][k][0]:4d} " + " ".join(f"{100 * t[k][1] / max(t[k][0], 1):9.1f}" for t in tabs))
