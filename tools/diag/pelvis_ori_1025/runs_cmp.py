import pandas as pd, numpy as np, sys
L = sys.argv[1]
runs = [("r_5815245b", "H32 ctl 88k (r7)"), ("r_7e981d85", "H48 88k (r9)"), ("r_6fec98c4", "H64 88k (r10)"),
        ("r_fc7b9608", "expo ctl 110k (r13)"), ("r_2139a048", "curric x1.4 110k (r13)"), ("r_f49e0de0", "expo ctl 116k"), ("r_e9157e1c", "curric ->1.0 116k"),
        ("r_ce12b7ed", "cyc2 128k (r14)"), ("r_2b02b12a", "r15 noclamp 136k"), ("r15_136", "r15 clamp 136k (bj27)"), ("s138", "r15ws 138k"), ("s148", "r15ws 148k"), ("s152", "r15ws 152k"), ("leader", "LEADER GMT v2")]
aer = ["tengkong", "erqijiao", "xuanfeng", "xuanzi", "tiaogao", "zhongtiao", "zhupaotiaoyue", "waibailian", "kongfan", "shoufan", "zhuanti"]
print(f"{'run':<24} {'succ%':>6} {'EE':>4} {'ori':>4} {'pos':>4} | succ-seg root_ori_max med/p90 | EEmax med | xy med | GROUND_GETUP ok/n | aerial ok/n (ori/pos/EE)")
for tag, name in runs:
    d = pd.read_csv(f"{L}/data/seg_{tag}.tsv", sep="\t")
    tr = d.termination_reason
    s = d[tr == "motion_complete"]
    gg = d.motion_name.str.startswith("GROUND_GETUP")
    ae = d.motion_name.str.lower().apply(lambda n: any(k in n for k in aer))
    a = d[ae]
    print(f"{name:<24} {100 * (tr == 'motion_complete').mean():6.2f} {(tr == 'body_pos').sum():4d} {(tr == 'root_ori').sum():4d} {(tr == 'root_pos').sum():4d} | "
          f"{s.root_ori_error_max.median():.3f}/{s.root_ori_error_max.quantile(.9):.3f}            | {s.termination_body_z_error_max_max.median():.3f}     | {s.root_xy_error_max.median():.2f}   | "
          f"{(tr[gg] == 'motion_complete').sum():3d}/{gg.sum():3d}          | {(a.termination_reason == 'motion_complete').sum()}/{len(a)} "
          f"({(a.termination_reason == 'root_ori').sum()}/{(a.termination_reason == 'root_pos').sum()}/{(a.termination_reason == 'body_pos').sum()})")
