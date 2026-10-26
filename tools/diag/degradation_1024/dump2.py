import json, csv, gzip, sys
csv.field_size_limit(10**9)
sel = json.load(open("/data2/supermimic/diag_1024/taskC/runs.json"))
ST = "/home/ubuntu/work/NOetix_GMT_eval/storage/runs"
keep = [s for s in sel if s[0] in ("a", "b", "L") or (s[0] == "s" and s[1] in (86000, 102000))]
hdr = None
with gzip.open("/data2/supermimic/diag_1024/taskC/seg_full.tsv.gz", "wt") as w:
    for tag, it, rid, sr in keep:
        rd = csv.DictReader(open(f"{ST}/{rid}/evaluation_per_segment.csv"))
        if hdr is None:
            cols = [c for c in rd.fieldnames if not c.endswith("_max") and not c.endswith("_max_max") and c not in ("motion_path", "category", "eval_unit", "body_mapping_source")
                    and (("__" not in c) or c.startswith("joint_") or c.startswith("body_local_lin_acc_error__") or c.startswith("body_local_pos_error__"))]
            cols += ["root_xy_error_max", "root_height_error_max", "body_z_error_max", "termination_body_z_error_max", "termination_body_pos_error_max"]
            hdr = cols
            w.write("\t".join(["tag", "it"] + cols) + "\n")
        for r in rd:
            w.write("\t".join([tag, str(it)] + [r.get(c, "") for c in hdr]) + "\n")
print(len(keep), len(hdr))
