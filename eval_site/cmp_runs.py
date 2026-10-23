"""Per-motion comparison of an evaluator report CSV with a stored site run (by run id prefix)."""
import csv
import glob
import sys

mine_csv, run_prefix = sys.argv[1], sys.argv[2]
cands = glob.glob(f"/home/ubuntu/work/NOetix_GMT_eval/storage/**/*{run_prefix}*/**/evaluation_per_segment.csv", recursive=True)
if not cands:
    cands = [p for p in glob.glob("/home/ubuntu/work/NOetix_GMT_eval/storage/**/evaluation_per_segment.csv", recursive=True) if run_prefix in p]
print("leader csv:", cands[:1])
mine = {r["motion_name"]: r for r in csv.DictReader(open(mine_csv))}
lead = {r["motion_name"]: r for r in csv.DictReader(open(cands[0]))}
print("leader run motions:", len(lead), "success:", sum(r["success"] == "True" for r in lead.values()))
for name, r in mine.items():
    l = lead.get(name, {})
    print(f"{name[:58]:58s} ours {r['success']:5s} {r['termination_reason']:15s} {float(r['length_ratio']):.2f} | "
          f"leader {l.get('success', '?'):5s} {l.get('termination_reason', ''):15s} {float(l.get('length_ratio', 'nan')):.2f}")
