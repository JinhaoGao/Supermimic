import csv, os, sys
ST = "/home/ubuntu/work/NOetix_GMT_eval/storage/runs"
LEADER = "130b4a731f3244ff877baee835fe256b"
rid = next(d for d in os.listdir(ST) if d.startswith(sys.argv[1]))
L = {r["segment_id"]: r for r in csv.DictReader(open(f"{ST}/{LEADER}/evaluation_per_segment.csv"))}
rows = [r for r in csv.DictReader(open(f"{ST}/{rid}/evaluation_per_segment.csv")) if r["termination_reason"] == "root_pos"]
print("path\teval_fail_frame\tsrc\tleader")
for r in rows:
  if int(r["start_frame"] or 0) != 0:
    sys.stderr.write(f"skip non-zero start {r['motion_name']}\n"); continue
  src = r["motion_path"].split("/g1_motionatlas_v4/")[1].split("/")[0]
  print(f"{r['motion_path']}\t{r['failure_step']}\t{src}\t{L.get(r['segment_id'], {}).get('termination_reason', '?')}")
