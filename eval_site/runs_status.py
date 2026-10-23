"""Status / headline numbers of site runs.  python runs_status.py <run_id> ..."""
import json
import sys
import urllib.request

for rid in sys.argv[1:]:
    d = json.load(urllib.request.urlopen(f"http://127.0.0.1:8010/api/v1/runs/{rid}"))
    s = d.get("summary") or {}
    head = {k: s.get(k) for k in ("success_rate", "progress_rate")} if s else {}
    print(rid[:8], d["status"], f"{d.get('progress_current')}/{d.get('progress_total')}", head, (d.get("policy_name") or "")[:60])
