import sys, glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
def load(d):
  ea = EventAccumulator(glob.glob(d + "/events.out.tfevents.*")[0], size_guidance={"scalars": 0}); ea.Reload()
  return {t: [(e.step, e.value) for e in ea.Scalars(t)] for t in ea.Tags()["scalars"]}
a, b = load(sys.argv[1]), load(sys.argv[2])
skip = lambda k: k.startswith("time/") or k == "train/wall_h"
ka = {k for k in a if not skip(k)}; kb = {k for k in b if not skip(k)}
print("keys only in base:", sorted(ka - kb), "only in new:", sorted(kb - ka))
diff = [k for k in sorted(ka & kb) if a[k] != b[k]]
print(f"{len(ka & kb)} common non-timing scalar tags; differing: {diff}")
