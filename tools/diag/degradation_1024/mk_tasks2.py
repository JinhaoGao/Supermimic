import json
T = json.load(open("/data2/supermimic/diag_1024/taskC/roll/tasks.json"))
base = [t for t in T if t["grp"] == "F212" and t["cond"] == "clean" and t["it"] in ("116000", "212000")]
out = []
for t in base:
    for c in ("pa", "ex", "act05", "act02"):
        u = dict(t); u["cond"] = c; out.append(u)
json.dump(out, open("/data2/supermimic/diag_1024/taskC/roll/tasks2.json", "w")); print(len(out))
