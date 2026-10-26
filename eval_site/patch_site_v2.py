"""SuperMimic goal_layout "q,qd,rot6d,ref_root" on the NOetix_GMT_eval site (10-26, user-approved): new observation profiles
supermimic_g1_v2 / e2_supermimic_v2 (goal frame + reference anchor height, body-frame lin / ang velocity, projected gravity:
reference side only, proprioceptive policies), config / catalog checks, plugin + sha pin, benchmark observation lists.
Run from the site root with the site's python. Backs every touched file up first; exact-string edits, all or nothing."""
import hashlib
import shutil
import sys
import time
from pathlib import Path

ROOT = Path.cwd()
PLUGIN_SRC = Path(sys.argv[1])  # the repo's eval_site/eval_supermimic.py
BK = ROOT / "backups" / time.strftime("supermimic_v2_%Y%m%d_%H%M%S")
EV = ROOT / "vendor/Noetix_GMT/scripts/evaluate_mujoco.py"
PLUGIN = ROOT / "vendor/Noetix_GMT/scripts/eval_supermimic.py"
CFG = ROOT / "backend/noetix_eval/config.py"
CAT = ROOT / "backend/noetix_eval/catalog.py"
OBS = ROOT / "config/observations"
staged: dict[Path, str] = {}


def edit(path: Path, old: str, new: str) -> None:
  s = staged.get(path) or path.read_text(encoding="utf-8")
  if s.count(old) != 1:
    sys.exit(f"{path}: expected exactly one match ({s.count(old)}) for:\n{old}")
  staged[path] = s.replace(old, new)


if (OBS / "supermimic_g1_v2.yaml").exists():
  sys.exit("supermimic_g1_v2.yaml already exists; refusing to patch twice")

# 1) config.py: SuperMimic goal frames may carry the 10 reference-root values
edit(CFG, '''            if groups["obs_hist"].frame_dim != 3 * nj + 6 or groups["goal"].frame_dim != 2 * nj + 6:
                raise ValueError(f"SuperMimic frames are obs_hist {3 * nj + 6} and goal {2 * nj + 6} wide")''',
     '''            # goal 2nj+6 = q, qd, rot6d; 2nj+16 adds the reference root motion (goal_layout ref_root, reference only)
            if groups["obs_hist"].frame_dim != 3 * nj + 6 or groups["goal"].frame_dim not in (2 * nj + 6, 2 * nj + 16):
                raise ValueError(
                    f"SuperMimic frames are obs_hist {3 * nj + 6} and goal {2 * nj + 6} (or {2 * nj + 16} with goal_layout ref_root) wide"
                )''')

# 2) catalog.py: the ONNX goal_layout metadata must match the profile's goal width
edit(CAT, '''        if "anchor_body_name" not in metadata:
            compatibility_errors.append("missing metadata: anchor_body_name (SuperMimic goal orientation)")''',
     '''        if "anchor_body_name" not in metadata:
            compatibility_errors.append("missing metadata: anchor_body_name (SuperMimic goal orientation)")
        goal_dim = dict(expected_shapes).get("goal", (0, 0))[1]
        want_layout = "q,qd,rot6d,ref_root" if goal_dim == 2 * observation.expected_action_dim + 16 else "q,qd,rot6d"
        got_layout = str(metadata.get("goal_layout") or "q,qd,rot6d").strip().strip("'\\"").replace(" ", "")
        if got_layout != want_layout:
            compatibility_errors.append(f"goal_layout {got_layout!r} != {want_layout!r} required by profile {observation.id}")''')

# 3) plugin + pin
src = PLUGIN_SRC.read_bytes()
new_sha = hashlib.sha256(src).hexdigest()
ev = EV.read_text(encoding="utf-8")
old_sha = ev.split('_SUPERMIMIC_SHA256 = "', 1)[1].split('"', 1)[0]
if hashlib.sha256(PLUGIN.read_bytes()).hexdigest() != old_sha:
  sys.exit("installed eval_supermimic.py does not match its pin; refusing")
edit(EV, f'_SUPERMIMIC_SHA256 = "{old_sha}"', f'_SUPERMIMIC_SHA256 = "{new_sha}"')

# 4) new observation profiles from the v1 ones
TERMS = ("      - {name: target_anchor_height, dim: 1, unit: m, frame: world}\n"
         "      - {name: target_anchor_lin_vel, dim: 3, unit: m_s, frame: ref_anchor}\n"
         "      - {name: target_anchor_ang_vel, dim: 3, unit: rad_s, frame: ref_anchor}\n"
         "      - {name: target_anchor_projected_gravity, dim: 3, unit: unit_vector, frame: ref_anchor}\n")
NOTE = ("# v2 (10-26): goal 每帧末尾再加参考锚点的世界高度(1)、参考锚点自身系线速度(3)/角速度(3)、参考锚点投影重力(3),\n"
        "# 全部只取自参考动作文件(不需要机器人位置/里程计),机器人侧输入仍是纯本体感知;ONNX metadata goal_layout = q,qd,rot6d,ref_root。\n")
new_files: dict[Path, str] = {}
for v1, v2, nj, name in (("supermimic_g1_v1", "supermimic_g1_v2", 29, "SuperMimic G1 v2 (obs_hist 10x93 + goal 10x74 = 64 + 参考根部运动 10)"),
                         ("e2_supermimic_v1", "e2_supermimic_v2", 23, "SuperMimic E2 v2 (obs_hist 10x75 + goal 10x62 = 52 + 参考根部运动 10)")):
  s = (OBS / f"{v1}.yaml").read_text(encoding="utf-8")
  P, G, K = 3 * nj + 6, 2 * nj + 6, 10
  for old, new in ((f"id: {v1}\n", f"id: {v2}\n"), (f"expected_input_dim: {K * P + K * G}\n", f"expected_input_dim: {K * P + K * (G + 10)}\n"),
                   (f"    frame_dim: {G}\n", f"    frame_dim: {G + 10}\n"),
                   ("      - {name: target_anchor_ori_b_rot6d, dim: 6, unit: rotation_6d, frame: robot_anchor}\n",
                    "      - {name: target_anchor_ori_b_rot6d, dim: 6, unit: rotation_6d, frame: robot_anchor}\n" + TERMS)):
    if s.count(old) != 1:
      sys.exit(f"{v1}.yaml: expected one {old!r}")
    s = s.replace(old, new)
  line0 = next(l for l in s.splitlines(keepends=True) if l.startswith("name: "))
  s = s.replace(line0, f"name: {name}\n" + NOTE, 1)
  new_files[OBS / f"{v2}.yaml"] = s

# 5) every benchmark that lists the v1 profile also lists v2
for b in sorted((ROOT / "config/benchmarks").glob("*.yaml")):
  s = b.read_text(encoding="utf-8")
  for v1, v2 in (("supermimic_g1_v1", "supermimic_g1_v2"), ("e2_supermimic_v1", "e2_supermimic_v2")):
    if f", {v1}]" in s and v2 not in s:
      edit(b, f", {v1}]", f", {v1}, {v2}]")

# write: backups first, then everything
BK.mkdir(parents=True)
for p in list(staged) + [PLUGIN]:
  dst = BK / p.relative_to(ROOT)
  dst.parent.mkdir(parents=True, exist_ok=True)
  shutil.copy2(p, dst)
for p, s in staged.items():
  p.write_text(s, encoding="utf-8")
PLUGIN.write_bytes(src)
for p, s in new_files.items():
  p.write_text(s, encoding="utf-8")
print("backup:", BK)
print("edited:", [str(p.relative_to(ROOT)) for p in staged], "| plugin sha", old_sha[:12], "->", new_sha[:12])
print("new:", [str(p.relative_to(ROOT)) for p in new_files])
