"""Contact-setting diagnosis for SuperMimic in the eval site's MuJoCo models (not an official run).

Variants of the official robot model (only colliding geoms are touched):
  official : as is (self-collision on, friction from the XML)
  noself   : robot geoms contype 0 / conaffinity 1 (collide with the floor only, like training)
  fric1    : sliding friction 1.0 on every colliding geom (training terrain friction; MuJoCo takes the max)
  noself_fric1 : both
Writes /tmp/claude_diag/models/<robot>_<variant>.xml and prints the evaluator command per variant.
"""
import os
import sys

import mujoco
import yaml

OUT = "/tmp/claude_diag/models"
os.makedirs(OUT, exist_ok=True)


def variant(robot_yaml, robot, name):
    path = yaml.safe_load(open(robot_yaml))["model_path"]
    spec = mujoco.MjSpec.from_file(path)
    base = os.path.dirname(os.path.abspath(path))
    spec.meshdir = os.path.join(base, spec.meshdir) if spec.meshdir else base
    if spec.texturedir:
        spec.texturedir = os.path.join(base, spec.texturedir)
    n_self = n_fric = 0
    for g in spec.geoms:
        if g.contype == 0 and g.conaffinity == 0:
            continue  # visual
        on_world = g.parent.name in ("world", "") or g.parent == spec.worldbody
        if "noself" in name:
            if on_world:
                g.contype, g.conaffinity = 1, 1
            else:
                g.contype, g.conaffinity = 0, 1
                n_self += 1
        if "fric1" in name:
            g.friction[0] = 1.0
            n_fric += 1
    xml = spec.to_xml()
    out = os.path.join(OUT, f"{robot}_{name}.xml")
    open(out, "w").write(xml)
    m = mujoco.MjModel.from_xml_path(out)  # compiles -> paths resolved
    print(f"{out}: robot geoms floor-only {n_self}, friction 1.0 on {n_fric}, nbody {m.nbody}, nu {m.nu}", file=sys.stderr)
    return out


for robot, ry in [("e2", "config/robots/e2_23dof_v1.yaml"), ("g1", "config/robots/g1_29dof_v1.yaml")]:
    for v in ("noself", "fric1", "noself_fric1"):
        variant(ry, robot, v)
