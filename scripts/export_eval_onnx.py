"""Stamp a SuperMimic bundle's ONNX for the team's MuJoCo evaluation site (NOetix_GMT_eval, profile
supermimic_g1_v1 / supermimic_e2_v1, evaluator policy_type supermimic).

  python scripts/export_eval_onnx.py --bundle ~/supermimic/bundle_eval_g1_atlas_v1_32000 --out g1.onnx \
      [--metric_bodies pelvis,left_hip_roll_link,...] [--damping training|actuator]

Needs only onnx (+ onnxruntime for the check). The bundle comes from export_bundle.py, whose policy.onnx
already carries joint_names, default_joint_pos, action_scale/offset, anchor_body_name, joint_stiffness,
joint_damping (actuator kd + passive joint damping) and joint_armature.

* body_names: the site scores MPKPE and the body-height termination on the ONNX `body_names`. The policy
  itself never reads them, so for a score comparable with the other policies of a benchmark pass that
  robot profile's tracked bodies (E2: 14 incl. left_hand/right_hand; SuperMimic E2 tracks 12 in training).
  The training set is kept as `train_body_names`. The first name must be the root body.
* --damping training (default): joint_damping = actuator kd + passive damping of the training model
  (= the dynamics the policy was trained with). --damping actuator: kd only (a real robot's motor PD).
"""

from __future__ import annotations

import argparse
import ast
import json
import os

import onnx


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--bundle", required=True)
  p.add_argument("--out", required=True)
  p.add_argument("--metric_bodies", default=None, help="comma separated; default = training tracked bodies")
  p.add_argument("--damping", choices=["training", "actuator"], default="training")
  a = p.parse_args()
  meta = json.load(open(os.path.join(a.bundle, "meta.json")))
  model = onnx.load(os.path.join(a.bundle, "policy.onnx"))
  props = {e.key: e.value for e in model.metadata_props}
  kd = meta["pd_kd"]
  passive = meta.get("joint_passive_damping", [0.0] * len(kd))
  damping = [x + y for x, y in zip(kd, passive)] if a.damping == "training" else list(kd)
  bodies = [b.strip() for b in a.metric_bodies.split(",")] if a.metric_bodies else list(meta["body_names"])
  if bodies[0] != meta["root_body_name"]:
    raise SystemExit(f"first metric body must be the root body {meta['root_body_name']!r}, got {bodies[0]!r}")
  props.update({
    "joint_stiffness": json.dumps(meta["pd_kp"]),
    "joint_damping": json.dumps(damping),
    "joint_armature": json.dumps(meta["joint_armature"]),
    "body_names": json.dumps(bodies),
    "train_body_names": json.dumps(meta["body_names"]),
    "eval_damping_mode": a.damping,
    "robot": meta["robot"],
    "source_ckpt": meta["ckpt"],
  })
  # The evaluator parses every value with ast.literal_eval (then a comma split); make sure lists survive.
  for k in ("joint_names", "joint_stiffness", "joint_damping", "joint_armature", "default_joint_pos", "action_scale", "body_names"):
    v = ast.literal_eval(props[k])
    assert isinstance(v, list) and len(v) > 0, k
  nj = len(json.loads(props["joint_names"]))
  for k in ("joint_stiffness", "joint_damping", "joint_armature", "default_joint_pos", "action_scale"):
    assert len(json.loads(props[k])) == nj, k
  del model.metadata_props[:]
  for k, v in props.items():
    e = model.metadata_props.add()
    e.key, e.value = k, v
  onnx.save(model, a.out)
  try:
    import numpy as np
    import onnxruntime as ort

    s = ort.InferenceSession(a.out, providers=["CPUExecutionProvider"])
    shapes = {i.name: i.shape for i in s.get_inputs()}
    out = s.run(None, {n: np.zeros(sh, dtype=np.float32) for n, sh in shapes.items()})[0]
    print(f"[eval-onnx] inputs {shapes} -> action {list(out.shape)}")
  except ImportError:
    pass
  print(f"[eval-onnx] wrote {a.out}: iteration {meta['iteration']}, damping={a.damping} "
        f"(kd + passive, max passive {max(passive):.3g})" if a.damping == "training" else f"[eval-onnx] wrote {a.out}: damping=actuator kd")
  print(f"[eval-onnx] metric bodies ({len(bodies)}): {bodies}")


if __name__ == "__main__":
  main()
