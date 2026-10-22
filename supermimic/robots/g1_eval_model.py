"""G1 as in the team's MuJoCo evaluation / deploy model (unitree_mujoco scene_29dof.xml, 35.11 kg robot mass).

mjlab's G1 (33.34 kg) differs from it in the waist: the waist-roll and waist-pitch joints sit 0.9 cm lower /
1.9 cm higher (the shoulders end up at the same height), and the waist links and torso have other inertials
(torso 7.818 -> 9.598 kg, the original carries the head). The original also has passive joint damping 0.05 and
joint friction 0.1 / 0.2 on every joint (mjlab: 0). ``patch_spec`` applies exactly these differences to mjlab's
G1 spec (values dumped from the compiled eval model on bj31, 2026-10-22); collision / visual geoms are untouched.
Actuators (gains, armature, effort limits) stay mjlab's.
"""

from __future__ import annotations

import math

import numpy as np

EVAL_BODIES = {
  "waist_yaw_link": {
    "mass": 0.244,
    "ipos": [
      0.003964,
      0.0,
      0.018769
    ],
    "iquat": [
      -0.017829102,
      0.628464055,
      0.028247102,
      0.777121069
    ],
    "inertia": [
      0.000158561,
      0.000124229,
      9.67669e-05
    ]
  },
  "waist_roll_link": {
    "pos": [
      -0.003964,
      0.0,
      0.035
    ],
    "quat": [
      1.0,
      0.0,
      0.0,
      0.0
    ],
    "mass": 0.047,
    "ipos": [
      0.0,
      -0.000236,
      0.010111
    ],
    "iquat": [
      0.999790017,
      0.020492,
      0.0,
      0.0
    ],
    "inertia": [
      7.515e-06,
      6.40206e-06,
      3.98394e-06
    ]
  },
  "torso_link": {
    "pos": [
      0.0,
      0.0,
      0.019
    ],
    "quat": [
      1.0,
      0.0,
      0.0,
      0.0
    ],
    "mass": 9.598,
    "ipos": [
      0.003317,
      0.000262,
      0.179856
    ],
    "iquat": [
      0.999830971,
      0.000376204,
      0.017989499,
      -0.00377704
    ],
    "inertia": [
      0.12407,
      0.111951,
      0.0325382
    ]
  },
  "left_shoulder_pitch_link": {
    "pos": [
      0.003956,
      0.10022,
      0.23778
    ],
    "quat": [
      0.99026414,
      0.13920102,
      1.3872e-05,
      -9.8687e-05
    ]
  },
  "right_shoulder_pitch_link": {
    "pos": [
      0.003956,
      -0.10021,
      0.23778
    ],
    "quat": [
      0.99026414,
      -0.13920102,
      1.3872e-05,
      9.8687e-05
    ]
  }
}

EVAL_JOINTS = {
  "left_hip_pitch_joint": {
    "damping": 0.05,
    "frictionloss": 0.2
  },
  "left_hip_roll_joint": {
    "damping": 0.05,
    "frictionloss": 0.2
  },
  "left_hip_yaw_joint": {
    "damping": 0.05,
    "frictionloss": 0.2
  },
  "left_knee_joint": {
    "damping": 0.05,
    "frictionloss": 0.2
  },
  "left_ankle_pitch_joint": {
    "damping": 0.05,
    "frictionloss": 0.2
  },
  "left_ankle_roll_joint": {
    "damping": 0.05,
    "frictionloss": 0.2
  },
  "right_hip_pitch_joint": {
    "damping": 0.05,
    "frictionloss": 0.2
  },
  "right_hip_roll_joint": {
    "damping": 0.05,
    "frictionloss": 0.2
  },
  "right_hip_yaw_joint": {
    "damping": 0.05,
    "frictionloss": 0.2
  },
  "right_knee_joint": {
    "damping": 0.05,
    "frictionloss": 0.2
  },
  "right_ankle_pitch_joint": {
    "damping": 0.05,
    "frictionloss": 0.2
  },
  "right_ankle_roll_joint": {
    "damping": 0.05,
    "frictionloss": 0.2
  },
  "waist_yaw_joint": {
    "damping": 0.05,
    "frictionloss": 0.2
  },
  "waist_roll_joint": {
    "damping": 0.05,
    "frictionloss": 0.2
  },
  "waist_pitch_joint": {
    "damping": 0.05,
    "frictionloss": 0.2
  },
  "left_shoulder_pitch_joint": {
    "damping": 0.05,
    "frictionloss": 0.2
  },
  "left_shoulder_roll_joint": {
    "damping": 0.05,
    "frictionloss": 0.2
  },
  "left_shoulder_yaw_joint": {
    "damping": 0.05,
    "frictionloss": 0.2
  },
  "left_elbow_joint": {
    "damping": 0.05,
    "frictionloss": 0.2
  },
  "left_wrist_roll_joint": {
    "damping": 0.05,
    "frictionloss": 0.2
  },
  "left_wrist_pitch_joint": {
    "damping": 0.05,
    "frictionloss": 0.1
  },
  "left_wrist_yaw_joint": {
    "damping": 0.05,
    "frictionloss": 0.1
  },
  "right_shoulder_pitch_joint": {
    "damping": 0.05,
    "frictionloss": 0.2
  },
  "right_shoulder_roll_joint": {
    "damping": 0.05,
    "frictionloss": 0.2
  },
  "right_shoulder_yaw_joint": {
    "damping": 0.05,
    "frictionloss": 0.2
  },
  "right_elbow_joint": {
    "damping": 0.05,
    "frictionloss": 0.2
  },
  "right_wrist_roll_joint": {
    "damping": 0.05,
    "frictionloss": 0.2
  },
  "right_wrist_pitch_joint": {
    "damping": 0.05,
    "frictionloss": 0.1
  },
  "right_wrist_yaw_joint": {
    "damping": 0.05,
    "frictionloss": 0.1
  }
}


def set_scalar(obj, attr: str, value: float) -> None:
  """Set a scalar MjSpec property; MuJoCo >= 3.16 stores some (joint damping) as arrays, XML scalar = element 0."""
  cur = getattr(obj, attr)
  if isinstance(cur, np.ndarray) and cur.size > 1:
    arr = np.zeros_like(cur, dtype=float)
    arr.flat[0] = value
    setattr(obj, attr, arr)
  else:
    setattr(obj, attr, value)


def patch_spec(spec_fn):
  def build():
    spec = spec_fn()
    for name, b in EVAL_BODIES.items():
      body = spec.body(name)
      if "pos" in b:
        body.pos = b["pos"]
        body.quat = b["quat"]
      if "mass" in b:
        body.mass = b["mass"]
        body.ipos = b["ipos"]
        body.iquat = b["iquat"]
        body.inertia = b["inertia"]
        body.fullinertia = [math.nan, 0.0, 0.0, 0.0, 0.0, 0.0]  # unset: use iquat + diagonal inertia
        body.explicitinertial = True
    for name, j in EVAL_JOINTS.items():
      joint = spec.joint(name)
      set_scalar(joint, "damping", j["damping"])
      set_scalar(joint, "frictionloss", j["frictionloss"])
    return spec

  return build
