"""Trainer._resume_diffs also reports the master-side settings that live outside every state_dict (10-26 handoff
checklist C-2): input_clip and the other --fixes items, action noise, UTD, the WM frame and the sampler options. A resume
that forgets --fixes would otherwise drop input_clip from 50 back to 10 without a word. CPU, torch only.

  python tests/test_resume_diffs_master.py
"""

import dataclasses
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # the repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_sampler_clamp  # noqa: E402,F401  (stubs mjlab / tensorboard via sys.modules when they are missing)

from supermimic.trainer import Trainer, TrainCfg, cfg_dict  # noqa: E402
from supermimic.world_model import WorldModelCfg  # noqa: E402

FIXES = dict(input_clip=50.0, clip_values=True, critic_weight_survival=True, imag_truncate=True, wm_prefix_mask=True,
             clean_prev_action=True)


def diffs(ck: TrainCfg | dict, cur: TrainCfg) -> list[str]:
  t = object.__new__(Trainer)
  t.cfg = cur
  d, rel = Trainer._resume_diffs(t, ck if isinstance(ck, dict) else cfg_dict(ck))
  assert rel == "same"
  return d


def test_same_setup_is_silent():
  for c in (TrainCfg(), TrainCfg(**FIXES), TrainCfg(sampler_cap=50.0, sampler_clusters=256, sampler_boot_mean=True)):
    assert diffs(c, c) == [], diffs(c, c)


def test_forgotten_fixes_are_reported():
  d = diffs(TrainCfg(**FIXES), TrainCfg())
  keys = sorted(x.split(" = ")[0] for x in d)
  assert keys == sorted(FIXES), keys
  assert "input_clip = 50.0 in the checkpoint, 10.0 here" in d


def test_sampler_and_wm_frame_are_reported():
  ck = TrainCfg(sampler_cap=50.0, sampler_clusters=256, sampler_cluster_mix=0.3, wm=WorldModelCfg(body_frame=True))
  d = diffs(ck, TrainCfg())
  keys = sorted(x.split(" = ")[0] for x in d)
  assert keys == ["sampler_cap", "sampler_cluster_mix", "sampler_clusters", "wm.body_frame"], keys
  # cluster alpha / mix are compared only while clusters are on somewhere
  assert diffs(TrainCfg(sampler_cluster_mix=0.2), TrainCfg()) == []


def test_old_checkpoints_take_defaults():
  ck = cfg_dict(TrainCfg())
  for k in list(FIXES) + ["action_noise_std", "env_steps_per_iter", "sampler_cap", "sampler_clusters", "sampler_boot_mean",
                         "sampler_cluster_alpha", "sampler_cluster_mix"]:
    ck.pop(k, None)
  ck["wm"].pop("body_frame", None)
  assert diffs(ck, TrainCfg()) == []
  assert diffs(ck, TrainCfg(**FIXES))[0].startswith("input_clip = 10.0 in the checkpoint, 50.0 here")


if __name__ == "__main__":
  for name, fn in list(globals().items()):
    if name.startswith("test_") and callable(fn):
      fn()
      print(f"ok  {name}")
