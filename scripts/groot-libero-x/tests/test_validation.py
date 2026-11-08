"""Pure tests for the LIBERO-X rollout alignment gate. No sim / network / GPU."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from validation import validate_rollout  # noqa: E402

from _fakes import build_rollout  # noqa: E402

from groot_obs import decode_action_step  # noqa: E402


def _v(rollout):
    return validate_rollout(rollout, decode_action_step)


class ValidationTest(unittest.TestCase):
    def test_aligned_rollout_passes(self) -> None:
        r = _v(build_rollout(success=True, n_policy=3, replan=8))
        self.assertTrue(r.all_passed, r.format())

    def test_aligned_with_features_passes(self) -> None:
        r = _v(build_rollout(with_features=True, n_policy=2, replan=4))
        self.assertTrue(r.all_passed, r.format())

    def test_num_steps_wait_does_not_break_init_mode(self) -> None:
        # sim_step == control_step even though num_steps_wait=10 (no settle steps)
        r = _v(build_rollout(n_policy=2, replan=4, num_steps_wait=10))
        self.assertTrue(r.all_passed, r.format())

    def test_broken_executed_action_fails(self) -> None:
        rec = build_rollout(n_policy=2, replan=4)
        rec.controls[3].executed_action = rec.controls[3].executed_action + 0.5
        self.assertFalse(_v(rec).all_passed)

    def test_non_sequential_policy_step_fails(self) -> None:
        rec = build_rollout(n_policy=2, replan=4)
        rec.policies[1].policy_step = 5
        self.assertFalse(_v(rec).all_passed)

    def test_video_frame_mismatch_fails(self) -> None:
        rec = build_rollout(n_policy=2, replan=4)
        rec.controls[2].video_frame_id = 99
        self.assertFalse(_v(rec).all_passed)

    def test_sim_step_ne_control_step_fails(self) -> None:
        rec = build_rollout(n_policy=2, replan=4)
        for c in rec.controls:
            c.sim_step = c.control_step + 10  # add a bogus wait offset
        self.assertFalse(_v(rec).all_passed)

    def test_sim_step_nonzero_start_fails(self) -> None:
        rec = build_rollout(n_policy=2, replan=4)
        for c in rec.controls:
            c.sim_step += 1  # starts at 1
        self.assertFalse(_v(rec).all_passed)

    def test_identical_base_wrist_fails(self) -> None:
        rec = build_rollout(with_features=True, n_policy=2, replan=3)
        rec.policies[0].features.wrist_image = rec.policies[0].features.base_image.copy()
        self.assertFalse(_v(rec).all_passed)

    def test_nonfinite_feature_fails(self) -> None:
        rec = build_rollout(with_features=True, n_policy=2, replan=3)
        rec.policies[0].features.language[0, 0] = np.float16("inf")
        self.assertFalse(_v(rec).all_passed)

    def test_total_mismatch_fails(self) -> None:
        rec = build_rollout(n_policy=2, replan=4)
        rec.controls.pop()
        self.assertFalse(_v(rec).all_passed)


if __name__ == "__main__":
    unittest.main()
