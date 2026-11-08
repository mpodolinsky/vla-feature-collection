"""Pure tests for identify_groot_features. No sim / network / GPU.

The LIBERO-X fine-tune keeps the same LIBERO_PANDA / Qwen3-VL token layout.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from feature_schema import identify_groot_features  # noqa: E402
from records import AlignmentError  # noqa: E402


def _good(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    lang = np.zeros((200, 2048), np.float32)
    lang[:9] = rng.standard_normal((9, 2048))
    mask = np.zeros((200,), bool)
    mask[:9] = True
    return {
        "base_image": rng.standard_normal((64, 2048)).astype(np.float32),
        "wrist_image": (rng.standard_normal((64, 2048)) + 4.0).astype(np.float32),
        "language": lang,
        "language_mask": mask,
        "language_len": 9,
        "state_features": rng.standard_normal((1536,)).astype(np.float32),
    }


class FeatureSchemaTest(unittest.TestCase):
    def test_good_features(self) -> None:
        f = identify_groot_features(_good(), np.zeros((16, 7), np.float32))
        self.assertEqual(f.base_image.shape, (64, 2048))
        self.assertEqual(f.language.shape, (200, 2048))
        self.assertEqual(f.state_features.shape, (1536,))
        self.assertEqual(f.language_len, 9)

    def test_missing_key(self) -> None:
        d = _good()
        del d["state_features"]
        with self.assertRaises(AlignmentError):
            identify_groot_features(d, np.zeros((16, 7), np.float32))

    def test_wrong_shape(self) -> None:
        d = _good()
        d["base_image"] = np.zeros((32, 2048), np.float32)
        with self.assertRaises(AlignmentError):
            identify_groot_features(d, np.zeros((16, 7), np.float32))

    def test_identical_base_wrist(self) -> None:
        d = _good()
        d["wrist_image"] = d["base_image"].copy()
        with self.assertRaises(AlignmentError):
            identify_groot_features(d, np.zeros((16, 7), np.float32))

    def test_all_false_mask(self) -> None:
        d = _good()
        d["language_mask"] = np.zeros((200,), bool)
        with self.assertRaises(AlignmentError):
            identify_groot_features(d, np.zeros((16, 7), np.float32))


if __name__ == "__main__":
    unittest.main()
