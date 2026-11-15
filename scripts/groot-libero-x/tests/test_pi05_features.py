"""Pure tests for the pi0.5 feature path: no state vector, 256 (or pooled) image tokens, extra layers."""

from __future__ import annotations

import dataclasses
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from constants import PI05_SPEC  # noqa: E402
from feature_schema import identify_groot_features  # noqa: E402
from pi05_obs import make_bridge  # noqa: E402
from records import AlignmentError  # noqa: E402
from serialization import load_rollout, save_rollout  # noqa: E402
from validation import validate_rollout  # noqa: E402

from _fakes import build_rollout  # noqa: E402


def reply(tokens=256, layers=("L17",), n_lang=52, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    mask = np.zeros((200,), np.bool_)
    mask[:n_lang] = True
    out = {"language_mask": mask, "language_len": np.int32(n_lang), "actions": np.zeros((50, 7), np.float32)}
    for i, tag in enumerate(layers):
        sfx = "" if i == 0 else f"@{tag}"
        lang = np.zeros((200, 2048), np.float16)
        lang[:n_lang] = rng.standard_normal((n_lang, 2048))
        out["base_image" + sfx] = rng.standard_normal((tokens, 2048)).astype(np.float16)
        out["wrist_image" + sfx] = (rng.standard_normal((tokens, 2048)) + 2).astype(np.float16)
        out["language" + sfx] = lang
    return out


class Pi05SchemaTest(unittest.TestCase):
    def test_accepts_primary_and_extra_layers_without_state(self) -> None:
        r = reply(layers=("L16", "L17", "Lfinal"))
        f = identify_groot_features(r, r["actions"], PI05_SPEC)
        self.assertIsNone(f.state_features)
        self.assertEqual(f.base_image.shape, (256, 2048))
        self.assertEqual(sorted(f.extra), sorted(["base_image@L17", "wrist_image@L17", "language@L17",
                                                  "base_image@Lfinal", "wrist_image@Lfinal", "language@Lfinal"]))

    def test_pooled_spec_and_wrong_token_count(self) -> None:
        spec = dataclasses.replace(PI05_SPEC, img_tokens=64)
        r = reply(tokens=64)
        self.assertEqual(identify_groot_features(r, r["actions"], spec).base_image.shape, (64, 2048))
        with self.assertRaises(AlignmentError):
            identify_groot_features(reply(tokens=256), np.zeros((50, 7), np.float32), spec)

    def test_extra_layer_shape_and_inf_are_rejected(self) -> None:
        r = reply(layers=("L16", "L17"))
        r["base_image@L17"] = r["base_image@L17"][:10]
        with self.assertRaises(AlignmentError):
            identify_groot_features(r, r["actions"], PI05_SPEC)
        r = reply(layers=("L16", "L17"))
        r["language@L17"][0, 0] = np.inf
        with self.assertRaises(AlignmentError):
            identify_groot_features(r, r["actions"], PI05_SPEC)


class Pi05ExpertFeatureTest(unittest.TestCase):
    def _reply(self, steps=10):
        r = reply(layers=("L16",))
        rng = np.random.default_rng(1)
        r["expert@L16"] = rng.standard_normal((steps, 50, 1024)).astype(np.float16)
        r["expert_t@L16"] = np.linspace(1.0, 0.1, steps).astype(np.float32)
        return r

    def test_expert_maps_are_kept_as_extras(self) -> None:
        r = self._reply()
        f = identify_groot_features(r, r["actions"], PI05_SPEC)
        self.assertEqual(f.extra["expert@L16"].shape, (10, 50, 1024))
        self.assertEqual(f.extra["expert@L16"].dtype, np.float16)
        self.assertEqual(f.extra["expert_t@L16"].dtype, np.float32)  # times are not rounded to fp16

    def test_bad_expert_shapes_and_inf_rejected(self) -> None:
        r = self._reply(); r["expert@L16"] = r["expert@L16"][0]
        with self.assertRaises(AlignmentError):
            identify_groot_features(r, r["actions"], PI05_SPEC)
        r = self._reply(); r["expert@L16"][3, 0, 0] = np.inf
        with self.assertRaises(AlignmentError):
            identify_groot_features(r, r["actions"], PI05_SPEC)

    def test_round_trip_through_npz(self) -> None:
        rec = build_rollout("LEVEL1", 0, 0, with_features=False, n_policy=2, replan=10, horizon=50)
        for i, pol in enumerate(rec.policies):
            r = self._reply(steps=4); r["expert@L16"] += i
            pol.features = identify_groot_features(r, pol.predicted_action_chunk, PI05_SPEC)
        with tempfile.TemporaryDirectory() as td:
            save_rollout(Path(td) / "ep", rec)
            with np.load(Path(td) / "ep" / "rollout.npz") as z:
                self.assertEqual(z["expert@L16"].shape, (2, 4, 50, 1024))
                self.assertEqual(z["expert_t@L16"].shape, (2, 4))
            back = load_rollout(Path(td) / "ep")
            self.assertTrue(np.array_equal(back.policies[1].features.extra["expert@L16"],
                                           rec.policies[1].features.extra["expert@L16"]))


class Pi05RoundTripTest(unittest.TestCase):
    def test_save_load_validate_with_extras_and_no_state(self) -> None:
        rec = build_rollout("LEVEL1", 0, 0, with_features=False, n_policy=2, replan=10, horizon=50)
        for i, pol in enumerate(rec.policies):
            r = reply(layers=("L16", "L17"), seed=i)
            pol.features = identify_groot_features(r, pol.predicted_action_chunk, PI05_SPEC)
        from groot_obs import decode_action_step as decode  # the fake rollout's executed actions use this
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "ep000"
            save_rollout(d, rec)
            with np.load(d / "rollout.npz") as z:
                self.assertNotIn("state_features", z.files)
                self.assertEqual(z["base_image"].shape, (2, 256, 2048))
                self.assertEqual(z["base_image@L17"].shape, (2, 256, 2048))
                self.assertEqual(int(z["img_tokens"]), 256)
                self.assertEqual(z["predicted_action_chunks"].shape, (2, 50, 7))
            back = load_rollout(d)
            f0 = back.policies[0].features
            self.assertIsNone(f0.state_features)
            self.assertIn("language@L17", f0.extra)
            self.assertTrue(np.array_equal(f0.extra["base_image@L17"], rec.policies[0].features.extra["base_image@L17"]))
            self.assertTrue(validate_rollout(back, decode).all_passed)


class Pi05BridgeTest(unittest.TestCase):
    def test_obs_layout_and_flip_conventions(self) -> None:
        img = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
        obs = {"agentview_image": img, "robot0_eye_in_hand_image": img, "robot0_eef_pos": [0.1, 0.2, 0.3],
               "robot0_eef_quat": [0.0, 0.0, 0.0, 1.0], "robot0_gripper_qpos": [0.04, -0.04]}
        nat, _ = make_bridge(native=True)
        nv, _ = make_bridge(native=False)
        a, b = nat(obs, "go"), nv(obs, "go")
        self.assertEqual(a["state"].shape, (8,))
        self.assertTrue(np.array_equal(a["image"], img[::-1]))
        self.assertTrue(np.array_equal(b["image"], img[::-1, ::-1]))
        self.assertEqual(a["prompt"], "go")

    def test_decode_clips(self) -> None:
        _, dec = make_bridge(native=True)
        self.assertEqual(dec([5, -5, 0.5, 0, 0, 0, 0.3]), [1.0, -1.0, 0.5, 0.0, 0.0, 0.0, 0.30000001192092896])


if __name__ == "__main__":
    unittest.main()
