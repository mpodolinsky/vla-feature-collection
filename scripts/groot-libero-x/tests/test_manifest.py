"""Pure tests for the LIBERO-X manifest + serialization round-trip. No sim/GPU."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import manifest as M  # noqa: E402
from serialization import load_rollout, save_rollout  # noqa: E402

from _fakes import build_rollout  # noqa: E402


class SerializationRoundTripTest(unittest.TestCase):
    def test_save_load_no_features(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "ep000"
            rec = build_rollout("LEVEL1", 0, 0, success=True, n_policy=2, replan=4)
            save_rollout(d, rec)
            self.assertTrue((d / "rollout.json").is_file())
            self.assertTrue((d / "rollout.npz").is_file())

            meta = json.loads((d / "rollout.json").read_text())
            self.assertEqual(meta["level"], "LEVEL1")
            self.assertEqual(meta["task_name"], "EXTENSION_KITCHEN_SCENE1_demo")
            self.assertEqual(meta["task_index"], 0)
            self.assertFalse(meta["has_features"])

            with np.load(d / "rollout.npz") as z:
                self.assertEqual(int(z["img_tokens"]), 64)
                self.assertEqual(int(z["hidden"]), 2048)
                n = int(z["n_control"])
                self.assertTrue(np.array_equal(z["video_frame_id"], np.arange(n)))
                self.assertTrue(np.array_equal(z["sim_step"], z["control_step"]))
                self.assertNotIn("base_image", z.files)

            back = load_rollout(d)
            self.assertEqual(back.level, "LEVEL1")
            self.assertEqual(back.n_control, rec.n_control)
            self.assertIsNone(back.policies[0].features)

    def test_save_load_with_features(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "ep000"
            rec = build_rollout("LEVEL2", 3, 1, with_features=True, n_policy=2, replan=3)
            save_rollout(d, rec)
            meta = json.loads((d / "rollout.json").read_text())
            self.assertTrue(meta["has_features"])
            with np.load(d / "rollout.npz") as z:
                self.assertEqual(z["base_image"].shape, (2, 64, 2048))
                self.assertEqual(z["language"].shape, (2, 200, 2048))
                self.assertEqual(z["language_mask"].shape, (2, 200))
                self.assertEqual(z["state_features"].shape, (2, 1536))
            back = load_rollout(d)
            self.assertIsNotNone(back.policies[0].features)
            self.assertEqual(back.policies[0].features.language_len, 7)


class ManifestTest(unittest.TestCase):
    def test_episode_dir_layout(self) -> None:
        d = M.episode_dir(Path("/out"), "LEVEL1", 4, "KITCHEN_SCENE1_x", 7)
        self.assertEqual(d, Path("/out/LEVEL1/04_KITCHEN_SCENE1_x/ep007"))

    def test_episode_is_complete(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "ep000"
            self.assertFalse(M.episode_is_complete(d))
            rec = build_rollout("LEVEL1", 0, 0, n_policy=1, replan=2)
            save_rollout(d, rec)
            self.assertFalse(M.episode_is_complete(d, expect_video=True))
            self.assertTrue(M.episode_is_complete(d, expect_video=False))
            (d / "rollout.mp4").write_bytes(b"x")
            (d / "wrist.mp4").write_bytes(b"x")
            self.assertTrue(M.episode_is_complete(d, expect_video=True))

    def test_rebuild_manifest_and_fields(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            for ti in range(2):
                for ep in range(2):
                    d = M.episode_dir(out, "LEVEL1", ti, "SCENE_demo", ep)
                    save_rollout(d, build_rollout("LEVEL1", ti, ep, n_policy=1, replan=2))
            rows = M.rebuild_manifest(out)
            self.assertEqual(len(rows), 4)
            self.assertEqual(set(rows[0]), set(M.MANIFEST_FIELDS))
            self.assertTrue((out / "manifest.csv").is_file())

    def test_find_next_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            stems = {("LEVEL1", 0): "SCENE_a", ("LEVEL1", 1): "SCENE_b"}
            save_rollout(M.episode_dir(out, "LEVEL1", 0, "SCENE_a", 0),
                         build_rollout("LEVEL1", 0, 0, n_policy=1, replan=2))
            (M.episode_dir(out, "LEVEL1", 0, "SCENE_a", 0) / "rollout.mp4").write_bytes(b"x")
            (M.episode_dir(out, "LEVEL1", 0, "SCENE_a", 0) / "wrist.mp4").write_bytes(b"x")
            nxt = M.find_next_incomplete(out, stems, 2, levels=("LEVEL1",))
            self.assertEqual(nxt, ("LEVEL1", 0, 1))


if __name__ == "__main__":
    unittest.main()
