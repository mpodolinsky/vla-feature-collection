"""Tests for ship_to_hub: verify-before-delete, resume markers, manifest. Uses a fake Hub, no network."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ship_to_hub as S  # noqa: E402
from manifest import SHIPPED_MARKER, episode_is_complete  # noqa: E402


class FakeApi:
    """Stores uploaded bytes in memory; ``mutate(rel, data) -> data`` lets a test corrupt a file."""

    def __init__(self, mutate=None, drop=()):
        self.store: dict[str, bytes] = {}
        self.commits = 0
        self.mutate, self.drop = mutate, set(drop)

    def create_commit(self, repo_id, repo_type, operations, commit_message):
        self.commits += 1
        for op in operations:
            if op.path_in_repo in self.drop:
                continue
            data = Path(op.path_or_fileobj).read_bytes()
            self.store[op.path_in_repo] = self.mutate(op.path_in_repo, data) if self.mutate else data
        return SimpleNamespace(oid=f"commit{self.commits}")

    def get_paths_info(self, repo_id, paths, expand, repo_type):
        out = []
        for p in paths:
            d = self.store.get(p)
            if d is None:
                continue
            lfs = {"sha256": hashlib.sha256(d).hexdigest()} if len(d) >= S.LFS_THRESHOLD else None
            out.append(SimpleNamespace(path=p, size=len(d), lfs=lfs,
                                       blob_id=hashlib.sha1(b"blob %d\0" % len(d) + d).hexdigest()))
        return out


def make_episode(out: Path, task="00_T", ep=0, *, heavy_bytes=300, video=True) -> Path:
    d = out / "LEVEL1" / task / f"ep{ep:03d}"
    d.mkdir(parents=True)
    (d / "rollout.json").write_text(json.dumps({"level": "LEVEL1", "task_index": 0, "task_name": task,
                                                "episode_index": ep, "rollout_id": f"{task}_{ep}", "success": False}))
    (d / "rollout.npz").write_bytes(bytes(range(256)) * (heavy_bytes // 256 + 1))
    if video:
        (d / "rollout.mp4").write_bytes(b"v" * heavy_bytes)
        (d / "wrist.mp4").write_bytes(b"w" * heavy_bytes)
    return d


class ShipTest(unittest.TestCase):
    def setUp(self) -> None:
        self._thr = S.LFS_THRESHOLD
        S.LFS_THRESHOLD = 200  # make the 300-byte files "large" so the sha256 path is exercised

    def tearDown(self) -> None:
        S.LFS_THRESHOLD = self._thr

    def test_pending_skips_incomplete_shipped_and_in_progress(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            ok = make_episode(out, "00_A", 0)
            make_episode(out, "00_A", 1, video=False)                      # wrist.mp4 / rollout.mp4 missing
            done = make_episode(out, "01_B", 0)
            (done / SHIPPED_MARKER).write_text("{}")
            busy = make_episode(out, "02_C", 0)
            (busy / "rollout.writing.npz").write_bytes(b"x")                # a save is in progress
            self.assertEqual([e.directory for e in S.pending_episodes(out)], [ok])

    def test_ship_verifies_then_deletes_heavy_files_and_keeps_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            d = make_episode(out)
            api = FakeApi()
            res = S.ship(out, "u/ds", api)
            self.assertEqual(res["episodes"], 1)
            self.assertEqual(sorted(p.name for p in d.iterdir()), ["rollout.json", SHIPPED_MARKER])
            self.assertIn("LEVEL1/00_T/ep000/rollout.npz", api.store)
            self.assertIn("manifest.csv", api.store)
            self.assertTrue(episode_is_complete(d))                          # collect.py will skip it
            self.assertIn("00_T", (out / "manifest.csv").read_text())        # manifest still lists it
            self.assertEqual(S.pending_episodes(out), [])
            self.assertEqual(json.loads((d / SHIPPED_MARKER).read_text())["repo_id"], "u/ds")

    def _assert_nothing_deleted(self, out: Path, d: Path) -> None:
        self.assertTrue((d / "rollout.npz").is_file() and (d / "rollout.mp4").is_file())
        self.assertFalse((d / SHIPPED_MARKER).exists())

    def test_size_mismatch_deletes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td); d = make_episode(out)
            with self.assertRaisesRegex(RuntimeError, "size mismatch"):
                S.ship(out, "u/ds", FakeApi(mutate=lambda rel, b: b[:-1] if rel.endswith("rollout.npz") else b))
            self._assert_nothing_deleted(out, d)

    def test_same_size_corruption_deletes_nothing(self) -> None:
        flip = lambda rel, b: bytes([b[0] ^ 1]) + b[1:] if rel.endswith("wrist.mp4") else b  # noqa: E731
        with tempfile.TemporaryDirectory() as td:
            out = Path(td); d = make_episode(out)
            with self.assertRaisesRegex(RuntimeError, "sha256 mismatch"):
                S.ship(out, "u/ds", FakeApi(mutate=flip))
            self._assert_nothing_deleted(out, d)

    def test_small_file_content_mismatch_deletes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td); d = make_episode(out)
            bad = lambda rel, b: b.replace(b"LEVEL1", b"LEVEL9") if rel.endswith("rollout.json") else b  # noqa: E731
            with self.assertRaisesRegex(RuntimeError, "blob sha1 mismatch"):
                S.ship(out, "u/ds", FakeApi(mutate=bad))
            self._assert_nothing_deleted(out, d)

    def test_missing_remote_file_deletes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td); d = make_episode(out)
            with self.assertRaisesRegex(RuntimeError, "missing on the Hub"):
                S.ship(out, "u/ds", FakeApi(drop={"LEVEL1/00_T/ep000/rollout.mp4"}))
            self._assert_nothing_deleted(out, d)

    def test_large_file_without_any_hash_is_not_trusted(self) -> None:
        class NoHash(FakeApi):
            def get_paths_info(self, *a, **k):
                return [SimpleNamespace(path=e.path, size=e.size, lfs=None, blob_id=e.blob_id)
                        for e in super().get_paths_info(*a, **k)]
        with tempfile.TemporaryDirectory() as td:
            out = Path(td); d = make_episode(out)
            with self.assertRaisesRegex(RuntimeError, "no hash available"):
                S.ship(out, "u/ds", NoHash())
            self._assert_nothing_deleted(out, d)

    def test_no_delete_keeps_files_but_marks_shipped(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td); d = make_episode(out)
            S.ship(out, "u/ds", FakeApi(), delete=False)
            self.assertTrue((d / "rollout.npz").is_file() and (d / SHIPPED_MARKER).is_file())

    def test_commits_are_split_by_size(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            for i in range(4):
                make_episode(out, f"0{i}_T", 0, heavy_bytes=1000)
            api = FakeApi()
            S.ship(out, "u/ds", api, max_commit_bytes=4000)
            self.assertGreaterEqual(api.commits, 3)                          # >= 2 data commits + manifest
            self.assertEqual(S.pending_episodes(out), [])

    def test_second_run_ships_only_new_episodes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            make_episode(out, "00_A", 0)
            api = FakeApi(); S.ship(out, "u/ds", api)
            make_episode(out, "01_B", 0)
            res = S.ship(out, "u/ds", api)
            self.assertEqual(res["episodes"], 1)
            manifest = (out / "manifest.csv").read_text()
            self.assertIn("00_A", manifest); self.assertIn("01_B", manifest)


if __name__ == "__main__":
    unittest.main()
