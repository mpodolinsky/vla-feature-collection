#!/usr/bin/env python3
"""Upload finished episodes to a Hugging Face dataset, verify them, then delete the heavy local files.

For every complete, not-yet-shipped episode directory ``<out>/<LEVEL>/<NN>_<task>/ep<NNN>/`` the whole
directory (``rollout.json``, ``rollout.npz``, ``rollout.mp4``, ``wrist.mp4``) is committed to the dataset
at the same relative path. Then every uploaded file is checked against the Hub (size, and the LFS sha256
or git blob sha1). Only if **all** files of the batch match are the heavy files deleted: ``shipped.json``
is written first, then ``rollout.npz`` / ``rollout.mp4`` / ``wrist.mp4`` are removed. ``rollout.json`` and
``manifest.csv`` stay, so the manifest is still complete and ``collect.py`` skips shipped episodes.

Any mismatch or error stops the run with nothing deleted. The Hub is the only copy afterwards.

    .venv/bin/python scripts/groot-libero-x/ship_to_hub.py --output-dir outputs/pi05-libero-x/collect \\
        --repo-id <user>/<dataset> [--create --private] [--max-commit-gb 10] [--no-delete]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifest import HEAVY_FILES, SHIPPED_MARKER, episode_is_shipped, rebuild_manifest  # noqa: E402

log = logging.getLogger("ship")
LFS_THRESHOLD = 10 * 1024 * 1024  # the Hub stores files above this in LFS/Xet (sha256 is exposed)


@dataclass
class Episode:
    directory: Path
    rel_dir: str  # path of the episode directory relative to the output dir (= path in the repo)

    def files(self) -> list[tuple[Path, str]]:
        names = ["rollout.json", *[n for n in HEAVY_FILES if (self.directory / n).is_file()]]
        return [(self.directory / n, f"{self.rel_dir}/{n}") for n in names]

    def heavy_paths(self) -> list[Path]:
        return [self.directory / n for n in HEAVY_FILES if (self.directory / n).is_file()]

    def nbytes(self) -> int:
        return sum(p.stat().st_size for p, _ in self.files())


def pending_episodes(output_dir: Path, *, expect_video: bool = True) -> list[Episode]:
    """Complete episodes (all files present) that have not been shipped yet, in path order."""
    output_dir = Path(output_dir)
    need = HEAVY_FILES if expect_video else ("rollout.npz",)
    eps = []
    for meta in sorted(output_dir.glob("*/*/ep*/rollout.json")):
        d = meta.parent
        if episode_is_shipped(d) or not all((d / n).is_file() for n in need):
            continue
        if any(p.name.endswith(".writing") or ".writing" in p.name for p in d.iterdir()):
            continue  # a save is in progress
        eps.append(Episode(d, d.relative_to(output_dir).as_posix()))
    return eps


def plan_commits(files: list[tuple[Path, str]], max_bytes: int) -> list[list[tuple[Path, str]]]:
    """Group files into commits of at most ``max_bytes`` (a single larger file gets its own commit)."""
    groups, cur, size = [], [], 0
    for item in files:
        n = item[0].stat().st_size
        if cur and size + n > max_bytes:
            groups.append(cur)
            cur, size = [], 0
        cur.append(item)
        size += n
    if cur:
        groups.append(cur)
    return groups


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def upload(api, repo_id: str, groups, *, message: str) -> list[str]:
    from huggingface_hub import CommitOperationAdd

    commits = []
    for i, group in enumerate(groups, 1):
        ops = [CommitOperationAdd(path_in_repo=rel, path_or_fileobj=str(path)) for path, rel in group]
        gb = sum(p.stat().st_size for p, _ in group) / 1e9
        t0 = time.time()
        info = api.create_commit(repo_id=repo_id, repo_type="dataset", operations=ops,
                                 commit_message=f"{message} ({i}/{len(groups)})")
        log.info("commit %d/%d: %d files, %.2f GB in %.0fs (%.0f MB/s)", i, len(groups), len(group), gb,
                 time.time() - t0, gb * 1000 / max(time.time() - t0, 1e-9))
        commits.append(str(getattr(info, "oid", "") or getattr(info, "commit_url", "")))
    return commits


def verify(api, repo_id: str, files: list[tuple[Path, str]], *, chunk: int = 200) -> list[str]:
    """Compare every local file with the Hub. Returns a list of problems; empty means all verified."""
    problems: list[str] = []
    remote = {}
    for i in range(0, len(files), chunk):
        paths = [rel for _, rel in files[i: i + chunk]]
        for entry in api.get_paths_info(repo_id, paths, expand=True, repo_type="dataset"):
            if hasattr(entry, "size"):
                remote[entry.path] = entry
    for path, rel in files:
        rf = remote.get(rel)
        if rf is None:
            problems.append(f"missing on the Hub: {rel}")
            continue
        local_size = path.stat().st_size
        if rf.size != local_size:
            problems.append(f"size mismatch {rel}: local {local_size} != remote {rf.size}")
            continue
        lfs = getattr(rf, "lfs", None)
        if lfs is not None:
            want = lfs["sha256"] if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
            if want is None or sha256_file(path) != want:
                problems.append(f"sha256 mismatch {rel}")
        elif local_size >= LFS_THRESHOLD:
            problems.append(f"no hash available for large file {rel}; refusing to treat it as verified")
        elif git_blob_sha1(path) != getattr(rf, "blob_id", None):
            problems.append(f"blob sha1 mismatch {rel}")
    return problems


def mark_shipped_and_prune(ep: Episode, repo_id: str, commits: list[str], *, delete: bool = True) -> int:
    """Write the marker first (so a crash can never leave an episode looking incomplete), then delete."""
    files = {rel: path.stat().st_size for path, rel in ep.files()}
    marker = {"repo_id": repo_id, "commits": commits, "files": files, "shipped_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    tmp = ep.directory / (SHIPPED_MARKER + ".tmp")
    tmp.write_text(json.dumps(marker, indent=1), encoding="utf-8")
    tmp.replace(ep.directory / SHIPPED_MARKER)
    freed = 0
    if delete:
        for p in ep.heavy_paths():
            freed += p.stat().st_size
            p.unlink()
    return freed


def ship(output_dir: Path, repo_id: str, api, *, max_commit_bytes: int = 10 * 10**9, delete: bool = True,
         limit_episodes: int | None = None, expect_video: bool = True) -> dict:
    """Upload + verify + prune every pending episode. Raises RuntimeError (nothing deleted) on any problem."""
    output_dir = Path(output_dir)
    eps = pending_episodes(output_dir, expect_video=expect_video)
    if limit_episodes is not None:
        eps = eps[:limit_episodes]
    if not eps:
        log.info("nothing to ship")
        return {"episodes": 0, "bytes": 0, "freed": 0}
    files = [f for ep in eps for f in ep.files()]
    total = sum(p.stat().st_size for p, _ in files)
    log.info("shipping %d episodes, %d files, %.2f GB to %s", len(eps), len(files), total / 1e9, repo_id)
    commits = upload(api, repo_id, plan_commits(files, max_commit_bytes), message=f"Add {len(eps)} episodes")
    problems = verify(api, repo_id, files)
    if problems:
        raise RuntimeError("verification failed, NOTHING was deleted:\n  " + "\n  ".join(problems[:20]))
    log.info("verified %d files on the Hub (size + hash)", len(files))
    freed = sum(mark_shipped_and_prune(ep, repo_id, commits, delete=delete) for ep in eps)
    rebuild_manifest(output_dir)
    manifest = output_dir / "manifest.csv"
    if manifest.is_file():  # keep the repo's manifest current (small; not part of the deletion)
        from huggingface_hub import CommitOperationAdd
        api.create_commit(repo_id=repo_id, repo_type="dataset", commit_message="Update manifest.csv",
                          operations=[CommitOperationAdd(path_in_repo="manifest.csv", path_or_fileobj=str(manifest))])
    log.info("deleted %.2f GB of local files" % (freed / 1e9) if delete else "kept local files (--no-delete)")
    return {"episodes": len(eps), "bytes": total, "freed": freed}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--repo-id", required=True, help="<user>/<dataset>")
    ap.add_argument("--create", action="store_true", help="create the dataset repo if it does not exist")
    ap.add_argument("--private", action="store_true", help="with --create: make it private (default public)")
    ap.add_argument("--max-commit-gb", type=float, default=10.0)
    ap.add_argument("--no-delete", action="store_true", help="upload and verify but keep the local files")
    ap.add_argument("--limit-episodes", type=int, default=None)
    args = ap.parse_args(argv)

    from huggingface_hub import HfApi

    api = HfApi()
    log.info("logged in as %s", api.whoami()["name"])
    if args.create:
        api.create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    out = ship(args.output_dir, args.repo_id, api, max_commit_bytes=int(args.max_commit_gb * 1e9),
               delete=not args.no_delete, limit_episodes=args.limit_episodes)
    log.info("done: %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
