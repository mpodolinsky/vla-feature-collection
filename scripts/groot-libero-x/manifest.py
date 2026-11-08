"""Global manifest + auto-resume for the GR00T LIBERO-X collection.

The grid is ``(level, task_index, episode)`` where
``task_index`` is the 0-based position of a task in the sampled list
(``tasks/<LEVEL>.txt`` or ``sample_tasks``). Layout:
``<output>/<LEVEL>/<NN>_<stem>/ep<NNN>/`` -- still two dirs deep, so the
``rebuild_manifest`` glob is unchanged.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

MANIFEST_FIELDS = (
    "level",
    "task_index",
    "task",
    "prompt",
    "episode",
    "rollout_id",
    "success",
    "n_policy",
    "n_control",
    "control_hz",
    "replan_steps",
    "sim_failure_category",
    "failing_predicate",
    "elapsed_seconds",
    "has_features",
    "dir",
)


def task_dirname(task_index: int, task_stem: str) -> str:
    return f"{task_index:02d}_{task_stem}"


def episode_dir(output_dir: Path, level: str, task_index: int, task_stem: str, episode: int) -> Path:
    return output_dir / level / task_dirname(task_index, task_stem) / f"ep{episode:03d}"


def episode_is_complete(directory: Path, *, expect_video: bool = True) -> bool:
    meta = directory / "rollout.json"
    npz = directory / "rollout.npz"
    if not (meta.is_file() and npz.is_file()):
        return False
    if expect_video and not (
        (directory / "rollout.mp4").is_file() and (directory / "wrist.mp4").is_file()
    ):
        return False
    try:
        json.loads(meta.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return False
    return True


def _row_from_meta(meta: dict, directory: Path, output_dir: Path) -> dict:
    task_stem = meta.get("task_name", "") or Path(meta.get("task_file", "")).stem
    return {
        "level": meta.get("level", ""),
        "task_index": meta.get("task_index", ""),
        "task": task_stem,
        "prompt": meta.get("instruction", ""),
        "episode": meta.get("episode_index", ""),
        "rollout_id": meta.get("rollout_id", ""),
        "success": bool(meta.get("success", False)),
        "n_policy": meta.get("n_policy", ""),
        "n_control": meta.get("n_control", ""),
        "control_hz": meta.get("control_hz", ""),
        "replan_steps": meta.get("replan_steps", ""),
        "sim_failure_category": meta.get("sim_failure_category", ""),
        "failing_predicate": meta.get("failing_predicate", ""),
        "elapsed_seconds": meta.get("elapsed_seconds", ""),
        "has_features": bool(meta.get("has_features", False)),
        "dir": str(directory.relative_to(output_dir)),
    }


def rebuild_manifest(output_dir: Path) -> list[dict]:
    """Scan every saved episode under ``output_dir`` and rewrite ``manifest.csv``
    atomically. Called after every episode so an interrupted run still leaves a
    complete manifest for whatever made it to disk."""
    output_dir = Path(output_dir)
    rows: list[dict] = []
    for meta_path in sorted(output_dir.glob("*/*/ep*/rollout.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        rows.append(_row_from_meta(meta, meta_path.parent, output_dir))

    manifest_path = output_dir / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest_path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(manifest_path)
    return rows


def find_next_incomplete(
    output_dir: Path,
    task_stems: dict[tuple[str, int], str],
    num_trials: int,
    *,
    levels: tuple[str, ...],
    expect_video: bool = True,
) -> tuple[str, int, int] | None:
    """First ``(level, task_index, episode)`` under ``output_dir`` that is not
    yet complete, iterating level-major then task then episode (the same order
    ``collect.py`` fills the grid). ``None`` when every cell is complete.

    ``task_stems`` is ``{(level, task_index): bddl_stem}`` for the sampled task
    list(s).
    """
    for level in levels:
        n_tasks = sum(1 for (lv, _idx) in task_stems if lv == level)
        for task_index in range(n_tasks):
            stem = task_stems.get((level, task_index))
            if stem is None:
                return level, task_index, 0
            for episode in range(num_trials):
                directory = episode_dir(output_dir, level, task_index, stem, episode)
                if not episode_is_complete(directory, expect_video=expect_video):
                    return level, task_index, episode
    return None
