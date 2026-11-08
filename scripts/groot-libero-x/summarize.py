#!/usr/bin/env python3
"""Per-level / per-task success-rate rollup for a LIBERO-X collection run.

(LIBERO-X has no normal/occluded pairing, so this is a per-level rollup.)
Reads the single ``manifest.csv`` written by ``collect.py`` and writes
``<output-dir>/summary.json``:

- overall SR
- per level: SR, episode count, sim_failure_category counts
- per (level, task): SR, episode count
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from constants import DEFAULT_COLLECT_DIR  # noqa: E402


def _truthy(value: str) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_COLLECT_DIR)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    manifest = args.output_dir / "manifest.csv"
    if not manifest.is_file():
        raise SystemExit(f"no manifest.csv under {args.output_dir}")

    with manifest.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise SystemExit("manifest.csv is empty")

    per_level: dict[str, dict] = defaultdict(lambda: {"n": 0, "succ": 0, "fail_cat": Counter()})
    per_task: dict[tuple[str, str], dict] = defaultdict(lambda: {"n": 0, "succ": 0})
    total_n = total_succ = 0

    for r in rows:
        level = r.get("level", "")
        task = r.get("task", "")
        ok = _truthy(r.get("success", ""))
        total_n += 1
        total_succ += int(ok)
        L = per_level[level]
        L["n"] += 1
        L["succ"] += int(ok)
        L["fail_cat"][r.get("sim_failure_category", "") or "unknown"] += 1
        T = per_task[(level, task)]
        T["n"] += 1
        T["succ"] += int(ok)

    summary = {
        "output_dir": str(args.output_dir),
        "episodes": total_n,
        "overall_success_rate": total_succ / total_n if total_n else 0.0,
        "per_level": {
            level: {
                "episodes": d["n"],
                "success_rate": d["succ"] / d["n"] if d["n"] else 0.0,
                "sim_failure_category": dict(d["fail_cat"]),
            }
            for level, d in sorted(per_level.items())
        },
        "per_task": {
            f"{level}/{task}": {
                "episodes": d["n"],
                "success_rate": d["succ"] / d["n"] if d["n"] else 0.0,
            }
            for (level, task), d in sorted(per_task.items())
        },
    }

    out = args.out or (args.output_dir / "summary.json")
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"overall SR {summary['overall_success_rate']:.1%}  ({total_n} episodes)")
    for level, d in summary["per_level"].items():
        print(f"  {level}: {d['success_rate']:.1%}  ({d['episodes']} eps)")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
