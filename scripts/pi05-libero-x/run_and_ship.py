#!/usr/bin/env python3
"""Collect pi0.5 features on LIBERO-X task by task and ship them to a Hugging Face dataset in batches.

One pi0.5 server is started for the whole run. For every task ``collect.py --policy pi05`` runs
``--n-rollouts`` episodes; once the finished-but-not-yet-uploaded episodes reach ``--ship-gb`` they are
uploaded, verified against the Hub (size + hash) and only then deleted locally (``ship_to_hub.py``).
The run resumes: finished and shipped episodes are skipped. Any error stops the run with nothing deleted.

Run it in the top-level venv:

    .venv/bin/python scripts/pi05-libero-x/run_and_ship.py --repo-id <user>/<dataset> --visibility public

Saved per policy call (defaults): backbone blocks 16 and 8 as full tokens, mean-pooled per-modality 6144-d
vectors of all 19 backbone layers, all 18 expert blocks at denoise steps 0, 5, 9, and the robot state.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "groot-libero-x"))

import ship_to_hub  # noqa: E402
from constants import TASKS_DIR  # noqa: E402

log = logging.getLogger("run_and_ship")


def port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def start_server(args, log_path: Path) -> subprocess.Popen:
    cmd = [str(REPO / "scripts/pi05-libero-x/.venv/bin/python"), str(REPO / "scripts/pi05-libero-x/serve_pi05.py"),
           "--port", str(args.port), "--with-features", "--layers", args.layers, "--pooled-layers", args.pooled_layers,
           "--expert-layers", args.expert_layers, "--expert-steps", args.expert_steps]
    cmd += ["--no-adapter"] if args.no_adapter else ["--adapter", args.adapter]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(cmd, stdout=log_path.open("w"), stderr=subprocess.STDOUT)
    deadline = time.time() + args.server_wait
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early, see {log_path}")
        if port_open(args.port):
            log.info("pi0.5 server ready on :%d (%s)", args.port, " ".join(cmd[2:]))
            return proc
        time.sleep(3)
    proc.terminate()
    raise RuntimeError(f"timed out waiting for the server, see {log_path}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", required=True, help="<user>/<dataset> on the Hub (created if missing)")
    ap.add_argument("--visibility", choices=("public", "private"), required=True,
                    help="visibility used when the dataset repo has to be created (no default on purpose)")
    ap.add_argument("--level", default="LEVEL1")
    ap.add_argument("--n-rollouts", type=int, default=5)
    ap.add_argument("--task-start", type=int, default=0)
    ap.add_argument("--task-stop", type=int, default=None)
    ap.add_argument("--ship-gb", type=float, default=50.0, help="upload once this much is waiting locally")
    ap.add_argument("--max-commit-gb", type=float, default=10.0)
    ap.add_argument("--min-free-gb", type=float, default=25.0, help="stop if free disk drops below this")
    ap.add_argument("--no-delete", action="store_true", help="upload and verify but keep the local files")
    ap.add_argument("--output-dir", type=Path, default=REPO / "outputs/pi05-libero-x/collect")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--layers", default="16,8", help="backbone blocks saved as full tokens (first = primary)")
    ap.add_argument("--pooled-layers", default="all")
    ap.add_argument("--expert-layers", default=",".join(str(i) for i in range(18)))
    ap.add_argument("--expert-steps", default="0,5,9")
    ap.add_argument("--adapter", default="podolinsky/pi05-libero-x-level123-lora")
    ap.add_argument("--no-adapter", action="store_true")
    ap.add_argument("--server-wait", type=int, default=900)
    args = ap.parse_args(argv)

    logs = REPO / "outputs/pi05-libero-x/logs"
    logs.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(logs / "run_and_ship.log")])

    from huggingface_hub import HfApi

    api = HfApi()
    log.info("Hugging Face user: %s", api.whoami()["name"])
    api.create_repo(args.repo_id, repo_type="dataset", private=args.visibility == "private", exist_ok=True)
    from huggingface_hub import CommitOperationAdd

    card = Path(__file__).with_name("DATASET_CARD.md")
    if card.is_file() and not api.file_exists(args.repo_id, "README.md", repo_type="dataset"):
        api.create_commit(repo_id=args.repo_id, repo_type="dataset", commit_message="Add dataset card",
                          operations=[CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(card))])
        log.info("uploaded the dataset card")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    n_tasks = len([ln for ln in (TASKS_DIR / f"{args.level}.txt").read_text().splitlines() if ln.strip()])
    stop = n_tasks if args.task_stop is None else min(args.task_stop, n_tasks)
    commit_bytes = int(args.max_commit_gb * 1e9)

    def waiting_bytes() -> int:
        return sum(e.nbytes() for e in ship_to_hub.pending_episodes(out))

    def ship_now() -> None:
        res = ship_to_hub.ship(out, args.repo_id, api, max_commit_bytes=commit_bytes, delete=not args.no_delete)
        log.info("shipped: %s | free disk %.0f GB", res, shutil.disk_usage(out).free / 1e9)

    server = start_server(args, logs / "pi05_server.log")
    env = {**os.environ, "MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl"),
           "PYOPENGL_PLATFORM": os.environ.get("PYOPENGL_PLATFORM", "egl")}
    try:
        for task in range(args.task_start, stop):
            if shutil.disk_usage(out).free < args.min_free_gb * 1e9:
                ship_now()
                if shutil.disk_usage(out).free < args.min_free_gb * 1e9:
                    raise RuntimeError("free disk below --min-free-gb even after shipping; stopping")
            cmd = [str(REPO / ".venv/bin/python"), str(REPO / "scripts/groot-libero-x/collect.py"), "--policy", "pi05",
                   "--port", str(args.port), "--level", args.level, "--n-rollouts", str(args.n_rollouts),
                   "--with-features", "--task-start", str(task), "--task-stop", str(task + 1), "--keep-going", "--fail-on-issues",
                   "--output-dir", str(out)]
            t0 = time.time()
            rc = subprocess.run(cmd, env=env).returncode
            if rc != 0:
                raise RuntimeError(f"collect.py failed for task {task} (exit {rc}); nothing was deleted")
            if server.poll() is not None:
                raise RuntimeError("the pi0.5 server died")
            w = waiting_bytes()
            log.info("task %d/%d done in %.0fs | %.1f GB waiting for upload | free disk %.0f GB",
                     task + 1, stop, time.time() - t0, w / 1e9, shutil.disk_usage(out).free / 1e9)
            if w >= args.ship_gb * 1e9:
                ship_now()
        ship_now()  # whatever is left
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
    log.info("all done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
