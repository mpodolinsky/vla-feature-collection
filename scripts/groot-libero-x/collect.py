#!/usr/bin/env python3
"""Collect aligned GR00T-N1.7 rollouts on LIBERO-X, one episode dir each.

Per episode: ``rollout.json`` + ``rollout.npz`` + ``rollout.mp4`` (agentview, 20 fps) +
``wrist.mp4``. A global ``manifest.csv`` is rewritten after every episode and
collection auto-resumes: a complete episode dir is skipped.

Defaults target the LIBERO-X fine-tune ``rohansiva/gr00t-libero-x-level123`` on
the task lists in ``tasks/<LEVEL>.txt``. Follows LIBERO-X's ``eval_template.py``
protocol (max_steps 1200, fresh env per episode, init-mode reset, native image
convention).

Run in the top-level venv (this is the websocket *client*; the GR00T model
runs in ``submodules/Isaac-GR00T/.venv`` behind ``scripts/groot-libero-x/serve.sh``):

    scripts/groot-libero-x/serve.sh checkpoints/gr00t-libero-x-level123 --with-features
    MUJOCO_GL=egl uv run python scripts/groot-libero-x/collect.py --level LEVEL1 --with-features
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import sys
from pathlib import Path

PI05_CHECKPOINT_LABEL = "podolinsky/pi05-libero-x-level123-lora"
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from constants import (  # noqa: E402
    CUSTOM_CHECKPOINT_LABEL,
    DEFAULT_COLLECT_DIR,
    GROOT_SPEC,
    PI05_SPEC,
    TASKS_DIR,
    LEVELS,
    EvalConfig,
)
from libero_x_env import load_task, make_env  # noqa: E402
from manifest import (  # noqa: E402
    episode_dir,
    episode_is_complete,
    find_next_incomplete,
    rebuild_manifest,
)
from recorder import save_episode  # noqa: E402
from rollout import run_episode  # noqa: E402
from sample_tasks import sample as sample_tasks  # noqa: E402
from serialization import load_rollout  # noqa: E402
from validation import validate_rollout  # noqa: E402

def _load_groot_obs(convention: str):
    """Set the convention env var *before* importing groot_obs (it reads it at import)."""
    if convention == "native":
        os.environ.setdefault("LIBEROX_NATIVE_CONVENTION", "1")
    elif convention == "nvidia":
        os.environ["LIBEROX_NATIVE_CONVENTION"] = "0"
    import groot_obs  # noqa: WPS433

    return groot_obs.build_flat_obs, groot_obs.decode_action_step


def _load_bridge(policy: str, convention: str):
    """(build_flat_obs, decode_action_step) for the chosen policy."""
    if policy == "pi05":
        from pi05_obs import make_bridge

        return make_bridge(native=convention == "native")
    return _load_groot_obs(convention)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--policy", choices=("groot", "pi05"), default="groot",
                   help="which policy server is on --port (pi05: scripts/pi05-libero-x/serve_pi05.py)")
    p.add_argument("--level", choices=LEVELS, default="LEVEL1", help="One level per invocation.")
    p.add_argument("--n-rollouts", type=int, default=10, help="Episodes per task.")
    p.add_argument("--seed", type=int, default=7, help="Only used when sampling tasks (no finetune list).")
    p.add_argument("--n-tasks", type=int, default=5, help="Sample size when there's no finetune list.")
    p.add_argument("--replan-steps", type=int, default=None,
                   help="actions executed per chunk (default: groot 25, pi05 10 = the adapter's n_action_steps)")
    p.add_argument("--max-steps", type=int, default=EvalConfig.max_steps)
    p.add_argument("--num-steps-wait", type=int, default=EvalConfig.num_steps_wait,
                   help="Horizon budget only -- no dummy settle steps are executed.")
    p.add_argument("--env-resolution", type=int, default=256)
    p.add_argument("--convention", choices=("native", "nvidia"), default="native",
                   help="native (default): meituan/LIBERO-X data convention (the custom "
                        "fine-tune). nvidia: NVIDIA's LIBERO convention (stock nvidia/GR00T-N1.7-LIBERO).")
    p.add_argument("--with-features", action="store_true",
                   help="Require + save the layer-16 backbone hidden states "
                        "(server must run with --with-features).")
    p.add_argument("--checkpoint-label", default=None,
                   help="Recorded in rollout.json (default: the policy's LIBERO-X checkpoint).")
    p.add_argument("--tasks-file", type=Path, default=None,
                   help="One BDDL filename per line. Default: tasks/<LEVEL>.txt "
                        "if it exists, else a fresh seeded sample.")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="default: outputs/groot-libero-x/collect or outputs/pi05-libero-x/collect")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--fresh", action="store_true", help="Re-collect episodes even if already complete.")
    p.add_argument("--keep-going", action="store_true", help="Continue past a per-episode error.")
    p.add_argument("--allow-wrap", action="store_true",
                   help="Allow episode index >= number of init states (wraps via modulo). "
                        "Default: skip those episodes.")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="groot-libero-x")
    p.add_argument("--wandb-entity", default=None)
    return p.parse_args(argv)


def _resolve_task_list(args: argparse.Namespace) -> list[str]:
    if args.tasks_file is not None:
        path = args.tasks_file
    else:
        path = TASKS_DIR / f"{args.level}.txt"
    if path.is_file():
        return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    logging.info("no task list at %s -- sampling %d tasks (seed %d)", path, args.n_tasks, args.seed)
    cfg = EvalConfig(level=args.level, seed=args.seed, n_tasks=args.n_tasks)
    chosen = sample_tasks(cfg)
    out = args.output_dir / f"tasks_{args.level}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(chosen) + "\n", encoding="utf-8")
    return chosen


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    if args.replan_steps is None:
        args.replan_steps = 10 if args.policy == "pi05" else EvalConfig.replan_steps
    if args.output_dir is None:
        args.output_dir = (DEFAULT_COLLECT_DIR.parent.parent / "pi05-libero-x" / "collect"
                           if args.policy == "pi05" else DEFAULT_COLLECT_DIR)
    if args.checkpoint_label is None:
        args.checkpoint_label = (PI05_CHECKPOINT_LABEL if args.policy == "pi05" else CUSTOM_CHECKPOINT_LABEL)
    output_dir = args.output_dir.resolve()
    expect_video = not args.no_video
    level = args.level

    build_flat_obs, decode_action_step = _load_bridge(args.policy, args.convention)
    native = args.convention == "native"

    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    client = WebsocketClientPolicy(args.host, args.port)
    meta = client.get_server_metadata() or {}
    server_metadata = str(meta)
    logging.info("GR00T policy server: %s", server_metadata)
    if args.with_features and not meta.get("with_features"):
        logging.error("--with-features requested but the server was not started with --with-features")
        return 2

    spec = GROOT_SPEC
    if args.policy == "pi05":  # token count depends on the server's --pool-image
        spec = dataclasses.replace(PI05_SPEC, img_tokens=int(meta.get("img_tokens", PI05_SPEC.img_tokens)),
                                   module=f"{PI05_SPEC.module}; layers={meta.get('layers')}")
        logging.info("pi0.5 features: layers=%s img_tokens=%d", meta.get("layers"), spec.img_tokens)

    task_files = _resolve_task_list(args)
    task_stems = {(level, i): Path(name).stem for i, name in enumerate(task_files)}
    logging.info("%s: %d tasks x %d rollouts (convention=%s, features=%s)",
                 level, len(task_files), args.n_rollouts, args.convention, args.with_features)

    run = None
    if args.wandb:
        import wandb

        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))

    n_done = n_skipped = n_errored = n_gate_failed = 0
    broken: set[int] = set()

    try:
        for task_index, task_file in enumerate(task_files):
            if task_index in broken:
                continue
            stem = task_stems[(level, task_index)]
            try:
                bddl_path, initial_states, instruction = load_task(
                    EvalConfig(level=level, env_resolution=args.env_resolution), task_file
                )
            except Exception:
                n_errored += 1
                logging.exception("could not load task %s (%s)", task_index, task_file)
                if not args.keep_going:
                    return 1
                broken.add(task_index)
                continue

            for episode in range(args.n_rollouts):
                directory = episode_dir(output_dir, level, task_index, stem, episode)
                if not args.fresh and episode_is_complete(directory, expect_video=expect_video):
                    n_skipped += 1
                    continue
                if episode >= len(initial_states) and not args.allow_wrap:
                    logging.warning("episode %d >= %d init states for %s t%d; skipping "
                                    "(pass --allow-wrap to wrap)",
                                    episode, len(initial_states), level, task_index)
                    continue

                env = None
                try:
                    state_vec = initial_states[episode % len(initial_states)]
                    cfg = EvalConfig(
                        host=args.host, port=args.port, level=level,
                        seed=args.seed, num_steps_wait=args.num_steps_wait,
                        replan_steps=int(args.replan_steps), env_resolution=args.env_resolution,
                        max_steps=args.max_steps, episode_index=episode,
                        task_name=stem, task_index=task_index,
                        save_video=expect_video, checkpoint_label=args.checkpoint_label,
                    )
                    env = make_env(cfg, bddl_path)
                    logging.info("collect %s t%d ep%d  replan=%d", level, task_index, episode, cfg.replan_steps)
                    rollout = run_episode(
                        env=env, client=client, state_vec=state_vec, ep_seed=episode,
                        instruction=instruction, cfg=cfg, task_file=str(bddl_path),
                        task_index=task_index, max_steps=args.max_steps,
                        server_metadata=server_metadata,
                        build_flat_obs=build_flat_obs, decode_action_step=decode_action_step,
                        native=native, spec=spec,
                    )
                    if args.with_features and rollout.policies and rollout.policies[0].features is None:
                        raise RuntimeError(
                            "--with-features set but the server returned no backbone features"
                        )
                    save_episode(directory, rollout, save_video=expect_video)

                    reloaded = load_rollout(directory)  # reload + alignment gate
                    report = validate_rollout(reloaded, decode_action_step)
                    report.add("Reload test", True)
                    if not report.all_passed:
                        n_gate_failed += 1
                        logging.error("ALIGNMENT GATE FAILED for %s\n%s", directory, report.format())
                        if not args.keep_going:
                            logging.error("stopping (pass --keep-going to continue)")
                            return 1
                    else:
                        logging.info(
                            "aligned OK: %s (success=%s, n_control=%d, n_policy=%d, features=%s, %.1fs)",
                            rollout.rollout_id, rollout.success, rollout.n_control, rollout.n_policy,
                            reloaded.policies[0].features is not None if reloaded.policies else False,
                            rollout.elapsed_seconds,
                        )

                    if run is not None:
                        run.log({"level": level, "task_index": task_index, "episode": episode,
                                 "success": int(rollout.success), "n_control": rollout.n_control,
                                 "n_policy": rollout.n_policy})

                    n_done += 1
                    rebuild_manifest(output_dir)
                except KeyboardInterrupt:
                    raise
                except Exception:
                    n_errored += 1
                    logging.exception("EPISODE ERRORED: %s t%d ep%d", level, task_index, episode)
                    rebuild_manifest(output_dir)
                    if not args.keep_going:
                        return 1
                finally:
                    if env is not None:
                        try:
                            env.close()
                        except Exception:
                            logging.exception("env.close() failed")
    finally:
        if run is not None:
            run.finish()

    rows = rebuild_manifest(output_dir)
    remaining = find_next_incomplete(
        output_dir, task_stems, args.n_rollouts, levels=(level,), expect_video=expect_video
    )
    logging.info(
        "collected=%d skipped=%d gate_failed=%d errored=%d  manifest rows=%d  next_incomplete=%s",
        n_done, n_skipped, n_gate_failed, n_errored, len(rows), remaining,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
