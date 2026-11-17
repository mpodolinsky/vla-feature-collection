#!/usr/bin/env python3
"""Quick success-rate check of a pi0.5 policy on LIBERO-X (client side).

Runs in the top-level venv (robosuite + LIBERO-X); the policy runs in
``scripts/pi05-libero-x/.venv`` behind ``serve_pi05.py`` and is reached over the openpi
websocket protocol. Same episode protocol as ``scripts/groot-libero-x/collect.py``: fresh
env per episode, LIBERO-X init-state reset (no dummy settle steps), ``max_steps=1200``.

Per level it samples ``--n-tasks`` tasks from ``scripts/groot-libero-x/tasks/<LEVEL>.txt``
(seeded) and runs ``--n-episodes`` init states each. Prints a success table and writes
``results.json`` (+ the first frame sent to the model for each convention check).

    MUJOCO_GL=egl uv run python scripts/pi05-libero-x/eval_quick.py \\
        --levels LEVEL1 LEVEL2 LEVEL3 --n-tasks 4 --n-episodes 2
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "groot-libero-x"))

from constants import EvalConfig, TASKS_DIR  # noqa: E402
from libero_x_env import load_task, make_env, quat2axisangle  # noqa: E402


def build_obs(obs: dict, prompt: str, native: bool) -> dict:
    """LIBERO obs -> pi0.5 server request. State = [eef_pos(3), axis-angle(3), gripper_qpos(2)]
    (the layout of the LIBERO-X LeRobot dataset the adapter was trained on)."""
    sl = (slice(None, None, -1),) if native else (slice(None, None, -1), slice(None, None, -1))
    img = np.ascontiguousarray(obs["agentview_image"][sl]).astype(np.uint8)
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][sl]).astype(np.uint8)
    state = np.concatenate([
        np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
        quat2axisangle(obs["robot0_eef_quat"]).astype(np.float32),
        np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
    ])
    return {"image": img, "wrist_image": wrist, "state": state, "prompt": str(prompt)}


def run_episode(env, client, state_vec, ep_seed, prompt, *, native, replan, max_steps, binarize_gripper):
    env.seed(ep_seed)
    env.reset()
    client.reset()
    obs = env.regenerate_obs_from_state(state_vec.numpy() if hasattr(state_vec, "numpy") else state_vec)
    plan: list[np.ndarray] = []
    n_infer, t_infer, success, step = 0, 0.0, False, 0
    first_req = None
    while not success and step < max_steps:
        if not plan:
            req = build_obs(obs, prompt, native)
            first_req = first_req or req
            t0 = time.time()
            actions = np.asarray(client.infer(req)["actions"], dtype=np.float32)
            t_infer += time.time() - t0
            n_infer += 1
            plan = list(actions[: min(replan, actions.shape[0])])
        a = plan.pop(0).copy()
        a = np.clip(a, -1.0, 1.0)
        if binarize_gripper:
            a[6] = 1.0 if a[6] > 0 else -1.0
        obs, _, done, _ = env.step(a.tolist())
        success = bool(done)
        step += 1
    return {"success": success, "steps": step, "n_infer": n_infer,
            "mean_infer_s": t_infer / max(n_infer, 1)}, first_req


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--levels", nargs="+", default=["LEVEL1", "LEVEL2", "LEVEL3"])
    ap.add_argument("--n-tasks", type=int, default=4, help="tasks sampled per level (seeded)")
    ap.add_argument("--n-episodes", type=int, default=2, help="init states per task")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--replan-steps", type=int, default=10, help="actions executed per chunk (adapter: n_action_steps=10)")
    ap.add_argument("--max-steps", type=int, default=EvalConfig.max_steps)
    ap.add_argument("--convention", choices=("native", "nvidia"), default="native")
    ap.add_argument("--binarize-gripper", action="store_true")
    ap.add_argument("--tag", default="adapter", help="label for this run in results.json")
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/pi05-libero-x/quick_eval"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    client = WebsocketClientPolicy(args.host, args.port)
    meta = client.get_server_metadata() or {}
    logging.info("server: %s", meta)
    out = args.output_dir / args.tag
    out.mkdir(parents=True, exist_ok=True)
    native = args.convention == "native"

    rows = []
    for level in args.levels:
        tasks = [t.strip() for t in (TASKS_DIR / f"{level}.txt").read_text().splitlines() if t.strip()]
        chosen = sorted(random.Random(args.seed).sample(tasks, min(args.n_tasks, len(tasks))))
        for task_file in chosen:
            cfg = EvalConfig(level=level, replan_steps=args.replan_steps, max_steps=args.max_steps)
            bddl, states, prompt = load_task(cfg, task_file)
            for ep in range(min(args.n_episodes, len(states))):
                env = make_env(cfg, bddl)
                try:
                    res, req = run_episode(env, client, states[ep], ep, prompt, native=native,
                                           replan=args.replan_steps, max_steps=args.max_steps,
                                           binarize_gripper=args.binarize_gripper)
                finally:
                    env.close()
                res.update(level=level, task=task_file, episode=ep, prompt=prompt)
                rows.append(res)
                logging.info("%s %-70.70s ep%d  success=%s steps=%d infer=%.0fms", level, task_file, ep,
                             res["success"], res["steps"], 1000 * res["mean_infer_s"])
                if len(rows) == 1 and req is not None:  # frame sent to the model, for orientation checks
                    import imageio.v2 as imageio
                    imageio.imwrite(out / "first_agentview.png", req["image"])
                    imageio.imwrite(out / "first_wrist.png", req["wrist_image"])
                (out / "results.json").write_text(json.dumps(
                    {"tag": args.tag, "args": {k: str(v) for k, v in vars(args).items()}, "server": str(meta),
                     "episodes": rows}, indent=1))

    print(f"\n== {args.tag}  convention={args.convention} replan={args.replan_steps} max_steps={args.max_steps}")
    tot = [0, 0]
    for level in args.levels:
        r = [x for x in rows if x["level"] == level]
        s = sum(x["success"] for x in r)
        tot[0] += s; tot[1] += len(r)
        print(f"{level}: {s}/{len(r)} = {100 * s / max(len(r), 1):.0f}%")
    print(f"ALL   : {tot[0]}/{tot[1]} = {100 * tot[0] / max(tot[1], 1):.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
