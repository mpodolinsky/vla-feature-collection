"""Shared builders for a correctly-aligned fake LIBERO-X rollout (no sim / GPU)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]           # scripts/groot-libero-x
sys.path.insert(0, str(ROOT))

from records import ControlRecord, GrootFeatures, PolicyRecord, RolloutRecord  # noqa: E402
from groot_obs import decode_action_step  # noqa: E402


def fake_features(n_lang: int = 7, *, seed: int = 0) -> GrootFeatures:
    rng = np.random.default_rng(seed)
    lang = np.zeros((200, 2048), np.float16)
    lang[:n_lang] = rng.standard_normal((n_lang, 2048)).astype(np.float16)
    mask = np.zeros((200,), np.bool_)
    mask[:n_lang] = True
    return GrootFeatures(
        base_image=rng.standard_normal((64, 2048)).astype(np.float16),
        wrist_image=(rng.standard_normal((64, 2048)) + 3.0).astype(np.float16),
        language=lang,
        language_mask=mask,
        language_len=n_lang,
        state_features=rng.standard_normal((1536,)).astype(np.float16),
        module="test",
        shapes={"base_image": (64, 2048)},
    )


def build_rollout(
    level: str = "LEVEL1",
    task_index: int = 0,
    episode: int = 0,
    *,
    success: bool = True,
    with_features: bool = False,
    n_policy: int = 3,
    replan: int = 8,
    horizon: int = 16,
    num_steps_wait: int = 10,
) -> RolloutRecord:
    rng = np.random.default_rng(42)
    policies: list[PolicyRecord] = []
    controls: list[ControlRecord] = []
    cs = 0
    for ps in range(n_policy):
        chunk = rng.uniform(-1, 1, (horizon, 7)).astype(np.float32)
        start = cs
        for j in range(replan):
            executed = np.asarray(decode_action_step(chunk[j]), dtype=np.float32)
            controls.append(
                ControlRecord(
                    control_step=cs, sim_step=cs, policy_step=ps,  # init-mode: sim == control
                    chunk_index=j, executed_action=executed, video_frame_id=cs,
                )
            )
            cs += 1
        policies.append(
            PolicyRecord(
                policy_step=ps, predicted_action_chunk=chunk,
                executed_control_step_start=start, executed_control_step_end=cs - 1,
                features=fake_features(seed=ps) if with_features else None,
            )
        )
    stem = "EXTENSION_KITCHEN_SCENE1_demo"
    return RolloutRecord(
        rollout_id=f"{level}__t{task_index:03d}__ep{episode:03d}__{stem}",
        level=level, task_name=stem, task_index=task_index,
        task_file=f"/x/{stem}.bddl", episode_index=episode,
        instruction="do the thing", convention="native",
        replan_steps=replan, success=success, max_steps=1200,
        seed=7, model_id="gr00t-n1.7-libero-x", checkpoint="rohansiva/gr00t-libero-x-level123",
        feature_source="backbone", feature_server="scripts/groot-libero-x/server/serve_groot_ws.py",
        feature_server_version="{'policy': 'gr00t-n1.7'}",
        control_hz=20.0, num_steps_wait=num_steps_wait,
        sim_failure_category="success" if success else "unsatisfied_goal",
        failing_predicate="" if success else "(on obj table)",
        failure_detail="" if success else "(on obj table)",
        elapsed_seconds=12.3, policies=policies, controls=controls,
    )
