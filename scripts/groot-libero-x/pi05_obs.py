"""Observation / action bridge for the pi0.5 (LeRobot) policy server.

Request layout is what ``scripts/pi05-libero-x/serve_pi05.py`` expects. The state is
``[eef_pos(3), axis-angle(3), gripper_qpos(2)]``, the layout of the LIBERO-X LeRobot dataset the
adapter was trained on. ``native`` (the adapter's data) flips the images vertically only;
``nvidia`` is the 180 degree convention of the stock ``lerobot/pi05_libero`` base.
"""

from __future__ import annotations

import numpy as np

from libero_x_env import quat2axisangle


def proprio(obs: dict) -> dict[str, np.ndarray]:
    """Robot state at a policy call, saved as ``robot_<key>``: ``state`` is exactly the 8-d vector the
    policy is given; ``joint_pos`` (7) and ``eef_quat`` (4, xyzw) are extras that are not model inputs."""
    out = {"state": np.concatenate([
        np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
        quat2axisangle(obs["robot0_eef_quat"]).astype(np.float32),
        np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
    ])}
    if "robot0_joint_pos" in obs:
        out["joint_pos"] = np.asarray(obs["robot0_joint_pos"], dtype=np.float32)
    out["eef_quat"] = np.asarray(obs["robot0_eef_quat"], dtype=np.float32)
    return out


def make_bridge(native: bool):
    """Returns ``(build_flat_obs, decode_action_step)`` for the given image convention."""
    sl = (slice(None, None, -1),) if native else (slice(None, None, -1), slice(None, None, -1))

    def build_flat_obs(obs: dict, prompt: str) -> dict:
        return {
            "image": np.ascontiguousarray(obs["agentview_image"][sl]).astype(np.uint8),
            "wrist_image": np.ascontiguousarray(obs["robot0_eye_in_hand_image"][sl]).astype(np.uint8),
            "state": proprio(obs)["state"],
            "prompt": str(prompt),
        }

    def decode_action_step(raw_action) -> list[float]:
        """One (7,) pi0.5 action row -> robosuite OSC_POSE action (clipped to [-1, 1])."""
        return np.clip(np.asarray(raw_action, dtype=np.float32), -1.0, 1.0).tolist()

    return build_flat_obs, decode_action_step
