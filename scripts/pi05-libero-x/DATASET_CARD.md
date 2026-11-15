---
license: other
license_name: see-licence-section
task_categories:
- robotics
tags:
- robotics
- libero
- libero-x
- pi05
- vision-language-action
- embeddings
- failure-detection
pretty_name: pi0.5 features on LIBERO-X Level 1
---

# pi0.5 rollouts and internal features on LIBERO-X, Level 1

Closed-loop rollouts of the LoRA-adapted **pi0.5** policy
[`podolinsky/pi05-libero-x-level123-lora`](https://huggingface.co/podolinsky/pi05-libero-x-level123-lora)
(base [`lerobot/pi05_libero`](https://huggingface.co/lerobot/pi05_libero), LeRobot 0.7.0) in the
[LIBERO-X](https://github.com/meituan/LIBERO-X) simulator, with the policy's hidden states saved at every
policy call. Collected with [`vla-feature-collection`](https://github.com/mpodolinsky/vla-feature-collection)
(`scripts/pi05-libero-x/run_and_ship.py`).

- **Tasks:** the 72 LEVEL1 tasks of the 218-task set (`scripts/groot-libero-x/tasks/LEVEL1.txt` in the code repo).
- **Episodes:** 5 per task (initial states 0-4), 1200 control steps at most, native LIBERO-X image convention.
- **Execution:** the policy predicts a 50-step chunk and 10 actions are executed before the next call, so there are
  at most 120 policy calls per episode.
- **The adapter is a one-epoch checkpoint with a low success rate**, so most episodes are failures.
  Success per episode is in `manifest.csv` (`success`).

## Layout

```
manifest.csv                                  one row per episode
LEVEL1/<NN>_<task>/ep<NNN>/rollout.json       metadata, per-call summary
                           rollout.npz        arrays (below)
                           rollout.mp4        agentview video, 20 fps
                           wrist.mp4          wrist-camera video, 20 fps
```

## `rollout.npz`

`P` = policy calls, `T` = control steps. Everything is fp16 unless noted. Layer numbers are **0-based block
outputs**; the backbone has 18 blocks (17 = last) and `final` is the output after the final norm.

| key | shape | what |
|---|---|---|
| `robot_state` | `(P, 8)` f32 | the policy's state input at each call: end-effector position (3), axis-angle (3), gripper fingers (2) |
| `robot_joint_pos`, `robot_eef_quat` | `(P, 7)`, `(P, 4)` f32 | joint positions, end-effector quaternion (xyzw); not model inputs |
| `pooled@all` | `(P, 19, 6144)` | backbone layers 0-17 and `final` in that order; per layer the mean over the agentview tokens, the mean over the wrist tokens and the mean over the real language tokens, concatenated in that order (3 x 2048) |
| `base_image`, `wrist_image` | `(P, 256, 2048)` | **full tokens of backbone block 16** (last block whose output the action expert reads): 16x16 patches per camera |
| `language`, `language_mask`, `language_len` | `(P, 200, 2048)`, `(P, 200)`, `(P,)` | block-16 language tokens, zero-padded; the discretised robot state is part of this text |
| `base_image@L8`, `wrist_image@L8`, `language@L8` | as above | the same for backbone block 8 |
| `expert@L<k>`, k = 0..17 | `(P, 3, 50, 1024)` | output of action-expert block k on the 50 action tokens at denoise steps 0, 5, 9 |
| `expert_t@L<k>` | `(P, 3)` f32 | flow time of those steps: `[1.0, 0.5, 0.1]` (pi0.5 convention: t=1 is pure noise, the update is `x <- x + dt*v` with dt = -0.1) |
| `predicted_action_chunks` | `(P, 50, 7)` f32 | the model's chunk, un-normalised |
| `executed_actions` | `(T, 7)` f32 | actions sent to the simulator (clipped to [-1, 1]) |
| `control_step`, `sim_step`, `policy_step`, `chunk_index`, `video_frame_id` | `(T,)` i32 | aligned clocks: control step `t` ran action `chunk_index[t]` of policy call `policy_step[t]` |

The backbone features come from the frozen PaliGemma (SigLIP + Gemma-2B) prefix pass over
`[agentview 256 | wrist 256 | empty camera 256 (dropped) | text 200]`. The LoRA only changes the action expert, so
the backbone features equal those of the base model; the expert states depend on the adapter and on the sampled
noise. Backbone block 17 and `final` are never read by the action expert.

## Licence and attribution

Mixed: the simulator tasks derive from LIBERO-X (Meituan, CC BY 4.0; please cite the LIBERO-X paper), the hidden
states were computed with Gemma-based weights (Gemma Terms of Use), and the code is a separate repository. Check the
upstream terms before reuse; this card does not grant additional rights.
