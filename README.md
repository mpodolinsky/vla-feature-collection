# vla-feature-collection

Roll out **GR00T-N1.7** on **[LIBERO-X](https://github.com/meituan/LIBERO-X)** and cache the
frozen VLM backbone's per-token hidden states ("embeddings") for every policy call, together
with actions, clocks, success and videos.

Policy and simulator run in separate venvs and talk over an openpi-style msgpack websocket:

```
top-level .venv (py3.11)                     submodules/Isaac-GR00T/.venv (py3.10)
  scripts/groot-libero-x/collect.py   ──ws──►  scripts/groot-libero-x/server/serve_groot_ws.py
  LIBERO-X env, rollout loop, writer           Gr00tPolicy + layer-16 feature capture
```

## Setup

Accept the [Cosmos-Reason2-2B](https://huggingface.co/nvidia/Cosmos-Reason2-2B) licence (the
GR00T backbone) and log in with `hf auth login`, then:

```bash
git clone --recurse-submodules <this repo>   # Isaac-GR00T @ n1.7-release, LIBERO-X
scripts/groot-libero-x/setup.sh              # both venvs + checkpoint (large: torch cu128, flash-attn)
```

Checkpoint: [`rohansiva/gr00t-libero-x-level123`](https://huggingface.co/rohansiva/gr00t-libero-x-level123)
(GR00T-N1.7 fine-tuned on LIBERO-X LEVEL1-3).

## Collect

```bash
# one command: starts the server if needed, collects, stops the server it started
WITH_FEATURES=1 scripts/groot-libero-x/collect.sh --level LEVEL1 --n-rollouts 10

# all three levels, then a per-level success summary
WITH_FEATURES=1 scripts/groot-libero-x/collect_all.sh

# manual: server in one terminal, client in another
scripts/groot-libero-x/serve.sh checkpoints/gr00t-libero-x-level123 --with-features
MUJOCO_GL=egl uv run python scripts/groot-libero-x/collect.py --level LEVEL1 --with-features
```

Tasks come from `scripts/groot-libero-x/tasks/LEVEL{1,2,3}.txt` (72 / 73 / 73 BDDL files;
`selection.csv` has the per-task metadata). Pass `--tasks-file` for another list.
Protocol follows LIBERO-X's `eval_template.py`: `max_steps=1200`, fresh env per episode,
init-state reset, native image convention. `collect.py` resumes: finished episodes are skipped.

## Output

`outputs/groot-libero-x/collect/<LEVEL>/<NN>_<task>/ep<NNN>/` with `rollout.json`,
`rollout.npz`, `rollout.mp4`, `wrist.mp4`, plus a `manifest.csv` (rewritten after every episode).
Each episode is reloaded and passed through an alignment gate before it enters the manifest.

`rollout.npz` with `--with-features` (P = number of policy calls, fp16), keyed row-for-row to `policy_step`:

| key | shape | |
|---|---|---|
| `base_image`, `wrist_image` | `(P, 64, 2048)` | layer-16 hidden states of the image tokens (2x2-merged patches) |
| `language`, `language_mask` | `(P, 200, 2048)`, `(P, 200)` | instruction tokens, zero-padded, real-vs-pad mask |
| `language_len` | `(P,)` | real instruction token count |
| `state_features` | `(P, 1536)` | the action head's embedded proprio vector |
| `executed_actions`, `predicted_action_chunks` | `(T, 7)`, `(P, 16, 7)` | what the sim ran / what the policy predicted |
| `control_step`, `sim_step`, `policy_step`, `chunk_index`, `video_frame_id` | `(T,)` | the aligned clocks |

Features are the **raw layer-16 residual stream** of the Cosmos-Reason2-2B (Qwen3-VL) backbone,
taken before the action head's `vlln` + VL self-attention.
`--replan-steps` (default 25) is how many actions of each 16-step chunk run before re-querying.

## Tests

Pure unit tests (no simulator, GPU or network):

```bash
cd scripts/groot-libero-x && python -m unittest discover -s tests
```

## pi0.5 hook (reference)

`scripts/pi05-hook/` holds the π0.5 counterpart of the feature server so the same capture can be
added for π0.5 on LIBERO-X later: `pi0_features.py` (a fork of openpi's `Pi0.sample_actions` that
also returns the prefix hidden states it already computes), `serve_pi05_with_features.py`
(feature-serving websocket server) and `verify_features.py` (numeric check against upstream).
They need an [openpi](https://github.com/Physical-Intelligence/openpi) checkout and venv at
`submodules/openpi` (not included) and are not wired into the collector.

## Credits and licence

Built on NVIDIA [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T), Meituan
[LIBERO-X](https://github.com/meituan/LIBERO-X) and Physical Intelligence
[openpi](https://github.com/Physical-Intelligence/openpi) (Apache-2.0; `server/msgpack_numpy.py` and
`pi05-hook/` derive from it). This repo has no licence file yet; check the upstream licences
before redistributing.
