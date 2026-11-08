#!/usr/bin/env bash
# One-time setup.
#
#   - init the submodules: Isaac-GR00T (n1.7-release, recursive) and LIBERO-X
#   - build the GR00T policy venv (submodules/Isaac-GR00T/.venv, Python 3.10)
#   - check gated backbone access (nvidia/Cosmos-Reason2-2B)
#   - download the LIBERO-X fine-tune checkpoint (rohansiva/gr00t-libero-x-level123)
#   - build the client venv (top-level .venv, Python 3.11: robosuite + mujoco + openpi-client)
#
# Prereq: accept the Cosmos-Reason2-2B licence while logged in as your HF user
#   https://huggingface.co/nvidia/Cosmos-Reason2-2B
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

GROOT_DIR="${REPO_ROOT}/submodules/Isaac-GR00T"
CKPT_REPO="${CKPT_REPO:-rohansiva/gr00t-libero-x-level123}"
CKPT_DIR="${REPO_ROOT}/checkpoints/$(basename "${CKPT_REPO}")"

command -v uv >/dev/null || { echo "uv not found -- install from https://astral.sh/uv" >&2; exit 1; }
command -v hf >/dev/null || { echo "hf (huggingface_hub) not found" >&2; exit 1; }
command -v ffmpeg >/dev/null || echo "WARN: ffmpeg not on PATH (torchcodec needs it: sudo apt install -y ffmpeg)"

echo ">> init submodules (Isaac-GR00T is large)"
git submodule update --init --recursive submodules/Isaac-GR00T submodules/LIBERO-X
test -d "${REPO_ROOT}/submodules/LIBERO-X/libero/libero_x/bddl/LEVEL1" || {
  echo "ERROR: submodules/LIBERO-X/libero/libero_x/bddl/LEVEL1 missing after init" >&2; exit 1; }

echo ">> GR00T venv: uv sync (python 3.10) -- large (torch cu128 + flash-attn)"
pushd "${GROOT_DIR}" >/dev/null
uv sync --python 3.10
uv pip install "websockets>=13" "msgpack>=1.0"
uv run python -c "import gr00t; from gr00t.policy.gr00t_policy import Gr00tPolicy; print('gr00t import ok')"
popd >/dev/null

echo ">> checking gated backbone access (nvidia/Cosmos-Reason2-2B)"
if ! hf download nvidia/Cosmos-Reason2-2B config.json --quiet >/dev/null 2>&1; then
  echo "ERROR: no access to nvidia/Cosmos-Reason2-2B (the GR00T VLM backbone)." >&2
  echo "  Request access while logged in as $(hf auth whoami 2>/dev/null | head -1):" >&2
  echo "    https://huggingface.co/nvidia/Cosmos-Reason2-2B" >&2
  exit 1
fi
hf download nvidia/Cosmos-Reason2-2B --quiet >/dev/null 2>&1 || true

echo ">> downloading ${CKPT_REPO} -> ${CKPT_DIR}"
hf download "${CKPT_REPO}" --local-dir "${CKPT_DIR}"
test -f "${CKPT_DIR}/config.json" || { echo "ERROR: ${CKPT_DIR}/config.json missing after download" >&2; exit 1; }

echo ">> client venv: uv sync (python 3.11)"
uv sync

echo
echo "Setup OK."
echo "  terminal 1:  scripts/groot-libero-x/serve.sh ${CKPT_DIR} --with-features"
echo "  terminal 2:  MUJOCO_GL=egl uv run python scripts/groot-libero-x/collect.py --level LEVEL1 --with-features"
echo "  or in one go: WITH_FEATURES=1 scripts/groot-libero-x/collect.sh --level LEVEL1"
