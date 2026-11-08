#!/usr/bin/env bash
# Launch the GR00T-N1.7 websocket policy server (openpi protocol, port 8000)
# against a local checkpoint directory. Runs in submodules/Isaac-GR00T/.venv;
# model load takes ~1-3 min.
#
#   scripts/groot-libero-x/serve.sh checkpoints/gr00t-libero-x-level123 --with-features
#   HOST=0.0.0.0 PORT=8000 scripts/groot-libero-x/serve.sh <dir> --no-strict
#
# --with-features also returns the layer-16 backbone hidden states.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GROOT_DIR="${REPO_ROOT}/submodules/Isaac-GR00T"
CKPT="${1:?usage: serve.sh <checkpoint-dir> [extra serve_groot_ws.py args]}"
shift || true
# Resolve to an absolute path before cd'ing into GROOT_DIR below -- a
# relative path would otherwise be interpreted relative to the wrong
# directory (and transformers silently reinterprets it as a HF repo id).
CKPT="$(cd "${CKPT}" && pwd)"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

[[ -d "${GROOT_DIR}/.git" || -f "${GROOT_DIR}/.git" ]] || {
  echo "submodules/Isaac-GR00T not initialised -- run scripts/groot-libero-x/setup.sh first" >&2; exit 1; }
[[ -f "${CKPT}/config.json" ]] || {
  echo "checkpoint missing: ${CKPT}/config.json" >&2; exit 1; }

export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
if [[ -z "${HF_TOKEN:-}" && -f "${HOME}/.cache/huggingface/token" ]]; then
  export HF_TOKEN="$(cat "${HOME}/.cache/huggingface/token")"
fi

cd "${GROOT_DIR}"
exec uv run python "${REPO_ROOT}/scripts/groot-libero-x/server/serve_groot_ws.py" \
  --model-path "${CKPT}" --host "${HOST}" --port "${PORT}" "$@"
