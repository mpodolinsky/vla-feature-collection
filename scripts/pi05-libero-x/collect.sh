#!/usr/bin/env bash
# One command: start the pi0.5 server (LeRobot venv) if the port is free, run
# scripts/groot-libero-x/collect.py --policy pi05, then stop only the server this script started.
#
#   scripts/pi05-libero-x/collect.sh --level LEVEL1 --n-rollouts 10
#   LAYERS=16,17,final POOL=2 scripts/pi05-libero-x/collect.sh --level LEVEL1
#
# Env: LAYERS  backbone block outputs to save, 0-based (default 16 = last block the action expert
#              reads; 17 = last block, never read; final = after the final norm); first = primary
#      POOLED_LAYERS  'all' (blocks 0-17 + final) or a comma list: per-modality mean-pooled 6144-d vectors (default none)
#      EXPERT_LAYERS  action-expert blocks to save, 0-based 0..17 (default none), e.g. 16
#      EXPERT_STEPS   'all' (10 denoise steps, default) or indices, 0 = first (t=1.0) ... 9 = last (t=0.1)
#      POOL    average-pool each camera's 16x16 token grid by this factor (default 1 = 256 tokens)
#      ADAPTER LoRA adapter (default podolinsky/pi05-libero-x-level123-lora); NO_ADAPTER=1 = base model
#      PI05_PORT (default 8000)   WAIT_SECS (default 900)   COLLECT_PYTHON (default top-level .venv)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
PORT="${PI05_PORT:-8000}"; WAIT_SECS="${WAIT_SECS:-900}"
PY="${COLLECT_PYTHON:-$REPO_ROOT/.venv/bin/python}"
SERVER_PY="$REPO_ROOT/scripts/pi05-libero-x/.venv/bin/python"
[[ -x "$PY" ]] || { echo "client venv missing: $PY (run: uv sync)" >&2; exit 1; }
[[ -x "$SERVER_PY" ]] || { echo "pi0.5 venv missing: run (cd scripts/pi05-libero-x && uv sync)" >&2; exit 1; }

SERVE_ARGS=(--port "$PORT" --with-features --layers "${LAYERS:-16}" --pool-image "${POOL:-1}")
[[ -n "${POOLED_LAYERS:-}" ]] && SERVE_ARGS+=(--pooled-layers "$POOLED_LAYERS")
[[ -n "${EXPERT_LAYERS:-}" ]] && SERVE_ARGS+=(--expert-layers "$EXPERT_LAYERS" --expert-steps "${EXPERT_STEPS:-all}")
[[ "${NO_ADAPTER:-0}" == "1" ]] && SERVE_ARGS+=(--no-adapter) || SERVE_ARGS+=(--adapter "${ADAPTER:-podolinsky/pi05-libero-x-level123-lora}")

port_open() { "$PY" - "$PORT" <<'PY'
import socket, sys
s = socket.socket(); s.settimeout(0.5)
try:
    sys.exit(0 if s.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)
finally:
    s.close()
PY
}

mkdir -p "$REPO_ROOT/outputs/pi05-libero-x/logs"
SERVER_LOG="$REPO_ROOT/outputs/pi05-libero-x/logs/pi05_server.log"
STARTED=0; SERVER_PID=""
cleanup() {
  if [[ "$STARTED" == "1" && -n "$SERVER_PID" ]]; then
    echo "Stopping pi0.5 server (pid $SERVER_PID)"; kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if port_open; then
  echo "Policy server already on :$PORT -- reusing it (will not stop it; its --layers/--pool-image apply)"
else
  echo "Starting pi0.5 server on :$PORT (layers: ${LAYERS:-16}, expert: ${EXPERT_LAYERS:-none}/${EXPERT_STEPS:-all}, pool: ${POOL:-1}, log: $SERVER_LOG)"
  "$SERVER_PY" "$REPO_ROOT/scripts/pi05-libero-x/serve_pi05.py" "${SERVE_ARGS[@]}" >"$SERVER_LOG" 2>&1 &
  SERVER_PID=$!; STARTED=1
  deadline=$((SECONDS + WAIT_SECS))
  while (( SECONDS < deadline )); do
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "server exited early:" >&2; tail -n 40 "$SERVER_LOG" >&2; exit 1; }
    port_open && { echo "pi0.5 server ready"; break; }
    sleep 3
  done
  port_open || { echo "timed out waiting for :$PORT" >&2; tail -n 40 "$SERVER_LOG" >&2; exit 1; }
fi

set +e
env MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}" \
  "$PY" "$REPO_ROOT/scripts/groot-libero-x/collect.py" --policy pi05 --port "$PORT" --with-features "$@"
STATUS=$?
set -e
exit "$STATUS"
