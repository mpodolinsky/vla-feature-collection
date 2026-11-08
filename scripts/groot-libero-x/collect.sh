#!/usr/bin/env bash
# One command: start the GR00T server (custom checkpoint) if port 8000 is free,
# run collect.py, then stop only the server this script started.
#
#   scripts/groot-libero-x/collect.sh --level LEVEL1 --n-rollouts 10
#   WITH_FEATURES=1 scripts/groot-libero-x/collect.sh --level LEVEL1
#   CKPT=checkpoints/other-ckpt scripts/groot-libero-x/collect.sh --level LEVEL2 --convention nvidia
#
# Env (optional):
#   GROOT_PORT (default 8000)   GROOT_SERVER_WAIT_SECS (default 900, model load is slow)
#   CKPT (default checkpoints/gr00t-libero-x-level123)
#   WITH_FEATURES=1  serve + collect the layer-16 backbone features
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

PORT="${GROOT_PORT:-8000}"
WAIT_SECS="${GROOT_SERVER_WAIT_SECS:-900}"
PY="${COLLECT_PYTHON:-$REPO_ROOT/.venv/bin/python}"
CKPT="${CKPT:-$REPO_ROOT/checkpoints/gr00t-libero-x-level123}"
SERVE_SCRIPT="$REPO_ROOT/scripts/groot-libero-x/serve.sh"
COLLECT_SCRIPT="$REPO_ROOT/scripts/groot-libero-x/collect.py"

[[ -x "$PY" ]] || { echo "COLLECT_PYTHON not executable: $PY" >&2; exit 1; }
[[ -f "$CKPT/config.json" ]] || { echo "checkpoint missing: $CKPT/config.json" >&2; exit 1; }

WITH_FEATURES_ARGS=()
[[ "${WITH_FEATURES:-0}" == "1" ]] && WITH_FEATURES_ARGS=(--with-features)

port_open() { "$PY" - "$PORT" <<'PY'
import socket, sys
s = socket.socket(); s.settimeout(0.5)
try:
    sys.exit(0 if s.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)
finally:
    s.close()
PY
}

mkdir -p "$REPO_ROOT/outputs/groot-libero-x/logs"
SERVER_LOG="$REPO_ROOT/outputs/groot-libero-x/logs/groot_server.log"
STARTED_SERVER=0
SERVER_PID=""
cleanup() {
  if [[ "$STARTED_SERVER" == "1" && -n "$SERVER_PID" ]]; then
    echo "Stopping GR00T server (pid $SERVER_PID)"
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if port_open; then
  echo "Policy server already on :$PORT -- reusing it (will not stop it)"
else
  echo "Starting GR00T server on :$PORT  (ckpt: $CKPT, features: ${WITH_FEATURES:-0}, log: $SERVER_LOG)"
  "$SERVE_SCRIPT" "$CKPT" "${WITH_FEATURES_ARGS[@]}" >"$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  STARTED_SERVER=1
  deadline=$((SECONDS + WAIT_SECS))
  while (( SECONDS < deadline )); do
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "server exited early:" >&2; tail -n 40 "$SERVER_LOG" >&2; exit 1; }
    port_open && { echo "GR00T server ready"; break; }
    sleep 3
  done
  port_open || { echo "timed out waiting for :$PORT" >&2; tail -n 40 "$SERVER_LOG" >&2; exit 1; }
fi

# Not exec: keep the cleanup trap alive so a server we started gets stopped.
set +e
env MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}" \
  "$PY" "$COLLECT_SCRIPT" "${WITH_FEATURES_ARGS[@]}" "$@"
STATUS=$?
set -e
exit "$STATUS"
