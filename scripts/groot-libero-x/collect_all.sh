#!/usr/bin/env bash
# Collect LEVEL1-3 with rohansiva/gr00t-libero-x-level123.
#
#   WITH_FEATURES=1 scripts/groot-libero-x/collect_all.sh
#   LEVELS="LEVEL1 LEVEL2" N_ROLLOUTS=5 scripts/groot-libero-x/collect_all.sh --no-video
#
# Loops the levels, each with its committed tasks/<LEVEL>.txt (72 / 73 / 73
# tasks). The server is started once by the
# first collect.sh call and reused (port already open) for the rest.
#
# Env: LEVELS (default "LEVEL1 LEVEL2 LEVEL3"), N_ROLLOUTS (default 10),
#      WITH_FEATURES, CKPT, GROOT_PORT -- passed through to collect.sh.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LEVELS="${LEVELS:-LEVEL1 LEVEL2 LEVEL3}"
N_ROLLOUTS="${N_ROLLOUTS:-10}"

for L in $LEVELS; do
  echo "======== collecting $L ($N_ROLLOUTS rollouts/task) ========"
  "$REPO_ROOT/scripts/groot-libero-x/collect.sh" --level "$L" --n-rollouts "$N_ROLLOUTS" --keep-going "$@"
done

echo "======== summary ========"
"$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/scripts/groot-libero-x/summarize.py"
