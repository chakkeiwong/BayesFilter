#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/chakwong/anaconda3/envs/tf-gpu/bin/python
OUTER_TERM_DEADLINE_SECONDS=28740
export OUTER_TERM_DEADLINE_SECONDS

read -r LAUNCH_STARTED_EPOCH LAUNCH_STARTED_MONOTONIC < <(
  "$PYTHON" -c 'import time; print(time.time(), time.monotonic())'
)
export LAUNCH_STARTED_EPOCH LAUNCH_STARTED_MONOTONIC

remaining_seconds="$($PYTHON -c '
import os
import time

remaining = (
    float(os.environ["LAUNCH_STARTED_MONOTONIC"])
    + float(os.environ["OUTER_TERM_DEADLINE_SECONDS"])
    - time.monotonic()
)
if remaining <= 0.0:
    raise SystemExit(124)
print(f"{remaining:.6f}")
')"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

exec /usr/bin/timeout --signal=TERM --kill-after=60s "${remaining_seconds}s" \
  /usr/bin/unshare \
    --user \
    --map-root-user \
    --mount \
    --pid \
    --fork \
    --mount-proc \
    --kill-child=TERM \
    /usr/bin/bash \
    "$script_dir/complete_highdim_leaderboard_outer_launch_boundary.sh" \
    "$@"
