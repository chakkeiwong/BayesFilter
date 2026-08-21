#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:?ROOT is required}"
exec "${TFGPU_PYTHON:-${CONDA_PREFIX:-/home/ubuntu/miniforge3/envs/tf-gpu}/bin/python}" \
  "$ROOT/scripts/complete_highdim_leaderboard_overnight_supervisor.py"
