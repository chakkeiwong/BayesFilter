#!/usr/bin/env bash
# Narrow fixed-repository campaign entry point; no arbitrary command execution.
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo 'Usage: run_neutra_rare_region_campaign.sh {run|resume|status|check}' >&2
  exit 2
fi
case "$1" in run|resume|status|check) ;; *) exit 2 ;; esac
cd /home/ubuntu/python/BayesFilter
export TF_FORCE_GPU_ALLOW_GROWTH=true
export CUDA_VISIBLE_DEVICES=-1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
exec /home/ubuntu/anaconda3/envs/tfgpu/bin/python /home/ubuntu/python/BayesFilter/scripts/run_neutra_rare_region_master.py "$1"
