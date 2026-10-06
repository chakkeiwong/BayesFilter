#!/usr/bin/env bash
# Fixed local research command surface. No arbitrary program or path arguments.
set -euo pipefail
if [[ $# -ne 1 ]]; then
    echo 'Usage: run_neutra_six_method_campaign.sh check|preflight|run|resume|status' >&2
    exit 2
fi
case "$1" in check|preflight|run|resume|status) ;; *) exit 2 ;; esac
cd /home/ubuntu/python/BayesFilter
export TF_FORCE_GPU_ALLOW_GROWTH=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=-1
export TF_NUM_INTRAOP_THREADS=2
export TF_NUM_INTEROP_THREADS=1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
exec /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
    /home/ubuntu/python/BayesFilter/scripts/run_neutra_six_method_master.py "$1"
