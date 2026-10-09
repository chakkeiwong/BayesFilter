#!/usr/bin/env bash
# Exact command surface for the scientific simple-mixture campaign.
set -euo pipefail
if [[ $# -ne 1 ]]; then
    echo 'Usage: run_neutra_scientific_campaign.sh check|preflight|price|run|resume|status|attribution|attribution-supervise|forward-reverse-plan|forward-reverse|forward-reverse-smoke' >&2
    exit 2
fi
case "$1" in check|preflight|price|run|resume|status|attribution|attribution-supervise|forward-reverse-plan|forward-reverse|forward-reverse-smoke) ;; *) exit 2 ;; esac
cd /home/ubuntu/python/BayesFilter
export PYTHONPATH=/home/ubuntu/python/BayesFilter${PYTHONPATH:+:${PYTHONPATH}}
export TF_FORCE_GPU_ALLOW_GROWTH=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_NUM_INTRAOP_THREADS=2
export TF_NUM_INTEROP_THREADS=1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
if [[ "$1" == attribution-supervise ]]; then
    exec /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
        /home/ubuntu/python/BayesFilter/scripts/supervise_neutra_attribution.py
fi
exec /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
    /home/ubuntu/python/BayesFilter/scripts/run_neutra_scientific_campaign_master.py "$1"
