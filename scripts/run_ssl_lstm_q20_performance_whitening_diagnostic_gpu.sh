#!/usr/bin/env bash
set -euo pipefail

# This launcher owns only the bounded diagnostic.  It establishes allocator
# policy before Python imports TensorFlow and never resumes a prior output.
GPU_ID="${BAYESFILTER_GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_ID}}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"

if [[ "${TF_FORCE_GPU_ALLOW_GROWTH}" != "true" ]]; then
  printf '%s\n' 'FAIL_LAUNCH: TF_FORCE_GPU_ALLOW_GROWTH must be true' >&2
  exit 2
fi

exec /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/docs/benchmarks/run_ssl_lstm_q20_performance_whitening_diagnostic_2026_09_02.py" \
  --gpu-id "${GPU_ID}" \
  --max-seconds "${BAYESFILTER_PERF_WHITENING_MAX_SECONDS:-1100}" \
  "$@"
