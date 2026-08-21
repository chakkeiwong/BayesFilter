#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/select_preferred_gpu_env.sh"
bayesfilter_select_preferred_gpu --preferred-gpu 1 --fallback-gpu 0 --maximum-utilization-percent 50 --minimum-free-mib 8192
export XLA_FLAGS=--xla_gpu_enable_triton_gemm=false MPLCONFIGDIR=/tmp/bayesfilter-matplotlib
exec /home/chakwong/anaconda3/envs/tftwogpu/bin/python docs/benchmarks/run_classifier_score_variance_bundle_20260815.py "$@"
