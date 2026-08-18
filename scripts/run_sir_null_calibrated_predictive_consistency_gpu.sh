#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/select_preferred_gpu_env.sh"
bayesfilter_select_preferred_gpu --preferred-gpu 1 --fallback-gpu 0 --maximum-utilization-percent 50 --minimum-free-mib 8192
export XLA_FLAGS=--xla_gpu_enable_triton_gemm=false MPLCONFIGDIR=/tmp/bayesfilter-matplotlib
PYTHON=/home/chakwong/anaconda3/envs/tftwogpu/bin/python
RUNNER=docs/benchmarks/run_sir_null_calibrated_predictive_consistency_20260814.py
case "${1:-}" in
  gaussian-smoke|sir-smoke) exec "$PYTHON" "$RUNNER" --kind "${1%-smoke}" --profile smoke --output-root "$2";;
  gaussian-full|sir-full) exec "$PYTHON" "$RUNNER" --kind "${1%-full}" --profile full --output-root "$2";;
  *) echo "usage: $0 {gaussian-smoke|gaussian-full|sir-smoke|sir-full} output-root" >&2; exit 2;;
esac
