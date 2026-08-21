#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/select_preferred_gpu_env.sh"
bayesfilter_select_preferred_gpu --preferred-gpu 1 --fallback-gpu 0 --maximum-utilization-percent 50 --minimum-free-mib 8192
export XLA_FLAGS=--xla_gpu_enable_triton_gemm=false MPLCONFIGDIR=/tmp/bayesfilter-matplotlib
PYTHON=/home/chakwong/anaconda3/envs/tftwogpu/bin/python
RUNNER=docs/benchmarks/run_sir_anchored_orthogonal_ratio_score_20260814.py
case "${1:-}" in
  exact-smoke) exec "$PYTHON" "$RUNNER" --stage exact_oracle --profile smoke --output-root "$2";;
  exact-full) exec "$PYTHON" "$RUNNER" --stage exact_oracle --profile full --output-root "$2";;
  sir-full) exec "$PYTHON" "$RUNNER" --stage sir --profile full --output-root "$2" --oracle-result "$3";;
  *) echo "usage: $0 {exact-smoke|exact-full|sir-full} output-root [oracle-result]" >&2; exit 2;;
esac
