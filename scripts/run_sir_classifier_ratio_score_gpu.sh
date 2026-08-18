#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/select_preferred_gpu_env.sh"
bayesfilter_select_preferred_gpu --preferred-gpu 1 --fallback-gpu 0 --maximum-utilization-percent 50 --minimum-free-mib 8192
export XLA_FLAGS=--xla_gpu_enable_triton_gemm=false
export MPLCONFIGDIR=/tmp/bayesfilter-matplotlib

PYTHON=/home/chakwong/anaconda3/envs/tftwogpu/bin/python
RUNNER=docs/benchmarks/run_sir_classifier_ratio_score_20260813.py

mode="${1:-}"
case "${mode}" in
  probe)
    exec "${PYTHON}" -c "import tensorflow as tf; from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth; print(configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)); print(tf.config.list_logical_devices('GPU'))"
    ;;
  exact-smoke)
    test "$#" -eq 2
    exec "${PYTHON}" "${RUNNER}" --stage exact_oracle --profile smoke --output-root "$2"
    ;;
  exact-full)
    test "$#" -eq 2
    exec "${PYTHON}" "${RUNNER}" --stage exact_oracle --profile full --output-root "$2"
    ;;
  sir-full)
    test "$#" -eq 3
    exec "${PYTHON}" "${RUNNER}" --stage sir --profile full --output-root "$2" --oracle-result "$3"
    ;;
  *)
    echo "usage: $0 {probe|exact-smoke|exact-full|sir-full} [output-root] [oracle-result]" >&2
    exit 2
    ;;
esac
