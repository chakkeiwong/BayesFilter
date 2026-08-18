#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/select_preferred_gpu_env.sh"
bayesfilter_select_preferred_gpu \
  --preferred-gpu 1 \
  --fallback-gpu 0 \
  --maximum-utilization-percent 50 \
  --minimum-free-mib 8192
export XLA_FLAGS=--xla_gpu_enable_triton_gemm=false
export MPLCONFIGDIR=/tmp/bayesfilter-matplotlib

printf 'selected nvidia-smi GPU %s (%s, %s, PCI %s), utilization=%s%%, memory=%s MiB\n' \
  "$BAYESFILTER_SELECTED_NVIDIA_SMI_INDEX" \
  "$BAYESFILTER_SELECTED_GPU_UUID" \
  "$BAYESFILTER_SELECTED_GPU_NAME" \
  "$BAYESFILTER_SELECTED_GPU_PCI_BUS_ID" \
  "$BAYESFILTER_SELECTED_GPU_UTILIZATION" \
  "$BAYESFILTER_SELECTED_GPU_MEMORY_USED_MIB/$BAYESFILTER_SELECTED_GPU_MEMORY_TOTAL_MIB" >&2

exec /home/chakwong/anaconda3/envs/tftwogpu/bin/python docs/benchmarks/run_classifier_score_path_count_bundle_20260815.py "$@"
