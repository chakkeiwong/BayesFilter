#!/usr/bin/env bash
set -euo pipefail

selection=$(
  /home/chakwong/anaconda3/envs/tftwogpu/bin/python \
    scripts/select_preferred_gpu.py \
    --preferred-gpu 1 \
    --fallback-gpu 0 \
    --maximum-utilization-percent 50 \
    --minimum-free-mib 8192
)
IFS=$'\t' read -r selected_index selected_uuid selected_name \
  selected_utilization selected_free_mib selection_reason <<< "$selection"
if [[ -z "$selected_index" || -z "$selected_uuid" ]]; then
  echo "GPU selector returned an incomplete selection: $selection" >&2
  exit 1
fi

export BAYESFILTER_SELECTED_NVIDIA_SMI_INDEX=$selected_index
export BAYESFILTER_SELECTED_GPU_UUID=$selected_uuid
export BAYESFILTER_SELECTED_GPU_NAME=$selected_name
export BAYESFILTER_SELECTED_GPU_UTILIZATION=$selected_utilization
export BAYESFILTER_SELECTED_GPU_FREE_MIB=$selected_free_mib
export BAYESFILTER_GPU_SELECTION_REASON=$selection_reason
export TF_FORCE_GPU_ALLOW_GROWTH=true
export CUDA_VISIBLE_DEVICES=$BAYESFILTER_SELECTED_GPU_UUID
export XLA_FLAGS=--xla_gpu_enable_triton_gemm=false
export MPLCONFIGDIR=/tmp/bayesfilter-matplotlib

printf 'selected nvidia-smi GPU %s (%s, %s), utilization=%s%%, free=%s MiB\n' \
  "$BAYESFILTER_SELECTED_NVIDIA_SMI_INDEX" \
  "$BAYESFILTER_SELECTED_GPU_UUID" \
  "$BAYESFILTER_SELECTED_GPU_NAME" \
  "$BAYESFILTER_SELECTED_GPU_UTILIZATION" \
  "$BAYESFILTER_SELECTED_GPU_FREE_MIB" >&2

exec /home/chakwong/anaconda3/envs/tftwogpu/bin/python docs/benchmarks/run_classifier_score_path_count_bundle_20260815.py "$@"
