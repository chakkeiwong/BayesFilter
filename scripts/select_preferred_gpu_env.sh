#!/usr/bin/env bash

# Source this helper and call bayesfilter_select_preferred_gpu before importing
# TensorFlow. The selected physical identity is exported as an NVIDIA UUID.
bayesfilter_select_preferred_gpu() {
  local helper_dir selector_python selection
  helper_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  selector_python="${BAYESFILTER_GPU_SELECTOR_PYTHON:-/home/chakwong/anaconda3/envs/tftwogpu/bin/python}"
  selection="$("${selector_python}" "${helper_dir}/select_preferred_gpu.py" "$@")"

  IFS=$'\t' read -r \
    BAYESFILTER_SELECTED_NVIDIA_SMI_INDEX \
    BAYESFILTER_SELECTED_GPU_UUID \
    BAYESFILTER_SELECTED_GPU_NAME \
    BAYESFILTER_SELECTED_GPU_PCI_BUS_ID \
    BAYESFILTER_SELECTED_GPU_UTILIZATION \
    BAYESFILTER_SELECTED_GPU_MEMORY_USED_MIB \
    BAYESFILTER_SELECTED_GPU_FREE_MIB \
    BAYESFILTER_SELECTED_GPU_MEMORY_TOTAL_MIB \
    BAYESFILTER_GPU_SELECTION_REASON <<< "${selection}"

  if [[ "${BAYESFILTER_SELECTED_GPU_UUID:-}" != GPU-* ]]; then
    echo "GPU selector returned an invalid UUID: ${selection}" >&2
    return 1
  fi

  export BAYESFILTER_SELECTED_NVIDIA_SMI_INDEX BAYESFILTER_SELECTED_GPU_UUID
  export BAYESFILTER_SELECTED_GPU_NAME BAYESFILTER_SELECTED_GPU_PCI_BUS_ID
  export BAYESFILTER_SELECTED_GPU_UTILIZATION BAYESFILTER_SELECTED_GPU_MEMORY_USED_MIB
  export BAYESFILTER_SELECTED_GPU_FREE_MIB BAYESFILTER_SELECTED_GPU_MEMORY_TOTAL_MIB
  export BAYESFILTER_GPU_SELECTION_REASON
  export CUDA_VISIBLE_DEVICES="${BAYESFILTER_SELECTED_GPU_UUID}"
  export TF_FORCE_GPU_ALLOW_GROWTH=true

  printf 'selected NVIDIA GPU %s (%s, %s, PCI %s), utilization=%s%%, memory=%s/%s MiB, reason=%s\n' \
    "${BAYESFILTER_SELECTED_NVIDIA_SMI_INDEX}" \
    "${BAYESFILTER_SELECTED_GPU_UUID}" \
    "${BAYESFILTER_SELECTED_GPU_NAME}" \
    "${BAYESFILTER_SELECTED_GPU_PCI_BUS_ID}" \
    "${BAYESFILTER_SELECTED_GPU_UTILIZATION}" \
    "${BAYESFILTER_SELECTED_GPU_MEMORY_USED_MIB}" \
    "${BAYESFILTER_SELECTED_GPU_MEMORY_TOTAL_MIB}" \
    "${BAYESFILTER_GPU_SELECTION_REASON}" >&2
}
