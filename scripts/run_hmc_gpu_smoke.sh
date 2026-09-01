#!/usr/bin/env bash
set -euo pipefail

mode="${1:-}"

if [[ "${mode}" != "mixed_precision_dpf_tf32_gpu0" ]]; then
  echo "usage: bash scripts/run_hmc_gpu_smoke.sh mixed_precision_dpf_tf32_gpu0" >&2
  exit 2
fi

python_bin="${TFGPU_PYTHON:-${CONDA_PREFIX:-/home/ubuntu/miniforge3/envs/tf-gpu}/bin/python}"
log_path="docs/benchmarks/logs/mixed-precision-hmc-smoke-fp32-tf32-gpu0-b1-t3-np8-d2-m2-active-odd-2026-06-17.log"
json_path="docs/benchmarks/experimental-batched-ledh-pfpf-ot-mixed-precision-hmc-smoke-fp32-tf32-gpu0-b1-t3-np8-d2-m2-active-odd-2026-06-17.json"
md_path="docs/benchmarks/experimental-batched-ledh-pfpf-ot-mixed-precision-hmc-smoke-fp32-tf32-gpu0-b1-t3-np8-d2-m2-active-odd-2026-06-17.md"

mkdir -p docs/benchmarks/logs

"${python_bin}" docs/benchmarks/run_experimental_batched_ledh_pfpf_ot_hmc_mechanics_smoke.py \
  --device-scope visible \
  --cuda-visible-devices 0 \
  --device /GPU:0 \
  --expect-device-kind gpu \
  --dtype float32 \
  --tf32-mode enabled \
  --batch-size 1 \
  --time-steps 3 \
  --num-particles 8 \
  --state-dim 2 \
  --obs-dim 2 \
  --transport-policy active-odd \
  --output "${json_path}" \
  --markdown-output "${md_path}" \
  > "${log_path}" 2>&1

echo "${json_path}"
