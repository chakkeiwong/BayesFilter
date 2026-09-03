#!/usr/bin/env bash
set -euo pipefail

# Repository-default GPU launcher.  It does not call an idle-GPU or approval
# probe; the Python runner performs the scientific and memory-growth checks.
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-/home/ubuntu/anaconda3/envs/tfgpu/bin/python}"
gpu_id="${BAYESFILTER_GPU_ID:-0}"
timeout_seconds="${BAYESFILTER_PHASE8_TIMEOUT_SECONDS:-1800}"
mode="${BAYESFILTER_PHASE8_MODE:-cost-pilot}"
principal_sqrt_backend="${BAYESFILTER_PHASE8_PRINCIPAL_SQRT_BACKEND:-compiled_custom_op}"
validation_size="${BAYESFILTER_PHASE8_VALIDATION_SIZE:-256}"
output_root="${BAYESFILTER_PHASE8_OUTPUT_ROOT:-${repo_root}/docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-08-29/phase8-calibration}"
attempt_label="${BAYESFILTER_PHASE8_ATTEMPT_LABEL:-attempt-$(date -u +%Y%m%dT%H%M%SZ)-gpu${gpu_id}}"
output_dir="${output_root}/${attempt_label}"

if [[ ! -x "${python_bin}" ]]; then
  printf 'ERROR: PYTHON_BIN is not executable: %s\n' "${python_bin}" >&2
  exit 2
fi
if [[ ! "${gpu_id}" =~ ^[0-9]+$ ]]; then
  printf 'ERROR: BAYESFILTER_GPU_ID must be a numeric physical GPU id: %s\n' "${gpu_id}" >&2
  exit 2
fi
if [[ ! "${timeout_seconds}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'ERROR: BAYESFILTER_PHASE8_TIMEOUT_SECONDS must be positive integer\n' >&2
  exit 2
fi
case "${mode}" in
  cost-pilot|target-localization) ;;
  *)
    printf 'ERROR: BAYESFILTER_PHASE8_MODE must be cost-pilot or target-localization: %s\n' "${mode}" >&2
    exit 2
    ;;
esac
case "${principal_sqrt_backend}" in
  compiled_custom_op|tensorflow_eigh_strict) ;;
  *)
    printf 'ERROR: BAYESFILTER_PHASE8_PRINCIPAL_SQRT_BACKEND is unsupported: %s\n' "${principal_sqrt_backend}" >&2
    exit 2
    ;;
esac
case "${validation_size}" in
  8|32|256) ;;
  *)
    printf 'ERROR: BAYESFILTER_PHASE8_VALIDATION_SIZE must be 8, 32, or 256: %s\n' "${validation_size}" >&2
    exit 2
    ;;
esac
if [[ ! "${attempt_label}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  printf 'ERROR: BAYESFILTER_PHASE8_ATTEMPT_LABEL must contain only [A-Za-z0-9._-]\n' >&2
  exit 2
fi
if [[ -e "${output_dir}" ]]; then
  printf 'ERROR: refusing to overwrite existing output directory: %s\n' "${output_dir}" >&2
  exit 2
fi
mkdir -p "${output_root}"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"
export BAYESFILTER_GPU_LAUNCH_MODE="repository_default_gpu_launcher"
export BAYESFILTER_GPU_TRUST_BASIS="${BAYESFILTER_GPU_TRUST_BASIS:-repository_default_gpu_route_external_boundary_unclassified}"

printf 'Launching Phase 8 mode %s on GPU %s\n' "${mode}" "${gpu_id}"
printf 'Mode: %s; principal-square-root backend: %s; validation size: %s\n' "${mode}" "${principal_sqrt_backend}" "${validation_size}"
printf 'Output: %s\n' "${output_dir}"
printf 'Memory policy: TF_FORCE_GPU_ALLOW_GROWTH=%s\n' "${TF_FORCE_GPU_ALLOW_GROWTH}"

exec timeout --signal=TERM --kill-after=60s "${timeout_seconds}s" \
  "${python_bin}" \
  "${repo_root}/docs/benchmarks/run_ssl_lstm_q20_tempered_rkl_transport_ensemble_phase8_2026_08_29.py" \
  --mode "${mode}" \
  --principal-sqrt-backend "${principal_sqrt_backend}" \
  --validation-size "${validation_size}" \
  --output-dir "${output_dir}"
