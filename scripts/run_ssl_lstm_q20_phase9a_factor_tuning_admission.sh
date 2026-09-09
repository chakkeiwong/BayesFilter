#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-/home/ubuntu/anaconda3/envs/tfgpu/bin/python}"
output_root="${repo_root}/docs/plans/artifacts/ssl-lstm-q20-factor-route-fresh-tuning-2026-09-04"
attempt_suffix="${BAYESFILTER_FACTOR_TUNING_ATTEMPT_ID:-attempt-$(date -u +%Y%m%dT%H%M%SZ)}"

if (($# != 1)); then
  printf 'usage: %s canary|full|r2|source-sync\n' "$0" >&2
  exit 2
fi

case "$1" in
  canary)
    profile="phase9a_factor_tuning_canary_v1"
    scope_start=3
    scope_limit=1
    material_cap_seconds=1800
    attempt_id="canary-${attempt_suffix}"
    ;;
  full)
    profile="phase9a_factor_tuning_full_v1"
    scope_start=0
    scope_limit=6
    material_cap_seconds=10000
    attempt_id="full-${attempt_suffix}"
    ;;
  r2)
    profile="phase9a_factor_tuning_full_r2_v1"
    scope_start=0
    scope_limit=6
    material_cap_seconds=8400
    attempt_id="r2-${attempt_suffix}"
    ;;
  source-sync)
    profile="phase9a_factor_tuning_full_source_sync_v1"
    scope_start=0
    scope_limit=6
    material_cap_seconds=4000
    attempt_id="source-sync-${attempt_suffix}"
    ;;
  *)
    printf 'unsupported mode %s; expected canary, full, r2, or source-sync\n' "$1" >&2
    exit 2
    ;;
esac

if [[ ! -x "${python_bin}" ]]; then
  printf 'missing executable Python interpreter: %s\n' "${python_bin}" >&2
  exit 2
fi

mkdir -p "${output_root}"
output_dir="${output_root}/${attempt_id}"
if [[ -e "${output_dir}" ]]; then
  printf 'refusing to overwrite existing output directory: %s\n' "${output_dir}" >&2
  exit 2
fi

# These settings are applied before the runner imports TensorFlow.
export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"
export TF32=1

set +e
timeout --signal=TERM --kill-after=120s "${material_cap_seconds}s" "${python_bin}" \
  "${repo_root}/docs/benchmarks/run_ssl_lstm_q20_phase9a_fresh_tuning_preflight_2026_08_31.py" \
  --profile "${profile}" \
  --scope-start "${scope_start}" \
  --scope-limit "${scope_limit}" \
  --output-dir "${output_dir}"
run_status=$?
set -e

if (( run_status != 0 )) && [[ -d "${output_dir}" ]] && [[ ! -e "${output_dir}/failure.json" ]]; then
  printf '{\n  "schema": "bayesfilter.ssl_lstm_q20.phase9a_failure.v2",\n  "status": "FAIL_PHASE9A_SCOPE_PREFLIGHT",\n  "error_type": "TimeoutExpired",\n  "error": "outer launcher timeout or terminated worker (exit %s)",\n  "failure_classification": "resource_or_execution",\n  "profile_id": "%s",\n  "principal_sqrt_backend": "tensorflow_eigh_strict_factor_cached",\n  "scope_start": %s,\n  "scope_limit": %s,\n  "material_cap_seconds": %s,\n  "command": "run_ssl_lstm_q20_phase9a_factor_tuning_admission.sh %s"\n}\n' \
    "${run_status}" "${profile}" "${scope_start}" "${scope_limit}" \
    "${material_cap_seconds}" "$1" > "${output_dir}/failure.json"
fi
if (( run_status != 0 )) && [[ -e "${output_dir}/failure.json" ]] && [[ ! -e "${output_dir}/run_manifest.json" ]]; then
  cp "${output_dir}/failure.json" "${output_dir}/run_manifest.json"
fi
exit "${run_status}"
