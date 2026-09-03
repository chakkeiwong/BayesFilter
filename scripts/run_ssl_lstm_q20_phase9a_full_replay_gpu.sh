#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-/home/ubuntu/anaconda3/envs/tfgpu/bin/python}"
output_root="${repo_root}/docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-09-02/phase9a-full-replay"
attempt_id="${BAYESFILTER_PHASE9A_ATTEMPT_ID:-attempt-$(date -u +%Y%m%dT%H%M%SZ)}"

if [[ -n "${BAYESFILTER_PHASE9A_OUTPUT_ROOT:-}" && "${BAYESFILTER_PHASE9A_OUTPUT_ROOT}" != "${output_root}" ]]; then
  printf 'output root is source-owned and cannot be overridden: %s\n' "${output_root}" >&2
  exit 2
fi
if [[ "${BAYESFILTER_GPU_ID:-0}" != "0" ]]; then
  printf '%s\n' 'Phase 9A full replay is source-bound to GPU0' >&2
  exit 2
fi

profile=""
scope_start=""
scope_limit=""
while (($#)); do
  case "$1" in
    --profile)
      (($# >= 2)) || { printf '%s\n' 'missing value for --profile' >&2; exit 2; }
      profile="$2"
      shift 2
      ;;
    --scope-start)
      (($# >= 2)) || { printf '%s\n' 'missing value for --scope-start' >&2; exit 2; }
      scope_start="$2"
      shift 2
      ;;
    --scope-limit)
      (($# >= 2)) || { printf '%s\n' 'missing value for --scope-limit' >&2; exit 2; }
      scope_limit="$2"
      shift 2
      ;;
    *)
      printf 'unsupported argument: %s\n' "$1" >&2
      exit 2
      ;;
  esac
done

case "${profile}:${scope_start}:${scope_limit}" in
  phase9a_full_replay_canary_v1:3:1)
    material_cap_seconds=1800
    ;;
  phase9a_full_replay_v1:0:6)
    material_cap_seconds=7800
    ;;
  *)
    printf '%s\n' \
      'profile/scope combination must be either' \
      'phase9a_full_replay_canary_v1 with scope 3/1 or' \
      'phase9a_full_replay_v1 with scope 0/6' >&2
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

export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"
export TF32="${TF32:-1}"

set +e
timeout --signal=TERM --kill-after=120s "${material_cap_seconds}s" "${python_bin}" \
  "${repo_root}/docs/benchmarks/run_ssl_lstm_q20_phase9a_fresh_tuning_preflight_2026_08_31.py" \
  --profile "${profile}" \
  --scope-start "${scope_start}" \
  --scope-limit "${scope_limit}" \
  --output-dir "${output_dir}"
run_status=$?
set -e

# TensorFlow may be inside a compiled call when the outer timeout fires.  Keep
# a durable failure receipt in that case without replacing runner output.
if (( run_status != 0 )) && [[ -d "${output_dir}" ]] && [[ ! -e "${output_dir}/failure.json" ]]; then
  printf '{\n  "schema": "bayesfilter.ssl_lstm_q20.phase9a_failure.v2",\n  "status": "FAIL_PHASE9A_SCOPE_PREFLIGHT",\n  "error_type": "TimeoutExpired",\n  "error": "outer launcher timeout or terminated worker (exit %s)",\n  "failure_classification": "resource_or_execution",\n  "profile_id": "%s",\n  "scope_start": %s,\n  "scope_limit": %s,\n  "material_cap_seconds": %s,\n  "command": "run_ssl_lstm_q20_phase9a_full_replay_gpu.sh"\n}\n' \
    "${run_status}" "${profile}" "${scope_start}" "${scope_limit}" "${material_cap_seconds}" \
    > "${output_dir}/failure.json"
fi
if (( run_status != 0 )) && [[ -e "${output_dir}/failure.json" ]] && [[ ! -e "${output_dir}/run_manifest.json" ]]; then
  cp "${output_dir}/failure.json" "${output_dir}/run_manifest.json"
fi
exit "${run_status}"
