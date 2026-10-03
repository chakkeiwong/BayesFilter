#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-/home/ubuntu/anaconda3/envs/tfgpu/bin/python}"
output_root="${BAYESFILTER_PHASE9A_OUTPUT_ROOT:-${repo_root}/docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-09-01/phase9a-chart1-beta0-repair}"
attempt_id="${BAYESFILTER_PHASE9A_ATTEMPT_ID:-attempt-$(date -u +%Y%m%dT%H%M%SZ)}"
output_dir="${output_root}/${attempt_id}"

if [[ ! -x "${python_bin}" ]]; then
  printf 'missing executable Python interpreter: %s\n' "${python_bin}" >&2
  exit 2
fi
if [[ -e "${output_dir}" ]]; then
  printf 'refusing to overwrite existing output directory: %s\n' "${output_dir}" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${BAYESFILTER_GPU_ID:-0}"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"
export TF32="${TF32:-1}"

set +e
timeout --signal=TERM --kill-after=120s 1800s "${python_bin}" "${repo_root}/docs/benchmarks/run_ssl_lstm_q20_phase9a_fresh_tuning_preflight_2026_08_31.py" \
  --profile chart1_beta0_repair_v4_fresh \
  --scope-start 3 \
  --scope-limit 1 \
  --output-dir "${output_dir}"
run_status=$?
set -e

# If the outer timeout had to terminate a TensorFlow call before Python could
# service SIGTERM, preserve a machine-readable resource failure in the same
# fresh directory.  Never replace a runner-produced failure or success record.
if (( run_status != 0 )) && [[ -d "${output_dir}" ]] && [[ ! -e "${output_dir}/failure.json" ]]; then
  printf '{\n  "schema": "bayesfilter.ssl_lstm_q20.phase9a_failure.v2",\n  "status": "FAIL_PHASE9A_SCOPE_PREFLIGHT",\n  "error_type": "TimeoutExpired",\n  "error": "outer launcher timeout or terminated worker (exit %s)",\n  "failure_classification": "resource_or_execution",\n  "profile_id": "chart1_beta0_repair_v4_fresh",\n  "scope_start": 3,\n  "scope_limit": 1,\n  "command": "run_ssl_lstm_q20_phase9a_chart1_beta0_repair_gpu.sh"\n}\n' "${run_status}" > "${output_dir}/failure.json"
fi
if (( run_status != 0 )) && [[ -e "${output_dir}/failure.json" ]] && [[ ! -e "${output_dir}/run_manifest.json" ]]; then
  cp "${output_dir}/failure.json" "${output_dir}/run_manifest.json"
fi
exit "${run_status}"
