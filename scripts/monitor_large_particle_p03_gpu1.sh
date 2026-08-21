#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${TFGPU_PYTHON:-${CONDA_PREFIX:-/home/ubuntu/miniforge3/envs/tf-gpu}/bin/python}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-300}"
SELECTED_GPU_INDEX="${SELECTED_GPU_INDEX:-1}"
GPU_MEMORY_LIMIT_MIB="${GPU_MEMORY_LIMIT_MIB:-2048}"
GPU_UTIL_LIMIT_PERCENT="${GPU_UTIL_LIMIT_PERCENT:-20}"
DATE_TAG="${DATE_TAG:-2026-06-21}"

ARTIFACT_DIR="docs/benchmarks/experimental-batched-ledh-pfpf-ot-large-particle-efficiency-p03-streaming-large-n-gpu-clean-rerun-children-${DATE_TAG}"
OUTPUT_JSON="docs/benchmarks/experimental-batched-ledh-pfpf-ot-large-particle-efficiency-p03-streaming-large-n-gpu-clean-rerun-${DATE_TAG}.json"
OUTPUT_MD="docs/benchmarks/experimental-batched-ledh-pfpf-ot-large-particle-efficiency-p03-streaming-large-n-gpu-clean-rerun-${DATE_TAG}.md"
LOG_PATH="docs/benchmarks/logs/large-particle-p03-gpu1-monitor-${DATE_TAG}.log"
SCRIPT_NAME="$(basename "$0")"

cd "$ROOT"
mkdir -p "$(dirname "$LOG_PATH")"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$LOG_PATH"
}

gpu_row() {
  nvidia-smi --id="${SELECTED_GPU_INDEX}" \
    --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader,nounits
}

gpu_uuid() {
  gpu_row | awk -F', ' '{print $2}'
}

compute_pids_for_uuid() {
  local uuid="$1"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv,noheader,nounits |
    awk -F', ' -v uuid="$uuid" '$1 == uuid {print $2}'
}

pid_args() {
  local pid="$1"
  ps -p "$pid" -o args= 2>/dev/null || true
}

is_our_p03_pid() {
  local pid="$1"
  local args
  args="$(pid_args "$pid")"
  [[ "$args" == *"benchmark_experimental_batched_ledh_pfpf_ot_large_particle_efficiency.py"* ]] && return 0
  [[ "$args" == *"benchmark_experimental_batched_ledh_pfpf_ot_streaming_lgssm.py"* && "$args" == *"$ARTIFACT_DIR"* ]] && return 0
  return 1
}

is_gpu_clean_before_launch() {
  local row uuid memory_used util pids
  row="$(gpu_row)"
  uuid="$(awk -F', ' '{print $2}' <<<"$row")"
  memory_used="$(awk -F', ' '{print $3}' <<<"$row")"
  util="$(awk -F', ' '{print $5}' <<<"$row")"
  mapfile -t pids < <(compute_pids_for_uuid "$uuid")
  log "GPU${SELECTED_GPU_INDEX} row: ${row}; compute_pids=${pids[*]:-none}"
  if (( memory_used >= GPU_MEMORY_LIMIT_MIB )); then
    log "GPU${SELECTED_GPU_INDEX} not clean: memory ${memory_used} MiB >= ${GPU_MEMORY_LIMIT_MIB} MiB"
    return 1
  fi
  if (( util >= GPU_UTIL_LIMIT_PERCENT )); then
    log "GPU${SELECTED_GPU_INDEX} not clean: utilization ${util}% >= ${GPU_UTIL_LIMIT_PERCENT}%"
    return 1
  fi
  if (( ${#pids[@]} > 0 )); then
    local pid
    for pid in "${pids[@]}"; do
      log "GPU${SELECTED_GPU_INDEX} not clean: pid=${pid} args=$(pid_args "$pid")"
    done
    return 1
  fi
  return 0
}

kill_our_p03_processes() {
  local pattern pids
  pattern="benchmark_experimental_batched_ledh_pfpf_ot_large_particle_efficiency.py|benchmark_experimental_batched_ledh_pfpf_ot_streaming_lgssm.py.*${ARTIFACT_DIR}"
  mapfile -t pids < <(ps -eo pid,args | awk -v pat="$pattern" '$0 ~ pat {print $1}')
  if (( ${#pids[@]} > 0 )); then
    log "Stopping this lane's P03 processes: ${pids[*]}"
    kill "${pids[@]}" 2>/dev/null || true
  fi
}

monitor_p03_for_contamination() {
  local parent_pid="$1"
  local uuid pids pid args contaminated
  uuid="$(gpu_uuid)"
  while kill -0 "$parent_pid" 2>/dev/null; do
    sleep 60
    contaminated=0
    mapfile -t pids < <(compute_pids_for_uuid "$uuid")
    for pid in "${pids[@]}"; do
      [[ -n "$pid" ]] || continue
      args="$(pid_args "$pid")"
      if ! is_our_p03_pid "$pid"; then
        log "Contamination detected on GPU${SELECTED_GPU_INDEX}: pid=${pid} args=${args}"
        contaminated=1
      fi
    done
    if (( contaminated )); then
      kill_our_p03_processes
      return 70
    fi
    log "P03 running; GPU${SELECTED_GPU_INDEX} compute_pids=${pids[*]:-none}"
  done
  return 0
}

run_p03() {
  local summary_json parent_pid monitor_status run_status
  summary_json="$("$PYTHON" - <<'PY'
import json
print(json.dumps({
    "gpu1": {
        "classification": "selected_clean_at_just_in_time_preflight",
        "memory_limit_mib": 2048,
        "utilization_limit_percent": 20,
    }
}, sort_keys=True))
PY
)"
  log "Launching clean-rerun P03 on physical GPU${SELECTED_GPU_INDEX}"
  "$PYTHON" docs/benchmarks/benchmark_experimental_batched_ledh_pfpf_ot_large_particle_efficiency.py \
    --run-kind streaming-ladder \
    --particle-counts 1000,5000,10000 \
    --optional-particle-counts 20000 \
    --batch-size 1 \
    --time-steps 80 \
    --state-dim 20 \
    --obs-dim 20 \
    --transport-policy active-all \
    --proposal-mode callback \
    --sinkhorn-iterations 4 \
    --row-chunk-size 1024 \
    --col-chunk-size 1024 \
    --particle-chunk-size 256 \
    --warmups 0 \
    --repeats 1 \
    --seed 20260621 \
    --device-scope visible \
    --cuda-visible-devices "${SELECTED_GPU_INDEX}" \
    --device /GPU:0 \
    --expect-device-kind gpu \
    --child-timeout-seconds 3600 \
    --phase-wall-time-budget-seconds 14400 \
    --optional-max-elapsed-before-seconds 7200 \
    --optional-max-reference-child-seconds 1200 \
    --selected-physical-gpu "${SELECTED_GPU_INDEX}" \
    --gpu-selection-reason "jit_clean_gpu${SELECTED_GPU_INDEX}_monitor_${SCRIPT_NAME}" \
    --nvidia-smi-summary-json "$summary_json" \
    --artifact-dir "$ARTIFACT_DIR" \
    --output "$OUTPUT_JSON" \
    --markdown-output "$OUTPUT_MD" >>"$LOG_PATH" 2>&1 &
  parent_pid="$!"
  monitor_p03_for_contamination "$parent_pid" &
  monitor_pid="$!"
  set +e
  wait "$parent_pid"
  run_status="$?"
  wait "$monitor_pid"
  monitor_status="$?"
  set -e
  monitor_status="${monitor_status:-0}"
  if (( run_status == 0 )); then
    log "P03 completed successfully; ignoring late monitor status ${monitor_status} after parent exit."
    return 0
  fi
  if (( monitor_status == 70 )); then
    log "P03 stopped because selected GPU became contaminated."
    return 70
  fi
  log "P03 exited with status ${run_status}; output=${OUTPUT_JSON}"
  return "$run_status"
}

log "Starting GPU${SELECTED_GPU_INDEX} monitor for P03 clean rerun; interval=${CHECK_INTERVAL_SECONDS}s"
while true; do
  if is_gpu_clean_before_launch; then
    run_p03
    exit "$?"
  fi
  log "GPU${SELECTED_GPU_INDEX} not clean yet; sleeping ${CHECK_INTERVAL_SECONDS}s"
  sleep "$CHECK_INTERVAL_SECONDS"
done
