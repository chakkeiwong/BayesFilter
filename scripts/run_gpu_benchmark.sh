#!/usr/bin/env bash
set -euo pipefail

mode="${1:-phase1_lgssm_full_row_manual_reverse}"

case "${mode}" in
  phase1_lgssm_full_row_manual_reverse)
    exec bash scripts/run_phase1_lgssm_full_row_manual_reverse_gpu.sh
    ;;
  ledh_phase7_ksc_tiny_smoke)
    MPLCONFIGDIR=/tmp python docs/benchmarks/benchmark_ledh_same_target_ksc_sv_value.py \
      --run-scope tiny-smoke \
      --time-steps 4 \
      --num-particles 128 \
      --batch-seeds 81120 \
      --transport-policy active-all \
      --sinkhorn-iterations 2 \
      --row-chunk-size 64 \
      --col-chunk-size 64 \
      --particle-chunk-size 64 \
      --history-mode full \
      --warmups 0 \
      --repeats 1 \
      --device /GPU:0 \
      --expect-device-kind gpu \
      --output docs/plans/ledh-phase7-ksc-sv-forward-scalar-tiny-smoke-artifact-2026-07-07.json \
      --markdown-output docs/plans/ledh-phase7-ksc-sv-forward-scalar-tiny-smoke-artifact-2026-07-07.md
    ;;
  ledh_phase7_ksc_full_row)
    MPLCONFIGDIR=/tmp python docs/benchmarks/benchmark_ledh_same_target_ksc_sv_value.py \
      --run-scope full-row-admission \
      --time-steps 1000 \
      --num-particles 10000 \
      --batch-seeds 81120,81121,81122,81123,81124 \
      --transport-policy active-all \
      --sinkhorn-iterations 10 \
      --row-chunk-size 512 \
      --col-chunk-size 512 \
      --particle-chunk-size 512 \
      --history-mode value-only \
      --warmups 0 \
      --repeats 1 \
      --device /GPU:0 \
      --expect-device-kind gpu \
      --output docs/plans/ledh-phase7-ksc-sv-forward-scalar-artifact-2026-07-07.json \
      --markdown-output docs/plans/ledh-phase7-ksc-sv-forward-scalar-artifact-2026-07-07.md
    ;;
  *)
    echo "unknown run_gpu_benchmark.sh mode: ${mode}" >&2
    exit 2
    ;;
esac
