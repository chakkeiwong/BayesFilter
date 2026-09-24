#!/usr/bin/env bash
set -euo pipefail
attempt="${1:?unique attempt required}"
mode="${2:-full}"
cd /home/chakwong/BayesFilter
export CUDA_VISIBLE_DEVICES=1
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_NUM_INTRAOP_THREADS=2
export TF_NUM_INTEROP_THREADS=1
extra=()
limit=2400
if [[ "$mode" == smoke ]]; then extra=(--smoke); limit=180; fi
mkdir -p docs/benchmarks/artifacts/observation_tt_downstream_filter_objective_20260916
mkdir -p docs/plans/artifacts/observation-tt-downstream-filter-objective-20260916-01
timeout "$((limit+100))s" /home/chakwong/anaconda3/envs/tftwogpu/bin/python \
  docs/benchmarks/run_observation_tt_independent_filtering.py \
  --output-root "docs/benchmarks/artifacts/observation_tt_downstream_filter_objective_20260916/${attempt}" \
  --wall-budget-seconds "$limit" "${extra[@]}" \
  > "docs/plans/artifacts/observation-tt-downstream-filter-objective-20260916-01/${attempt}.log" 2>&1
