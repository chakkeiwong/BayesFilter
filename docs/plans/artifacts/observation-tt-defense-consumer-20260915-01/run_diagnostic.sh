#!/usr/bin/env bash
set -euo pipefail
attempt="${1:?unique attempt required}"
cd /home/chakwong/BayesFilter
export CUDA_VISIBLE_DEVICES=GPU-d54fdcfc-c6ed-dbe7-25c7-93f737e0f93a
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_NUM_INTRAOP_THREADS=2
export TF_NUM_INTEROP_THREADS=1
timeout 300s /home/chakwong/anaconda3/envs/tftwogpu/bin/python \
  docs/benchmarks/diagnose_observation_tt_defense_consumer.py \
  --output-root "docs/benchmarks/artifacts/observation_tt_defense_consumer_20260915/${attempt}" \
  --wall-budget-seconds 290 \
  > "docs/plans/artifacts/observation-tt-defense-consumer-20260915-01/${attempt}.log" 2>&1
