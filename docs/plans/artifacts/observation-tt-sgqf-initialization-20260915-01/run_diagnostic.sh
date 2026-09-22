#!/usr/bin/env bash
set -euo pipefail
attempt="${1:?unique attempt suffix required}"
if [[ ! "$attempt" =~ ^[0-9]{2}$ ]]; then exit 2; fi
cd /home/chakwong/BayesFilter
export CUDA_VISIBLE_DEVICES=GPU-d54fdcfc-c6ed-dbe7-25c7-93f737e0f93a
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_NUM_INTRAOP_THREADS=2
export TF_NUM_INTEROP_THREADS=1
timeout 3660s /home/chakwong/anaconda3/envs/tftwogpu/bin/python \
  docs/benchmarks/diagnose_observation_tt_sgqf_initialization.py \
  --output-root "docs/benchmarks/artifacts/observation_tt_sgqf_initialization_20260915/attempt-${attempt}" \
  --wall-budget-seconds 3600 \
  > "docs/plans/artifacts/observation-tt-sgqf-initialization-20260915-01/attempt-${attempt}.log" 2>&1
