#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo "usage: $0 {gaussian|sir} output-root" >&2
  exit 2
fi
kind=$1
root=$2
mkdir -p "$root"
for bundle in $(seq 0 9); do
  output=$(printf '%s/bundle_%02d' "$root" "$bundle")
  if [[ -f "$output/result.json" ]]; then
    continue
  fi
  bash scripts/run_classifier_score_variance_bundle_gpu.sh \
    --kind "$kind" --bundle "$bundle" --profile full --output "$output"
done
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/bayesfilter-matplotlib \
/home/chakwong/anaconda3/envs/tftwogpu/bin/python \
  docs/benchmarks/aggregate_classifier_score_variance_20260815.py \
  --root "$root" --bootstrap-replicates 5000
