#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 3 ]]; then
  echo "usage: $0 {gaussian|sir} {16384|32768} output-root" >&2
  exit 2
fi
kind=$1
count=$2
root=$3
if [[ "$count" != "16384" && "$count" != "32768" ]]; then
  echo "count must be 16384 or 32768" >&2
  exit 2
fi
mkdir -p "$root"
for bundle in $(seq 0 9); do
  output=$(printf '%s/bundle_%02d' "$root" "$bundle")
  if [[ -f "$output/result.json" ]]; then
    continue
  fi
  bash scripts/run_classifier_score_path_count_bundle_gpu.sh \
    --kind "$kind" --bundle "$bundle" --path-count "$count" \
    --profile full --output "$output"
done
