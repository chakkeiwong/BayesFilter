#!/usr/bin/env bash
set -euo pipefail

export TF_FORCE_GPU_ALLOW_GROWTH=true
export CUDA_VISIBLE_DEVICES=1

PYTHON=/home/ubuntu/anaconda3/envs/tfgpu/bin/python
RUNNER=docs/benchmarks/run_neutra_paper_d100_training_2026_08_13.py
ROOT=docs/plans/artifacts/weighted-forward-kl-positive-controls-2026-08-12/paper-d100
REPLAY="$ROOT/gaussian-replay-r1"
CONSTANTS="$ROOT/source-r1/paper_ill_cond_gaussian_d100_constants.json"

"$PYTHON" "$RUNNER" \
  --output-root "$ROOT/gaussian-reverse-lr1e-3-canary-r2" \
  --replay-root "$REPLAY" \
  --gaussian-constants "$CONSTANTS" \
  --target paper_ill_cond_gaussian \
  --objective reverse_kl \
  --device 1 \
  --updates 200 \
  --batch-size 4096 \
  --checkpoint-every 50 \
  --hidden-width 100 \
  --stages 3 \
  --learning-rate 1e-3 \
  --learning-rate-schedule constant \
  --initialization-seed-offset 111

"$PYTHON" "$RUNNER" \
  --output-root "$ROOT/gaussian-reverse-lr1e-2-canary-r1" \
  --replay-root "$REPLAY" \
  --gaussian-constants "$CONSTANTS" \
  --target paper_ill_cond_gaussian \
  --objective reverse_kl \
  --device 1 \
  --updates 200 \
  --batch-size 4096 \
  --checkpoint-every 50 \
  --hidden-width 100 \
  --stages 3 \
  --learning-rate 1e-2 \
  --learning-rate-schedule paper_piecewise \
  --initialization-seed-offset 111

"$PYTHON" "$RUNNER" \
  --output-root "$ROOT/gaussian-forward-lr1e-3-canary-r1" \
  --replay-root "$REPLAY" \
  --gaussian-constants "$CONSTANTS" \
  --target paper_ill_cond_gaussian \
  --objective forward_kl \
  --device 1 \
  --updates 200 \
  --batch-size 4096 \
  --checkpoint-every 50 \
  --hidden-width 100 \
  --stages 3 \
  --learning-rate 1e-3 \
  --learning-rate-schedule constant \
  --initialization-seed-offset 211

"$PYTHON" "$RUNNER" \
  --output-root "$ROOT/gaussian-forward-lr1e-2-canary-r1" \
  --replay-root "$REPLAY" \
  --gaussian-constants "$CONSTANTS" \
  --target paper_ill_cond_gaussian \
  --objective forward_kl \
  --device 1 \
  --updates 200 \
  --batch-size 4096 \
  --checkpoint-every 50 \
  --hidden-width 100 \
  --stages 3 \
  --learning-rate 1e-2 \
  --learning-rate-schedule paper_piecewise \
  --initialization-seed-offset 211
