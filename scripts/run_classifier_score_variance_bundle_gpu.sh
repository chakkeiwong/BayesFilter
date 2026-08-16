#!/usr/bin/env bash
set -euo pipefail
export TF_FORCE_GPU_ALLOW_GROWTH=true CUDA_VISIBLE_DEVICES=1 XLA_FLAGS=--xla_gpu_enable_triton_gemm=false MPLCONFIGDIR=/tmp/bayesfilter-matplotlib
exec /home/chakwong/anaconda3/envs/tftwogpu/bin/python docs/benchmarks/run_classifier_score_variance_bundle_20260815.py "$@"
