#!/usr/bin/env python3
"""Test if dense vs streaming discrepancy is systematic across seeds - simplified version."""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
sys.path.insert(0, '/home/chakwong/BayesFilter')

import tensorflow as tf
import numpy as np

from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
MEMORY_POLICY = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)

# Enable TF32
tf.config.experimental.enable_tensor_float_32_execution(True)

# Import the actual runner's _row function and setup
from docs.benchmarks.run_sqmc_rerun_corrected_filter_20260906 import (
    build_full_horizon_models,
    _make_evaluator,
    ROUTE_CONTROLS,
)

print("Loading Austria SIR model...")
with tf.device("/CPU:0"):
    models = build_full_horizon_models(include_references=False)

model = next(item for item in models if item.row_id == "austria_sir_T20")
N = 1008
route = "repaired_permutation"
reset = "trust_region"

print(f"\nTesting {route} at N={N} across 10 seeds...")
print("(This takes ~2 min per seed, ~20 min total)\n")

results = []

for seed in range(97701, 97711):  # 10 seeds
    # Make evaluators for both transport modes
    evaluator_dense = _make_evaluator(model, N, route, reset, "dense")
    evaluator_streaming = _make_evaluator(model, N, route, reset, "streaming")

    # Generate identical inputs on CPU
    with tf.device("/CPU:0"):
        controls = ROUTE_CONTROLS[route]
        d = model.callbacks.state_dimension
        horizon = int(model.observations.shape[0])

        # Use TF stateless random to ensure reproducibility
        key = tf.constant([seed, 0], dtype=tf.int32)
        initial_noise = tf.random.stateless_normal([N, d], key, dtype=tf.float32)

        key = tf.constant([seed, 1], dtype=tf.int32)
        process_noise = tf.random.stateless_normal([horizon, N, d], key, dtype=tf.float32)

        key = tf.constant([seed, 2], dtype=tf.int32)
        ancestor_uniforms = tf.random.stateless_uniform([horizon, N], key, dtype=tf.float32)

        key = tf.constant([seed, 3], dtype=tf.int32)
        design = tf.random.stateless_normal([N, d], key, dtype=tf.float32)

        # Fixed state map (from MAP_LOCATION/MAP_BASE_SCALE in runner)
        location = tf.zeros([d], dtype=tf.float32)
        scale = tf.ones([d], dtype=tf.float32)

    # Run dense
    value_dense, _, diagnostics_dense = evaluator_dense(
        model.theta,
        model.observations,
        initial_noise,
        process_noise,
        ancestor_uniforms,
        location,
        scale,
        design,
    )

    # Run streaming
    value_streaming, _, diagnostics_streaming = evaluator_streaming(
        model.theta,
        model.observations,
        initial_noise,
        process_noise,
        ancestor_uniforms,
        location,
        scale,
        design,
    )

    diff = float(value_streaming.numpy() - value_dense.numpy())
    results.append({
        'seed': seed,
        'dense': float(value_dense.numpy()),
        'streaming': float(value_streaming.numpy()),
        'diff': diff,
    })
    print(f"Seed {seed}: dense={value_dense.numpy():.6f}, streaming={value_streaming.numpy():.6f}, diff={diff:+.6f}")

print("\n" + "="*70)
print("STATISTICAL ANALYSIS")
print("="*70)

diffs = [r['diff'] for r in results]
mean_diff = np.mean(diffs)
std_diff = np.std(diffs, ddof=1)
min_diff = min(diffs)
max_diff = max(diffs)

print(f"\nMean difference (streaming - dense): {mean_diff:+.6f}")
print(f"Std dev of differences: {std_diff:.6f}")
print(f"Min difference: {min_diff:+.6f}")
print(f"Max difference: {max_diff:+.6f}")
print(f"Range: {max_diff - min_diff:.6f}")

print("\n" + "="*70)
print("SYMMETRY TEST")
print("="*70)

positive_count = sum(1 for d in diffs if d > 0)
negative_count = sum(1 for d in diffs if d < 0)
zero_count = sum(1 for d in diffs if abs(d) < 1e-9)

print(f"\nPositive diffs (streaming > dense): {positive_count}")
print(f"Negative diffs (streaming < dense): {negative_count}")
print(f"Zero diffs (exact match): {zero_count}")

if abs(mean_diff) < 0.01:
    print(f"\n✓ Error is CENTERED AROUND ZERO (mean = {mean_diff:+.6f})")
    print("  → Neither mode has systematic bias")
else:
    print(f"\n✗ Error is BIASED: mean = {mean_diff:+.6f}")
    print("  → One mode is systematically different")

if std_diff < abs(mean_diff) * 0.5 if mean_diff != 0 else False:
    print(f"\n✓ Error is SYSTEMATIC (std={std_diff:.6f} < 0.5*|mean|)")
    print("  → Difference is consistent across seeds")
else:
    print(f"\n✗ Error is NOISY (std={std_diff:.6f}, std/|mean|={std_diff/abs(mean_diff) if mean_diff!=0 else 'inf'})")
    print("  → Difference varies significantly across seeds")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)

if zero_count == 10:
    print("\n✓ Dense and streaming are BITWISE IDENTICAL across all seeds")
    print("  → No discrepancy exists; smoke test anomaly was measurement error")
elif abs(mean_diff) < 0.01 and std_diff < 0.1:
    print("\n✓ Error is SYMMETRIC and SMALL (pure FP roundoff)")
    print("  → Either mode is acceptable; choose based on chunk-rule compliance")
elif abs(mean_diff) > 0.1:
    print(f"\n✗ Error is LARGE and {'SYSTEMATIC' if std_diff < abs(mean_diff)*0.5 else 'NOISY'}")
    print("  → Requires further investigation or TF32 disable")
