#!/usr/bin/env python3
"""Debug streaming vs dense cost_scale discrepancy."""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import numpy as np

# Enable TF32 to match production
tf.config.experimental.enable_tensor_float_32_execution(True)

# Generate a small test cloud
np.random.seed(97701)
N = 1008
d = 18  # Austria SIR state dimension
particles = tf.constant(np.random.randn(N, d).astype(np.float32), dtype=tf.float32)
weights = tf.ones([N], dtype=tf.float32) / float(N)

print("=== DENSE COST_SCALE (full N×N matrix) ===")
deltas = particles[:, None, :] - particles[None, :, :]
cost = tf.reduce_sum(tf.square(deltas), axis=-1)
cost_scale_dense = tf.maximum(
    tf.reduce_mean(cost), tf.constant(1.0e-3, particles.dtype)
)
print(f"cost_scale_dense = {cost_scale_dense.numpy():.6f}")
print(f"cost mean = {tf.reduce_mean(cost).numpy():.6f}")
print(f"cost min/max = {tf.reduce_min(cost).numpy():.6f} / {tf.reduce_max(cost).numpy():.6f}")

print("\n=== STREAMING COST_SCALE (chunked accumulation, K=1008) ===")
# Mimic streaming path with K=N (one chunk)
row_chunk_size = N
col_chunk_size = N
row_blocks = N // row_chunk_size  # = 1
col_blocks = N // col_chunk_size  # = 1

# Streaming accumulates sum(cost), not mean(cost)
total_cost = tf.reduce_sum(cost)
count_squared = tf.cast(N, tf.float32) ** 2
cost_scale_streaming_k1008 = tf.maximum(
    total_cost / count_squared, tf.constant(1.0e-3, tf.float32)
)
print(f"cost_scale_streaming (K={N}) = {cost_scale_streaming_k1008.numpy():.6f}")
print(f"total_cost / N^2 = {(total_cost / count_squared).numpy():.6f}")

print("\n=== STREAMING COST_SCALE (chunked accumulation, K=2016 if N=4032) ===")
# For N=4032, streaming uses K=2016 (2 blocks)
N_large = 4032
d_large = 18
particles_large = tf.constant(np.random.randn(N_large, d_large).astype(np.float32), dtype=tf.float32)
K_large = 2016

# Dense cost_scale for N=4032
deltas_large = particles_large[:, None, :] - particles_large[None, :, :]
cost_large = tf.reduce_sum(tf.square(deltas_large), axis=-1)
cost_scale_dense_large = tf.maximum(
    tf.reduce_mean(cost_large), tf.constant(1.0e-3, particles_large.dtype)
)
print(f"\nDense (N={N_large}): cost_scale = {cost_scale_dense_large.numpy():.6f}")

# Streaming cost_scale for N=4032, K=2016
total_cost_large = tf.reduce_sum(cost_large)
count_squared_large = tf.cast(N_large, tf.float32) ** 2
cost_scale_streaming_large = tf.maximum(
    total_cost_large / count_squared_large, tf.constant(1.0e-3, tf.float32)
)
print(f"Streaming (N={N_large}, K={K_large}): cost_scale = {cost_scale_streaming_large.numpy():.6f}")

print("\n=== VERDICT ===")
print("Dense computes: mean(cost)")
print("Streaming computes: sum(cost) / N^2  (which equals mean(cost))")
print("These SHOULD be identical.")
print()
print(f"Difference at N=1008: {abs(cost_scale_dense.numpy() - cost_scale_streaming_k1008.numpy()):.9f}")
print(f"Difference at N=4032: {abs(cost_scale_dense_large.numpy() - cost_scale_streaming_large.numpy()):.9f}")

print("\n=== BUT: Check if streaming bypasses itself at K=N ===")
print("Per line 200 of genut_guided_proposal_tf.py:")
print("if row_blocks == 1 and col_blocks == 1:")
print("    return _dense_sinkhorn_barycentric_value(...)")
print()
print(f"At N=1008, K=1008: row_blocks={row_blocks}, col_blocks={col_blocks}")
print("→ Streaming SHOULD call dense internally (identical results)")
print()
print(f"At N=4032, K=2016: row_blocks=2, col_blocks=2")
print("→ Streaming DOES NOT call dense (different accumulation order)")

print("\n=== TF32 STATUS ===")
print(f"TF32 enabled: {tf.config.experimental.tensor_float_32_execution_enabled()}")

print("\n=== HYPOTHESIS ===")
print("At N=1008, streaming→dense internally → results should match")
print("At N=4032, streaming uses chunked path → accumulation order differs")
print("BUT: sum(cost)/N^2 vs mean(cost) should still be algebraically identical.")
print()
print("Check: TF32 precision loss in chunked accumulation vs full reduction?")
print("Or: different operation order causing FP32 roundoff amplification?")
