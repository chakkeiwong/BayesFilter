#!/usr/bin/env python3
"""Verify that streaming→dense bypass is actually taken at K=N."""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf
import numpy as np

# Import the actual functions
import sys
sys.path.insert(0, '/home/chakwong/BayesFilter')
from bayesfilter.highdim.genut_guided_proposal_tf import (
    _streaming_sinkhorn_barycentric_value,
    _dense_sinkhorn_barycentric_value,
)

# Match smoke test conditions
N = 1008
d = 18
seed = 97701

np.random.seed(seed)
particles = tf.constant(np.random.randn(N, d).astype(np.float32), dtype=tf.float32)
weights = tf.ones([N], dtype=tf.float32) / float(N)

epsilon = 8.0
sinkhorn_steps = 8
balance_steps = 8

print("=== DENSE PATH ===")
dense_result = _dense_sinkhorn_barycentric_value(
    particles,
    weights,
    epsilon=epsilon,
    sinkhorn_steps=sinkhorn_steps,
    balance_steps=balance_steps,
)
print(f"barycentric[0,:3] = {dense_result['barycentric'][0,:3].numpy()}")
print(f"row_mass[0] = {dense_result['row_mass'][0].numpy():.6f}")

print("\n=== STREAMING PATH (K=N, should bypass to dense internally) ===")
streaming_result = _streaming_sinkhorn_barycentric_value(
    particles,
    weights,
    epsilon=epsilon,
    sinkhorn_steps=sinkhorn_steps,
    balance_steps=balance_steps,
    row_chunk_size=N,
    col_chunk_size=N,
)
print(f"barycentric[0,:3] = {streaming_result['barycentric'][0,:3].numpy()}")
print(f"row_mass[0] = {streaming_result['row_mass'][0].numpy():.6f}")

print("\n=== COMPARISON ===")
barycentric_diff = tf.reduce_max(tf.abs(dense_result['barycentric'] - streaming_result['barycentric']))
row_mass_diff = tf.reduce_max(tf.abs(dense_result['row_mass'] - streaming_result['row_mass']))
print(f"Max barycentric difference: {barycentric_diff.numpy():.9e}")
print(f"Max row_mass difference: {row_mass_diff.numpy():.9e}")

if barycentric_diff < 1e-6 and row_mass_diff < 1e-6:
    print("\n✓ Bypass is working correctly (results are identical)")
else:
    print("\n✗ Bypass FAILED or not taken (results differ)")
