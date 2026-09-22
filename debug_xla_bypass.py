#!/usr/bin/env python3
"""Test streaming→dense bypass under XLA compilation."""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf
import numpy as np
import sys
sys.path.insert(0, '/home/chakwong/BayesFilter')

from bayesfilter.highdim.genut_guided_proposal_tf import _restore_cloud_primal

# Enable TF32 to match production
tf.config.experimental.enable_tensor_float_32_execution(True)

# Match smoke test conditions
N = 1008
d = 18
seed = 97701

np.random.seed(seed)
particles = tf.constant(np.random.randn(N, d).astype(np.float32), dtype=tf.float32)
weights = tf.ones([N], dtype=tf.float32) / float(N)
design = tf.constant(np.random.randn(N, d).astype(np.float32), dtype=tf.float32)

epsilon = 8.0
sinkhorn_steps = 8
balance_steps = 8
ridge = 1.0e-5

print("=== EAGER MODE (already verified to be identical) ===")
print("Skipping...")

print("\n=== XLA COMPILED MODE ===")

@tf.function(jit_compile=True, autograph=False)
def compiled_dense():
    return _restore_cloud_primal(
        particles,
        weights,
        design,
        epsilon=epsilon,
        sinkhorn_steps=sinkhorn_steps,
        balance_steps=balance_steps,
        ridge=ridge,
        reset_policy="contract_e",
        dual_cap_enabled=True,
        dual_cap_diagonal_steps=3,
        dual_cap_diagonal_strength=0.15,
        dual_cap_pairwise_steps=3,
        dual_cap_pairwise_strength=0.01,
        dual_cap_pairwise_particle_rms_cap=1.5,
        dual_cap_coordinate_cap=0.97,
        dual_cap_coordinate_cap_power=6,
        trust_region_enabled=True,
        trust_region_lm_damping=1.0e-2,
        trust_region_lm_scale_floor=1.0e-4,
        trust_region_radius=0.5,
        transport_plan_mode="dense",
        transport_row_chunk_size=None,
        transport_col_chunk_size=None,
        marginal_tolerance=1.0e-4,
    )

@tf.function(jit_compile=True, autograph=False)
def compiled_streaming():
    return _restore_cloud_primal(
        particles,
        weights,
        design,
        epsilon=epsilon,
        sinkhorn_steps=sinkhorn_steps,
        balance_steps=balance_steps,
        ridge=ridge,
        reset_policy="contract_e",
        dual_cap_enabled=True,
        dual_cap_diagonal_steps=3,
        dual_cap_diagonal_strength=0.15,
        dual_cap_pairwise_steps=3,
        dual_cap_pairwise_strength=0.01,
        dual_cap_pairwise_particle_rms_cap=1.5,
        dual_cap_coordinate_cap=0.97,
        dual_cap_coordinate_cap_power=6,
        trust_region_enabled=True,
        trust_region_lm_damping=1.0e-2,
        trust_region_lm_scale_floor=1.0e-4,
        trust_region_radius=0.5,
        transport_plan_mode="streaming",
        transport_row_chunk_size=N,
        transport_col_chunk_size=N,
        marginal_tolerance=1.0e-4,
    )

print("Compiling dense...")
dense_result = compiled_dense()
print(f"Dense particles[0,:3] = {dense_result['particles'][0,:3].numpy()}")
print(f"Dense reset_valid = {dense_result['reset_valid'].numpy()}")

print("\nCompiling streaming...")
streaming_result = compiled_streaming()
print(f"Streaming particles[0,:3] = {streaming_result['particles'][0,:3].numpy()}")
print(f"Streaming reset_valid = {streaming_result['reset_valid'].numpy()}")

print("\n=== COMPARISON ===")
particles_diff = tf.reduce_max(tf.abs(dense_result['particles'] - streaming_result['particles']))
print(f"Max particles difference: {particles_diff.numpy():.9e}")

if particles_diff < 1e-6:
    print("\n✓ XLA compiled results are identical")
else:
    print(f"\n✗ XLA compiled results DIFFER by {particles_diff.numpy():.6f}")
    print("\nThis confirms the XLA graph topology difference hypothesis.")
    print("Even though streaming→dense bypass works in eager mode,")
    print("XLA compiles different graphs for the two entry points.")
