#!/usr/bin/env python3
"""Test if dense vs streaming discrepancy is systematic across seeds."""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
sys.path.insert(0, '/home/chakwong/BayesFilter')

import tensorflow as tf
import numpy as np

from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
MEMORY_POLICY = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)

from bayesfilter.highdim.ledh_pfpf_genut_initial_rqmc_tf import finite_value_standard_score_initial_rqmc
from docs.benchmarks.run_ledh_pfpf_genut_full_sqmc_full_horizons import (
    build_full_horizon_models,
    campaign_inputs,
)

# Enable TF32
tf.config.experimental.enable_tensor_float_32_execution(True)

# Load Austria SIR model (match smoke runner)
models = build_full_horizon_models(include_references=False)
austria_sir = [m for m in models if m['id'] == 'austria_sir_20'][0]
callbacks = austria_sir['callbacks']
theta = austria_sir['theta']
observations = austria_sir['observations']
N = 1008
d = austria_sir['latent_dim']
horizon = austria_sir['horizon']

# Test parameters (match repaired_permutation)
epsilon = 8.0
sinkhorn_steps = 8
balance_steps = 8
ridge = 1.0e-5

results = []

print("Testing 10 seeds...")
for seed in range(97701, 97711):  # 10 seeds
    np.random.seed(seed)

    # Generate inputs using campaign_inputs (matches runner)
    inputs = campaign_inputs(austria_sir, seed)

    initial_noise = inputs['initial']
    process_noise = inputs['process']
    design = inputs['design']
    ancestor_uniforms = inputs['ancestors']
    state_map_location = inputs['location']
    state_map_scale = inputs['scale']

    # Dense path
    value_dense, _, _ = finite_value_standard_score_initial_rqmc(
        callbacks,
        theta,
        observations,
        initial_noise,
        process_noise,
        design,
        ancestry_policy="hilbert_permutation_one_to_one",
        process_ancestor_uniforms=ancestor_uniforms,
        hilbert_bits=12,
        state_map_policy="fixed_supplied",
        state_map_location=state_map_location,
        state_map_scale=state_map_scale,
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
        score_child_block_size=126,
        transport_plan_mode="dense",
        transport_row_chunk_size=None,
        transport_col_chunk_size=None,
        functional_time_loop=True,
        epsilon=epsilon,
        sinkhorn_steps=sinkhorn_steps,
        balance_steps=balance_steps,
        ridge=ridge,
        marginal_tolerance=1.0e-4,
    )

    # Streaming path (K=N, should bypass)
    value_streaming, _, _ = finite_value_standard_score_initial_rqmc(
        callbacks,
        theta,
        observations,
        initial_noise,
        process_noise,
        design,
        ancestry_policy="hilbert_permutation_one_to_one",
        process_ancestor_uniforms=ancestor_uniforms,
        hilbert_bits=12,
        state_map_policy="fixed_supplied",
        state_map_location=state_map_location,
        state_map_scale=state_map_scale,
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
        score_child_block_size=126,
        transport_plan_mode="streaming",
        transport_row_chunk_size=N,
        transport_col_chunk_size=N,
        functional_time_loop=True,
        epsilon=epsilon,
        sinkhorn_steps=sinkhorn_steps,
        balance_steps=balance_steps,
        ridge=ridge,
        marginal_tolerance=1.0e-4,
    )

    diff = float(value_streaming.numpy() - value_dense.numpy())
    results.append({
        'seed': seed,
        'dense': float(value_dense.numpy()),
        'streaming': float(value_streaming.numpy()),
        'diff': diff,
    })
    print(f"  Seed {seed}: dense={value_dense.numpy():.6f}, streaming={value_streaming.numpy():.6f}, diff={diff:+.6f}")

print("\n=== STATISTICAL ANALYSIS ===")
diffs = [r['diff'] for r in results]
mean_diff = np.mean(diffs)
std_diff = np.std(diffs, ddof=1)
min_diff = min(diffs)
max_diff = max(diffs)

print(f"Mean difference (streaming - dense): {mean_diff:+.6f}")
print(f"Std dev of differences: {std_diff:.6f}")
print(f"Min difference: {min_diff:+.6f}")
print(f"Max difference: {max_diff:+.6f}")
print(f"Range: {max_diff - min_diff:.6f}")

print("\n=== SYMMETRY TEST ===")
positive_count = sum(1 for d in diffs if d > 0)
negative_count = sum(1 for d in diffs if d < 0)
print(f"Positive diffs (streaming > dense): {positive_count}")
print(f"Negative diffs (streaming < dense): {negative_count}")
print(f"Zero diffs (exact match): {10 - positive_count - negative_count}")

if abs(mean_diff) < 0.001:
    print("\n✓ Error is CENTERED AROUND ZERO (symmetric)")
else:
    print(f"\n✗ Error is BIASED: mean = {mean_diff:+.6f}")

if std_diff / abs(mean_diff) < 0.1 if mean_diff != 0 else False:
    print("✓ Error is SYSTEMATIC (low variance)")
else:
    print(f"✗ Error is RANDOM or NOISY (std/mean = {std_diff / abs(mean_diff) if mean_diff != 0 else 'inf'})")
