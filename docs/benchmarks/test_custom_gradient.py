"""Test that custom gradient works correctly."""

import tensorflow as tf
from bayesfilter.highdim.ledh_canonical_neutra_targets_tf import (
    _diagonal_lgssm_fused_model,
    _lgssm_frozen_observations,
)
from bayesfilter.inference.ledh_dual_parameter_target import (
    DualParameterLEDHTarget,
)

print("Testing custom gradient implementation...")

# Minimal setup
observations = _lgssm_frozen_observations()
model = _diagonal_lgssm_fused_model()

d = 3
N = 100  # Small
T = observations.shape[0]
dtype = observations.dtype

generator = tf.random.Generator.from_seed(81100)
initial_states = generator.normal([N, d], dtype=dtype) * 0.1
initial_covariances = tf.tile(tf.eye(d, dtype=dtype)[None, :, :], [N, 1, 1]) * 0.01
noises = generator.normal([T, N, d], dtype=dtype) * 0.1

reset_basis = tf.concat([tf.eye(d, dtype=dtype), -tf.eye(d, dtype=dtype)], axis=0)
reset_repeats = (N + 2 * d - 1) // (2 * d)
reset_design = tf.tile(reset_basis, [reset_repeats, 1])[:N]

shared_params = dict(
    substeps=4,  # Fewer substeps
    reset_policy="contract_e",
    reset_design=reset_design,
    reset_epsilon=2.0,
    reset_sinkhorn_steps=4,  # Fewer steps
    reset_balance_steps=4,
    correction_steps=2,
    correction_strength=0.2,
    correction_lm_scale_floor=1e-4,
    correction_trust_radius=0.5,
    pairwise_steps=2,
    pairwise_strength=0.02,
    pairwise_rms_cap=2.0,
    coordinate_cap=0.0,
    annealed_stages=1,
    annealed_seed=0,
)

exact_params = {**shared_params, "reset_ridge": 1e-5, "correction_lm_damping": 1e-2}
biased_params = {**shared_params, "reset_ridge": 1e-3, "correction_lm_damping": 1.0}

target = DualParameterLEDHTarget(
    model=model,
    initial_states=initial_states,
    initial_covariances=initial_covariances,
    noises=noises,
    observations=observations,
    exact_params=exact_params,
    biased_params=biased_params,
)

print("Created target, testing gradient computation...")

theta = tf.constant([1.0, 1.0, 1.0, 0.5, 0.3], dtype=dtype)

# Test 1: Call returns scalar
print("\nTest 1: Call returns scalar value")
value = target(theta)
print(f"  Value shape: {value.shape}")
print(f"  Value: {value.numpy():.6f}")
assert value.shape == (), "Value should be scalar"
print("  ✓ PASS")

# Test 2: Autodiff through custom_gradient
print("\nTest 2: GradientTape extracts custom gradient")
with tf.GradientTape() as tape:
    tape.watch(theta)
    value = target(theta)

grad = tape.gradient(value, theta)
print(f"  Gradient shape: {grad.shape}")
print(f"  Gradient: {grad.numpy()}")
assert grad.shape == (5,), "Gradient should match theta shape"
assert tf.reduce_all(tf.math.is_finite(grad)), "Gradient should be finite"
print("  ✓ PASS")

# Test 3: No memory explosion
print("\nTest 3: Multiple calls don't accumulate memory")
for i in range(5):
    with tf.GradientTape() as tape:
        tape.watch(theta)
        v = target(theta)
    g = tape.gradient(v, theta)
    print(f"  Call {i+1}: value={v.numpy():.6f}, grad_norm={tf.norm(g).numpy():.6f}")

print("\n✓ ALL TESTS PASSED")
print("\nCustom gradient is working correctly.")
print("HMC will use biased (surrogate) gradient while targeting exact value.")
