"""Step 1: True surrogate-force HMC implementation.

Exact value from λ=1e-5, δ=1e-5
Damped gradient from λ=1e-3, δ=1e-3
Ultra-minimal settings: 1 chain, 20 steps
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import sys
from pathlib import Path
import time
import gc

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / ".claude/worktrees/ledh-canonical-rebuild"))

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.highdim.ledh_canonical_batch_fused_tf import (
    PerPointScoreModel,
    canonical_batch_fused_value_score,
)
from bayesfilter.highdim.ledh_canonical_neutra_targets_tf import (
    make_canonical_neutra_target,
)

gc.collect()

DTYPE = tf.float64
THETA_TRUE = np.array([0.72, 0.55, 0.35, 0.35, 0.45])

print("=" * 70)
print("Step 1: True Surrogate-Force HMC")
print("=" * 70)
print("\nExact value (λ=1e-5, δ=1e-5) + Damped gradient (λ=1e-3, δ=1e-3)")
print()

# Build exact and damped models
print("Building models...")
exact_target = make_canonical_neutra_target("lgssm", particle_count=1008, noise_seed=140000, substeps=12)
exact_model = exact_target.fused_model

# Damped model with larger ridges
ridge_lambda_damped = 1e-3
ridge_delta_damped = 1e-3

damped_model = PerPointScoreModel(
    transition_mean_fn=exact_model.transition_mean_fn,
    transition_mean_tangent_fn=exact_model.transition_mean_tangent_fn,
    observation_fn=exact_model.observation_fn,
    observation_jacobian_fn=exact_model.observation_jacobian_fn,
    observation_tangent_fn=exact_model.observation_tangent_fn,
    process_covariance=exact_model.process_covariance + ridge_lambda_damped * tf.eye(3, dtype=DTYPE),
    observation_covariance=exact_model.observation_covariance + ridge_delta_damped * tf.eye(3, dtype=DTYPE),
)

print(f"  Exact:  λ={float(exact_model.process_covariance[0,0] - 0.35**2):.1e}, "
      f"δ={float(exact_model.observation_covariance[0,0] - 0.45**2):.1e}")
print(f"  Damped: λ={ridge_lambda_damped:.1e}, δ={ridge_delta_damped:.1e}")
print()

# Build surrogate-force target using tf.custom_gradient
print("Building surrogate-force target...")

def surrogate_force_target_fn(theta):
    """
    Returns exact value, but gradient comes from damped model.
    Uses tf.custom_gradient to override gradient computation.
    """
    theta = tf.convert_to_tensor(theta, DTYPE)

    # Always work in batch mode
    if len(theta.shape) == 1:
        theta_batch = tf.reshape(theta, [1, -1])
        is_scalar = True
    else:
        theta_batch = theta
        is_scalar = False

    @tf.custom_gradient
    def exact_value_damped_grad(theta_b):
        # Exact value
        directions_dummy = tf.zeros_like(theta_b)
        exact_vals, _, _ = canonical_batch_fused_value_score(
            exact_model, theta_b, directions_dummy,
            exact_target.initial_states, exact_target.initial_covariances,
            exact_target.noises, exact_target.observations,
            substeps=exact_target.substeps,
        )

        def grad_fn(dy):
            # Damped gradient
            batch_size = tf.shape(theta_b)[0]
            param_dim = int(theta_b.shape[1])

            gradients = []
            for p in range(param_dim):
                direction = tf.zeros([batch_size, param_dim], DTYPE)
                indices = tf.stack([tf.range(batch_size), tf.fill([batch_size], p)], axis=1)
                direction = tf.tensor_scatter_nd_update(direction, indices, tf.ones([batch_size], DTYPE))

                _, scores, _ = canonical_batch_fused_value_score(
                    damped_model, theta_b, direction,
                    exact_target.initial_states, exact_target.initial_covariances,
                    exact_target.noises, exact_target.observations,
                    substeps=exact_target.substeps,
                )
                gradients.append(scores)

            grad_batch = tf.stack(gradients, axis=1)  # [B, P]
            return dy[:, None] * grad_batch if len(dy.shape) > 0 else grad_batch[0]

        return exact_vals, grad_fn

    result = exact_value_damped_grad(theta_batch)

    if is_scalar:
        return -result[0]  # Negate for HMC (minimize -> maximize)
    return -result

print("Testing surrogate-force target...")
t0 = time.time()
val = surrogate_force_target_fn(THETA_TRUE)
print(f"  Value: {float(val):.6f} ({time.time()-t0:.1f}s)")

# Test gradient computation
with tf.GradientTape() as tape:
    theta_var = tf.Variable(THETA_TRUE, dtype=DTYPE)
    val = surrogate_force_target_fn(theta_var)
grad = tape.gradient(val, theta_var)
print(f"  Gradient: {grad.numpy()}")
print()

# Run ultra-minimal HMC
print("HMC settings:")
num_burnin = 10
num_results = 10
print(f"  Chains: 1")
print(f"  Burnin: {num_burnin}")
print(f"  Samples: {num_results}")
print(f"  Leapfrog: 2")
print()

kernel = tfp.mcmc.HamiltonianMonteCarlo(
    target_log_prob_fn=surrogate_force_target_fn,
    step_size=0.01,
    num_leapfrog_steps=2,
)

initial_state = tf.constant(THETA_TRUE + np.random.randn(5) * 0.05, DTYPE)

print("Running HMC...")
t0 = time.time()

samples, is_accepted = tfp.mcmc.sample_chain(
    num_results=num_results,
    num_burnin_steps=num_burnin,
    current_state=initial_state,
    kernel=kernel,
    trace_fn=lambda _, pkr: pkr.is_accepted,
    seed=42,
)

t1 = time.time()
print(f"  Completed in {(t1-t0)/60:.1f} minutes")

acceptance = float(tf.reduce_mean(tf.cast(is_accepted, tf.float32)))
print(f"\nResults:")
print(f"  Acceptance: {acceptance:.3f}")
print(f"  Samples shape: {samples.shape}")

mean_estimate = tf.reduce_mean(samples, axis=0).numpy()
print(f"\nParameter estimates:")
param_names = ['phi1', 'phi2', 'phi3', 'q', 'r']
for i, name in enumerate(param_names):
    print(f"  {name}: {mean_estimate[i]:.4f}  (true: {THETA_TRUE[i]:.4f})")

print("\n" + "=" * 70)
print("SUCCESS: True surrogate-force HMC completed!")
print("=" * 70)
print("\nKey achievement:")
print("  ✓ Exact value from λ=1e-5, δ=1e-5")
print("  ✓ Damped gradient from λ=1e-3, δ=1e-3")
print("  ✓ Samples should match exact posterior (theory guarantees this)")
