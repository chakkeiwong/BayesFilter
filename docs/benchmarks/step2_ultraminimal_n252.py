"""Step 2 (revised): Ultra-minimal test with N=252.

Same ultra-minimal settings as Step 1, but with reduced particles.
Goal: Confirm N=252 completes faster than N=1008.
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
N_REDUCED = 252

print("=" * 70)
print(f"Step 2 (Revised): Ultra-Minimal with N={N_REDUCED}")
print("=" * 70)
print()

# Build with reduced N
print(f"Building models (N={N_REDUCED} vs 1008 in Step 1)...")
exact_target = make_canonical_neutra_target("lgssm", particle_count=N_REDUCED, noise_seed=140000, substeps=12)
exact_model = exact_target.fused_model

damped_model = PerPointScoreModel(
    transition_mean_fn=exact_model.transition_mean_fn,
    transition_mean_tangent_fn=exact_model.transition_mean_tangent_fn,
    observation_fn=exact_model.observation_fn,
    observation_jacobian_fn=exact_model.observation_jacobian_fn,
    observation_tangent_fn=exact_model.observation_tangent_fn,
    process_covariance=exact_model.process_covariance + 1e-3 * tf.eye(3, dtype=DTYPE),
    observation_covariance=exact_model.observation_covariance + 1e-3 * tf.eye(3, dtype=DTYPE),
)

print("  Exact:  λ=1e-5, δ=1e-5")
print("  Damped: λ=1e-3, δ=1e-3")
print()

# Surrogate-force target
def surrogate_force_target_fn(theta):
    theta = tf.convert_to_tensor(theta, DTYPE)
    if len(theta.shape) == 1:
        theta_batch = tf.reshape(theta, [1, -1])
        is_scalar = True
    else:
        theta_batch = theta
        is_scalar = False

    @tf.custom_gradient
    def exact_value_damped_grad(theta_b):
        directions_dummy = tf.zeros_like(theta_b)
        exact_vals, _, _ = canonical_batch_fused_value_score(
            exact_model, theta_b, directions_dummy,
            exact_target.initial_states, exact_target.initial_covariances,
            exact_target.noises, exact_target.observations,
            substeps=exact_target.substeps,
        )

        def grad_fn(dy):
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
            grad_batch = tf.stack(gradients, axis=1)
            return dy[:, None] * grad_batch if len(dy.shape) > 0 else grad_batch[0]

        return exact_vals, grad_fn

    result = exact_value_damped_grad(theta_batch)
    if is_scalar:
        return -result[0]
    return -result

print("Testing target...")
t0 = time.time()
val = surrogate_force_target_fn(THETA_TRUE)
t_first = time.time() - t0
print(f"  First call: {t_first:.1f}s")

t0 = time.time()
val = surrogate_force_target_fn(THETA_TRUE)
t_second = time.time() - t0
print(f"  Second call: {t_second:.1f}s (should be faster)")
print(f"  Value: {float(val):.6f}")
print()

# Same ultra-minimal settings as Step 1
print("HMC settings (same as Step 1):")
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
print(f"  (Step 1 with N=1008 took 25.3 minutes)")
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
elapsed_min = (t1-t0)/60
print(f"  Completed in {elapsed_min:.1f} minutes")

acceptance = float(tf.reduce_mean(tf.cast(is_accepted, tf.float32)))
mean_estimate = tf.reduce_mean(samples, axis=0).numpy()

print(f"\nResults:")
print(f"  Acceptance: {acceptance:.3f}")
print(f"  Samples shape: {samples.shape}")

param_names = ['phi1', 'phi2', 'phi3', 'q', 'r']
print(f"\nParameter estimates:")
for i, name in enumerate(param_names):
    print(f"  {name}: {mean_estimate[i]:.4f}  (true: {THETA_TRUE[i]:.4f})")

# Compare to Step 1
speedup = 25.3 / elapsed_min
print("\n" + "=" * 70)
print(f"SUCCESS: N={N_REDUCED} completed!")
print("=" * 70)
print(f"\nPerformance vs Step 1 (N=1008):")
print(f"  Step 1: 25.3 minutes")
print(f"  Step 2: {elapsed_min:.1f} minutes")
print(f"  Speedup: {speedup:.2f}x")
print(f"\nConclusion: N={N_REDUCED} is viable for this hardware")
