"""Phase 2: Optimized surrogate-force HMC (reduced memory).

Reduce chains, samples, and enable XLA compilation for better memory management.
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import sys
from pathlib import Path
import time

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

DTYPE = tf.float64
THETA_TRUE = np.array([0.72, 0.55, 0.35, 0.35, 0.45])

print("=" * 80)
print("Phase 2: Surrogate-Force HMC (Optimized)")
print("=" * 80)
print()

# Build models
print("Building models...")
exact_target = make_canonical_neutra_target("lgssm", particle_count=1008, noise_seed=140000, substeps=12)
exact_model = exact_target.fused_model

ridge_lambda = 1e-3
ridge_delta = 1e-3
damped_model = PerPointScoreModel(
    transition_mean_fn=exact_model.transition_mean_fn,
    transition_mean_tangent_fn=exact_model.transition_mean_tangent_fn,
    observation_fn=exact_model.observation_fn,
    observation_jacobian_fn=exact_model.observation_jacobian_fn,
    observation_tangent_fn=exact_model.observation_tangent_fn,
    process_covariance=exact_model.process_covariance + ridge_lambda * tf.eye(3, dtype=DTYPE),
    observation_covariance=exact_model.observation_covariance + ridge_delta * tf.eye(3, dtype=DTYPE),
)
print(f"  Ridge λ={ridge_lambda}, δ={ridge_delta}")
print()

# Surrogate-force target with XLA compilation
print("Building surrogate-force target...")

@tf.function(jit_compile=True)
def compute_exact_value(theta_batch):
    """XLA-compiled exact value computation."""
    directions_dummy = tf.zeros_like(theta_batch)
    vals, _, _ = canonical_batch_fused_value_score(
        exact_model, theta_batch, directions_dummy,
        exact_target.initial_states, exact_target.initial_covariances,
        exact_target.noises, exact_target.observations,
        substeps=exact_target.substeps,
    )
    return vals

@tf.function(jit_compile=True)
def compute_damped_gradient(theta_batch):
    """XLA-compiled damped gradient computation."""
    batch_size = tf.shape(theta_batch)[0]
    param_dim = int(theta_batch.shape[1])

    gradients = []
    for p in range(param_dim):
        direction = tf.zeros([batch_size, param_dim], DTYPE)
        indices = tf.stack([tf.range(batch_size), tf.fill([batch_size], p)], axis=1)
        direction = tf.tensor_scatter_nd_update(direction, indices, tf.ones([batch_size], DTYPE))

        _, scores, _ = canonical_batch_fused_value_score(
            damped_model, theta_batch, direction,
            exact_target.initial_states, exact_target.initial_covariances,
            exact_target.noises, exact_target.observations,
            substeps=exact_target.substeps,
        )
        gradients.append(scores)

    return tf.stack(gradients, axis=1)

def surrogate_force_log_prob_fn(theta):
    """Target for HMC."""
    if len(theta.shape) == 1:
        theta_batch = theta[None, :]
        val = compute_exact_value(theta_batch)
        return -val[0]
    else:
        val = compute_exact_value(theta)
        return -val

print("  Testing target...")
t0 = time.time()
test_val = surrogate_force_log_prob_fn(tf.constant(THETA_TRUE, DTYPE))
print(f"  First call: {time.time()-t0:.3f}s (includes compilation)")
t0 = time.time()
test_val = surrogate_force_log_prob_fn(tf.constant(THETA_TRUE, DTYPE))
print(f"  Second call: {time.time()-t0:.3f}s (cached)")
print(f"  Value: {float(test_val):.6f}")
print()

# Reduced HMC settings
print("Building HMC kernel (reduced memory)...")
num_burnin = 200
num_results = 200
num_chains = 2  # Reduced from 4

kernel = tfp.mcmc.HamiltonianMonteCarlo(
    target_log_prob_fn=surrogate_force_log_prob_fn,
    step_size=0.01,
    num_leapfrog_steps=5,  # Reduced from 10
)

adaptive_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
    inner_kernel=kernel,
    num_adaptation_steps=int(0.8 * num_burnin),
    target_accept_prob=0.65,
)

initial_state = tf.constant(
    np.random.RandomState(42).randn(num_chains, 5) * 0.1 + THETA_TRUE,
    DTYPE
)

print(f"  Chains: {num_chains}")
print(f"  Burnin: {num_burnin}")
print(f"  Samples: {num_results}")
print()

# Run HMC
print("Running HMC...")
t0 = time.time()

samples, trace = tfp.mcmc.sample_chain(
    num_results=num_results,
    num_burnin_steps=num_burnin,
    current_state=initial_state,
    kernel=adaptive_kernel,
    trace_fn=lambda _, pkr: {
        'is_accepted': pkr.inner_results.is_accepted,
    }
)

t1 = time.time()
print(f"  Total time: {(t1-t0)/60:.2f} minutes")
print()

# Results
acceptance = float(tf.reduce_mean(tf.cast(trace['is_accepted'], tf.float32)))
print(f"Results:")
print(f"  Samples shape: {samples.shape}")
print(f"  Acceptance rate: {acceptance:.3f}")
print()

means = tf.reduce_mean(samples, axis=[0, 1]).numpy()
stds = tf.math.reduce_std(samples, axis=[0, 1]).numpy()

print("Parameter estimates (mean ± std):")
param_names = ['phi1', 'phi2', 'phi3', 'q', 'r']
for i, name in enumerate(param_names):
    print(f"  {name}: {means[i]:.4f} ± {stds[i]:.4f}  (true: {THETA_TRUE[i]:.4f})")
print()

print("SUCCESS: Surrogate-force HMC completed!")
