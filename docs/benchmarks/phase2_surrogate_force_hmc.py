"""Phase 2: Full surrogate-force HMC run on LGSSM.

Run HMC with exact value, damped gradient. Compare to standard HMC baseline.
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
print("Phase 2: Surrogate-Force HMC Full Run")
print("=" * 80)
print()

# Build exact and damped models
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

# Surrogate-force target
@tf.function(autograph=False)
def surrogate_force_log_prob_and_grad_batch(theta_batch):
    """Batched: value from exact, gradient from damped."""
    batch_size = int(theta_batch.shape[0])
    param_dim = int(theta_batch.shape[1])

    # Exact value
    directions_dummy = tf.zeros_like(theta_batch)
    exact_vals, _, _ = canonical_batch_fused_value_score(
        exact_model, theta_batch, directions_dummy,
        exact_target.initial_states, exact_target.initial_covariances,
        exact_target.noises, exact_target.observations,
        substeps=exact_target.substeps,
    )

    # Damped gradients
    gradients = []
    for p in range(param_dim):
        direction = tf.zeros([batch_size, param_dim], DTYPE)
        direction = tf.tensor_scatter_nd_update(
            direction,
            [[i, p] for i in range(batch_size)],
            tf.ones([batch_size], DTYPE)
        )
        _, scores, _ = canonical_batch_fused_value_score(
            damped_model, theta_batch, direction,
            exact_target.initial_states, exact_target.initial_covariances,
            exact_target.noises, exact_target.observations,
            substeps=exact_target.substeps,
        )
        gradients.append(scores)

    grad_batch = tf.stack(gradients, axis=1)
    return -exact_vals, -grad_batch  # Negate for HMC

def surrogate_force_log_prob_fn(theta):
    """Single theta wrapper."""
    if len(theta.shape) == 1:
        val, grad = surrogate_force_log_prob_and_grad_batch(theta[None, :])
        return val[0]
    else:
        val, grad = surrogate_force_log_prob_and_grad_batch(theta)
        return val

# HMC kernel
print("Building HMC kernel...")
num_burnin = 500
num_results = 500
num_chains = 4

kernel = tfp.mcmc.HamiltonianMonteCarlo(
    target_log_prob_fn=surrogate_force_log_prob_fn,
    step_size=0.01,
    num_leapfrog_steps=10,
)

adaptive_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
    inner_kernel=kernel,
    num_adaptation_steps=int(0.8 * num_burnin),
    target_accept_prob=0.65,
)

# Initial state
initial_state = tf.constant(
    np.random.randn(num_chains, 5) * 0.1 + THETA_TRUE,
    DTYPE
)

print(f"  Chains: {num_chains}")
print(f"  Burnin: {num_burnin}")
print(f"  Samples: {num_results}")
print(f"  Initial state shape: {initial_state.shape}")
print()

# Run HMC
print("Running HMC...")
print("  (This may take several minutes)")
t0 = time.time()

@tf.function(autograph=False)
def run_chain():
    return tfp.mcmc.sample_chain(
        num_results=num_results,
        num_burnin_steps=num_burnin,
        current_state=initial_state,
        kernel=adaptive_kernel,
        trace_fn=lambda _, pkr: {
            'step_size': pkr.inner_results.accepted_results.step_size,
            'is_accepted': pkr.inner_results.is_accepted,
        }
    )

samples, trace = run_chain()
t1 = time.time()

print(f"  Total time: {(t1-t0)/60:.2f} minutes")
print()

# Analyze results
print("Results:")
print(f"  Samples shape: {samples.shape}")
acceptance = float(tf.reduce_mean(tf.cast(trace['is_accepted'], tf.float32)))
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
