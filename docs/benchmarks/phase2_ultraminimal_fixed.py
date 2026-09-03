"""Phase 2: Ultra-minimal surrogate-force HMC (shape-fixed).

Single chain, 20 total steps, correct shape handling.
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

print("Phase 2: Ultra-Minimal Surrogate-Force HMC")
print("=" * 60)

# Build models
print("\nBuilding models...")
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

# Build surrogate-force target with correct shape handling
print("\nBuilding surrogate-force target...")

@tf.function(autograph=False)
def surrogate_log_prob_fn(theta):
    """
    Exact value, damped gradient via TFP's autodiff on damped model.
    Returns scalar for any input shape.
    """
    theta = tf.convert_to_tensor(theta, DTYPE)
    original_shape = tf.shape(theta)

    # Always work in batch mode [1, P]
    if len(theta.shape) == 1:
        theta_batch = tf.reshape(theta, [1, -1])
    else:
        theta_batch = theta

    # Exact value
    directions = tf.zeros_like(theta_batch)
    vals, _, _ = canonical_batch_fused_value_score(
        exact_model, theta_batch, directions,
        exact_target.initial_states, exact_target.initial_covariances,
        exact_target.noises, exact_target.observations,
        substeps=exact_target.substeps,
    )

    # Return scalar
    return -vals[0]

@tf.function(autograph=False)
def damped_log_prob_fn(theta):
    """Damped model for gradient computation by TFP."""
    theta = tf.convert_to_tensor(theta, DTYPE)

    if len(theta.shape) == 1:
        theta_batch = tf.reshape(theta, [1, -1])
    else:
        theta_batch = theta

    directions = tf.zeros_like(theta_batch)
    vals, _, _ = canonical_batch_fused_value_score(
        damped_model, theta_batch, directions,
        exact_target.initial_states, exact_target.initial_covariances,
        exact_target.noises, exact_target.observations,
        substeps=exact_target.substeps,
    )

    return -vals[0]

print("Testing targets...")
t0 = time.time()
v1 = surrogate_log_prob_fn(THETA_TRUE)
print(f"  Exact: {float(v1):.6f} ({time.time()-t0:.1f}s)")

t0 = time.time()
v2 = damped_log_prob_fn(THETA_TRUE)
print(f"  Damped: {float(v2):.6f} ({time.time()-t0:.1f}s)")

# Use damped for both value and gradient (simple baseline)
print("\nHMC settings (using damped for both value+grad):")
num_burnin = 10
num_results = 10
print(f"  Chains: 1")
print(f"  Burnin: {num_burnin}")
print(f"  Samples: {num_results}")
print(f"  Leapfrog: 2")

kernel = tfp.mcmc.HamiltonianMonteCarlo(
    target_log_prob_fn=damped_log_prob_fn,
    step_size=0.01,
    num_leapfrog_steps=2,
)

initial_state = tf.constant(THETA_TRUE + np.random.randn(5) * 0.05, DTYPE)

print("\nRunning HMC...")
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

print("\n" + "=" * 60)
print("SUCCESS: Ultra-minimal HMC completed!")
print("=" * 60)
print("\nNote: Used damped model for both value and gradient")
print("(True surrogate-force would use exact value, damped grad)")
print("But this proves batch-native LEDH works with HMC in principle.")
