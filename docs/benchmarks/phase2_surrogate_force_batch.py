"""Phase 2: Surrogate-force HMC using damped LEDH for gradient.

Strategy: Create two separate PerPointScoreModel instances with different
process/observation covariances (exact vs damped), then build a custom
gradient adapter that calls exact model for value, damped model for score.
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import sys
from pathlib import Path
import json
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
print("Phase 2: Surrogate-Force HMC (Damped Gradient)")
print("=" * 80)
print()

# Build exact target first
print("Building exact LEDH target...")
exact_target = make_canonical_neutra_target(
    "lgssm",
    particle_count=1008,
    noise_seed=140000,
    substeps=12,
)
print(f"  Model: {exact_target.model_id}")
print()

# Now build damped model by cloning the fused_model with modified covariances
print("Building damped LEDH model...")
exact_model = exact_target.fused_model

# Damping: add ridge to covariances
ridge_lambda = 1e-3
ridge_delta = 1e-3

damped_process_cov = exact_model.process_covariance + ridge_lambda * tf.eye(3, dtype=DTYPE)
damped_obs_cov = exact_model.observation_covariance + ridge_delta * tf.eye(3, dtype=DTYPE)

damped_model = PerPointScoreModel(
    transition_mean_fn=exact_model.transition_mean_fn,
    transition_mean_tangent_fn=exact_model.transition_mean_tangent_fn,
    observation_fn=exact_model.observation_fn,
    observation_jacobian_fn=exact_model.observation_jacobian_fn,
    observation_tangent_fn=exact_model.observation_tangent_fn,
    process_covariance=damped_process_cov,
    observation_covariance=damped_obs_cov,
)
print(f"  Exact process variance: {float(exact_model.process_covariance[0,0]):.6f}")
print(f"  Damped process variance: {float(damped_model.process_covariance[0,0]):.6f}")
print(f"  Exact obs variance: {float(exact_model.observation_covariance[0,0]):.6f}")
print(f"  Damped obs variance: {float(damped_model.observation_covariance[0,0]):.6f}")
print()

# Test both models
print("Testing exact vs damped models...")
theta_test = tf.constant([THETA_TRUE], DTYPE)  # [1, 5]
directions = tf.constant([tf.eye(5, dtype=DTYPE)[0].numpy()], DTYPE)  # [1, 5]

exact_val, exact_score, _ = canonical_batch_fused_value_score(
    exact_model, theta_test, directions,
    exact_target.initial_states,
    exact_target.initial_covariances,
    exact_target.noises,
    exact_target.observations,
    substeps=exact_target.substeps,
)

damped_val, damped_score, _ = canonical_batch_fused_value_score(
    damped_model, theta_test, directions,
    exact_target.initial_states,
    exact_target.initial_covariances,
    exact_target.noises,
    exact_target.observations,
    substeps=exact_target.substeps,
)

print(f"  Exact value: {float(exact_val[0]):.6f}")
print(f"  Damped value: {float(damped_val[0]):.6f}")
print(f"  Value difference: {float(exact_val[0] - damped_val[0]):.6f}")
print()
print(f"  Exact score[0]: {float(exact_score[0]):.6f}")
print(f"  Damped score[0]: {float(damped_score[0]):.6f}")
print(f"  Score difference: {float(exact_score[0] - damped_score[0]):.6f}")
print()

# Build surrogate-force adapter
print("Building surrogate-force HMC target...")

def surrogate_force_log_prob_and_grad(theta):
    """Value from exact, gradient from damped."""
    theta = tf.convert_to_tensor(theta, DTYPE)

    # Handle both batched and unbatched
    if len(theta.shape) == 1:
        theta_batch = theta[None, :]  # [1, P]
        is_single = True
    else:
        theta_batch = theta
        is_single = False

    batch_size = int(theta_batch.shape[0])
    param_dim = int(theta_batch.shape[1])

    # Get exact value (gradient ignored)
    directions_dummy = tf.zeros_like(theta_batch)
    exact_vals, _, _ = canonical_batch_fused_value_score(
        exact_model, theta_batch, directions_dummy,
        exact_target.initial_states,
        exact_target.initial_covariances,
        exact_target.noises,
        exact_target.observations,
        substeps=exact_target.substeps,
    )

    # Get damped gradients (value ignored)
    # Need to call once per parameter direction
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
            exact_target.initial_states,
            exact_target.initial_covariances,
            exact_target.noises,
            exact_target.observations,
            substeps=exact_target.substeps,
        )
        gradients.append(scores)

    grad_batch = tf.stack(gradients, axis=1)  # [B, P]

    if is_single:
        return -exact_vals[0], -grad_batch[0]  # Negate for HMC (minimize -> maximize)
    else:
        return -exact_vals, -grad_batch

# Test surrogate-force
print("Testing surrogate-force adapter...")
t0 = time.time()
val, grad = surrogate_force_log_prob_and_grad(THETA_TRUE)
t1 = time.time()

print(f"  Time: {t1-t0:.3f}s")
print(f"  Value: {float(val):.6f}")
print(f"  Gradient: {grad.numpy()}")
print()

print("SUCCESS: Surrogate-force adapter works!")
print()
print("Next: Run HMC with this target")
