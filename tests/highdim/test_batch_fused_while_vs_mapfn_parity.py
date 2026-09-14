"""Parity test: nested while_loop vs nested map_fn for batch fused canonical score.

This test verifies that replacing tf.map_fn with tf.while_loop + scatter_update
preserves numerical parity before measuring XLA performance.

Success criteria:
- values match within rtol=5e-4
- scores match within rtol=5e-4
- Both implementations pass primal invariance assertion
"""

from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import tensorflow as tf

from bayesfilter.highdim.ledh_canonical_batch_fused_tf import (
    PerPointScoreModel,
    canonical_batch_fused_value_score,
    canonical_batch_fused_value_score_whileloop,
)

DTYPE = tf.float64


def _fused_model():
    """Create a simple nonlinear model for testing (from test_ledh_canonical_batch_fused.py)."""
    def transition_mean_fn(theta, points):
        # per-point theta rows [M, P]
        return points + theta[:, 0:1] * tf.sin(points)

    def transition_mean_tangent_fn(theta, points, d_points, d_theta):
        return (
            d_theta[:, 0:1] * tf.sin(points)
            + d_points
            + theta[:, 0:1] * tf.cos(points) * d_points
        )

    def observation_fn(points):
        return points

    def observation_jacobian_fn(points):
        return tf.broadcast_to(
            tf.eye(2, dtype=DTYPE), [tf.shape(points)[0], 2, 2]
        )

    def observation_tangent_fn(points, d_points):
        return d_points

    return PerPointScoreModel(
        transition_mean_fn=transition_mean_fn,
        transition_mean_tangent_fn=transition_mean_tangent_fn,
        observation_fn=observation_fn,
        observation_jacobian_fn=observation_jacobian_fn,
        observation_tangent_fn=observation_tangent_fn,
        process_covariance=tf.constant(0.4 * np.eye(2), DTYPE),
        observation_covariance=tf.constant(0.6 * np.eye(2), DTYPE),
    )


def _fixture(seed: int, n: int = 24, horizon: int = 5):
    """Generate test fixture (from test_ledh_canonical_batch_fused.py)."""
    rng = np.random.default_rng(seed)
    initial = tf.constant(rng.standard_normal((n, 2)), DTYPE)
    covs = tf.constant(np.stack([np.eye(2)] * n), DTYPE)
    noises = tf.constant(rng.standard_normal((horizon, n, 2)), DTYPE)
    observations = tf.constant(rng.standard_normal((horizon, 2)), DTYPE)
    return initial, covs, noises, observations


def test_batch_fused_mapfn_baseline():
    """Verify current map_fn implementation runs (baseline for future parity test)."""

    # Small fixture: B=2, K=5, N=24, T=5, d=2
    initial, covs, noises, observations = _fixture(seed=42, n=24, horizon=5)
    model = _fused_model()

    batch_size = 2
    k_count = 5
    param_dim = 1  # Model has 1 parameter

    # Generate theta and directions
    rng = np.random.default_rng(43)
    theta = tf.constant(rng.standard_normal((batch_size, param_dim)), DTYPE)
    directions = tf.constant(rng.standard_normal((batch_size, k_count, param_dim)), DTYPE)

    # Current implementation (map_fn, k_batch_mode="sequential")
    values_mapfn, scores_mapfn, diag = canonical_batch_fused_value_score(
        model=model,
        theta=theta,
        theta_directions=directions,
        initial_states=initial,
        initial_covariances=covs,
        noises=noises,
        observations=observations,
        substeps=2,
        k_batch_mode="sequential",
    )

    # Verify output shapes and finiteness
    assert values_mapfn.shape == (batch_size,)
    assert scores_mapfn.shape == (batch_size, k_count)
    assert tf.reduce_all(tf.math.is_finite(values_mapfn))
    assert tf.reduce_all(tf.math.is_finite(scores_mapfn))
    assert tf.reduce_all(diag["program_valid"])

    print(f"✓ Baseline map_fn implementation runs successfully")
    print(f"  Batch size: {batch_size}, K: {k_count}, N: {initial.shape[0]}, T: {noises.shape[0]}")
    print(f"  values: {values_mapfn.numpy()}")
    print(f"  scores[0]: {scores_mapfn[0].numpy()}")
    print(f"  All program_valid: {diag['program_valid'].numpy()}")


def test_batch_fused_mapfn_vs_whileloop_parity():
    """Compare map_fn vs while_loop implementation - must match within rtol=5e-4."""

    # Small fixture: B=2, K=5, N=24, T=5, d=2
    initial, covs, noises, observations = _fixture(seed=42, n=24, horizon=5)
    model = _fused_model()

    batch_size = 2
    k_count = 5
    param_dim = 1

    # Generate theta and directions
    rng = np.random.default_rng(43)
    theta = tf.constant(rng.standard_normal((batch_size, param_dim)), DTYPE)
    directions = tf.constant(rng.standard_normal((batch_size, k_count, param_dim)), DTYPE)

    common_kwargs = dict(
        theta=theta,
        theta_directions=directions,
        initial_states=initial,
        initial_covariances=covs,
        noises=noises,
        observations=observations,
        substeps=2,
    )

    # map_fn implementation (current)
    values_mapfn, scores_mapfn, diag_mapfn = canonical_batch_fused_value_score(
        model=model, k_batch_mode="sequential", **common_kwargs
    )

    # while_loop implementation (new)
    values_while, scores_while, diag_while = canonical_batch_fused_value_score_whileloop(
        model=model, **common_kwargs
    )

    # Parity checks
    value_err = tf.reduce_max(tf.abs(values_mapfn - values_while) / (tf.abs(values_mapfn) + 1e-12))
    score_err = tf.reduce_max(tf.abs(scores_mapfn - scores_while) / (tf.abs(scores_mapfn) + 1e-12))

    print(f"✓ Both implementations run successfully")
    print(f"  Batch size: {batch_size}, K: {k_count}, N: {initial.shape[0]}, T: {noises.shape[0]}")
    print(f"  map_fn values: {values_mapfn.numpy()}")
    print(f"  while values:  {values_while.numpy()}")
    print(f"  Value relative error: {value_err.numpy():.6e}")
    print(f"  Score relative error: {score_err.numpy():.6e}")

    # Success criteria: rtol < 5e-4
    assert value_err < 5e-4, f"Value parity failed: {value_err.numpy():.6e} >= 5e-4"
    assert score_err < 5e-4, f"Score parity failed: {score_err.numpy():.6e} >= 5e-4"

    print(f"\n✓ PARITY PASSED: Both implementations match within rtol=5e-4")


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 1.5: TensorArray elimination parity test")
    print("=" * 60)
    print("\nTest 1: Baseline map_fn")
    print("-" * 60)
    test_batch_fused_mapfn_baseline()
    print("\nTest 2: map_fn vs while_loop parity")
    print("-" * 60)
    test_batch_fused_mapfn_vs_whileloop_parity()
