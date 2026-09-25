"""XLA compilation test for while_loop variant.

Verifies that the while_loop implementation compiles successfully under XLA
and measures the speedup vs eager and graph modes.
"""

from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import time
import numpy as np
import tensorflow as tf

from bayesfilter.highdim.ledh_canonical_batch_fused_tf import (
    PerPointScoreModel,
    canonical_batch_fused_value_score,
    canonical_batch_fused_value_score_whileloop,
)

DTYPE = tf.float64


def _fused_model():
    """Create a simple nonlinear model for testing."""
    def transition_mean_fn(theta, points):
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
    """Generate test fixture."""
    rng = np.random.default_rng(seed)
    initial = tf.constant(rng.standard_normal((n, 2)), DTYPE)
    covs = tf.constant(np.stack([np.eye(2)] * n), DTYPE)
    noises = tf.constant(rng.standard_normal((horizon, n, 2)), DTYPE)
    observations = tf.constant(rng.standard_normal((horizon, 2)), DTYPE)
    return initial, covs, noises, observations


def test_xla_compilation():
    """Test XLA compilation and measure speedup."""

    initial, covs, noises, observations = _fixture(seed=42, n=24, horizon=5)
    model = _fused_model()

    batch_size = 1  # Start with B=1 for simplicity
    k_count = 5
    param_dim = 1

    rng = np.random.default_rng(43)
    theta = tf.constant(rng.standard_normal((batch_size, param_dim)), DTYPE)
    directions = tf.constant(rng.standard_normal((batch_size, k_count, param_dim)), DTYPE)

    common_kwargs = dict(
        model=model,
        theta=theta,
        theta_directions=directions,
        initial_states=initial,
        initial_covariances=covs,
        noises=noises,
        observations=observations,
        substeps=2,
    )

    # 1. Eager mode (map_fn baseline)
    print("1. Eager mode (map_fn baseline)")
    start = time.perf_counter()
    for _ in range(5):
        values_eager, scores_eager, _ = canonical_batch_fused_value_score(
            k_batch_mode="sequential", **common_kwargs
        )
    eager_time = (time.perf_counter() - start) / 5
    print(f"   Time: {eager_time:.4f}s")

    # 2. Eager mode (while_loop)
    print("\n2. Eager mode (while_loop)")
    start = time.perf_counter()
    for _ in range(5):
        values_while_eager, scores_while_eager, _ = canonical_batch_fused_value_score_whileloop(
            **common_kwargs
        )
    while_eager_time = (time.perf_counter() - start) / 5
    print(f"   Time: {while_eager_time:.4f}s")
    print(f"   vs eager map_fn: {eager_time/while_eager_time:.2f}×")

    # 3. Graph mode (while_loop)
    print("\n3. Graph mode (while_loop)")

    # Bind model outside tf.function to avoid tracing issues
    @tf.function
    def graph_while(theta, directions, initial, covs, noises, observations):
        return canonical_batch_fused_value_score_whileloop(
            model=model,
            theta=theta,
            theta_directions=directions,
            initial_states=initial,
            initial_covariances=covs,
            noises=noises,
            observations=observations,
            substeps=2,
        )

    # Warmup
    _ = graph_while(theta, directions, initial, covs, noises, observations)

    start = time.perf_counter()
    for _ in range(5):
        values_graph, scores_graph, _ = graph_while(theta, directions, initial, covs, noises, observations)
    graph_time = (time.perf_counter() - start) / 5
    print(f"   Time: {graph_time:.4f}s")
    print(f"   vs eager map_fn: {eager_time/graph_time:.2f}×")

    # 4. XLA mode (while_loop) - THE KEY TEST
    print("\n4. XLA mode (while_loop) - KEY TEST")

    @tf.function(jit_compile=True)
    def xla_while(theta, directions, initial, covs, noises, observations):
        return canonical_batch_fused_value_score_whileloop(
            model=model,
            theta=theta,
            theta_directions=directions,
            initial_states=initial,
            initial_covariances=covs,
            noises=noises,
            observations=observations,
            substeps=2,
        )

    try:
        # Warmup - this will trigger XLA compilation
        print("   Compiling...")
        compile_start = time.perf_counter()
        _ = xla_while(theta, directions, initial, covs, noises, observations)
        compile_time = time.perf_counter() - compile_start
        print(f"   ✓ XLA compilation successful ({compile_time:.2f}s)")

        # Measure steady-state
        start = time.perf_counter()
        for _ in range(5):
            values_xla, scores_xla, _ = xla_while(theta, directions, initial, covs, noises, observations)
        xla_time = (time.perf_counter() - start) / 5
        print(f"   Time: {xla_time:.4f}s")
        print(f"   vs eager map_fn: {eager_time/xla_time:.2f}×")
        print(f"   vs graph while: {graph_time/xla_time:.2f}×")

        # Verify numerical parity
        value_err = tf.reduce_max(tf.abs(values_eager - values_xla) / (tf.abs(values_eager) + 1e-12))
        score_err = tf.reduce_max(tf.abs(scores_eager - scores_xla) / (tf.abs(scores_eager) + 1e-12))
        print(f"   Value parity: {value_err.numpy():.6e}")
        print(f"   Score parity: {score_err.numpy():.6e}")

        assert value_err < 1e-10, f"XLA value parity failed: {value_err.numpy()}"
        assert score_err < 1e-10, f"XLA score parity failed: {score_err.numpy()}"

        print(f"\n{'='*60}")
        print(f"✓ XLA COMPILATION SUCCESSFUL")
        print(f"✓ PHASE 1.5 COMPLETE - TensorArray eliminated")
        print(f"{'='*60}")

    except Exception as e:
        print(f"\n   ✗ XLA compilation FAILED:")
        print(f"   {type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 1.5: XLA compilation test")
    print("=" * 60)
    print()
    test_xla_compilation()
