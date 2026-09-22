from __future__ import annotations

import pytest
import tensorflow as tf

from bayesfilter.highdim.recursive_moment_map_tf import (
    affine_forward_inverse_residual,
    build_lagged_moment_map,
    cholesky_tangent,
    make_weighted_transition_moment_kernel,
    require_valid_moment_map,
    weighted_transition_moment_tangent,
)


DTYPE = tf.float64


def _linear_inputs() -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    weights = tf.fill([4], tf.constant(0.25, DTYPE))
    means = tf.constant(
        [[-0.8, 0.4], [-0.2, -0.5], [0.5, 0.1], [1.0, -0.3]], DTYPE
    )
    transition = tf.constant([[0.72, 0.08], [-0.04, 0.65]], DTYPE)
    offset = tf.constant([0.05, -0.1], DTYPE)
    process = tf.constant([[0.25, 0.03], [0.03, 0.18]], DTYPE)
    conditional_means = tf.einsum("ij,nj->ni", transition, means) + offset
    conditional_covariances = tf.broadcast_to(process[None, :, :], [4, 2, 2])
    return weights, means, conditional_means, conditional_covariances


def test_linear_law_of_total_covariance_and_map() -> None:
    weights, means, conditional_means, conditional_covariances = _linear_inputs()
    result = make_weighted_transition_moment_kernel(
        particle_count=4, state_dim=2, jit_compile=False
    )(weights, conditional_means, conditional_covariances)
    transition = tf.constant([[0.72, 0.08], [-0.04, 0.65]], DTYPE)
    offset = tf.constant([0.05, -0.1], DTYPE)
    process = tf.constant([[0.25, 0.03], [0.03, 0.18]], DTYPE)
    mean = tf.einsum("n,nd->d", weights, means)
    covariance = tf.einsum(
        "ij,jk,lk->il", transition,
        tf.einsum("n,ni,nj->ij", weights, means - mean, means - mean),
        transition,
    ) + process
    expected_mean = tf.einsum("ij,j->i", transition, mean) + offset
    tf.debugging.assert_near(result["predicted_mean"], expected_mean, atol=2e-13)
    tf.debugging.assert_near(result["predicted_covariance"], covariance, atol=2e-13)
    assert bool(result["valid"].numpy())
    built = build_lagged_moment_map(
        weights, conditional_means, conditional_covariances, jit_compile=False
    )
    assert bool(built["valid"].numpy())
    assert built["coordinate_map"].dimension == 2


def test_affine_map_round_trip_is_finite() -> None:
    weights, _, conditional_means, conditional_covariances = _linear_inputs()
    built = build_lagged_moment_map(
        weights, conditional_means, conditional_covariances, jit_compile=False
    )
    points = tf.constant([[0.0, 0.0], [1.0, -0.5], [-0.7, 0.2]], DTYPE)
    residual = affine_forward_inverse_residual(built["coordinate_map"], points)
    assert bool(residual["physical_finite"].numpy())
    assert bool(residual["reference_finite"].numpy())
    assert float(residual["forward_inverse_max_abs"].numpy()) <= 2e-14


def test_invalid_weight_and_covariance_are_rejected() -> None:
    weights, _, conditional_means, conditional_covariances = _linear_inputs()
    bad_weights = tf.tensor_scatter_nd_update(weights, [[0]], [0.3])
    result_weights = make_weighted_transition_moment_kernel(
        particle_count=4, state_dim=2, jit_compile=False
    )(bad_weights, conditional_means, conditional_covariances)
    assert not bool(result_weights["valid"].numpy())
    with pytest.raises(ValueError, match="invalid transition moments"):
        require_valid_moment_map(result_weights)

    bad_covariance = tf.tensor_scatter_nd_update(
        conditional_covariances,
        [[0, 0, 0]],
        [tf.constant(-1.0, DTYPE)],
    )
    result_covariance = make_weighted_transition_moment_kernel(
        particle_count=4, state_dim=2, jit_compile=False
    )(weights, conditional_means, bad_covariance)
    assert not bool(result_covariance["valid"].numpy())
    with pytest.raises(ValueError, match="invalid transition moments"):
        require_valid_moment_map(result_covariance)


def test_moment_tangent_matches_central_difference() -> None:
    weights, _, means, covariances = _linear_inputs()
    weight_tangent = tf.constant([0.1, -0.1, 0.0, 0.0], DTYPE)
    mean_tangent = tf.constant(
        [[0.2, -0.1], [0.0, 0.1], [0.1, 0.0], [-0.1, 0.2]], DTYPE
    )
    covariance_tangent = tf.broadcast_to(
        tf.constant([[0.03, 0.01], [0.01, -0.02]], DTYPE)[None, :, :], [4, 2, 2]
    )
    tangent = weighted_transition_moment_tangent(
        weights, weight_tangent, means, mean_tangent,
        covariances, covariance_tangent,
    )

    def value(eps: float) -> dict[str, tf.Tensor]:
        return weighted_transition_moment_tangent(
            weights + eps * weight_tangent,
            tf.zeros_like(weight_tangent),
            means + eps * mean_tangent,
            tf.zeros_like(mean_tangent),
            covariances + eps * covariance_tangent,
            tf.zeros_like(covariance_tangent),
        )

    eps = 1.0e-6
    plus = value(eps)
    minus = value(-eps)
    fd_mean = (plus["predicted_mean"] - minus["predicted_mean"]) / (2.0 * eps)
    fd_covariance = (
        plus["predicted_covariance"] - minus["predicted_covariance"]
    ) / (2.0 * eps)
    tf.debugging.assert_near(
        tangent["predicted_mean_tangent"], fd_mean, atol=2e-10, rtol=2e-10
    )
    tf.debugging.assert_near(
        tangent["predicted_covariance_tangent"], fd_covariance,
        atol=2e-10, rtol=2e-10,
    )


def test_cholesky_tangent_reconstructs_covariance_tangent() -> None:
    covariance = tf.constant([[2.0, 0.2], [0.2, 1.5]], DTYPE)
    tangent = tf.constant([[0.3, -0.1], [-0.1, 0.25]], DTYPE)
    result = cholesky_tangent(covariance, tangent)
    tf.debugging.assert_near(result["reconstruction_tangent"], tangent, atol=2e-12)


def test_xla_and_eager_kernel_parity() -> None:
    weights, _, conditional_means, conditional_covariances = _linear_inputs()
    eager = make_weighted_transition_moment_kernel(
        particle_count=4, state_dim=2, jit_compile=False
    )(weights, conditional_means, conditional_covariances)
    xla = make_weighted_transition_moment_kernel(
        particle_count=4, state_dim=2, jit_compile=True
    )(weights, conditional_means, conditional_covariances)
    for name in ("predicted_mean", "predicted_covariance", "predicted_cholesky"):
        tf.debugging.assert_near(eager[name], xla[name], atol=2e-12, rtol=2e-12)
    assert bool(eager["valid"].numpy()) == bool(xla["valid"].numpy())
