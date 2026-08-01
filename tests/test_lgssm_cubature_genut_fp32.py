from __future__ import annotations

import inspect

import numpy as np
import pytest
import tensorflow as tf

from docs.benchmarks.run_lgssm_cubature_genut_fp32 import (
    FD_MIN_STEP,
    FD_EPS,
    STATE_DIM,
    _central_difference_score,
    _kalman_value,
    _kalman_value_score,
    _particle_value_score_recursive,
    _restore_cloud_jvp,
    _sinkhorn_barycentric,
    _sinkhorn_barycentric_jvp,
    _particle_value,
    cubature_design,
    contract_e_gaussian_design,
    genut_gaussian_design,
)
from docs.benchmarks.tune_lgssm_cubature_genut_fp32 import _representative_points


def _moments(points: tf.Tensor) -> tuple[np.ndarray, np.ndarray]:
    mean = tf.reduce_mean(points, axis=0)
    centered = points - mean[None, :]
    covariance = tf.transpose(centered) @ centered / tf.cast(tf.shape(points)[0], tf.float32)
    return mean.numpy(), covariance.numpy()


def test_cubature_design_has_zero_mean_and_identity_covariance() -> None:
    points = cubature_design(dim=STATE_DIM, num_particles=1008)
    mean, covariance = _moments(points)
    np.testing.assert_allclose(mean, np.zeros(STATE_DIM), atol=1e-6)
    np.testing.assert_allclose(covariance, np.eye(STATE_DIM), atol=1e-6)


def test_gaussian_genut_design_is_cubature_design() -> None:
    cubature = cubature_design(dim=STATE_DIM, num_particles=1008)
    genut, metadata = genut_gaussian_design(dim=STATE_DIM, num_particles=1008)
    np.testing.assert_array_equal(cubature.numpy(), genut.numpy())
    assert metadata["central_weight"] == 0.0
    assert metadata["central_point_omitted"] is True


def test_design_requires_divisibility() -> None:
    with pytest.raises(ValueError, match="divisible"):
        cubature_design(dim=STATE_DIM, num_particles=1000)
    with pytest.raises(ValueError, match="divisibility"):
        genut_gaussian_design(dim=STATE_DIM, num_particles=1000)


def test_sinkhorn_shape_and_marginals_on_small_fixture() -> None:
    particles = tf.reshape(tf.range(12, dtype=tf.float32), [4, 3])
    weights = tf.constant([0.1, 0.2, 0.3, 0.4], tf.float32)
    barycentric, row_residual, col_residual = _sinkhorn_barycentric(particles, weights)
    assert barycentric.shape == particles.shape
    assert float(row_residual.numpy()) < 0.02
    assert float(col_residual.numpy()) < 0.02


def test_particle_route_returns_finite_equal_weight_reset_path() -> None:
    design = cubature_design(dim=STATE_DIM, num_particles=12)
    theta = tf.constant([0.72, 0.55, 0.35, 0.35, 0.45], tf.float32)
    observations = tf.constant([[0.1, -0.2, 0.3], [0.2, 0.4, -0.1]], tf.float32)
    initial_noise = tf.random.stateless_normal([12, STATE_DIM], seed=[3, 4])
    process_noise = tf.random.stateless_normal([2, 12, STATE_DIM], seed=[5, 6])
    value, mean_cov, row, col = _particle_value(
        theta, observations, initial_noise, process_noise, design
    )
    assert bool(tf.reduce_all(tf.math.is_finite([value, mean_cov, row, col])).numpy())


def test_central_difference_uses_value_only_callable() -> None:
    def value_only(theta: tf.Tensor, _observations: tf.Tensor) -> dict[str, tf.Tensor]:
        return {"value": tf.reduce_sum(theta * theta)}

    theta = tf.constant([0.4, -0.7, 0.2], tf.float32)
    score, steps = _central_difference_score(
        value_only, theta, tf.zeros([1], tf.float32)
    )
    np.testing.assert_allclose(score.numpy(), 2.0 * theta.numpy(), rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(
        steps.numpy(), np.maximum(FD_MIN_STEP, FD_EPS * np.abs(theta.numpy()))
    )


def test_representative_points_are_deterministic_and_in_domain() -> None:
    first, metadata_first = _representative_points()
    second, metadata_second = _representative_points()
    np.testing.assert_array_equal(first.numpy(), second.numpy())
    assert metadata_first == metadata_second
    values = first.numpy()
    assert np.all(values[:, :STATE_DIM] >= 0.25)
    assert np.all(values[:, :STATE_DIM] <= 0.85)
    assert np.all(values[:, STATE_DIM:] >= 0.25)
    assert np.all(values[:, STATE_DIM:] <= 0.65)


def test_recursive_score_matches_same_value_central_difference() -> None:
    particle_count = 12
    theta = tf.constant([0.72, 0.55, 0.35, 0.35, 0.45], tf.float32)
    observations = tf.constant([[0.1, -0.2, 0.3], [0.2, 0.4, -0.1]], tf.float32)
    initial_noise = tf.random.stateless_normal(
        [particle_count, STATE_DIM], seed=[3, 4]
    )
    process_noise = tf.random.stateless_normal(
        [2, particle_count, STATE_DIM], seed=[5, 6]
    )
    design = cubature_design(dim=STATE_DIM, num_particles=particle_count)
    value, score, *_ = _particle_value_score_recursive(
        theta, observations, initial_noise, process_noise, design
    )

    def value_only(
        current_theta: tf.Tensor, current_observations: tf.Tensor
    ) -> dict[str, tf.Tensor]:
        current_value, *_ = _particle_value(
            current_theta,
            current_observations,
            initial_noise,
            process_noise,
            design,
        )
        return {"value": current_value}

    finite_difference, _ = _central_difference_score(
        value_only, theta, observations
    )
    tf.debugging.assert_near(value, value_only(theta, observations)["value"], atol=1e-6)
    tf.debugging.assert_near(score, finite_difference, atol=3e-4, rtol=3e-4)


def test_recursive_score_route_contains_no_tensorflow_autodiff() -> None:
    source = "\n".join(
        inspect.getsource(function)
        for function in (
            _particle_value_score_recursive,
            _sinkhorn_barycentric_jvp,
            _restore_cloud_jvp,
        )
    )
    assert "GradientTape" not in source
    assert "ForwardAccumulator" not in source
    assert "tf.autodiff" not in source


def test_analytic_kalman_score_matches_value_and_central_difference() -> None:
    theta = tf.constant([0.72, 0.55, 0.35, 0.35, 0.45], tf.float32)
    observations = tf.constant([[0.1, -0.2, 0.3], [0.2, 0.4, -0.1]], tf.float32)
    value, score = _kalman_value_score(theta, observations)
    expected_value = _kalman_value(theta, observations)

    def value_only(
        current_theta: tf.Tensor, current_observations: tf.Tensor
    ) -> dict[str, tf.Tensor]:
        return {"value": _kalman_value(current_theta, current_observations)}

    finite_difference, _ = _central_difference_score(
        value_only,
        tf.cast(theta, tf.float64),
        tf.cast(observations, tf.float64),
        relative_step=1e-5,
        minimum_step=1e-6,
    )
    tf.debugging.assert_near(value, expected_value, atol=1e-12, rtol=1e-12)
    tf.debugging.assert_near(score, finite_difference, atol=1e-7, rtol=1e-7)


def test_contract_e_gaussian_design_is_centered_and_time_varying() -> None:
    design, metadata = contract_e_gaussian_design(
        horizon=3, num_particles=12, particle_seed=17
    )
    tf.debugging.assert_near(
        tf.reduce_mean(design, axis=1), tf.zeros([3, STATE_DIM]), atol=1e-6
    )
    assert design.shape == (3, 12, STATE_DIM)
    assert not bool(tf.reduce_all(design[0] == design[1]).numpy())
    assert metadata["residual_design_id"] == (
        "contract_e_residual_centered_population_scaled_v1"
    )
