import numpy as np
import tensorflow as tf

from bayesfilter.testing import (
    dense_projection_first_step,
    make_affine_gaussian_structural_oracle_tf,
    make_nonlinear_accumulation_model_tf,
    make_univariate_nonlinear_growth_model_tf,
)


def test_dense_projection_first_step_is_finite_for_model_b() -> None:
    model = make_nonlinear_accumulation_model_tf()
    reference = dense_projection_first_step(
        model,
        tf.constant([0.10], dtype=tf.float64),
        nodes_per_dim=9,
    )

    assert np.isfinite(reference.log_likelihood.numpy())
    assert reference.predicted_mean.shape == (2,)
    assert reference.predicted_covariance.shape == (2, 2)
    assert reference.observation_covariance.shape == (1, 1)
    np.testing.assert_allclose(reference.deterministic_residual.numpy(), 0.0, atol=1e-12)


def test_dense_projection_first_step_is_finite_for_model_c() -> None:
    model = make_univariate_nonlinear_growth_model_tf()
    reference = dense_projection_first_step(
        model,
        tf.constant([0.40], dtype=tf.float64),
        nodes_per_dim=9,
    )

    assert np.isfinite(reference.log_likelihood.numpy())
    assert reference.predicted_mean.shape == (2,)
    assert reference.filtered_covariance.shape == (2, 2)
    np.testing.assert_allclose(reference.deterministic_residual.numpy(), 0.0, atol=1e-12)


def test_dense_projection_matches_linear_oracle_first_step_moments_for_model_a() -> None:
    model = make_affine_gaussian_structural_oracle_tf()
    reference = dense_projection_first_step(
        model,
        tf.constant([0.20], dtype=tf.float64),
        nodes_per_dim=5,
    )

    transition_covariance = (
        model.innovation_matrix
        @ model.innovation_covariance
        @ tf.transpose(model.innovation_matrix)
    )
    expected_mean = model.transition_offset + tf.linalg.matvec(
        model.transition_matrix,
        model.initial_mean,
    )
    expected_covariance = (
        model.transition_matrix
        @ model.initial_covariance
        @ tf.transpose(model.transition_matrix)
        + transition_covariance
    )
    expected_observation_mean = model.observation_offset + tf.linalg.matvec(
        model.observation_matrix,
        expected_mean,
    )
    expected_observation_covariance = (
        model.observation_matrix
        @ expected_covariance
        @ tf.transpose(model.observation_matrix)
        + model.observation_covariance
    )

    np.testing.assert_allclose(reference.predicted_mean.numpy(), expected_mean.numpy(), atol=1e-12)
    np.testing.assert_allclose(
        reference.predicted_covariance.numpy(),
        expected_covariance.numpy(),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        reference.observation_mean.numpy(),
        expected_observation_mean.numpy(),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        reference.observation_covariance.numpy(),
        expected_observation_covariance.numpy(),
        atol=1e-12,
    )
