import numpy as np
import tensorflow as tf

from bayesfilter import affine_structural_to_linear_gaussian_tf
from bayesfilter.linear.kalman_tf import tf_linear_gaussian_log_likelihood
from bayesfilter.nonlinear.svd_cut_tf import tf_svd_cut4_filter
from bayesfilter.nonlinear.cut_tf import tf_cut4g_sigma_point_rule
from bayesfilter.nonlinear.sigma_points_tf import tf_svd_sigma_point_filter
from bayesfilter.testing import (
    dense_projection_first_step,
    make_affine_gaussian_structural_oracle_tf,
    make_nonlinear_accumulation_model_tf,
    make_univariate_nonlinear_growth_model_tf,
    model_a_observations_tf,
    model_b_observations_tf,
    model_c_observations_tf,
    sigma_point_projection_first_step,
)


def test_model_a_value_filters_match_exact_kalman_reference() -> None:
    model = make_affine_gaussian_structural_oracle_tf()
    observations = model_a_observations_tf()
    linear = affine_structural_to_linear_gaussian_tf(model)
    exact = tf_linear_gaussian_log_likelihood(
        observations,
        linear,
        backend="tf_cholesky",
        jitter=tf.constant(0.0, dtype=tf.float64),
    )

    cubature = tf_svd_sigma_point_filter(
        observations,
        model,
        backend="tf_svd_cubature",
        innovation_floor=tf.constant(1e-12, dtype=tf.float64),
    )
    ukf = tf_svd_sigma_point_filter(
        observations,
        model,
        backend="tf_svd_ukf",
        innovation_floor=tf.constant(1e-12, dtype=tf.float64),
    )
    cut4 = tf_svd_cut4_filter(
        observations,
        model,
        innovation_floor=tf.constant(1e-12, dtype=tf.float64),
    )

    for result in (cubature, ukf, cut4):
        np.testing.assert_allclose(
            result.log_likelihood.numpy(),
            exact.log_likelihood.numpy(),
            atol=1e-8,
        )
        np.testing.assert_allclose(
            result.diagnostics.extra["deterministic_residual"].numpy(),
            0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            result.diagnostics.extra["support_residual"].numpy(),
            0.0,
            atol=1e-12,
        )


def test_model_b_value_filters_are_finite_and_report_residuals() -> None:
    model = make_nonlinear_accumulation_model_tf()
    observations = model_b_observations_tf()
    results = [
        tf_svd_sigma_point_filter(observations, model, backend="tf_svd_cubature"),
        tf_svd_sigma_point_filter(observations, model, backend="tf_svd_ukf"),
        tf_svd_cut4_filter(observations, model),
    ]

    for result in results:
        assert np.isfinite(result.log_likelihood.numpy())
        np.testing.assert_allclose(
            result.diagnostics.extra["deterministic_residual"].numpy(),
            0.0,
            atol=1e-12,
        )
        assert result.diagnostics.extra["point_count"].numpy() > 0


def test_model_c_value_filters_are_finite_and_report_phase_residuals() -> None:
    model = make_univariate_nonlinear_growth_model_tf()
    observations = model_c_observations_tf()
    results = [
        tf_svd_sigma_point_filter(observations, model, backend="tf_svd_cubature"),
        tf_svd_sigma_point_filter(observations, model, backend="tf_svd_ukf"),
        tf_svd_cut4_filter(observations, model),
    ]

    for result in results:
        assert np.isfinite(result.log_likelihood.numpy())
        np.testing.assert_allclose(
            result.diagnostics.extra["deterministic_residual"].numpy(),
            0.0,
            atol=1e-12,
        )


def test_model_b_first_step_sigma_point_observation_moments_near_dense_projection() -> None:
    model = make_nonlinear_accumulation_model_tf()
    observation = model_b_observations_tf()[0]
    dense = dense_projection_first_step(model, observation, nodes_per_dim=11)
    cut4_projection = sigma_point_projection_first_step(
        model,
        observation,
        sigma_rule=tf_cut4g_sigma_point_rule(
            model.partition.state_dim + model.partition.innovation_dim
        ),
    )

    np.testing.assert_allclose(
        cut4_projection.observation_mean.numpy(),
        dense.observation_mean.numpy(),
        atol=2e-3,
        rtol=2e-3,
    )
    np.testing.assert_allclose(
        cut4_projection.observation_covariance.numpy(),
        dense.observation_covariance.numpy(),
        atol=5e-3,
        rtol=5e-3,
    )
    np.testing.assert_allclose(
        cut4_projection.deterministic_residual.numpy(),
        0.0,
        atol=1e-12,
    )


def test_model_value_filter_graph_parity_for_model_b() -> None:
    model = make_nonlinear_accumulation_model_tf()
    observations = model_b_observations_tf()

    eager = tf_svd_sigma_point_filter(
        observations,
        model,
        backend="tf_svd_cubature",
    ).log_likelihood

    @tf.function(reduce_retracing=True)
    def compiled(obs: tf.Tensor) -> tf.Tensor:
        return tf_svd_sigma_point_filter(
            obs,
            model,
            backend="tf_svd_cubature",
        ).log_likelihood

    graph = compiled(observations)
    np.testing.assert_allclose(graph.numpy(), eager.numpy(), atol=1e-12)
