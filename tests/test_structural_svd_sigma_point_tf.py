import numpy as np
import tensorflow as tf

from bayesfilter import StatePartition, affine_structural_to_linear_gaussian_tf
from bayesfilter.linear.kalman_tf import tf_linear_gaussian_log_likelihood
from bayesfilter.nonlinear.sigma_points_tf import (
    tf_svd_sigma_point_filter,
    tf_svd_sigma_point_log_likelihood,
    tf_svd_sigma_point_placement,
    tf_unit_sigma_point_rule,
)
from bayesfilter.structural_tf import make_affine_structural_tf


def _affine_ar2_model(*, sigma: float = 0.25, innovation_variance: float = 1.0):
    partition = StatePartition(
        state_names=("m", "lag_m"),
        stochastic_indices=(0,),
        deterministic_indices=(1,),
        innovation_dim=1,
    )
    return make_affine_structural_tf(
        partition=partition,
        initial_mean=tf.zeros([2], dtype=tf.float64),
        initial_covariance=tf.eye(2, dtype=tf.float64),
        transition_offset=tf.zeros([2], dtype=tf.float64),
        transition_matrix=tf.constant([[0.35, -0.10], [1.0, 0.0]], dtype=tf.float64),
        innovation_matrix=tf.constant([[sigma], [0.0]], dtype=tf.float64),
        innovation_covariance=tf.constant([[innovation_variance]], dtype=tf.float64),
        observation_offset=tf.zeros([1], dtype=tf.float64),
        observation_matrix=tf.constant([[1.0, 0.0]], dtype=tf.float64),
        observation_covariance=tf.constant([[0.15**2]], dtype=tf.float64),
    )


def test_structural_svd_cubature_matches_linear_qr_on_affine_lgssm() -> None:
    structural = _affine_ar2_model()
    linear = affine_structural_to_linear_gaussian_tf(structural)
    observations = tf.constant([[0.2], [0.05], [-0.1], [0.15], [0.0]], dtype=tf.float64)

    sigma_point = tf_svd_sigma_point_filter(
        observations,
        structural,
        backend="tf_svd_cubature",
        innovation_floor=tf.constant(1e-12, dtype=tf.float64),
        return_filtered=True,
    )
    linear_value = tf_linear_gaussian_log_likelihood(
        observations,
        linear,
        backend="tf_cholesky",
        jitter=tf.constant(0.0, dtype=tf.float64),
    )

    np.testing.assert_allclose(
        sigma_point.log_likelihood.numpy(),
        linear_value.log_likelihood.numpy(),
        atol=1e-8,
    )
    assert sigma_point.filtered_means.shape == (5, 2)
    assert sigma_point.metadata.filter_name == "tf_svd_cubature_filter"
    assert sigma_point.metadata.differentiability_status == "value_only"
    assert sigma_point.diagnostics.regularization.derivative_target == "blocked"
    assert sigma_point.diagnostics.extra["rule"] == "cubature"
    np.testing.assert_allclose(
        sigma_point.diagnostics.extra["deterministic_residual"].numpy(),
        0.0,
        atol=1e-12,
    )


def test_structural_svd_ukf_matches_linear_qr_on_affine_lgssm() -> None:
    structural = _affine_ar2_model()
    linear = affine_structural_to_linear_gaussian_tf(structural)
    observations = tf.constant([[0.2], [0.05], [-0.1]], dtype=tf.float64)

    sigma_point = tf_svd_sigma_point_filter(
        observations,
        structural,
        backend="tf_svd_ukf",
        innovation_floor=tf.constant(1e-12, dtype=tf.float64),
    )
    linear_value = tf_linear_gaussian_log_likelihood(
        observations,
        linear,
        backend="tf_cholesky",
        jitter=tf.constant(0.0, dtype=tf.float64),
    )

    np.testing.assert_allclose(
        sigma_point.log_likelihood.numpy(),
        linear_value.log_likelihood.numpy(),
        atol=1e-8,
    )
    assert sigma_point.diagnostics.extra["rule"] == "unscented"


def test_structural_svd_sigma_points_preserve_deterministic_completion_support() -> None:
    structural = _affine_ar2_model(sigma=0.0, innovation_variance=0.0)
    aug_mean = tf.concat(
        [structural.initial_mean, tf.zeros([structural.partition.innovation_dim], dtype=tf.float64)],
        axis=0,
    )
    aug_covariance = tf.linalg.diag(tf.constant([1.0, 1.0, 1.0], dtype=tf.float64))
    rule = tf_unit_sigma_point_rule(3, rule="cubature")

    aug_points, diagnostics = tf_svd_sigma_point_placement(
        aug_mean,
        aug_covariance,
        rule,
        singular_floor=tf.constant(0.0, dtype=tf.float64),
    )
    previous_points = aug_points[:, :2]
    innovation_points = aug_points[:, 2:]
    next_points = structural.transition(previous_points, innovation_points)
    residuals = structural.deterministic_residual(previous_points, innovation_points, next_points)

    assert diagnostics.rank.shape == ()
    np.testing.assert_allclose(residuals.numpy(), np.zeros([6, 1]), atol=1e-14)
    np.testing.assert_allclose(next_points.numpy()[:, 1], previous_points.numpy()[:, 0], atol=1e-14)


def test_structural_svd_sigma_point_static_shape_graph_reuse() -> None:
    structural = _affine_ar2_model()
    observations_a = tf.constant([[0.2], [0.05], [-0.1]], dtype=tf.float64)
    observations_b = tf.constant([[0.1], [0.0], [0.2]], dtype=tf.float64)

    @tf.function(reduce_retracing=True)
    def compiled(observations: tf.Tensor) -> tf.Tensor:
        value, _means, _covs, _diagnostics = tf_svd_sigma_point_log_likelihood(
            observations,
            structural,
            rule="cubature",
            innovation_floor=tf.constant(1e-12, dtype=tf.float64),
        )
        return value

    first = compiled(observations_a)
    second = compiled(observations_b)

    assert np.isfinite(first.numpy())
    assert np.isfinite(second.numpy())
    assert len(compiled._list_all_concrete_functions_for_serialization()) == 1


def test_structural_svd_sigma_point_reports_rank_deficient_placement() -> None:
    structural = _affine_ar2_model(sigma=0.0, innovation_variance=0.0)
    observations = tf.constant([[0.2]], dtype=tf.float64)

    result = tf_svd_sigma_point_filter(
        observations,
        structural,
        backend="tf_svd_cubature",
        innovation_floor=tf.constant(1e-12, dtype=tf.float64),
    )

    assert int(result.diagnostics.extra["max_integration_rank"].numpy()) < 3
    np.testing.assert_allclose(
        result.diagnostics.extra["support_residual"].numpy(),
        0.0,
        atol=1e-12,
    )
