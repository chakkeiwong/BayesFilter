import numpy as np
import tensorflow as tf

from bayesfilter import StatePartition, affine_structural_to_linear_gaussian_tf
from bayesfilter.linear.kalman_tf import tf_linear_gaussian_log_likelihood
from bayesfilter.nonlinear.svd_cut_tf import (
    tf_svd_cut4_filter,
    tf_svd_cut4_log_likelihood,
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


def test_svd_cut4_matches_linear_value_on_affine_lgssm() -> None:
    structural = _affine_ar2_model()
    linear = affine_structural_to_linear_gaussian_tf(structural)
    observations = tf.constant([[0.2], [0.05], [-0.1], [0.15]], dtype=tf.float64)

    cut = tf_svd_cut4_filter(
        observations,
        structural,
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
        cut.log_likelihood.numpy(),
        linear_value.log_likelihood.numpy(),
        atol=1e-8,
    )
    assert cut.filtered_means.shape == (4, 2)
    assert cut.metadata.filter_name == "tf_svd_cut4_filter"
    assert cut.metadata.differentiability_status == "value_only"
    assert cut.diagnostics.regularization.derivative_target == "blocked"
    assert cut.diagnostics.extra["rule"] == "CUT4-G"
    assert int(cut.diagnostics.extra["point_count"].numpy()) == 14
    assert int(cut.diagnostics.extra["polynomial_degree"].numpy()) == 5


def test_svd_cut4_rank_deficient_support_and_point_count_diagnostics() -> None:
    structural = _affine_ar2_model(sigma=0.0, innovation_variance=0.0)
    observations = tf.constant([[0.2]], dtype=tf.float64)

    cut = tf_svd_cut4_filter(
        observations,
        structural,
        innovation_floor=tf.constant(1e-12, dtype=tf.float64),
    )

    assert int(cut.diagnostics.extra["point_count"].numpy()) == 14
    assert int(cut.diagnostics.extra["max_integration_rank"].numpy()) < 3
    np.testing.assert_allclose(
        cut.diagnostics.extra["support_residual"].numpy(),
        0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        cut.diagnostics.extra["deterministic_residual"].numpy(),
        0.0,
        atol=1e-12,
    )


def test_svd_cut4_static_shape_graph_reuse() -> None:
    structural = _affine_ar2_model()
    observations_a = tf.constant([[0.2], [0.05]], dtype=tf.float64)
    observations_b = tf.constant([[0.1], [0.0]], dtype=tf.float64)

    @tf.function(reduce_retracing=True)
    def compiled(observations: tf.Tensor) -> tf.Tensor:
        value, _means, _covs, _diagnostics = tf_svd_cut4_log_likelihood(
            observations,
            structural,
            innovation_floor=tf.constant(1e-12, dtype=tf.float64),
        )
        return value

    first = compiled(observations_a)
    second = compiled(observations_b)

    assert np.isfinite(first.numpy())
    assert np.isfinite(second.numpy())
    assert len(compiled._list_all_concrete_functions_for_serialization()) == 1
