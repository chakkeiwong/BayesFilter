import numpy as np
import tensorflow as tf

from bayesfilter import StatePartition
from bayesfilter.linear.kalman_svd_tf import tf_svd_kalman_log_likelihood
from bayesfilter.nonlinear.sigma_points_tf import tf_svd_sigma_point_log_likelihood
from bayesfilter.nonlinear.svd_cut_derivatives_tf import tf_svd_cut4_score_hessian
from bayesfilter.nonlinear.svd_cut_tf import tf_svd_cut4_log_likelihood
from bayesfilter.structural_tf import make_affine_structural_tf


def _linear_inputs():
    observations = tf.constant([[0.3, -0.1], [0.2, 0.05]], dtype=tf.float64)
    return {
        "observations": observations,
        "transition_offset": tf.constant([0.1], dtype=tf.float64),
        "transition_matrix": tf.constant([[0.8]], dtype=tf.float64),
        "transition_covariance": tf.constant([[0.05]], dtype=tf.float64),
        "observation_offset": tf.constant([0.0, 0.2], dtype=tf.float64),
        "observation_matrix": tf.constant([[1.0], [0.5]], dtype=tf.float64),
        "observation_covariance": tf.constant([[0.08, 0.01], [0.01, 0.10]], dtype=tf.float64),
        "initial_state_mean": tf.constant([0.0], dtype=tf.float64),
        "initial_state_covariance": tf.constant([[0.4]], dtype=tf.float64),
        "jitter": tf.constant(1e-9, dtype=tf.float64),
        "singular_floor": tf.constant(1e-12, dtype=tf.float64),
    }


def _structural_model(params: tf.Tensor | None = None):
    if params is None:
        phi1 = tf.constant(0.35, dtype=tf.float64)
        sigma = tf.constant(0.25, dtype=tf.float64)
    else:
        phi1 = params[0]
        sigma = params[1]
    partition = StatePartition(
        state_names=("m", "lag_m"),
        stochastic_indices=(0,),
        deterministic_indices=(1,),
        innovation_dim=1,
    )
    return make_affine_structural_tf(
        partition=partition,
        initial_mean=tf.constant([0.1, -0.2], dtype=tf.float64),
        initial_covariance=tf.linalg.diag(tf.constant([1.2, 0.7], dtype=tf.float64)),
        transition_offset=tf.zeros([2], dtype=tf.float64),
        transition_matrix=tf.stack(
            [
                tf.stack([phi1, tf.constant(-0.12, dtype=tf.float64)]),
                tf.constant([1.0, 0.0], dtype=tf.float64),
            ]
        ),
        innovation_matrix=tf.reshape(tf.stack([sigma, tf.constant(0.0, dtype=tf.float64)]), [2, 1]),
        innovation_covariance=tf.constant([[0.43]], dtype=tf.float64),
        observation_offset=tf.zeros([1], dtype=tf.float64),
        observation_matrix=tf.constant([[1.0, 0.25]], dtype=tf.float64),
        observation_covariance=tf.constant([[0.19]], dtype=tf.float64),
    )


def test_tf_svd_linear_value_cpu_compiled_parity() -> None:
    inputs = _linear_inputs()
    eager = tf_svd_kalman_log_likelihood(**inputs)[0]

    @tf.function(reduce_retracing=True)
    def compiled(observations: tf.Tensor) -> tf.Tensor:
        local = dict(inputs)
        local["observations"] = observations
        return tf_svd_kalman_log_likelihood(**local)[0]

    graph = compiled(inputs["observations"])

    np.testing.assert_allclose(graph.numpy(), eager.numpy(), atol=1e-12)
    assert len(compiled._list_all_concrete_functions_for_serialization()) == 1


def test_svd_sigma_point_value_cpu_compiled_parity() -> None:
    observations = tf.constant([[0.2], [-0.05], [0.15]], dtype=tf.float64)
    model = _structural_model()
    eager = tf_svd_sigma_point_log_likelihood(
        observations,
        model,
        rule="cubature",
        innovation_floor=tf.constant(1e-12, dtype=tf.float64),
    )[0]

    @tf.function(reduce_retracing=True)
    def compiled(obs: tf.Tensor) -> tf.Tensor:
        return tf_svd_sigma_point_log_likelihood(
            obs,
            model,
            rule="cubature",
            innovation_floor=tf.constant(1e-12, dtype=tf.float64),
        )[0]

    graph = compiled(observations)

    np.testing.assert_allclose(graph.numpy(), eager.numpy(), atol=1e-12)
    assert len(compiled._list_all_concrete_functions_for_serialization()) == 1


def test_svd_cut_value_cpu_compiled_parity() -> None:
    observations = tf.constant([[0.2], [-0.05], [0.15]], dtype=tf.float64)
    model = _structural_model()
    eager = tf_svd_cut4_log_likelihood(
        observations,
        model,
        innovation_floor=tf.constant(1e-12, dtype=tf.float64),
    )[0]

    @tf.function(reduce_retracing=True)
    def compiled(obs: tf.Tensor) -> tf.Tensor:
        return tf_svd_cut4_log_likelihood(
            obs,
            model,
            innovation_floor=tf.constant(1e-12, dtype=tf.float64),
        )[0]

    graph = compiled(observations)

    np.testing.assert_allclose(graph.numpy(), eager.numpy(), atol=1e-12)
    assert len(compiled._list_all_concrete_functions_for_serialization()) == 1


def test_svd_cut_derivative_cpu_compiled_parity() -> None:
    observations = tf.constant([[0.2], [-0.05], [0.15]], dtype=tf.float64)
    params = tf.constant([0.31, 0.27], dtype=tf.float64)
    eager = tf_svd_cut4_score_hessian(observations, params, _structural_model)

    @tf.function(reduce_retracing=True)
    def compiled(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        result = tf_svd_cut4_score_hessian(observations, theta, _structural_model)
        return result.log_likelihood, result.score, result.hessian

    graph_value, graph_score, graph_hessian = compiled(params)

    np.testing.assert_allclose(graph_value.numpy(), eager.log_likelihood.numpy(), atol=1e-12)
    np.testing.assert_allclose(graph_score.numpy(), eager.score.numpy(), atol=1e-12)
    np.testing.assert_allclose(graph_hessian.numpy(), eager.hessian.numpy(), atol=1e-12)
    assert len(compiled._list_all_concrete_functions_for_serialization()) == 1
