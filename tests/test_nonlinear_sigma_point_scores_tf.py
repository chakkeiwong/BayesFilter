import numpy as np
import pytest
import tensorflow as tf

from bayesfilter import StatePartition
from bayesfilter.nonlinear.svd_cut_tf import tf_svd_cut4_log_likelihood
from bayesfilter.nonlinear.svd_sigma_point_derivatives_tf import (
    TFStructuralFirstDerivatives,
    tf_svd_cubature_score,
    tf_svd_cut4_score,
    tf_svd_ukf_score,
)
from bayesfilter.nonlinear.sigma_points_tf import tf_svd_sigma_point_log_likelihood
from bayesfilter.structural_tf import make_affine_structural_tf
from bayesfilter.testing import (
    make_nonlinear_accumulation_first_derivatives_tf,
    make_nonlinear_accumulation_model_tf,
    make_univariate_nonlinear_growth_first_derivatives_tf,
    make_univariate_nonlinear_growth_model_tf,
    model_b_observations_tf,
    model_c_observations_tf,
)
from bayesfilter.testing.tf_svd_cut_autodiff_oracle import (
    tf_svd_cut4_score_hessian_autodiff_oracle,
)


OBSERVATIONS = tf.constant([[0.2], [-0.05], [0.15]], dtype=tf.float64)


def _smooth_affine_model_and_derivatives(params: tf.Tensor):
    phi = params[0]
    sigma = params[1]
    obs_scale = params[2]
    partition = StatePartition(
        state_names=("x", "lag_x"),
        stochastic_indices=(0,),
        deterministic_indices=(1,),
        innovation_dim=1,
    )
    transition_matrix = tf.stack(
        [
            tf.stack([phi, tf.constant(-0.12, dtype=tf.float64)]),
            tf.constant([1.0, 0.0], dtype=tf.float64),
        ]
    )
    innovation_matrix = tf.reshape(
        tf.stack([sigma, tf.constant(0.0, dtype=tf.float64)]),
        [2, 1],
    )
    observation_matrix = tf.reshape(tf.stack([obs_scale, 0.25]), [1, 2])
    model = make_affine_structural_tf(
        partition=partition,
        initial_mean=tf.constant([0.1, -0.2], dtype=tf.float64),
        initial_covariance=tf.linalg.diag(tf.constant([1.2, 0.7], dtype=tf.float64)),
        transition_offset=tf.zeros([2], dtype=tf.float64),
        transition_matrix=transition_matrix,
        innovation_matrix=innovation_matrix,
        innovation_covariance=tf.constant([[0.43]], dtype=tf.float64),
        observation_offset=tf.zeros([1], dtype=tf.float64),
        observation_matrix=observation_matrix,
        observation_covariance=tf.constant([[0.19]], dtype=tf.float64),
    )
    p = 3
    state_dim = 2
    innovation_dim = 1
    observation_dim = 1
    d_transition_matrix = tf.zeros([p, state_dim, state_dim], dtype=tf.float64)
    d_transition_matrix = tf.tensor_scatter_nd_update(
        d_transition_matrix,
        [[0, 0, 0]],
        [tf.constant(1.0, dtype=tf.float64)],
    )
    d_innovation_matrix = tf.zeros([p, state_dim, innovation_dim], dtype=tf.float64)
    d_innovation_matrix = tf.tensor_scatter_nd_update(
        d_innovation_matrix,
        [[1, 0, 0]],
        [tf.constant(1.0, dtype=tf.float64)],
    )
    d_observation_matrix = tf.zeros([p, observation_dim, state_dim], dtype=tf.float64)
    d_observation_matrix = tf.tensor_scatter_nd_update(
        d_observation_matrix,
        [[2, 0, 0]],
        [tf.constant(1.0, dtype=tf.float64)],
    )

    def transition_state_jacobian(previous: tf.Tensor, innovation: tf.Tensor) -> tf.Tensor:
        del innovation
        point_count = tf.shape(previous)[0]
        return tf.broadcast_to(transition_matrix[tf.newaxis, :, :], [point_count, 2, 2])

    def transition_innovation_jacobian(
        previous: tf.Tensor,
        innovation: tf.Tensor,
    ) -> tf.Tensor:
        del previous
        point_count = tf.shape(innovation)[0]
        return tf.broadcast_to(innovation_matrix[tf.newaxis, :, :], [point_count, 2, 1])

    def d_transition(previous: tf.Tensor, innovation: tf.Tensor) -> tf.Tensor:
        return (
            tf.einsum("pij,rj->pri", d_transition_matrix, previous)
            + tf.einsum("piq,rq->pri", d_innovation_matrix, innovation)
        )

    def observation_state_jacobian(states: tf.Tensor) -> tf.Tensor:
        point_count = tf.shape(states)[0]
        return tf.broadcast_to(observation_matrix[tf.newaxis, :, :], [point_count, 1, 2])

    def d_observation(states: tf.Tensor) -> tf.Tensor:
        return tf.einsum("pmj,rj->prm", d_observation_matrix, states)

    derivatives = TFStructuralFirstDerivatives(
        d_initial_mean=tf.zeros([p, state_dim], dtype=tf.float64),
        d_initial_covariance=tf.zeros([p, state_dim, state_dim], dtype=tf.float64),
        d_innovation_covariance=tf.zeros([p, innovation_dim, innovation_dim], dtype=tf.float64),
        d_observation_covariance=tf.zeros([p, observation_dim, observation_dim], dtype=tf.float64),
        transition_state_jacobian_fn=transition_state_jacobian,
        transition_innovation_jacobian_fn=transition_innovation_jacobian,
        d_transition_fn=d_transition,
        observation_state_jacobian_fn=observation_state_jacobian,
        d_observation_fn=d_observation,
        name="smooth_affine_test_derivatives",
    )
    return model, derivatives


def _sigma_value(params: tf.Tensor, rule: str) -> tf.Tensor:
    model, _derivatives = _smooth_affine_model_and_derivatives(params)
    value, _means, _covariances, _diagnostics = tf_svd_sigma_point_log_likelihood(
        OBSERVATIONS,
        model,
        rule=rule,
        innovation_floor=tf.constant(1e-12, dtype=tf.float64),
    )
    return value


def _cut_value(params: tf.Tensor) -> tf.Tensor:
    model, _derivatives = _smooth_affine_model_and_derivatives(params)
    value, _means, _covariances, _diagnostics = tf_svd_cut4_log_likelihood(
        OBSERVATIONS,
        model,
        innovation_floor=tf.constant(1e-12, dtype=tf.float64),
    )
    return value


def _model_b_and_derivatives(params: tf.Tensor):
    model = make_nonlinear_accumulation_model_tf(
        rho=params[0],
        sigma=params[1],
        beta=params[2],
    )
    derivatives = make_nonlinear_accumulation_first_derivatives_tf(
        rho=params[0],
        sigma=params[1],
        beta=params[2],
    )
    return model, derivatives


def _model_b_value(params: tf.Tensor, backend: str) -> tf.Tensor:
    model, _derivatives = _model_b_and_derivatives(params)
    if backend == "tf_svd_cut4":
        value, _means, _covariances, _diagnostics = tf_svd_cut4_log_likelihood(
            model_b_observations_tf(),
            model,
            innovation_floor=tf.constant(1e-12, dtype=tf.float64),
        )
        return value
    rule = "cubature" if backend == "tf_svd_cubature" else "unscented"
    value, _means, _covariances, _diagnostics = tf_svd_sigma_point_log_likelihood(
        model_b_observations_tf(),
        model,
        rule=rule,
        innovation_floor=tf.constant(1e-12, dtype=tf.float64),
    )
    return value


def _model_c_and_derivatives(params: tf.Tensor, *, phase_variance: tf.Tensor | float):
    model = make_univariate_nonlinear_growth_model_tf(
        process_sigma=params[0],
        observation_sigma=params[1],
        initial_variance=params[2],
        initial_phase_variance=phase_variance,
    )
    derivatives = make_univariate_nonlinear_growth_first_derivatives_tf(
        process_sigma=params[0],
        observation_sigma=params[1],
    )
    return model, derivatives


def _model_c_value(
    params: tf.Tensor,
    backend: str,
    *,
    phase_variance: tf.Tensor | float,
) -> tf.Tensor:
    model, _derivatives = _model_c_and_derivatives(
        params,
        phase_variance=phase_variance,
    )
    if backend == "tf_svd_cut4":
        value, _means, _covariances, _diagnostics = tf_svd_cut4_log_likelihood(
            model_c_observations_tf(),
            model,
            innovation_floor=tf.constant(1e-12, dtype=tf.float64),
        )
        return value
    rule = "cubature" if backend == "tf_svd_cubature" else "unscented"
    value, _means, _covariances, _diagnostics = tf_svd_sigma_point_log_likelihood(
        model_c_observations_tf(),
        model,
        rule=rule,
        innovation_floor=tf.constant(1e-12, dtype=tf.float64),
    )
    return value


def _model_c_structural_fixed_support_value(params: tf.Tensor, backend: str) -> tf.Tensor:
    return _model_c_value(
        params,
        backend,
        phase_variance=tf.constant(0.0, dtype=tf.float64),
    )


def _finite_difference_score(value_fn, theta: np.ndarray, step: float = 1e-5):
    theta = np.asarray(theta, dtype=np.float64)
    score = np.zeros(theta.size, dtype=np.float64)
    for i in range(theta.size):
        direction = np.zeros(theta.size, dtype=np.float64)
        direction[i] = step
        score[i] = (
            float(value_fn(tf.constant(theta + direction, dtype=tf.float64)).numpy())
            - float(value_fn(tf.constant(theta - direction, dtype=tf.float64)).numpy())
        ) / (2.0 * step)
    return score


@pytest.mark.parametrize(
    ("score_fn", "rule", "filter_name"),
    [
        (tf_svd_cubature_score, "cubature", "tf_svd_cubature_score"),
        (tf_svd_ukf_score, "unscented", "tf_svd_ukf_score"),
    ],
)
def test_svd_sigma_point_analytic_score_matches_finite_difference(
    score_fn,
    rule,
    filter_name,
) -> None:
    params = tf.constant([0.31, 0.27, 1.05], dtype=tf.float64)
    model, derivatives = _smooth_affine_model_and_derivatives(params)
    result = score_fn(
        OBSERVATIONS,
        model,
        derivatives,
        innovation_floor=tf.constant(1e-12, dtype=tf.float64),
        spectral_gap_tolerance=tf.constant(1e-7, dtype=tf.float64),
    )
    finite_difference = _finite_difference_score(
        lambda values: _sigma_value(values, rule),
        params.numpy(),
    )

    np.testing.assert_allclose(result.score.numpy(), finite_difference, rtol=2e-4, atol=2e-4)
    assert result.hessian is None
    assert result.metadata.filter_name == filter_name
    assert (
        result.metadata.differentiability_status
        == "analytic_score_smooth_branch_hessian_deferred"
    )
    assert result.diagnostics.extra["derivative_method"] == "analytic_first_order_smooth_branch"
    np.testing.assert_allclose(
        result.diagnostics.extra["factor_derivative_reconstruction_residual"].numpy(),
        0.0,
        atol=1e-9,
    )


def test_svd_cut4_analytic_score_matches_finite_difference_and_oracle_score() -> None:
    params = tf.constant([0.31, 0.27, 1.05], dtype=tf.float64)
    model, derivatives = _smooth_affine_model_and_derivatives(params)
    analytic = tf_svd_cut4_score(
        OBSERVATIONS,
        model,
        derivatives,
        innovation_floor=tf.constant(1e-12, dtype=tf.float64),
        spectral_gap_tolerance=tf.constant(1e-7, dtype=tf.float64),
    )
    finite_difference = _finite_difference_score(_cut_value, params.numpy())
    oracle = tf_svd_cut4_score_hessian_autodiff_oracle(
        OBSERVATIONS,
        params,
        lambda values: _smooth_affine_model_and_derivatives(values)[0],
        innovation_floor=tf.constant(1e-12, dtype=tf.float64),
        spectral_gap_tolerance=tf.constant(1e-7, dtype=tf.float64),
    )

    np.testing.assert_allclose(analytic.score.numpy(), finite_difference, rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(analytic.score.numpy(), oracle.score.numpy(), atol=1e-9)
    assert analytic.hessian is None
    assert analytic.metadata.filter_name == "tf_svd_cut4_score"
    assert analytic.diagnostics.extra["hessian_status"] == "deferred"


def test_svd_cut4_analytic_score_blocks_active_floor() -> None:
    params = tf.constant([0.31, 0.27, 1.05], dtype=tf.float64)
    model, derivatives = _smooth_affine_model_and_derivatives(params)

    with pytest.raises(tf.errors.InvalidArgumentError, match="blocked_active_floor"):
        tf_svd_cut4_score(
            OBSERVATIONS,
            model,
            derivatives,
            placement_floor=tf.constant(10.0, dtype=tf.float64),
        )


def test_svd_cut4_analytic_score_blocks_weak_spectral_gap() -> None:
    params = tf.constant([0.31, 0.27, 1.05], dtype=tf.float64)
    model, derivatives = _smooth_affine_model_and_derivatives(params)

    with pytest.raises(tf.errors.InvalidArgumentError, match="blocked_weak_spectral_gap"):
        tf_svd_cut4_score(
            OBSERVATIONS,
            model,
            derivatives,
            spectral_gap_tolerance=tf.constant(10.0, dtype=tf.float64),
        )


def test_svd_cubature_analytic_score_graph_parity() -> None:
    params = tf.constant([0.31, 0.27, 1.05], dtype=tf.float64)
    model, derivatives = _smooth_affine_model_and_derivatives(params)
    eager = tf_svd_cubature_score(OBSERVATIONS, model, derivatives)

    @tf.function(reduce_retracing=True)
    def compiled() -> tuple[tf.Tensor, tf.Tensor]:
        result = tf_svd_cubature_score(OBSERVATIONS, model, derivatives)
        return result.log_likelihood, result.score

    graph_value, graph_score = compiled()
    np.testing.assert_allclose(graph_value.numpy(), eager.log_likelihood.numpy(), atol=1e-12)
    np.testing.assert_allclose(graph_score.numpy(), eager.score.numpy(), atol=1e-12)


@pytest.mark.parametrize(
    ("score_fn", "backend"),
    [
        (tf_svd_cubature_score, "tf_svd_cubature"),
        (tf_svd_ukf_score, "tf_svd_ukf"),
        (tf_svd_cut4_score, "tf_svd_cut4"),
    ],
)
def test_model_b_analytic_score_matches_finite_difference(score_fn, backend) -> None:
    params = tf.constant([0.70, 0.25, 0.80], dtype=tf.float64)
    model, derivatives = _model_b_and_derivatives(params)
    analytic = score_fn(
        model_b_observations_tf(),
        model,
        derivatives,
        innovation_floor=tf.constant(1e-12, dtype=tf.float64),
        spectral_gap_tolerance=tf.constant(1e-8, dtype=tf.float64),
    )
    finite_difference = _finite_difference_score(
        lambda values: _model_b_value(values, backend),
        params.numpy(),
        step=2e-5,
    )

    np.testing.assert_allclose(analytic.score.numpy(), finite_difference, rtol=5e-4, atol=5e-4)
    assert analytic.hessian is None
    assert analytic.diagnostics.extra["derivative_provider"] == (
        "model_b_nonlinear_accumulation_first_derivatives"
    )
    assert analytic.diagnostics.extra["derivative_method"] == "analytic_first_order_smooth_branch"
    np.testing.assert_allclose(
        analytic.diagnostics.extra["deterministic_residual"].numpy(),
        0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    ("score_fn", "backend"),
    [
        (tf_svd_cubature_score, "tf_svd_cubature"),
        (tf_svd_ukf_score, "tf_svd_ukf"),
        (tf_svd_cut4_score, "tf_svd_cut4"),
    ],
)
def test_model_c_smooth_phase_analytic_score_matches_finite_difference(
    score_fn,
    backend,
) -> None:
    params = tf.constant([1.0, 1.0, 0.20], dtype=tf.float64)
    phase_variance = tf.constant(0.05, dtype=tf.float64)
    model, derivatives = _model_c_and_derivatives(
        params,
        phase_variance=phase_variance,
    )
    analytic = score_fn(
        model_c_observations_tf(),
        model,
        derivatives,
        innovation_floor=tf.constant(1e-12, dtype=tf.float64),
        spectral_gap_tolerance=tf.constant(1e-8, dtype=tf.float64),
    )
    finite_difference = _finite_difference_score(
        lambda values: _model_c_value(values, backend, phase_variance=phase_variance),
        params.numpy(),
        step=1e-5,
    )

    np.testing.assert_allclose(analytic.score.numpy(), finite_difference, rtol=1e-3, atol=1e-3)
    assert analytic.hessian is None
    assert analytic.diagnostics.extra["derivative_provider"] == (
        "model_c_nonlinear_growth_first_derivatives"
    )
    assert analytic.diagnostics.extra["derivative_method"] == "analytic_first_order_smooth_branch"
    np.testing.assert_allclose(
        analytic.diagnostics.extra["deterministic_residual"].numpy(),
        0.0,
        atol=1e-12,
    )


def test_model_c_default_zero_phase_variance_blocks_smooth_score_branch() -> None:
    params = tf.constant([1.0, 1.0, 0.20], dtype=tf.float64)
    model, derivatives = _model_c_and_derivatives(
        params,
        phase_variance=tf.constant(0.0, dtype=tf.float64),
    )

    with pytest.raises(tf.errors.InvalidArgumentError, match="blocked_active_floor"):
        tf_svd_cut4_score(
            model_c_observations_tf(),
            model,
            derivatives,
            placement_floor=tf.constant(0.0, dtype=tf.float64),
            innovation_floor=tf.constant(1e-12, dtype=tf.float64),
            spectral_gap_tolerance=tf.constant(1e-8, dtype=tf.float64),
        )


@pytest.mark.parametrize(
    ("score_fn", "backend"),
    [
        (tf_svd_cubature_score, "tf_svd_cubature"),
        (tf_svd_ukf_score, "tf_svd_ukf"),
        (tf_svd_cut4_score, "tf_svd_cut4"),
    ],
)
def test_model_c_default_structural_fixed_support_score_matches_finite_difference(
    score_fn,
    backend,
) -> None:
    params = tf.constant([1.0, 1.0, 0.20], dtype=tf.float64)
    model, derivatives = _model_c_and_derivatives(
        params,
        phase_variance=tf.constant(0.0, dtype=tf.float64),
    )
    analytic = score_fn(
        model_c_observations_tf(),
        model,
        derivatives,
        innovation_floor=tf.constant(1e-12, dtype=tf.float64),
        spectral_gap_tolerance=tf.constant(1e-8, dtype=tf.float64),
        allow_fixed_null_support=True,
    )
    finite_difference = _finite_difference_score(
        lambda values: _model_c_structural_fixed_support_value(values, backend),
        params.numpy(),
        step=1e-5,
    )

    np.testing.assert_allclose(analytic.score.numpy(), finite_difference, rtol=1e-3, atol=1e-3)
    assert analytic.hessian is None
    assert analytic.metadata.differentiability_status == (
        "analytic_score_structural_fixed_support_hessian_deferred"
    )
    assert analytic.diagnostics.extra["derivative_method"] == (
        "analytic_first_order_structural_fixed_support"
    )
    assert analytic.diagnostics.extra["derivative_branch"] == (
        "structural_fixed_support_no_active_floor"
    )
    assert analytic.diagnostics.extra["sigma_point_variable"] == (
        "pre_transition_structural"
    )
    assert int(analytic.diagnostics.extra["structural_null_count"].numpy()) == 1
    assert int(analytic.diagnostics.extra["placement_floor_count"].numpy()) == 0
    np.testing.assert_allclose(
        analytic.diagnostics.extra["structural_null_covariance_residual"].numpy(),
        0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        analytic.diagnostics.extra["fixed_null_derivative_residual"].numpy(),
        0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        analytic.diagnostics.extra["deterministic_residual"].numpy(),
        0.0,
        atol=1e-12,
    )


def test_model_c_structural_fixed_support_blocks_positive_placement_floor() -> None:
    params = tf.constant([1.0, 1.0, 0.20], dtype=tf.float64)
    model, derivatives = _model_c_and_derivatives(
        params,
        phase_variance=tf.constant(0.0, dtype=tf.float64),
    )

    with pytest.raises(tf.errors.InvalidArgumentError, match="blocked_active_floor"):
        tf_svd_cut4_score(
            model_c_observations_tf(),
            model,
            derivatives,
            placement_floor=tf.constant(1e-10, dtype=tf.float64),
            innovation_floor=tf.constant(1e-12, dtype=tf.float64),
            allow_fixed_null_support=True,
        )


def test_model_c_structural_fixed_support_blocks_moving_null_covariance() -> None:
    params = tf.constant([1.0, 1.0, 0.20], dtype=tf.float64)
    model, derivatives = _model_c_and_derivatives(
        params,
        phase_variance=tf.constant(1e-11, dtype=tf.float64),
    )

    with pytest.raises(
        tf.errors.InvalidArgumentError,
        match="blocked_structural_null_covariance",
    ):
        tf_svd_cut4_score(
            model_c_observations_tf(),
            model,
            derivatives,
            innovation_floor=tf.constant(1e-12, dtype=tf.float64),
            rank_tolerance=tf.constant(1e-10, dtype=tf.float64),
            fixed_null_tolerance=tf.constant(1e-12, dtype=tf.float64),
            allow_fixed_null_support=True,
        )
