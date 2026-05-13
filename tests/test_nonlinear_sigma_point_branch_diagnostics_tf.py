import numpy as np
import pytest
import tensorflow as tf

from bayesfilter import StatePartition
from bayesfilter.nonlinear.svd_sigma_point_derivatives_tf import TFStructuralFirstDerivatives
from bayesfilter.structural_tf import make_affine_structural_tf
from bayesfilter.testing import (
    make_nonlinear_accumulation_first_derivatives_tf,
    make_nonlinear_accumulation_model_tf,
    make_univariate_nonlinear_growth_first_derivatives_tf,
    make_univariate_nonlinear_growth_model_tf,
    model_b_observations_tf,
    model_c_observations_tf,
    nonlinear_sigma_point_diagnostic_snapshot,
    nonlinear_sigma_point_score_branch_summary,
    nonlinear_sigma_point_value_branch_summary,
    tf_nonlinear_sigma_point_score,
    tf_nonlinear_sigma_point_value_filter,
)


BACKENDS = ("tf_svd_cubature", "tf_svd_ukf", "tf_svd_cut4")


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
        name="smooth_affine_branch_diagnostics_derivatives",
    )
    return model, derivatives


def _parameterized_model_a(params: tf.Tensor, *, repeated_spectrum: bool = False):
    phi = params[0]
    sigma = params[1]
    obs_scale = params[2]
    partition = StatePartition(
        state_names=("x", "lag_x"),
        stochastic_indices=(0,),
        deterministic_indices=(1,),
        innovation_dim=1,
    )
    initial_covariance = (
        tf.eye(2, dtype=tf.float64)
        if repeated_spectrum
        else tf.linalg.diag(tf.constant([1.2, 0.7], dtype=tf.float64))
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
    return make_affine_structural_tf(
        partition=partition,
        initial_mean=tf.constant([0.1, -0.2], dtype=tf.float64),
        initial_covariance=initial_covariance,
        transition_offset=tf.zeros([2], dtype=tf.float64),
        transition_matrix=transition_matrix,
        innovation_matrix=innovation_matrix,
        innovation_covariance=tf.constant([[0.43]], dtype=tf.float64),
        observation_offset=tf.zeros([1], dtype=tf.float64),
        observation_matrix=observation_matrix,
        observation_covariance=tf.constant([[0.19]], dtype=tf.float64),
    )


def _parameterized_model_b(params: tf.Tensor):
    return make_nonlinear_accumulation_model_tf(
        rho=params[0],
        sigma=params[1],
        beta=params[2],
    )


def _parameterized_model_b_derivatives(params: tf.Tensor):
    return make_nonlinear_accumulation_first_derivatives_tf(
        rho=params[0],
        sigma=params[1],
        beta=params[2],
    )


def _parameterized_model_c(params: tf.Tensor):
    return make_univariate_nonlinear_growth_model_tf(
        process_sigma=params[0],
        observation_sigma=params[1],
        initial_variance=params[2],
    )


def _parameterized_model_c_smooth_phase(params: tf.Tensor):
    return make_univariate_nonlinear_growth_model_tf(
        process_sigma=params[0],
        observation_sigma=params[1],
        initial_variance=params[2],
        initial_phase_variance=tf.constant(0.05, dtype=tf.float64),
    )


def _parameterized_model_c_derivatives(params: tf.Tensor):
    return make_univariate_nonlinear_growth_first_derivatives_tf(
        process_sigma=params[0],
        observation_sigma=params[1],
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_nonlinear_value_backends_share_required_diagnostic_vocabulary(backend) -> None:
    result = tf_nonlinear_sigma_point_value_filter(
        model_b_observations_tf(),
        make_nonlinear_accumulation_model_tf(),
        backend=backend,
    )
    snapshot = nonlinear_sigma_point_diagnostic_snapshot(result, mode="value")

    assert snapshot.backend == backend
    assert snapshot.mode == "value"
    assert snapshot.point_count > 0
    assert snapshot.polynomial_degree >= 3
    assert snapshot.max_integration_rank > 0
    assert snapshot.placement_floor_count == 0
    assert snapshot.innovation_floor_count == 0
    assert snapshot.placement_psd_projection_residual >= 0.0
    assert snapshot.innovation_psd_projection_residual >= 0.0
    assert snapshot.support_residual >= 0.0
    np.testing.assert_allclose(snapshot.deterministic_residual, 0.0, atol=1e-12)
    assert np.isfinite(snapshot.min_placement_eigen_gap)
    assert np.isfinite(snapshot.implemented_covariance_trace)
    assert snapshot.implemented_covariance_trace > 0.0
    assert snapshot.regularization_derivative_target == "blocked"
    assert snapshot.derivative_branch == "value_only"


@pytest.mark.parametrize("backend", BACKENDS)
def test_model_a_score_backends_share_required_branch_diagnostics(backend) -> None:
    observations = tf.constant([[0.2], [-0.05], [0.15]], dtype=tf.float64)
    params = tf.constant([0.31, 0.27, 1.05], dtype=tf.float64)
    model, derivatives = _smooth_affine_model_and_derivatives(params)
    result = tf_nonlinear_sigma_point_score(
        observations,
        model,
        derivatives,
        backend=backend,
        innovation_floor=tf.constant(1e-12, dtype=tf.float64),
        spectral_gap_tolerance=tf.constant(1e-7, dtype=tf.float64),
    )
    snapshot = nonlinear_sigma_point_diagnostic_snapshot(result, mode="score")

    expected_backend = f"{backend}_score" if backend != "tf_svd_cut4" else "tf_svd_cut4_score"
    assert snapshot.backend == expected_backend
    assert snapshot.mode == "score"
    assert snapshot.point_count > 0
    assert snapshot.polynomial_degree >= 3
    assert snapshot.regularization_derivative_target == "implemented_regularized_law"
    assert snapshot.derivative_branch == "smooth_simple_spectrum_no_active_floor"
    assert snapshot.derivative_method == "analytic_first_order_smooth_branch"
    np.testing.assert_allclose(snapshot.deterministic_residual, 0.0, atol=1e-12)


def test_value_branch_summaries_cover_models_a_b_c_and_all_backends() -> None:
    grids = [
        (
            tf.constant([[0.30, 0.24, 1.00], [0.35, 0.27, 1.05]], dtype=tf.float64),
            _parameterized_model_a,
            tf.constant([[0.2], [-0.05]], dtype=tf.float64),
        ),
        (
            tf.constant([[0.66, 0.23, 0.75], [0.70, 0.25, 0.80]], dtype=tf.float64),
            _parameterized_model_b,
            model_b_observations_tf(),
        ),
        (
            tf.constant([[0.90, 1.00, 0.20], [1.00, 1.10, 0.25]], dtype=tf.float64),
            _parameterized_model_c,
            model_c_observations_tf(),
        ),
    ]

    for backend in BACKENDS:
        for parameter_grid, builder, observations in grids:
            summary = nonlinear_sigma_point_value_branch_summary(
                observations,
                parameter_grid,
                builder,
                backend=backend,
            )
            assert summary.backend == backend
            assert summary.mode == "value"
            assert summary.total_count == int(parameter_grid.shape[0])
            assert summary.ok_count == summary.total_count
            assert summary.active_floor_count == 0
            assert summary.nonfinite_count == 0
            assert summary.ok_fraction == 1.0
            assert summary.max_point_count > 0
            assert summary.max_integration_rank > 0
            np.testing.assert_allclose(summary.max_deterministic_residual, 0.0, atol=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
def test_score_branch_summary_covers_affine_model_b_and_smooth_model_c(
    backend,
) -> None:
    cases = [
        (
            tf.constant([[0.29, 0.25, 1.00], [0.31, 0.27, 1.05]], dtype=tf.float64),
            lambda values: _smooth_affine_model_and_derivatives(values)[0],
            lambda values: _smooth_affine_model_and_derivatives(values)[1],
            tf.constant([[0.2], [-0.05], [0.15]], dtype=tf.float64),
        ),
        (
            tf.constant([[0.68, 0.24, 0.78], [0.70, 0.25, 0.80]], dtype=tf.float64),
            _parameterized_model_b,
            _parameterized_model_b_derivatives,
            model_b_observations_tf(),
        ),
        (
            tf.constant([[0.95, 1.00, 0.18], [1.00, 1.02, 0.20]], dtype=tf.float64),
            _parameterized_model_c_smooth_phase,
            _parameterized_model_c_derivatives,
            model_c_observations_tf(),
        ),
    ]

    for parameter_grid, model_builder, derivative_builder, observations in cases:
        summary = nonlinear_sigma_point_score_branch_summary(
            observations,
            parameter_grid,
            model_builder,
            derivative_builder,
            backend=backend,
            spectral_gap_tolerance=tf.constant(1e-8, dtype=tf.float64),
        )

        assert summary.mode == "score"
        assert summary.total_count == int(parameter_grid.shape[0])
        assert summary.ok_count == summary.total_count
        assert summary.active_floor_count == 0
        assert summary.weak_spectral_gap_count == 0
        assert summary.nonfinite_count == 0
        assert summary.ok_fraction == 1.0
        assert summary.max_point_count > 0
        np.testing.assert_allclose(summary.max_deterministic_residual, 0.0, atol=1e-12)


def test_default_model_c_score_branch_summary_counts_active_floor_blocker() -> None:
    parameter_grid = tf.constant([[1.0, 1.0, 0.20]], dtype=tf.float64)

    summary = nonlinear_sigma_point_score_branch_summary(
        model_c_observations_tf(),
        parameter_grid,
        _parameterized_model_c,
        _parameterized_model_c_derivatives,
        backend="tf_svd_cut4",
        spectral_gap_tolerance=tf.constant(1e-8, dtype=tf.float64),
    )

    assert summary.total_count == 1
    assert summary.ok_count == 0
    assert summary.active_floor_count == 1
    assert summary.weak_spectral_gap_count == 0


@pytest.mark.parametrize("backend", BACKENDS)
def test_default_model_c_structural_fixed_support_score_branch_summary_passes(
    backend,
) -> None:
    parameter_grid = tf.constant([[1.0, 1.0, 0.20]], dtype=tf.float64)

    summary = nonlinear_sigma_point_score_branch_summary(
        model_c_observations_tf(),
        parameter_grid,
        _parameterized_model_c,
        _parameterized_model_c_derivatives,
        backend=backend,
        spectral_gap_tolerance=tf.constant(1e-8, dtype=tf.float64),
        allow_fixed_null_support=True,
    )

    assert summary.total_count == 1
    assert summary.ok_count == 1
    assert summary.active_floor_count == 0
    assert summary.weak_spectral_gap_count == 0
    assert summary.nonfinite_count == 0


@pytest.mark.parametrize("backend", BACKENDS)
def test_default_model_c_structural_fixed_support_diagnostics(backend) -> None:
    params = tf.constant([1.0, 1.0, 0.20], dtype=tf.float64)
    result = tf_nonlinear_sigma_point_score(
        model_c_observations_tf(),
        _parameterized_model_c(params),
        _parameterized_model_c_derivatives(params),
        backend=backend,
        spectral_gap_tolerance=tf.constant(1e-8, dtype=tf.float64),
        allow_fixed_null_support=True,
    )
    snapshot = nonlinear_sigma_point_diagnostic_snapshot(result, mode="score")

    assert snapshot.derivative_branch == "structural_fixed_support_no_active_floor"
    assert snapshot.derivative_method == "analytic_first_order_structural_fixed_support"
    assert snapshot.sigma_point_variable == "pre_transition_structural"
    assert snapshot.structural_null_count == 1
    assert snapshot.placement_floor_count == 0
    np.testing.assert_allclose(snapshot.structural_null_covariance_residual, 0.0, atol=1e-12)
    np.testing.assert_allclose(snapshot.fixed_null_derivative_residual, 0.0, atol=1e-12)
    np.testing.assert_allclose(snapshot.deterministic_residual, 0.0, atol=1e-12)


def test_score_branch_summary_counts_weak_spectral_gap_blocker() -> None:
    observations = tf.constant([[0.2]], dtype=tf.float64)
    parameter_grid = tf.constant([[0.31, 0.27, 1.05]], dtype=tf.float64)

    def repeated_model(values: tf.Tensor):
        del values
        return _parameterized_model_a(
            tf.constant([0.31, 0.27, 1.05], dtype=tf.float64),
            repeated_spectrum=True,
        )

    summary = nonlinear_sigma_point_score_branch_summary(
        observations,
        parameter_grid,
        repeated_model,
        lambda values: _smooth_affine_model_and_derivatives(values)[1],
        backend="tf_svd_cut4",
        spectral_gap_tolerance=tf.constant(1e-7, dtype=tf.float64),
    )

    assert summary.total_count == 1
    assert summary.ok_count == 0
    assert summary.weak_spectral_gap_count == 1
