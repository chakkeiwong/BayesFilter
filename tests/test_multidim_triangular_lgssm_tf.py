from __future__ import annotations

import inspect
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import tensorflow as tf

from bayesfilter.testing import multidim_triangular_lgssm_tf as triangular


def test_lower_triangular_contract_truth_materializes_expected_model() -> None:
    contract = triangular.load_lower_triangular_lgssm_contract()
    raw = triangular.raw_truth_from_contract(contract)
    materialized = triangular.materialize_lower_triangular_lgssm_from_raw(raw, contract)

    expected_transition = tf.constant(
        [
            [0.62, 0.0, 0.0, 0.0],
            [0.18, 0.48, 0.0, 0.0],
            [-0.10, 0.14, 0.30, 0.0],
            [0.06, -0.08, 0.11, 0.16],
        ],
        dtype=tf.float64,
    )
    expected_process_std = tf.constant([0.30, 0.26, 0.22, 0.18], dtype=tf.float64)
    expected_observation_std = tf.constant([0.12, 0.11, 0.10, 0.09], dtype=tf.float64)

    np.testing.assert_allclose(
        materialized.transition_matrix.numpy(),
        expected_transition.numpy(),
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        materialized.process_std.numpy(),
        expected_process_std.numpy(),
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        materialized.observation_std.numpy(),
        expected_observation_std.numpy(),
        atol=1.0e-14,
    )
    assert materialized.model.observation_matrix.shape == (4, 4)
    np.testing.assert_allclose(
        materialized.model.observation_matrix.numpy(),
        np.eye(4),
        atol=0.0,
    )


def test_stationary_covariance_matches_phase2_fixture_and_residual() -> None:
    contract = triangular.load_lower_triangular_lgssm_contract()
    fixture = triangular.load_lower_triangular_lgssm_synthetic_data()
    raw = tf.constant(fixture["raw_truth"], dtype=tf.float64)
    materialized = triangular.materialize_lower_triangular_lgssm_from_raw(raw, contract)
    expected = tf.constant(
        fixture["constrained_truth"]["stationary_initial_covariance"],
        dtype=tf.float64,
    )
    residual = triangular.lyapunov_residual_tf(
        materialized.transition_matrix,
        materialized.stationary_covariance,
        materialized.model.transition_covariance,
    )

    np.testing.assert_allclose(
        materialized.stationary_covariance.numpy(),
        expected.numpy(),
        atol=1.0e-14,
    )
    assert float(tf.reduce_max(tf.abs(residual)).numpy()) <= 1.0e-14
    assert float(tf.reduce_min(tf.linalg.eigvalsh(materialized.stationary_covariance)).numpy()) > 0.0


def test_first_derivatives_match_finite_difference_blocks() -> None:
    contract = triangular.load_lower_triangular_lgssm_contract()
    raw = triangular.raw_truth_from_contract(contract)
    materialized = triangular.materialize_lower_triangular_lgssm_with_first_derivatives(
        raw,
        contract,
    )
    assert materialized.derivatives is not None
    eps = tf.constant(1.0e-5, dtype=tf.float64)
    for index in range(18):
        basis = tf.one_hot(index, 18, dtype=tf.float64)
        plus = triangular.materialize_lower_triangular_lgssm_from_raw(
            raw + eps * basis,
            contract,
        )
        minus = triangular.materialize_lower_triangular_lgssm_from_raw(
            raw - eps * basis,
            contract,
        )
        np.testing.assert_allclose(
            materialized.derivatives.d_transition_matrix[index].numpy(),
            ((plus.transition_matrix - minus.transition_matrix) / (2.0 * eps)).numpy(),
            rtol=1.0e-6,
            atol=1.0e-7,
        )
        np.testing.assert_allclose(
            materialized.derivatives.d_transition_covariance[index].numpy(),
            (
                (
                    plus.model.transition_covariance
                    - minus.model.transition_covariance
                )
                / (2.0 * eps)
            ).numpy(),
            rtol=1.0e-6,
            atol=1.0e-7,
        )
        np.testing.assert_allclose(
            materialized.derivatives.d_observation_covariance[index].numpy(),
            (
                (
                    plus.model.observation_covariance
                    - minus.model.observation_covariance
                )
                / (2.0 * eps)
            ).numpy(),
            rtol=1.0e-6,
            atol=1.0e-7,
        )
        np.testing.assert_allclose(
            materialized.derivatives.d_initial_covariance[index].numpy(),
            (
                (
                    plus.stationary_covariance
                    - minus.stationary_covariance
                )
                / (2.0 * eps)
            ).numpy(),
            rtol=2.0e-5,
            atol=2.0e-7,
        )


def test_lower_triangular_log_prob_score_matches_finite_difference() -> None:
    contract = triangular.load_lower_triangular_lgssm_contract()
    fixture = triangular.load_lower_triangular_lgssm_synthetic_data()
    raw = tf.constant(fixture["raw_truth"], dtype=tf.float64)
    observations = triangular.lower_triangular_lgssm_observations_from_fixture(fixture)
    value, score, likelihood_value, likelihood_score = (
        triangular.lower_triangular_lgssm_log_prob_and_score(
            raw,
            observations,
            contract,
        )
    )

    assert value.shape == ()
    assert score.shape == (18,)
    assert likelihood_value.shape == ()
    assert likelihood_score.shape == (18,)
    assert bool(tf.math.is_finite(value).numpy())
    assert bool(tf.reduce_all(tf.math.is_finite(score)).numpy())
    eps = tf.constant(1.0e-5, dtype=tf.float64)
    columns = []
    for index in range(18):
        basis = tf.one_hot(index, 18, dtype=tf.float64)
        plus, _plus_score, _plus_likelihood, _plus_likelihood_score = (
            triangular.lower_triangular_lgssm_log_prob_and_score(
                raw + eps * basis,
                observations,
                contract,
            )
        )
        minus, _minus_score, _minus_likelihood, _minus_likelihood_score = (
            triangular.lower_triangular_lgssm_log_prob_and_score(
                raw - eps * basis,
                observations,
                contract,
            )
        )
        columns.append((plus - minus) / (2.0 * eps))
    finite_difference = tf.stack(columns, axis=0)

    np.testing.assert_allclose(
        score.numpy(),
        finite_difference.numpy(),
        rtol=2.0e-4,
        atol=2.0e-4,
    )


def test_lower_triangular_materialization_jit_compiles_cpu_hidden() -> None:
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
    contract = triangular.load_lower_triangular_lgssm_contract()
    raw = triangular.raw_truth_from_contract(contract)

    @tf.function(jit_compile=True)
    def compiled(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        materialized = triangular.materialize_lower_triangular_lgssm_from_raw(theta, contract)
        residual = triangular.lyapunov_residual_tf(
            materialized.transition_matrix,
            materialized.stationary_covariance,
            materialized.model.transition_covariance,
        )
        return materialized.transition_matrix, residual

    transition, residual = compiled(raw)

    assert transition.shape == (4, 4)
    assert residual.shape == (4, 4)
    assert float(tf.reduce_max(tf.abs(residual)).numpy()) <= 1.0e-14


def test_lower_triangular_value_score_jit_compiles_cpu_hidden() -> None:
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
    contract = triangular.load_lower_triangular_lgssm_contract()
    fixture = triangular.load_lower_triangular_lgssm_synthetic_data()
    raw = tf.constant(fixture["raw_truth"], dtype=tf.float64)
    observations = triangular.lower_triangular_lgssm_observations_from_fixture(fixture)

    @tf.function(jit_compile=True)
    def compiled(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        value, score, _likelihood, _likelihood_score = (
            triangular.lower_triangular_lgssm_log_prob_and_score(
                theta,
                observations,
                contract,
            )
        )
        return value, score

    value, score = compiled(raw)

    assert value.shape == ()
    assert score.shape == (18,)
    assert bool(tf.math.is_finite(value).numpy())
    assert bool(tf.reduce_all(tf.math.is_finite(score)).numpy())


def test_lower_triangular_value_score_uses_svd_graph_status_backend() -> None:
    contract = triangular.load_lower_triangular_lgssm_contract()
    fixture = triangular.load_lower_triangular_lgssm_synthetic_data()
    raw = tf.constant(fixture["raw_truth"], dtype=tf.float64)
    observations = triangular.lower_triangular_lgssm_observations_from_fixture(fixture)

    value, score, likelihood, likelihood_score, status = (
        triangular.lower_triangular_lgssm_log_prob_score_status(
            raw,
            observations,
            contract,
        )
    )

    assert value.shape == ()
    assert score.shape == (18,)
    assert likelihood.shape == ()
    assert likelihood_score.shape == (18,)
    assert int(status["status_code"].numpy()) == 0
    assert bool(status["valid_pre_regularized_score"].numpy())
    assert int(status["floor_count_value"].numpy()) == 0
    assert float(status["min_innovation_eigenvalue"].numpy()) > 0.0
    assert float(status["innovation_condition_estimate"].numpy()) >= 1.0


def test_lower_triangular_runtime_uses_no_generic_autodiff() -> None:
    source = inspect.getsource(triangular)
    for forbidden in ("GradientTape", "jacobian", "batch_jacobian"):
        assert forbidden not in source


def test_lower_triangular_hmc_target_demotes_qr_derivative_path() -> None:
    source = inspect.getsource(triangular)
    assert "tf_qr_linear_gaussian_score" not in source
    assert "kalman_qr_derivatives_tf" not in source
    assert "tf_svd_linear_gaussian_score_first_order_graph_status" in source
