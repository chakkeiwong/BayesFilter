from __future__ import annotations

import ast
import inspect
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.testing import multidim_triangular_lgssm_batched_tf as batched
from bayesfilter.testing.multidim_triangular_lgssm_tf import (
    gaussian_raw_prior_log_prob_and_score,
    load_lower_triangular_lgssm_contract,
    materialize_lower_triangular_lgssm_with_first_derivatives,
    raw_truth_from_contract,
)


def _batch() -> tuple[dict, tf.Tensor]:
    contract = load_lower_triangular_lgssm_contract()
    truth = raw_truth_from_contract(contract)
    offsets = tf.constant(
        [
            [0.0] * 18,
            [0.02, -0.01, 0.01, -0.02, 0.01, 0.0, -0.01, 0.02, -0.02, 0.01, 0.01, -0.01, 0.02, -0.02, 0.01, -0.01, 0.02, -0.02],
            [-0.01, 0.02, -0.02, 0.01, -0.01, 0.02, 0.01, -0.02, 0.0, 0.01, -0.02, 0.02, -0.01, 0.01, -0.02, 0.01, -0.01, 0.02],
        ],
        tf.float64,
    )
    return contract, truth[tf.newaxis, :] + offsets


def _assert_row_matches_scalar(actual, row: int, raw: tf.Tensor, contract: dict) -> None:
    expected = materialize_lower_triangular_lgssm_with_first_derivatives(raw, contract)
    assert expected.derivatives is not None
    pairs = (
        (actual.initial_mean[row], expected.model.initial_mean),
        (actual.initial_covariance[row], expected.model.initial_covariance),
        (actual.transition_offset[row], expected.model.transition_offset),
        (actual.transition_matrix[row], expected.model.transition_matrix),
        (actual.transition_covariance[row], expected.model.transition_covariance),
        (actual.observation_offset[row], expected.model.observation_offset),
        (actual.observation_matrix[row], expected.model.observation_matrix),
        (actual.observation_covariance[row], expected.model.observation_covariance),
        (actual.d_initial_mean[row], expected.derivatives.d_initial_mean),
        (actual.d_initial_covariance[row], expected.derivatives.d_initial_covariance),
        (actual.d_transition_offset[row], expected.derivatives.d_transition_offset),
        (actual.d_transition_matrix[row], expected.derivatives.d_transition_matrix),
        (actual.d_transition_covariance[row], expected.derivatives.d_transition_covariance),
        (actual.d_observation_offset[row], expected.derivatives.d_observation_offset),
        (actual.d_observation_matrix[row], expected.derivatives.d_observation_matrix),
        (actual.d_observation_covariance[row], expected.derivatives.d_observation_covariance),
    )
    for observed, wanted in pairs:
        np.testing.assert_allclose(observed.numpy(), wanted.numpy(), rtol=2e-13, atol=2e-13)


def test_batch_materialization_shapes_and_scalar_row_parity() -> None:
    contract, raw = _batch()
    result = batched.materialize_lower_triangular_lgssm_batch(raw, contract)

    assert result.transition_matrix.shape == (3, 4, 4)
    assert result.d_transition_matrix.shape == (3, 18, 4, 4)
    assert result.d_initial_covariance.shape == (3, 18, 4, 4)
    _assert_row_matches_scalar(result, 0, raw[0], contract)
    _assert_row_matches_scalar(result, 1, raw[1], contract)
    _assert_row_matches_scalar(result, 2, raw[2], contract)


def test_batch_prior_matches_scalar_rows() -> None:
    contract, raw = _batch()
    values, scores = batched.gaussian_raw_prior_log_prob_and_score_batch(raw, contract)
    value0, score0 = gaussian_raw_prior_log_prob_and_score(raw[0], contract)
    value1, score1 = gaussian_raw_prior_log_prob_and_score(raw[1], contract)
    value2, score2 = gaussian_raw_prior_log_prob_and_score(raw[2], contract)
    np.testing.assert_allclose(values.numpy(), [value0.numpy(), value1.numpy(), value2.numpy()], rtol=0.0, atol=2e-14)
    np.testing.assert_allclose(scores.numpy(), tf.stack((score0, score1, score2)).numpy(), rtol=0.0, atol=2e-14)


def test_row_permutation_equivariance() -> None:
    contract, raw = _batch()
    permutation = tf.constant([2, 0, 1], tf.int32)
    direct = batched.materialize_lower_triangular_lgssm_batch(raw, contract)
    permuted = batched.materialize_lower_triangular_lgssm_batch(tf.gather(raw, permutation), contract)
    np.testing.assert_allclose(permuted.initial_covariance.numpy(), tf.gather(direct.initial_covariance, permutation).numpy(), rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(permuted.d_initial_covariance.numpy(), tf.gather(direct.d_initial_covariance, permutation).numpy(), rtol=2e-13, atol=2e-13)


def test_primal_and_derivative_lyapunov_residuals() -> None:
    contract, raw = _batch()
    result = batched.materialize_lower_triangular_lgssm_batch(raw, contract)
    transition_p = result.transition_matrix[:, tf.newaxis, :, :]
    stationary_p = result.initial_covariance[:, tf.newaxis, :, :]
    primal = result.initial_covariance - result.transition_matrix @ result.initial_covariance @ tf.linalg.matrix_transpose(result.transition_matrix) - result.transition_covariance
    derivative = result.d_initial_covariance - (
        result.d_transition_matrix @ stationary_p @ tf.linalg.matrix_transpose(transition_p)
        + transition_p @ result.d_initial_covariance @ tf.linalg.matrix_transpose(transition_p)
        + transition_p @ stationary_p @ tf.linalg.matrix_transpose(result.d_transition_matrix)
        + result.d_transition_covariance
    )
    assert float(tf.reduce_max(tf.abs(primal)).numpy()) <= 2e-15
    assert float(tf.reduce_max(tf.abs(derivative)).numpy()) <= 3e-15


@pytest.mark.parametrize(
    ("parameter", "field", "derivative_field"),
    (
        (0, "transition_matrix", "d_transition_matrix"),
        (10, "transition_covariance", "d_transition_covariance"),
        (14, "observation_covariance", "d_observation_covariance"),
        (0, "initial_covariance", "d_initial_covariance"),
        (10, "initial_covariance", "d_initial_covariance"),
    ),
)
def test_representative_derivatives_match_central_difference(
    parameter: int,
    field: str,
    derivative_field: str,
) -> None:
    contract, raw = _batch()
    center = raw[1:2]
    direction = tf.one_hot(parameter, 18, dtype=tf.float64)[tf.newaxis, :]
    epsilon = tf.constant(1e-5, tf.float64)
    points = tf.concat((center + epsilon * direction, center - epsilon * direction), axis=0)
    finite = batched.materialize_lower_triangular_lgssm_batch(points, contract)
    analytical = batched.materialize_lower_triangular_lgssm_batch(center, contract)
    numeric = (getattr(finite, field)[0] - getattr(finite, field)[1]) / (2.0 * epsilon)
    expected = getattr(analytical, derivative_field)[0, parameter]
    np.testing.assert_allclose(numeric.numpy(), expected.numpy(), rtol=2e-8, atol=2e-9)


def test_graph_dynamic_batch_and_cpu_xla() -> None:
    contract, raw = _batch()

    @tf.function(input_signature=[tf.TensorSpec((None, 18), tf.float64)])
    def dynamic(values):
        result = batched.materialize_lower_triangular_lgssm_batch(values, contract)
        return result.initial_covariance, result.d_initial_covariance

    @tf.function(input_signature=[tf.TensorSpec((3, 18), tf.float64)], jit_compile=True)
    def compiled(values):
        result = batched.materialize_lower_triangular_lgssm_batch(values, contract)
        return result.transition_matrix, result.initial_covariance, result.d_initial_covariance

    dynamic_values = dynamic(raw[:2])
    compiled_values = compiled(raw)
    assert dynamic_values[0].shape == (2, 4, 4)
    assert compiled_values[2].shape == (3, 18, 4, 4)
    assert all(bool(tf.reduce_all(tf.math.is_finite(value)).numpy()) for value in compiled_values)


def test_algorithmic_source_has_no_mapping_loop_numpy_or_host_callback() -> None:
    source = inspect.getsource(batched)
    tree = ast.parse(source)
    assert "numpy" not in source.lower()
    assert "tf.map_fn" not in source
    assert "tf.vectorized_map" not in source
    assert "tf.py_function" not in source
    assert "tf.numpy_function" not in source
    assert not any(isinstance(node, (ast.For, ast.AsyncFor, ast.While)) for node in ast.walk(tree))

