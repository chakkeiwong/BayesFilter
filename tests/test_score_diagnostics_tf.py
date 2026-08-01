from __future__ import annotations

import math

import pytest
import tensorflow as tf

from bayesfilter.score_diagnostics_tf import tf_score_comparison_diagnostics


def _diagnostics(candidate: tf.Tensor):
    return tf_score_comparison_diagnostics(
        candidate_score=candidate,
        reference_score_increments=tf.constant(
            [[1.0, 2.0, 0.0], [-0.5, 1.0, 0.0]], tf.float64
        ),
        diagonal_shrinkage=tf.constant(0.25, tf.float64),
        base_ridge=tf.constant(0.4, tf.float64),
        ridge_floor=tf.constant(0.0, tf.float64),
        ridge_scale_diagonal=tf.constant([1.0, 4.0, 9.0], tf.float64),
    )


def test_average_opg_and_total_metric_match_direct_formula() -> None:
    candidate = tf.constant([0.6, 2.8, 0.3], tf.float64)
    result = _diagnostics(candidate)
    increments = tf.constant([[1.0, 2.0, 0.0], [-0.5, 1.0, 0.0]], tf.float64)
    average = tf.transpose(increments) @ increments / 2.0
    shrunk = 0.75 * average + 0.25 * tf.linalg.diag(tf.linalg.diag_part(average))
    expected_total = 2.0 * shrunk + 0.4 * tf.linalg.diag(
        tf.constant([1.0, 4.0, 9.0], tf.float64)
    )
    tf.debugging.assert_near(result.average_opg, average, atol=1e-15, rtol=1e-15)
    tf.debugging.assert_near(
        result.total_metric, expected_total, atol=1e-15, rtol=1e-15
    )
    tf.debugging.assert_near(
        result.reference_score, tf.constant([0.5, 3.0, 0.0], tf.float64)
    )
    assert not bool(result.ridge_floor_active.numpy())


def test_regularized_metric_is_positive_definite_when_horizon_is_below_dimension() -> None:
    result = _diagnostics(tf.constant([0.5, 3.0, 0.0], tf.float64))
    assert int(tf.math.count_nonzero(result.average_opg_eigenvalues > 1e-12)) <= 2
    assert bool(tf.reduce_all(result.total_metric_eigenvalues > 0.0).numpy())
    assert float(result.total_metric_condition_proxy.numpy()) >= 1.0
    tf.debugging.assert_near(result.rms_total_metric_error, 0.0)


def test_batched_candidates_and_norm_diagnostics() -> None:
    candidates = tf.constant(
        [[0.5, 3.0, 0.0], [1.5, 3.0, 0.0]], tf.float64
    )
    result = _diagnostics(candidates)
    assert result.absolute_error_norm.shape == (2,)
    assert result.rms_total_metric_error.shape == (2,)
    tf.debugging.assert_near(
        result.absolute_error_norm, tf.constant([0.0, 1.0], tf.float64)
    )
    assert math.isclose(
        float(result.relative_total_score_norm_error[1].numpy()),
        1.0 / math.sqrt(9.25),
        rel_tol=1e-15,
    )


def test_zero_reference_denominators_are_reported_undefined() -> None:
    result = tf_score_comparison_diagnostics(
        candidate_score=tf.constant([1.0, 0.0], tf.float64),
        reference_score_increments=tf.zeros([2, 2], tf.float64),
        diagonal_shrinkage=0.0,
        base_ridge=1.0,
        ridge_floor=0.0,
        ridge_scale_diagonal=tf.ones([2], tf.float64),
    )
    assert math.isnan(float(result.relative_total_score_norm_error.numpy()))
    assert math.isnan(float(result.relative_increment_energy_error.numpy()))
    assert math.isfinite(float(result.rms_total_metric_error.numpy()))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("diagonal_shrinkage", -0.1),
        ("diagonal_shrinkage", 1.1),
        ("base_ridge", 0.0),
        ("ridge_floor", -0.1),
    ],
)
def test_invalid_hyperparameters_fail_closed(field: str, value: float) -> None:
    kwargs = {
        "candidate_score": tf.zeros([2], tf.float64),
        "reference_score_increments": tf.ones([2, 2], tf.float64),
        "diagonal_shrinkage": 0.0,
        "base_ridge": 1.0,
        "ridge_floor": 0.0,
        "ridge_scale_diagonal": tf.ones([2], tf.float64),
    }
    kwargs[field] = value
    with pytest.raises(tf.errors.InvalidArgumentError):
        tf_score_comparison_diagnostics(**kwargs)


def test_nonpositive_ridge_scale_fails_closed() -> None:
    with pytest.raises(tf.errors.InvalidArgumentError):
        tf_score_comparison_diagnostics(
            candidate_score=tf.zeros([2], tf.float64),
            reference_score_increments=tf.ones([2, 2], tf.float64),
            diagonal_shrinkage=0.0,
            base_ridge=1.0,
            ridge_floor=0.0,
            ridge_scale_diagonal=tf.constant([1.0, 0.0], tf.float64),
        )


def test_xla_cpu_smoke() -> None:
    @tf.function(jit_compile=True)
    def compiled(candidate: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        result = _diagnostics(candidate)
        return result.rms_total_metric_error, result.total_metric_eigenvalues

    error, eigenvalues = compiled(tf.constant([0.6, 2.8, 0.3], tf.float64))
    assert bool(tf.math.is_finite(error).numpy())
    assert bool(tf.reduce_all(eigenvalues > 0.0).numpy())
