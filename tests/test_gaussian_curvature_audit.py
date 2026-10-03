"""Independent analytic oracles for a guide-curvature diagnostic, not normality."""

import math
from dataclasses import replace

import pytest
import tensorflow as tf
from bayesfilter.inference.gaussian_curvature_audit import (
    GaussianCurvatureAuditConfig,
    audit_gaussian_curvature,
    binomial_nonpass_upper,
    compare_curvature_steps,
)


def _config(**changes):
    return replace(GaussianCurvatureAuditConfig(probe_count=8, max_nonpass_probability=0.5), **changes)


def _gaussian(precision, mean=None):
    precision = tf.constant(precision, tf.float64)
    dimension = int(precision.shape[0])
    mean = tf.zeros([dimension], tf.float64) if mean is None else tf.constant(mean, tf.float64)

    @tf.function(input_signature=[tf.TensorSpec([None, dimension], tf.float64)], autograph=False)
    def target(points):
        centered = points - mean
        scores = -tf.matmul(centered, precision)
        return 0.5 * tf.reduce_sum(centered * scores, axis=1), scores, tf.ones(tf.shape(points)[0], tf.bool)

    return target


@pytest.mark.parametrize("eigenvalue,passed", [(0.25, True), (4.0, True), (0.249, False), (4.001, False), (1.0e-12, False), (-1.0, False)])
def test_exact_matrix_band_includes_boundaries_without_response_division(eigenvalue, passed):
    matrix = tf.linalg.diag(tf.constant([1.0, eigenvalue], tf.float64))
    result = compare_curvature_steps(matrix, matrix)
    assert bool(result["passed"]) is passed


def test_boundary_with_numerical_uncertainty_is_inconclusive():
    matrix = tf.eye(2, dtype=tf.float64) * 0.25
    result = compare_curvature_steps(matrix, matrix, roundoff_margin=1e-10)
    assert bool(result["inconclusive"])
    assert not bool(result["passed"])


def test_full_spectrum_catches_one_stiff_direction_in_142_dimensions():
    vector = tf.ones([142], tf.float64) / math.sqrt(142)
    matrix = tf.eye(142, dtype=tf.float64) + 8.0 * tf.tensordot(vector, vector, axes=0)
    result = compare_curvature_steps(matrix, matrix)
    assert float(result["maximum"]) == pytest.approx(9.0)
    assert not bool(result["passed"])


def test_nonsymmetric_score_cannot_pass_after_symmetrization():
    matrix = tf.constant([[1.0, 0.3], [-0.3, 1.0]], tf.float64)
    result = compare_curvature_steps(matrix, matrix)
    assert bool(result["inconclusive"])
    assert float(result["antisymmetry"]) > 0.4


def test_step_disagreement_is_numerical_inconclusive():
    result = compare_curvature_steps(tf.eye(2, dtype=tf.float64), tf.eye(2, dtype=tf.float64) * 1.02)
    assert not bool(result["resolved"])


def test_correlated_affine_gaussian_preserves_position_factor_orientation():
    factor = tf.constant([[2.0, 0.0], [0.7, 0.3]], tf.float64)
    covariance = factor @ tf.transpose(factor)
    precision = tf.linalg.inv(covariance)
    target = _gaussian(precision, [3.0, -2.0])
    result = audit_gaussian_curvature(target, [3.0, -2.0], factor, config=_config())
    assert result["passed"]
    for point in result["point_diagnostics"]:
        assert float(point["minimum"]) == pytest.approx(1.0, abs=1e-9)
        assert float(point["maximum"]) == pytest.approx(1.0, abs=1e-9)


def test_analytic_nonlinear_hessian_and_sign():
    coefficient = 0.03

    def quartic(points):
        values = -tf.reduce_sum(points**2 / 2 + coefficient * points**4 / 4, axis=1)
        scores = -points - coefficient * points**3
        return values, scores, tf.ones(tf.shape(points)[0], tf.bool)

    result = audit_gaussian_curvature(quartic, [0.0, 0.0], tf.eye(2, dtype=tf.float64), config=_config())
    assert result["passed"]
    for latent, point in zip(result["latent_points"], result["point_diagnostics"]):
        expected = tf.linalg.diag(1 + 3 * coefficient * latent**2)
        assert float(tf.reduce_max(tf.abs(point["fine_curvature"] - expected))) < 1e-7


def test_probe_design_replays_inside_unit_ball_and_is_not_gaussian_cloud():
    target = _gaussian(tf.eye(12, dtype=tf.float64))
    config = _config(probe_count=64, max_nonpass_probability=0.05)
    first = audit_gaussian_curvature(target, tf.zeros([12], tf.float64), tf.eye(12, dtype=tf.float64), config=config)
    second = audit_gaussian_curvature(target, tf.zeros([12], tf.float64), tf.eye(12, dtype=tf.float64), config=config)
    assert bool(tf.reduce_all(first["latent_points"] == second["latent_points"]))
    radii = tf.linalg.norm(first["latent_points"], axis=1)
    assert float(tf.reduce_max(radii)) <= 1.0
    assert 0.7 < float(tf.reduce_mean(radii)) < 1.0
    assert first["passed"] and second["passed"]


@pytest.mark.parametrize("failure", ["sentinel", "nan_value", "nan_score"])
def test_invalid_target_rows_stop_without_replacement_and_are_recorded(failure):
    recorded = []

    def target(points):
        values = tf.zeros([8], tf.float64)
        scores = -points
        eligible = tf.ones([8], tf.bool)
        if failure == "sentinel":
            eligible = tf.tensor_scatter_nd_update(eligible, [[3]], [False])
        elif failure == "nan_value":
            values = tf.tensor_scatter_nd_update(values, [[3]], tf.constant([math.nan], tf.float64))
        else:
            scores = tf.tensor_scatter_nd_update(scores, [[3, 0]], tf.constant([math.nan], tf.float64))
        return values, scores, eligible

    result = audit_gaussian_curvature(target, [0.0, 0.0], tf.eye(2, dtype=tf.float64), config=_config(),
                                     record_batch=lambda *row: recorded.append(row))
    assert not result["passed"]
    assert result["target_calls"] == len(recorded) == 1
    assert result["physical_rows"] == 8
    assert result["status"] in {"ineligible_target", "nonfinite_target"}


def test_padding_is_counted_and_recorded_without_changing_iid_probe_count():
    batches = []
    result = audit_gaussian_curvature(_gaussian([[1.0]]), [0.0], [[1.0]], config=_config(probe_count=9),
                                     record_batch=lambda *batch: batches.append(batch))
    assert result["physical_rows"] == 16 + 9 * 8
    assert result["logical_rows"] == 9 + 9 * 4
    assert result["target_calls"] == len(batches) == 11
    assert len(result["point_diagnostics"]) == 9


def test_preflight_budget_failure_makes_no_target_call():
    def unexpected(points):
        raise AssertionError("budget must reject before callback")

    with pytest.raises(ValueError, match="before target calls"):
        audit_gaussian_curvature(unexpected, [0.0, 0.0], tf.eye(2, dtype=tf.float64), config=_config(max_physical_rows=1))


def test_probability_bound_matches_zero_failure_formula_and_independent_binomial_oracle():
    assert binomial_nonpass_upper(0, 64) == pytest.approx(1 - 0.05**(1 / 64), abs=1e-13)
    assert binomial_nonpass_upper(64, 64) == 1.0
    upper = binomial_nonpass_upper(2, 64)
    cumulative = sum(math.comb(64, count) * upper**count * (1 - upper)**(64 - count) for count in range(3))
    assert cumulative == pytest.approx(0.05, abs=1e-12)
    assert binomial_nonpass_upper(1, 64) > 0.05


def test_small_response_is_a_finite_band_failure():
    result = audit_gaussian_curvature(_gaussian([[1e-10]]), [0.0], [[1.0]], config=_config())
    assert not result["passed"]
    assert result["nonpasses"] == 8
    assert all(math.isfinite(float(point["minimum"])) for point in result["point_diagnostics"])


def test_numerically_unresolvable_large_linear_score_cannot_pass():
    result = audit_gaussian_curvature(_gaussian([[1.0]], [1e14]), [0.0], [[1.0]], config=_config())
    assert not result["passed"]
    assert all(bool(point["inconclusive"]) for point in result["point_diagnostics"])


def test_smooth_incorrect_score_cannot_pass_with_in_band_hessian():
    def wrong_score(points):
        return -tf.reduce_sum(points**2, axis=1) / 2, -2 * points, tf.ones(tf.shape(points)[0], tf.bool)

    result = audit_gaussian_curvature(wrong_score, [0.0, 0.0], tf.eye(2, dtype=tf.float64), config=_config())
    assert not result["passed"]
    assert all(float(point["maximum"]) == pytest.approx(2.0) for point in result["point_diagnostics"])
    assert all(not bool(point["value_gradient_consistent"]) for point in result["point_diagnostics"])
