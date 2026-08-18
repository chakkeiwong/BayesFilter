from __future__ import annotations

import math
import importlib.util

import pytest
import tensorflow as tf

from bayesfilter.independent_score import null_calibration_tf as null


def test_conformal_order_rank_has_declared_marginal_and_tolerance_modes() -> None:
    assert null.conformal_order_rank(200, coverage=0.95) == 191
    assert null.conformal_order_rank(200, coverage=0.95, tolerance_confidence=0.95) == 196


def test_conformal_threshold_is_order_statistic_not_score_scale_rule() -> None:
    scores = tf.range(1.0, 201.0, dtype=tf.float64)
    tf.debugging.assert_equal(null.conformal_threshold(scores), tf.constant(191.0, tf.float64))


def test_svd_geometry_handles_dependence_and_reports_rank() -> None:
    base = tf.constant([[0.0, 0.0, 0.0], [1.0, 1.0, 2.0], [2.0, 4.0, 4.0], [3.0, 9.0, 6.0], [4.0, 16.0, 8.0], [5.0, 25.0, 10.0]], tf.float64)
    geometry = null.fit_svd_geometry(base)
    assert geometry.rank == 2
    assert geometry.omitted_variance.numpy() >= 0.0
    assert geometry.distance(base).shape == (6,)


def test_svd_geometry_fails_closed_for_rank_one_null() -> None:
    with pytest.raises(ValueError, match="rank"):
        null.fit_svd_geometry(tf.constant([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], tf.float64))


def test_clopper_pearson_audit_veto_is_not_an_upper_bound_gate() -> None:
    assert null.audit_failure(33, 500)["falsified"] is False
    assert null.audit_failure(34, 500)["falsified"] is True
    assert null.audit_failure(0, 500)["one_sided_lower_bound"] == 0.0


def test_zero_mean_diagnostic_reports_simultaneous_result() -> None:
    values = tf.random.stateless_normal([128, 3], [19, 23], dtype=tf.float64)
    result = null.zero_mean_max_t_diagnostic(values, bootstrap_replicates=128)
    assert result["mean"].shape == (3,)
    assert math.isfinite(float(result["critical_value"].numpy()))
    assert isinstance(result["contains_zero"], bool)


def test_null_partitions_are_independent_by_contract() -> None:
    # This is intentionally a contract test for the runner's frozen domains.
    from pathlib import Path

    source = Path(__file__).resolve().parents[2] / "docs/benchmarks/run_sir_null_calibrated_predictive_consistency_20260814.py"
    if source.exists():
        text = source.read_text(encoding="utf-8")
        assert "NULL_FIT_DOMAIN = 70" in text
        assert "NULL_CALIBRATION_DOMAIN = 80" in text
        assert "NULL_AUDIT_DOMAIN = 90" in text
        assert "claim\": \"joint_same_parameter_predictive_coverage_only\"" in text
        assert "gaussian_exact_diagnostic" in text
        assert "SIR" not in text.split("nonclaims", 1)[0] or "predictive" in text.split("nonclaims", 1)[0]
        assert "validate_domains()" in text
        assert "frozen_classifier_head_admission_veto" in text
        assert "BLOCKED_HEAD_ADMISSION" in text
        assert 'hard_names = ("finite", "temperature_positive", "optimizer_complete")' in text
        assert "score_interpretability_all_passed" in text


def test_exchangeable_toy_audit_is_not_falsified() -> None:
    fit = tf.random.stateless_normal([500, 3], [31, 1], dtype=tf.float64)
    calibration = tf.random.stateless_normal([200, 3], [31, 2], dtype=tf.float64)
    audit = tf.random.stateless_normal([500, 3], [31, 3], dtype=tf.float64)
    geometry = null.fit_svd_geometry(fit)
    threshold = null.conformal_threshold(geometry.distance(calibration))
    failures = int(tf.reduce_sum(tf.cast(geometry.distance(audit) > threshold, tf.int32)).numpy())
    assert null.audit_failure(failures, 500)["falsified"] is False
