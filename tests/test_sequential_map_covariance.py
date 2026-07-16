from __future__ import annotations

import json

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import (
    SEQUENTIAL_MAP_COVARIANCE_NONCLAIMS,
    SequentialMapCovarianceConfig,
    estimate_sequential_map_covariance,
)


def _quadratic_target(precision: np.ndarray, mode: np.ndarray):
    precision_tf = tf.constant(precision, dtype=tf.float64)
    mode_tf = tf.constant(mode, dtype=tf.float64)

    def value_and_score(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        theta = tf.convert_to_tensor(theta, dtype=tf.float64)
        delta = theta - mode_tf
        score = -tf.linalg.matvec(precision_tf, delta)
        value = 0.5 * tf.reduce_sum(delta * score)
        return value, score

    return value_and_score


def _batched_quadratic_target(precision: np.ndarray, mode: np.ndarray):
    precision_tf = tf.constant(precision, dtype=tf.float64)
    mode_tf = tf.constant(mode, dtype=tf.float64)

    def value_and_score(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        delta = tf.convert_to_tensor(theta, tf.float64) - mode_tf[None, :]
        score = -tf.einsum("ij,bj->bi", precision_tf, delta)
        return 0.5 * tf.reduce_sum(delta * score, axis=1), score

    return value_and_score


def test_rotated_quadratic_recovers_mode_and_fresh_covariance() -> None:
    rotation = np.array([[0.8, -0.6], [0.6, 0.8]])
    precision = rotation @ np.diag([2.0, 7.0]) @ rotation.T
    mode = np.array([0.35, -0.22])
    scale = np.array([0.5, 2.0])
    config = SequentialMapCovarianceConfig(
        terminal_score_max_abs=1.0e-8,
        initial_radius=0.2,
        search_sample_count=8,
        regression_sample_count=18,
        terminal_sample_count=18,
        max_attempts=4,
        max_exact_evaluations=256,
        seed=(2026, 715),
    )

    first = estimate_sequential_map_covariance(
        _quadratic_target(precision, mode),
        [np.zeros(2)],
        scale=scale,
        config=config,
    )
    second = estimate_sequential_map_covariance(
        _quadratic_target(precision, mode),
        [np.zeros(2)],
        scale=scale,
        config=config,
    )

    assert first.accepted is True
    assert first.status == "usable"
    np.testing.assert_allclose(first.map_candidate, mode, atol=1.0e-8)
    np.testing.assert_allclose(first.precision, precision, atol=1.0e-7)
    np.testing.assert_allclose(first.covariance, np.linalg.inv(precision), atol=1.0e-7)
    assert abs(first.precision[0, 1]) > 1.0
    assert first.diagnostics["terminal_fit_fresh"] is True
    assert first.diagnostics["terminal_seed"] != first.diagnostics["search_seed"]
    assert first.diagnostics["exact_evaluations"] <= config.max_exact_evaluations
    assert first.diagnostics["precision_coordinate_system"] == "theta"
    assert first.diagnostics["regression_coordinate_system"] == "z"
    assert first.payload() == second.payload()
    assert tuple(first.payload()["nonclaims"]) == SEQUENTIAL_MAP_COVARIANCE_NONCLAIMS


def test_nonstationary_locator_fails_closed_at_evaluation_budget() -> None:
    mode = np.array([0.4, -0.3])

    def quartic(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        delta = tf.convert_to_tensor(theta, tf.float64) - tf.constant(mode, tf.float64)
        return -tf.reduce_sum(delta**4), -4.0 * delta**3

    result = estimate_sequential_map_covariance(
        quartic,
        [np.zeros(2)],
        config=SequentialMapCovarianceConfig(
            terminal_score_max_abs=1.0e-12,
            locator_max_iterations=1,
            max_exact_evaluations=64,
        ),
    )

    assert result.accepted is False
    assert result.status == "maximum_exact_evaluations"
    assert result.map_candidate is not None
    assert result.precision is None
    assert result.covariance is None


def test_malformed_score_fails_closed() -> None:
    def malformed(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        del theta
        return tf.constant(0.0, tf.float64), tf.zeros([3], tf.float64)

    with pytest.raises(ValueError, match="one entry per parameter"):
        estimate_sequential_map_covariance(malformed, [np.zeros(2)])


def test_nonlinear_canary_recovers_after_truncated_locator() -> None:
    def rosenbrock(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        x, y = tf.unstack(tf.convert_to_tensor(theta, tf.float64))
        objective = (1.0 - x) ** 2 + 20.0 * (y - x**2) ** 2
        gradient_x = -2.0 * (1.0 - x) - 80.0 * x * (y - x**2)
        gradient_y = 40.0 * (y - x**2)
        return -objective, -tf.stack([gradient_x, gradient_y])

    result = estimate_sequential_map_covariance(
        rosenbrock,
        [np.zeros(2)],
        config=SequentialMapCovarianceConfig(
            terminal_score_max_abs=2.0e-2,
            initial_radius=0.25,
            search_sample_count=16,
            regression_sample_count=48,
            terminal_sample_count=48,
            locator_max_iterations=1,
            max_attempts=8,
            max_exact_evaluations=800,
            score_holdout_relative_rmse=0.8,
            max_stalled_attempts=5,
            seed=(2026, 716),
        ),
    )

    assert result.accepted is True
    assert result.diagnostics["terminal_max_abs_scaled_score"] <= 2.0e-2
    assert result.diagnostics["history"]
    assert any(
        row.get("recentered")
        or row.get("radius_action") == "contract"
        or row.get("action") == "proposal_rejected"
        for row in result.diagnostics["history"]
    )
    assert result.diagnostics["terminal_fit"]["status"] == "usable"


def test_rank_deficient_terminal_fit_fails_closed() -> None:
    result = estimate_sequential_map_covariance(
        _quadratic_target(np.eye(3), np.zeros(3)),
        [np.zeros(3)],
        config=SequentialMapCovarianceConfig(
            terminal_score_max_abs=1.0e-8,
            terminal_sample_count=2,
            max_attempts=1,
            max_exact_evaluations=64,
        ),
    )
    assert result.accepted is False
    assert result.status == "sequential_refinement_without_terminal_geometry"


def test_terminal_projection_veto_fails_closed() -> None:
    def saddle(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        x, y = tf.unstack(tf.convert_to_tensor(theta, tf.float64))
        return -x**2 + y**2, tf.stack([-2.0 * x, 2.0 * y])

    result = estimate_sequential_map_covariance(
        saddle,
        [np.zeros(2)],
        config=SequentialMapCovarianceConfig(
            terminal_score_max_abs=1.0e-8,
            terminal_sample_count=24,
            terminal_projection_relative_frobenius_cap=1.0e-6,
            max_exact_evaluations=128,
        ),
    )
    assert result.accepted is False
    assert result.status == "terminal_projection_exceeds_cap"


def test_no_finite_start_fails_closed() -> None:
    def nonfinite(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        theta = tf.convert_to_tensor(theta, tf.float64)
        return tf.constant(float("nan"), tf.float64), tf.fill(
            tf.shape(theta), tf.constant(float("nan"), tf.float64)
        )

    result = estimate_sequential_map_covariance(nonfinite, [np.zeros(2)])
    assert result.accepted is False
    assert result.status == "no_finite_locator_candidate"


def test_locator_uses_start_centered_standardized_coordinates() -> None:
    visited = []

    def target(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        theta = tf.convert_to_tensor(theta, tf.float64)
        visited.append(np.asarray(theta.numpy(), dtype=float))
        mode = tf.constant([1000.0, 1.0e-3], tf.float64)
        scale = tf.constant([100.0, 1.0e-4], tf.float64)
        delta = (theta - mode) / scale
        return -0.5 * tf.reduce_sum(delta**2), -delta / scale

    result = estimate_sequential_map_covariance(
        target,
        [np.array([900.0, 0.9e-3])],
        scale=np.array([100.0, 1.0e-4]),
        config=SequentialMapCovarianceConfig(
            terminal_score_max_abs=1.0e-8,
            terminal_sample_count=18,
            max_exact_evaluations=128,
        ),
    )
    assert result.accepted is True
    assert result.diagnostics["locator"][0]["coordinate_system"] == (
        "start_centered_prior_standardized_smooth_box"
    )
    start = np.array([900.0, 0.9e-3])
    scale = np.array([100.0, 1.0e-4])
    standardized = np.asarray([(row - start) / scale for row in visited])
    assert np.max(np.abs(standardized)) <= 4.0 + 1.0e-12


def test_batched_cloud_route_matches_scalar_result() -> None:
    precision = np.array([[3.0, 0.7], [0.7, 2.0]])
    mode = np.array([0.2, -0.1])
    config = SequentialMapCovarianceConfig(
        terminal_score_max_abs=1.0e-8,
        terminal_sample_count=18,
        max_exact_evaluations=128,
        seed=(2026, 718),
    )
    scalar = estimate_sequential_map_covariance(
        _quadratic_target(precision, mode), [np.zeros(2)], config=config
    )
    batched = estimate_sequential_map_covariance(
        _quadratic_target(precision, mode),
        [np.zeros(2)],
        batched_value_and_score_fn=_batched_quadratic_target(precision, mode),
        config=config,
    )
    assert scalar.accepted is True and batched.accepted is True
    np.testing.assert_allclose(batched.map_candidate, scalar.map_candidate, atol=1.0e-12)
    np.testing.assert_allclose(batched.precision, scalar.precision, atol=1.0e-10)
    np.testing.assert_allclose(batched.covariance, scalar.covariance, atol=1.0e-12)
    assert batched.diagnostics["exact_evaluations"] == scalar.diagnostics["exact_evaluations"]


def test_native_batched_locator_recovers_same_quadratic_mode() -> None:
    precision = np.array([[2.5, 0.4], [0.4, 1.5]])
    mode = np.array([0.15, -0.08])
    starts = [np.zeros(2), np.array([-0.3, 0.2])]
    config = SequentialMapCovarianceConfig(
        terminal_score_max_abs=1.0e-8,
        terminal_sample_count=18,
        max_exact_evaluations=256,
        seed=(2026, 719),
    )
    result = estimate_sequential_map_covariance(
        _quadratic_target(precision, mode),
        starts,
        batched_value_and_score_fn=_batched_quadratic_target(precision, mode),
        batched_locator_value_and_score_fn=_batched_quadratic_target(
            precision, mode
        ),
        config=config,
    )
    assert result.accepted is True
    np.testing.assert_allclose(result.map_candidate, mode, atol=1.0e-8)
    assert all(row["native_batched_locator"] for row in result.diagnostics["locator"])


def test_locator_gradient_tolerance_is_explicit_and_recorded() -> None:
    precision = np.array([[2.5, 0.4], [0.4, 1.5]])
    mode = np.array([0.15, -0.08])
    config = SequentialMapCovarianceConfig(
        terminal_score_max_abs=1.0e-8,
        locator_gradient_tolerance=0.05,
        terminal_sample_count=18,
        max_exact_evaluations=256,
        seed=(2026, 720),
    )
    result = estimate_sequential_map_covariance(
        _quadratic_target(precision, mode),
        [np.zeros(2), np.array([-0.3, 0.2])],
        batched_value_and_score_fn=_batched_quadratic_target(precision, mode),
        batched_locator_value_and_score_fn=_batched_quadratic_target(
            precision, mode
        ),
        config=config,
    )

    assert result.accepted is True
    assert result.diagnostics["terminal_max_abs_scaled_score"] <= 1.0e-8
    assert all(
        row["gradient_tolerance"] == 0.05
        for row in result.diagnostics["locator"]
    )


def test_locator_gradient_tolerance_must_be_positive_finite() -> None:
    with pytest.raises(ValueError, match="locator_gradient_tolerance"):
        SequentialMapCovarianceConfig(locator_gradient_tolerance=0.0)


def test_progress_callback_records_locator_and_terminal_stages() -> None:
    events = []
    precision = np.array([[2.5, 0.4], [0.4, 1.5]])
    mode = np.array([0.15, -0.08])
    result = estimate_sequential_map_covariance(
        _quadratic_target(precision, mode),
        [np.zeros(2), np.array([-0.3, 0.2])],
        batched_value_and_score_fn=_batched_quadratic_target(precision, mode),
        batched_locator_value_and_score_fn=_batched_quadratic_target(
            precision, mode
        ),
        config=SequentialMapCovarianceConfig(
            terminal_score_max_abs=1.0e-8,
            terminal_sample_count=18,
            max_exact_evaluations=256,
            seed=(2026, 721),
        ),
        progress_callback=events.append,
    )

    assert result.accepted is True
    stages = [event["stage"] for event in events]
    assert stages[0] == "initializer_started"
    assert "locator_objective_completed" in stages
    assert "locator_completed" in stages
    assert "candidate_selected" in stages
    assert "terminal_fit_started" in stages
    assert "terminal_fit_completed" in stages
    assert stages[-1] == "initializer_completed"
    json.dumps(events)


def test_locator_stopping_condition_is_explicit_and_validated() -> None:
    config = SequentialMapCovarianceConfig(
        locator_stopping_condition="converged_any"
    )
    assert config.locator_stopping_condition == "converged_any"
    with pytest.raises(ValueError, match="locator_stopping_condition"):
        SequentialMapCovarianceConfig(locator_stopping_condition="automatic")


def test_center_first_stationary_center_skips_locator_and_fits_geometry() -> None:
    events = []
    precision = np.array([[3.0, 0.4], [0.4, 2.0]])
    mode = np.array([0.2, -0.1])
    result = estimate_sequential_map_covariance(
        _quadratic_target(precision, mode),
        [mode],
        batched_value_and_score_fn=_batched_quadratic_target(precision, mode),
        batched_locator_value_and_score_fn=lambda theta: (_ for _ in ()).throw(
            AssertionError("center-first must not call the locator")
        ),
        config=SequentialMapCovarianceConfig(
            locator_policy="center_first",
            terminal_score_max_abs=1.0e-8,
            terminal_sample_count=18,
            max_exact_evaluations=128,
            seed=(2026, 722),
        ),
        progress_callback=events.append,
    )

    assert result.accepted is True
    np.testing.assert_allclose(result.map_candidate, mode, atol=1.0e-12)
    np.testing.assert_allclose(result.precision, precision, atol=1.0e-10)
    assert result.diagnostics["locator"] == [
        {
            "finite": True,
            "coordinate_system": "reviewed_exact_center",
            "locator_policy": "center_first",
            "locator_skipped": True,
            "skip_reason": "exact_center_admission",
        }
    ]
    assert "locator_skipped_center_first" in [event["stage"] for event in events]
    assert "locator_objective_completed" not in [event["stage"] for event in events]


def test_center_first_nonstationary_center_uses_local_refinement() -> None:
    precision = np.array([[2.5, 0.3], [0.3, 1.5]])
    mode = np.array([0.15, -0.08])
    result = estimate_sequential_map_covariance(
        _quadratic_target(precision, mode),
        [np.zeros(2)],
        batched_value_and_score_fn=_batched_quadratic_target(precision, mode),
        config=SequentialMapCovarianceConfig(
            locator_policy="center_first",
            terminal_score_max_abs=1.0e-8,
            search_sample_count=8,
            regression_sample_count=18,
            terminal_sample_count=18,
            max_attempts=4,
            max_exact_evaluations=256,
            seed=(2026, 723),
        ),
    )

    assert result.accepted is True
    np.testing.assert_allclose(result.map_candidate, mode, atol=1.0e-8)
    assert result.diagnostics["history"]


def test_center_first_requires_one_center_and_policy_is_validated() -> None:
    with pytest.raises(ValueError, match="exactly one center"):
        estimate_sequential_map_covariance(
            _quadratic_target(np.eye(2), np.zeros(2)),
            [np.zeros(2), np.ones(2)],
            config=SequentialMapCovarianceConfig(locator_policy="center_first"),
        )
    with pytest.raises(ValueError, match="locator_policy"):
        SequentialMapCovarianceConfig(locator_policy="automatic")


def test_default_locator_policy_remains_multistart() -> None:
    config = SequentialMapCovarianceConfig()
    assert config.locator_policy == "multistart"


def test_terminal_fit_attempt_cap_is_enforced() -> None:
    calls = 0

    def flat_quartic(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        nonlocal calls
        calls += 1
        value = tf.convert_to_tensor(theta, tf.float64)
        # The center is stationary, but a quartic score has no usable local
        # linear curvature at that center, so every terminal fit is rejected.
        return -tf.reduce_sum(value**4), -4.0 * value**3

    result = estimate_sequential_map_covariance(
        flat_quartic,
        [np.zeros(2)],
        config=SequentialMapCovarianceConfig(
            locator_policy="center_first",
            terminal_score_max_abs=1.0e-8,
            terminal_sample_count=18,
            max_attempts=8,
            max_exact_evaluations=512,
            score_holdout_relative_rmse=1.0e-12,
            max_terminal_fit_attempts=2,
        ),
    )

    assert result.accepted is False
    terminal_rows = [
        row
        for row in result.diagnostics["history"]
        if row.get("action") == "terminal_fit_rejected"
    ]
    assert len(terminal_rows) == 2
    assert calls > 0


def test_terminal_fit_attempt_cap_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_terminal_fit_attempts"):
        SequentialMapCovarianceConfig(max_terminal_fit_attempts=0)
