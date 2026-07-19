from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference.quadratic_geometry import (
    LOW_RANK_SPD_QUADRATIC_GEOMETRY_NONCLAIMS,
    LowRankSPDQuadraticGeometryConfig,
    _solve_spd_quadratic_trust_region,
    fit_low_rank_spd_quadratic_geometry,
)


def _quadratic_target(
    precision: np.ndarray,
    *,
    mode: np.ndarray | None = None,
    shift: float = 0.0,
):
    precision_tf = tf.constant(precision, dtype=tf.float64)
    mode_tf = tf.zeros([precision.shape[0]], dtype=tf.float64) if mode is None else tf.constant(mode, dtype=tf.float64)

    def value_and_score(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        theta = tf.reshape(tf.convert_to_tensor(theta, dtype=tf.float64), [-1])
        delta = theta - mode_tf
        value = tf.constant(float(shift), dtype=tf.float64) - 0.5 * tf.tensordot(
            delta,
            tf.linalg.matvec(precision_tf, delta),
            axes=1,
        )
        score = -tf.linalg.matvec(precision_tf, delta)
        return value, score

    return value_and_score


def _batched_quadratic_target(
    precision: np.ndarray,
    *,
    mode: np.ndarray | None = None,
    shift: float = 0.0,
):
    precision_tf = tf.constant(precision, dtype=tf.float64)
    mode_tf = (
        tf.zeros([precision.shape[0]], dtype=tf.float64)
        if mode is None
        else tf.constant(mode, dtype=tf.float64)
    )

    def value_and_score(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        theta = tf.convert_to_tensor(theta, dtype=tf.float64)
        delta = theta - mode_tf[tf.newaxis, :]
        score = -tf.einsum("ij,bj->bi", precision_tf, delta)
        value = tf.constant(float(shift), dtype=tf.float64) + 0.5 * tf.reduce_sum(
            delta * score, axis=1
        )
        return value, score

    return value_and_score


def test_synthetic_low_rank_spd_quadratic_recovers_precision() -> None:
    q = np.eye(4, 2)
    true_precision = 1.4 * np.eye(4) + (q * np.array([2.0, 0.7])) @ q.T
    result = fit_low_rank_spd_quadratic_geometry(
        _quadratic_target(true_precision),
        np.zeros(4),
        config=LowRankSPDQuadraticGeometryConfig(
            rank=2,
            sample_count=260,
            pilot_direction_count=512,
            trust_radius=0.8,
            eigenvalue_floor=0.2,
            max_condition_number=100.0,
            holdout_rmse_abs_tolerance=5.0e-2,
            holdout_rmse_rel_tolerance=5.0e-2,
            seed=(11, 22),
        ),
    )

    assert result.accepted is True
    assert result.status == "usable"
    assert result.precision is not None
    np.testing.assert_allclose(result.precision, true_precision, rtol=0.12, atol=0.18)
    assert result.payload()["precision_eigen_summary"]["positive"] is True
    assert result.payload()["diagnostics"]["holdout_rmse"] < 5.0e-2
    assert result.payload()["diagnostics"]["finite_sample_count"] >= 5 * (
        1 + 4 + 1 + 2
    )


def test_undersampled_regression_is_rejected() -> None:
    result = fit_low_rank_spd_quadratic_geometry(
        _quadratic_target(np.eye(3)),
        np.zeros(3),
        config=LowRankSPDQuadraticGeometryConfig(
            rank=1,
            sample_count=10,
            min_samples_per_parameter=5,
            seed=(1, 2),
        ),
    )

    assert result.accepted is False
    assert result.status == "insufficient_finite_samples"
    diagnostics = result.payload()["diagnostics"]
    assert diagnostics["finite_sample_count"] < diagnostics["required_finite_samples"]


def test_spd_and_condition_cap_are_enforced_by_construction() -> None:
    result = fit_low_rank_spd_quadratic_geometry(
        _quadratic_target(np.diag([1.0, 4.0, 9.0])),
        np.zeros(3),
        config=LowRankSPDQuadraticGeometryConfig(
            rank=2,
            sample_count=180,
            eigenvalue_floor=0.5,
            max_condition_number=3.0,
            holdout_rmse_abs_tolerance=10.0,
            seed=(3, 4),
        ),
    )

    assert result.accepted is True
    summary = result.payload()["precision_eigen_summary"]
    assert summary["positive"] is True
    assert summary["min"] >= 0.5 - 1.0e-8
    assert summary["condition_number"] <= 3.0 * (1.0 + 1.0e-6)


def test_nonfinite_values_do_not_silently_pass_sample_gate() -> None:
    def value_and_score(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        theta = tf.reshape(tf.convert_to_tensor(theta, dtype=tf.float64), [-1])
        value = -0.5 * tf.reduce_sum(tf.square(theta))
        value = tf.where(theta[0] > 0.0, tf.constant(np.nan, dtype=tf.float64), value)
        return value, -theta

    result = fit_low_rank_spd_quadratic_geometry(
        value_and_score,
        np.zeros(2),
        config=LowRankSPDQuadraticGeometryConfig(
            rank=1,
            sample_count=20,
            min_samples_per_parameter=5,
            seed=(5, 6),
        ),
    )

    assert result.accepted is False
    assert result.status == "insufficient_finite_samples"
    diagnostics = result.payload()["diagnostics"]
    assert diagnostics["nonfinite_sample_count"] > 0


def test_bad_holdout_fit_is_rejected() -> None:
    def quartic_target(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        theta = tf.reshape(tf.convert_to_tensor(theta, dtype=tf.float64), [-1])
        value = -tf.reduce_sum(tf.pow(theta, 4))
        score = -4.0 * tf.pow(theta, 3)
        return value, score

    result = fit_low_rank_spd_quadratic_geometry(
        quartic_target,
        np.zeros(2),
        config=LowRankSPDQuadraticGeometryConfig(
            rank=1,
            sample_count=100,
            trust_radius=1.0,
            eigenvalue_floor=0.1,
            holdout_rmse_abs_tolerance=1.0e-8,
            holdout_rmse_rel_tolerance=1.0e-8,
            seed=(7, 8),
        ),
    )

    assert result.accepted is False
    assert result.status == "holdout_fit_rejected"
    assert result.payload()["diagnostics"]["holdout_passed"] is False


def test_center_refinement_accepts_nearby_mode() -> None:
    precision = np.diag([2.0, 3.0])
    mode = np.array([0.1, -0.05])
    result = fit_low_rank_spd_quadratic_geometry(
        _quadratic_target(precision, mode=mode),
        np.zeros(2),
        config=LowRankSPDQuadraticGeometryConfig(
            rank=1,
            sample_count=160,
            trust_radius=1.0,
            eigenvalue_floor=0.2,
            max_condition_number=20.0,
            holdout_rmse_abs_tolerance=1.0e-4,
            seed=(9, 10),
        ),
    )

    assert result.accepted is True
    assert result.center_refinement_accepted is True
    assert result.refined_center is not None
    np.testing.assert_allclose(result.refined_center, mode, atol=0.08)


def test_center_refinement_rejects_out_of_trust_mode() -> None:
    precision = np.eye(2)
    mode = np.array([3.0, 0.0])
    result = fit_low_rank_spd_quadratic_geometry(
        _quadratic_target(precision, mode=mode),
        np.zeros(2),
        config=LowRankSPDQuadraticGeometryConfig(
            rank=1,
            sample_count=160,
            trust_radius=0.5,
            eigenvalue_floor=0.2,
            max_condition_number=20.0,
            holdout_rmse_abs_tolerance=1.0e-4,
            seed=(12, 13),
        ),
    )

    assert result.accepted is True
    assert result.center_refinement_accepted is False
    assert result.refined_center is None
    assert "outside_trust_radius" in result.payload()["diagnostics"]["center_refinement"]["reason"]


def test_seed_and_payload_are_deterministic() -> None:
    kwargs = {
        "value_and_score_fn": _quadratic_target(np.diag([1.5, 2.0, 3.0])),
        "center": np.zeros(3),
        "config": LowRankSPDQuadraticGeometryConfig(
            rank=2,
            sample_count=180,
            seed=(123, 456),
        ),
    }
    result_a = fit_low_rank_spd_quadratic_geometry(**kwargs)
    result_b = fit_low_rank_spd_quadratic_geometry(**kwargs)

    assert result_a.status == result_b.status
    assert result_a.payload()["diagnostics"]["artifact_hash"] == result_b.payload()["diagnostics"]["artifact_hash"]
    assert LOW_RANK_SPD_QUADRATIC_GEOMETRY_NONCLAIMS[-1] == (
        "not source-faithful Zhao-Cui evidence"
    )


def test_batched_design_route_matches_scalar_geometry() -> None:
    precision = np.array(
        [[3.0, 0.25, 0.0], [0.25, 1.5, 0.1], [0.0, 0.1, 0.8]],
        dtype=float,
    )
    mode = np.array([0.08, -0.04, 0.03])
    config = LowRankSPDQuadraticGeometryConfig(
        rank=2,
        sample_count=180,
        pilot_direction_count=96,
        trust_radius=0.8,
        eigenvalue_floor=0.1,
        max_condition_number=100.0,
        holdout_rmse_abs_tolerance=0.1,
        holdout_rmse_rel_tolerance=1.0e-8,
        seed=(2026, 714),
    )
    scalar = fit_low_rank_spd_quadratic_geometry(
        _quadratic_target(precision, mode=mode),
        np.zeros(3),
        config=config,
    )
    batched = fit_low_rank_spd_quadratic_geometry(
        _quadratic_target(precision, mode=mode),
        np.zeros(3),
        batched_value_and_score_fn=_batched_quadratic_target(
            precision, mode=mode
        ),
        config=config,
    )

    assert scalar.accepted is True
    assert batched.accepted is True
    np.testing.assert_allclose(batched.precision, scalar.precision, atol=1.0e-12)
    np.testing.assert_allclose(batched.covariance, scalar.covariance, atol=1.0e-12)
    np.testing.assert_allclose(
        batched.refined_center, scalar.refined_center, atol=1.0e-12
    )
    scalar_diagnostics = scalar.payload()["diagnostics"]
    batched_diagnostics = batched.payload()["diagnostics"]
    assert scalar_diagnostics["design_evaluation_route"] == (
        "scalar_value_and_score_loop"
    )
    assert batched_diagnostics["design_evaluation_route"] == (
        "batched_value_and_score"
    )
    assert batched_diagnostics["pilot"]["evaluation_route"] == (
        "batched_value_and_score"
    )
    assert batched_diagnostics["pilot"]["evaluation_batch_size"] == 192
    assert batched_diagnostics["artifact_hash"] == scalar_diagnostics["artifact_hash"]


def test_malformed_batched_design_output_fails_closed() -> None:
    def malformed(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        theta = tf.convert_to_tensor(theta, dtype=tf.float64)
        return tf.zeros([1], tf.float64), tf.zeros_like(theta)

    result = fit_low_rank_spd_quadratic_geometry(
        _quadratic_target(np.eye(2)),
        np.zeros(2),
        batched_value_and_score_fn=malformed,
        config=LowRankSPDQuadraticGeometryConfig(
            rank=1,
            sample_count=40,
            min_samples_per_parameter=2,
            pilot_direction_count=16,
            seed=(15, 16),
        ),
    )

    assert result.accepted is False
    assert result.status == "insufficient_finite_samples"
    assert result.diagnostics["finite_sample_count"] == 0
    assert result.diagnostics["design_evaluation_route"] == (
        "batched_value_and_score"
    )


def test_spd_quadratic_trust_region_uses_interior_newton_step() -> None:
    precision = np.diag([2.0, 4.0])
    linear = np.array([0.2, -0.4])

    result = _solve_spd_quadratic_trust_region(
        precision,
        linear,
        radius=0.5,
    )

    assert result["status"] == "usable"
    assert result["boundary_active"] is False
    assert result["lagrange_multiplier"] == 0.0
    np.testing.assert_allclose(result["step"], [0.1, -0.1], atol=1.0e-14)
    assert result["predicted_improvement"] > 0.0


def test_spd_quadratic_trust_region_solves_boundary_not_component_clip() -> None:
    precision = np.diag([1.0, 4.0])
    linear = np.array([2.0, 1.0])
    radius = 0.3

    result = _solve_spd_quadratic_trust_region(
        precision,
        linear,
        radius=radius,
    )

    assert result["status"] == "usable"
    assert result["boundary_active"] is True
    assert result["lagrange_multiplier"] > 0.0
    np.testing.assert_allclose(np.linalg.norm(result["step"]), radius, atol=1.0e-12)
    assert not np.array_equal(result["step"], np.clip([2.0, 0.25], -radius, radius))
    residual = (
        precision + result["lagrange_multiplier"] * np.eye(2)
    ) @ result["step"] - linear
    np.testing.assert_allclose(residual, np.zeros(2), atol=2.0e-12)


def test_constrained_refinement_accepts_exact_boundary_quadratic_step() -> None:
    precision = np.diag([2.0, 4.0])
    mode = np.array([0.3, -0.2])
    result = fit_low_rank_spd_quadratic_geometry(
        _quadratic_target(precision, mode=mode),
        np.zeros(2),
        config=LowRankSPDQuadraticGeometryConfig(
            rank=1,
            sample_count=180,
            pilot_direction_count=512,
            trust_radius=0.1,
            pilot_radius=0.05,
            eigenvalue_floor=0.1,
            max_condition_number=100.0,
            holdout_rmse_abs_tolerance=5.0e-2,
            holdout_rmse_rel_tolerance=1.0e-8,
            constrain_center_refinement_to_trust_region=True,
            seed=(2026, 715),
        ),
    )

    assert result.accepted is True
    assert result.center_refinement_accepted is True
    refinement = result.diagnostics["center_refinement"]
    assert refinement["step_method"] == "exact_spd_quadratic_trust_region"
    assert refinement["boundary_active"] is True
    assert refinement["actual_improvement"] > 0.0
    assert refinement["predicted_improvement"] > 0.0
    np.testing.assert_allclose(refinement["z_norm"], 0.1, atol=1.0e-10)


def test_default_refinement_policy_remains_unconstrained() -> None:
    config = LowRankSPDQuadraticGeometryConfig()
    assert config.constrain_center_refinement_to_trust_region is False
    assert config.payload()["constrain_center_refinement_to_trust_region"] is False
