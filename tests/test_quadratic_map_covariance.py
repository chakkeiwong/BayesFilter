from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import (
    ITERATIVE_QUADRATIC_MAP_COVARIANCE_NONCLAIMS,
    QUADRATIC_MAP_COVARIANCE_NONCLAIMS,
    IterativeQuadraticMapCovarianceConfig,
    LowRankSPDQuadraticGeometryConfig,
    QuadraticMapCovarianceLocatorConfig,
    QuadraticMapCovarianceMassConfig,
    estimate_iterative_quadratic_map_covariance,
    estimate_quadratic_map_covariance,
)


def _quadratic_target(
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
        value = 0.5 * tf.reduce_sum(delta * score, axis=1)
        return value, score

    return value_and_score


def _geometry_config(**kwargs) -> LowRankSPDQuadraticGeometryConfig:
    return LowRankSPDQuadraticGeometryConfig(
        rank=kwargs.pop("rank", 2),
        sample_count=kwargs.pop("sample_count", 220),
        pilot_direction_count=kwargs.pop("pilot_direction_count", 256),
        trust_radius=kwargs.pop("trust_radius", 1.0),
        eigenvalue_floor=kwargs.pop("eigenvalue_floor", 0.1),
        max_condition_number=kwargs.pop("max_condition_number", 100.0),
        holdout_rmse_abs_tolerance=kwargs.pop("holdout_rmse_abs_tolerance", 1.0e-4),
        holdout_rmse_rel_tolerance=kwargs.pop("holdout_rmse_rel_tolerance", 1.0e-4),
        seed=kwargs.pop("seed", (2026, 708)),
        **kwargs,
    )


def test_quadratic_initializer_recovers_gaussian_mode_and_covariance() -> None:
    precision = np.diag([2.0, 4.0])
    mode = np.array([0.18, -0.12])

    result = estimate_quadratic_map_covariance(
        _quadratic_target(precision, mode=mode),
        np.array([0.0, 0.0]),
        locator_config=QuadraticMapCovarianceLocatorConfig(enabled=False),
        quadratic_config=_geometry_config(
            rank=1,
            sample_count=220,
            pilot_direction_count=1024,
            holdout_rmse_abs_tolerance=7.0e-2,
        ),
        mass_config=QuadraticMapCovarianceMassConfig(
            jitter=0.0,
            eigenvalue_floor=0.1,
            max_condition_number=100.0,
        ),
    )

    assert result.accepted is True
    assert result.status == "usable"
    assert result.map_candidate_role == "quadratic_surrogate_map_candidate"
    assert result.map_candidate is not None
    np.testing.assert_allclose(result.map_candidate, mode, atol=0.06)
    np.testing.assert_allclose(result.precision, precision, atol=0.08, rtol=0.05)
    np.testing.assert_allclose(
        result.covariance,
        np.linalg.inv(precision),
        atol=0.04,
        rtol=0.06,
    )

    payload = result.payload()
    assert payload["covariance_source"] == (
        "low_rank_spd_quadratic_geometry_precision_theta_coordinates"
    )
    assert payload["diagnostics"]["covariance_authority"] == "covariance_from_precision"
    assert payload["locator_diagnostics"]["uses_optimizer_inverse_hessian"] is False
    assert "not HMC readiness evidence" in payload["nonclaims"]


def test_scaled_quadratic_initializer_returns_original_coordinate_mass() -> None:
    precision_theta = np.diag([4.0, 0.25])
    mode = np.array([0.12, -0.08])
    scale = np.array([0.5, 2.0])
    expected_precision_z = scale[:, np.newaxis] * precision_theta * scale[np.newaxis, :]

    result = estimate_quadratic_map_covariance(
        _quadratic_target(precision_theta, mode=mode),
        np.array([0.0, 0.0]),
        scale=scale,
        locator_config=QuadraticMapCovarianceLocatorConfig(enabled=False),
        quadratic_config=_geometry_config(
            rank=1,
            sample_count=260,
            pilot_direction_count=1024,
            holdout_rmse_abs_tolerance=8.0e-2,
            seed=(13, 14),
        ),
        mass_config=QuadraticMapCovarianceMassConfig(
            jitter=0.0,
            eigenvalue_floor=0.1,
            max_condition_number=100.0,
        ),
    )

    assert result.accepted is True
    assert result.geometry is not None
    np.testing.assert_allclose(
        result.geometry.precision,
        expected_precision_z,
        atol=0.12,
        rtol=0.06,
    )
    np.testing.assert_allclose(result.precision, precision_theta, atol=0.12, rtol=0.06)
    np.testing.assert_allclose(
        result.covariance,
        np.linalg.inv(precision_theta),
        atol=0.04,
        rtol=0.08,
    )
    payload = result.payload()
    assert payload["diagnostics"]["geometry_precision_coordinate_system"] == "z"
    assert payload["diagnostics"]["mass_precision_coordinate_system"] == "theta"
    assert payload["diagnostics"]["mass_covariance_coordinate_system"] == "theta"
    assert payload["diagnostics"]["scale_all_ones"] is False


def test_initializer_forwards_batched_design_callback_without_changing_result() -> None:
    precision = np.diag([2.0, 4.0])
    mode = np.array([0.18, -0.12])
    kwargs = {
        "initial_position": np.zeros(2),
        "locator_config": QuadraticMapCovarianceLocatorConfig(enabled=False),
        "quadratic_config": _geometry_config(
            rank=1,
            sample_count=220,
            pilot_direction_count=256,
            holdout_rmse_abs_tolerance=7.0e-2,
            seed=(17, 18),
        ),
        "mass_config": QuadraticMapCovarianceMassConfig(
            jitter=0.0,
            eigenvalue_floor=0.1,
            max_condition_number=100.0,
        ),
    }
    scalar = estimate_quadratic_map_covariance(
        _quadratic_target(precision, mode=mode),
        **kwargs,
    )
    batched = estimate_quadratic_map_covariance(
        _quadratic_target(precision, mode=mode),
        batched_value_and_score_fn=_batched_quadratic_target(
            precision, mode=mode
        ),
        **kwargs,
    )

    assert scalar.accepted is True
    assert batched.accepted is True
    np.testing.assert_allclose(batched.map_candidate, scalar.map_candidate, atol=1.0e-12)
    np.testing.assert_allclose(batched.precision, scalar.precision, atol=1.0e-12)
    np.testing.assert_allclose(batched.covariance, scalar.covariance, atol=1.0e-12)
    assert batched.geometry is not None
    assert batched.geometry.diagnostics["design_evaluation_route"] == (
        "batched_value_and_score"
    )


def test_iterative_initializer_reaches_terminal_center_with_bounded_steps() -> None:
    precision = np.diag([2.0, 4.0])
    mode = np.array([0.32, -0.24])
    callbacks = []
    fit_starts = []

    result = estimate_iterative_quadratic_map_covariance(
        _quadratic_target(precision, mode=mode),
        np.zeros(2),
        locator_config=QuadraticMapCovarianceLocatorConfig(enabled=False),
        quadratic_config=_geometry_config(
            rank=1,
            sample_count=220,
            pilot_direction_count=512,
            trust_radius=0.1,
            pilot_radius=0.05,
            holdout_rmse_abs_tolerance=7.0e-2,
            constrain_center_refinement_to_trust_region=True,
            seed=(19, 20),
        ),
        mass_config=QuadraticMapCovarianceMassConfig(
            jitter=0.0,
            eigenvalue_floor=0.1,
            max_condition_number=100.0,
        ),
        iterative_config=IterativeQuadraticMapCovarianceConfig(
            max_refinement_steps=8,
            terminal_score_max_abs=1.0e-8,
        ),
        fit_start_callback=lambda index, center: fit_starts.append(
            (index, center.copy())
        ),
        iteration_callback=callbacks.append,
    )

    assert result.accepted is True
    assert result.status == "usable"
    assert result.map_candidate_role == "iterative_terminal_exact_score_center"
    assert result.map_candidate is not None
    np.testing.assert_allclose(result.map_candidate, mode, atol=1.0e-8)
    np.testing.assert_allclose(result.precision, precision, atol=0.08, rtol=0.05)
    np.testing.assert_allclose(
        result.covariance,
        np.linalg.inv(precision),
        atol=0.04,
        rtol=0.06,
    )
    assert len(callbacks) == len(result.iterations)
    assert len(fit_starts) == len(result.iterations)
    for expected_index, ((fit_index, fit_center), row) in enumerate(
        zip(fit_starts, result.iterations, strict=True)
    ):
        assert fit_index == expected_index
        np.testing.assert_array_equal(fit_center, row["center"])
    assert len(result.iterations) >= 3
    assert result.iterations[-1]["terminal_before_fit"] is True
    accepted_moves = result.iterations[:-1]
    assert all(row["center_refinement"]["accepted"] for row in accepted_moves)
    assert all(
        row["center_refinement"]["actual_improvement"] > 0.0
        for row in accepted_moves
    )
    assert all(
        row["center_refinement"]["refined_score_norm"]
        < row["center_refinement"]["center_score_norm"]
        for row in accepted_moves
    )
    assert result.diagnostics["covariance_authority"] == "covariance_from_precision"
    assert tuple(result.payload()["nonclaims"]) == (
        ITERATIVE_QUADRATIC_MAP_COVARIANCE_NONCLAIMS
    )


def test_iterative_initializer_requires_constrained_refinement() -> None:
    with pytest.raises(ValueError, match="requires constrained center refinement"):
        estimate_iterative_quadratic_map_covariance(
            _quadratic_target(np.eye(2)),
            np.zeros(2),
            locator_config=QuadraticMapCovarianceLocatorConfig(enabled=False),
            quadratic_config=_geometry_config(
                rank=1,
                sample_count=80,
                constrain_center_refinement_to_trust_region=False,
            ),
        )


def test_iterative_initializer_fails_closed_at_refinement_budget() -> None:
    result = estimate_iterative_quadratic_map_covariance(
        _quadratic_target(np.eye(2), mode=np.array([0.2, -0.2])),
        np.zeros(2),
        locator_config=QuadraticMapCovarianceLocatorConfig(enabled=False),
        quadratic_config=_geometry_config(
            rank=1,
            sample_count=120,
            pilot_direction_count=256,
            trust_radius=0.05,
            pilot_radius=0.02,
            holdout_rmse_abs_tolerance=7.0e-2,
            constrain_center_refinement_to_trust_region=True,
            seed=(21, 22),
        ),
        iterative_config=IterativeQuadraticMapCovarianceConfig(
            max_refinement_steps=1,
            terminal_score_max_abs=1.0e-8,
        ),
    )

    assert result.accepted is False
    assert result.status == "maximum_refinement_steps_without_terminal_score"
    assert len(result.iterations) == 2
    assert result.precision is None
    assert result.covariance is None


def test_enabled_locator_is_finite_locator_only_not_covariance_authority() -> None:
    precision = np.diag([1.5, 3.0])
    mode = np.array([0.2, -0.1])

    result = estimate_quadratic_map_covariance(
        _quadratic_target(precision, mode=mode),
        np.array([0.0, 0.0]),
        locator_config=QuadraticMapCovarianceLocatorConfig(
            enabled=True,
            max_iterations=20,
        ),
        quadratic_config=_geometry_config(
            rank=1,
            sample_count=220,
            pilot_direction_count=1024,
            holdout_rmse_abs_tolerance=7.0e-2,
            seed=(5, 6),
        ),
        mass_config=QuadraticMapCovarianceMassConfig(
            jitter=0.0,
            eigenvalue_floor=0.1,
            max_condition_number=100.0,
        ),
    )

    assert result.accepted is True
    diagnostics = result.payload()["locator_diagnostics"]
    assert diagnostics["optimizer_role"] == "finite_neighborhood_locator_only"
    assert diagnostics["uses_optimizer_inverse_hessian"] is False
    assert result.payload()["diagnostics"]["optimizer_authority"] == "locator_only"
    assert result.covariance_source == (
        "low_rank_spd_quadratic_geometry_precision_theta_coordinates"
    )


def test_locator_exception_falls_back_to_initial_position_for_geometry(monkeypatch) -> None:
    from bayesfilter.inference import quadratic_map_covariance as qmc

    def raising_lbfgs(*args, **kwargs):
        raise RuntimeError("synthetic locator-only failure")

    monkeypatch.setattr(qmc.tfp.optimizer, "lbfgs_minimize", raising_lbfgs)

    result = estimate_quadratic_map_covariance(
        _quadratic_target(np.eye(2)),
        np.zeros(2),
        locator_config=QuadraticMapCovarianceLocatorConfig(
            enabled=True,
            max_iterations=2,
        ),
        quadratic_config=_geometry_config(
            rank=1,
            sample_count=24,
            min_samples_per_parameter=2,
            holdout_fraction=0.0,
            seed=(7, 8),
        ),
    )

    assert result.locator_diagnostics["accepted_optimizer_position"] is False
    assert np.array_equal(result.locator_position, np.zeros(2))
    assert result.diagnostics["optimizer_authority"] == "locator_only"
    assert result.covariance_source == (
        "low_rank_spd_quadratic_geometry_precision_theta_coordinates"
    )


def test_nonfinite_initial_position_target_fails_closed() -> None:
    def value_and_score(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        theta = tf.reshape(tf.convert_to_tensor(theta, dtype=tf.float64), [-1])
        return tf.constant(np.nan, dtype=tf.float64), -theta

    result = estimate_quadratic_map_covariance(
        value_and_score,
        np.zeros(2),
        locator_config=QuadraticMapCovarianceLocatorConfig(enabled=False),
        quadratic_config=_geometry_config(rank=1, sample_count=120),
    )

    assert result.accepted is False
    assert result.status == "initial_value_or_score_nonfinite"
    assert result.precision is None
    assert result.covariance is None
    assert result.covariance_source is None


def test_insufficient_samples_reject_without_mass_matrix() -> None:
    result = estimate_quadratic_map_covariance(
        _quadratic_target(np.eye(3)),
        np.zeros(3),
        locator_config=QuadraticMapCovarianceLocatorConfig(enabled=False),
        quadratic_config=LowRankSPDQuadraticGeometryConfig(
            rank=1,
            sample_count=10,
            min_samples_per_parameter=5,
            seed=(9, 10),
        ),
    )

    assert result.accepted is False
    assert result.status == "geometry_insufficient_finite_samples"
    assert result.mass_matrix is None
    assert result.payload()["geometry"]["status"] == "insufficient_finite_samples"


def test_payload_nonclaims_and_exported_constant_are_consistent() -> None:
    assert QUADRATIC_MAP_COVARIANCE_NONCLAIMS[-1] == (
        "not source-faithful Zhao-Cui evidence"
    )
    result = estimate_quadratic_map_covariance(
        _quadratic_target(np.eye(1)),
        np.zeros(1),
        locator_config=QuadraticMapCovarianceLocatorConfig(enabled=False),
        quadratic_config=_geometry_config(
            rank=1,
            sample_count=80,
            pilot_direction_count=32,
            seed=(11, 12),
        ),
    )

    payload = result.payload()
    assert tuple(payload["nonclaims"]) == QUADRATIC_MAP_COVARIANCE_NONCLAIMS
    assert payload["diagnostics"]["reports_hmc_convergence"] is False
