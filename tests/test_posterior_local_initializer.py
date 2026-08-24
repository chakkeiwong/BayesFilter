from __future__ import annotations

import json

import numpy as np
import tensorflow as tf

from bayesfilter.inference.fixed_center_curvature import (
    FixedCenterCurvatureThresholds,
)
from bayesfilter.inference.joint_center import JointCenterLocatorConfig
from bayesfilter.inference.posterior_local_initializer import (
    POSTERIOR_LOCAL_INITIALIZER_NONCLAIMS,
    PosteriorLocalInitializerConfig,
    initialize_posterior_local_location_scale,
)
from bayesfilter.inference.quadratic_geometry import (
    LowRankSPDQuadraticGeometryConfig,
)


def _thresholds(dimension: int) -> FixedCenterCurvatureThresholds:
    return FixedCenterCurvatureThresholds(
        selection_holdout_relative_rmse_cap=1.0e-8,
        audit_relative_rmse_cap=1.0e-8,
        projection_relative_frobenius_cap=1.0e-8,
        generalized_eigenvalue_spread_cap=1.01,
        trace_normalized_frobenius_cap=1.0e-7,
        trace_normalized_operator_cap=1.0e-7,
        principal_angle_degrees_cap=1.0e-5,
        principal_subspace_rank=dimension,
        require_raw_spd=True,
    )


def _movement_config(seed: int = 20260825) -> LowRankSPDQuadraticGeometryConfig:
    return LowRankSPDQuadraticGeometryConfig(
        rank=1,
        sample_count=40,
        min_samples_per_parameter=1,
        trust_radius=1.0,
        pilot_radius=0.2,
        holdout_fraction=0.25,
        eigenvalue_floor=1.0e-4,
        max_condition_number=1.0e6,
        fit_max_iterations=100,
        fit_tolerance=1.0e-10,
        # Movement geometry is deliberately low rank and need only be a stable
        # trust-region proposal model. Exact covariance recovery is required
        # below from the separate terminal fixed-center fit.
        holdout_rmse_abs_tolerance=2.0e-1,
        holdout_rmse_rel_tolerance=2.0e-1,
        center_score_improvement_factor=0.999,
        center_log_prob_tolerance=1.0e-10,
        constrain_center_refinement_to_trust_region=True,
        seed=seed,
    )


def _initializer_config(
    *, locator_gradient_tolerance: float = 1.0e-10
) -> PosteriorLocalInitializerConfig:
    return PosteriorLocalInitializerConfig(
        locator_box_radius=4.0,
        locator_config=JointCenterLocatorConfig(
            max_iterations=40,
            max_line_search_iterations=20,
            gradient_tolerance=locator_gradient_tolerance,
            max_objective_evaluations=401,
            jit_compile=False,
        ),
        min_movement_fits=2,
        max_movement_attempts=3,
        max_curvature_attempts=2,
        curvature_radius=0.08,
        training_rows_per_replicate=8,
        selection_rows_per_replicate=8,
        audit_rows=4,
        factor_max=1,
        max_exact_evaluations=2000,
        seed=20260825,
    )


def _gaussian_callbacks(
    mean: np.ndarray,
    covariance: np.ndarray,
):
    precision = np.linalg.inv(covariance)

    def scalar(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        delta = tf.reshape(tf.cast(theta, tf.float64), [-1]) - mean
        precision_tf = tf.constant(precision, tf.float64)
        score = -tf.linalg.matvec(precision_tf, delta)
        value = -0.5 * tf.tensordot(delta, -score, axes=1)
        return value, score

    def batched(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        delta = tf.cast(theta, tf.float64) - mean[None, :]
        precision_tf = tf.constant(precision, tf.float64)
        scores = -tf.linalg.matmul(delta, precision_tf)
        values = -0.5 * tf.reduce_sum(delta * (-scores), axis=1)
        return values, scores

    return scalar, batched


def test_posterior_local_initializer_is_lazily_exported_from_inference() -> None:
    from bayesfilter import inference

    assert inference.PosteriorLocalInitializerConfig is PosteriorLocalInitializerConfig
    assert (
        inference.initialize_posterior_local_location_scale
        is initialize_posterior_local_location_scale
    )
    assert "PosteriorLocalInitializerResult" in inference.__all__


def test_gaussian_recovers_location_and_physical_covariance_under_scaling() -> None:
    mean = np.array([0.7, -0.4])
    covariance = np.array([[0.5, 0.12], [0.12, 1.2]])
    scale = np.array([0.5, 2.0])
    scalar, batched = _gaussian_callbacks(mean, covariance)

    result = initialize_posterior_local_location_scale(
        scalar,
        np.array([-1.0, 1.0]),
        scale=scale,
        batched_value_and_score_fn=batched,
        config=_initializer_config(),
        movement_config=_movement_config(),
        curvature_thresholds=_thresholds(2),
    )

    assert result.accepted is True
    assert result.status == "usable_posterior_local_initializer"
    np.testing.assert_allclose(result.center, mean, atol=1.0e-7)
    np.testing.assert_allclose(result.covariance_theta, covariance, atol=1.0e-7)
    np.testing.assert_allclose(
        result.precision_z,
        np.diag(scale) @ np.linalg.inv(covariance) @ np.diag(scale),
        atol=1.0e-7,
    )
    np.testing.assert_allclose(
        result.marginal_standard_deviations, np.sqrt(np.diag(covariance))
    )
    np.testing.assert_allclose(result.initial_output_shift, mean, atol=1.0e-7)
    np.testing.assert_allclose(
        result.initial_output_scale_log,
        0.5 * np.log(np.diag(covariance)),
        atol=1.0e-7,
    )
    assert len(result.movement_fits) >= 2
    assert len({row["seed"] for row in result.movement_fits}) == len(
        result.movement_fits
    )
    assert result.curvature is not None
    np.testing.assert_array_equal(result.curvature.center, result.center)
    assert tuple(result.payload()["nonclaims"]) == POSTERIOR_LOCAL_INITIALIZER_NONCLAIMS
    json.dumps(result.payload(include_arrays=True))


def test_cloud_winner_forces_fresh_fit_before_covariance_handoff() -> None:
    mean = np.array([0.6])
    covariance = np.array([[0.4]])
    scalar, batched = _gaussian_callbacks(mean, covariance)
    config = _initializer_config(locator_gradient_tolerance=1.0e6)

    result = initialize_posterior_local_location_scale(
        scalar,
        np.array([0.0]),
        scale=np.ones(1),
        batched_value_and_score_fn=batched,
        config=config,
        movement_config=_movement_config(seed=20260826),
        curvature_thresholds=_thresholds(1),
    )

    assert result.accepted is True
    assert result.movement_fits[0]["center_moved"] is True
    assert result.movement_fits[0]["covariance_handoff_eligible"] is False
    assert len(result.movement_fits) >= 2
    assert result.movement_fits[1]["seed"] != result.movement_fits[0]["seed"]
    np.testing.assert_allclose(result.center, mean, atol=1.0e-7)
    np.testing.assert_array_equal(result.curvature.center, result.center)


def test_finite_rejection_sentinel_and_status_disagreement_fail_closed() -> None:
    def scalar(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        x = tf.reshape(tf.cast(theta, tf.float64), [-1])
        valid = x[0] <= 0.1
        return tf.cond(
            valid,
            lambda: (-0.5 * x[0] ** 2, -x),
            lambda: (tf.constant(-1.0e100, tf.float64), tf.zeros_like(x)),
        )

    def batched(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        x = tf.cast(theta, tf.float64)
        valid = x[:, 0] <= 0.1
        values = tf.where(valid, -0.5 * x[:, 0] ** 2, -1.0e100)
        scores = tf.where(valid[:, None], -x, tf.zeros_like(x))
        return values, scores

    def eligible(theta: tf.Tensor) -> tf.Tensor:
        return tf.reshape(tf.cast(theta, tf.float64), [-1])[0] <= 0.1

    def batched_eligible(theta: tf.Tensor) -> tf.Tensor:
        return tf.cast(theta, tf.float64)[:, 0] <= 0.1

    result = initialize_posterior_local_location_scale(
        scalar,
        np.array([0.0]),
        scale=np.ones(1),
        batched_value_and_score_fn=batched,
        eligibility_fn=eligible,
        batched_eligibility_fn=batched_eligible,
        config=_initializer_config(locator_gradient_tolerance=1.0e6),
        movement_config=_movement_config(seed=20260827),
        curvature_thresholds=_thresholds(1),
    )

    assert result.accepted is False
    assert result.status == "eligibility_contract_mismatch"
    assert result.covariance_theta is None
    assert result.diagnostics["eligibility_contract"]["mismatch_rows"] > 0
    assert result.diagnostics["hmc_rejection_policy_permitted"] is False
