from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf
import bayesfilter.inference.sequential_map_covariance as sequential
import bayesfilter.inference.factor_correlation_geometry as factor_geometry

from bayesfilter.inference.factor_correlation_geometry import (
    FactorCorrelationGeometryConfig,
    factor_correlation_covariance,
    fit_factor_correlation_score_geometry,
)
from bayesfilter.inference.sequential_map_covariance import (
    SequentialMapCovarianceConfig,
    dimension_scaled_search_count,
    estimate_sequential_map_covariance,
)


def _score_rows(precision: tf.Tensor, offsets: tf.Tensor) -> tf.Tensor:
    return -tf.einsum("ij,bj->bi", precision, offsets)


def test_one_factor_covariance_has_declared_variances_and_correlations() -> None:
    standard_deviations = tf.constant([0.8, 1.2, 1.5], tf.float64)
    loadings = tf.constant([[0.4], [-0.2], [0.3]], tf.float64)
    covariance = factor_correlation_covariance(standard_deviations, loadings)
    correlation = covariance / (
        standard_deviations[:, None] * standard_deviations[None, :]
    )

    np.testing.assert_allclose(
        tf.linalg.diag_part(covariance).numpy(),
        np.square(standard_deviations.numpy()),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        correlation.numpy()[np.triu_indices(3, 1)],
        (loadings @ tf.transpose(loadings)).numpy()[np.triu_indices(3, 1)],
        atol=1.0e-12,
    )
    assert float(tf.reduce_min(tf.linalg.eigvalsh(covariance)).numpy()) > 0.0


def test_two_factor_covariance_matches_latent_factor_identity() -> None:
    standard_deviations = tf.constant([1.0, 1.1, 0.9, 1.3], tf.float64)
    loadings = tf.constant(
        [[0.4, 0.0], [0.2, 0.3], [-0.3, 0.15], [0.1, -0.25]],
        tf.float64,
    )
    covariance = factor_correlation_covariance(standard_deviations, loadings)
    correlation = covariance / (
        standard_deviations[:, None] * standard_deviations[None, :]
    )
    expected = loadings @ tf.transpose(loadings)
    expected = tf.linalg.set_diag(expected, tf.ones([4], tf.float64))

    np.testing.assert_allclose(correlation.numpy(), expected.numpy(), atol=1.0e-12)
    assert float(tf.reduce_min(tf.linalg.eigvalsh(covariance)).numpy()) > 0.0


def test_one_factor_score_fit_recovers_covariance_and_sign_normalization() -> None:
    rng = np.random.default_rng(17)
    dimension = 5
    deviations = tf.constant(np.linspace(0.8, 1.2, dimension), tf.float64)
    loadings = tf.constant([[0.35], [-0.2], [0.15], [-0.3], [0.1]], tf.float64)
    covariance = factor_correlation_covariance(deviations, loadings)
    precision = tf.linalg.inv(covariance)
    offsets = tf.constant(rng.normal(size=(70, dimension)) * 0.1, tf.float64)
    scores = _score_rows(precision, offsets)

    result = fit_factor_correlation_score_geometry(
        tf.zeros([dimension], tf.float64),
        offsets[:45],
        scores[:45],
        offsets[45:],
        scores[45:],
        config=FactorCorrelationGeometryConfig(
            factor_count=1,
            holdout_score_relative_rmse=1.0e-3,
        ),
    )

    assert result.accepted is True
    assert result.status == "usable"
    assert result.parameter_count == 2 * dimension
    anchor = result.anchor_indices[0]
    assert result.loadings is not None and result.loadings[anchor, 0] > 0.0
    np.testing.assert_allclose(result.covariance_z, covariance.numpy(), atol=2.0e-5)


def test_two_factor_score_fit_recovers_identified_covariance() -> None:
    rng = np.random.default_rng(23)
    dimension = 9
    deviations = tf.constant(np.linspace(0.75, 1.25, dimension), tf.float64)
    loadings = tf.constant(rng.normal(size=(dimension, 2)) * 0.12, tf.float64)
    covariance = factor_correlation_covariance(deviations, loadings)
    precision = tf.linalg.inv(covariance)
    offsets = tf.constant(rng.normal(size=(80, dimension)) * 0.1, tf.float64)
    scores = _score_rows(precision, offsets)

    result = fit_factor_correlation_score_geometry(
        tf.zeros([dimension], tf.float64),
        offsets[:50],
        scores[:50],
        offsets[50:],
        scores[50:],
        config=FactorCorrelationGeometryConfig(
            factor_count=2,
            holdout_score_relative_rmse=1.0e-3,
            max_iterations=300,
        ),
    )

    assert result.accepted is True
    assert result.parameter_count == 3 * dimension - 1
    assert result.diagnostics["prediction_jacobian_rank"] == result.parameter_count
    anchor_a, anchor_b = result.anchor_indices
    assert result.loadings is not None
    assert abs(result.loadings[anchor_a, 1]) <= 1.0e-12
    assert result.loadings[anchor_a, 0] > 0.0
    assert result.loadings[anchor_b, 1] > 0.0
    np.testing.assert_allclose(result.covariance_z, covariance.numpy(), atol=2.0e-4)


def test_two_factor_fit_rejects_dimensionally_unidentified_case() -> None:
    dimension = 4
    offsets = tf.eye(dimension, dtype=tf.float64)
    result = fit_factor_correlation_score_geometry(
        tf.zeros([dimension], tf.float64),
        offsets,
        -offsets,
        -offsets,
        offsets,
        config=FactorCorrelationGeometryConfig(factor_count=2),
    )

    assert result.accepted is False
    assert result.status == "factor_parameterization_dimensionally_unidentified"


def test_two_factor_fit_rejects_rank_deficient_prediction_jacobian(monkeypatch) -> None:
    dimension = 5
    rng = np.random.default_rng(41)
    offsets = tf.constant(rng.normal(size=(30, dimension)) * 0.1, tf.float64)
    precision = tf.eye(dimension, dtype=tf.float64)
    scores = _score_rows(precision, offsets)
    monkeypatch.setattr(
        factor_geometry,
        "_prediction_jacobian_diagnostics",
        lambda *_args, **_kwargs: (3 * dimension - 2, 10.0),
    )

    result = fit_factor_correlation_score_geometry(
        tf.zeros([dimension], tf.float64),
        offsets[:20],
        scores[:20],
        offsets[20:],
        scores[20:],
        config=FactorCorrelationGeometryConfig(factor_count=2),
    )

    assert result.accepted is False
    assert result.status == "second_factor_unidentified"
    assert result.diagnostics["second_factor_identified"] is False


def test_dimension_scaled_search_rule_is_even_and_matches_boundaries() -> None:
    assert dimension_scaled_search_count(1) == 2
    assert dimension_scaled_search_count(10) == 100
    assert dimension_scaled_search_count(11) == 100
    assert dimension_scaled_search_count(20) == 124
    assert dimension_scaled_search_count(100) == 506
    with pytest.raises(ValueError, match="dimension"):
        dimension_scaled_search_count(0)


def test_factor_covariance_rejects_loading_row_ball_boundary() -> None:
    deviations = tf.ones([2], tf.float64)
    margin = 1.0e-6
    boundary = np.sqrt(1.0 - margin)
    with pytest.raises(tf.errors.InvalidArgumentError):
        factor_correlation_covariance(
            deviations,
            tf.constant([[boundary], [0.0]], tf.float64),
            loading_margin=margin,
        )


def test_factor_covariance_construction_compiles_with_xla() -> None:
    @tf.function(
        input_signature=(
            tf.TensorSpec([4], tf.float64),
            tf.TensorSpec([4, 2], tf.float64),
        ),
        jit_compile=True,
    )
    def compiled(deviations: tf.Tensor, loadings: tf.Tensor) -> tf.Tensor:
        return factor_correlation_covariance(deviations, loadings)

    deviations = tf.constant([1.0, 1.1, 0.9, 1.2], tf.float64)
    loadings = tf.constant(
        [[0.3, 0.0], [0.1, 0.2], [-0.2, 0.1], [0.05, -0.2]],
        tf.float64,
    )
    eager = factor_correlation_covariance(deviations, loadings)
    xla = compiled(deviations, loadings)
    np.testing.assert_allclose(xla.numpy(), eager.numpy(), atol=1.0e-12)


def test_structured_sequential_policy_reuses_search_scores_and_keeps_fresh_terminal_fit() -> None:
    dimension = 3
    deviations = tf.constant([0.8, 1.1, 1.3], tf.float64)
    loadings = tf.constant([[0.35], [-0.2], [0.15]], tf.float64)
    covariance = factor_correlation_covariance(deviations, loadings)
    precision = tf.linalg.inv(covariance)
    mode = tf.constant([0.16, -0.08, 0.12], tf.float64)

    def scalar(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        delta = tf.convert_to_tensor(theta, tf.float64) - mode
        score = -tf.linalg.matvec(precision, delta)
        return 0.5 * tf.tensordot(delta, score, axes=1), score

    def batched(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        delta = tf.convert_to_tensor(theta, tf.float64) - mode[None, :]
        score = -tf.einsum("ij,bj->bi", precision, delta)
        return 0.5 * tf.reduce_sum(delta * score, axis=1), score

    result = estimate_sequential_map_covariance(
        scalar,
        [np.zeros(dimension)],
        batched_value_and_score_fn=batched,
        config=SequentialMapCovarianceConfig(
            terminal_score_max_abs=1.0e-7,
            initial_radius=0.25,
            terminal_sample_count=24,
            max_attempts=8,
            max_exact_evaluations=512,
            refinement_geometry_policy="factor_correlation",
            dimension_scaled_search=True,
            orthogonal_antithetic_search=True,
            reuse_search_scores=True,
            structured_fresh_sample_multiplier=4,
            structured_max_factors=1,
            structured_holdout_score_relative_rmse=1.0e-4,
            require_proposal_score_reduction=False,
            stop_on_stalled_attempts=False,
            locator_policy="center_first",
            seed=(2026, 716),
        ),
    )

    assert result.accepted is True
    assert result.status == "usable"
    np.testing.assert_allclose(result.map_candidate, mode.numpy(), atol=1.0e-6)
    assert result.diagnostics["terminal_fit_fresh"] is True
    attempts = [row for row in result.diagnostics["history"] if "fit" in row]
    assert attempts
    assert any(row["fit"].get("reused_training_count", 0) > 0 for row in attempts)


def test_reuse_filter_includes_radius_boundary_and_rejects_zero_outside_nonfinite() -> None:
    dimension = 3
    center = tf.zeros([dimension], tf.float64)
    scale = tf.ones([dimension], tf.float64)
    center_score = tf.zeros([dimension], tf.float64)
    search_theta = tf.constant(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.5001, 0.0, 0.0],
            [float("nan"), 0.0, 0.0],
        ],
        tf.float64,
    )
    search_scores = -search_theta

    def target(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        row = tf.convert_to_tensor(theta, tf.float64)
        return -0.5 * tf.reduce_sum(row**2), -row

    data, evaluations = sequential._structured_factor_fit_data(
        target,
        center,
        center_score,
        scale,
        search_theta=search_theta,
        search_scores=search_scores,
        dimension=dimension,
        radius=0.5,
        fresh_sample_count=4 * dimension,
        seed=(2026, 724),
        evaluations=4,
        batched_value_and_score_fn=None,
        reuse_search_scores=True,
    )

    assert data["reused_training_count"] == 1
    np.testing.assert_allclose(data["training_offsets_z"][-1].numpy(), [0.5, 0.0, 0.0])
    assert evaluations == 4 + 4 * dimension


def test_structured_policy_escalates_from_rejected_one_factor_to_two_factors(
    monkeypatch,
) -> None:
    dimension = 3
    mode = tf.constant([0.12, -0.08, 0.05], tf.float64)

    def scalar(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        delta = tf.convert_to_tensor(theta, tf.float64) - mode
        return -0.5 * tf.reduce_sum(delta**2), -delta

    def batched(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        delta = tf.convert_to_tensor(theta, tf.float64) - mode[None, :]
        return -0.5 * tf.reduce_sum(delta**2, axis=1), -delta

    def controlled_fit(data, *, factor_count, config):
        del data, config
        if factor_count == 1:
            return {
                "status": "holdout_score_fit_rejected",
                "factor_count": 1,
            }
        return {
            "status": "usable",
            "factor_count": 2,
            "projected_precision_z": np.eye(dimension),
            "projection_relative_frobenius": 0.0,
        }

    monkeypatch.setattr(sequential, "_fit_factor_from_data", controlled_fit)
    result = estimate_sequential_map_covariance(
        scalar,
        [np.zeros(dimension)],
        batched_value_and_score_fn=batched,
        config=SequentialMapCovarianceConfig(
            locator_policy="center_first",
            terminal_score_max_abs=1.0e-8,
            terminal_sample_count=24,
            max_attempts=2,
            max_exact_evaluations=256,
            refinement_geometry_policy="factor_correlation",
            dimension_scaled_search=True,
            orthogonal_antithetic_search=True,
            reuse_search_scores=True,
            structured_fresh_sample_multiplier=4,
            structured_max_factors=2,
            require_proposal_score_reduction=False,
            stop_on_stalled_attempts=False,
            seed=(2026, 717),
        ),
    )

    assert result.accepted is True
    attempts = [row for row in result.diagnostics["history"] if "fit" in row]
    proposal_rows = attempts[0]["proposal_attempts"]
    assert proposal_rows[0] == {
        "factor_count": 1,
        "fit_status": "holdout_score_fit_rejected",
        "proposal_evaluated": False,
    }
    assert proposal_rows[1]["factor_count"] == 2
    assert proposal_rows[1]["proposal_evaluated"] is True
    assert proposal_rows[1]["accepted"] is True
