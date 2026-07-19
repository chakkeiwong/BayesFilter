from __future__ import annotations

import numpy as np

from bayesfilter.inference.block_score_geometry import (
    BLOCK_SCORE_GEOMETRY_NONCLAIMS,
    BlockScoreGeometryConfig,
    ScoreGeometryBlock,
    fit_block_diagonal_score_geometry,
)


def _blocks() -> tuple[ScoreGeometryBlock, ...]:
    return (
        ScoreGeometryBlock("first", 0, 2),
        ScoreGeometryBlock("second", 2, 5),
    )


def _cloud(
    precision: np.ndarray,
    center_score: np.ndarray,
    *,
    rows: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    offsets = rng.normal(size=(rows, center_score.size)) * 0.05
    scores = center_score[None, :] - offsets @ precision.T
    return offsets, scores


def _fit(
    precision: np.ndarray,
    *,
    config: BlockScoreGeometryConfig,
):
    center_score = np.linspace(-0.2, 0.3, precision.shape[0])
    training = [_cloud(precision, center_score, rows=16, seed=100 + i) for i in range(2)]
    selection = [_cloud(precision, center_score, rows=12, seed=200 + i) for i in range(2)]
    audit = _cloud(precision, center_score, rows=14, seed=300)
    return fit_block_diagonal_score_geometry(
        center_score_z=center_score,
        training_offsets_z=np.stack([item[0] for item in training]),
        training_scores_z=np.stack([item[1] for item in training]),
        selection_offsets_z=np.stack([item[0] for item in selection]),
        selection_scores_z=np.stack([item[1] for item in selection]),
        audit_offsets_z=audit[0],
        audit_scores_z=audit[1],
        blocks=_blocks(),
        config=config,
    )


def _config(**overrides) -> BlockScoreGeometryConfig:
    values = {
        "ridge": 1.0e-12,
        "max_condition_number": 1.0e6,
        "selection_relative_rmse_cap": 1.0e-8,
        "audit_relative_rmse_cap": 1.0e-8,
        "unexplained_response_fraction_cap": 1.0e-8,
        "generalized_eigenvalue_spread_cap": 1.01,
        "trace_normalized_frobenius_cap": 1.0e-8,
        "trace_normalized_operator_cap": 1.0e-8,
        "principal_angle_degrees_cap": 1.0e-6,
        "principal_subspace_rank": 4,
    }
    values.update(overrides)
    return BlockScoreGeometryConfig(**values)


def test_exact_block_score_geometry_qualifies_and_preserves_orientation() -> None:
    precision = np.array(
        [
            [3.0, 0.4, 0.0, 0.0, 0.0],
            [0.4, 1.5, 0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.2, -0.1],
            [0.0, 0.0, 0.2, 1.2, 0.3],
            [0.0, 0.0, -0.1, 0.3, 0.9],
        ]
    )

    result = _fit(precision, config=_config())

    assert result.accepted is True
    assert result.status == "qualified_for_hmc_initialization"
    np.testing.assert_allclose(result.precision_z, precision, rtol=1.0e-9, atol=1.0e-9)
    np.testing.assert_allclose(result.precision_z @ result.covariance_z, np.eye(5), atol=1.0e-10)
    assert tuple(result.nonclaims) == BLOCK_SCORE_GEOMETRY_NONCLAIMS

    scale = np.array([0.5, 2.0, 1.5, 0.8, 1.2])
    physical = result.position_geometry(scale)
    expected_covariance = np.diag(scale) @ np.linalg.inv(precision) @ np.diag(scale)
    np.testing.assert_allclose(physical["covariance"], expected_covariance, atol=1.0e-9)
    np.testing.assert_allclose(
        physical["precision"] @ physical["covariance"], np.eye(5), atol=1.0e-10
    )
    np.testing.assert_allclose(
        physical["factor"] @ physical["factor"].T,
        physical["covariance"],
        atol=1.0e-10,
    )


def test_off_block_curvature_fails_complete_score_audit() -> None:
    precision = np.diag([2.0, 1.5, 1.2, 1.8, 2.2])
    precision[0, 2] = precision[2, 0] = 0.8
    precision[1, 4] = precision[4, 1] = -0.6

    result = _fit(
        precision,
        config=_config(
            selection_relative_rmse_cap=0.10,
            audit_relative_rmse_cap=0.10,
            unexplained_response_fraction_cap=0.10,
            trace_normalized_frobenius_cap=0.20,
            trace_normalized_operator_cap=0.20,
            generalized_eigenvalue_spread_cap=3.0,
            principal_angle_degrees_cap=30.0,
        ),
    )

    assert result.accepted is False
    assert result.status in {
        "selection_score_fit_rejected",
        "audit_score_fit_rejected",
        "offblock_curvature_rejected",
    }
    assert result.diagnostics["audit_unexplained_response_fraction"] > 0.10


def test_raw_non_spd_block_is_not_rescued_by_projection() -> None:
    precision = np.diag([2.0, -0.5, 1.2, 1.8, 2.2])

    result = _fit(
        precision,
        config=_config(
            selection_relative_rmse_cap=1.0,
            audit_relative_rmse_cap=1.0,
            unexplained_response_fraction_cap=1.0,
            trace_normalized_frobenius_cap=1.0,
            trace_normalized_operator_cap=1.0,
            generalized_eigenvalue_spread_cap=100.0,
            principal_angle_degrees_cap=90.0,
        ),
    )

    assert result.accepted is False
    assert result.status == "raw_block_precision_not_spd"
    assert any(
        block["raw_nonpositive_eigenvalue_count"] > 0
        for replicate in result.diagnostics["replicates"]
        for block in replicate["blocks"]
    )


def test_rank_deficient_block_design_fails_closed() -> None:
    precision = np.eye(5)
    center_score = np.zeros(5)
    offsets = np.zeros((2, 8, 5))
    scores = np.zeros_like(offsets)

    result = fit_block_diagonal_score_geometry(
        center_score_z=center_score,
        training_offsets_z=offsets,
        training_scores_z=scores,
        selection_offsets_z=np.ones((2, 4, 5)) * 0.01,
        selection_scores_z=np.stack(
            [center_score - np.ones((4, 5)) * 0.01 for _ in range(2)]
        ),
        audit_offsets_z=np.ones((4, 5)) * 0.02,
        audit_scores_z=center_score - np.ones((4, 5)) * 0.02,
        blocks=_blocks(),
        config=_config(
            selection_relative_rmse_cap=1.0,
            audit_relative_rmse_cap=1.0,
            unexplained_response_fraction_cap=1.0,
        ),
    )

    assert result.accepted is False
    assert result.status == "block_design_rank_deficient"
