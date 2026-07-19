from __future__ import annotations

import numpy as np
import pytest

from bayesfilter.inference.dense_directional_score_geometry import (
    DENSE_DIRECTIONAL_GEOMETRY_NONCLAIMS,
    DenseDirectionalScoreGeometryConfig,
    arithmetic_precision_consensus,
    compare_dense_directional_precision_stability,
    directional_prediction_relative_frobenius,
    fit_dense_directional_score_geometry,
)


def _frame(dimension: int, *, radius: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    q *= np.where(np.diag(r) >= 0.0, 1.0, -1.0)[None, :]
    positive = radius * q.T
    return np.stack((positive, -positive), axis=1).reshape(2 * dimension, dimension)


def _scores(
    precision: np.ndarray,
    center_score: np.ndarray,
    offsets: np.ndarray,
    *,
    even_quadratic: float = 0.0,
) -> np.ndarray:
    linear = center_score[None, :] - offsets @ precision.T
    return linear + even_quadratic * np.square(offsets)


def _fit(precision: np.ndarray, *, seed: int = 11, **config_overrides):
    center = np.linspace(-0.2, 0.3, precision.shape[0])
    offsets = _frame(precision.shape[0], radius=0.05, seed=seed)
    config = DenseDirectionalScoreGeometryConfig(**config_overrides)
    return fit_dense_directional_score_geometry(
        center_score_z=center,
        antithetic_offsets_z=offsets,
        antithetic_scores_z=_scores(precision, center, offsets),
        config=config,
    )


def test_rotated_dense_spd_is_recovered_with_correct_orientation() -> None:
    rng = np.random.default_rng(3)
    q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    precision = q @ np.diag([0.7, 1.0, 1.8, 2.5, 4.0, 7.0]) @ q.T

    result = _fit(precision)

    assert result.valid is True
    assert result.accepted is True
    assert result.status == "dense_directional_candidate_usable"
    np.testing.assert_allclose(result.directional_precision_raw, precision, atol=2.0e-14)
    np.testing.assert_allclose(result.central_symmetric_precision, precision, atol=2.0e-14)
    assert result.diagnostics["central_symmetric_raw_spd"] is True
    assert result.diagnostics["symmetry_projection_relative_frobenius"] < 1.0e-14
    assert result.diagnostics["solve_relative_frobenius_residual"] < 1.0e-14
    assert tuple(result.nonclaims) == DENSE_DIRECTIONAL_GEOMETRY_NONCLAIMS
    payload = result.payload(include_matrices=True)
    assert payload["directional_precision_raw"] is not None
    assert payload["diagnostics"]["covariance_built"] is False
    assert payload["diagnostics"]["hmc_operation_called"] is False


def test_indefinite_raw_symmetric_map_is_not_projected_or_rescued() -> None:
    precision = np.diag([2.0, 1.0, -0.25, 3.0])

    result = _fit(precision)

    assert result.valid is True
    assert result.accepted is False
    assert result.status == "central_symmetric_raw_not_spd"
    assert result.diagnostics["central_symmetric_raw_nonpositive_eigenvalue_count"] == 1
    np.testing.assert_allclose(result.central_symmetric_precision, precision, atol=1.0e-14)
    assert result.diagnostics["eigenvalue_projection_used"] is False


def test_asymmetric_map_is_validly_rejected_before_spd_interpretation() -> None:
    precision = np.array(
        [[2.0, 1.2, 0.0], [-0.4, 2.0, 0.3], [0.0, -0.2, 1.0]],
        dtype=np.float64,
    )

    result = _fit(precision, symmetry_projection_relative_frobenius_cap=0.02)

    assert result.valid is True
    assert result.accepted is False
    assert result.status == "symmetry_projection_burden_rejected"
    np.testing.assert_allclose(result.directional_precision_raw, precision, atol=1.0e-14)
    assert result.central_symmetric_precision is None
    assert result.central_symmetric_raw_eigenvalues is None
    assert result.diagnostics["central_symmetric_interpreted"] is False


def test_swapped_antithetic_scores_catches_response_sign_error() -> None:
    precision = np.diag([1.0, 2.0, 3.0])
    center = np.zeros(3)
    offsets = _frame(3, radius=0.05, seed=8)
    scores = _scores(precision, center, offsets).reshape(3, 2, 3)[:, ::-1].reshape(6, 3)

    result = fit_dense_directional_score_geometry(
        center_score_z=center,
        antithetic_offsets_z=offsets,
        antithetic_scores_z=scores,
    )

    np.testing.assert_allclose(result.directional_precision_raw, -precision, atol=1.0e-14)
    assert result.status == "central_symmetric_raw_not_spd"


@pytest.mark.parametrize("defect", ["pairs", "rank", "nonfinite", "shape"])
def test_malformed_frames_fail_as_harness_errors(defect: str) -> None:
    dimension = 4
    center = np.zeros(dimension)
    offsets = _frame(dimension, radius=0.05, seed=5)
    scores = _scores(np.eye(dimension), center, offsets)
    if defect == "pairs":
        offsets[1, 0] += 1.0e-3
    elif defect == "rank":
        offsets[2] = offsets[0]
        offsets[3] = -offsets[2]
    elif defect == "nonfinite":
        scores[0, 0] = np.nan
    else:
        offsets = offsets[:-2]

    with pytest.raises(ValueError):
        fit_dense_directional_score_geometry(
            center_score_z=center,
            antithetic_offsets_z=offsets,
            antithetic_scores_z=scores,
        )


def test_even_response_is_explanatory_and_does_not_change_central_fit() -> None:
    precision = np.diag([1.0, 2.0, 4.0])
    center = np.zeros(3)
    offsets = _frame(3, radius=0.05, seed=14)

    result = fit_dense_directional_score_geometry(
        center_score_z=center,
        antithetic_offsets_z=offsets,
        antithetic_scores_z=_scores(
            precision, center, offsets, even_quadratic=200.0
        ),
    )

    assert result.accepted is True
    np.testing.assert_allclose(result.directional_precision_raw, precision, atol=1.0e-14)
    assert result.diagnostics["even_response_relative_frobenius"] > 1.0


def test_holdout_and_arithmetic_consensus_formulas_are_exact() -> None:
    left = np.diag([1.0, 2.0, 3.0])
    right = np.diag([1.4, 1.8, 2.6])
    truth = np.diag([1.1, 2.1, 2.7])
    center = np.array([0.2, -0.1, 0.3])
    offsets = _frame(3, radius=0.05, seed=20)
    scores = _scores(truth, center, offsets)
    positive = offsets[0::2]
    response = 0.5 * (scores[1::2] - scores[0::2])

    consensus = arithmetic_precision_consensus((left, right))
    np.testing.assert_array_equal(consensus, (left + right) / 2.0)
    observed = directional_prediction_relative_frobenius(consensus, offsets, scores)
    expected = np.linalg.norm(positive @ consensus.T - response) / np.linalg.norm(response)
    assert observed == pytest.approx(expected, rel=0.0, abs=1.0e-16)

    other_offsets = _frame(3, radius=0.05, seed=21)
    other_scores = _scores(truth, center, other_offsets)
    mean_error = (
        directional_prediction_relative_frobenius(consensus, offsets, scores)
        + directional_prediction_relative_frobenius(
            consensus, other_offsets, other_scores
        )
    ) / 2.0
    assert mean_error == pytest.approx(
        np.mean(
            [
                directional_prediction_relative_frobenius(consensus, offsets, scores),
                directional_prediction_relative_frobenius(
                    consensus, other_offsets, other_scores
                ),
            ]
        )
    )


def test_holdout_and_consensus_reject_nonsymmetric_or_nonfinite_precision() -> None:
    offsets = _frame(2, radius=0.05, seed=24)
    scores = _scores(np.eye(2), np.zeros(2), offsets)
    asymmetric = np.array([[1.0, 0.2], [0.0, 1.0]])

    with pytest.raises(ValueError, match="symmetric"):
        directional_prediction_relative_frobenius(asymmetric, offsets, scores)
    with pytest.raises(ValueError, match="symmetric"):
        arithmetic_precision_consensus((np.eye(2), asymmetric))
    with pytest.raises(ValueError, match="finite"):
        arithmetic_precision_consensus((np.eye(2), np.diag([1.0, np.nan])))


@pytest.mark.parametrize("defect", ["reordered_pair", "rank", "radius", "orthogonality"])
def test_holdout_rejects_malformed_directional_frames(defect: str) -> None:
    dimension = 4
    precision = np.diag([1.0, 2.0, 3.0, 4.0])
    center = np.zeros(dimension)
    offsets = _frame(dimension, radius=0.05, seed=25)
    scores = _scores(precision, center, offsets)
    if defect == "reordered_pair":
        offsets[[1, 2]] = offsets[[2, 1]]
        scores[[1, 2]] = scores[[2, 1]]
    elif defect == "rank":
        offsets[2] = offsets[0]
        offsets[3] = -offsets[2]
        scores = _scores(precision, center, offsets)
    elif defect == "radius":
        offsets[2:4] *= 0.5
        scores = _scores(precision, center, offsets)
    else:
        positive = offsets[0::2].copy()
        positive[1] += 1.0e-6 * positive[0]
        positive *= 0.05 / np.linalg.norm(positive, axis=1, keepdims=True)
        offsets = np.stack((positive, -positive), axis=1).reshape(8, 4)
        scores = _scores(precision, center, offsets)

    with pytest.raises(ValueError):
        directional_prediction_relative_frobenius(precision, offsets, scores)


def test_stability_comparison_requires_two_accepted_raw_spd_candidates() -> None:
    config = DenseDirectionalScoreGeometryConfig(
        principal_subspace_rank=2,
        trace_normalized_frobenius_cap=0.1,
        trace_normalized_operator_cap=0.1,
    )
    left = _fit(np.diag([1.0, 2.0, 3.0]), seed=30)
    right = _fit(np.diag([1.01, 1.99, 3.0]), seed=31)

    comparison = compare_dense_directional_precision_stability(
        left, right, config=config
    )

    assert comparison["passed"] is True
    assert comparison["metrics"]["generalized_eigenvalues"] is not None
    rejected = _fit(np.diag([1.0, -1.0, 3.0]), seed=32)
    with pytest.raises(ValueError, match="raw-SPD"):
        compare_dense_directional_precision_stability(
            left, rejected, config=config
        )


def test_numerical_orthogonality_gate_is_invalid_not_candidate_rejection() -> None:
    dimension = 3
    center = np.zeros(dimension)
    offsets = _frame(dimension, radius=0.05, seed=40)
    positive = offsets[0::2].copy()
    positive[1] += 1.0e-8 * positive[0]
    positive *= 0.05 / np.linalg.norm(positive, axis=1, keepdims=True)
    offsets = np.stack((positive, -positive), axis=1).reshape(6, 3)
    scores = _scores(np.eye(dimension), center, offsets)

    result = fit_dense_directional_score_geometry(
        center_score_z=center,
        antithetic_offsets_z=offsets,
        antithetic_scores_z=scores,
    )

    assert result.valid is False
    assert result.accepted is False
    assert result.status == "directional_reconstruction_numerically_invalid"
    assert result.central_symmetric_precision is None
