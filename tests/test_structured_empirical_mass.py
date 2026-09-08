from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference.mass_matrix import structured_covariance_from_empirical


def test_diagonal_empirical_mass_is_positive_and_diagonal():
    empirical = np.asarray([[4.0, 1.5], [1.5, 0.25]])
    result = structured_covariance_from_empirical(
        empirical,
        diagonal=True,
        eigenvalue_floor=1.0e-3,
    )
    assert result.matrix_kind == "diagonal"
    assert tf.is_tensor(result.covariance)
    assert result.regularization_report["numerical_backend"] == "tensorflow"
    np.testing.assert_allclose(result.covariance, np.diag(np.diag(result.covariance)))
    assert np.min(np.linalg.eigvalsh(result.covariance)) >= 1.0e-3


def test_structural_blocks_zero_cross_block_entries_and_preserve_within_block():
    empirical = np.asarray(
        [
            [2.0, 0.5, 0.8],
            [0.5, 1.0, 0.4],
            [0.8, 0.4, 3.0],
        ]
    )
    result = structured_covariance_from_empirical(
        empirical,
        blocks=(
            {"name": "first", "start": 0, "stop": 2},
            {"name": "second", "start": 2, "stop": 3},
        ),
        shrinkage=0.0,
    )
    np.testing.assert_allclose(result.covariance[:2, :2], empirical[:2, :2])
    np.testing.assert_allclose(result.covariance[:2, 2], 0.0)
    np.testing.assert_allclose(result.covariance[2, :2], 0.0)
    assert result.regularization_report["cross_block_entries_zero"] is True


def test_structural_blocks_must_be_a_complete_ordered_partition():
    with pytest.raises(ValueError, match="complete partition"):
        structured_covariance_from_empirical(
            np.eye(3),
            blocks=({"name": "short", "start": 0, "stop": 2},),
        )


def test_structured_empirical_mass_rejects_nonfinite_geometry():
    with pytest.raises(ValueError, match="must be finite"):
        structured_covariance_from_empirical(
            tf.constant([[1.0, float("nan")], [0.0, 1.0]], tf.float64),
            diagonal=True,
        )
