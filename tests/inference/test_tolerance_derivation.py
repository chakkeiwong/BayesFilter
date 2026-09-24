"""Phase 0 Task 0.1 smoke tests for tolerance derivation.

Checks the derived tolerances land in the ranges the Option A decision
records, and that incompatible dtype/backend pairs fail closed.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import tensorflow as tf

from bayesfilter.inference.tolerance_derivation import (
    condition_number_from_matrix,
    derive_golden_master_tolerance,
    derive_parity_tolerance,
    derive_tolerance_from_matrix,
)


def test_parity_tolerance_float32_tf32_in_documented_range():
    """kappa=1e3, float32/TF32 -> ~1e-1 per the docstring formula."""
    tolerance = derive_parity_tolerance(1.0e3, "float32", "TF32")
    # 1e3 * 1e-5 * 10 = 1e-1
    assert tolerance == pytest.approx(1.0e-1, rel=1e-9)


def test_parity_tolerance_scales_linearly_in_condition_number():
    """Doubling the condition number doubles the tolerance."""
    base = derive_parity_tolerance(1.0e2, "float32", "TF32")
    doubled = derive_parity_tolerance(2.0e2, "float32", "TF32")
    assert doubled == pytest.approx(2.0 * base, rel=1e-9)


def test_tf32_is_looser_than_fp32():
    """TF32's reduced mantissa must produce a looser tolerance than FP32."""
    tf32 = derive_parity_tolerance(1.0e3, "float32", "TF32")
    fp32 = derive_parity_tolerance(1.0e3, "float32", "FP32")
    assert tf32 > fp32


def test_golden_master_tolerance_float64_in_documented_range():
    """kappa=1e3, float64 -> 2.2e-11 (1e3 * 2.2e-16 * 100)."""
    tolerance = derive_golden_master_tolerance(1.0e3, "float64")
    assert tolerance == pytest.approx(2.2e-11, rel=1e-9)
    assert 1.0e-13 <= tolerance <= 1.0e-10


def test_golden_master_rejects_float32():
    """Golden master is a float64 bitwise check; float32 must fail closed."""
    with pytest.raises(ValueError, match="requires float64"):
        derive_golden_master_tolerance(1.0e3, "float32")  # type: ignore[arg-type]


def test_incompatible_dtype_backend_fails_closed():
    """float32 cannot run on the FP64 backend, and float64 cannot run TF32."""
    with pytest.raises(ValueError, match="incompatible"):
        derive_parity_tolerance(1.0e3, "float32", "FP64")
    with pytest.raises(ValueError, match="incompatible"):
        derive_parity_tolerance(1.0e3, "float64", "TF32")


def test_condition_number_of_identity_is_one():
    identity = tf.eye(4, dtype=tf.float64)
    assert condition_number_from_matrix(identity) == pytest.approx(1.0, rel=1e-9)


def test_condition_number_of_diagonal_is_ratio():
    """diag([1, 4, 9]) has sigma_max/sigma_min = 9."""
    matrix = tf.linalg.diag(tf.constant([1.0, 4.0, 9.0], tf.float64))
    assert condition_number_from_matrix(matrix) == pytest.approx(9.0, rel=1e-9)


def test_singular_matrix_widens_tolerance_instead_of_dividing_by_zero():
    """A singular input must return the conservative 1e10 estimate."""
    singular = tf.linalg.diag(tf.constant([1.0, 0.0], tf.float64))
    assert condition_number_from_matrix(singular) == pytest.approx(1.0e10, rel=1e-6)


def test_derive_tolerance_from_matrix_matches_manual_path():
    matrix = tf.linalg.diag(tf.constant([1.0, 4.0, 9.0], tf.float64))
    from_matrix = derive_tolerance_from_matrix(matrix, "float32", "TF32")
    manual = derive_parity_tolerance(9.0, "float32", "TF32")
    assert from_matrix == pytest.approx(manual, rel=1e-9)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
