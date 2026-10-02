"""Tolerance derivation for cross-lane parity and golden-master checks.

Derives relative tolerances from condition numbers, dtype, and execution backend.
Implements Option A decision from ledh-surrogate-hmc-executable-master-program.
"""

import tensorflow as tf
from typing import Literal


def derive_parity_tolerance(
    condition_number: float,
    dtype: Literal["float32", "float64"] = "float32",
    backend: Literal["TF32", "FP32", "FP64"] = "TF32",
) -> float:
    """
    Derive relative tolerance for cross-lane parity checks.

    Args:
        condition_number: Estimated condition number of the operation
        dtype: "float32" or "float64"
        backend: "TF32" (TensorFlow default on A100/H100), "FP32", or "FP64"

    Returns:
        Relative tolerance for np.allclose or tf.debugging.assert_near

    Formula:
        tolerance = condition_number × effective_epsilon × safety_factor

    For float32 TF32 (production regime):
        - float32 machine epsilon ≈ 1.2e-7
        - TF32 matmul reduces mantissa: 10 bits vs 23, effective eps ≈ 1e-5
        - safety_factor = 10 (accumulated rounding across operations)
        - Typical condition_number = 1e2 to 1e4 for particle filters
        - Result: tolerance = 1e-2 to 1e0

    For float32 FP32:
        - machine epsilon ≈ 1.2e-7
        - safety_factor = 10
        - Result: tolerance = 1e-5 to 1e-3

    For float64 FP64:
        - machine epsilon ≈ 2.2e-16
        - safety_factor = 10
        - Result: tolerance = 1e-13 to 1e-11
    """
    if dtype == "float32":
        machine_eps = 1.2e-7

        if backend == "TF32":
            # TF32: 10-bit mantissa vs float32's 23-bit
            # Precision loss: 2^(-10) / 2^(-23) ≈ 2^13 ≈ 8192
            # Effective epsilon: 1.2e-7 * 8192 ≈ 1e-3
            # Conservative: use 1e-5 (allows 2 orders of magnitude margin)
            effective_eps = 1e-5
        elif backend == "FP32":
            effective_eps = machine_eps
        else:
            raise ValueError(f"backend={backend} incompatible with dtype={dtype}")

    elif dtype == "float64":
        machine_eps = 2.2e-16

        if backend == "FP64":
            effective_eps = machine_eps
        else:
            raise ValueError(f"backend={backend} incompatible with dtype={dtype}")
    else:
        raise ValueError(f"Unknown dtype: {dtype}")

    safety_factor = 10.0
    tolerance = condition_number * effective_eps * safety_factor

    return float(tolerance)


def derive_golden_master_tolerance(
    condition_number: float,
    dtype: Literal["float64"] = "float64",
) -> float:
    """
    Derive tolerance for golden master (bitwise-reproducibility check).

    Golden masters test deterministic CPU-only float64 operations.
    Use for regression tests that verify "refactor changed nothing."

    Args:
        condition_number: Estimated condition number
        dtype: "float64" (required for golden master)

    Returns:
        Absolute tolerance (relative tolerance × typical scale)

    For float64 CPU-only:
        - machine_epsilon ≈ 2.2e-16
        - Deterministic ops → tolerance = condition_number × eps × safety
        - safety_factor = 100 (more operations than parity checks)
        - Typical result: 1e-12 to 1e-10
    """
    if dtype != "float64":
        raise ValueError(
            f"Golden master requires float64, got {dtype}. "
            "Golden masters are bitwise-reproducibility checks for CPU-only "
            "deterministic operations."
        )

    machine_eps = 2.2e-16
    safety_factor = 100.0  # More accumulated operations than parity checks

    tolerance = condition_number * machine_eps * safety_factor

    return float(tolerance)


def condition_number_from_matrix(matrix: tf.Tensor) -> float:
    """
    Estimate the spectral condition number of a matrix.

    TensorFlow backend (CLAUDE.md Backend Rule): this is admission logic on
    the runtime path, not diagnostic code, so it must not use NumPy.

    Args:
        matrix: Input matrix, shape [..., M, N]

    Returns:
        Condition number sigma_max / sigma_min. Returns 1e10 as a conservative
        estimate when the matrix is numerically singular, so that a singular
        input widens the tolerance rather than producing a division blow-up.
    """
    singular_values = tf.linalg.svd(matrix, compute_uv=False)

    sigma_max = tf.reduce_max(singular_values)
    sigma_min = tf.reduce_min(singular_values)

    singular_floor = tf.cast(1.0e-14, singular_values.dtype)

    condition_number = tf.where(
        sigma_min < singular_floor,
        tf.cast(1.0e10, singular_values.dtype),
        sigma_max / tf.maximum(sigma_min, singular_floor),
    )

    return float(condition_number.numpy())


def derive_tolerance_from_matrix(
    matrix: tf.Tensor,
    dtype: Literal["float32", "float64"] = "float32",
    backend: Literal["TF32", "FP32", "FP64"] = "TF32",
) -> float:
    """
    Derive tolerance from a matrix's measured condition number.

    Use when the condition number is not known a priori.

    Args:
        matrix: Input matrix, shape [..., M, N]
        dtype: Target dtype for the computation
        backend: Execution backend

    Returns:
        Relative tolerance appropriate for operations on this matrix
    """
    condition_number = condition_number_from_matrix(matrix)
    return derive_parity_tolerance(condition_number, dtype, backend)
