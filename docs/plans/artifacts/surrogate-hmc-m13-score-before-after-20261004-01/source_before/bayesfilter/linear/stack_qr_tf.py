"""Standalone batched QR stack kernel for the factor SR-UKF route.

The kernel factors ``A @ A.T`` by applying thin QR to ``A.T``.  It contains no
covariance-to-factor fallback; callers provide the residual stack directly.
"""

from __future__ import annotations

from typing import Any

import tensorflow as tf


DEFAULT_RELATIVE_PIVOT_TOLERANCE = 1.0e-12


def _as_float(value: Any, name: str) -> tf.Tensor:
    tensor = tf.convert_to_tensor(value, dtype=tf.float64, name=name)
    return tensor


def _batched_transpose(value: tf.Tensor) -> tf.Tensor:
    return tf.linalg.matrix_transpose(value)


def _positive_qr(matrix: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Return thin QR factors with a deterministic positive diagonal."""

    matrix = _as_float(matrix, "matrix")
    tf.debugging.assert_rank(matrix, 3, message="QR matrix must have rank three")
    tf.debugging.assert_all_finite(matrix, "QR matrix contains NaN or Inf")
    rows = matrix.shape[-2]
    cols = matrix.shape[-1]
    if rows is None or cols is None or int(rows) < int(cols):
        raise ValueError("QR stack must have at least as many rows as columns")
    q, r = tf.linalg.qr(matrix, full_matrices=False)
    diagonal = tf.linalg.diag_part(r)
    tf.debugging.assert_all_finite(diagonal, "QR diagonal contains NaN or Inf")
    tf.debugging.assert_greater(
        tf.abs(diagonal),
        tf.zeros_like(diagonal),
        message="qr_pivot_nonpositive",
    )
    signs = tf.where(diagonal >= 0.0, tf.ones_like(diagonal), -tf.ones_like(diagonal))
    q = q * signs[:, tf.newaxis, :]
    r = signs[:, :, tf.newaxis] * r
    return q, r, tf.abs(diagonal)


def _relative_pivots(stack: tf.Tensor, pivots: tf.Tensor) -> tf.Tensor:
    row_norms = tf.linalg.norm(stack, axis=-1)
    safe_norms = tf.maximum(row_norms, tf.constant(1.0e-300, tf.float64))
    return tf.reduce_min(pivots / safe_norms, axis=-1)


def batched_stack_qr_lower(
    stack: tf.Tensor,
    d_stack: tf.Tensor | None = None,
    *,
    compute_covariance_diagnostics: bool = True,
    relative_pivot_tolerance: float = DEFAULT_RELATIVE_PIVOT_TOLERANCE,
) -> tuple[tf.Tensor, tf.Tensor | None, dict[str, tf.Tensor]]:
    """Factor a batched horizontal stack and optionally its first derivatives.

    Parameters
    ----------
    stack:
        Tensor with shape ``[B, N, K]`` and ``K >= N``.
    d_stack:
        Optional tensor with shape ``[B, P, N, K]``.
    """

    stack = _as_float(stack, "stack")
    if stack.shape.rank != 3:
        raise ValueError("stack must have shape [batch, dimension, columns]")
    batch, dimension, columns = stack.shape.as_list()
    if batch is None or dimension is None or columns is None:
        raise ValueError("stack dimensions must be statically known")
    if columns < dimension:
        raise ValueError("stack must have at least as many columns as rows")
    if relative_pivot_tolerance < 0.0:
        raise ValueError("relative_pivot_tolerance must be nonnegative")
    q, r, pivots = _positive_qr(_batched_transpose(stack))
    factor = _batched_transpose(r)
    scale = tf.maximum(tf.linalg.norm(stack, axis=[-2, -1]), tf.constant(1.0e-300, tf.float64))
    relative_min_pivot = tf.reduce_min(pivots, axis=-1) / scale
    tf.debugging.assert_greater_equal(
        relative_min_pivot,
        tf.cast(relative_pivot_tolerance, tf.float64),
        message="qr_relative_pivot_below_tolerance",
    )
    d_factor = None

    if d_stack is not None:
        d_stack = _as_float(d_stack, "d_stack")
        if d_stack.shape.rank != 4:
            raise ValueError("d_stack must have shape [batch, parameter, dimension, columns]")
        if tuple(d_stack.shape.as_list()[::2]) != (batch, dimension):
            raise ValueError("d_stack batch and dimension do not match stack")
        if d_stack.shape[2] != dimension or d_stack.shape[3] != columns:
            raise ValueError("d_stack spatial dimensions do not match stack")
        tf.debugging.assert_all_finite(d_stack, "d_stack contains NaN or Inf")
        d_matrix = tf.linalg.matrix_transpose(d_stack)
        d_r_rows = []
        for parameter_index in range(int(d_stack.shape[1])):
            d_a = d_matrix[:, parameter_index, :, :]
            d_a_t = tf.linalg.matrix_transpose(d_a)
            solved_t = tf.linalg.triangular_solve(
                tf.linalg.matrix_transpose(r),
                d_a_t,
                lower=True,
            )
            solved = tf.linalg.matrix_transpose(solved_t)
            e = tf.einsum("bki,bkj->bij", q, solved)
            lower = tf.linalg.band_part(e, -1, 0) - tf.linalg.diag(tf.linalg.diag_part(e))
            omega = lower - tf.linalg.matrix_transpose(lower)
            d_r_rows.append(tf.einsum("bij,bjk->bik", e - omega, r))
        d_r = tf.stack(d_r_rows, axis=1)
        d_factor = tf.linalg.matrix_transpose(d_r)
        tf.debugging.assert_all_finite(d_factor, "QR derivative contains NaN or Inf")

    # This residual is valid without constructing a covariance and is the
    # runtime diagnostic used by the admitted SR-UKF route.
    stack_reconstructed = tf.einsum("bki,bij->bkj", q, r)
    diagnostics = {
        "minimum_qr_pivot": tf.reduce_min(pivots, axis=-1),
        "relative_qr_pivot": relative_min_pivot,
        "factor_diagonal": tf.linalg.diag_part(factor),
        "stack_reconstruction_residual": tf.linalg.norm(
            tf.linalg.matrix_transpose(stack) - stack_reconstructed, axis=[-2, -1]
        ),
        "factor_reconstruction_metric": tf.constant(
            "direct_stack_qr_residual" if not compute_covariance_diagnostics
            else "covariance_reconstruction_residual"
        ),
    }
    if compute_covariance_diagnostics:
        diagnostics["factor_reconstruction_residual"] = tf.linalg.norm(
            factor @ tf.linalg.matrix_transpose(factor)
            - stack @ tf.linalg.matrix_transpose(stack),
            axis=[-2, -1],
        )
    else:
        # Keep the legacy key available to callers that aggregate diagnostics,
        # while making its runtime meaning explicit in the metric field.
        diagnostics["factor_reconstruction_residual"] = diagnostics["stack_reconstruction_residual"]
    if d_factor is not None:
        d_matrix = tf.linalg.matrix_transpose(d_stack)
        d_r = tf.linalg.matrix_transpose(d_factor)
        residual = d_matrix - tf.einsum("bki,bpij->bpkj", q, d_r)
        dp = d_stack.shape[1]
        flat_r = tf.repeat(tf.linalg.matrix_transpose(r), dp, axis=0)
        flat_rhs = tf.reshape(tf.linalg.matrix_transpose(residual), [batch * dp, dimension, columns])
        flat_dq = tf.linalg.triangular_solve(flat_r, flat_rhs, lower=True)
        d_q = tf.reshape(tf.linalg.matrix_transpose(flat_dq), [batch, dp, columns, dimension])
        d_qr = tf.einsum("bpki,bij->bpkj", d_q, r) + tf.einsum("bki,bpij->bpkj", q, d_r)
        diagnostics["qr_derivative_reconstruction_residual"] = tf.linalg.norm(
            d_matrix - d_qr, axis=[-2, -1]
        )
        if compute_covariance_diagnostics:
            reconstructed = tf.einsum("bpij,bkj->bpik", d_factor, factor) + tf.einsum(
                "bij,bpkj->bpik", factor, d_factor
            )
            d_covariance = tf.einsum("bpik,bkj->bpij", d_stack, tf.linalg.matrix_transpose(stack)) + tf.einsum(
                "bik,bpkj->bpij", stack, tf.linalg.matrix_transpose(d_stack)
            )
            diagnostics["factor_derivative_reconstruction_residual"] = tf.linalg.norm(
                reconstructed - d_covariance, axis=[-2, -1]
            )
        else:
            diagnostics["factor_derivative_reconstruction_residual"] = diagnostics[
                "qr_derivative_reconstruction_residual"
            ]
    return factor, d_factor, diagnostics


__all__ = ["DEFAULT_RELATIVE_PIVOT_TOLERANCE", "batched_stack_qr_lower"]
