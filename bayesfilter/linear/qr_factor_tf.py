"""TensorFlow QR and Cholesky factor derivative helpers."""

from __future__ import annotations

import tensorflow as tf

from bayesfilter.linear.dtypes_tf import as_float_tensor, common_floating_dtype


def symmetrize(matrix: tf.Tensor) -> tf.Tensor:
    """Return the symmetric part of a matrix."""

    return 0.5 * (matrix + tf.linalg.matrix_transpose(matrix))


def factor_solve(factor: tf.Tensor, rhs: tf.Tensor) -> tf.Tensor:
    """Solve ``(factor @ factor.T) x = rhs`` with a lower triangular factor."""

    dtype = common_floating_dtype(factor, rhs, context="factor_solve inputs")
    factor = as_float_tensor(factor, dtype, name="factor")
    rhs = as_float_tensor(rhs, dtype, name="rhs")
    if rhs.shape.rank == 1:
        rhs_matrix = rhs[:, tf.newaxis]
        first = tf.linalg.triangular_solve(factor, rhs_matrix, lower=True)
        second = tf.linalg.triangular_solve(tf.linalg.matrix_transpose(factor), first, lower=False)
        return second[:, 0]
    first = tf.linalg.triangular_solve(factor, rhs, lower=True)
    return tf.linalg.triangular_solve(tf.linalg.matrix_transpose(factor), first, lower=False)


def trace_factor_solve(factor: tf.Tensor, matrix: tf.Tensor) -> tf.Tensor:
    """Return ``trace(inv(factor @ factor.T) @ matrix)``."""

    return tf.linalg.trace(factor_solve(factor, matrix))


def right_solve_upper(matrix: tf.Tensor, upper: tf.Tensor) -> tf.Tensor:
    """Return ``matrix @ inv(upper)`` without explicitly forming the inverse."""

    solved_t = tf.linalg.triangular_solve(
        tf.linalg.matrix_transpose(upper),
        tf.linalg.matrix_transpose(matrix),
        lower=True,
    )
    return tf.linalg.matrix_transpose(solved_t)


def qr_positive(matrix: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Thin QR with a positive diagonal in the triangular factor."""

    dtype = common_floating_dtype(matrix, context="qr_positive matrix")
    matrix = as_float_tensor(matrix, dtype, name="matrix")
    q, r = tf.linalg.qr(matrix, full_matrices=False)
    signs = tf.sign(tf.linalg.diag_part(r))
    signs = tf.where(tf.equal(signs, 0.0), tf.ones_like(signs), signs)
    return q * signs[tf.newaxis, :], signs[:, tf.newaxis] * r


def omega_from_a(a: tf.Tensor) -> tf.Tensor:
    """Return the skew component used by the first-order QR derivative split."""

    lower = tf.linalg.band_part(a, -1, 0) - tf.linalg.diag(tf.linalg.diag_part(a))
    return lower - tf.linalg.matrix_transpose(lower)


def gamma_from_b_and_c(b: tf.Tensor, c: tf.Tensor) -> tf.Tensor:
    """Return the second-order QR split operator."""

    lower_b = tf.linalg.band_part(b, -1, 0) - tf.linalg.diag(tf.linalg.diag_part(b))
    upper_c = tf.linalg.band_part(c, 0, -1) - tf.linalg.diag(tf.linalg.diag_part(c))
    return lower_b + upper_c - tf.linalg.matrix_transpose(lower_b) + 0.5 * tf.linalg.diag(
        tf.linalg.diag_part(c)
    )


def qr_factor_derivatives(
    matrix: tf.Tensor,
    dmatrix: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Return first derivatives of a positive-diagonal thin QR factorization."""

    q, r = qr_positive(matrix)
    dmatrix_r_inv = right_solve_upper(dmatrix, r)
    a = tf.linalg.matrix_transpose(q) @ dmatrix_r_inv
    omega = omega_from_a(a)
    dr = (a - omega) @ r
    identity_rows = tf.eye(tf.shape(q)[0], dtype=q.dtype)
    dq = q @ omega + (identity_rows - q @ tf.linalg.matrix_transpose(q)) @ dmatrix_r_inv
    return q, r, dq, dr


def qr_factor_second_derivatives(
    matrix: tf.Tensor,
    dmatrix_i: tf.Tensor,
    dmatrix_j: tf.Tensor,
    d2matrix_ij: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Return second derivatives of a positive-diagonal thin QR factorization."""

    q, r, dq_i, dr_i = qr_factor_derivatives(matrix, dmatrix_i)
    _, _, dq_j, dr_j = qr_factor_derivatives(matrix, dmatrix_j)
    effective = d2matrix_ij - dq_i @ dr_j - dq_j @ dr_i
    effective_r_inv = right_solve_upper(effective, r)
    b = tf.linalg.matrix_transpose(q) @ effective_r_inv
    c = -tf.linalg.matrix_transpose(dq_i) @ dq_j - tf.linalg.matrix_transpose(dq_j) @ dq_i
    gamma = gamma_from_b_and_c(b, c)
    d2r = (b - gamma) @ r
    identity_rows = tf.eye(tf.shape(q)[0], dtype=q.dtype)
    d2q = q @ gamma + (identity_rows - q @ tf.linalg.matrix_transpose(q)) @ effective_r_inv
    return q, r, d2q, d2r


def qr_factor_full_derivatives(
    matrix: tf.Tensor,
    dmatrix: tf.Tensor,
    d2matrix: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Return QR factors and all first/second derivatives for a parameter grid."""

    q, r, dq, dr = qr_factor_derivatives(matrix, dmatrix)
    effective = d2matrix - dq[:, None] @ dr[None, :] - dq[None, :] @ dr[:, None]
    scaled = right_solve_upper(effective, r)
    b = tf.linalg.matrix_transpose(q) @ scaled
    c = -(tf.linalg.matrix_transpose(dq[:, None]) @ dq[None, :]
          + tf.linalg.matrix_transpose(dq[None, :]) @ dq[:, None])
    gamma = gamma_from_b_and_c(b, c)
    d2r = (b - gamma) @ r
    d2q = q @ gamma + scaled - q @ b
    return q, r, dq, dr, d2q, d2r


def transpose_factor_derivatives(
    r: tf.Tensor,
    dr: tf.Tensor,
    d2r: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Transpose a QR upper factor and its derivative arrays."""

    return tf.linalg.matrix_transpose(r), tf.linalg.matrix_transpose(dr), tf.linalg.matrix_transpose(d2r)


def stack_qr_lower_factor_derivatives(
    stack: tf.Tensor,
    dstack: tf.Tensor,
    d2stack: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Factor ``stack @ stack.T`` by QR of ``stack.T`` and differentiate it."""

    matrix = tf.linalg.matrix_transpose(stack)
    dmatrix = tf.linalg.matrix_transpose(dstack)
    d2matrix = tf.linalg.matrix_transpose(d2stack)
    _, r, _, dr, _, d2r = qr_factor_full_derivatives(matrix, dmatrix, d2matrix)
    factor, dfactor, d2factor = transpose_factor_derivatives(r, dr, d2r)
    return factor, dfactor, d2factor, tf.reduce_min(tf.linalg.diag_part(factor))


def stack_qr_lower_factor_first_derivatives(
    stack: tf.Tensor,
    dstack: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Factor ``stack @ stack.T`` and return only first derivatives."""

    _, r, _, dr = qr_factor_derivatives(
        tf.linalg.matrix_transpose(stack), tf.linalg.matrix_transpose(dstack)
    )
    factor = tf.linalg.matrix_transpose(r)
    return factor, tf.linalg.matrix_transpose(dr), tf.reduce_min(tf.linalg.diag_part(factor))


def cholesky_factor(covariance: tf.Tensor, jitter: tf.Tensor | float = 0.0) -> tf.Tensor:
    """Return a lower Cholesky factor of a symmetrized covariance matrix."""

    dtype = common_floating_dtype(covariance, jitter, context="cholesky_factor inputs")
    covariance = symmetrize(as_float_tensor(covariance, dtype, name="covariance"))
    jitter_tensor = as_float_tensor(jitter, dtype, name="jitter")
    return tf.linalg.cholesky(
        covariance + jitter_tensor * tf.eye(tf.shape(covariance)[0], dtype=dtype)
    )


def lower_factor_from_horizontal_stack(stack: tf.Tensor) -> tf.Tensor:
    """Return lower factor ``L`` such that ``L L.T = stack stack.T``."""

    _, r = qr_positive(tf.linalg.matrix_transpose(stack))
    return tf.linalg.matrix_transpose(r)


def cholesky_factor_derivatives(
    covariance: tf.Tensor,
    dcovariance: tf.Tensor,
    d2covariance: tf.Tensor,
    jitter: tf.Tensor | float = 0.0,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Differentiate ``covariance + jitter I = L L.T``."""

    dtype = common_floating_dtype(
        covariance,
        dcovariance,
        d2covariance,
        jitter,
        context="cholesky_factor_derivatives inputs",
    )
    covariance = symmetrize(as_float_tensor(covariance, dtype, name="covariance"))
    dcovariance = as_float_tensor(dcovariance, dtype, name="dcovariance")
    d2covariance = as_float_tensor(d2covariance, dtype, name="d2covariance")
    factor = cholesky_factor(covariance, jitter=jitter)
    dfactor = _cholesky_direction(factor, symmetrize(dcovariance))
    effective = (symmetrize(d2covariance)
                 - dfactor[:, None] @ tf.linalg.matrix_transpose(dfactor[None, :])
                 - dfactor[None, :] @ tf.linalg.matrix_transpose(dfactor[:, None]))
    return factor, dfactor, _cholesky_direction(factor, effective)


def _cholesky_direction(factor: tf.Tensor, direction: tf.Tensor) -> tf.Tensor:
    """Analytical Cholesky tangent over arbitrary leading derivative axes."""
    left = tf.linalg.triangular_solve(factor, direction, lower=True)
    scaled = right_solve_upper(left, tf.linalg.matrix_transpose(factor))
    phi = tf.linalg.band_part(scaled, -1, 0) - 0.5 * tf.linalg.diag(tf.linalg.diag_part(scaled))
    return factor @ phi


def cholesky_factor_first_derivatives(
    covariance: tf.Tensor,
    dcovariance: tf.Tensor,
    jitter: tf.Tensor | float = 0.0,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Differentiate ``covariance + jitter I = L L.T`` to first order."""

    dtype = common_floating_dtype(
        covariance,
        dcovariance,
        jitter,
        context="cholesky_factor_first_derivatives inputs",
    )
    covariance = symmetrize(as_float_tensor(covariance, dtype, name="covariance"))
    dcovariance = as_float_tensor(dcovariance, dtype, name="dcovariance")
    factor = cholesky_factor(covariance, jitter=jitter)
    return factor, _cholesky_direction(factor, symmetrize(dcovariance))


def factor_covariance_derivatives(
    factor: tf.Tensor,
    dfactor: tf.Tensor,
    d2factor: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Convert factor derivatives into covariance derivatives."""

    dtype = common_floating_dtype(
        factor,
        dfactor,
        d2factor,
        context="factor_covariance_derivatives inputs",
    )
    factor = as_float_tensor(factor, dtype, name="factor")
    dfactor = as_float_tensor(dfactor, dtype, name="dfactor")
    d2factor = as_float_tensor(d2factor, dtype, name="d2factor")
    covariance = factor @ tf.linalg.matrix_transpose(factor)
    dcovariance = symmetrize(dfactor @ tf.linalg.matrix_transpose(factor)
                             + factor @ tf.linalg.matrix_transpose(dfactor))
    d2covariance = symmetrize(
        d2factor @ tf.linalg.matrix_transpose(factor)
        + dfactor[:, None] @ tf.linalg.matrix_transpose(dfactor[None, :])
        + dfactor[None, :] @ tf.linalg.matrix_transpose(dfactor[:, None])
        + factor @ tf.linalg.matrix_transpose(d2factor)
    )
    return covariance, dcovariance, d2covariance


def factor_covariance_first_derivatives(
    factor: tf.Tensor,
    dfactor: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Convert first factor derivatives into first covariance derivatives."""

    dtype = common_floating_dtype(
        factor,
        dfactor,
        context="factor_covariance_first_derivatives inputs",
    )
    factor = as_float_tensor(factor, dtype, name="factor")
    dfactor = as_float_tensor(dfactor, dtype, name="dfactor")
    covariance = factor @ tf.linalg.matrix_transpose(factor)
    return covariance, symmetrize(dfactor @ tf.linalg.matrix_transpose(factor)
                                 + factor @ tf.linalg.matrix_transpose(dfactor))


def stack_covariance_derivatives(
    stack: tf.Tensor,
    dstack: tf.Tensor,
    d2stack: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Return derivatives of ``stack @ stack.T`` for reconstruction checks."""

    return factor_covariance_derivatives(stack, dstack, d2stack)


def factor_derivative_reconstruction_errors(
    factor: tf.Tensor,
    dfactor: tf.Tensor,
    d2factor: tf.Tensor,
    covariance: tf.Tensor,
    dcovariance: tf.Tensor,
    d2covariance: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Return max first- and second-order covariance reconstruction errors."""

    reconstructed_covariance, reconstructed_dcovariance, reconstructed_d2covariance = (
        factor_covariance_derivatives(factor, dfactor, d2factor)
    )
    first_error = tf.linalg.norm(reconstructed_covariance - covariance)
    first_error = tf.maximum(
        first_error,
        tf.linalg.norm(reconstructed_dcovariance - dcovariance),
    )
    second_error = tf.linalg.norm(reconstructed_d2covariance - d2covariance)
    return first_error, second_error
