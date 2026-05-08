"""TensorFlow QR and Cholesky factor derivative helpers."""

from __future__ import annotations

import tensorflow as tf


def symmetrize(matrix: tf.Tensor) -> tf.Tensor:
    """Return the symmetric part of a matrix."""

    return 0.5 * (matrix + tf.linalg.matrix_transpose(matrix))


def factor_solve(factor: tf.Tensor, rhs: tf.Tensor) -> tf.Tensor:
    """Solve ``(factor @ factor.T) x = rhs`` with a lower triangular factor."""

    factor = tf.convert_to_tensor(factor, dtype=tf.float64)
    rhs = tf.convert_to_tensor(rhs, dtype=tf.float64)
    if rhs.shape.rank == 1:
        rhs_matrix = rhs[:, tf.newaxis]
        first = tf.linalg.triangular_solve(factor, rhs_matrix, lower=True)
        second = tf.linalg.triangular_solve(tf.transpose(factor), first, lower=False)
        return second[:, 0]
    first = tf.linalg.triangular_solve(factor, rhs, lower=True)
    return tf.linalg.triangular_solve(tf.transpose(factor), first, lower=False)


def trace_factor_solve(factor: tf.Tensor, matrix: tf.Tensor) -> tf.Tensor:
    """Return ``trace(inv(factor @ factor.T) @ matrix)``."""

    return tf.linalg.trace(factor_solve(factor, matrix))


def right_solve_upper(matrix: tf.Tensor, upper: tf.Tensor) -> tf.Tensor:
    """Return ``matrix @ inv(upper)`` without explicitly forming the inverse."""

    solved_t = tf.linalg.triangular_solve(
        tf.transpose(upper),
        tf.transpose(matrix),
        lower=True,
    )
    return tf.transpose(solved_t)


def qr_positive(matrix: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Thin QR with a positive diagonal in the triangular factor."""

    matrix = tf.convert_to_tensor(matrix, dtype=tf.float64)
    q, r = tf.linalg.qr(matrix, full_matrices=False)
    signs = tf.sign(tf.linalg.diag_part(r))
    signs = tf.where(tf.equal(signs, 0.0), tf.ones_like(signs), signs)
    return q * signs[tf.newaxis, :], signs[:, tf.newaxis] * r


def omega_from_a(a: tf.Tensor) -> tf.Tensor:
    """Return the skew component used by the first-order QR derivative split."""

    lower = tf.linalg.band_part(a, -1, 0) - tf.linalg.diag(tf.linalg.diag_part(a))
    return lower - tf.transpose(lower)


def gamma_from_b_and_c(b: tf.Tensor, c: tf.Tensor) -> tf.Tensor:
    """Return the second-order QR split operator."""

    lower_b = tf.linalg.band_part(b, -1, 0) - tf.linalg.diag(tf.linalg.diag_part(b))
    upper_c = tf.linalg.band_part(c, 0, -1) - tf.linalg.diag(tf.linalg.diag_part(c))
    return lower_b + upper_c - tf.transpose(lower_b) + 0.5 * tf.linalg.diag(
        tf.linalg.diag_part(c)
    )


def qr_factor_derivatives(
    matrix: tf.Tensor,
    dmatrix: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Return first derivatives of a positive-diagonal thin QR factorization."""

    q, r = qr_positive(matrix)
    dmatrix_r_inv = right_solve_upper(dmatrix, r)
    a = tf.transpose(q) @ dmatrix_r_inv
    omega = omega_from_a(a)
    dr = (a - omega) @ r
    identity_rows = tf.eye(tf.shape(q)[0], dtype=tf.float64)
    dq = q @ omega + (identity_rows - q @ tf.transpose(q)) @ dmatrix_r_inv
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
    b = tf.transpose(q) @ effective_r_inv
    c = -tf.transpose(dq_i) @ dq_j - tf.transpose(dq_j) @ dq_i
    gamma = gamma_from_b_and_c(b, c)
    d2r = (b - gamma) @ r
    identity_rows = tf.eye(tf.shape(q)[0], dtype=tf.float64)
    d2q = q @ gamma + (identity_rows - q @ tf.transpose(q)) @ effective_r_inv
    return q, r, d2q, d2r


def qr_factor_full_derivatives(
    matrix: tf.Tensor,
    dmatrix: tf.Tensor,
    d2matrix: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Return QR factors and all first/second derivatives for a parameter grid."""

    parameter_dim = int(dmatrix.shape[0])
    q, r = qr_positive(matrix)
    dq_values = []
    dr_values = []
    for i in range(parameter_dim):
        _, _, dq_i, dr_i = qr_factor_derivatives(matrix, dmatrix[i])
        dq_values.append(dq_i)
        dr_values.append(dr_i)
    d2q_rows = []
    d2r_rows = []
    for i in range(parameter_dim):
        d2q_values = []
        d2r_values = []
        for j in range(parameter_dim):
            _, _, d2q_ij, d2r_ij = qr_factor_second_derivatives(
                matrix,
                dmatrix[i],
                dmatrix[j],
                d2matrix[i, j],
            )
            d2q_values.append(d2q_ij)
            d2r_values.append(d2r_ij)
        d2q_rows.append(tf.stack(d2q_values, axis=0))
        d2r_rows.append(tf.stack(d2r_values, axis=0))
    return (
        q,
        r,
        tf.stack(dq_values, axis=0),
        tf.stack(dr_values, axis=0),
        tf.stack(d2q_rows, axis=0),
        tf.stack(d2r_rows, axis=0),
    )


def transpose_factor_derivatives(
    r: tf.Tensor,
    dr: tf.Tensor,
    d2r: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Transpose a QR upper factor and its derivative arrays."""

    return tf.transpose(r), tf.linalg.matrix_transpose(dr), tf.linalg.matrix_transpose(d2r)


def stack_qr_lower_factor_derivatives(
    stack: tf.Tensor,
    dstack: tf.Tensor,
    d2stack: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Factor ``stack @ stack.T`` by QR of ``stack.T`` and differentiate it."""

    matrix = tf.transpose(stack)
    dmatrix = tf.linalg.matrix_transpose(dstack)
    d2matrix = tf.linalg.matrix_transpose(d2stack)
    _, r, _, dr, _, d2r = qr_factor_full_derivatives(matrix, dmatrix, d2matrix)
    factor, dfactor, d2factor = transpose_factor_derivatives(r, dr, d2r)
    return factor, dfactor, d2factor, tf.reduce_min(tf.linalg.diag_part(factor))


def cholesky_factor(covariance: tf.Tensor, jitter: tf.Tensor | float = 0.0) -> tf.Tensor:
    """Return a lower Cholesky factor of a symmetrized covariance matrix."""

    covariance = symmetrize(tf.convert_to_tensor(covariance, dtype=tf.float64))
    jitter_tensor = tf.cast(jitter, tf.float64)
    return tf.linalg.cholesky(
        covariance + jitter_tensor * tf.eye(tf.shape(covariance)[0], dtype=tf.float64)
    )


def lower_factor_from_horizontal_stack(stack: tf.Tensor) -> tf.Tensor:
    """Return lower factor ``L`` such that ``L L.T = stack stack.T``."""

    _, r = qr_positive(tf.transpose(stack))
    return tf.transpose(r)


def cholesky_factor_derivatives(
    covariance: tf.Tensor,
    dcovariance: tf.Tensor,
    d2covariance: tf.Tensor,
    jitter: tf.Tensor | float = 0.0,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Differentiate ``covariance + jitter I = L L.T``."""

    covariance = symmetrize(tf.convert_to_tensor(covariance, dtype=tf.float64))
    dcovariance = tf.convert_to_tensor(dcovariance, dtype=tf.float64)
    d2covariance = tf.convert_to_tensor(d2covariance, dtype=tf.float64)
    factor = cholesky_factor(covariance, jitter=jitter)
    parameter_dim = int(dcovariance.shape[0])
    dfactor_values = []
    for i in range(parameter_dim):
        left = tf.linalg.triangular_solve(
            factor,
            symmetrize(dcovariance[i]),
            lower=True,
        )
        b_i = right_solve_upper(left, tf.transpose(factor))
        g_i = tf.linalg.band_part(b_i, -1, 0) - 0.5 * tf.linalg.diag(
            tf.linalg.diag_part(b_i)
        )
        dfactor_values.append(factor @ g_i)
    dfactor = tf.stack(dfactor_values, axis=0)

    d2factor_rows = []
    for i in range(parameter_dim):
        d2factor_values = []
        for j in range(parameter_dim):
            effective = (
                symmetrize(d2covariance[i, j])
                - dfactor[i] @ tf.transpose(dfactor[j])
                - dfactor[j] @ tf.transpose(dfactor[i])
            )
            left = tf.linalg.triangular_solve(factor, effective, lower=True)
            c_ij = right_solve_upper(left, tf.transpose(factor))
            h_ij = tf.linalg.band_part(c_ij, -1, 0) - 0.5 * tf.linalg.diag(
                tf.linalg.diag_part(c_ij)
            )
            d2factor_values.append(factor @ h_ij)
        d2factor_rows.append(tf.stack(d2factor_values, axis=0))
    return factor, dfactor, tf.stack(d2factor_rows, axis=0)


def factor_covariance_derivatives(
    factor: tf.Tensor,
    dfactor: tf.Tensor,
    d2factor: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Convert factor derivatives into covariance derivatives."""

    factor = tf.convert_to_tensor(factor, dtype=tf.float64)
    dfactor = tf.convert_to_tensor(dfactor, dtype=tf.float64)
    d2factor = tf.convert_to_tensor(d2factor, dtype=tf.float64)
    covariance = factor @ tf.transpose(factor)
    parameter_dim = int(dfactor.shape[0])
    dcovariance_values = []
    d2covariance_rows = []
    for i in range(parameter_dim):
        dcovariance_values.append(
            symmetrize(dfactor[i] @ tf.transpose(factor) + factor @ tf.transpose(dfactor[i]))
        )
        d2covariance_values = []
        for j in range(parameter_dim):
            d2covariance_values.append(
                symmetrize(
                    d2factor[i, j] @ tf.transpose(factor)
                    + dfactor[i] @ tf.transpose(dfactor[j])
                    + dfactor[j] @ tf.transpose(dfactor[i])
                    + factor @ tf.transpose(d2factor[i, j])
                )
            )
        d2covariance_rows.append(tf.stack(d2covariance_values, axis=0))
    return covariance, tf.stack(dcovariance_values, axis=0), tf.stack(d2covariance_rows, axis=0)


def stack_covariance_derivatives(
    stack: tf.Tensor,
    dstack: tf.Tensor,
    d2stack: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Return derivatives of ``stack @ stack.T`` for reconstruction checks."""

    stack = tf.convert_to_tensor(stack, dtype=tf.float64)
    dstack = tf.convert_to_tensor(dstack, dtype=tf.float64)
    d2stack = tf.convert_to_tensor(d2stack, dtype=tf.float64)
    covariance = stack @ tf.transpose(stack)
    parameter_dim = int(dstack.shape[0])
    dcovariance_values = []
    d2covariance_rows = []
    for i in range(parameter_dim):
        dcovariance_values.append(
            symmetrize(dstack[i] @ tf.transpose(stack) + stack @ tf.transpose(dstack[i]))
        )
        d2covariance_values = []
        for j in range(parameter_dim):
            d2covariance_values.append(
                symmetrize(
                    d2stack[i, j] @ tf.transpose(stack)
                    + dstack[i] @ tf.transpose(dstack[j])
                    + dstack[j] @ tf.transpose(dstack[i])
                    + stack @ tf.transpose(d2stack[i, j])
                )
            )
        d2covariance_rows.append(tf.stack(d2covariance_values, axis=0))
    return covariance, tf.stack(dcovariance_values, axis=0), tf.stack(d2covariance_rows, axis=0)


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
