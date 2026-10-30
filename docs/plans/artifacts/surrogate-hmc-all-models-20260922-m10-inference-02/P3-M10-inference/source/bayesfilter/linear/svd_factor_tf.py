"""TensorFlow eigen/SVD helpers for PSD covariance solves."""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf


@dataclass(frozen=True)
class PrincipalSqrtFirstDerivativeDiagnostics:
    """Strict-SPD principal-square-root factor and first-derivative diagnostics."""

    eigenvalues: tf.Tensor
    floored_eigenvalues: tf.Tensor
    eigenvectors: tf.Tensor
    factor: tf.Tensor
    d_factor: tf.Tensor
    implemented_covariance: tf.Tensor
    floor_count: tf.Tensor
    min_eigenvalue: tf.Tensor
    psd_projection_residual: tf.Tensor
    sylvester_residual: tf.Tensor
    reconstruction_residual: tf.Tensor


def symmetrize(matrix: tf.Tensor) -> tf.Tensor:
    """Return the symmetric part of a covariance-like matrix."""

    matrix = tf.convert_to_tensor(matrix, dtype=tf.float64)
    return 0.5 * (matrix + tf.linalg.matrix_transpose(matrix))


def psd_eigh(
    covariance: tf.Tensor,
    singular_floor: tf.Tensor | float,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Return PSD eigensystem and implemented covariance under eigenvalue floor."""

    covariance = symmetrize(covariance)
    floor = tf.cast(singular_floor, tf.float64)
    eigenvalues, eigenvectors = tf.linalg.eigh(covariance)
    floored_eigenvalues = tf.maximum(eigenvalues, floor)
    implemented_covariance = (
        eigenvectors
        @ tf.linalg.diag(floored_eigenvalues)
        @ tf.linalg.matrix_transpose(eigenvectors)
    )
    projection_residual = tf.linalg.norm(implemented_covariance - covariance)
    return (
        eigenvalues,
        floored_eigenvalues,
        eigenvectors,
        implemented_covariance,
        projection_residual,
    )


def psd_eigh_graph_status(
    covariance: tf.Tensor,
    singular_floor: tf.Tensor | float,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Return a PSD eigensystem after guarding nonfinite eigensolver inputs.

    This is a graph-status helper for callers that must return tensor
    diagnostics instead of letting ``tf.linalg.eigh`` abort on invalid
    covariance entries.  When the covariance is nonfinite, the eigensolver sees
    a benign identity covariance only so graph execution can complete; callers
    must treat the returned score/value payload as blocked diagnostics.
    """

    covariance = symmetrize(covariance)
    floor = tf.cast(singular_floor, tf.float64)
    covariance_is_finite = tf.reduce_all(tf.math.is_finite(covariance))
    floor_is_finite = tf.math.is_finite(floor)
    valid_eigensolver_input = tf.logical_and(covariance_is_finite, floor_is_finite)
    dimension = tf.shape(covariance)[0]
    safe_floor = tf.cond(
        floor_is_finite,
        lambda: floor,
        lambda: tf.constant(1.0, dtype=tf.float64),
    )
    benign_scale = tf.maximum(tf.abs(safe_floor), tf.constant(1.0, dtype=tf.float64))
    benign_covariance = benign_scale * tf.eye(dimension, dtype=tf.float64)
    safe_covariance = tf.cond(
        valid_eigensolver_input,
        lambda: covariance,
        lambda: benign_covariance,
    )
    eigenvalues, eigenvectors = tf.linalg.eigh(safe_covariance)
    floored_eigenvalues = tf.maximum(eigenvalues, safe_floor)
    implemented_covariance = (
        eigenvectors
        @ tf.linalg.diag(floored_eigenvalues)
        @ tf.linalg.matrix_transpose(eigenvectors)
    )
    residual = tf.cond(
        valid_eigensolver_input,
        lambda: tf.linalg.norm(implemented_covariance - covariance),
        lambda: tf.constant(float("nan"), dtype=tf.float64),
    )
    return (
        eigenvalues,
        floored_eigenvalues,
        eigenvectors,
        implemented_covariance,
        residual,
        tf.logical_not(valid_eigensolver_input),
    )


def principal_sqrt_frechet_derivative_from_eigh(
    eigenvectors: tf.Tensor,
    sqrt_eigenvalues: tf.Tensor,
    d_covariance: tf.Tensor,
) -> tf.Tensor:
    """Return the strict-SPD principal-square-root Frechet derivative.

    The derivative is expressed in the eigenbasis of the strict-SPD covariance
    through the Sylvester/Frechet identity and therefore avoids eigen-gap
    denominators associated with eigenvector derivatives.
    """

    eigenvectors = tf.convert_to_tensor(eigenvectors, dtype=tf.float64)
    sqrt_eigenvalues = tf.convert_to_tensor(sqrt_eigenvalues, dtype=tf.float64)
    d_covariance = symmetrize(tf.convert_to_tensor(d_covariance, dtype=tf.float64))
    transformed_rhs = (
        tf.linalg.matrix_transpose(eigenvectors)[tf.newaxis, :, :]
        @ d_covariance
        @ eigenvectors[tf.newaxis, :, :]
    )
    denominator = sqrt_eigenvalues[:, tf.newaxis] + sqrt_eigenvalues[tf.newaxis, :]
    transformed_derivative = transformed_rhs / denominator[tf.newaxis, :, :]
    return symmetrize(
        eigenvectors[tf.newaxis, :, :]
        @ transformed_derivative
        @ tf.linalg.matrix_transpose(eigenvectors)[tf.newaxis, :, :]
    )


def strict_spd_principal_sqrt_first_derivatives(
    covariance: tf.Tensor,
    d_covariance: tf.Tensor,
    *,
    singular_floor: tf.Tensor | float,
    label: str,
    lyapunov_tolerance: tf.Tensor | float = 1.0e-10,
) -> PrincipalSqrtFirstDerivativeDiagnostics:
    """Return strict-SPD principal-square-root factor derivatives.

    This helper is intentionally strict: it blocks when a floor is active or the
    covariance is not strictly SPD.  The derivative contract is the Sylvester
    derivative of the principal square root of the same covariance used to place
    points or solve innovation systems on the promoted principal-square-root
    branch.
    """

    covariance = symmetrize(tf.convert_to_tensor(covariance, dtype=tf.float64))
    d_covariance = symmetrize(tf.convert_to_tensor(d_covariance, dtype=tf.float64))
    singular_floor = tf.convert_to_tensor(singular_floor, dtype=tf.float64)
    (
        eigenvalues,
        floored_eigenvalues,
        eigenvectors,
        _implemented_covariance,
        psd_projection_residual,
    ) = psd_eigh(covariance, singular_floor)
    active_floors = floor_count(eigenvalues, singular_floor)
    min_eigenvalue = tf.reduce_min(eigenvalues)
    assertions = [
        tf.debugging.assert_equal(
            active_floors,
            tf.constant(0, dtype=tf.int32),
            message=f"blocked_active_floor: {label} floor is active",
        ),
        tf.debugging.assert_greater(
            min_eigenvalue,
            singular_floor,
            message=f"blocked_non_spd_principal_sqrt: {label} is not strict SPD",
        ),
        tf.debugging.assert_all_finite(
            eigenvalues,
            f"blocked_nonfinite_factor: {label} eigenvalues are nonfinite",
        ),
    ]
    with tf.control_dependencies(assertions):
        eigenvalues = tf.identity(eigenvalues)
        floored_eigenvalues = tf.identity(floored_eigenvalues)
        eigenvectors = tf.identity(eigenvectors)

    sqrt_eigenvalues = tf.sqrt(floored_eigenvalues)
    factor = (
        eigenvectors
        @ tf.linalg.diag(sqrt_eigenvalues)
        @ tf.linalg.matrix_transpose(eigenvectors)
    )
    d_factor = principal_sqrt_frechet_derivative_from_eigh(
        eigenvectors,
        sqrt_eigenvalues,
        d_covariance,
    )
    implemented_covariance = symmetrize(factor @ tf.linalg.matrix_transpose(factor))
    psd_projection_residual = tf.linalg.norm(implemented_covariance - covariance)
    sylvester_residual = tf.linalg.norm(
        factor[tf.newaxis, :, :] @ d_factor
        + d_factor @ factor[tf.newaxis, :, :]
        - d_covariance,
        axis=[-2, -1],
    )
    reconstructed = (
        tf.matmul(d_factor, factor[tf.newaxis, :, :], transpose_b=True)
        + tf.matmul(factor[tf.newaxis, :, :], d_factor, transpose_b=True)
    )
    reconstruction_residual = tf.linalg.norm(
        reconstructed - d_covariance,
        axis=[-2, -1],
    )
    max_residual = tf.maximum(sylvester_residual, reconstruction_residual)
    lyapunov_tolerance = tf.convert_to_tensor(lyapunov_tolerance, dtype=tf.float64)
    with tf.control_dependencies(
        [
            tf.debugging.assert_less_equal(
                tf.reduce_max(max_residual),
                lyapunov_tolerance,
                message=f"blocked_principal_sqrt_reconstruction: {label} derivative reconstruction failed",
            )
        ]
    ):
        d_factor = tf.identity(d_factor)
    return PrincipalSqrtFirstDerivativeDiagnostics(
        eigenvalues=eigenvalues,
        floored_eigenvalues=floored_eigenvalues,
        eigenvectors=eigenvectors,
        factor=factor,
        d_factor=d_factor,
        implemented_covariance=implemented_covariance,
        floor_count=active_floors,
        min_eigenvalue=min_eigenvalue,
        psd_projection_residual=psd_projection_residual,
        sylvester_residual=tf.reduce_max(sylvester_residual),
        reconstruction_residual=tf.reduce_max(reconstruction_residual),
    )


def eigh_solve(
    eigenvectors: tf.Tensor,
    eigenvalues: tf.Tensor,
    rhs: tf.Tensor,
) -> tf.Tensor:
    """Solve with an eigensystem whose eigenvalues are already floored."""

    eigenvectors = tf.convert_to_tensor(eigenvectors, dtype=tf.float64)
    eigenvalues = tf.convert_to_tensor(eigenvalues, dtype=tf.float64)
    rhs = tf.convert_to_tensor(rhs, dtype=tf.float64)
    if rhs.shape.rank == 1:
        projected = tf.linalg.matvec(tf.linalg.matrix_transpose(eigenvectors), rhs)
        scaled = projected / eigenvalues
        return tf.linalg.matvec(eigenvectors, scaled)
    projected = tf.linalg.matrix_transpose(eigenvectors) @ rhs
    scaled = projected / eigenvalues[:, tf.newaxis]
    return eigenvectors @ scaled


def eigh_logdet(eigenvalues: tf.Tensor) -> tf.Tensor:
    """Return log determinant from already-floored eigenvalues."""

    return tf.reduce_sum(tf.math.log(tf.convert_to_tensor(eigenvalues, dtype=tf.float64)))


def floor_count(eigenvalues: tf.Tensor, singular_floor: tf.Tensor | float) -> tf.Tensor:
    """Count eigenvalues at or below the floor branch."""

    floor = tf.cast(singular_floor, tf.float64)
    return tf.reduce_sum(tf.cast(eigenvalues <= floor, tf.int32))
