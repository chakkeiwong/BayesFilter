"""TensorFlow eigen/SVD helpers for PSD covariance solves."""

from __future__ import annotations

import tensorflow as tf


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
