"""Analytical affine-innovation coordinates for the shared canonical executor.

The declared transition is x = anchor + B e, e ~ N(0, S). Densities and
flow determinants use innovation Lebesgue measure. This does not implement
nonlinear completion manifolds. See the structural integration plan of
2026-09-22 for the proposal and tangent derivation.
"""
from dataclasses import dataclass

import tensorflow as tf

import bayesfilter.highdim.ledh_small_matrix_tf as small_matrix

from bayesfilter.highdim.ledh_numerical_safety_tf import (
    all_finite, finite_or_zero, safe_cholesky,
)


@dataclass(frozen=True)
class AffineInnovationGeometry:
    matrix: tf.Tensor
    tangent: tf.Tensor
    left_inverse: tf.Tensor
    left_inverse_tangent: tf.Tensor
    value_valid: tf.Tensor
    force_valid: tf.Tensor

    def coordinates(self, points, d_points, anchors, d_anchors):
        residual = points - anchors
        return (tf.einsum("ij,nj->ni", self.left_inverse, residual),
                tf.einsum("ij,nj->ni", self.left_inverse_tangent, residual)
                + tf.einsum("ij,nj->ni", self.left_inverse, d_points-d_anchors))

    def complete(self, innovations, d_innovations, anchors, d_anchors):
        return (anchors + tf.einsum("ij,nj->ni", self.matrix, innovations),
                d_anchors + tf.einsum("ij,nj->ni", self.tangent, innovations)
                + tf.einsum("ij,nj->ni", self.matrix, d_innovations))

    def project_covariance(self, covariance, d_covariance):
        j, dj = self.left_inverse, self.left_inverse_tangent
        return (tf.einsum("ai,nij,bj->nab", j, covariance, j),
                tf.einsum("ai,nij,bj->nab", dj, covariance, j)
                + tf.einsum("ai,nij,bj->nab", j, d_covariance, j)
                + tf.einsum("ai,nij,bj->nab", j, covariance, dj))

    def state_covariance(self, covariance, d_covariance):
        b, db = self.matrix, self.tangent
        return (b @ covariance @ tf.transpose(b),
                db @ covariance @ tf.transpose(b)
                + b @ d_covariance @ tf.transpose(b)
                + b @ covariance @ tf.transpose(db))


def affine_innovation_geometry(matrix, tangent, *, state_dim, innovation_dim):
    """Checked full-column-rank chart; no ridge or extra stochastic dimension."""
    matrix = tf.convert_to_tensor(matrix)
    if not 0 < innovation_dim <= state_dim:
        raise ValueError("innovation dimension must be positive and no larger than state dimension")
    if matrix.shape != tf.TensorShape([state_dim, innovation_dim]):
        raise ValueError("innovation_matrix must have shape [state_dim, innovation_dim]")
    tangent = tf.zeros_like(matrix) if tangent is None else tf.ensure_shape(tangent, matrix.shape)
    gram = tf.transpose(matrix) @ matrix
    valid, factor = safe_cholesky(gram, "innovation_basis_gram")
    valid = valid & all_finite(matrix)
    force_valid = valid & all_finite(tangent)
    b = tf.where(valid, matrix, tf.eye(state_dim, innovation_dim, dtype=matrix.dtype))
    db = tf.where(force_valid, finite_or_zero(tangent), tf.zeros_like(matrix))
    j = small_matrix.cholesky_solve(factor, tf.transpose(b))
    d_gram = tf.transpose(db) @ b + tf.transpose(b) @ db
    dj = small_matrix.cholesky_solve(factor, tf.transpose(db) - d_gram @ j)
    return AffineInnovationGeometry(b, db, j, dj, valid, force_valid & all_finite(dj))
