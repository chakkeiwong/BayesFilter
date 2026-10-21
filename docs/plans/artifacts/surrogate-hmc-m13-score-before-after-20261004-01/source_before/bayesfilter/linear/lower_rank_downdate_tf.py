"""Batched lower-factor rank-one downdates for the direct SR-UKF route."""

from __future__ import annotations

import tensorflow as tf


def batched_lower_rank_downdate(
    factor: tf.Tensor,
    vectors: tf.Tensor,
    d_factor: tf.Tensor | None = None,
    d_vectors: tf.Tensor | None = None,
):
    """Apply sequential lower-factor rank-one downdates.

    Static state/observation dimensions are unrolled in Python; batch and
    parameter proposal axes remain TensorFlow tensors throughout.
    """
    factor = tf.convert_to_tensor(factor, dtype=tf.float64)
    vectors = tf.convert_to_tensor(vectors, dtype=tf.float64)
    if factor.shape.rank != 3 or vectors.shape.rank != 3:
        raise ValueError("factor must be [B,N,N] and vectors must be [B,N,M]")
    batch, n, n2 = factor.shape.as_list()
    if None in (batch, n, n2) or n != n2 or vectors.shape[0] != batch or vectors.shape[1] != n:
        raise ValueError("factor/vectors dimensions must be static and compatible")
    columns = vectors.shape[2]
    if columns is None:
        raise ValueError("vector count must be static")
    tf.debugging.assert_all_finite(factor, "factor contains NaN or Inf")
    tf.debugging.assert_all_finite(vectors, "downdate vectors contain NaN or Inf")

    has_derivatives = d_factor is not None or d_vectors is not None
    if has_derivatives:
        if d_factor is None or d_vectors is None:
            raise ValueError("d_factor and d_vectors must be supplied together")
        d_factor = tf.convert_to_tensor(d_factor, dtype=tf.float64)
        d_vectors = tf.convert_to_tensor(d_vectors, dtype=tf.float64)
        if d_factor.shape.rank != 4 or d_vectors.shape.rank != 4:
            raise ValueError("derivatives must have rank four [B,P,...]")
        if d_factor.shape[0] != batch or d_factor.shape[2:] != factor.shape[1:]:
            raise ValueError("d_factor must be [B,P,N,N]")
        if d_vectors.shape[0] != batch or d_vectors.shape[2:] != vectors.shape[1:]:
            raise ValueError("d_vectors must be [B,P,N,M]")
        if d_factor.shape[1] != d_vectors.shape[1]:
            raise ValueError("derivative parameter dimensions must match")
        tf.debugging.assert_all_finite(d_factor, "d_factor contains NaN or Inf")
        tf.debugging.assert_all_finite(d_vectors, "d_vectors contain NaN or Inf")

    current = factor
    current_vectors = vectors
    current_d_factor = d_factor
    current_d_vectors = d_vectors
    margins = []
    relative_margins = []

    for j in range(columns):
        for k in range(n):
            lkk = current[:, k, k]
            xk = current_vectors[:, k, j]
            margin = lkk * lkk - xk * xk
            tf.debugging.assert_all_finite(margin, "downdate margin contains NaN or Inf")
            tf.debugging.assert_greater(margin, tf.zeros_like(margin), message="downdate_margin_nonpositive")
            margins.append(margin)
            relative_margins.append(
                margin / tf.maximum(lkk * lkk, tf.constant(1.0e-300, tf.float64))
            )
            r = tf.sqrt(margin)
            c = r / lkk
            s = xk / lkk
            old_column = current[:, :, k]
            old_vector = current_vectors[:, :, j]
            a_old = old_column[:, k + 1 :]
            u_old = old_vector[:, k + 1 :]
            a_new = (a_old - s[:, None] * u_old) / c[:, None]
            u_new = c[:, None] * u_old - s[:, None] * a_new
            new_column = tf.concat([old_column[:, :k], r[:, None], a_new], axis=1)
            new_vector = tf.concat([old_vector[:, : k + 1], u_new], axis=1)
            current = tf.concat(
                [current[:, :, :k], new_column[:, :, None], current[:, :, k + 1 :]],
                axis=2,
            )
            current_vectors = tf.concat(
                [current_vectors[:, :, :j], new_vector[:, :, None], current_vectors[:, :, j + 1 :]],
                axis=2,
            )

            if has_derivatives:
                dlkk = current_d_factor[:, :, k, k]
                dxk = current_d_vectors[:, :, k, j]
                l_p = lkk[:, None]
                x_p = xk[:, None]
                r_p = r[:, None]
                dr = (l_p * dlkk - x_p * dxk) / r_p
                dc = (dr * l_p - r_p * dlkk) / (l_p * l_p)
                ds = (dxk * l_p - x_p * dlkk) / (l_p * l_p)
                old_d_column = current_d_factor[:, :, :, k]
                old_d_vector = current_d_vectors[:, :, :, j]
                da_old = old_d_column[:, :, k + 1 :]
                du_old = old_d_vector[:, :, k + 1 :]
                a_old_p = old_column[:, k + 1 :][:, None, :]
                u_old_p = old_vector[:, k + 1 :][:, None, :]
                c_p = c[:, None, None]
                s_p = s[:, None, None]
                da_new = (
                    (da_old - ds[:, :, None] * u_old_p - s_p * du_old) * c_p
                    - (a_old_p - s_p * u_old_p) * dc[:, :, None]
                ) / (c_p * c_p)
                du_new = (
                    dc[:, :, None] * u_old_p
                    + c_p * du_old
                    - ds[:, :, None] * a_new[:, None, :]
                    - s_p * da_new
                )
                new_d_column = tf.concat(
                    [old_d_column[:, :, :k], dr[:, :, None], da_new], axis=2
                )
                new_d_vector = tf.concat(
                    [old_d_vector[:, :, : k + 1], du_new], axis=2
                )
                current_d_factor = tf.concat(
                    [
                        current_d_factor[:, :, :, :k],
                        new_d_column[:, :, :, None],
                        current_d_factor[:, :, :, k + 1 :],
                    ],
                    axis=3,
                )
                current_d_vectors = tf.concat(
                    [
                        current_d_vectors[:, :, :, :j],
                        new_d_vector[:, :, :, None],
                        current_d_vectors[:, :, :, j + 1 :],
                    ],
                    axis=3,
                )

    diagonal = tf.linalg.diag_part(current)
    tf.debugging.assert_all_finite(current, "downdated factor contains NaN or Inf")
    tf.debugging.assert_greater(diagonal, tf.zeros_like(diagonal), message="downdate diagonal nonpositive")
    min_margin = tf.reduce_min(tf.stack(margins, axis=1), axis=1)
    min_relative_margin = tf.reduce_min(tf.stack(relative_margins, axis=1), axis=1)
    diagnostics = {
        "minimum_downdate_margin": min_margin,
        "relative_downdate_margin": min_relative_margin,
        "downdate_failed": tf.zeros([batch], dtype=tf.bool),
    }
    return current, current_d_factor if has_derivatives else None, diagnostics


__all__ = ["batched_lower_rank_downdate"]
