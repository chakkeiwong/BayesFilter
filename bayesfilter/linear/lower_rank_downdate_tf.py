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

    Rotation coordinates are a sequential TensorFlow loop with fixed-shape
    state; chain and parameter directions are tensor-batched.
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

    if not has_derivatives:
        d_factor = tf.zeros([batch, 0, n, n], tf.float64)
        d_vectors = tf.zeros([batch, 0, n, columns], tf.float64)
    coordinates = tf.range(n)
    inf = tf.fill([batch], tf.constant(float("inf"), tf.float64))

    def step(index, current, current_vectors, current_d_factor, current_d_vectors,
             min_margin, min_relative_margin):
        j, k = index // n, index % n
        lkk = tf.gather(tf.linalg.diag_part(current), k, axis=-1)
        old_column = tf.gather(current, k, axis=2)
        old_vector = tf.gather(current_vectors, j, axis=2)
        xk = tf.gather(old_vector, k, axis=1)
        margin = lkk * lkk - xk * xk
        tf.debugging.assert_greater(margin, tf.zeros_like(margin), message="downdate_margin_nonpositive")
        r = tf.sqrt(margin)
        c, s = r / lkk, xk / lkk
        a_new = (old_column - s[:, None] * old_vector) / c[:, None]
        u_new = c[:, None] * old_vector - s[:, None] * a_new
        new_column = tf.where(coordinates[None, :] > k, a_new,
                              tf.where(coordinates[None, :] == k, r[:, None], old_column))
        new_vector = tf.where(coordinates[None, :] > k, u_new, old_vector)
        column_mask = (coordinates == k)[None, None, :]
        vector_mask = (tf.range(columns) == j)[None, None, :]
        current = tf.where(column_mask, new_column[:, :, None], current)
        current_vectors = tf.where(vector_mask, new_vector[:, :, None], current_vectors)
        if has_derivatives:
            old_d_column = tf.gather(current_d_factor, k, axis=3)
            old_d_vector = tf.gather(current_d_vectors, j, axis=3)
            dlkk = tf.gather(old_d_column, k, axis=2)
            dxk = tf.gather(old_d_vector, k, axis=2)
            lp, xp, rp = lkk[:, None], xk[:, None], r[:, None]
            dr = (lp * dlkk - xp * dxk) / rp
            dc = (dr * lp - rp * dlkk) / (lp * lp)
            ds = (dxk * lp - xp * dlkk) / (lp * lp)
            cp, sp = c[:, None, None], s[:, None, None]
            da_new = ((old_d_column - ds[..., None] * old_vector[:, None, :]
                       - sp * old_d_vector) * cp
                      - (old_column[:, None, :] - sp * old_vector[:, None, :]) * dc[..., None]) / (cp * cp)
            du_new = (dc[..., None] * old_vector[:, None, :] + cp * old_d_vector
                      - ds[..., None] * a_new[:, None, :] - sp * da_new)
            new_d_column = tf.where(coordinates[None, None, :] > k, da_new,
                                    tf.where(coordinates[None, None, :] == k, dr[..., None], old_d_column))
            new_d_vector = tf.where(coordinates[None, None, :] > k, du_new, old_d_vector)
            current_d_factor = tf.where(column_mask[:, None], new_d_column[..., None], current_d_factor)
            current_d_vectors = tf.where(vector_mask[:, None], new_d_vector[..., None], current_d_vectors)
        return (index + 1, current, current_vectors, current_d_factor, current_d_vectors,
                tf.minimum(min_margin, margin), tf.minimum(min_relative_margin,
                    margin / tf.maximum(lkk*lkk, tf.constant(1e-300, tf.float64))))

    _, current, _, current_d_factor, _, min_margin, min_relative_margin = tf.while_loop(
        lambda index, *_: index < columns * n, step,
        (tf.constant(0), factor, vectors, d_factor, d_vectors, inf, inf),
        parallel_iterations=1, maximum_iterations=columns*n,
    )

    diagonal = tf.linalg.diag_part(current)
    tf.debugging.assert_all_finite(current, "downdated factor contains NaN or Inf")
    tf.debugging.assert_greater(diagonal, tf.zeros_like(diagonal), message="downdate diagonal nonpositive")
    diagnostics = {
        "minimum_downdate_margin": min_margin,
        "relative_downdate_margin": min_relative_margin,
        "downdate_failed": ~(tf.math.is_finite(min_margin) & (min_margin > 0.0)),
    }
    return current, current_d_factor if has_derivatives else None, diagnostics


__all__ = ["batched_lower_rank_downdate"]
