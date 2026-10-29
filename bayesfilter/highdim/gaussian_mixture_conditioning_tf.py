"""Exact Gaussian-product mixture moments and analytical directional tangents.

Means/covariances have shapes [batch, dimension] and [batch, dimension,
dimension]; mixture coefficients have no batch axis and are frozen. This is
an optional proposal construction, not an exact nonlinear filtering update.
"""

import math

import tensorflow as tf


def _symmetric(matrix):
    return 0.5 * (matrix + tf.linalg.matrix_transpose(matrix))


def _factor(matrix):
    dimension = tf.shape(matrix)[-1]
    identity = tf.eye(dimension, batch_shape=tf.shape(matrix)[:-2], dtype=matrix.dtype)
    epsilon = tf.constant(2.0 ** (-52 if matrix.dtype == tf.float64 else -23), matrix.dtype)
    finite = tf.reduce_all(tf.math.is_finite(matrix), axis=[-2, -1])
    scale = tf.reduce_max(tf.abs(matrix), axis=[-2, -1])
    asymmetry = tf.reduce_max(tf.abs(matrix - tf.linalg.matrix_transpose(matrix)), axis=[-2, -1])
    symmetric_valid = asymmetry <= 32.0 * epsilon * tf.cast(dimension, matrix.dtype) * scale
    safe = tf.where((finite & symmetric_valid)[..., None, None], _symmetric(matrix), identity)
    eigenvalues = tf.linalg.eigvalsh(safe)
    margin = tf.math.divide_no_nan(eigenvalues[..., 0], tf.reduce_max(tf.abs(eigenvalues), axis=-1))
    valid = finite & symmetric_valid & (margin > epsilon * tf.cast(dimension, matrix.dtype))
    factor = tf.linalg.cholesky(tf.where(valid[..., None, None], safe, identity))
    return factor, tf.reduce_all(valid), tf.reduce_min(margin)


def gaussian_mixture_product_moments(mean, covariance, d_mean, d_covariance,
                                    logits, centers, component_covariances):
    """Return mean, covariance, their tangents, validity and minimum SPD margin.

    Invalid matrices use identity solely to keep a rejected computation finite.
    No accepted covariance is ridged, clipped or replaced. Call inside a traced
    numerical consumer with an explicit input signature.
    """
    dimension = tf.shape(mean)[-1]
    identity = tf.eye(dimension, dtype=mean.dtype)
    factor, valid, margin = _factor(covariance)
    component_factor, component_valid, component_margin = _factor(component_covariances)
    precision = tf.linalg.cholesky_solve(factor, tf.broadcast_to(identity, tf.shape(covariance)))
    component_precision = tf.linalg.cholesky_solve(
        component_factor, tf.broadcast_to(identity, tf.shape(component_covariances)))
    d_precision = -tf.matmul(tf.matmul(precision, d_covariance), precision)
    product_precision = precision[:, None] + component_precision[None]
    product_factor, product_valid, product_margin = _factor(product_precision)
    product_covariance = tf.linalg.cholesky_solve(
        product_factor, tf.broadcast_to(identity, tf.shape(product_precision)))
    d_product_covariance = -tf.matmul(
        tf.matmul(product_covariance, d_precision[:, None]), product_covariance)
    natural = (tf.linalg.matvec(precision, mean)[:, None]
               + tf.linalg.matvec(component_precision, centers)[None])
    d_natural = (tf.linalg.matvec(d_precision, mean)
                 + tf.linalg.matvec(precision, d_mean))[:, None]
    product_mean = tf.linalg.matvec(product_covariance, natural)
    d_product_mean = (tf.linalg.matvec(d_product_covariance, natural)
                      + tf.linalg.matvec(product_covariance, d_natural))
    sum_covariance = covariance[:, None] + component_covariances[None]
    sum_factor, sum_valid, sum_margin = _factor(sum_covariance)
    residual = mean[:, None] - centers[None]
    solved_residual = tf.linalg.cholesky_solve(sum_factor, residual[..., None])[..., 0]
    log_density = -0.5 * (
        tf.reduce_sum(residual * solved_residual, axis=-1)
        + 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(sum_factor)), axis=-1)
        + tf.cast(dimension, mean.dtype) * tf.constant(math.log(2.0 * math.pi), mean.dtype))
    responsibilities = tf.nn.softmax(logits[None] + log_density, axis=-1)
    sum_precision = tf.linalg.cholesky_solve(
        sum_factor, tf.broadcast_to(identity, tf.shape(sum_covariance)))
    d_log_density = (-tf.reduce_sum(solved_residual * d_mean[:, None], axis=-1)
                     + 0.5 * (tf.reduce_sum(solved_residual * tf.linalg.matvec(
                         d_covariance[:, None], solved_residual), axis=-1)
                         - tf.einsum("bkij,bji->bk", sum_precision, d_covariance)))
    d_responsibilities = responsibilities * (
        d_log_density - tf.reduce_sum(responsibilities * d_log_density, axis=-1, keepdims=True))
    output_mean = tf.einsum("bk,bki->bi", responsibilities, product_mean)
    d_output_mean = (tf.einsum("bk,bki->bi", d_responsibilities, product_mean)
                     + tf.einsum("bk,bki->bi", responsibilities, d_product_mean))
    centered = product_mean - output_mean[:, None]
    d_centered = d_product_mean - d_output_mean[:, None]
    second_moments = product_covariance + centered[..., :, None] * centered[..., None, :]
    d_second_moments = (d_product_covariance
                        + d_centered[..., :, None] * centered[..., None, :]
                        + centered[..., :, None] * d_centered[..., None, :])
    output_covariance = _symmetric(tf.einsum("bk,bkij->bij", responsibilities, second_moments))
    d_output_covariance = _symmetric(
        tf.einsum("bk,bkij->bij", d_responsibilities, second_moments)
        + tf.einsum("bk,bkij->bij", responsibilities, d_second_moments))
    _, output_valid, output_margin = _factor(output_covariance)
    valid &= component_valid & product_valid & sum_valid & output_valid
    for value in (mean, d_mean, d_covariance, logits, centers, output_mean,
                  d_output_mean, d_output_covariance):
        valid &= tf.reduce_all(tf.math.is_finite(value))
    margin = tf.reduce_min(tf.stack([margin, component_margin, product_margin, sum_margin, output_margin]))
    return output_mean, output_covariance, d_output_mean, d_output_covariance, valid, margin
