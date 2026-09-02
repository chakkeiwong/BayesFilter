"""Finite higher-moment correction for the opt-in Contract E candidate.

The map in this module is deliberately bounded and finite.  It is not a claim
that a nonconvex exact moment projection exists for every cloud.  The returned
JVP differentiates the complete executed map without TensorFlow autodiff.
"""

from __future__ import annotations

import tensorflow as tf

from bayesfilter.highdim.genut_shape_lm_tf import (
    necessary_marginal_feasibility,
    scaled_lm_coefficients_jvp,
    smooth_rms_cap_jvp,
)
from bayesfilter.highdim.ledh_contract_e_reset_tf import (
    _cholesky_jvp as _contract_e_cholesky_jvp,
)


def _sym(value: tf.Tensor) -> tf.Tensor:
    return 0.5 * (value + tf.linalg.matrix_transpose(value))


def _sym_tangent(value: tf.Tensor) -> tf.Tensor:
    return 0.5 * (value + tf.transpose(value, [1, 0, 2]))


def _right_solve(lower: tf.Tensor, rows: tf.Tensor) -> tf.Tensor:
    return tf.linalg.matrix_transpose(
        tf.linalg.triangular_solve(lower, tf.linalg.matrix_transpose(rows))
    )


def _right_solve_jvp(
    lower: tf.Tensor,
    lower_tangent: tf.Tensor,
    rows: tf.Tensor,
    rows_tangent: tf.Tensor,
) -> tf.Tensor:
    solved = _right_solve(lower, rows)
    rhs_tangent = rows_tangent - tf.einsum(
        "ni,jip->njp", solved, lower_tangent
    )
    batched_lower = tf.broadcast_to(
        lower[None, :, :],
        [tf.shape(rows_tangent)[-1], tf.shape(lower)[0], tf.shape(lower)[1]],
    )
    batched_rhs = tf.transpose(rhs_tangent, [2, 0, 1])
    solved_tangent = _right_solve(batched_lower, batched_rhs)
    return tf.transpose(solved_tangent, [1, 2, 0])


# Relative PSD floor for empirical-covariance factorizations (2026-08-26).
#
# An empirical covariance of a weighted cloud is symmetric PSD in exact
# arithmetic, but under annealed weight concentration its smallest
# eigenvalue lands at roundoff (measured: -6.4e-16 against a largest
# eigenvalue of 2.33 on the Austria production score lane), which makes
# `tf.linalg.cholesky` emit NaNs and the downstream spectral diagnostic
# raise an opaque GPU eigendecomposition error.
#
# Two separable corrections, deliberately classified:
#   symmetrization  - the matrix is mathematically symmetric, so
#                     averaging with its transpose removes accumulation
#                     asymmetry without changing the represented object;
#   relative ridge  - delta * tr(C)/d * I, the scale- and
#                     dimension-aware form derived for the reset ridge
#                     (registry gap A5). This DOES shift the computed
#                     factor and is therefore a numerics-altering
#                     protection; its non-harm evaluation is recorded in
#                     the campaign ledger. An absolute floor was rejected
#                     here: it silently expires as cloud scale grows.
#
# The ridge tangent (delta * tr(dC)/d * I) is returned alongside so the
# hand-derived JVP differentiates the matrix that was actually factored.
RELATIVE_PSD_FLOOR = 1.0e-12


def _relative_psd_covariance(
    covariance: tf.Tensor,
    covariance_tangent: tf.Tensor,
    *,
    relative_floor: float = RELATIVE_PSD_FLOOR,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Symmetrize and relatively-ridge a covariance and its tangent."""

    symmetric = _sym(covariance)
    dtype = symmetric.dtype
    dim = tf.cast(tf.shape(symmetric)[-1], dtype)
    eye = tf.eye(tf.shape(symmetric)[-1], dtype=dtype)
    delta = tf.cast(relative_floor, dtype)
    scale = tf.linalg.trace(symmetric) / dim
    ridged = symmetric + delta * scale * eye
    symmetric_tangent = _sym_tangent(covariance_tangent)
    scale_tangent = (
        tf.einsum("iip->p", symmetric_tangent) / dim
    )
    ridged_tangent = symmetric_tangent + (
        delta * eye[:, :, None] * scale_tangent[None, None, :]
    )
    return ridged, ridged_tangent


def _cholesky_jvp(chol: tf.Tensor, matrix_tangent: tf.Tensor) -> tf.Tensor:
    parameter_count = tf.shape(matrix_tangent)[-1]
    batched_chol = tf.broadcast_to(
        chol[None, :, :],
        [parameter_count, tf.shape(chol)[0], tf.shape(chol)[1]],
    )
    batched_tangent = tf.transpose(matrix_tangent, [2, 0, 1])
    return tf.transpose(
        _contract_e_cholesky_jvp(batched_chol, batched_tangent), [1, 2, 0]
    )


def _weighted_moments_jvp(
    points: tf.Tensor,
    weights: tf.Tensor,
    points_tangent: tf.Tensor,
    weights_tangent: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    mean = tf.reduce_sum(weights[:, None] * points, axis=0)
    mean_tangent = tf.reduce_sum(
        weights_tangent[:, None, :] * points[:, :, None]
        + weights[:, None, None] * points_tangent,
        axis=0,
    )
    centered = points - mean[None, :]
    centered_tangent = points_tangent - mean_tangent[None, :, :]
    covariance = tf.einsum("n,ni,nj->ij", weights, centered, centered)
    covariance_tangent = tf.einsum(
        "np,ni,nj->ijp", weights_tangent, centered, centered
    )
    covariance_tangent += tf.einsum(
        "n,nip,nj->ijp", weights, centered_tangent, centered
    )
    covariance_tangent += tf.einsum(
        "n,ni,njp->ijp", weights, centered, centered_tangent
    )
    return mean, _sym(covariance), mean_tangent, _sym_tangent(covariance_tangent)


def _uniform_moments_jvp(
    points: tf.Tensor, points_tangent: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    count = tf.cast(tf.shape(points)[0], points.dtype)
    mean = tf.reduce_mean(points, axis=0)
    mean_tangent = tf.reduce_mean(points_tangent, axis=0)
    centered = points - mean[None, :]
    centered_tangent = points_tangent - mean_tangent[None, :, :]
    uniform = tf.ones_like(points[:, 0]) / count
    covariance = tf.einsum("n,ni,nj->ij", uniform, centered, centered)
    covariance_tangent = tf.einsum(
        "n,nip,nj->ijp", uniform, centered_tangent, centered
    ) + tf.einsum(
        "n,ni,njp->ijp", uniform, centered, centered_tangent
    )
    return mean, _sym(covariance), mean_tangent, _sym_tangent(covariance_tangent)


def _weighted_diag_moments_jvp(
    source: tf.Tensor,
    weights: tf.Tensor,
    source_tangent: tf.Tensor,
    weights_tangent: tf.Tensor,
    mean: tf.Tensor,
    mean_tangent: tf.Tensor,
    chol: tf.Tensor,
    chol_tangent: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    centered = source - mean[None, :]
    centered_tangent = source_tangent - mean_tangent[None, :, :]
    standardized = _right_solve(chol, centered)
    standardized_tangent = _right_solve_jvp(
        chol, chol_tangent, centered, centered_tangent
    )
    skew = tf.reduce_sum(weights[:, None] * tf.pow(standardized, 3.0), axis=0)
    kurt = tf.reduce_sum(weights[:, None] * tf.pow(standardized, 4.0), axis=0)
    skew_tangent = tf.reduce_sum(
        weights_tangent[:, None, :] * tf.pow(standardized[:, :, None], 3.0)
        + weights[:, None, None]
        * 3.0
        * tf.pow(standardized[:, :, None], 2.0)
        * standardized_tangent,
        axis=0,
    )
    kurt_tangent = tf.reduce_sum(
        weights_tangent[:, None, :] * tf.pow(standardized[:, :, None], 4.0)
        + weights[:, None, None]
        * 4.0
        * tf.pow(standardized[:, :, None], 3.0)
        * standardized_tangent,
        axis=0,
    )
    return skew, kurt, skew_tangent, kurt_tangent


def _weighted_pair_moments_jvp(
    standardized: tf.Tensor,
    weights: tf.Tensor,
    standardized_tangent: tf.Tensor,
    weights_tangent: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Return ordered E[z_i^2 z_j] and symmetric E[z_i^2 z_j^2]."""

    squared = tf.square(standardized)
    squared_tangent = 2.0 * standardized[:, :, None] * standardized_tangent
    co_skew = tf.einsum("n,ni,nj->ij", weights, squared, standardized)
    co_skew_tangent = tf.einsum(
        "np,ni,nj->ijp", weights_tangent, squared, standardized
    )
    co_skew_tangent += tf.einsum(
        "n,nip,nj->ijp", weights, squared_tangent, standardized
    )
    co_skew_tangent += tf.einsum(
        "n,ni,njp->ijp", weights, squared, standardized_tangent
    )
    co_kurtosis = tf.einsum("n,ni,nj->ij", weights, squared, squared)
    co_kurtosis_tangent = tf.einsum(
        "np,ni,nj->ijp", weights_tangent, squared, squared
    )
    co_kurtosis_tangent += tf.einsum(
        "n,nip,nj->ijp", weights, squared_tangent, squared
    )
    co_kurtosis_tangent += tf.einsum(
        "n,ni,njp->ijp", weights, squared, squared_tangent
    )
    off_diagonal = 1.0 - tf.eye(tf.shape(standardized)[1], dtype=standardized.dtype)
    return (
        co_skew * off_diagonal,
        co_kurtosis * off_diagonal,
        co_skew_tangent * off_diagonal[:, :, None],
        co_kurtosis_tangent * off_diagonal[:, :, None],
    )


def weighted_shape_targets_jvp(
    points: tf.Tensor,
    weights: tf.Tensor,
    points_tangent: tf.Tensor,
    weights_tangent: tf.Tensor,
) -> dict[str, tf.Tensor]:
    """Return standardized diagonal and full pairwise moments with total JVP."""

    points = tf.convert_to_tensor(points)
    weights = tf.convert_to_tensor(weights, points.dtype)
    points_tangent = tf.convert_to_tensor(points_tangent, points.dtype)
    weights_tangent = tf.convert_to_tensor(weights_tangent, points.dtype)
    mean, covariance, mean_tangent, covariance_tangent = _weighted_moments_jvp(
        points, weights, points_tangent, weights_tangent
    )
    covariance, covariance_tangent = _relative_psd_covariance(
        covariance, covariance_tangent
    )
    chol = tf.linalg.cholesky(covariance)
    chol_tangent = _cholesky_jvp(chol, covariance_tangent)
    skew, kurtosis, skew_tangent, kurtosis_tangent = _weighted_diag_moments_jvp(
        points,
        weights,
        points_tangent,
        weights_tangent,
        mean,
        mean_tangent,
        chol,
        chol_tangent,
    )
    centered = points - mean[None, :]
    centered_tangent = points_tangent - mean_tangent[None, :, :]
    standardized = _right_solve(chol, centered)
    standardized_tangent = _right_solve_jvp(
        chol, chol_tangent, centered, centered_tangent
    )
    co_skew, co_kurtosis, co_skew_tangent, co_kurtosis_tangent = (
        _weighted_pair_moments_jvp(
            standardized, weights, standardized_tangent, weights_tangent
        )
    )
    return {
        "skew": skew,
        "kurtosis": kurtosis,
        "skew_tangent": skew_tangent,
        "kurtosis_tangent": kurtosis_tangent,
        "pairwise_co_skew": co_skew,
        "pairwise_co_kurtosis": co_kurtosis,
        "pairwise_co_skew_tangent": co_skew_tangent,
        "pairwise_co_kurtosis_tangent": co_kurtosis_tangent,
    }


def affine_restore_cloud_jvp(
    source: tf.Tensor,
    weights: tf.Tensor,
    source_tangent: tf.Tensor,
    weights_tangent: tf.Tensor,
    points: tf.Tensor,
    points_tangent: tf.Tensor,
) -> dict[str, tf.Tensor]:
    """Restore a uniform cloud to the weighted source mean/covariance."""

    target_mean, target_cov, target_mean_tangent, target_cov_tangent = (
        _weighted_moments_jvp(
            source, weights, source_tangent, weights_tangent
        )
    )
    current_mean, current_cov, current_mean_tangent, current_cov_tangent = (
        _uniform_moments_jvp(points, points_tangent)
    )
    # Class-B guard (2026-08-27): annealed concentration can collapse the
    # covariances in the affine restore; apply relative ridge to both.
    target_cov_safe, target_cov_tangent_safe = _relative_psd_covariance(
        target_cov, target_cov_tangent, relative_floor=RELATIVE_PSD_FLOOR
    )
    current_cov_safe, current_cov_tangent_safe = _relative_psd_covariance(
        current_cov, current_cov_tangent, relative_floor=RELATIVE_PSD_FLOOR
    )
    target_chol = tf.linalg.cholesky(target_cov_safe)
    current_chol = tf.linalg.cholesky(current_cov_safe)
    target_chol_tangent = _cholesky_jvp(target_chol, target_cov_tangent_safe)
    current_chol_tangent = _cholesky_jvp(current_chol, current_cov_tangent_safe)
    centered = points - current_mean[None, :]
    centered_tangent = points_tangent - current_mean_tangent[None, :, :]
    standardized = _right_solve(current_chol, centered)
    standardized_tangent = _right_solve_jvp(
        current_chol,
        current_chol_tangent,
        centered,
        centered_tangent,
    )
    output = target_mean[None, :] + tf.linalg.matmul(
        standardized, target_chol, transpose_b=True
    )
    output_tangent = (
        target_mean_tangent[None, :, :]
        + tf.einsum("nip,ji->njp", standardized_tangent, target_chol)
        + tf.einsum("ni,jip->njp", standardized, target_chol_tangent)
    )
    observed_mean, observed_cov, _, _ = _uniform_moments_jvp(
        output, output_tangent
    )
    maximum_mean_residual = tf.reduce_max(tf.abs(observed_mean - target_mean))
    maximum_covariance_residual = tf.reduce_max(
        tf.abs(observed_cov - target_cov)
    )
    mean_scale = tf.maximum(
        tf.reduce_max(tf.abs(target_mean)), tf.cast(1.0, source.dtype)
    )
    covariance_scale = tf.maximum(
        tf.reduce_max(tf.abs(target_cov)), tf.cast(1.0, source.dtype)
    )
    return {
        "particles": output,
        "particles_tangent": output_tangent,
        "maximum_mean_residual": maximum_mean_residual,
        "maximum_covariance_residual": maximum_covariance_residual,
        "maximum_normalized_mean_residual": maximum_mean_residual / mean_scale,
        "maximum_normalized_covariance_residual": (
            maximum_covariance_residual / covariance_scale
        ),
        "valid": tf.reduce_all(tf.math.is_finite(output))
        & tf.reduce_all(tf.math.is_finite(output_tangent)),
    }
def _uniform_pair_moments_jvp(
    standardized: tf.Tensor,
    standardized_tangent: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    count = tf.cast(tf.shape(standardized)[0], standardized.dtype)
    weights = tf.ones([tf.shape(standardized)[0]], standardized.dtype) / count
    weights_tangent = tf.zeros(
        [tf.shape(standardized)[0], tf.shape(standardized_tangent)[-1]],
        standardized.dtype,
    )
    return _weighted_pair_moments_jvp(
        standardized, weights, standardized_tangent, weights_tangent
    )


def _standardize_uniform_jvp(
    points: tf.Tensor,
    points_tangent: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    mean, covariance, mean_tangent, covariance_tangent = _uniform_moments_jvp(
        points, points_tangent
    )
    covariance, covariance_tangent = _relative_psd_covariance(
        covariance, covariance_tangent
    )
    chol = tf.linalg.cholesky(covariance)
    chol_tangent = _cholesky_jvp(chol, covariance_tangent)
    centered = points - mean[None, :]
    centered_tangent = points_tangent - mean_tangent[None, :, :]
    return (
        _right_solve(chol, centered),
        _right_solve_jvp(
            chol, chol_tangent, centered, centered_tangent
        ),
    )


def _projected_moment_tensors_jvp(
    projected: tf.Tensor,
    weights: tf.Tensor,
    projected_tangent: tf.Tensor,
    weights_tangent: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Return complete projected third/fourth moments and total JVPs."""

    third = tf.einsum("n,na,nb,nc->abc", weights, projected, projected, projected)
    third_tangent = tf.einsum(
        "np,na,nb,nc->abcp", weights_tangent, projected, projected, projected
    )
    third_tangent += tf.einsum(
        "n,nap,nb,nc->abcp", weights, projected_tangent, projected, projected
    )
    third_tangent += tf.einsum(
        "n,na,nbp,nc->abcp", weights, projected, projected_tangent, projected
    )
    third_tangent += tf.einsum(
        "n,na,nb,ncp->abcp", weights, projected, projected, projected_tangent
    )
    fourth = tf.einsum(
        "n,na,nb,nc,ne->abce", weights, projected, projected, projected, projected
    )
    fourth_tangent = tf.einsum(
        "np,na,nb,nc,ne->abcep",
        weights_tangent,
        projected,
        projected,
        projected,
        projected,
    )
    fourth_tangent += tf.einsum(
        "n,nap,nb,nc,ne->abcep",
        weights,
        projected_tangent,
        projected,
        projected,
        projected,
    )
    fourth_tangent += tf.einsum(
        "n,na,nbp,nc,ne->abcep",
        weights,
        projected,
        projected_tangent,
        projected,
        projected,
    )
    fourth_tangent += tf.einsum(
        "n,na,nb,ncp,ne->abcep",
        weights,
        projected,
        projected,
        projected_tangent,
        projected,
    )
    fourth_tangent += tf.einsum(
        "n,na,nb,nc,nep->abcep",
        weights,
        projected,
        projected,
        projected,
        projected_tangent,
    )
    return third, fourth, third_tangent, fourth_tangent


def _projected_residuals_jvp(
    source_standardized: tf.Tensor,
    weights: tf.Tensor,
    source_standardized_tangent: tf.Tensor,
    weights_tangent: tf.Tensor,
    current_standardized: tf.Tensor,
    current_standardized_tangent: tf.Tensor,
    basis: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    source_projected = tf.linalg.matmul(source_standardized, basis)
    source_projected_tangent = tf.einsum(
        "ndp,dr->nrp", source_standardized_tangent, basis
    )
    current_projected = tf.linalg.matmul(current_standardized, basis)
    current_projected_tangent = tf.einsum(
        "ndp,dr->nrp", current_standardized_tangent, basis
    )
    target3, target4, target3_tangent, target4_tangent = (
        _projected_moment_tensors_jvp(
            source_projected,
            weights,
            source_projected_tangent,
            weights_tangent,
        )
    )
    count = tf.cast(tf.shape(current_standardized)[0], current_standardized.dtype)
    uniform = tf.ones([tf.shape(current_standardized)[0]], current_standardized.dtype) / count
    uniform_tangent = tf.zeros(
        [tf.shape(current_standardized)[0], tf.shape(current_standardized_tangent)[-1]],
        current_standardized.dtype,
    )
    current3, current4, current3_tangent, current4_tangent = (
        _projected_moment_tensors_jvp(
            current_projected,
            uniform,
            current_projected_tangent,
            uniform_tangent,
        )
    )
    return (
        target3 - current3,
        target4 - current4,
        target3_tangent - current3_tangent,
        target4_tangent - current4_tangent,
        current_projected,
        current_projected_tangent,
    )


def _projected_cumulant_iteration_jvp(
    source_standardized: tf.Tensor,
    weights: tf.Tensor,
    source_standardized_tangent: tf.Tensor,
    weights_tangent: tf.Tensor,
    current_standardized: tf.Tensor,
    current_standardized_tangent: tf.Tensor,
    basis: tf.Tensor,
    *,
    strength: float,
    floor: float,
) -> tuple[tf.Tensor, tf.Tensor]:
    residual3, residual4, residual3_tangent, residual4_tangent, projected, projected_tangent = (
        _projected_residuals_jvp(
            source_standardized,
            weights,
            source_standardized_tangent,
            weights_tangent,
            current_standardized,
            current_standardized_tangent,
            basis,
        )
    )
    direction3 = 3.0 * tf.einsum("abc,nb,nc->na", residual3, projected, projected)
    direction3_tangent = 3.0 * (
        tf.einsum("abcp,nb,nc->nap", residual3_tangent, projected, projected)
        + tf.einsum("abc,nbp,nc->nap", residual3, projected_tangent, projected)
        + tf.einsum("abc,nb,ncp->nap", residual3, projected, projected_tangent)
    )
    direction4 = 4.0 * tf.einsum(
        "abce,nb,nc,ne->na", residual4, projected, projected, projected
    )
    direction4_tangent = 4.0 * (
        tf.einsum(
            "abcep,nb,nc,ne->nap", residual4_tangent, projected, projected, projected
        )
        + tf.einsum(
            "abce,nbp,nc,ne->nap", residual4, projected_tangent, projected, projected
        )
        + tf.einsum(
            "abce,nb,ncp,ne->nap", residual4, projected, projected_tangent, projected
        )
        + tf.einsum(
            "abce,nb,nc,nep->nap", residual4, projected, projected, projected_tangent
        )
    )
    direction = tf.linalg.matmul(direction3 + direction4, basis, transpose_b=True)
    direction_tangent = tf.einsum(
        "nrp,dr->ndp", direction3_tangent + direction4_tangent, basis
    )
    direction_mean = tf.reduce_mean(direction, axis=0)
    direction_mean_tangent = tf.reduce_mean(direction_tangent, axis=0)
    centered_direction = direction - direction_mean[None, :]
    centered_direction_tangent = direction_tangent - direction_mean_tangent[None, :, :]
    cross = tf.reduce_mean(
        current_standardized[:, :, None] * centered_direction[:, None, :], axis=0
    )
    cross_tangent = tf.reduce_mean(
        current_standardized_tangent[:, :, None, :]
        * centered_direction[:, None, :, None]
        + current_standardized[:, :, None, None]
        * centered_direction_tangent[:, None, :, :],
        axis=0,
    )
    symmetric_cross = _sym(cross)
    symmetric_cross_tangent = _sym_tangent(cross_tangent)
    affine_projected = centered_direction - tf.linalg.matmul(
        current_standardized, symmetric_cross
    )
    affine_projected_tangent = (
        centered_direction_tangent
        - tf.einsum(
            "ndp,de->nep", current_standardized_tangent, symmetric_cross
        )
        - tf.einsum(
            "nd,dep->nep", current_standardized, symmetric_cross_tangent
        )
    )
    rms = tf.sqrt(
        tf.reduce_mean(tf.square(affine_projected))
        + tf.cast(floor, current_standardized.dtype)
    )
    rms_tangent = tf.reduce_mean(
        affine_projected[:, :, None] * affine_projected_tangent, axis=[0, 1]
    ) / rms
    normalized = affine_projected / rms
    normalized_tangent = (
        affine_projected_tangent / rms
        - affine_projected[:, :, None]
        * rms_tangent[None, None, :]
        / tf.square(rms)
    )
    corrected = current_standardized + tf.cast(strength, current_standardized.dtype) * normalized
    corrected_tangent = current_standardized_tangent + tf.cast(
        strength, current_standardized.dtype
    ) * normalized_tangent
    return _standardize_uniform_jvp(corrected, corrected_tangent)


def _pairwise_shape_iteration_jvp(
    standardized: tf.Tensor,
    standardized_tangent: tf.Tensor,
    target_co_skew: tf.Tensor,
    target_co_kurtosis: tf.Tensor,
    target_co_skew_tangent: tf.Tensor,
    target_co_kurtosis_tangent: tf.Tensor,
    target_co_skew_mask: tf.Tensor,
    target_co_kurtosis_mask: tf.Tensor,
    *,
    strength: float,
    floor: float,
    particle_rms_cap: float,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Take one bounded residual-gradient step over all coordinate pairs."""

    co_skew, co_kurtosis, co_skew_tangent, co_kurtosis_tangent = (
        _uniform_pair_moments_jvp(standardized, standardized_tangent)
    )
    residual3 = target_co_skew_mask * (target_co_skew - co_skew)
    residual4 = target_co_kurtosis_mask * (target_co_kurtosis - co_kurtosis)
    residual3_tangent = target_co_skew_mask[:, :, None] * (
        target_co_skew_tangent - co_skew_tangent
    )
    residual4_tangent = target_co_kurtosis_mask[:, :, None] * (
        target_co_kurtosis_tangent - co_kurtosis_tangent
    )

    squared = tf.square(standardized)
    squared_tangent = 2.0 * standardized[:, :, None] * standardized_tangent
    row3 = tf.linalg.matmul(standardized, residual3, transpose_b=True)
    row3_tangent = (
        tf.einsum("nip,ji->njp", standardized_tangent, residual3)
        + tf.einsum("ni,jip->njp", standardized, residual3_tangent)
    )
    column3 = tf.linalg.matmul(squared, residual3)
    column3_tangent = (
        tf.einsum("nip,ij->njp", squared_tangent, residual3)
        + tf.einsum("ni,ijp->njp", squared, residual3_tangent)
    )
    row4 = tf.linalg.matmul(squared, residual4, transpose_b=True)
    row4_tangent = (
        tf.einsum("nip,ji->njp", squared_tangent, residual4)
        + tf.einsum("ni,jip->njp", squared, residual4_tangent)
    )
    dimension_scale = tf.cast(
        tf.maximum(tf.shape(standardized)[1] - 1, 1), standardized.dtype
    )
    direction = (
        2.0 * standardized * row3 + column3 + 2.0 * standardized * row4
    ) / dimension_scale
    direction_tangent = (
        2.0
        * (
            standardized_tangent * row3[:, :, None]
            + standardized[:, :, None] * row3_tangent
        )
        + column3_tangent
        + 2.0
        * (
            standardized_tangent * row4[:, :, None]
            + standardized[:, :, None] * row4_tangent
        )
    ) / dimension_scale

    direction_mean = tf.reduce_mean(direction, axis=0)
    direction_mean_tangent = tf.reduce_mean(direction_tangent, axis=0)
    centered_direction = direction - direction_mean[None, :]
    centered_direction_tangent = (
        direction_tangent - direction_mean_tangent[None, :, :]
    )
    cross = tf.reduce_mean(
        standardized[:, :, None] * centered_direction[:, None, :], axis=0
    )
    cross_tangent = tf.reduce_mean(
        standardized_tangent[:, :, None, :]
        * centered_direction[:, None, :, None]
        + standardized[:, :, None, None]
        * centered_direction_tangent[:, None, :, :],
        axis=0,
    )
    symmetric_cross = _sym(cross)
    symmetric_cross_tangent = _sym_tangent(cross_tangent)
    projected = centered_direction - tf.linalg.matmul(
        standardized, symmetric_cross
    )
    # Avoid an XLA GEMM autotuner layout failure for the small d x d tangent
    # projection while retaining the exact contractions over coordinate i.
    tangent_projection = tf.reduce_sum(
        standardized_tangent[:, :, None, :]
        * symmetric_cross[None, :, :, None],
        axis=1,
    )
    matrix_tangent_projection = tf.reduce_sum(
        standardized[:, :, None, None]
        * symmetric_cross_tangent[None, :, :, :],
        axis=1,
    )
    projected_tangent = (
        centered_direction_tangent
        - tangent_projection
        - matrix_tangent_projection
    )

    rms = tf.sqrt(
        tf.reduce_mean(tf.square(projected))
        + tf.cast(floor, standardized.dtype)
    )
    rms_tangent = tf.reduce_mean(
        projected[:, :, None] * projected_tangent, axis=[0, 1]
    ) / rms
    normalized_direction = projected / rms
    normalized_direction_tangent = (
        projected_tangent / rms
        - projected[:, :, None] * rms_tangent[None, None, :] / tf.square(rms)
    )
    pre_cap_particle_rms = tf.sqrt(
        tf.reduce_mean(tf.square(normalized_direction), axis=1)
    )
    if particle_rms_cap > 0.0:
        cap = tf.cast(particle_rms_cap, standardized.dtype)
        row_mean_square = tf.reduce_mean(
            tf.square(normalized_direction), axis=1
        )
        row_mean_square_tangent = 2.0 * tf.reduce_mean(
            normalized_direction[:, :, None] * normalized_direction_tangent,
            axis=1,
        )
        scale_base = 1.0 + row_mean_square / tf.square(cap)
        row_scale = tf.math.rsqrt(scale_base)
        row_scale_tangent = (
            -0.5
            * tf.pow(scale_base[:, None], -1.5)
            * row_mean_square_tangent
            / tf.square(cap)
        )
        normalized_direction_tangent = (
            normalized_direction_tangent * row_scale[:, None, None]
            + normalized_direction[:, :, None] * row_scale_tangent[:, None, :]
        )
        normalized_direction *= row_scale[:, None]
    else:
        row_scale = tf.ones_like(pre_cap_particle_rms)
    post_cap_particle_rms = tf.sqrt(
        tf.reduce_mean(tf.square(normalized_direction), axis=1)
    )
    corrected = standardized + tf.cast(strength, standardized.dtype) * normalized_direction
    corrected_tangent = (
        standardized_tangent
        + tf.cast(strength, standardized.dtype) * normalized_direction_tangent
    )
    output, output_tangent = _standardize_uniform_jvp(corrected, corrected_tangent)
    return (
        output,
        output_tangent,
        tf.reduce_max(pre_cap_particle_rms),
        tf.reduce_max(post_cap_particle_rms),
        tf.reduce_min(row_scale),
    )


def _shape_iteration_jvp(
    points: tf.Tensor,
    points_tangent: tf.Tensor,
    target_skew: tf.Tensor,
    target_kurt: tf.Tensor,
    target_skew_tangent: tf.Tensor,
    target_kurt_tangent: tf.Tensor,
    *,
    strength: float,
    floor: float,
    lm_damping: float,
    lm_scale_floor: float,
    trust_radius: float,
) -> tuple[
    tf.Tensor,
    tf.Tensor,
    tf.Tensor,
    tf.Tensor,
    tf.Tensor,
    tf.Tensor,
    tf.Tensor,
]:
    mean, covariance, mean_tangent, covariance_tangent = _uniform_moments_jvp(
        points, points_tangent
    )
    covariance, covariance_tangent = _relative_psd_covariance(
        covariance, covariance_tangent
    )
    chol = tf.linalg.cholesky(covariance)
    chol_tangent = _cholesky_jvp(chol, covariance_tangent)
    centered = points - mean[None, :]
    centered_tangent = points_tangent - mean_tangent[None, :, :]
    u = _right_solve(chol, centered)
    u_tangent = _right_solve_jvp(chol, chol_tangent, centered, centered_tangent)
    m3 = tf.reduce_mean(tf.pow(u, 3.0), axis=0)
    m4 = tf.reduce_mean(tf.pow(u, 4.0), axis=0)
    m3_tangent = tf.reduce_mean(3.0 * tf.pow(u[:, :, None], 2.0) * u_tangent, axis=0)
    m4_tangent = tf.reduce_mean(4.0 * tf.pow(u[:, :, None], 3.0) * u_tangent, axis=0)
    d3 = target_skew - m3
    d4 = target_kurt - m4
    d3_tangent = target_skew_tangent - m3_tangent
    d4_tangent = target_kurt_tangent - m4_tangent

    # Project the moment gradients away from the affine mean/scale tangent
    # space.  Unlike odd/even Hermite updates, these directions can change
    # skewness and kurtosis from a symmetric standardized cloud.
    direction3 = tf.square(u) - 1.0 - m3[None, :] * u
    direction4 = tf.pow(u, 3.0) - m3[None, :] - m4[None, :] * u
    direction3_tangent = (
        2.0 * u[:, :, None] * u_tangent
        - m3_tangent[None, :, :] * u[:, :, None]
        - m3[None, :, None] * u_tangent
    )
    direction4_tangent = (
        3.0 * tf.square(u)[:, :, None] * u_tangent
        - m3_tangent[None, :, :]
        - m4_tangent[None, :, :] * u[:, :, None]
        - m4[None, :, None] * u_tangent
    )

    j33 = tf.reduce_mean(3.0 * tf.square(u) * direction3, axis=0)
    j34 = tf.reduce_mean(3.0 * tf.square(u) * direction4, axis=0)
    j43 = tf.reduce_mean(4.0 * tf.pow(u, 3.0) * direction3, axis=0)
    j44 = tf.reduce_mean(4.0 * tf.pow(u, 3.0) * direction4, axis=0)
    j33_tangent = tf.reduce_mean(
        6.0 * u[:, :, None] * u_tangent * direction3[:, :, None]
        + 3.0 * tf.square(u)[:, :, None] * direction3_tangent,
        axis=0,
    )
    j34_tangent = tf.reduce_mean(
        6.0 * u[:, :, None] * u_tangent * direction4[:, :, None]
        + 3.0 * tf.square(u)[:, :, None] * direction4_tangent,
        axis=0,
    )
    j43_tangent = tf.reduce_mean(
        12.0 * tf.square(u)[:, :, None] * u_tangent * direction3[:, :, None]
        + 4.0 * tf.pow(u, 3.0)[:, :, None] * direction3_tangent,
        axis=0,
    )
    j44_tangent = tf.reduce_mean(
        12.0 * tf.square(u)[:, :, None] * u_tangent * direction4[:, :, None]
        + 4.0 * tf.pow(u, 3.0)[:, :, None] * direction4_tangent,
        axis=0,
    )
    jacobian = tf.stack(
        [tf.stack([j33, j34], axis=-1), tf.stack([j43, j44], axis=-1)],
        axis=-2,
    )
    jacobian_tangent = tf.stack(
        [
            tf.stack([j33_tangent, j34_tangent], axis=-2),
            tf.stack([j43_tangent, j44_tangent], axis=-2),
        ],
        axis=-3,
    )
    residual = tf.stack([d3, d4], axis=-1)
    residual_tangent = tf.stack([d3_tangent, d4_tangent], axis=-2)
    if lm_damping > 0.0:
        lm = scaled_lm_coefficients_jvp(
            jacobian,
            residual,
            jacobian_tangent,
            residual_tangent,
            strength=strength,
            damping=lm_damping,
            scale_floor=lm_scale_floor,
        )
        coefficient = lm["coefficient"]
        coefficient_tangent = lm["coefficient_tangent"]
        maximum_scaled_system_condition = tf.reduce_max(
            lm["scaled_system_condition"]
        )
    else:
        normal = tf.linalg.matmul(jacobian, jacobian, transpose_a=True)
        normal += tf.cast(floor, points.dtype) * tf.eye(
            2, batch_shape=[tf.shape(points)[1]], dtype=points.dtype
        )
        rhs = tf.linalg.matvec(jacobian, residual, transpose_a=True)
        coefficient = tf.cast(strength, points.dtype) * tf.linalg.solve(
            normal, rhs[:, :, None]
        )[:, :, 0]
        normal_tangent = (
            tf.einsum("daip,daj->dijp", jacobian_tangent, jacobian)
            + tf.einsum("dai,dajp->dijp", jacobian, jacobian_tangent)
        )
        rhs_tangent = (
            tf.einsum("daip,da->dip", jacobian_tangent, residual)
            + tf.einsum("dai,dap->dip", jacobian, residual_tangent)
        )
        coefficient_rhs_tangent = (
            rhs_tangent
            - tf.einsum(
                "dijp,dj->dip",
                normal_tangent,
                coefficient / tf.cast(strength, points.dtype),
            )
            if strength > 0.0
            else tf.zeros_like(rhs_tangent)
        )
        parameter_count = tf.shape(points_tangent)[-1]
        normal_batch = tf.broadcast_to(
            normal[None, :, :, :],
            [parameter_count, tf.shape(normal)[0], 2, 2],
        )
        coefficient_tangent = tf.cast(strength, points.dtype) * tf.transpose(
            tf.linalg.solve(
                normal_batch,
                tf.transpose(coefficient_rhs_tangent, [2, 0, 1])[:, :, :, None],
            )[:, :, :, 0],
            [1, 2, 0],
        )
        maximum_scaled_system_condition = tf.zeros([], points.dtype)
    coefficient3 = coefficient[:, 0]
    coefficient4 = coefficient[:, 1]
    coefficient3_tangent = coefficient_tangent[:, 0, :]
    coefficient4_tangent = coefficient_tangent[:, 1, :]
    displacement = (
        direction3 * coefficient3[None, :]
        + direction4 * coefficient4[None, :]
    )
    displacement_tangent = (
        direction3_tangent * coefficient3[None, :, None]
        + direction3[:, :, None] * coefficient3_tangent[None, :, :]
        + direction4_tangent * coefficient4[None, :, None]
        + direction4[:, :, None] * coefficient4_tangent[None, :, :]
    )
    pre_cap_rms = tf.sqrt(tf.reduce_mean(tf.square(displacement), axis=1))
    if trust_radius > 0.0:
        capped = smooth_rms_cap_jvp(
            displacement, displacement_tangent, radius=trust_radius
        )
        displacement = capped["displacement"]
        displacement_tangent = capped["displacement_tangent"]
        post_cap_rms = capped["post_rms"]
    else:
        post_cap_rms = pre_cap_rms
    corrected = u + displacement
    corrected_tangent = (
        u_tangent
        + displacement_tangent
    )
    corrected_mean, corrected_cov, corrected_mean_tangent, corrected_cov_tangent = _uniform_moments_jvp(
        corrected, corrected_tangent
    )
    corrected_chol = tf.linalg.cholesky(corrected_cov)
    corrected_chol_tangent = _cholesky_jvp(corrected_chol, corrected_cov_tangent)
    output = _right_solve(corrected_chol, corrected - corrected_mean[None, :])
    output_tangent = _right_solve_jvp(
        corrected_chol,
        corrected_chol_tangent,
        corrected - corrected_mean[None, :],
        corrected_tangent - corrected_mean_tangent[None, :, :],
    )
    return (
        output,
        output_tangent,
        target_skew - tf.reduce_mean(tf.pow(output, 3.0), axis=0),
        target_kurt - tf.reduce_mean(tf.pow(output, 4.0), axis=0),
        maximum_scaled_system_condition,
        tf.reduce_max(pre_cap_rms),
        tf.reduce_max(post_cap_rms),
    )


def higher_moment_shape_jvp(
    source: tf.Tensor,
    weights: tf.Tensor,
    source_tangent: tf.Tensor,
    weights_tangent: tf.Tensor,
    points: tf.Tensor,
    points_tangent: tf.Tensor,
    *,
    correction_steps: int,
    strength: float,
    floor: float,
    diagonal_lm_damping: float = 0.0,
    diagonal_lm_scale_floor: float = 1.0e-6,
    diagonal_trust_radius: float = 0.0,
    pairwise_correction_steps: int = 0,
    pairwise_strength: float = 0.0,
    pairwise_floor: float = 1.0e-6,
    pairwise_particle_rms_cap: float = 0.0,
    coordinatewise_bounded_cap: float = 0.0,
    coordinatewise_bounded_cap_power: int = 8,
    coordinatewise_standardized_cap: float = 0.0,
    coordinatewise_standardized_cap_power: int = 8,
    projected_cumulant_basis: tf.Tensor | None = None,
    projected_cumulant_correction_steps: int = 0,
    projected_cumulant_strength: float = 0.0,
    projected_cumulant_floor: float = 1.0e-6,
    explicit_target_skew: tf.Tensor | None = None,
    explicit_target_kurtosis: tf.Tensor | None = None,
    explicit_target_skew_tangent: tf.Tensor | None = None,
    explicit_target_kurtosis_tangent: tf.Tensor | None = None,
    explicit_target_pairwise_co_skew: tf.Tensor | None = None,
    explicit_target_pairwise_co_kurtosis: tf.Tensor | None = None,
    explicit_target_pairwise_co_skew_tangent: tf.Tensor | None = None,
    explicit_target_pairwise_co_kurtosis_tangent: tf.Tensor | None = None,
    pairwise_target_mask: tf.Tensor | None = None,
    pairwise_co_skew_target_mask: tf.Tensor | None = None,
    pairwise_co_kurtosis_target_mask: tf.Tensor | None = None,
) -> dict[str, tf.Tensor]:
    """Apply bounded diagonal, pairwise, and projected moment corrections.

    Explicit targets are an all-or-none opt-in group.  They replace only the
    standardized shape targets; weighted particle mean and covariance remain
    the affine restoration targets.
    """
    if (
        correction_steps < 0
        or strength < 0.0
        or floor <= 0.0
        or diagonal_lm_damping < 0.0
        or diagonal_lm_scale_floor <= 0.0
        or diagonal_trust_radius < 0.0
        or pairwise_correction_steps < 0
        or pairwise_strength < 0.0
        or pairwise_floor <= 0.0
        or pairwise_particle_rms_cap < 0.0
        or coordinatewise_bounded_cap < 0.0
        or coordinatewise_bounded_cap >= 1.0
        or coordinatewise_bounded_cap_power < 2
        or coordinatewise_bounded_cap_power % 2 != 0
        or coordinatewise_standardized_cap < 0.0
        or coordinatewise_standardized_cap >= 1.0
        or coordinatewise_standardized_cap_power < 2
        or coordinatewise_standardized_cap_power % 2 != 0
        or projected_cumulant_correction_steps < 0
        or projected_cumulant_strength < 0.0
        or projected_cumulant_floor <= 0.0
    ):
        raise ValueError("invalid higher-moment correction controls")
    if coordinatewise_bounded_cap > 0.0 and coordinatewise_standardized_cap > 0.0:
        raise ValueError(
            "bounded-chart and generic standardized coordinate caps are mutually exclusive"
        )
    if projected_cumulant_correction_steps > 0 and projected_cumulant_basis is None:
        raise ValueError("projected_cumulant_basis is required for nonzero projected steps")
    state_dim = tf.shape(source)[1]
    mean, covariance, mean_tangent, covariance_tangent = _weighted_moments_jvp(
        source, weights, source_tangent, weights_tangent
    )
    covariance, covariance_tangent = _relative_psd_covariance(
        covariance, covariance_tangent
    )
    target_chol = tf.linalg.cholesky(covariance)
    target_chol_tangent = _cholesky_jvp(target_chol, covariance_tangent)
    target_skew, target_kurt, target_skew_tangent, target_kurt_tangent = _weighted_diag_moments_jvp(
        source, weights, source_tangent, weights_tangent, mean, mean_tangent, target_chol, target_chol_tangent
    )
    feasibility = necessary_marginal_feasibility(
        target_skew, target_kurt, tf.shape(source)[0]
    )
    source_centered = source - mean[None, :]
    source_centered_tangent = source_tangent - mean_tangent[None, :, :]
    source_standardized = _right_solve(target_chol, source_centered)
    source_standardized_tangent = _right_solve_jvp(
        target_chol,
        target_chol_tangent,
        source_centered,
        source_centered_tangent,
    )
    projected_basis = None
    if projected_cumulant_basis is not None:
        projected_basis = tf.convert_to_tensor(
            projected_cumulant_basis, dtype=source.dtype
        )
        if projected_basis.shape.rank != 2:
            raise ValueError("projected_cumulant_basis must be a matrix")
        if (
            source.shape[1] is not None
            and projected_basis.shape[0] is not None
            and source.shape[1] != projected_basis.shape[0]
        ):
            raise ValueError("projected_cumulant_basis has the wrong state dimension")
        basis_gram = tf.linalg.matmul(projected_basis, projected_basis, transpose_a=True)
        projected_basis_valid = tf.reduce_max(
            tf.abs(
                basis_gram
                - tf.eye(tf.shape(projected_basis)[1], dtype=source.dtype)
            )
        ) <= tf.cast(2.0e-4, source.dtype)
    else:
        projected_basis_valid = tf.constant(True)
    (
        target_co_skew,
        target_co_kurtosis,
        target_co_skew_tangent,
        target_co_kurtosis_tangent,
    ) = _weighted_pair_moments_jvp(
        source_standardized,
        weights,
        source_standardized_tangent,
        weights_tangent,
    )
    explicit_targets = (
        explicit_target_skew,
        explicit_target_kurtosis,
        explicit_target_skew_tangent,
        explicit_target_kurtosis_tangent,
        explicit_target_pairwise_co_skew,
        explicit_target_pairwise_co_kurtosis,
        explicit_target_pairwise_co_skew_tangent,
        explicit_target_pairwise_co_kurtosis_tangent,
    )
    explicit_count = sum(value is not None for value in explicit_targets)
    if explicit_count not in (0, len(explicit_targets)):
        raise ValueError("explicit higher-moment targets must be supplied as one complete group")
    target_source_id = tf.constant(0, tf.int32)
    off_diagonal = 1.0 - tf.eye(tf.shape(source)[1], dtype=source.dtype)
    target_co_skew_mask = off_diagonal
    target_co_kurtosis_mask = off_diagonal
    if explicit_count:
        expected_vector = target_skew.shape
        expected_matrix = target_co_skew.shape
        expected_vector_tangent = target_skew_tangent.shape
        expected_matrix_tangent = target_co_skew_tangent.shape
        target_skew = _explicit_target(
            explicit_target_skew, expected_vector, source.dtype, "explicit_target_skew"
        )
        target_kurt = _explicit_target(
            explicit_target_kurtosis, expected_vector, source.dtype, "explicit_target_kurtosis"
        )
        target_skew_tangent = _explicit_target(
            explicit_target_skew_tangent,
            expected_vector_tangent,
            source.dtype,
            "explicit_target_skew_tangent",
        )
        target_kurt_tangent = _explicit_target(
            explicit_target_kurtosis_tangent,
            expected_vector_tangent,
            source.dtype,
            "explicit_target_kurtosis_tangent",
        )
        target_co_skew = _explicit_target(
            explicit_target_pairwise_co_skew,
            expected_matrix,
            source.dtype,
            "explicit_target_pairwise_co_skew",
        )
        target_co_kurtosis = _explicit_target(
            explicit_target_pairwise_co_kurtosis,
            expected_matrix,
            source.dtype,
            "explicit_target_pairwise_co_kurtosis",
        )
        target_co_skew_tangent = _explicit_target(
            explicit_target_pairwise_co_skew_tangent,
            expected_matrix_tangent,
            source.dtype,
            "explicit_target_pairwise_co_skew_tangent",
        )
        target_co_kurtosis_tangent = _explicit_target(
            explicit_target_pairwise_co_kurtosis_tangent,
            expected_matrix_tangent,
            source.dtype,
            "explicit_target_pairwise_co_kurtosis_tangent",
        )
        target_source_id = tf.constant(1, tf.int32)
    if pairwise_target_mask is not None:
        if (
            pairwise_co_skew_target_mask is not None
            or pairwise_co_kurtosis_target_mask is not None
        ):
            raise ValueError(
                "pairwise_target_mask cannot be combined with moment-specific masks"
            )
        common_mask = _explicit_target(
            pairwise_target_mask,
            target_co_skew.shape,
            source.dtype,
            "pairwise_target_mask",
        )
        common_mask = tf.clip_by_value(
            common_mask, tf.cast(0.0, source.dtype), tf.cast(1.0, source.dtype)
        ) * off_diagonal
        target_co_skew_mask = common_mask
        target_co_kurtosis_mask = common_mask
    if pairwise_co_skew_target_mask is not None:
        target_co_skew_mask = _explicit_target(
            pairwise_co_skew_target_mask,
            target_co_skew.shape,
            source.dtype,
            "pairwise_co_skew_target_mask",
        )
        target_co_skew_mask = tf.clip_by_value(
            target_co_skew_mask,
            tf.cast(0.0, source.dtype),
            tf.cast(1.0, source.dtype),
        ) * off_diagonal
    if pairwise_co_kurtosis_target_mask is not None:
        target_co_kurtosis_mask = _explicit_target(
            pairwise_co_kurtosis_target_mask,
            target_co_kurtosis.shape,
            source.dtype,
            "pairwise_co_kurtosis_target_mask",
        )
        target_co_kurtosis_mask = tf.clip_by_value(
            target_co_kurtosis_mask,
            tf.cast(0.0, source.dtype),
            tf.cast(1.0, source.dtype),
        ) * off_diagonal

    if (
        correction_steps == 0
        and pairwise_correction_steps == 0
        and projected_cumulant_correction_steps == 0
    ):
        output_standardized, output_standardized_tangent = _standardize_uniform_jvp(
            points, points_tangent
        )
        output_skew = tf.reduce_mean(tf.pow(output_standardized, 3.0), axis=0)
        output_kurt = tf.reduce_mean(tf.pow(output_standardized, 4.0), axis=0)
        output_co_skew, output_co_kurtosis, _, _ = _uniform_pair_moments_jvp(
            output_standardized, output_standardized_tangent
        )
        valid = tf.reduce_all(tf.math.is_finite(points)) & tf.reduce_all(
            tf.math.is_finite(points_tangent)
        )
        return {
            "particles": points,
            "particles_tangent": points_tangent,
            "target_skew": target_skew,
            "target_kurtosis": target_kurt,
            "target_pairwise_co_skew": target_co_skew,
            "target_pairwise_co_kurtosis": target_co_kurtosis,
            "skew_residual": target_skew - output_skew,
            "kurtosis_residual": target_kurt - output_kurt,
            "pairwise_co_skew_residual": target_co_skew_mask
            * (target_co_skew - output_co_skew),
            "pairwise_co_kurtosis_residual": (
                target_co_kurtosis_mask
                * (target_co_kurtosis - output_co_kurtosis)
            ),
            "pairwise_target_mask": tf.maximum(
                target_co_skew_mask, target_co_kurtosis_mask
            ),
            "pairwise_co_skew_target_mask": target_co_skew_mask,
            "pairwise_co_kurtosis_target_mask": target_co_kurtosis_mask,
            "maximum_pairwise_pre_cap_particle_rms": tf.zeros([], source.dtype),
            "maximum_pairwise_post_cap_particle_rms": tf.zeros([], source.dtype),
            "minimum_pairwise_particle_cap_scale": tf.ones([], source.dtype),
            "target_source_id": target_source_id,
            "projected_cumulant_residual_norm": tf.zeros([], source.dtype),
            "projected_cumulant_third_residual_norm": tf.zeros([], source.dtype),
            "projected_cumulant_fourth_residual_norm": tf.zeros([], source.dtype),
            "valid": valid & projected_basis_valid,
            "maximum_coordinatewise_pre_cap_absolute": tf.reduce_max(
                tf.abs(points)
            ),
            "maximum_coordinatewise_post_cap_absolute": tf.reduce_max(
                tf.abs(points)
            ),
            "mean_coordinatewise_cap_displacement": tf.zeros([], source.dtype),
            "fraction_coordinatewise_cap_active": tf.zeros([], source.dtype),
            "minimum_coordinatewise_cap_derivative": tf.ones([], source.dtype),
            "minimum_pearson_feasibility_margin": tf.reduce_min(
                feasibility["pearson_margin"]
            ),
            "minimum_finite_particle_upper_margin": tf.reduce_min(
                feasibility["finite_particle_upper_margin"]
            ),
            "maximum_diagonal_scaled_system_condition": tf.zeros([], source.dtype),
            "maximum_diagonal_pre_cap_particle_rms": tf.zeros([], source.dtype),
            "maximum_diagonal_post_cap_particle_rms": tf.zeros([], source.dtype),
        }

    steps = tf.constant(correction_steps, tf.int32)
    residual3 = tf.zeros([state_dim], source.dtype)
    residual4 = tf.zeros([state_dim], source.dtype)
    maximum_diagonal_condition = tf.zeros([], source.dtype)
    maximum_diagonal_pre_cap_rms = tf.zeros([], source.dtype)
    maximum_diagonal_post_cap_rms = tf.zeros([], source.dtype)

    def body(index, current, current_tangent, _r3, _r4, max_condition, max_pre, max_post):
        (
            next_points,
            next_tangent,
            next_r3,
            next_r4,
            next_condition,
            next_pre,
            next_post,
        ) = _shape_iteration_jvp(
            current, current_tangent, target_skew, target_kurt,
            target_skew_tangent, target_kurt_tangent,
            strength=strength,
            floor=floor,
            lm_damping=diagonal_lm_damping,
            lm_scale_floor=diagonal_lm_scale_floor,
            trust_radius=diagonal_trust_radius,
        )
        return (
            index + 1,
            next_points,
            next_tangent,
            next_r3,
            next_r4,
            tf.maximum(max_condition, next_condition),
            tf.maximum(max_pre, next_pre),
            tf.maximum(max_post, next_post),
        )

    # Initialize the loop state with a standardization of the input cloud.
    initial_mean, initial_cov, initial_mean_tangent, initial_cov_tangent = _uniform_moments_jvp(points, points_tangent)
    initial_cov_safe, initial_cov_tangent_safe = _relative_psd_covariance(
        initial_cov, initial_cov_tangent, relative_floor=RELATIVE_PSD_FLOOR
    )
    initial_chol = tf.linalg.cholesky(initial_cov_safe)
    initial_chol_tangent = _cholesky_jvp(initial_chol, initial_cov_tangent_safe)
    initial_standardized = _right_solve(initial_chol, points - initial_mean[None, :])
    initial_standardized_tangent = _right_solve_jvp(
        initial_chol, initial_chol_tangent, points - initial_mean[None, :], points_tangent - initial_mean_tangent[None, :, :]
    )

    (
        _,
        standardized,
        standardized_tangent,
        residual3,
        residual4,
        maximum_diagonal_condition,
        maximum_diagonal_pre_cap_rms,
        maximum_diagonal_post_cap_rms,
    ) = tf.while_loop(
        lambda index, *_: index < steps,
        body,
        (
            tf.zeros([], tf.int32),
            initial_standardized,
            initial_standardized_tangent,
            residual3,
            residual4,
            maximum_diagonal_condition,
            maximum_diagonal_pre_cap_rms,
            maximum_diagonal_post_cap_rms,
        ),
        parallel_iterations=1,
    )
    # Scalar states have no off-diagonal pair moments. Skip the loop entirely
    # so a nonzero pairwise control is an exact structural no-op at d=1.
    pairwise_steps = tf.where(
        state_dim > 1,
        tf.constant(pairwise_correction_steps, tf.int32),
        tf.zeros([], tf.int32),
    )

    def pairwise_body(
        index,
        current,
        current_tangent,
        maximum_pre_cap_rms,
        maximum_post_cap_rms,
        minimum_cap_scale,
    ):
        (
            next_points,
            next_tangent,
            next_pre_cap_rms,
            next_post_cap_rms,
            next_minimum_cap_scale,
        ) = _pairwise_shape_iteration_jvp(
            current,
            current_tangent,
            target_co_skew,
            target_co_kurtosis,
            target_co_skew_tangent,
            target_co_kurtosis_tangent,
            target_co_skew_mask,
            target_co_kurtosis_mask,
            strength=pairwise_strength,
            floor=pairwise_floor,
            particle_rms_cap=pairwise_particle_rms_cap,
        )
        return (
            index + 1,
            next_points,
            next_tangent,
            tf.maximum(maximum_pre_cap_rms, next_pre_cap_rms),
            tf.maximum(maximum_post_cap_rms, next_post_cap_rms),
            tf.minimum(minimum_cap_scale, next_minimum_cap_scale),
        )

    (
        _,
        standardized,
        standardized_tangent,
        maximum_pairwise_pre_cap_particle_rms,
        maximum_pairwise_post_cap_particle_rms,
        minimum_pairwise_particle_cap_scale,
    ) = tf.while_loop(
        lambda index, *_: index < pairwise_steps,
        pairwise_body,
        (
            tf.zeros([], tf.int32),
            standardized,
            standardized_tangent,
            tf.zeros([], source.dtype),
            tf.zeros([], source.dtype),
            tf.ones([], source.dtype),
        ),
        parallel_iterations=1,
    )
    projected_steps = tf.constant(projected_cumulant_correction_steps, tf.int32)

    def projected_body(index, current, current_tangent):
        next_points, next_tangent = _projected_cumulant_iteration_jvp(
            source_standardized,
            weights,
            source_standardized_tangent,
            weights_tangent,
            current,
            current_tangent,
            projected_basis,
            strength=projected_cumulant_strength,
            floor=projected_cumulant_floor,
        )
        return index + 1, next_points, next_tangent

    if projected_basis is not None:
        _, standardized, standardized_tangent = tf.while_loop(
            lambda index, *_: index < projected_steps,
            projected_body,
            (tf.zeros([], tf.int32), standardized, standardized_tangent),
            parallel_iterations=1,
        )
    # The Austria bounded-teacher route caps its local chart through
    # ``coordinatewise_bounded_cap``.  This separate opt-in applies the same
    # smooth map to the final standardized coordinates for generic models,
    # where no bounded teacher frame is available.  It is an extension and
    # deliberately does not change the historical/default route.
    standardized_pre_cap = standardized
    standardized_pre_cap_tangent = standardized_tangent
    if coordinatewise_standardized_cap > 0.0:
        standardized_cap = tf.cast(coordinatewise_standardized_cap, source.dtype)
        standardized_power = tf.cast(
            coordinatewise_standardized_cap_power, source.dtype
        )
        standardized_scaled_power = tf.pow(
            standardized_pre_cap / standardized_cap, standardized_power
        )
        standardized_denominator = tf.pow(
            1.0 + standardized_scaled_power, 1.0 / standardized_power
        )
        standardized = standardized_pre_cap / standardized_denominator
        standardized_derivative = tf.pow(
            1.0 + standardized_scaled_power,
            -1.0 / standardized_power - 1.0,
        )
        standardized_tangent = (
            standardized_derivative[:, :, None] * standardized_pre_cap_tangent
        )
    raw_output = mean[None, :] + tf.linalg.matmul(
        standardized, target_chol, transpose_b=True
    )
    raw_output_tangent = (
        mean_tangent[None, :, :]
        + tf.einsum("nip,ji->njp", standardized_tangent, target_chol)
        + tf.einsum("ni,jip->njp", standardized, target_chol_tangent)
    )
    if coordinatewise_standardized_cap > 0.0:
        standardized_restore = affine_restore_cloud_jvp(
            source,
            weights,
            source_tangent,
            weights_tangent,
            raw_output,
            raw_output_tangent,
        )
        raw_output = standardized_restore["particles"]
        raw_output_tangent = standardized_restore["particles_tangent"]
        standardized_restore_valid = standardized_restore["valid"]
    else:
        standardized_restore_valid = tf.constant(True)
    coordinatewise_pre_cap = raw_output
    coordinatewise_pre_cap_tangent = raw_output_tangent
    if coordinatewise_bounded_cap > 0.0:
        cap = tf.cast(coordinatewise_bounded_cap, source.dtype)
        power = tf.cast(coordinatewise_bounded_cap_power, source.dtype)
        scaled_power = tf.pow(coordinatewise_pre_cap / cap, power)
        cap_denominator = tf.pow(1.0 + scaled_power, 1.0 / power)
        coordinatewise_cap = coordinatewise_pre_cap / cap_denominator
        # Differentiate the odd smooth cap directly. The even power avoids
        # sign ambiguity while retaining a smooth, bounded map.
        cap_derivative = tf.pow(1.0 + scaled_power, -1.0 / power - 1.0)
        coordinatewise_cap_tangent = (
            cap_derivative[:, :, None] * coordinatewise_pre_cap_tangent
        )
    else:
        coordinatewise_cap = coordinatewise_pre_cap
        coordinatewise_cap_tangent = coordinatewise_pre_cap_tangent
        cap_derivative = tf.ones_like(coordinatewise_pre_cap)
    output = coordinatewise_cap
    output_tangent = coordinatewise_cap_tangent
    output_standardized, output_standardized_tangent = _standardize_uniform_jvp(
        output,
        output_tangent,
    )
    output_skew = tf.reduce_mean(tf.pow(output_standardized, 3.0), axis=0)
    output_kurt = tf.reduce_mean(tf.pow(output_standardized, 4.0), axis=0)
    output_co_skew, output_co_kurtosis, _, _ = _uniform_pair_moments_jvp(
        output_standardized, output_standardized_tangent
    )
    if projected_basis is None:
        projected_residual3 = tf.zeros([1, 1, 1], source.dtype)
        projected_residual4 = tf.zeros([1, 1, 1, 1], source.dtype)
    else:
        (
            projected_residual3,
            projected_residual4,
            _,
            _,
            _,
            _,
        ) = _projected_residuals_jvp(
            source_standardized,
            weights,
            source_standardized_tangent,
            weights_tangent,
            output_standardized,
            output_standardized_tangent,
            projected_basis,
        )
    valid = (
        tf.reduce_all(tf.math.is_finite(output))
        & tf.reduce_all(tf.math.is_finite(output_tangent))
        & standardized_restore_valid
    )
    return {
        "particles": output,
        "particles_tangent": output_tangent,
        "target_skew": target_skew,
        "target_kurtosis": target_kurt,
        "target_pairwise_co_skew": target_co_skew,
        "target_pairwise_co_kurtosis": target_co_kurtosis,
        "skew_residual": target_skew - output_skew,
        "kurtosis_residual": target_kurt - output_kurt,
        "pairwise_co_skew_residual": target_co_skew_mask
        * (target_co_skew - output_co_skew),
        "pairwise_co_kurtosis_residual": target_co_kurtosis_mask
        * (target_co_kurtosis - output_co_kurtosis),
        "pairwise_target_mask": tf.maximum(
            target_co_skew_mask, target_co_kurtosis_mask
        ),
        "pairwise_co_skew_target_mask": target_co_skew_mask,
        "pairwise_co_kurtosis_target_mask": target_co_kurtosis_mask,
        "maximum_pairwise_pre_cap_particle_rms": (
            maximum_pairwise_pre_cap_particle_rms
        ),
        "maximum_pairwise_post_cap_particle_rms": (
            maximum_pairwise_post_cap_particle_rms
        ),
        "minimum_pairwise_particle_cap_scale": (
            minimum_pairwise_particle_cap_scale
        ),
        "maximum_coordinatewise_pre_cap_absolute": tf.reduce_max(
            tf.abs(
                standardized_pre_cap
                if coordinatewise_standardized_cap > 0.0
                else coordinatewise_pre_cap
            )
        ),
        "maximum_coordinatewise_post_cap_absolute": tf.reduce_max(
            tf.abs(
                standardized
                if coordinatewise_standardized_cap > 0.0
                else coordinatewise_cap
            )
        ),
        "mean_coordinatewise_cap_displacement": tf.reduce_mean(
            tf.abs(
                (standardized - standardized_pre_cap)
                if coordinatewise_standardized_cap > 0.0
                else (coordinatewise_cap - coordinatewise_pre_cap)
            )
        ),
        "fraction_coordinatewise_cap_active": tf.reduce_mean(
            tf.cast(
                tf.abs(
                    (standardized - standardized_pre_cap)
                    if coordinatewise_standardized_cap > 0.0
                    else (coordinatewise_cap - coordinatewise_pre_cap)
                )
                > 1.0e-7,
                source.dtype,
            )
        ),
        "minimum_coordinatewise_cap_derivative": tf.reduce_min(
            tf.abs(
                standardized_derivative
                if coordinatewise_standardized_cap > 0.0
                else cap_derivative
            )
        ),
        "target_source_id": target_source_id,
        "projected_cumulant_residual_norm": tf.sqrt(
            tf.reduce_sum(tf.square(projected_residual3))
            + tf.reduce_sum(tf.square(projected_residual4))
        ),
        "projected_cumulant_third_residual_norm": tf.linalg.norm(
            projected_residual3
        ),
        "projected_cumulant_fourth_residual_norm": tf.linalg.norm(
            projected_residual4
        ),
        "minimum_pearson_feasibility_margin": tf.reduce_min(
            feasibility["pearson_margin"]
        ),
        "minimum_finite_particle_upper_margin": tf.reduce_min(
            feasibility["finite_particle_upper_margin"]
        ),
        "maximum_diagonal_scaled_system_condition": (
            maximum_diagonal_condition
        ),
        "maximum_diagonal_pre_cap_particle_rms": (
            maximum_diagonal_pre_cap_rms
        ),
        "maximum_diagonal_post_cap_particle_rms": (
            maximum_diagonal_post_cap_rms
        ),
        "valid": valid & projected_basis_valid,
    }


def _explicit_target(
    value: tf.Tensor | None,
    expected_shape: tf.TensorShape,
    dtype: tf.DType,
    name: str,
) -> tf.Tensor:
    if value is None:
        raise ValueError(f"{name} is required")
    tensor = tf.convert_to_tensor(value, dtype=dtype)
    if not tensor.shape.is_compatible_with(expected_shape):
        raise ValueError(f"{name} has shape {tensor.shape}, expected {expected_shape}")
    return tf.ensure_shape(tensor, expected_shape)


__all__ = [
    "affine_restore_cloud_jvp",
    "higher_moment_shape_jvp",
    "weighted_shape_targets_jvp",
]
