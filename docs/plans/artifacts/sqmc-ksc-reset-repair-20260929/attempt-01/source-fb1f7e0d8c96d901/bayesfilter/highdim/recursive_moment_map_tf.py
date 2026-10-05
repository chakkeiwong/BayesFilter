"""Generic lagged transition-moment maps for recursive filtering diagnostics.

The kernel in this module is deliberately independent of a model, proposal,
or TT representation.  A caller supplies one conditional transition mean and
covariance for each carried-cloud row.  The kernel applies the law of total
covariance, checks the resulting matrix, and returns a lower-Cholesky affine
map.  A map constructed at time ``t`` is intended to be frozen while the
current target is fitted; rebuilding it from the current fit would define a
different finite program.

The repeated arithmetic uses fixed-shape TensorFlow functions.  NumPy,
sample-wise numerical Python loops, and implicit pfor are intentionally absent.
This module is a diagnostic candidate and does not assert posterior or
likelihood correctness.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import tensorflow as tf

from bayesfilter.highdim.filtering import AffineCoordinateMap


DTYPE = tf.float64
ROUTE_ID = "recursive_lagged_transition_moment_map_v1"
ROUTE_CLASSIFICATION = "extension_or_invention_candidate_diagnostic_only"


def _symmetrize(matrix: tf.Tensor) -> tf.Tensor:
    return 0.5 * (matrix + tf.linalg.matrix_transpose(matrix))


def _safe_identity(size: int) -> tf.Tensor:
    return tf.eye(int(size), dtype=DTYPE)


@dataclass(frozen=True)
class MomentMapConfig:
    """Numerical validity policy for the fixed-shape moment kernel."""

    weight_tolerance: float = 1.0e-10
    covariance_psd_tolerance: float = 1.0e-12

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.weight_tolerance)) or float(self.weight_tolerance) <= 0.0:
            raise ValueError("weight_tolerance must be finite and positive")
        if not math.isfinite(float(self.covariance_psd_tolerance)) or float(self.covariance_psd_tolerance) < 0.0:
            raise ValueError("covariance_psd_tolerance must be finite and nonnegative")


def make_weighted_transition_moment_kernel(
    *,
    particle_count: int,
    state_dim: int,
    config: MomentMapConfig | None = None,
    jit_compile: bool = True,
):
    """Build a fixed-shape law-of-total-covariance kernel.

    Inputs have shapes ``[N]``, ``[N,D]``, and ``[N,D,D]`` for normalized
    weights, conditional means, and conditional covariances.  The output
    contains a scalar ``valid`` flag.  Invalid inputs use an identity matrix
    only to keep the compiled algebra finite; callers must reject ``valid=False``
    and therefore cannot mistake the identity for an accepted repair.
    """

    particle_count = int(particle_count)
    state_dim = int(state_dim)
    if particle_count < 1 or state_dim < 1:
        raise ValueError("particle_count and state_dim must be positive")
    cfg = MomentMapConfig() if config is None else config
    identity = _safe_identity(state_dim)
    weight_tol = tf.constant(float(cfg.weight_tolerance), DTYPE)
    covariance_tol = tf.constant(float(cfg.covariance_psd_tolerance), DTYPE)

    @tf.function(
        input_signature=[
            tf.TensorSpec([particle_count], DTYPE),
            tf.TensorSpec([particle_count, state_dim], DTYPE),
            tf.TensorSpec([particle_count, state_dim, state_dim], DTYPE),
        ],
        jit_compile=bool(jit_compile),
        autograph=False,
        reduce_retracing=True,
    )
    def kernel(
        weights: tf.Tensor,
        conditional_means: tf.Tensor,
        conditional_covariances: tf.Tensor,
    ) -> Mapping[str, tf.Tensor]:
        weights = tf.ensure_shape(weights, [particle_count])
        conditional_means = tf.ensure_shape(
            conditional_means, [particle_count, state_dim]
        )
        conditional_covariances = tf.ensure_shape(
            conditional_covariances, [particle_count, state_dim, state_dim]
        )
        all_finite = (
            tf.reduce_all(tf.math.is_finite(weights))
            & tf.reduce_all(tf.math.is_finite(conditional_means))
            & tf.reduce_all(tf.math.is_finite(conditional_covariances))
        )
        weight_sum = tf.reduce_sum(weights)
        weight_scale = tf.maximum(tf.constant(1.0, DTYPE), tf.abs(weight_sum))
        weights_valid = (
            all_finite
            & tf.reduce_all(weights >= tf.constant(0.0, DTYPE))
            & (tf.abs(weight_sum - tf.constant(1.0, DTYPE)) <= weight_tol * weight_scale)
        )
        safe_weights = tf.where(
            weights_valid,
            weights,
            tf.fill([particle_count], tf.constant(1.0 / particle_count, DTYPE)),
        )
        safe_means = tf.where(
            all_finite,
            conditional_means,
            tf.zeros_like(conditional_means),
        )
        safe_covariances = tf.where(
            all_finite,
            _symmetrize(conditional_covariances),
            tf.broadcast_to(identity[tf.newaxis, :, :], [particle_count, state_dim, state_dim]),
        )
        conditional_eigenvalues = tf.linalg.eigvalsh(safe_covariances)
        conditional_minimum = tf.reduce_min(conditional_eigenvalues, axis=1)
        conditional_psd_valid = tf.reduce_all(conditional_minimum >= -covariance_tol)

        predicted_mean = tf.einsum("n,nd->d", safe_weights, safe_means)
        centered = safe_means - predicted_mean[tf.newaxis, :]
        between = tf.einsum("ni,nj->nij", centered, centered)
        predicted_covariance_raw = tf.einsum(
            "n,nij->ij", safe_weights, safe_covariances + between
        )
        predicted_covariance = _symmetrize(predicted_covariance_raw)
        predicted_finite = tf.reduce_all(tf.math.is_finite(predicted_covariance))

        raw_eigenvalues = tf.linalg.eigvalsh(
            tf.where(predicted_finite, predicted_covariance, identity)
        )
        raw_minimum = tf.reduce_min(raw_eigenvalues)
        predicted_spd_valid = predicted_finite & (raw_minimum > tf.constant(0.0, DTYPE))
        safe_covariance = tf.where(predicted_spd_valid, predicted_covariance, identity)
        cholesky = tf.linalg.cholesky(safe_covariance)
        eigenvalues = tf.linalg.eigvalsh(safe_covariance)
        minimum_eigenvalue = tf.reduce_min(eigenvalues)
        maximum_eigenvalue = tf.reduce_max(eigenvalues)
        condition_number = tf.where(
            predicted_spd_valid,
            maximum_eigenvalue / minimum_eigenvalue,
            tf.constant(-1.0, DTYPE),
        )
        valid = (
            weights_valid
            & conditional_psd_valid
            & predicted_spd_valid
            & tf.reduce_all(tf.math.is_finite(predicted_mean))
            & tf.reduce_all(tf.math.is_finite(cholesky))
        )
        reconstruction = tf.matmul(cholesky, cholesky, transpose_b=True)
        reconstruction_error = tf.reduce_max(
            tf.abs(reconstruction - safe_covariance)
        )
        return {
            "predicted_mean": predicted_mean,
            "predicted_covariance": safe_covariance,
            "predicted_covariance_raw": predicted_covariance,
            "predicted_cholesky": cholesky,
            "minimum_eigenvalue": minimum_eigenvalue,
            "raw_minimum_eigenvalue": tf.where(
                predicted_finite, raw_minimum, tf.constant(-1.0, DTYPE)
            ),
            "maximum_eigenvalue": maximum_eigenvalue,
            "condition_number": condition_number,
            "weight_sum": weight_sum,
            "conditional_minimum_eigenvalue": tf.reduce_min(conditional_minimum),
            "reconstruction_error": reconstruction_error,
            "weights_valid": weights_valid,
            "conditional_psd_valid": conditional_psd_valid,
            "predicted_spd_valid": predicted_spd_valid,
            "valid": valid,
            "finite": valid,
        }

    return kernel


def require_valid_moment_map(result: Mapping[str, tf.Tensor]) -> None:
    """Reject the identity fallback emitted for invalid compiled rows."""

    value = result.get("valid")
    if value is None:
        raise ValueError("moment-map result does not expose a validity flag")
    if not bool(tf.convert_to_tensor(value).numpy()):
        details = {
            name: bool(tf.convert_to_tensor(result[name]).numpy())
            for name in ("weights_valid", "conditional_psd_valid", "predicted_spd_valid")
            if name in result
        }
        raise ValueError(f"invalid transition moments: {details}")


def build_lagged_moment_map(
    weights: tf.Tensor,
    conditional_means: tf.Tensor,
    conditional_covariances: tf.Tensor,
    *,
    jit_compile: bool = True,
    config: MomentMapConfig | None = None,
    kernel=None,
) -> Mapping[str, object]:
    """Evaluate the kernel and construct a checked frozen affine map.

    A caller that rebuilds the map at several times can pass one kernel built
    for the fixed ``[N,D]`` shape.  This keeps the TensorFlow trace stable
    while preserving the lagged value semantics.
    """

    weights = tf.convert_to_tensor(weights, DTYPE)
    means = tf.convert_to_tensor(conditional_means, DTYPE)
    covariances = tf.convert_to_tensor(conditional_covariances, DTYPE)
    if weights.shape.rank != 1 or means.shape.rank != 2 or covariances.shape.rank != 3:
        raise ValueError("moment inputs must have ranks 1, 2, and 3")
    if means.shape[0] != weights.shape[0] or covariances.shape[:2] != means.shape:
        raise ValueError("moment input shapes are inconsistent")
    if covariances.shape[1] != means.shape[1]:
        raise ValueError("conditional covariance state dimension mismatch")
    if kernel is None:
        kernel = make_weighted_transition_moment_kernel(
            particle_count=int(weights.shape[0]),
            state_dim=int(means.shape[1]),
            config=config,
            jit_compile=jit_compile,
        )
    result = kernel(weights, means, covariances)
    require_valid_moment_map(result)
    coordinate_map = AffineCoordinateMap(
        offset=result["predicted_mean"], matrix=result["predicted_cholesky"]
    )
    return {
        **result,
        "coordinate_map": coordinate_map,
        "route_id": ROUTE_ID,
        "route_classification": ROUTE_CLASSIFICATION,
        "lagged": True,
    }


def affine_forward_inverse_residual(
    coordinate_map: AffineCoordinateMap,
    reference_points: tf.Tensor,
) -> Mapping[str, tf.Tensor]:
    """Return round-trip residuals for fixed reference points."""

    reference = tf.convert_to_tensor(reference_points, DTYPE)
    physical, _ = coordinate_map.forward(reference)
    recovered, _ = coordinate_map.inverse(physical)
    forward_inverse = tf.reduce_max(tf.abs(recovered - reference))
    return {
        "forward_inverse_max_abs": forward_inverse,
        "physical_finite": tf.reduce_all(tf.math.is_finite(physical)),
        "reference_finite": tf.reduce_all(tf.math.is_finite(recovered)),
    }


def weighted_transition_moment_tangent(
    weights: tf.Tensor,
    weight_tangent: tf.Tensor,
    conditional_means: tf.Tensor,
    mean_tangent: tf.Tensor,
    conditional_covariances: tf.Tensor,
    covariance_tangent: tf.Tensor,
) -> Mapping[str, tf.Tensor]:
    """Differentiate the finite moment contraction analytically.

    This helper is a value/tangent diagnostic.  It does not choose ranks,
    maps, or samples and therefore does not by itself define an adaptive
    total-gradient route.
    """

    weights = tf.convert_to_tensor(weights, DTYPE)
    weight_tangent = tf.convert_to_tensor(weight_tangent, DTYPE)
    means = tf.convert_to_tensor(conditional_means, DTYPE)
    mean_tangent = tf.convert_to_tensor(mean_tangent, DTYPE)
    covariances = _symmetrize(tf.convert_to_tensor(conditional_covariances, DTYPE))
    covariance_tangent = _symmetrize(tf.convert_to_tensor(covariance_tangent, DTYPE))
    if weights.shape.rank != 1 or weight_tangent.shape != weights.shape:
        raise ValueError("weights and weight_tangent must have shape [N]")
    if means.shape.rank != 2 or mean_tangent.shape != means.shape:
        raise ValueError("means and mean_tangent must have shape [N,D]")
    if covariances.shape.rank != 3 or covariance_tangent.shape != covariances.shape:
        raise ValueError("covariances and covariance_tangent must have shape [N,D,D]")
    predicted_mean = tf.einsum("n,nd->d", weights, means)
    predicted_mean_tangent = tf.einsum(
        "n,nd->d", weight_tangent, means
    ) + tf.einsum("n,nd->d", weights, mean_tangent)
    centered = means - predicted_mean[tf.newaxis, :]
    centered_tangent = mean_tangent - predicted_mean_tangent[tf.newaxis, :]
    component = covariances + tf.einsum("ni,nj->nij", centered, centered)
    component_tangent = covariance_tangent + tf.einsum(
        "ni,nj->nij", centered_tangent, centered
    ) + tf.einsum("ni,nj->nij", centered, centered_tangent)
    predicted_covariance = _symmetrize(tf.einsum("n,nij->ij", weights, component))
    predicted_covariance_tangent = _symmetrize(
        tf.einsum("n,nij->ij", weight_tangent, component)
        + tf.einsum("n,nij->ij", weights, component_tangent)
    )
    return {
        "predicted_mean": predicted_mean,
        "predicted_mean_tangent": predicted_mean_tangent,
        "predicted_covariance": predicted_covariance,
        "predicted_covariance_tangent": predicted_covariance_tangent,
    }


def cholesky_tangent(
    covariance: tf.Tensor, covariance_tangent: tf.Tensor
) -> Mapping[str, tf.Tensor]:
    """Return the analytic lower-Cholesky tangent for an SPD covariance."""

    covariance = _symmetrize(tf.convert_to_tensor(covariance, DTYPE))
    covariance_tangent = _symmetrize(tf.convert_to_tensor(covariance_tangent, DTYPE))
    if covariance.shape.rank != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be square")
    if covariance_tangent.shape != covariance.shape:
        raise ValueError("covariance_tangent shape mismatch")
    chol = tf.linalg.cholesky(covariance)
    left = tf.linalg.triangular_solve(chol, covariance_tangent, lower=True)
    scaled = tf.transpose(
        tf.linalg.triangular_solve(chol, tf.transpose(left), lower=True)
    )
    lower = tf.linalg.band_part(scaled, -1, 0)
    diagonal = tf.linalg.diag_part(scaled)
    phi = lower - tf.linalg.diag(tf.linalg.diag_part(lower))
    phi = phi + 0.5 * tf.linalg.diag(diagonal)
    tangent = tf.matmul(chol, phi)
    return {
        "cholesky": chol,
        "cholesky_tangent": tangent,
        "reconstruction_tangent": tf.matmul(tangent, chol, transpose_b=True)
        + tf.matmul(chol, tangent, transpose_b=True),
    }


__all__ = [
    "DTYPE",
    "ROUTE_ID",
    "ROUTE_CLASSIFICATION",
    "MomentMapConfig",
    "make_weighted_transition_moment_kernel",
    "require_valid_moment_map",
    "build_lagged_moment_map",
    "affine_forward_inverse_residual",
    "weighted_transition_moment_tangent",
    "cholesky_tangent",
]
