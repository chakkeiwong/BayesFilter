"""Optional frozen backward-information proposals for the shared LEDH executor.

Implements eqs. bg-gaussian, bg-mixture and their analytical total tangents
in ledh_backward_guidance.tex. No pilot fit, smoothing path, categorical draw,
IWSG score, model-score accuracy or canonical admission is implied.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import tensorflow as tf


def _symmetric(matrix):
    return 0.5 * (matrix + tf.linalg.matrix_transpose(matrix))


def _symmetry_valid(matrix):
    epsilon = 2.0 ** (-23 if matrix.dtype == tf.float32 else -52)
    scale = tf.maximum(tf.reduce_max(tf.abs(matrix)), tf.ones([], matrix.dtype))
    tolerance = tf.constant(32.0 * epsilon, dtype=matrix.dtype) * scale
    return tf.reduce_max(tf.abs(matrix - tf.linalg.matrix_transpose(matrix))) <= tolerance


def _checked_cholesky(matrix):
    symmetric = _symmetric(matrix)
    factor = tf.linalg.cholesky(symmetric)
    valid = tf.reduce_all(tf.math.is_finite(matrix))
    valid = tf.logical_and(valid, _symmetry_valid(matrix))
    valid = tf.logical_and(valid, tf.reduce_all(tf.math.is_finite(factor)))
    valid = tf.logical_and(
        valid, tf.reduce_all(tf.linalg.diag_part(factor) > 0)
    )
    return factor, valid


@dataclass(frozen=True)
class FrozenQuadraticBackwardGuide:
    """A fixed [T,d,d] precision and [T,d] linear-coefficient schedule."""

    precision: tf.Tensor
    linear: tf.Tensor

    @classmethod
    def from_controlled_smc(cls, quadratic, linear):
        """Adapt exp(-x'Ax-beta'x-c) to H=2A, b=-beta."""
        return cls(2.0 * tf.convert_to_tensor(quadratic), -tf.convert_to_tensor(linear))

    def validate(self, horizon, dimension, dtype):
        if self.precision.shape != (horizon, dimension, dimension):
            raise ValueError("Guide precision must have shape [T,d,d]")
        if self.linear.shape != (horizon, dimension):
            raise ValueError("Guide linear coefficient must have shape [T,d]")
        if self.precision.dtype != dtype or self.linear.dtype != dtype:
            raise ValueError("Guide and state dtypes must match")

    def evaluate(self, time, anchors, d_anchors):
        count = tf.shape(anchors)[0]
        precision = self.precision[time]
        valid = tf.reduce_all(tf.math.is_finite(precision))
        valid = tf.logical_and(valid, tf.reduce_all(tf.math.is_finite(self.linear[time])))
        valid = tf.logical_and(valid, _symmetry_valid(precision))
        valid = tf.logical_and(
            valid, tf.reduce_min(tf.linalg.eigvalsh(_symmetric(precision))) >= 0
        )
        matrices = tf.broadcast_to(precision[None], [count, *precision.shape])
        vectors = tf.broadcast_to(self.linear[time][None], tf.shape(anchors))
        return matrices, vectors, tf.zeros_like(matrices), tf.zeros_like(vectors), valid


@dataclass(frozen=True)
class FrozenMixtureBackwardGuide:
    """Frozen [T,K] logits, [T,K,d] centers and [T,K,d,d] covariances.

    A [T] boolean mask disables guidance where future information is constant.
    Coefficients are fixed; responsibilities retain their moving-anchor tangent.
    """

    logits: tf.Tensor
    centers: tf.Tensor
    covariances: tf.Tensor
    active: tf.Tensor

    def validate(self, horizon, dimension, dtype):
        if self.logits.shape.rank != 2 or self.logits.shape[0] != horizon:
            raise ValueError("Mixture logits must have shape [T,K]")
        components = self.logits.shape[1]
        if not components or self.centers.shape != (horizon, components, dimension):
            raise ValueError("Mixture centers must have shape [T,K,d]")
        if self.covariances.shape != (horizon, components, dimension, dimension):
            raise ValueError("Mixture covariances must have shape [T,K,d,d]")
        if self.active.shape != (horizon,) or self.active.dtype != tf.bool:
            raise ValueError("Mixture active mask must be boolean [T]")
        if any(item.dtype != dtype for item in (self.logits, self.centers, self.covariances)):
            raise ValueError("Guide and state dtypes must match")

    def evaluate(self, time, anchors, d_anchors):
        factor, valid = _checked_cholesky(self.covariances[time])
        components = tf.shape(factor)[0]
        dimension = anchors.shape[-1]
        precision = tf.linalg.cholesky_solve(
            factor, tf.eye(dimension, batch_shape=[components], dtype=anchors.dtype)
        )
        residual = self.centers[time][None] - anchors[:, None]
        spatial_score = tf.einsum("kij,nkj->nki", precision, residual)
        log_density = -0.5 * (
            tf.einsum("nki,nki->nk", residual, spatial_score)
            + 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(factor)), axis=-1)
            + tf.constant(dimension * math.log(2.0 * math.pi), dtype=anchors.dtype)
        )
        responsibilities = tf.nn.softmax(log_density + self.logits[time][None], axis=-1)
        d_logits = tf.einsum("nki,ni->nk", spatial_score, d_anchors)
        d_responsibilities = responsibilities * (
            d_logits - tf.reduce_sum(responsibilities * d_logits, axis=-1, keepdims=True)
        )
        precision_centers = tf.einsum("kij,kj->ki", precision, self.centers[time])
        matrices = tf.einsum("nk,kij->nij", responsibilities, precision)
        vectors = tf.einsum("nk,ki->ni", responsibilities, precision_centers)
        d_matrices = tf.einsum("nk,kij->nij", d_responsibilities, precision)
        d_vectors = tf.einsum("nk,ki->ni", d_responsibilities, precision_centers)
        valid = tf.logical_and(valid, tf.reduce_all(tf.math.is_finite(self.logits[time])))
        valid = tf.logical_and(valid, tf.reduce_all(tf.math.is_finite(self.centers[time])))
        active = self.active[time]
        return (
            tf.where(active, matrices, tf.zeros_like(matrices)),
            tf.where(active, vectors, tf.zeros_like(vectors)),
            tf.where(active, d_matrices, tf.zeros_like(d_matrices)),
            tf.where(active, d_vectors, tf.zeros_like(d_vectors)),
            valid,
        )


def guided_gaussian_draw(guide, time, anchors, d_anchors, covariance, d_covariance, noise):
    """Return U, dU, log q0(U), d log q0(U) and a finite-validity flag."""
    matrices, vectors, d_matrices, d_vectors, valid = guide.evaluate(time, anchors, d_anchors)
    dimension = anchors.shape[-1]
    identity = tf.eye(dimension, dtype=anchors.dtype)
    process_factor, valid_process = _checked_cholesky(covariance)
    precision = tf.linalg.cholesky_solve(process_factor, identity)
    covariance_tangent = tf.zeros_like(covariance) if d_covariance is None else d_covariance
    d_precision = -precision @ covariance_tangent @ precision
    guided_precision = precision[None] + matrices
    precision_factor, valid_precision = _checked_cholesky(guided_precision)
    proposal_covariance = tf.linalg.cholesky_solve(
        precision_factor, tf.broadcast_to(identity, tf.shape(guided_precision))
    )
    d_guided_precision = d_precision[None] + d_matrices
    d_proposal_covariance = -proposal_covariance @ d_guided_precision @ proposal_covariance
    natural_mean = tf.einsum("ij,nj->ni", precision, anchors) + vectors
    proposal_mean = tf.einsum("nij,nj->ni", proposal_covariance, natural_mean)
    d_natural_mean = (
        tf.einsum("ij,nj->ni", d_precision, anchors)
        + tf.einsum("ij,nj->ni", precision, d_anchors)
        + d_vectors
    )
    d_proposal_mean = tf.einsum(
        "nij,nj->ni",
        proposal_covariance,
        d_natural_mean - tf.einsum("nij,nj->ni", d_guided_precision, proposal_mean),
    )
    proposal_factor, valid_proposal = _checked_cholesky(proposal_covariance)
    intermediate = tf.linalg.triangular_solve(proposal_factor, d_proposal_covariance)
    congruence = tf.linalg.matrix_transpose(
        tf.linalg.triangular_solve(proposal_factor, tf.linalg.matrix_transpose(intermediate))
    )
    lower = tf.linalg.band_part(congruence, -1, 0)
    phi = lower - 0.5 * tf.linalg.diag(tf.linalg.diag_part(congruence))
    d_factor = proposal_factor @ phi
    points = proposal_mean + tf.einsum("nij,nj->ni", proposal_factor, noise)
    d_points = d_proposal_mean + tf.einsum("nij,nj->ni", d_factor, noise)
    residual = points - proposal_mean
    d_residual = d_points - d_proposal_mean
    precision_residual = tf.einsum("nij,nj->ni", guided_precision, residual)
    log_density = -0.5 * (
        tf.reduce_sum(residual * precision_residual, axis=-1)
        + 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(proposal_factor)), axis=-1)
        + tf.constant(dimension * math.log(2.0 * math.pi), dtype=anchors.dtype)
    )
    d_log_density = (
        -tf.reduce_sum(precision_residual * d_residual, axis=-1)
        + 0.5 * tf.einsum(
            "ni,nij,nj->n", precision_residual, d_proposal_covariance, precision_residual
        )
        - 0.5 * tf.linalg.trace(guided_precision @ d_proposal_covariance)
    )
    valid = tf.logical_and(valid, valid_process)
    valid = tf.logical_and(valid, valid_precision)
    valid = tf.logical_and(valid, valid_proposal)
    valid = tf.logical_and(valid, tf.reduce_all(tf.math.is_finite(points)))
    valid = tf.logical_and(valid, tf.reduce_all(tf.math.is_finite(d_points)))
    valid = tf.logical_and(valid, tf.reduce_all(tf.math.is_finite(log_density)))
    valid = tf.logical_and(valid, tf.reduce_all(tf.math.is_finite(d_log_density)))
    invalid = tf.cast(float("nan"), anchors.dtype)
    return (
        tf.where(valid, points, invalid),
        tf.where(valid, d_points, invalid),
        tf.where(valid, log_density, invalid),
        tf.where(valid, d_log_density, invalid),
        valid,
    )


def build_backward_guided_ledh_kernel(
    model_factory, guide, *, horizon, particle_count, state_dimension,
    observation_dimension, parameter_dimension, dtype=tf.float32,
    jit_compile=True, **options
):
    """Build one fixed-signature candidate using the shared full LEDH executor.

    model_factory(theta, direction) supplies the existing directional callbacks.
    No canonical identity or tuning artifact is issued by this factory.
    """
    from bayesfilter.highdim.ledh_canonical_score_tf import _value_and_analytical_score_impl

    if dtype not in (tf.float32, tf.float64):
        raise ValueError("Guided LEDH requires float32 or float64")
    if options.get("importance_weight_policy", "ancestor") != "ancestor":
        raise ValueError("Backward-guided proposals require ancestor importance weights")
    if options.get("annealed_stages", 1) != 1:
        raise ValueError("Backward-guided proposals require one annealing stage")
    if options.get("ancestry_policy", "existing_one_to_one") != "existing_one_to_one":
        raise ValueError("Backward-guided proposals require existing_one_to_one ancestry")
    if options.get("reset_policy") != "contract_e":
        raise ValueError("Guided LEDH requires the shared Contract-E reset")
    if options.get("correction_steps", 0) < 1 or options.get("pairwise_steps", 0) < 1:
        raise ValueError("Guided LEDH requires the existing full higher-moment correction")
    if options.get("coordinate_cap", 0.0) <= 0:
        raise ValueError("Guided LEDH requires an explicit positive coordinate cap")
    guide.validate(horizon, state_dimension, dtype)
    signature = [
        tf.TensorSpec([parameter_dimension], dtype),
        tf.TensorSpec([parameter_dimension], dtype),
        tf.TensorSpec([particle_count, state_dimension], dtype),
        tf.TensorSpec([particle_count, state_dimension, state_dimension], dtype),
        tf.TensorSpec([horizon, particle_count, state_dimension], dtype),
        tf.TensorSpec([horizon, observation_dimension], dtype),
        tf.TensorSpec([particle_count, state_dimension], dtype),
        tf.TensorSpec([particle_count, state_dimension, state_dimension], dtype),
    ]

    @tf.function(input_signature=signature, jit_compile=jit_compile, autograph=False)
    def kernel(theta, direction, states, covariances, noises, observations, d_states, d_covariances):
        return _value_and_analytical_score_impl(
            model_factory(theta, direction), theta, states, covariances, noises,
            observations, with_score=True, initial_state_tangent=d_states,
            initial_covariance_tangent=d_covariances, backward_guide=guide, **options
        )

    return kernel
