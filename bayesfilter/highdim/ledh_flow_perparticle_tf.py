"""Per-particle LEDH flow with dual-state pseudo-time integration (P2).

Faithful discrete Algorithm 1 flow per ch19c (eqs. auxiliary-step,
actual-step, theta-product): per-particle coefficients

  A^i(lambda) = -0.5 P^i H_i^T (lambda H_i P^i H_i^T + R)^{-1} H_i
  b^i(lambda) = (I + 2 lambda A^i)[(I + lambda A^i) P^i H_i^T R^{-1}
                (z - e_i) + A^i xbar_i]

with H_i, e_i linearized at the CURRENT AUXILIARY (zero-noise anchor)
state, P^i the per-particle UKF-predicted covariance, and xbar_i the
particle's prior mean. Both the auxiliary and the actual state advance by
the same local affine step; the forward determinant accumulates
log|det(I + eps A)| per substep.

Fidelity note (recorded 2026-08-21): the pre-existing
``batched_ledh_flow_core_tf`` in the experiments tree implements a ONE-SHOT
closed-form Gaussian conditioning map, not the pseudo-time integration of
Algorithm 1; it also shares one covariance across particles. This module
supersedes it for the canonical lane.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import tensorflow as tf

from bayesfilter.ops.legacy_fraction_tf import legacy_fraction
from bayesfilter.ops.slogdet_tf import determinant_tf

Tensor = tf.Tensor

DEFAULT_SUBSTEPS = 32
DEFAULT_JITTER = 1.0e-12


def _sym(value: Tensor) -> Tensor:
    return 0.5 * (value + tf.linalg.matrix_transpose(value))


def ledh_flow_per_particle(
    *,
    anchor_states: Tensor,
    pre_flow_states: Tensor,
    predicted_covariances: Tensor,
    observation: Tensor,
    observation_fn: Callable[[Tensor], Tensor],
    observation_jacobian_fn: Callable[[Tensor], Tensor],
    observation_covariance: Tensor,
    prior_means: Tensor | None = None,
    substeps: int = DEFAULT_SUBSTEPS,
    jitter: float = DEFAULT_JITTER,
) -> dict[str, Tensor]:
    """Run the dual-state LEDH flow for every particle.

    Shapes: states [N, d]; covariances [N, d, d]; observation [o];
    observation_fn: [M, d] -> [M, o]; observation_jacobian_fn: [M, d] ->
    [M, o, d]. Returns post-flow states, forward log-det, the total affine
    matrix of the realized per-particle map, the pre-flow proposal log
    density, and the auxiliary terminal states.
    """

    anchors = tf.convert_to_tensor(anchor_states)
    dtype = anchors.dtype
    actual = tf.convert_to_tensor(pre_flow_states, dtype)
    covariances = _sym(tf.convert_to_tensor(predicted_covariances, dtype))
    observation = tf.convert_to_tensor(observation, dtype)
    obs_cov = tf.convert_to_tensor(observation_covariance, dtype)
    if prior_means is None:
        prior_means = anchors
    prior_means = tf.convert_to_tensor(prior_means, dtype)
    dim = int(anchors.shape[1])
    obs_dim = int(observation.shape[-1])
    count = tf.shape(anchors)[0]
    eye = tf.eye(dim, dtype=dtype)
    eps = tf.cast(1.0 / substeps, dtype)

    obs_cov_chol = tf.linalg.cholesky(
        obs_cov + tf.cast(jitter, dtype) * tf.eye(obs_dim, dtype=dtype)
    )

    # Pre-flow proposal log density: N(pre_flow; prior_mean, P^i).
    prior_chol = tf.linalg.cholesky(
        covariances + tf.cast(jitter, dtype) * eye
    )
    delta = actual - prior_means
    solved = tf.linalg.triangular_solve(
        prior_chol, delta[:, :, None]
    )[:, :, 0]
    log_two_pi = tf.constant(math.log(2.0 * math.pi), dtype=dtype)
    pre_flow_log_density = -0.5 * (
        tf.reduce_sum(tf.square(solved), axis=1)
        + tf.cast(dim, dtype) * log_two_pi
    ) - tf.reduce_sum(
        tf.math.log(tf.linalg.diag_part(prior_chol)), axis=1
    )

    auxiliary = anchors
    total_matrix = tf.broadcast_to(eye, [count, dim, dim])
    total_offset = tf.zeros_like(anchors)
    forward_log_det = tf.zeros([count], dtype)

    def substep(step_index, auxiliary, actual, total_matrix, total_offset, forward_log_det):
        lam = legacy_fraction(step_index + 1, substeps, dtype)
        h_jac = observation_jacobian_fn(auxiliary)
        h_val = observation_fn(auxiliary)
        residual_e = h_val - tf.einsum("nod,nd->no", h_jac, auxiliary)

        php = tf.einsum(
            "nod,nde,nqe->noq", h_jac, covariances, h_jac
        )
        innovation = lam * php + obs_cov[None]
        innovation_chol = tf.linalg.cholesky(
            _sym(innovation)
            + tf.cast(jitter, dtype) * tf.eye(obs_dim, dtype=dtype)
        )
        ph_t = tf.einsum("nde,noe->ndo", covariances, h_jac)
        solved_h = tf.linalg.cholesky_solve(innovation_chol, h_jac)
        # A = -0.5 P H^T (lam H P H^T + R)^{-1} H : [N, d, d]
        a_matrix = -0.5 * tf.linalg.matmul(ph_t, solved_h)

        z_minus_e = observation[None, :] - residual_e
        r_inv_z = tf.linalg.cholesky_solve(
            obs_cov_chol[None] if obs_cov_chol.shape.rank == 2 else obs_cov_chol,
            z_minus_e[:, :, None],
        )[:, :, 0]
        phr_z = tf.einsum("ndo,no->nd", ph_t, r_inv_z)
        inner = (
            phr_z
            + lam * tf.einsum("nde,ne->nd", a_matrix, phr_z)
            + tf.einsum("nde,ne->nd", a_matrix, prior_means)
        )
        b_vector = inner + 2.0 * lam * tf.einsum(
            "nde,ne->nd", a_matrix, inner
        )

        step_matrix = eye[None] + eps * a_matrix
        auxiliary = (
            tf.einsum("nde,ne->nd", step_matrix, auxiliary) + eps * b_vector
        )
        actual = (
            tf.einsum("nde,ne->nd", step_matrix, actual) + eps * b_vector
        )
        total_matrix = tf.einsum("nde,nef->ndf", step_matrix, total_matrix)
        total_offset = (
            tf.einsum("nde,ne->nd", step_matrix, total_offset)
            + eps * b_vector
        )
        forward_log_det += tf.math.log(
            tf.abs(determinant_tf(step_matrix))
        )
        return step_index + 1, auxiliary, actual, total_matrix, total_offset, forward_log_det

    _, auxiliary, actual, total_matrix, total_offset, forward_log_det = tf.while_loop(
        lambda step_index, *_: step_index < substeps, substep,
        (tf.constant(0), auxiliary, actual, total_matrix, total_offset, forward_log_det),
        parallel_iterations=1, maximum_iterations=substeps)

    return {
        "post_flow_states": actual,
        "auxiliary_states": auxiliary,
        "forward_log_det": forward_log_det,
        "total_affine_matrix": total_matrix,
        "total_affine_offset": total_offset,
        "pre_flow_log_density": pre_flow_log_density,
    }


__all__ = ["ledh_flow_per_particle"]
