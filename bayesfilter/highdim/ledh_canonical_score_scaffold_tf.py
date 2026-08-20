"""P4 scaffold: flow value + analytical parameter tangent, LGSSM slice.

Scaffold naming per rule R-D (expiry: absorbed into
``ledh_canonical_score_tf`` when all stages land; G-2 enforces).

Derivation (note S3, LGSSM slice, theta scalar scaling F = theta*F0):

  P^i(theta) = F P_prev F^T + Q            (UKF exact on linear dynamics)
  dP^i = dF P F^T + F P dF^T
  anchor a = F x_prev; da = dF x_prev
  per substep with lam fixed:
    S = lam H P H^T + R;      dS = lam H dP H^T
    K = P H^T S^{-1};         dK = dP H^T S^{-1} - P H^T S^{-1} dS S^{-1}
    A = -0.5 K H;             dA = -0.5 dK H
    zres = z - (h(a) - H a) = z  (linear H: residual e = 0 offset)
    phrz = P H^T R^{-1} z_eff; dphrz = dP H^T R^{-1} z_eff + P H^T R^{-1} dz_eff
    inner = phrz + lam A phrz + A m;  d(inner) by product rule
    b = inner + 2 lam A inner; db by product rule
    state step x <- (I + eps A) x + eps b
    dx <- (I + eps A) dx + eps dA x + eps db
  All matrix operations elementwise-analytical; no autodiff.
"""

from __future__ import annotations

import tensorflow as tf

Tensor = tf.Tensor


def flow_value_and_parameter_tangent_lgssm(
    theta: Tensor,
    f0: Tensor,
    process_covariance: Tensor,
    observation_matrix: Tensor,
    observation_covariance: Tensor,
    states: Tensor,
    covariances: Tensor,
    noise: Tensor,
    observation: Tensor,
    *,
    substeps: int,
    with_tangent: bool,
) -> tuple[Tensor, Tensor | None]:
    """Return post-flow states and (optionally) d(post_flow)/dtheta.

    theta: [1] scalar parameter scaling the transition matrix.
    """

    dtype = states.dtype
    dim = int(states.shape[1])
    count = tf.shape(states)[0]
    transition = theta[0] * f0
    d_transition = f0  # d(theta*F0)/dtheta

    process_chol = tf.linalg.cholesky(process_covariance)
    anchors = tf.einsum("ij,nj->ni", transition, states)
    d_anchors = tf.einsum("ij,nj->ni", d_transition, states)
    pre_flow = anchors + tf.einsum("ij,nj->ni", process_chol, noise)
    d_pre_flow = d_anchors

    predicted = (
        tf.einsum("ij,njk,lk->nil", transition, covariances, transition)
        + process_covariance[None]
    )
    d_predicted = tf.einsum(
        "ij,njk,lk->nil", d_transition, covariances, transition
    ) + tf.einsum("ij,njk,lk->nil", transition, covariances, d_transition)

    eye = tf.eye(dim, dtype=dtype)
    eps = tf.constant(1.0 / substeps, dtype=dtype)
    h = observation_matrix
    r = observation_covariance
    r_chol = tf.linalg.cholesky(r)
    r_inv_z = tf.linalg.cholesky_solve(r_chol, observation[:, None])[:, 0]

    actual = pre_flow
    d_actual = d_pre_flow
    prior_means = anchors
    d_prior_means = d_anchors

    for step_index in range(substeps):
        lam = tf.constant((step_index + 1) / substeps, dtype=dtype)
        php = tf.einsum("oi,nij,pj->nop", h, predicted, h)
        d_php = tf.einsum("oi,nij,pj->nop", h, d_predicted, h)
        innovation = lam * php + r[None]
        innovation_chol = tf.linalg.cholesky(innovation)
        ph_t = tf.einsum("nij,oj->nio", predicted, h)
        d_ph_t = tf.einsum("nij,oj->nio", d_predicted, h)
        # K_lam = P H^T S^{-1}
        k_lam = tf.linalg.matrix_transpose(
            tf.linalg.cholesky_solve(
                innovation_chol, tf.linalg.matrix_transpose(ph_t)
            )
        )
        d_s = lam * d_php
        # d(K_lam) = dP H^T S^{-1} - K_lam dS S^{-1}
        # implemented via cholesky_solve on the transposed systems:
        term_one = tf.linalg.matrix_transpose(
            tf.linalg.cholesky_solve(
                innovation_chol, tf.linalg.matrix_transpose(d_ph_t)
            )
        )
        k_ds = tf.einsum("nio,nop->nip", k_lam, d_s)
        term_two = tf.linalg.matrix_transpose(
            tf.linalg.cholesky_solve(
                innovation_chol, tf.linalg.matrix_transpose(k_ds)
            )
        )
        d_k_lam = term_one - term_two
        a_matrix = -0.5 * tf.einsum("nio,oj->nij", k_lam, h)
        d_a_matrix = -0.5 * tf.einsum("nio,oj->nij", d_k_lam, h)

        phrz = tf.einsum("nio,o->ni", ph_t, r_inv_z)
        d_phrz = tf.einsum("nio,o->ni", d_ph_t, r_inv_z)
        a_phrz = tf.einsum("nij,nj->ni", a_matrix, phrz)
        d_a_phrz = tf.einsum("nij,nj->ni", d_a_matrix, phrz) + tf.einsum(
            "nij,nj->ni", a_matrix, d_phrz
        )
        a_mean = tf.einsum("nij,nj->ni", a_matrix, prior_means)
        d_a_mean = tf.einsum("nij,nj->ni", d_a_matrix, prior_means) + tf.einsum(
            "nij,nj->ni", a_matrix, d_prior_means
        )
        inner = phrz + lam * a_phrz + a_mean
        d_inner = d_phrz + lam * d_a_phrz + d_a_mean
        a_inner = tf.einsum("nij,nj->ni", a_matrix, inner)
        d_a_inner = tf.einsum("nij,nj->ni", d_a_matrix, inner) + tf.einsum(
            "nij,nj->ni", a_matrix, d_inner
        )
        b_vector = inner + 2.0 * lam * a_inner
        d_b_vector = d_inner + 2.0 * lam * d_a_inner

        new_actual = (
            actual
            + eps * (tf.einsum("nij,nj->ni", a_matrix, actual) + b_vector)
        )
        if with_tangent:
            d_actual = (
                d_actual
                + eps
                * (
                    tf.einsum("nij,nj->ni", d_a_matrix, actual)
                    + tf.einsum("nij,nj->ni", a_matrix, d_actual)
                    + d_b_vector
                )
            )
        actual = new_actual

    if with_tangent:
        return actual, d_actual[..., None]
    return actual, None


__all__ = ["flow_value_and_parameter_tangent_lgssm"]
