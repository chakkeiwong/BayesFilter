"""Fused batch-native canonical value/score (NeuTra-eligible lane).

Batch strategy (recorded): every per-particle operation in the canonical
program (UKF sigma propagation, flow coefficients, density evaluations,
tangent recursions) is POINTWISE over particles, so theta rows are fused
by flattening [B, N] -> [B*N] with per-point theta rows; only the
weight-normalization reductions (logsumexp, softmax) are per-row, applied
via reshape to [B, N]. This preserves the single semantic authority: the
arithmetic per point is IDENTICAL to the single-cloud lane, satisfying
the anti-fork rule with batch-size-1 parity by construction, while the
execution is one fused tensor program (no Python row loop) — eligible
under the NeuTra batch-native training rule.

Model contract: `PerPointScoreModel` callbacks receive theta of shape
[M, P] aligned with points [M, d] (broadcast-elementwise semantics).

NO autodiff (C-9): score is the analytical recursion, fused.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import math

import tensorflow as tf

Tensor = tf.Tensor


@dataclass(frozen=True)
class PerPointScoreModel:
    transition_mean_fn: Callable[[Tensor, Tensor], Tensor]
    transition_mean_tangent_fn: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]
    observation_fn: Callable[[Tensor], Tensor]
    observation_jacobian_fn: Callable[[Tensor], Tensor]
    observation_tangent_fn: Callable[[Tensor, Tensor], Tensor]
    process_covariance: Tensor
    observation_covariance: Tensor


def _unscented_weights(dim: int, dtype):
    lam = float(dim) - dim  # alpha=1, kappa=0 -> lam = 0
    count = 2 * dim + 1
    rest = 1.0 / (2.0 * dim)
    mean_w = tf.concat(
        [tf.constant([0.0], dtype=dtype), tf.constant([rest] * (count - 1), dtype=dtype)],
        axis=0,
    )
    cov_w = tf.concat(
        [tf.constant([2.0], dtype=dtype), tf.constant([rest] * (count - 1), dtype=dtype)],
        axis=0,
    )
    return mean_w, cov_w, float(dim)


def canonical_batch_fused_value_score(
    model: PerPointScoreModel,
    theta: Tensor,
    theta_directions: Tensor,
    initial_states: Tensor,
    initial_covariances: Tensor,
    noises: Tensor,
    observations: Tensor,
    *,
    substeps: int,
    jitter: float = 1.0e-12,
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    """Fused value [B] and single-direction score [B].

    theta: [B, P]; theta_directions: [B, P] (the score direction per row —
    multi-parameter scores sweep directions across calls, matching the
    analytical-score convention). Frozen inputs shared across rows.
    """

    theta = tf.convert_to_tensor(theta)
    dtype = tf.convert_to_tensor(initial_states).dtype
    theta = tf.cast(theta, dtype)
    directions = tf.cast(theta_directions, dtype)
    batch = int(theta.shape[0])
    n = int(initial_states.shape[0])
    dim = int(initial_states.shape[1])
    m = batch * n
    mean_w, cov_w, scale = _unscented_weights(dim, dtype)
    eye = tf.eye(dim, dtype=dtype)
    eps = tf.constant(1.0 / substeps, dtype=dtype)
    process_chol = tf.linalg.cholesky(model.process_covariance)
    obs_dim = int(observations.shape[1])
    obs_chol = tf.linalg.cholesky(model.observation_covariance)
    r_inv = tf.linalg.cholesky_solve(obs_chol, tf.eye(obs_dim, dtype=dtype))
    horizon = int(observations.shape[0])
    log_two_pi = tf.constant(math.log(2.0 * math.pi), dtype=dtype)

    theta_flat = tf.reshape(
        tf.tile(theta[:, None, :], [1, n, 1]), [m, -1]
    )
    dtheta_flat = tf.reshape(
        tf.tile(directions[:, None, :], [1, n, 1]), [m, -1]
    )
    states = tf.tile(initial_states, [batch, 1])
    d_states = tf.zeros_like(states)
    covariances = tf.tile(initial_covariances, [batch, 1, 1])
    d_covariances = tf.zeros_like(covariances)
    total = tf.zeros([batch], dtype)
    d_total = tf.zeros([batch], dtype)

    def gaussian_log_and_tangent(points, d_points, means, d_means, chol):
        residual = points - means
        solved = tf.linalg.triangular_solve(
            tf.broadcast_to(chol, [m, *chol.shape]), residual[:, :, None]
        )[:, :, 0]
        k = int(chol.shape[0])
        log_norm = tf.cast(k, dtype) * log_two_pi + 2.0 * tf.reduce_sum(
            tf.math.log(tf.linalg.diag_part(chol))
        )
        value = -0.5 * (tf.reduce_sum(tf.square(solved), axis=1) + log_norm)
        d_residual = (d_points if d_points is not None else 0.0) - (
            d_means if d_means is not None else 0.0
        )
        d_solved = tf.linalg.triangular_solve(
            tf.broadcast_to(chol, [m, *chol.shape]), d_residual[:, :, None]
        )[:, :, 0]
        return value, -tf.reduce_sum(solved * d_solved, axis=1)

    def chol_diff(chol, d_matrix):
        inv_d = tf.linalg.triangular_solve(chol, d_matrix)
        inv_d_inv_t = tf.linalg.matrix_transpose(
            tf.linalg.triangular_solve(
                chol, tf.linalg.matrix_transpose(inv_d)
            )
        )
        lower = tf.linalg.band_part(inv_d_inv_t, -1, 0)
        phi = lower - 0.5 * tf.linalg.diag(
            tf.linalg.diag_part(inv_d_inv_t)
        )
        return tf.linalg.matmul(chol, phi)

    for time_index in range(horizon):
        observation = observations[time_index]
        noise = tf.tile(noises[time_index], [batch, 1])

        # --- S1 fused UKF predict with tangent
        stabilized = 0.5 * (
            covariances + tf.linalg.matrix_transpose(covariances)
        ) + tf.constant(jitter, dtype=dtype) * eye
        d_stab = 0.5 * (
            d_covariances + tf.linalg.matrix_transpose(d_covariances)
        )
        chol_s = tf.linalg.cholesky(tf.cast(scale, dtype) * stabilized)
        d_chol_s = chol_diff(chol_s, tf.cast(scale, dtype) * d_stab)
        offsets = tf.linalg.matrix_transpose(chol_s)
        d_offsets = tf.linalg.matrix_transpose(d_chol_s)
        points = tf.concat(
            [states[:, None, :], states[:, None, :] + offsets, states[:, None, :] - offsets],
            axis=1,
        )
        d_points = tf.concat(
            [d_states[:, None, :], d_states[:, None, :] + d_offsets, d_states[:, None, :] - d_offsets],
            axis=1,
        )
        s_count = 2 * dim + 1
        flat_pts = tf.reshape(points, [m * s_count, dim])
        flat_dpts = tf.reshape(d_points, [m * s_count, dim])
        theta_rep = tf.reshape(
            tf.tile(theta_flat[:, None, :], [1, s_count, 1]),
            [m * s_count, -1],
        )
        dtheta_rep = tf.reshape(
            tf.tile(dtheta_flat[:, None, :], [1, s_count, 1]),
            [m * s_count, -1],
        )
        pushed = tf.reshape(
            model.transition_mean_fn(theta_rep, flat_pts),
            [m, s_count, dim],
        )
        d_pushed = tf.reshape(
            model.transition_mean_tangent_fn(
                theta_rep, flat_pts, flat_dpts, dtheta_rep
            ),
            [m, s_count, dim],
        )
        predicted_means = tf.einsum("s,msd->md", mean_w, pushed)
        d_predicted_means = tf.einsum("s,msd->md", mean_w, d_pushed)
        centered = pushed - predicted_means[:, None, :]
        d_centered = d_pushed - d_predicted_means[:, None, :]
        predicted_covs = 0.5 * (
            tf.einsum("s,msi,msj->mij", cov_w, centered, centered)
            + tf.einsum("s,msj,msi->mij", cov_w, centered, centered)
        ) + model.process_covariance[None]
        d_pc_raw = tf.einsum(
            "s,msi,msj->mij", cov_w, d_centered, centered
        ) + tf.einsum("s,msi,msj->mij", cov_w, centered, d_centered)
        d_predicted_covs = 0.5 * (
            d_pc_raw + tf.linalg.matrix_transpose(d_pc_raw)
        )

        # --- S2 anchors + pre-flow
        anchors = model.transition_mean_fn(theta_flat, states)
        d_anchors = model.transition_mean_tangent_fn(
            theta_flat, states, d_states, dtheta_flat
        )
        pre_flow = anchors + tf.einsum("ij,mj->mi", process_chol, noise)
        d_pre_flow = d_anchors

        # --- S3 fused flow with tangent (linear/affine H per model set)
        r_inv_obs = tf.linalg.matvec(r_inv, observation)
        actual, d_actual = pre_flow, d_pre_flow
        auxiliary, d_auxiliary = anchors, d_anchors
        log_det = tf.zeros([m], dtype)
        d_log_det = tf.zeros([m], dtype)
        for step_index in range(substeps):
            lam = tf.constant((step_index + 1) / substeps, dtype=dtype)
            h_jac = model.observation_jacobian_fn(auxiliary)
            php = tf.einsum("moi,mij,mpj->mop", h_jac, predicted_covs, h_jac)
            d_php = tf.einsum(
                "moi,mij,mpj->mop", h_jac, d_predicted_covs, h_jac
            )
            innovation_chol = tf.linalg.cholesky(
                lam * php + model.observation_covariance[None]
            )
            ph_t = tf.einsum("mij,moj->mio", predicted_covs, h_jac)
            d_ph_t = tf.einsum("mij,moj->mio", d_predicted_covs, h_jac)
            k_lam = tf.linalg.matrix_transpose(
                tf.linalg.cholesky_solve(
                    innovation_chol, tf.linalg.matrix_transpose(ph_t)
                )
            )
            d_s = lam * d_php
            term_one = tf.linalg.matrix_transpose(
                tf.linalg.cholesky_solve(
                    innovation_chol, tf.linalg.matrix_transpose(d_ph_t)
                )
            )
            k_ds = tf.einsum("mio,mop->mip", k_lam, d_s)
            term_two = tf.linalg.matrix_transpose(
                tf.linalg.cholesky_solve(
                    innovation_chol, tf.linalg.matrix_transpose(k_ds)
                )
            )
            d_k_lam = term_one - term_two
            a_matrix = -0.5 * tf.einsum("mio,moj->mij", k_lam, h_jac)
            d_a_matrix = -0.5 * tf.einsum("mio,moj->mij", d_k_lam, h_jac)
            h_val = model.observation_fn(auxiliary)
            d_h_val = model.observation_tangent_fn(auxiliary, d_auxiliary)
            residual_e = h_val - tf.einsum("moi,mi->mo", h_jac, auxiliary)
            d_residual_e = d_h_val - tf.einsum(
                "moi,mi->mo", h_jac, d_auxiliary
            )
            z_eff = observation[None, :] - residual_e
            d_z_eff = -d_residual_e
            r_inv_z_eff = tf.einsum("op,mp->mo", r_inv, z_eff)
            d_r_inv_z_eff = tf.einsum("op,mp->mo", r_inv, d_z_eff)
            phrz = tf.einsum("mio,mo->mi", ph_t, r_inv_z_eff)
            d_phrz = tf.einsum(
                "mio,mo->mi", d_ph_t, r_inv_z_eff
            ) + tf.einsum("mio,mo->mi", ph_t, d_r_inv_z_eff)
            a_phrz = tf.einsum("mij,mj->mi", a_matrix, phrz)
            d_a_phrz = tf.einsum(
                "mij,mj->mi", d_a_matrix, phrz
            ) + tf.einsum("mij,mj->mi", a_matrix, d_phrz)
            a_mean = tf.einsum("mij,mj->mi", a_matrix, anchors)
            d_a_mean = tf.einsum(
                "mij,mj->mi", d_a_matrix, anchors
            ) + tf.einsum("mij,mj->mi", a_matrix, d_anchors)
            inner = phrz + lam * a_phrz + a_mean
            d_inner = d_phrz + lam * d_a_phrz + d_a_mean
            a_inner = tf.einsum("mij,mj->mi", a_matrix, inner)
            d_a_inner = tf.einsum(
                "mij,mj->mi", d_a_matrix, inner
            ) + tf.einsum("mij,mj->mi", a_matrix, d_inner)
            b_vector = inner + 2.0 * lam * a_inner
            d_b_vector = d_inner + 2.0 * lam * d_a_inner
            new_actual = actual + eps * (
                tf.einsum("mij,mj->mi", a_matrix, actual) + b_vector
            )
            d_actual = d_actual + eps * (
                tf.einsum("mij,mj->mi", d_a_matrix, actual)
                + tf.einsum("mij,mj->mi", a_matrix, d_actual)
                + d_b_vector
            )
            new_aux = auxiliary + eps * (
                tf.einsum("mij,mj->mi", a_matrix, auxiliary) + b_vector
            )
            d_auxiliary = d_auxiliary + eps * (
                tf.einsum("mij,mj->mi", d_a_matrix, auxiliary)
                + tf.einsum("mij,mj->mi", a_matrix, d_auxiliary)
                + d_b_vector
            )
            actual, auxiliary = new_actual, new_aux
            step_matrix = eye[None] + eps * a_matrix
            # XLA-compatible det/inverse-trace: MatrixDeterminant and
            # MatrixInverse lack tf2xla kernels; QR (supported HLO) gives
            # log|det M| = sum log|diag R| and the trace term via
            # triangular solve: tr(M^{-1} dA) = tr(R^{-1} Q^T dA).
            q_factor, r_factor = tf.linalg.qr(step_matrix)
            log_det += tf.reduce_sum(
                tf.math.log(tf.abs(tf.linalg.diag_part(r_factor))),
                axis=1,
            )
            qt_da = tf.einsum("mji,mjk->mik", q_factor, d_a_matrix)
            solved_da = tf.linalg.triangular_solve(
                r_factor, qt_da, lower=False
            )
            d_log_det += eps * tf.linalg.trace(solved_da)
        children, d_children = actual, d_actual

        # --- S4 weight assembly + S8 per-row reduction
        transition_log, d_transition_log = gaussian_log_and_tangent(
            children, d_children, anchors, d_anchors, process_chol
        )
        observed = model.observation_fn(children)
        d_observed = model.observation_tangent_fn(children, d_children)
        obs_target = tf.broadcast_to(observation[None, :], tf.shape(observed))
        observation_log, d_observation_log = gaussian_log_and_tangent(
            obs_target, None, observed, d_observed, obs_chol
        )
        proposal_log, d_proposal_log = gaussian_log_and_tangent(
            pre_flow, d_pre_flow, anchors, d_anchors, process_chol
        )
        logits = tf.reshape(
            transition_log + observation_log + log_det - proposal_log,
            [batch, n],
        ) - tf.cast(tf.math.log(float(n)), dtype)
        d_logits = tf.reshape(
            d_transition_log + d_observation_log + d_log_det - d_proposal_log,
            [batch, n],
        )
        increment = tf.reduce_logsumexp(logits, axis=1)
        softmax = tf.exp(logits - increment[:, None])
        total += increment
        d_total += tf.reduce_sum(softmax * d_logits, axis=1)

        # --- S5 fused UKF update with tangent -> next covariances
        stab_p = 0.5 * (
            predicted_covs + tf.linalg.matrix_transpose(predicted_covs)
        ) + tf.constant(jitter, dtype=dtype) * eye
        d_stab_p = 0.5 * (
            d_predicted_covs + tf.linalg.matrix_transpose(d_predicted_covs)
        )
        chol_p = tf.linalg.cholesky(tf.cast(scale, dtype) * stab_p)
        d_chol_p = chol_diff(chol_p, tf.cast(scale, dtype) * d_stab_p)
        offs = tf.linalg.matrix_transpose(chol_p)
        d_offs = tf.linalg.matrix_transpose(d_chol_p)
        upoints = tf.concat(
            [
                predicted_means[:, None, :],
                predicted_means[:, None, :] + offs,
                predicted_means[:, None, :] - offs,
            ],
            axis=1,
        )
        d_upoints = tf.concat(
            [
                d_predicted_means[:, None, :],
                d_predicted_means[:, None, :] + d_offs,
                d_predicted_means[:, None, :] - d_offs,
            ],
            axis=1,
        )
        flat_u = tf.reshape(upoints, [m * s_count, dim])
        flat_du = tf.reshape(d_upoints, [m * s_count, dim])
        uobserved = tf.reshape(
            model.observation_fn(flat_u), [m, s_count, obs_dim]
        )
        d_uobserved = tf.reshape(
            model.observation_tangent_fn(flat_u, flat_du),
            [m, s_count, obs_dim],
        )
        uobs_means = tf.einsum("s,mso->mo", mean_w, uobserved)
        d_uobs_means = tf.einsum("s,mso->mo", mean_w, d_uobserved)
        cx = upoints - predicted_means[:, None, :]
        d_cx = d_upoints - d_predicted_means[:, None, :]
        cy = uobserved - uobs_means[:, None, :]
        d_cy = d_uobserved - d_uobs_means[:, None, :]
        s_cov = 0.5 * (
            tf.einsum("s,msi,msj->mij", cov_w, cy, cy)
            + tf.einsum("s,msj,msi->mij", cov_w, cy, cy)
        ) + model.observation_covariance[None]
        d_s_raw = tf.einsum(
            "s,msi,msj->mij", cov_w, d_cy, cy
        ) + tf.einsum("s,msi,msj->mij", cov_w, cy, d_cy)
        d_s_cov = 0.5 * (d_s_raw + tf.linalg.matrix_transpose(d_s_raw))
        c_cov = tf.einsum("s,msi,msj->mij", cov_w, cx, cy)
        d_c_cov = tf.einsum(
            "s,msi,msj->mij", cov_w, d_cx, cy
        ) + tf.einsum("s,msi,msj->mij", cov_w, cx, d_cy)
        s_chol = tf.linalg.cholesky(
            s_cov
            + tf.constant(jitter, dtype=dtype)
            * tf.eye(obs_dim, dtype=dtype)
        )
        gain = tf.linalg.matrix_transpose(
            tf.linalg.cholesky_solve(
                s_chol, tf.linalg.matrix_transpose(c_cov)
            )
        )
        residual_m = d_c_cov - tf.einsum("mio,mop->mip", gain, d_s_cov)
        d_gain = tf.linalg.matrix_transpose(
            tf.linalg.cholesky_solve(
                s_chol, tf.linalg.matrix_transpose(residual_m)
            )
        )
        ksk = tf.einsum("mio,mop,mjp->mij", gain, s_cov, gain)
        d_ksk = (
            tf.einsum("mio,mop,mjp->mij", d_gain, s_cov, gain)
            + tf.einsum("mio,mop,mjp->mij", gain, d_s_cov, gain)
            + tf.einsum("mio,mop,mjp->mij", gain, s_cov, d_gain)
        )
        post = predicted_covs - ksk
        d_post = d_predicted_covs - d_ksk
        covariances = 0.5 * (post + tf.linalg.matrix_transpose(post))
        d_covariances = 0.5 * (
            d_post + tf.linalg.matrix_transpose(d_post)
        )
        states, d_states = children, d_children

    valid = tf.math.is_finite(total) & tf.math.is_finite(d_total)
    nan = tf.cast(float("nan"), dtype)
    return (
        tf.where(valid, total, nan),
        tf.where(valid, d_total, nan),
        {"program_valid": valid},
    )


__all__ = ["PerPointScoreModel", "canonical_batch_fused_value_score"]
