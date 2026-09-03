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
    # Optional non-Gaussian observation density for the WEIGHT path
    # (mirrors the single-cloud extension 2026-08-23; the Gaussian fields
    # above remain the FLOW's proposal-design inputs). Per-point
    # signatures: fn(theta_rows, points, observation) -> [M];
    # tangent(theta_rows, points, observation, d_points, d_theta_rows).
    observation_log_density_fn: Callable[[Tensor, Tensor, Tensor], Tensor] | None = None
    observation_log_density_tangent_fn: Callable[[Tensor, Tensor, Tensor, Tensor, Tensor], Tensor] | None = None


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


def _gaussian_log_and_tangent(points, d_points, means, d_means, chol, log_norm, m, k_count, dtype):
    """Gaussian log density and tangent, vectorized over K directions.

    Args:
        points: [m, dim] primal points
        d_points: [K, m, dim] tangent points or None
        means: [m, dim] primal means
        d_means: [K, m, dim] tangent means or None
        chol: [dim, dim] or [obs_dim, obs_dim] Cholesky factor
        log_norm: scalar, precomputed k·log(2π) + 2·Σ log diag(chol)
        m: batch size
        k_count: number of directions
        dtype: tensor dtype

    Returns:
        value: [m] log density
        d_value: [K, m] tangent log density
    """
    residual = points - means
    solved = tf.linalg.triangular_solve(
        tf.broadcast_to(chol, [m, *chol.shape]), residual[:, :, None]
    )[:, :, 0]
    value = -0.5 * (tf.reduce_sum(tf.square(solved), axis=1) + log_norm)

    # Tangent: vectorized over K
    if d_points is not None or d_means is not None:
        if d_points is None:
            d_residual = -d_means
        elif d_means is None:
            d_residual = d_points
        else:
            d_residual = d_points - d_means
        # Solve for all K directions at once by reshaping
        d_residual_flat = tf.reshape(d_residual, [k_count * m, -1])
        d_solved_flat = tf.linalg.triangular_solve(
            tf.broadcast_to(chol, [k_count * m, *chol.shape]), d_residual_flat[:, :, None]
        )[:, :, 0]
        d_solved = tf.reshape(d_solved_flat, [k_count, m, -1])
        d_value = -tf.reduce_sum(solved[None, :, :] * d_solved, axis=2)
    else:
        d_value = tf.zeros([k_count, m], dtype)

    return value, d_value


def _chol_diff(chol, d_matrix):
    """Tangent of Cholesky factor w.r.t. matrix perturbation.

    Args:
        chol: [m, dim, dim] lower triangular Cholesky factor
        d_matrix: [m, dim, dim] symmetric perturbation to the matrix

    Returns:
        [m, dim, dim] lower triangular perturbation to chol
    """
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
    result = tf.linalg.matmul(chol, phi)
    return result


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
    """Fused value [B] and multi-direction score [B, K].

    theta: [B, P]; theta_directions: [B, K, P] or [B, P].
    When rank-2, promoted to [B, 1, P] and score squeezed to [B] for
    backward compatibility. Frozen inputs shared across rows.
    """

    theta = tf.convert_to_tensor(theta)
    dtype = tf.convert_to_tensor(initial_states).dtype
    theta = tf.cast(theta, dtype)
    directions_input = tf.cast(theta_directions, dtype)

    # Capture module-level helper for nested function access
    chol_diff = _chol_diff

    # Rank promotion for backward compatibility
    if directions_input.shape.rank == 2:
        directions = directions_input[:, None, :]
        squeeze_output = True
    else:
        directions = directions_input
        squeeze_output = False

    # K must be statically known for Python-level unrolling
    k_shape = directions.shape.as_list()
    if k_shape[1] is None:
        raise ValueError("K dimension (directions.shape[1]) must be statically known")
    k_count = k_shape[1]
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

    # Precompute log_norm for fixed Cholesky factors (Phase 3 item 3.2)
    process_log_norm = tf.cast(dim, dtype) * log_two_pi + 2.0 * tf.reduce_sum(
        tf.math.log(tf.linalg.diag_part(process_chol))
    )
    obs_log_norm = tf.cast(obs_dim, dtype) * log_two_pi + 2.0 * tf.reduce_sum(
        tf.math.log(tf.linalg.diag_part(obs_chol))
    )

    theta_flat = tf.reshape(
        tf.tile(theta[:, None, :], [1, n, 1]), [m, -1]
    )
    # dtheta_flat now [K, m, P] for K directions
    dtheta_flat = tf.reshape(
        tf.tile(directions[:, :, None, :], [1, 1, n, 1]), [k_count, m, -1]
    )
    states = tf.tile(initial_states, [batch, 1])
    d_states = tf.zeros([k_count, m, dim], dtype)
    covariances = tf.tile(initial_covariances, [batch, 1, 1])
    d_covariances = tf.zeros([k_count, m, dim, dim], dtype)
    total = tf.zeros([batch], dtype)
    d_total = tf.zeros([k_count, batch], dtype)

    def _step_body(t, states, d_states, covariances, d_covariances, total, d_total):
        observation = tf.gather(observations, t)
        noise = tf.tile(tf.gather(noises, t), [batch, 1])

        # --- S1 fused UKF predict with tangent
        stabilized = 0.5 * (
            covariances + tf.linalg.matrix_transpose(covariances)
        ) + tf.constant(jitter, dtype=dtype) * eye

        # Compute primal prediction once
        chol_s = tf.linalg.cholesky(tf.cast(scale, dtype) * stabilized)
        chol_s.set_shape([m, dim, dim])  # Set static shape immediately

        # Debug: verify chol_s shape
        tf.debugging.assert_equal(
            tf.shape(chol_s),
            tf.constant([m, dim, dim], tf.int32),
            message="chol_s has wrong shape"
        )

        # Force offsets to be a fresh tensor with explicit identity to prevent TF confusion
        primal_chol_transpose = tf.linalg.matrix_transpose(chol_s)
        primal_chol_transpose.set_shape([m, dim, dim])
        offsets = tf.identity(primal_chol_transpose)
        offsets.set_shape([m, dim, dim])

        # Debug: verify offsets shape
        tf.debugging.assert_equal(
            tf.shape(offsets),
            tf.constant([m, dim, dim], tf.int32),
            message="offsets has wrong shape"
        )

        # MARKER: primal points construction
        # Force explicit shapes before concat to help TF shape inference
        states_expanded = states[:, None, :]  # [m, 1, dim]
        states_expanded.set_shape([m, 1, dim])
        offsets_add = states_expanded + offsets  # [m, dim, dim]
        offsets_add.set_shape([m, dim, dim])
        offsets_sub = states_expanded - offsets  # [m, dim, dim]
        offsets_sub.set_shape([m, dim, dim])

        points = tf.concat(
            [states_expanded,
             offsets_add,
             offsets_sub],
            axis=1,
        )
        s_count = 2 * dim + 1
        flat_pts = tf.reshape(points, [m * s_count, dim])
        theta_rep = tf.reshape(
            tf.tile(theta_flat[:, None, :], [1, s_count, 1]),
            [m * s_count, -1],
        )
        pushed = tf.reshape(
            model.transition_mean_fn(theta_rep, flat_pts),
            [m, s_count, dim],
        )
        predicted_means = tf.einsum("s,msd->md", mean_w, pushed)
        centered = pushed - predicted_means[:, None, :]
        predicted_covs = 0.5 * (
            tf.einsum("s,msi,msj->mij", cov_w, centered, centered)
            + tf.einsum("s,msj,msi->mij", cov_w, centered, centered)
        ) + model.process_covariance[None]

        # Tangent: vectorized over directions using tf.vectorized_map
        def compute_tangent_prediction(k_inputs):
            """Compute tangent prediction for one direction k."""
            d_cov_k, d_state_k, dtheta_flat_k = k_inputs

            d_stab = 0.5 * (
                d_cov_k + tf.linalg.matrix_transpose(d_cov_k)
            )
            d_chol_s = chol_diff(chol_s, tf.cast(scale, dtype) * d_stab)
            d_offsets = tf.linalg.matrix_transpose(d_chol_s)

            d_state_k_expanded = d_state_k[:, None, :]  # [m, 1, dim]
            d_offsets_add = d_state_k_expanded + d_offsets  # [m, dim, dim]
            d_offsets_sub = d_state_k_expanded - d_offsets  # [m, dim, dim]

            d_points = tf.concat(
                [d_state_k_expanded, d_offsets_add, d_offsets_sub],
                axis=1,
            )
            flat_dpts = tf.reshape(d_points, [m * s_count, dim])
            dtheta_rep = tf.reshape(
                tf.tile(dtheta_flat_k[:, None, :], [1, s_count, 1]),
                [m * s_count, -1],
            )
            d_pushed = tf.reshape(
                model.transition_mean_tangent_fn(
                    theta_rep, flat_pts, flat_dpts, dtheta_rep
                ),
                [m, s_count, dim],
            )
            d_predicted_means = tf.einsum("s,msd->md", mean_w, d_pushed)
            d_centered = d_pushed - d_predicted_means[:, None, :]
            d_pc_raw = tf.einsum(
                "s,msi,msj->mij", cov_w, d_centered, centered
            ) + tf.einsum("s,msi,msj->mij", cov_w, centered, d_centered)
            d_predicted_covs = 0.5 * (
                d_pc_raw + tf.linalg.matrix_transpose(d_pc_raw)
            )
            return d_predicted_means, d_predicted_covs

        # Stack tangent results using tf.vectorized_map
        d_predicted_means, d_predicted_covs = tf.vectorized_map(
            compute_tangent_prediction,
            (d_covariances, d_states, dtheta_flat),
        )

        # --- S2 anchors + pre-flow
        anchors = model.transition_mean_fn(theta_flat, states)
        pre_flow = anchors + tf.einsum("ij,mj->mi", process_chol, noise)

        # Tangent: vectorized over directions
        def compute_tangent_anchors(k_inputs):
            d_state_k, dtheta_flat_k = k_inputs
            d_anchors = model.transition_mean_tangent_fn(
                theta_flat, states, d_state_k, dtheta_flat_k
            )
            d_pre_flow = d_anchors
            return d_anchors, d_pre_flow

        d_anchors, d_pre_flow = tf.vectorized_map(
            compute_tangent_anchors,
            (d_states, dtheta_flat),
        )

        # --- S3 fused flow with tangent (linear/affine H per model set)
        def _flow_body(s, actual, d_actual, auxiliary, d_auxiliary, log_det, d_log_det):
            lam = (tf.cast(s, dtype) + 1.0) / tf.cast(substeps, dtype)

            # Primal computation
            h_jac = model.observation_jacobian_fn(auxiliary)
            php = tf.einsum("moi,mij,mpj->mop", h_jac, predicted_covs, h_jac)
            innovation_chol = tf.linalg.cholesky(
                lam * php + model.observation_covariance[None]
            )
            ph_t = tf.einsum("mij,moj->mio", predicted_covs, h_jac)
            k_lam = tf.linalg.matrix_transpose(
                tf.linalg.cholesky_solve(
                    innovation_chol, tf.linalg.matrix_transpose(ph_t)
                )
            )
            a_matrix = -0.5 * tf.einsum("mio,moj->mij", k_lam, h_jac)
            h_val = model.observation_fn(auxiliary)
            residual_e = h_val - tf.einsum("moi,mi->mo", h_jac, auxiliary)
            z_eff = observation[None, :] - residual_e
            r_inv_z_eff = tf.einsum("op,mp->mo", r_inv, z_eff)
            phrz = tf.einsum("mio,mo->mi", ph_t, r_inv_z_eff)
            a_phrz = tf.einsum("mij,mj->mi", a_matrix, phrz)
            a_mean = tf.einsum("mij,mj->mi", a_matrix, anchors)
            inner = phrz + lam * a_phrz + a_mean
            a_inner = tf.einsum("mij,mj->mi", a_matrix, inner)
            b_vector = inner + 2.0 * lam * a_inner
            new_actual = actual + eps * (
                tf.einsum("mij,mj->mi", a_matrix, actual) + b_vector
            )
            new_aux = auxiliary + eps * (
                tf.einsum("mij,mj->mi", a_matrix, auxiliary) + b_vector
            )
            step_matrix = eye[None] + eps * a_matrix
            q_factor, r_factor = tf.linalg.qr(step_matrix)
            ldet_inc = tf.reduce_sum(
                tf.math.log(tf.abs(tf.linalg.diag_part(r_factor))),
                axis=1,
            )

            # Tangent: loop over directions
            new_d_actual_list = []
            new_d_auxiliary_list = []
            # Tangent: vectorized over directions
            def compute_tangent_flow_step(k_inputs):
                """Compute tangent flow step for one direction k."""
                d_predicted_cov_k, d_auxiliary_k, d_anchors_k, d_actual_k, dtheta_flat_k = k_inputs

                d_php = tf.einsum(
                    "moi,mij,mpj->mop", h_jac, d_predicted_cov_k, h_jac
                )
                d_ph_t = tf.einsum("mij,moj->mio", d_predicted_cov_k, h_jac)
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
                d_a_matrix = -0.5 * tf.einsum("mio,moj->mij", d_k_lam, h_jac)
                d_h_val = model.observation_tangent_fn(auxiliary, d_auxiliary_k)
                d_residual_e = d_h_val - tf.einsum(
                    "moi,mi->mo", h_jac, d_auxiliary_k
                )
                d_z_eff = -d_residual_e
                d_r_inv_z_eff = tf.einsum("op,mp->mo", r_inv, d_z_eff)
                d_phrz = tf.einsum(
                    "mio,mo->mi", d_ph_t, r_inv_z_eff
                ) + tf.einsum("mio,mo->mi", ph_t, d_r_inv_z_eff)
                d_a_phrz = tf.einsum(
                    "mij,mj->mi", d_a_matrix, phrz
                ) + tf.einsum("mij,mj->mi", a_matrix, d_phrz)
                d_a_mean = tf.einsum(
                    "mij,mj->mi", d_a_matrix, anchors
                ) + tf.einsum("mij,mj->mi", a_matrix, d_anchors_k)
                d_inner = d_phrz + lam * d_a_phrz + d_a_mean
                d_a_inner = tf.einsum(
                    "mij,mj->mi", d_a_matrix, inner
                ) + tf.einsum("mij,mj->mi", a_matrix, d_inner)
                d_b_vector = d_inner + 2.0 * lam * d_a_inner
                new_d_actual_k = d_actual_k + eps * (
                    tf.einsum("mij,mj->mi", d_a_matrix, actual)
                    + tf.einsum("mij,mj->mi", a_matrix, d_actual_k)
                    + d_b_vector
                )
                new_d_auxiliary_k = d_auxiliary_k + eps * (
                    tf.einsum("mij,mj->mi", d_a_matrix, auxiliary)
                    + tf.einsum("mij,mj->mi", a_matrix, d_auxiliary_k)
                    + d_b_vector
                )
                qt_da = tf.einsum("mji,mjk->mik", q_factor, d_a_matrix)
                solved_da = tf.linalg.triangular_solve(
                    r_factor, qt_da, lower=False
                )
                dldet_inc = eps * tf.linalg.trace(solved_da)
                return new_d_actual_k, new_d_auxiliary_k, dldet_inc

            new_d_actual, new_d_auxiliary, d_ldet_inc = tf.vectorized_map(
                compute_tangent_flow_step,
                (d_predicted_covs, d_auxiliary, d_anchors, d_actual, dtheta_flat),
            )

            return (s + 1, new_actual, new_d_actual, new_aux, new_d_auxiliary,
                    log_det + ldet_inc, d_log_det + d_ldet_inc)

        _, actual, d_actual, auxiliary, d_auxiliary, log_det, d_log_det = tf.while_loop(
            cond=lambda s, *_: s < substeps,
            body=_flow_body,
            loop_vars=(
                tf.constant(0, tf.int32),
                pre_flow,
                d_pre_flow,
                anchors,
                d_anchors,
                tf.zeros([m], dtype),
                tf.zeros([k_count, m], dtype),
            ),
            shape_invariants=(
                tf.TensorShape([]),
                tf.TensorShape([m, dim]),
                tf.TensorShape([k_count, m, dim]),
                tf.TensorShape([m, dim]),
                tf.TensorShape([k_count, m, dim]),
                tf.TensorShape([m]),
                tf.TensorShape([k_count, m]),
            ),
            maximum_iterations=substeps,
            parallel_iterations=1,
        )
        children, d_children = actual, d_actual

        # --- S4 weight assembly + S8 per-row reduction
        transition_log, d_transition_log = _gaussian_log_and_tangent(
            children, d_children, anchors, d_anchors, process_chol, process_log_norm, m, k_count, dtype
        )
        if model.observation_log_density_fn is not None:
            observation_log = model.observation_log_density_fn(
                theta_flat, children, observation
            )
            # Tangent: vectorized over directions
            def compute_tangent_obs_log(k_inputs):
                d_children_k, dtheta_flat_k = k_inputs
                return model.observation_log_density_tangent_fn(
                    theta_flat, children, observation, d_children_k, dtheta_flat_k
                )
            d_observation_log = tf.vectorized_map(
                compute_tangent_obs_log,
                (d_children, dtheta_flat),
            )
        else:
            observed = model.observation_fn(children)
            obs_target = tf.broadcast_to(
                observation[None, :], tf.shape(observed)
            )
            # Tangent: vectorized over directions
            def compute_tangent_observed(d_children_k):
                return model.observation_tangent_fn(children, d_children_k)
            d_observed = tf.vectorized_map(
                compute_tangent_observed,
                d_children,
            )
            observation_log, d_observation_log = _gaussian_log_and_tangent(
                obs_target, None, observed, d_observed, obs_chol, obs_log_norm, m, k_count, dtype
            )
        proposal_log, d_proposal_log = _gaussian_log_and_tangent(
            pre_flow, d_pre_flow, anchors, d_anchors, process_chol, process_log_norm, m, k_count, dtype
        )
        logits = tf.reshape(
            transition_log + observation_log + log_det - proposal_log,
            [batch, n],
        ) - tf.cast(tf.math.log(float(n)), dtype)
        d_logits = tf.reshape(
            d_transition_log + d_observation_log + d_log_det - d_proposal_log,
            [k_count, batch, n],
        )
        increment = tf.reduce_logsumexp(logits, axis=1)
        softmax = tf.exp(logits - increment[:, None])
        new_total = total + increment
        new_d_total = d_total + tf.reduce_sum(softmax[None, :, :] * d_logits, axis=2)

        # --- S5 fused UKF update with tangent -> next covariances
        stab_p = 0.5 * (
            predicted_covs + tf.linalg.matrix_transpose(predicted_covs)
        ) + tf.constant(jitter, dtype=dtype) * eye
        chol_p = tf.linalg.cholesky(tf.cast(scale, dtype) * stab_p)
        offs = tf.linalg.matrix_transpose(chol_p)
        upoints = tf.concat(
            [
                predicted_means[:, None, :],
                predicted_means[:, None, :] + offs,
                predicted_means[:, None, :] - offs,
            ],
            axis=1,
        )
        flat_u = tf.reshape(upoints, [m * s_count, dim])
        uobserved = tf.reshape(
            model.observation_fn(flat_u), [m, s_count, obs_dim]
        )
        uobs_means = tf.einsum("s,mso->mo", mean_w, uobserved)
        cx = upoints - predicted_means[:, None, :]
        cy = uobserved - uobs_means[:, None, :]
        s_cov = 0.5 * (
            tf.einsum("s,msi,msj->mij", cov_w, cy, cy)
            + tf.einsum("s,msj,msi->mij", cov_w, cy, cy)
        ) + model.observation_covariance[None]
        c_cov = tf.einsum("s,msi,msj->mij", cov_w, cx, cy)
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
        ksk = tf.einsum("mio,mop,mjp->mij", gain, s_cov, gain)
        post = predicted_covs - ksk
        new_covariances = 0.5 * (post + tf.linalg.matrix_transpose(post))

        # Tangent: vectorized over directions
        def compute_tangent_update(k_inputs):
            """Compute tangent UKF update for one direction k."""
            d_predicted_means_k, d_predicted_covs_k = k_inputs

            d_stab_p = 0.5 * (
                d_predicted_covs_k + tf.linalg.matrix_transpose(d_predicted_covs_k)
            )
            d_chol_p = chol_diff(chol_p, tf.cast(scale, dtype) * d_stab_p)
            d_offs = tf.linalg.matrix_transpose(d_chol_p)
            d_upoints = tf.concat(
                [
                    d_predicted_means_k[:, None, :],
                    d_predicted_means_k[:, None, :] + d_offs,
                    d_predicted_means_k[:, None, :] - d_offs,
                ],
                axis=1,
            )
            flat_du = tf.reshape(d_upoints, [m * s_count, dim])
            d_uobserved = tf.reshape(
                model.observation_tangent_fn(flat_u, flat_du),
                [m, s_count, obs_dim],
            )
            d_uobs_means = tf.einsum("s,mso->mo", mean_w, d_uobserved)
            d_cx = d_upoints - d_predicted_means_k[:, None, :]
            d_cy = d_uobserved - d_uobs_means[:, None, :]
            d_s_raw = tf.einsum(
                "s,msi,msj->mij", cov_w, d_cy, cy
            ) + tf.einsum("s,msi,msj->mij", cov_w, cy, d_cy)
            d_s_cov = 0.5 * (d_s_raw + tf.linalg.matrix_transpose(d_s_raw))
            d_c_cov = tf.einsum(
                "s,msi,msj->mij", cov_w, d_cx, cy
            ) + tf.einsum("s,msi,msj->mij", cov_w, cx, d_cy)
            residual_m = d_c_cov - tf.einsum("mio,mop->mip", gain, d_s_cov)
            d_gain = tf.linalg.matrix_transpose(
                tf.linalg.cholesky_solve(
                    s_chol, tf.linalg.matrix_transpose(residual_m)
                )
            )
            d_ksk = (
                tf.einsum("mio,mop,mjp->mij", d_gain, s_cov, gain)
                + tf.einsum("mio,mop,mjp->mij", gain, d_s_cov, gain)
                + tf.einsum("mio,mop,mjp->mij", gain, s_cov, d_gain)
            )
            d_post = d_predicted_covs_k - d_ksk
            new_d_cov = 0.5 * (
                d_post + tf.linalg.matrix_transpose(d_post)
            )
            return new_d_cov

        new_d_covariances = tf.vectorized_map(
            compute_tangent_update,
            (d_predicted_means, d_predicted_covs),
        )

        return (t + 1, children, d_children, new_covariances, new_d_covariances,
                new_total, new_d_total)

    _, states, d_states, covariances, d_covariances, total, d_total = tf.while_loop(
        cond=lambda t, *_: t < horizon,
        body=_step_body,
        loop_vars=(
            tf.constant(0, tf.int32),
            states,
            d_states,
            covariances,
            d_covariances,
            total,
            d_total,
        ),
        shape_invariants=(
            tf.TensorShape([]),
            tf.TensorShape([m, dim]),
            tf.TensorShape([k_count, m, dim]),
            tf.TensorShape([m, dim, dim]),
            tf.TensorShape([k_count, m, dim, dim]),
            tf.TensorShape([batch]),
            tf.TensorShape([k_count, batch]),
        ),
        maximum_iterations=horizon,
        parallel_iterations=1,
    )

    # d_total is [K, batch], transpose to [batch, K]
    d_total = tf.transpose(d_total, [1, 0])

    # Squeeze if rank-2 input
    if squeeze_output:
        d_total = tf.squeeze(d_total, axis=1)

    valid = tf.math.is_finite(total) & tf.reduce_all(tf.math.is_finite(d_total), axis=-1 if not squeeze_output else 0)
    nan = tf.cast(float("nan"), dtype)
    return (
        tf.where(valid, total, nan),
        tf.where(valid[:, None] if not squeeze_output else valid, d_total, nan),
        {"program_valid": valid},
    )


__all__ = ["PerPointScoreModel", "canonical_batch_fused_value_score"]
