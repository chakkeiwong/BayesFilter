"""Canonical LEDH analytical recursive score (P4 assembly).

Registered claim-bearing score entry point
(``ledh_alg1_contract.ENTRY_POINTS``). Composes the oracle-gated stage
derivatives (derivation note 2026-08-21): S1 UKF-predict tangent (Cholesky
Phi differential), S2 anchor chaining, S3 flow-map + log-det tangent, S4
PF-PF weight tangent, S5 UKF-update tangent (gain differential), S8
softmax increment accumulation — with the covariance recursion CHAINED
(each step consumes the previous step's posterior covariance and its
parameter tangent).

NO autodiff anywhere in this module (contract C-9). The autodiff oracle
lives in ``ledh_canonical_autodiff_oracle_tf`` and judges this module's
gates; it never ships.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import tensorflow as tf

from bayesfilter.highdim.ledh_canonical_score_stages_tf import (
    _gaussian_log_density_and_tangent,
    ukf_predict_with_parameter_tangent,
    ukf_update_with_parameter_tangent,
)

Tensor = tf.Tensor


@dataclass(frozen=True)
class NonlinearScoreModel:
    transition_mean_fn: Callable[[Tensor, Tensor], Tensor]
    transition_mean_tangent_fn: Callable[[Tensor, Tensor, Tensor], Tensor]
    observation_fn: Callable[[Tensor], Tensor]
    observation_jacobian_fn: Callable[[Tensor], Tensor]
    observation_tangent_fn: Callable[[Tensor, Tensor], Tensor]
    process_covariance: Tensor
    observation_covariance: Tensor
    # Optional NON-GAUSSIAN observation density for the WEIGHT/VALUE path
    # (the Gaussian observation_fn/covariance above remain the FLOW's
    # proposal-design inputs, corrected by the PF-PF identity). When set,
    # the tangent callback must return the TOTAL parameter tangent
    # (closure carries the score direction, same convention as the
    # transition tangent). Added 2026-08-23 after the KSC/generalized-SV
    # fidelity defects: heteroskedastic and mixture observation families.
    observation_log_density_fn: Callable[[Tensor, Tensor, Tensor], Tensor] | None = None
    observation_log_density_tangent_fn: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor] | None = None
    # Q(theta)/R(theta) support (Q1.3): direction-tangents of the noise
    # covariances (closure carries the score direction; None => constant).
    process_covariance_tangent_fn: Callable[[Tensor], Tensor] | None = None
    observation_covariance_tangent_fn: Callable[[Tensor], Tensor] | None = None
    transition_log_density_fn: Callable[[Tensor, Tensor, Tensor], Tensor] | None = None
    transition_log_density_tangent_fn: Callable[[Tensor, Tensor, Tensor, Tensor, Tensor], Tensor] | None = None


def canonical_value_and_analytical_score(
    model: NonlinearScoreModel,
    theta: Tensor,
    initial_states: Tensor,
    initial_covariances: Tensor,
    noises: Tensor,
    observations: Tensor,
    *,
    substeps: int,
    with_score: bool,
    reset_policy: str = "none",
    reset_design: Tensor | None = None,
    reset_epsilon: float = 2.0,
    reset_sinkhorn_steps: int = 8,
    reset_balance_steps: int = 8,
    reset_ridge: float = 1.0e-5,
    correction_steps: int = 0,
    correction_strength: float = 0.2,
    correction_lm_damping: float = 1.0e-2,
    correction_trust_radius: float = 0.5,
    pairwise_steps: int = 0,
    pairwise_strength: float = 0.02,
    pairwise_rms_cap: float = 2.0,
    coordinate_cap: float = 0.0,
) -> tuple[Tensor, Tensor | None]:
    """T-step canonical value and analytical score (single parameter dir).

    reset_policy="contract_e" runs the FULL per-step program (S6: Sinkhorn
    + Contract-E reset with analytical tangent; S7: optional dual-cap
    trust-region correction via the general implementation's hand-derived
    JVPs). reset_policy="none" is the historical gated slice.

    Multi-parameter models call this per direction; the per-direction
    tangent callbacks close over the direction (same convention as the
    repository's model tangent adapters).
    """

    dtype = initial_states.dtype
    horizon = int(observations.shape[0])
    dim = int(initial_states.shape[1])
    count = tf.shape(initial_states)[0]
    obs_dim = int(observations.shape[1])
    eye = tf.eye(dim, dtype=dtype)
    eps = tf.constant(1.0 / substeps, dtype=dtype)
    process_chol = tf.linalg.cholesky(model.process_covariance)
    obs_chol = tf.linalg.cholesky(model.observation_covariance)
    r_inv = tf.linalg.cholesky_solve(
        obs_chol, tf.eye(obs_dim, dtype=dtype)
    )

    d_q = (
        model.process_covariance_tangent_fn(theta)
        if model.process_covariance_tangent_fn is not None
        else None
    )
    d_r = (
        model.observation_covariance_tangent_fn(theta)
        if model.observation_covariance_tangent_fn is not None
        else None
    )
    d_process_chol = (
        _cholesky_forward_diff_local(process_chol, d_q)
        if d_q is not None
        else None
    )
    d_r_inv = (
        -tf.linalg.matmul(tf.linalg.matmul(r_inv, d_r), r_inv)
        if d_r is not None
        else None
    )
    states = initial_states
    d_states = tf.zeros_like(states)
    covariances = initial_covariances
    d_covariances = tf.zeros_like(covariances)
    total = tf.zeros([], dtype)
    d_total = tf.zeros([], dtype)

    def mean_fn(points):
        return model.transition_mean_fn(theta, points)

    def mean_tangent_fn(points, d_points):
        return model.transition_mean_tangent_fn(theta, points, d_points)

    for time_index in range(horizon):
        observation = observations[time_index]
        noise = noises[time_index]

        # S1: UKF predict with chained covariance tangent
        (
            predicted_means,
            predicted_covs,
            d_predicted_means,
            d_predicted_covs,
        ) = ukf_predict_with_parameter_tangent(
            states,
            covariances,
            d_states,
            d_covariances,
            mean_fn,
            mean_tangent_fn,
            model.process_covariance,
            d_process_noise_covariance=d_q,
        )

        # S2: anchor + pre-flow (total tangent via model callback)
        anchors = mean_fn(states)
        d_anchors = mean_tangent_fn(states, d_states)
        pre_flow = anchors + tf.einsum("ij,nj->ni", process_chol, noise)
        d_pre_flow = d_anchors
        if d_process_chol is not None:
            d_pre_flow = d_pre_flow + tf.einsum(
                "ij,nj->ni", d_process_chol, noise
            )

        # S3: flow with per-particle predicted covariance AND its tangent.
        # H is evaluated along the auxiliary path; its state dependence
        # enters through the model jacobian at the moving anchor.
        r_inv_z = tf.linalg.matvec(r_inv, observation)
        actual, d_actual = pre_flow, d_pre_flow
        auxiliary, d_auxiliary = anchors, d_anchors
        log_det = tf.zeros([count], dtype)
        d_log_det = tf.zeros([count], dtype)
        for step_index in range(substeps):
            lam = tf.constant((step_index + 1) / substeps, dtype=dtype)
            h_jac = model.observation_jacobian_fn(auxiliary)
            # For state-independent H (linear observation), d(H)=0; models
            # with state-dependent H supply the jacobian's tangent through
            # observation_tangent_fn acting on the auxiliary tangent. The
            # canonical six-model set all have linear/affine observation
            # maps (recorded assumption; conformance C-8 fixtures cover
            # linear H; a curvature-H model extension would add d_h_jac).
            php = tf.einsum("noi,nij,npj->nop", h_jac, predicted_covs, h_jac)
            d_php = tf.einsum(
                "noi,nij,npj->nop", h_jac, d_predicted_covs, h_jac
            )
            innovation_chol = tf.linalg.cholesky(
                lam * php + model.observation_covariance[None]
            )
            ph_t = tf.einsum("nij,noj->nio", predicted_covs, h_jac)
            d_ph_t = tf.einsum("nij,noj->nio", d_predicted_covs, h_jac)
            k_lam = tf.linalg.matrix_transpose(
                tf.linalg.cholesky_solve(
                    innovation_chol, tf.linalg.matrix_transpose(ph_t)
                )
            )
            d_s = lam * d_php
            if d_r is not None:
                d_s = d_s + d_r[None]
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
            a_matrix = -0.5 * tf.einsum("nio,noj->nij", k_lam, h_jac)
            d_a_matrix = -0.5 * tf.einsum("nio,noj->nij", d_k_lam, h_jac)

            h_val = model.observation_fn(auxiliary)
            d_h_val = model.observation_tangent_fn(auxiliary, d_auxiliary)
            residual_e = h_val - tf.einsum("noi,ni->no", h_jac, auxiliary)
            d_residual_e = d_h_val - tf.einsum(
                "noi,ni->no", h_jac, d_auxiliary
            )
            z_eff = observation[None, :] - residual_e
            d_z_eff = -d_residual_e
            r_inv_z_eff = tf.einsum("op,np->no", r_inv, z_eff)
            d_r_inv_z_eff = tf.einsum("op,np->no", r_inv, d_z_eff)
            if d_r_inv is not None:
                d_r_inv_z_eff = d_r_inv_z_eff + tf.einsum(
                    "op,np->no", d_r_inv, z_eff
                )
            phrz = tf.einsum("nio,no->ni", ph_t, r_inv_z_eff)
            d_phrz = tf.einsum("nio,no->ni", d_ph_t, r_inv_z_eff) + tf.einsum(
                "nio,no->ni", ph_t, d_r_inv_z_eff
            )
            a_phrz = tf.einsum("nij,nj->ni", a_matrix, phrz)
            d_a_phrz = tf.einsum(
                "nij,nj->ni", d_a_matrix, phrz
            ) + tf.einsum("nij,nj->ni", a_matrix, d_phrz)
            a_mean = tf.einsum("nij,nj->ni", a_matrix, anchors)
            d_a_mean = tf.einsum(
                "nij,nj->ni", d_a_matrix, anchors
            ) + tf.einsum("nij,nj->ni", a_matrix, d_anchors)
            inner = phrz + lam * a_phrz + a_mean
            d_inner = d_phrz + lam * d_a_phrz + d_a_mean
            a_inner = tf.einsum("nij,nj->ni", a_matrix, inner)
            d_a_inner = tf.einsum(
                "nij,nj->ni", d_a_matrix, inner
            ) + tf.einsum("nij,nj->ni", a_matrix, d_inner)
            b_vector = inner + 2.0 * lam * a_inner
            d_b_vector = d_inner + 2.0 * lam * d_a_inner

            new_actual = actual + eps * (
                tf.einsum("nij,nj->ni", a_matrix, actual) + b_vector
            )
            d_actual = d_actual + eps * (
                tf.einsum("nij,nj->ni", d_a_matrix, actual)
                + tf.einsum("nij,nj->ni", a_matrix, d_actual)
                + d_b_vector
            )
            new_auxiliary = auxiliary + eps * (
                tf.einsum("nij,nj->ni", a_matrix, auxiliary) + b_vector
            )
            d_auxiliary = d_auxiliary + eps * (
                tf.einsum("nij,nj->ni", d_a_matrix, auxiliary)
                + tf.einsum("nij,nj->ni", a_matrix, d_auxiliary)
                + d_b_vector
            )
            actual, auxiliary = new_actual, new_auxiliary
            step_matrix = eye[None] + eps * a_matrix
            log_det += tf.math.log(tf.abs(tf.linalg.det(step_matrix)))
            step_inv = tf.linalg.inv(step_matrix)
            d_log_det += eps * tf.linalg.trace(
                tf.einsum("nij,njk->nik", step_inv, d_a_matrix)
            )

        children, d_children = actual, d_actual

        # S4: weight assembly (transition density tangent needs the total
        # tangent of the transition mean AT the ancestor: d_anchors).
        if model.transition_log_density_fn is not None:
            transition_log = model.transition_log_density_fn(
                theta, children, anchors
            )
            d_transition_log = model.transition_log_density_tangent_fn(
                theta, children, anchors, d_children, d_anchors
            )
        else:
            transition_log, d_transition_log = (
                _gaussian_log_density_and_tangent(
                    children, d_children, anchors, d_anchors, process_chol
                )
            )
        if model.observation_log_density_fn is not None:
            observation_log = model.observation_log_density_fn(
                theta, children, observation
            )
            d_observation_log = model.observation_log_density_tangent_fn(
                theta, children, observation, d_children
            )
        else:
            observed = model.observation_fn(children)
            d_observed = model.observation_tangent_fn(children, d_children)
            obs_target = tf.broadcast_to(
                observation[None, :], tf.shape(observed)
            )
            observation_log, d_observation_log = (
                _gaussian_log_density_and_tangent(
                    obs_target, None, observed, d_observed, obs_chol
                )
            )
        if model.transition_log_density_fn is not None:
            proposal_log = model.transition_log_density_fn(
                theta, pre_flow, anchors
            )
            d_proposal_log = model.transition_log_density_tangent_fn(
                theta, pre_flow, anchors, d_pre_flow, d_anchors
            )
        else:
            proposal_log, d_proposal_log = (
                _gaussian_log_density_and_tangent(
                    pre_flow, d_pre_flow, anchors, d_anchors, process_chol
                )
            )
        weights_log = -tf.math.log(tf.cast(count, dtype)) * tf.ones(
            [count], dtype
        )
        logits = (
            weights_log
            + transition_log
            + observation_log
            + log_det
            - proposal_log
        )
        d_logits = (
            d_transition_log + d_observation_log + d_log_det - d_proposal_log
        )
        increment = tf.reduce_logsumexp(logits)
        softmax = tf.exp(logits - increment)
        total += increment
        d_total += tf.reduce_sum(softmax * d_logits)

        # S5: UKF update with chained tangent -> next step's covariances
        (
            _post_means,
            post_covs,
            _d_post_means,
            d_post_covs,
        ) = ukf_update_with_parameter_tangent(
            predicted_means,
            predicted_covs,
            d_predicted_means,
            d_predicted_covs,
            model.observation_fn,
            model.observation_tangent_fn,
            model.observation_covariance,
            observation,
            d_observation_covariance=d_r,
        )

        if reset_policy == "contract_e":
            from bayesfilter.highdim.ledh_canonical_reset_score_tf import (
                sinkhorn_contract_e_reset_with_tangent,
            )

            step_weights = softmax
            d_step_weights = softmax * (
                d_logits - tf.reduce_sum(softmax * d_logits)
            )
            reset_states, d_reset_states = (
                sinkhorn_contract_e_reset_with_tangent(
                    children,
                    d_children,
                    step_weights,
                    d_step_weights,
                    reset_design,
                    epsilon=reset_epsilon,
                    sinkhorn_steps=reset_sinkhorn_steps,
                    balance_steps=reset_balance_steps,
                    ridge=reset_ridge,
                )
            )
            if correction_steps > 0 or pairwise_steps > 0:
                from bayesfilter.highdim.higher_moment_contract_e import (
                    higher_moment_shape_jvp,
                )

                corrected = higher_moment_shape_jvp(
                    children,
                    step_weights,
                    d_children[:, :, None],
                    d_step_weights[:, None],
                    reset_states,
                    d_reset_states[:, :, None],
                    correction_steps=correction_steps,
                    strength=correction_strength,
                    floor=1.0e-5,
                    diagonal_lm_damping=correction_lm_damping,
                    diagonal_lm_scale_floor=1.0e-4,
                    diagonal_trust_radius=correction_trust_radius,
                    pairwise_correction_steps=pairwise_steps,
                    pairwise_strength=pairwise_strength,
                    pairwise_floor=1.0e-5,
                    pairwise_particle_rms_cap=pairwise_rms_cap,
                    coordinatewise_standardized_cap=coordinate_cap,
                    coordinatewise_standardized_cap_power=8,
                )
                reset_states = corrected["particles"]
                d_reset_states = corrected["particles_tangent"][:, :, 0]
            states, d_states = reset_states, d_reset_states
        else:
            states, d_states = children, d_children
        covariances, d_covariances = post_covs, d_post_covs

    if with_score:
        return total, d_total[None]
    return total, None


def _cholesky_forward_diff_local(chol: Tensor, d_matrix: Tensor) -> Tensor:
    inv_d = tf.linalg.triangular_solve(chol, d_matrix)
    inv_d_inv_t = tf.linalg.matrix_transpose(
        tf.linalg.triangular_solve(chol, tf.linalg.matrix_transpose(inv_d))
    )
    lower = tf.linalg.band_part(inv_d_inv_t, -1, 0)
    phi = lower - 0.5 * tf.linalg.diag(tf.linalg.diag_part(inv_d_inv_t))
    return tf.linalg.matmul(chol, phi)


__all__ = [
    "NonlinearScoreModel",
    "canonical_value_and_analytical_score",
]
