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

from bayesfilter.highdim.ledh_canonical_score_scaffold_tf import (
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
) -> tuple[Tensor, Tensor | None]:
    """T-step canonical value and analytical score (single parameter dir).

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
        )

        # S2: anchor + pre-flow (total tangent via model callback)
        anchors = mean_fn(states)
        d_anchors = mean_tangent_fn(states, d_states)
        pre_flow = anchors + tf.einsum("ij,nj->ni", process_chol, noise)
        d_pre_flow = d_anchors

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
        transition_log, d_transition_log = _gaussian_log_density_and_tangent(
            children, d_children, anchors, d_anchors, process_chol
        )
        observed = model.observation_fn(children)
        d_observed = model.observation_tangent_fn(children, d_children)
        obs_target = tf.broadcast_to(
            observation[None, :], tf.shape(observed)
        )
        observation_log, d_observation_log = _gaussian_log_density_and_tangent(
            obs_target, None, observed, d_observed, obs_chol
        )
        proposal_log, d_proposal_log = _gaussian_log_density_and_tangent(
            pre_flow, d_pre_flow, anchors, d_anchors, process_chol
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
        )

        states, d_states = children, d_children
        covariances, d_covariances = post_covs, d_post_covs

    if with_score:
        return total, d_total[None]
    return total, None


__all__ = [
    "NonlinearScoreModel",
    "canonical_value_and_analytical_score",
]
