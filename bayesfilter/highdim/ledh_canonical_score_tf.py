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
    flow_substeps: int = 24,
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
    correction_lm_scale_floor: float = 1.0e-4,
    correction_trust_radius: float = 0.5,
    pairwise_steps: int = 0,
    pairwise_strength: float = 0.02,
    pairwise_rms_cap: float = 2.0,
    coordinate_cap: float = 0.0,
    coordinate_cap_power: int = 8,
    annealed_stages: int = 1,
    annealed_seed: int = 0,
) -> tuple[Tensor, Tensor | None]:
    """T-step canonical value and analytical score (single parameter dir).

    reset_policy="contract_e" runs the FULL per-step program (S6: Sinkhorn
    + Contract-E reset with analytical tangent; S7: optional dual-cap
    trust-region correction via the general implementation's hand-derived
    JVPs). reset_policy="none" is the historical gated slice.

    ESTIMAND WARNING (2026-08-25, fidelity ledger): with
    reset_policy="none" and annealed_stages=1, each step assumes uniform
    incoming weights but never resamples, so the returned VALUE is NOT a
    log-likelihood estimator for horizons T > 1 (measured on the frozen
    LGSSM anchor: N-independent score bias +4.2 at T=50). That slice is
    a derivative-parity/diagnostic object ONLY. Claim-bearing value or
    score cells must use the production program (contract_e reset, and
    annealed mode where the scope's calibration says so).

    annealed_stages > 1 selects the within-step annealed telescope (Q1.2):
    tempered flow stages (P/k, R*k) with systematic resampling between
    stages and the SMC normalizer telescope as the step increment; stage
    tangents are analytical, and realized resampling indices are held
    fixed in the tangent (the same convention the oracle differentiates).
    The score-lane annealed mode uses the uncapped flow prior; the value
    lane's eigenvalue cap (`flow_prior_cap`) is an efficiency lever not
    wired here.

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
    eps = tf.constant(1.0 / flow_substeps, dtype=dtype)
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

        # Per-step density evaluators shared by S4 and the annealed
        # telescope (model callback path or Gaussian fallback).
        def _transition_density(points, d_points, means, d_means):
            if model.transition_log_density_fn is not None:
                return (
                    model.transition_log_density_fn(theta, points, means),
                    model.transition_log_density_tangent_fn(
                        theta, points, means, d_points, d_means
                    ),
                )
            return _gaussian_log_density_and_tangent(
                points, d_points, means, d_means, process_chol
            )

        def _observation_density(points, d_points):
            if model.observation_log_density_fn is not None:
                return (
                    model.observation_log_density_fn(
                        theta, points, observation
                    ),
                    model.observation_log_density_tangent_fn(
                        theta, points, observation, d_points
                    ),
                )
            observed = model.observation_fn(points)
            d_observed = model.observation_tangent_fn(points, d_points)
            obs_target = tf.broadcast_to(
                observation[None, :], tf.shape(observed)
            )
            return _gaussian_log_density_and_tangent(
                obs_target, None, observed, d_observed, obs_chol
            )

        if annealed_stages > 1:
            # Q1.2: within-step annealed telescope with analytical stage
            # tangents. Structure mirrors the value lane: tempered flow
            # (P/k, R*k) per stage, systematic resampling of the ancestry
            # triple between stages, increment = SMC normalizer telescope.
            # Tangent convention: realized resampling indices are FIXED
            # (piecewise-constant in theta almost everywhere) — the same
            # function the forward-autodiff oracle differentiates, since
            # index selection passes through non-differentiable searches.
            k_f = tf.cast(annealed_stages, dtype)
            current, d_current = pre_flow, d_pre_flow
            stage_anchors, d_stage_anchors = anchors, d_anchors
            stage_pm, d_stage_pm = predicted_means, d_predicted_means
            stage_pc, d_stage_pc = predicted_covs, d_predicted_covs
            prev_trans, d_prev_trans = _transition_density(
                current, d_current, stage_anchors, d_stage_anchors
            )
            prev_obs, d_prev_obs = _observation_density(current, d_current)
            for stage in range(1, annealed_stages + 1):
                moved, d_moved, stage_log_det, d_stage_log_det = (
                    _flow_substeps_with_tangent(
                        model,
                        current,
                        d_current,
                        stage_anchors,
                        d_stage_anchors,
                        stage_pc / k_f,
                        d_stage_pc / k_f,
                        observation,
                        model.observation_covariance * k_f,
                        None if d_r is None else d_r * k_f,
                        r_inv / k_f,
                        None if d_r_inv is None else d_r_inv / k_f,
                        substeps=flow_substeps,
                        eye=eye,
                    )
                )
                new_trans, d_new_trans = _transition_density(
                    moved, d_moved, stage_anchors, d_stage_anchors
                )
                new_obs, d_new_obs = _observation_density(moved, d_moved)
                fraction = tf.cast(stage / annealed_stages, dtype)
                prev_fraction = tf.cast(
                    (stage - 1) / annealed_stages, dtype
                )
                stage_logits = (
                    new_trans
                    + fraction * new_obs
                    + stage_log_det
                    - prev_trans
                    - prev_fraction * prev_obs
                )
                d_stage_logits = (
                    d_new_trans
                    + fraction * d_new_obs
                    + d_stage_log_det
                    - d_prev_trans
                    - prev_fraction * d_prev_obs
                )
                stage_norm = tf.reduce_logsumexp(stage_logits)
                stage_soft = tf.exp(stage_logits - stage_norm)
                total += stage_norm - tf.math.log(tf.cast(count, dtype))
                d_total += tf.reduce_sum(stage_soft * d_stage_logits)
                # Systematic resampling, all-TF (backend rule): stateless
                # seed keyed on (step, stage) so the oracle and analytic
                # calls realize the SAME indices.
                offset = tf.random.stateless_uniform(
                    [],
                    seed=tf.constant(
                        [
                            annealed_seed,
                            time_index * annealed_stages + stage,
                        ],
                        tf.int32,
                    ),
                    dtype=dtype,
                )
                positions = (
                    offset + tf.cast(tf.range(count), dtype)
                ) / tf.cast(count, dtype)
                cumulative = tf.cumsum(stage_soft)
                idx = tf.minimum(
                    tf.searchsorted(cumulative, positions), count - 1
                )
                current, d_current = (
                    tf.gather(moved, idx),
                    tf.gather(d_moved, idx),
                )
                stage_anchors, d_stage_anchors = (
                    tf.gather(stage_anchors, idx),
                    tf.gather(d_stage_anchors, idx),
                )
                stage_pm, d_stage_pm = (
                    tf.gather(stage_pm, idx),
                    tf.gather(d_stage_pm, idx),
                )
                stage_pc, d_stage_pc = (
                    tf.gather(stage_pc, idx),
                    tf.gather(d_stage_pc, idx),
                )
                prev_trans, d_prev_trans = _transition_density(
                    current, d_current, stage_anchors, d_stage_anchors
                )
                prev_obs, d_prev_obs = _observation_density(
                    current, d_current
                )
            children, d_children = current, d_current
            predicted_means, d_predicted_means = stage_pm, d_stage_pm
            predicted_covs, d_predicted_covs = stage_pc, d_stage_pc
            # Post-resampling weights are uniform with zero tangent (the
            # reset branch consumes softmax / d_logits).
            softmax = tf.ones([count], dtype) / tf.cast(count, dtype)
            d_logits = tf.zeros([count], dtype)
        else:
            # S3: flow with per-particle predicted covariance AND its
            # tangent (shared helper), then S4: weight assembly (the
            # transition density tangent needs the total tangent of the
            # transition mean AT the ancestor: d_anchors).
            children, d_children, log_det, d_log_det = (
                _flow_substeps_with_tangent(
                    model,
                    pre_flow,
                    d_pre_flow,
                    anchors,
                    d_anchors,
                    predicted_covs,
                    d_predicted_covs,
                    observation,
                    model.observation_covariance,
                    d_r,
                    r_inv,
                    d_r_inv,
                    substeps=flow_substeps,
                    eye=eye,
                )
            )
            transition_log, d_transition_log = _transition_density(
                children, d_children, anchors, d_anchors
            )
            observation_log, d_observation_log = _observation_density(
                children, d_children
            )
            proposal_log, d_proposal_log = _transition_density(
                pre_flow, d_pre_flow, anchors, d_anchors
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
                d_transition_log
                + d_observation_log
                + d_log_det
                - d_proposal_log
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
                    diagonal_lm_scale_floor=correction_lm_scale_floor,
                    diagonal_trust_radius=correction_trust_radius,
                    pairwise_correction_steps=pairwise_steps,
                    pairwise_strength=pairwise_strength,
                    pairwise_floor=1.0e-5,
                    pairwise_particle_rms_cap=pairwise_rms_cap,
                    coordinatewise_standardized_cap=coordinate_cap,
                    coordinatewise_standardized_cap_power=coordinate_cap_power,
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


def _flow_substeps_with_tangent(
    model: NonlinearScoreModel,
    actual: Tensor,
    d_actual: Tensor,
    prior_means: Tensor,
    d_prior_means: Tensor,
    predicted_covs: Tensor,
    d_predicted_covs: Tensor,
    observation: Tensor,
    observation_covariance: Tensor,
    d_observation_covariance: Tensor | None,
    r_inv: Tensor,
    d_r_inv: Tensor | None,
    *,
    substeps: int,
    eye: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """S3 substep loop with analytical tangent, shared by the base path
    and the annealed telescope's tempered stages (which pass P/k, R*k and
    the correspondingly scaled tangents/inverse). H is evaluated along
    the auxiliary path; its state dependence enters through the model
    jacobian at the moving anchor. Returns (post_flow, tangent, log_det,
    d_log_det)."""

    dtype = actual.dtype
    count = tf.shape(actual)[0]
    eps = tf.constant(1.0 / substeps, dtype=dtype)
    auxiliary, d_auxiliary = prior_means, d_prior_means
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
            lam * php + observation_covariance[None]
        )
        ph_t = tf.einsum("nij,noj->nio", predicted_covs, h_jac)
        d_ph_t = tf.einsum("nij,noj->nio", d_predicted_covs, h_jac)
        k_lam = tf.linalg.matrix_transpose(
            tf.linalg.cholesky_solve(
                innovation_chol, tf.linalg.matrix_transpose(ph_t)
            )
        )
        d_s = lam * d_php
        if d_observation_covariance is not None:
            d_s = d_s + d_observation_covariance[None]
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
        a_mean = tf.einsum("nij,nj->ni", a_matrix, prior_means)
        d_a_mean = tf.einsum(
            "nij,nj->ni", d_a_matrix, prior_means
        ) + tf.einsum("nij,nj->ni", a_matrix, d_prior_means)
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
    return actual, d_actual, log_det, d_log_det


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
