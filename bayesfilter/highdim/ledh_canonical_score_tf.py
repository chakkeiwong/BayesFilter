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

from dataclasses import dataclass
from typing import Callable, Mapping

import tensorflow as tf

from bayesfilter.highdim.ledh_canonical_score_stages_tf import (
    _gaussian_log_density_and_tangent,
    ukf_predict_with_parameter_tangent,
    ukf_update_with_parameter_tangent,
)

Tensor = tf.Tensor

ANCESTRY_POLICIES = (
    "existing_one_to_one",
    "hilbert_inverse_cdf",
    "hilbert_permutation_one_to_one",
)
STATE_MAP_POLICIES = ("adaptive_empirical", "fixed_supplied")


def _sqmc_ancestor_indices(
    states: Tensor,
    incoming_log_weights: Tensor,
    ancestor_uniforms: Tensor,
    *,
    ancestry_policy: str,
    state_map_location: Tensor,
    state_map_scale: Tensor,
    hilbert_bits: int,
    state_map_policy: str,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Delegate realized fixed-index ancestry to the established SQMC helper."""
    from bayesfilter.highdim.ledh_pfpf_genut_initial_rqmc_tf import (
        _transition_ancestors,
    )

    result = _transition_ancestors(
        states,
        tf.nn.softmax(incoming_log_weights),
        ancestor_uniforms,
        ancestry_policy=ancestry_policy,
        state_map_location=state_map_location,
        state_map_scale=state_map_scale,
        hilbert_bits=hilbert_bits,
        state_map_policy=state_map_policy,
    )
    ancestry_valid = tf.constant(True)
    if ancestry_policy == "hilbert_permutation_one_to_one":
        ancestry_valid = (
            result["equal_weight_valid"]
            & result["ancestry_permutation_valid"]
        )
    return (
        result["selected_row_identities"],
        result["hilbert_ties"],
        result["state_map_saturation"],
        ancestry_valid,
    )


@dataclass(frozen=True)
class NonlinearScoreModel:
    transition_mean_fn: Callable[[Tensor, Tensor], Tensor]
    transition_mean_tangent_fn: Callable[[Tensor, Tensor, Tensor], Tensor]
    observation_fn: Callable[[Tensor], Tensor]
    observation_jacobian_fn: Callable[[Tensor], Tensor]
    observation_tangent_fn: Callable[[Tensor, Tensor], Tensor]
    process_covariance: Tensor
    observation_covariance: Tensor
    # Total tangent of H_theta(x_theta).  None is valid only when the
    # observation Jacobian is constant in both theta and x along the executed
    # direction.  The observation-value tangent alone cannot recover dH.
    observation_jacobian_tangent_fn: Callable[[Tensor, Tensor], Tensor] | None = None
    # Optional NON-GAUSSIAN observation density for the WEIGHT/VALUE path
    # (the Gaussian observation_fn/covariance above remain the FLOW's
    # proposal-design inputs, corrected by the PF-PF identity). When set,
    # the tangent callback must return the TOTAL parameter tangent
    # (closure carries the score direction, same convention as the
    # transition tangent). Added 2026-08-23 after the KSC/generalized-SV
    # fidelity defects: heteroskedastic and mixture observation families.
    observation_log_density_fn: Callable[[Tensor, Tensor, Tensor], Tensor] | None = None
    observation_log_density_tangent_fn: (
        Callable[[Tensor, Tensor, Tensor, Tensor], Tensor] | None
    ) = None
    # Q(theta)/R(theta) support (Q1.3): direction-tangents of the noise
    # covariances (closure carries the score direction; None => constant).
    process_covariance_tangent_fn: Callable[[Tensor], Tensor] | None = None
    observation_covariance_tangent_fn: Callable[[Tensor], Tensor] | None = None
    transition_log_density_fn: Callable[[Tensor, Tensor, Tensor], Tensor] | None = None
    transition_log_density_tangent_fn: (
        Callable[[Tensor, Tensor, Tensor, Tensor, Tensor], Tensor] | None
    ) = None


def _value_and_analytical_score_impl(
    model: NonlinearScoreModel,
    theta: Tensor,
    initial_states: Tensor,
    initial_covariances: Tensor,
    noises: Tensor,
    observations: Tensor,
    *,
    flow_substeps: int = 24,
    with_score: bool,
    return_trace: bool = False,
    observation_factor_override: Callable[
        [int, Tensor, Tensor, Tensor, Tensor, Tensor], tuple[Tensor, Tensor]
    ]
    | None = None,
    post_reset_transform: Callable[
        [int, Tensor, Tensor, Tensor, Tensor],
        tuple[
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Mapping[str, Tensor],
        ],
    ]
    | None = None,
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
    ancestry_policy: str = "existing_one_to_one",
    process_ancestor_uniforms: Tensor | None = None,
    state_map_location: Tensor | None = None,
    state_map_scale: Tensor | None = None,
    hilbert_bits: int = 12,
    state_map_policy: str = "adaptive_empirical",
) -> (
    tuple[Tensor, Tensor | None]
    | tuple[Tensor, Tensor | None, tuple[dict[str, Tensor | int], ...]]
):
    """Shared T-step LEDH analytical executor (single parameter direction).

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

    ``observation_factor_override`` is repository-internal composition: it
    receives ``(time_index, points, d_points, observation, base_log,
    d_base_log)`` and returns the component log factors and their total
    tangents.  The registered canonical wrapper never supplies it.

    ``return_trace=True`` is an opt-in diagnostic surface.  It returns a third
    value containing the actual per-step clouds, weights, tangents, observation,
    and post-reset state produced by this same finite program.  The trace is
    not part of the default return contract and does not feed back into the
    value or score computation.
    """

    dtype = initial_states.dtype
    if return_trace and annealed_stages != 1:
        raise ValueError(
            "return_trace currently supports only annealed_stages=1; "
            "the annealed telescope needs a separate prior-observation factorization"
        )
    if observation_factor_override is not None and annealed_stages != 1:
        raise ValueError(
            "an observation-factor override currently supports only annealed_stages=1"
        )
    if post_reset_transform is not None and reset_policy != "contract_e":
        raise ValueError("a post-reset transform requires reset_policy='contract_e'")
    if post_reset_transform is not None and annealed_stages != 1:
        raise ValueError("a post-reset transform currently requires annealed_stages=1")
    horizon = int(observations.shape[0])
    dim = int(initial_states.shape[1])
    particle_count = int(initial_states.shape[0])
    count = tf.shape(initial_states)[0]
    obs_dim = int(observations.shape[1])
    if ancestry_policy not in ANCESTRY_POLICIES:
        raise ValueError(f"unsupported ancestry policy: {ancestry_policy}")
    if state_map_policy not in STATE_MAP_POLICIES:
        raise ValueError(f"unsupported state-map policy: {state_map_policy}")
    if ancestry_policy != "existing_one_to_one" and reset_policy != "contract_e":
        raise ValueError("SQMC ancestry requires reset_policy='contract_e'")
    if process_ancestor_uniforms is None:
        process_ancestor_uniforms = tf.zeros([horizon, particle_count], dtype)
    process_ancestor_uniforms = tf.ensure_shape(
        tf.cast(process_ancestor_uniforms, dtype), [horizon, particle_count]
    )
    if state_map_location is None:
        state_map_location = tf.zeros([dim], dtype)
    if state_map_scale is None:
        state_map_scale = tf.ones([dim], dtype)
    state_map_location = tf.ensure_shape(
        tf.cast(state_map_location, dtype), [dim]
    )
    state_map_scale = tf.ensure_shape(tf.cast(state_map_scale, dtype), [dim])
    if state_map_policy == "fixed_supplied":
        tf.debugging.assert_positive(state_map_scale)
    eye = tf.eye(dim, dtype=dtype)
    process_chol = tf.linalg.cholesky(model.process_covariance)
    obs_chol = tf.linalg.cholesky(model.observation_covariance)
    r_inv = tf.linalg.cholesky_solve(obs_chol, tf.eye(obs_dim, dtype=dtype))

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
        _cholesky_forward_diff_local(process_chol, d_q) if d_q is not None else None
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
    uniform_log_weights = -tf.math.log(tf.cast(count, dtype)) * tf.ones([count], dtype)
    incoming_log_weights = uniform_log_weights
    d_incoming_log_weights = tf.zeros_like(uniform_log_weights)
    total = tf.zeros([], dtype)
    d_total = tf.zeros([], dtype)
    program_valid = tf.constant(True)
    trace = [] if return_trace else None

    def mean_fn(points):
        return model.transition_mean_fn(theta, points)

    def mean_tangent_fn(points, d_points):
        return model.transition_mean_tangent_fn(theta, points, d_points)

    for time_index in range(horizon):
        (
            ancestor_indices,
            hilbert_ties,
            state_map_saturation,
            ancestry_valid,
        ) = _sqmc_ancestor_indices(
            states,
            incoming_log_weights,
            process_ancestor_uniforms[time_index],
            ancestry_policy=ancestry_policy,
            state_map_location=state_map_location,
            state_map_scale=state_map_scale,
            hilbert_bits=hilbert_bits,
            state_map_policy=state_map_policy,
        )
        program_valid = program_valid & ancestry_valid
        states = tf.gather(states, ancestor_indices)
        d_states = tf.gather(d_states, ancestor_indices)
        covariances = tf.gather(covariances, ancestor_indices)
        d_covariances = tf.gather(d_covariances, ancestor_indices)
        incoming_log_weights = tf.gather(incoming_log_weights, ancestor_indices)
        d_incoming_log_weights = tf.gather(
            d_incoming_log_weights, ancestor_indices
        )

        observation = observations[time_index]
        noise = noises[time_index]
        step_incoming_log_weights = incoming_log_weights
        d_step_incoming_log_weights = d_incoming_log_weights

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
            d_pre_flow = d_pre_flow + tf.einsum("ij,nj->ni", d_process_chol, noise)

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
                points,
                d_points,
                means,
                d_means,
                process_chol,
                d_covariance=d_q,
            )

        def _observation_density(points, d_points):
            if model.observation_log_density_fn is not None:
                base_result = (
                    model.observation_log_density_fn(theta, points, observation),
                    model.observation_log_density_tangent_fn(
                        theta, points, observation, d_points
                    ),
                )
            else:
                observed = model.observation_fn(points)
                d_observed = model.observation_tangent_fn(points, d_points)
                obs_target = tf.broadcast_to(observation[None, :], tf.shape(observed))
                base_result = _gaussian_log_density_and_tangent(
                    obs_target,
                    None,
                    observed,
                    d_observed,
                    obs_chol,
                    d_covariance=d_r,
                )
            if observation_factor_override is None:
                return base_result
            return observation_factor_override(
                time_index,
                points,
                d_points,
                observation,
                base_result[0],
                base_result[1],
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
                prev_fraction = tf.cast((stage - 1) / annealed_stages, dtype)
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
                positions = (offset + tf.cast(tf.range(count), dtype)) / tf.cast(
                    count, dtype
                )
                cumulative = tf.cumsum(stage_soft)
                idx = tf.minimum(tf.searchsorted(cumulative, positions), count - 1)
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
                prev_obs, d_prev_obs = _observation_density(current, d_current)
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
            children, d_children, log_det, d_log_det = _flow_substeps_with_tangent(
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
            transition_log, d_transition_log = _transition_density(
                children, d_children, anchors, d_anchors
            )
            observation_log, d_observation_log = _observation_density(
                children, d_children
            )
            proposal_log, d_proposal_log = _transition_density(
                pre_flow, d_pre_flow, anchors, d_anchors
            )
            weights_log = step_incoming_log_weights
            prior_observation_logits = (
                weights_log + transition_log + log_det - proposal_log
            )
            d_prior_observation_logits = (
                d_step_incoming_log_weights
                + d_transition_log
                + d_log_det
                - d_proposal_log
            )
            prior_observation_normalizer = tf.reduce_logsumexp(prior_observation_logits)
            prior_observation_weights = tf.exp(
                prior_observation_logits - prior_observation_normalizer
            )
            d_prior_observation_normalizer = tf.reduce_sum(
                prior_observation_weights * d_prior_observation_logits
            )
            d_prior_observation_weights = prior_observation_weights * (
                d_prior_observation_logits - d_prior_observation_normalizer
            )
            logits = prior_observation_logits + observation_log
            d_logits = d_prior_observation_logits + d_observation_log
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

        step_weights = softmax
        d_step_weights = softmax * (d_logits - tf.reduce_sum(softmax * d_logits))
        reset_transport = tf.eye(count, dtype=dtype)
        d_reset_transport = tf.zeros_like(reset_transport)
        post_reset_record: Mapping[str, Tensor] = {}
        higher_moment_record: Mapping[str, Tensor] = {
            "higher_moment_valid": tf.constant(True),
            "higher_moment_pairwise_configured": tf.constant(False),
            "higher_moment_pairwise_target_mask": tf.zeros([dim, dim], dtype),
            "higher_moment_pairwise_co_skew_residual": tf.zeros([dim, dim], dtype),
            "higher_moment_pairwise_co_kurtosis_residual": tf.zeros([dim, dim], dtype),
            "higher_moment_maximum_pairwise_pre_cap_particle_rms": tf.zeros([], dtype),
            "higher_moment_maximum_pairwise_post_cap_particle_rms": tf.zeros([], dtype),
            "higher_moment_minimum_pairwise_particle_cap_scale": tf.ones([], dtype),
            "higher_moment_maximum_coordinatewise_pre_cap_absolute": tf.zeros(
                [], dtype
            ),
            "higher_moment_maximum_coordinatewise_post_cap_absolute": tf.zeros(
                [], dtype
            ),
            "higher_moment_mean_coordinatewise_cap_displacement": tf.zeros([], dtype),
            "higher_moment_fraction_coordinatewise_cap_active": tf.zeros([], dtype),
            "higher_moment_minimum_coordinatewise_cap_derivative": tf.ones([], dtype),
        }
        if reset_policy == "contract_e":
            from bayesfilter.highdim.ledh_canonical_reset_score_tf import (
                sinkhorn_contract_e_reset_triple_with_tangent,
            )

            (
                reset_states,
                d_reset_states,
                reset_covariances,
                d_reset_covariances,
                reset_transport,
                d_reset_transport,
            ) = sinkhorn_contract_e_reset_triple_with_tangent(
                children,
                d_children,
                post_covs,
                d_post_covs,
                step_weights,
                d_step_weights,
                reset_design,
                epsilon=reset_epsilon,
                sinkhorn_steps=reset_sinkhorn_steps,
                balance_steps=reset_balance_steps,
                ridge=reset_ridge,
            )
            reset_row_error = tf.reduce_max(
                tf.abs(tf.reduce_sum(reset_transport, axis=1) - 1.0)
            )
            reset_column_residual = (
                tf.reduce_mean(reset_transport, axis=0) - step_weights
            )
            reset_column_tv_error = 0.5 * tf.reduce_sum(
                tf.abs(reset_column_residual)
            )
            d_reset_column_residual = (
                tf.reduce_mean(d_reset_transport, axis=0) - d_step_weights
            )
            reset_valid = (
                tf.reduce_all(tf.math.is_finite(reset_states))
                & tf.reduce_all(tf.math.is_finite(d_reset_states))
                & tf.reduce_all(tf.math.is_finite(reset_covariances))
                & tf.reduce_all(tf.math.is_finite(d_reset_covariances))
                & tf.reduce_all(tf.math.is_finite(reset_transport))
                & tf.reduce_all(tf.math.is_finite(d_reset_transport))
                & tf.reduce_all(tf.math.is_finite(d_reset_column_residual))
                & (reset_row_error <= tf.cast(1.0e-6, dtype))
                & (reset_column_tv_error <= tf.cast(1.0e-4, dtype))
            )
            program_valid = program_valid & reset_valid
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
                program_valid = program_valid & corrected["valid"]
                reset_states = corrected["particles"]
                d_reset_states = corrected["particles_tangent"][:, :, 0]
                higher_moment_record = {
                    "higher_moment_valid": corrected["valid"],
                    "higher_moment_pairwise_configured": tf.constant(
                        bool(pairwise_steps > 0 and dim > 1)
                    ),
                    "higher_moment_pairwise_target_mask": corrected[
                        "pairwise_target_mask"
                    ],
                    "higher_moment_pairwise_co_skew_residual": corrected[
                        "pairwise_co_skew_residual"
                    ],
                    "higher_moment_pairwise_co_kurtosis_residual": corrected[
                        "pairwise_co_kurtosis_residual"
                    ],
                    "higher_moment_maximum_pairwise_pre_cap_particle_rms": corrected[
                        "maximum_pairwise_pre_cap_particle_rms"
                    ],
                    "higher_moment_maximum_pairwise_post_cap_particle_rms": corrected[
                        "maximum_pairwise_post_cap_particle_rms"
                    ],
                    "higher_moment_minimum_pairwise_particle_cap_scale": corrected[
                        "minimum_pairwise_particle_cap_scale"
                    ],
                    "higher_moment_maximum_coordinatewise_pre_cap_absolute": corrected[
                        "maximum_coordinatewise_pre_cap_absolute"
                    ],
                    "higher_moment_maximum_coordinatewise_post_cap_absolute": corrected[
                        "maximum_coordinatewise_post_cap_absolute"
                    ],
                    "higher_moment_mean_coordinatewise_cap_displacement": corrected[
                        "mean_coordinatewise_cap_displacement"
                    ],
                    "higher_moment_fraction_coordinatewise_cap_active": corrected[
                        "fraction_coordinatewise_cap_active"
                    ],
                    "higher_moment_minimum_coordinatewise_cap_derivative": corrected[
                        "minimum_coordinatewise_cap_derivative"
                    ],
                }
            states, d_states = reset_states, d_reset_states
            covariances, d_covariances = (
                reset_covariances,
                d_reset_covariances,
            )
            incoming_log_weights = uniform_log_weights
            d_incoming_log_weights = tf.zeros_like(uniform_log_weights)
            if post_reset_transform is not None:
                (
                    states,
                    d_states,
                    covariances,
                    d_covariances,
                    incoming_log_weights,
                    d_incoming_log_weights,
                    post_reset_record,
                ) = post_reset_transform(
                    time_index,
                    states,
                    d_states,
                    covariances,
                    d_covariances,
                )
        else:
            states, d_states = children, d_children
            covariances, d_covariances = post_covs, d_post_covs
            incoming_log_weights = uniform_log_weights
            d_incoming_log_weights = tf.zeros_like(uniform_log_weights)

        if trace is not None:
            step_record = {
                "time_index": time_index,
                "ancestor_indices": ancestor_indices,
                "hilbert_tie_count": hilbert_ties,
                "state_map_saturation_rate": state_map_saturation,
                "incoming_log_weights": step_incoming_log_weights,
                "d_incoming_log_weights": d_step_incoming_log_weights,
                "pre_flow": pre_flow,
                "d_pre_flow": d_pre_flow,
                "children": children,
                "d_children": d_children,
                "posterior_weights": step_weights,
                "d_posterior_weights": d_step_weights,
                "prior_observation_weights": prior_observation_weights,
                "d_prior_observation_weights": d_prior_observation_weights,
                "prior_observation_logits": prior_observation_logits,
                "d_prior_observation_logits": d_prior_observation_logits,
                "prior_observation_log_normalizer": prior_observation_normalizer,
                "d_prior_observation_log_normalizer": d_prior_observation_normalizer,
                "observation_log_density": observation_log,
                "d_observation_log_density": d_observation_log,
                "posterior_logits": logits,
                "d_posterior_logits": d_logits,
                "observation": observation,
                "predicted_covariances": predicted_covs,
                "d_predicted_covariances": d_predicted_covs,
                "post_covariances": post_covs,
                "covariances_after_reset": covariances,
                "d_covariances_after_reset": d_covariances,
                "reset_transport": reset_transport,
                "d_reset_transport": d_reset_transport,
                "states_after_reset": states,
                "d_states_after_reset": d_states,
                "outgoing_log_weights": incoming_log_weights,
                "d_outgoing_log_weights": d_incoming_log_weights,
            }
            step_record.update(higher_moment_record)
            step_record.update(post_reset_record)
            trace.append(step_record)

    program_valid = (
        program_valid
        & tf.math.is_finite(total)
        & tf.math.is_finite(d_total)
        & tf.reduce_all(tf.math.is_finite(states))
        & tf.reduce_all(tf.math.is_finite(covariances))
    )
    invalid = tf.cast(float("nan"), dtype)
    total = tf.where(program_valid, total, invalid)
    d_total = tf.where(program_valid, d_total, invalid)
    score = d_total[None] if with_score else None
    if trace is not None:
        return total, score, tuple(trace)
    if with_score:
        return total, d_total[None]
    return total, None


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
    return_trace: bool = False,
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
    ancestry_policy: str = "existing_one_to_one",
    process_ancestor_uniforms: Tensor | None = None,
    state_map_location: Tensor | None = None,
    state_map_scale: Tensor | None = None,
    hilbert_bits: int = 12,
    state_map_policy: str = "adaptive_empirical",
) -> (
    tuple[Tensor, Tensor | None]
    | tuple[Tensor, Tensor | None, tuple[dict[str, Tensor | int], ...]]
):
    """Registered canonical atom-program value and analytical total score.

    This wrapper deliberately offers no observation-factor substitution.  It
    invokes the shared executor with the model's actual observation density,
    preserving the canonical target and its default two-value return contract.
    ``reset_policy='none'`` remains a diagnostic slice; claim-bearing calls
    must explicitly select Contract-E and the required correction settings.
    """

    return _value_and_analytical_score_impl(
        model,
        theta,
        initial_states,
        initial_covariances,
        noises,
        observations,
        flow_substeps=flow_substeps,
        with_score=with_score,
        return_trace=return_trace,
        observation_factor_override=None,
        post_reset_transform=None,
        reset_policy=reset_policy,
        reset_design=reset_design,
        reset_epsilon=reset_epsilon,
        reset_sinkhorn_steps=reset_sinkhorn_steps,
        reset_balance_steps=reset_balance_steps,
        reset_ridge=reset_ridge,
        correction_steps=correction_steps,
        correction_strength=correction_strength,
        correction_lm_damping=correction_lm_damping,
        correction_lm_scale_floor=correction_lm_scale_floor,
        correction_trust_radius=correction_trust_radius,
        pairwise_steps=pairwise_steps,
        pairwise_strength=pairwise_strength,
        pairwise_rms_cap=pairwise_rms_cap,
        coordinate_cap=coordinate_cap,
        coordinate_cap_power=coordinate_cap_power,
        annealed_stages=annealed_stages,
        annealed_seed=annealed_seed,
        ancestry_policy=ancestry_policy,
        process_ancestor_uniforms=process_ancestor_uniforms,
        state_map_location=state_map_location,
        state_map_scale=state_map_scale,
        hilbert_bits=hilbert_bits,
        state_map_policy=state_map_policy,
    )


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
        d_h_jac = (
            model.observation_jacobian_tangent_fn(auxiliary, d_auxiliary)
            if model.observation_jacobian_tangent_fn is not None
            else tf.zeros_like(h_jac)
        )
        php = tf.einsum("noi,nij,npj->nop", h_jac, predicted_covs, h_jac)
        d_php = (
            tf.einsum(
                "noi,nij,npj->nop",
                d_h_jac,
                predicted_covs,
                h_jac,
            )
            + tf.einsum(
                "noi,nij,npj->nop",
                h_jac,
                d_predicted_covs,
                h_jac,
            )
            + tf.einsum(
                "noi,nij,npj->nop",
                h_jac,
                predicted_covs,
                d_h_jac,
            )
        )
        innovation_chol = tf.linalg.cholesky(lam * php + observation_covariance[None])
        ph_t = tf.einsum("nij,noj->nio", predicted_covs, h_jac)
        d_ph_t = tf.einsum("nij,noj->nio", d_predicted_covs, h_jac) + tf.einsum(
            "nij,noj->nio", predicted_covs, d_h_jac
        )
        k_lam = tf.linalg.matrix_transpose(
            tf.linalg.cholesky_solve(innovation_chol, tf.linalg.matrix_transpose(ph_t))
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
            tf.linalg.cholesky_solve(innovation_chol, tf.linalg.matrix_transpose(k_ds))
        )
        d_k_lam = term_one - term_two
        a_matrix = -0.5 * tf.einsum("nio,noj->nij", k_lam, h_jac)
        d_a_matrix = -0.5 * (
            tf.einsum("nio,noj->nij", d_k_lam, h_jac)
            + tf.einsum("nio,noj->nij", k_lam, d_h_jac)
        )

        h_val = model.observation_fn(auxiliary)
        d_h_val = model.observation_tangent_fn(auxiliary, d_auxiliary)
        residual_e = h_val - tf.einsum("noi,ni->no", h_jac, auxiliary)
        d_residual_e = (
            d_h_val
            - tf.einsum("noi,ni->no", d_h_jac, auxiliary)
            - tf.einsum("noi,ni->no", h_jac, d_auxiliary)
        )
        z_eff = observation[None, :] - residual_e
        d_z_eff = -d_residual_e
        r_inv_z_eff = tf.einsum("op,np->no", r_inv, z_eff)
        d_r_inv_z_eff = tf.einsum("op,np->no", r_inv, d_z_eff)
        if d_r_inv is not None:
            d_r_inv_z_eff = d_r_inv_z_eff + tf.einsum("op,np->no", d_r_inv, z_eff)
        phrz = tf.einsum("nio,no->ni", ph_t, r_inv_z_eff)
        d_phrz = tf.einsum("nio,no->ni", d_ph_t, r_inv_z_eff) + tf.einsum(
            "nio,no->ni", ph_t, d_r_inv_z_eff
        )
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
        q_factor, r_factor = tf.linalg.qr(step_matrix)
        log_det += tf.reduce_sum(
            tf.math.log(tf.abs(tf.linalg.diag_part(r_factor))), axis=1
        )
        q_transpose_d_a = tf.linalg.matmul(q_factor, d_a_matrix, transpose_a=True)
        solved_d_a = tf.linalg.triangular_solve(r_factor, q_transpose_d_a, lower=False)
        d_log_det += eps * tf.linalg.trace(solved_d_a)
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
