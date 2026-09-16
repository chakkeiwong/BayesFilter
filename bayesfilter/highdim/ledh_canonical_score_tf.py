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
    initial_state_tangent: Tensor | None = None,
    initial_covariance_tangent: Tensor | None = None,
    observation_factor_override: Callable[
        [Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
        tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Mapping[str, Tensor]]
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
    moment_provider: tuple[Callable, Callable] | None = None,
    moment_schedule: Mapping[str, Tensor] | None = None,
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

    # Phase 1 while-loop conversion constraints (2026-09-14)
    if annealed_stages != 1:
        raise ValueError(
            "Phase 1 while-loop conversion requires annealed_stages=1; "
            "nested loop support deferred to future phase"
        )
    horizon = int(observations.shape[0])
    dim = int(initial_states.shape[1])
    count = tf.shape(initial_states)[0]
    obs_dim = int(observations.shape[1])
    eye = tf.eye(dim, dtype=dtype)
    process_chol = tf.linalg.cholesky(model.process_covariance)
    obs_chol = tf.linalg.cholesky(model.observation_covariance)
    r_inv = tf.linalg.cholesky_solve(obs_chol, tf.eye(obs_dim, dtype=dtype))

    d_q = (
        model.process_covariance_tangent_fn(theta)
        if model.process_covariance_tangent_fn is not None and theta is not None
        else None
    )
    d_r = (
        model.observation_covariance_tangent_fn(theta)
        if model.observation_covariance_tangent_fn is not None and theta is not None
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
    d_states = (tf.zeros_like(states) if initial_state_tangent is None else
                tf.ensure_shape(tf.convert_to_tensor(initial_state_tangent, dtype=dtype), states.shape))
    covariances = initial_covariances
    d_covariances = (tf.zeros_like(covariances) if initial_covariance_tangent is None else
                     tf.ensure_shape(tf.convert_to_tensor(initial_covariance_tangent, dtype=dtype), covariances.shape))
    uniform_log_weights = -tf.math.log(tf.cast(count, dtype)) * tf.ones([count], dtype)
    incoming_log_weights = uniform_log_weights
    d_incoming_log_weights = tf.zeros_like(uniform_log_weights)
    total = tf.zeros([], dtype)
    d_total = tf.zeros([], dtype)

    def mean_fn(points):
        return model.transition_mean_fn(theta, points)

    def mean_tangent_fn(points, d_points):
        return model.transition_mean_tangent_fn(theta, points, d_points)

    # Phase 1 while-loop: convert time loop (2026-09-14)
    # Reference: deleted implementation at 5cc59cfa~1, lines 691-714
    def time_loop_cond(t, _states, _d_states, _covs, _d_covs, _in_log_w, _d_in_log_w, _total, _d_total):
        return t < horizon

    moment_predict, moment_update = (moment_provider if moment_provider is not None else
                                    (ukf_predict_with_parameter_tangent, ukf_update_with_parameter_tangent))
    if moment_schedule is not None and moment_provider is not None:
        raise ValueError("choose a recursive provider or an independent filter schedule")

    def scheduled_moments(t, prefix):
        return tuple(tf.broadcast_to(moment_schedule[prefix + name][t],
                     [count, dim, dim] if "covariances" in name else [count, dim])
                     for name in ("means", "covariances", "d_means", "d_covariances"))

    def predict_at(t, *args, **kwargs):
        return scheduled_moments(t, "predicted_") if moment_schedule is not None else moment_predict(*args, **kwargs)

    def update_at(t, *args, **kwargs):
        return scheduled_moments(t, "post_") if moment_schedule is not None else moment_update(*args, **kwargs)

    def time_loop_body(t, states, d_states, covariances, d_covariances, incoming_log_weights, d_incoming_log_weights, total, d_total):
        observation = observations[t]
        noise = noises[t]
        step_incoming_log_weights = incoming_log_weights
        d_step_incoming_log_weights = d_incoming_log_weights

        # S1: UKF predict with chained covariance tangent
        (
            predicted_means,
            predicted_covs,
            d_predicted_means,
            d_predicted_covs,
            valid_predict,
        ) = predict_at(t,
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

        observation_record = {}

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
            result = observation_factor_override(t, points, d_points,
                                                 observation, *base_result)
            if len(result) == 3:
                observation_record.update(result[2])
            return result[:2]

        # annealed_stages > 1 removed by constraint
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
        total = total + increment
        d_total = d_total + tf.reduce_sum(softmax * d_logits)

        # S5: UKF update with chained tangent -> next step's covariances
        (
            _post_means,
            post_covs,
            _d_post_means,
            d_post_covs,
            valid_update,
        ) = update_at(t,
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
                valid_reset,
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
            if correction_steps > 0 or pairwise_steps > 0:
                from bayesfilter.highdim.ledh_unified_correction_tf import (
                    batched_higher_moment_shape_jvp,
                )

                batched_corrected = batched_higher_moment_shape_jvp(
                    children[None, ...],
                    step_weights[None, ...],
                    d_children[None, None, ...],
                    d_step_weights[None, None, ...],
                    reset_states[None, ...],
                    d_reset_states[None, None, ...],
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
                corrected = {
                    key: (
                        tf.transpose(value[:, 0], [1, 2, 0])
                        if key == "particles_tangent"
                        else value[0]
                    )
                    for key, value in batched_corrected.items()
                }
                tf.debugging.assert_equal(
                    corrected["valid"],
                    True,
                    message="higher-moment Contract-E correction is invalid",
                )
                # XLA may elide Assert ops. Propagate invalidity into the
                # returned scalar even at the final reset, where no subsequent
                # observation would otherwise expose an invalid correction.
                neg_inf = tf.constant(float("-inf"), dtype)
                total = tf.where(corrected["valid"], total, neg_inf)
                d_total = tf.where(corrected["valid"], d_total, tf.zeros([], dtype))
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
                # Retain every diagnostic computed by the shared correction.
                # These tensors never alter states, weights or their tangents.
                higher_moment_record.update({
                    "higher_moment_" + key: value
                    for key, value in corrected.items()
                    if key not in ("particles", "particles_tangent")
                })
            new_states, new_d_states = reset_states, d_reset_states
            new_covariances, new_d_covariances = (
                reset_covariances,
                d_reset_covariances,
            )
            new_incoming_log_weights = uniform_log_weights
            new_d_incoming_log_weights = tf.zeros_like(uniform_log_weights)
        else:
            new_states, new_d_states = children, d_children
            new_covariances, new_d_covariances = post_covs, d_post_covs
            new_incoming_log_weights = uniform_log_weights
            new_d_incoming_log_weights = tf.zeros_like(uniform_log_weights)

        if post_reset_transform is not None:
            (new_states, new_d_states, new_covariances, new_d_covariances,
             new_incoming_log_weights, new_d_incoming_log_weights,
             post_reset_record) = post_reset_transform(
                 t, new_states, new_d_states, new_covariances, new_d_covariances)
        callback_valid = (observation_record.get("observation_factor_valid", tf.constant(True))
                          & post_reset_record.get("post_reset_valid", tf.constant(True)))

        # Combine all validity flags from numerical operations
        # Reduce batch-shaped validity flags to scalars for single-trajectory context
        numerical_valid = tf.reduce_all(valid_predict) & tf.reduce_all(valid_update)
        if reset_policy == "contract_e":
            numerical_valid = numerical_valid & tf.reduce_all(valid_reset)

        # Overall validity: both callbacks and numerical operations must succeed
        overall_valid = tf.reduce_all(callback_valid) & numerical_valid

        # When invalid, return -inf for log probability (NaN would propagate differently)
        neg_inf = tf.constant(float("-inf"), dtype)
        total = tf.where(overall_valid, total, neg_inf)
        d_total = tf.where(overall_valid, d_total, tf.zeros([], dtype))

        result = (
            t + 1,
            new_states,
            new_d_states,
            new_covariances,
            new_d_covariances,
            new_incoming_log_weights,
            new_d_incoming_log_weights,
            total,
            d_total,
        )
        if return_trace:
            step_record = {
                "time_index": t,
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
                "post_means": _post_means,
                "d_post_means": _d_post_means,
                "d_post_covariances": d_post_covs,
                "covariances_after_reset": new_covariances,
                "d_covariances_after_reset": new_d_covariances,
                "reset_transport": reset_transport,
                "d_reset_transport": d_reset_transport,
                "states_after_reset": new_states,
                "d_states_after_reset": new_d_states,
                "outgoing_log_weights": new_incoming_log_weights,
                "d_outgoing_log_weights": new_d_incoming_log_weights,
            }
            step_record.update(higher_moment_record)
            step_record.update(post_reset_record)
            step_record.update(observation_record)
            return result, step_record
        return result


    initial_loop = (tf.constant(0, tf.int32), states, d_states, covariances,
                    d_covariances, incoming_log_weights, d_incoming_log_weights,
                    total, d_total)
    state_shapes = (tf.TensorShape([]), tf.TensorShape([None, dim]),
                    tf.TensorShape([None, dim]), tf.TensorShape([None, dim, dim]),
                    tf.TensorShape([None, dim, dim]), tf.TensorShape([None]),
                    tf.TensorShape([None]), tf.TensorShape([]), tf.TensorShape([]))
    if return_trace:
        # One traced numerical step; TensorArrays carry all diagnostic fields
        # through the time loop. Python callbacks must return their metadata,
        # never append loop tensors to a Python container outside this body.
        step_kernel = tf.function(time_loop_body, autograph=False,
                                  input_signature=[tf.TensorSpec(v.shape, v.dtype)
                                                   for v in initial_loop])
        template = step_kernel.get_concrete_function().structured_outputs[1]
        buffers = tf.nest.map_structure(
            lambda value: tf.TensorArray(value.dtype, size=horizon,
                                         element_shape=value.shape), template)

        def traced_body(*args):
            result, record = step_kernel(*args[:-1])
            buffers = tf.nest.map_structure(lambda buf, value: buf.write(args[0], value),
                                            args[-1], record)
            return (*result, buffers)

        loop_result = tf.while_loop(
            lambda *args: args[0] < horizon, traced_body,
            (*initial_loop, buffers), maximum_iterations=horizon,
            parallel_iterations=1)
        stacked = tf.nest.map_structure(lambda buf: buf.stack(), loop_result[-1])
        trace = tuple(tf.nest.map_structure(lambda value: value[index], stacked)
                      for index in range(horizon))
        total, d_total = loop_result[-3:-1]
        return total, d_total[None] if with_score else None, trace
    loop_result = tf.while_loop(
        time_loop_cond, time_loop_body, initial_loop,
        shape_invariants=state_shapes, maximum_iterations=horizon,
        parallel_iterations=1)
    total, d_total = loop_result[-2:]
    return total, d_total[None] if with_score else None


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
    initial_state_tangent: Tensor | None = None,
    initial_covariance_tangent: Tensor | None = None,
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

    Supplied initial-state and covariance tangents carry dependence of a
    parameterized initial law into the same analytical recursion. Omitting
    them declares fixed initial inputs; it does not differentiate their
    construction. Their direction must match the model tangent callbacks.
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
        initial_state_tangent=initial_state_tangent,
        initial_covariance_tangent=initial_covariance_tangent,
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
    )


def _flow_substep_body_impl(
    step_index: tf.Tensor,
    actual: Tensor,
    d_actual: Tensor,
    auxiliary: Tensor,
    d_auxiliary: Tensor,
    log_det: Tensor,
    d_log_det: Tensor,
    model: NonlinearScoreModel,
    prior_means: Tensor,
    d_prior_means: Tensor,
    predicted_covs: Tensor,
    d_predicted_covs: Tensor,
    observation: Tensor,
    observation_covariance: Tensor,
    d_observation_covariance: Tensor | None,
    r_inv: Tensor,
    d_r_inv: Tensor | None,
    substeps: tf.Tensor,
    eye: Tensor,
    eps: Tensor,
) -> tuple[tf.Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Single flow substep with analytical tangent.

    Extracted body for tf.while_loop conversion (Phase 3.5.2).

    Returns:
        (step_index + 1, updated_actual, updated_d_actual, updated_auxiliary,
         updated_d_auxiliary, updated_log_det, updated_d_log_det)
    """
    dtype = actual.dtype
    step_int = tf.cast(step_index, tf.int32)
    lam = tf.cast(step_index + 1, dtype) / tf.cast(substeps, dtype)
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
    new_d_actual = d_actual + eps * (
        tf.einsum("nij,nj->ni", d_a_matrix, actual)
        + tf.einsum("nij,nj->ni", a_matrix, d_actual)
        + d_b_vector
    )
    new_auxiliary = auxiliary + eps * (
        tf.einsum("nij,nj->ni", a_matrix, auxiliary) + b_vector
    )
    new_d_auxiliary = d_auxiliary + eps * (
        tf.einsum("nij,nj->ni", d_a_matrix, auxiliary)
        + tf.einsum("nij,nj->ni", a_matrix, d_auxiliary)
        + d_b_vector
    )
    step_matrix = eye[None] + eps * a_matrix
    q_factor, r_factor = tf.linalg.qr(step_matrix)
    new_log_det = log_det + tf.reduce_sum(
        tf.math.log(tf.abs(tf.linalg.diag_part(r_factor))), axis=1
    )
    q_transpose_d_a = tf.linalg.matmul(q_factor, d_a_matrix, transpose_a=True)
    solved_d_a = tf.linalg.triangular_solve(r_factor, q_transpose_d_a, lower=False)
    new_d_log_det = d_log_det + eps * tf.linalg.trace(solved_d_a)

    return (
        step_index + 1,
        new_actual,
        new_d_actual,
        new_auxiliary,
        new_d_auxiliary,
        new_log_det,
        new_d_log_det,
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
    substeps_tensor = tf.constant(substeps, dtype=tf.int32)
    auxiliary, d_auxiliary = prior_means, d_prior_means
    log_det = tf.zeros([count], dtype)
    d_log_det = tf.zeros([count], dtype)

    # Phase 3.5.2: tf.while_loop conversion for substep loop
    def substep_cond(i, *_):
        return i < substeps_tensor

    def substep_body_wrapper(i, act, d_act, aux, d_aux, ld, d_ld):
        return _flow_substep_body_impl(
            i,
            act,
            d_act,
            aux,
            d_aux,
            ld,
            d_ld,
            model,
            prior_means,
            d_prior_means,
            predicted_covs,
            d_predicted_covs,
            observation,
            observation_covariance,
            d_observation_covariance,
            r_inv,
            d_r_inv,
            substeps_tensor,
            eye,
            eps,
        )

    initial_loop_vars = (
        tf.constant(0, dtype=tf.int32),
        actual,
        d_actual,
        auxiliary,
        d_auxiliary,
        log_det,
        d_log_det,
    )

    _, actual, d_actual, auxiliary, d_auxiliary, log_det, d_log_det = tf.while_loop(
        cond=substep_cond,
        body=substep_body_wrapper,
        loop_vars=initial_loop_vars,
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
