"""Native execution of the existing single-cloud LEDH value recurrence.

This is an execution repair, not canonical/scientific admission. The program
uses the original UKF, flow, weight and reset authorities, with supplied random
inputs. Callback configuration must remain fixed for the owner lifetime; all
observations and random inputs are dynamic operands of one enclosing program.
"""

from __future__ import annotations

import math

import tensorflow as tf

from bayesfilter.highdim.genut_guided_proposal_tf import _restore_cloud_primal
from bayesfilter.highdim.ledh_flow_perparticle_tf import ledh_flow_per_particle
from bayesfilter.highdim.ledh_ukf_lifecycle_tf import (
    ukf_predict_per_particle,
    ukf_update_per_particle,
)
from bayesfilter.ops.legacy_fraction_tf import legacy_fraction


def systematic_ancestor_indices(weights, uniform):
    """Original sequential cumulative sum and left-search systematic rule."""
    count = weights.shape[0]

    def accumulate(index, total, cumulative):
        total = total + weights[index]
        return (
            index + 1,
            total,
            tf.tensor_scatter_nd_update(cumulative, [[index]], [total]),
        )

    cumulative = tf.while_loop(
        lambda index, *_: index < count,
        accumulate,
        (tf.constant(0), tf.zeros([], weights.dtype), tf.zeros_like(weights)),
        parallel_iterations=1,
    )[2]
    cumulative = tf.tensor_scatter_nd_update(
        cumulative, [[count - 1]], [tf.cast(1.0, weights.dtype)]
    )
    # NumPy promotes arange + the PCG64 uniform to float64, including for FP32 weights.
    positions = (uniform + tf.cast(tf.range(count), tf.float64)) / tf.constant(
        count, tf.float64
    )
    return tf.searchsorted(
        tf.cast(cumulative, tf.float64), positions, side="left", out_type=tf.int32
    )


def make_canonical_value_program(
    callbacks,
    observation_spec,
    *,
    particle_count,
    flow_substeps=24,
    temper_stages=1,
    annealed_resampling=False,
    flow_prior_cap=float("inf"),
    epsilon=2.0,
    sinkhorn_steps=8,
    balance_steps=8,
    ridge=1e-5,
    dual_cap_enabled=False,
    trust_region_enabled=False,
    trust_region_lm_damping=1e-2,
    trust_region_lm_scale_floor=1e-4,
    trust_region_radius=0.5,
    jit_compile=True,
):
    """Build a caller-owned fixed-signature value kernel, defaulting to XLA.

    Inputs: observations, initial normals, per-time process normals and per-time
    PCG64 stage uniforms. Return numeric, fixed-capacity diagnostics plus count.
    A host adapter must trim completed diagnostic history and attach model_id;
    the existing public adapter is not integrated with this program yet.
    Explicit ``jit_compile=False`` is a diagnostic reference exception only.
    """
    dtype = observation_spec.dtype
    horizon, observed = observation_spec.shape.as_list()
    dimension = callbacks.state_dim
    stage_count = max(1, int(temper_stages))
    if horizon is None or horizon < 1 or observed != callbacks.observation_dim:
        raise ValueError("observations require static positive [T, observation_dim]")
    if particle_count < 1 or flow_substeps < 1:
        raise ValueError("particle_count and flow_substeps must be positive")
    signature = [
        observation_spec,
        tf.TensorSpec([particle_count, dimension], dtype),
        tf.TensorSpec([horizon, particle_count, dimension], dtype),
        tf.TensorSpec([horizon, stage_count], tf.float64),
    ]

    @tf.function(input_signature=signature, jit_compile=jit_compile, autograph=False)
    def program(observations, initial_noise, process_noises, resampling_uniforms):
        initial_covariance = tf.convert_to_tensor(callbacks.initial_covariance, dtype)
        chol = tf.linalg.cholesky(initial_covariance)
        states = callbacks.initial_mean[None, :] + tf.linalg.matvec(
            tf.broadcast_to(chol, [particle_count, dimension, dimension]), initial_noise
        )
        covariances = tf.broadcast_to(
            initial_covariance, [particle_count, dimension, dimension]
        )
        weights = tf.fill([particle_count], tf.cast(1.0 / particle_count, dtype))
        process_covariance = tf.convert_to_tensor(
            callbacks.process_noise_covariance, dtype
        )
        process_chol = tf.linalg.cholesky(process_covariance)
        observation_covariance = tf.convert_to_tensor(
            callbacks.observation_covariance, dtype
        )
        design = tf.concat(
            [tf.eye(dimension, dtype=dtype), -tf.eye(dimension, dtype=dtype)], axis=0
        )
        design = tf.tile(design, [particle_count // (2 * dimension) + 1, 1])[
            :particle_count
        ]
        nan = tf.cast(float("nan"), dtype)

        def flow_step(current, anchors, prior, observation, time_index):
            return ledh_flow_per_particle(
                anchor_states=anchors,
                pre_flow_states=current,
                predicted_covariances=prior / tf.cast(stage_count, dtype),
                observation=observation,
                observation_fn=lambda p: callbacks.observation_fn(p, time_index),
                observation_jacobian_fn=lambda p: callbacks.observation_jacobian_fn(
                    p, time_index
                ),
                observation_covariance=observation_covariance
                * tf.cast(stage_count, dtype),
                prior_means=anchors,
                substeps=flow_substeps,
            )

        def finish_step(time_index, states, weights, predicted_means, predicted_covs):
            observation = observations[time_index]
            anchors = callbacks.transition_mean_fn(states, time_index)
            pre_flow = anchors + tf.linalg.matvec(
                tf.broadcast_to(process_chol, [particle_count, dimension, dimension]),
                process_noises[time_index],
            )
            if math.isfinite(flow_prior_cap):
                eigenvalues, vectors = tf.linalg.eigh(predicted_covs)
                prior = tf.einsum(
                    "nij,nj,nkj->nik",
                    vectors,
                    tf.minimum(eigenvalues, tf.cast(flow_prior_cap, dtype)),
                    vectors,
                )
            else:
                prior = predicted_covs
            if annealed_resampling:
                prior_obs = callbacks.observation_log_density_fn(
                    pre_flow, observation, time_index
                )
                prior_trans = callbacks.transition_log_density_fn(
                    pre_flow, states, time_index
                )

                def annealed_body(
                    stage,
                    current,
                    anchors,
                    ancestors,
                    prior,
                    means,
                    covs,
                    increment,
                    ess,
                    prev_obs,
                    prev_trans,
                ):
                    flow = flow_step(current, anchors, prior, observation, time_index)
                    moved = flow["post_flow_states"]
                    obs_log = callbacks.observation_log_density_fn(
                        moved, observation, time_index
                    )
                    trans_log = callbacks.transition_log_density_fn(
                        moved, ancestors, time_index
                    )
                    # Match Python stage/stage_count followed by tf.cast (float32
                    # intermediate for a Python float, including in FP64 programs).
                    fraction = legacy_fraction(stage, stage_count, dtype)
                    previous = legacy_fraction(stage - 1, stage_count, dtype)
                    logits = (
                        trans_log
                        + tf.cast(fraction, dtype) * obs_log
                        + flow["forward_log_det"]
                        - prev_trans
                        - tf.cast(previous, dtype) * prev_obs
                    )
                    increment += tf.reduce_logsumexp(
                        logits - tf.math.log(tf.cast(particle_count, dtype))
                    )
                    stage_weights = tf.exp(logits - tf.reduce_logsumexp(logits))
                    ess = tf.minimum(ess, 1.0 / tf.reduce_sum(tf.square(stage_weights)))
                    indices = systematic_ancestor_indices(
                        stage_weights, resampling_uniforms[time_index, stage - 1]
                    )
                    current = tf.gather(moved, indices)
                    anchors, ancestors, prior = (
                        tf.gather(anchors, indices),
                        tf.gather(ancestors, indices),
                        tf.gather(prior, indices),
                    )
                    means, covs = tf.gather(means, indices), tf.gather(covs, indices)
                    prev_obs = callbacks.observation_log_density_fn(
                        current, observation, time_index
                    )
                    prev_trans = callbacks.transition_log_density_fn(
                        current, ancestors, time_index
                    )
                    return (
                        stage + 1,
                        current,
                        anchors,
                        ancestors,
                        prior,
                        means,
                        covs,
                        increment,
                        ess,
                        prev_obs,
                        prev_trans,
                    )

                stage_result = tf.while_loop(
                    lambda stage, *_: stage <= stage_count,
                    annealed_body,
                    (
                        tf.constant(1),
                        pre_flow,
                        anchors,
                        states,
                        prior,
                        predicted_means,
                        predicted_covs,
                        tf.zeros([], dtype),
                        tf.cast(particle_count, dtype),
                        prior_obs,
                        prior_trans,
                    ),
                    parallel_iterations=1,
                )
                children, predicted_means, predicted_covs = (
                    stage_result[1],
                    stage_result[5],
                    stage_result[6],
                )
                increment, ess = stage_result[7], stage_result[8]
                step_weights = tf.fill(
                    [particle_count], tf.cast(1.0 / particle_count, dtype)
                )
            else:

                def composed_body(stage, current, log_det):
                    flow = flow_step(current, anchors, prior, observation, time_index)
                    return (
                        stage + 1,
                        flow["post_flow_states"],
                        log_det + flow["forward_log_det"],
                    )

                _, children, log_det = tf.while_loop(
                    lambda stage, *_: stage < stage_count,
                    composed_body,
                    (tf.constant(0), pre_flow, tf.zeros([particle_count], dtype)),
                    parallel_iterations=1,
                )
                transition_log = callbacks.transition_log_density_fn(
                    children, states, time_index
                )
                observation_log = callbacks.observation_log_density_fn(
                    children, observation, time_index
                )
                proposal_log = callbacks.transition_log_density_fn(
                    pre_flow, states, time_index
                )
                logits = (
                    tf.math.log(weights)
                    + transition_log
                    + observation_log
                    + log_det
                    - proposal_log
                )
                increment = tf.reduce_logsumexp(logits)
                step_weights = tf.exp(logits - increment)
                ess = 1.0 / tf.reduce_sum(tf.square(step_weights))
            step_valid = (
                tf.reduce_all(tf.math.is_finite(children))
                & tf.math.is_finite(increment)
                & tf.reduce_all(tf.math.is_finite(step_weights))
            )
            safe_weights = tf.where(
                step_valid,
                step_weights,
                tf.fill([particle_count], tf.cast(1.0 / particle_count, dtype)),
            )
            safe_children = tf.where(
                tf.math.is_finite(children), children, tf.zeros_like(children)
            )
            _, post_covs = ukf_update_per_particle(
                predicted_means,
                predicted_covs,
                lambda p: callbacks.observation_fn(p, time_index),
                observation_covariance,
                observation,
            )
            restored = _restore_cloud_primal(
                tf.cast(safe_children, tf.float32),
                tf.cast(safe_weights, tf.float32),
                tf.cast(design, tf.float32),
                epsilon=epsilon,
                sinkhorn_steps=sinkhorn_steps,
                balance_steps=balance_steps,
                ridge=ridge,
                reset_policy="contract_e",
                dual_cap_enabled=dual_cap_enabled,
                trust_region_enabled=trust_region_enabled,
                trust_region_lm_damping=trust_region_lm_damping,
                trust_region_lm_scale_floor=trust_region_lm_scale_floor,
                trust_region_radius=trust_region_radius,
            )
            states = tf.cast(restored["particles"], dtype)
            return (
                states,
                post_covs,
                tf.where(step_valid, increment, nan),
                step_valid & tf.reduce_all(tf.math.is_finite(states)),
                ess,
                tf.cast(restored["post_quotient_column_tv_error"], dtype),
                restored["marginal_valid"],
                restored["reset_valid"],
                tf.cast(restored["maximum_diagonal_scaled_system_condition"], dtype),
            )

        def body(
            index,
            states,
            covariances,
            total,
            valid,
            ess,
            marginal_tv,
            marginal_ok,
            completed,
            reset_ok,
            reset_condition,
        ):
            def abort():
                return (
                    tf.constant(horizon),
                    states,
                    covariances,
                    nan,
                    tf.constant(False),
                    ess,
                    marginal_tv,
                    marginal_ok,
                    completed,
                    reset_ok,
                    reset_condition,
                )

            def predict():
                means, covs = ukf_predict_per_particle(
                    states,
                    covariances,
                    lambda p: callbacks.transition_mean_fn(p, index),
                    process_covariance,
                )

                def advance():
                    next_states, next_covs, increment, step_valid, step_ess, tv, ok, reset_valid, condition = (
                        finish_step(index, states, weights, means, covs)
                    )
                    return (
                        index + 1,
                        next_states,
                        next_covs,
                        total + increment,
                        valid & step_valid,
                        tf.tensor_scatter_nd_update(ess, [[index]], [step_ess]),
                        tf.tensor_scatter_nd_update(marginal_tv, [[index]], [tv]),
                        tf.tensor_scatter_nd_update(marginal_ok, [[index]], [ok]),
                        completed + 1,
                        tf.tensor_scatter_nd_update(reset_ok, [[index]], [reset_valid]),
                        tf.tensor_scatter_nd_update(reset_condition, [[index]], [condition]),
                    )

                finite = tf.reduce_all(tf.math.is_finite(means)) & tf.reduce_all(
                    tf.math.is_finite(covs)
                )
                return tf.cond(finite, advance, abort)

            finite = tf.reduce_all(tf.math.is_finite(states)) & tf.reduce_all(
                tf.math.is_finite(covariances)
            )
            return tf.cond(finite, predict, abort)

        result = tf.while_loop(
            lambda index, *_: index < horizon,
            body,
            (
                tf.constant(0),
                states,
                covariances,
                tf.zeros([], dtype),
                tf.constant(True),
                tf.fill([horizon], nan),
                tf.fill([horizon], nan),
                tf.zeros([horizon], tf.bool),
                tf.constant(0),
                tf.zeros([horizon], tf.bool),
                tf.fill([horizon], nan),
            ),
            parallel_iterations=1,
        )
        _, _, _, total, valid, ess, marginal_tv, marginal_ok, completed, reset_ok, reset_condition = result
        mask = tf.range(horizon) < completed
        all_resets_valid = (completed == horizon) & tf.reduce_all(reset_ok)
        return {
            "value": tf.where(valid, total, nan),
            "program_valid": valid,
            "per_step_ess": ess,
            "dual_cap_active": tf.constant(dual_cap_enabled),
            "per_step_marginal_tv_error": marginal_tv,
            "per_step_marginal_valid": marginal_ok,
            "max_marginal_tv_error": tf.where(
                completed > 0,
                tf.reduce_max(
                    tf.where(mask, marginal_tv, tf.cast(float("-inf"), dtype))
                ),
                nan,
            ),
            "all_marginals_valid": (completed > 0) & tf.reduce_all(~mask | marginal_ok),
            "marginal_steps_completed": completed,
            "per_step_reset_valid": reset_ok,
            "all_resets_valid": all_resets_valid,
            "numerical_valid": valid & all_resets_valid,
            "per_step_reset_scaled_system_condition": reset_condition,
        }

    return program
