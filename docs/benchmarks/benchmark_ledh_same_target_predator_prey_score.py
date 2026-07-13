"""Same-target predator-prey LEDH score repair helpers.

This module is the score-side sibling of
``benchmark_ledh_same_target_predator_prey_value``.  The current/default score
route is the compact forward-sensitivity recurrence for the same realized
finite-``N`` LEDH log-likelihood scalar.  The older reverse/manual VJP route is
kept only as historical diagnostic code and must not be full-admitted.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import argparse
import copy
import json
import platform
import subprocess
import time
from typing import Any, Mapping


_PRE_DEVICE_SCOPE = os.environ.get("BAYESFILTER_LEDHD_SCORE_DEVICE_SCOPE")
if _PRE_DEVICE_SCOPE == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

from bayesfilter import highdim
from bayesfilter.highdim.ledh_forward_contract import (
    LEDH_OUTPUT_TENSOR_FIELD_LOG_LIKELIHOOD,
    LEDH_TARGET_SCALAR_OBSERVED_DATA_LOG_LIKELIHOOD,
    PREDATOR_PREY_PARAMETER_ORDER,
    PREDATOR_PREY_ROW_ID,
    validate_ledh_forward_scalar_artifact,
)
from bayesfilter.highdim.ledh_score_contract import (
    LEDH_SCORE_ADMISSION_STATUS_BLOCKED_NOT_RUN,
    LEDH_SCORE_ADMISSION_STATUS_FULL,
    LEDH_SCORE_ADMISSION_STATUS_TINY,
    LEDH_SCORE_ARTIFACT_SCHEMA_VERSION,
    LEDH_SCORE_COMPACT_PREDATOR_PREY_PROVENANCE,
    LEDH_SCORE_MEMORY_STYLE_PREDATOR_PREY_PROVENANCE,
    LEDH_SCORE_TARGET_KIND_REALIZED_FINITE_N_ESTIMATOR,
    LEDH_SCORE_VALUE_ROUTE_STATUS_SAME,
    validate_ledh_score_artifact,
)
from docs.benchmarks import benchmark_ledh_same_target_predator_prey_value as value_mod
from docs.benchmarks import benchmark_p8p_parameterized_sir_gradient as p8p
from experiments.dpf_implementation.tf_tfp.filters import (
    experimental_batched_ledh_pfpf_ot_tf as core_tf,
)
from experiments.dpf_implementation.tf_tfp.resampling import annealed_transport_tf


DTYPE = tf.float32
PREDATOR_PREY_MANUAL_SCORE_ROUTE_ID = (
    "manual_total_vjp_no_autodiff_same_scalar_predator_prey_ledh_pfpf_ot"
)
PREDATOR_PREY_MEMORY_STYLE_SCORE_ROUTE_ID = LEDH_SCORE_MEMORY_STYLE_PREDATOR_PREY_PROVENANCE
PREDATOR_PREY_COMPACT_SCORE_ROUTE_ID = LEDH_SCORE_COMPACT_PREDATOR_PREY_PROVENANCE
PARAMETER_NAMES = PREDATOR_PREY_PARAMETER_ORDER
TRUTH_THETA = (0.6, 114.0, 25.0, 0.3, 0.5, 0.5)
MANUAL_SCORE_COMPONENT_NAMES = (
    "transition_mean_total",
)


def _configure_precision(args: argparse.Namespace) -> dict[str, Any]:
    global DTYPE
    dtype_name = getattr(args, "dtype", "float32")
    tf32_mode = getattr(args, "tf32_mode", "enabled")
    DTYPE = tf.float64 if dtype_name == "float64" else tf.float32
    value_mod.DTYPE = DTYPE
    p8p.DTYPE = DTYPE
    core_tf.DTYPE = DTYPE
    annealed_transport_tf.DTYPE = DTYPE
    if tf32_mode != "default":
        tf.config.experimental.enable_tensor_float_32_execution(tf32_mode == "enabled")
    metadata = core_tf.precision_policy_metadata()
    metadata.update(
        {
            "dtype": dtype_name,
            "tf_dtype": DTYPE.name,
            "tf32_mode": tf32_mode,
            "tf32_execution_enabled": bool(
                tf.config.experimental.tensor_float_32_execution_enabled()
            ),
        }
    )
    return metadata


def _score_precision_metadata(precision: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dtype": str(precision.get("dtype")),
        "active_dtype": str(precision.get("active_dtype")),
        "tf_dtype": str(precision.get("tf_dtype")),
        "tf32_mode": str(precision.get("tf32_mode")),
        "tf32_execution_enabled": bool(precision.get("tf32_execution_enabled")),
    }


def _as_theta(theta: tf.Tensor | list[float] | tuple[float, ...]) -> tf.Tensor:
    tensor = tf.convert_to_tensor(theta, dtype=DTYPE)
    if tensor.shape.rank != 1 or int(tensor.shape[0]) != len(PARAMETER_NAMES):
        raise ValueError("predator-prey theta must have shape [6]")
    return tensor


def _as_state_matrix(state: tf.Tensor) -> tf.Tensor:
    tensor = tf.convert_to_tensor(state, dtype=DTYPE)
    if tensor.shape.rank == 1:
        tensor = tensor[tf.newaxis, :]
    if tensor.shape.rank is None or tensor.shape.rank < 2 or int(tensor.shape[-1]) != 2:
        raise ValueError("predator-prey state must have trailing shape [...,2]")
    return tensor


def _predator_prey_rhs_tf(theta: tf.Tensor, state: tf.Tensor) -> tf.Tensor:
    theta = _as_theta(theta)
    state = _as_state_matrix(state)
    r, k_capacity, a_half, s_rate, u_rate, v_rate = tf.unstack(theta)
    prey = state[..., 0]
    predator = state[..., 1]
    denominator = a_half + prey
    interaction = prey * predator / denominator
    d_prey = r * prey * (1.0 - prey / k_capacity) - s_rate * interaction
    d_predator = u_rate * interaction - v_rate * predator
    return tf.stack([d_prey, d_predator], axis=-1)


def _predator_prey_rhs_vjp_tf(
    theta: tf.Tensor,
    state: tf.Tensor,
    upstream: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Manual VJP of the predator-prey RHS.

    Returns ``(bar_state, bar_theta)`` for the physical theta order
    ``(r,K,a,s,u,v)``.
    """

    theta = _as_theta(theta)
    state = _as_state_matrix(state)
    upstream = _as_state_matrix(upstream)
    r, k_capacity, a_half, s_rate, u_rate, v_rate = tf.unstack(theta)
    prey = state[..., 0]
    predator = state[..., 1]
    bar_d_prey = upstream[..., 0]
    bar_d_predator = upstream[..., 1]

    denominator = a_half + prey
    denominator_sq = tf.square(denominator)
    interaction = prey * predator / denominator

    logistic = prey * (1.0 - prey / k_capacity)
    reduce_axes = tf.range(tf.rank(prey))
    bar_r = tf.reduce_sum(bar_d_prey * logistic, axis=reduce_axes)
    bar_logistic = bar_d_prey * r
    bar_prey = bar_logistic * (1.0 - 2.0 * prey / k_capacity)
    bar_k = tf.reduce_sum(
        bar_logistic * tf.square(prey) / tf.square(k_capacity),
        axis=reduce_axes,
    )

    bar_s = tf.reduce_sum(-bar_d_prey * interaction, axis=reduce_axes)
    bar_u = tf.reduce_sum(bar_d_predator * interaction, axis=reduce_axes)
    bar_v = tf.reduce_sum(-bar_d_predator * predator, axis=reduce_axes)
    bar_predator = -bar_d_predator * v_rate

    bar_interaction = -bar_d_prey * s_rate + bar_d_predator * u_rate
    bar_prey += bar_interaction * predator * a_half / denominator_sq
    bar_predator += bar_interaction * prey / denominator
    bar_a = tf.reduce_sum(
        -bar_interaction * prey * predator / denominator_sq,
        axis=reduce_axes,
    )

    bar_state = tf.stack([bar_prey, bar_predator], axis=-1)
    bar_theta = tf.stack([bar_r, bar_k, bar_a, bar_s, bar_u, bar_v])
    return bar_state, bar_theta


def _predator_prey_rk4_step_with_aux_tf(
    theta: tf.Tensor,
    state: tf.Tensor,
    step: tf.Tensor,
) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
    theta = _as_theta(theta)
    state = _as_state_matrix(state)
    step = tf.convert_to_tensor(step, dtype=DTYPE)
    half = tf.constant(0.5, dtype=DTYPE)
    k1 = _predator_prey_rhs_tf(theta, state)
    k2_input = state + half * step * k1
    k2 = _predator_prey_rhs_tf(theta, k2_input)
    k3_input = state + half * step * k2
    k3 = _predator_prey_rhs_tf(theta, k3_input)
    k4_input = state + step * k3
    k4 = _predator_prey_rhs_tf(theta, k4_input)
    next_state = state + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return next_state, {
        "state": state,
        "k1": k1,
        "k2_input": k2_input,
        "k2": k2,
        "k3_input": k3_input,
        "k3": k3,
        "k4_input": k4_input,
    }


def _predator_prey_rk4_step_vjp_tf(
    theta: tf.Tensor,
    aux: dict[str, tf.Tensor],
    upstream: tf.Tensor,
    step: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Manual VJP of one RK4 step."""

    theta = _as_theta(theta)
    upstream = _as_state_matrix(upstream)
    step = tf.convert_to_tensor(step, dtype=DTYPE)
    half = tf.constant(0.5, dtype=DTYPE)
    sixth = step / 6.0
    third = step / 3.0

    bar_state = upstream
    bar_k1 = sixth * upstream
    bar_k2 = third * upstream
    bar_k3 = third * upstream
    bar_k4 = sixth * upstream
    bar_theta = tf.zeros([len(PARAMETER_NAMES)], dtype=DTYPE)

    bar_k4_input, theta_part = _predator_prey_rhs_vjp_tf(
        theta,
        aux["k4_input"],
        bar_k4,
    )
    bar_theta += theta_part
    bar_state += bar_k4_input
    bar_k3 += step * bar_k4_input

    bar_k3_input, theta_part = _predator_prey_rhs_vjp_tf(
        theta,
        aux["k3_input"],
        bar_k3,
    )
    bar_theta += theta_part
    bar_state += bar_k3_input
    bar_k2 += half * step * bar_k3_input

    bar_k2_input, theta_part = _predator_prey_rhs_vjp_tf(
        theta,
        aux["k2_input"],
        bar_k2,
    )
    bar_theta += theta_part
    bar_state += bar_k2_input
    bar_k1 += half * step * bar_k2_input

    bar_k1_input, theta_part = _predator_prey_rhs_vjp_tf(
        theta,
        aux["state"],
        bar_k1,
    )
    bar_theta += theta_part
    bar_state += bar_k1_input
    return bar_state, bar_theta


def _predator_prey_transition_mean_with_aux_tf(
    theta: tf.Tensor,
    state: tf.Tensor,
) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
    model = highdim.p30_predator_prey_fixture_model()
    theta = _as_theta(theta)
    running = _as_state_matrix(state)
    step = tf.cast(model.delta, DTYPE) / tf.cast(int(model._rk4_substeps), DTYPE)
    aux_rows: dict[str, list[tf.Tensor]] = {
        "state": [],
        "k1": [],
        "k2_input": [],
        "k2": [],
        "k3_input": [],
        "k3": [],
        "k4_input": [],
    }
    for _ in range(int(model._rk4_substeps)):
        running, aux = _predator_prey_rk4_step_with_aux_tf(theta, running, step)
        for key, value in aux.items():
            aux_rows[key].append(value)
    return running, {key: tf.stack(values, axis=0) for key, values in aux_rows.items()}


def _predator_prey_transition_mean_vjp_tf(
    theta: tf.Tensor,
    aux: dict[str, tf.Tensor],
    upstream: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    model = highdim.p30_predator_prey_fixture_model()
    theta = _as_theta(theta)
    bar_state = _as_state_matrix(upstream)
    bar_theta = tf.zeros([len(PARAMETER_NAMES)], dtype=DTYPE)
    step = tf.cast(model.delta, DTYPE) / tf.cast(int(model._rk4_substeps), DTYPE)
    for index in range(int(model._rk4_substeps) - 1, -1, -1):
        step_aux = {key: value[index] for key, value in aux.items()}
        bar_state, theta_part = _predator_prey_rk4_step_vjp_tf(
            theta,
            step_aux,
            bar_state,
            step,
        )
        bar_theta += theta_part
    return bar_state, bar_theta


def predator_prey_dynamics_vjp_diagnostic() -> dict[str, Any]:
    """Return a tiny deterministic diagnostic for the manual dynamics VJP."""

    theta = tf.constant(TRUTH_THETA, dtype=DTYPE)
    state = tf.constant([[50.0, 5.0], [80.0, 3.0]], dtype=DTYPE)
    upstream = tf.constant([[0.7, -0.2], [-0.3, 0.5]], dtype=DTYPE)
    rhs = _predator_prey_rhs_tf(theta, state)
    rhs_bar_state, rhs_bar_theta = _predator_prey_rhs_vjp_tf(theta, state, upstream)
    mean, aux = _predator_prey_transition_mean_with_aux_tf(theta, state)
    mean_bar_state, mean_bar_theta = _predator_prey_transition_mean_vjp_tf(
        theta,
        aux,
        upstream,
    )
    return {
        "route_id": PREDATOR_PREY_MANUAL_SCORE_ROUTE_ID,
        "parameter_names": list(PARAMETER_NAMES),
        "rhs": rhs,
        "rhs_bar_state": rhs_bar_state,
        "rhs_bar_theta": rhs_bar_theta,
        "transition_mean": mean,
        "transition_mean_bar_state": mean_bar_state,
        "transition_mean_bar_theta": mean_bar_theta,
    }


def _make_transition_noise_tensor(args: argparse.Namespace) -> tf.Tensor:
    time_rows = []
    for time_index in range(int(args.time_steps)):
        batch_rows = []
        for seed in args.batch_seeds:
            batch_rows.append(
                tf.random.stateless_normal(
                    [int(args.num_particles), value_mod.STATE_DIM],
                    seed=value_mod._seed_pair(  # noqa: SLF001
                        int(seed),
                        tf.constant(1120 + int(time_index), dtype=tf.int32),
                    ),
                    dtype=DTYPE,
                )
            )
        time_rows.append(tf.stack(batch_rows, axis=0))
    return tf.stack(time_rows, axis=1)


def _require_manual_score_args(args: argparse.Namespace) -> None:
    if args.transport_plan_mode != "streaming":
        raise ValueError("predator-prey score repair requires streaming transport")
    if args.transport_ad_mode != "full":
        raise ValueError("predator-prey score repair requires transport_ad_mode='full'")
    if args.transport_gradient_mode not in {
        core_tf.MANUAL_STREAMING_FINITE_TRANSPORT_GRADIENT_MODE,
        core_tf.MANUAL_STREAMING_BLOCKWISE_VJP_FINITE_TRANSPORT_GRADIENT_MODE,
    }:
        raise ValueError("predator-prey score repair requires a manual streaming transport VJP")
    if args.transport_policy == "no-resampling":
        raise ValueError("predator-prey admitted score requires active relaxed Sinkhorn transport")


def _forward_transport_tf(
    *,
    post_flow: tf.Tensor,
    normalized_log_weights: tf.Tensor,
    mask: tf.Tensor,
    args: argparse.Namespace,
) -> tuple[tf.Tensor, tf.Tensor]:
    return p8p._manual_forward_transport_tf(  # noqa: SLF001
        post_flow=post_flow,
        normalized_log_weights=normalized_log_weights,
        mask=mask,
        args=args,
    )


def _transport_vjp_tf(
    *,
    post_flow: tf.Tensor,
    normalized_log_weights: tf.Tensor,
    mask: tf.Tensor,
    args: argparse.Namespace,
    upstream_particles: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    return p8p._manual_transport_vjp_tf(  # noqa: SLF001
        post_flow=post_flow,
        normalized_log_weights=normalized_log_weights,
        mask=mask,
        args=args,
        upstream_particles=upstream_particles,
    )


def _manual_value_and_score_from_components(
    args: argparse.Namespace,
    theta_values: list[float] | tuple[float, ...],
    *,
    return_score_decomposition: bool = False,
) -> dict[str, tf.Tensor]:
    """Historical reverse/manual predator-prey score diagnostic."""

    _configure_precision(args)
    _require_manual_score_args(args)
    if len(args.batch_seeds) != 1:
        raise ValueError("predator-prey score repair evaluates one seed at a time")
    theta = tf.constant([float(value) for value in theta_values], dtype=DTYPE)
    tensors, _semantics = value_mod._build_predator_prey_tensors(args)  # noqa: SLF001
    observations = tf.cast(tensors["observations"], DTYPE)
    fixed_resampling_mask = tf.convert_to_tensor(tensors["fixed_resampling_mask"], dtype=tf.bool)
    transition_covariance = tf.cast(tensors["transition_covariance"], DTYPE)
    observation_covariance = tf.cast(tensors["observation_covariance"], DTYPE)
    process_chol = tf.linalg.cholesky(transition_covariance)
    transition_noise = _make_transition_noise_tensor(args)
    particles = tf.cast(tensors["initial_particles"], DTYPE)
    batch_size = int(particles.shape[0])
    num_particles = int(particles.shape[1])
    state_dim = int(particles.shape[2])
    time_steps = int(observations.shape[0])
    if state_dim != value_mod.STATE_DIM:
        raise ValueError("predator-prey score route expects state_dim=2")

    identity_jacobian = tf.eye(value_mod.OBS_DIM, value_mod.STATE_DIM, dtype=DTYPE)
    h_jacobian = tf.tile(
        identity_jacobian[tf.newaxis, tf.newaxis, :, :],
        [batch_size, num_particles, 1, 1],
    )
    log_weights = core_tf.uniform_log_weights(batch_size, num_particles)
    log_likelihood = tf.zeros([batch_size], dtype=DTYPE)
    records: list[dict[str, Any]] = []

    for time_index in range(time_steps):
        observation = observations[time_index]
        prior_means, transition_aux = _predator_prey_transition_mean_with_aux_tf(
            theta,
            particles,
        )
        noise = transition_noise[:, time_index, :, :]
        pre_flow = prior_means + tf.einsum("bnd,bed->bne", noise, process_chol)
        residual = observation[tf.newaxis, tf.newaxis, :] - pre_flow
        flow, flow_aux = core_tf._batched_ledh_linearized_flow_with_aux_tf(  # noqa: SLF001
            pre_flow_particles=pre_flow,
            prior_means=prior_means,
            observation_jacobian=h_jacobian,
            observation_residual=residual,
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance,
        )
        post_flow = flow.post_flow_particles
        transition_log_density = core_tf._batched_gaussian_logpdf(  # noqa: SLF001
            post_flow - prior_means,
            transition_covariance,
        )
        predicted_observation = post_flow
        observation_log_density = core_tf._batched_gaussian_logpdf(  # noqa: SLF001
            predicted_observation - observation[tf.newaxis, tf.newaxis, :],
            observation_covariance,
        )
        corrected_log_weights = (
            log_weights
            + transition_log_density
            + observation_log_density
            - flow.pre_flow_log_density
            + flow.forward_log_det
        )
        weights, incremental = core_tf._normalize_log_weights(corrected_log_weights)  # noqa: SLF001
        normalized_log_weights = tf.math.log(
            tf.maximum(weights, core_tf._log_weight_floor())  # noqa: SLF001
        )
        mask = fixed_resampling_mask[:, time_index]
        next_particles, next_log_weights = _forward_transport_tf(
            post_flow=post_flow,
            normalized_log_weights=normalized_log_weights,
            mask=mask,
            args=args,
        )
        records.append(
            {
                "prior_means": prior_means,
                "post_flow": post_flow,
                "predicted_observation": predicted_observation,
                "observation": observation,
                "corrected_log_weights": corrected_log_weights,
                "normalized_log_weights": normalized_log_weights,
                "mask": mask,
                "transition_aux": transition_aux,
                "flow_aux": flow_aux,
            }
        )
        particles = next_particles
        log_weights = next_log_weights
        log_likelihood = log_likelihood + incremental

    bar_particles = tf.zeros_like(particles)
    bar_log_weights = tf.zeros_like(log_weights)
    per_seed_score = tf.zeros([batch_size, len(PARAMETER_NAMES)], dtype=DTYPE)
    component_scores = tf.zeros(
        [len(MANUAL_SCORE_COMPONENT_NAMES), batch_size, len(PARAMETER_NAMES)],
        dtype=DTYPE,
    )
    for record in reversed(records):
        mask = record["mask"]
        post_flow = record["post_flow"]
        normalized_log_weights = record["normalized_log_weights"]
        bar_post_transport, bar_normalized_from_transport = _transport_vjp_tf(
            post_flow=post_flow,
            normalized_log_weights=normalized_log_weights,
            mask=mask,
            args=args,
            upstream_particles=bar_particles,
        )
        inactive = tf.logical_not(mask)
        bar_post = bar_post_transport + tf.where(
            inactive[:, None, None],
            bar_particles,
            tf.zeros_like(bar_particles),
        )
        bar_normalized_log_weights = bar_normalized_from_transport + tf.where(
            inactive[:, None],
            bar_log_weights,
            tf.zeros_like(bar_log_weights),
        )
        bar_corrected, _weights, _incremental, _floor_active = (
            core_tf._normalize_log_weights_with_floor_vjp(  # noqa: SLF001
                record["corrected_log_weights"],
                bar_normalized_log_weights,
                tf.ones([batch_size], dtype=DTYPE),
            )
        )
        correction_bars = core_tf._log_weight_correction_vjp(bar_corrected)  # noqa: SLF001
        next_bar_log_weights = correction_bars["current_log_weights"]

        transition_vjp = core_tf._transition_gaussian_log_density_vjp(  # noqa: SLF001
            post_flow,
            record["prior_means"],
            transition_covariance,
            correction_bars["transition_log_density"],
        )
        observation_vjp = core_tf._observation_gaussian_log_density_vjp(  # noqa: SLF001
            record["predicted_observation"],
            record["observation"],
            observation_covariance,
            correction_bars["observation_log_density"],
            residual_convention="model_minus_observation",
        )
        bar_post += transition_vjp["x_next"] + observation_vjp["predicted_observation"]

        flow_vjp = core_tf._batched_ledh_linearized_flow_vjp(  # noqa: SLF001
            record["flow_aux"],
            bar_post,
            correction_bars["pre_flow_log_density"],
            correction_bars["forward_log_det"],
        )
        bar_pre_flow = flow_vjp.pre_flow_particles - flow_vjp.observation_residual
        bar_prior = (
            transition_vjp["transition_mean"]
            + flow_vjp.prior_means
            + bar_pre_flow
        )
        bar_particles, bar_theta = _predator_prey_transition_mean_vjp_tf(
            theta,
            record["transition_aux"],
            bar_prior,
        )
        per_seed_score += tf.tile(bar_theta[tf.newaxis, :], [batch_size, 1])
        if return_score_decomposition:
            component_scores += bar_theta[tf.newaxis, tf.newaxis, :]
        bar_log_weights = next_bar_log_weights

    result = {
        "objective": tf.reduce_mean(log_likelihood),
        "log_likelihood": log_likelihood,
        "gradient_tensor": tf.reduce_mean(per_seed_score, axis=0),
        "per_seed_gradient": per_seed_score,
        "score_route": PREDATOR_PREY_MEMORY_STYLE_SCORE_ROUTE_ID,
        "historical_manual_score_route": PREDATOR_PREY_MANUAL_SCORE_ROUTE_ID,
        "no_autodiff_score_route": True,
        "value_score_route_status": "same_route_value_score",
    }
    if return_score_decomposition:
        result["manual_score_components"] = component_scores
    return result


def _manual_value_only_from_components(
    args: argparse.Namespace,
    theta_values: list[float] | tuple[float, ...],
) -> dict[str, tf.Tensor]:
    """Forward-only value for same-scalar finite differences."""

    _configure_precision(args)
    _require_manual_score_args(args)
    if len(args.batch_seeds) != 1:
        raise ValueError("predator-prey score repair evaluates one seed at a time")
    theta = tf.constant([float(value) for value in theta_values], dtype=DTYPE)
    tensors, _semantics = value_mod._build_predator_prey_tensors(args)  # noqa: SLF001
    observations = tf.cast(tensors["observations"], DTYPE)
    fixed_resampling_mask = tf.convert_to_tensor(tensors["fixed_resampling_mask"], dtype=tf.bool)
    transition_covariance = tf.cast(tensors["transition_covariance"], DTYPE)
    observation_covariance = tf.cast(tensors["observation_covariance"], DTYPE)
    process_chol = tf.linalg.cholesky(transition_covariance)
    transition_noise = _make_transition_noise_tensor(args)
    particles = tf.cast(tensors["initial_particles"], DTYPE)
    batch_size = int(particles.shape[0])
    num_particles = int(particles.shape[1])
    identity_jacobian = tf.eye(value_mod.OBS_DIM, value_mod.STATE_DIM, dtype=DTYPE)
    h_jacobian = tf.tile(
        identity_jacobian[tf.newaxis, tf.newaxis, :, :],
        [batch_size, num_particles, 1, 1],
    )
    log_weights = core_tf.uniform_log_weights(batch_size, num_particles)
    log_likelihood = tf.zeros([batch_size], dtype=DTYPE)
    for time_index in range(int(observations.shape[0])):
        observation = observations[time_index]
        prior_means, _transition_aux = _predator_prey_transition_mean_with_aux_tf(
            theta,
            particles,
        )
        noise = transition_noise[:, time_index, :, :]
        pre_flow = prior_means + tf.einsum("bnd,bed->bne", noise, process_chol)
        residual = observation[tf.newaxis, tf.newaxis, :] - pre_flow
        flow, _flow_aux = core_tf._batched_ledh_linearized_flow_with_aux_tf(  # noqa: SLF001
            pre_flow_particles=pre_flow,
            prior_means=prior_means,
            observation_jacobian=h_jacobian,
            observation_residual=residual,
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance,
        )
        post_flow = flow.post_flow_particles
        transition_log_density = core_tf._batched_gaussian_logpdf(  # noqa: SLF001
            post_flow - prior_means,
            transition_covariance,
        )
        observation_log_density = core_tf._batched_gaussian_logpdf(  # noqa: SLF001
            post_flow - observation[tf.newaxis, tf.newaxis, :],
            observation_covariance,
        )
        corrected_log_weights = (
            log_weights
            + transition_log_density
            + observation_log_density
            - flow.pre_flow_log_density
            + flow.forward_log_det
        )
        weights, incremental = core_tf._normalize_log_weights(corrected_log_weights)  # noqa: SLF001
        normalized_log_weights = tf.math.log(
            tf.maximum(weights, core_tf._log_weight_floor())  # noqa: SLF001
        )
        mask = fixed_resampling_mask[:, time_index]
        particles, log_weights = _forward_transport_tf(
            post_flow=post_flow,
            normalized_log_weights=normalized_log_weights,
            mask=mask,
            args=args,
        )
        log_likelihood = log_likelihood + incremental
    return {
        "objective": tf.reduce_mean(log_likelihood),
        "log_likelihood": log_likelihood,
    }


def _normalize_log_weights_jvp(
    corrected_log_weights: tf.Tensor,
    d_corrected_log_weights: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    corrected_log_weights = tf.convert_to_tensor(corrected_log_weights, dtype=DTYPE)
    d_corrected_log_weights = tf.convert_to_tensor(d_corrected_log_weights, dtype=DTYPE)
    weights, incremental = core_tf._normalize_log_weights(corrected_log_weights)  # noqa: SLF001
    d_incremental = tf.reduce_sum(weights[:, :, None] * d_corrected_log_weights, axis=1)
    normalized_log_weights = tf.math.log(
        tf.maximum(weights, core_tf._log_weight_floor())  # noqa: SLF001
    )
    floor_active = weights <= core_tf._log_weight_floor()  # noqa: SLF001
    d_normalized = tf.where(
        floor_active[:, :, None],
        tf.zeros_like(d_corrected_log_weights),
        d_corrected_log_weights - d_incremental[:, None, :],
    )
    return normalized_log_weights, d_normalized, incremental, d_incremental


def _batched_gaussian_logpdf_jvp(
    residuals: tf.Tensor,
    covariance: tf.Tensor,
    d_residuals: tf.Tensor,
    d_covariance: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    residuals = tf.convert_to_tensor(residuals, dtype=DTYPE)
    covariance = tf.convert_to_tensor(covariance, dtype=DTYPE)
    d_residuals = tf.convert_to_tensor(d_residuals, dtype=DTYPE)
    d_covariance = tf.convert_to_tensor(d_covariance, dtype=DTYPE)
    chol = tf.linalg.cholesky(covariance)
    precision = tf.linalg.cholesky_solve(
        chol,
        tf.eye(int(covariance.shape[-1]), dtype=DTYPE)[tf.newaxis, :, :],
    )
    solved = tf.einsum("bij,bnj->bni", precision, residuals)
    value = core_tf._batched_gaussian_logpdf(residuals, covariance)  # noqa: SLF001
    covariance_bar_per_particle = tf.constant(0.5, dtype=DTYPE) * (
        tf.einsum("bni,bnj->bnij", solved, solved)
        - precision[:, tf.newaxis, :, :]
    )
    tangent = (
        -tf.reduce_sum(solved[:, :, :, tf.newaxis] * d_residuals, axis=2)
        + tf.reduce_sum(
            covariance_bar_per_particle[:, :, :, :, tf.newaxis]
            * d_covariance[:, tf.newaxis, :, :, :],
            axis=[2, 3],
        )
    )
    return value, tangent


def _cholesky_jvp_matrix(matrix: tf.Tensor, d_matrix: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    matrix = tf.convert_to_tensor(matrix, dtype=DTYPE)
    d_matrix = tf.convert_to_tensor(d_matrix, dtype=DTYPE)
    chol = tf.linalg.cholesky(matrix)
    tangent_columns = []
    for index in range(len(PARAMETER_NAMES)):
        tangent = d_matrix[..., index]
        inner = tf.linalg.triangular_solve(chol, tangent, lower=True)
        inner = tf.linalg.triangular_solve(chol, tf.linalg.matrix_transpose(inner), lower=True)
        inner = tf.linalg.matrix_transpose(inner)
        lower = tf.linalg.band_part(inner, -1, 0)
        diag = tf.linalg.diag(tf.linalg.diag_part(lower) * tf.constant(0.5, dtype=DTYPE))
        phi = lower - tf.linalg.diag(tf.linalg.diag_part(lower)) + diag
        tangent_columns.append(tf.matmul(chol, phi))
    return chol, tf.stack(tangent_columns, axis=-1)


def _filterflow_scale_jvp(particles: tf.Tensor, d_particles: tf.Tensor) -> tf.Tensor:
    particles = tf.convert_to_tensor(particles, dtype=DTYPE)
    d_particles = tf.convert_to_tensor(d_particles, dtype=DTYPE)
    num_particles = tf.cast(tf.shape(particles)[1], DTYPE)
    dimension = tf.cast(tf.shape(particles)[2], DTYPE)
    mean = tf.reduce_mean(particles, axis=1, keepdims=True)
    d_mean = tf.reduce_mean(d_particles, axis=1, keepdims=True)
    centered = particles - mean
    d_centered = d_particles - d_mean
    variance = tf.reduce_mean(centered * centered, axis=1)
    d_variance = (tf.constant(2.0, dtype=DTYPE) / num_particles) * tf.reduce_sum(
        centered[:, :, :, None] * d_centered,
        axis=1,
    )
    std = tf.sqrt(variance)
    safe_std = tf.where(std > 0.0, std, tf.ones_like(std))
    d_std = tf.where(
        std[:, :, None] > 0.0,
        d_variance / (tf.constant(2.0, dtype=DTYPE) * safe_std[:, :, None]),
        tf.zeros_like(d_variance),
    )
    diameter = tf.reduce_max(std, axis=1)
    max_mask = tf.cast(std == diameter[:, None], DTYPE)
    max_count = tf.reduce_sum(max_mask, axis=1, keepdims=True)
    d_diameter = tf.reduce_sum(
        d_std * max_mask[:, :, None] / max_count[:, :, None],
        axis=1,
    )
    active = tf.cast(diameter != 0.0, DTYPE)
    return tf.sqrt(dimension) * d_diameter * active[:, None]


def _filterflow_epsilon_start_jvp(scaled_x: tf.Tensor, d_scaled_x: tf.Tensor) -> tf.Tensor:
    scaled_x = tf.convert_to_tensor(scaled_x, dtype=DTYPE)
    d_scaled_x = tf.convert_to_tensor(d_scaled_x, dtype=DTYPE)
    max_value = tf.reduce_max(scaled_x, axis=[1, 2])
    min_value = tf.reduce_min(scaled_x, axis=[1, 2])
    coordinate_range = max_value - min_value
    max_mask = tf.cast(scaled_x == max_value[:, None, None], DTYPE)
    min_mask = tf.cast(scaled_x == min_value[:, None, None], DTYPE)
    max_count = tf.reduce_sum(max_mask, axis=[1, 2], keepdims=True)
    min_count = tf.reduce_sum(min_mask, axis=[1, 2], keepdims=True)
    d_max = tf.reduce_sum(
        d_scaled_x * max_mask[:, :, :, None] / max_count[:, :, :, None],
        axis=[1, 2],
    )
    d_min = tf.reduce_sum(
        d_scaled_x * min_mask[:, :, :, None] / min_count[:, :, :, None],
        axis=[1, 2],
    )
    active = tf.cast(
        coordinate_range * coordinate_range >= tf.constant(1.0e-6, DTYPE),
        DTYPE,
    )
    return tf.constant(2.0, dtype=DTYPE) * coordinate_range[:, None] * (d_max - d_min) * active[:, None]


def _compact_forward_transport_jvp_tf(
    *,
    post_flow: tf.Tensor,
    normalized_log_weights: tf.Tensor,
    d_post_flow: tf.Tensor,
    d_normalized_log_weights: tf.Tensor,
    mask: tf.Tensor,
    args: argparse.Namespace,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    if args.transport_policy == "no-resampling":
        return post_flow, normalized_log_weights, d_post_flow, d_normalized_log_weights
    if args.transport_gradient_mode != core_tf.MANUAL_STREAMING_FINITE_TRANSPORT_GRADIENT_MODE:
        raise ValueError("compact predator-prey score requires manual streaming finite transport")
    if args.transport_ad_mode != "full":
        raise ValueError("compact predator-prey score requires transport_ad_mode='full'")
    batch_size, num_particles, _state_dim = core_tf._static_shape(  # noqa: SLF001
        post_flow,
        "post_flow",
    )
    center = tf.reduce_mean(post_flow, axis=1, keepdims=True)
    d_center = tf.reduce_mean(d_post_flow, axis=1, keepdims=True)
    scale = annealed_transport_tf._filterflow_scale(post_flow)  # noqa: SLF001
    d_scale = _filterflow_scale_jvp(post_flow, d_post_flow)
    scaled_x = (post_flow - center) / scale[:, None, None]
    d_scaled_x = (
        (d_post_flow - d_center) / scale[:, None, None, None]
        - (post_flow - center)[:, :, :, None]
        * d_scale[:, None, None, :]
        / (scale[:, None, None, None] * scale[:, None, None, None])
    )
    epsilon = tf.convert_to_tensor(args.sinkhorn_epsilon, dtype=DTYPE)
    epsilon0 = annealed_transport_tf._filterflow_epsilon_start(scaled_x)  # noqa: SLF001
    d_epsilon0 = _filterflow_epsilon_start_jvp(scaled_x, d_scaled_x)
    scaling = tf.convert_to_tensor(args.annealed_scaling, dtype=DTYPE)
    steps = core_tf._manual_dense_finite_steps(args.sinkhorn_iterations)  # noqa: SLF001
    transported, d_transported, _row_residual = (
        annealed_transport_tf._filterflow_manual_streaming_finite_transport_value_and_jvp_total(  # noqa: SLF001
            scaled_x,
            post_flow,
            normalized_log_weights,
            d_scaled_x,
            d_post_flow,
            d_normalized_log_weights,
            d_epsilon0,
            epsilon,
            epsilon0,
            scaling,
            steps=steps,
            row_chunk_size=args.row_chunk_size,
            col_chunk_size=args.col_chunk_size,
        )
    )
    raw_transport = core_tf.batched_annealed_transport_core_tf(
        post_flow,
        normalized_log_weights,
        mask,
        epsilon=args.sinkhorn_epsilon,
        scaling=args.annealed_scaling,
        convergence_threshold=args.annealed_convergence_threshold,
        max_iterations=args.sinkhorn_iterations,
        transport_gradient_mode="raw",
        transport_plan_mode="streaming",
        transport_ad_mode="stabilized",
        row_chunk_size=args.row_chunk_size,
        col_chunk_size=args.col_chunk_size,
    )
    uniform_log_weights = core_tf.uniform_log_weights(batch_size, num_particles)
    d_uniform = tf.zeros_like(d_normalized_log_weights)
    next_d_particles = tf.where(mask[:, None, None, None], d_transported, d_post_flow)
    next_d_log_weights = tf.where(mask[:, None, None], d_uniform, d_normalized_log_weights)
    del transported, uniform_log_weights
    return raw_transport.particles, raw_transport.log_weights, next_d_particles, next_d_log_weights


def _predator_prey_rhs_jvp_tf(
    theta: tf.Tensor,
    state: tf.Tensor,
    d_state: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    theta = _as_theta(theta)
    state = _as_state_matrix(state)
    d_state = tf.convert_to_tensor(d_state, dtype=DTYPE)
    rhs = _predator_prey_rhs_tf(theta, state)
    r, k_capacity, a_half, s_rate, u_rate, v_rate = tf.unstack(theta)
    prey = state[..., 0]
    predator = state[..., 1]
    d_prey_state = d_state[..., 0, :]
    d_predator_state = d_state[..., 1, :]
    eye = tf.eye(len(PARAMETER_NAMES), dtype=DTYPE)
    d_r, d_k, d_a, d_s, d_u, d_v = tf.unstack(eye, axis=0)
    denominator = a_half + prey
    denominator_sq = tf.square(denominator)
    interaction = prey * predator / denominator
    d_interaction = (
        predator[..., None] * a_half / denominator_sq[..., None] * d_prey_state
        + prey[..., None] / denominator[..., None] * d_predator_state
        - prey[..., None]
        * predator[..., None]
        / denominator_sq[..., None]
        * d_a
    )
    logistic = prey * (tf.constant(1.0, dtype=DTYPE) - prey / k_capacity)
    d_logistic = (
        (tf.constant(1.0, dtype=DTYPE) - tf.constant(2.0, dtype=DTYPE) * prey / k_capacity)[..., None]
        * d_prey_state
        + tf.square(prey)[..., None] / tf.square(k_capacity) * d_k
    )
    d_rhs_prey = (
        d_r * logistic[..., None]
        + r * d_logistic
        - d_s * interaction[..., None]
        - s_rate * d_interaction
    )
    d_rhs_predator = (
        d_u * interaction[..., None]
        + u_rate * d_interaction
        - d_v * predator[..., None]
        - v_rate * d_predator_state
    )
    return rhs, tf.stack([d_rhs_prey, d_rhs_predator], axis=-2)


def _predator_prey_transition_mean_jvp_tf(
    theta: tf.Tensor,
    state: tf.Tensor,
    d_state: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    model = highdim.p30_predator_prey_fixture_model()
    theta = _as_theta(theta)
    running = _as_state_matrix(state)
    d_running = tf.convert_to_tensor(d_state, dtype=DTYPE)
    step = tf.cast(model.delta, DTYPE) / tf.cast(int(model._rk4_substeps), DTYPE)
    half = tf.constant(0.5, dtype=DTYPE)
    sixth = step / tf.constant(6.0, dtype=DTYPE)
    for _ in range(int(model._rk4_substeps)):
        k1, d_k1 = _predator_prey_rhs_jvp_tf(theta, running, d_running)
        k2_input = running + half * step * k1
        d_k2_input = d_running + half * step * d_k1
        k2, d_k2 = _predator_prey_rhs_jvp_tf(theta, k2_input, d_k2_input)
        k3_input = running + half * step * k2
        d_k3_input = d_running + half * step * d_k2
        k3, d_k3 = _predator_prey_rhs_jvp_tf(theta, k3_input, d_k3_input)
        k4_input = running + step * k3
        d_k4_input = d_running + step * d_k3
        k4, d_k4 = _predator_prey_rhs_jvp_tf(theta, k4_input, d_k4_input)
        running = running + sixth * (
            k1 + tf.constant(2.0, dtype=DTYPE) * k2 + tf.constant(2.0, dtype=DTYPE) * k3 + k4
        )
        d_running = d_running + sixth * (
            d_k1
            + tf.constant(2.0, dtype=DTYPE) * d_k2
            + tf.constant(2.0, dtype=DTYPE) * d_k3
            + d_k4
        )
    return running, d_running


def _compact_ledh_flow_jvp_tf(
    *,
    pre_flow: tf.Tensor,
    d_pre_flow: tf.Tensor,
    prior_means: tf.Tensor,
    d_prior_means: tf.Tensor,
    observation: tf.Tensor,
    observation_jacobian: tf.Tensor,
    observation_covariance: tf.Tensor,
    transition_covariance: tf.Tensor,
) -> tuple[core_tf.LEDHFlowTensors, tf.Tensor, tf.Tensor, tf.Tensor]:
    residual = observation[tf.newaxis, tf.newaxis, :] - pre_flow
    d_residual = -d_pre_flow
    flow, aux = core_tf._batched_ledh_linearized_flow_with_aux_tf(  # noqa: SLF001
        pre_flow_particles=pre_flow,
        prior_means=prior_means,
        observation_jacobian=observation_jacobian,
        observation_residual=residual,
        transition_covariance=transition_covariance,
        observation_covariance=observation_covariance,
    )
    batch_size, _num_particles, state_dim = core_tf._static_shape(pre_flow, "pre_flow")  # noqa: SLF001
    obs_dim = value_mod.OBS_DIM
    param_dim = len(PARAMETER_NAMES)
    prior_precision = aux.prior_precision
    obs_precision = aux.obs_precision
    post_covariance = aux.post_covariance
    info = aux.info
    d_post_precision = tf.zeros(
        [batch_size, tf.shape(pre_flow)[1], state_dim, state_dim, param_dim],
        dtype=DTYPE,
    )
    d_post_covariance = -tf.einsum(
        "bnij,bnjkp,bnkl->bnilp",
        post_covariance,
        d_post_precision,
        post_covariance,
    )
    d_pseudo_observation = (
        tf.einsum("bnod,bndp->bnop", observation_jacobian, d_pre_flow)
        + d_residual
    )
    d_info = (
        tf.einsum("bde,bnep->bndp", prior_precision, d_prior_means)
        + tf.einsum("bnod,boq,bnqp->bndp", observation_jacobian, obs_precision, d_pseudo_observation)
    )
    d_post_mean = (
        tf.einsum("bndep,bne->bndp", d_post_covariance, info)
        + tf.einsum("bnde,bnep->bndp", post_covariance, d_info)
    )
    post_chol, d_post_chol = _cholesky_jvp_matrix(post_covariance, d_post_covariance)
    prior_inv = aux.prior_inv
    affine_transform = aux.affine_transform
    d_affine_transform = tf.einsum("bnijp,bjk->bnikp", d_post_chol, prior_inv)
    delta = pre_flow - prior_means
    d_delta = d_pre_flow - d_prior_means
    d_post_flow = (
        d_post_mean
        + tf.einsum("bnijp,bnj->bnip", d_affine_transform, delta)
        + tf.einsum("bnij,bnjp->bnip", affine_transform, d_delta)
    )
    d_logdet_post_chol = tf.reduce_sum(
        tf.einsum("bniip->bnip", d_post_chol)
        / tf.linalg.diag_part(post_chol)[:, :, :, None],
        axis=2,
    )
    _pre_density, d_pre_density = _batched_gaussian_logpdf_jvp(
        pre_flow - prior_means,
        transition_covariance,
        d_pre_flow - d_prior_means,
        tf.zeros([batch_size, state_dim, state_dim, param_dim], dtype=DTYPE),
    )
    del obs_dim
    return flow, d_post_flow, d_pre_density, d_logdet_post_chol


def _compact_value_and_score_from_components(
    args: argparse.Namespace,
    theta_values: list[float] | tuple[float, ...],
) -> dict[str, tf.Tensor]:
    """Predator-prey same-scalar score with compact forward sensitivity."""

    _configure_precision(args)
    _require_manual_score_args(args)
    theta = tf.constant([float(value) for value in theta_values], dtype=DTYPE)
    tensors, _semantics = value_mod._build_predator_prey_tensors(args)  # noqa: SLF001
    observations = tf.cast(tensors["observations"], DTYPE)
    fixed_resampling_mask = tf.convert_to_tensor(tensors["fixed_resampling_mask"], dtype=tf.bool)
    transition_covariance = tf.cast(tensors["transition_covariance"], DTYPE)
    observation_covariance = tf.cast(tensors["observation_covariance"], DTYPE)
    process_chol = tf.linalg.cholesky(transition_covariance)
    transition_noise = _make_transition_noise_tensor(args)
    particles = tf.cast(tensors["initial_particles"], DTYPE)
    batch_size = int(particles.shape[0])
    num_particles = int(particles.shape[1])
    state_dim = int(particles.shape[2])
    param_dim = len(PARAMETER_NAMES)
    identity_jacobian = tf.eye(value_mod.OBS_DIM, value_mod.STATE_DIM, dtype=DTYPE)
    h_jacobian = tf.tile(
        identity_jacobian[tf.newaxis, tf.newaxis, :, :],
        [batch_size, num_particles, 1, 1],
    )
    d_particles = tf.zeros([batch_size, num_particles, state_dim, param_dim], dtype=DTYPE)
    log_weights = core_tf.uniform_log_weights(batch_size, num_particles)
    d_log_weights = tf.zeros([batch_size, num_particles, param_dim], dtype=DTYPE)
    log_likelihood = tf.zeros([batch_size], dtype=DTYPE)
    d_log_likelihood = tf.zeros([batch_size, param_dim], dtype=DTYPE)

    for time_index in range(int(observations.shape[0])):
        observation = observations[time_index]
        prior_means, d_prior_means = _predator_prey_transition_mean_jvp_tf(
            theta,
            particles,
            d_particles,
        )
        noise = transition_noise[:, time_index, :, :]
        pre_flow = prior_means + tf.einsum("bnd,bed->bne", noise, process_chol)
        d_pre_flow = d_prior_means
        flow, d_post_flow, d_pre_log_density, d_forward_log_det = _compact_ledh_flow_jvp_tf(
            pre_flow=pre_flow,
            d_pre_flow=d_pre_flow,
            prior_means=prior_means,
            d_prior_means=d_prior_means,
            observation=observation,
            observation_jacobian=h_jacobian,
            observation_covariance=observation_covariance,
            transition_covariance=transition_covariance,
        )
        post_flow = flow.post_flow_particles
        transition_log_density, d_transition_log_density = _batched_gaussian_logpdf_jvp(
            post_flow - prior_means,
            transition_covariance,
            d_post_flow - d_prior_means,
            tf.zeros([batch_size, state_dim, state_dim, param_dim], dtype=DTYPE),
        )
        observation_log_density, d_observation_log_density = _batched_gaussian_logpdf_jvp(
            post_flow - observation[tf.newaxis, tf.newaxis, :],
            observation_covariance,
            d_post_flow,
            tf.zeros([batch_size, value_mod.OBS_DIM, value_mod.OBS_DIM, param_dim], dtype=DTYPE),
        )
        corrected_log_weights = (
            log_weights
            + transition_log_density
            + observation_log_density
            - flow.pre_flow_log_density
            + flow.forward_log_det
        )
        d_corrected_log_weights = (
            d_log_weights
            + d_transition_log_density
            + d_observation_log_density
            - d_pre_log_density
            + d_forward_log_det
        )
        (
            normalized_log_weights,
            d_normalized_log_weights,
            incremental,
            d_incremental,
        ) = _normalize_log_weights_jvp(corrected_log_weights, d_corrected_log_weights)
        mask = fixed_resampling_mask[:, time_index]
        particles, log_weights, d_particles, d_log_weights = _compact_forward_transport_jvp_tf(
            post_flow=post_flow,
            normalized_log_weights=normalized_log_weights,
            d_post_flow=d_post_flow,
            d_normalized_log_weights=d_normalized_log_weights,
            mask=mask,
            args=args,
        )
        log_likelihood = log_likelihood + incremental
        d_log_likelihood = d_log_likelihood + d_incremental

    return {
        "objective": tf.reduce_mean(log_likelihood),
        "log_likelihood": log_likelihood,
        "gradient_tensor": tf.reduce_mean(d_log_likelihood, axis=0),
        "per_seed_gradient": d_log_likelihood,
        "score_route": PREDATOR_PREY_COMPACT_SCORE_ROUTE_ID,
        "no_autodiff_score_route": True,
        "value_score_route_status": "same_route_value_score",
        "batch_seeds": [int(seed) for seed in args.batch_seeds],
        "time_steps": int(args.time_steps),
        "num_particles": int(args.num_particles),
        "transport": {
            "value_core_mode": "compact_forward_sensitivity_same_scalar",
            "transport_plan_mode": args.transport_plan_mode,
            "transport_ad_mode": args.transport_ad_mode,
            "gradient_mode": args.transport_gradient_mode,
            "row_chunk_size": int(args.row_chunk_size),
            "col_chunk_size": int(args.col_chunk_size),
            "particle_chunk_size": int(args.particle_chunk_size),
            "sinkhorn_iterations": int(args.sinkhorn_iterations),
            "sinkhorn_epsilon": float(args.sinkhorn_epsilon),
            "dense_transport_matrix_materialized": False,
        },
    }


def _single_seed_args(args: argparse.Namespace, seed: int) -> argparse.Namespace:
    clone = copy.copy(args)
    clone.batch_seeds = [int(seed)]
    return clone


def _manual_value_and_score_across_seeds(
    args: argparse.Namespace,
    theta_values: list[float] | tuple[float, ...],
) -> dict[str, tf.Tensor]:
    """Historical reverse/manual across-seed score diagnostic only."""

    seed_results = [
        _manual_value_and_score_from_components(
            _single_seed_args(args, seed),
            theta_values,
        )
        for seed in args.batch_seeds
    ]
    log_likelihood = tf.concat([result["log_likelihood"] for result in seed_results], axis=0)
    per_seed_gradient = tf.stack(
        [tf.reshape(result["gradient_tensor"], [-1]) for result in seed_results],
        axis=0,
    )
    return {
        "objective": tf.reduce_mean(log_likelihood),
        "log_likelihood": log_likelihood,
        "gradient_tensor": tf.reduce_mean(per_seed_gradient, axis=0),
        "per_seed_gradient": per_seed_gradient,
        "score_route": PREDATOR_PREY_MEMORY_STYLE_SCORE_ROUTE_ID,
        "historical_manual_score_route": PREDATOR_PREY_MANUAL_SCORE_ROUTE_ID,
        "historical_compact_score_route": PREDATOR_PREY_COMPACT_SCORE_ROUTE_ID,
        "no_autodiff_score_route": True,
        "value_score_route_status": "same_route_value_score",
    }


def _manual_objective_across_seeds(
    args: argparse.Namespace,
    theta_values: list[float] | tuple[float, ...],
) -> tf.Tensor:
    return _value_objective_across_seeds(args, theta_values)


def _value_objective_across_seeds(
    args: argparse.Namespace,
    theta_values: list[float] | tuple[float, ...],
) -> tf.Tensor:
    """Value-only same-scalar objective used for finite differences."""

    seed_values = [
        _manual_value_only_from_components(_single_seed_args(args, seed), theta_values)[
            "log_likelihood"
        ]
        for seed in args.batch_seeds
    ]
    return tf.reduce_mean(tf.concat(seed_values, axis=0))


def _coordinate_fd_score_diagnostic(
    args: argparse.Namespace,
    theta_values: list[float] | tuple[float, ...],
    *,
    fd_step: float = 1.0e-4,
    atol: float = 5.0e-3,
    rtol: float = 5.0e-3,
) -> dict[str, Any]:
    precision = _configure_precision(args)
    theta = tf.constant([float(value) for value in theta_values], dtype=DTYPE)
    step = tf.constant(float(fd_step), dtype=DTYPE)
    base = _compact_value_and_score_from_components(args, theta.numpy().tolist())
    fd_values = []
    for index in range(len(PARAMETER_NAMES)):
        basis = tf.one_hot(index, len(PARAMETER_NAMES), dtype=DTYPE)
        plus_objective = _value_objective_across_seeds(
            args,
            (theta + step * basis).numpy().tolist(),
        )
        minus_objective = _value_objective_across_seeds(
            args,
            (theta - step * basis).numpy().tolist(),
        )
        fd_values.append((plus_objective - minus_objective) / (2.0 * step))
    fd_score = tf.stack(fd_values)
    score = tf.convert_to_tensor(base["gradient_tensor"], dtype=DTYPE)
    abs_error = tf.abs(score - fd_score)
    rel_error = abs_error / tf.maximum(
        tf.maximum(tf.abs(score), tf.abs(fd_score)),
        tf.constant(1.0e-12, dtype=DTYPE),
    )
    max_abs_error = tf.reduce_max(abs_error)
    max_rel_error = tf.reduce_max(rel_error)
    passed = bool(
        (
            max_abs_error <= tf.constant(float(atol), dtype=DTYPE)
            or max_rel_error <= tf.constant(float(rtol), dtype=DTYPE)
        ).numpy()
    )
    return {
        "status": "pass" if passed else "fail",
        "score": score,
        "fd_score": fd_score,
        "max_abs_error": max_abs_error,
        "max_rel_error": max_rel_error,
        "per_coordinate_abs_error": abs_error,
        "per_coordinate_rel_error": rel_error,
        "atol": float(atol),
        "rtol": float(rtol),
        "base": base,
        "parameter_names": list(PARAMETER_NAMES),
        "fd_step": float(step.numpy()),
        "score_precision": _score_precision_metadata(precision),
    }


def _score_artifact_from_diagnostic(
    diagnostic: dict[str, Any],
    *,
    source_value_artifact: dict[str, Any],
    source_value_artifact_path: str,
    require_all_parameter_correctness: bool = False,
    memory_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    value_core = validate_ledh_forward_scalar_artifact(
        source_value_artifact,
        expected_row_id=PREDATOR_PREY_ROW_ID,
        require_admitted=True,
    )
    base_raw = diagnostic.get("base")
    if not isinstance(base_raw, Mapping):
        raise ValueError("predator-prey compact diagnostic must include base mapping")
    base = base_raw
    if base.get("score_route") != PREDATOR_PREY_COMPACT_SCORE_ROUTE_ID:
        raise ValueError("predator-prey compact diagnostic must use compact score route")
    if base.get("no_autodiff_score_route") is not True:
        raise ValueError("predator-prey compact diagnostic must declare no_autodiff_score_route")
    if base.get("value_score_route_status") != LEDH_SCORE_VALUE_ROUTE_STATUS_SAME:
        raise ValueError("predator-prey compact diagnostic must be same_route_value_score")
    if tuple(diagnostic.get("parameter_names", ())) != tuple(PARAMETER_NAMES):
        raise ValueError("predator-prey compact diagnostic parameter_names must match parameter order")
    score = tf.convert_to_tensor(base["gradient_tensor"], dtype=DTYPE)
    memory = dict(memory_diagnostics or {})
    memory_pass = bool(memory.get("n10000_memory_pass") is True)
    if "source" not in memory and memory:
        memory["source"] = "trusted_gpu_score_memory_artifact"
    if require_all_parameter_correctness:
        if diagnostic.get("status") != "pass":
            raise ValueError("predator-prey all-parameter correctness status must pass")
        if tuple(diagnostic.get("parameter_names", PARAMETER_NAMES)) != tuple(PARAMETER_NAMES):
            raise ValueError("predator-prey all-parameter correctness parameter_names mismatch")
        if not memory_pass:
            raise ValueError("predator-prey full admission requires N=10000 memory pass")
        if int(base.get("num_particles", -1)) != int(value_core["num_particles"]):
            raise ValueError("predator-prey compact full admission requires N=10000 diagnostic shape")
        if int(base.get("time_steps", -1)) != int(value_core["time_steps"]):
            raise ValueError("predator-prey compact full admission requires full time_steps")
        if tuple(int(seed) for seed in base.get("batch_seeds", ())) != tuple(
            int(seed) for seed in value_core["batch_seeds"]
        ):
            raise ValueError("predator-prey compact full admission requires full batch_seeds")
    score_correctness = {
        "kind": "same_scalar_finite_difference",
        "status": str(diagnostic.get("status", "fail")),
        "max_abs_error": float(
            tf.convert_to_tensor(diagnostic["max_abs_error"], dtype=DTYPE).numpy()
        ),
        "max_rel_error": float(
            tf.convert_to_tensor(diagnostic["max_rel_error"], dtype=DTYPE).numpy()
        ),
    }
    for optional_key in ("fd_step", "atol", "rtol"):
        if optional_key in diagnostic:
            score_correctness[optional_key] = float(diagnostic[optional_key])
    artifact = {
        "schema_version": LEDH_SCORE_ARTIFACT_SCHEMA_VERSION,
        "row_id": PREDATOR_PREY_ROW_ID,
        "source_value_artifact": source_value_artifact_path,
        "score_target_kind": LEDH_SCORE_TARGET_KIND_REALIZED_FINITE_N_ESTIMATOR,
        "target_scalar": LEDH_TARGET_SCALAR_OBSERVED_DATA_LOG_LIKELIHOOD,
        "target_output_tensor_field": LEDH_OUTPUT_TENSOR_FIELD_LOG_LIKELIHOOD,
        "target_observation_policy": value_core["target_observation_policy"],
        "theta_coordinate_system": value_core["theta_coordinate_system"],
        "score_parameter_names": list(PARAMETER_NAMES),
        "score": [float(value) for value in score.numpy().reshape(-1)],
        "score_derivative_provenance": str(
            base.get("score_route", PREDATOR_PREY_COMPACT_SCORE_ROUTE_ID)
        ),
        "value_score_route_status": LEDH_SCORE_VALUE_ROUTE_STATUS_SAME,
        "value_score_same_transport_algorithm": True,
        "no_autodiff_score_route": True,
        "uses_gradient_tape": False,
        "uses_forward_accumulator": False,
        "uses_stopped_partial_derivative": False,
        "score_correctness": score_correctness,
        "score_admission_status": (
            LEDH_SCORE_ADMISSION_STATUS_FULL
            if require_all_parameter_correctness
            else LEDH_SCORE_ADMISSION_STATUS_TINY
            if diagnostic.get("status") == "pass"
            else LEDH_SCORE_ADMISSION_STATUS_BLOCKED_NOT_RUN
        ),
        "score_precision": _score_precision_metadata(
            diagnostic.get("score_precision", {})
        ),
        "memory_diagnostics": memory,
    }
    if require_all_parameter_correctness:
        validate_ledh_score_artifact(
            artifact,
            source_value_artifact=source_value_artifact,
            expected_row_id=PREDATOR_PREY_ROW_ID,
            require_admitted=True,
        )
    return artifact


def _parse_int_csv(value: str) -> list[int]:
    parsed = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if not parsed:
        raise ValueError("expected at least one integer seed")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--batch-seeds", default="81120")
    parser.add_argument("--time-steps", type=int, default=1)
    parser.add_argument("--num-particles", type=int, default=2)
    parser.add_argument("--transport-policy", choices=("active-all", "active-odd", "no-resampling"), default="active-all")
    parser.add_argument("--sinkhorn-iterations", type=int, default=1)
    parser.add_argument("--sinkhorn-epsilon", type=float, default=1.0)
    parser.add_argument("--annealed-scaling", type=float, default=0.9)
    parser.add_argument("--annealed-convergence-threshold", type=float, default=1.0e-3)
    parser.add_argument("--transport-plan-mode", choices=("streaming", "dense"), default="streaming")
    parser.add_argument(
        "--transport-gradient-mode",
        choices=(
            core_tf.MANUAL_STREAMING_FINITE_TRANSPORT_GRADIENT_MODE,
            core_tf.MANUAL_STREAMING_BLOCKWISE_VJP_FINITE_TRANSPORT_GRADIENT_MODE,
        ),
        default=core_tf.MANUAL_STREAMING_FINITE_TRANSPORT_GRADIENT_MODE,
    )
    parser.add_argument(
        "--transport-ad-mode",
        choices=("full",),
        default="full",
    )
    parser.add_argument("--row-chunk-size", type=int, default=2)
    parser.add_argument("--col-chunk-size", type=int, default=2)
    parser.add_argument("--particle-chunk-size", type=int, default=2)
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float32")
    parser.add_argument("--tf32-mode", choices=("default", "enabled", "disabled"), default="enabled")
    parser.add_argument("--fd-step", type=float, default=1.0e-4)
    parser.add_argument("--score-fd-atol", type=float, default=5.0e-3)
    parser.add_argument("--score-fd-rtol", type=float, default=5.0e-3)
    parser.add_argument("--source-value-artifact", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output", default=None)
    parser.add_argument("--admit-full", action="store_true")
    parser.add_argument("--memory-budget-mib", type=float, default=14000.0)
    args = parser.parse_args()
    args.batch_seeds = _parse_int_csv(args.batch_seeds)
    return args


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unavailable"


def _gpu_memory_peak_mib() -> float | None:
    try:
        info = tf.config.experimental.get_memory_info("GPU:0")
    except (RuntimeError, ValueError):
        return None
    peak = info.get("peak")
    if peak is None:
        return None
    return float(peak) / (1024.0 * 1024.0)


def _write_markdown(path: Path, artifact: dict[str, Any], json_path: Path) -> None:
    lines = [
        "# Predator-Prey Same-Target LEDH Score",
        "",
        f"- JSON artifact: `{json_path}`",
        f"- Row: `{artifact['row_id']}`",
        f"- Score admission status: `{artifact['score_admission_status']}`",
        f"- Correctness: `{artifact['score_correctness']}`",
        f"- Memory diagnostics: `{artifact['memory_diagnostics']}`",
        "",
        "## Nonclaims",
        "",
        "- This artifact is not HMC readiness evidence.",
        "- This artifact is not posterior correctness evidence.",
        "- This artifact is not a scientific superiority claim.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    source_path = Path(args.source_value_artifact)
    source_value = json.loads(source_path.read_text(encoding="utf-8"))
    start = time.perf_counter()
    diagnostic = _coordinate_fd_score_diagnostic(
        args,
        list(TRUTH_THETA),
        fd_step=float(args.fd_step),
        atol=float(args.score_fd_atol),
        rtol=float(args.score_fd_rtol),
    )
    elapsed = time.perf_counter() - start
    diagnostic["parameter_names"] = list(PARAMETER_NAMES)
    peak_mib = _gpu_memory_peak_mib()
    memory_pass = (
        bool(args.admit_full)
        and peak_mib is not None
        and peak_mib <= float(args.memory_budget_mib)
    )
    memory = {
        "n10000_memory_pass": memory_pass,
        "peak_mib": peak_mib,
        "budget_mib": float(args.memory_budget_mib),
    }
    artifact = _score_artifact_from_diagnostic(
        diagnostic,
        source_value_artifact=source_value,
        source_value_artifact_path=str(source_path),
        require_all_parameter_correctness=bool(args.admit_full),
        memory_diagnostics=memory,
    )
    artifact.update(
        {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "host": platform.node(),
            "git_commit": _git_commit(),
            "elapsed_seconds": elapsed,
            "shape": {
                "time_steps": int(args.time_steps),
                "num_particles": int(args.num_particles),
                "batch_seed_count": len(args.batch_seeds),
            },
            "batch_seeds": [int(seed) for seed in args.batch_seeds],
            "fd_step": float(args.fd_step),
            "per_coordinate_fd_score": [
                float(value)
                for value in tf.convert_to_tensor(diagnostic["fd_score"], dtype=DTYPE)
                .numpy()
                .reshape(-1)
            ],
            "per_coordinate_abs_error": [
                float(value)
                for value in tf.convert_to_tensor(
                    diagnostic["per_coordinate_abs_error"],
                    dtype=DTYPE,
                )
                .numpy()
                .reshape(-1)
            ],
            "nonclaims": [
                "not HMC readiness evidence",
                "not posterior correctness evidence",
                "not scientific superiority evidence",
                "not runtime ranking evidence",
            ],
        }
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_output is not None:
        _write_markdown(Path(args.markdown_output), artifact, output_path)
    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
