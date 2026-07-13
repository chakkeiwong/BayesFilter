"""Same-target generalized-SV LEDH score runner.

This module differentiates the finite-N generalized-SV LEDH value scalar used by
``benchmark_ledh_same_target_generalized_sv_value``.  The target likelihood is
the raw observation model

``y_t | x_t ~ Normal(0, exp(tau * x_t))``.

The log-square Gaussianized observation is used only as the LEDH proposal
surface.  The score route is compact forward sensitivity: it carries particles,
log weights, particle tangents, log-weight tangents, log likelihood, and
log-likelihood tangents forward.  Historical reverse-record/manual-total-VJP
routes remain diagnostic only and are not used here.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence


_PRE_DEVICE_SCOPE = os.environ.get("BAYESFILTER_LEDHD_SCORE_DEVICE_SCOPE")
if _PRE_DEVICE_SCOPE == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.highdim.ledh_forward_contract import (
    GENERALIZED_SV_PARAMETER_ORDER,
    GENERALIZED_SV_ROW_ID,
    LEDH_OUTPUT_TENSOR_FIELD_LOG_LIKELIHOOD,
    LEDH_TARGET_SCALAR_OBSERVED_DATA_LOG_LIKELIHOOD,
    validate_ledh_forward_scalar_artifact,
)
from bayesfilter.highdim.ledh_score_contract import (
    LEDH_SCORE_ADMISSION_STATUS_BLOCKED_NOT_RUN,
    LEDH_SCORE_ADMISSION_STATUS_FULL,
    LEDH_SCORE_ADMISSION_STATUS_TINY,
    LEDH_SCORE_ARTIFACT_SCHEMA_VERSION,
    LEDH_SCORE_COMPACT_GENERALIZED_SV_PROVENANCE,
    LEDH_SCORE_TARGET_KIND_REALIZED_FINITE_N_ESTIMATOR,
    LEDH_SCORE_VALUE_ROUTE_STATUS_SAME,
    validate_ledh_score_artifact,
)
from docs.benchmarks import benchmark_ledh_same_target_actual_sv_score as actual_score
from docs.benchmarks import benchmark_ledh_same_target_generalized_sv_value as value_mod
from experiments.dpf_implementation.tf_tfp.filters import (
    experimental_batched_ledh_pfpf_ot_streaming_tf as streaming_tf,
)
from experiments.dpf_implementation.tf_tfp.filters import (
    experimental_batched_ledh_pfpf_ot_tf as core_tf,
)
from experiments.dpf_implementation.tf_tfp.resampling import annealed_transport_tf


DTYPE = tf.float64
PARAMETER_NAMES = GENERALIZED_SV_PARAMETER_ORDER
TRUTH_THETA = tuple(value_mod.TRUTH_THETA)
GENERALIZED_SV_COMPACT_SCORE_ROUTE_ID = LEDH_SCORE_COMPACT_GENERALIZED_SV_PROVENANCE
_STD_NORMAL64 = tfp.distributions.Normal(
    loc=tf.constant(0.0, dtype=tf.float64),
    scale=tf.constant(1.0, dtype=tf.float64),
)


def _configure_precision(args: argparse.Namespace) -> dict[str, Any]:
    global DTYPE
    DTYPE = tf.float64 if getattr(args, "dtype", "float64") == "float64" else tf.float32
    value_mod.DTYPE = DTYPE
    actual_score.DTYPE = DTYPE
    core_tf.DTYPE = DTYPE
    streaming_tf.DTYPE = DTYPE
    annealed_transport_tf.DTYPE = DTYPE
    if getattr(args, "tf32_mode", "default") != "default":
        tf.config.experimental.enable_tensor_float_32_execution(args.tf32_mode == "enabled")
    metadata = core_tf.precision_policy_metadata()
    metadata.update(
        {
            "dtype": getattr(args, "dtype", "float64"),
            "tf_dtype": DTYPE.name,
            "tf32_mode": getattr(args, "tf32_mode", "default"),
            "tf32_execution_enabled": bool(
                tf.config.experimental.tensor_float_32_execution_enabled()
            ),
        }
    )
    return metadata


def _as_theta(theta: tf.Tensor | Sequence[float]) -> tf.Tensor:
    tensor = tf.convert_to_tensor(theta, dtype=DTYPE)
    if tensor.shape.rank != 1 or int(tensor.shape[0]) != len(PARAMETER_NAMES):
        raise ValueError("generalized-SV theta must have shape [3]")
    return tensor


def _gamma_tau_mu(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    theta = _as_theta(theta)
    z_gamma = theta[0]
    log_tau = theta[1]
    mu = theta[2]
    gamma = tf.cast(_STD_NORMAL64.cdf(tf.cast(z_gamma, tf.float64)), DTYPE)
    dgamma_dz = tf.cast(_STD_NORMAL64.prob(tf.cast(z_gamma, tf.float64)), DTYPE)
    tau = tf.exp(log_tau)
    d_gamma = tf.constant([1.0, 0.0, 0.0], dtype=DTYPE) * dgamma_dz
    d_tau = tf.constant([0.0, 1.0, 0.0], dtype=DTYPE) * tau
    return gamma, tau, mu, d_gamma, d_tau


def _stationary_variance(gamma: tf.Tensor) -> tf.Tensor:
    return tf.constant(1.0, dtype=DTYPE) / (
        tf.constant(1.0, dtype=DTYPE) - tf.square(gamma)
    )


def _stationary_variance_jvp(gamma: tf.Tensor, d_gamma: tf.Tensor) -> tf.Tensor:
    denominator = tf.constant(1.0, dtype=DTYPE) - tf.square(gamma)
    return (
        tf.constant(2.0, dtype=DTYPE)
        * gamma
        / tf.square(denominator)
        * d_gamma
    )


def _make_initial_noise_tensor(args: argparse.Namespace) -> tf.Tensor:
    rows = []
    for seed in args.batch_seeds:
        rows.append(
            tf.random.stateless_normal(
                [int(args.num_particles), value_mod.STATE_DIM],
                seed=value_mod._seed_pair(int(seed), 500),  # noqa: SLF001
                dtype=DTYPE,
            )
        )
    return tf.stack(rows, axis=0)


def _make_proposal_noise_tensor(args: argparse.Namespace) -> tf.Tensor:
    time_rows = []
    for time_index in range(int(args.time_steps)):
        batch_rows = []
        for seed in args.batch_seeds:
            batch_rows.append(
                tf.random.stateless_normal(
                    [int(args.num_particles), value_mod.STATE_DIM],
                    seed=value_mod._seed_pair(  # noqa: SLF001
                        int(seed),
                        tf.constant(7000 + int(time_index), dtype=tf.int32),
                    ),
                    dtype=DTYPE,
                )
            )
        time_rows.append(tf.stack(batch_rows, axis=0))
    return tf.stack(time_rows, axis=1)


def _require_compact_score_args(args: argparse.Namespace) -> None:
    if args.transport_plan_mode != "streaming":
        raise ValueError("generalized-SV compact score requires streaming transport")
    if args.transport_ad_mode != "full":
        raise ValueError("generalized-SV compact score requires transport_ad_mode='full'")
    if args.transport_gradient_mode != core_tf.MANUAL_STREAMING_FINITE_TRANSPORT_GRADIENT_MODE:
        raise ValueError("generalized-SV compact score requires manual streaming finite transport")
    if args.transport_policy == "no-resampling":
        raise ValueError("generalized-SV admitted score requires active relaxed Sinkhorn transport")


def _forward_transport_tf(
    *,
    post_flow: tf.Tensor,
    normalized_log_weights: tf.Tensor,
    mask: tf.Tensor,
    args: argparse.Namespace,
) -> tuple[tf.Tensor, tf.Tensor]:
    transported = core_tf.batched_annealed_transport_core_tf(
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
    return transported.particles, transported.log_weights


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


def _log_normal_logpdf_jvp(
    value: tf.Tensor,
    mean: tf.Tensor,
    variance: tf.Tensor,
    d_value: tf.Tensor,
    d_mean: tf.Tensor,
    d_variance: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    value = tf.convert_to_tensor(value, dtype=DTYPE)
    mean = tf.convert_to_tensor(mean, dtype=DTYPE)
    variance = tf.convert_to_tensor(variance, dtype=DTYPE)
    d_value = tf.convert_to_tensor(d_value, dtype=DTYPE)
    d_mean = tf.convert_to_tensor(d_mean, dtype=DTYPE)
    d_variance = tf.convert_to_tensor(d_variance, dtype=DTYPE)
    residual = value - mean
    density = value_mod._log_normal_logpdf(value, mean, variance)  # noqa: SLF001
    residual_bar = -residual / variance
    variance_bar = -tf.constant(0.5, dtype=DTYPE) / variance + (
        tf.constant(0.5, dtype=DTYPE) * tf.square(residual) / tf.square(variance)
    )
    tangent = residual_bar[..., None] * (d_value - d_mean) + variance_bar[..., None] * d_variance
    return density, tangent


def _raw_zero_mean_normal_log_density_jvp(
    *,
    raw_observation: tf.Tensor,
    target_state: tf.Tensor,
    tau: tf.Tensor,
    d_target_state: tf.Tensor,
    d_tau: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    target_state = tf.convert_to_tensor(target_state, dtype=DTYPE)
    d_target_state = tf.convert_to_tensor(d_target_state, dtype=DTYPE)
    log_scale = tf.constant(0.5, dtype=DTYPE) * tau * target_state
    density = value_mod._raw_zero_mean_normal_log_density_from_log_scale(  # noqa: SLF001
        raw_observation,
        log_scale,
    )
    d_log_scale = tf.constant(0.5, dtype=DTYPE) * (
        target_state[:, :, None] * tf.reshape(d_tau, [1, 1, len(PARAMETER_NAMES)])
        + tau * d_target_state
    )
    y = tf.reshape(tf.convert_to_tensor(raw_observation, dtype=DTYPE), [1])[0]
    score_log_scale = -tf.constant(1.0, dtype=DTYPE) + tf.square(y) * tf.exp(
        -tf.constant(2.0, dtype=DTYPE) * log_scale
    )
    tangent = score_log_scale[:, :, None] * d_log_scale
    return density, tangent


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
        raise ValueError("compact generalized-SV score requires manual streaming finite transport")
    if args.transport_ad_mode != "full":
        raise ValueError("compact generalized-SV score requires transport_ad_mode='full'")
    batch_size, num_particles, _state_dim = core_tf._static_shape(post_flow, "post_flow")  # noqa: SLF001
    center = tf.reduce_mean(post_flow, axis=1, keepdims=True)
    d_center = tf.reduce_mean(d_post_flow, axis=1, keepdims=True)
    scale = annealed_transport_tf._filterflow_scale(post_flow)  # noqa: SLF001
    d_scale = actual_score._filterflow_scale_jvp(post_flow, d_post_flow)  # noqa: SLF001
    scaled_x = (post_flow - center) / scale[:, None, None]
    d_scaled_x = (
        (d_post_flow - d_center) / scale[:, None, None, None]
        - (post_flow - center)[:, :, :, None]
        * d_scale[:, None, None, :]
        / (scale[:, None, None, None] * scale[:, None, None, None])
    )
    epsilon = tf.convert_to_tensor(args.sinkhorn_epsilon, dtype=DTYPE)
    epsilon0 = annealed_transport_tf._filterflow_epsilon_start(scaled_x)  # noqa: SLF001
    d_epsilon0 = actual_score._filterflow_epsilon_start_jvp(scaled_x, d_scaled_x)  # noqa: SLF001
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
    d_uniform_log_weights = tf.zeros_like(d_normalized_log_weights)
    next_d_particles = tf.where(mask[:, None, None, None], d_transported, d_post_flow)
    next_d_log_weights = tf.where(
        mask[:, None, None],
        d_uniform_log_weights,
        d_normalized_log_weights,
    )
    return raw_transport.particles, raw_transport.log_weights, next_d_particles, next_d_log_weights


def _compact_ledh_flow_jvp_tf(
    *,
    pre_flow_particles: tf.Tensor,
    d_pre_flow_particles: tf.Tensor,
    ancestors: tf.Tensor,
    d_ancestors: tf.Tensor,
    observation: tf.Tensor,
    gamma: tf.Tensor,
    tau: tf.Tensor,
    mu: tf.Tensor,
    d_gamma: tf.Tensor,
    d_tau: tf.Tensor,
    flow_observation_covariance: tf.Tensor,
    particle_chunk_size: int,
) -> tuple[streaming_tf.StreamingLEDHFlowTensors, tf.Tensor, tf.Tensor, tf.Tensor]:
    x0 = tf.convert_to_tensor(pre_flow_particles, dtype=DTYPE)
    d_x0 = tf.convert_to_tensor(d_pre_flow_particles, dtype=DTYPE)
    ancestors = tf.convert_to_tensor(ancestors, dtype=DTYPE)
    d_ancestors = tf.convert_to_tensor(d_ancestors, dtype=DTYPE)
    batch_size, num_particles, state_dim = core_tf._static_shape(x0, "pre_flow_particles")  # noqa: SLF001
    if state_dim != 1:
        raise ValueError("generalized-SV compact score expects one-dimensional state")
    param_dim = len(PARAMETER_NAMES)
    chunk_size = int(particle_chunk_size)
    num_blocks = (int(num_particles) + chunk_size - 1) // chunk_size
    transition_matrix = tf.ones([batch_size, 1, 1], dtype=DTYPE)
    proposal_covariance = tf.ones([batch_size, 1, 1], dtype=DTYPE)

    post_blocks = []
    d_post_blocks = []
    pre_log_blocks = []
    d_pre_log_blocks = []
    logdet_blocks = []
    d_logdet_blocks = []
    d_mu = tf.constant([0.0, 0.0, 1.0], dtype=DTYPE)

    for block_index in range(num_blocks):
        start = block_index * chunk_size
        pre_chunk = streaming_tf._slice_axis1_padded_3d(  # noqa: SLF001
            x0,
            tf.constant(start, dtype=tf.int32),
            chunk_size,
        )
        ancestor_chunk = streaming_tf._slice_axis1_padded_3d(  # noqa: SLF001
            ancestors,
            tf.constant(start, dtype=tf.int32),
            chunk_size,
        )
        d_pre_chunk = actual_score._zero_padded_axis1_3d(  # noqa: SLF001
            tf.reshape(d_x0, [batch_size, num_particles, state_dim * param_dim]),
            start,
            chunk_size,
        )
        d_pre_chunk = tf.reshape(d_pre_chunk, [batch_size, chunk_size, state_dim, param_dim])
        d_ancestor_chunk = actual_score._zero_padded_axis1_3d(  # noqa: SLF001
            tf.reshape(d_ancestors, [batch_size, num_particles, state_dim * param_dim]),
            start,
            chunk_size,
        )
        d_ancestor_chunk = tf.reshape(
            d_ancestor_chunk,
            [batch_size, chunk_size, state_dim, param_dim],
        )

        def observation_fn(points: tf.Tensor) -> tf.Tensor:
            return tau * points

        def observation_jacobian_fn(points: tf.Tensor) -> tf.Tensor:
            chunk_particles = points.shape[1]
            if chunk_particles is None:
                raise ValueError("generalized-SV adapter requires static particle chunk dimension")
            return tf.ones([batch_size, int(chunk_particles), 1, 1], dtype=DTYPE) * tau

        def observation_residual_fn(h_ref: tf.Tensor, obs: tf.Tensor) -> tf.Tensor:
            return obs[tf.newaxis, tf.newaxis, :] - h_ref

        def chunk_prior_mean_fn(chunk_ancestors: tf.Tensor) -> tf.Tensor:
            return mu + gamma * (chunk_ancestors - mu)

        flow, aux = actual_score._core_flow_with_streaming_forward_aux_tf(  # noqa: SLF001
            pre_flow_particles=pre_chunk,
            ancestors=ancestor_chunk,
            observation=observation,
            transition_matrix=transition_matrix,
            transition_covariance=proposal_covariance,
            observation_covariance=flow_observation_covariance,
            observation_fn=observation_fn,
            observation_jacobian_fn=observation_jacobian_fn,
            observation_residual_fn=observation_residual_fn,
            prior_mean_fn=chunk_prior_mean_fn,
        )
        prior_means = mu + gamma * (ancestor_chunk - mu)
        d_prior_means = (
            gamma * d_ancestor_chunk
            + (ancestor_chunk - mu)[:, :, :, None] * tf.reshape(d_gamma, [1, 1, 1, param_dim])
            + (tf.constant(1.0, dtype=DTYPE) - gamma) * tf.reshape(d_mu, [1, 1, 1, param_dim])
        )
        d_h_jac = tf.reshape(d_tau, [1, 1, 1, param_dim])
        d_residual = -(
            tau * d_pre_chunk[:, :, 0, :]
            + pre_chunk[:, :, 0, None] * tf.reshape(d_tau, [1, 1, param_dim])
        )
        d_pseudo_observation = (
            tau * d_pre_chunk[:, :, 0, :]
            + pre_chunk[:, :, 0, None] * tf.reshape(d_tau, [1, 1, param_dim])
            + d_residual
        )
        del d_pseudo_observation

        obs_precision = tf.reshape(aux.obs_precision[:, 0, 0], [batch_size])
        post_precision = tf.reshape(aux.post_precision_stable[:, :, 0, 0], [batch_size, chunk_size])
        post_covariance = tf.reshape(aux.post_covariance[:, :, 0, 0], [batch_size, chunk_size])
        d_post_precision = (
            tf.constant(2.0, dtype=DTYPE)
            * tau
            * obs_precision[:, None, None]
            * tf.reshape(d_tau, [1, 1, param_dim])
        )
        d_post_covariance = -tf.square(post_covariance)[:, :, None] * d_post_precision
        pseudo_observation = tf.reshape(aux.pseudo_observation[:, :, 0], [batch_size, chunk_size])
        prior_precision = tf.reshape(aux.prior_precision[:, 0, 0], [batch_size])
        d_info = (
            prior_precision[:, None, None] * d_prior_means[:, :, 0, :]
            + obs_precision[:, None, None]
            * (
                d_h_jac * pseudo_observation[:, :, None, None]
            )[:, :, 0, :]
        )
        info = tf.reshape(aux.info[:, :, 0], [batch_size, chunk_size])
        d_post_mean = d_post_covariance * info[:, :, None] + post_covariance[:, :, None] * d_info
        delta = pre_chunk[:, :, 0] - prior_means[:, :, 0]
        d_delta = d_pre_chunk[:, :, 0, :] - d_prior_means[:, :, 0, :]
        post_scale = tf.sqrt(post_covariance)
        d_post_scale = d_post_covariance / (tf.constant(2.0, dtype=DTYPE) * post_scale[:, :, None])
        d_post = d_post_mean + d_post_scale * delta[:, :, None] + post_scale[:, :, None] * d_delta
        d_forward_log_det = d_post_scale / post_scale[:, :, None]
        _pre_log_density, d_pre_log_density = _log_normal_logpdf_jvp(
            pre_chunk[:, :, 0],
            prior_means[:, :, 0],
            tf.constant(1.0, dtype=DTYPE),
            d_pre_chunk[:, :, 0, :],
            d_prior_means[:, :, 0, :],
            tf.zeros([batch_size, 1, param_dim], dtype=DTYPE),
        )

        post_blocks.append(flow.post_flow_particles)
        d_post_blocks.append(d_post[:, :, None, :])
        pre_log_blocks.append(flow.pre_flow_log_density)
        d_pre_log_blocks.append(d_pre_log_density)
        logdet_blocks.append(flow.forward_log_det)
        d_logdet_blocks.append(d_forward_log_det)

    post_flow = tf.reshape(
        tf.transpose(tf.stack(post_blocks, axis=0), [1, 0, 2, 3]),
        [batch_size, num_blocks * chunk_size, state_dim],
    )[:, :num_particles, :]
    d_post_flow = tf.reshape(
        tf.transpose(tf.stack(d_post_blocks, axis=0), [1, 0, 2, 3, 4]),
        [batch_size, num_blocks * chunk_size, state_dim, param_dim],
    )[:, :num_particles, :, :]
    pre_log = tf.reshape(
        tf.transpose(tf.stack(pre_log_blocks, axis=0), [1, 0, 2]),
        [batch_size, num_blocks * chunk_size],
    )[:, :num_particles]
    d_pre_log = tf.reshape(
        tf.transpose(tf.stack(d_pre_log_blocks, axis=0), [1, 0, 2, 3]),
        [batch_size, num_blocks * chunk_size, param_dim],
    )[:, :num_particles, :]
    logdet = tf.reshape(
        tf.transpose(tf.stack(logdet_blocks, axis=0), [1, 0, 2]),
        [batch_size, num_blocks * chunk_size],
    )[:, :num_particles]
    d_logdet = tf.reshape(
        tf.transpose(tf.stack(d_logdet_blocks, axis=0), [1, 0, 2, 3]),
        [batch_size, num_blocks * chunk_size, param_dim],
    )[:, :num_particles, :]
    post_flow.set_shape([batch_size, num_particles, state_dim])
    d_post_flow.set_shape([batch_size, num_particles, state_dim, param_dim])
    pre_log.set_shape([batch_size, num_particles])
    d_pre_log.set_shape([batch_size, num_particles, param_dim])
    logdet.set_shape([batch_size, num_particles])
    d_logdet.set_shape([batch_size, num_particles, param_dim])
    return (
        streaming_tf.StreamingLEDHFlowTensors(
            post_flow_particles=post_flow,
            pre_flow_log_density=pre_log,
            forward_log_det=logdet,
        ),
        d_post_flow,
        d_pre_log,
        d_logdet,
    )


def _common_tensors(args: argparse.Namespace) -> tuple[dict[str, tf.Tensor], dict[str, Any]]:
    tensors, semantics = value_mod._build_generalized_sv_tensors(args)  # noqa: SLF001
    return {
        "observations": tf.cast(tensors["observations"], DTYPE),
        "raw_observations": tf.cast(tensors["raw_observations"], DTYPE),
        "fixed_resampling_mask": tf.convert_to_tensor(tensors["fixed_resampling_mask"], dtype=tf.bool),
        "flow_observation_covariance": tf.tile(
            tf.cast(tensors["flow_observation_covariance"], DTYPE),
            [len(args.batch_seeds), 1, 1],
        ),
    }, semantics


def _initial_particles_and_jvp(
    args: argparse.Namespace,
    gamma: tf.Tensor,
    mu: tf.Tensor,
    d_gamma: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    batch_size = len(args.batch_seeds)
    num_particles = int(args.num_particles)
    state_dim = value_mod.STATE_DIM
    param_dim = len(PARAMETER_NAMES)
    stationary_variance = _stationary_variance(gamma)
    d_stationary_variance = _stationary_variance_jvp(gamma, d_gamma)
    stationary_scale = tf.sqrt(stationary_variance)
    d_stationary_scale = d_stationary_variance / (
        tf.constant(2.0, dtype=DTYPE) * stationary_scale
    )
    initial_noise = _make_initial_noise_tensor(args)
    particles = mu + stationary_scale * initial_noise
    d_mu = tf.constant([0.0, 0.0, 1.0], dtype=DTYPE)
    d_particles = (
        tf.reshape(d_mu, [1, 1, 1, param_dim])
        + initial_noise[:, :, :, None] * tf.reshape(d_stationary_scale, [1, 1, 1, param_dim])
    )
    particles.set_shape([batch_size, num_particles, state_dim])
    d_particles.set_shape([batch_size, num_particles, state_dim, param_dim])
    return particles, d_particles


def _compact_value_and_score_from_components(
    args: argparse.Namespace,
    theta_values: Sequence[float],
) -> dict[str, tf.Tensor]:
    """Generalized-SV same-scalar score with compact forward sensitivity."""

    _configure_precision(args)
    _require_compact_score_args(args)
    theta = tf.constant([float(value) for value in theta_values], dtype=DTYPE)
    gamma, tau, mu, d_gamma, d_tau = _gamma_tau_mu(theta)
    tensors, _semantics = _common_tensors(args)
    observations = tensors["observations"]
    raw_observations = tensors["raw_observations"]
    fixed_resampling_mask = tensors["fixed_resampling_mask"]
    flow_observation_covariance = tensors["flow_observation_covariance"]
    proposal_noise = _make_proposal_noise_tensor(args)
    particles, d_particles = _initial_particles_and_jvp(args, gamma, mu, d_gamma)
    batch_size = len(args.batch_seeds)
    num_particles = int(args.num_particles)
    param_dim = len(PARAMETER_NAMES)
    d_mu = tf.constant([0.0, 0.0, 1.0], dtype=DTYPE)
    log_weights = core_tf.uniform_log_weights(batch_size, num_particles)
    d_log_weights = tf.zeros([batch_size, num_particles, param_dim], dtype=DTYPE)
    log_likelihood = tf.zeros([batch_size], dtype=DTYPE)
    d_log_likelihood = tf.zeros([batch_size, param_dim], dtype=DTYPE)

    for time_index in range(int(args.time_steps)):
        observation = observations[time_index]
        raw_observation = raw_observations[time_index]
        proposal_mean = mu + gamma * (particles - mu)
        d_proposal_mean = (
            gamma * d_particles
            + (particles - mu)[:, :, :, None] * tf.reshape(d_gamma, [1, 1, 1, param_dim])
            + (tf.constant(1.0, dtype=DTYPE) - gamma) * tf.reshape(d_mu, [1, 1, 1, param_dim])
        )
        noise = proposal_noise[:, time_index, :, :]
        pre_flow = proposal_mean + noise
        d_pre_flow = d_proposal_mean
        flow, d_post_flow, d_pre_flow_log_density, d_forward_log_det = _compact_ledh_flow_jvp_tf(
            pre_flow_particles=pre_flow,
            d_pre_flow_particles=d_pre_flow,
            ancestors=particles,
            d_ancestors=d_particles,
            observation=observation,
            gamma=gamma,
            tau=tau,
            mu=mu,
            d_gamma=d_gamma,
            d_tau=d_tau,
            flow_observation_covariance=flow_observation_covariance,
            particle_chunk_size=int(args.particle_chunk_size),
        )
        post_flow = flow.post_flow_particles
        target_state = post_flow[:, :, 0]
        d_target_state = d_post_flow[:, :, 0, :]
        previous_state = particles[:, :, 0]
        d_previous_state = d_particles[:, :, 0, :]
        transition_mean = mu + gamma * (previous_state - mu)
        d_transition_mean = (
            gamma * d_previous_state
            + (previous_state - mu)[:, :, None] * tf.reshape(d_gamma, [1, 1, param_dim])
            + (tf.constant(1.0, dtype=DTYPE) - gamma) * tf.reshape(d_mu, [1, 1, param_dim])
        )
        target_transition, d_target_transition = _log_normal_logpdf_jvp(
            target_state,
            transition_mean,
            tf.constant(1.0, dtype=DTYPE),
            d_target_state,
            d_transition_mean,
            tf.zeros([batch_size, 1, param_dim], dtype=DTYPE),
        )
        target_observation, d_target_observation = _raw_zero_mean_normal_log_density_jvp(
            raw_observation=raw_observation,
            target_state=target_state,
            tau=tau,
            d_target_state=d_target_state,
            d_tau=d_tau,
        )
        corrected_log_weights = (
            log_weights
            + target_transition
            + target_observation
            - flow.pre_flow_log_density
            + flow.forward_log_det
        )
        d_corrected_log_weights = (
            d_log_weights
            + d_target_transition
            + d_target_observation
            - d_pre_flow_log_density
            + d_forward_log_det
        )
        (
            normalized_log_weights,
            d_normalized_log_weights,
            incremental,
            d_incremental,
        ) = _normalize_log_weights_jvp(corrected_log_weights, d_corrected_log_weights)
        particles, log_weights, d_particles, d_log_weights = _compact_forward_transport_jvp_tf(
            post_flow=post_flow,
            normalized_log_weights=normalized_log_weights,
            d_post_flow=d_post_flow,
            d_normalized_log_weights=d_normalized_log_weights,
            mask=fixed_resampling_mask[:, time_index],
            args=args,
        )
        log_likelihood = log_likelihood + incremental
        d_log_likelihood = d_log_likelihood + d_incremental

    return {
        "objective": tf.reduce_mean(log_likelihood),
        "log_likelihood": log_likelihood,
        "gradient_tensor": tf.reduce_mean(d_log_likelihood, axis=0),
        "per_seed_gradient": d_log_likelihood,
        "score_route": GENERALIZED_SV_COMPACT_SCORE_ROUTE_ID,
        "no_autodiff_score_route": True,
        "value_score_route_status": "same_route_value_score",
    }


def _manual_value_only_from_components(
    args: argparse.Namespace,
    theta_values: Sequence[float],
) -> dict[str, tf.Tensor]:
    _configure_precision(args)
    _require_compact_score_args(args)
    theta = tf.constant([float(value) for value in theta_values], dtype=DTYPE)
    gamma, tau, mu, _d_gamma, _d_tau = _gamma_tau_mu(theta)
    tensors, _semantics = _common_tensors(args)
    proposal_noise = _make_proposal_noise_tensor(args)
    initial_noise = _make_initial_noise_tensor(args)
    stationary_scale = tf.sqrt(_stationary_variance(gamma))
    particles = mu + stationary_scale * initial_noise
    observations = tensors["observations"]
    raw_observations = tensors["raw_observations"]
    fixed_resampling_mask = tensors["fixed_resampling_mask"]
    flow_observation_covariance = tensors["flow_observation_covariance"]
    batch_size = len(args.batch_seeds)
    num_particles = int(args.num_particles)
    log_weights = core_tf.uniform_log_weights(batch_size, num_particles)
    log_likelihood = tf.zeros([batch_size], dtype=DTYPE)
    transition_matrix = tf.ones([batch_size, 1, 1], dtype=DTYPE)
    transition_covariance = tf.ones([batch_size, 1, 1], dtype=DTYPE)

    for time_index in range(int(args.time_steps)):
        observation = observations[time_index]
        proposal_mean = mu + gamma * (particles - mu)
        pre_flow = proposal_mean + proposal_noise[:, time_index, :, :]

        def observation_fn(points: tf.Tensor) -> tf.Tensor:
            return tau * points

        def observation_jacobian_fn(points: tf.Tensor) -> tf.Tensor:
            chunk_particles = points.shape[1]
            if chunk_particles is None:
                raise ValueError("generalized-SV adapter requires static particle chunk dimension")
            return tf.ones([batch_size, int(chunk_particles), 1, 1], dtype=DTYPE) * tau

        def observation_residual_fn(h_ref: tf.Tensor, obs: tf.Tensor) -> tf.Tensor:
            return obs[tf.newaxis, tf.newaxis, :] - h_ref

        def prior_mean_fn(ancestors: tf.Tensor) -> tf.Tensor:
            return mu + gamma * (ancestors - mu)

        flow, _aux = actual_score._streaming_flow_with_aux_tf(  # noqa: SLF001
            pre_flow_particles=pre_flow,
            ancestors=particles,
            observation=observation,
            transition_matrix=transition_matrix,
            transition_covariance=transition_covariance,
            observation_covariance=flow_observation_covariance,
            observation_fn=observation_fn,
            observation_jacobian_fn=observation_jacobian_fn,
            observation_residual_fn=observation_residual_fn,
            prior_mean_fn=prior_mean_fn,
            particle_chunk_size=int(args.particle_chunk_size),
        )
        post_flow = flow.post_flow_particles
        target_state = post_flow[:, :, 0]
        target_transition = value_mod._log_normal_logpdf(  # noqa: SLF001
            target_state,
            mu + gamma * (particles[:, :, 0] - mu),
            tf.constant(1.0, dtype=DTYPE),
        )
        target_observation = value_mod._raw_zero_mean_normal_log_density_from_log_scale(  # noqa: SLF001
            raw_observations[time_index],
            tf.constant(0.5, dtype=DTYPE) * tau * target_state,
        )
        corrected_log_weights = (
            log_weights
            + target_transition
            + target_observation
            - flow.pre_flow_log_density
            + flow.forward_log_det
        )
        weights, incremental = core_tf._normalize_log_weights(corrected_log_weights)  # noqa: SLF001
        normalized_log_weights = tf.math.log(
            tf.maximum(weights, core_tf._log_weight_floor())  # noqa: SLF001
        )
        particles, log_weights = _forward_transport_tf(
            post_flow=post_flow,
            normalized_log_weights=normalized_log_weights,
            mask=fixed_resampling_mask[:, time_index],
            args=args,
        )
        log_likelihood = log_likelihood + incremental
    return {"objective": tf.reduce_mean(log_likelihood), "log_likelihood": log_likelihood}


def _single_seed_args(args: argparse.Namespace, seed: int) -> argparse.Namespace:
    clone = copy.copy(args)
    clone.batch_seeds = [int(seed)]
    return clone


def _manual_value_and_score_across_seeds(
    args: argparse.Namespace,
    theta_values: Sequence[float],
) -> dict[str, tf.Tensor]:
    seed_results = [
        _compact_value_and_score_from_components(_single_seed_args(args, seed), theta_values)
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
        "score_route": GENERALIZED_SV_COMPACT_SCORE_ROUTE_ID,
        "no_autodiff_score_route": True,
        "value_score_route_status": "same_route_value_score",
    }


def _manual_objective_across_seeds(
    args: argparse.Namespace,
    theta_values: Sequence[float],
) -> tf.Tensor:
    seed_values = [
        _manual_value_only_from_components(_single_seed_args(args, seed), theta_values)[
            "log_likelihood"
        ]
        for seed in args.batch_seeds
    ]
    return tf.reduce_mean(tf.concat(seed_values, axis=0))


def _coordinate_fd_score_diagnostic(
    args: argparse.Namespace,
    theta_values: Sequence[float],
    *,
    fd_step: float = 1.0e-4,
    atol: float = 5.0e-3,
    rtol: float = 5.0e-3,
) -> dict[str, Any]:
    _configure_precision(args)
    theta = tf.constant([float(value) for value in theta_values], dtype=DTYPE)
    step = tf.constant(float(fd_step), dtype=DTYPE)
    base = _manual_value_and_score_across_seeds(args, theta.numpy().tolist())
    fd_values = []
    for index in range(len(PARAMETER_NAMES)):
        basis = tf.one_hot(index, len(PARAMETER_NAMES), dtype=DTYPE)
        plus_objective = _manual_objective_across_seeds(
            args,
            (theta + step * basis).numpy().tolist(),
        )
        minus_objective = _manual_objective_across_seeds(
            args,
            (theta - step * basis).numpy().tolist(),
        )
        fd_values.append((plus_objective - minus_objective) / (tf.constant(2.0, dtype=DTYPE) * step))
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
        "fd_step": float(fd_step),
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
        expected_row_id=GENERALIZED_SV_ROW_ID,
        require_admitted=True,
    )
    base = diagnostic["base"]
    score = tf.convert_to_tensor(base["gradient_tensor"], dtype=DTYPE)
    memory = dict(memory_diagnostics or {})
    memory_pass = bool(memory.get("n10000_memory_pass") is True)
    if require_all_parameter_correctness:
        if diagnostic.get("status") != "pass":
            raise ValueError("generalized-SV all-parameter correctness status must pass")
        if tuple(diagnostic.get("parameter_names", PARAMETER_NAMES)) != tuple(PARAMETER_NAMES):
            raise ValueError("generalized-SV all-parameter correctness parameter_names mismatch")
        if not memory_pass:
            raise ValueError("generalized-SV full admission requires N=10000 memory pass")
    artifact = {
        "schema_version": LEDH_SCORE_ARTIFACT_SCHEMA_VERSION,
        "row_id": GENERALIZED_SV_ROW_ID,
        "source_value_artifact": source_value_artifact_path,
        "score_target_kind": LEDH_SCORE_TARGET_KIND_REALIZED_FINITE_N_ESTIMATOR,
        "target_scalar": LEDH_TARGET_SCALAR_OBSERVED_DATA_LOG_LIKELIHOOD,
        "target_output_tensor_field": LEDH_OUTPUT_TENSOR_FIELD_LOG_LIKELIHOOD,
        "target_observation_policy": value_core["target_observation_policy"],
        "theta_coordinate_system": value_core["theta_coordinate_system"],
        "score_parameter_names": list(PARAMETER_NAMES),
        "score": [float(value) for value in score.numpy().reshape(-1)],
        "score_derivative_provenance": str(
            base.get("score_route", GENERALIZED_SV_COMPACT_SCORE_ROUTE_ID)
        ),
        "value_score_route_status": LEDH_SCORE_VALUE_ROUTE_STATUS_SAME,
        "value_score_same_transport_algorithm": True,
        "no_autodiff_score_route": True,
        "uses_gradient_tape": False,
        "uses_forward_accumulator": False,
        "uses_stopped_partial_derivative": False,
        "claims_exact_native_actual_sv_likelihood": False,
        "score_correctness": {
            "kind": "same_scalar_finite_difference",
            "status": str(diagnostic.get("status", "fail")),
            "max_abs_error": float(tf.convert_to_tensor(diagnostic["max_abs_error"], dtype=DTYPE).numpy()),
            "max_rel_error": float(tf.convert_to_tensor(diagnostic["max_rel_error"], dtype=DTYPE).numpy()),
            "fd_step": float(diagnostic.get("fd_step", math.nan)),
            "atol": float(diagnostic.get("atol", math.nan)),
            "rtol": float(diagnostic.get("rtol", math.nan)),
        },
        "score_admission_status": (
            LEDH_SCORE_ADMISSION_STATUS_FULL
            if require_all_parameter_correctness
            else LEDH_SCORE_ADMISSION_STATUS_TINY
            if diagnostic.get("status") == "pass"
            else LEDH_SCORE_ADMISSION_STATUS_BLOCKED_NOT_RUN
        ),
        "memory_diagnostics": memory,
    }
    if require_all_parameter_correctness:
        validate_ledh_score_artifact(
            artifact,
            source_value_artifact=source_value_artifact,
            expected_row_id=GENERALIZED_SV_ROW_ID,
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
    parser.add_argument("--time-steps", type=int, default=2)
    parser.add_argument("--num-particles", type=int, default=64)
    parser.add_argument("--transport-policy", choices=("active-all", "active-odd", "no-resampling"), default="active-all")
    parser.add_argument("--sinkhorn-iterations", type=int, default=2)
    parser.add_argument("--sinkhorn-epsilon", type=float, default=1.0)
    parser.add_argument("--annealed-scaling", type=float, default=0.9)
    parser.add_argument("--annealed-convergence-threshold", type=float, default=1.0e-3)
    parser.add_argument("--flow-observation-variance", type=float, default=value_mod.DEFAULT_FLOW_OBSERVATION_VARIANCE)
    parser.add_argument("--transport-plan-mode", choices=("streaming", "dense"), default="streaming")
    parser.add_argument(
        "--transport-gradient-mode",
        choices=(core_tf.MANUAL_STREAMING_FINITE_TRANSPORT_GRADIENT_MODE,),
        default=core_tf.MANUAL_STREAMING_FINITE_TRANSPORT_GRADIENT_MODE,
    )
    parser.add_argument("--transport-ad-mode", choices=("full",), default="full")
    parser.add_argument("--row-chunk-size", type=int, default=64)
    parser.add_argument("--col-chunk-size", type=int, default=64)
    parser.add_argument("--particle-chunk-size", type=int, default=64)
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--tf32-mode", choices=("default", "enabled", "disabled"), default="disabled")
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
    if args.time_steps <= 0 or args.num_particles <= 1:
        raise ValueError("time_steps must be positive and num_particles must exceed one")
    if args.row_chunk_size <= 0 or args.col_chunk_size <= 0 or args.particle_chunk_size <= 0:
        raise ValueError("chunk sizes must be positive")
    if args.admit_full:
        exact_full = (
            tuple(args.batch_seeds) == tuple(value_mod.FULL_ROW_BATCH_SEEDS)
            and int(args.num_particles) == value_mod.FULL_ROW_NUM_PARTICLES
            and int(args.time_steps) == value_mod.FULL_ROW_TIME_STEPS
        )
        if not exact_full:
            raise ValueError("full score admission requires exact generalized-SV full-row shape")
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
        "# Generalized-SV Same-Target LEDH Score",
        "",
        f"- JSON artifact: `{json_path}`",
        f"- Row: `{artifact['row_id']}`",
        f"- Score admission status: `{artifact['score_admission_status']}`",
        f"- Correctness: `{artifact['score_correctness']}`",
        f"- Memory diagnostics: `{artifact['memory_diagnostics']}`",
        "",
        "## Nonclaims",
        "",
    ]
    lines.extend(f"- {claim}" for claim in artifact["nonclaims"])
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
            "target_semantics": {
                "target_observation_policy": "source_route_prior_mean_generalized_sv",
                "target_observation_density": "raw_zero_mean_generalized_sv_prior_mean_normal_log_density",
                "flow_observation_policy": "log_square_gaussian_surrogate_for_ledh_flow_only",
                "flow_is_proposal_surface_only": True,
                "actual_sv_evidence_used": False,
                "native_generalized_sv_dense_fixture_used": False,
                "sp500_returns_used_as_benchmark_observations": False,
                "author_defaults_used_as_truth": False,
                "ksc_mixture_used": False,
            },
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
            "per_coordinate_rel_error": [
                float(value)
                for value in tf.convert_to_tensor(
                    diagnostic["per_coordinate_rel_error"],
                    dtype=DTYPE,
                )
                .numpy()
                .reshape(-1)
            ],
            "nonclaims": [
                "not full generalized-SV score admission",
                "not actual-SV admission",
                "not KSC surrogate likelihood evidence",
                "not native generalized-SV dense fixture evidence",
                "not SP500 benchmark-observation evidence",
                "not author-default truth evidence",
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
