"""Same-target actual-SV LEDH score repair helpers.

This module is the score-side sibling of
``benchmark_ledh_same_target_actual_sv_value``.  The current/default score
route is the compact forward-sensitivity recurrence for the same transformed
actual-SV finite-``N`` LEDH log-likelihood scalar.  The older reverse/manual
VJP route is retained only as historical diagnostic code and must not be
full-admitted.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping


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
    ACTUAL_SV_ROW_ID,
    LEDH_OUTPUT_TENSOR_FIELD_LOG_LIKELIHOOD,
    LEDH_TARGET_SCALAR_OBSERVED_DATA_LOG_LIKELIHOOD,
    SV_SYNTHETIC_PARAMETER_ORDER,
    validate_ledh_forward_scalar_artifact,
)
from bayesfilter.highdim.ledh_score_contract import (
    LEDH_SCORE_ADMISSION_STATUS_BLOCKED_NOT_RUN,
    LEDH_SCORE_ADMISSION_STATUS_FULL,
    LEDH_SCORE_ADMISSION_STATUS_TINY,
    LEDH_SCORE_ARTIFACT_SCHEMA_VERSION,
    LEDH_SCORE_COMPACT_ACTUAL_SV_PROVENANCE,
    LEDH_SCORE_MEMORY_STYLE_ACTUAL_SV_PROVENANCE,
    LEDH_SCORE_TARGET_KIND_REALIZED_FINITE_N_ESTIMATOR,
    LEDH_SCORE_VALUE_ROUTE_STATUS_SAME,
    validate_ledh_score_artifact,
)
from bayesfilter.highdim.sv_mixture_cut4 import exact_log_chi_square_log_density
from docs.benchmarks import benchmark_ledh_same_target_actual_sv_value as value_mod
from docs.benchmarks import benchmark_p8p_parameterized_sir_gradient as p8p
from experiments.dpf_implementation.tf_tfp.filters import (
    experimental_batched_ledh_pfpf_ot_tf as core_tf,
)
from experiments.dpf_implementation.tf_tfp.filters import (
    experimental_batched_ledh_pfpf_ot_streaming_tf as streaming_tf,
)
from experiments.dpf_implementation.tf_tfp.resampling import annealed_transport_tf


DTYPE = tf.float32
ACTUAL_SV_MANUAL_SCORE_ROUTE_ID = (
    "manual_total_vjp_no_autodiff_same_scalar_actual_sv_ledh_pfpf_ot"
)
ACTUAL_SV_MEMORY_STYLE_SCORE_ROUTE_ID = LEDH_SCORE_MEMORY_STYLE_ACTUAL_SV_PROVENANCE
ACTUAL_SV_COMPACT_SCORE_ROUTE_ID = LEDH_SCORE_COMPACT_ACTUAL_SV_PROVENANCE
PARAMETER_NAMES = SV_SYNTHETIC_PARAMETER_ORDER
TRUTH_THETA = tuple(value_mod.TRUTH_THETA)
_STD_NORMAL = tfp.distributions.Normal(
    loc=tf.constant(0.0, dtype=tf.float64),
    scale=tf.constant(1.0, dtype=tf.float64),
)


@dataclass(frozen=True)
class _StreamingFlowAux:
    block_aux: list[core_tf._BatchedLEDHLinearizedFlowAux]  # noqa: SLF001
    particle_chunk_size: int
    batch_size: int
    num_particles: int
    state_dim: int


def _zero_padded_axis1_3d(tensor: tf.Tensor, start: int, chunk_size: int) -> tf.Tensor:
    value = tf.convert_to_tensor(tensor, dtype=DTYPE)
    indices = tf.range(start, start + chunk_size, dtype=tf.int32)
    num_particles = tf.shape(value)[1]
    safe_indices = tf.minimum(indices, num_particles - 1)
    gathered = tf.gather(value, safe_indices, axis=1)
    active = indices < num_particles
    return tf.where(active[None, :, None], gathered, tf.zeros_like(gathered))


def _zero_padded_axis1_2d(tensor: tf.Tensor, start: int, chunk_size: int) -> tf.Tensor:
    value = tf.convert_to_tensor(tensor, dtype=DTYPE)
    indices = tf.range(start, start + chunk_size, dtype=tf.int32)
    num_particles = tf.shape(value)[1]
    safe_indices = tf.minimum(indices, num_particles - 1)
    gathered = tf.gather(value, safe_indices, axis=1)
    active = indices < num_particles
    return tf.where(active[None, :], gathered, tf.zeros_like(gathered))


def _core_flow_with_streaming_forward_aux_tf(
    *,
    pre_flow_particles: tf.Tensor,
    ancestors: tf.Tensor,
    observation: tf.Tensor,
    transition_matrix: tf.Tensor,
    transition_covariance: tf.Tensor,
    observation_covariance: tf.Tensor,
    observation_fn: Callable[[tf.Tensor], tf.Tensor],
    observation_jacobian_fn: Callable[[tf.Tensor], tf.Tensor],
    observation_residual_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
    prior_mean_fn: Callable[[tf.Tensor], tf.Tensor] | None = None,
    jitter: float | tf.Tensor = 1.0e-9,
) -> tuple[core_tf.BatchedLEDHFlowTensors, core_tf._BatchedLEDHLinearizedFlowAux]:  # noqa: SLF001
    """Run the same LEDH core arithmetic as the streaming value route with aux."""

    x0 = core_tf._to_float_tensor(pre_flow_particles, "pre_flow_particles")  # noqa: SLF001
    ancestors = core_tf._to_float_tensor(ancestors, "ancestors")  # noqa: SLF001
    observation = core_tf._to_float_tensor(observation, "observation")  # noqa: SLF001
    transition_matrix = core_tf._to_float_tensor(transition_matrix, "transition_matrix")  # noqa: SLF001
    transition_covariance_stable = core_tf._stabilize_batch_covariance(  # noqa: SLF001
        transition_covariance,
        jitter,
        "transition_covariance",
    )
    observation_covariance_stable = core_tf._stabilize_batch_covariance(  # noqa: SLF001
        observation_covariance,
        jitter,
        "observation_covariance",
    )
    core_tf._require_static_rank(x0, 3, "pre_flow_particles")  # noqa: SLF001
    core_tf._require_static_rank(ancestors, 3, "ancestors")  # noqa: SLF001
    core_tf._require_static_rank(transition_matrix, 3, "transition_matrix")  # noqa: SLF001
    core_tf._require_static_rank(transition_covariance_stable, 3, "transition_covariance")  # noqa: SLF001
    core_tf._require_static_rank(observation_covariance_stable, 3, "observation_covariance")  # noqa: SLF001
    batch_size, num_particles, state_dim = core_tf._static_shape(  # noqa: SLF001
        x0,
        "pre_flow_particles",
    )
    if core_tf._static_shape(ancestors, "ancestors") != (  # noqa: SLF001
        batch_size,
        num_particles,
        state_dim,
    ):
        raise ValueError("ancestors shape mismatch")
    core_tf._require_square_batch(transition_matrix, batch_size, state_dim, "transition_matrix")  # noqa: SLF001
    core_tf._require_square_batch(  # noqa: SLF001
        transition_covariance_stable,
        batch_size,
        state_dim,
        "transition_covariance",
    )
    observation_dim = core_tf._static_shape(  # noqa: SLF001
        observation_covariance_stable,
        "observation_covariance",
    )[1]
    core_tf._require_square_batch(  # noqa: SLF001
        observation_covariance_stable,
        batch_size,
        observation_dim,
        "observation_covariance",
    )

    if prior_mean_fn is None:
        prior_means = tf.einsum("bnj,bdj->bnd", ancestors, transition_matrix)
    else:
        prior_means = core_tf._to_float_tensor(  # noqa: SLF001
            prior_mean_fn(ancestors),
            "prior_mean_fn output",
        )
        core_tf._require_static_rank(prior_means, 3, "prior_mean_fn output")  # noqa: SLF001
        if core_tf._static_shape(prior_means, "prior_mean_fn output") != (  # noqa: SLF001
            batch_size,
            num_particles,
            state_dim,
        ):
            raise ValueError("prior_mean_fn output shape mismatch")

    delta = x0 - prior_means
    pre_flow_log_density = core_tf._batched_gaussian_logpdf(  # noqa: SLF001
        delta,
        transition_covariance_stable,
    )
    prior_chol = tf.linalg.cholesky(transition_covariance_stable)
    prior_precision = tf.linalg.cholesky_solve(
        prior_chol,
        core_tf._tile_eye(batch_size, state_dim),  # noqa: SLF001
    )
    obs_chol = tf.linalg.cholesky(observation_covariance_stable)
    obs_precision = tf.linalg.cholesky_solve(
        obs_chol,
        core_tf._tile_eye(batch_size, observation_dim),  # noqa: SLF001
    )

    h_ref = core_tf._to_float_tensor(observation_fn(x0), "observation_fn output")  # noqa: SLF001
    h_jac = core_tf._to_float_tensor(  # noqa: SLF001
        observation_jacobian_fn(x0),
        "observation_jacobian_fn output",
    )
    residual = core_tf._to_float_tensor(  # noqa: SLF001
        observation_residual_fn(h_ref, observation),
        "observation_residual_fn output",
    )
    core_tf._require_static_rank(h_ref, 3, "observation_fn output")  # noqa: SLF001
    core_tf._require_static_rank(h_jac, 4, "observation_jacobian_fn output")  # noqa: SLF001
    core_tf._require_static_rank(residual, 3, "observation_residual_fn output")  # noqa: SLF001
    core_tf._require_observation_callback_shapes(  # noqa: SLF001
        h_ref=h_ref,
        h_jac=h_jac,
        residual=residual,
        batch_size=batch_size,
        num_particles=num_particles,
        state_dim=state_dim,
        observation_dim=observation_dim,
    )

    pseudo_observation = tf.einsum("bnod,bnd->bno", h_jac, x0) + residual
    post_precision = prior_precision[:, None, :, :] + tf.einsum(
        "bnod,boq,bnqe->bnde",
        h_jac,
        obs_precision,
        h_jac,
    )
    post_precision_stable = core_tf._stabilize_batch_covariance(  # noqa: SLF001
        post_precision,
        jitter,
        "post_precision",
    )
    post_precision_chol = tf.linalg.cholesky(post_precision_stable)
    post_covariance_unstabilized = tf.linalg.cholesky_solve(
        post_precision_chol,
        tf.broadcast_to(
            tf.eye(state_dim, dtype=DTYPE),
            tf.shape(post_precision_stable),
        ),
    )
    post_covariance = core_tf._stabilize_batch_covariance(  # noqa: SLF001
        post_covariance_unstabilized,
        jitter,
        "post_covariance",
    )
    info = tf.einsum("bde,bne->bnd", prior_precision, prior_means) + tf.einsum(
        "bnod,boq,bnq->bnd",
        h_jac,
        obs_precision,
        pseudo_observation,
    )
    post_mean = tf.einsum("bnde,bne->bnd", post_covariance, info)
    post_chol = tf.linalg.cholesky(post_covariance)
    prior_inv = tf.linalg.triangular_solve(
        prior_chol,
        core_tf._tile_eye(batch_size, state_dim),  # noqa: SLF001
    )
    affine_transform = tf.einsum("bnij,bjk->bnik", post_chol, prior_inv)
    post_flow_particles = post_mean + tf.einsum(
        "bnij,bnj->bni",
        affine_transform,
        delta,
    )
    forward_log_det = tf.reduce_sum(
        tf.math.log(tf.linalg.diag_part(post_chol)),
        axis=-1,
    ) - tf.reduce_sum(
        tf.math.log(tf.linalg.diag_part(prior_chol)),
        axis=-1,
    )[:, None]
    flow = core_tf.BatchedLEDHFlowTensors(
        post_flow_particles=post_flow_particles,
        pre_flow_log_density=pre_flow_log_density,
        forward_log_det=forward_log_det,
        local_posterior_means=post_mean,
        local_posterior_covariances=post_covariance,
    )
    aux = core_tf._BatchedLEDHLinearizedFlowAux(  # noqa: SLF001
        x0=x0,
        prior_means=prior_means,
        observation_jacobian=h_jac,
        observation_residual=residual,
        transition_covariance=transition_covariance,
        observation_covariance=observation_covariance,
        transition_covariance_stable=transition_covariance_stable,
        observation_covariance_stable=observation_covariance_stable,
        prior_chol=prior_chol,
        prior_precision=prior_precision,
        obs_precision=obs_precision,
        pseudo_observation=pseudo_observation,
        post_precision=post_precision,
        post_precision_stable=post_precision_stable,
        post_covariance_unstabilized=post_covariance_unstabilized,
        post_covariance=post_covariance,
        post_chol=post_chol,
        prior_inv=prior_inv,
        affine_transform=affine_transform,
        delta=delta,
        info=info,
    )
    return flow, aux


def _streaming_flow_with_aux_tf(
    *,
    pre_flow_particles: tf.Tensor,
    ancestors: tf.Tensor,
    observation: tf.Tensor,
    transition_matrix: tf.Tensor,
    transition_covariance: tf.Tensor,
    observation_covariance: tf.Tensor,
    observation_fn: Callable[[tf.Tensor], tf.Tensor],
    observation_jacobian_fn: Callable[[tf.Tensor], tf.Tensor],
    observation_residual_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
    prior_mean_fn: Callable[[tf.Tensor], tf.Tensor] | None = None,
    particle_chunk_size: int,
) -> tuple[streaming_tf.StreamingLEDHFlowTensors, _StreamingFlowAux]:
    """Mirror streaming value chunking while retaining per-block VJP aux."""

    if particle_chunk_size <= 0:
        raise ValueError("particle_chunk_size must be positive")
    x0 = core_tf._to_float_tensor(pre_flow_particles, "pre_flow_particles")  # noqa: SLF001
    ancestors = core_tf._to_float_tensor(ancestors, "ancestors")  # noqa: SLF001
    core_tf._require_static_rank(x0, 3, "pre_flow_particles")  # noqa: SLF001
    core_tf._require_static_rank(ancestors, 3, "ancestors")  # noqa: SLF001
    batch_size, num_particles, state_dim = core_tf._static_shape(  # noqa: SLF001
        x0,
        "pre_flow_particles",
    )
    if core_tf._static_shape(ancestors, "ancestors") != (  # noqa: SLF001
        batch_size,
        num_particles,
        state_dim,
    ):
        raise ValueError("ancestors shape mismatch")

    chunk_size = int(particle_chunk_size)
    num_blocks = (int(num_particles) + chunk_size - 1) // chunk_size
    post_blocks = []
    pre_log_blocks = []
    logdet_blocks = []
    aux_blocks: list[core_tf._BatchedLEDHLinearizedFlowAux] = []  # noqa: SLF001
    for block_index in range(num_blocks):
        particle_start = tf.constant(block_index * chunk_size, dtype=tf.int32)
        pre_chunk = streaming_tf._slice_axis1_padded_3d(  # noqa: SLF001
            x0,
            particle_start,
            chunk_size,
        )
        ancestor_chunk = streaming_tf._slice_axis1_padded_3d(  # noqa: SLF001
            ancestors,
            particle_start,
            chunk_size,
        )
        flow, aux = _core_flow_with_streaming_forward_aux_tf(
            pre_flow_particles=pre_chunk,
            ancestors=ancestor_chunk,
            observation=observation,
            transition_matrix=transition_matrix,
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance,
            observation_fn=observation_fn,
            observation_jacobian_fn=observation_jacobian_fn,
            observation_residual_fn=observation_residual_fn,
            prior_mean_fn=prior_mean_fn,
        )
        post_blocks.append(flow.post_flow_particles)
        pre_log_blocks.append(flow.pre_flow_log_density)
        logdet_blocks.append(flow.forward_log_det)
        aux_blocks.append(aux)

    post_flow = tf.reshape(
        tf.transpose(tf.stack(post_blocks, axis=0), [1, 0, 2, 3]),
        [batch_size, num_blocks * chunk_size, state_dim],
    )[:, :num_particles, :]
    pre_log = tf.reshape(
        tf.transpose(tf.stack(pre_log_blocks, axis=0), [1, 0, 2]),
        [batch_size, num_blocks * chunk_size],
    )[:, :num_particles]
    logdet = tf.reshape(
        tf.transpose(tf.stack(logdet_blocks, axis=0), [1, 0, 2]),
        [batch_size, num_blocks * chunk_size],
    )[:, :num_particles]
    post_flow.set_shape([batch_size, num_particles, state_dim])
    pre_log.set_shape([batch_size, num_particles])
    logdet.set_shape([batch_size, num_particles])
    return (
        streaming_tf.StreamingLEDHFlowTensors(
            post_flow_particles=post_flow,
            pre_flow_log_density=pre_log,
            forward_log_det=logdet,
        ),
        _StreamingFlowAux(
            block_aux=aux_blocks,
            particle_chunk_size=chunk_size,
            batch_size=batch_size,
            num_particles=num_particles,
            state_dim=state_dim,
        ),
    )


def _streaming_flow_vjp(
    aux: _StreamingFlowAux,
    post_flow_particles_upstream: tf.Tensor,
    pre_flow_log_density_upstream: tf.Tensor,
    forward_log_det_upstream: tf.Tensor,
) -> core_tf.BatchedLEDHFlowVJPTensors:
    bar_pre_blocks = []
    bar_prior_blocks = []
    bar_residual_blocks = []
    bar_transition_covariance = None
    bar_observation_covariance = None
    chunk_size = int(aux.particle_chunk_size)
    for block_index, block_aux in enumerate(aux.block_aux):
        start = block_index * chunk_size
        block_vjp = core_tf._batched_ledh_linearized_flow_vjp(  # noqa: SLF001
            block_aux,
            _zero_padded_axis1_3d(post_flow_particles_upstream, start, chunk_size),
            _zero_padded_axis1_2d(pre_flow_log_density_upstream, start, chunk_size),
            _zero_padded_axis1_2d(forward_log_det_upstream, start, chunk_size),
        )
        real_count = min(chunk_size, aux.num_particles - start)
        if real_count > 0:
            bar_pre_blocks.append(block_vjp.pre_flow_particles[:, :real_count, :])
            bar_prior_blocks.append(block_vjp.prior_means[:, :real_count, :])
            bar_residual_blocks.append(block_vjp.observation_residual[:, :real_count, :])
        if bar_transition_covariance is None:
            bar_transition_covariance = block_vjp.transition_covariance
            bar_observation_covariance = block_vjp.observation_covariance
        else:
            bar_transition_covariance += block_vjp.transition_covariance
            bar_observation_covariance += block_vjp.observation_covariance

    return core_tf.BatchedLEDHFlowVJPTensors(
        pre_flow_particles=tf.concat(bar_pre_blocks, axis=1),
        prior_means=tf.concat(bar_prior_blocks, axis=1),
        observation_jacobian=tf.zeros(
            [aux.batch_size, aux.num_particles, 1, aux.state_dim],
            dtype=DTYPE,
        ),
        observation_residual=tf.concat(bar_residual_blocks, axis=1),
        transition_covariance=tf.convert_to_tensor(bar_transition_covariance, dtype=DTYPE),
        observation_covariance=tf.convert_to_tensor(bar_observation_covariance, dtype=DTYPE),
    )


def _configure_precision(args: argparse.Namespace) -> dict[str, Any]:
    global DTYPE
    dtype_name = getattr(args, "dtype", "float32")
    tf32_mode = getattr(args, "tf32_mode", "enabled")
    DTYPE = tf.float64 if dtype_name == "float64" else tf.float32
    value_mod.DTYPE = DTYPE
    p8p.DTYPE = DTYPE
    core_tf.DTYPE = DTYPE
    streaming_tf.DTYPE = DTYPE
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
        raise ValueError("actual-SV theta must have shape [2]")
    return tensor


def _gamma_beta(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    theta = _as_theta(theta)
    z_gamma = theta[0]
    log_beta = theta[1]
    gamma = tf.cast(_STD_NORMAL.cdf(tf.cast(z_gamma, tf.float64)), DTYPE)
    dgamma_dtheta = tf.cast(_STD_NORMAL.prob(tf.cast(z_gamma, tf.float64)), DTYPE)
    beta = tf.exp(log_beta)
    return gamma, beta, dgamma_dtheta


def _stationary_variance(gamma: tf.Tensor, sigma: tf.Tensor) -> tf.Tensor:
    return tf.square(sigma) / (tf.constant(1.0, dtype=DTYPE) - tf.square(gamma))


def _stationary_variance_theta_score(gamma: tf.Tensor, dgamma_dtheta: tf.Tensor, sigma: tf.Tensor) -> tf.Tensor:
    denominator = tf.constant(1.0, dtype=DTYPE) - tf.square(gamma)
    d_variance_d_gamma = (
        tf.constant(2.0, dtype=DTYPE)
        * tf.square(sigma)
        * gamma
        / tf.square(denominator)
    )
    return d_variance_d_gamma * dgamma_dtheta


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


def _require_manual_score_args(args: argparse.Namespace) -> None:
    if args.transport_plan_mode != "streaming":
        raise ValueError("actual-SV score repair requires streaming transport")
    if args.transport_ad_mode != "full":
        raise ValueError("actual-SV score repair requires transport_ad_mode='full'")
    if args.transport_gradient_mode not in {
        core_tf.MANUAL_STREAMING_FINITE_TRANSPORT_GRADIENT_MODE,
        core_tf.MANUAL_STREAMING_BLOCKWISE_VJP_FINITE_TRANSPORT_GRADIENT_MODE,
    }:
        raise ValueError("actual-SV score repair requires a manual streaming transport VJP")
    if args.transport_policy == "no-resampling":
        raise ValueError("actual-SV admitted score requires active relaxed Sinkhorn transport")


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
    tangent = (
        residual_bar[..., None] * (d_value - d_mean)
        + variance_bar[..., None] * d_variance
    )
    return density, tangent


def _actual_sv_target_observation_jvp(
    *,
    observation: tf.Tensor,
    target_state: tf.Tensor,
    beta: tf.Tensor,
    d_target_state: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    target_state = tf.convert_to_tensor(target_state, dtype=DTYPE)
    d_target_state = tf.convert_to_tensor(d_target_state, dtype=DTYPE)
    residual = (
        observation[0]
        - tf.constant(2.0, dtype=DTYPE) * tf.math.log(beta)
        - target_state
    )
    density = tf.cast(
        exact_log_chi_square_log_density(tf.cast(residual, tf.float64)),
        DTYPE,
    )
    d_residual = -d_target_state - tf.reshape(
        tf.constant([0.0, 2.0], dtype=DTYPE),
        [1, 1, len(PARAMETER_NAMES)],
    )
    log_chi_score_residual = tf.constant(0.5, dtype=DTYPE) * (
        tf.constant(1.0, dtype=DTYPE) - tf.exp(residual)
    )
    tangent = log_chi_score_residual[:, :, None] * d_residual
    return density, tangent


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


def _filterflow_epsilon_start_jvp(
    scaled_x: tf.Tensor,
    d_scaled_x: tf.Tensor,
) -> tf.Tensor:
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
        raise ValueError("compact actual-SV score requires manual streaming finite transport")
    if args.transport_ad_mode != "full":
        raise ValueError("compact actual-SV score requires transport_ad_mode='full'")
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
    d_uniform_log_weights = tf.zeros_like(d_normalized_log_weights)
    next_particles = raw_transport.particles
    next_log_weights = raw_transport.log_weights
    next_d_particles = tf.where(mask[:, None, None, None], d_transported, d_post_flow)
    next_d_log_weights = tf.where(
        mask[:, None, None],
        d_uniform_log_weights,
        d_normalized_log_weights,
    )
    return next_particles, next_log_weights, next_d_particles, next_d_log_weights


def _compact_streaming_flow_jvp_tf(
    *,
    pre_flow_particles: tf.Tensor,
    d_pre_flow_particles: tf.Tensor,
    ancestors: tf.Tensor,
    d_ancestors: tf.Tensor,
    observation: tf.Tensor,
    gamma: tf.Tensor,
    d_gamma: tf.Tensor,
    beta: tf.Tensor,
    proposal_variance: tf.Tensor,
    d_proposal_variance: tf.Tensor,
    is_initial: bool,
    flow_observation_covariance: tf.Tensor,
    particle_chunk_size: int,
) -> tuple[streaming_tf.StreamingLEDHFlowTensors, tf.Tensor, tf.Tensor, tf.Tensor]:
    x0 = tf.convert_to_tensor(pre_flow_particles, dtype=DTYPE)
    d_x0 = tf.convert_to_tensor(d_pre_flow_particles, dtype=DTYPE)
    ancestors = tf.convert_to_tensor(ancestors, dtype=DTYPE)
    d_ancestors = tf.convert_to_tensor(d_ancestors, dtype=DTYPE)
    batch_size, num_particles, state_dim = core_tf._static_shape(x0, "pre_flow_particles")  # noqa: SLF001
    if state_dim != 1:
        raise ValueError("actual-SV compact score expects one-dimensional state")
    param_dim = len(PARAMETER_NAMES)
    chunk_size = int(particle_chunk_size)
    num_blocks = (int(num_particles) + chunk_size - 1) // chunk_size
    transition_matrix = tf.ones([batch_size, 1, 1], dtype=DTYPE)
    proposal_covariance = tf.ones([batch_size, 1, 1], dtype=DTYPE) * proposal_variance

    post_blocks = []
    d_post_blocks = []
    pre_log_blocks = []
    d_pre_log_blocks = []
    logdet_blocks = []
    d_logdet_blocks = []

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
        d_pre_chunk = _zero_padded_axis1_3d(
            tf.reshape(d_x0, [batch_size, num_particles, state_dim * param_dim]),
            start,
            chunk_size,
        )
        d_pre_chunk = tf.reshape(d_pre_chunk, [batch_size, chunk_size, state_dim, param_dim])
        d_ancestor_chunk = _zero_padded_axis1_3d(
            tf.reshape(d_ancestors, [batch_size, num_particles, state_dim * param_dim]),
            start,
            chunk_size,
        )
        d_ancestor_chunk = tf.reshape(
            d_ancestor_chunk,
            [batch_size, chunk_size, state_dim, param_dim],
        )

        def observation_fn(points: tf.Tensor) -> tf.Tensor:
            return points + tf.reshape(
                tf.constant(2.0, dtype=DTYPE) * tf.math.log(beta),
                [1, 1, 1],
            )

        def observation_jacobian_fn(points: tf.Tensor) -> tf.Tensor:
            chunk_particles = points.shape[1]
            if chunk_particles is None:
                raise ValueError("actual-SV adapter requires static particle chunk dimension")
            return tf.ones([batch_size, int(chunk_particles), 1, 1], dtype=DTYPE)

        def observation_residual_fn(h_ref: tf.Tensor, obs: tf.Tensor) -> tf.Tensor:
            return obs[tf.newaxis, tf.newaxis, :] - h_ref

        def chunk_prior_mean_fn(chunk_ancestors: tf.Tensor) -> tf.Tensor:
            if is_initial:
                return tf.zeros_like(chunk_ancestors)
            return gamma * chunk_ancestors

        flow, aux = _core_flow_with_streaming_forward_aux_tf(
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
        if is_initial:
            prior_means = tf.zeros_like(pre_chunk)
            d_prior_means = tf.zeros_like(d_pre_chunk)
        else:
            prior_means = gamma * ancestor_chunk
            d_prior_means = (
                gamma * d_ancestor_chunk
                + ancestor_chunk[:, :, :, None]
                * tf.reshape(d_gamma, [1, 1, 1, param_dim])
            )

        prior_variance = tf.reshape(aux.transition_covariance_stable[:, 0, 0], [batch_size])
        obs_variance = tf.reshape(aux.observation_covariance_stable[:, 0, 0], [batch_size])
        prior_precision = tf.reshape(aux.prior_precision[:, 0, 0], [batch_size])
        obs_precision = tf.reshape(aux.obs_precision[:, 0, 0], [batch_size])
        residual = observation[0] - pre_chunk[:, :, 0] - tf.constant(2.0, dtype=DTYPE) * tf.math.log(beta)
        d_residual = -d_pre_chunk[:, :, 0, :] - tf.reshape(
            tf.constant([0.0, 2.0], dtype=DTYPE),
            [1, 1, param_dim],
        )
        pseudo_observation = pre_chunk[:, :, 0] + residual
        d_pseudo_observation = d_pre_chunk[:, :, 0, :] + d_residual
        post_precision = prior_precision[:, None] + obs_precision[:, None]
        d_prior_precision = -d_proposal_variance / tf.square(prior_variance)[:, None]
        d_post_precision = d_prior_precision
        post_covariance = tf.reshape(aux.post_covariance[:, :, 0, 0], [batch_size, chunk_size])
        d_post_covariance = -tf.square(post_covariance)[:, :, None] * d_post_precision[:, None, :]
        info = prior_precision[:, None] * prior_means[:, :, 0] + obs_precision[:, None] * pseudo_observation
        d_info = (
            d_prior_precision[:, None, :] * prior_means[:, :, 0, None]
            + prior_precision[:, None, None] * d_prior_means[:, :, 0, :]
            + obs_precision[:, None, None] * d_pseudo_observation
        )
        post_mean = flow.local_posterior_means[:, :, 0]
        d_post_mean = d_post_covariance * info[:, :, None] + post_covariance[:, :, None] * d_info
        delta = pre_chunk[:, :, 0] - prior_means[:, :, 0]
        d_delta = d_pre_chunk[:, :, 0, :] - d_prior_means[:, :, 0, :]
        prior_scale = tf.sqrt(prior_variance)
        post_scale = tf.sqrt(post_covariance)
        affine = post_scale / prior_scale[:, None]
        d_post_scale = d_post_covariance / (tf.constant(2.0, dtype=DTYPE) * post_scale[:, :, None])
        d_prior_scale = d_proposal_variance / (
            tf.constant(2.0, dtype=DTYPE) * prior_scale[:, None]
        )
        d_affine = (
            d_post_scale / prior_scale[:, None, None]
            - post_scale[:, :, None]
            * d_prior_scale[:, None, :]
            / tf.square(prior_scale)[:, None, None]
        )
        d_post = d_post_mean + d_affine * delta[:, :, None] + affine[:, :, None] * d_delta
        d_forward_log_det = (
            d_post_scale / post_scale[:, :, None]
            - d_prior_scale[:, None, :] / prior_scale[:, None, None]
        )
        _pre_log_density, d_pre_log_density = _log_normal_logpdf_jvp(
            pre_chunk[:, :, 0],
            prior_means[:, :, 0],
            prior_variance[:, None],
            d_pre_chunk[:, :, 0, :],
            d_prior_means[:, :, 0, :],
            d_proposal_variance[:, None, :],
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


def _compact_value_and_score_from_components(
    args: argparse.Namespace,
    theta_values: list[float] | tuple[float, ...],
) -> dict[str, tf.Tensor]:
    """Actual-SV same-scalar value and score with compact forward sensitivity."""

    _configure_precision(args)
    _require_manual_score_args(args)
    theta = tf.constant([float(value) for value in theta_values], dtype=DTYPE)
    gamma, beta, dgamma_dtheta = _gamma_beta(theta)
    sigma = tf.constant(1.0, dtype=DTYPE)
    stationary_variance = _stationary_variance(gamma, sigma)
    d_stationary_variance_scalar = _stationary_variance_theta_score(
        gamma,
        dgamma_dtheta,
        sigma,
    )
    d_gamma = tf.constant([1.0, 0.0], dtype=DTYPE) * dgamma_dtheta

    tensors, _semantics = value_mod._build_actual_sv_tensors(args)  # noqa: SLF001
    observations = tf.cast(tensors["observations"], DTYPE)
    fixed_resampling_mask = tf.convert_to_tensor(tensors["fixed_resampling_mask"], dtype=tf.bool)
    flow_observation_covariance = tf.tile(
        tf.cast(tensors["flow_observation_covariance"], DTYPE),
        [len(args.batch_seeds), 1, 1],
    )
    proposal_noise = _make_proposal_noise_tensor(args)
    batch_size = len(args.batch_seeds)
    num_particles = int(args.num_particles)
    state_dim = value_mod.STATE_DIM
    param_dim = len(PARAMETER_NAMES)
    particles = tf.cast(tensors["initial_particles"], DTYPE)
    if tuple(particles.shape.as_list()) != (batch_size, num_particles, state_dim):
        raise ValueError("actual-SV compact score initial_particles shape mismatch")
    initial_noise = particles / tf.sqrt(stationary_variance)
    d_particles = initial_noise[:, :, :, None] * tf.reshape(
        tf.stack(
            [
                d_stationary_variance_scalar
                / (tf.constant(2.0, dtype=DTYPE) * tf.sqrt(stationary_variance)),
                tf.constant(0.0, dtype=DTYPE),
            ]
        ),
        [1, 1, 1, param_dim],
    )
    log_weights = core_tf.uniform_log_weights(batch_size, num_particles)
    d_log_weights = tf.zeros([batch_size, num_particles, param_dim], dtype=DTYPE)
    log_likelihood = tf.zeros([batch_size], dtype=DTYPE)
    d_log_likelihood = tf.zeros([batch_size, param_dim], dtype=DTYPE)

    for time_index in range(int(args.time_steps)):
        observation = observations[time_index]
        is_initial = time_index == 0
        noise = proposal_noise[:, time_index, :, :]
        if is_initial:
            proposal_mean = tf.zeros_like(particles)
            d_proposal_mean = tf.zeros_like(d_particles)
            proposal_variance = stationary_variance
            d_proposal_variance = tf.tile(
                tf.reshape(
                    tf.stack(
                        [
                            d_stationary_variance_scalar,
                            tf.constant(0.0, dtype=DTYPE),
                        ]
                    ),
                    [1, param_dim],
                ),
                [batch_size, 1],
            )
        else:
            proposal_mean = gamma * particles
            d_proposal_mean = (
                gamma * d_particles
                + particles[:, :, :, None] * tf.reshape(d_gamma, [1, 1, 1, param_dim])
            )
            proposal_variance = tf.square(sigma)
            d_proposal_variance = tf.zeros([batch_size, param_dim], dtype=DTYPE)
        pre_flow = proposal_mean + tf.sqrt(proposal_variance) * noise
        d_pre_flow = d_proposal_mean + noise[:, :, :, None] * tf.reshape(
            d_proposal_variance / (tf.constant(2.0, dtype=DTYPE) * tf.sqrt(proposal_variance)),
            [batch_size, 1, 1, param_dim],
        )

        flow, d_post_flow, d_pre_flow_log_density, d_forward_log_det = (
            _compact_streaming_flow_jvp_tf(
                pre_flow_particles=pre_flow,
                d_pre_flow_particles=d_pre_flow,
                ancestors=particles,
                d_ancestors=d_particles,
                observation=observation,
                gamma=gamma,
                d_gamma=d_gamma,
                beta=beta,
                proposal_variance=proposal_variance,
                d_proposal_variance=d_proposal_variance,
                is_initial=is_initial,
                flow_observation_covariance=flow_observation_covariance,
                particle_chunk_size=int(args.particle_chunk_size),
            )
        )
        post_flow = flow.post_flow_particles
        target_state = post_flow[:, :, 0]
        d_target_state = d_post_flow[:, :, 0, :]
        if is_initial:
            target_transition, d_target_transition = _log_normal_logpdf_jvp(
                target_state,
                tf.constant(0.0, dtype=DTYPE),
                stationary_variance,
                d_target_state,
                tf.zeros_like(d_target_state),
                d_proposal_variance[:, None, :],
            )
        else:
            transition_mean = gamma * particles[:, :, 0]
            d_transition_mean = (
                gamma * d_particles[:, :, 0, :]
                + particles[:, :, 0, None] * tf.reshape(d_gamma, [1, 1, param_dim])
            )
            target_transition, d_target_transition = _log_normal_logpdf_jvp(
                target_state,
                transition_mean,
                tf.square(sigma),
                d_target_state,
                d_transition_mean,
                tf.zeros([batch_size, 1, param_dim], dtype=DTYPE),
            )
        target_observation, d_target_observation = _actual_sv_target_observation_jvp(
            observation=observation,
            target_state=target_state,
            beta=beta,
            d_target_state=d_target_state,
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
        "score_route": ACTUAL_SV_COMPACT_SCORE_ROUTE_ID,
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


def _target_transition_vjp(
    *,
    time_index: int,
    target_state: tf.Tensor,
    previous_particles: tf.Tensor,
    gamma: tf.Tensor,
    sigma: tf.Tensor,
    stationary_variance: tf.Tensor,
    dgamma_dtheta: tf.Tensor,
    upstream: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    state = target_state[:, :, 0]
    bar = tf.convert_to_tensor(upstream, dtype=DTYPE)
    batch_size = int(target_state.shape[0])
    score = tf.zeros([batch_size, len(PARAMETER_NAMES)], dtype=DTYPE)
    if int(time_index) == 0:
        bar_state = -bar * state / stationary_variance
        gamma_score = (
            -gamma / (tf.constant(1.0, dtype=DTYPE) - tf.square(gamma))
            + gamma * tf.square(state) / tf.square(sigma)
        ) * dgamma_dtheta
        score += tf.stack(
            [
                tf.reduce_sum(bar * gamma_score, axis=1),
                tf.zeros([batch_size], dtype=DTYPE),
            ],
            axis=1,
        )
        return bar_state[:, :, None], tf.zeros_like(previous_particles), score

    previous = previous_particles[:, :, 0]
    residual = state - gamma * previous
    inv_var = tf.constant(1.0, dtype=DTYPE) / tf.square(sigma)
    bar_state = -bar * residual * inv_var
    bar_previous = bar * gamma * residual * inv_var
    gamma_score = residual * previous * inv_var * dgamma_dtheta
    score += tf.stack(
        [
            tf.reduce_sum(bar * gamma_score, axis=1),
            tf.zeros([batch_size], dtype=DTYPE),
        ],
        axis=1,
    )
    return bar_state[:, :, None], bar_previous[:, :, None], score


def _target_observation_vjp(
    *,
    observation: tf.Tensor,
    target_state: tf.Tensor,
    beta: tf.Tensor,
    upstream: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    state = target_state[:, :, 0]
    residual = observation[0] - tf.constant(2.0, dtype=DTYPE) * tf.math.log(beta) - state
    bar = tf.convert_to_tensor(upstream, dtype=DTYPE)
    log_chi_score_residual = tf.constant(0.5, dtype=DTYPE) * (
        tf.constant(1.0, dtype=DTYPE) - tf.exp(residual)
    )
    bar_residual = bar * log_chi_score_residual
    bar_state = -bar_residual
    beta_score = -tf.constant(2.0, dtype=DTYPE) * bar_residual
    score = tf.stack(
        [
            tf.zeros([int(target_state.shape[0])], dtype=DTYPE),
            tf.reduce_sum(beta_score, axis=1),
        ],
        axis=1,
    )
    return bar_state[:, :, None], score


def _manual_value_and_score_from_components(
    args: argparse.Namespace,
    theta_values: list[float] | tuple[float, ...],
) -> dict[str, tf.Tensor]:
    """Historical reverse/manual actual-SV score diagnostic."""

    _configure_precision(args)
    _require_manual_score_args(args)
    if len(args.batch_seeds) != 1:
        raise ValueError("actual-SV score repair evaluates one seed at a time")
    theta = tf.constant([float(value) for value in theta_values], dtype=DTYPE)
    gamma, beta, dgamma_dtheta = _gamma_beta(theta)
    sigma = tf.constant(1.0, dtype=DTYPE)
    stationary_variance = _stationary_variance(gamma, sigma)

    tensors, _semantics = value_mod._build_actual_sv_tensors(args)  # noqa: SLF001
    observations = tf.cast(tensors["observations"], DTYPE)
    fixed_resampling_mask = tf.convert_to_tensor(tensors["fixed_resampling_mask"], dtype=tf.bool)
    flow_observation_covariance = tf.tile(
        tf.cast(tensors["flow_observation_covariance"], DTYPE),
        [len(args.batch_seeds), 1, 1],
    )
    proposal_noise = _make_proposal_noise_tensor(args)
    batch_size = len(args.batch_seeds)
    num_particles = int(args.num_particles)
    state_dim = value_mod.STATE_DIM
    particles = tf.cast(tensors["initial_particles"], DTYPE)
    if tuple(particles.shape.as_list()) != (batch_size, num_particles, state_dim):
        raise ValueError("actual-SV score initial_particles shape mismatch")
    log_weights = core_tf.uniform_log_weights(batch_size, num_particles)
    log_likelihood = tf.zeros([batch_size], dtype=DTYPE)
    transition_matrix = tf.ones([batch_size, 1, 1], dtype=DTYPE)
    records: list[dict[str, Any]] = []

    for time_index in range(int(args.time_steps)):
        observation = observations[time_index]
        is_initial = time_index == 0
        if is_initial:
            proposal_mean = tf.zeros_like(particles)
            proposal_variance = stationary_variance
        else:
            proposal_mean = gamma * particles
            proposal_variance = tf.square(sigma)
        proposal_covariance = tf.ones([batch_size, 1, 1], dtype=DTYPE) * proposal_variance
        noise = proposal_noise[:, time_index, :, :]
        pre_flow = proposal_mean + tf.sqrt(proposal_variance) * noise

        def observation_fn(points: tf.Tensor) -> tf.Tensor:
            return points + tf.reshape(
                tf.constant(2.0, dtype=DTYPE) * tf.math.log(beta),
                [1, 1, 1],
            )

        def observation_jacobian_fn(points: tf.Tensor) -> tf.Tensor:
            chunk_particles = points.shape[1]
            if chunk_particles is None:
                raise ValueError("actual-SV adapter requires static particle chunk dimension")
            return tf.ones([batch_size, int(chunk_particles), 1, 1], dtype=DTYPE)

        def observation_residual_fn(h_ref: tf.Tensor, obs: tf.Tensor) -> tf.Tensor:
            return obs[tf.newaxis, tf.newaxis, :] - h_ref

        def chunk_prior_mean_fn(ancestors: tf.Tensor) -> tf.Tensor:
            if is_initial:
                return tf.zeros_like(ancestors)
            return gamma * ancestors

        flow, flow_aux = _streaming_flow_with_aux_tf(
            pre_flow_particles=pre_flow,
            ancestors=particles,
            observation=observation,
            transition_matrix=transition_matrix,
            transition_covariance=proposal_covariance,
            observation_covariance=flow_observation_covariance,
            observation_fn=observation_fn,
            observation_jacobian_fn=observation_jacobian_fn,
            observation_residual_fn=observation_residual_fn,
            prior_mean_fn=chunk_prior_mean_fn,
            particle_chunk_size=int(args.particle_chunk_size),
        )
        post_flow = flow.post_flow_particles
        target_state = post_flow[:, :, 0]
        if is_initial:
            target_transition = value_mod._log_normal_logpdf(  # noqa: SLF001
                target_state,
                tf.constant(0.0, dtype=DTYPE),
                stationary_variance,
            )
        else:
            target_transition = value_mod._log_normal_logpdf(  # noqa: SLF001
                target_state,
                gamma * particles[:, :, 0],
                tf.square(sigma),
            )
        exact_residual = (
            observation[0]
            - tf.constant(2.0, dtype=DTYPE) * tf.math.log(beta)
            - target_state
        )
        target_observation = tf.cast(
            exact_log_chi_square_log_density(tf.cast(exact_residual, tf.float64)),
            DTYPE,
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
        mask = fixed_resampling_mask[:, time_index]
        next_particles, next_log_weights = _forward_transport_tf(
            post_flow=post_flow,
            normalized_log_weights=normalized_log_weights,
            mask=mask,
            args=args,
        )
        records.append(
            {
                "time_index": time_index,
                "particles": particles,
                "proposal_mean": proposal_mean,
                "proposal_variance": proposal_variance,
                "proposal_covariance": proposal_covariance,
                "noise": noise,
                "post_flow": post_flow,
                "observation": observation,
                "corrected_log_weights": corrected_log_weights,
                "normalized_log_weights": normalized_log_weights,
                "mask": mask,
                "flow_aux": flow_aux,
            }
        )
        particles = next_particles
        log_weights = next_log_weights
        log_likelihood = log_likelihood + incremental

    bar_particles = tf.zeros_like(particles)
    bar_log_weights = tf.zeros_like(log_weights)
    per_seed_score = tf.zeros([batch_size, len(PARAMETER_NAMES)], dtype=DTYPE)
    d_stationary_variance = _stationary_variance_theta_score(gamma, dgamma_dtheta, sigma)

    for record in reversed(records):
        time_index = int(record["time_index"])
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

        transition_bar_post, transition_bar_previous, transition_score = _target_transition_vjp(
            time_index=time_index,
            target_state=post_flow,
            previous_particles=record["particles"],
            gamma=gamma,
            sigma=sigma,
            stationary_variance=stationary_variance,
            dgamma_dtheta=dgamma_dtheta,
            upstream=correction_bars["transition_log_density"],
        )
        observation_bar_post, observation_score = _target_observation_vjp(
            observation=record["observation"],
            target_state=post_flow,
            beta=beta,
            upstream=correction_bars["observation_log_density"],
        )
        bar_post += transition_bar_post + observation_bar_post
        per_seed_score += transition_score + observation_score

        flow_vjp = _streaming_flow_vjp(
            record["flow_aux"],
            bar_post,
            correction_bars["pre_flow_log_density"],
            correction_bars["forward_log_det"],
        )
        bar_pre_flow = flow_vjp.pre_flow_particles - flow_vjp.observation_residual
        flow_beta_score = -tf.constant(2.0, dtype=DTYPE) * tf.reduce_sum(
            flow_vjp.observation_residual[:, :, 0],
            axis=1,
        )
        per_seed_score += tf.stack(
            [tf.zeros([batch_size], dtype=DTYPE), flow_beta_score],
            axis=1,
        )

        bar_proposal_mean = flow_vjp.prior_means + bar_pre_flow
        if time_index == 0:
            bar_variance = tf.reduce_sum(flow_vjp.transition_covariance[:, 0, 0], axis=0)
            bar_variance += tf.reduce_sum(
                bar_pre_flow[:, :, 0]
                * record["noise"][:, :, 0]
                * tf.constant(0.5, dtype=DTYPE)
                / tf.sqrt(stationary_variance),
                axis=[0, 1],
            )
            per_seed_score += tf.stack(
                [
                    tf.fill([batch_size], bar_variance * d_stationary_variance),
                    tf.zeros([batch_size], dtype=DTYPE),
                ],
                axis=1,
            )
            bar_particles = transition_bar_previous
        else:
            theta_from_mean = tf.reduce_sum(
                bar_proposal_mean[:, :, 0]
                * record["particles"][:, :, 0]
                * dgamma_dtheta,
                axis=1,
            )
            bar_particles = (
                transition_bar_previous
                + gamma * bar_proposal_mean
            )
            per_seed_score += tf.stack(
                [theta_from_mean, tf.zeros([batch_size], dtype=DTYPE)],
                axis=1,
            )
        bar_log_weights = next_bar_log_weights

    return {
        "objective": tf.reduce_mean(log_likelihood),
        "log_likelihood": log_likelihood,
        "gradient_tensor": tf.reduce_mean(per_seed_score, axis=0),
        "per_seed_gradient": per_seed_score,
        "score_route": ACTUAL_SV_MEMORY_STYLE_SCORE_ROUTE_ID,
        "historical_manual_score_route": ACTUAL_SV_MANUAL_SCORE_ROUTE_ID,
        "no_autodiff_score_route": True,
        "value_score_route_status": "same_route_value_score",
    }


def _manual_value_only_from_components(
    args: argparse.Namespace,
    theta_values: list[float] | tuple[float, ...],
) -> dict[str, tf.Tensor]:
    _configure_precision(args)
    _require_manual_score_args(args)
    if len(args.batch_seeds) != 1:
        raise ValueError("actual-SV score repair evaluates one seed at a time")
    theta = tf.constant([float(value) for value in theta_values], dtype=DTYPE)
    gamma, beta, _dgamma_dtheta = _gamma_beta(theta)
    sigma = tf.constant(1.0, dtype=DTYPE)
    stationary_variance = _stationary_variance(gamma, sigma)
    tensors, _semantics = value_mod._build_actual_sv_tensors(args)  # noqa: SLF001
    observations = tf.cast(tensors["observations"], DTYPE)
    fixed_resampling_mask = tf.convert_to_tensor(tensors["fixed_resampling_mask"], dtype=tf.bool)
    flow_observation_covariance = tf.tile(
        tf.cast(tensors["flow_observation_covariance"], DTYPE),
        [len(args.batch_seeds), 1, 1],
    )
    proposal_noise = _make_proposal_noise_tensor(args)
    batch_size = len(args.batch_seeds)
    num_particles = int(args.num_particles)
    particles = tf.cast(tensors["initial_particles"], DTYPE)
    if tuple(particles.shape.as_list()) != (batch_size, num_particles, value_mod.STATE_DIM):
        raise ValueError("actual-SV score initial_particles shape mismatch")
    log_weights = core_tf.uniform_log_weights(batch_size, num_particles)
    log_likelihood = tf.zeros([batch_size], dtype=DTYPE)
    transition_matrix = tf.ones([batch_size, 1, 1], dtype=DTYPE)
    for time_index in range(int(args.time_steps)):
        observation = observations[time_index]
        is_initial = time_index == 0
        if time_index == 0:
            proposal_mean = tf.zeros_like(particles)
            proposal_variance = stationary_variance
        else:
            proposal_mean = gamma * particles
            proposal_variance = tf.square(sigma)
        proposal_covariance = tf.ones([batch_size, 1, 1], dtype=DTYPE) * proposal_variance
        noise = proposal_noise[:, time_index, :, :]
        pre_flow = proposal_mean + tf.sqrt(proposal_variance) * noise

        def observation_fn(points: tf.Tensor) -> tf.Tensor:
            return points + tf.reshape(
                tf.constant(2.0, dtype=DTYPE) * tf.math.log(beta),
                [1, 1, 1],
            )

        def observation_jacobian_fn(points: tf.Tensor) -> tf.Tensor:
            chunk_particles = points.shape[1]
            if chunk_particles is None:
                raise ValueError("actual-SV adapter requires static particle chunk dimension")
            return tf.ones([batch_size, int(chunk_particles), 1, 1], dtype=DTYPE)

        def observation_residual_fn(h_ref: tf.Tensor, obs: tf.Tensor) -> tf.Tensor:
            return obs[tf.newaxis, tf.newaxis, :] - h_ref

        def chunk_prior_mean_fn(ancestors: tf.Tensor) -> tf.Tensor:
            if is_initial:
                return tf.zeros_like(ancestors)
            return gamma * ancestors

        flow, _flow_aux = _streaming_flow_with_aux_tf(
            pre_flow_particles=pre_flow,
            ancestors=particles,
            observation=observation,
            transition_matrix=transition_matrix,
            transition_covariance=proposal_covariance,
            observation_covariance=flow_observation_covariance,
            observation_fn=observation_fn,
            observation_jacobian_fn=observation_jacobian_fn,
            observation_residual_fn=observation_residual_fn,
            prior_mean_fn=chunk_prior_mean_fn,
            particle_chunk_size=int(args.particle_chunk_size),
        )
        post_flow = flow.post_flow_particles
        target_state = post_flow[:, :, 0]
        if time_index == 0:
            target_transition = value_mod._log_normal_logpdf(  # noqa: SLF001
                target_state,
                tf.constant(0.0, dtype=DTYPE),
                stationary_variance,
            )
        else:
            target_transition = value_mod._log_normal_logpdf(  # noqa: SLF001
                target_state,
                gamma * particles[:, :, 0],
                tf.square(sigma),
            )
        exact_residual = observation[0] - 2.0 * tf.math.log(beta) - target_state
        target_observation = tf.cast(
            exact_log_chi_square_log_density(tf.cast(exact_residual, tf.float64)),
            DTYPE,
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
    theta_values: list[float] | tuple[float, ...],
) -> dict[str, tf.Tensor]:
    """Historical reverse/manual across-seed score diagnostic only."""

    seed_results = [
        _manual_value_and_score_from_components(_single_seed_args(args, seed), theta_values)
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
        "score_route": ACTUAL_SV_MEMORY_STYLE_SCORE_ROUTE_ID,
        "historical_manual_score_route": ACTUAL_SV_MANUAL_SCORE_ROUTE_ID,
        "historical_compact_score_route": ACTUAL_SV_COMPACT_SCORE_ROUTE_ID,
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
        expected_row_id=ACTUAL_SV_ROW_ID,
        require_admitted=True,
    )
    base_raw = diagnostic.get("base")
    if not isinstance(base_raw, Mapping):
        raise ValueError("actual-SV compact diagnostic must include base mapping")
    base = base_raw
    if base.get("score_route") != ACTUAL_SV_COMPACT_SCORE_ROUTE_ID:
        raise ValueError("actual-SV compact diagnostic must use compact score route")
    if base.get("no_autodiff_score_route") is not True:
        raise ValueError("actual-SV compact diagnostic must declare no_autodiff_score_route")
    if base.get("value_score_route_status") != LEDH_SCORE_VALUE_ROUTE_STATUS_SAME:
        raise ValueError("actual-SV compact diagnostic must be same_route_value_score")
    if tuple(diagnostic.get("parameter_names", ())) != tuple(PARAMETER_NAMES):
        raise ValueError("actual-SV compact diagnostic parameter_names must match parameter order")
    score = tf.convert_to_tensor(base["gradient_tensor"], dtype=DTYPE)
    memory = dict(memory_diagnostics or {})
    memory_pass = bool(memory.get("n10000_memory_pass") is True)
    if "source" not in memory and memory:
        memory["source"] = "trusted_gpu_score_memory_artifact"
    if require_all_parameter_correctness:
        if diagnostic.get("status") != "pass":
            raise ValueError("actual-SV all-parameter correctness status must pass")
        if tuple(diagnostic.get("parameter_names", PARAMETER_NAMES)) != tuple(PARAMETER_NAMES):
            raise ValueError("actual-SV all-parameter correctness parameter_names mismatch")
        if not memory_pass:
            raise ValueError("actual-SV full admission requires N=10000 memory pass")
        if int(base.get("num_particles", -1)) != int(value_core["num_particles"]):
            raise ValueError("actual-SV compact full admission requires N=10000 diagnostic shape")
        if int(base.get("time_steps", -1)) != int(value_core["time_steps"]):
            raise ValueError("actual-SV compact full admission requires full time_steps")
        if tuple(int(seed) for seed in base.get("batch_seeds", ())) != tuple(
            int(seed) for seed in value_core["batch_seeds"]
        ):
            raise ValueError("actual-SV compact full admission requires full batch_seeds")
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
        "row_id": ACTUAL_SV_ROW_ID,
        "source_value_artifact": source_value_artifact_path,
        "score_target_kind": LEDH_SCORE_TARGET_KIND_REALIZED_FINITE_N_ESTIMATOR,
        "target_scalar": LEDH_TARGET_SCALAR_OBSERVED_DATA_LOG_LIKELIHOOD,
        "target_output_tensor_field": LEDH_OUTPUT_TENSOR_FIELD_LOG_LIKELIHOOD,
        "target_observation_policy": value_core["target_observation_policy"],
        "theta_coordinate_system": value_core["theta_coordinate_system"],
        "score_parameter_names": list(PARAMETER_NAMES),
        "score": [float(value) for value in score.numpy().reshape(-1)],
        "score_derivative_provenance": str(
            base.get("score_route", ACTUAL_SV_COMPACT_SCORE_ROUTE_ID)
        ),
        "value_score_route_status": LEDH_SCORE_VALUE_ROUTE_STATUS_SAME,
        "value_score_same_transport_algorithm": True,
        "no_autodiff_score_route": True,
        "uses_gradient_tape": False,
        "uses_forward_accumulator": False,
        "uses_stopped_partial_derivative": False,
        "claims_exact_native_actual_sv_likelihood": False,
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
            expected_row_id=ACTUAL_SV_ROW_ID,
            require_admitted=True,
        )
    return artifact


def _score_artifact_from_value_score(
    value_score: dict[str, tf.Tensor],
    *,
    source_value_artifact: dict[str, Any],
    source_value_artifact_path: str,
    memory_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    value_core = validate_ledh_forward_scalar_artifact(
        source_value_artifact,
        expected_row_id=ACTUAL_SV_ROW_ID,
        require_admitted=True,
    )
    score = tf.convert_to_tensor(value_score["gradient_tensor"], dtype=DTYPE)
    memory = dict(memory_diagnostics or {})
    if "source" not in memory and memory:
        memory["source"] = "trusted_gpu_score_memory_artifact"
    return {
        "schema_version": LEDH_SCORE_ARTIFACT_SCHEMA_VERSION,
        "row_id": ACTUAL_SV_ROW_ID,
        "source_value_artifact": source_value_artifact_path,
        "score_target_kind": LEDH_SCORE_TARGET_KIND_REALIZED_FINITE_N_ESTIMATOR,
        "target_scalar": LEDH_TARGET_SCALAR_OBSERVED_DATA_LOG_LIKELIHOOD,
        "target_output_tensor_field": LEDH_OUTPUT_TENSOR_FIELD_LOG_LIKELIHOOD,
        "target_observation_policy": value_core["target_observation_policy"],
        "theta_coordinate_system": value_core["theta_coordinate_system"],
        "score_parameter_names": list(PARAMETER_NAMES),
        "score": [float(value) for value in score.numpy().reshape(-1)],
        "score_derivative_provenance": str(
            value_score.get("score_route", ACTUAL_SV_COMPACT_SCORE_ROUTE_ID)
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
            "status": "not_run",
            "reason": "no_fd_value_score_diagnostic_only",
        },
        "score_admission_status": LEDH_SCORE_ADMISSION_STATUS_BLOCKED_NOT_RUN,
        "memory_diagnostics": memory,
    }


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
    parser.add_argument("--flow-observation-variance", type=float, default=math.pi * math.pi / 2.0)
    parser.add_argument("--transport-plan-mode", choices=("streaming", "dense"), default="streaming")
    parser.add_argument(
        "--transport-gradient-mode",
        choices=(
            core_tf.MANUAL_STREAMING_FINITE_TRANSPORT_GRADIENT_MODE,
            core_tf.MANUAL_STREAMING_BLOCKWISE_VJP_FINITE_TRANSPORT_GRADIENT_MODE,
        ),
        default=core_tf.MANUAL_STREAMING_FINITE_TRANSPORT_GRADIENT_MODE,
    )
    parser.add_argument("--transport-ad-mode", choices=("full",), default="full")
    parser.add_argument("--row-chunk-size", type=int, default=64)
    parser.add_argument("--col-chunk-size", type=int, default=64)
    parser.add_argument("--particle-chunk-size", type=int, default=64)
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
    parser.add_argument(
        "--diagnostic-mode",
        choices=("coordinate-fd", "value-score-only"),
        default="coordinate-fd",
    )
    args = parser.parse_args()
    args.batch_seeds = _parse_int_csv(args.batch_seeds)
    if args.time_steps <= 0 or args.num_particles <= 1:
        raise ValueError("time_steps must be positive and num_particles must exceed one")
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
        "# Actual-SV Same-Target LEDH Score",
        "",
        f"- JSON artifact: `{json_path}`",
        f"- Row: `{artifact['row_id']}`",
        f"- Score admission status: `{artifact['score_admission_status']}`",
        f"- Correctness: `{artifact['score_correctness']}`",
        f"- Memory diagnostics: `{artifact['memory_diagnostics']}`",
        "",
        "## Nonclaims",
        "",
        "- This artifact is not full actual-SV score admission.",
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
    if args.diagnostic_mode == "coordinate-fd":
        diagnostic = _coordinate_fd_score_diagnostic(
            args,
            list(TRUTH_THETA),
            fd_step=float(args.fd_step),
            atol=float(args.score_fd_atol),
            rtol=float(args.score_fd_rtol),
        )
    else:
        diagnostic = {
            "status": "not_run",
            "base": _compact_value_and_score_from_components(args, list(TRUTH_THETA)),
        }
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
    if args.diagnostic_mode == "coordinate-fd":
        artifact = _score_artifact_from_diagnostic(
            diagnostic,
            source_value_artifact=source_value,
            source_value_artifact_path=str(source_path),
            require_all_parameter_correctness=bool(args.admit_full),
            memory_diagnostics=memory,
        )
    else:
        if args.admit_full:
            raise ValueError("value-score-only diagnostic cannot admit full score")
        artifact = _score_artifact_from_value_score(
            diagnostic["base"],
            source_value_artifact=source_value,
            source_value_artifact_path=str(source_path),
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
            "diagnostic_mode": args.diagnostic_mode,
            "nonclaims": [
                "not full actual-SV score admission",
                "not KSC surrogate likelihood evidence",
                "not raw Gaussian observation likelihood evidence",
                "not augmented-noise Gaussian-closure evidence",
                "not HMC readiness evidence",
                "not posterior correctness evidence",
                "not scientific superiority evidence",
                "not runtime ranking evidence",
            ],
        }
    )
    if args.diagnostic_mode == "coordinate-fd":
        artifact.update(
            {
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
