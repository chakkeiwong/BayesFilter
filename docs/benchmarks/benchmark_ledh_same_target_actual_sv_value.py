"""Exact-transformed actual-SV LEDH-PFPF-OT value runner.

This runner targets the actual-SV row
``zhao_cui_sv_actual_nongaussian_T1000``.  By default it emits a tiny
``tiny_executed_not_full_row`` adapter-smoke artifact.  Full row admission
requires the explicit ``--run-scope full-row-admission`` guard and exact
``N=10000,T=1000`` settings.

The target correction is exact transformed actual SV:

``z_t = log(y_t**2)``,
``z_t - 2 log(beta) - x_t ~ log(chi_square_1)``.

The Gaussianized LEDH observation surface is a proposal surface only.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


_PRE_PARSER = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
_PRE_PARSER.add_argument("--device-scope", choices=("cpu", "visible"), default="visible")
_PRE_PARSER.add_argument("--cuda-visible-devices", default=None)
_PRE_ARGS, _UNKNOWN = _PRE_PARSER.parse_known_args()
if _PRE_ARGS.device_scope == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
elif _PRE_ARGS.cuda_visible_devices is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _PRE_ARGS.cuda_visible_devices
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

from bayesfilter.highdim.ledh_forward_contract import (
    ACTUAL_SV_ROW_ID,
    LEDH_FORWARD_ADMISSION_STATUS_ADMITTED,
    LEDH_FORWARD_ADMISSION_STATUS_BLOCKED_NONFINITE,
    LEDH_FORWARD_ADMISSION_STATUS_TINY,
    LEDH_FORWARD_SCALAR_ARTIFACT_SCHEMA_VERSION,
    make_actual_sv_forward_contract,
    validate_ledh_forward_scalar_artifact,
)
from bayesfilter.highdim.models import StochasticVolatilitySSM
from bayesfilter.highdim.sv_mixture_cut4 import (
    exact_log_chi_square_log_density,
    exact_transformed_sv_observations,
)
from experiments.dpf_implementation.tf_tfp.filters import (
    experimental_batched_ledh_pfpf_ot_streaming_tf as streaming_tf,
)
from experiments.dpf_implementation.tf_tfp.filters import (
    experimental_batched_ledh_pfpf_ot_tf as core_tf,
)
from experiments.dpf_implementation.tf_tfp.resampling import annealed_transport_tf
from scripts.filtering_value_gradient_benchmark_generate_p8_datasets import _sv_dataset


DATASET_SEED = 81101
TRUTH_THETA = [0.2533471031357997, -0.916290731874155]
FULL_ROW_BATCH_SEEDS = (81120, 81121, 81122, 81123, 81124)
FULL_ROW_NUM_PARTICLES = 10000
FULL_ROW_TIME_STEPS = 1000
STATE_DIM = 1
OBS_DIM = 1
DTYPE = tf.float32
BASE_NONCLAIMS = (
    "not score admission",
    "not score correctness",
    "not KSC surrogate likelihood evidence",
    "not raw Gaussian observation likelihood evidence",
    "not augmented-noise Gaussian-closure evidence",
    "not generalized-SV admission",
    "not HMC readiness evidence",
    "not posterior correctness evidence",
    "not scientific superiority evidence",
    "not runtime ranking evidence",
)
TINY_NONCLAIMS = (
    "not full actual-SV row admission",
    *BASE_NONCLAIMS,
)
FULL_ROW_NONCLAIMS = (
    "not score admission",
    "not score correctness",
    "not KSC surrogate likelihood evidence",
    "not raw Gaussian observation likelihood evidence",
    "not augmented-noise Gaussian-closure evidence",
    "not generalized-SV admission",
    "not HMC readiness evidence",
    "not posterior correctness evidence",
    "not scientific superiority evidence",
    "not runtime ranking evidence",
)


def _parse_int_csv(value: str) -> list[int]:
    entries = [item.strip() for item in str(value).split(",") if item.strip()]
    if not entries:
        raise ValueError("expected at least one integer")
    return [int(item) for item in entries]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--run-scope",
        choices=("tiny-smoke", "full-row-admission"),
        default="tiny-smoke",
    )
    parser.add_argument("--batch-seeds", default="81120")
    parser.add_argument("--time-steps", type=int, default=4)
    parser.add_argument("--num-particles", type=int, default=128)
    parser.add_argument(
        "--transport-policy",
        choices=("active-all", "active-odd", "no-resampling"),
        default="active-all",
    )
    parser.add_argument("--sinkhorn-iterations", type=int, default=2)
    parser.add_argument("--sinkhorn-epsilon", type=float, default=1.0)
    parser.add_argument("--annealed-scaling", type=float, default=0.9)
    parser.add_argument("--annealed-convergence-threshold", type=float, default=1.0e-3)
    parser.add_argument("--flow-observation-variance", type=float, default=math.pi * math.pi / 2.0)
    parser.add_argument("--row-chunk-size", type=int, default=64)
    parser.add_argument("--col-chunk-size", type=int, default=64)
    parser.add_argument("--particle-chunk-size", type=int, default=64)
    parser.add_argument("--history-mode", choices=("full", "value-only"), default="full")
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float32")
    parser.add_argument("--tf32-mode", choices=("default", "enabled", "disabled"), default="enabled")
    parser.add_argument("--device", default="/GPU:0")
    parser.add_argument("--device-scope", choices=("cpu", "visible"), default=_PRE_ARGS.device_scope)
    parser.add_argument("--cuda-visible-devices", default=_PRE_ARGS.cuda_visible_devices)
    parser.add_argument("--expect-device-kind", choices=("any", "cpu", "gpu"), default="gpu")
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output", default=None)
    args = parser.parse_args()
    args.batch_seeds = _parse_int_csv(args.batch_seeds)
    if args.time_steps <= 0 or args.time_steps > FULL_ROW_TIME_STEPS:
        raise ValueError("time_steps must be in 1..1000")
    if args.num_particles <= 1:
        raise ValueError("num_particles must be greater than one")
    if args.sinkhorn_iterations <= 0:
        raise ValueError("sinkhorn_iterations must be positive")
    if args.flow_observation_variance <= 0.0:
        raise ValueError("flow_observation_variance must be positive")
    if args.warmups < 0 or args.repeats <= 0:
        raise ValueError("warmups must be nonnegative and repeats must be positive")
    if args.row_chunk_size <= 0 or args.col_chunk_size <= 0 or args.particle_chunk_size <= 0:
        raise ValueError("chunk sizes must be positive")
    if args.run_scope == "tiny-smoke" and _full_row_requested(args):
        raise ValueError("adapter smoke runner must not run the full actual-SV row")
    if args.run_scope == "full-row-admission" and not _exact_full_row_requested(args):
        raise ValueError(
            "full-row-admission requires exact actual-SV full row settings: "
            "T=1000, N=10000, seeds=81120,81121,81122,81123,81124"
        )
    return args


def _full_row_requested(args: argparse.Namespace) -> bool:
    return (
        tuple(int(seed) for seed in args.batch_seeds) == FULL_ROW_BATCH_SEEDS
        and int(args.num_particles) >= FULL_ROW_NUM_PARTICLES
        and int(args.time_steps) >= FULL_ROW_TIME_STEPS
    )


def _exact_full_row_requested(args: argparse.Namespace) -> bool:
    return (
        tuple(int(seed) for seed in args.batch_seeds) == FULL_ROW_BATCH_SEEDS
        and int(args.num_particles) == FULL_ROW_NUM_PARTICLES
        and int(args.time_steps) == FULL_ROW_TIME_STEPS
    )


def _configure_precision(args: argparse.Namespace) -> dict[str, Any]:
    global DTYPE
    DTYPE = tf.float64 if args.dtype == "float64" else tf.float32
    core_tf.DTYPE = DTYPE
    streaming_tf.DTYPE = DTYPE
    annealed_transport_tf.DTYPE = DTYPE
    if args.tf32_mode != "default":
        tf.config.experimental.enable_tensor_float_32_execution(args.tf32_mode == "enabled")
    metadata = core_tf.precision_policy_metadata()
    metadata.update(
        {
            "dtype": args.dtype,
            "tf_dtype": DTYPE.name,
            "tf32_mode": args.tf32_mode,
            "tf32_execution_enabled": bool(
                tf.config.experimental.tensor_float_32_execution_enabled()
            ),
        }
    )
    return metadata


def _configure_gpus() -> tuple[list[str], list[str]]:
    physical_gpus = tf.config.list_physical_devices("GPU")
    for gpu in physical_gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    logical_gpus = tf.config.list_logical_devices("GPU")
    return ([str(device) for device in physical_gpus], [str(device) for device in logical_gpus])


def _gpu_memory_info() -> dict[str, Any]:
    try:
        return dict(tf.config.experimental.get_memory_info("GPU:0"))
    except (ValueError, RuntimeError):
        return {"status": "unavailable"}


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


def _summary(timings: list[float]) -> dict[str, float]:
    if not timings:
        return {}
    return {
        "min": min(timings),
        "median": statistics.median(timings),
        "mean": statistics.fmean(timings),
        "max": max(timings),
    }


def _seed_pair(seed: int, salt: int | tf.Tensor) -> tf.Tensor:
    salt_tensor = tf.cast(salt, tf.int32)
    return tf.stack(
        [
            tf.constant(int(seed) % 2147483647, dtype=tf.int32),
            tf.math.floormod(salt_tensor, tf.constant(2147483647, dtype=tf.int32)),
        ]
    )


def _make_fixed_resampling_mask(batch_size: int, time_steps: int, policy: str) -> tf.Tensor:
    if policy == "active-all":
        mask = tf.ones([batch_size, time_steps], dtype=tf.bool)
    elif policy == "active-odd":
        active = tf.equal(tf.math.floormod(tf.range(time_steps), 2), 1)
        mask = tf.tile(active[tf.newaxis, :], [batch_size, 1])
    else:
        mask = tf.zeros([batch_size, time_steps], dtype=tf.bool)
    return mask


def _log_normal_logpdf(value: tf.Tensor, mean: tf.Tensor, variance: tf.Tensor) -> tf.Tensor:
    return -0.5 * (
        tf.math.log(tf.constant(2.0 * math.pi, dtype=DTYPE))
        + tf.math.log(variance)
        + tf.square(value - mean) / variance
    )


def _build_actual_sv_tensors(args: argparse.Namespace) -> tuple[dict[str, tf.Tensor], dict[str, Any]]:
    model = StochasticVolatilitySSM(sigma=1.0)
    theta64 = tf.constant(TRUTH_THETA, dtype=tf.float64)
    parameters64 = model.physical_parameters(theta64)
    gamma64 = parameters64["gamma"]
    beta64 = parameters64["beta"]
    sigma64 = parameters64["sigma"]
    dataset = _sv_dataset(DATASET_SEED)
    raw_observations64 = tf.convert_to_tensor(dataset["observations"], dtype=tf.float64)[: args.time_steps]
    transformed64 = exact_transformed_sv_observations(raw_observations64)
    gamma = tf.cast(gamma64, DTYPE)
    beta = tf.cast(beta64, DTYPE)
    sigma = tf.cast(sigma64, DTYPE)
    stationary_variance = tf.square(sigma) / (tf.constant(1.0, dtype=DTYPE) - tf.square(gamma))
    stationary_scale = tf.sqrt(stationary_variance)
    batch_size = len(args.batch_seeds)
    initial_particles = tf.stack(
        [
            stationary_scale
            * tf.random.stateless_normal(
                [args.num_particles, STATE_DIM],
                seed=_seed_pair(int(seed), 500),
                dtype=DTYPE,
            )
            for seed in args.batch_seeds
        ],
        axis=0,
    )
    tensors = {
        "observations": tf.cast(transformed64, DTYPE),
        "raw_observations": tf.cast(raw_observations64, DTYPE),
        "initial_particles": initial_particles,
        "proposal_seed_bases": tf.constant(
            [int(seed) % 2147483647 for seed in args.batch_seeds],
            dtype=tf.int32,
        ),
        "fixed_resampling_mask": _make_fixed_resampling_mask(
            batch_size,
            args.time_steps,
            args.transport_policy,
        ),
        "gamma": gamma,
        "beta": beta,
        "sigma": sigma,
        "stationary_variance": stationary_variance,
        "process_variance": tf.square(sigma),
        "flow_observation_covariance": tf.reshape(
            tf.constant(args.flow_observation_variance, dtype=DTYPE),
            [1, 1, 1],
        ),
        "identity_transition_matrix": tf.ones([batch_size, 1, 1], dtype=DTYPE),
        "theta": tf.constant(TRUTH_THETA, dtype=DTYPE),
    }
    semantics = {
        "row_id": ACTUAL_SV_ROW_ID,
        "dataset_seed": DATASET_SEED,
        "truth_theta": list(TRUTH_THETA),
        "truth_theta_coordinate": "synthetic_unconstrained",
        "truth_physical": {
            "gamma": float(gamma64.numpy()),
            "beta": float(beta64.numpy()),
            "sigma": float(sigma64.numpy()),
        },
        "target_transform": "exact_log_y_square",
        "transform_offset": 0.0,
        "target_observation_policy": "transformed_actual_sv_log_y_square",
        "flow_observation_policy": "gaussianized_exact_log_square_actual_sv_flow_observation",
        "target_density_used_for_correction": True,
        "flow_is_proposal_surface_only": True,
        "time_zero_density": "initial_stationary_sv_prior",
        "positive_time_density": "sv_transition_density",
        "target_observation_density": "exact_log_chi_square_log_density",
        "raw_observation_zero_count": int(
            tf.reduce_sum(tf.cast(tf.equal(raw_observations64, 0.0), tf.int32)).numpy()
        ),
        "legacy_raw_gaussian_callback_used": False,
        "ksc_mixture_used": False,
        "augmented_noise_gaussian_closure_used": False,
    }
    return tensors, semantics


def _validate_device(outputs: tuple[tf.Tensor, ...], expect_device_kind: str) -> list[str]:
    devices = [tensor.device for tensor in outputs]
    if expect_device_kind == "gpu":
        if not all("GPU" in device.upper() for device in devices):
            raise RuntimeError(f"expected GPU outputs, got {devices}")
    elif expect_device_kind == "cpu":
        if not all("CPU" in device.upper() for device in devices):
            raise RuntimeError(f"expected CPU outputs, got {devices}")
    return devices


def _materialize(*tensors: tf.Tensor) -> None:
    for tensor in tensors:
        tensor.numpy()


def _weighted_mean_and_variance(particles: tf.Tensor, weights: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    mean = tf.reduce_sum(weights[:, :, None] * particles, axis=1)
    centered = particles - mean[:, None, :]
    variance = tf.reduce_sum(weights[:, :, None] * tf.square(centered), axis=1)
    return mean, variance


def _normalize_log_weights(log_weights: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    normalizer = tf.reduce_logsumexp(log_weights, axis=1)
    weights = tf.exp(log_weights - normalizer[:, None])
    return weights, normalizer


def _uniform_log_weights(batch_size: int, num_particles: int) -> tf.Tensor:
    return tf.fill(
        [batch_size, num_particles],
        -tf.math.log(tf.constant(float(num_particles), dtype=DTYPE)),
    )


def _log_weight_floor() -> tf.Tensor:
    if DTYPE == tf.float32:
        return tf.constant(1.0e-30, dtype=DTYPE)
    return tf.constant(1.0e-300, dtype=DTYPE)


def _actual_sv_value_core(
    *,
    observations: tf.Tensor,
    initial_particles: tf.Tensor,
    proposal_seed_bases: tf.Tensor,
    fixed_resampling_mask: tf.Tensor,
    gamma: tf.Tensor,
    beta: tf.Tensor,
    sigma: tf.Tensor,
    stationary_variance: tf.Tensor,
    flow_observation_covariance: tf.Tensor,
    sinkhorn_epsilon: float | tf.Tensor,
    annealed_scaling: float | tf.Tensor,
    annealed_convergence_threshold: float | tf.Tensor,
    sinkhorn_iterations: int | tf.Tensor,
    row_chunk_size: int,
    col_chunk_size: int,
    particle_chunk_size: int,
    return_history: bool,
) -> streaming_tf.StreamingLEDHPFPFOTValueTensors:
    particles = tf.convert_to_tensor(initial_particles, dtype=DTYPE)
    proposal_seed_bases = tf.convert_to_tensor(proposal_seed_bases, dtype=tf.int32)
    observations = tf.convert_to_tensor(observations, dtype=DTYPE)
    fixed_resampling_mask = tf.convert_to_tensor(fixed_resampling_mask, dtype=tf.bool)
    batch_size = int(particles.shape[0])
    num_particles = int(particles.shape[1])
    state_dim = int(particles.shape[2])
    time_steps = int(observations.shape[0])
    transition_matrix = tf.ones([batch_size, 1, 1], dtype=DTYPE)
    log_weights = _uniform_log_weights(batch_size, num_particles)
    log_likelihood = tf.zeros([batch_size], dtype=DTYPE)
    max_row_residual = tf.constant(0.0, dtype=DTYPE)
    max_column_residual = tf.constant(0.0, dtype=DTYPE)
    if return_history:
        means_ta = tf.TensorArray(
            dtype=DTYPE,
            size=time_steps,
            element_shape=tf.TensorShape([batch_size, state_dim]),
        )
        variances_ta = tf.TensorArray(
            dtype=DTYPE,
            size=time_steps,
            element_shape=tf.TensorShape([batch_size, state_dim]),
        )
        ess_ta = tf.TensorArray(
            dtype=DTYPE,
            size=time_steps,
            element_shape=tf.TensorShape([batch_size]),
        )

    def body(
        time_index: tf.Tensor,
        current_particles: tf.Tensor,
        current_log_weights: tf.Tensor,
        current_log_likelihood: tf.Tensor,
        current_max_row: tf.Tensor,
        current_max_column: tf.Tensor,
        means: tf.TensorArray | tf.Tensor,
        variances: tf.TensorArray | tf.Tensor,
        ess_values: tf.TensorArray | tf.Tensor,
    ):
        observation = observations[time_index]
        is_initial = tf.equal(time_index, 0)
        proposal_mean = tf.cond(
            is_initial,
            lambda: tf.zeros_like(current_particles),
            lambda: gamma * current_particles,
        )
        proposal_variance = tf.cond(
            is_initial,
            lambda: stationary_variance,
            lambda: tf.square(sigma),
        )
        proposal_covariance = tf.ones([batch_size, 1, 1], dtype=DTYPE) * proposal_variance
        salt = tf.math.floormod(
            tf.constant(7000, dtype=tf.int32) + time_index,
            tf.constant(2147483647, dtype=tf.int32),
        )

        def proposal_noise_for_seed(seed_base: tf.Tensor) -> tf.Tensor:
            return tf.random.stateless_normal(
                [num_particles, state_dim],
                seed=tf.stack([seed_base, salt]),
                dtype=DTYPE,
            )

        noise = tf.map_fn(
            proposal_noise_for_seed,
            proposal_seed_bases,
            fn_output_signature=tf.TensorSpec([num_particles, state_dim], dtype=DTYPE),
        )
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
            return tf.cond(
                is_initial,
                lambda: tf.zeros_like(ancestors),
                lambda: gamma * ancestors,
            )

        flow = streaming_tf.batched_ledh_flow_streaming_particles_tf(
            pre_flow_particles=pre_flow,
            ancestors=current_particles,
            observation=observation,
            transition_matrix=transition_matrix,
            transition_covariance=proposal_covariance,
            observation_covariance=tf.tile(flow_observation_covariance, [batch_size, 1, 1]),
            observation_fn=observation_fn,
            observation_jacobian_fn=observation_jacobian_fn,
            observation_residual_fn=observation_residual_fn,
            prior_mean_fn=chunk_prior_mean_fn,
            particle_chunk_size=particle_chunk_size,
        )
        post_flow = flow.post_flow_particles
        target_state = post_flow[:, :, 0]
        initial_log = _log_normal_logpdf(target_state, tf.constant(0.0, dtype=DTYPE), stationary_variance)
        transition_log = _log_normal_logpdf(
            target_state,
            gamma * current_particles[:, :, 0],
            tf.square(sigma),
        )
        target_transition = tf.cond(is_initial, lambda: initial_log, lambda: transition_log)
        residual = observation[0] - tf.constant(2.0, dtype=DTYPE) * tf.math.log(beta) - target_state
        target_observation = tf.cast(
            exact_log_chi_square_log_density(tf.cast(residual, tf.float64)),
            DTYPE,
        )
        corrected_log_weights = (
            current_log_weights
            + target_transition
            + target_observation
            - flow.pre_flow_log_density
            + flow.forward_log_det
        )
        weights, incremental = _normalize_log_weights(corrected_log_weights)
        next_log_likelihood = current_log_likelihood + incremental
        ess = 1.0 / tf.reduce_sum(weights * weights, axis=1)
        mean, variance = _weighted_mean_and_variance(post_flow, weights)
        normalized_log_weights = tf.math.log(tf.maximum(weights, _log_weight_floor()))
        mask = fixed_resampling_mask[:, time_index]

        def do_transport() -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
            transported = core_tf.batched_annealed_transport_core_tf(
                post_flow,
                normalized_log_weights,
                mask,
                epsilon=sinkhorn_epsilon,
                scaling=annealed_scaling,
                convergence_threshold=annealed_convergence_threshold,
                max_iterations=sinkhorn_iterations,
                transport_gradient_mode="raw",
                transport_plan_mode="streaming",
                transport_ad_mode="stabilized",
                row_chunk_size=row_chunk_size,
                col_chunk_size=col_chunk_size,
            )
            return (
                transported.particles,
                transported.log_weights,
                transported.max_row_residual,
                transported.max_column_residual,
            )

        def skip_transport() -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
            zero = tf.constant(0.0, dtype=DTYPE)
            return post_flow, normalized_log_weights, zero, zero

        next_particles, next_log_weights, row_residual, column_residual = tf.cond(
            tf.reduce_any(mask),
            do_transport,
            skip_transport,
        )
        next_max_row = tf.maximum(current_max_row, row_residual)
        next_max_column = tf.maximum(current_max_column, column_residual)
        if return_history:
            means = means.write(time_index, mean)
            variances = variances.write(time_index, variance)
            ess_values = ess_values.write(time_index, ess)
        return (
            time_index + 1,
            next_particles,
            next_log_weights,
            next_log_likelihood,
            next_max_row,
            next_max_column,
            means,
            variances,
            ess_values,
        )

    if return_history:
        initial_means = means_ta
        initial_variances = variances_ta
        initial_ess = ess_ta
    else:
        initial_means = tf.constant(0.0, dtype=DTYPE)
        initial_variances = tf.constant(0.0, dtype=DTYPE)
        initial_ess = tf.constant(0.0, dtype=DTYPE)

    (
        _,
        final_particles,
        _final_log_weights,
        final_log_likelihood,
        final_max_row,
        final_max_column,
        final_means,
        final_variances,
        final_ess,
    ) = tf.while_loop(
        lambda time_index, *_args: time_index < time_steps,
        body,
        loop_vars=(
            tf.constant(0, dtype=tf.int32),
            particles,
            log_weights,
            log_likelihood,
            max_row_residual,
            max_column_residual,
            initial_means,
            initial_variances,
            initial_ess,
        ),
        parallel_iterations=1,
        maximum_iterations=time_steps,
    )
    if return_history:
        filtered_means = final_means.stack()
        filtered_variances = final_variances.stack()
        ess_by_time = final_ess.stack()
    else:
        filtered_means = tf.zeros([time_steps, batch_size, state_dim], dtype=DTYPE)
        filtered_variances = tf.zeros([time_steps, batch_size, state_dim], dtype=DTYPE)
        ess_by_time = tf.zeros([time_steps, batch_size], dtype=DTYPE)
    return streaming_tf.StreamingLEDHPFPFOTValueTensors(
        log_likelihood=final_log_likelihood,
        filtered_means=filtered_means,
        filtered_variances=filtered_variances,
        ess_by_time=ess_by_time,
        final_particles=final_particles,
        max_row_residual=final_max_row,
        max_column_residual=final_max_column,
    )


def _write_markdown(path: Path, result: dict[str, Any], json_path: Path) -> None:
    lines = [
        "# Phase 5 Actual-SV Tiny Adapter Smoke Artifact",
        "",
        f"- JSON artifact: `{json_path}`",
        f"- Row id: `{result['row_id']}`",
        f"- Admission status: `{result['admission_status']}`",
        f"- Target scalar: `{result['target_scalar']}`",
        f"- Target observation policy: `{result['target_observation_policy']}`",
        f"- Flow observation policy: `{result['flow_observation_policy']}`",
        f"- Target density used for correction: `{result['target_density_used_for_correction']}`",
        f"- Batch seeds: `{result['batch_seeds']}`",
        f"- Num particles: `{result['num_particles']}`",
        f"- Time steps: `{result['time_steps']}`",
        f"- Log likelihood by seed: `{result['log_likelihood_by_seed']}`",
        f"- Output devices: `{result['output_devices']}`",
        f"- Finite output: `{result['finite_output']}`",
        "",
        "## Nonclaims",
        "",
    ]
    lines.extend(f"- {claim}" for claim in result["nonclaims"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    precision = _configure_precision(args)
    physical_gpus, logical_gpus = _configure_gpus()
    tensors, actual_sv_semantics = _build_actual_sv_tensors(args)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def compiled_outputs() -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        value = _actual_sv_value_core(
            observations=tensors["observations"],
            initial_particles=tensors["initial_particles"],
            proposal_seed_bases=tensors["proposal_seed_bases"],
            fixed_resampling_mask=tensors["fixed_resampling_mask"],
            gamma=tensors["gamma"],
            beta=tensors["beta"],
            sigma=tensors["sigma"],
            stationary_variance=tensors["stationary_variance"],
            flow_observation_covariance=tensors["flow_observation_covariance"],
            sinkhorn_epsilon=args.sinkhorn_epsilon,
            annealed_scaling=args.annealed_scaling,
            annealed_convergence_threshold=args.annealed_convergence_threshold,
            sinkhorn_iterations=args.sinkhorn_iterations,
            row_chunk_size=args.row_chunk_size,
            col_chunk_size=args.col_chunk_size,
            particle_chunk_size=args.particle_chunk_size,
            return_history=args.history_mode == "full",
        )
        return (
            value.log_likelihood,
            value.filtered_means,
            value.filtered_variances,
            value.ess_by_time,
            value.final_particles,
        )

    with tf.device(args.device):
        memory_before = _gpu_memory_info()
        start = time.perf_counter()
        outputs = compiled_outputs()
        _materialize(*outputs)
        compile_and_first = time.perf_counter() - start
        for _ in range(args.warmups):
            _materialize(*compiled_outputs())
        timings: list[float] = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            outputs = compiled_outputs()
            _materialize(*outputs)
            timings.append(time.perf_counter() - start)
        memory_after = _gpu_memory_info()

    log_likelihood, filtered_means, filtered_variances, ess_by_time, final_particles = outputs
    output_devices = _validate_device((log_likelihood,), args.expect_device_kind)
    history_returned = args.history_mode == "full"
    finite_output = bool(tf.reduce_all(tf.math.is_finite(log_likelihood)).numpy())
    if history_returned:
        finite_output = bool(
            finite_output
            and tf.reduce_all(tf.math.is_finite(filtered_means)).numpy()
            and tf.reduce_all(tf.math.is_finite(filtered_variances)).numpy()
            and tf.reduce_all(tf.math.is_finite(ess_by_time)).numpy()
            and tf.reduce_all(tf.math.is_finite(final_particles)).numpy()
        )
    full_row_scope = args.run_scope == "full-row-admission"
    forward_contract = make_actual_sv_forward_contract(
        time_steps=args.time_steps,
        num_particles=args.num_particles,
        batch_seeds=args.batch_seeds,
        full_leaderboard_row=full_row_scope,
    ).to_manifest()
    log_likelihood_values = [float(value) for value in log_likelihood.numpy().reshape(-1)]
    average_values = [value / float(args.time_steps) for value in log_likelihood_values]
    artifact: dict[str, Any] = {
        "schema_version": LEDH_FORWARD_SCALAR_ARTIFACT_SCHEMA_VERSION,
        "timestamp_utc": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "host": platform.node(),
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "git_commit": _git_commit(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_gpus": physical_gpus,
        "logical_gpus": logical_gpus,
        "device": args.device,
        "device_scope": args.device_scope,
        "expect_device_kind": args.expect_device_kind,
        "precision": precision,
        "run_scope": args.run_scope,
        "row_id": ACTUAL_SV_ROW_ID,
        "shape": {
            "batch_size": len(args.batch_seeds),
            "batch_seed_count": len(args.batch_seeds),
            "time_steps": args.time_steps,
            "num_particles": args.num_particles,
            "state_dim": STATE_DIM,
            "obs_dim": OBS_DIM,
        },
        "batch_seeds": [int(seed) for seed in args.batch_seeds],
        "num_particles": args.num_particles,
        "time_steps": args.time_steps,
        "full_batched_evaluation_per_warm_call": True,
        "history_mode": args.history_mode,
        "return_history": history_returned,
        "forward_contract": forward_contract,
        "target_scalar": forward_contract["target_scalar"],
        "target_output_tensor_field": forward_contract["output_tensor_field"],
        "target_density_fields": forward_contract["target_density_fields"],
        "proposal_flow_fields": forward_contract["proposal_flow_fields"],
        "correction_formula": forward_contract["correction_formula"],
        "theta_values": list(TRUTH_THETA),
        "theta_coordinate_system": "synthetic_unconstrained",
        "flow_observation_policy": "gaussianized_exact_log_square_actual_sv_flow_observation",
        "target_observation_policy": "transformed_actual_sv_log_y_square",
        "target_density_used_for_correction": True,
        "actual_sv_semantics": actual_sv_semantics,
        "transport_policy": args.transport_policy,
        "transport": {
            "plan_mode": "streaming",
            "gradient_mode": "raw",
            "row_chunk_size": args.row_chunk_size,
            "col_chunk_size": args.col_chunk_size,
            "particle_chunk_size": args.particle_chunk_size,
            "sinkhorn_iterations": args.sinkhorn_iterations,
            "sinkhorn_epsilon": args.sinkhorn_epsilon,
            "annealed_scaling": args.annealed_scaling,
            "annealed_convergence_threshold": args.annealed_convergence_threshold,
            "dense_transport_matrix_materialized": False,
        },
        "flow_observation_variance": args.flow_observation_variance,
        "compile_and_first_call_seconds": compile_and_first,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "warm_call_timings_seconds": timings,
        "warm_call_timing_summary_seconds": _summary(timings),
        "gpu_memory_info_before": memory_before,
        "gpu_memory_info_after": memory_after,
        "output_devices": output_devices,
        "output_shape": list(log_likelihood.numpy().shape),
        "history_shapes": {
            "filtered_means": list(filtered_means.numpy().shape),
            "filtered_variances": list(filtered_variances.numpy().shape),
            "ess_by_time": list(ess_by_time.numpy().shape),
            "final_particles": list(final_particles.numpy().shape),
        },
        "log_likelihood_by_seed": log_likelihood_values,
        "average_log_likelihood_by_seed": average_values,
        "finite_output": finite_output,
        "admission_status": LEDH_FORWARD_ADMISSION_STATUS_TINY,
        "normalization_checks": {
            "row_id": True,
            "seed_count_matches_values": len(log_likelihood_values) == len(args.batch_seeds),
            "time_steps_tiny": args.time_steps < FULL_ROW_TIME_STEPS,
            "num_particles_tiny": args.num_particles < FULL_ROW_NUM_PARTICLES,
            "not_full_row_requested": not _full_row_requested(args),
            "exact_full_row_requested": _exact_full_row_requested(args),
            "full_row_admission_scope": full_row_scope,
            "finite_log_likelihood": finite_output,
            "target_density_used_for_correction": True,
            "exact_transform_offset_zero": actual_sv_semantics["transform_offset"] == 0.0,
            "legacy_raw_gaussian_callback_used": False,
            "ksc_mixture_used": False,
            "augmented_noise_gaussian_closure_used": False,
        },
        "nonclaims": list(FULL_ROW_NONCLAIMS if full_row_scope else TINY_NONCLAIMS),
    }
    if full_row_scope:
        artifact["admission_status"] = (
            LEDH_FORWARD_ADMISSION_STATUS_ADMITTED
            if finite_output
            else LEDH_FORWARD_ADMISSION_STATUS_BLOCKED_NONFINITE
        )
    normalized_core = validate_ledh_forward_scalar_artifact(
        artifact,
        expected_row_id=ACTUAL_SV_ROW_ID,
        require_admitted=full_row_scope,
    )
    artifact["validator_normalized_core"] = normalized_core

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_output is not None:
        _write_markdown(Path(args.markdown_output), artifact, output_path)
    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
