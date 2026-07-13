"""Source-route generalized-SV LEDH-PFPF-OT value runner.

This runner targets the generalized-SV row
``zhao_cui_generalized_sv_synthetic_from_estimated_values``.  By default it emits a tiny
``tiny_executed_not_full_row`` adapter-smoke artifact.  Full row admission
requires the explicit ``--run-scope full-row-admission`` guard and exact
``N=10000,T=1008`` settings.

The target correction is the source-route prior-mean generalized-SV raw
observation likelihood:

``y_t | x_t ~ Normal(0, exp(tau * x_t))``.

The log-square Gaussianized LEDH observation surface is a proposal surface
only. It is not the target likelihood.
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
    GENERALIZED_SV_ROW_ID,
    LEDH_FORWARD_ADMISSION_STATUS_ADMITTED,
    LEDH_FORWARD_ADMISSION_STATUS_BLOCKED_NONFINITE,
    LEDH_FORWARD_ADMISSION_STATUS_TINY,
    LEDH_FORWARD_SCALAR_ARTIFACT_SCHEMA_VERSION,
    make_generalized_sv_forward_contract,
    validate_ledh_forward_scalar_artifact,
)
from experiments.dpf_implementation.tf_tfp.filters import (
    experimental_batched_ledh_pfpf_ot_streaming_tf as streaming_tf,
)
from experiments.dpf_implementation.tf_tfp.filters import (
    experimental_batched_ledh_pfpf_ot_tf as core_tf,
)
from experiments.dpf_implementation.tf_tfp.resampling import annealed_transport_tf
from scripts.filtering_value_gradient_benchmark_generate_p8_datasets import (
    _generalized_sv_prior_mean_dataset,
)


DATASET_SEED = 81105
TRUTH_THETA = [1.0824113944610982, -2.076793740349318, 0.0]
FULL_ROW_BATCH_SEEDS = (81120, 81121, 81122, 81123, 81124)
FULL_ROW_NUM_PARTICLES = 10000
FULL_ROW_TIME_STEPS = 1008
STATE_DIM = 1
OBS_DIM = 1
DTYPE = tf.float32
LOG_SQUARE_FLOW_OFFSET = 1.0e-6
DEFAULT_FLOW_OBSERVATION_VARIANCE = 2.0
BASE_NONCLAIMS = (
    "not score admission",
    "not score correctness",
    "not actual-SV admission",
    "not KSC admission",
    "not KSC surrogate likelihood evidence",
    "not native generalized-SV dense fixture evidence",
    "not SP500 benchmark-observation evidence",
    "not author-default truth evidence",
    "not HMC readiness evidence",
    "not posterior correctness evidence",
    "not scientific superiority evidence",
    "not runtime ranking evidence",
)
TINY_NONCLAIMS = (
    "not full generalized-SV row admission",
    *BASE_NONCLAIMS,
)
FULL_ROW_NONCLAIMS = (
    "not score admission",
    "not score correctness",
    "not actual-SV admission",
    "not KSC admission",
    "not KSC surrogate likelihood evidence",
    "not native generalized-SV dense fixture evidence",
    "not SP500 benchmark-observation evidence",
    "not author-default truth evidence",
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
    parser.add_argument("--flow-observation-variance", type=float, default=DEFAULT_FLOW_OBSERVATION_VARIANCE)
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
        raise ValueError("time_steps must be in 1..1008")
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
        raise ValueError("adapter smoke runner must not run the full generalized-SV row")
    if args.run_scope == "full-row-admission" and not _exact_full_row_requested(args):
        raise ValueError(
            "full-row-admission requires exact generalized-SV full row settings: "
            "T=1008, N=10000, seeds=81120,81121,81122,81123,81124"
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


def _normal01_cdf(value: tf.Tensor) -> tf.Tensor:
    return 0.5 * (
        tf.constant(1.0, dtype=value.dtype)
        + tf.math.erf(value / tf.sqrt(tf.constant(2.0, dtype=value.dtype)))
    )


def _log_square_flow_observations(raw_observations: tf.Tensor) -> tf.Tensor:
    raw = tf.convert_to_tensor(raw_observations, dtype=tf.float64)
    return tf.math.log(tf.square(raw) + tf.constant(LOG_SQUARE_FLOW_OFFSET, dtype=tf.float64))


def _raw_zero_mean_normal_log_density_from_log_scale(
    observation: tf.Tensor,
    log_scale: tf.Tensor,
) -> tf.Tensor:
    y = tf.reshape(tf.convert_to_tensor(observation, dtype=DTYPE), [1])[0]
    log_scale = tf.convert_to_tensor(log_scale, dtype=DTYPE)
    standardized = y * tf.exp(-log_scale)
    return -0.5 * tf.math.log(tf.constant(2.0 * math.pi, dtype=DTYPE)) - log_scale - 0.5 * tf.square(standardized)


def _build_generalized_sv_tensors(args: argparse.Namespace) -> tuple[dict[str, tf.Tensor], dict[str, Any]]:
    theta64 = tf.constant(TRUTH_THETA, dtype=tf.float64)
    gamma64 = _normal01_cdf(theta64[0])
    tau64 = tf.exp(theta64[1])
    mu64 = theta64[2]
    dataset = _generalized_sv_prior_mean_dataset(DATASET_SEED)
    raw_observations64 = tf.convert_to_tensor(dataset["observations"], dtype=tf.float64)[: args.time_steps]
    flow_observations64 = _log_square_flow_observations(raw_observations64)
    gamma = tf.cast(gamma64, DTYPE)
    tau = tf.cast(tau64, DTYPE)
    mu = tf.cast(mu64, DTYPE)
    stationary_variance = tf.constant(1.0, dtype=DTYPE) / (tf.constant(1.0, dtype=DTYPE) - tf.square(gamma))
    stationary_scale = tf.sqrt(stationary_variance)
    batch_size = len(args.batch_seeds)
    initial_particles = tf.stack(
        [
            mu
            + stationary_scale
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
        "observations": tf.cast(flow_observations64, DTYPE),
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
        "tau": tau,
        "mu": mu,
        "stationary_variance": stationary_variance,
        "process_variance": tf.constant(1.0, dtype=DTYPE),
        "flow_observation_covariance": tf.reshape(
            tf.constant(args.flow_observation_variance, dtype=DTYPE),
            [1, 1, 1],
        ),
        "identity_transition_matrix": tf.ones([batch_size, 1, 1], dtype=DTYPE),
        "theta": tf.constant(TRUTH_THETA, dtype=DTYPE),
    }
    semantics = {
        "row_id": GENERALIZED_SV_ROW_ID,
        "dataset_seed": DATASET_SEED,
        "truth_theta": list(TRUTH_THETA),
        "truth_theta_coordinate": "source_route_active_transformed_prior_mean",
        "truth_physical": {
            "gamma": float(gamma64.numpy()),
            "tau_or_sigma": float(tau64.numpy()),
            "mu_or_log_beta_center_coordinate": float(mu64.numpy()),
            "phi": 0.0,
            "a": 0.0,
            "delta": 0.0,
            "nu1": "inf",
            "nu2": "inf",
        },
        "source_route": "zhao_cui_svmodels_prior_mean_synthetic",
        "target_observation_policy": "source_route_prior_mean_generalized_sv",
        "flow_observation_policy": "log_square_gaussian_surrogate_for_ledh_flow_only",
        "flow_observation_transform": "log(y_t^2 + 1e-6)",
        "flow_observation_offset": LOG_SQUARE_FLOW_OFFSET,
        "flow_observation_variance": float(args.flow_observation_variance),
        "target_density_used_for_correction": True,
        "flow_is_proposal_surface_only": True,
        "time_zero_density": "transition_from_stationary_previous_state",
        "positive_time_density": "source_route_ar1_transition_density",
        "target_observation_density": "raw_zero_mean_generalized_sv_prior_mean_normal_log_density",
        "state_timing_convention": (
            "initial particles are stationary previous states; every recorded observation, "
            "including t=0, first applies the AR(1) transition before target weighting"
        ),
        "raw_observation_zero_count": int(
            tf.reduce_sum(tf.cast(tf.equal(raw_observations64, 0.0), tf.int32)).numpy()
        ),
        "actual_sv_evidence_used": False,
        "native_generalized_sv_dense_fixture_used": False,
        "sp500_returns_used_as_benchmark_observations": False,
        "author_defaults_used_as_truth": False,
        "ksc_mixture_used": False,
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


def _generalized_sv_value_core(
    *,
    observations: tf.Tensor,
    raw_observations: tf.Tensor,
    initial_particles: tf.Tensor,
    proposal_seed_bases: tf.Tensor,
    fixed_resampling_mask: tf.Tensor,
    gamma: tf.Tensor,
    tau: tf.Tensor,
    mu: tf.Tensor,
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
    initial_particles = tf.convert_to_tensor(initial_particles, dtype=DTYPE)
    observations = tf.convert_to_tensor(observations, dtype=DTYPE)
    raw_observations = tf.convert_to_tensor(raw_observations, dtype=DTYPE)
    fixed_resampling_mask = tf.convert_to_tensor(fixed_resampling_mask, dtype=tf.bool)
    proposal_seed_bases = tf.convert_to_tensor(proposal_seed_bases, dtype=tf.int32)
    batch_size = int(initial_particles.shape[0])
    num_particles = int(initial_particles.shape[1])
    state_dim = int(initial_particles.shape[2])
    transition_matrix = tf.ones([batch_size, 1, 1], dtype=DTYPE)
    process_variance = tf.constant(1.0, dtype=DTYPE)
    transition_covariance = tf.ones([batch_size, 1, 1], dtype=DTYPE) * process_variance
    observation_covariance = tf.tile(flow_observation_covariance, [batch_size, 1, 1])

    def pre_flow_step_fn(current_particles: tf.Tensor, time_index: tf.Tensor) -> tf.Tensor:
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
        return mu + gamma * (current_particles - mu) + noise

    def observation_fn(points: tf.Tensor) -> tf.Tensor:
        return tau * points

    def observation_jacobian_fn(points: tf.Tensor) -> tf.Tensor:
        chunk_particles = points.shape[1]
        if chunk_particles is None:
            raise ValueError("generalized-SV adapter requires static particle chunk dimension")
        return tf.ones([batch_size, int(chunk_particles), 1, 1], dtype=DTYPE) * tau

    def observation_residual_fn(h_ref: tf.Tensor, obs: tf.Tensor) -> tf.Tensor:
        return obs[tf.newaxis, tf.newaxis, :] - h_ref

    def prior_mean_fn(ancestors: tf.Tensor, time_index: tf.Tensor) -> tf.Tensor:
        del time_index
        return mu + gamma * (ancestors - mu)

    def transition_log_density_fn(
        post_flow: tf.Tensor,
        previous_particles: tf.Tensor,
        time_index: tf.Tensor,
    ) -> tf.Tensor:
        del time_index
        target_state = post_flow[:, :, 0]
        previous_state = previous_particles[:, :, 0]
        return _log_normal_logpdf(
            target_state,
            mu + gamma * (previous_state - mu),
            process_variance,
        )

    def observation_log_density_fn(
        post_flow: tf.Tensor,
        observation: tf.Tensor,
        time_index: tf.Tensor,
    ) -> tf.Tensor:
        del observation
        target_state = post_flow[:, :, 0]
        raw_observation = raw_observations[time_index]
        log_scale = tf.constant(0.5, dtype=DTYPE) * tau * target_state
        return _raw_zero_mean_normal_log_density_from_log_scale(
            raw_observation,
            log_scale,
        )

    return streaming_tf.streaming_batched_ledh_pfpf_ot_value_core_tf(
        observations=observations,
        initial_particles=initial_particles,
        fixed_resampling_mask=fixed_resampling_mask,
        transition_matrix=transition_matrix,
        transition_covariance=transition_covariance,
        observation_covariance=observation_covariance,
        observation_fn=observation_fn,
        observation_jacobian_fn=observation_jacobian_fn,
        observation_residual_fn=observation_residual_fn,
        transition_log_density_fn=transition_log_density_fn,
        observation_log_density_fn=observation_log_density_fn,
        prior_mean_fn=prior_mean_fn,
        pre_flow_step_fn=pre_flow_step_fn,
        sinkhorn_epsilon=sinkhorn_epsilon,
        annealed_scaling=annealed_scaling,
        annealed_convergence_threshold=annealed_convergence_threshold,
        sinkhorn_iterations=sinkhorn_iterations,
        transport_gradient_mode="raw",
        transport_plan_mode="streaming",
        transport_ad_mode="stabilized",
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
        particle_chunk_size=particle_chunk_size,
        skip_transport_when_no_active=True,
        return_history=return_history,
    )


def _write_markdown(path: Path, result: dict[str, Any], json_path: Path) -> None:
    lines = [
        "# Phase 6 Generalized-SV Forward Scalar Artifact",
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
    tensors, generalized_sv_semantics = _build_generalized_sv_tensors(args)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def compiled_outputs() -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        value = _generalized_sv_value_core(
            observations=tensors["observations"],
            raw_observations=tensors["raw_observations"],
            initial_particles=tensors["initial_particles"],
            proposal_seed_bases=tensors["proposal_seed_bases"],
            fixed_resampling_mask=tensors["fixed_resampling_mask"],
            gamma=tensors["gamma"],
            tau=tensors["tau"],
            mu=tensors["mu"],
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
    forward_contract = make_generalized_sv_forward_contract(
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
        "row_id": GENERALIZED_SV_ROW_ID,
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
        "theta_coordinate_system": "source_route_active_transformed_prior_mean",
        "flow_observation_policy": "log_square_gaussian_surrogate_for_ledh_flow_only",
        "target_observation_policy": "source_route_prior_mean_generalized_sv",
        "target_density_used_for_correction": True,
        "generalized_sv_semantics": generalized_sv_semantics,
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
            "source_route_target_policy": True,
            "flow_surface_is_proposal_only": generalized_sv_semantics["flow_is_proposal_surface_only"],
            "flow_offset_matches_contract": (
                generalized_sv_semantics["flow_observation_offset"] == LOG_SQUARE_FLOW_OFFSET
            ),
            "actual_sv_evidence_used": False,
            "native_generalized_sv_dense_fixture_used": False,
            "sp500_returns_used_as_benchmark_observations": False,
            "author_defaults_used_as_truth": False,
            "ksc_mixture_used": False,
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
        expected_row_id=GENERALIZED_SV_ROW_ID,
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
