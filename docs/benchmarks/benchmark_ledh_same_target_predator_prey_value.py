"""Same-target LEDH-PFPF-OT value runner for the predator-prey T20 row.

This runner targets ``zhao_cui_predator_prey_T20`` with the additive-Gaussian
RK4 predator-prey target, dataset seed ``81104``, and physical theta
``(r, K, a, s, u, v) = (0.6, 114, 25, 0.3, 0.5, 0.5)``.

It emits only the observed-data forward likelihood scalar.  It does not emit or
admit scores.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
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

from bayesfilter import highdim
from bayesfilter.highdim.ledh_forward_contract import (
    LEDH_FORWARD_ADMISSION_STATUS_ADMITTED,
    LEDH_FORWARD_ADMISSION_STATUS_HISTORICAL_RAW,
    LEDH_FORWARD_ADMISSION_STATUS_TINY,
    LEDH_FORWARD_SCALAR_ARTIFACT_SCHEMA_VERSION,
    PREDATOR_PREY_ROW_ID,
    make_predator_prey_forward_contract,
    validate_ledh_forward_scalar_artifact,
)
from bayesfilter.highdim.ledh_historical_raw_policy import (
    require_historical_raw_diagnostic_opt_in,
)
from experiments.dpf_implementation.tf_tfp.filters import (
    experimental_batched_ledh_pfpf_ot_streaming_tf as streaming_tf,
)
from experiments.dpf_implementation.tf_tfp.filters import (
    experimental_batched_ledh_pfpf_ot_tf as core_tf,
)
from experiments.dpf_implementation.tf_tfp.resampling import annealed_transport_tf
from scripts.filtering_value_gradient_benchmark_generate_p8_datasets import (
    _predator_prey_dataset,
)


DATASET_SEED = 81104
TRUTH_THETA = [0.6, 114.0, 25.0, 0.3, 0.5, 0.5]
FULL_ROW_BATCH_SEEDS = (81120, 81121, 81122, 81123, 81124)
FULL_ROW_NUM_PARTICLES = 10000
FULL_ROW_TIME_STEPS = 20
FULL_ROW_TRANSPORT_POLICY = "active-all"
FULL_ROW_SINKHORN_ITERATIONS = 10
FULL_ROW_SINKHORN_EPSILON = 1.0
STATE_DIM = 2
OBS_DIM = 2
DTYPE = tf.float32
NONCLAIMS = (
    "not score admission",
    "not score correctness",
    "not exact nonlinear likelihood correctness evidence",
    "not Zhao-Cui TT/SIRT source-faithfulness evidence",
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
    parser.add_argument("--batch-seeds", default=",".join(str(seed) for seed in FULL_ROW_BATCH_SEEDS))
    parser.add_argument("--time-steps", type=int, default=FULL_ROW_TIME_STEPS)
    parser.add_argument("--num-particles", type=int, default=FULL_ROW_NUM_PARTICLES)
    parser.add_argument(
        "--transport-policy",
        choices=("active-all", "active-odd", "no-resampling"),
        default=FULL_ROW_TRANSPORT_POLICY,
    )
    parser.add_argument("--sinkhorn-iterations", type=int, default=FULL_ROW_SINKHORN_ITERATIONS)
    parser.add_argument("--sinkhorn-epsilon", type=float, default=FULL_ROW_SINKHORN_EPSILON)
    parser.add_argument("--annealed-scaling", type=float, default=0.9)
    parser.add_argument("--annealed-convergence-threshold", type=float, default=1.0e-3)
    parser.add_argument("--row-chunk-size", type=int, default=512)
    parser.add_argument("--col-chunk-size", type=int, default=512)
    parser.add_argument("--particle-chunk-size", type=int, default=512)
    parser.add_argument(
        "--history-mode",
        choices=("full", "value-only"),
        default="value-only",
    )
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float32")
    parser.add_argument(
        "--tf32-mode",
        choices=("default", "enabled", "disabled"),
        default="enabled",
    )
    parser.add_argument("--device", default="/GPU:0")
    parser.add_argument("--device-scope", choices=("cpu", "visible"), default=_PRE_ARGS.device_scope)
    parser.add_argument("--cuda-visible-devices", default=_PRE_ARGS.cuda_visible_devices)
    parser.add_argument("--expect-device-kind", choices=("any", "cpu", "gpu"), default="gpu")
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output", default=None)
    parser.add_argument("--historical-raw-diagnostic", action="store_true")
    args = parser.parse_args()
    args.batch_seeds = _parse_int_csv(args.batch_seeds)
    if args.time_steps <= 0 or args.time_steps > FULL_ROW_TIME_STEPS:
        raise ValueError("time_steps must be in 1..20")
    if args.num_particles <= 1:
        raise ValueError("num_particles must be greater than one")
    if args.sinkhorn_iterations <= 0:
        raise ValueError("sinkhorn_iterations must be positive")
    if args.warmups < 0 or args.repeats <= 0:
        raise ValueError("warmups must be nonnegative and repeats must be positive")
    if args.row_chunk_size <= 0 or args.col_chunk_size <= 0 or args.particle_chunk_size <= 0:
        raise ValueError("chunk sizes must be positive")
    return args


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


def _batched_gaussian_logpdf(residuals: tf.Tensor, covariance: tf.Tensor) -> tf.Tensor:
    residuals = tf.convert_to_tensor(residuals, dtype=DTYPE)
    covariance = tf.convert_to_tensor(covariance, dtype=DTYPE)
    chol = tf.linalg.cholesky(covariance)
    solved = tf.linalg.matrix_transpose(
        tf.linalg.cholesky_solve(chol, tf.linalg.matrix_transpose(residuals))
    )
    quad = tf.reduce_sum(solved * residuals, axis=-1)
    logdet = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)), axis=-1)
    dim = tf.cast(residuals.shape[-1], DTYPE)
    return -0.5 * (
        dim * tf.math.log(tf.constant(2.0 * 3.141592653589793, dtype=DTYPE))
        + logdet[:, None]
        + quad
    )


def _build_predator_prey_tensors(args: argparse.Namespace) -> tuple[dict[str, tf.Tensor], dict[str, Any]]:
    model = highdim.p30_predator_prey_fixture_model()
    dataset = _predator_prey_dataset(DATASET_SEED)
    observations = tf.cast(tf.convert_to_tensor(dataset["observations"], dtype=tf.float64)[: args.time_steps], DTYPE)
    batch_size = len(args.batch_seeds)
    initial_chol = tf.linalg.cholesky(tf.cast(model.initial_covariance, DTYPE))
    initial_mean = tf.cast(model.initial_mean, DTYPE)
    initial_particles = tf.stack(
        [
            initial_mean
            + tf.random.stateless_normal(
                [args.num_particles, STATE_DIM],
                seed=_seed_pair(int(seed), 120),
                dtype=DTYPE,
            )
            @ tf.transpose(initial_chol)
            for seed in args.batch_seeds
        ],
        axis=0,
    )
    process_covariance = tf.tile(
        tf.cast(model.process_covariance, DTYPE)[tf.newaxis, :, :],
        [batch_size, 1, 1],
    )
    observation_covariance = tf.tile(
        tf.cast(model.observation_covariance, DTYPE)[tf.newaxis, :, :],
        [batch_size, 1, 1],
    )
    tensors = {
        "observations": observations,
        "initial_particles": initial_particles,
        "fixed_resampling_mask": _make_fixed_resampling_mask(
            batch_size,
            args.time_steps,
            args.transport_policy,
        ),
        "transition_matrix": tf.tile(tf.eye(STATE_DIM, dtype=DTYPE)[tf.newaxis, :, :], [batch_size, 1, 1]),
        "transition_covariance": process_covariance,
        "observation_covariance": observation_covariance,
    }
    semantics = {
        "row_id": PREDATOR_PREY_ROW_ID,
        "dataset_seed": DATASET_SEED,
        "state_dimension": STATE_DIM,
        "observation_dimension": OBS_DIM,
        "theta_coordinate_system": "physical",
        "truth_theta": list(TRUTH_THETA),
        "target_density_used_for_correction": True,
        "flow_observation_contract": "identity_state_gaussian_flow_observation",
        "target_observation_policy": "additive_gaussian_predator_prey",
        "model_family": "PredatorPreySSM",
        "rk4_delta": float(model.delta.numpy()),
        "rk4_internal_step": float(model.rk4_internal_step.numpy()),
        "rk4_substeps": int(model._rk4_substeps),
        "process_covariance": [[float(item) for item in row] for row in model.process_covariance.numpy().tolist()],
        "observation_covariance": [[float(item) for item in row] for row in model.observation_covariance.numpy().tolist()],
        "initial_covariance": [[float(item) for item in row] for row in model.initial_covariance.numpy().tolist()],
        "experimental_streaming_adapter_route": (
            "batched predator-prey RK4-prior LEDH-PFPF-OT streaming adapter "
            "with target transition and observation densities in the correction"
        ),
        "scalar_algorithm1_ukf_covariance_lifecycle_parity_claim": False,
    }
    return tensors, semantics


def _make_predator_prey_callbacks(tensors: dict[str, tf.Tensor], seeds: list[int], args: argparse.Namespace):
    model = highdim.p30_predator_prey_fixture_model()
    theta = tf.constant(TRUTH_THETA, dtype=DTYPE)
    r, k_capacity, a_half, s_rate, u_rate, v_rate = tf.unstack(theta)
    substeps = int(model._rk4_substeps)
    step_size = tf.cast(model.delta, DTYPE) / tf.cast(substeps, DTYPE)
    process_chol = tf.linalg.cholesky(tf.cast(model.process_covariance, DTYPE))
    identity_jacobian = tf.eye(OBS_DIM, STATE_DIM, dtype=DTYPE)
    batch_size = len(seeds)
    num_particles = args.num_particles

    def rhs(points: tf.Tensor) -> tf.Tensor:
        prey = points[:, :, 0]
        predator = points[:, :, 1]
        denominator = a_half + prey
        interaction = prey * predator / denominator
        d_prey = r * prey * (tf.constant(1.0, dtype=DTYPE) - prey / k_capacity) - s_rate * interaction
        d_predator = u_rate * interaction - v_rate * predator
        return tf.stack([d_prey, d_predator], axis=2)

    def transition_mean(points: tf.Tensor, time_index: tf.Tensor) -> tf.Tensor:
        del time_index
        state = tf.cast(points, DTYPE)
        for _ in range(substeps):
            k1 = rhs(state)
            k2 = rhs(state + tf.constant(0.5, dtype=DTYPE) * step_size * k1)
            k3 = rhs(state + tf.constant(0.5, dtype=DTYPE) * step_size * k2)
            k4 = rhs(state + step_size * k3)
            state = state + (step_size / tf.constant(6.0, dtype=DTYPE)) * (
                k1
                + tf.constant(2.0, dtype=DTYPE) * k2
                + tf.constant(2.0, dtype=DTYPE) * k3
                + k4
            )
        return state

    def pre_flow_step(points: tf.Tensor, time_index: tf.Tensor) -> tf.Tensor:
        mean = transition_mean(points, time_index)
        noise_rows = []
        for seed in seeds:
            noise_rows.append(
                tf.random.stateless_normal(
                    [num_particles, STATE_DIM],
                    seed=_seed_pair(int(seed), tf.constant(1120, dtype=tf.int32) + time_index),
                    dtype=DTYPE,
                )
            )
        noise_tensor = tf.stack(noise_rows, axis=0)
        return mean + tf.einsum("bnd,ed->bne", noise_tensor, process_chol)

    def observation_fn(points: tf.Tensor) -> tf.Tensor:
        return tf.identity(points)

    def observation_jacobian_fn(points: tf.Tensor) -> tf.Tensor:
        chunk_particles = points.shape[1]
        if chunk_particles is None:
            raise ValueError("predator-prey runner requires static particle chunk dimension")
        return tf.tile(
            identity_jacobian[tf.newaxis, tf.newaxis, :, :],
            [batch_size, int(chunk_particles), 1, 1],
        )

    def observation_residual_fn(h_ref: tf.Tensor, observation: tf.Tensor) -> tf.Tensor:
        return observation[tf.newaxis, tf.newaxis, :] - h_ref

    def transition_log_density_fn(x_next: tf.Tensor, x_prev: tf.Tensor, time_index: tf.Tensor) -> tf.Tensor:
        residual = x_next - transition_mean(x_prev, time_index)
        return _batched_gaussian_logpdf(residual, tf.cast(tensors["transition_covariance"], DTYPE))

    def observation_log_density_fn(x: tf.Tensor, observation: tf.Tensor, time_index: tf.Tensor) -> tf.Tensor:
        del time_index
        residual = observation_fn(x) - observation[tf.newaxis, tf.newaxis, :]
        return _batched_gaussian_logpdf(residual, tf.cast(tensors["observation_covariance"], DTYPE))

    return {
        "prior_mean_fn": transition_mean,
        "pre_flow_step_fn": pre_flow_step,
        "observation_fn": observation_fn,
        "observation_jacobian_fn": observation_jacobian_fn,
        "observation_residual_fn": observation_residual_fn,
        "transition_log_density_fn": transition_log_density_fn,
        "observation_log_density_fn": observation_log_density_fn,
    }


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


def _full_row_requested(args: argparse.Namespace) -> bool:
    return (
        args.time_steps == FULL_ROW_TIME_STEPS
        and args.num_particles == FULL_ROW_NUM_PARTICLES
        and tuple(args.batch_seeds) == FULL_ROW_BATCH_SEEDS
        and args.transport_policy == FULL_ROW_TRANSPORT_POLICY
        and args.sinkhorn_iterations == FULL_ROW_SINKHORN_ITERATIONS
        and float(args.sinkhorn_epsilon) == FULL_ROW_SINKHORN_EPSILON
    )


def _write_markdown(path: Path, result: dict[str, Any], json_path: Path) -> None:
    lines = [
        "# Predator-Prey Same-Target LEDH Forward Scalar",
        "",
        f"- JSON artifact: `{json_path}`",
        f"- Row: `{result['row_id']}`",
        f"- Shape: `{result['shape']}`",
        f"- Output devices: `{result['output_devices']}`",
        f"- Precision: `{result['precision']}`",
        f"- History mode: `{result['history_mode']}`",
        f"- Compile plus first call seconds: `{result['compile_and_first_call_seconds']}`",
        f"- Warm-call timing summary seconds: `{result['warm_call_timing_summary_seconds']}`",
        f"- Finite output: `{result['finite_output']}`",
        f"- Admission status: `{result['admission_status']}`",
        "",
        "## Nonclaims",
        "",
    ]
    lines.extend(f"- {claim}" for claim in result["nonclaims"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    raise RuntimeError("ARCHIVAL_WRONG_TRANSPORT_CHUNK_POLICY: this route is preserved only as provenance and cannot emit new evidence")
    args = _parse_args()
    require_historical_raw_diagnostic_opt_in(
        args, route_name="predator-prey raw-barycentric value runner"
    )
    precision = _configure_precision(args)
    physical_gpus, logical_gpus = _configure_gpus()
    tensors, predator_prey_semantics = _build_predator_prey_tensors(args)
    adapter_callbacks = _make_predator_prey_callbacks(tensors, args.batch_seeds, args)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def compiled_outputs() -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        value = streaming_tf.streaming_batched_ledh_pfpf_ot_value_core_tf(
            observations=tensors["observations"],
            initial_particles=tensors["initial_particles"],
            fixed_resampling_mask=tensors["fixed_resampling_mask"],
            transition_matrix=tensors["transition_matrix"],
            transition_covariance=tensors["transition_covariance"],
            observation_covariance=tensors["observation_covariance"],
            observation_fn=adapter_callbacks["observation_fn"],
            observation_jacobian_fn=adapter_callbacks["observation_jacobian_fn"],
            observation_residual_fn=adapter_callbacks["observation_residual_fn"],
            transition_log_density_fn=adapter_callbacks["transition_log_density_fn"],
            observation_log_density_fn=adapter_callbacks["observation_log_density_fn"],
            prior_mean_fn=adapter_callbacks["prior_mean_fn"],
            pre_flow_step_fn=adapter_callbacks["pre_flow_step_fn"],
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

    log_likelihood, filtered_means, filtered_variances, ess_by_time = outputs
    output_devices = _validate_device((log_likelihood,), args.expect_device_kind)
    history_returned = args.history_mode == "full"
    finite_output = bool(tf.reduce_all(tf.math.is_finite(log_likelihood)).numpy())
    if history_returned:
        finite_output = bool(
            finite_output
            and tf.reduce_all(tf.math.is_finite(filtered_means)).numpy()
            and tf.reduce_all(tf.math.is_finite(filtered_variances)).numpy()
            and tf.reduce_all(tf.math.is_finite(ess_by_time)).numpy()
        )
    full_row = _full_row_requested(args)
    forward_contract = make_predator_prey_forward_contract(
        time_steps=args.time_steps,
        num_particles=args.num_particles,
        batch_seeds=args.batch_seeds,
        full_leaderboard_row=full_row,
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
        "row_id": PREDATOR_PREY_ROW_ID,
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
        "theta_coordinate_system": "physical",
        "flow_observation_policy": "predator_prey_identity_state_gaussian_flow_observation",
        "target_observation_policy": "additive_gaussian_predator_prey",
        "target_density_used_for_correction": True,
        "predator_prey_semantics": predator_prey_semantics,
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
        },
        "log_likelihood_by_seed": log_likelihood_values,
        "average_log_likelihood_by_seed": average_values,
        "finite_output": finite_output,
        "admission_status": LEDH_FORWARD_ADMISSION_STATUS_HISTORICAL_RAW,
        "normalization_checks": {
            "row_id": True,
            "batch_seeds": tuple(args.batch_seeds) == FULL_ROW_BATCH_SEEDS,
            "seed_count_matches_values": len(log_likelihood_values) == len(args.batch_seeds),
            "time_steps": args.time_steps == FULL_ROW_TIME_STEPS,
            "num_particles": args.num_particles == FULL_ROW_NUM_PARTICLES,
            "finite_log_likelihood": finite_output,
            "target_density_used_for_correction": True,
            "predator_prey_semantics_row_id": predator_prey_semantics["row_id"] == PREDATOR_PREY_ROW_ID,
            "full_row_requested": full_row,
        },
        "nonclaims": list(NONCLAIMS),
    }
    normalized_core = validate_ledh_forward_scalar_artifact(
        artifact,
        expected_row_id=PREDATOR_PREY_ROW_ID,
        require_admitted=False,
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
