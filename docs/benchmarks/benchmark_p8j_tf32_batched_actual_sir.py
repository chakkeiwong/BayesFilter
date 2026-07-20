"""Actual-SIR d18 TF32/GPU probe for experimental batched LEDH-PFPF-OT.

This is a P8j diagnostic harness.  It deliberately reuses the SIR callbacks
from ``scripts/filtering_value_gradient_benchmark_run_p8d_numeric.py`` and is
not a leaderboard runner.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import statistics
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
    make_fixed_sir_logscale_forward_contract,
)
from experiments.dpf_implementation.tf_tfp.filters import (
    experimental_batched_ledh_pfpf_ot_streaming_tf as streaming_tf,
)
from experiments.dpf_implementation.tf_tfp.filters import (
    experimental_batched_ledh_pfpf_ot_tf as core_tf,
)
from experiments.dpf_implementation.tf_tfp.resampling import annealed_transport_tf
from scripts.filtering_value_gradient_benchmark_run_p8d_numeric import (
    SIR_ROW,
    _dpf_sir_callbacks,
    _git_commit,
    _sir_observations,
)


DEFAULT_SCALAR_COMPARATOR_SECONDS = 789.755664
DEFAULT_SCALAR_COMPARATOR_SCOPE = (
    "P8j Phase 5d scalar actual-SIR adaptive LEDH OT N=64 "
    "five-seed trusted-GPU run"
)
DTYPE = tf.float32
NONCLAIMS = (
    "actual SIR d18 TF32/GPU feasibility probe only",
    "production/default target by owner directive",
    "not SIR d18 particle-count adequacy evidence",
    "not leaderboard completion",
    "not MC-SE adequacy evidence",
    "not exact likelihood correctness",
    "not DPF gradient correctness",
    "not HMC/NUTS readiness",
    "not Zhao-Cui TT/SIRT or MATLAB parity",
)


def _parse_int_csv(value: str) -> list[int]:
    entries = [item.strip() for item in str(value).split(",") if item.strip()]
    if not entries:
        raise ValueError("expected at least one integer")
    return [int(item) for item in entries]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--batch-seeds", default="81120,81121,81122,81123,81124")
    parser.add_argument("--time-steps", type=int, default=20)
    parser.add_argument("--num-particles", type=int, default=64)
    parser.add_argument(
        "--transport-policy",
        choices=("active-all", "active-odd", "no-resampling"),
        default="active-all",
    )
    parser.add_argument("--sinkhorn-iterations", type=int, default=10)
    parser.add_argument("--sinkhorn-epsilon", type=float, default=1.0)
    parser.add_argument("--annealed-scaling", type=float, default=0.9)
    parser.add_argument("--annealed-convergence-threshold", type=float, default=1.0e-3)
    parser.add_argument("--row-chunk-size", type=int, default=128)
    parser.add_argument("--col-chunk-size", type=int, default=128)
    parser.add_argument("--particle-chunk-size", type=int, default=64)
    parser.add_argument(
        "--history-mode",
        choices=("full", "value-only"),
        default="full",
        help="Return full filtered history diagnostics or likelihood value only.",
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed-salt", type=int, default=4242)
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
    args = parser.parse_args()
    args.batch_seeds = _parse_int_csv(args.batch_seeds)
    if args.time_steps <= 0:
        raise ValueError("time_steps must be positive")
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


def _summary(timings: list[float]) -> dict[str, float]:
    if not timings:
        return {}
    return {
        "min": min(timings),
        "median": statistics.median(timings),
        "mean": statistics.fmean(timings),
        "max": max(timings),
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


def _seed_pair(seed: int, salt: int) -> tf.Tensor:
    return tf.constant(
        [int(seed) % 2147483647, int(salt) % 2147483647],
        dtype=tf.int32,
    )


def _materialize(*tensors: tf.Tensor) -> None:
    for tensor in tensors:
        tensor.numpy()


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


def _make_fixed_resampling_mask(batch_size: int, time_steps: int, policy: str) -> tf.Tensor:
    if policy == "active-all":
        mask = tf.ones([batch_size, time_steps], dtype=tf.bool)
    elif policy == "active-odd":
        active = tf.equal(tf.math.floormod(tf.range(time_steps), 2), 1)
        mask = tf.tile(active[tf.newaxis, :], [batch_size, 1])
    else:
        mask = tf.zeros([batch_size, time_steps], dtype=tf.bool)
    return mask


def _build_actual_sir_tensors(args: argparse.Namespace) -> tuple[dict[str, tf.Tensor], dict[str, Any]]:
    callbacks = _dpf_sir_callbacks()
    metadata = dict(callbacks["sir_model_metadata"])
    adapter = dict(callbacks["ledh_observation_adapter"])
    observations = _sir_observations()[: args.time_steps]
    batch_size = len(args.batch_seeds)
    initial_particles = tf.stack(
        [
            tf.cast(
                callbacks["initial_sample"](args.num_particles, int(seed)),
                dtype=DTYPE,
            )
            for seed in args.batch_seeds
        ],
        axis=0,
    )
    initial_shape = initial_particles.shape.as_list()
    if len(initial_shape) != 3:
        raise ValueError("initial SIR particles must have shape [B,N,D]")
    state_dim = int(initial_shape[2])
    obs_dim = int(observations.shape[1])
    if state_dim != 18 or obs_dim != 9:
        raise ValueError(f"unexpected SIR shape D={state_dim}, M={obs_dim}")
    if args.time_steps > int(_sir_observations().shape[0]):
        raise ValueError("time_steps exceeds available SIR observations")

    transition_covariance = tf.tile(
        tf.cast(callbacks["process_noise_covariance_fn"](initial_particles[0, 0], 0), DTYPE)[
            tf.newaxis, :, :
        ],
        [batch_size, 1, 1],
    )
    observation_covariance = tf.tile(
        tf.cast(callbacks["observation_covariance_fn"](0), DTYPE)[tf.newaxis, :, :],
        [batch_size, 1, 1],
    )
    tensors = {
        "observations": tf.cast(observations, DTYPE),
        "initial_particles": initial_particles,
        "fixed_resampling_mask": _make_fixed_resampling_mask(
            batch_size,
            args.time_steps,
            args.transport_policy,
        ),
        "transition_matrix": tf.tile(tf.eye(state_dim, dtype=DTYPE)[tf.newaxis, :, :], [batch_size, 1, 1]),
        "transition_covariance": transition_covariance,
        "observation_covariance": observation_covariance,
    }
    sir_semantics = {
        "row_id": metadata["row_id"],
        "state_dimension": int(metadata["state_dimension"]),
        "observation_dimension": int(metadata["observation_dimension"]),
        "process_noise_policy": metadata["process_noise_policy"],
        "flow_observation_contract": adapter["flow_observation_contract"],
        "target_density_used_for_correction": bool(adapter["target_density_used_for_correction"]),
        "adapter_classification": adapter["adapter_classification"],
        "actual_sir_callbacks_used": True,
        "actual_sir_callback_usage": (
            "initial_sample, process/observation covariance callbacks, "
            "observation metadata, and route metadata"
        ),
        "graph_compatible_sir_transition_copy_used": True,
        "graph_compatible_sir_transition_source": (
            "bayesfilter.highdim.zhao_cui_sir_austria_model RK4 "
            "zhao_cui_sir_step equations, copied into TensorFlow graph form "
            "because the model validation path uses eager .numpy() checks"
        ),
        "nonlinear_prior_mean_hook_used": True,
        "scalar_algorithm1_ukf_covariance_lifecycle_parity_claim": False,
        "experimental_streaming_adapter_route": (
            "batched nonlinear-prior LEDH-PFPF-OT streaming adapter with fixed "
            "process covariance, not scalar Li-Coates Algorithm 1 UKF "
            "covariance lifecycle parity"
        ),
    }
    return tensors, sir_semantics


def _make_actual_sir_callbacks(
    callbacks: dict[str, Any],
    tensors: dict[str, tf.Tensor],
    seeds: list[int],
    args: argparse.Namespace,
):
    model = highdim.zhao_cui_sir_austria_model()
    kappa = tf.cast(model.kappa, DTYPE)
    nu = tf.cast(model.nu, DTYPE)
    adjacency = tf.cast(model._adjacency_matrix, DTYPE)
    neighbor_degree = tf.cast(model._neighbor_degree, DTYPE)
    substeps = int(model._rk4_substeps)
    step_size = tf.cast(model.delta, DTYPE) / tf.cast(substeps, DTYPE)
    process_chol = tf.linalg.cholesky(
        tf.cast(callbacks["process_noise_covariance_fn"](tf.zeros([18], dtype=tf.float64), 0), DTYPE)
    )
    selector = tf.cast(
        tf.one_hot(tf.constant(range(1, 18, 2), dtype=tf.int32), depth=18, dtype=tf.float64),
        DTYPE,
    )
    batch_size = len(seeds)
    num_particles = args.num_particles

    def apply_sir_process_noise_policy(points: tf.Tensor) -> tf.Tensor:
        susceptible = tf.maximum(points[:, :, 0::2], tf.constant(0.0, dtype=DTYPE))
        infectious = points[:, :, 1::2]
        reshaped = tf.reshape(tf.stack([susceptible, infectious], axis=3), tf.shape(points))
        reshaped.set_shape(points.shape)
        return reshaped

    def sir_rhs(points: tf.Tensor) -> tf.Tensor:
        susceptible = points[:, :, 0::2]
        infectious = points[:, :, 1::2]
        susceptible_neighbor = (
            tf.linalg.matmul(susceptible, adjacency, transpose_b=True)
            - susceptible * neighbor_degree[tf.newaxis, tf.newaxis, :]
        )
        infectious_neighbor = (
            tf.linalg.matmul(infectious, adjacency, transpose_b=True)
            - infectious * neighbor_degree[tf.newaxis, tf.newaxis, :]
        )
        infection = kappa[tf.newaxis, tf.newaxis, :] * susceptible * infectious
        d_susceptible = -infection + tf.constant(0.5, dtype=DTYPE) * susceptible_neighbor
        d_infectious = (
            infection
            - nu[tf.newaxis, tf.newaxis, :] * infectious
            + tf.constant(0.5, dtype=DTYPE) * infectious_neighbor
        )
        reshaped = tf.reshape(
            tf.stack([d_susceptible, d_infectious], axis=3),
            tf.shape(points),
        )
        reshaped.set_shape(points.shape)
        return reshaped

    def transition_mean(points: tf.Tensor, time_index: tf.Tensor) -> tf.Tensor:
        del time_index
        state = tf.cast(points, DTYPE)
        for _ in range(substeps):
            k1 = sir_rhs(state)
            k2 = sir_rhs(state + tf.constant(0.5, dtype=DTYPE) * step_size * k1)
            k3 = sir_rhs(state + tf.constant(0.5, dtype=DTYPE) * step_size * k2)
            k4 = sir_rhs(state + tf.constant(0.5, dtype=DTYPE) * step_size * k3)
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
        for batch_index, seed in enumerate(seeds):
            seed_tensor = tf.stack(
                [
                    tf.constant(int(seed) % 2147483647, dtype=tf.int32),
                    tf.math.floormod(
                        tf.constant(1140, dtype=tf.int32) + time_index,
                        tf.constant(2147483647, dtype=tf.int32),
                    ),
                ]
            )
            noise = tf.random.stateless_normal(
                [num_particles, 18],
                seed=seed_tensor,
                dtype=DTYPE,
            )
            noise_rows.append(noise)
        noise_tensor = tf.stack(noise_rows, axis=0)
        pushed = mean + tf.einsum("bnd,ed->bne", noise_tensor, process_chol)
        return apply_sir_process_noise_policy(pushed)

    def observation_fn(points: tf.Tensor) -> tf.Tensor:
        return tf.gather(points, list(range(1, 18, 2)), axis=2)

    def observation_jacobian_fn(points: tf.Tensor) -> tf.Tensor:
        chunk_particles = points.shape[1]
        if chunk_particles is None:
            raise ValueError("actual-SIR probe requires static particle chunk dimension")
        return tf.tile(
            selector[tf.newaxis, tf.newaxis, :, :],
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


def _write_markdown(path: Path, result: dict[str, Any], json_path: Path) -> None:
    lines = [
        "# P8j TF32 Batched Actual-SIR Probe",
        "",
        f"- JSON artifact: `{json_path}`",
        f"- Row: `{result['sir_semantics']['row_id']}`",
        f"- Shape: `{result['shape']}`",
        f"- Output devices: `{result['output_devices']}`",
        f"- Precision: `{result['precision']}`",
        f"- History mode: `{result['history_mode']}`",
        f"- Compile plus first call seconds: `{result['compile_and_first_call_seconds']}`",
        f"- Warm-call timing summary seconds: `{result['warm_call_timing_summary_seconds']}`",
        f"- Speedup vs scalar comparator: `{result['speedup_vs_scalar_comparator_mean_warm_call']}`",
        f"- Finite output: `{result['finite_output']}`",
        "",
        "## Nonclaims",
        "",
    ]
    lines.extend(f"- {claim}" for claim in result["nonclaims"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    raise RuntimeError("ARCHIVAL_WRONG_TRANSPORT_CHUNK_POLICY: this route is preserved only as provenance and cannot emit new evidence")
    args = _parse_args()
    precision = _configure_precision(args)
    physical_gpus, logical_gpus = _configure_gpus()
    callbacks = _dpf_sir_callbacks()
    global tensors
    tensors, sir_semantics = _build_actual_sir_tensors(args)
    adapter_callbacks = _make_actual_sir_callbacks(callbacks, tensors, args.batch_seeds, args)

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
    warm_summary = _summary(timings)
    mean_warm = warm_summary.get("mean")
    runtime_gate_applicable = bool(
        args.expect_device_kind == "gpu"
        and all("GPU" in device.upper() for device in output_devices)
    )
    speedup = (
        DEFAULT_SCALAR_COMPARATOR_SECONDS / mean_warm
        if runtime_gate_applicable and mean_warm and mean_warm > 0.0
        else None
    )
    history_returned = args.history_mode == "full"
    finite_output = bool(tf.reduce_all(tf.math.is_finite(log_likelihood)).numpy())
    if history_returned:
        finite_output = bool(
            finite_output
            and tf.reduce_all(tf.math.is_finite(filtered_means)).numpy()
            and tf.reduce_all(tf.math.is_finite(filtered_variances)).numpy()
            and tf.reduce_all(tf.math.is_finite(ess_by_time)).numpy()
        )
    forward_contract = make_fixed_sir_logscale_forward_contract(
        time_steps=args.time_steps,
        num_particles=args.num_particles,
        batch_seeds=args.batch_seeds,
        full_leaderboard_row=(
            args.time_steps == 20
            and args.num_particles == 10000
            and tuple(args.batch_seeds) == (81120, 81121, 81122, 81123, 81124)
            and args.transport_policy == "active-all"
            and args.sinkhorn_iterations == 10
            and float(args.sinkhorn_epsilon) == 1.0
        ),
    ).to_manifest()
    result: dict[str, Any] = {
        "schema_version": "filter_bench.p8j_tf32_batched_actual_sir_probe.v1",
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
        "shape": {
            "batch_size": len(args.batch_seeds),
            "batch_seed_count": len(args.batch_seeds),
            "time_steps": args.time_steps,
            "num_particles": args.num_particles,
            "state_dim": 18,
            "obs_dim": 9,
        },
        "batch_seeds": [int(seed) for seed in args.batch_seeds],
        "full_batched_evaluation_per_warm_call": True,
        "history_mode": args.history_mode,
        "return_history": history_returned,
        "forward_contract": forward_contract,
        "target_scalar": forward_contract["target_scalar"],
        "target_output_tensor_field": forward_contract["output_tensor_field"],
        "target_density_fields": forward_contract["target_density_fields"],
        "proposal_flow_fields": forward_contract["proposal_flow_fields"],
        "correction_formula": forward_contract["correction_formula"],
        "sir_semantics": sir_semantics,
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
        "warm_call_timing_summary_seconds": warm_summary,
        "scalar_comparator_runtime_seconds": DEFAULT_SCALAR_COMPARATOR_SECONDS,
        "scalar_comparator_scope": DEFAULT_SCALAR_COMPARATOR_SCOPE,
        "runtime_gate_applicable": runtime_gate_applicable,
        "speedup_vs_scalar_comparator_mean_warm_call": speedup,
        "gpu_memory_info_before": memory_before,
        "gpu_memory_info_after": memory_after,
        "output_devices": output_devices,
        "output_shape": list(log_likelihood.numpy().shape),
        "history_shapes": {
            "filtered_means": list(filtered_means.numpy().shape),
            "filtered_variances": list(filtered_variances.numpy().shape),
            "ess_by_time": list(ess_by_time.numpy().shape),
        },
        "log_likelihood": [float(value) for value in log_likelihood.numpy().reshape(-1)],
        "ess_min_by_seed": (
            [float(value) for value in tf.reduce_min(ess_by_time, axis=0).numpy().reshape(-1)]
            if history_returned
            else None
        ),
        "ess_summary_available": history_returned,
        "finite_output": finite_output,
        "primary_pass_5x_runtime_gate": bool(
            runtime_gate_applicable
            and finite_output
            and speedup is not None
            and speedup >= 5.0
        ),
        "nonclaims": list(NONCLAIMS),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_output is not None:
        _write_markdown(Path(args.markdown_output), result, output_path)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
