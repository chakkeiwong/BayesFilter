"""LGSSM scale benchmark for experimental batched LEDH-PFPF-OT.

This harness intentionally lives beside, not inside, the experimental
implementation.  It measures a synthetic LGSSM-shaped likelihood evaluation at
larger state and particle dimensions while preserving benchmark-only fixture
generation and reporting boundaries.
"""

from __future__ import annotations

import argparse
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

import numpy as np
import tensorflow as tf

from experiments.dpf_implementation.tf_tfp.filters import experimental_batched_ledh_pfpf_ot_tf
from experiments.dpf_implementation.tf_tfp.resampling import annealed_transport_tf
from experiments.dpf_implementation.tf_tfp.filters.experimental_batched_ledh_pfpf_ot_tf import (
    BatchedLEDHPFPFOTValueTensors,
    batched_ledh_pfpf_ot_value_core_tf,
)


DEFAULT_DTYPE_NAME = "float32"
DEFAULT_TF32_MODE = "enabled"
DTYPE = tf.float32

NONCLAIMS = (
    "single synthetic LGSSM-shaped fixture only",
    "no production default readiness claim",
    "no CPU/GPU ranking claim",
    "no scalar parity claim",
    "no active transport gradient validation claim",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--mode", choices=("compiled-value",), default="compiled-value")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--time-steps", type=int, default=200)
    parser.add_argument("--num-particles", type=int, default=1000)
    parser.add_argument("--state-dim", type=int, default=20)
    parser.add_argument("--obs-dim", type=int, default=20)
    parser.add_argument(
        "--transport-policy",
        choices=("active-all", "active-odd", "no-resampling"),
        default="active-all",
    )
    parser.add_argument("--sinkhorn-iterations", type=int, default=10)
    parser.add_argument("--sinkhorn-epsilon", type=float, default=0.5)
    parser.add_argument("--annealed-scaling", type=float, default=0.9)
    parser.add_argument("--annealed-convergence-threshold", type=float, default=1.0e-3)
    parser.add_argument(
        "--transport-plan-mode",
        choices=("dense", "streaming"),
        default="dense",
    )
    parser.add_argument("--row-chunk-size", type=int, default=1024)
    parser.add_argument("--col-chunk-size", type=int, default=1024)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--dtype", choices=("float64", "float32"), default=DEFAULT_DTYPE_NAME)
    parser.add_argument(
        "--tf32-mode",
        choices=("default", "enabled", "disabled"),
        default=DEFAULT_TF32_MODE,
    )
    parser.add_argument("--device", default="/GPU:0")
    parser.add_argument(
        "--device-scope",
        choices=("cpu", "visible"),
        default=_PRE_ARGS.device_scope,
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default=_PRE_ARGS.cuda_visible_devices,
    )
    parser.add_argument(
        "--expect-device-kind",
        choices=("any", "cpu", "gpu"),
        default="gpu",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output", default=None)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.time_steps <= 0:
        raise ValueError("time_steps must be positive")
    if args.num_particles <= 1:
        raise ValueError("num_particles must be greater than 1")
    if args.state_dim <= 0 or args.obs_dim <= 0:
        raise ValueError("state_dim and obs_dim must be positive")
    if args.sinkhorn_iterations <= 0:
        raise ValueError("sinkhorn_iterations must be positive")
    if args.warmups < 0 or args.repeats <= 0:
        raise ValueError("warmups must be nonnegative and repeats must be positive")
    if args.row_chunk_size <= 0 or args.col_chunk_size <= 0:
        raise ValueError("row_chunk_size and col_chunk_size must be positive")
    return args


def _configure_precision(args: argparse.Namespace) -> dict[str, Any]:
    global DTYPE
    DTYPE = tf.float64 if args.dtype == "float64" else tf.float32
    experimental_batched_ledh_pfpf_ot_tf.DTYPE = DTYPE
    annealed_transport_tf.DTYPE = DTYPE
    if args.tf32_mode != "default":
        tf.config.experimental.enable_tensor_float_32_execution(args.tf32_mode == "enabled")
    metadata = experimental_batched_ledh_pfpf_ot_tf.precision_policy_metadata()
    metadata.update({
        "dtype": args.dtype,
        "tf_dtype": DTYPE.name,
        "tf32_mode": args.tf32_mode,
        "tf32_execution_enabled": bool(
            tf.config.experimental.tensor_float_32_execution_enabled()
        ),
    })
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


def _stable_lgssm_fixture(args: argparse.Namespace) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(args.seed)
    batch_size = args.batch_size
    time_steps = args.time_steps
    num_particles = args.num_particles
    state_dim = args.state_dim
    obs_dim = args.obs_dim

    batch = np.arange(batch_size, dtype=np.float64)
    state_grid = np.linspace(-1.0, 1.0, state_dim, dtype=np.float64)
    particle_grid = np.linspace(-1.0, 1.0, num_particles, dtype=np.float64)

    initial_particles = (
        0.08 * rng.standard_normal((batch_size, num_particles, state_dim))
        + 0.03 * state_grid[None, None, :]
        + 0.01 * particle_grid[None, :, None]
        + 0.0001 * batch[:, None, None]
    )

    diagonal = 0.86 + 0.08 * np.linspace(0.0, 1.0, state_dim, dtype=np.float64)
    transition_matrix = np.zeros((batch_size, state_dim, state_dim), dtype=np.float64)
    for row in range(batch_size):
        transition_matrix[row] = np.diag(diagonal + 0.00001 * row)
        upper = 0.006 * np.eye(state_dim, k=1, dtype=np.float64)
        lower = -0.004 * np.eye(state_dim, k=-1, dtype=np.float64)
        transition_matrix[row] += upper + lower

    q_diag = 0.04 + 0.004 * np.linspace(0.0, 1.0, state_dim, dtype=np.float64)
    r_diag = 0.06 + 0.006 * np.linspace(0.0, 1.0, obs_dim, dtype=np.float64)
    transition_covariance = np.tile(np.diag(q_diag)[None, :, :], (batch_size, 1, 1))
    observation_covariance = np.tile(np.diag(r_diag)[None, :, :], (batch_size, 1, 1))

    observation_matrix = np.zeros((batch_size, obs_dim, state_dim), dtype=np.float64)
    for row in range(batch_size):
        for obs_index in range(obs_dim):
            state_index = obs_index % state_dim
            observation_matrix[row, obs_index, state_index] = 1.0
            if state_dim > 1:
                observation_matrix[row, obs_index, (state_index + 1) % state_dim] = 0.025

    transitioned_initial = np.einsum(
        "bnj,bdj->bnd",
        initial_particles,
        transition_matrix,
    )
    time_grid = np.arange(time_steps, dtype=np.float64)
    time_wave = 0.012 * np.sin(0.017 * time_grid[:, None] * (np.arange(state_dim) + 1))
    particle_wave = 0.006 * np.cos(0.11 * particle_grid[:, None] * (np.arange(state_dim) + 1))
    pre_flow_particles = (
        transitioned_initial[:, None, :, :]
        + time_wave[None, :, None, :]
        + particle_wave[None, None, :, :]
    )

    observations = 0.05 * np.sin(
        0.023 * time_grid[:, None] * (np.arange(obs_dim, dtype=np.float64)[None, :] + 1.0)
    )
    observations += 0.02 * np.cos(
        0.011 * time_grid[:, None] * (np.arange(obs_dim, dtype=np.float64)[None, :] + 2.0)
    )

    if args.transport_policy == "active-all":
        fixed_resampling_mask = np.ones((batch_size, time_steps), dtype=bool)
    elif args.transport_policy == "active-odd":
        mask = (np.arange(time_steps)[None, :] % 2) == 1
        fixed_resampling_mask = np.broadcast_to(mask, (batch_size, time_steps)).copy()
    else:
        fixed_resampling_mask = np.zeros((batch_size, time_steps), dtype=bool)

    return {
        "observations": observations,
        "initial_particles": initial_particles,
        "pre_flow_particles": pre_flow_particles,
        "fixed_resampling_mask": fixed_resampling_mask,
        "transition_matrix": transition_matrix,
        "transition_covariance": transition_covariance,
        "observation_covariance": observation_covariance,
        "observation_matrix": observation_matrix,
    }


def _to_tensors(fixture: dict[str, np.ndarray]) -> dict[str, tf.Tensor]:
    tensors: dict[str, tf.Tensor] = {}
    for name, value in fixture.items():
        dtype = tf.bool if value.dtype == np.bool_ else DTYPE
        tensors[name] = tf.constant(value, dtype=dtype)
    return tensors


def _make_observation_fn(observation_matrix: tf.Tensor):
    def _observation(points: tf.Tensor) -> tf.Tensor:
        return tf.einsum("bmd,bnd->bnm", observation_matrix, points)

    return _observation


def _make_observation_jacobian_fn(observation_matrix: tf.Tensor):
    def _observation_jacobian(points: tf.Tensor) -> tf.Tensor:
        batch_size = points.shape[0]
        num_particles = points.shape[1]
        if batch_size is None or num_particles is None:
            raise ValueError("benchmark fixture requires static batch and particle dimensions")
        return tf.tile(observation_matrix[:, None, :, :], [1, num_particles, 1, 1])

    return _observation_jacobian


def _observation_residual(h_ref: tf.Tensor, observation: tf.Tensor) -> tf.Tensor:
    return observation[None, None, :] - h_ref


def _batched_gaussian_logpdf(residuals: tf.Tensor, covariance: tf.Tensor) -> tf.Tensor:
    chol = tf.linalg.cholesky(covariance)
    solved = tf.linalg.matrix_transpose(
        tf.linalg.cholesky_solve(chol, tf.linalg.matrix_transpose(residuals))
    )
    quad = tf.reduce_sum(solved * residuals, axis=-1)
    logdet = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)), axis=-1)
    dim = tf.cast(residuals.shape[-1], DTYPE)
    return -0.5 * (
        dim * tf.math.log(tf.constant(2.0 * np.pi, DTYPE))
        + logdet[:, None]
        + quad
    )


def _make_transition_log_density(
    transition_matrix: tf.Tensor,
    transition_covariance: tf.Tensor,
):
    def _transition_log_density(
        x_next: tf.Tensor,
        x_prev: tf.Tensor,
        _time_index: tf.Tensor,
    ) -> tf.Tensor:
        del _time_index
        mean = tf.einsum("bnj,bdj->bnd", x_prev, transition_matrix)
        return _batched_gaussian_logpdf(x_next - mean, transition_covariance)

    return _transition_log_density


def _make_observation_log_density(
    observation_matrix: tf.Tensor,
    observation_covariance: tf.Tensor,
):
    def _observation_log_density(
        x: tf.Tensor,
        observation: tf.Tensor,
        _time_index: tf.Tensor,
    ) -> tf.Tensor:
        del _time_index
        predicted = tf.einsum("bmd,bnd->bnm", observation_matrix, x)
        return _batched_gaussian_logpdf(
            predicted - observation[None, None, :],
            observation_covariance,
        )

    return _observation_log_density


def _value_from_tensors(
    tensors: dict[str, tf.Tensor],
    args: argparse.Namespace,
) -> BatchedLEDHPFPFOTValueTensors:
    return batched_ledh_pfpf_ot_value_core_tf(
        observations=tensors["observations"],
        initial_particles=tensors["initial_particles"],
        pre_flow_particles=tensors["pre_flow_particles"],
        fixed_resampling_mask=tensors["fixed_resampling_mask"],
        transition_matrix=tensors["transition_matrix"],
        transition_covariance=tensors["transition_covariance"],
        observation_covariance=tensors["observation_covariance"],
        observation_fn=_make_observation_fn(tensors["observation_matrix"]),
        observation_jacobian_fn=_make_observation_jacobian_fn(tensors["observation_matrix"]),
        observation_residual_fn=_observation_residual,
        transition_log_density_fn=_make_transition_log_density(
            tensors["transition_matrix"],
            tensors["transition_covariance"],
        ),
        observation_log_density_fn=_make_observation_log_density(
            tensors["observation_matrix"],
            tensors["observation_covariance"],
        ),
        sinkhorn_epsilon=args.sinkhorn_epsilon,
        annealed_scaling=args.annealed_scaling,
        annealed_convergence_threshold=args.annealed_convergence_threshold,
        sinkhorn_iterations=args.sinkhorn_iterations,
        transport_gradient_mode="raw",
        transport_plan_mode=args.transport_plan_mode,
        row_chunk_size=args.row_chunk_size,
        col_chunk_size=args.col_chunk_size,
    )


def _materialize(*tensors: tf.Tensor) -> None:
    for tensor in tensors:
        tensor.numpy()


def _validate_device(outputs: tuple[tf.Tensor, ...], expect_device_kind: str) -> list[str]:
    devices = [tensor.device for tensor in outputs]
    if expect_device_kind == "gpu":
        if not all("GPU" in device.upper() for device in devices):
            raise RuntimeError(f"expected GPU outputs, got {devices}")
    elif expect_device_kind == "cpu":
        if not all("CPU" in device.upper() for device in devices):
            raise RuntimeError(f"expected CPU outputs, got {devices}")
    return devices


def _summary(timings: list[float]) -> dict[str, float]:
    if not timings:
        return {}
    return {
        "min": min(timings),
        "median": statistics.median(timings),
        "max": max(timings),
    }


def _write_markdown(path: Path, result: dict[str, Any], json_path: Path) -> None:
    lines = [
        "# Experimental Batched LEDH-PFPF-OT LGSSM Scale Benchmark",
        "",
        f"- JSON artifact: `{json_path}`",
        f"- Mode: `{result['mode']}`",
        f"- Shape: B={result['batch_size']}, T={result['time_steps']}, "
        f"N={result['num_particles']}, state_dim={result['state_dim']}, "
        f"obs_dim={result['obs_dim']}",
        f"- Device request: `{result['device']}`",
        f"- Output devices: `{result['output_devices']}`",
        f"- Precision: `{result['precision']}`",
        f"- Transport plan mode: `{result['transport_plan_mode']}`",
        f"- JIT compiled: `{result['jit_compile']}`",
        f"- Compile plus first call seconds: `{result['compile_and_first_call_seconds']}`",
        f"- Warm-call timing summary seconds: `{result['warm_call_timing_summary_seconds']}`",
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
    fixture = _stable_lgssm_fixture(args)
    tensors = _to_tensors(fixture)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def compiled_value() -> tf.Tensor:
        return _value_from_tensors(tensors, args).log_likelihood

    with tf.device(args.device):
        start = time.perf_counter()
        value = compiled_value()
        _materialize(value)
        compile_and_first = time.perf_counter() - start

        for _ in range(args.warmups):
            _materialize(compiled_value())

        timings: list[float] = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            value = compiled_value()
            _materialize(value)
            timings.append(time.perf_counter() - start)

    value_np = value.numpy()
    output_devices = _validate_device((value,), args.expect_device_kind)
    result: dict[str, Any] = {
        "mode": args.mode,
        "batch_size": args.batch_size,
        "time_steps": args.time_steps,
        "num_particles": args.num_particles,
        "state_dim": args.state_dim,
        "obs_dim": args.obs_dim,
        "transport_policy": args.transport_policy,
        "transport_plan_mode": args.transport_plan_mode,
        "row_chunk_size": args.row_chunk_size,
        "col_chunk_size": args.col_chunk_size,
        "sinkhorn_iterations": args.sinkhorn_iterations,
        "sinkhorn_epsilon": args.sinkhorn_epsilon,
        "annealed_scaling": args.annealed_scaling,
        "annealed_convergence_threshold": args.annealed_convergence_threshold,
        "device": args.device,
        "device_scope": args.device_scope,
        "cuda_visible_devices": args.cuda_visible_devices,
        "expect_device_kind": args.expect_device_kind,
        "precision": precision,
        "physical_gpus": physical_gpus,
        "logical_gpus": logical_gpus,
        "gpu_memory_info": _gpu_memory_info(),
        "jit_compile": True,
        "compiled_unit": "experimental_batched_ledh_pfpf_ot_lgssm_value",
        "compile_and_first_call_seconds": compile_and_first,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "warm_call_timings_seconds": timings,
        "warm_call_timing_summary_seconds": _summary(timings),
        "output_devices": output_devices,
        "output_shape": list(value_np.shape),
        "finite_output": bool(np.isfinite(value_np).all()),
        "log_likelihood_preview": [float(v) for v in value_np.reshape(-1)[: min(8, value_np.size)]],
        "tensorflow_version": tf.__version__,
        "python_version": platform.python_version(),
        "nonclaims": list(NONCLAIMS),
    }
    if not result["finite_output"]:
        raise FloatingPointError("compiled benchmark emitted non-finite value")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_output is not None:
        markdown_output = Path(args.markdown_output)
        markdown_output.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(markdown_output, result, output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
