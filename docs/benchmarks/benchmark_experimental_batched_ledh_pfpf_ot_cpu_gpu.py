"""Benchmark experimental batched LEDH-PFPF-OT value and value+score.

This standalone harness exercises only the additive experimental module in
``experiments.dpf_implementation.tf_tfp.filters.experimental_batched_ledh_pfpf_ot_tf``.
It keeps the time axis sequential and batches independent parameter rows along
the leading axis.
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
_PRE_PARSER.add_argument(
    "--device-scope",
    choices=("cpu", "visible"),
    default="cpu",
    help="Hide GPU for CPU runs or leave configured devices visible.",
)
_PRE_PARSER.add_argument(
    "--cuda-visible-devices",
    default=None,
    help="Set CUDA_VISIBLE_DEVICES before TensorFlow import.",
)
_PRE_ARGS, _ = _PRE_PARSER.parse_known_args()
if _PRE_ARGS.cuda_visible_devices is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _PRE_ARGS.cuda_visible_devices
elif _PRE_ARGS.device_scope == "cpu":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
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
    batched_ledh_pfpf_ot_value_and_score_tf,
    batched_ledh_pfpf_ot_value_core_tf,
)


DTYPE = tf.float64
experimental_batched_ledh_pfpf_ot_tf.DTYPE = DTYPE
annealed_transport_tf.DTYPE = DTYPE


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--mode",
        choices=(
            "compiled-value",
            "compiled-value-score",
            "scalar-compiled-value-loop",
            "parity",
        ),
        required=True,
    )
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--time-steps", type=int, default=3)
    parser.add_argument("--num-particles", type=int, default=4)
    parser.add_argument("--state-dim", type=int, default=1)
    parser.add_argument("--obs-dim", type=int, default=1)
    parser.add_argument("--parameter-dim", type=int, default=3)
    parser.add_argument(
        "--transport-policy",
        choices=("active", "no-resampling"),
        default="active",
    )
    parser.add_argument(
        "--transport-gradient-mode",
        choices=("filterflow_clipped", "filterflow_custom_op", "raw"),
        default="raw",
    )
    parser.add_argument(
        "--transport-plan-mode",
        choices=("dense", "streaming"),
        default="dense",
    )
    parser.add_argument("--row-chunk-size", type=int, default=1024)
    parser.add_argument("--col-chunk-size", type=int, default=1024)
    parser.add_argument("--sinkhorn-iterations", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", default="/CPU:0")
    parser.add_argument(
        "--device-scope",
        choices=("cpu", "visible"),
        default=_PRE_ARGS.device_scope,
        help="Hide GPU for CPU runs or leave configured devices visible.",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default=_PRE_ARGS.cuda_visible_devices,
        help="Set CUDA_VISIBLE_DEVICES before TensorFlow import.",
    )
    parser.add_argument(
        "--expect-device-kind",
        choices=("any", "cpu", "gpu"),
        default="any",
        help="Fail closed if output tensors are not placed as expected.",
    )
    parser.add_argument("--value-rtol", type=float, default=1.0e-10)
    parser.add_argument("--value-atol", type=float, default=1.0e-10)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output", default=None)
    args = parser.parse_args()
    if args.transport_plan_mode == "streaming" and args.transport_gradient_mode != "raw":
        raise ValueError("streaming transport requires --transport-gradient-mode raw")
    if args.row_chunk_size <= 0 or args.col_chunk_size <= 0:
        raise ValueError("row_chunk_size and col_chunk_size must be positive")
    return args


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


def _stable_fixture(args: argparse.Namespace) -> dict[str, np.ndarray]:
    if args.state_dim != 1 or args.obs_dim != 1:
        raise ValueError("this diagnostic fixture currently expects state_dim=obs_dim=1")
    if args.parameter_dim != 3:
        raise ValueError("this diagnostic fixture expects parameter_dim=3")
    if args.batch_size <= 0 or args.time_steps <= 0 or args.num_particles <= 1:
        raise ValueError("batch_size/time_steps must be positive and num_particles > 1")

    batch = np.arange(args.batch_size, dtype=np.float64)
    particles = np.linspace(-0.30, 0.45, args.num_particles, dtype=np.float64)
    initial_particles = particles[None, :, None] + 0.002 * batch[:, None, None]
    theta = np.stack(
        [
            0.82 + 0.0002 * batch,
            np.log(0.16 + 0.00005 * batch),
            np.log(0.25 + 0.00005 * batch),
        ],
        axis=1,
    )
    transition_matrix = theta[:, 0:1, None]
    time_grid = np.arange(args.time_steps, dtype=np.float64)
    offsets = np.linspace(-0.04, 0.05, args.num_particles, dtype=np.float64)
    pre_flow_particles = (
        transition_matrix[:, None, :, :] * initial_particles[:, None, :, :]
        + offsets[None, None, :, None]
        + 0.002 * time_grid[None, :, None, None]
    )
    observations = (0.03 + 0.02 * np.sin(0.7 * time_grid))[:, None]
    if args.transport_policy == "active":
        mask = (np.arange(args.time_steps)[None, :] % 2) == 1
        fixed_resampling_mask = np.broadcast_to(mask, (args.batch_size, args.time_steps)).copy()
    else:
        fixed_resampling_mask = np.zeros((args.batch_size, args.time_steps), dtype=bool)
    return {
        "theta": theta,
        "observations": observations,
        "initial_particles": initial_particles,
        "pre_flow_particles": pre_flow_particles,
        "fixed_resampling_mask": fixed_resampling_mask,
    }


def _to_tensors(fixture: dict[str, np.ndarray]) -> dict[str, tf.Tensor]:
    tensors: dict[str, tf.Tensor] = {}
    for name, value in fixture.items():
        dtype = tf.bool if value.dtype == np.bool_ else DTYPE
        tensors[name] = tf.constant(value, dtype=dtype)
    return tensors


def _observation(points: tf.Tensor) -> tf.Tensor:
    return points


def _observation_jacobian(points: tf.Tensor) -> tf.Tensor:
    batch_size = points.shape[0]
    num_particles = points.shape[1]
    if batch_size is None or num_particles is None:
        raise ValueError("benchmark fixture requires static batch and particle dimensions")
    return tf.ones([batch_size, num_particles, 1, 1], dtype=DTYPE)


def _observation_residual(h_ref: tf.Tensor, observation: tf.Tensor) -> tf.Tensor:
    return observation[None, None, :] - h_ref


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
        residual = x_next - mean
        variance = transition_covariance[:, 0, 0]
        quad = residual[:, :, 0] * residual[:, :, 0] / variance[:, None]
        return -0.5 * (
            tf.math.log(tf.constant(2.0 * np.pi, dtype=DTYPE))
            + tf.math.log(variance)[:, None]
            + quad
        )

    return _transition_log_density


def _make_observation_log_density(observation_covariance: tf.Tensor):
    def _observation_log_density(
        x: tf.Tensor,
        observation: tf.Tensor,
        _time_index: tf.Tensor,
    ) -> tf.Tensor:
        del _time_index
        variance = observation_covariance[:, 0, 0]
        residual = x[:, :, 0] - observation[0]
        return -0.5 * (
            tf.math.log(tf.constant(2.0 * np.pi, dtype=DTYPE))
            + tf.math.log(variance)[:, None]
            + residual * residual / variance[:, None]
        )

    return _observation_log_density


def _value_from_theta(
    theta: tf.Tensor,
    tensors: dict[str, tf.Tensor],
    args: argparse.Namespace,
) -> BatchedLEDHPFPFOTValueTensors:
    transition_matrix = theta[:, 0:1, None]
    transition_covariance = tf.exp(theta[:, 1:2])[:, :, None]
    observation_covariance = tf.exp(theta[:, 2:3])[:, :, None]
    return batched_ledh_pfpf_ot_value_core_tf(
        observations=tensors["observations"],
        initial_particles=tensors["initial_particles"],
        pre_flow_particles=tensors["pre_flow_particles"],
        fixed_resampling_mask=tensors["fixed_resampling_mask"],
        transition_matrix=transition_matrix,
        transition_covariance=transition_covariance,
        observation_covariance=observation_covariance,
        observation_fn=_observation,
        observation_jacobian_fn=_observation_jacobian,
        observation_residual_fn=_observation_residual,
        transition_log_density_fn=_make_transition_log_density(
            transition_matrix,
            transition_covariance,
        ),
        observation_log_density_fn=_make_observation_log_density(
            observation_covariance,
        ),
        sinkhorn_iterations=args.sinkhorn_iterations,
        transport_gradient_mode=args.transport_gradient_mode,
        transport_plan_mode=args.transport_plan_mode,
        row_chunk_size=args.row_chunk_size,
        col_chunk_size=args.col_chunk_size,
    )


def _value_score_from_theta(
    theta: tf.Tensor,
    tensors: dict[str, tf.Tensor],
    args: argparse.Namespace,
) -> tuple[tf.Tensor, tf.Tensor]:
    result = batched_ledh_pfpf_ot_value_and_score_tf(
        theta,
        lambda values: _value_from_theta(values, tensors, args),
    )
    return result.log_likelihood, result.score


def _materialize(*tensors: tf.Tensor) -> None:
    for tensor in tensors:
        _ = tensor.numpy()


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "min_seconds": min(values),
        "median_seconds": statistics.median(values),
        "mean_seconds": statistics.fmean(values),
        "max_seconds": max(values),
    }


def _validate_device(
    *,
    expect_device_kind: str,
    physical_gpus: list[str],
    outputs: tuple[tf.Tensor, ...],
) -> None:
    devices = [tensor.device for tensor in outputs]
    if expect_device_kind == "gpu":
        if not physical_gpus:
            raise RuntimeError("expected a GPU run, but TensorFlow sees no physical GPUs")
        if not all("GPU" in device.upper() for device in devices):
            raise RuntimeError(f"expected GPU tensor placement, got {devices}")
    if expect_device_kind == "cpu" and not all("CPU" in device.upper() for device in devices):
        raise RuntimeError(f"expected CPU tensor placement, got {devices}")


def _base_result(
    args: argparse.Namespace,
    physical_gpus: list[str],
    logical_gpus: list[str],
) -> dict[str, Any]:
    return {
        "timestamp_utc": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "host": platform.node(),
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_arg": args.device,
        "device_scope": args.device_scope,
        "expect_device_kind": args.expect_device_kind,
        "physical_gpus": physical_gpus,
        "logical_gpus": logical_gpus,
        "shape": {
            "batch_size": args.batch_size,
            "time_steps": args.time_steps,
            "num_particles": args.num_particles,
            "state_dim": args.state_dim,
            "obs_dim": args.obs_dim,
            "parameter_dim": args.parameter_dim,
        },
        "transport": {
            "policy": args.transport_policy,
            "gradient_mode": args.transport_gradient_mode,
            "plan_mode": args.transport_plan_mode,
            "row_chunk_size": args.row_chunk_size,
            "col_chunk_size": args.col_chunk_size,
            "sinkhorn_iterations": args.sinkhorn_iterations,
        },
        "notes": [
            "Single-shape diagnostic only; not a production benchmark.",
            "Experimental opt-in DPF batching path only.",
            "Time remains sequential; the batch axis parallelizes independent parameter rows.",
            "Benchmarks do not establish classical PF score correctness or posterior validity.",
        ],
    }


def _finish_result(
    *,
    args: argparse.Namespace,
    physical_gpus: list[str],
    logical_gpus: list[str],
    mode: str,
    outputs: tuple[tf.Tensor, ...],
    timings: list[float],
    compile_and_first_call_seconds: float,
    compiled_unit: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    arrays = [tensor.numpy() for tensor in outputs]
    finite = bool(all(np.isfinite(array).all() for array in arrays))
    if not finite:
        raise RuntimeError("nonfinite benchmark output")
    _validate_device(
        expect_device_kind=args.expect_device_kind,
        physical_gpus=physical_gpus,
        outputs=outputs,
    )
    result = _base_result(args, physical_gpus, logical_gpus)
    result.update(
        {
            "mode": mode,
            "compiler": {
                "tf_function": True,
                "jit_compile": True,
                "compile_and_first_call_seconds": compile_and_first_call_seconds,
                "warm_calls_exclude_compile": True,
                "compiled_unit": compiled_unit,
            },
            "timing_policy": {
                "first_call_includes_trace_compile_and_initialization": True,
                "warmups": args.warmups,
                "repeats": args.repeats,
                "materialization": "all output tensors materialized with .numpy()",
            },
            "warm_call_summary": _summary(timings),
            "finite_outputs": finite,
            "output_devices": [tensor.device for tensor in outputs],
            "output_shapes": [list(array.shape) for array in arrays],
            "gpu_memory_info_after": _gpu_memory_info(),
        }
    )
    if arrays:
        result["value_sum"] = float(np.sum(arrays[0]))
        result["value_vector_full"] = [float(value) for value in arrays[0].reshape(-1)]
        result["value_vector_preview"] = _preview_vector(arrays[0])
    if len(arrays) > 1:
        result["score_abs_max"] = float(np.max(np.abs(arrays[1])))
        result["score_first_row"] = [float(value) for value in arrays[1][0].reshape(-1)]
    if extra:
        result.update(extra)
    return result


def _run_compiled_value(
    args: argparse.Namespace,
    tensors: dict[str, tf.Tensor],
    physical_gpus: list[str],
    logical_gpus: list[str],
) -> dict[str, Any]:
    @tf.function(jit_compile=True, reduce_retracing=True)
    def compiled(theta: tf.Tensor) -> tf.Tensor:
        return _value_from_theta(theta, tensors, args).log_likelihood

    with tf.device(args.device):
        start = time.perf_counter()
        value = compiled(tensors["theta"])
        _materialize(value)
        compile_and_first = time.perf_counter() - start
        for _ in range(args.warmups):
            _materialize(compiled(tensors["theta"]))
        timings: list[float] = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            value = compiled(tensors["theta"])
            _materialize(value)
            timings.append(time.perf_counter() - start)
    return _finish_result(
        args=args,
        physical_gpus=physical_gpus,
        logical_gpus=logical_gpus,
        mode="compiled-value",
        outputs=(value,),
        timings=timings,
        compile_and_first_call_seconds=compile_and_first,
        compiled_unit="batched_value",
    )


def _run_compiled_value_score(
    args: argparse.Namespace,
    tensors: dict[str, tf.Tensor],
    physical_gpus: list[str],
    logical_gpus: list[str],
) -> dict[str, Any]:
    @tf.function(jit_compile=True, reduce_retracing=True)
    def compiled(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        return _value_score_from_theta(theta, tensors, args)

    with tf.device(args.device):
        start = time.perf_counter()
        value, score = compiled(tensors["theta"])
        _materialize(value, score)
        compile_and_first = time.perf_counter() - start
        for _ in range(args.warmups):
            _materialize(*compiled(tensors["theta"]))
        timings: list[float] = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            value, score = compiled(tensors["theta"])
            _materialize(value, score)
            timings.append(time.perf_counter() - start)
    return _finish_result(
        args=args,
        physical_gpus=physical_gpus,
        logical_gpus=logical_gpus,
        mode="compiled-value-score",
        outputs=(value, score),
        timings=timings,
        compile_and_first_call_seconds=compile_and_first,
        compiled_unit="batched_value_score",
    )


def _run_scalar_compiled_value_loop(
    args: argparse.Namespace,
    tensors: dict[str, tf.Tensor],
    physical_gpus: list[str],
    logical_gpus: list[str],
) -> dict[str, Any]:
    @tf.function(jit_compile=True, reduce_retracing=True)
    def compiled_scalar(
        theta: tf.Tensor,
        initial_particles: tf.Tensor,
        pre_flow_particles: tf.Tensor,
        fixed_resampling_mask: tf.Tensor,
    ) -> tf.Tensor:
        scalar_tensors = {
            "theta": theta,
            "observations": tensors["observations"],
            "initial_particles": initial_particles,
            "pre_flow_particles": pre_flow_particles,
            "fixed_resampling_mask": fixed_resampling_mask,
        }
        return _value_from_theta(theta, scalar_tensors, args).log_likelihood

    def run_loop() -> tf.Tensor:
        values = []
        for row in range(args.batch_size):
            values.append(
                compiled_scalar(
                    tensors["theta"][row : row + 1],
                    tensors["initial_particles"][row : row + 1],
                    tensors["pre_flow_particles"][row : row + 1],
                    tensors["fixed_resampling_mask"][row : row + 1],
                )[0]
            )
        return tf.stack(values, axis=0)

    with tf.device(args.device):
        start = time.perf_counter()
        value = run_loop()
        _materialize(value)
        compile_and_first = time.perf_counter() - start
        for _ in range(args.warmups):
            _materialize(run_loop())
        timings: list[float] = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            value = run_loop()
            _materialize(value)
            timings.append(time.perf_counter() - start)
    return _finish_result(
        args=args,
        physical_gpus=physical_gpus,
        logical_gpus=logical_gpus,
        mode="scalar-compiled-value-loop",
        outputs=(value,),
        timings=timings,
        compile_and_first_call_seconds=compile_and_first,
        compiled_unit="one_scalar_value_row",
        extra={
            "compiler_extra": {
                "python_loop_over_rows_in_benchmark_harness": True,
                "scalar_loop_is_comparator_not_production_path": True,
            }
        },
    )


def _run_parity(
    args: argparse.Namespace,
    tensors: dict[str, tf.Tensor],
    physical_gpus: list[str],
    logical_gpus: list[str],
) -> dict[str, Any]:
    batched = _run_compiled_value(args, tensors, physical_gpus, logical_gpus)
    scalar = _run_scalar_compiled_value_loop(args, tensors, physical_gpus, logical_gpus)
    batched_value = np.asarray(batched["value_vector_full"], dtype=np.float64)
    scalar_value = np.asarray(scalar["value_vector_full"], dtype=np.float64)
    max_abs_delta = float(np.max(np.abs(batched_value - scalar_value)))
    max_rel_delta = float(
        np.max(np.abs(batched_value - scalar_value) / np.maximum(1.0, np.abs(scalar_value)))
    )
    passed = max_abs_delta <= args.value_atol or max_rel_delta <= args.value_rtol
    result = _base_result(args, physical_gpus, logical_gpus)
    result.update(
        {
            "mode": "parity",
            "compiler": {
                "tf_function": True,
                "jit_compile": True,
                "compiled_unit": "batched_value_and_scalar_value_loop",
                "warm_calls_exclude_compile": True,
            },
            "batched_warm_call_summary": batched["warm_call_summary"],
            "scalar_loop_warm_call_summary": scalar["warm_call_summary"],
            "value_max_abs_delta": max_abs_delta,
            "value_max_rel_delta": max_rel_delta,
            "value_atol": args.value_atol,
            "value_rtol": args.value_rtol,
            "parity_passed": passed,
            "finite_outputs": bool(batched["finite_outputs"] and scalar["finite_outputs"]),
            "value_vector_preview": _preview_vector(batched_value),
            "scalar_value_vector_preview": _preview_vector(scalar_value),
        }
    )
    if not passed:
        raise RuntimeError(
            f"compiled batched/scalar value parity failed: abs={max_abs_delta}, rel={max_rel_delta}"
        )
    return result


def _preview_vector(values: np.ndarray, *, limit: int = 8) -> dict[str, Any]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size <= 2 * limit:
        preview = [float(value) for value in flat]
    else:
        preview = (
            [float(value) for value in flat[:limit]]
            + ["..."]
            + [float(value) for value in flat[-limit:]]
        )
    return {
        "size": int(flat.size),
        "values": preview,
    }


def _write_markdown(path: Path, result: dict[str, Any], json_path: Path) -> None:
    summary = result.get("warm_call_summary") or result.get("batched_warm_call_summary")
    lines = [
        f"# Experimental Batched LEDH-PFPF-OT Benchmark: {result['mode']}",
        "",
        f"Authoritative JSON artifact: `{json_path}`.",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Mode | `{result['mode']}` |",
        f"| Device arg | `{result['device_arg']}` |",
        f"| Device scope | `{result['device_scope']}` |",
        f"| CUDA_VISIBLE_DEVICES | `{result['cuda_visible_devices']}` |",
        f"| TensorFlow | `{result['tensorflow']}` |",
        f"| Shape | `{result['shape']}` |",
        f"| Transport | `{result['transport']}` |",
        f"| Compiler | `{result.get('compiler')}` |",
        f"| Timing summary | `{summary}` |",
        f"| Finite outputs | `{result.get('finite_outputs')}` |",
        "",
        "Non-claims: no production/default readiness, no classical particle-filter score "
        "correctness, no posterior validity, and no HMC/NeuTra readiness.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    raise RuntimeError("ARCHIVAL_WRONG_TRANSPORT_CHUNK_POLICY: this route is preserved only as provenance and cannot emit new evidence")
    args = _parse_args()
    physical_gpus, logical_gpus = _configure_gpus()
    fixture = _stable_fixture(args)
    tensors = _to_tensors(fixture)

    if args.mode == "compiled-value":
        result = _run_compiled_value(args, tensors, physical_gpus, logical_gpus)
    elif args.mode == "compiled-value-score":
        result = _run_compiled_value_score(args, tensors, physical_gpus, logical_gpus)
    elif args.mode == "scalar-compiled-value-loop":
        result = _run_scalar_compiled_value_loop(args, tensors, physical_gpus, logical_gpus)
    else:
        result = _run_parity(args, tensors, physical_gpus, logical_gpus)

    result.pop("value_vector_full", None)

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
