"""Generic transport-core benchmark for P8m.

This benchmark uses synthetic particles and normalized log weights to exercise
the existing batched TensorFlow transport core.  It deliberately avoids model
callbacks so it can profile generic transport behavior rather than SIR-specific
adapter code.
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

from experiments.dpf_implementation.tf_tfp.filters import (
    experimental_batched_ledh_pfpf_ot_tf as core_tf,
)
from experiments.dpf_implementation.tf_tfp.resampling import annealed_transport_tf
from scripts.filtering_value_gradient_benchmark_run_p8d_numeric import _git_commit


DTYPE = tf.float32
NONCLAIMS = (
    "generic synthetic transport-core benchmark only",
    "not SIR d18 or any model-specific evidence",
    "not particle-count adequacy evidence",
    "not leaderboard completion",
    "not exact likelihood correctness",
    "not DPF gradient correctness",
    "not HMC/NUTS readiness",
    "not production/default readiness",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-particles", type=int, default=64)
    parser.add_argument("--state-dim", type=int, default=3)
    parser.add_argument("--seed", type=int, default=90210)
    parser.add_argument(
        "--transport-policy",
        choices=("active-all", "no-resampling"),
        default="active-all",
    )
    parser.add_argument("--sinkhorn-iterations", type=int, default=10)
    parser.add_argument("--sinkhorn-epsilon", type=float, default=1.0)
    parser.add_argument("--annealed-scaling", type=float, default=0.9)
    parser.add_argument("--annealed-convergence-threshold", type=float, default=1.0e-3)
    parser.add_argument("--row-chunk-size", type=int, default=1024)
    parser.add_argument("--col-chunk-size", type=int, default=1024)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
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
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.num_particles <= 1:
        raise ValueError("num-particles must be greater than one")
    if args.state_dim <= 0:
        raise ValueError("state-dim must be positive")
    if args.sinkhorn_iterations <= 0:
        raise ValueError("sinkhorn-iterations must be positive")
    if args.row_chunk_size <= 0 or args.col_chunk_size <= 0:
        raise ValueError("chunk sizes must be positive")
    if args.warmups < 0 or args.repeats <= 0:
        raise ValueError("warmups must be nonnegative and repeats must be positive")
    return args


def _configure_precision(args: argparse.Namespace) -> dict[str, Any]:
    global DTYPE
    DTYPE = tf.float64 if args.dtype == "float64" else tf.float32
    core_tf.DTYPE = DTYPE
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


def _build_synthetic_inputs(args: argparse.Namespace) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    seed = tf.constant([int(args.seed) % 2147483647, 1729], dtype=tf.int32)
    particles = tf.random.stateless_normal(
        [args.batch_size, args.num_particles, args.state_dim],
        seed=seed,
        dtype=DTYPE,
    )
    raw_logits = tf.random.stateless_normal(
        [args.batch_size, args.num_particles],
        seed=seed + tf.constant([0, 1], dtype=tf.int32),
        dtype=DTYPE,
    )
    log_weights = raw_logits - tf.reduce_logsumexp(raw_logits, axis=1, keepdims=True)
    if args.transport_policy == "active-all":
        mask = tf.ones([args.batch_size], dtype=tf.bool)
    else:
        mask = tf.zeros([args.batch_size], dtype=tf.bool)
    return particles, log_weights, mask


def _write_markdown(path: Path, result: dict[str, Any], json_path: Path) -> None:
    lines = [
        "# P8m Generic Transport-Core Benchmark",
        "",
        f"- JSON artifact: `{json_path}`",
        f"- Shape: `{result['shape']}`",
        f"- Output devices: `{result['output_devices']}`",
        f"- Precision: `{result['precision']}`",
        f"- Transport: `{result['transport']}`",
        f"- Compile plus first call seconds: `{result['compile_and_first_call_seconds']}`",
        f"- Warm-call timing summary seconds: `{result['warm_call_timing_summary_seconds']}`",
        f"- Finite output: `{result['finite_output']}`",
        f"- Max row residual: `{result['max_row_residual']}`",
        f"- Max column residual: `{result['max_column_residual']}`",
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
    particles, log_weights, mask = _build_synthetic_inputs(args)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def compiled_outputs() -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        result = core_tf.batched_annealed_transport_core_tf(
            particles,
            log_weights,
            mask,
            epsilon=args.sinkhorn_epsilon,
            scaling=args.annealed_scaling,
            convergence_threshold=args.annealed_convergence_threshold,
            max_iterations=args.sinkhorn_iterations,
            transport_gradient_mode="raw",
            transport_plan_mode="streaming",
            row_chunk_size=args.row_chunk_size,
            col_chunk_size=args.col_chunk_size,
        )
        return (
            result.particles,
            result.log_weights,
            result.max_row_residual,
            result.max_column_residual,
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

    transported_particles, out_log_weights, max_row_residual, max_column_residual = outputs
    output_devices = _validate_device((transported_particles, out_log_weights), args.expect_device_kind)
    finite_output = bool(
        tf.reduce_all(tf.math.is_finite(transported_particles)).numpy()
        and tf.reduce_all(tf.math.is_finite(out_log_weights)).numpy()
        and tf.reduce_all(tf.math.is_finite(max_row_residual)).numpy()
        and tf.reduce_all(tf.math.is_finite(max_column_residual)).numpy()
    )
    log_weight_normalization_residual = tf.reduce_max(
        tf.abs(tf.reduce_logsumexp(out_log_weights, axis=1))
    )
    warm_summary = _summary(timings)
    result: dict[str, Any] = {
        "schema_version": "filter_bench.p8m_transport_core.v1",
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
            "model_family": "synthetic_transport_core",
            "batch_size": args.batch_size,
            "num_particles": args.num_particles,
            "state_dim": args.state_dim,
        },
        "seed": args.seed,
        "transport_policy": args.transport_policy,
        "transport": {
            "plan_mode": "streaming",
            "gradient_mode": "raw",
            "row_chunk_size": args.row_chunk_size,
            "col_chunk_size": args.col_chunk_size,
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
        "gpu_memory_info_before": memory_before,
        "gpu_memory_info_after": memory_after,
        "output_devices": output_devices,
        "output_shapes": {
            "particles": list(transported_particles.numpy().shape),
            "log_weights": list(out_log_weights.numpy().shape),
        },
        "finite_output": finite_output,
        "max_row_residual": float(max_row_residual.numpy()),
        "max_column_residual": float(max_column_residual.numpy()),
        "log_weight_normalization_residual": float(log_weight_normalization_residual.numpy()),
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
