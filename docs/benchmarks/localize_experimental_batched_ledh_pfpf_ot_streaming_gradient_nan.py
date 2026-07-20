"""Localize non-finite gradients in streaming FilterFlow transport.

This is a Phase 4 diagnostic artifact for the experimental streaming
LEDH-PFPF-OT path. It is not a benchmark and does not establish HMC readiness.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Callable


_PRE_PARSER = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
_PRE_PARSER.add_argument("--device-scope", choices=("cpu", "visible"), default="cpu")
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

from experiments.dpf_implementation.tf_tfp.resampling import annealed_transport_tf as at


NONCLAIMS = (
    "NaN-localization diagnostic only",
    "dense transport is a tiny reference, not a scalable implementation",
    "no HMC readiness claim",
    "no posterior validity claim",
    "no production/default readiness claim",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-particles", type=int, default=8)
    parser.add_argument("--state-dim", type=int, default=2)
    parser.add_argument("--row-chunk-size", type=int, default=32)
    parser.add_argument("--col-chunk-size", type=int, default=32)
    parser.add_argument("--sinkhorn-iterations", type=int, default=3)
    parser.add_argument("--sinkhorn-epsilon", type=float, default=0.5)
    parser.add_argument("--annealed-scaling", type=float, default=0.9)
    parser.add_argument("--annealed-convergence-threshold", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--no-jit-compile", action="store_true")
    parser.add_argument("--device", default="/CPU:0")
    parser.add_argument("--device-scope", choices=("cpu", "visible"), default=_PRE_ARGS.device_scope)
    parser.add_argument("--cuda-visible-devices", default=_PRE_ARGS.cuda_visible_devices)
    parser.add_argument("--expect-device-kind", choices=("any", "cpu", "gpu"), default="cpu")
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output", default=None)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.num_particles <= 1:
        raise ValueError("num_particles must be greater than one")
    if args.state_dim <= 0:
        raise ValueError("state_dim must be positive")
    if args.row_chunk_size <= 0 or args.col_chunk_size <= 0:
        raise ValueError("chunk sizes must be positive")
    if args.sinkhorn_iterations <= 0:
        raise ValueError("sinkhorn_iterations must be positive")
    return args


def _configure_dtype(name: str) -> tf.DType:
    dtype = tf.float64 if name == "float64" else tf.float32
    at.DTYPE = dtype
    return dtype


def _fixture(args: argparse.Namespace, dtype: tf.DType) -> dict[str, tf.Tensor]:
    rng = np.random.default_rng(args.seed)
    batch_size = args.batch_size
    num_particles = args.num_particles
    state_dim = args.state_dim
    state_grid = np.linspace(-0.5, 0.5, state_dim, dtype=np.float64)
    particle_grid = np.linspace(-1.0, 1.0, num_particles, dtype=np.float64)
    x = (
        0.08 * rng.standard_normal((batch_size, num_particles, state_dim))
        + 0.03 * state_grid[None, None, :]
        + 0.01 * particle_grid[None, :, None]
    )
    raw_logw = 0.15 * rng.standard_normal((batch_size, num_particles))
    logw = raw_logw - np.logaddexp.reduce(raw_logw, axis=1)[:, None]
    particles = x + 0.02 * np.cos(particle_grid[None, :, None] + state_grid[None, None, :])
    scale = np.std(x, axis=1).max(axis=1)
    scale = np.where(scale == 0.0, 1.0, scale) * np.sqrt(state_dim)
    scaled_x = x / scale[:, None, None]
    return {
        "x": tf.constant(x, dtype=dtype),
        "scaled_x": tf.constant(scaled_x, dtype=dtype),
        "particles": tf.constant(particles, dtype=dtype),
        "logw": tf.constant(logw, dtype=dtype),
    }


def _summary(name: str, tensor: tf.Tensor) -> dict[str, Any]:
    array = tensor.numpy()
    finite = np.isfinite(array)
    return {
        "name": name,
        "shape": list(array.shape),
        "finite": bool(finite.all()),
        "num_nonfinite": int(array.size - finite.sum()),
        "min": float(np.nanmin(array)) if array.size else 0.0,
        "max": float(np.nanmax(array)) if array.size else 0.0,
        "l2_norm": float(np.linalg.norm(np.nan_to_num(array))),
        "preview": [float(v) for v in array.reshape(-1)[:8]],
    }


def _tensor_finite(*tensors: tf.Tensor) -> bool:
    return all(bool(tf.reduce_all(tf.math.is_finite(tensor)).numpy()) for tensor in tensors)


def _device_check(outputs: tuple[tf.Tensor, ...], expect_device_kind: str) -> list[str]:
    devices = [tensor.device for tensor in outputs]
    if expect_device_kind == "gpu":
        if not all("GPU" in device.upper() for device in devices):
            raise RuntimeError(f"expected GPU outputs, got {devices}")
    elif expect_device_kind == "cpu":
        if not all("CPU" in device.upper() for device in devices):
            raise RuntimeError(f"expected CPU outputs, got {devices}")
    return devices


def _run_probe(
    name: str,
    fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tuple[tf.Tensor, ...]],
    scaled_x: tf.Tensor,
    particles: tf.Tensor,
    logw: tf.Tensor,
    *,
    jit_compile: bool,
    expect_device_kind: str,
) -> dict[str, Any]:
    @tf.function(jit_compile=jit_compile, reduce_retracing=True)
    def compiled(
        local_scaled_x: tf.Tensor,
        local_particles: tf.Tensor,
        local_logw: tf.Tensor,
    ) -> tuple[tuple[tf.Tensor, ...], tf.Tensor, tf.Tensor, tf.Tensor]:
        with tf.GradientTape() as tape:
            tape.watch([local_scaled_x, local_particles, local_logw])
            outputs = fn(local_scaled_x, local_particles, local_logw)
            objective = tf.add_n([tf.reduce_sum(output) for output in outputs])
        grads = tape.gradient(
            objective,
            [local_scaled_x, local_particles, local_logw],
            unconnected_gradients=tf.UnconnectedGradients.ZERO,
        )
        return outputs, grads[0], grads[1], grads[2]

    try:
        outputs, grad_x, grad_particles, grad_logw = compiled(scaled_x, particles, logw)
        material = tuple(outputs) + (grad_x, grad_particles, grad_logw)
        for tensor in material:
            tensor.numpy()
        devices = _device_check(material, expect_device_kind)
        output_summaries = [
            _summary(f"output_{index}", output)
            for index, output in enumerate(outputs)
        ]
        grad_summaries = [
            _summary("grad_scaled_x", grad_x),
            _summary("grad_particles", grad_particles),
            _summary("grad_logw", grad_logw),
        ]
        return {
            "name": name,
            "status": "ok",
            "output_finite": _tensor_finite(*outputs),
            "gradient_finite": _tensor_finite(grad_x, grad_particles, grad_logw),
            "outputs": output_summaries,
            "gradients": grad_summaries,
            "output_devices": devices,
        }
    except Exception as exc:  # noqa: BLE001 - artifact should capture diagnostic failures.
        return {
            "name": name,
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "output_finite": False,
            "gradient_finite": False,
        }


def _write_markdown(path: Path, result: dict[str, Any], json_path: Path) -> None:
    lines = [
        "# Streaming Transport Gradient NaN Localization",
        "",
        f"- JSON artifact: `{json_path}`",
        f"- Overall passed: `{result['overall_passed']}`",
        f"- JIT compile: `{result['jit_compile']}`",
        "",
        "## Probes",
        "",
        "| probe | status | outputs finite | gradients finite | first non-finite gradient |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for probe in result["probes"]:
        first_bad = "none"
        for grad in probe.get("gradients", []):
            if not grad["finite"]:
                first_bad = grad["name"]
                break
        lines.append(
            f"| {probe['name']} | {probe['status']} | {probe['output_finite']} | "
            f"{probe['gradient_finite']} | `{first_bad}` |"
        )
    lines.extend(["", "## Nonclaims", ""])
    lines.extend(f"- {claim}" for claim in result["nonclaims"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    raise RuntimeError("ARCHIVAL_WRONG_TRANSPORT_CHUNK_POLICY: this route is preserved only as provenance and cannot emit new evidence")
    args = _parse_args()
    dtype = _configure_dtype(args.dtype)
    physical_gpus = [str(device) for device in tf.config.list_physical_devices("GPU")]
    logical_gpus = [str(device) for device in tf.config.list_logical_devices("GPU")]
    fixture = _fixture(args, dtype)
    eps = tf.constant(args.sinkhorn_epsilon, dtype=dtype)
    scaling = tf.constant(args.annealed_scaling, dtype=dtype)
    threshold = tf.constant(args.annealed_convergence_threshold, dtype=dtype)
    max_iter = tf.constant(args.sinkhorn_iterations, dtype=tf.int32)
    n = tf.constant(args.num_particles, dtype=tf.int32)
    float_n = tf.cast(n, dtype)
    uniform_log = -tf.math.log(float_n) * tf.ones_like(fixture["logw"])
    row_chunk_size = args.row_chunk_size
    col_chunk_size = args.col_chunk_size

    def dense_transport(
        scaled_x: tf.Tensor,
        particles: tf.Tensor,
        logw: tf.Tensor,
    ) -> tuple[tf.Tensor, ...]:
        matrix, iterations = at._filterflow_exact_transport_matrix(
            scaled_x,
            logw,
            eps,
            scaling,
            threshold,
            max_iter,
            n,
        )
        carried = tf.linalg.matmul(matrix, particles)
        return carried, tf.cast(iterations, dtype)

    def streaming_softmin(
        scaled_x: tf.Tensor,
        _particles: tf.Tensor,
        logw: tf.Tensor,
    ) -> tuple[tf.Tensor, ...]:
        value = at._filterflow_streaming_softmin(
            eps,
            scaled_x,
            tf.stop_gradient(scaled_x),
            logw,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )
        return (value,)

    def streaming_potentials(
        scaled_x: tf.Tensor,
        _particles: tf.Tensor,
        logw: tf.Tensor,
    ) -> tuple[tf.Tensor, ...]:
        alpha, beta, alpha_x, beta_y, iterations = at._filterflow_streaming_sinkhorn_potentials(
            logw,
            scaled_x,
            uniform_log,
            scaled_x,
            eps,
            scaling,
            threshold,
            max_iter,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )
        return alpha, beta, alpha_x, beta_y, tf.cast(iterations, dtype)

    def streaming_column_normalizer(
        scaled_x: tf.Tensor,
        _particles: tf.Tensor,
        logw: tf.Tensor,
    ) -> tuple[tf.Tensor, ...]:
        alpha, beta, _, _, _ = at._filterflow_streaming_sinkhorn_potentials(
            logw,
            scaled_x,
            uniform_log,
            scaled_x,
            eps,
            scaling,
            threshold,
            max_iter,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )
        value = at._filterflow_streaming_column_log_normalizer(
            scaled_x,
            alpha,
            beta,
            eps,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )
        return (value,)

    def streaming_transport_from_potentials(
        scaled_x: tf.Tensor,
        particles: tf.Tensor,
        logw: tf.Tensor,
    ) -> tuple[tf.Tensor, ...]:
        alpha, beta, _, _, _ = at._filterflow_streaming_sinkhorn_potentials(
            logw,
            scaled_x,
            uniform_log,
            scaled_x,
            eps,
            scaling,
            threshold,
            max_iter,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )
        transported, row_residual = at._filterflow_streaming_transport_from_potentials(
            scaled_x,
            particles,
            alpha,
            beta,
            eps,
            logw,
            float_n,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )
        return transported, row_residual[None]

    def streaming_transport(
        scaled_x: tf.Tensor,
        particles: tf.Tensor,
        logw: tf.Tensor,
    ) -> tuple[tf.Tensor, ...]:
        transported, iterations, row_residual, column_residual = at._filterflow_streaming_transport(
            scaled_x,
            particles,
            logw,
            eps,
            scaling,
            threshold,
            max_iter,
            n,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )
        return (
            transported,
            tf.cast(iterations, dtype)[None],
            row_residual[None],
            column_residual[None],
        )

    probes = []
    with tf.device(args.device):
        for name, fn in (
            ("dense_transport_reference", dense_transport),
            ("streaming_softmin", streaming_softmin),
            ("streaming_sinkhorn_potentials", streaming_potentials),
            ("streaming_column_log_normalizer", streaming_column_normalizer),
            ("streaming_transport_from_potentials", streaming_transport_from_potentials),
            ("streaming_transport", streaming_transport),
        ):
            probes.append(
                _run_probe(
                    name,
                    fn,
                    fixture["scaled_x"],
                    fixture["particles"],
                    fixture["logw"],
                    jit_compile=not args.no_jit_compile,
                    expect_device_kind=args.expect_device_kind,
                )
            )

    overall_passed = all(
        probe["status"] == "ok"
        and probe["output_finite"]
        and probe["gradient_finite"]
        for probe in probes
    )
    result = {
        "timestamp_utc": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "host": platform.node(),
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_gpus": physical_gpus,
        "logical_gpus": logical_gpus,
        "device": args.device,
        "device_scope": args.device_scope,
        "dtype": args.dtype,
        "jit_compile": not args.no_jit_compile,
        "shape": {
            "batch_size": args.batch_size,
            "num_particles": args.num_particles,
            "state_dim": args.state_dim,
        },
        "transport": {
            "sinkhorn_iterations": args.sinkhorn_iterations,
            "sinkhorn_epsilon": args.sinkhorn_epsilon,
            "annealed_scaling": args.annealed_scaling,
            "annealed_convergence_threshold": args.annealed_convergence_threshold,
            "row_chunk_size": args.row_chunk_size,
            "col_chunk_size": args.col_chunk_size,
        },
        "probes": probes,
        "overall_passed": bool(overall_passed),
        "nonclaims": list(NONCLAIMS),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_output is not None:
        markdown_output = Path(args.markdown_output)
        markdown_output.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(markdown_output, result, output)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["overall_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
