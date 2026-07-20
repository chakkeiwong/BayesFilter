"""Phase 1 baseline transport diagnostics for scalable OT work.

This script exercises the current TensorFlow dense/streaming
FilterFlow-style annealed transport baseline on deterministic fixtures.  It is
not a candidate benchmark and does not make speedup claims.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


_PRE_PARSER = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
_PRE_PARSER.add_argument("--device-scope", choices=("cpu", "visible"), default="cpu")
_PRE_PARSER.add_argument("--cuda-visible-devices", default=None)
_PRE_ARGS, _UNKNOWN = _PRE_PARSER.parse_known_args()
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

from experiments.dpf_implementation.tf_tfp.resampling import annealed_transport_tf
from experiments.dpf_implementation.tf_tfp.resampling.annealed_transport_tf import (
    annealed_transport_resample_tf,
)


DTYPE = tf.float64
annealed_transport_tf.DTYPE = DTYPE

NONCLAIMS = (
    "Phase 1 baseline transport diagnostics only",
    "no scalable candidate correctness claim",
    "no speedup claim",
    "no posterior validity claim",
    "no production default change",
    "no statistical ranking",
    "no GPU performance claim",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--epsilon", type=float, default=0.5)
    parser.add_argument("--scaling", type=float, default=0.9)
    parser.add_argument("--convergence-threshold", type=float, default=1.0e-3)
    parser.add_argument("--max-iterations", type=int, default=12)
    parser.add_argument("--row-chunk-size", type=int, default=4)
    parser.add_argument("--col-chunk-size", type=int, default=4)
    parser.add_argument("--device", default="/CPU:0")
    parser.add_argument("--device-scope", choices=("cpu", "visible"), default=_PRE_ARGS.device_scope)
    parser.add_argument("--cuda-visible-devices", default=_PRE_ARGS.cuda_visible_devices)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output", default=None)
    args = parser.parse_args()
    if args.epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if not 0.0 < args.scaling <= 1.0:
        raise ValueError("scaling must be in (0, 1]")
    if args.convergence_threshold <= 0.0:
        raise ValueError("convergence_threshold must be positive")
    if args.max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if args.row_chunk_size <= 0 or args.col_chunk_size <= 0:
        raise ValueError("chunk sizes must be positive")
    return args


def _git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return completed.stdout.strip()


def _fixture_tiny_manual() -> tuple[np.ndarray, np.ndarray]:
    particles = np.array(
        [
            [[-0.45, 0.00, 0.20], [-0.25, 0.15, -0.10], [-0.05, -0.20, 0.30],
             [0.10, 0.25, -0.25], [0.28, -0.15, 0.05], [0.50, 0.05, -0.15]]
        ],
        dtype=np.float64,
    )
    raw = np.array([[0.0, -0.3, -0.7, -1.1, -1.6, -2.0]], dtype=np.float64)
    log_weights = raw - np.log(np.sum(np.exp(raw), axis=1, keepdims=True))
    return particles, log_weights


def _fixture_small_parity() -> tuple[np.ndarray, np.ndarray]:
    batch_size, num_particles, state_dim = 2, 16, 5
    i = np.arange(num_particles, dtype=np.float64)
    d = np.arange(state_dim, dtype=np.float64)
    b = np.arange(batch_size, dtype=np.float64)
    particles = (
        0.20 * np.sin(0.37 * i[None, :, None] + 0.19 * d[None, None, :])
        + 0.11 * np.cos(0.23 * i[None, :, None] * (d[None, None, :] + 1.0))
        + 0.015 * b[:, None, None]
    )
    raw = -0.06 * i[None, :] + 0.08 * np.sin(0.4 * i[None, :] + 0.3 * b[:, None])
    log_weights = raw - np.log(np.sum(np.exp(raw), axis=1, keepdims=True))
    return particles.astype(np.float64), log_weights.astype(np.float64)


def _orthonormal_basis(state_dim: int, rank: int) -> np.ndarray:
    grid = np.arange(state_dim, dtype=np.float64)[:, None]
    cols = []
    for k in range(rank):
        col = np.sin((k + 1) * 0.31 * grid[:, 0]) + np.cos((k + 2) * 0.17 * grid[:, 0])
        cols.append(col)
    matrix = np.stack(cols, axis=1)
    q, _ = np.linalg.qr(matrix)
    return q[:, :rank]


def _fixture_high_dim_low_rank() -> tuple[np.ndarray, np.ndarray]:
    num_particles, state_dim, rank = 64, 32, 4
    i = np.linspace(-1.0, 1.0, num_particles, dtype=np.float64)
    coeffs = np.stack(
        [
            i,
            i * i - np.mean(i * i),
            np.sin(np.pi * i),
            np.cos(0.5 * np.pi * i) - np.mean(np.cos(0.5 * np.pi * i)),
        ],
        axis=1,
    )
    basis = _orthonormal_basis(state_dim, rank)
    particles = coeffs @ basis.T
    particles = particles[None, :, :]
    raw = -0.04 * np.arange(num_particles, dtype=np.float64)[None, :]
    log_weights = raw - np.log(np.sum(np.exp(raw), axis=1, keepdims=True))
    return particles.astype(np.float64), log_weights.astype(np.float64)


def _fixture_high_dim_locality() -> tuple[np.ndarray, np.ndarray]:
    num_particles, state_dim, clusters = 64, 32, 4
    cluster_ids = np.arange(num_particles) // (num_particles // clusters)
    within = np.arange(num_particles) % (num_particles // clusters)
    centers = _orthonormal_basis(state_dim, clusters).T
    centers = 0.75 * centers[cluster_ids]
    offsets = np.zeros((num_particles, state_dim), dtype=np.float64)
    offsets[:, 0] = 0.015 * within
    offsets[:, 1] = 0.010 * np.sin(0.7 * within)
    offsets[:, 2] = 0.010 * np.cos(0.5 * within)
    particles = (centers + offsets)[None, :, :]
    raw = -0.02 * np.arange(num_particles, dtype=np.float64)[None, :]
    raw += 0.03 * np.sin(0.3 * np.arange(num_particles, dtype=np.float64)[None, :])
    log_weights = raw - np.log(np.sum(np.exp(raw), axis=1, keepdims=True))
    return particles.astype(np.float64), log_weights.astype(np.float64)


def _fixtures() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        "tiny_manual": _fixture_tiny_manual(),
        "small_parity": _fixture_small_parity(),
        "high_dim_low_rank": _fixture_high_dim_low_rank(),
        "high_dim_locality": _fixture_high_dim_locality(),
    }


def _float(value: Any) -> float:
    array = np.asarray(value)
    return float(array.reshape(-1)[0])


def _tensor_finite(value: tf.Tensor) -> bool:
    return bool(tf.reduce_all(tf.math.is_finite(value)).numpy())


def _run_transport(
    particles_np: np.ndarray,
    log_weights_np: np.ndarray,
    args: argparse.Namespace,
    *,
    plan_mode: str,
) -> tuple[dict[str, Any], tf.Tensor, tf.Tensor]:
    particles = tf.constant(particles_np, dtype=DTYPE)
    log_weights = tf.constant(log_weights_np, dtype=DTYPE)
    start = time.perf_counter()
    with tf.device(args.device):
        result = annealed_transport_resample_tf(
            particles,
            log_weights,
            epsilon=args.epsilon,
            scaling=args.scaling,
            convergence_threshold=args.convergence_threshold,
            max_iterations=args.max_iterations,
            transport_gradient_mode="raw",
            transport_plan_mode=plan_mode,
            row_chunk_size=args.row_chunk_size,
            col_chunk_size=args.col_chunk_size,
        )
    wall_time = time.perf_counter() - start
    diagnostics = dict(result.diagnostics)
    transport = tf.convert_to_tensor(result.transport_matrix, dtype=DTYPE)
    transported = tf.convert_to_tensor(result.particles, dtype=DTYPE)
    out_log_weights = tf.convert_to_tensor(result.log_weights, dtype=DTYPE)
    record: dict[str, Any] = {
        "plan_mode": plan_mode,
        "wall_time_seconds": wall_time,
        "particles_shape": transported.shape.as_list(),
        "log_weights_shape": out_log_weights.shape.as_list(),
        "transport_matrix_shape": transport.shape.as_list(),
        "finite_particles": _tensor_finite(transported),
        "finite_log_weights": _tensor_finite(out_log_weights),
        "transported_particle_norm": _float(tf.linalg.norm(transported)),
        "log_weight_normalization_residual": _float(
            tf.reduce_max(tf.abs(tf.reduce_logsumexp(out_log_weights, axis=1)))
        ),
        "diagnostics": diagnostics,
    }
    if plan_mode == "dense":
        source_weights = tf.exp(log_weights)
        num_particles = tf.cast(tf.shape(log_weights)[1], DTYPE)
        row_residual = tf.reduce_max(tf.abs(tf.reduce_sum(transport, axis=2) - 1.0))
        col_target = source_weights * num_particles
        col_residual = tf.reduce_max(tf.abs(tf.reduce_sum(transport, axis=1) - col_target))
        record.update(
            {
                "finite_transport_matrix": _tensor_finite(transport),
                "transport_mass": _float(tf.reduce_sum(transport)),
                "recomputed_row_residual": _float(row_residual),
                "recomputed_column_residual": _float(col_residual),
            }
        )
    else:
        record.update(
            {
                "transport_object": "not_materialized",
                "not_materialized_reason": "streaming_no_dense_matrix",
                "finite_transport_matrix": None,
            }
        )
    return record, transported, out_log_weights


def _compare_dense_streaming(dense_particles: tf.Tensor, streaming_particles: tf.Tensor) -> dict[str, Any]:
    diff = dense_particles - streaming_particles
    return {
        "max_abs_transported_particle_error": _float(tf.reduce_max(tf.abs(diff))),
        "rms_transported_particle_error": _float(tf.sqrt(tf.reduce_mean(tf.square(diff)))),
        "finite_difference": _tensor_finite(diff),
    }


def _build_result(args: argparse.Namespace) -> dict[str, Any]:
    fixture_records: dict[str, Any] = {}
    hard_vetoes: list[str] = []
    for name, (particles_np, log_weights_np) in _fixtures().items():
        dense_record, dense_particles, _dense_logw = _run_transport(
            particles_np,
            log_weights_np,
            args,
            plan_mode="dense",
        )
        streaming_record, streaming_particles, _streaming_logw = _run_transport(
            particles_np,
            log_weights_np,
            args,
            plan_mode="streaming",
        )
        comparison = _compare_dense_streaming(dense_particles, streaming_particles)
        fixture_records[name] = {
            "input_shape": {
                "particles": list(particles_np.shape),
                "log_weights": list(log_weights_np.shape),
            },
            "input_log_weight_normalization_residual": float(
                np.max(np.abs(np.log(np.sum(np.exp(log_weights_np), axis=1))))
            ),
            "dense": dense_record,
            "streaming": streaming_record,
            "dense_vs_streaming": comparison,
        }
        if not dense_record["finite_particles"]:
            hard_vetoes.append(f"{name}:dense_nonfinite_particles")
        if not dense_record.get("finite_transport_matrix"):
            hard_vetoes.append(f"{name}:dense_nonfinite_or_missing_transport")
        if not streaming_record["finite_particles"]:
            hard_vetoes.append(f"{name}:streaming_nonfinite_particles")
        if not comparison["finite_difference"]:
            hard_vetoes.append(f"{name}:dense_streaming_nonfinite_difference")
        if not math.isfinite(comparison["max_abs_transported_particle_error"]):
            hard_vetoes.append(f"{name}:dense_streaming_error_nonfinite")
    return {
        "status": "PASS" if not hard_vetoes else "FAIL",
        "hard_vetoes": hard_vetoes,
        "timestamp_utc": _dt.datetime.now(tz=_dt.UTC).isoformat(),
        "git_commit": _git_commit(),
        "python": sys.version,
        "platform": platform.platform(),
        "tensorflow_version": tf.__version__,
        "device_scope": args.device_scope,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": args.device,
        "dtype": "tf.float64",
        "settings": {
            "epsilon": args.epsilon,
            "scaling": args.scaling,
            "convergence_threshold": args.convergence_threshold,
            "max_iterations": args.max_iterations,
            "row_chunk_size": args.row_chunk_size,
            "col_chunk_size": args.col_chunk_size,
            "transport_gradient_mode": "raw",
        },
        "fixtures": fixture_records,
        "nonclaims": NONCLAIMS,
    }


def _write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Phase 1 Baseline Fixture Diagnostics",
        "",
        f"- Status: `{result['status']}`",
        f"- Git commit: `{result['git_commit']}`",
        f"- TensorFlow: `{result['tensorflow_version']}`",
        f"- Device scope: `{result['device_scope']}`",
        f"- CUDA_VISIBLE_DEVICES: `{result['cuda_visible_devices']}`",
        "",
        "## Fixture Summary",
        "",
        "| Fixture | Dense finite | Streaming finite | Dense row residual | Dense column residual | Max dense-streaming particle error |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, fixture in result["fixtures"].items():
        dense = fixture["dense"]
        streaming = fixture["streaming"]
        comp = fixture["dense_vs_streaming"]
        lines.append(
            "| {name} | `{dense_finite}` | `{streaming_finite}` | `{row:.6e}` | `{col:.6e}` | `{err:.6e}` |".format(
                name=name,
                dense_finite=dense["finite_particles"],
                streaming_finite=streaming["finite_particles"],
                row=dense["recomputed_row_residual"],
                col=dense["recomputed_column_residual"],
                err=comp["max_abs_transported_particle_error"],
            )
        )
    lines.extend(
        [
            "",
            "## Hard Vetoes",
            "",
            "`{}`".format(result["hard_vetoes"]),
            "",
            "## Non-Claims",
            "",
        ]
    )
    for nonclaim in result["nonclaims"]:
        lines.append(f"- {nonclaim}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    raise RuntimeError("ARCHIVAL_WRONG_TRANSPORT_CHUNK_POLICY: this route is preserved only as provenance and cannot emit new evidence")
    args = _parse_args()
    result = _build_result(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_output:
        markdown = Path(args.markdown_output)
        markdown.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(result, markdown)
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
