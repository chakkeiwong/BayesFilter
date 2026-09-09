#!/usr/bin/env python3
"""Benchmark strict versus cached eigensystem q=20 target evaluation.

This is a diagnostic benchmark only.  It compares identical batch-native
TensorFlow target calls, records compile and steady-state time, and checks
value/score/status parity.  It does not train a transport or make an HMC or
posterior claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BACKENDS = (
    "tensorflow_eigh_strict",
    "tensorflow_eigh_strict_cached",
    "tensorflow_eigh_strict_factor_cached",
)
VALUE_PARITY_RTOL = 1.0e-10
SCORE_PARITY_RTOL = 1.0e-9
PARITY_ATOL = 1.0e-10

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _json_value(value: Any) -> Any:
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        return _json_value(value.tolist())
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _count_eigh_ops(concrete: Any) -> int:
    if hasattr(concrete, "get_concrete_function"):
        concrete = concrete.get_concrete_function()
    graph_def = concrete.graph.as_graph_def()
    return sum(
        node.op == "SelfAdjointEigV2"
        for function in graph_def.library.function
        for node in function.node_def
    )


def _allocator_info(tf: Any, physical: tuple[Any, ...]) -> dict[str, int] | None:
    if not physical:
        return None
    try:
        info = tf.config.experimental.get_memory_info("GPU:0")
    except (AttributeError, RuntimeError, ValueError):
        return None
    return {str(key): int(value) for key, value in info.items()}


def _git_metadata() -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(
                ("git", *args), cwd=ROOT, text=True, stderr=subprocess.STDOUT
            ).strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            return f"unavailable:{type(exc).__name__}"

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "worktree_dirty": bool(status),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def _source_sha256(relative_path: str) -> str:
    path = ROOT / relative_path
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cpu", action="store_true", help="hide GPUs before TensorFlow import")
    parser.add_argument("--no-jit", action="store_true", help="diagnostic non-XLA exception")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--row-offset-scale",
        type=float,
        default=0.0,
        help="deterministic per-row perturbation scale for a varied fixture",
    )
    parser.add_argument(
        "--candidate-backend",
        action="append",
        choices=BACKENDS[1:],
        dest="candidate_backends",
        help=(
            "candidate backend(s) required for the exit status; may be repeated. "
            "The default evaluates every diagnostic candidate."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    started_campaign = time.perf_counter()
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        # The promotion plan binds GPU evidence to the first physical device.
        # Reject an externally supplied multi-device selection rather than
        # allowing device placement to become an unrecorded variable.
        requested_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if requested_devices in (None, ""):
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        elif requested_devices != "0":
            raise ValueError(
                "factor-route promotion benchmark requires CUDA_VISIBLE_DEVICES=0"
            )
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "").lower() != "true":
        raise RuntimeError("TF_FORCE_GPU_ALLOW_GROWTH=true is required")

    import tensorflow as tf

    from bayesfilter.runtime.gpu_memory_policy import (
        configure_tensorflow_gpu_memory_growth,
    )

    jit_compile = not args.no_jit
    if args.batch_size <= 0 or args.repeats <= 0:
        raise ValueError("batch size and repeats must be positive")
    if not (float(args.row_offset_scale) >= 0.0):
        raise ValueError("row offset scale must be nonnegative")
    candidate_backends = tuple(args.candidate_backends or BACKENDS[1:])
    memory_policy = None
    if args.cpu:
        physical = tuple(tf.config.list_physical_devices("GPU"))
    else:
        memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
        physical = tuple(tf.config.list_physical_devices("GPU"))
        tf.config.experimental.enable_tensor_float_32_execution(True)
    if not physical and not args.cpu:
        raise RuntimeError("no GPU is visible; rerun with --cpu for a diagnostic CPU benchmark")
    logical = tuple(tf.config.list_logical_devices("GPU"))

    # Import the target only after allocator policy configuration.  Its module
    # constants are TensorFlow tensors and can otherwise initialize a GPU.
    from bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf import (
        batch_native_complexity_posterior_target,
    )

    center = tf.constant([0.35, -0.08, 0.65, 0.05], tf.float64)
    row = tf.cast(tf.range(args.batch_size), tf.float64)
    offsets = tf.stack(
        (
            tf.sin(0.71 * row),
            tf.cos(0.53 * row),
            tf.sin(0.37 * row + 0.2),
            tf.cos(0.29 * row - 0.1),
        ),
        axis=1,
    ) * tf.constant(float(args.row_offset_scale), tf.float64)
    theta = center[tf.newaxis, :] + offsets
    records: dict[str, Any] = {}
    tensors: dict[str, tuple[tf.Tensor, tf.Tensor, dict[str, Any]]] = {}
    evaluated_backends = (BACKENDS[0],) + tuple(candidate_backends)
    for backend in evaluated_backends:
        target = batch_native_complexity_posterior_target(
            20,
            jit_compile=jit_compile,
            principal_sqrt_backend=backend,
        )
        started = time.perf_counter()
        likelihood, likelihood_score, prior, prior_score, status = (
            target.batch_prior_likelihood_value_score_status(theta)
        )
        likelihood = tf.convert_to_tensor(likelihood, tf.float64)
        score = tf.convert_to_tensor(likelihood_score, tf.float64)
        value = likelihood + tf.convert_to_tensor(prior, tf.float64)
        score = score + tf.convert_to_tensor(prior_score, tf.float64)
        compile_seconds = time.perf_counter() - started
        steady_seconds: list[float] = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            likelihood, likelihood_score, prior, prior_score, status = (
                target.batch_prior_likelihood_value_score_status(theta)
            )
            likelihood = tf.convert_to_tensor(likelihood, tf.float64)
            score = tf.convert_to_tensor(likelihood_score, tf.float64)
            value = likelihood + tf.convert_to_tensor(prior, tf.float64)
            score = score + tf.convert_to_tensor(prior_score, tf.float64)
            steady_seconds.append(time.perf_counter() - started)
        compiled = target._compiled_component_batches[args.batch_size]
        status_payload = {
            key: _json_value(value)
            for key, value in status.items()
            if key in {
                "status_code",
                "valid_pre_regularized_score",
                "principal_sqrt_target_row_class_code",
                "principal_sqrt_target_valid_count",
                "principal_sqrt_target_classified_invalid_count",
                "min_placement_eigen_gap",
                "min_innovation_eigen_gap",
                "min_placement_eigenvalue",
                "min_innovation_eigenvalue",
            }
        }
        records[backend] = {
            "compile_seconds": compile_seconds,
            "steady_seconds": steady_seconds,
            "steady_mean_seconds": sum(steady_seconds) / len(steady_seconds),
            "eigh_ops_in_graph_library": _count_eigh_ops(compiled),
            "trace_count": int(compiled.experimental_get_tracing_count()),
            "likelihood_finite": bool(tf.reduce_all(tf.math.is_finite(likelihood)).numpy()),
            "value_finite": bool(tf.reduce_all(tf.math.is_finite(value)).numpy()),
            "score_finite": bool(tf.reduce_all(tf.math.is_finite(score)).numpy()),
            "value_max_abs": float(tf.reduce_max(tf.abs(value)).numpy()),
            "score_max_abs": float(tf.reduce_max(tf.abs(score)).numpy()),
            "allocator_bytes_after_backend": _allocator_info(tf, physical),
            "batch_native": bool(target.supports_retained_flat_batch),
            "target_scope": target.target_scope,
            "status": status_payload,
        }
        tensors[backend] = (value, score, status)

    strict_value, strict_score, strict_status = tensors[BACKENDS[0]]
    strict_record = records[BACKENDS[0]]
    strict_value_scale = max(float(strict_record["value_max_abs"]), 1.0e-300)
    strict_score_scale = max(float(strict_record["score_max_abs"]), 1.0e-300)

    def compare(candidate: str) -> dict[str, Any]:
        candidate_value, candidate_score, candidate_status = tensors[candidate]
        value_difference = tf.abs(candidate_value - strict_value)
        score_difference = tf.abs(candidate_score - strict_score)
        value_tolerance = tf.constant(PARITY_ATOL, tf.float64) + tf.constant(
            VALUE_PARITY_RTOL, tf.float64
        ) * tf.abs(strict_value)
        score_tolerance = tf.constant(PARITY_ATOL, tf.float64) + tf.constant(
            SCORE_PARITY_RTOL, tf.float64
        ) * tf.abs(strict_score)
        score_flat_index = tf.argmax(
            tf.reshape(score_difference, [-1]), output_type=tf.int32
        )
        score_location = tf.unravel_index(score_flat_index, tf.shape(score_difference))
        score_ratio = score_difference / score_tolerance
        score_ratio_flat_index = tf.argmax(
            tf.reshape(score_ratio, [-1]), output_type=tf.int32
        )
        score_ratio_location = tf.unravel_index(
            score_ratio_flat_index, tf.shape(score_ratio)
        )
        value_flat_index = tf.argmax(
            tf.reshape(value_difference, [-1]), output_type=tf.int32
        )
        value_location = tf.unravel_index(value_flat_index, tf.shape(value_difference))
        return {
            "candidate": candidate,
            "max_abs_value": float(tf.reduce_max(value_difference).numpy()),
            "max_abs_score": float(tf.reduce_max(score_difference).numpy()),
            "max_abs_value_over_strict_scale": float(
                tf.reduce_max(value_difference).numpy()
            )
            / strict_value_scale,
            "max_abs_score_over_strict_scale": float(
                tf.reduce_max(score_difference).numpy()
            )
            / strict_score_scale,
            "value_rtol": VALUE_PARITY_RTOL,
            "score_rtol": SCORE_PARITY_RTOL,
            "atol": PARITY_ATOL,
            "value_within_tolerance": bool(
                tf.reduce_all(value_difference <= value_tolerance).numpy()
            ),
            "score_within_tolerance": bool(
                tf.reduce_all(score_difference <= score_tolerance).numpy()
            ),
            "max_score_location": [int(item) for item in score_location.numpy()],
            "strict_score_at_max": float(
                tf.gather_nd(strict_score, score_location).numpy()
            ),
            "candidate_score_at_max": float(
                tf.gather_nd(candidate_score, score_location).numpy()
            ),
            "max_score_tolerance_ratio": float(
                tf.gather_nd(score_ratio, score_ratio_location).numpy()
            ),
            "max_score_tolerance_ratio_location": [
                int(item) for item in score_ratio_location.numpy()
            ],
            "strict_score_at_max_tolerance_ratio": float(
                tf.gather_nd(strict_score, score_ratio_location).numpy()
            ),
            "candidate_score_at_max_tolerance_ratio": float(
                tf.gather_nd(candidate_score, score_ratio_location).numpy()
            ),
            "max_value_location": [int(item) for item in value_location.numpy()],
            "status_code_equal": bool(
                tf.reduce_all(
                    tf.equal(
                        candidate_status["status_code"], strict_status["status_code"]
                    )
                ).numpy()
            ),
            "row_class_equal": bool(
                tf.reduce_all(
                    tf.equal(
                        candidate_status["principal_sqrt_target_row_class_code"],
                        strict_status["principal_sqrt_target_row_class_code"],
                    )
                ).numpy()
            ),
        }

    parity_by_backend = {
        backend: compare(backend) for backend in candidate_backends
    }
    # Keep the original top-level fields while retaining the candidate map for
    # future additions to this diagnostic.
    parity = dict(parity_by_backend[candidate_backends[0]])
    parity["candidates"] = parity_by_backend
    candidate_passes = all(
        row["status_code_equal"]
        and row["row_class_equal"]
        and row["value_within_tolerance"]
        and row["score_within_tolerance"]
        for row in parity_by_backend.values()
    )
    payload = {
        "schema": "bayesfilter.ssl_lstm_q20.eigh_reuse_benchmark.v1",
        "status": (
            "PASS_DIAGNOSTIC_PARITY"
            if candidate_passes
            else "FAIL_DIAGNOSTIC_PARITY"
        ),
        "python": sys.executable,
        "platform": platform.platform(),
        "jit_compile": jit_compile,
        "batch_size": args.batch_size,
        "row_offset_scale": float(args.row_offset_scale),
        "repeats": args.repeats,
        "evaluated_backends": list(evaluated_backends),
        "required_candidate_backends": list(candidate_backends),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "tf_force_gpu_allow_growth": os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", ""),
        "physical_gpus": [str(device) for device in physical],
        "logical_gpus": [str(device) for device in logical],
        "memory_policy": _json_value(memory_policy),
        "plan_path": "docs/plans/bayesfilter-ssl-lstm-q20-factor-route-promotion-test-plan-2026-09-04.md",
        "command": list(sys.argv),
        "git": _git_metadata(),
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "seed_ledger": {
            "policy": "deterministic_fixture_no_random_draws",
            "random_seeds": [],
        },
        "source_provenance": {
            "files": {
                "benchmark": {
                    "path": str(Path(__file__).relative_to(ROOT)),
                    "sha256": _source_sha256(
                        str(Path(__file__).relative_to(ROOT))
                    ),
                },
                "target_module": {
                    "path": "bayesfilter/nonlinear/ssl_lstm_complexity_batched_target_tf.py",
                    "sha256": _source_sha256(
                        "bayesfilter/nonlinear/ssl_lstm_complexity_batched_target_tf.py"
                    ),
                },
                "eigensystem_module": {
                    "path": "bayesfilter/nonlinear/experimental_batched_svd_sigma_point_tf.py",
                    "sha256": _source_sha256(
                        "bayesfilter/nonlinear/experimental_batched_svd_sigma_point_tf.py"
                    ),
                },
            }
        },
        "records": records,
        "parity": parity,
        "elapsed_seconds": time.perf_counter() - started_campaign,
        "nonclaims": [
            "no whitening",
            "no mode discovery",
            "no posterior correctness",
            "no convergence",
            "no sampler superiority",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(args.output), "parity": parity}, sort_keys=True))
    return 0 if payload["status"] == "PASS_DIAGNOSTIC_PARITY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
