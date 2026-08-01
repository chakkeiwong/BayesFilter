#!/usr/bin/env python3
"""Trusted GPU/XLA timing cell for the exact batch-native LGSSM target."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PLAN = (
    "docs/plans/"
    "bayesfilter-neutra-batch-native-training-phase6-trusted-gpu-performance-"
    "subplan-2026-07-14.md"
)
TRUST_BASIS = "owner_designated_managed_session_visible_gpu_trusted"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--timed-calls", type=int, default=3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = run_cell(
        batch_size=int(args.batch_size),
        timed_calls=int(args.timed_calls),
        output=_repo_path(args.output),
    )
    print(
        json.dumps(
            {
                "passed": payload["passed"],
                "batch_size": payload["batch_size"],
                "warm_seconds": payload["timing"]["warm_compile_seconds"],
                "steady_seconds": payload["timing"]["steady_call_seconds"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


def run_cell(*, batch_size: int, timed_calls: int, output: Path) -> Mapping[str, Any]:
    if batch_size < 2:
        raise ValueError("batch size must be at least two")
    if timed_calls <= 0:
        raise ValueError("timed calls must be positive")
    if output.exists():
        raise ValueError(f"refusing to overwrite artifact: {output}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
        raise RuntimeError("trusted GPU cell cannot run with CUDA hidden")

    import tensorflow as tf

    physical = tf.config.list_physical_devices("GPU")
    if not physical:
        raise RuntimeError("TensorFlow sees no trusted GPU")
    for device in physical:
        try:
            tf.config.experimental.set_memory_growth(device, True)
        except RuntimeError:
            pass
    tf.config.set_soft_device_placement(False)

    from bayesfilter.inference.neutra_batching import bind_batch_native_neutra_target
    from bayesfilter.testing.deterministic_lgssm_exact_target_tf import (
        load_deterministic_lgssm_exact_target,
    )

    bundle = load_deterministic_lgssm_exact_target()
    binding = bind_batch_native_neutra_target(
        bundle.adapter,
        target_signature=bundle.target_signature,
    )
    perturbation = tf.random.stateless_normal(
        (batch_size, bundle.adapter.parameter_dim),
        seed=(20260714, 1600 + batch_size),
        stddev=tf.constant(1.0e-3, tf.float64),
        dtype=tf.float64,
    )
    points = bundle.raw_truth[tf.newaxis, :] + perturbation
    points = tf.tensor_scatter_nd_update(
        points,
        tf.constant([[0]], tf.int32),
        bundle.raw_truth[tf.newaxis, :],
    )

    @tf.function(
        input_signature=[tf.TensorSpec((batch_size, 18), tf.float64)],
        jit_compile=True,
    )
    def compiled(values):
        value, score, status = binding.invoke(values)
        return (
            value,
            score,
            status["status_code"],
            status["valid_pre_regularized_score"],
            status["floor_count_value"],
            status["min_innovation_eigenvalue"],
            status["innovation_condition_estimate"],
        )

    started = time.perf_counter()
    warm = compiled(points)
    _synchronize(warm)
    warm_seconds = time.perf_counter() - started
    steady_seconds: list[float] = []
    terminal = warm
    for _call in range(timed_calls):
        started = time.perf_counter()
        terminal = compiled(points)
        _synchronize(terminal)
        steady_seconds.append(time.perf_counter() - started)

    value, score, status_code, valid, floor_count, minimum, condition = terminal
    all_finite = bool(
        tf.reduce_all(
            tf.concat(
                (
                    tf.reshape(tf.math.is_finite(value), (-1,)),
                    tf.reshape(tf.math.is_finite(score), (-1,)),
                    tf.reshape(tf.math.is_finite(minimum), (-1,)),
                    tf.reshape(tf.math.is_finite(condition), (-1,)),
                ),
                axis=0,
            )
        ).numpy()
    )
    all_valid = bool(tf.reduce_all(valid).numpy())
    status_all_zero = bool(tf.reduce_all(tf.equal(status_code, 0)).numpy())
    no_floors = bool(tf.reduce_all(tf.equal(floor_count, 0)).numpy())
    output_devices = tuple(str(item.device) for item in terminal)
    all_outputs_gpu = all("GPU" in device.upper() for device in output_devices)
    concrete = compiled.get_concrete_function()
    operation_types = tuple(
        sorted({operation.type for operation in concrete.graph.get_operations()})
    )
    passed = bool(
        all_finite and all_valid and status_all_zero and no_floors and all_outputs_gpu
    )
    memory = _gpu_memory(tf)
    payload: dict[str, Any] = {
        "schema": "bayesfilter.neutra.batch_native_target_gpu_cell.v1",
        "passed": passed,
        "decision": (
            "ADMIT_BATCH_TARGET_GPU_TIMING_CELL"
            if passed
            else "REJECT_BATCH_TARGET_GPU_TIMING_CELL"
        ),
        "plan": PLAN,
        "trust_basis": TRUST_BASIS,
        "batch_size": batch_size,
        "timed_calls": timed_calls,
        "target_signature": bundle.target_signature,
        "adapter_signature": bundle.adapter.adapter_signature(),
        "batch_binding": binding.payload(),
        "timing": {
            "warm_compile_seconds": warm_seconds,
            "steady_call_seconds": steady_seconds,
            "steady_mean_seconds": sum(steady_seconds) / len(steady_seconds),
            "synchronization": "materialize_reduced_value_score_status_scalars",
        },
        "validity": {
            "all_finite": all_finite,
            "all_valid_pre_regularized": all_valid,
            "status_all_zero": status_all_zero,
            "floor_count_all_zero": no_floors,
            "status_nonzero_count": int(
                tf.reduce_sum(tf.cast(tf.not_equal(status_code, 0), tf.int32)).numpy()
            ),
            "minimum_innovation_eigenvalue": float(tf.reduce_min(minimum).numpy()),
            "maximum_innovation_condition_estimate": float(
                tf.reduce_max(condition).numpy()
            ),
        },
        "runtime": {
            "tensorflow_version": tf.__version__,
            "physical_gpus": tuple(str(item) for item in physical),
            "logical_gpus": tuple(
                str(item) for item in tf.config.list_logical_devices("GPU")
            ),
            "output_devices": output_devices,
            "all_outputs_gpu": all_outputs_gpu,
            "jit_compile": True,
            "tf32_enabled": bool(
                tf.config.experimental.tensor_float_32_execution_enabled()
            ),
            "gpu_memory": memory,
            "graph_operation_types": operation_types,
            "while_operation_types": tuple(
                item for item in operation_types if "While" in item
            ),
            "map_operation_types": tuple(
                item for item in operation_types if "Map" in item
            ),
            "callback_operation_types": tuple(
                item for item in operation_types if "PyFunc" in item
            ),
        },
        "command": tuple(sys.argv),
        "environment": {
            "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "python": sys.executable,
        },
        "git": {
            "commit": _command(("git", "rev-parse", "HEAD")),
            "status_short": _command(("git", "status", "--short")),
        },
        "source_sha256": {
            path: _file_sha256(ROOT / path)
            for path in (
                "bayesfilter/inference/neutra_batching.py",
                "bayesfilter/linear/batched_kalman_svd_derivatives_tf.py",
                "bayesfilter/testing/multidim_triangular_lgssm_batched_tf.py",
                "bayesfilter/testing/deterministic_lgssm_exact_target_tf.py",
            )
        },
        "nonclaims": (
            "target timing is not full NeuTra training timing",
            "three calls do not support statistical batch-size ranking",
            "no transport quality, posterior correctness, HMC readiness, or scientific validity claim",
        ),
    }
    payload["artifact_hash"] = f"sha256:{_stable_hash(payload)}"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not passed:
        raise RuntimeError("trusted GPU batch target cell failed validity checks")
    return payload


def _synchronize(outputs: tuple[Any, ...]) -> None:
    import tensorflow as tf

    scalar = tf.add_n(
        (
            tf.reduce_sum(tf.cast(outputs[0], tf.float64)),
            tf.reduce_sum(tf.cast(outputs[1], tf.float64)),
            tf.reduce_sum(tf.cast(outputs[2], tf.float64)),
            tf.reduce_sum(tf.cast(outputs[3], tf.float64)),
        )
    )
    scalar.numpy()


def _gpu_memory(tf: Any) -> Mapping[str, int] | None:
    try:
        return {
            str(key): int(value)
            for key, value in tf.config.experimental.get_memory_info("GPU:0").items()
        }
    except Exception:
        return None


def _command(command: tuple[str, ...]) -> str:
    return subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
