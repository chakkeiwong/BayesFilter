#!/usr/bin/env python3
"""Transparent, no-write probes for isolating gateway command admission.

The script never starts a child process and never writes an artifact. The
``shape`` mode accepts arbitrary trailing arguments and reports their size and
digest, which makes it possible to compare a long Phase 52-shaped argv with a
real boundary request without invoking the boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Any


SCHEMA = "bayesfilter.gateway_admission_probe.v1"


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True), flush=True)


def _argv_summary() -> dict[str, Any]:
    raw = sys.argv[1:]
    encoded = [value.encode("utf-8") for value in raw]
    joined = b"\0".join(encoded)
    return {
        "argv_count": len(raw),
        "argv_bytes_with_separators": sum(len(value) + 1 for value in encoded),
        "argv_sha256": hashlib.sha256(joined).hexdigest(),
    }


def _base_payload(mode: str) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": "PASS_GATEWAY_PROBE",
        "mode": mode,
        "pid": os.getpid(),
        "python": sys.executable,
        "python_version": sys.version,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "tf_force_gpu_allow_growth": os.environ.get(
            "TF_FORCE_GPU_ALLOW_GROWTH", "unset"
        ),
    }


def _tensorflow_cpu_probe() -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("tensorflow_cpu mode requires CUDA_VISIBLE_DEVICES=-1")
    import tensorflow as tf

    physical = tuple(tf.config.list_physical_devices("GPU"))
    logical = tuple(tf.config.list_logical_devices("GPU"))
    if physical or logical:
        raise RuntimeError(
            f"CPU-hidden TensorFlow probe unexpectedly sees GPU devices: {physical}, {logical}"
        )
    return {
        "tensorflow": tf.__version__,
        "physical_gpus": [str(device) for device in physical],
        "logical_gpus": [str(device) for device in logical],
        "tensorflow_imported": True,
    }


def _tensorflow_gpu_probe() -> dict[str, Any]:
    if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "").lower() != "true":
        raise RuntimeError("tensorflow_gpu mode requires TF_FORCE_GPU_ALLOW_GROWTH=true")

    import tensorflow as tf

    physical = tuple(tf.config.list_physical_devices("GPU"))
    if not physical:
        raise RuntimeError("tensorflow_gpu mode requires a visible physical GPU")
    policy_devices = []
    for device in physical:
        tf.config.experimental.set_memory_growth(device, True)
        enabled = bool(tf.config.experimental.get_memory_growth(device))
        if not enabled:
            raise RuntimeError(f"memory growth verification failed for {device}")
        policy_devices.append({"device": str(device), "memory_growth": enabled})
    policy = {
        "schema": "bayesfilter.tensorflow.gpu_memory_policy.probe.v1",
        "mode": "memory_growth",
        "physical_devices": policy_devices,
        "all_physical_devices_memory_growth": True,
        "configured_before_logical_device_initialization": True,
        "tf_force_gpu_allow_growth": os.environ["TF_FORCE_GPU_ALLOW_GROWTH"],
    }

    @tf.function(jit_compile=True)
    def tiny_xla_workload() -> Any:
        left = tf.ones((128, 128), dtype=tf.float32)
        return tf.linalg.matmul(left, left)

    product = tiny_xla_workload()
    return {
        "tensorflow": tf.__version__,
        "tensorflow_gpu_memory_policy": policy,
        "logical_gpus": [
            str(device) for device in tf.config.list_logical_devices("GPU")
        ],
        "operation_device": str(product.device),
        "jit_compile": True,
        "tensorflow_imported": True,
        "gpu_workload_completed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("python", "shape", "sleep", "tensorflow_cpu", "tensorflow_gpu"),
    )
    parser.add_argument("--seconds", type=float, default=0.0)
    args, trailing = parser.parse_known_args()

    if args.mode == "sleep":
        if not 0.0 <= args.seconds <= 5.0:
            raise ValueError("sleep mode accepts seconds in [0, 5]")
        time.sleep(args.seconds)

    payload = _base_payload(args.mode)
    payload.update(_argv_summary())
    payload["trailing_argument_count"] = len(trailing)
    payload["trailing_arguments_are_uninterpreted"] = True

    if args.mode == "shape":
        payload["purpose"] = "argv_shape_only_no_tensorflow_no_child_process"
    elif args.mode == "python":
        payload["purpose"] = "interpreter_only_no_tensorflow_no_child_process"
    elif args.mode == "sleep":
        payload["purpose"] = "bounded_sleep_no_tensorflow_no_child_process"
        payload["slept_seconds"] = args.seconds
    elif args.mode == "tensorflow_cpu":
        payload.update(_tensorflow_cpu_probe())
    else:
        payload.update(_tensorflow_gpu_probe())

    _emit(payload)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"gateway_admission_probe_error: {exc}", file=sys.stderr, flush=True)
        raise
