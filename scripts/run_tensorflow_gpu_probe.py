#!/usr/bin/env python3
"""Bounded UUID-pinned TensorFlow GPU and allocator probe."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from bayesfilter.runtime.gpu_provenance import (
    query_nvidia_smi_gpus,
    selected_nvidia_gpu,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {args.output}")
    if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "").lower() != "true":
        raise RuntimeError("TF_FORCE_GPU_ALLOW_GROWTH=true is required")

    # Resolve the stable physical identity before TensorFlow initializes CUDA.
    nvidia_gpu = selected_nvidia_gpu(query_nvidia_smi_gpus())

    import tensorflow as tf

    from bayesfilter.runtime.gpu_memory_policy import (
        configure_tensorflow_gpu_memory_growth,
    )

    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    physical = tf.config.list_physical_devices("GPU")
    if len(physical) != 1:
        raise RuntimeError(f"UUID-pinned probe requires one physical GPU, got {physical}")

    started = time.perf_counter()
    with tf.device("/GPU:0"):
        left = tf.ones((1024, 1024), dtype=tf.float32)
        product = tf.linalg.matmul(left, left)
        checksum = float(tf.reduce_sum(product).numpy())
    allocator = tf.config.experimental.get_memory_info("GPU:0")
    if "GPU:0" not in product.device or int(allocator["peak"]) <= 0:
        raise RuntimeError(
            f"operation did not prove GPU use: device={product.device}, allocator={allocator}"
        )
    logical = tf.config.list_logical_devices("GPU")
    processes = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory,gpu_uuid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()

    payload = {
        "status": "pass",
        "kind": "bayesfilter_tensorflow_gpu_probe",
        "tensorflow": tf.__version__,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "selected_nvidia_gpu": nvidia_gpu,
        "tensorflow_gpu_memory_policy": memory_policy,
        "physical_gpus": [device.name for device in physical],
        "logical_gpus": [device.name for device in logical],
        "operation_device": product.device,
        "allocator_bytes": {key: int(value) for key, value in allocator.items()},
        "checksum": checksum,
        "elapsed_seconds": time.perf_counter() - started,
        "process_pid": os.getpid(),
        "nvidia_compute_processes_while_live": [row for row in processes if row.strip()],
        "nonclaims": [
            "bounded GPU infrastructure probe only",
            "no filtering, likelihood, HMC, convergence, performance, or scientific claim",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
