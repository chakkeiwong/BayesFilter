#!/usr/bin/env python3
"""Fail-closed trusted-GPU preflight for Contract E--TP Phase 4."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


ARGS = _parse()
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.runtime.gpu_memory_policy import (  # noqa: E402
    configure_tensorflow_gpu_memory_growth,
)

import tensorflow as tf  # noqa: E402


MEMORY_POLICY = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)


def main() -> int:
    output = ARGS.output if ARGS.output.is_absolute() else ROOT / ARGS.output
    if output.exists():
        raise FileExistsError(output)
    if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") != "true":
        raise RuntimeError("GPU probe requires TF_FORCE_GPU_ALLOW_GROWTH=true")
    started = time.perf_counter()
    physical = tuple(tf.config.list_physical_devices("GPU"))
    logical = tuple(tf.config.list_logical_devices("GPU"))
    if not physical or not logical:
        raise RuntimeError("Phase 4 trusted-GPU probe requires a logical GPU")
    with tf.device("/GPU:0"):
        value = tf.linalg.matmul(
            tf.eye(2, dtype=tf.float64), tf.ones([2, 1], dtype=tf.float64)
        )
    _ = value.numpy()
    if "GPU:0" not in value.device.upper():
        raise RuntimeError(f"probe tensor did not execute on GPU: {value.device}")
    memory = tf.config.experimental.get_memory_info("GPU:0")
    payload = {
        "schema": "bayesfilter.contract_e_tp.clean_xla_phase4_gpu_probe.v1",
        "status": "PASS_TRUSTED_GPU_PREFLIGHT",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "python": sys.version,
        "platform": platform.platform(),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "N/A"),
        "tensorflow_version": tf.__version__,
        "physical_gpus": [str(item) for item in physical],
        "logical_gpus": [str(item) for item in logical],
        "probe_output_device": value.device,
        "gpu_memory_policy": MEMORY_POLICY,
        "gpu_allocator_current_bytes": int(memory["current"]),
        "gpu_allocator_peak_bytes": int(memory["peak"]),
        "tf32": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
        "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        "command": " ".join(sys.argv),
        "wall_time_seconds": time.perf_counter() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "device": value.device}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
