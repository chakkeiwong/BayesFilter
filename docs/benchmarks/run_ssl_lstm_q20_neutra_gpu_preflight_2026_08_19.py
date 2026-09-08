#!/usr/bin/env python3
"""Record the trusted GPU/XLA preflight for q=20 NeuTra execution."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
PLAN = ROOT / (
    "docs/plans/"
    "bayesfilter-ssl-lstm-q20-neutra-global-mixing-execution-plan-2026-08-19.md"
)
DEFAULT_OUTPUT = ROOT / (
    "docs/plans/artifacts/"
    "ssl-lstm-q20-neutra-global-mixing-execution-2026-08-19/r1/"
    "gpu-preflight.json"
)
TRUST_BASIS = "owner_designated_managed_session_visible_gpu_trusted"


class GPUPreflightError(RuntimeError):
    """Raised when the trusted q=20 GPU preflight fails closed."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "numpy"):
        return _json_ready(value.numpy())
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise GPUPreflightError(f"refusing to overwrite GPU preflight: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise GPUPreflightError(f"stale GPU preflight temporary exists: {temporary}")
    temporary.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="ascii",
    )
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="1")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if not str(args.device).isdigit():
        raise SystemExit("device must be one nonnegative physical GPU index")
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists():
        raise GPUPreflightError(f"refusing to overwrite GPU preflight: {output}")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    started = time.perf_counter()
    started_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    smi = subprocess.run(
        (
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ),
        check=False,
        capture_output=True,
        text=True,
    )

    import tensorflow as tf

    from bayesfilter.runtime.gpu_memory_policy import (
        configure_tensorflow_gpu_memory_growth,
    )

    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    physical = tuple(tf.config.list_physical_devices("GPU"))
    logical = tuple(tf.config.list_logical_devices("GPU"))
    if len(physical) != 1 or len(logical) != 1:
        raise GPUPreflightError(
            f"expected one masked physical/logical GPU, found {physical} / {logical}"
        )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def compiled_probe(values: Any) -> Any:
        return tf.linalg.matmul(values, values)

    with tf.device("/GPU:0"):
        result = compiled_probe(tf.ones((16, 16), tf.float64))
    tf.debugging.assert_all_finite(result, "GPU/XLA preflight result")
    expected = tf.fill((16, 16), tf.constant(16.0, tf.float64))
    value_residual = tf.reduce_max(tf.abs(result - expected))
    # IEEE float64 epsilon with a conservative reduction-depth safety factor.
    value_tolerance = (
        tf.constant(4096.0 * 2.220446049250313e-16, tf.float64)
        * tf.maximum(tf.constant(1.0, tf.float64), tf.reduce_max(tf.abs(expected)))
    )
    tf.debugging.assert_less_equal(
        value_residual,
        value_tolerance,
        message="GPU/XLA preflight value exceeded the float64 roundoff bound",
    )
    if "GPU:0" not in result.device.upper():
        raise GPUPreflightError(f"compiled probe did not execute on GPU: {result.device}")
    allocator = tf.config.experimental.get_memory_info("GPU:0")
    if int(allocator.get("peak", 0)) <= 0:
        raise GPUPreflightError("TensorFlow GPU allocator did not report positive peak use")
    if bool(tf.config.experimental.tensor_float_32_execution_enabled()):
        raise GPUPreflightError("TF32 must remain disabled for the q=20 float64 route")

    payload = {
        "schema": "bayesfilter.ssl_lstm.q20_neutra_gpu_preflight.v1",
        "status": "GPU_PREFLIGHT_PASSED",
        "timestamp_started_utc": started_utc,
        "timestamp_completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "wall_time_seconds": time.perf_counter() - started,
        "command": [sys.executable, *sys.argv],
        "cwd": Path.cwd(),
        "environment": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "git_commit": subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "git_dirty": bool(
            subprocess.run(
                ("git", "status", "--short"),
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        ),
        "plan": {"path": PLAN, "sha256": _sha256(PLAN)},
        "runner": {"path": SCRIPT, "sha256": _sha256(SCRIPT)},
        "managed_session_trust_basis": TRUST_BASIS,
        "requested_physical_device_selector": str(args.device),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "visible_physical_gpus": [str(device) for device in physical],
        "visible_logical_gpus": [str(device) for device in logical],
        "memory_policy": memory_policy,
        "jit_compile": True,
        "concrete_function_count": len(
            compiled_probe._list_all_concrete_functions_for_serialization()
        ),
        "dtype": "float64",
        "tf32_enabled": False,
        "result_device": result.device,
        "result_finite": True,
        "value_maximum_absolute_residual": value_residual,
        "value_roundoff_tolerance": value_tolerance,
        "value_roundoff_tolerance_provenance": (
            "4096_times_ieee_float64_epsilon_times_max_1_or_expected_scale"
        ),
        "allocator_bytes": {
            "current": int(allocator.get("current", 0)),
            "peak": int(allocator.get("peak", 0)),
        },
        "nvidia_smi": {
            "exit_code": smi.returncode,
            "rows": [line for line in smi.stdout.splitlines() if line.strip()],
            "stderr": smi.stderr.strip(),
        },
        "data_version": "N/A_device_only_preflight",
        "random_seeds": "N/A_deterministic_constant_matmul",
        "output_artifact": output,
        "nonclaims": [
            "device preflight is not NeuTra training evidence",
            "device preflight is not HMC, posterior, or scientific evidence",
        ],
    }
    _write_atomic(output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "result_device": payload["result_device"],
                "output": output.as_posix(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
