#!/usr/bin/env python3
"""Run and record the trusted TensorFlow GPU/XLA launch preflight."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import subprocess
import sys
import hashlib
import time
from pathlib import Path
from typing import Sequence


TRUST_BASIS = "owner_designated_managed_session_visible_gpu_trusted"
PLAN_PATH = (
    "docs/plans/bayesfilter-complete-highdim-leaderboard-"
    "phase2-ledh-fulltime-seed81120-subplan-2026-07-11.md"
)
RESULT_PATH = (
    "docs/plans/bayesfilter-complete-highdim-leaderboard-"
    "phase2-ledh-fulltime-seed81120-result-2026-07-11.md"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite GPU preflight artifact: {args.output}")
    started = time.perf_counter()
    started_utc = dt.datetime.now(dt.timezone.utc).isoformat()

    smi = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    import tensorflow as tf

    tf.config.experimental.enable_tensor_float_32_execution(True)

    @tf.function(jit_compile=True)
    def compiled_matmul(x: tf.Tensor) -> tf.Tensor:
        return tf.linalg.matmul(x, x)

    physical = [str(device) for device in tf.config.list_physical_devices("GPU")]
    logical = [str(device) for device in tf.config.list_logical_devices("GPU")]
    output = compiled_matmul(tf.ones([16, 16], dtype=tf.float32))
    output_device = str(output.device)
    tf32_enabled = bool(tf.config.experimental.tensor_float_32_execution_enabled())
    output_finite = bool(tf.reduce_all(tf.math.is_finite(output)).numpy())
    passed = bool(
        smi.returncode == 0
        and physical
        and logical
        and "GPU" in output_device.upper()
        and tf32_enabled
        and output_finite
    )
    record = {
        "schema_version": "bayesfilter.complete_highdim_leaderboard.gpu_preflight.v1",
        "timestamp_started_utc": started_utc,
        "timestamp_completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "wall_time_seconds": time.perf_counter() - started,
        "command": [sys.executable, __file__, "--output", str(args.output)],
        "preflight_script_path": str(Path(__file__).resolve()),
        "preflight_script_sha256": _sha256(Path(__file__).resolve()),
        "cwd": str(Path.cwd()),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "git_status_short": subprocess.check_output(
            ["git", "status", "--short"], text=True
        ),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "tensorflow_version": tf.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_trust_basis": TRUST_BASIS,
        "nvidia_smi_command": [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        "nvidia_smi_exit_code": smi.returncode,
        "nvidia_smi_pass": smi.returncode == 0,
        "nvidia_smi_rows": [line for line in smi.stdout.splitlines() if line.strip()],
        "nvidia_smi_stderr": smi.stderr.strip(),
        "physical_gpus": physical,
        "logical_gpus": logical,
        "jit_compile": True,
        "dtype": "float32",
        "tf32_execution_enabled": tf32_enabled,
        "output_device": output_device,
        "output_finite": output_finite,
        "output_artifact": str(args.output),
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "data_version": "N/A_device_only_preflight",
        "random_seeds": "N/A_deterministic_constant_matmul",
        "evidence_role": "continuation_veto_device_preflight",
        "preflight_pass": passed,
        "nonclaims": [
            "device preflight is not leaderboard completion",
            "device preflight is not posterior, HMC, or scientific validity evidence"
        ],
    }
    _write_atomic(args.output, record)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
