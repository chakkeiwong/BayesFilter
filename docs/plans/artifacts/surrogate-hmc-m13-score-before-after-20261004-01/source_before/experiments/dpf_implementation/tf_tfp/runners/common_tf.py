"""Shared helpers for TF/TFP experimental DPF runners."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tensorflow as tf
import tensorflow_probability as tfp


REPO_ROOT = Path(__file__).resolve().parents[4]
REPORT_DIR = REPO_ROOT / "experiments" / "dpf_implementation" / "reports"
OUTPUT_DIR = REPORT_DIR / "outputs"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def environment_manifest(
    *,
    command: str,
    pre_import_cuda_visible_devices: str | None,
) -> dict[str, Any]:
    return {
        **git_manifest(),
        "python_version": platform.python_version(),
        "package_versions": {
            "tensorflow": tf.__version__,
            "tensorflow_probability": tfp.__version__,
        },
        "cpu_only": True,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "pre_import_cuda_visible_devices": pre_import_cuda_visible_devices,
        "gpu_devices_visible": [str(device) for device in tf.config.list_physical_devices("GPU")],
        "command": command,
    }


def git_manifest() -> dict[str, str]:
    return {
        "branch": _git(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "commit": _git(["git", "rev-parse", "HEAD"]),
        "dirty_state_summary": _git(["git", "status", "--short"]) or "clean",
    }


def _git(args: list[str]) -> str:
    completed = subprocess.run(args, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def stable_digest(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def tensor_to_json(value: tf.Tensor) -> Any:
    return tf.cast(value, tf.float64).numpy().tolist()


def scalar(value: tf.Tensor) -> float:
    return float(tf.cast(value, tf.float64).numpy())


def finite_tensor(value: tf.Tensor) -> bool:
    return bool(tf.reduce_all(tf.math.is_finite(tf.cast(value, tf.float64))).numpy())


def rmse_tf(a: tf.Tensor, b: tf.Tensor) -> float:
    diff = tf.cast(a, tf.float64) - tf.cast(b, tf.float64)
    return scalar(tf.sqrt(tf.reduce_mean(diff * diff)))


def max_sinkhorn_residual(diagnostics: list[dict[str, Any]]) -> float:
    residuals = []
    for diag in diagnostics:
        if diag.get("resampling_method") in {
            "finite_sinkhorn_relaxed_tf",
            "fixed_target_sinkhorn_local_comparator_tf",
        }:
            residuals.append(float(diag.get("max_row_residual", 0.0)))
            residuals.append(float(diag.get("max_column_residual", 0.0)))
            residuals.append(float(diag.get("total_mass_residual", 0.0)))
        elif diag.get("resampling_method") == "filterflow_style_annealed_transport_tf":
            residuals.append(float(diag.get("max_column_residual", 0.0)))
    return max(residuals) if residuals else 0.0


def wall_time_call(fn):
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start
