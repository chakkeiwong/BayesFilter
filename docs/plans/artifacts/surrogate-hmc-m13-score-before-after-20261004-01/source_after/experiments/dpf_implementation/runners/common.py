"""Shared helpers for experimental DPF runners."""

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

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
REPORT_DIR = REPO_ROOT / "experiments" / "dpf_implementation" / "reports"
OUTPUT_DIR = REPORT_DIR / "outputs"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def environment_manifest(
    *,
    pre_import_cuda_visible_devices: str | None,
    pre_import_gpu_hiding_assertion: bool,
) -> dict[str, Any]:
    manifest = git_manifest()
    return {
        **manifest,
        "python_version": platform.python_version(),
        "package_versions": {"numpy": np.__version__},
        "cpu_only": True,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "pre_import_cuda_visible_devices": pre_import_cuda_visible_devices,
        "pre_import_gpu_hiding_assertion": pre_import_gpu_hiding_assertion,
        "gpu_devices_visible": [],
    }


def wall_time_call(fn):
    started = time.perf_counter()
    result = fn()
    runtime = time.perf_counter() - started
    return result, runtime


def finite_list(values: Any) -> bool:
    return bool(np.all(np.isfinite(np.asarray(values, dtype=np.float64))))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean((arr_a - arr_b) ** 2)))


def max_sinkhorn_residual(diagnostics: list[dict[str, Any]]) -> float:
    residuals = []
    for diag in diagnostics:
        if diag.get("resampling_method") == "finite_sinkhorn_relaxed":
            residuals.append(float(diag.get("max_row_residual", 0.0)))
            residuals.append(float(diag.get("max_column_residual", 0.0)))
            residuals.append(float(diag.get("total_mass_residual", 0.0)))
    return float(max(residuals) if residuals else 0.0)
