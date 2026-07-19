#!/usr/bin/env python3
"""Localize Phase 8 terminal-covariance projection-only failures."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import shlex
import subprocess
import sys
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import tensorflow as tf


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bayesfilter.nonlinear.ssl_lstm_predictive_tf as predictive  # noqa: E402


PLAN_PATH = Path(
    "docs/plans/bayesfilter-ssl-lstm-neutra-phase-8-predictive-design-refresh-"
    "plan-2026-07-17.md"
)
RESULT_PATH = Path(
    "docs/plans/bayesfilter-ssl-lstm-neutra-phase-8-predictive-design-refresh-"
    "result-2026-07-17.md"
)
SCRIPT_PATH = Path(__file__).resolve().relative_to(ROOT)
PILOT_SCRIPT_PATH = Path(
    "docs/benchmarks/run_ssl_lstm_neutra_phase8_target_pilot_2026_07_17.py"
)
OUTPUT_ROOT = Path(
    "docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/"
    "phase-8-predictive-design"
)
EXPECTED_FAILURE_STATUS = predictive.STATUS_PROJECTION


class TerminalDiagnosticError(RuntimeError):
    """Raised when the bounded terminal diagnostic is invalid."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(_absolute(path).read_bytes()).hexdigest()


def _load_pilot_harness() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "phase8_target_pilot_for_terminal_diagnostic", _absolute(PILOT_SCRIPT_PATH)
    )
    if spec is None or spec.loader is None:
        raise TerminalDiagnosticError("cannot load Phase 8 target-pilot harness")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0.0 else "-Infinity"
    if hasattr(value, "numpy"):
        return _json_safe(value.numpy())
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    absolute = _absolute(path)
    absolute.parent.mkdir(parents=True, exist_ok=True)
    if absolute.exists():
        raise TerminalDiagnosticError(f"refusing to overwrite receipt: {path}")
    text = json.dumps(
        _json_safe(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"
    absolute.write_text(text, encoding="ascii")


def _git(*arguments: str) -> str:
    environment = dict(os.environ)
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    return subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _trace_count(program: Any) -> int | None:
    method = getattr(program, "experimental_get_tracing_count", None)
    return None if method is None else int(method())


def _covariance_rows(terminal: Any) -> list[dict[str, Any]]:
    status = [int(value) for value in tf.unstack(terminal.status)]
    rows = []
    for index, code in enumerate(status):
        tau = float(terminal.psd_tolerance[index])
        rows.append(
            {
                "index": index,
                "status": code,
                "minimum_eigenvalue": float(terminal.minimum_eigenvalue[index]),
                "psd_tolerance": tau,
                "negative_eigenvalue_ratio": max(
                    0.0, -float(terminal.minimum_eigenvalue[index]) / tau
                ),
                "symmetry_residual": float(terminal.symmetry_residual[index]),
                "symmetry_ratio": float(terminal.symmetry_residual[index]) / tau,
                "projection_residual": float(terminal.projection_residual[index]),
                "projection_ratio": float(terminal.projection_residual[index]) / tau,
                "factor_reconstruction_residual": float(
                    terminal.factor_reconstruction_residual[index]
                ),
                "factor_reconstruction_ratio": float(
                    terminal.factor_reconstruction_residual[index]
                )
                / tau,
            }
        )
    return rows


def _cpu_reaudit(raw_covariances: tf.Tensor) -> Any:
    fields: list[list[tf.Tensor]] = [[] for _ in range(12)]
    with tf.device("/CPU:0"):
        copied = tf.identity(raw_covariances)
        for covariance in tf.unstack(copied):
            audited = predictive._audit_terminal_covariance(covariance)
            for index, value in enumerate(audited):
                fields[index].append(value)
        stacked = tuple(tf.stack(values, axis=0) for values in fields)
    return predictive.SSLLSTMTerminalState(
        mean=tf.zeros((int(raw_covariances.shape[0]), 3), tf.float64),
        raw_covariance=stacked[0],
        symmetrized_covariance=stacked[1],
        implemented_covariance=stacked[2],
        factor=stacked[3],
        raw_eigenvalues=stacked[4],
        clipped_eigenvalues=stacked[5],
        minimum_eigenvalue=stacked[6],
        psd_tolerance=stacked[7],
        symmetry_residual=stacked[8],
        projection_residual=stacked[9],
        factor_reconstruction_residual=stacked[10],
        filter_log_likelihood=tf.zeros((int(raw_covariances.shape[0]),), tf.float64),
        a1_filter_log_likelihood=tf.zeros((int(raw_covariances.shape[0]),), tf.float64),
        target_value=tf.zeros((int(raw_covariances.shape[0]),), tf.float64),
        total_value=tf.zeros((int(raw_covariances.shape[0]),), tf.float64),
        filter_parity_residual=tf.zeros((int(raw_covariances.shape[0]),), tf.float64),
        total_parity_residual=tf.zeros((int(raw_covariances.shape[0]),), tf.float64),
        filter_parity_tolerance=tf.zeros((int(raw_covariances.shape[0]),), tf.float64),
        total_parity_tolerance=tf.zeros((int(raw_covariances.shape[0]),), tf.float64),
        full_parameters=tf.zeros((int(raw_covariances.shape[0]), 24), tf.float64),
        status=stacked[11],
    )


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    failures = [row for row in rows if row["status"] != predictive.STATUS_VALID]
    return {
        "point_count": len(rows),
        "failure_count": len(failures),
        "failure_indices": [row["index"] for row in failures],
        "failure_statuses": [row["status"] for row in failures],
        "maximum_symmetry_ratio": max(row["symmetry_ratio"] for row in rows),
        "maximum_projection_ratio": max(row["projection_ratio"] for row in rows),
        "maximum_factor_reconstruction_ratio": max(
            row["factor_reconstruction_ratio"] for row in rows
        ),
        "maximum_negative_eigenvalue_ratio": max(
            row["negative_eigenvalue_ratio"] for row in rows
        ),
        "failure_rows": failures,
    }


def run_diagnostic(*, output: Path, wall_cap_seconds: float) -> dict[str, Any]:
    if not math.isfinite(wall_cap_seconds) or wall_cap_seconds <= 0.0:
        raise TerminalDiagnosticError("wall cap must be positive and finite")
    started_at = _now()
    started = time.perf_counter()
    pilot = _load_pilot_harness()
    entry = pilot.validate_entry_receipts()
    latent, archive_audit = pilot.read_frozen_pilot_prefix(entry["phase7"])
    theta, mapping = pilot.map_pilot_to_theta("fresh-g", latent["fresh-g"])
    flat_theta = tf.reshape(theta, [4 * pilot.PILOT_DRAWS_PER_CHAIN, 4])
    config = predictive.SSLLSTMForecastConfig()
    config.assert_evidence_config()
    program = predictive.ssl_lstm_terminal_compiled_program(config, 256)
    tensors = tuple(program(flat_theta))
    terminal = predictive._terminal_from_tensors(tensors)
    gpu_rows = _covariance_rows(terminal)
    gpu_summary = _summary(gpu_rows)
    if gpu_summary["failure_count"] == 0:
        raise TerminalDiagnosticError("the projection failure did not reproduce")
    if set(gpu_summary["failure_statuses"]) != {EXPECTED_FAILURE_STATUS}:
        raise TerminalDiagnosticError(
            f"terminal diagnostic found non-projection failures: {gpu_summary['failure_statuses']}"
        )
    if not all("GPU:" in str(value.device) for value in tensors):
        raise TerminalDiagnosticError("terminal diagnostic outputs are not GPU resident")
    trace_count = _trace_count(program)
    if trace_count != 1:
        raise TerminalDiagnosticError("terminal diagnostic retraced")
    cpu_terminal = _cpu_reaudit(terminal.raw_covariance)
    cpu_rows = _covariance_rows(cpu_terminal)
    cpu_summary = _summary(cpu_rows)
    if time.perf_counter() - started > wall_cap_seconds:
        raise TerminalDiagnosticError("terminal diagnostic wall cap exceeded")

    gpu_failures = {row["index"]: row for row in gpu_rows if row["status"] != 0}
    cpu_by_index = {row["index"]: row for row in cpu_rows}
    paired_failures = [
        {
            "index": index,
            "gpu": gpu_failures[index],
            "cpu_reaudit": cpu_by_index[index],
        }
        for index in sorted(gpu_failures)
    ]
    projection_only = (
        set(gpu_summary["failure_statuses"]) == {predictive.STATUS_PROJECTION}
        and gpu_summary["maximum_negative_eigenvalue_ratio"] <= 1.0
        and gpu_summary["maximum_symmetry_ratio"] <= 1.0
        and gpu_summary["maximum_factor_reconstruction_ratio"] <= 16.0
    )
    decision = (
        "PHASE8_TERMINAL_PROJECTION_TOLERANCE_REPAIR_CANDIDATE"
        if projection_only
        else "PHASE8_TERMINAL_COVARIANCE_MATERIAL_BLOCKER"
    )
    wall_time = time.perf_counter() - started
    payload = {
        "schema": "bayesfilter.ssl_lstm_neutra.phase8_terminal_projection_diagnostic.v1",
        "status": "PASSED_DIAGNOSTIC",
        "decision": decision,
        "failed_target_pilot": {
            "command_was_frozen": True,
            "exit_code": 1,
            "receipt_written": False,
            "failure_stage": "fresh_g_terminal_extraction_before_any_forecast_or_h_execution",
            "automatic_pilot_retry": False,
        },
        "scope": {
            "chart_mapped_and_evaluated": "fresh-g",
            "h_prefix_tensor_deserialized_by_shared_archive_reader": True,
            "h_mapped_or_terminal_evaluated": False,
            "forecast_executed": False,
            "predictive_difference_computed": False,
            "confirmation_suffix_selected": False,
        },
        "mapping": mapping,
        "archive_integrity": archive_audit,
        "gpu_xla_audit": gpu_summary,
        "cpu_tensorflow_reaudit": cpu_summary,
        "paired_gpu_failure_rows": paired_failures,
        "classification": {
            "projection_only": projection_only,
            "status_code": predictive.STATUS_PROJECTION,
            "status_name": "STATUS_PROJECTION",
            "current_projection_multiplier": 8.0,
            "current_factor_reconstruction_multiplier": 16.0,
            "interpretation": (
                "localization_only; repair requires a separately reviewed numerical change"
            ),
        },
        "source_bindings": {
            "plan": {"path": PLAN_PATH.as_posix(), "sha256": _sha256(PLAN_PATH)},
            "runner": {"path": SCRIPT_PATH.as_posix(), "sha256": _sha256(SCRIPT_PATH)},
            "pilot_runner": {
                "path": PILOT_SCRIPT_PATH.as_posix(),
                "sha256": _sha256(PILOT_SCRIPT_PATH),
            },
            "forecast": {
                "path": "bayesfilter/nonlinear/ssl_lstm_predictive_tf.py",
                "sha256": _sha256(Path("bayesfilter/nonlinear/ssl_lstm_predictive_tf.py")),
            },
        },
        "run_manifest": {
            "command": " ".join(shlex.quote(item) for item in (sys.executable, *sys.argv)),
            "cwd": str(ROOT),
            "interpreter": sys.executable,
            "python_version": platform.python_version(),
            "tensorflow": tf.__version__,
            "tensorflow_probability": __import__("tensorflow_probability").__version__,
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "physical_devices": [device.name for device in tf.config.list_physical_devices("GPU")],
            "logical_devices": [device.name for device in tf.config.list_logical_devices("GPU")],
            "jit_compile": True,
            "terminal_compile_trace_count": trace_count,
            "dtype": "float64",
            "tf32_enabled": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
            "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
            "git_commit": _git("rev-parse", "HEAD").strip(),
            "git_dirty": bool(_git("status", "--porcelain").strip()),
            "random_seeds": "N/A deterministic replay of Phase 7 retained points",
            "started_at_utc": started_at,
            "completed_at_utc": _now(),
            "wall_time_seconds": wall_time,
            "wall_cap_seconds": wall_cap_seconds,
            "output_path": output.as_posix(),
            "plan_path": PLAN_PATH.as_posix(),
            "result_path": RESULT_PATH.as_posix(),
        },
        "nonclaims": [
            "terminal-covariance localization only; no forecast executed",
            "no G/H predictive comparison, equivalence, ranking, or posterior claim",
            "CPU/GPU differences alone do not authorize changing a numerical tolerance",
            "the failed target pilot remains failed and was not retried",
        ],
    }
    _write_json(output, payload)
    return payload


def _require_gpu() -> None:
    physical = tf.config.list_physical_devices("GPU")
    if not physical:
        raise TerminalDiagnosticError("terminal diagnostic requires a visible trusted GPU")
    for gpu in physical:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    if not tf.config.list_logical_devices("GPU"):
        raise TerminalDiagnosticError("terminal diagnostic requires a logical GPU")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-cap-seconds", type=float, required=True)
    args = parser.parse_args(argv)
    _require_gpu()
    with tf.device("/GPU:0"):
        payload = run_diagnostic(
            output=args.output, wall_cap_seconds=float(args.wall_cap_seconds)
        )
    print(
        "JSON_SUMMARY "
        + json.dumps(
            {
                "decision": payload["decision"],
                "gpu_failure_count": payload["gpu_xla_audit"]["failure_count"],
                "cpu_failure_count": payload["cpu_tensorflow_reaudit"]["failure_count"],
                "wall_time_seconds": payload["run_manifest"]["wall_time_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
