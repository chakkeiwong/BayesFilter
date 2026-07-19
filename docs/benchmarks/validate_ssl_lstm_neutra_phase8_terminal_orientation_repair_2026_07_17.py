#!/usr/bin/env python3
"""Exact-prefix GPU validation for the Phase 8 staged covariance audit."""

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
DIAGNOSTIC_SCRIPT_PATH = Path(
    "docs/benchmarks/diagnose_ssl_lstm_neutra_phase8_terminal_projection_2026_07_17.py"
)
DECOMPOSITION_RECEIPT_PATH = Path(
    "docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/"
    "phase-8-predictive-design/raw-covariance-decomposition-diagnostic.json"
)
DECOMPOSITION_RECEIPT_SHA256 = (
    "32d1051b667df210c7eae4d174731a28fb7d8b9b26c13d590d5742d15f082fbd"
)


class TerminalRepairValidationError(RuntimeError):
    """Raised when exact-prefix terminal repair validation fails."""


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(_absolute(path).read_bytes()).hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, _absolute(path))
    if spec is None or spec.loader is None:
        raise TerminalRepairValidationError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _strict_json(path: Path) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            if key in result:
                raise TerminalRepairValidationError(
                    f"duplicate JSON key {key!r}: {path}"
                )
            result[key] = value
        return result

    def reject(value: str) -> None:
        raise TerminalRepairValidationError(
            f"nonfinite JSON constant {value!r}: {path}"
        )

    value = json.loads(
        _absolute(path).read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=reject,
    )
    if not isinstance(value, dict):
        raise TerminalRepairValidationError(f"expected JSON object: {path}")
    return value


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
        raise TerminalRepairValidationError(f"refusing to overwrite receipt: {path}")
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


def validate_decomposition_receipt() -> dict[str, Any]:
    if _sha256(DECOMPOSITION_RECEIPT_PATH) != DECOMPOSITION_RECEIPT_SHA256:
        raise TerminalRepairValidationError("decomposition receipt byte identity drift")
    receipt = _strict_json(DECOMPOSITION_RECEIPT_PATH)
    if (
        receipt.get("status") != "PASSED_DIAGNOSTIC"
        or receipt.get("decision")
        != "PHASE8_NO_PRINCIPAL_DECOMPOSITION_REPAIR_IDENTIFIED"
        or receipt.get("candidate_summary", {}).get(
            "svd_right_passed_unchanged_gates"
        )
        is not False
        or receipt.get("candidate_summary", {}).get(
            "svd_left_passed_unchanged_gates"
        )
        is not False
        or receipt.get("scope", {}).get("forecast_executed") is not False
        or receipt.get("scope", {}).get("production_source_modified_by_diagnostic")
        is not False
    ):
        raise TerminalRepairValidationError("decomposition diagnostic decision drift")
    return receipt


def _terminal_summary(terminal: Any) -> dict[str, Any]:
    statuses = [int(value) for value in tf.unstack(terminal.status)]
    tau = terminal.psd_tolerance
    projection_ratio = terminal.projection_residual / tau
    factor_ratio = terminal.factor_reconstruction_residual / tau
    symmetry_ratio = terminal.symmetry_residual / tau
    negative_ratio = tf.maximum(
        tf.zeros_like(terminal.minimum_eigenvalue),
        -terminal.minimum_eigenvalue / tau,
    )
    finite_fields = (
        terminal.mean,
        terminal.raw_covariance,
        terminal.symmetrized_covariance,
        terminal.implemented_covariance,
        terminal.factor,
        terminal.raw_eigenvalues,
        terminal.clipped_eigenvalues,
        terminal.minimum_eigenvalue,
        tau,
        terminal.symmetry_residual,
        terminal.projection_residual,
        terminal.factor_reconstruction_residual,
        terminal.filter_log_likelihood,
        terminal.a1_filter_log_likelihood,
        terminal.target_value,
        terminal.total_value,
        terminal.filter_parity_residual,
        terminal.total_parity_residual,
        terminal.filter_parity_tolerance,
        terminal.total_parity_tolerance,
        terminal.full_parameters,
    )
    all_finite = all(
        bool(tf.reduce_all(tf.math.is_finite(value)).numpy()) for value in finite_fields
    )
    devices = sorted({str(value.device) for value in finite_fields})
    return {
        "point_count": len(statuses),
        "all_statuses_zero": all(value == predictive.STATUS_VALID for value in statuses),
        "nonzero_status_indices": [
            index for index, value in enumerate(statuses) if value != predictive.STATUS_VALID
        ],
        "nonzero_statuses": [value for value in statuses if value != predictive.STATUS_VALID],
        "all_fields_finite": all_finite,
        "minimum_raw_eigenvalue": float(tf.reduce_min(terminal.minimum_eigenvalue)),
        "maximum_negative_eigenvalue_ratio": float(tf.reduce_max(negative_ratio)),
        "maximum_symmetry_ratio": float(tf.reduce_max(symmetry_ratio)),
        "maximum_projection_ratio": float(tf.reduce_max(projection_ratio)),
        "maximum_factor_reconstruction_ratio": float(tf.reduce_max(factor_ratio)),
        "output_devices": devices,
        "passed": (
            all(value == predictive.STATUS_VALID for value in statuses)
            and all_finite
            and float(tf.reduce_max(negative_ratio)) <= 1.0
            and float(tf.reduce_max(symmetry_ratio)) <= 1.0
            and float(tf.reduce_max(projection_ratio)) <= 8.0
            and float(tf.reduce_max(factor_ratio)) <= 16.0
            and bool(devices)
            and all("GPU:" in device for device in devices)
        ),
    }


def run_validation(*, output: Path, wall_cap_seconds: float) -> dict[str, Any]:
    if not math.isfinite(wall_cap_seconds) or wall_cap_seconds <= 0.0:
        raise TerminalRepairValidationError("wall cap must be positive and finite")
    started_at = _now()
    started = time.perf_counter()
    decomposition = validate_decomposition_receipt()
    pilot = _load_module(
        PILOT_SCRIPT_PATH, "phase8_target_pilot_for_terminal_repair_validation"
    )
    entry = pilot.validate_entry_receipts()
    latent, archive_audit = pilot.read_frozen_pilot_prefix(entry["phase7"])
    config = predictive.SSLLSTMForecastConfig()
    config.assert_evidence_config()
    program = predictive.ssl_lstm_terminal_compiled_program(config, 256)
    covariance_program = predictive.ssl_lstm_terminal_covariance_audit_compiled_program(
        256
    )
    charts: dict[str, Any] = {}
    for chart in ("fresh-g", "fresh-h"):
        theta, mapping = pilot.map_pilot_to_theta(chart, latent[chart])
        flat_theta = tf.reshape(theta, [256, 4])
        terminal = predictive.extract_ssl_lstm_terminal_states(flat_theta, config)
        summary = _terminal_summary(terminal)
        if not summary["passed"]:
            raise TerminalRepairValidationError(
                f"terminal orientation repair failed exact-prefix validation: {chart} {summary}"
            )
        charts[chart] = {"mapping": mapping, "terminal": summary}
        if time.perf_counter() - started > wall_cap_seconds:
            raise TerminalRepairValidationError("terminal validation wall cap exceeded")
    trace_counts = {
        "terminal_filter": _trace_count(program),
        "staged_covariance_audit": _trace_count(covariance_program),
    }
    if trace_counts != {"terminal_filter": 1, "staged_covariance_audit": 1}:
        raise TerminalRepairValidationError(
            f"staged terminal validation retraced: {trace_counts}"
        )
    wall_time = time.perf_counter() - started
    payload = {
        "schema": "bayesfilter.ssl_lstm_neutra.phase8_staged_covariance_audit_validation.v1",
        "status": "PASSED",
        "decision": "PHASE8_STAGED_COVARIANCE_AUDIT_VALIDATED_PILOT_REPAIR_ELIGIBLE",
        "decomposition_binding": {
            "path": DECOMPOSITION_RECEIPT_PATH.as_posix(),
            "sha256": DECOMPOSITION_RECEIPT_SHA256,
            "decision": decomposition["decision"],
            "original_failed_g_indices": decomposition[
                "prior_diagnostic_binding"
            ]["failure_indices"],
        },
        "scope": {
            "charts_evaluated": ["fresh-g", "fresh-h"],
            "points_per_chart": 256,
            "total_terminal_points": 512,
            "forecast_executed": False,
            "predictive_difference_computed": False,
            "confirmation_suffix_selected": False,
            "target_pilot_retried": False,
        },
        "archive_integrity": archive_audit,
        "charts": charts,
        "compile_trace_counts": trace_counts,
        "repair_contract": {
            "filter_stage_still_returns_raw_covariance_and_parity": True,
            "covariance_audit_is_separate_batched_xla_program": True,
            "only_covariance_derived_fields_and_status_bits_are_replaced": True,
        },
        "unchanged_gates": {
            "covariance_roundoff_multiplier": predictive.COVARIANCE_ROUNDOFF_MULTIPLIER,
            "projection_multiplier": 8.0,
            "factor_reconstruction_multiplier": 16.0,
            "material_negative_ratio_max": 1.0,
        },
        "source_bindings": {
            "plan": {"path": PLAN_PATH.as_posix(), "sha256": _sha256(PLAN_PATH)},
            "runner": {"path": SCRIPT_PATH.as_posix(), "sha256": _sha256(SCRIPT_PATH)},
            "pilot_runner": {
                "path": PILOT_SCRIPT_PATH.as_posix(),
                "sha256": _sha256(PILOT_SCRIPT_PATH),
            },
            "diagnostic_runner": {
                "path": DIAGNOSTIC_SCRIPT_PATH.as_posix(),
                "sha256": _sha256(DIAGNOSTIC_SCRIPT_PATH),
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
            "staged terminal covariance audit validation only; no forecast executed",
            "no predictive equivalence, sampler ranking, posterior, or model claim",
            "a pass only makes a separately recorded target-pilot repair eligible",
            "the failed pilot and rejected orientation candidate remain authoritative history",
        ],
    }
    _write_json(output, payload)
    return payload


def _require_gpu() -> None:
    physical = tf.config.list_physical_devices("GPU")
    if not physical:
        raise TerminalRepairValidationError("terminal validation requires a visible GPU")
    for gpu in physical:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    if not tf.config.list_logical_devices("GPU"):
        raise TerminalRepairValidationError("terminal validation requires a logical GPU")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-cap-seconds", type=float, required=True)
    args = parser.parse_args(argv)
    _require_gpu()
    with tf.device("/GPU:0"):
        payload = run_validation(
            output=args.output, wall_cap_seconds=float(args.wall_cap_seconds)
        )
    print(
        "JSON_SUMMARY "
        + json.dumps(
            {
                "decision": payload["decision"],
                "g_projection_ratio": payload["charts"]["fresh-g"]["terminal"][
                    "maximum_projection_ratio"
                ],
                "h_projection_ratio": payload["charts"]["fresh-h"]["terminal"][
                    "maximum_projection_ratio"
                ],
                "wall_time_seconds": payload["run_manifest"]["wall_time_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
