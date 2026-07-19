#!/usr/bin/env python3
"""Compare GPU/XLA decompositions on exact failing Phase 8 covariances."""

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
DIAGNOSTIC_RECEIPT_PATH = Path(
    "docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/"
    "phase-8-predictive-design/terminal-projection-diagnostic.json"
)
DIAGNOSTIC_RECEIPT_SHA256 = (
    "ea2ded5e9e3321c18048a4306606c3d2dcd12fbc728050152ebef0ba521c0bcc"
)
EXPECTED_FAILURE_INDICES = (33, 68, 144, 189, 200, 201)


class RawDecompositionDiagnosticError(RuntimeError):
    """Raised when the decomposition diagnostic is invalid."""


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(_absolute(path).read_bytes()).hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, _absolute(path))
    if spec is None or spec.loader is None:
        raise RawDecompositionDiagnosticError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _strict_json(path: Path) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            if key in result:
                raise RawDecompositionDiagnosticError(
                    f"duplicate JSON key {key!r}: {path}"
                )
            result[key] = value
        return result

    def reject(value: str) -> None:
        raise RawDecompositionDiagnosticError(
            f"nonfinite JSON constant {value!r}: {path}"
        )

    value = json.loads(
        _absolute(path).read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=reject,
    )
    if not isinstance(value, dict):
        raise RawDecompositionDiagnosticError(f"expected JSON object: {path}")
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
        raise RawDecompositionDiagnosticError(f"refusing to overwrite receipt: {path}")
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


def validate_diagnostic_receipt() -> dict[str, Any]:
    if _sha256(DIAGNOSTIC_RECEIPT_PATH) != DIAGNOSTIC_RECEIPT_SHA256:
        raise RawDecompositionDiagnosticError("terminal diagnostic byte identity drift")
    receipt = _strict_json(DIAGNOSTIC_RECEIPT_PATH)
    if (
        receipt.get("decision")
        != "PHASE8_TERMINAL_PROJECTION_TOLERANCE_REPAIR_CANDIDATE"
        or tuple(receipt.get("gpu_xla_audit", {}).get("failure_indices", ()))
        != EXPECTED_FAILURE_INDICES
        or receipt.get("classification", {}).get("projection_only") is not True
        or receipt.get("scope", {}).get("forecast_executed") is not False
    ):
        raise RawDecompositionDiagnosticError("terminal diagnostic state drift")
    return receipt


def _frobenius(matrix: tf.Tensor) -> tf.Tensor:
    return tf.sqrt(tf.reduce_sum(tf.square(matrix), axis=[-2, -1]))


def _decomposition_kernel(raw_covariances: tf.Tensor) -> tuple[tf.Tensor, ...]:
    sym = 0.5 * (raw_covariances + tf.transpose(raw_covariances, [0, 2, 1]))
    scale = tf.maximum(tf.ones([tf.shape(sym)[0]], tf.float64), _frobenius(sym))
    tau = tf.constant(
        predictive.COVARIANCE_ROUNDOFF_MULTIPLIER * predictive.FLOAT64_EPSILON,
        tf.float64,
    ) * scale

    eigenvalues, eigenvectors = tf.linalg.eigh(sym)
    diagonal = tf.linalg.diag(eigenvalues)
    transpose_eigenvectors = tf.transpose(eigenvectors, [0, 2, 1])
    eigen_column = eigenvectors @ diagonal @ transpose_eigenvectors
    eigen_row = transpose_eigenvectors @ diagonal @ eigenvectors
    eigen_column_residual = _frobenius(eigen_column - sym) / tau
    eigen_row_residual = _frobenius(eigen_row - sym) / tau
    eigen_equation_column = _frobenius(sym @ eigenvectors - eigenvectors @ diagonal) / tau
    eigen_equation_row = _frobenius(
        sym @ transpose_eigenvectors - transpose_eigenvectors @ diagonal
    ) / tau
    eigen_orthogonality = _frobenius(
        transpose_eigenvectors @ eigenvectors
        - tf.eye(3, batch_shape=[tf.shape(sym)[0]], dtype=tf.float64)
    ) / tau

    singular_values, left_vectors, right_vectors = tf.linalg.svd(
        sym, full_matrices=True, compute_uv=True
    )
    diagonal_singular = tf.linalg.diag(singular_values)
    transpose_right = tf.transpose(right_vectors, [0, 2, 1])
    svd_reconstruction = left_vectors @ diagonal_singular @ transpose_right
    svd_reconstruction_residual = _frobenius(svd_reconstruction - sym) / tau
    svd_left_orthogonality = _frobenius(
        tf.transpose(left_vectors, [0, 2, 1]) @ left_vectors
        - tf.eye(3, batch_shape=[tf.shape(sym)[0]], dtype=tf.float64)
    ) / tau
    svd_right_orthogonality = _frobenius(
        transpose_right @ right_vectors
        - tf.eye(3, batch_shape=[tf.shape(sym)[0]], dtype=tf.float64)
    ) / tau

    right_rotated = transpose_right @ sym @ right_vectors
    right_signed_values = tf.linalg.diag_part(right_rotated)
    right_off_diagonal = right_rotated - tf.linalg.diag(right_signed_values)
    right_eigen_equation = _frobenius(
        sym @ right_vectors - right_vectors @ tf.linalg.diag(right_signed_values)
    ) / tau
    right_off_diagonal_residual = _frobenius(right_off_diagonal) / tau
    right_clipped = tf.maximum(right_signed_values, tf.zeros_like(right_signed_values))
    right_covariance = right_vectors @ tf.linalg.diag(right_clipped) @ transpose_right
    right_factor = (
        right_vectors
        @ tf.linalg.diag(tf.sqrt(right_clipped))
        @ transpose_right
    )
    right_projection_residual = _frobenius(right_covariance - sym) / tau
    right_factor_residual = _frobenius(
        right_factor @ tf.transpose(right_factor, [0, 2, 1]) - right_covariance
    ) / tau

    transpose_left = tf.transpose(left_vectors, [0, 2, 1])
    left_rotated = transpose_left @ sym @ left_vectors
    left_signed_values = tf.linalg.diag_part(left_rotated)
    left_off_diagonal = left_rotated - tf.linalg.diag(left_signed_values)
    left_eigen_equation = _frobenius(
        sym @ left_vectors - left_vectors @ tf.linalg.diag(left_signed_values)
    ) / tau
    left_off_diagonal_residual = _frobenius(left_off_diagonal) / tau
    left_clipped = tf.maximum(left_signed_values, tf.zeros_like(left_signed_values))
    left_covariance = left_vectors @ tf.linalg.diag(left_clipped) @ transpose_left
    left_factor = (
        left_vectors @ tf.linalg.diag(tf.sqrt(left_clipped)) @ transpose_left
    )
    left_projection_residual = _frobenius(left_covariance - sym) / tau
    left_factor_residual = _frobenius(
        left_factor @ tf.transpose(left_factor, [0, 2, 1]) - left_covariance
    ) / tau

    cholesky = tf.linalg.cholesky(sym)
    cholesky_residual = _frobenius(
        cholesky @ tf.transpose(cholesky, [0, 2, 1]) - sym
    ) / tau
    return (
        tau,
        eigenvalues,
        eigen_column_residual,
        eigen_row_residual,
        eigen_equation_column,
        eigen_equation_row,
        eigen_orthogonality,
        singular_values,
        svd_reconstruction_residual,
        svd_left_orthogonality,
        svd_right_orthogonality,
        right_signed_values,
        right_eigen_equation,
        right_off_diagonal_residual,
        right_projection_residual,
        right_factor_residual,
        left_signed_values,
        left_eigen_equation,
        left_off_diagonal_residual,
        left_projection_residual,
        left_factor_residual,
        cholesky_residual,
    )


_DECOMPOSITION_PROGRAM = tf.function(
    _decomposition_kernel,
    input_signature=[tf.TensorSpec([256, 3, 3], tf.float64)],
    autograph=False,
    jit_compile=True,
    reduce_retracing=True,
)


def _rows(outputs: tuple[tf.Tensor, ...], indices: tuple[int, ...]) -> list[dict[str, Any]]:
    names = (
        "tau",
        "eigenvalues",
        "eigen_column_residual_ratio",
        "eigen_row_residual_ratio",
        "eigen_equation_column_ratio",
        "eigen_equation_row_ratio",
        "eigen_orthogonality_ratio",
        "singular_values",
        "svd_reconstruction_residual_ratio",
        "svd_left_orthogonality_ratio",
        "svd_right_orthogonality_ratio",
        "svd_right_signed_values",
        "svd_right_eigen_equation_ratio",
        "svd_right_off_diagonal_ratio",
        "svd_right_projection_ratio",
        "svd_right_factor_ratio",
        "svd_left_signed_values",
        "svd_left_eigen_equation_ratio",
        "svd_left_off_diagonal_ratio",
        "svd_left_projection_ratio",
        "svd_left_factor_ratio",
        "cholesky_reconstruction_ratio",
    )
    rows = []
    for index in indices:
        rows.append(
            {
                "index": index,
                **{
                    name: _json_safe(value[index])
                    for name, value in zip(names, outputs, strict=True)
                },
            }
        )
    return rows


def run_diagnostic(*, output: Path, wall_cap_seconds: float) -> dict[str, Any]:
    if not math.isfinite(wall_cap_seconds) or wall_cap_seconds <= 0.0:
        raise RawDecompositionDiagnosticError("wall cap must be positive and finite")
    started_at = _now()
    started = time.perf_counter()
    prior = validate_diagnostic_receipt()
    pilot = _load_module(
        PILOT_SCRIPT_PATH, "phase8_target_pilot_for_raw_decomposition_diagnostic"
    )
    entry = pilot.validate_entry_receipts()
    latent, archive_audit = pilot.read_frozen_pilot_prefix(entry["phase7"])
    theta, mapping = pilot.map_pilot_to_theta("fresh-g", latent["fresh-g"])
    flat_theta = tf.reshape(theta, [256, 4])
    config = predictive.SSLLSTMForecastConfig()
    config.assert_evidence_config()
    terminal_program = predictive.ssl_lstm_terminal_compiled_program(config, 256)
    terminal_tensors = tuple(terminal_program(flat_theta))
    terminal = predictive._terminal_from_tensors(terminal_tensors)
    statuses = tuple(int(value) for value in tf.unstack(terminal.status))
    observed = tuple(index for index, value in enumerate(statuses) if value != 0)
    if observed != EXPECTED_FAILURE_INDICES or any(
        statuses[index] != predictive.STATUS_PROJECTION for index in observed
    ):
        raise RawDecompositionDiagnosticError(
            f"exact projection-only failures did not reproduce: {observed}"
        )
    outputs = tuple(_DECOMPOSITION_PROGRAM(terminal.raw_covariance))
    if not all("GPU:" in str(value.device) for value in outputs):
        raise RawDecompositionDiagnosticError("decomposition outputs are not GPU resident")
    trace_counts = {
        "terminal": _trace_count(terminal_program),
        "decomposition": _trace_count(_DECOMPOSITION_PROGRAM),
    }
    if trace_counts != {"terminal": 1, "decomposition": 1}:
        raise RawDecompositionDiagnosticError(
            f"decomposition diagnostic trace gate failed: {trace_counts}"
        )
    rows = _rows(outputs, EXPECTED_FAILURE_INDICES)
    all_finite = all(
        bool(tf.reduce_all(tf.math.is_finite(value)).numpy()) for value in outputs
    )
    if not all_finite:
        raise RawDecompositionDiagnosticError("decomposition diagnostic is nonfinite")
    right_pass = all(
        row["svd_right_projection_ratio"] <= 8.0
        and row["svd_right_factor_ratio"] <= 16.0
        and row["svd_right_eigen_equation_ratio"] <= 8.0
        for row in rows
    )
    left_pass = all(
        row["svd_left_projection_ratio"] <= 8.0
        and row["svd_left_factor_ratio"] <= 16.0
        and row["svd_left_eigen_equation_ratio"] <= 8.0
        for row in rows
    )
    if right_pass and not left_pass:
        decision = "PHASE8_SVD_RIGHT_VECTOR_PRINCIPAL_REPAIR_CANDIDATE"
    elif left_pass and not right_pass:
        decision = "PHASE8_SVD_LEFT_VECTOR_PRINCIPAL_REPAIR_CANDIDATE"
    elif right_pass and left_pass:
        decision = "PHASE8_SVD_BOTH_VECTOR_REPAIR_CANDIDATES"
    else:
        decision = "PHASE8_NO_PRINCIPAL_DECOMPOSITION_REPAIR_IDENTIFIED"
    if time.perf_counter() - started > wall_cap_seconds:
        raise RawDecompositionDiagnosticError("decomposition diagnostic wall cap exceeded")
    wall_time = time.perf_counter() - started
    payload = {
        "schema": "bayesfilter.ssl_lstm_neutra.phase8_raw_covariance_decomposition_diagnostic.v1",
        "status": "PASSED_DIAGNOSTIC",
        "decision": decision,
        "prior_diagnostic_binding": {
            "path": DIAGNOSTIC_RECEIPT_PATH.as_posix(),
            "sha256": DIAGNOSTIC_RECEIPT_SHA256,
            "failure_indices": prior["gpu_xla_audit"]["failure_indices"],
        },
        "scope": {
            "chart_mapped_and_terminal_evaluated": "fresh-g",
            "failing_indices_compared": list(EXPECTED_FAILURE_INDICES),
            "h_mapped_or_evaluated": False,
            "forecast_executed": False,
            "predictive_difference_computed": False,
            "confirmation_suffix_selected": False,
            "production_source_modified_by_diagnostic": False,
        },
        "mapping": mapping,
        "archive_integrity": archive_audit,
        "failure_rows": rows,
        "candidate_summary": {
            "svd_right_passed_unchanged_gates": right_pass,
            "svd_left_passed_unchanged_gates": left_pass,
            "projection_ratio_max": 8.0,
            "factor_ratio_max": 16.0,
            "eigen_equation_ratio_max": 8.0,
            "all_outputs_finite": all_finite,
        },
        "compile_trace_counts": trace_counts,
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
            "raw covariance decomposition localization only",
            "no forecast, predictive comparison, equivalence, sampler, or posterior claim",
            "a candidate diagnostic does not authorize production implementation",
            "no tolerance was changed and the failed pilot was not retried",
        ],
    }
    _write_json(output, payload)
    return payload


def _require_gpu() -> None:
    physical = tf.config.list_physical_devices("GPU")
    if not physical:
        raise RawDecompositionDiagnosticError("decomposition diagnostic requires a GPU")
    for gpu in physical:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    if not tf.config.list_logical_devices("GPU"):
        raise RawDecompositionDiagnosticError("decomposition diagnostic requires a logical GPU")


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
                "right_pass": payload["candidate_summary"][
                    "svd_right_passed_unchanged_gates"
                ],
                "left_pass": payload["candidate_summary"][
                    "svd_left_passed_unchanged_gates"
                ],
                "wall_time_seconds": payload["run_manifest"]["wall_time_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
