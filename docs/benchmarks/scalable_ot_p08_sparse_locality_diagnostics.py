"""Phase 8 sparse/locality diagnostics for dense Phase 1 transport plans.

This script replays the Phase 1 TensorFlow dense baseline fixtures and measures
whether their materialized transport matrices have enough row-wise support
concentration to justify a later sparse/screened/localized prototype.

It is not a sparse solver and it does not make speedup, ranking, posterior
correctness, or default-readiness claims.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


_PRE_PARSER = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
_PRE_PARSER.add_argument("--device-scope", choices=("cpu", "visible"), default="cpu")
_PRE_PARSER.add_argument("--cuda-visible-devices", default=None)
_PRE_ARGS, _UNKNOWN = _PRE_PARSER.parse_known_args()
if _PRE_ARGS.cuda_visible_devices is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _PRE_ARGS.cuda_visible_devices
elif _PRE_ARGS.device_scope == "cpu":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import tensorflow as tf

from docs.benchmarks.scalable_ot_p01_baseline_fixture_diagnostics import _fixtures
from experiments.dpf_implementation.tf_tfp.resampling import annealed_transport_tf
from experiments.dpf_implementation.tf_tfp.resampling.annealed_transport_tf import (
    annealed_transport_resample_tf,
)


DTYPE = tf.float64
annealed_transport_tf.DTYPE = DTYPE

MASS_THRESHOLDS = (0.90, 0.95, 0.99, 0.999)
TRUNCATION_MASS_THRESHOLD = 0.99
ROW_RESIDUAL_THRESHOLD = 5.0e-3
COLUMN_RESIDUAL_THRESHOLD = 5.0e-2
PARTICLE_ERROR_THRESHOLD = 5.0e-2

NONCLAIMS = (
    "Phase 8 dense-plan locality diagnostic only",
    "not a sparse solver implementation",
    "no sparse speedup claim",
    "no ranking claim",
    "no production default change",
    "no posterior correctness claim",
    "no general scalability claim",
    "source availability is not locality evidence",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--epsilon", type=float, default=0.5)
    parser.add_argument("--scaling", type=float, default=0.9)
    parser.add_argument("--convergence-threshold", type=float, default=1.0e-3)
    parser.add_argument("--max-iterations", type=int, default=12)
    parser.add_argument("--row-chunk-size", type=int, default=4)
    parser.add_argument("--col-chunk-size", type=int, default=4)
    parser.add_argument("--device", default="/CPU:0")
    parser.add_argument("--device-scope", choices=("cpu", "visible"), default=_PRE_ARGS.device_scope)
    parser.add_argument("--cuda-visible-devices", default=_PRE_ARGS.cuda_visible_devices)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output", required=True)
    args = parser.parse_args()
    if args.epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if not 0.0 < args.scaling <= 1.0:
        raise ValueError("scaling must be in (0, 1]")
    if args.convergence_threshold <= 0.0:
        raise ValueError("convergence_threshold must be positive")
    if args.max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if args.row_chunk_size <= 0 or args.col_chunk_size <= 0:
        raise ValueError("chunk sizes must be positive")
    return args


def _git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return completed.stdout.strip()


def _float(value: Any) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _tensor_finite(value: tf.Tensor) -> bool:
    return bool(tf.reduce_all(tf.math.is_finite(value)).numpy())


def _stable_prefix_count(row: np.ndarray, threshold: float) -> int:
    row_mass = float(np.sum(row))
    if row_mass <= 0.0:
        return 0
    order = np.argsort(-row, kind="mergesort")
    cumulative = np.cumsum(row[order])
    index = int(np.searchsorted(cumulative, threshold * row_mass, side="left"))
    return min(index + 1, row.shape[0])


def _support_summary(counts: np.ndarray, n_particles: int) -> dict[str, Any]:
    flat = counts.reshape(-1)
    return {
        "median": float(np.median(flat)),
        "p90": float(np.percentile(flat, 90.0)),
        "max": int(np.max(flat)),
        "mean": float(np.mean(flat)),
        "median_fraction_of_n": float(np.median(flat) / n_particles),
        "p90_fraction_of_n": float(np.percentile(flat, 90.0) / n_particles),
        "per_row_counts": flat.astype(int).tolist(),
    }


def _support_count_diagnostics(transport: np.ndarray) -> dict[str, Any]:
    batch_size, num_rows, num_cols = transport.shape
    by_threshold: dict[str, Any] = {}
    counts_by_threshold: dict[float, np.ndarray] = {}
    for threshold in MASS_THRESHOLDS:
        counts = np.zeros([batch_size, num_rows], dtype=np.int64)
        for batch in range(batch_size):
            for row_index in range(num_rows):
                counts[batch, row_index] = _stable_prefix_count(
                    transport[batch, row_index, :],
                    threshold,
                )
        counts_by_threshold[threshold] = counts
        by_threshold[f"{threshold:.3f}"] = _support_summary(counts, num_cols)
    return {
        "definition": (
            "For each row, masses are sorted descending with numpy mergesort; "
            "k_i(t) is the first stable prefix whose cumulative mass reaches "
            "t times that row mass; ties are not expanded beyond that prefix."
        ),
        "by_threshold": by_threshold,
        "counts_99": counts_by_threshold[TRUNCATION_MASS_THRESHOLD],
    }


def _nearest_neighbor_mass_diagnostics(
    particles: np.ndarray,
    transport: np.ndarray,
) -> dict[str, Any]:
    batch_size, num_particles, _state_dim = particles.shape
    k_values = sorted(
        {
            1,
            2,
            4,
            8,
            int(math.ceil(0.25 * num_particles)),
            int(math.ceil(0.50 * num_particles)),
            num_particles,
        }
    )
    k_values = [k for k in k_values if 1 <= k <= num_particles]
    mass_by_k: dict[str, list[float]] = {str(k): [] for k in k_values}
    for batch in range(batch_size):
        diff = particles[batch, :, None, :] - particles[batch, None, :, :]
        distances = np.linalg.norm(diff, axis=2)
        for row_index in range(num_particles):
            row = transport[batch, row_index, :]
            row_mass = float(np.sum(row))
            order = np.argsort(distances[row_index, :], kind="mergesort")
            for k in k_values:
                captured = float(np.sum(row[order[:k]]))
                fraction = captured / row_mass if row_mass > 0.0 else 0.0
                mass_by_k[str(k)].append(fraction)
    return {
        "definition": (
            "For each target row, source columns are sorted by Euclidean "
            "distance from the original row particle; reported values are "
            "retained transport-mass fractions."
        ),
        "k_values": k_values,
        "by_k": {
            key: {
                "min": float(np.min(values)),
                "median": float(np.median(values)),
                "p10": float(np.percentile(values, 10.0)),
                "mean": float(np.mean(values)),
            }
            for key, values in mass_by_k.items()
        },
    }


def _truncate_rows_at_mass(
    transport: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    batch_size, num_rows, num_cols = transport.shape
    truncated = np.zeros_like(transport)
    counts = np.zeros([batch_size, num_rows], dtype=np.int64)
    for batch in range(batch_size):
        for row_index in range(num_rows):
            row = transport[batch, row_index, :]
            row_mass = float(np.sum(row))
            count = _stable_prefix_count(row, threshold)
            counts[batch, row_index] = count
            if count == 0:
                continue
            order = np.argsort(-row, kind="mergesort")
            keep = order[:count]
            truncated[batch, row_index, keep] = row[keep]
            retained_mass = float(np.sum(truncated[batch, row_index, :]))
            if retained_mass > 0.0:
                truncated[batch, row_index, :] *= row_mass / retained_mass
    return truncated, counts


def _run_dense(
    particles_np: np.ndarray,
    log_weights_np: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    particles = tf.constant(particles_np, dtype=DTYPE)
    log_weights = tf.constant(log_weights_np, dtype=DTYPE)
    start = time.perf_counter()
    with tf.device(args.device):
        result = annealed_transport_resample_tf(
            particles,
            log_weights,
            epsilon=args.epsilon,
            scaling=args.scaling,
            convergence_threshold=args.convergence_threshold,
            max_iterations=args.max_iterations,
            transport_gradient_mode="raw",
            transport_plan_mode="dense",
            row_chunk_size=args.row_chunk_size,
            col_chunk_size=args.col_chunk_size,
        )
    wall_time = time.perf_counter() - start
    transport = tf.convert_to_tensor(result.transport_matrix, dtype=DTYPE)
    transported = tf.convert_to_tensor(result.particles, dtype=DTYPE)
    source_weights = tf.exp(log_weights)
    num_particles = tf.cast(tf.shape(log_weights)[1], DTYPE)
    row_residual = tf.reduce_max(tf.abs(tf.reduce_sum(transport, axis=2) - 1.0))
    column_target = source_weights * num_particles
    column_residual = tf.reduce_max(tf.abs(tf.reduce_sum(transport, axis=1) - column_target))
    dense_record = {
        "wall_time_seconds": wall_time,
        "transport_matrix_shape": transport.shape.as_list(),
        "particles_shape": transported.shape.as_list(),
        "finite_transport_matrix": _tensor_finite(transport),
        "finite_particles": _tensor_finite(transported),
        "max_row_residual": _float(row_residual),
        "max_column_residual": _float(column_residual),
        "diagnostics": dict(result.diagnostics),
    }
    return dense_record, transport.numpy(), transported.numpy()


def _truncation_diagnostics(
    particles: np.ndarray,
    log_weights: np.ndarray,
    transport: np.ndarray,
    dense_particles: np.ndarray,
    counts_99: np.ndarray,
) -> dict[str, Any]:
    truncated, truncation_counts = _truncate_rows_at_mass(
        transport,
        TRUNCATION_MASS_THRESHOLD,
    )
    if not np.array_equal(counts_99, truncation_counts):
        raise AssertionError("99% support counts changed between support and truncation diagnostics")
    num_particles = transport.shape[2]
    source_weights = np.exp(log_weights)
    column_target = source_weights * float(num_particles)
    row_residual = float(np.max(np.abs(np.sum(truncated, axis=2) - 1.0)))
    column_residual = float(np.max(np.abs(np.sum(truncated, axis=1) - column_target)))
    truncated_particles = np.matmul(truncated, particles)
    diff = truncated_particles - dense_particles
    return {
        "truncation_mass_threshold": TRUNCATION_MASS_THRESHOLD,
        "definition": (
            "Retain the stable minimal per-row 99% mass support, zero all "
            "other entries, then renormalize each retained row to its original "
            "dense row sum before applying the matrix to particles."
        ),
        "max_row_residual": row_residual,
        "max_column_residual": column_residual,
        "max_transported_particle_error": float(np.max(np.abs(diff))),
        "rms_transported_particle_error": float(np.sqrt(np.mean(np.square(diff)))),
        "finite_truncated_matrix": bool(np.all(np.isfinite(truncated))),
        "finite_truncated_particles": bool(np.all(np.isfinite(truncated_particles))),
        "nonzero_entries": int(np.count_nonzero(truncated)),
        "dense_entries": int(np.prod(transport.shape)),
        "nonzero_fraction_of_dense": float(np.count_nonzero(truncated) / np.prod(transport.shape)),
    }


def _fixture_diagnostics(
    fixture_name: str,
    particles_np: np.ndarray,
    log_weights_np: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    dense_record, transport, dense_particles = _run_dense(particles_np, log_weights_np, args)
    support = _support_count_diagnostics(transport)
    nearest = _nearest_neighbor_mass_diagnostics(particles_np, transport)
    truncation = _truncation_diagnostics(
        particles_np,
        log_weights_np,
        transport,
        dense_particles,
        support["counts_99"],
    )
    n_particles = int(transport.shape[2])
    median_threshold = max(8, int(math.ceil(0.25 * n_particles)))
    p90_threshold = max(16, int(math.ceil(0.50 * n_particles)))
    support_99 = support["by_threshold"]["0.990"]
    threshold_checks = {
        "n_particles": n_particles,
        "median_99_support_threshold": median_threshold,
        "p90_99_support_threshold": p90_threshold,
        "median_99_support_pass": bool(support_99["median"] <= median_threshold),
        "p90_99_support_pass": bool(support_99["p90"] <= p90_threshold),
        "row_residual_threshold": ROW_RESIDUAL_THRESHOLD,
        "column_residual_threshold": COLUMN_RESIDUAL_THRESHOLD,
        "particle_error_threshold": PARTICLE_ERROR_THRESHOLD,
        "truncated_row_residual_pass": bool(truncation["max_row_residual"] <= ROW_RESIDUAL_THRESHOLD),
        "truncated_column_residual_pass": bool(
            truncation["max_column_residual"] <= COLUMN_RESIDUAL_THRESHOLD
        ),
        "truncated_particle_error_pass": bool(
            truncation["max_transported_particle_error"] <= PARTICLE_ERROR_THRESHOLD
        ),
        "finite_dense_and_truncated_pass": bool(
            dense_record["finite_transport_matrix"]
            and dense_record["finite_particles"]
            and truncation["finite_truncated_matrix"]
            and truncation["finite_truncated_particles"]
        ),
    }
    threshold_checks["advance_thresholds_pass"] = bool(
        threshold_checks["median_99_support_pass"]
        and threshold_checks["p90_99_support_pass"]
        and threshold_checks["truncated_row_residual_pass"]
        and threshold_checks["truncated_column_residual_pass"]
        and threshold_checks["truncated_particle_error_pass"]
        and threshold_checks["finite_dense_and_truncated_pass"]
    )
    support["counts_99"] = support["counts_99"].astype(int).tolist()
    return {
        "input_shape": {
            "particles": list(particles_np.shape),
            "log_weights": list(log_weights_np.shape),
        },
        "dense": dense_record,
        "support_count_diagnostics": support,
        "nearest_neighbor_mass_diagnostics": nearest,
        "truncation_diagnostics": truncation,
        "threshold_checks": threshold_checks,
        "fixture_decision": (
            "advance_sparse_prototype_candidate"
            if threshold_checks["advance_thresholds_pass"]
            else "SPARSE_LOCALITY_DIAGNOSTIC_BLOCKS_SPARSE_IMPLEMENTATION_FOR_NOW"
        ),
        "fixture_name": fixture_name,
    }


def _build_result(args: argparse.Namespace) -> dict[str, Any]:
    fixture_results: dict[str, Any] = {}
    hard_vetoes: list[str] = []
    promotion_vetoes: list[str] = []
    for fixture_name, (particles_np, log_weights_np) in _fixtures().items():
        fixture = _fixture_diagnostics(fixture_name, particles_np, log_weights_np, args)
        fixture_results[fixture_name] = fixture
        checks = fixture["threshold_checks"]
        if not checks["finite_dense_and_truncated_pass"]:
            hard_vetoes.append(f"{fixture_name}:nonfinite_dense_or_truncated_artifact")
        if not checks["advance_thresholds_pass"]:
            if not checks["median_99_support_pass"]:
                promotion_vetoes.append(f"{fixture_name}:median_99_support_too_large")
            if not checks["p90_99_support_pass"]:
                promotion_vetoes.append(f"{fixture_name}:p90_99_support_too_large")
            if not checks["truncated_row_residual_pass"]:
                promotion_vetoes.append(f"{fixture_name}:truncated_row_residual_too_large")
            if not checks["truncated_column_residual_pass"]:
                promotion_vetoes.append(f"{fixture_name}:truncated_column_residual_too_large")
            if not checks["truncated_particle_error_pass"]:
                promotion_vetoes.append(f"{fixture_name}:truncated_particle_error_too_large")

    diagnostic_completed = not hard_vetoes
    advance = diagnostic_completed and not promotion_vetoes
    phase8_decision = (
        "advance_sparse_prototype_candidate"
        if advance
        else "SPARSE_LOCALITY_DIAGNOSTIC_BLOCKS_SPARSE_IMPLEMENTATION_FOR_NOW"
    )
    status = "PASS" if diagnostic_completed else "FAIL"
    phase8_status = (
        "PHASE_8_SPARSE_LOCALITY_DIAGNOSTIC_COMPLETED_ADVANCE_SPARSE_PROTOTYPE_CANDIDATE"
        if advance
        else (
            "PHASE_8_SPARSE_LOCALITY_DIAGNOSTIC_COMPLETED_BLOCKS_SPARSE_IMPLEMENTATION_FOR_NOW"
            if diagnostic_completed
            else "PHASE_8_SPARSE_LOCALITY_DIAGNOSTIC_FAILED_HARD_VETO"
        )
    )
    summary = {
        "max_dense_row_residual": max(
            fixture["dense"]["max_row_residual"] for fixture in fixture_results.values()
        ),
        "max_dense_column_residual": max(
            fixture["dense"]["max_column_residual"] for fixture in fixture_results.values()
        ),
        "max_truncated_row_residual": max(
            fixture["truncation_diagnostics"]["max_row_residual"]
            for fixture in fixture_results.values()
        ),
        "max_truncated_column_residual": max(
            fixture["truncation_diagnostics"]["max_column_residual"]
            for fixture in fixture_results.values()
        ),
        "max_truncated_particle_error": max(
            fixture["truncation_diagnostics"]["max_transported_particle_error"]
            for fixture in fixture_results.values()
        ),
        "max_99_support_p90_fraction_of_n": max(
            fixture["support_count_diagnostics"]["by_threshold"]["0.990"]["p90_fraction_of_n"]
            for fixture in fixture_results.values()
        ),
        "min_99_support_median_fraction_of_n": min(
            fixture["support_count_diagnostics"]["by_threshold"]["0.990"]["median_fraction_of_n"]
            for fixture in fixture_results.values()
        ),
    }
    return {
        "phase8_status": phase8_status,
        "status": status,
        "phase8_decision": phase8_decision,
        "diagnostic_completed": diagnostic_completed,
        "hard_vetoes": hard_vetoes,
        "promotion_vetoes": promotion_vetoes,
        "semantic_class": "reference_only_diagnostic",
        "source_route": "source_reference_only",
        "implementation_scope": "dense_plan_locality_and_truncation_diagnostic",
        "baseline_comparator": "phase1_dense_streaming_baseline_2026_06_17",
        "thresholds": {
            "median_99_support": "max(8, ceil(0.25 * N))",
            "p90_99_support": "max(16, ceil(0.50 * N))",
            "truncated_max_row_residual": ROW_RESIDUAL_THRESHOLD,
            "truncated_max_column_residual": COLUMN_RESIDUAL_THRESHOLD,
            "truncated_max_transported_particle_error": PARTICLE_ERROR_THRESHOLD,
            "finite_dense_and_truncated_particles": True,
        },
        "definitions": {
            "N": "transport_matrix.shape[2], the source-particle count",
            "orientation": "target rows, source columns, Phase 1 scaled dense transport",
            "minimal_per_row_99_support": (
                "stable descending row-mass prefix whose cumulative mass first "
                "reaches 99% of the dense row mass; ties are not expanded"
            ),
            "truncated_transport_role": (
                "diagnostic-only materialization used to evaluate locality; "
                "not a sparse solver implementation"
            ),
        },
        "summary": summary,
        "settings": {
            "epsilon": args.epsilon,
            "scaling": args.scaling,
            "convergence_threshold": args.convergence_threshold,
            "max_iterations": args.max_iterations,
            "row_chunk_size": args.row_chunk_size,
            "col_chunk_size": args.col_chunk_size,
            "transport_gradient_mode": "raw",
            "transport_plan_mode": "dense",
        },
        "manifest": {
            "git_commit": _git_commit(),
            "timestamp_utc": _dt.datetime.now(tz=_dt.UTC).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "tensorflow_version": tf.__version__,
            "device_scope": args.device_scope,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device": args.device,
            "dtype": "tf.float64",
            "command": "scalable_ot_p08_sparse_locality_diagnostics.py",
            "plan_path": (
                "docs/plans/bayesfilter-dpf-ledh-pfpf-ot-scalable-ot-"
                "p08-sparse-localized-diagnostic-subplan-2026-06-17.md"
            ),
        },
        "fixtures": fixture_results,
        "diagnostic_roles": {
            "finite_dense_and_truncated_pass": "hard_veto",
            "median_99_support_pass": "promotion_veto",
            "p90_99_support_pass": "promotion_veto",
            "truncated_row_residual_pass": "promotion_veto",
            "truncated_column_residual_pass": "promotion_veto",
            "truncated_particle_error_pass": "promotion_veto",
            "nearest_neighbor_mass_diagnostics": "explanatory",
            "support_curves_90_95_999": "explanatory",
            "nonzero_fraction_of_dense": "explanatory",
        },
        "nonclaims": list(NONCLAIMS),
    }


def _write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Phase 8 Sparse/Localized Locality Diagnostics",
        "",
        f"- Status: `{result['status']}`",
        f"- Phase 8 status: `{result['phase8_status']}`",
        f"- Decision: `{result['phase8_decision']}`",
        f"- Semantic class: `{result['semantic_class']}`",
        f"- Implementation scope: `{result['implementation_scope']}`",
        f"- Hard vetoes: `{result['hard_vetoes']}`",
        f"- Promotion vetoes: `{result['promotion_vetoes']}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| max dense row residual | `{result['summary']['max_dense_row_residual']:.6e}` |",
        f"| max dense column residual | `{result['summary']['max_dense_column_residual']:.6e}` |",
        f"| max truncated row residual | `{result['summary']['max_truncated_row_residual']:.6e}` |",
        f"| max truncated column residual | `{result['summary']['max_truncated_column_residual']:.6e}` |",
        f"| max truncated particle error | `{result['summary']['max_truncated_particle_error']:.6e}` |",
        f"| max 99% support p90 fraction of N | `{result['summary']['max_99_support_p90_fraction_of_n']:.6e}` |",
        "",
        "## Fixture Rows",
        "",
        "| Fixture | Decision | N | Median k99 | P90 k99 | Trunc row residual | Trunc column residual | Trunc max particle error | Nonzero fraction |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for fixture_name, fixture in result["fixtures"].items():
        support_99 = fixture["support_count_diagnostics"]["by_threshold"]["0.990"]
        truncation = fixture["truncation_diagnostics"]
        checks = fixture["threshold_checks"]
        lines.append(
            "| {fixture} | `{decision}` | {n} | `{median:.3f}` | `{p90:.3f}` | `{row:.6e}` | `{col:.6e}` | `{err:.6e}` | `{frac:.6e}` |".format(
                fixture=fixture_name,
                decision=fixture["fixture_decision"],
                n=checks["n_particles"],
                median=support_99["median"],
                p90=support_99["p90"],
                row=truncation["max_row_residual"],
                col=truncation["max_column_residual"],
                err=truncation["max_transported_particle_error"],
                frac=truncation["nonzero_fraction_of_dense"],
            )
        )
    lines.extend(
        [
            "",
            "## Threshold Definitions",
            "",
            "- `N = transport_matrix.shape[2]`, the source-particle count.",
            "- `k_i(t)` uses a deterministic stable descending sort of row mass.",
            "- Ties are not expanded beyond the first stable prefix reaching the threshold.",
            "- The truncated transport is diagnostic-only and is not a sparse solver.",
            "",
            "## Non-Claims",
            "",
        ]
    )
    for nonclaim in result["nonclaims"]:
        lines.append(f"- {nonclaim}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    raise RuntimeError("ARCHIVAL_WRONG_TRANSPORT_CHUNK_POLICY: this route is preserved only as provenance and cannot emit new evidence")
    args = _parse_args()
    result = _build_result(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_ready(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = Path(args.markdown_output)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    _write_markdown(result, markdown)
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
