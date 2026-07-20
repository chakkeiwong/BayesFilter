"""Phase 11 reduced-rank Nystrom ladder diagnostics.

This script exercises the TensorFlow Nystrom approximate-kernel transport on
deterministic Phase 1 fixtures plus one deterministic LEDH-like smoke fixture.
It writes a manifest with one Phase 3-valid candidate record per fixture/rank
row.  Runtime and memory fields are explanatory only.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
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

from docs.benchmarks.scalable_ot_candidate_result_schema import (
    CandidateResultRecord,
    TransportObjectRecord,
    validate_candidate_result,
)
from docs.benchmarks.scalable_ot_p01_baseline_fixture_diagnostics import _fixtures as _phase1_fixtures
from experiments.dpf_implementation.tf_tfp.resampling import annealed_transport_tf
from experiments.dpf_implementation.tf_tfp.resampling import nystrom_transport_tf
from experiments.dpf_implementation.tf_tfp.resampling.annealed_transport_tf import (
    annealed_transport_resample_tf,
)
from experiments.dpf_implementation.tf_tfp.resampling.nystrom_transport_tf import (
    nystrom_transport_resample_tf,
)


DTYPE = tf.float64
annealed_transport_tf.DTYPE = DTYPE
nystrom_transport_tf.DTYPE = DTYPE

PHASE11_STATUS_PASSED = "PHASE_11_REDUCED_RANK_NYSTROM_LADDER_PASSED_DIAGNOSTIC_ONLY"
PHASE11_STATUS_NOT_PROMOTED = "PHASE_11_REDUCED_RANK_NYSTROM_LADDER_COMPLETED_CANDIDATE_NOT_PROMOTED"
PHASE11_STATUS_BLOCKED = "PHASE_11_REDUCED_RANK_NYSTROM_LADDER_BLOCKED"

VALIDITY_ROW_COLUMN_THRESHOLD = 5.0e-2
VIABILITY_MAX_ERROR_THRESHOLD = 7.5e-2
VIABILITY_RMS_ERROR_THRESHOLD = 3.0e-2
PROMOTION_FIXTURES = ("tiny_manual", "small_parity", "high_dim_low_rank", "ledh_specific_smoke")
EXPLANATORY_DENSE_ERROR_FIXTURES = ("high_dim_locality",)

BASELINE_COMPARATOR = "phase1_dense_streaming_baseline_2026_06_17_dense_reference"
PLAN_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-ledh-pfpf-ot-scalable-ot-agent-a-reduced-rank-nystrom-ladder-plan-2026-06-18.md"
)
NONCLAIMS = (
    "Phase 11 reduced-rank Nystrom diagnostics only",
    "no speedup claim",
    "no ranking claim",
    "no production default change",
    "no posterior correctness claim",
    "no HMC readiness claim",
    "no public API readiness claim",
    "no production readiness claim",
    "no statistically supported ranking",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--epsilon", type=float, default=0.5)
    parser.add_argument("--baseline-scaling", type=float, default=0.9)
    parser.add_argument("--baseline-convergence-threshold", type=float, default=1.0e-3)
    parser.add_argument("--baseline-max-iterations", type=int, default=12)
    parser.add_argument("--baseline-row-chunk-size", type=int, default=4)
    parser.add_argument("--baseline-col-chunk-size", type=int, default=4)
    parser.add_argument("--nystrom-max-iterations", type=int, default=160)
    parser.add_argument("--nystrom-convergence-threshold", type=float, default=1.0e-4)
    parser.add_argument("--cholesky-jitter", type=float, default=1.0e-8)
    parser.add_argument("--denominator-floor", type=float, default=1.0e-30)
    parser.add_argument("--fixtures", default="all")
    parser.add_argument("--ranks", default="plan")
    parser.add_argument("--device", default="/CPU:0")
    parser.add_argument("--device-scope", choices=("cpu", "visible"), default=_PRE_ARGS.device_scope)
    parser.add_argument("--cuda-visible-devices", default=_PRE_ARGS.cuda_visible_devices)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output", required=True)
    args = parser.parse_args()
    if args.epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if args.baseline_max_iterations <= 0 or args.nystrom_max_iterations <= 0:
        raise ValueError("iteration counts must be positive")
    if args.baseline_row_chunk_size <= 0 or args.baseline_col_chunk_size <= 0:
        raise ValueError("baseline chunk sizes must be positive")
    if args.nystrom_convergence_threshold <= 0.0:
        raise ValueError("nystrom convergence threshold must be positive")
    if args.cholesky_jitter < 0.0:
        raise ValueError("cholesky jitter must be non-negative")
    if args.denominator_floor <= 0.0:
        raise ValueError("denominator floor must be positive")
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


def _ledh_specific_smoke_fixture() -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build a deterministic LEDH-like post-flow particle geometry.

    The fixture uses no runtime random draws.  It combines a smooth latent
    curve, two flow-like shears, low-amplitude high-dimensional harmonics, and
    deterministic uneven weights to make rank sensitivity visible without
    claiming posterior correctness.
    """

    batch_size, num_particles, state_dim, latent_dim = 1, 32, 12, 3
    t = np.linspace(-1.0, 1.0, num_particles, dtype=np.float64)
    latent = np.stack(
        [
            t,
            np.sin(1.3 * np.pi * t),
            np.cos(0.7 * np.pi * t) - np.mean(np.cos(0.7 * np.pi * t)),
        ],
        axis=1,
    )
    grid = np.arange(state_dim, dtype=np.float64)[:, None]
    basis_raw = np.concatenate(
        [
            np.sin(0.23 * (grid + 1.0) * (np.arange(latent_dim, dtype=np.float64)[None, :] + 1.0)),
            np.cos(0.17 * (grid + 1.0) * (np.arange(latent_dim, dtype=np.float64)[None, :] + 2.0)),
        ],
        axis=1,
    )
    q, _ = np.linalg.qr(basis_raw)
    basis = q[:, :latent_dim]
    particles = latent @ basis.T
    shear = np.zeros_like(particles)
    shear[:, 0] = 0.12 * particles[:, 1] * particles[:, 2]
    shear[:, 1] = 0.08 * particles[:, 0] ** 2
    harmonic = 0.025 * np.sin(2.7 * t[:, None] + 0.31 * np.arange(state_dim, dtype=np.float64)[None, :])
    cluster_shift = np.where(np.arange(num_particles)[:, None] < num_particles // 2, -0.08, 0.08)
    particles = particles + shear + harmonic + cluster_shift * basis[:, 0][None, :]
    particles = particles[None, :, :]
    raw = -0.05 * np.arange(num_particles, dtype=np.float64)[None, :]
    raw += 0.11 * np.sin(1.7 * t[None, :]) - 0.04 * np.cos(2.3 * t[None, :])
    log_weights = raw - np.log(np.sum(np.exp(raw), axis=1, keepdims=True))
    summary = {
        "construction": (
            "deterministic latent curve embedded in 12 dimensions with flow-like "
            "shear, harmonic perturbation, two deterministic clusters, and fixed uneven weights"
        ),
        "batch_size": batch_size,
        "num_particles": num_particles,
        "state_dim": state_dim,
        "latent_dim": latent_dim,
        "runtime_random_draws": 0,
        "weight_entropy": float(-np.sum(np.exp(log_weights) * log_weights)),
        "particle_norm": float(np.linalg.norm(particles)),
    }
    return particles.astype(np.float64), log_weights.astype(np.float64), summary


def _fixtures_with_ledh() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    fixtures = dict(_phase1_fixtures())
    particles, log_weights, _summary = _ledh_specific_smoke_fixture()
    fixtures["ledh_specific_smoke"] = (particles, log_weights)
    return fixtures


def _fixture_summaries() -> dict[str, Any]:
    _particles, _log_weights, ledh_summary = _ledh_specific_smoke_fixture()
    return {
        "ledh_specific_smoke": ledh_summary,
        "phase1_fixtures": {
            "source": "docs/benchmarks/scalable_ot_p01_baseline_fixture_diagnostics.py::_fixtures",
            "names": ["tiny_manual", "small_parity", "high_dim_low_rank", "high_dim_locality"],
        },
    }


def _planned_ranks(fixture_name: str, num_particles: int) -> list[int]:
    if fixture_name == "tiny_manual":
        ranks = [1, 2, 3, num_particles]
    elif fixture_name == "small_parity":
        ranks = [2, 4, 8, num_particles]
    else:
        ranks = [2, 4, 8, 16, num_particles]
    return sorted({rank for rank in ranks if 0 < rank <= num_particles})


def _rank_list(spec: str, fixture_name: str, num_particles: int) -> list[int]:
    if spec == "plan":
        return _planned_ranks(fixture_name, num_particles)
    if spec == "full":
        return [num_particles]
    ranks = sorted({int(item) for item in spec.split(",") if item.strip()})
    if not ranks:
        raise ValueError("at least one rank is required")
    invalid = [rank for rank in ranks if rank <= 0 or rank > num_particles]
    if invalid:
        raise ValueError(f"invalid ranks for particle count {num_particles}: {invalid}")
    return ranks


def _selected_fixtures(spec: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    fixtures = _fixtures_with_ledh()
    if spec == "all":
        return fixtures
    selected = [item.strip() for item in spec.split(",") if item.strip()]
    missing = sorted(set(selected).difference(fixtures))
    if missing:
        raise ValueError(f"unknown fixtures: {missing}")
    return {name: fixtures[name] for name in selected}


def _float(value: Any) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _tensor_finite(value: tf.Tensor) -> bool:
    return bool(tf.reduce_all(tf.math.is_finite(value)).numpy())


def _run_dense(
    particles_np: np.ndarray,
    log_weights_np: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], tf.Tensor]:
    particles = tf.constant(particles_np, dtype=DTYPE)
    log_weights = tf.constant(log_weights_np, dtype=DTYPE)
    start = time.perf_counter()
    with tf.device(args.device):
        result = annealed_transport_resample_tf(
            particles,
            log_weights,
            epsilon=args.epsilon,
            scaling=args.baseline_scaling,
            convergence_threshold=args.baseline_convergence_threshold,
            max_iterations=args.baseline_max_iterations,
            transport_gradient_mode="raw",
            transport_plan_mode="dense",
            row_chunk_size=args.baseline_row_chunk_size,
            col_chunk_size=args.baseline_col_chunk_size,
        )
    wall_time = time.perf_counter() - start
    transport = tf.convert_to_tensor(result.transport_matrix, dtype=DTYPE)
    source_weights = tf.exp(log_weights)
    num_particles = tf.cast(tf.shape(log_weights)[1], DTYPE)
    row_residual = tf.reduce_max(tf.abs(tf.reduce_sum(transport, axis=2) - 1.0))
    column_target = source_weights * num_particles
    column_residual = tf.reduce_max(tf.abs(tf.reduce_sum(transport, axis=1) - column_target))
    transported = tf.convert_to_tensor(result.particles, dtype=DTYPE)
    return {
        "wall_time_seconds": wall_time,
        "particles_shape": transported.shape.as_list(),
        "transport_matrix_shape": transport.shape.as_list(),
        "finite_particles": _tensor_finite(transported),
        "finite_transport_matrix": _tensor_finite(transport),
        "max_row_residual": _float(row_residual),
        "max_column_residual": _float(column_residual),
        "diagnostics": dict(result.diagnostics),
    }, transported


def _memory_proxy(batch_size: int, num_particles: int, rank: int) -> dict[str, Any]:
    dense_entries = batch_size * num_particles * num_particles
    factor_entries = batch_size * (num_particles * rank + rank * rank + 2 * num_particles)
    return {
        "role": "explanatory_only_until_validity_gates_pass",
        "dense_materialized_entries": dense_entries,
        "kernel_factor_entries": factor_entries,
        "entry_ratio_vs_dense": float(factor_entries / dense_entries),
        "formula": "B*(N*r + r*r + 2*N) factor/scaling entries vs B*N*N dense entries",
    }


def _run_nystrom(
    particles_np: np.ndarray,
    log_weights_np: np.ndarray,
    args: argparse.Namespace,
    *,
    fixture_name: str,
    rank: int,
    dense_particles: tf.Tensor,
) -> dict[str, Any]:
    particles = tf.constant(particles_np, dtype=DTYPE)
    log_weights = tf.constant(log_weights_np, dtype=DTYPE)
    start = time.perf_counter()
    with tf.device(args.device):
        result = nystrom_transport_resample_tf(
            particles,
            log_weights,
            rank=rank,
            epsilon=args.epsilon,
            max_iterations=args.nystrom_max_iterations,
            convergence_threshold=args.nystrom_convergence_threshold,
            cholesky_jitter=args.cholesky_jitter,
            denominator_floor=args.denominator_floor,
        )
    wall_time = time.perf_counter() - start
    candidate_particles = tf.convert_to_tensor(result.particles, dtype=DTYPE)
    diff = candidate_particles - dense_particles
    max_error = tf.reduce_max(tf.abs(diff))
    rms_error = tf.sqrt(tf.reduce_mean(tf.square(diff)))
    diagnostics = dict(result.diagnostics)
    row_residual = float(diagnostics["max_row_residual"])
    column_residual = float(diagnostics["max_column_residual"])
    dense_max_error = _float(max_error)
    dense_rms_error = _float(rms_error)
    shape_pass = candidate_particles.shape == dense_particles.shape
    finite_dense_errors = np.isfinite(dense_max_error) and np.isfinite(dense_rms_error)
    validity_pass = bool(
        bool(diagnostics["finite_particles"])
        and bool(diagnostics["finite_factors"])
        and shape_pass
        and np.isfinite(row_residual)
        and np.isfinite(column_residual)
        and row_residual <= VALIDITY_ROW_COLUMN_THRESHOLD
        and column_residual <= VALIDITY_ROW_COLUMN_THRESHOLD
        and finite_dense_errors
    )
    is_full_rank = rank == particles_np.shape[1]
    dense_reference_threshold_pass = bool(
        validity_pass
        and not is_full_rank
        and dense_max_error <= VIABILITY_MAX_ERROR_THRESHOLD
        and dense_rms_error <= VIABILITY_RMS_ERROR_THRESHOLD
    )
    reduced_rank_viability_pass = bool(
        dense_reference_threshold_pass
        and fixture_name in PROMOTION_FIXTURES
    )
    row = {
        "fixture": fixture_name,
        "rank": rank,
        "rank_label": "full" if is_full_rank else str(rank),
        "is_full_rank": is_full_rank,
        "wall_time_seconds": wall_time,
        "runtime_proxy_role": "explanatory_only_until_validity_gates_pass",
        "memory_proxy": _memory_proxy(particles_np.shape[0], particles_np.shape[1], rank),
        "particles_shape": candidate_particles.shape.as_list(),
        "transport_object": {
            "kind": "kernel_factors",
            "materialized": False,
            "factor_shapes": diagnostics["factor_shapes"],
            "not_materialized_reason": "kernel_factors_nonmaterialized",
            "orientation": "source_rows_target_columns",
            "semantic_output": "full_state_particles",
        },
        "finite_particles": bool(diagnostics["finite_particles"]),
        "finite_factors": bool(diagnostics["finite_factors"]),
        "shape_pass": bool(shape_pass),
        "max_row_residual": row_residual,
        "max_column_residual": column_residual,
        "dense_reference_max_abs_particle_error": dense_max_error,
        "dense_reference_rms_particle_error": dense_rms_error,
        "dense_reference_error_role": (
            "promotion_criterion"
            if fixture_name in PROMOTION_FIXTURES
            else "explanatory"
        ),
        "dense_reference_threshold_pass": dense_reference_threshold_pass,
        "validity_pass": validity_pass,
        "reduced_rank_viability_pass": reduced_rank_viability_pass,
        "diagnostics": diagnostics,
    }
    row["candidate_record"] = _candidate_schema_record(row)
    return row


def _candidate_schema_record(row: dict[str, Any]) -> dict[str, Any]:
    diagnostics = dict(row["diagnostics"])
    record = CandidateResultRecord(
        candidate_id=f"phase11_reduced_rank_nystrom_{row['fixture']}_rank_{row['rank_label']}",
        source_status="source_locked",
        semantic_class="approximate_kernel",
        source_route="fixed_hmc_adaptation",
        baseline_comparator=BASELINE_COMPARATOR,
        transport_object=TransportObjectRecord(
            kind="kernel_factors",
            materialized=False,
            factor_shapes=diagnostics["factor_shapes"],
            not_materialized_reason="kernel_factors_nonmaterialized",
            orientation="source_rows_target_columns",
            semantic_output="full_state_particles",
        ),
        diagnostics={
            "fixture": row["fixture"],
            "rank": row["rank"],
            "rank_label": row["rank_label"],
            "is_full_rank": row["is_full_rank"],
            "validity_pass": row["validity_pass"],
            "reduced_rank_viability_pass": row["reduced_rank_viability_pass"],
            "finite_particles": row["finite_particles"],
            "finite_factors": row["finite_factors"],
            "shape_pass": row["shape_pass"],
            "max_row_residual": row["max_row_residual"],
            "max_column_residual": row["max_column_residual"],
            "dense_reference_max_abs_particle_error": row["dense_reference_max_abs_particle_error"],
            "dense_reference_rms_particle_error": row["dense_reference_rms_particle_error"],
            "dense_reference_error_role": row["dense_reference_error_role"],
            "dense_reference_threshold_pass": row["dense_reference_threshold_pass"],
            "runtime_proxy_seconds": row["wall_time_seconds"],
            "runtime_proxy_role": row["runtime_proxy_role"],
            "memory_proxy": row["memory_proxy"],
            "landmark_rule": diagnostics["landmark_rule"],
            "landmark_indices": diagnostics["landmark_indices"],
            "factor_shapes": diagnostics["factor_shapes"],
            "source_route_components": diagnostics["source_route_components"],
            "iterations_used": diagnostics["iterations_used"],
            "max_factor_diag_error": diagnostics["max_factor_diag_error"],
            "min_factor_diagonal": diagnostics["min_factor_diagonal"],
            "min_kernel_denominator": diagnostics["min_kernel_denominator"],
            "denominator_floor_hits": diagnostics["denominator_floor_hits"],
        },
        diagnostic_roles={
            "validity_pass": "hard_veto",
            "finite_particles": "hard_veto",
            "finite_factors": "hard_veto",
            "shape_pass": "hard_veto",
            "max_row_residual": "hard_veto",
            "max_column_residual": "hard_veto",
            "dense_reference_max_abs_particle_error": row["dense_reference_error_role"],
            "dense_reference_rms_particle_error": row["dense_reference_error_role"],
            "dense_reference_threshold_pass": row["dense_reference_error_role"],
            "reduced_rank_viability_pass": "promotion_criterion",
            "runtime_proxy_seconds": "explanatory",
            "runtime_proxy_role": "explanatory",
            "memory_proxy": "explanatory",
            "landmark_rule": "repair_trigger",
            "landmark_indices": "repair_trigger",
            "factor_shapes": "explanatory",
            "source_route_components": "explanatory",
            "iterations_used": "explanatory",
            "max_factor_diag_error": "repair_trigger",
            "min_factor_diagonal": "repair_trigger",
            "min_kernel_denominator": "repair_trigger",
            "denominator_floor_hits": "repair_trigger",
        },
        execution_manifest={
            "git_commit": _git_commit(),
            "plan_path": PLAN_PATH,
            "device_scope": "cpu" if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" else "visible",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "dtype": "tf.float64",
            "fixture": row["fixture"],
            "rank": row["rank"],
            "baseline_comparator": BASELINE_COMPARATOR,
        },
        nonclaims=list(NONCLAIMS),
    ).to_dict()
    validate_candidate_result(record)
    return record


def _build_result(args: argparse.Namespace) -> dict[str, Any]:
    selected_fixtures = _selected_fixtures(args.fixtures)
    fixture_results: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    hard_vetoes: list[str] = []
    schema_warnings: list[str] = []
    start = time.perf_counter()
    for fixture_name, (particles_np, log_weights_np) in selected_fixtures.items():
        dense_record, dense_particles = _run_dense(particles_np, log_weights_np, args)
        ranks = _rank_list(args.ranks, fixture_name, particles_np.shape[1])
        nystrom_rows = [
            _run_nystrom(
                particles_np,
                log_weights_np,
                args,
                fixture_name=fixture_name,
                rank=rank,
                dense_particles=dense_particles,
            )
            for rank in ranks
        ]
        for row in nystrom_rows:
            schema_warnings.extend(validate_candidate_result(row["candidate_record"]))
        all_rows.extend(nystrom_rows)
        fixture_results[fixture_name] = {
            "input_shape": {
                "particles": list(particles_np.shape),
                "log_weights": list(log_weights_np.shape),
            },
            "rank_grid": ranks,
            "dense": dense_record,
            "nystrom": nystrom_rows,
        }
        if not dense_record["finite_particles"] or not dense_record["finite_transport_matrix"]:
            hard_vetoes.append(f"{fixture_name}:dense_baseline_nonfinite")
        for row in nystrom_rows:
            if not row["validity_pass"]:
                hard_vetoes.append(f"{fixture_name}:rank_{row['rank_label']}:validity_failed")
    viability_by_fixture = {}
    viable_reduced_ranks = {}
    dense_reference_threshold_reduced_ranks = {}
    for fixture_name, fixture in fixture_results.items():
        rows = fixture["nystrom"]
        viable = [
            row["rank_label"]
            for row in rows
            if row["reduced_rank_viability_pass"]
        ]
        threshold_hits = [
            row["rank_label"]
            for row in rows
            if row["dense_reference_threshold_pass"]
        ]
        viable_reduced_ranks[fixture_name] = viable
        dense_reference_threshold_reduced_ranks[fixture_name] = threshold_hits
        if fixture_name in PROMOTION_FIXTURES:
            viability_by_fixture[fixture_name] = bool(viable)
        else:
            viability_by_fixture[fixture_name] = None
    selected_promotion_fixtures = [
        name for name in PROMOTION_FIXTURES if name in fixture_results
    ]
    validity_pass = not hard_vetoes
    viability_pass = validity_pass and all(
        viability_by_fixture[name] for name in selected_promotion_fixtures
    )
    if not validity_pass:
        phase11_status = PHASE11_STATUS_BLOCKED
        status = "FAIL"
    elif viability_pass:
        phase11_status = PHASE11_STATUS_PASSED
        status = "PASS"
    else:
        phase11_status = PHASE11_STATUS_NOT_PROMOTED
        status = "PASS"
    wall_time = time.perf_counter() - start
    summary = {
        "max_row_residual": max(row["max_row_residual"] for row in all_rows),
        "max_column_residual": max(row["max_column_residual"] for row in all_rows),
        "max_dense_reference_particle_error": max(
            row["dense_reference_max_abs_particle_error"] for row in all_rows
        ),
        "max_dense_reference_rms_error": max(
            row["dense_reference_rms_particle_error"] for row in all_rows
        ),
        "validity_pass": validity_pass,
        "viability_pass": viability_pass,
        "viability_by_fixture": viability_by_fixture,
        "viable_reduced_ranks": viable_reduced_ranks,
        "dense_reference_threshold_reduced_ranks": dense_reference_threshold_reduced_ranks,
        "record_count": len(all_rows),
        "candidate_record_count": len(all_rows),
        "schema_warnings": schema_warnings,
        "wall_time_seconds": wall_time,
    }
    manifest = {
        "git_commit": _git_commit(),
        "timestamp_utc": _dt.datetime.now(tz=_dt.UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "tensorflow_version": tf.__version__,
        "device_scope": args.device_scope,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": args.device,
        "dtype": "tf.float64",
        "command": "scalable_ot_p11_reduced_rank_nystrom_ladder_diagnostics.py",
        "plan_path": PLAN_PATH,
        "baseline_comparator": BASELINE_COMPARATOR,
    }
    candidate_records = [
        row["candidate_record"]
        for fixture in fixture_results.values()
        for row in fixture["nystrom"]
    ]
    return {
        "phase11_status": phase11_status,
        "status": status,
        "hard_vetoes": hard_vetoes,
        "summary": summary,
        "thresholds": {
            "row_column_residual_hard_veto": VALIDITY_ROW_COLUMN_THRESHOLD,
            "dense_reference_max_abs_promotion": VIABILITY_MAX_ERROR_THRESHOLD,
            "dense_reference_rms_promotion": VIABILITY_RMS_ERROR_THRESHOLD,
            "promotion_fixtures": list(PROMOTION_FIXTURES),
            "explanatory_dense_error_fixtures": list(EXPLANATORY_DENSE_ERROR_FIXTURES),
        },
        "source_route_components": {
            "nystrom_factors": "source_faithful",
            "low_rank_scaling": "source_faithful",
            "filterflow_cost_scaling_adapter": "fixed_hmc_adaptation",
            "deterministic_landmarks": "fixed_hmc_adaptation",
            "cholesky_jitter": "fixed_hmc_adaptation",
        },
        "source_anchors": {
            "nystrom_factors": ".localsource/1812.05189-src/sections/nystrom.tex lines 10-27",
            "adaptive_context_only": ".localsource/1812.05189-src/sections/nystrom.tex lines 121-172",
            "sinkhorn_scaling": ".localsource/1812.05189-src/sections/sinkhorn.tex lines 8-24 and 41-50",
            "pot_lowrank_reference": ".localsource/scalable_ot_code_audit/POT/ot/lowrank.py lines 530-730",
            "pot_empirical_reference": ".localsource/scalable_ot_code_audit/POT/ot/bregman/_empirical.py lines 766-865",
            "local_notation": (
                "docs/plans/bayesfilter-dpf-ledh-pfpf-ot-scalable-ot-self-contained-survey-paper-2026-06-17.tex "
                "lines 437-545"
            ),
        },
        "settings": {
            "epsilon": args.epsilon,
            "baseline_scaling": args.baseline_scaling,
            "baseline_convergence_threshold": args.baseline_convergence_threshold,
            "baseline_max_iterations": args.baseline_max_iterations,
            "baseline_row_chunk_size": args.baseline_row_chunk_size,
            "baseline_col_chunk_size": args.baseline_col_chunk_size,
            "nystrom_max_iterations": args.nystrom_max_iterations,
            "nystrom_convergence_threshold": args.nystrom_convergence_threshold,
            "cholesky_jitter": args.cholesky_jitter,
            "denominator_floor": args.denominator_floor,
            "fixtures": args.fixtures,
            "ranks": args.ranks,
        },
        "fixture_summaries": _fixture_summaries(),
        "manifest": manifest,
        "fixtures": fixture_results,
        "candidate_records": candidate_records,
        "nonclaims": list(NONCLAIMS),
    }


def _write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Phase 11 Reduced-Rank Nystrom Ladder Diagnostics",
        "",
        f"- Status: `{result['status']}`",
        f"- Phase 11 status: `{result['phase11_status']}`",
        f"- Validity pass: `{result['summary']['validity_pass']}`",
        f"- Viability pass: `{result['summary']['viability_pass']}`",
        f"- Hard vetoes: `{result['hard_vetoes']}`",
        f"- Candidate records: `{result['summary']['candidate_record_count']}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| max row residual | `{result['summary']['max_row_residual']:.6e}` |",
        f"| max column residual | `{result['summary']['max_column_residual']:.6e}` |",
        f"| max dense-reference particle error | `{result['summary']['max_dense_reference_particle_error']:.6e}` |",
        f"| max dense-reference RMS error | `{result['summary']['max_dense_reference_rms_error']:.6e}` |",
        f"| wall time seconds | `{result['summary']['wall_time_seconds']:.6e}` |",
        "",
        "## Viable Reduced Ranks",
        "",
        "| Fixture | Viable reduced ranks |",
        "| --- | --- |",
    ]
    for fixture_name, ranks in result["summary"]["viable_reduced_ranks"].items():
        lines.append(f"| `{fixture_name}` | `{ranks}` |")
    lines.extend(
        [
            "",
            "## Dense-Reference Threshold Hits",
            "",
            "These ranks meet the numeric dense-reference screen; `high_dim_locality` remains explanatory.",
            "",
            "| Fixture | Reduced ranks meeting dense-reference screen |",
            "| --- | --- |",
        ]
    )
    for fixture_name, ranks in result["summary"]["dense_reference_threshold_reduced_ranks"].items():
        lines.append(f"| `{fixture_name}` | `{ranks}` |")
    lines.extend(
        [
            "",
            "## LEDH-Specific Smoke Fixture",
            "",
            "| Field | Value |",
            "| --- | --- |",
        ]
    )
    for key, value in result["fixture_summaries"]["ledh_specific_smoke"].items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(
        [
            "",
            "## Fixture Rows",
            "",
            (
                "| Fixture | Rank | Valid | Reduced-rank viable | Row residual | Column residual | "
                "Max dense error | RMS dense error | Memory entry ratio |"
            ),
            "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for fixture_name, fixture in result["fixtures"].items():
        for row in fixture["nystrom"]:
            lines.append(
                "| {fixture} | {rank} | `{valid}` | `{viable}` | `{row_res:.6e}` | `{col_res:.6e}` | "
                "`{max_err:.6e}` | `{rms_err:.6e}` | `{ratio:.6e}` |".format(
                    fixture=fixture_name,
                    rank=row["rank_label"],
                    valid=row["validity_pass"],
                    viable=row["reduced_rank_viability_pass"],
                    row_res=row["max_row_residual"],
                    col_res=row["max_column_residual"],
                    max_err=row["dense_reference_max_abs_particle_error"],
                    rms_err=row["dense_reference_rms_particle_error"],
                    ratio=row["memory_proxy"]["entry_ratio_vs_dense"],
                )
            )
    lines.extend(["", "## Non-Claims", ""])
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
    if result["phase11_status"] == PHASE11_STATUS_BLOCKED:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
