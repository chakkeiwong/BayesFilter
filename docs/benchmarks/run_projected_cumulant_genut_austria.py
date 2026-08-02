#!/usr/bin/env python3
"""Tune and compare projected-cumulant GenUT corrections on Austria SIR."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth


PLAN = "docs/plans/bayesfilter-projected-cumulant-genut-austria-rank-comparison-plan-2026-08-01.md"
SCHEMA = "bayesfilter.projected_cumulant_genut_austria.v1"
RANKS = (4, 6, 8)
TUNING_SEEDS = (98301, 98302)
CLAIM_SEEDS_1008 = tuple(range(98201, 98217))
CLAIM_SEEDS_4032 = (98201, 98202, 98203)
SKETCH_COUNT = 16
RESIDUAL_TOLERANCE = 5.0e-4
DISPLACEMENT_VETO = 2.0
BASE_CONTROLS = {
    "epsilon": 8.0,
    "sinkhorn_steps": 16,
    "balance_steps": 16,
    "ridge": 1.0e-5,
    "higher_moment_correction_steps": 4,
    "higher_moment_strength": 0.2,
    "higher_moment_floor": 1.0e-5,
    "pairwise_moment_correction_steps": 0,
    "pairwise_moment_strength": 0.0,
    "pairwise_moment_floor": 1.0e-5,
    "projected_cumulant_correction_steps": 0,
    "projected_cumulant_strength": 0.0,
    "projected_cumulant_floor": 1.0e-5,
}
PAIRWISE_CONTROLS = {
    **BASE_CONTROLS,
    "pairwise_moment_correction_steps": 4,
    "pairwise_moment_strength": 0.02,
}
PROJECTED_GRID = tuple(
    {
        **BASE_CONTROLS,
        "projected_cumulant_correction_steps": steps,
        "projected_cumulant_strength": strength,
    }
    for steps in (1, 2)
    for strength in (0.0025, 0.005, 0.01)
)
SGQF_COMPARATOR = {
    "value": -682.3480055392419,
    "score": [28.739453057371584, -106.65885657030441, 9.43117639262833],
    "role": "descriptive comparator, not an exact Austria oracle",
}


def _safe(value: Any) -> Any:
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _target(particles: int) -> dict[str, Any]:
    from bayesfilter.highdim.cubature_genut_candidate import cubature_design
    from docs.benchmarks.run_moment_retuned_genut_whole_leaderboard import _build_targets

    target = dict(_build_targets()["austria_sir_T20"])
    target["design"] = cubature_design(dim=18, num_particles=particles)
    return target


def _sketch_directions() -> tf.Tensor:
    raw = tf.random.stateless_normal([18, SKETCH_COUNT], [20260801, 41], dtype=tf.float32)
    basis, _ = tf.linalg.qr(raw, full_matrices=False)
    return basis


def _make_evaluator(
    target: dict[str, Any],
    particles: int,
    controls: dict[str, Any],
    *,
    basis: tf.Tensor | None = None,
    sketches: tf.Tensor | None = None,
):
    from bayesfilter.highdim.cubature_genut_filter import finite_value_score

    horizon = 20

    @tf.function(jit_compile=True, reduce_retracing=True)
    def evaluate(theta, observations, initial_noise, process_noise, design):
        theta = tf.ensure_shape(theta, [3])
        observations = tf.ensure_shape(observations, [horizon, 9])
        initial_noise = tf.ensure_shape(initial_noise, [particles, 18])
        process_noise = tf.ensure_shape(process_noise, [horizon, particles, 18])
        design = tf.ensure_shape(design, [particles, 18])
        with tf.device("/GPU:0"):
            return finite_value_score(
                target["adapter"],
                theta,
                observations,
                initial_noise,
                process_noise,
                design,
                epsilon=float(controls["epsilon"]),
                sinkhorn_steps=int(controls["sinkhorn_steps"]),
                balance_steps=int(controls["balance_steps"]),
                ridge=float(controls["ridge"]),
                transition_before_first_observation=True,
                higher_moment_correction_steps=int(
                    controls["higher_moment_correction_steps"]
                ),
                higher_moment_strength=float(controls["higher_moment_strength"]),
                higher_moment_floor=float(controls["higher_moment_floor"]),
                pairwise_moment_correction_steps=int(
                    controls["pairwise_moment_correction_steps"]
                ),
                pairwise_moment_strength=float(controls["pairwise_moment_strength"]),
                pairwise_moment_floor=float(controls["pairwise_moment_floor"]),
                projected_cumulant_basis=basis,
                projected_cumulant_correction_steps=int(
                    controls["projected_cumulant_correction_steps"]
                ),
                projected_cumulant_strength=float(
                    controls["projected_cumulant_strength"]
                ),
                projected_cumulant_floor=float(
                    controls["projected_cumulant_floor"]
                ),
                projected_cumulant_sketch_directions=sketches,
            )

    return evaluate


def _noise(seed: int, particles: int) -> tuple[tf.Tensor, tf.Tensor]:
    initial = tf.random.stateless_normal([particles, 18], [seed, 101], dtype=tf.float32)
    process = tf.random.stateless_normal([20, particles, 18], [seed, 102], dtype=tf.float32)
    return initial, process


def _evaluate(
    evaluator: Any,
    target: dict[str, Any],
    observations: tf.Tensor,
    seed: int,
    particles: int,
    *,
    include_mode_score: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    initial, process = _noise(seed, particles)
    value, score, diagnostics = evaluator(
        target["theta"],
        tf.cast(observations, tf.float32),
        initial,
        process,
        target["design"],
    )
    score_residual = tf.reduce_max(
        tf.abs(tf.reduce_sum(diagnostics["score_increments"], axis=0) - score)
    )
    finite = bool(diagnostics["program_valid"].numpy()) and bool(
        tf.math.is_finite(value).numpy()
    ) and bool(tf.reduce_all(tf.math.is_finite(score)).numpy())
    row = {
        "seed": seed,
        "finite": finite,
        "program_valid": bool(diagnostics["program_valid"].numpy()),
        "value": float(value.numpy()) if finite else None,
        "score": [float(item) for item in score.numpy()] if finite else None,
        "score_increment_sum_residual": float(score_residual.numpy()) if finite else None,
        "max_mean_residual": float(diagnostics["max_mean_residual"].numpy()),
        "max_row_residual": float(diagnostics["max_row_residual"].numpy()),
        "max_col_residual": float(diagnostics["max_col_residual"].numpy()),
        "maximum_normalized_shape_displacement": float(
            diagnostics["maximum_normalized_shape_displacement"].numpy()
        ),
        "maximum_projected_cumulant_residual": float(
            diagnostics["maximum_projected_cumulant_residual"].numpy()
        ),
        "maximum_projected_cumulant_third_residual": float(
            diagnostics["maximum_projected_cumulant_third_residual"].numpy()
        ),
        "maximum_projected_cumulant_fourth_residual": float(
            diagnostics["maximum_projected_cumulant_fourth_residual"].numpy()
        ),
        "minimum_row_mass": float(diagnostics["minimum_row_mass"].numpy()),
        "minimum_covariance_gap_eigenvalue": float(
            diagnostics["minimum_covariance_gap_eigenvalue"].numpy()
        ),
        "maximum_post_quotient_column_tv_error": float(
            diagnostics["maximum_post_quotient_column_tv_error"].numpy()
        ),
        "device": str(value.device),
        "wall_time_seconds": time.perf_counter() - started,
    }
    if include_mode_score:
        row["mode_score"] = _safe(diagnostics["projected_cumulant_mode_score"])
    row["validity_checks"] = _validity_checks(row)
    row["valid"] = all(row["validity_checks"].values())
    return row


def _validity_checks(row: dict[str, Any]) -> dict[str, bool]:
    residuals = [
        row["max_mean_residual"],
        row["max_row_residual"],
        row["max_col_residual"],
    ]
    if row["score_increment_sum_residual"] is not None:
        residuals.append(row["score_increment_sum_residual"])
    return {
        "finite_value_score_and_program": bool(row["finite"]),
        "gpu_device": "GPU" in row["device"].upper(),
        "residual_tolerance": max(residuals) < RESIDUAL_TOLERANCE,
        "displacement_veto": (
            row["maximum_normalized_shape_displacement"] <= DISPLACEMENT_VETO
        ),
    }


def _valid(row: dict[str, Any]) -> bool:
    return all(_validity_checks(row).values())


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    finite_rows = [row for row in rows if row["finite"]]
    result: dict[str, Any] = {
        "count": len(rows),
        "finite_count": len(finite_rows),
        "all_valid": len(finite_rows) == len(rows) and all(_valid(row) for row in rows),
    }
    if len(finite_rows) != len(rows) or not rows:
        return result
    vectors = [[row["value"], *row["score"]] for row in rows]
    labels = ("value", "score_0", "score_1", "score_2")
    for index, label in enumerate(labels):
        values = [vector[index] for vector in vectors]
        result[label] = {
            "mean": statistics.mean(values),
            "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            "mean_mcse": (
                statistics.stdev(values) / math.sqrt(len(values))
                if len(values) > 1
                else 0.0
            ),
        }
    result["mean_wall_time_seconds"] = statistics.mean(
        row["wall_time_seconds"] for row in rows
    )
    result["maximum_displacement"] = max(
        row["maximum_normalized_shape_displacement"] for row in rows
    )
    result["maximum_projected_residual"] = max(
        row["maximum_projected_cumulant_residual"] for row in rows
    )
    return result


def _partition_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize particle-seed variation without pooling dataset variation."""

    result = _summary(rows)
    dataset_ids = sorted({int(row["dataset_index"]) for row in rows})
    per_dataset = [
        [row for row in rows if int(row["dataset_index"]) == dataset_id]
        for dataset_id in dataset_ids
    ]
    if not result["all_valid"] or any(len(group) < 2 for group in per_dataset):
        return result
    labels = ("value", "score_0", "score_1", "score_2")
    for index, label in enumerate(labels):
        variances = []
        for group in per_dataset:
            values = [
                [row["value"], *row["score"]][index]
                for row in group
            ]
            variances.append(statistics.variance(values))
        within_variance = statistics.mean(variances)
        result[label]["sample_sd"] = math.sqrt(within_variance)
        result[label]["mean_mcse"] = math.sqrt(within_variance / len(rows))
    result["variance_definition"] = (
        "mean within-observation-dataset particle-seed variance"
    )
    return result


def _mode_score_rows(
    target: dict[str, Any], particles: int, datasets: list[tf.Tensor]
) -> tuple[list[dict[str, Any]], tf.Tensor]:
    evaluator = _make_evaluator(
        target,
        particles,
        BASE_CONTROLS,
        sketches=_sketch_directions(),
    )
    rows = []
    for dataset_index, data in enumerate(datasets):
        for seed in TUNING_SEEDS:
            row = _evaluate(
                evaluator,
                target,
                data,
                seed,
                particles,
                include_mode_score=True,
            )
            row["dataset_index"] = dataset_index
            rows.append(row)
    score = tf.reduce_sum(
        tf.stack([tf.constant(row["mode_score"], tf.float32) for row in rows]), axis=0
    )
    return rows, score


def _write_basis_failure(
    output_root: Path,
    *,
    particles: int,
    calibration_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    reason: str,
) -> None:
    payload = {
        "schema": f"{SCHEMA}.failure",
        "plan": PLAN,
        "status": "BASIS_CAPTURE_HARD_VETO",
        "reason": reason,
        "particles": particles,
        "residual_tolerance": RESIDUAL_TOLERANCE,
        "displacement_veto": DISPLACEMENT_VETO,
        "calibration_rows": calibration_rows,
        "validation_rows": validation_rows,
    }
    (output_root / "failure_result.json").write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_checkpoint(
    output_root: Path,
    *,
    stage: str,
    particles: int,
    basis: dict[str, Any] | None = None,
    tuning: dict[str, Any] | None = None,
    claim: dict[str, Any] | None = None,
) -> None:
    payload = {
        "schema": f"{SCHEMA}.checkpoint",
        "plan": PLAN,
        "stage": stage,
        "particles": particles,
        "basis": basis or {},
        "tuning": tuning or {},
        "claim": claim or {},
    }
    (output_root / "checkpoint.json").write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _canonical_basis(mode_score: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    eigenvalues, eigenvectors = tf.linalg.eigh(mode_score)
    eigenvalues = tf.reverse(eigenvalues, axis=[-1])
    eigenvectors = tf.reverse(eigenvectors, axis=[-1])
    basis = eigenvectors[:, :, :8]
    largest = tf.argmax(tf.abs(basis), axis=1, output_type=tf.int32)
    time_index = tf.broadcast_to(tf.range(tf.shape(basis)[0])[:, None], tf.shape(largest))
    rank_index = tf.broadcast_to(tf.range(tf.shape(basis)[2])[None, :], tf.shape(largest))
    indices = tf.stack([time_index, largest, rank_index], axis=-1)
    signs = tf.where(tf.gather_nd(basis, indices) < 0.0, -1.0, 1.0)
    return basis * signs[:, None, :], eigenvalues


def _basis_diagnostics(
    calibration_score: tf.Tensor,
    validation_score: tf.Tensor,
    calibration_basis: tf.Tensor,
    validation_basis: tf.Tensor,
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    total_trace = tf.reduce_sum(tf.linalg.trace(validation_score))
    for rank in RANKS:
        basis = calibration_basis[:, :, :rank]
        captured = tf.einsum("tdr,tde,ter->t", basis, validation_score, basis)
        singular = tf.linalg.svd(
            tf.linalg.matmul(
                basis,
                validation_basis[:, :, :rank],
                transpose_a=True,
            ),
            compute_uv=False,
        )
        angles = tf.acos(tf.clip_by_value(singular, 0.0, 1.0)) * (180.0 / math.pi)
        rows[str(rank)] = {
            "validation_explained_energy": float(
                (tf.reduce_sum(captured) / tf.maximum(total_trace, 1.0e-12)).numpy()
            ),
            "maximum_principal_angle_degrees": float(tf.reduce_max(angles).numpy()),
            "mean_principal_angle_degrees": float(tf.reduce_mean(angles).numpy()),
        }
    return rows


def _validation_rows(
    evaluator: Any,
    target: dict[str, Any],
    particles: int,
) -> list[dict[str, Any]]:
    rows = []
    for dataset_index, data in enumerate(target["validation"]):
        for seed in TUNING_SEEDS:
            row = _evaluate(evaluator, target, data, seed, particles)
            row["dataset_index"] = dataset_index
            rows.append(row)
    return rows


def _tune_rank(
    target: dict[str, Any],
    particles: int,
    basis: tf.Tensor,
    baseline_summary: dict[str, Any],
) -> dict[str, Any]:
    candidates = []
    for controls in PROJECTED_GRID:
        evaluator = _make_evaluator(target, particles, controls, basis=basis)
        rows = _validation_rows(evaluator, target, particles)
        summary = _partition_summary(rows)
        hard_valid = summary["all_valid"]
        score_ok = bool(
            hard_valid
            and all(
                summary[f"score_{index}"]["sample_sd"]
                <= baseline_summary[f"score_{index}"]["sample_sd"]
                for index in range(3)
            )
        )
        value_ok = bool(
            hard_valid
            and summary["value"]["sample_sd"]
            <= 1.25 * baseline_summary["value"]["sample_sd"]
            and abs(summary["value"]["mean"] - baseline_summary["value"]["mean"])
            <= baseline_summary["value"]["mean_mcse"]
        )
        candidates.append(
            {
                "controls": controls,
                "rows": rows,
                "summary": summary,
                "hard_valid": hard_valid,
                "score_veto_pass": score_ok,
                "value_veto_pass": value_ok,
                "eligible": hard_valid and score_ok and value_ok,
            }
        )
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    valid = [candidate for candidate in candidates if candidate["hard_valid"]]
    if not valid:
        return {
            "selection_status": "NO_MECHANICALLY_VALID_PROJECTED_CONTROL",
            "selected_controls": None,
            "candidates": candidates,
            "claim_data_read_during_selection": False,
        }

    def key(candidate: dict[str, Any]) -> tuple[Any, ...]:
        summary = candidate["summary"]
        ratios = [
            summary[f"score_{index}"]["sample_sd"]
            / max(baseline_summary[f"score_{index}"]["sample_sd"], 1.0e-12)
            for index in range(3)
        ]
        return (
            max(ratios),
            sum(ratio * ratio for ratio in ratios),
            summary["maximum_projected_residual"],
            summary["maximum_displacement"],
            candidate["controls"]["projected_cumulant_strength"],
            candidate["controls"]["projected_cumulant_correction_steps"],
        )

    selected = min(eligible or valid, key=key)
    return {
        "selection_status": (
            "ELIGIBLE_PROJECTED_CONTROL_SELECTED"
            if eligible
            else "INELIGIBLE_DIAGNOSTIC_REPRESENTATIVE"
        ),
        "selected_controls": selected["controls"],
        "candidates": candidates,
        "claim_data_read_during_selection": False,
    }


def _bootstrap_variance_ratio(
    baseline: list[dict[str, Any]], candidate: list[dict[str, Any]], draws: int = 5000
) -> dict[str, Any]:
    if not all(row["finite"] for row in baseline + candidate) or len(baseline) < 3:
        return {"available": False}
    baseline_by_seed = {row["seed"]: row for row in baseline}
    candidate_by_seed = {row["seed"]: row for row in candidate}
    common_seeds = sorted(set(baseline_by_seed) & set(candidate_by_seed))
    if len(common_seeds) < 3 or len(common_seeds) != len(baseline) or len(common_seeds) != len(candidate):
        return {
            "available": False,
            "reason": "baseline and candidate do not have the same complete seed set",
            "common_seed_count": len(common_seeds),
        }
    baseline = [baseline_by_seed[seed] for seed in common_seeds]
    candidate = [candidate_by_seed[seed] for seed in common_seeds]
    rng = random.Random(20260801)

    def ratio(indices: list[int]) -> float:
        base_sd = [
            statistics.stdev(baseline[index]["score"][coordinate] for index in indices)
            for coordinate in range(3)
        ]
        candidate_sd = [
            statistics.stdev(candidate[index]["score"][coordinate] for index in indices)
            for coordinate in range(3)
        ]
        return statistics.geometric_mean(
            (candidate_sd[index] / base_sd[index]) ** 2 for index in range(3)
        )

    samples = []
    while len(samples) < draws:
        indices = [rng.randrange(len(baseline)) for _ in baseline]
        try:
            value = ratio(indices)
        except (statistics.StatisticsError, ZeroDivisionError):
            continue
        if math.isfinite(value):
            samples.append(value)
    samples.sort()
    return {
        "available": True,
        "point": ratio(list(range(len(baseline)))),
        "ci95_lower": samples[int(0.025 * draws)],
        "ci95_upper": samples[int(0.975 * draws) - 1],
        "draws": draws,
    }


def _render(payload: dict[str, Any]) -> str:
    lines = [
        "# Projected-Cumulant GenUT Austria Comparison",
        "",
        f"- particles: `{payload['scope']['particles']}`",
        f"- hard_valid: `{payload['hard_valid']}`",
        "",
        "| Arm | Valid | Value mean (SD) | Score SD | SGQF value gap |",
        "|---|---:|---:|---|---:|",
    ]
    for arm_id, arm in payload["claim"].items():
        summary = arm["summary"]
        if not summary.get("all_valid", False):
            lines.append(f"| {arm_id} | False | unavailable | unavailable | unavailable |")
            continue
        score_sd = [summary[f"score_{index}"]["sample_sd"] for index in range(3)]
        lines.append(
            f"| {arm_id} | True | {summary['value']['mean']:.6g} "
            f"({summary['value']['sample_sd']:.6g}) | `{score_sd}` | "
            f"{summary['value']['mean'] - SGQF_COMPARATOR['value']:.6g} |"
        )
    lines += [
        "",
        "SGQF is a descriptive comparator, not an exact nonlinear Austria oracle.",
    ]
    return "\n".join(lines) + "\n"


def run(output_root: Path, *, particles: int, smoke: bool) -> dict[str, Any]:
    started = time.perf_counter()
    output_root.mkdir(parents=True, exist_ok=False)
    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    logical = tf.config.list_logical_devices("GPU")
    if not logical:
        raise RuntimeError("projected-cumulant campaign requires a logical GPU")
    if particles % 36:
        raise ValueError("Austria exact cubature design requires particles divisible by 36")
    target = _target(particles)

    calibration_rows, calibration_score = _mode_score_rows(
        target, particles, target["calibration"][:1] if smoke else target["calibration"]
    )
    validation_rows, validation_score = _mode_score_rows(
        target, particles, target["validation"][:1] if smoke else target["validation"]
    )
    if not all(_valid(row) for row in calibration_rows + validation_rows):
        _write_basis_failure(
            output_root,
            particles=particles,
            calibration_rows=calibration_rows,
            validation_rows=validation_rows,
            reason="diagonal basis-capture route failed a declared validity check",
        )
        raise RuntimeError("basis diagnostic diagonal route is invalid; see failure_result.json")
    calibration_basis, calibration_eigenvalues = _canonical_basis(calibration_score)
    validation_basis, validation_eigenvalues = _canonical_basis(validation_score)
    basis_diagnostics = _basis_diagnostics(
        calibration_score,
        validation_score,
        calibration_basis,
        validation_basis,
    )
    baseline_summary = _partition_summary(validation_rows)
    basis_artifact = {
        "diagnostics": basis_diagnostics,
        "calibration_rows": calibration_rows,
        "validation_rows": validation_rows,
    }
    _write_checkpoint(
        output_root,
        stage="BASIS_VALIDATION_COMPLETE",
        particles=particles,
        basis=basis_artifact,
    )

    tuning = {}
    if smoke:
        controls = PROJECTED_GRID[0]
        for rank in RANKS:
            tuning[str(rank)] = {
                "selection_status": "SMOKE_FIXED_FIRST_CONTROL",
                "selected_controls": controls,
                "candidates": [],
                "claim_data_read_during_selection": False,
            }
    else:
        for rank in RANKS:
            tuning[str(rank)] = _tune_rank(
                target,
                particles,
                calibration_basis[:, :, :rank],
                baseline_summary,
            )
            _write_checkpoint(
                output_root,
                stage=f"RANK_{rank}_TUNING_COMPLETE",
                particles=particles,
                basis=basis_artifact,
                tuning=tuning,
            )
        if not any(
            item["selected_controls"] is not None for item in tuning.values()
        ):
            raise RuntimeError(
                "all projected ranks failed mechanics tuning; see checkpoint.json"
            )

    claim_seeds = (98201,) if smoke else (
        CLAIM_SEEDS_1008 if particles == 1008 else CLAIM_SEEDS_4032
    )
    claim: dict[str, Any] = {}
    arms: list[tuple[str, dict[str, Any], tf.Tensor | None]] = [
        ("diagonal", BASE_CONTROLS, None),
        ("pairwise", PAIRWISE_CONTROLS, None),
    ]
    for rank in RANKS:
        controls = tuning[str(rank)]["selected_controls"]
        if controls is None:
            claim[f"projected_r{rank}"] = {
                "controls": None,
                "rows": [],
                "summary": {
                    "count": 0,
                    "finite_count": 0,
                    "all_valid": False,
                },
                "status": "NOT_RUN_NO_MECHANICALLY_VALID_TUNED_CONTROL",
            }
            continue
        arms.append(
            (f"projected_r{rank}", controls, calibration_basis[:, :, :rank])
        )
    for arm_id, controls, basis in arms:
        evaluator = _make_evaluator(target, particles, controls, basis=basis)
        rows = [
            _evaluate(evaluator, target, target["observations"], seed, particles)
            for seed in claim_seeds
        ]
        claim[arm_id] = {
            "controls": controls,
            "rows": rows,
            "summary": _summary(rows),
        }
        _write_checkpoint(
            output_root,
            stage=f"CLAIM_ARM_{arm_id}_COMPLETE",
            particles=particles,
            basis=basis_artifact,
            tuning=tuning,
            claim=claim,
        )
    if not smoke and particles == 1008:
        for rank in RANKS:
            arm_id = f"projected_r{rank}"
            claim[arm_id]["variance_ratio_to_diagonal"] = _bootstrap_variance_ratio(
                claim["diagonal"]["rows"], claim[arm_id]["rows"]
            )
            claim[arm_id]["variance_ratio_to_pairwise"] = _bootstrap_variance_ratio(
                claim["pairwise"]["rows"], claim[arm_id]["rows"]
            )

    payload = {
        "schema": SCHEMA,
        "plan": PLAN,
        "scope": {
            "target": "austria_sir_T20",
            "particles": particles,
            "ranks": RANKS,
            "claim_seeds": claim_seeds,
            "tuning_seeds": TUNING_SEEDS,
            "calibration_observation_seeds": [91141, 91142],
            "validation_observation_seeds": [91241, 91242],
            "state_dimension": 18,
            "parameter_dimension": 3,
            "horizon": 20,
            "dtype": "float32",
            "tf32": True,
            "jit_compile": True,
            "residual_tolerance": RESIDUAL_TOLERANCE,
            "displacement_veto": DISPLACEMENT_VETO,
            "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
            "source_observation_sha256": target["source_observation_sha256"],
        },
        "basis": {
            "sketch_count": SKETCH_COUNT,
            "sketch_seed": [20260801, 41],
            "third_order_sketch_scale": math.sqrt(15.0),
            "fourth_order_sketch_scale": math.sqrt(96.0),
            "calibration_basis_rank8": _safe(calibration_basis),
            "calibration_eigenvalues": _safe(calibration_eigenvalues),
            "validation_eigenvalues": _safe(validation_eigenvalues),
            "diagnostics": basis_diagnostics,
            "calibration_rows": calibration_rows,
            "validation_rows": validation_rows,
        },
        "tuning": tuning,
        "claim": claim,
        "sgqf_comparator": SGQF_COMPARATOR,
        "hard_valid": all(
            arm["summary"].get("all_valid", False) for arm in claim.values()
        ),
        "memory_policy": _safe(memory_policy),
        "device": [device.name for device in logical],
        "gpu_allocator": {
            key: int(value)
            for key, value in tf.config.experimental.get_memory_info("GPU:0").items()
        },
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "wall_time_seconds": time.perf_counter() - started,
        "manifest": {
            "command": [sys.executable, *sys.argv],
            "output_root": str(output_root),
            "environment": sys.prefix,
            "host": platform.node(),
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "source_sha256": {
                PLAN: _sha256(ROOT / PLAN),
                Path(__file__).relative_to(ROOT).as_posix(): _sha256(Path(__file__)),
                "bayesfilter/highdim/higher_moment_contract_e.py": _sha256(
                    ROOT / "bayesfilter/highdim/higher_moment_contract_e.py"
                ),
                "bayesfilter/highdim/cubature_genut_filter.py": _sha256(
                    ROOT / "bayesfilter/highdim/cubature_genut_filter.py"
                ),
            },
        },
        "review": {
            "local_skeptical_audit": "PASS_AFTER_REVISION",
            "claude_status": "unavailable_workspace_trust_gate_before_health_probe",
            "claude_verdict_claimed": False,
        },
        "nonclaims": [
            "no exact nonlinear Austria value or score oracle",
            "score SD is not absolute score error",
            "SGQF is a comparator, not an oracle",
            "N=4032 three-seed differences are descriptive only",
            "not source-faithful Zhao-Cui, HMC-ready, or default-ready",
        ],
    }
    (output_root / "result.json").write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_root / "result.md").write_text(_render(payload), encoding="utf-8")
    (output_root / "run_manifest.json").write_text(
        json.dumps(_safe(payload["manifest"]), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--particles", type=int, choices=(1008, 4032), required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    payload = run(args.output_root.resolve(), particles=args.particles, smoke=args.smoke)
    print(
        json.dumps(
            {
                "output": str(args.output_root.resolve()),
                "hard_valid": payload["hard_valid"],
                "wall_time_seconds": payload["wall_time_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
