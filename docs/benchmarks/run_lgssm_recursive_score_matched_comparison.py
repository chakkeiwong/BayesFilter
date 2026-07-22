#!/usr/bin/env python3
"""Matched T=2,10,50 recursive-score comparison against Kalman."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
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

from docs.benchmarks import run_lgssm_cubature_genut_fp32 as runner


SCHEMA_VERSION = "bayesfilter.lgssm_recursive_score_matched_comparison.v1"
CAMPAIGN_ID = "lgssm-recursive-score-matched-t2-t10-t50-20260721"
PLAN_PATH = Path(
    "docs/plans/"
    "bayesfilter-lgssm-recursive-score-matched-three-horizon-comparison-plan-2026-07-21.md"
)
METHODS = ("contract_e_gaussian", "cubature")
METHOD_LABELS = {
    "contract_e_gaussian": "Contract E Gaussian residual",
    "cubature": "Cubature = Gaussian GenUT",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=runner.ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _interval(values: list[float]) -> dict[str, float]:
    if len(values) != len(runner.PARTICLE_SEEDS):
        raise ValueError("matched comparison requires exactly 16 seeds")
    mean = statistics.mean(values)
    standard_deviation = statistics.stdev(values)
    standard_error = standard_deviation / math.sqrt(len(values))
    radius = runner.CRITICAL_VALUE * standard_error
    return {
        "mean": mean,
        "standard_deviation": standard_deviation,
        "standard_error": standard_error,
        "critical_value": runner.CRITICAL_VALUE,
        "lower": mean - radius,
        "upper": mean + radius,
    }


def _method_summary(rows: list[dict[str, Any]], method: str, horizon: int) -> dict[str, Any]:
    coordinate_relative_intervals = {
        label: _interval([row["coordinate_relative_error"][index] for row in rows])
        for index, label in enumerate(runner.LABELS)
    }
    value_error_interval = _interval([row["value_error"] for row in rows])
    physical_score_error_intervals = {
        label: _interval([row["score_error"][index] for row in rows])
        for index, label in enumerate(runner.LABELS[1:])
    }
    physical_score_intervals = {
        label: _interval([row["particle_score"][index] for row in rows])
        for index, label in enumerate(runner.LABELS[1:])
    }
    hard_valid = all(
        row["finite"]
        and row["bitwise_replay"]
        and row["score_route"] == runner.SCORE_ROUTE_ID
        and row["no_autodiff_score_route"]
        and not row["fd_audit_executed"]
        and row["kalman_score_route"] == "analytic_recursive_kalman_score"
        and row["reset_mean_cov_residual"] < 5.0e-4
        and row["sinkhorn_row_residual"] < 5.0e-4
        and row["sinkhorn_col_residual"] < 5.0e-4
        for row in rows
    )
    return {
        "method": method,
        "method_label": METHOD_LABELS[method],
        "horizon": horizon,
        "particle_seeds": list(runner.PARTICLE_SEEDS),
        "hard_valid": hard_valid,
        "kalman_value": rows[0]["kalman_value"],
        "kalman_physical_score": rows[0]["kalman_score"],
        "mean_particle_value": statistics.mean(row["particle_value"] for row in rows),
        "mean_particle_physical_score": [
            statistics.mean(row["particle_score"][index] for row in rows)
            for index in range(runner.PARAMETER_DIM)
        ],
        "value_error_interval": value_error_interval,
        "physical_score_intervals": physical_score_intervals,
        "physical_score_error_intervals": physical_score_error_intervals,
        "coordinate_relative_error_intervals": coordinate_relative_intervals,
        "mean_wall_time_seconds": statistics.mean(row["wall_time_seconds"] for row in rows),
    }


def _paired_summary(
    gaussian_rows: list[dict[str, Any]], cubature_rows: list[dict[str, Any]], horizon: int
) -> dict[str, Any]:
    if [row["particle_seed"] for row in gaussian_rows] != [
        row["particle_seed"] for row in cubature_rows
    ]:
        raise ValueError("paired arms do not have identical ordered seeds")
    relative_absolute_error_deltas = {
        label: _interval(
            [
                abs(cubature["coordinate_relative_error"][index])
                - abs(gaussian["coordinate_relative_error"][index])
                for gaussian, cubature in zip(gaussian_rows, cubature_rows, strict=True)
            ]
        )
        for index, label in enumerate(runner.LABELS)
    }
    physical_absolute_score_error_deltas = {
        label: _interval(
            [
                abs(cubature["score_error"][index])
                - abs(gaussian["score_error"][index])
                for gaussian, cubature in zip(gaussian_rows, cubature_rows, strict=True)
            ]
        )
        for index, label in enumerate(runner.LABELS[1:])
    }
    return {
        "horizon": horizon,
        "delta_definition": "abs_error_cubature_minus_abs_error_contract_e_gaussian",
        "negative_favors": "cubature",
        "coordinate_relative_absolute_error_delta_intervals": relative_absolute_error_deltas,
        "physical_absolute_score_error_delta_intervals": physical_absolute_score_error_deltas,
    }


def _format_interval(interval: dict[str, float], *, percent: bool = False) -> str:
    scale = 100.0 if percent else 1.0
    suffix = "%" if percent else ""
    return (
        f"{scale * interval['mean']:+.3f}{suffix} "
        f"[{scale * interval['lower']:+.3f},{scale * interval['upper']:+.3f}]{suffix}"
    )


def run(output_root: Path) -> dict[str, Any]:
    device = runner._configure_gpu(jit_compile=False)
    theta = tf.constant(runner.THETA_VALUES, tf.float32)
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    by_horizon_method: dict[tuple[int, str], list[dict[str, Any]]] = {}

    cubature = runner.cubature_design(
        dim=runner.STATE_DIM, num_particles=runner.NUM_PARTICLES
    )
    genut, _ = runner.genut_gaussian_design(
        dim=runner.STATE_DIM, num_particles=runner.NUM_PARTICLES
    )
    cubature_genut_identical = bool(tf.reduce_all(cubature == genut).numpy())
    if not cubature_genut_identical:
        raise RuntimeError("Gaussian GenUT is not identical to Cubature")

    for horizon in runner.HORIZONS:
        observations = runner._lgssm_observations(theta, horizon)
        for method in METHODS:
            rows = [
                runner._evaluate_method(
                    method,
                    horizon,
                    observations,
                    particle_seed=seed,
                    jit_compile=False,
                    diagnostics=False,
                )
                for seed in runner.PARTICLE_SEEDS
            ]
            by_horizon_method[(horizon, method)] = rows
            results.extend(rows)
            summaries.append(_method_summary(rows, method, horizon))
        paired.append(
            _paired_summary(
                by_horizon_method[(horizon, "contract_e_gaussian")],
                by_horizon_method[(horizon, "cubature")],
                horizon,
            )
        )

    memory = tf.config.experimental.get_memory_info("GPU:0")
    hard_valid = all(summary["hard_valid"] for summary in summaries)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": CAMPAIGN_ID,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "git_commit": _git_commit(),
        "source_sha256": {
            "runner": _sha256(
                runner.ROOT / "docs/benchmarks/run_lgssm_cubature_genut_fp32.py"
            ),
            "campaign": _sha256(Path(__file__).resolve()),
            "plan": _sha256(runner.ROOT / PLAN_PATH),
        },
        "plan": str((runner.ROOT / PLAN_PATH).resolve()),
        "device": device,
        "configuration": {
            "horizons": list(runner.HORIZONS),
            "particle_count": runner.NUM_PARTICLES,
            "particle_seeds": list(runner.PARTICLE_SEEDS),
            "dataset_seed": runner.DATASET_SEED,
            "theta": list(runner.THETA_VALUES),
            "dtype": "float32",
            "tf32_enabled": True,
            "jit_compile": False,
            "epsilon": runner.EPSILON,
            "sinkhorn_steps": runner.SINKHORN_STEPS,
            "ridge": runner.RIDGE,
            "candidate_score_route": runner.SCORE_ROUTE_ID,
            "kalman_score_route": "analytic_recursive_kalman_score",
            "finite_difference_runtime_score": False,
            "cubature_genut_identical": cubature_genut_identical,
            "executed_particle_methods": list(METHODS),
            "reported_alias": "genut -> cubature",
        },
        "results": results,
        "summaries": summaries,
        "paired_comparisons": paired,
        "hard_valid": hard_valid,
        "wall_time_seconds": time.perf_counter() - started,
        "gpu_allocator": {key: int(value) for key, value in memory.items()},
        "inference_status": {
            "hard_veto_screen": "passed" if hard_valid else "failed",
            "statistically_supported_ranking": "coordinate_specific_only_if_paired_interval_excludes_zero",
            "descriptive_only_differences": "all other observed means and runtimes",
            "default_readiness": False,
            "next_evidence_needed": "nonlinear target validation and XLA route",
        },
        "nonclaims": [
            "no broad method superiority claim",
            "no nonlinear-model or NAWM claim",
            "no HMC or XLA readiness claim",
            "Gaussian GenUT is not independent evidence from Cubature",
        ],
    }
    output_root.mkdir(parents=True, exist_ok=False)
    json_path = output_root / "result.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Matched LGSSM Recursive-Score Comparison",
        "",
        f"- hard_valid: `{hard_valid}`",
        f"- wall time: `{payload['wall_time_seconds']:.1f} s`",
        f"- peak GPU allocator: `{payload['gpu_allocator']['peak']}` bytes",
        "- GenUT: verified bitwise-identical design alias of Cubature",
        "",
        "## Raw Physical-Score Error",
        "",
        "| T | Method | phi1 | phi2 | phi3 | q_scale | r_scale |",
        "|---:|---|---|---|---|---|---|",
    ]
    for horizon in runner.HORIZONS:
        for method in METHODS:
            summary = next(
                item for item in summaries
                if item["horizon"] == horizon and item["method"] == method
            )
            intervals = summary["physical_score_error_intervals"]
            fields = [
                str(horizon),
                summary["method_label"],
                *[_format_interval(intervals[label]) for label in runner.LABELS[1:]],
            ]
            lines.append("| " + " | ".join(fields) + " |")
    lines.extend(
        [
            "",
            "Raw physical-score errors are the primary comparison object. The JSON",
            "also contains coordinate-wise relative errors and paired absolute-error",
            "delta intervals. Relative errors are descriptive near zero reference",
            "coordinates. Negative paired deltas favor",
            "Cubature; positive paired deltas favor Gaussian-residual Contract E.",
        ]
    )
    (output_root / "result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output_root),
                "hard_valid": hard_valid,
                "wall_time_seconds": payload["wall_time_seconds"],
            },
            indent=2,
        )
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.output_root.resolve())


if __name__ == "__main__":
    main()
