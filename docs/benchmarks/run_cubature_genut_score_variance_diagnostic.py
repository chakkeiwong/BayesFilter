#!/usr/bin/env python3
"""Diagnostic score-variance comparison for the candidate SV route."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

from bayesfilter.highdim.cubature_genut_adapters import (
    exact_transformed_sv_candidate_adapter,
)
from bayesfilter.highdim.cubature_genut_candidate import cubature_design
from bayesfilter.highdim.cubature_genut_filter import finite_value_score


SEEDS = tuple(range(9000, 9016))
HORIZON = 4
THETA = tf.constant([0.25, -0.15], tf.float32)
OBSERVATIONS = tf.constant([[0.2], [-0.1], [0.3], [-0.4]], tf.float32)


def _evaluate(num_particles: int, seed: int, sign: float) -> tuple[float, list[float], dict[str, float]]:
    initial = sign * tf.random.stateless_normal([num_particles, 1], seed=[seed, 51])
    process = sign * tf.random.stateless_normal(
        [HORIZON, num_particles, 1], seed=[seed, 52]
    )
    design = cubature_design(dim=1, num_particles=num_particles)
    value, score, diagnostics = finite_value_score(
        exact_transformed_sv_candidate_adapter(),
        THETA,
        OBSERVATIONS,
        initial,
        process,
        design,
    )
    return (
        float(value.numpy()),
        [float(item) for item in score.numpy()],
        {
            "max_mean_residual": float(diagnostics["max_mean_residual"].numpy()),
            "max_row_residual": float(diagnostics["max_row_residual"].numpy()),
            "max_col_residual": float(diagnostics["max_col_residual"].numpy()),
        },
    )


def _summary(rows: list[list[float]]) -> dict[str, object]:
    labels = ("theta_gamma", "theta_log_beta")
    return {
        label: {
            "mean": statistics.mean(row[index] for row in rows),
            "standard_deviation": statistics.stdev(row[index] for row in rows),
            "minimum": min(row[index] for row in rows),
            "maximum": max(row[index] for row in rows),
        }
        for index, label in enumerate(labels)
    }


def run(output_root: Path) -> dict[str, object]:
    started = time.perf_counter()
    adapter = exact_transformed_sv_candidate_adapter()
    del adapter
    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for num_particles in (12, 24):
        independent_scores: list[list[float]] = []
        antithetic_scores: list[list[float]] = []
        independent_values: list[float] = []
        antithetic_values: list[float] = []
        max_residual = 0.0
        for seed in SEEDS:
            value, score, diagnostics = _evaluate(num_particles, seed, 1.0)
            anti_value, anti_score, anti_diagnostics = _evaluate(num_particles, seed, -1.0)
            independent_scores.append(score)
            independent_values.append(value)
            antithetic_scores.append(
                [(left + right) / 2.0 for left, right in zip(score, anti_score)]
            )
            antithetic_values.append((value + anti_value) / 2.0)
            max_residual = max(
                max_residual,
                diagnostics["max_mean_residual"],
                diagnostics["max_row_residual"],
                diagnostics["max_col_residual"],
                anti_diagnostics["max_mean_residual"],
                anti_diagnostics["max_row_residual"],
                anti_diagnostics["max_col_residual"],
            )
        independent_summary = _summary(independent_scores)
        antithetic_summary = _summary(antithetic_scores)
        variance_ratios = {
            label: antithetic_summary[label]["standard_deviation"]
            / independent_summary[label]["standard_deviation"]
            for label in independent_summary
        }
        summaries.append(
            {
                "particle_count": num_particles,
                "independent": independent_summary,
                "antithetic": antithetic_summary,
                "standard_deviation_ratio_antithetic_over_independent": variance_ratios,
                "independent_value_mean": statistics.mean(independent_values),
                "antithetic_value_mean": statistics.mean(antithetic_values),
                "max_residual": max_residual,
                "finite": all(math.isfinite(value) for value in independent_values + antithetic_values),
            }
        )
        for seed, score, anti_score in zip(SEEDS, independent_scores, antithetic_scores):
            rows.append(
                {
                    "particle_count": num_particles,
                    "seed": seed,
                    "independent_score": score,
                    "antithetic_pair_mean_score": anti_score,
                }
            )
    payload = {
        "schema_version": "bayesfilter.cubature_genut_score_variance_diagnostic.v1",
        "campaign_id": "cubature-genut-score-variance-sv-t4-20260721",
        "host": platform.node(),
        "dtype": "float32",
        "tf32_enabled": False,
        "jit_compile": False,
        "gpu_hidden": True,
        "horizon": HORIZON,
        "theta": [float(value) for value in THETA.numpy()],
        "seeds": list(SEEDS),
        "summaries": summaries,
        "rows": rows,
        "hard_valid": all(item["finite"] and item["max_residual"] < 1.0e-2 for item in summaries),
        "wall_time_seconds": time.perf_counter() - started,
        "nonclaims": [
            "variance reduction is diagnostic only",
            "no exact filtering or nonlinear default claim",
            "no leaderboard admission or method superiority claim",
        ],
    }
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "result.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Candidate Score Variance Diagnostic",
        "",
        f"- hard_valid: `{payload['hard_valid']}`",
        f"- wall time: `{payload['wall_time_seconds']:.3f} s`",
        "",
        "| N | Coordinate | Independent SD | Antithetic SD | Ratio |",
        "|---:|---|---:|---:|---:|",
    ]
    for summary in summaries:
        for label in ("theta_gamma", "theta_log_beta"):
            lines.append(
                f"| {summary['particle_count']} | {label} | "
                f"{summary['independent'][label]['standard_deviation']:.6g} | "
                f"{summary['antithetic'][label]['standard_deviation']:.6g} | "
                f"{summary['standard_deviation_ratio_antithetic_over_independent'][label]:.6g} |"
            )
    (output_root / "result.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"output": str(output_root), "hard_valid": payload["hard_valid"]}, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.output_root.resolve())


if __name__ == "__main__":
    main()
