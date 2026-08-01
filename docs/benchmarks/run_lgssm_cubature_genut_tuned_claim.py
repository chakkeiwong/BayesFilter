#!/usr/bin/env python3
"""Run the untouched tuned Cubature/GenUT LGSSM claim."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from docs.benchmarks import run_lgssm_cubature_genut_fp32 as runner
from docs.benchmarks.tune_lgssm_cubature_genut_fp32 import (
    CLAIM_SEEDS,
    HORIZON,
    SCHEMA_VERSION as TUNING_SCHEMA_VERSION,
)


SCHEMA_VERSION = "bayesfilter.lgssm_cubature_genut_tuned_claim.v3"
CAMPAIGN_ID = "lgssm-cubature-genut-recursive-score-tuned-claim-20260721"


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


def _configure_gpu() -> dict[str, Any]:
    physical = tf.config.list_physical_devices("GPU")
    if not physical:
        raise RuntimeError("claim requires a visible GPU")
    for device in physical:
        tf.config.experimental.set_memory_growth(device, True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    logical = tf.config.list_logical_devices("GPU")
    if not logical:
        raise RuntimeError("GPU initialization produced no logical GPU")
    return {
        "physical_devices": [device.name for device in physical],
        "logical_devices": [device.name for device in logical],
        "memory_growth": True,
        "tf32_mode": "enabled",
        "dtype": "float32",
        "jit_compile": False,
        "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
    }


def _load_tuning(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != TUNING_SCHEMA_VERSION:
        raise ValueError("unexpected tuning artifact schema")
    scope = payload.get("scope", {})
    expected = {
        "model_id": "canonical_lgssm_m3",
        "dataset_seed": runner.DATASET_SEED,
        "horizon": HORIZON,
        "particle_count": runner.NUM_PARTICLES,
        "state_dimension": runner.STATE_DIM,
        "dtype": "float32",
        "tf32_enabled": True,
        "jit_compile": False,
    }
    for key, value in expected.items():
        if scope.get(key) != value:
            raise ValueError(f"tuning scope mismatch for {key}: {scope.get(key)!r}")
    if payload.get("partitions", {}).get("claim") != list(CLAIM_SEEDS):
        raise ValueError("tuning claim partition mismatch")
    score_route = payload.get("score_route", {})
    if score_route.get("candidate") != runner.SCORE_ROUTE_ID:
        raise ValueError("tuning artifact does not bind the recursive score route")
    if not score_route.get("fd_restricted_to_representative_tuning_points"):
        raise ValueError("tuning artifact does not restrict FD to representative points")
    representative_points = payload.get("representative_points", {})
    if representative_points.get("count", 0) < 1 or not representative_points.get(
        "theta_values"
    ):
        raise ValueError("tuning artifact has no representative points")
    controls = payload.get("selected_controls")
    if not isinstance(controls, dict):
        raise ValueError("tuning artifact has no selected controls")
    if set(controls) != {"epsilon", "sinkhorn_steps", "ridge"}:
        raise ValueError("tuning controls do not match the claim control family")
    current_runner_hash = _sha256(
        runner.ROOT / "docs/benchmarks/run_lgssm_cubature_genut_fp32.py"
    )
    if payload.get("source_sha256", {}).get("runner") != current_runner_hash:
        raise ValueError("tuning artifact was generated from a different runner source")
    return payload


def _summary(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    labels = runner.LABELS
    relative_rows = [row["relative_error"] for row in rows]
    intervals = {
        label: runner._interval([row[index] for row in relative_rows])
        for index, label in enumerate(labels)
    }
    hard_valid = all(
        row["finite"]
        and row["bitwise_replay"]
        and row["score_route"] == runner.SCORE_ROUTE_ID
        and row["no_autodiff_score_route"]
        and not row["fd_audit_executed"]
        and row["reset_mean_cov_residual"] < 5.0e-4
        and row["sinkhorn_row_residual"] < 5.0e-4
        and row["sinkhorn_col_residual"] < 5.0e-4
        for row in rows
    )
    return {
        "method": method,
        "horizon": HORIZON,
        "particle_seeds": list(CLAIM_SEEDS),
        "kalman_value": rows[0]["kalman_value"],
        "kalman_score": rows[0]["kalman_score"],
        "kalman_hmc_score": rows[0]["kalman_hmc_score"],
        "relative_error_intervals": intervals,
        "screen": runner._screen(intervals, hard_valid),
        "screen_margins": {"value": runner.VALUE_MARGIN, "score": runner.SCORE_MARGIN},
        "hard_valid": hard_valid,
        "all_finite": all(row["finite"] for row in rows),
        "all_bitwise_replay": all(row["bitwise_replay"] for row in rows),
        "mean_particle_value": sum(row["particle_value"] for row in rows) / len(rows),
        "mean_particle_hmc_score": [
            sum(row["particle_hmc_score"][index] for row in rows) / len(rows)
            for index in range(runner.STATE_DIM + 2)
        ],
    }


def run(output_root: Path, tuning_path: Path) -> dict[str, Any]:
    tuning = _load_tuning(tuning_path)
    controls = tuning["selected_controls"]
    device = _configure_gpu()
    observations = runner._lgssm_observations(
        tf.constant(runner.THETA_VALUES, tf.float32), HORIZON
    )
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for method in ("cubature", "genut"):
        rows = [
            runner._evaluate_method(
                method,
                HORIZON,
                observations,
                particle_seed=seed,
                epsilon=float(controls["epsilon"]),
                sinkhorn_steps=int(controls["sinkhorn_steps"]),
                ridge=float(controls["ridge"]),
                jit_compile=False,
                diagnostics=False,
            )
            for seed in CLAIM_SEEDS
        ]
        results.extend(rows)
        summaries.append(_summary(rows, method))
    memory = tf.config.experimental.get_memory_info("GPU:0")
    hard_valid = all(item["hard_valid"] for item in summaries)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": CAMPAIGN_ID,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "git_commit": _git_commit(),
        "source_sha256": {
            "runner": _sha256(runner.ROOT / "docs/benchmarks/run_lgssm_cubature_genut_fp32.py"),
            "claim": _sha256(Path(__file__).resolve()),
            "tuning": _sha256(tuning_path),
        },
        "device": device,
        "configuration": {
            "state_dim": runner.STATE_DIM,
            "num_particles": runner.NUM_PARTICLES,
            "horizon": HORIZON,
            "dataset_seed": runner.DATASET_SEED,
            "claim_seeds": list(CLAIM_SEEDS),
            "theta": list(runner.THETA_VALUES),
            "controls": controls,
            "comparison_metric": "previous_lgssm_hmc_relative_error_simultaneous_ci",
            "comparison_labels": list(runner.LABELS),
            "critical_value": runner.CRITICAL_VALUE,
            "value_margin": runner.VALUE_MARGIN,
            "hmc_score_margin": runner.SCORE_MARGIN,
            "dtype": "float32",
            "tf32_mode": "enabled",
            "jit_compile": False,
            "score_route": runner.SCORE_ROUTE_ID,
            "representative_point_tuning": True,
            "finite_difference_runtime_score": False,
        },
        "tuning_artifact": {
            "path": str(tuning_path.resolve()),
            "sha256": _sha256(tuning_path),
            "selected_candidate_id": tuning["selected_candidate_id"],
            "selection_objective": tuning["selection_objective"],
        },
        "results": results,
        "summaries": summaries,
        "hard_valid": hard_valid,
        "wall_time_seconds": time.perf_counter() - started,
        "gpu_allocator": {key: int(value) for key, value in memory.items()},
        "nonclaims": [
            "grid-optimal only, not globally optimal",
            "no method superiority claim",
            "no exact filtering or nonlinear-model claim",
        ],
    }
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_root / "result.md").write_text(
        "# Tuned Cubature/GenUT LGSSM Claim\n\n"
        f"- hard_valid: {hard_valid}\n"
        f"- selected controls: `{json.dumps(controls, sort_keys=True)}`\n"
        f"- Cubature screen: `{summaries[0]['screen']}`\n"
        f"- GenUT screen: `{summaries[1]['screen']}`\n"
        f"- GPU peak bytes: {memory['peak']}\n\n"
        "The JSON contains all per-seed six-coordinate relative errors and the "
        "frozen tuning-artifact binding.\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output_root), "controls": controls, "screens": [s["screen"] for s in summaries]}, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tuning-artifact", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.output_root.resolve(), args.tuning_artifact.resolve())


if __name__ == "__main__":
    main()
