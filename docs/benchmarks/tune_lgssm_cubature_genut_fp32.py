#!/usr/bin/env python3
"""Scope-specific calibration/validation tuner for Cubature/GenUT LGSSM."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

import tensorflow as tf

from docs.benchmarks import run_lgssm_cubature_genut_fp32 as runner


SCHEMA_VERSION = "bayesfilter.lgssm_cubature_genut_tuning.v3"
CAMPAIGN_ID = "lgssm-cubature-genut-recursive-score-fd-audit-tuning-20260721"
HORIZON = 50
CALIBRATION_SEEDS = tuple(range(82300, 82304))
VALIDATION_SEEDS = tuple(range(82310, 82314))
CLAIM_SEEDS = tuple(range(82320, 82336))
REPRESENTATIVE_POINT_COUNT = 6
REPRESENTATIVE_POINT_SEED = (82400, 917)
PHI_BOUNDS = (0.25, 0.85)
SCALE_BOUNDS = (0.25, 0.65)
# Raw relative error is ill-conditioned when a reference score component is
# near zero. These floors affect selection only; raw errors remain recorded.
TUNING_ERROR_SCALE_FLOORS = (1.0, 0.1, 0.1, 0.1, 0.1, 0.1)
FD_AUDIT_ATOL = 5.0e-2
FD_AUDIT_RTOL = 5.0e-2
GRID = tuple(
    {
        "epsilon": epsilon,
        "sinkhorn_steps": sinkhorn_steps,
        "ridge": ridge,
    }
    for epsilon in (0.5, 1.0, 2.0, 4.0)
    for sinkhorn_steps in (4, 8, 16)
    for ridge in (1.0e-6, 1.0e-5)
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _configure_gpu() -> dict[str, Any]:
    physical = tf.config.list_physical_devices("GPU")
    if not physical:
        raise RuntimeError("tuning requires a visible GPU")
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


def _observations() -> tf.Tensor:
    return runner._lgssm_observations(
        tf.constant(runner.THETA_VALUES, tf.float32), HORIZON
    )


def _representative_points() -> tuple[tf.Tensor, dict[str, Any]]:
    """Create the fixed valid theta points used by every grid candidate."""
    raw = tf.random.stateless_uniform(
        [REPRESENTATIVE_POINT_COUNT, runner.STATE_DIM + 2],
        seed=REPRESENTATIVE_POINT_SEED,
        dtype=tf.float32,
    )
    phi = PHI_BOUNDS[0] + (PHI_BOUNDS[1] - PHI_BOUNDS[0]) * raw[:, :runner.STATE_DIM]
    scales = SCALE_BOUNDS[0] + (SCALE_BOUNDS[1] - SCALE_BOUNDS[0]) * raw[:, runner.STATE_DIM:]
    points = tf.concat([phi, scales], axis=1)
    metadata = {
        "count": REPRESENTATIVE_POINT_COUNT,
        "seed": list(REPRESENTATIVE_POINT_SEED),
        "phi_bounds": list(PHI_BOUNDS),
        "scale_bounds": list(SCALE_BOUNDS),
        "theta_values": points.numpy().tolist(),
    }
    return points, metadata


def _evaluate_partition(
    controls: dict[str, Any],
    seeds: tuple[int, ...],
    observations: tf.Tensor,
    representative_points: tf.Tensor,
) -> dict[str, Any]:
    rows = []
    for seed in seeds:
        for point_index, point in enumerate(tf.unstack(representative_points)):
            row = runner._evaluate_method(
                "cubature",
                HORIZON,
                observations,
                particle_seed=seed,
                epsilon=float(controls["epsilon"]),
                sinkhorn_steps=int(controls["sinkhorn_steps"]),
                ridge=float(controls["ridge"]),
                jit_compile=False,
                diagnostics=True,
                theta_values=tuple(float(value) for value in point.numpy()),
            )
            row["representative_point_index"] = point_index
            row["representative_theta"] = [float(value) for value in point.numpy()]
            reference = [
                abs(row["kalman_value"]),
                *[abs(value) for value in row["kalman_hmc_score"]],
            ]
            raw_error = [
                row["value_error"],
                *[
                    value * chain
                    for value, chain in zip(
                        row["score_error"],
                        (
                            1.0 - row["representative_theta"][0] ** 2,
                            1.0 - row["representative_theta"][1] ** 2,
                            1.0 - row["representative_theta"][2] ** 2,
                            row["representative_theta"][3],
                            row["representative_theta"][4],
                        ),
                        strict=True,
                    )
                ],
            ]
            row["tuning_error"] = [
                error / max(scale, floor)
                for error, scale, floor in zip(
                    raw_error, reference, TUNING_ERROR_SCALE_FLOORS, strict=True
                )
            ]
            recursive_score = row["particle_score"]
            fd_score = row["finite_difference_score"]
            coordinate_scaled_errors = [
                abs(recursive - finite_difference)
                / (
                    FD_AUDIT_ATOL
                    + FD_AUDIT_RTOL
                    * max(abs(recursive), abs(finite_difference))
                )
                for recursive, finite_difference in zip(
                    recursive_score, fd_score, strict=True
                )
            ]
            row["recursive_fd_coordinate_scaled_error"] = coordinate_scaled_errors
            row["recursive_fd_scaled_error"] = max(coordinate_scaled_errors)
            row["recursive_fd_pass"] = all(
                value <= 1.0 for value in coordinate_scaled_errors
            )
            rows.append(row)
    labels = runner.LABELS
    means = {
        label: sum(row["tuning_error"][index] for row in rows) / len(rows)
        for index, label in enumerate(labels)
    }
    hard_valid = all(
        row["finite"]
        and row["bitwise_replay"]
        and row["score_route"] == runner.SCORE_ROUTE_ID
        and row["no_autodiff_score_route"]
        and row["fd_audit_executed"]
        and row["recursive_fd_pass"]
        and all(math.isfinite(value) for value in row["tuning_error"])
        and all(math.isfinite(value) for value in row["relative_error"])
        and row["reset_mean_cov_residual"] < 5.0e-4
        and row["sinkhorn_row_residual"] < 5.0e-4
        and row["sinkhorn_col_residual"] < 5.0e-4
        for row in rows
    )
    max_target_error = max(abs(value) for value in means.values())
    max_fd_parity_error = max(row["recursive_fd_scaled_error"] for row in rows)
    objective = max(max_target_error, max_fd_parity_error) if hard_valid else math.inf
    squared_mean = sum(value * value for value in means.values()) if hard_valid else math.inf
    return {
        "controls": dict(controls),
        "seeds": list(seeds),
        "rows": rows,
        "mean_relative_error": means,
        "max_abs_mean_relative_error": objective,
        "max_abs_mean_target_error": max_target_error,
        "max_recursive_fd_scaled_error": max_fd_parity_error,
        "sum_squared_mean_relative_error": squared_mean,
        "hard_valid": hard_valid,
        "representative_point_count": int(representative_points.shape[0]),
        "max_recursive_fd_abs_error": max(
            row["finite_difference_max_abs_error"] for row in rows
        ),
        "max_recursive_fd_relative_error": max(
            row["finite_difference_max_relative_error"] for row in rows
        ),
    }


def tune(output_root: Path) -> dict[str, Any]:
    device = _configure_gpu()
    observations = _observations()
    representative_points, representative_metadata = _representative_points()
    started = time.perf_counter()
    candidates = []
    for index, controls in enumerate(GRID):
        calibration = _evaluate_partition(
            controls, CALIBRATION_SEEDS, observations, representative_points
        )
        validation = _evaluate_partition(
            controls, VALIDATION_SEEDS, observations, representative_points
        )
        candidates.append(
            {
                "candidate_id": index,
                "controls": dict(controls),
                "calibration": calibration,
                "validation": validation,
                "valid_for_selection": calibration["hard_valid"] and validation["hard_valid"],
            }
        )
    valid = [item for item in candidates if item["valid_for_selection"]]
    if not valid:
        raise RuntimeError("all Cubature candidates failed calibration/validation")
    selected = min(
        valid,
        key=lambda item: (
            item["validation"]["max_abs_mean_relative_error"],
            item["validation"]["sum_squared_mean_relative_error"],
            item["controls"]["sinkhorn_steps"],
            item["controls"]["ridge"],
        ),
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": CAMPAIGN_ID,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _git_commit(),
        "source_sha256": {
            "runner": _sha256(ROOT / "docs/benchmarks/run_lgssm_cubature_genut_fp32.py"),
            "tuner": _sha256(Path(__file__).resolve()),
        },
        "device": device,
        "scope": {
            "model_id": "canonical_lgssm_m3",
            "dataset_seed": runner.DATASET_SEED,
            "horizon": HORIZON,
            "particle_count": runner.NUM_PARTICLES,
            "state_dimension": runner.STATE_DIM,
            "dtype": "float32",
            "tf32_enabled": True,
            "jit_compile": False,
            "reset_design_family": "cubature_genut_gaussian",
        },
        "partitions": {
            "calibration": list(CALIBRATION_SEEDS),
            "validation": list(VALIDATION_SEEDS),
            "claim": list(CLAIM_SEEDS),
        },
        "representative_points": representative_metadata,
        "score_route": {
            "candidate": runner.SCORE_ROUTE_ID,
            "reference": "central_finite_difference_kalman_value_program",
            "implementation_audit": "central_finite_difference_same_candidate_value_program",
            "relative_step": runner.FD_EPS,
            "minimum_step": runner.FD_MIN_STEP,
            "autodiff_used_for_selection": False,
            "fd_audit_atol": FD_AUDIT_ATOL,
            "fd_audit_rtol": FD_AUDIT_RTOL,
            "fd_restricted_to_representative_tuning_points": True,
            "tuning_error_scale_floors": list(TUNING_ERROR_SCALE_FLOORS),
        },
        "grid": [dict(item) for item in GRID],
        "candidates": candidates,
        "selected_candidate_id": selected["candidate_id"],
        "selected_controls": selected["controls"],
        "selection_objective": {
            "name": "max_of_representative_target_error_and_recursive_fd_scaled_error",
            "validation_value": selected["validation"]["max_abs_mean_relative_error"],
            "validation_target_error": selected["validation"]["max_abs_mean_target_error"],
            "validation_recursive_fd_scaled_error": selected["validation"][
                "max_recursive_fd_scaled_error"
            ],
            "validation_sum_squared": selected["validation"]["sum_squared_mean_relative_error"],
        },
        "wall_time_seconds": time.perf_counter() - started,
        "hard_valid": True,
        "nonclaims": [
            "grid-optimal only, not globally optimal",
            "tuning artifact only; no claim-run admission by itself",
            "no method superiority or exact-filtering claim",
        ],
    }
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "tuning.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_root / "selected_controls.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "scope": payload["scope"],
                "partitions": payload["partitions"],
                "selected_candidate_id": payload["selected_candidate_id"],
                "selected_controls": payload["selected_controls"],
                "selection_objective": payload["selection_objective"],
                "representative_points": payload["representative_points"],
                "score_route": payload["score_route"],
                "source_sha256": payload["source_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"selected_controls": selected["controls"], "output": str(output_root)}, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    tune(args.output_root.resolve())


if __name__ == "__main__":
    main()
