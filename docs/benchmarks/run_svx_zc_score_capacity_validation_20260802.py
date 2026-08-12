#!/usr/bin/env python3
"""Validate SVX-ZC score consistency at the value-tuning neighborhood."""

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
from typing import Any, Mapping

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", "/tmp")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

import docs.benchmarks.run_contract_e_tp_phase6_zhao_cui_comparator as comparator
from bayesfilter.highdim.zhao_cui_fixed_adjacent_tt_tf import (
    scalar_adjacent_state_fixed_tt_score,
)


PLAN = "docs/plans/bayesfilter-svx-zc-score-capacity-validation-plan-2026-08-02.md"
ROW = "actual_sv"
DATA_SEED = 81101
HORIZON = 10
COORDINATE_HALF_WIDTH = 8.0
FD_STEPS = (1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4)
CELLS = ((10, 2, 25), (12, 2, 25), (10, 4, 25), (10, 2, 29), (10, 2, 33))
MAX_WALL_TIME_SECONDS = 30.0 * 60.0
PARAMETER_NAMES = ("gamma_unconstrained", "log_beta")


def _json_value(value: Any) -> Any:
    if isinstance(value, tf.Tensor):
        if value.shape.rank == 0:
            if value.dtype == tf.bool:
                return bool(value.numpy())
            if value.dtype.is_integer:
                return int(value.numpy())
            if value.dtype.is_floating:
                scalar = float(value.numpy())
                return scalar if math.isfinite(scalar) else str(scalar)
            return str(value.numpy())
        return [_json_value(item) for item in tf.unstack(value)]
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, (str, int, float, bool)):
        return _json_value(enum_value)
    return value


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _git_status() -> str:
    return subprocess.check_output(
        ["git", "status", "--short"], cwd=ROOT, text=True
    ).strip()


def _sha256_tensor(value: tf.Tensor) -> str:
    return hashlib.sha256(tf.io.serialize_tensor(value).numpy()).hexdigest()


def _fd_payload(result: Any) -> dict[str, Any]:
    rows = []
    grouped: dict[int, list[Any]] = {}
    for row in result.finite_difference_table.rows:
        grouped.setdefault(int(row.parameter_index), []).append(row)
        rows.append(
            {
                "parameter_index": int(row.parameter_index),
                "parameter": PARAMETER_NAMES[int(row.parameter_index)],
                "h": float(row.h.numpy()),
                "score": float(row.analytic_gradient.numpy()),
                "finite_difference": float(row.centered_difference.numpy()),
                "absolute_error": float(row.abs_error.numpy()),
                "relative_error_for_policy": float(row.rel_error.numpy()),
                "branch_hash_base": row.branch_hash_base,
                "branch_hash_plus": row.branch_hash_plus,
                "branch_hash_minus": row.branch_hash_minus,
                "row_status": row.row_status.value,
            }
        )

    threshold = 0.05 * math.sqrt(len(PARAMETER_NAMES))
    stable_windows = []
    selected_fd = []
    for parameter_index, parameter_name in enumerate(PARAMETER_NAMES):
        parameter_rows = sorted(
            grouped.get(parameter_index, []),
            key=lambda row: float(row.h.numpy()),
            reverse=True,
        )
        relative = []
        for row in parameter_rows:
            score = float(row.analytic_gradient.numpy())
            finite_difference = float(row.centered_difference.numpy())
            error = abs(score - finite_difference) / max(
                abs(score), abs(finite_difference), 1.0e-12
            )
            relative.append((row, error))
        windows = []
        for (left, left_error), (right, right_error) in zip(
            relative[:-1], relative[1:]
        ):
            compatible = (
                left.row_status.value == "VALID"
                and right.row_status.value == "VALID"
                and left_error <= threshold
                and right_error <= threshold
            )
            shape_ok = right_error <= left_error or abs(
                right_error - left_error
            ) <= 0.1 * threshold
            windows.append(
                {
                    "h_large": float(left.h.numpy()),
                    "h_small": float(right.h.numpy()),
                    "relative_error_large": left_error,
                    "relative_error_small": right_error,
                    "pass": compatible and shape_ok,
                }
            )
        passing = [window for window in windows if window["pass"]]
        stable_windows.append(
            {
                "parameter": parameter_name,
                "threshold": threshold,
                "windows": windows,
                "status": "pass" if passing else "fail",
            }
        )
        if passing:
            selected_h = passing[-1]["h_small"]
            selected = next(
                row
                for row in parameter_rows
                if math.isclose(float(row.h.numpy()), selected_h)
            )
        else:
            selected = min(
                parameter_rows,
                key=lambda row: abs(
                    float(row.analytic_gradient.numpy())
                    - float(row.centered_difference.numpy())
                ),
            )
        selected_fd.append(float(selected.centered_difference.numpy()))

    score = [float(value) for value in result.score.numpy()]
    relative_errors = [
        abs(a - b) / max(abs(a), abs(b), 1.0e-12)
        for a, b in zip(score, selected_fd)
    ]
    branch_hashes = {
        row["branch_hash_base"] for row in rows
    } | {row["branch_hash_plus"] for row in rows} | {
        row["branch_hash_minus"] for row in rows
    }
    all_rows_valid = all(row["row_status"] == "VALID" for row in rows)
    stable_pass = all(item["status"] == "pass" for item in stable_windows)
    policy_pass = all(error <= threshold for error in relative_errors)
    return {
        "steps": list(FD_STEPS),
        "rows": rows,
        "selected_fd_for_policy": selected_fd,
        "relative_errors_for_policy": relative_errors,
        "threshold": threshold,
        "all_rows_valid": all_rows_valid,
        "branch_hash_count": len(branch_hashes),
        "stable_windows": stable_windows,
        "stable_window_pass": stable_pass,
        "policy_pass": policy_pass,
        "status": "pass" if all_rows_valid and len(branch_hashes) == 1 and stable_pass and policy_pass else "fail",
    }


def _run_cell(
    model: Any,
    theta: tf.Tensor,
    observations: tf.Tensor,
    raw_observations: tf.Tensor,
    degree: int,
    rank: int,
    order: int,
    started: float,
) -> dict[str, Any]:
    initial_cores, adjacent_cores, ukf_manifest = comparator._ukf_initial_cores(
        model=model,
        theta=theta,
        raw_observations=raw_observations,
        degree=degree,
        order=order,
        rank=rank,
        coordinate_half_width=COORDINATE_HALF_WIDTH,
    )
    seed = f"svx-zc-score-capacity-20260802:d{degree}:r{rank}:o{order}"
    config = comparator._comparator_config(
        degree=degree,
        order=order,
        rank=rank,
        seed=seed,
        transition_before_first_observation=False,
        coordinate_half_width=COORDINATE_HALF_WIDTH,
        density_tau=0.0,
        initial_cores=initial_cores,
        adjacent_initial_cores=adjacent_cores,
        initialization_rule=str(ukf_manifest["initializer_rule"]),
    )
    result = scalar_adjacent_state_fixed_tt_score(
        model,
        theta,
        observations,
        config,
        finite_difference_h=FD_STEPS,
        fixture_id=f"svx-zc-score-capacity.d{degree}.r{rank}.o{order}",
        branch_seed_prefix=seed,
    )
    fd = _fd_payload(result)
    value = float(result.log_likelihood.numpy())
    score = [float(value) for value in result.score.numpy()]
    finite = math.isfinite(value) and all(math.isfinite(value) for value in score)
    hard_pass = finite and fd["status"] == "pass"
    return {
        "schema": "bayesfilter.svx_zc.score_capacity.cell.v1",
        "capacity": {"degree": degree, "rank": rank, "order": order},
        "status": "PASS_DERIVATIVE_CONSISTENCY" if hard_pass else "BLOCKED_DERIVATIVE_CONSISTENCY",
        "value": value,
        "score": score,
        "finite_value_and_score": finite,
        "finite_difference": fd,
        "compatibility_hash": result.diagnostics["compatibility_hash"],
        "branch_identity_hash": result.branch_identity.hash.value,
        "ukf_initializer": _json_value(ukf_manifest),
        "diagnostics": _json_value(result.diagnostics),
        "wall_time_seconds": time.perf_counter() - started,
    }


def run(output_root: Path) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing root: {output_root}")
    output_root.mkdir(parents=True)
    campaign_started = time.perf_counter()
    model, theta, observations = comparator._row_inputs(ROW, HORIZON)
    raw_observations = tf.convert_to_tensor(
        comparator._sv_dataset(DATA_SEED)["observations"], dtype=tf.float64
    )[:HORIZON]
    cells = []
    errors = []
    for degree, rank, order in CELLS:
        if time.perf_counter() - campaign_started > MAX_WALL_TIME_SECONDS:
            errors.append("wall-time budget exhausted")
            break
        cell_started = time.perf_counter()
        try:
            cell = _run_cell(
                model, theta, observations, raw_observations,
                degree, rank, order, cell_started
            )
        except Exception as exc:  # preserve a structured failure artifact
            cell = {
                "schema": "bayesfilter.svx_zc.score_capacity.cell.v1",
                "capacity": {"degree": degree, "rank": rank, "order": order},
                "status": "HARNESS_OR_NUMERICAL_ERROR",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "wall_time_seconds": time.perf_counter() - cell_started,
            }
            errors.append(f"d{degree}r{rank}o{order}: {type(exc).__name__}: {exc}")
        cells.append(cell)
        (output_root / f"cell-d{degree}-r{rank}-o{order}.json").write_text(
            json.dumps(_json_value(cell), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    center = next((cell for cell in cells if cell.get("capacity") == {"degree": 10, "rank": 2, "order": 25}), None)
    comparisons = []
    if center and "score" in center:
        center_score = center["score"]
        for cell in cells:
            if cell is center or "score" not in cell:
                continue
            delta = [float(a - b) for a, b in zip(cell["score"], center_score)]
            scale = [max(abs(a), abs(b), 1.0e-12) for a, b in zip(cell["score"], center_score)]
            comparisons.append({
                "center_capacity": center["capacity"],
                "other_capacity": cell["capacity"],
                "score_delta_other_minus_center": delta,
                "componentwise_relative_delta": [abs(d) / s for d, s in zip(delta, scale)],
                "max_componentwise_relative_delta": max(abs(d) / s for d, s in zip(delta, scale)),
                "interpretation": "descriptive_only_not_a_promotion_criterion",
            })
    all_derivative_pass = bool(cells) and len(cells) == len(CELLS) and all(
        cell.get("status") == "PASS_DERIVATIVE_CONSISTENCY" for cell in cells
    )
    result = {
        "schema": "bayesfilter.svx_zc.score_capacity.result.v1",
        "status": "DERIVATIVE_CONSISTENT_FOR_TESTED_CELLS" if all_derivative_pass else "SCORE_VALIDATION_BLOCKED",
        "decision": (
            "The score is internally consistent with the tested finite likelihood programs; cross-capacity changes remain descriptive."
            if all_derivative_pass
            else "Do not promote score consistency; inspect failed cell diagnostics."
        ),
        "research_question": "same-scalar derivative consistency and local score capacity sensitivity",
        "cells": cells,
        "capacity_comparisons": comparisons,
        "errors": errors,
        "nonclaims": [
            "not exact score correctness",
            "not score convergence outside tested cells",
            "not exact likelihood accuracy",
            "not HMC readiness",
            "not posterior correctness",
            "not GPU/XLA readiness",
            "not production readiness",
            "not cross-scope transfer",
        ],
        "run_manifest": {
            "schema": "bayesfilter.svx_zc.score_capacity.run_manifest.v1",
            "git_commit": _git_commit(),
            "git_status_short": _git_status(),
            "command": " ".join(sys.argv),
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "tensorflow_version": tf.__version__,
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV", "not_detected"),
            "device": "CPU-only; CUDA_VISIBLE_DEVICES=-1",
            "gpu_devices_intentionally_hidden": True,
            "dtype": "float64",
            "jit_compile": False,
            "data_seed": DATA_SEED,
            "horizon": HORIZON,
            "cells": CELLS,
            "finite_difference_h": FD_STEPS,
            "plan": PLAN,
            "output_root": str(output_root.relative_to(ROOT)),
            "completed_wall_time_seconds": time.perf_counter() - campaign_started,
            "trust_basis": "deliberate_cpu_only_diagnostic_exception",
        },
    }
    (output_root / "result.json").write_text(
        json.dumps(_json_value(result), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_root / "run_manifest.json").write_text(
        json.dumps(_json_value(result["run_manifest"]), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    result = run(root)
    print(json.dumps({"status": result["status"], "output_root": str(root)}))


if __name__ == "__main__":
    main()
