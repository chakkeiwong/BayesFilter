#!/usr/bin/env python3
"""Oracle-free retuning trial for the existing higher-moment reset.

The filtering algorithm is unchanged.  This runner reuses the established
model fixtures and claim protocol, but selects controls by the normalized
diagonal skewness/kurtosis residual emitted by the finite program.  Dense
oracles, where available, are retained only for post-run diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

import docs.benchmarks.run_genut_transport_repair_regressions as base
import docs.benchmarks.run_higher_moment_contract_e_regressions as established


PLAN = Path(
    "docs/plans/bayesfilter-higher-moment-genut-retuning-trial-plan-2026-07-23.md"
)
OUTPUT_SCHEMA = "bayesfilter.higher_moment_genut_retuning_trial.v1"
DISPLACEMENT_VETO = 2.0
RESIDUAL_TOLERANCE = 5.0e-4
CONTROL_GRID = tuple(
    {
        "epsilon": 2.0,
        "sinkhorn_steps": 8,
        "balance_steps": 8,
        "ridge": 1.0e-5,
        "higher_moment_correction_steps": steps,
        "higher_moment_strength": strength,
        "higher_moment_floor": 1.0e-5,
    }
    for steps in (0, 1, 2, 4)
    for strength in (0.02, 0.05, 0.10, 0.20)
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evaluate(evaluate, *arguments) -> dict[str, Any]:
    value, score, diagnostics = evaluate(*arguments)
    score_sum_error = tf.reduce_max(
        tf.abs(tf.reduce_sum(diagnostics["score_increments"], axis=0) - score)
    )
    finite = (
        bool(diagnostics["program_valid"].numpy())
        and bool(tf.math.is_finite(value).numpy())
        and bool(tf.reduce_all(tf.math.is_finite(score)).numpy())
    )
    return {
        "value": float(value.numpy()) if finite else None,
        "score": [float(item) for item in score.numpy()] if finite else None,
        "finite": finite,
        "program_valid": bool(diagnostics["program_valid"].numpy()),
        "max_mean_residual": float(diagnostics["max_mean_residual"].numpy()),
        "max_row_residual": float(diagnostics["max_row_residual"].numpy()),
        "max_col_residual": float(diagnostics["max_col_residual"].numpy()),
        "max_transition_residual": float(
            diagnostics["max_transition_residual"].numpy()
        ),
        "score_increment_sum_residual": float(score_sum_error.numpy()),
        "minimum_row_mass": float(diagnostics["minimum_row_mass"].numpy()),
        "maximum_post_quotient_column_tv_error": float(
            diagnostics["maximum_post_quotient_column_tv_error"].numpy()
        ),
        "minimum_covariance_gap_eigenvalue": float(
            diagnostics["minimum_covariance_gap_eigenvalue"].numpy()
        ),
        "maximum_skew_residual": float(
            diagnostics["maximum_skew_residual"].numpy()
        ),
        "maximum_kurtosis_residual": float(
            diagnostics["maximum_kurtosis_residual"].numpy()
        ),
        "maximum_shape_displacement": float(
            diagnostics["maximum_shape_displacement"].numpy()
        ),
        "maximum_normalized_shape_displacement": float(
            diagnostics["maximum_normalized_shape_displacement"].numpy()
        ),
        "mean_normalized_shape_residual_objective": float(
            diagnostics["mean_normalized_shape_residual_objective"].numpy()
        ),
        "device": str(value.device),
    }


def _valid(row: dict[str, Any]) -> bool:
    return (
        bool(row["finite"])
        and "GPU" in str(row["device"]).upper()
        and max(
            row["max_mean_residual"],
            row["max_row_residual"],
            row["max_col_residual"],
            row["max_transition_residual"],
            row["score_increment_sum_residual"],
        ) < RESIDUAL_TOLERANCE
        and row["maximum_normalized_shape_displacement"] <= DISPLACEMENT_VETO
    )


def _variance_objective(rows: list[dict[str, Any]], scales: tuple[float, ...]) -> float:
    vectors = [[row["value"], *row["score"]] for row in rows]
    return max(
        statistics.variance(vector[index] for vector in vectors) / (scales[index] ** 2)
        for index in range(len(scales))
    )


def _tune(*, name, evaluator_factory, datasets, particle_seeds, arguments, scales):
    candidates = []
    for controls in CONTROL_GRID:
        evaluate = evaluator_factory(controls)
        objectives: dict[str, float | None] = {}
        variance_objectives: dict[str, float | None] = {}
        eligible = True
        maximum_displacement = 0.0
        for partition, observations_set in datasets.items():
            moment_rows = []
            variance_rows = []
            for observations in observations_set:
                rows = [
                    _evaluate(evaluate, *arguments(observations, seed))
                    for seed in particle_seeds
                ]
                eligible = eligible and all(_valid(row) for row in rows)
                if all(row["finite"] for row in rows):
                    moment_rows.append(
                        statistics.mean(
                            row["mean_normalized_shape_residual_objective"]
                            for row in rows
                        )
                    )
                    variance_rows.append(_variance_objective(rows, scales))
                maximum_displacement = max(
                    maximum_displacement,
                    *(row["maximum_normalized_shape_displacement"] for row in rows),
                )
            objectives[partition] = (
                statistics.mean(moment_rows)
                if len(moment_rows) == len(observations_set)
                else None
            )
            variance_objectives[partition] = (
                statistics.mean(variance_rows)
                if len(variance_rows) == len(observations_set)
                else None
            )
        candidates.append(
            {
                "controls": controls,
                "objectives": objectives,
                "variance_objectives": variance_objectives,
                "maximum_normalized_shape_displacement": maximum_displacement,
                "eligible": eligible,
            }
        )
    eligible = [row for row in candidates if row["eligible"]]
    if not eligible:
        raise RuntimeError(f"no eligible controls for {name}")
    selected = min(
        eligible,
        key=lambda row: (
            row["objectives"]["validation"],
            row["objectives"]["calibration"],
            row["variance_objectives"]["validation"],
            float(row["controls"]["higher_moment_strength"]),
            int(row["controls"]["higher_moment_correction_steps"]),
        ),
    )
    return {
        "scope": name,
        "selection_objective": "mean normalized diagonal skewness+kurtosis residual",
        "secondary_objective": "scaled conditional value and recursive score variance",
        "particle_seeds": particle_seeds,
        "control_grid": CONTROL_GRID,
        "selected_controls": selected["controls"],
        "selected_objectives": selected["objectives"],
        "selected_variance_objectives": selected["variance_objectives"],
        "candidates": candidates,
        "displacement_veto": DISPLACEMENT_VETO,
        "claim_data_read_during_selection": False,
    }


def _run_suite() -> dict[str, Any]:
    # The established fixture functions resolve their tuning/evaluation helpers
    # from the imported base module. Replace only those helpers for this trial.
    base._make_evaluator = established._make_evaluator
    base._evaluate = _evaluate
    base._valid = _valid
    base._tune = _tune
    return {
        "lgssm": base._run_lgssm(),
        "fresh_exact_sv": base._run_sv(),
        "predator_prey": base._run_predator_prey(),
    }


def _add_identities(suite: dict[str, Any]) -> None:
    for model_id, section in (
        ("diagonal_lgssm", suite["lgssm"]),
        ("exact_transformed_sv", suite["fresh_exact_sv"]),
        ("predator_prey_additive_gaussian", suite["predator_prey"]),
    ):
        scopes = section.items() if model_id == "diagonal_lgssm" else (("fixed", section),)
        for _, scope in scopes:
            scope["route_identity"] = established._identity(
                model_id=model_id,
                target_id=f"{model_id}_moment_retuned_candidate_v1",
                horizon=int(scope["scope"]["horizon"]),
                particle_count=int(scope["scope"]["particle_count"]),
                state_dimension=(
                    3 if model_id == "diagonal_lgssm"
                    else 1 if model_id == "exact_transformed_sv" else 2
                ),
                parameter_count=len(scope["labels"]) - 1,
                design_family="genut",
                controls=scope["tuning"]["selected_controls"],
                prepared_data_id=str(scope["observation_sha256"]),
            )


def run(output_root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    output_root.mkdir(parents=True, exist_ok=False)
    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    logical = tf.config.list_logical_devices("GPU")
    if not logical:
        raise RuntimeError("retuning trial requires a logical GPU")
    suite = _run_suite()
    _add_identities(suite)
    sir = established._run_sir()
    payload = {
        "schema_version": OUTPUT_SCHEMA,
        "plan": PLAN.as_posix(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "device": {
            "logical_devices": [device.name for device in logical],
            "dtype": "float32",
            "tf32_enabled": True,
            "jit_compile": True,
            "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        },
        "memory_policy": dict(memory_policy),
        "control_grid": CONTROL_GRID,
        "selection_objective": "mean normalized diagonal skewness+kurtosis residual",
        "secondary_objective": "scaled conditional value and recursive score variance",
        "runtime_score": "recursive forward sensitivity; no autodiff or runtime finite differences",
        "displacement_veto": DISPLACEMENT_VETO,
        "particle_policy": "N>1000",
        **suite,
        "sir": sir,
        "hard_valid": True,
        "nonclaims": [
            "moment residual is not an exact distributional or score oracle",
            "dense/Kalman references are post-run diagnostics only",
            "predator-prey score is descriptive without an exact oracle",
            "Austria SIR row is the existing fixed SGQF value/score target and is not a GenUT trial",
            "no default, HMC, leaderboard, superiority, or NAWM promotion",
        ],
        "gpu_allocator": {
            key: int(value)
            for key, value in tf.config.experimental.get_memory_info("GPU:0").items()
        },
        "wall_time_seconds": time.perf_counter() - started,
        "run_manifest": {
            "command": [sys.executable, *sys.argv],
            "environment": sys.prefix,
            "host": platform.node(),
            "python": sys.version.split()[0],
            "tensorflow": tf.__version__,
            "plan": PLAN.as_posix(),
            "source_sha256": {
                PLAN.as_posix(): _sha256(ROOT / PLAN),
                Path(__file__).relative_to(ROOT).as_posix(): _sha256(Path(__file__)),
                "bayesfilter/highdim/cubature_genut_filter.py": _sha256(
                    ROOT / "bayesfilter/highdim/cubature_genut_filter.py"
                ),
                "bayesfilter/highdim/higher_moment_contract_e.py": _sha256(
                    ROOT / "bayesfilter/highdim/higher_moment_contract_e.py"
                ),
            },
        },
    }
    (output_root / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    payload = run(args.output_root.resolve())
    print(json.dumps({
        "output": str(args.output_root.resolve()),
        "hard_valid": payload["hard_valid"],
        "wall_time_seconds": payload["wall_time_seconds"],
    }))


if __name__ == "__main__":
    main()
