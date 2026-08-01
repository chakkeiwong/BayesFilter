#!/usr/bin/env python3
"""Candidate higher-moment Contract E regressions on the established suite.

This runner imports the historical GenUT harness only for its model fixtures,
oracle construction, and comparison conventions.  It does not modify or
overwrite historical artifacts.  The candidate controls are included in the
repository-issued route identity and are selected on disjoint calibration and
validation observations.
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

BASE = None
PLAN = Path(
    "docs/plans/bayesfilter-higher-moment-contract-e-genut-campaign-plan-2026-07-23.md"
)
RESIDUAL_TOLERANCE = 5.0e-4


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary(values: list[float]) -> dict[str, float | int]:
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    half = 2.131449545559323 * sd / (len(values) ** 0.5)
    return {
        "count": len(values),
        "mean": mean,
        "sample_sd": sd,
        "standard_error": sd / (len(values) ** 0.5),
        "ci95_lower": mean - half,
        "ci95_upper": mean + half,
    }


def _make_evaluator(adapter: Any, *, particle_count: int, state_dimension: int,
                    parameter_count: int, horizon: int,
                    controls: dict[str, float | int],
                    transition_before_first_observation: bool = True):
    from bayesfilter.highdim.cubature_genut_filter import finite_value_score

    @tf.function(jit_compile=True, reduce_retracing=True)
    def evaluate(theta, observations, initial, process, design):
        theta = tf.ensure_shape(theta, [parameter_count])
        observations = tf.ensure_shape(observations, [horizon, state_dimension])
        initial = tf.ensure_shape(initial, [particle_count, state_dimension])
        process = tf.ensure_shape(process, [horizon, particle_count, state_dimension])
        design = tf.ensure_shape(design, [particle_count, state_dimension])
        with tf.device("/GPU:0"):
            return finite_value_score(
                adapter, theta, observations, initial, process, design,
                epsilon=float(controls["epsilon"]),
                sinkhorn_steps=int(controls["sinkhorn_steps"]),
                balance_steps=int(controls["balance_steps"]),
                ridge=float(controls["ridge"]),
                transition_before_first_observation=transition_before_first_observation,
                higher_moment_correction_steps=int(controls["higher_moment_correction_steps"]),
                higher_moment_strength=float(controls["higher_moment_strength"]),
                higher_moment_floor=float(controls["higher_moment_floor"]),
            )
    return evaluate


def _evaluate(evaluate, *arguments) -> dict[str, Any]:
    value, score, diagnostics = evaluate(*arguments)
    score_sum_error = tf.reduce_max(
        tf.abs(tf.reduce_sum(diagnostics["score_increments"], axis=0) - score)
    )
    valid = (
        bool(diagnostics["program_valid"].numpy())
        and bool(tf.math.is_finite(value).numpy())
        and bool(tf.reduce_all(tf.math.is_finite(score)).numpy())
    )
    return {
        "value": float(value.numpy()) if valid else None,
        "score": [float(v) for v in score.numpy()] if valid else None,
        "finite": valid,
        "program_valid": bool(diagnostics["program_valid"].numpy()),
        "max_mean_residual": float(diagnostics["max_mean_residual"].numpy()),
        "max_row_residual": float(diagnostics["max_row_residual"].numpy()),
        "max_col_residual": float(diagnostics["max_col_residual"].numpy()),
        "max_transition_residual": float(diagnostics["max_transition_residual"].numpy()),
        "score_increment_sum_residual": float(score_sum_error.numpy()),
        "minimum_row_mass": float(diagnostics["minimum_row_mass"].numpy()),
        "maximum_post_quotient_column_tv_error": float(
            diagnostics["maximum_post_quotient_column_tv_error"].numpy()
        ),
        "minimum_covariance_gap_eigenvalue": float(
            diagnostics["minimum_covariance_gap_eigenvalue"].numpy()
        ),
        "maximum_skew_residual": float(diagnostics["maximum_skew_residual"].numpy()),
        "maximum_kurtosis_residual": float(
            diagnostics["maximum_kurtosis_residual"].numpy()
        ),
        "device": str(value.device),
    }


def _valid(row: dict[str, Any]) -> bool:
    return bool(row["finite"]) and "GPU" in row["device"].upper() and max(
        row["max_mean_residual"], row["max_row_residual"], row["max_col_residual"],
        row["max_transition_residual"], row["score_increment_sum_residual"],
    ) < RESIDUAL_TOLERANCE


def _tune(*, name: str, evaluator_factory, datasets, particle_seeds,
          arguments, scales, controls_grid):
    candidates = []
    for controls in controls_grid:
        evaluate = evaluator_factory(controls)
        objectives = {}
        eligible = True
        for partition, observations_set in datasets.items():
            objective_rows = []
            for observations in observations_set:
                rows = []
                for seed in particle_seeds:
                    row = _evaluate(evaluate, *arguments(observations, seed))
                    rows.append(row)
                    eligible = eligible and _valid(row)
                if not all(row["finite"] for row in rows):
                    eligible = False
                    continue
                vectors = [[row["value"], *row["score"]] for row in rows]
                objective_rows.append(max(
                    statistics.variance(vector[i] for vector in vectors)
                    / (scales[i] * scales[i])
                    for i in range(len(scales))
                ))
            objectives[partition] = (
                statistics.mean(objective_rows)
                if len(objective_rows) == len(observations_set) else None
            )
        candidates.append({
            "controls": controls,
            "objectives": objectives,
            "eligible": eligible,
        })
    eligible = [row for row in candidates if row["eligible"]]
    if not eligible:
        raise RuntimeError(f"no eligible controls for {name}")
    selected = min(
        eligible,
        key=lambda row: (
            row["objectives"]["validation"],
            row["objectives"]["calibration"],
            int(row["controls"]["higher_moment_correction_steps"]),
            float(row["controls"]["higher_moment_strength"]),
        ),
    )
    return {
        "scope": name,
        "selection_objective": "scaled conditional value and recursive score variance",
        "candidate_controls_grid": controls_grid,
        "calibration_particle_seeds": particle_seeds,
        "selected_controls": selected["controls"],
        "candidates": candidates,
        "claim_data_read_during_selection": False,
    }


def _identity(model_id: str, target_id: str, horizon: int, particle_count: int,
              state_dimension: int, parameter_count: int, design_family: str,
              controls: dict[str, Any], prepared_data_id: str):
    from bayesfilter.highdim.cubature_genut_candidate import (
        CandidateRouteScope,
        issue_repository_candidate_route_identity,
        validate_repository_candidate_route_identity,
    )

    identity = issue_repository_candidate_route_identity(
        CandidateRouteScope(
            model_id=model_id, target_id=target_id, horizon=horizon,
            particle_count=particle_count, state_dimension=state_dimension,
            parameter_count=parameter_count, dtype="float32",
            tf32_enabled=True, jit_compile=True, design_family=design_family,
            control_family_id="higher_moment_contract_e_candidate_v1",
        ),
        prepared_data_id=prepared_data_id,
        residual_design_id=f"fixed_{design_family}_candidate_n{particle_count}",
        controls={key: str(value) for key, value in controls.items()},
        adapter_id={
            "exact_transformed_sv": "exact_transformed_sv_v1",
            "predator_prey_additive_gaussian": "predator_prey_additive_gaussian_v1",
            "diagonal_lgssm": "diagonal_lgssm_v1",
        }[model_id],
    )
    validate_repository_candidate_route_identity(identity)
    return identity.to_dict()


def _run_suite(controls_grid):
    global BASE
    import docs.benchmarks.run_genut_transport_repair_regressions as base
    BASE = base
    base._make_evaluator = lambda adapter, **kwargs: _make_evaluator(
        adapter, **kwargs
    )
    base._evaluate = _evaluate
    base._valid = _valid
    base.CONTROL_GRID = controls_grid
    return {
        "lgssm": base._run_lgssm(),
        "fresh_exact_sv": base._run_sv(),
        "predator_prey": base._run_predator_prey(),
    }


def _run_sir() -> dict[str, Any]:
    from bayesfilter.highdim.fixed_sir_sgqf_tf import make_fixed_sir_sgqf_route
    route = make_fixed_sir_sgqf_route()
    @tf.function(jit_compile=True)
    def evaluate(observations):
        with tf.device("/GPU:0"):
            return route.value_only_status()
    value, status = evaluate(route.observations)
    prior_path = ROOT / (
        "docs/benchmarks/artifacts/sgqf_whole_highdim_leaderboard_repair_20260722/"
        "attempt02/fixed-sir/gpu/result.json"
    )
    prior = json.loads(prior_path.read_text())
    return {
        "scope": {"model": "actual_austria_sir", "horizon": 20, "state_dimension": 18},
        "route_identity": route.route_identity,
        "value": float(value.numpy()),
        "status": {key: int(item.numpy()) for key, item in status.items()},
        "device": str(value.device),
        "prior_artifact": prior_path.relative_to(ROOT).as_posix(),
        "prior_value": float(prior["gpu_xla"]["value"]),
        "difference_from_prior": float(value.numpy()) - float(prior["gpu_xla"]["value"]),
        "score": "not applicable: fixed parameter route",
    }


def run(output_root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    output_root.mkdir(parents=True, exist_ok=False)
    from bayesfilter.runtime.gpu_memory_policy import (
        configure_tensorflow_gpu_memory_growth,
    )

    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    if not tf.config.list_logical_devices("GPU"):
        raise RuntimeError("candidate campaign requires a logical GPU")
    controls_grid = tuple(
        {
            "epsilon": epsilon, "sinkhorn_steps": steps,
            "balance_steps": balance, "ridge": ridge,
            "higher_moment_correction_steps": hm_steps,
            "higher_moment_strength": strength,
            "higher_moment_floor": 1.0e-5,
        }
        for epsilon in (2.0, 4.0)
        for steps, balance in ((4, 4), (8, 8))
        for ridge in (1.0e-6, 1.0e-5)
        for hm_steps, strength in ((1, 0.02), (2, 0.05))
    )
    suite = _run_suite(controls_grid)
    for model_id, section in (
        ("diagonal_lgssm", suite["lgssm"]),
        ("exact_transformed_sv", suite["fresh_exact_sv"]),
        ("predator_prey_additive_gaussian", suite["predator_prey"]),
    ):
        for horizon_key, scope in (
            section.items() if model_id == "diagonal_lgssm" else [("fixed", section)]
        ):
            controls = scope["tuning"]["selected_controls"]
            scope["route_identity"] = _identity(
                model_id=model_id,
                target_id=f"{model_id}_higher_moment_candidate_v1",
                horizon=int(scope["scope"]["horizon"]),
                particle_count=int(scope["scope"]["particle_count"]),
                state_dimension=(
                    3 if model_id == "diagonal_lgssm"
                    else 1 if model_id == "exact_transformed_sv" else 2
                ),
                parameter_count=len(scope["labels"]) - 1,
                design_family="genut",
                controls=controls,
                prepared_data_id=str(scope["observation_sha256"]),
            )
    sir = _run_sir()
    payload = {
        "schema_version": "bayesfilter.higher_moment_contract_e_regressions.v1",
        "plan": PLAN.as_posix(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "device": {
            "logical_devices": [item.name for item in tf.config.list_logical_devices("GPU")],
            "dtype": "float32", "tf32_enabled": True, "jit_compile": True,
            "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        },
        "memory_policy": dict(memory_policy),
        "controls_grid": controls_grid,
        "runtime_score": "recursive forward sensitivity; no autodiff or runtime finite differences",
        "particle_policy": "N>1000",
        **suite,
        "sir": sir,
        "hard_valid": True,
        "nonclaims": [
            "candidate is not canonical Contract E or classical sigma-point GenUT",
            "no exact nonlinear likelihood or posterior-score theorem",
            "predator-prey score is descriptive without an exact oracle",
            "Austria SIR has no free parameter score and is value-only",
            "no default, HMC, leaderboard, or NAWM promotion",
        ],
        "gpu_allocator": {
            key: int(value)
            for key, value in tf.config.experimental.get_memory_info("GPU:0").items()
        },
        "wall_time_seconds": time.perf_counter() - started,
        "run_manifest": {
            "command": [sys.executable, *sys.argv],
            "environment": sys.prefix, "host": platform.node(),
            "python": sys.version.split()[0], "tensorflow": tf.__version__,
            "source_sha256": {
                Path(__file__).relative_to(ROOT).as_posix(): _sha256(Path(__file__)),
                "bayesfilter/highdim/higher_moment_contract_e.py": _sha256(
                    ROOT / "bayesfilter/highdim/higher_moment_contract_e.py"
                ),
                "bayesfilter/highdim/cubature_genut_filter.py": _sha256(
                    ROOT / "bayesfilter/highdim/cubature_genut_filter.py"
                ),
                PLAN.as_posix(): _sha256(ROOT / PLAN),
            },
        },
    }
    (output_root / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return payload


def main():
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
