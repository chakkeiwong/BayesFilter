#!/usr/bin/env python3
"""Historical non-DGP engineering diagnostic for the candidate SV adapter.

The observations are arbitrary direct-Normal transformed values, not draws
from the declared SV DGP.  This script is ineligible for SV tuning, accuracy,
bias, score, ranking, or promotion evidence and fails closed unless the caller
explicitly requests historical engineering-only execution.
"""

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

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf


N = 12
T_VALUES = (2, 10, 50)
CALIBRATION_SEEDS = (3301, 3302)
VALIDATION_SEEDS = (3311, 3312)
CLAIM_SEEDS = tuple(range(3320, 3336))
THETA_VALUES = (0.25, -0.15)
CONTROLS = tuple(
    {"epsilon": epsilon, "sinkhorn_steps": steps, "ridge": ridge}
    for epsilon in (1.0, 2.0)
    for steps in (4, 8)
    for ridge in (1.0e-5, 1.0e-4)
)
VALUE_BUDGET = 0.25
SCORE_BUDGET = 1.0
T_CRITICAL_95_16 = 2.131449545559323


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _observations(horizon: int) -> tf.Tensor:
    with tf.device("/CPU:0"):
        return tf.random.stateless_normal(
            [horizon, 1], [7000 + horizon, 17], dtype=tf.float32
        )


def _inputs(particle_count: int, horizon: int, seed: int):
    with tf.device("/CPU:0"):
        initial = tf.random.stateless_normal(
            [particle_count, 1], [seed, 101], dtype=tf.float32
        )
        process = tf.random.stateless_normal(
            [horizon, particle_count, 1], [seed, 102], dtype=tf.float32
        )
        design = tf.cast(
            tf.repeat(
                tf.concat([tf.ones([1, 1]), -tf.ones([1, 1])], axis=0),
                repeats=particle_count // 2,
                axis=0,
            ),
            tf.float32,
        )
    return initial, process, design


def _make_candidate(adapter, controls):
    from bayesfilter.highdim.cubature_genut_filter import finite_value_score

    @tf.function(jit_compile=True, reduce_retracing=True)
    def evaluate(theta, observations_, initial_, process_, design_):
        with tf.device("/GPU:0"):
            return finite_value_score(
                adapter,
                theta,
                observations_,
                initial_,
                process_,
                design_,
                epsilon=float(controls["epsilon"]),
                sinkhorn_steps=int(controls["sinkhorn_steps"]),
                ridge=float(controls["ridge"]),
            )

    return evaluate


def _dense_value(observations: tf.Tensor, theta: tf.Tensor, order: int = 257) -> tf.Tensor:
    """Independent same-target dense reference; diagnostic code only."""
    from bayesfilter.highdim.sv_mixture_cut4 import (
        ExactTransformedSVSSM,
        exact_transformed_sv_scalar_dense_reference,
    )

    with tf.device("/CPU:0"):
        result = exact_transformed_sv_scalar_dense_reference(
            ExactTransformedSVSSM(sigma=1.0),
            tf.cast(theta, tf.float64),
            tf.exp(0.5 * tf.cast(observations, tf.float64)),
            order=order,
            radius=8.0,
        )
    return result.log_likelihood


def _row(
    evaluate,
    theta: tf.Tensor,
    particle_count: int,
    horizon: int,
    seed: int,
    controls: dict[str, object],
    dense_value: tf.Tensor,
    *,
    include_fd: bool,
) -> dict[str, object]:
    observations = _observations(horizon)
    initial, process, design = _inputs(particle_count, horizon, seed)
    try:
        tf.config.experimental.reset_memory_stats("GPU:0")
    except (AttributeError, RuntimeError, tf.errors.InvalidArgumentError):
        pass
    row_started = time.perf_counter()
    value, score, diagnostics = evaluate(
        theta, observations, initial, process, design
    )
    fd = []
    if include_fd:
        for index in range(2):
            step = tf.constant(2.0e-3, tf.float32)
            plus = tf.tensor_scatter_nd_add(theta, [[index]], [step])
            minus = tf.tensor_scatter_nd_sub(theta, [[index]], [step])
            plus_value = evaluate(plus, observations, initial, process, design)[0]
            minus_value = evaluate(minus, observations, initial, process, design)[0]
            fd.append(float(((plus_value - minus_value) / (2.0 * step)).numpy()))
    memory = tf.config.experimental.get_memory_info("GPU:0")
    device = str(value.device)
    return {
        "horizon": horizon,
        "seed": seed,
        "controls": controls,
        "value": float(value.numpy()),
        "score": [float(item) for item in score.numpy().tolist()],
        "fd_probe_score": fd,
        "dense_value": float(dense_value.numpy()),
        "value_error_to_dense": float((value - tf.cast(dense_value, tf.float32)).numpy()),
        "fd_max_abs_error": (
            float(
                max(
                    abs(float(score[index].numpy()) - fd[index])
                    for index in range(2)
                )
            )
            if include_fd
            else 0.0
        ),
        "max_mean_residual": float(diagnostics["max_mean_residual"].numpy()),
        "max_row_residual": float(diagnostics["max_row_residual"].numpy()),
        "max_col_residual": float(diagnostics["max_col_residual"].numpy()),
        "finite": bool(tf.math.is_finite(value).numpy()) and bool(tf.reduce_all(tf.math.is_finite(score)).numpy()),
        "elapsed_seconds": time.perf_counter() - row_started,
        "device": device,
        "gpu_placement": "GPU" in device.upper(),
        "gpu_allocator": {key: int(item) for key, item in memory.items()},
    }


def _interval(values: list[float]) -> dict[str, float]:
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    mcse = sd / math.sqrt(len(values))
    critical = T_CRITICAL_95_16 if len(values) == 16 else 1.96
    half_width = critical * mcse
    return {
        "mean": mean,
        "sd": sd,
        "mcse": mcse,
        "critical_value": critical,
        "half_width": half_width,
        "lower": mean - half_width,
        "upper": mean + half_width,
    }


def run(
    output_root: Path,
    *,
    particle_count: int = N,
    horizons: tuple[int, ...] = T_VALUES,
    calibration_seeds: tuple[int, ...] = CALIBRATION_SEEDS,
    validation_seeds: tuple[int, ...] = VALIDATION_SEEDS,
    claim_seeds: tuple[int, ...] = CLAIM_SEEDS,
    historical_nondgp_engineering_only: bool = False,
) -> dict[str, object]:
    if not historical_nondgp_engineering_only:
        raise RuntimeError(
            "revoked non-DGP fixture: use a declared SV-DGP dataset; "
            "historical mechanics require --historical-nondgp-engineering-only"
        )
    started = time.perf_counter()
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if particle_count < 2 or particle_count % 2 != 0:
        raise ValueError("scalar Cubature particle count must be divisible by 2")
    if not horizons or any(horizon < 1 for horizon in horizons):
        raise ValueError("at least one positive horizon is required")
    if set(calibration_seeds) & set(validation_seeds):
        raise ValueError("calibration and validation seeds must be disjoint")
    if (set(calibration_seeds) | set(validation_seeds)) & set(claim_seeds):
        raise ValueError("claim seeds must be untouched by tuning")
    physical = tf.config.list_physical_devices("GPU")
    if not physical:
        raise RuntimeError("exact-SV claim diagnostic requires a visible GPU")
    tf.config.set_soft_device_placement(False)
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    from bayesfilter.highdim.cubature_genut_adapters import exact_transformed_sv_candidate_adapter
    adapter = exact_transformed_sv_candidate_adapter()
    with tf.device("/CPU:0"):
        theta = tf.constant(THETA_VALUES, tf.float32)
    dense_values = {
        horizon: _dense_value(_observations(horizon), theta) for horizon in horizons
    }
    tuning_horizon = max(horizons)

    candidates = []
    for controls in CONTROLS:
        evaluate = _make_candidate(adapter, controls)
        calibration = [
            _row(
                evaluate,
                theta,
                particle_count,
                tuning_horizon,
                seed,
                controls,
                dense_values[tuning_horizon],
                include_fd=True,
            )
            for seed in calibration_seeds
        ]
        validation = [
            _row(
                evaluate,
                theta,
                particle_count,
                tuning_horizon,
                seed,
                controls,
                dense_values[tuning_horizon],
                include_fd=False,
            )
            for seed in validation_seeds
        ]
        validation_errors = [row["value_error_to_dense"] for row in validation]
        validation_mean_error = statistics.mean(validation_errors)
        validation_rmse = math.sqrt(
            statistics.mean(value * value for value in validation_errors)
        )
        maximum_fd_error = max(row["fd_max_abs_error"] for row in calibration)
        maximum_residual = max(
            row["max_row_residual"] + row["max_col_residual"]
            for row in validation
        )
        selection_key = (
            abs(validation_mean_error),
            validation_rmse,
            maximum_fd_error,
            maximum_residual,
            int(controls["sinkhorn_steps"]),
            float(controls["ridge"]),
        )
        candidates.append(
            {
                "controls": controls,
                "calibration": calibration,
                "validation": validation,
                "validation_mean_error": validation_mean_error,
                "validation_rmse": validation_rmse,
                "maximum_fd_error": maximum_fd_error,
                "maximum_residual": maximum_residual,
                "selection_key": list(selection_key),
            }
        )
    selected = min(candidates, key=lambda item: tuple(item["selection_key"]))
    selected_evaluate = _make_candidate(adapter, selected["controls"])
    tuning = {
        "schema_version": "bayesfilter.cubature_genut_exact_sv_tuning.v1",
        "scientific_eligibility": "ineligible_nondgp_engineering_only",
        "eligible_for_sv_tuning_or_claim": False,
        "scope": {
            "model_id": "exact_transformed_sv_candidate_v1",
            "horizon": list(horizons),
            "particle_count": particle_count,
            "state_dimension": 1,
            "dtype": "float32",
            "tf32_enabled": True,
            "jit_compile": True,
        },
        "partitions": {
            "calibration": list(calibration_seeds),
            "validation": list(validation_seeds),
            "claim": list(claim_seeds),
        },
        "selected_controls": selected["controls"],
        "candidate_count": len(candidates),
        "selection_objective": "lexicographic(abs validation mean value error, validation RMSE, calibration recursive-vs-FD error, validation reset residual, work)",
        "selected_candidate_summary": {
            key: selected[key]
            for key in (
                "validation_mean_error",
                "validation_rmse",
                "maximum_fd_error",
                "maximum_residual",
                "selection_key",
            )
        },
        "candidate_summaries": [
            {
                key: item[key]
                for key in (
                    "controls",
                    "validation_mean_error",
                    "validation_rmse",
                    "maximum_fd_error",
                    "maximum_residual",
                    "selection_key",
                )
            }
            for item in candidates
        ],
        "fd_role": "representative calibration validation only; never runtime score",
        "frozen_before_claim": True,
    }
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "tuning.json").write_text(
        json.dumps(tuning, indent=2, sort_keys=True) + "\n"
    )
    claim_rows = [
        _row(
            selected_evaluate,
            theta,
            particle_count,
            horizon,
            seed,
            selected["controls"],
            dense_values[horizon],
            include_fd=False,
        )
        for horizon in horizons
        for seed in claim_seeds
    ]
    claim_summary = []
    for horizon in horizons:
        rows = [row for row in claim_rows if row["horizon"] == horizon]
        value_interval = _interval(
            [row["value_error_to_dense"] for row in rows]
        )
        score_intervals = [
            _interval([row["score"][index] for row in rows]) for index in range(2)
        ]
        claim_summary.append(
            {
                "horizon": horizon,
                "value_error_interval": value_interval,
                "score_intervals": score_intervals,
                "scientific_eligibility": "ineligible_nondgp_engineering_only",
                "value_accuracy_pass": False,
                "score_accuracy_pass": False,
                "hard_valid": False,
                "engineering_valid": all(
                    row["finite"]
                    and row["gpu_placement"]
                    and row["max_row_residual"] < 1.0e-2
                    and row["max_col_residual"] < 1.0e-2
                    for row in rows
                ),
                "max_abs_value_error": max(
                    abs(row["value_error_to_dense"]) for row in rows
                ),
            }
        )

    from bayesfilter.highdim.cubature_genut_candidate import (
        CandidateRouteScope,
        issue_repository_candidate_route_identity,
        validate_repository_candidate_route_identity,
    )

    prepared_hash = hashlib.sha256(
        b"".join(
            tf.io.serialize_tensor(_observations(horizon)).numpy()
            for horizon in horizons
        )
    ).hexdigest()
    route_identity = issue_repository_candidate_route_identity(
        CandidateRouteScope(
            model_id="exact_transformed_sv",
            target_id="exact_log_chi_square_transformed_sv_v1",
            horizon=max(horizons),
            particle_count=particle_count,
            state_dimension=1,
            parameter_count=2,
            dtype="float32",
            tf32_enabled=True,
            jit_compile=True,
            design_family="cubature",
            control_family_id="epsilon_sinkhorn_steps_ridge_v1",
        ),
        prepared_data_id=prepared_hash,
        residual_design_id=f"replicated_cubature_d1_n{particle_count}",
        controls={key: str(value) for key, value in selected["controls"].items()},
        adapter_id="exact_transformed_sv_v1",
    )
    validate_repository_candidate_route_identity(route_identity)
    payload = {
        "schema_version": "bayesfilter.cubature_genut_exact_sv_claim_diagnostic.v1",
        "campaign_id": "cubature-genut-exact-sv-model-claim-20260721",
        "host": platform.node(),
        "physical_devices": [item.name for item in physical],
        "memory_policy": dict(memory_policy),
        "dtype": "float32",
        "tf32_enabled": True,
        "jit_compile": True,
        "theta": list(THETA_VALUES),
        "tuning": tuning,
        "claim_rows": claim_rows,
        "claim_summary": claim_summary,
        "route_identity": route_identity.to_dict(),
        "dense_reference": {
            "status": "diagnostic_tuning_comparator_and_claim_accuracy_reference",
            "target": "exact_log_chi_square_transformed_sv",
            "dtype": "float64",
            "quadrature_order": 257,
            "radius": 8.0,
            "not_a_candidate_runtime_dependency": True,
        },
        "comparator": {"status": "BLOCKED_SAME_TARGET_COMPARATOR", "reason": "Contract E scalar-SV proposal/reset scope is not identical to this candidate finite program"},
        "engineering_valid": all(item["engineering_valid"] for item in claim_summary),
        "scientific_eligibility": "ineligible_nondgp_engineering_only",
        "target_accuracy_valid": False,
        "score_accuracy_valid": False,
        "hard_valid": False,
        "run_manifest": {
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "command": list(sys.argv),
            "python": sys.version,
            "tensorflow": tf.__version__,
            "environment": "tf-gpu",
            "cpu_gpu_status": "GPU/XLA candidate; CPU float64 dense diagnostic reference",
            "data_version": "stateless_synthetic_observations_v1",
            "particle_count": particle_count,
            "horizons": list(horizons),
            "calibration_seeds": list(calibration_seeds),
            "validation_seeds": list(validation_seeds),
            "claim_seeds": list(claim_seeds),
            "plan_file": "docs/plans/bayesfilter-cubature-genut-exact-sv-n1000-plan-2026-07-21.md",
            "tuning_file": str(output_root / "tuning.json"),
            "result_file": str(output_root / "result.json"),
            "started_utc": started_utc,
            "wall_seconds": time.perf_counter() - started,
        },
        "nonclaims": [
            "historical non-DGP engineering mechanics only",
            "wrong and irrelevant for SV accuracy, bias, tuning, score, or ranking",
            "no method ranking or leaderboard/default admission",
        ],
    }
    (output_root / "result.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output_root), "hard_valid": payload["hard_valid"], "selected_controls": selected["controls"]}, indent=2))
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--particle-count", type=int, default=N)
    parser.add_argument("--horizons", nargs="+", type=int, default=list(T_VALUES))
    parser.add_argument("--calibration-seeds", nargs="+", type=int, default=list(CALIBRATION_SEEDS))
    parser.add_argument("--validation-seeds", nargs="+", type=int, default=list(VALIDATION_SEEDS))
    parser.add_argument("--claim-seeds", nargs="+", type=int, default=list(CLAIM_SEEDS))
    parser.add_argument("--historical-nondgp-engineering-only", action="store_true")
    args = parser.parse_args()
    run(
        args.output_root.resolve(),
        particle_count=args.particle_count,
        horizons=tuple(args.horizons),
        calibration_seeds=tuple(args.calibration_seeds),
        validation_seeds=tuple(args.validation_seeds),
        claim_seeds=tuple(args.claim_seeds),
        historical_nondgp_engineering_only=args.historical_nondgp_engineering_only,
    )
