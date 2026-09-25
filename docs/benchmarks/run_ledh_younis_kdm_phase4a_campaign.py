"""Bounded paired LGSSM campaign for the Phase 4A KDM score question.

The runner compares the full-feedback KDM-FINITE score with the actual
Contract-E atom endpoint on the same fixed streams and with an independent
scalar Kalman score oracle.  It compiles one atom kernel and one positive-
bandwidth kernel per fixed (N,T) cell, so the rho grid does not create a new
XLA program for every bandwidth.

This is a diagnostic campaign.  It does not promote KDM-FINITE, change the
canonical route, or claim DSGE/HMC validity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shlex
import subprocess
import sys
import time
from typing import Any

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import tensorflow as tf

from bayesfilter.runtime.gpu_memory_policy import (
    configure_tensorflow_gpu_memory_growth,
)


PLAN_PATH = (
    "docs/plans/"
    "bayesfilter-ledh-younis-kdm-phase4a-campaign-amendment-20260908.md"
)
TRUST_BASIS = "owner_designated_managed_session_visible_gpu_trusted"
RHO_GRID = (0.0, 0.025, 0.05, 0.10, 0.20, 0.40, 0.80, 1.20, 1.60)
TRUE_PHI = 0.75
PROCESS_VARIANCE = 0.10
OBSERVATION_VARIANCE = 0.20
INITIAL_VARIANCE = 1.0
CALIBRATION_SEED_BASE = 100_000
VALIDATION_SEED_BASE = 200_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_output(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), check=True, capture_output=True, text=True
    ).stdout.strip()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _parse_int_list(value: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not values or any(item < 1 for item in values):
        raise ValueError("integer list must contain positive values")
    return values


def _parse_float_list(value: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not values or any(item < 0.0 for item in values):
        raise ValueError("rho list must contain nonnegative values")
    return values


def _seed(seed: int, salt: int) -> tf.Tensor:
    return tf.constant([int(seed), int(salt)], dtype=tf.int32)


def _path_seed(
    split: str,
    particle_count: int,
    horizon: int,
    replicate: int,
) -> int:
    if split == "calibration":
        base = CALIBRATION_SEED_BASE
    elif split == "validation":
        base = VALIDATION_SEED_BASE
    else:
        raise ValueError(f"unknown split: {split}")
    if particle_count < 1 or horizon < 1 or replicate < 0:
        raise ValueError("seed coordinates are outside their declared domain")
    return base + particle_count * 100 + horizon * 10 + replicate


def _validate_seed_schedule(
    cells: list[tuple[int, int]],
    calibration_reps: int,
    validation_reps: int,
) -> None:
    calibration = [
        _path_seed("calibration", particle_count, horizon, replicate)
        for particle_count, horizon in cells
        for replicate in range(calibration_reps)
    ]
    validation = [
        _path_seed("validation", particle_count, horizon, replicate)
        for particle_count, horizon in cells
        for replicate in range(validation_reps)
    ]
    if len(set(calibration)) != len(calibration):
        raise ValueError("calibration seed schedule contains collisions")
    if len(set(validation)) != len(validation):
        raise ValueError("validation seed schedule contains collisions")
    if not set(calibration).isdisjoint(validation):
        raise ValueError("calibration and validation seed schedules overlap")


def _make_model(dtype: tf.dtypes.DType) -> NonlinearScoreModel:
    from bayesfilter.highdim.ledh_canonical_score_tf import NonlinearScoreModel

    process_covariance = tf.constant([[PROCESS_VARIANCE]], dtype=dtype)
    observation_covariance = tf.constant([[OBSERVATION_VARIANCE]], dtype=dtype)

    def transition_mean_fn(theta: tf.Tensor, points: tf.Tensor) -> tf.Tensor:
        return theta[0] * points

    def transition_mean_tangent_fn(
        theta: tf.Tensor, points: tf.Tensor, d_points: tf.Tensor
    ) -> tf.Tensor:
        # The direction is one in the scalar theta coordinate.  This is the
        # total tangent of phi*x(theta), including the explicit phi term.
        return points + theta[0] * d_points

    def observation_fn(points: tf.Tensor) -> tf.Tensor:
        return points

    def observation_jacobian_fn(points: tf.Tensor) -> tf.Tensor:
        return tf.broadcast_to(
            tf.ones([1, 1], dtype=dtype),
            [tf.shape(points)[0], 1, 1],
        )

    def observation_tangent_fn(
        points: tf.Tensor, d_points: tf.Tensor
    ) -> tf.Tensor:
        return d_points

    def observation_jacobian_tangent_fn(
        points: tf.Tensor, d_points: tf.Tensor
    ) -> tf.Tensor:
        return tf.zeros([tf.shape(points)[0], 1, 1], dtype=dtype)

    return NonlinearScoreModel(
        transition_mean_fn=transition_mean_fn,
        transition_mean_tangent_fn=transition_mean_tangent_fn,
        observation_fn=observation_fn,
        observation_jacobian_fn=observation_jacobian_fn,
        observation_tangent_fn=observation_tangent_fn,
        process_covariance=process_covariance,
        observation_covariance=observation_covariance,
        observation_jacobian_tangent_fn=observation_jacobian_tangent_fn,
    )


def _make_path(
    seed: int, particle_count: int, horizon: int, dtype: tf.dtypes.DType
) -> dict[str, tf.Tensor]:
    """Generate one exact LGSSM observation path and paired filter streams."""

    with tf.device("/GPU:0"):
        initial_states = tf.sqrt(tf.constant(INITIAL_VARIANCE, dtype=dtype)) * tf.random.stateless_normal(
            [particle_count, 1], seed=_seed(seed, 101), dtype=dtype
        )
        noises = tf.random.stateless_normal(
            [horizon, particle_count, 1], seed=_seed(seed, 102), dtype=dtype
        )
        process_shocks = tf.random.stateless_normal(
            [horizon], seed=_seed(seed, 103), dtype=dtype
        )
        observation_shocks = tf.random.stateless_normal(
            [horizon], seed=_seed(seed, 104), dtype=dtype
        )
        # The oracle uses x_0 ~ N(0, initial_variance); generate observations
        # from that same initial law rather than silently starting at zero.
        state = tf.sqrt(tf.constant(INITIAL_VARIANCE, dtype=dtype)) * tf.random.stateless_normal(
            [], seed=_seed(seed, 105), dtype=dtype
        )
        observations = []
        process_scale = tf.sqrt(tf.constant(PROCESS_VARIANCE, dtype=dtype))
        observation_scale = tf.sqrt(tf.constant(OBSERVATION_VARIANCE, dtype=dtype))
        true_phi = tf.constant(TRUE_PHI, dtype=dtype)
        for process_shock, observation_shock in zip(
            tf.unstack(process_shocks), tf.unstack(observation_shocks)
        ):
            state = true_phi * state + process_scale * process_shock
            observations.append(state + observation_scale * observation_shock)
        observations_tensor = tf.stack(observations)
        initial_covariances = tf.fill(
            [particle_count, 1, 1], tf.constant(INITIAL_VARIANCE, dtype=dtype)
        )
        observation_matrix = tf.constant([[1.0]], dtype=dtype)
        d_observation_matrix = tf.zeros([1, 1], dtype=dtype)
    return {
        "initial_states": initial_states,
        "initial_covariances": initial_covariances,
        "noises": noises,
        "observations": tf.ensure_shape(observations_tensor, [horizon]),
        "observation_matrix": observation_matrix,
        "d_observation_matrix": d_observation_matrix,
    }


def _reset_design(particle_count: int, dtype: tf.dtypes.DType) -> tf.Tensor:
    if particle_count % 2 != 0:
        raise ValueError("particle_count must be even for the declared design")
    return tf.tile(tf.constant([[1.0], [-1.0]], dtype=dtype), [particle_count // 2, 1])


def _options(particle_count: int, dtype: tf.dtypes.DType) -> dict[str, Any]:
    return {
        "flow_substeps": 6,
        "reset_policy": "contract_e",
        "reset_design": _reset_design(particle_count, dtype),
        "reset_sinkhorn_steps": 4,
        "reset_balance_steps": 2,
        "correction_steps": 1,
        "pairwise_steps": 1,
        "coordinate_cap": 0.95,
    }


def _make_kernels(
    particle_count: int,
    horizon: int,
    dtype: tf.dtypes.DType,
    jit_compile: bool,
) -> tuple[Any, Any]:
    from bayesfilter.highdim.ledh_younis_kdm_integrated_tf import (
        make_integrated_linear_gaussian_kdm_kernel,
    )

    model = _make_model(dtype)
    options = _options(particle_count, dtype)
    atom_kernel = make_integrated_linear_gaussian_kdm_kernel(
        model=model,
        theta_dimension=1,
        particle_count=particle_count,
        state_dimension=1,
        observation_dimension=1,
        horizon=horizon,
        canonical_options=options,
        bandwidth_is_zero=True,
        dtype=dtype,
        jit_compile=jit_compile,
    )
    positive_kernel = make_integrated_linear_gaussian_kdm_kernel(
        model=model,
        theta_dimension=1,
        particle_count=particle_count,
        state_dimension=1,
        observation_dimension=1,
        horizon=horizon,
        canonical_options=options,
        bandwidth_is_zero=False,
        dtype=dtype,
        jit_compile=jit_compile,
    )
    return atom_kernel, positive_kernel


def _kernel_inputs(
    path: dict[str, tf.Tensor],
    particle_count: int,
    horizon: int,
    rho: float,
    dtype: tf.dtypes.DType,
) -> tuple[tf.Tensor, ...]:
    theta = tf.constant([TRUE_PHI], dtype=dtype)
    if rho == 0.0:
        bandwidths = tf.zeros([horizon, particle_count, 1, 1], dtype=dtype)
    else:
        bandwidths = tf.fill(
            [horizon, particle_count, 1, 1],
            tf.constant(PROCESS_VARIANCE * rho * rho, dtype=dtype),
        )
    return (
        theta,
        path["initial_states"],
        path["initial_covariances"],
        path["noises"],
        path["observations"][..., tf.newaxis],
        path["observation_matrix"],
        path["d_observation_matrix"],
        bandwidths,
        tf.zeros_like(bandwidths),
    )


def _evaluate(
    atom_kernel: Any,
    positive_kernel: Any,
    path: dict[str, tf.Tensor],
    particle_count: int,
    horizon: int,
    rho: float,
    dtype: tf.dtypes.DType,
) -> dict[str, Any]:
    inputs = _kernel_inputs(path, particle_count, horizon, rho, dtype)
    started = time.perf_counter()
    output = (atom_kernel if rho == 0.0 else positive_kernel)(*inputs)
    value = float(output["value"].numpy())
    score = float(tf.reshape(output["score"], []).numpy())
    return {
        "rho": rho,
        "value": value,
        "score": score,
        "valid": bool(output["valid"].numpy()),
        "elapsed_seconds": time.perf_counter() - started,
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _sample_variance(values: list[float]) -> float:
    if len(values) < 2:
        return float("nan")
    mean = _mean(values)
    return sum((value - mean) ** 2 for value in values) / (len(values) - 1)


def _bootstrap_ci(
    values: list[float], *, seed: int, replicates: int
) -> tuple[float, float]:
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(values)
    bootstrap_means = []
    for _ in range(replicates):
        bootstrap_means.append(
            sum(values[rng.randrange(n)] for _ in range(n)) / n
        )
    bootstrap_means.sort()
    lower_index = max(0, int(0.025 * replicates) - 1)
    upper_index = min(replicates - 1, int(0.975 * replicates))
    return bootstrap_means[lower_index], bootstrap_means[upper_index]


def _paired_error_summary(
    candidate_squared_errors: list[float],
    atom_squared_errors: list[float],
    *,
    bootstrap_seed: int,
    bootstrap_reps: int,
) -> dict[str, Any]:
    if len(candidate_squared_errors) != len(atom_squared_errors):
        raise ValueError("candidate and atom error vectors must have equal length")
    if len(candidate_squared_errors) < 2:
        raise ValueError("paired summary requires at least two observations")
    differences = [
        candidate - atom
        for candidate, atom in zip(candidate_squared_errors, atom_squared_errors)
    ]
    atom_mse = _mean(atom_squared_errors)
    candidate_mse = _mean(candidate_squared_errors)
    ci_lower, ci_upper = _bootstrap_ci(
        differences,
        seed=bootstrap_seed,
        replicates=bootstrap_reps,
    )
    return {
        "mse": candidate_mse,
        "relative_gain": (atom_mse - candidate_mse) / max(atom_mse, 1.0e-30),
        "paired_difference_mean": _mean(differences),
        "paired_difference_mcse": math.sqrt(
            _sample_variance(differences) / len(differences)
        ),
        "paired_difference_bootstrap_ci95": [ci_lower, ci_upper],
    }


def _cell_result(
    *,
    particle_count: int,
    horizon: int,
    rhos: tuple[float, ...],
    calibration_reps: int,
    validation_reps: int,
    bootstrap_reps: int,
    dtype: tf.dtypes.DType,
    jit_compile: bool,
    retain_all_validation_rhos: bool,
    rows_path: Path,
) -> dict[str, Any]:
    from bayesfilter.highdim.ledh_younis_kdm_lgssm_reference_tf import (
        scalar_lgssm_value_and_score,
    )

    atom_kernel, positive_kernel = _make_kernels(
        particle_count, horizon, dtype, jit_compile
    )
    calibration_rows = []
    calibration_mse = {rho: [] for rho in rhos}
    calibration_valid = {rho: True for rho in rhos}
    atom_first_call_seconds = None
    positive_first_call_seconds = None

    for replicate in range(calibration_reps):
        seed = _path_seed("calibration", particle_count, horizon, replicate)
        path = _make_path(seed, particle_count, horizon, dtype)
        oracle_value, oracle_score = scalar_lgssm_value_and_score(
            tf.constant([TRUE_PHI], dtype), path["observations"]
        )
        oracle_score_float = float(oracle_score.numpy())
        row = {
            "split": "calibration",
            "replicate": replicate,
            "seed": seed,
            "oracle_value": float(oracle_value.numpy()),
            "oracle_score": oracle_score_float,
            "candidates": {},
        }
        for rho in rhos:
            candidate = _evaluate(
                atom_kernel,
                positive_kernel,
                path,
                particle_count,
                horizon,
                rho,
                dtype,
            )
            row["candidates"][str(rho)] = candidate
            if replicate == 0 and rho == 0.0:
                atom_first_call_seconds = candidate["elapsed_seconds"]
            if replicate == 0 and rho != 0.0 and positive_first_call_seconds is None:
                positive_first_call_seconds = candidate["elapsed_seconds"]
            calibration_valid[rho] = calibration_valid[rho] and candidate["valid"]
            if candidate["valid"] and math.isfinite(candidate["score"]):
                calibration_mse[rho].append(
                    (candidate["score"] - oracle_score_float) ** 2
                )
        calibration_rows.append(row)
        with rows_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    valid_rhos = [
        rho
        for rho in rhos
        if calibration_valid[rho]
        and len(calibration_mse[rho]) == calibration_reps
    ]
    if not valid_rhos:
        raise RuntimeError("no valid rho survived calibration")
    selected_rho = min(valid_rhos, key=lambda rho: _mean(calibration_mse[rho]))

    validation_differences = []
    validation_rows = []
    validation_rho_squared_errors = {rho: [] for rho in rhos}
    for replicate in range(validation_reps):
        seed = _path_seed("validation", particle_count, horizon, replicate)
        path = _make_path(seed, particle_count, horizon, dtype)
        oracle_value, oracle_score = scalar_lgssm_value_and_score(
            tf.constant([TRUE_PHI], dtype), path["observations"]
        )
        oracle_score_float = float(oracle_score.numpy())
        atom = _evaluate(
            atom_kernel, positive_kernel, path, particle_count, horizon, 0.0, dtype
        )
        validation_candidates = {"0.0": atom}
        if retain_all_validation_rhos:
            for rho in rhos:
                if rho == 0.0:
                    continue
                validation_candidates[str(rho)] = _evaluate(
                    atom_kernel,
                    positive_kernel,
                    path,
                    particle_count,
                    horizon,
                    rho,
                    dtype,
                )
        else:
            validation_candidates[str(selected_rho)] = _evaluate(
                atom_kernel,
                positive_kernel,
                path,
                particle_count,
                horizon,
                selected_rho,
                dtype,
            )
        selected = validation_candidates[str(selected_rho)]
        if not atom["valid"] or not selected["valid"]:
            raise RuntimeError(
                f"validation validity veto at N={particle_count}, T={horizon}, "
                f"replicate={replicate}"
            )
        if retain_all_validation_rhos:
            for rho in rhos:
                candidate = validation_candidates[str(rho)]
                if not candidate["valid"] or not math.isfinite(candidate["score"]):
                    raise RuntimeError(
                        f"validation rho validity veto at N={particle_count}, "
                        f"T={horizon}, replicate={replicate}, rho={rho}"
                    )
                validation_rho_squared_errors[rho].append(
                    (candidate["score"] - oracle_score_float) ** 2
                )
        atom_error = (atom["score"] - oracle_score_float) ** 2
        selected_error = (selected["score"] - oracle_score_float) ** 2
        difference = selected_error - atom_error
        validation_differences.append(difference)
        row = {
            "split": "validation",
            "replicate": replicate,
            "seed": seed,
            "oracle_value": float(oracle_value.numpy()),
            "oracle_score": oracle_score_float,
            "atom": atom,
            "selected": selected,
            "atom_squared_error": atom_error,
            "selected_squared_error": selected_error,
            "paired_squared_error_difference": difference,
        }
        if retain_all_validation_rhos:
            row["validation_candidates"] = validation_candidates
            row["validation_squared_errors"] = {
                str(rho): validation_rho_squared_errors[rho][-1]
                for rho in rhos
            }
        validation_rows.append(row)
        with rows_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    atom_squared_errors = [
        row["atom_squared_error"] for row in validation_rows
    ]
    selected_squared_errors = [
        row["selected_squared_error"] for row in validation_rows
    ]
    selected_summary = _paired_error_summary(
        selected_squared_errors,
        atom_squared_errors,
        bootstrap_seed=particle_count * 1000 + horizon,
        bootstrap_reps=bootstrap_reps,
    )
    atom_mse = _mean(atom_squared_errors)
    selected_mse = selected_summary["mse"]
    relative_gain = selected_summary["relative_gain"]
    ci_lower, ci_upper = selected_summary["paired_difference_bootstrap_ci95"]
    statistically_supported = ci_upper < 0.0
    practical_threshold_pass = relative_gain >= 0.05
    if statistically_supported and practical_threshold_pass:
        verdict = "PROMOTION_NOMINATION_ONLY"
    else:
        verdict = "NO_PROMOTION_EVIDENCE"
    return {
        "particle_count": particle_count,
        "horizon": horizon,
        "rhos": list(rhos),
        "calibration_repetitions": calibration_reps,
        "validation_repetitions": validation_reps,
        "selected_rho": selected_rho,
        "first_call_seconds": {
            "atom": atom_first_call_seconds,
            "positive": positive_first_call_seconds,
        },
        "calibration_mse": {
            str(rho): _mean(calibration_mse[rho]) for rho in valid_rhos
        },
        "validation_atom_mse": atom_mse,
        "validation_selected_mse": selected_mse,
        "validation_relative_gain": relative_gain,
        "validation_paired_difference_mean": selected_summary[
            "paired_difference_mean"
        ],
        "validation_paired_difference_mcse": selected_summary[
            "paired_difference_mcse"
        ],
        "validation_paired_difference_bootstrap_ci95": [ci_lower, ci_upper],
        "validation_all_rho_mse": (
            {
                str(rho): _mean(validation_rho_squared_errors[rho])
                for rho in rhos
            }
            if retain_all_validation_rhos
            else None
        ),
        "validation_all_rho_diagnostics": (
            {
                str(rho): _paired_error_summary(
                    validation_rho_squared_errors[rho],
                    atom_squared_errors,
                    bootstrap_seed=(
                        particle_count * 100_000 + horizon * 100 + index
                    ),
                    bootstrap_reps=bootstrap_reps,
                )
                for index, rho in enumerate(rhos)
            }
            if retain_all_validation_rhos
            else None
        ),
        "retained_all_validation_rhos": retain_all_validation_rhos,
        "statistically_supported_ranking": statistically_supported,
        "practical_threshold_pass": practical_threshold_pass,
        "verdict": verdict,
        "nonclaims": [
            "not a canonical-default or HMC promotion",
            "not evidence for DSGE or degenerate-support validity",
            "not evidence that positive bandwidth equals ATOM-FINITE",
            "all-rho holdout inspection is explanatory and cannot retune this holdout",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--particle-counts", default="32,128")
    parser.add_argument("--horizons", default="5,20")
    parser.add_argument(
        "--rho-grid", default=",".join(str(value) for value in RHO_GRID)
    )
    parser.add_argument("--calibration-reps", type=int, default=20)
    parser.add_argument("--validation-reps", type=int, default=100)
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument(
        "--retain-all-validation-rhos",
        action="store_true",
        help="retain and summarize every rho on validation paths for diagnosis",
    )
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument(
        "--tf32-mode", choices=("enabled", "disabled"), default="disabled"
    )
    parser.add_argument(
        "--jit-compile", choices=("true", "false"), default="true"
    )
    arguments = parser.parse_args()

    if arguments.calibration_reps < 2 or arguments.validation_reps < 2:
        raise ValueError("both calibration and validation need at least two paths")
    if arguments.bootstrap_reps < 1000:
        raise ValueError("bootstrap-reps must be at least 1000")
    particle_counts = _parse_int_list(arguments.particle_counts)
    horizons = _parse_int_list(arguments.horizons)
    rhos = _parse_float_list(arguments.rho_grid)
    cells = [(n, horizon) for n in particle_counts for horizon in horizons]
    if arguments.max_cells is not None:
        if arguments.max_cells < 1:
            raise ValueError("max-cells must be positive")
        cells = cells[: arguments.max_cells]
    _validate_seed_schedule(
        cells,
        arguments.calibration_reps,
        arguments.validation_reps,
    )

    output = arguments.output
    output.parent.mkdir(parents=True, exist_ok=True)
    rows_path = output.with_name("rows.jsonl")
    if rows_path.exists():
        raise FileExistsError(f"refusing to append to existing {rows_path}")

    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(
        arguments.tf32_mode == "enabled"
    )
    logical_devices = tf.config.list_logical_devices("GPU")
    if len(logical_devices) != 1:
        raise RuntimeError("campaign requires exactly one visible logical GPU")
    dtype = tf.float64 if arguments.dtype == "float64" else tf.float32
    jit_compile = arguments.jit_compile == "true"
    started = time.time()
    wall_start = time.perf_counter()
    source_root = Path(__file__).resolve().parents[2]
    source_paths = (
        Path("bayesfilter/highdim/ledh_canonical_score_tf.py"),
        Path("bayesfilter/highdim/ledh_canonical_score_stages_tf.py"),
        Path("bayesfilter/highdim/ledh_younis_kdm_integrated_tf.py"),
        Path("bayesfilter/highdim/ledh_younis_kdm_lgssm_reference_tf.py"),
        Path(__file__).relative_to(source_root),
        Path(PLAN_PATH),
    )
    manifest = {
        "schema": "bayesfilter.ledh_younis_kdm_phase4a_campaign.v1",
        "role": "paired_lgssm_score_error_diagnostic_campaign",
        "question": "Does KDM-FINITE reduce score MSE relative to the Contract-E atom endpoint against an independent Kalman score?",
        "baseline": "ATOM-FINITE full Contract-E/GenUT/dual-cap analytical endpoint",
        "candidate": "KDM-FINITE full-feedback Gaussian observation weighting",
        "oracle": "independent scalar AR(1) Kalman value and analytical score",
        "promotion_criterion": "validation paired squared-error bootstrap CI upper bound < 0 and relative MSE gain >= 5 percent",
        "promotion_vetoes": [
            "invalid endpoint output",
            "nonfinite score",
            "missing disjoint split",
            "stale or mismatched artifact",
        ],
        "nonclaims": [
            "does not establish canonical-default readiness",
            "does not establish HMC or DSGE validity",
            "does not implement Phase 4B complete mixture resampling",
        ],
        "particle_counts": list(particle_counts),
        "horizons": list(horizons),
        "rho_grid": list(rhos),
        "calibration_repetitions": arguments.calibration_reps,
        "validation_repetitions": arguments.validation_reps,
        "bootstrap_repetitions": arguments.bootstrap_reps,
        "retain_all_validation_rhos": arguments.retain_all_validation_rhos,
        "cells": [list(cell) for cell in cells],
        "dtype": dtype.name,
        "jit_compile": jit_compile,
        "tf32_mode": arguments.tf32_mode,
        "tf32_enabled": bool(
            tf.config.experimental.tensor_float_32_execution_enabled()
        ),
        "tensorflow_version": tf.__version__,
        "python": sys.executable,
        "environment_prefix": sys.prefix,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "logical_devices": [device.name for device in logical_devices],
        "gpu_memory_policy": memory_policy,
        "trust_basis": TRUST_BASIS,
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_status_short": _git_output("status", "--short"),
        "command": shlex.join((sys.executable, *sys.argv)),
        "plan_file": PLAN_PATH,
        "result_file": str(output),
        "rows_file": str(rows_path),
        "started_unix": started,
        "source_sha256": {
            str(path): _sha256(source_root / path) for path in source_paths
        },
    }
    _write_json(output.with_name("manifest.json"), manifest)

    cell_results = []
    try:
        for index, (particle_count, horizon) in enumerate(cells):
            result = _cell_result(
                particle_count=particle_count,
                horizon=horizon,
                rhos=rhos,
                calibration_reps=arguments.calibration_reps,
                validation_reps=arguments.validation_reps,
                bootstrap_reps=arguments.bootstrap_reps,
                dtype=dtype,
                jit_compile=jit_compile,
                retain_all_validation_rhos=arguments.retain_all_validation_rhos,
                rows_path=rows_path,
            )
            cell_results.append(result)
            _write_json(
                output.with_name("progress.json"),
                {
                    "completed_cells": index + 1,
                    "total_cells": len(cells),
                    "cell_results": cell_results,
                },
            )
            print(
                f"N={particle_count} T={horizon} selected rho={result['selected_rho']} "
                f"verdict={result['verdict']}",
                flush=True,
            )
        payload = {
            **manifest,
            "status": "COMPLETED",
            "cell_results": cell_results,
            "wall_time_seconds": time.perf_counter() - wall_start,
        }
        _write_json(output, payload)
    except Exception as exc:
        failure = {
            **manifest,
            "status": "FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "completed_cell_results": cell_results,
            "wall_time_seconds": time.perf_counter() - wall_start,
        }
        _write_json(output.with_name("failure.json"), failure)
        raise
    print(f"Result: {output}")


if __name__ == "__main__":
    main()
