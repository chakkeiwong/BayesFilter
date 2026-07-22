#!/usr/bin/env python3
"""Scope-tuned GenUT continuation for the canonical predator-prey T20 row."""

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
from typing import Any, Callable

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf


PLAN = Path(
    "docs/plans/"
    "bayesfilter-genut-predator-prey-leaderboard-continuation-plan-2026-07-22.md"
)
RESULT_NOTE = Path(
    "docs/plans/"
    "bayesfilter-genut-predator-prey-leaderboard-continuation-result-2026-07-22.md"
)
CAMPAIGN_ID = "genut-predator-prey-leaderboard-continuation-20260722"
SCHEMA_VERSION = "bayesfilter.genut_predator_prey_leaderboard_continuation.v1"
ROW_ID = "zhao_cui_predator_prey_T20"
HORIZON = 20
THETA = (0.6, 114.0, 25.0, 0.3, 0.5, 0.5)
PARAMETER_LABELS = ("r", "K", "a", "s", "u", "v")
LABELS = ("value", *PARAMETER_LABELS)
PARTICLE_COUNTS = (96, 384, 1002)
TUNING_PARTICLE_COUNT = 1002
PF_REFERENCE_COUNTS = (65_536, 262_144)
TUNING_REFERENCE_COUNT = PF_REFERENCE_COUNTS[0]
T_CRITICAL_95 = {
    3: 3.182446305284263,
    7: 2.3646242515927844,
    15: 2.131449545559323,
}
RESIDUAL_TOLERANCE = 5.0e-4
FD_RELATIVE_TOLERANCE = 5.0e-2
SCORE_SCALES = (50.0, 25.0, 2.0, 1.0, 25.0, 10.0, 10.0)
CONTROL_GRID = tuple(
    {"epsilon": epsilon, "sinkhorn_steps": steps, "ridge": ridge}
    for epsilon in (2.0, 4.0)
    for steps in (4, 8)
    for ridge in (1.0e-6, 1.0e-5)
)
DGP_SEEDS = {
    "calibration": (95101, 95102),
    "validation": (95201, 95202),
    "claim": (81104,),
}
TUNING_PARTICLE_SEEDS = tuple(range(96101, 96105))
TUNING_REFERENCE_SEEDS = tuple(range(97001, 97009))
CLAIM_PARTICLE_SEEDS = tuple(range(97201, 97217))
CLAIM_REFERENCE_SEEDS = tuple(range(97301, 97317))
FD_STEPS = (0.004, 0.4, 0.1, 0.004, 0.004, 0.004)
FD_OFFSETS = (-1.0, 0.0, 1.0)
FD_OFFSET_VECTOR = (0.02, 1.0, 0.5, 0.02, 0.02, 0.02)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_sha256(value: tf.Tensor) -> str:
    return hashlib.sha256(tf.io.serialize_tensor(value).numpy()).hexdigest()


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _dataset(seed: int) -> dict[str, tf.Tensor]:
    from scripts.filtering_value_gradient_benchmark_generate_p8_datasets import (
        _predator_prey_dataset,
    )

    raw = _predator_prey_dataset(seed)
    return {
        "states": tf.cast(tf.convert_to_tensor(raw["states"]), tf.float32),
        "observations": tf.cast(
            tf.convert_to_tensor(raw["observations"]), tf.float32
        ),
    }


def _particle_noise(seed: int, particle_count: int) -> tuple[tf.Tensor, tf.Tensor]:
    return (
        tf.random.stateless_normal(
            [particle_count, 2], [seed, 101], dtype=tf.float32
        ),
        tf.random.stateless_normal(
            [HORIZON, particle_count, 2], [seed, 102], dtype=tf.float32
        ),
    )


def _genut_design(particle_count: int) -> tf.Tensor:
    from bayesfilter.highdim.cubature_genut_candidate import (
        gaussian_genut_design,
        replicate_positive_genut,
    )

    return replicate_positive_genut(
        gaussian_genut_design(dim=2), num_particles=particle_count
    )


def _make_genut_evaluator(
    particle_count: int, controls: dict[str, float | int]
) -> Callable[..., tuple[tf.Tensor, tf.Tensor, dict[str, tf.Tensor]]]:
    from bayesfilter.highdim.cubature_genut_adapters import (
        predator_prey_candidate_adapter,
    )
    from bayesfilter.highdim.cubature_genut_filter import finite_value_score

    adapter = predator_prey_candidate_adapter()

    @tf.function(jit_compile=True, reduce_retracing=True)
    def evaluate(theta, observations, initial_noise, process_noise, design):
        theta = tf.ensure_shape(theta, [6])
        observations = tf.ensure_shape(observations, [HORIZON, 2])
        initial_noise = tf.ensure_shape(initial_noise, [particle_count, 2])
        process_noise = tf.ensure_shape(
            process_noise, [HORIZON, particle_count, 2]
        )
        design = tf.ensure_shape(design, [particle_count, 2])
        with tf.device("/GPU:0"):
            return finite_value_score(
                adapter,
                theta,
                observations,
                initial_noise,
                process_noise,
                design,
                epsilon=float(controls["epsilon"]),
                sinkhorn_steps=int(controls["sinkhorn_steps"]),
                ridge=float(controls["ridge"]),
                transition_before_first_observation=False,
            )

    return evaluate


def _evaluate_genut(
    evaluate: Callable[..., tuple[tf.Tensor, tf.Tensor, dict[str, tf.Tensor]]],
    theta: tf.Tensor,
    observations: tf.Tensor,
    initial_noise: tf.Tensor,
    process_noise: tf.Tensor,
    design: tf.Tensor,
) -> dict[str, object]:
    value, score, diagnostics = evaluate(
        theta, observations, initial_noise, process_noise, design
    )
    score_sum_residual = tf.reduce_max(
        tf.abs(tf.reduce_sum(diagnostics["score_increments"], axis=0) - score)
    )
    maximum_residual = tf.reduce_max(
        tf.stack(
            [
                diagnostics["max_mean_residual"],
                diagnostics["max_row_residual"],
                diagnostics["max_col_residual"],
                score_sum_residual,
            ]
        )
    )
    return {
        "value": float(value.numpy()),
        "score": [float(item) for item in score.numpy()],
        "finite": bool(tf.math.is_finite(value).numpy())
        and bool(tf.reduce_all(tf.math.is_finite(score)).numpy()),
        "max_mean_residual": float(diagnostics["max_mean_residual"].numpy()),
        "max_row_residual": float(diagnostics["max_row_residual"].numpy()),
        "max_col_residual": float(diagnostics["max_col_residual"].numpy()),
        "score_increment_sum_residual": float(score_sum_residual.numpy()),
        "maximum_residual": float(maximum_residual.numpy()),
        "device": str(value.device),
    }


def _physical_to_source(physical: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    from bayesfilter.testing.predator_prey_ukf_neutra_target_tf import (
        PP_PARAMETER_LOWER,
        PP_PARAMETER_UPPER,
    )

    values = tf.convert_to_tensor(physical, tf.float64)
    probability = (values - PP_PARAMETER_LOWER) / (
        PP_PARAMETER_UPPER - PP_PARAMETER_LOWER
    )
    source = tf.sqrt(tf.constant(2.0, tf.float64)) * tf.math.erfinv(
        2.0 * probability - 1.0
    )
    density = tf.exp(-0.5 * tf.square(source)) / tf.sqrt(
        tf.constant(2.0 * math.pi, tf.float64)
    )
    derivative = (PP_PARAMETER_UPPER - PP_PARAMETER_LOWER) * density
    return source, derivative


def _make_pf_evaluator(
    particle_count: int, seed_count: int
) -> Callable[..., dict[str, tf.Tensor]]:
    from bayesfilter.testing.predator_prey_bootstrap_pf_reference_tf import (
        predator_prey_bootstrap_pf_reference,
    )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def evaluate(source_theta, observations, seeds):
        source_theta = tf.ensure_shape(source_theta, [1, 6])
        observations = tf.ensure_shape(observations, [HORIZON, 2])
        seeds = tf.ensure_shape(seeds, [seed_count])
        with tf.device("/GPU:0"):
            return predator_prey_bootstrap_pf_reference(
                source_theta,
                observations=tf.cast(observations, tf.float64),
                seeds=seeds,
                num_particles=particle_count,
            )

    return evaluate


def _evaluate_pf(
    evaluate: Callable[..., dict[str, tf.Tensor]],
    observations: tf.Tensor,
    seeds: tuple[int, ...],
) -> dict[str, object]:
    source, _ = _physical_to_source(tf.constant(THETA, tf.float64))
    result = evaluate(
        source[None, :], observations, tf.constant(seeds, tf.int32)
    )
    return {
        "values": [float(item) for item in result["log_likelihood"].numpy()],
        "minimum_ess": [float(item) for item in result["minimum_ess"].numpy()],
        "minimum_state": [float(item) for item in result["minimum_state"].numpy()],
        "finite": [bool(item) for item in result["finite"].numpy()],
        "device": str(result["log_likelihood"].device),
    }


def _summary(values: list[float]) -> dict[str, float | int]:
    if len(values) < 2:
        raise ValueError("uncertainty summary requires at least two values")
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    se = sd / math.sqrt(len(values))
    degrees_of_freedom = len(values) - 1
    try:
        critical = T_CRITICAL_95[degrees_of_freedom]
    except KeyError as exc:
        raise ValueError(
            f"no predeclared Student critical value for df={degrees_of_freedom}"
        ) from exc
    half = critical * se
    return {
        "count": len(values),
        "degrees_of_freedom": degrees_of_freedom,
        "critical_value": critical,
        "mean": mean,
        "sample_sd": sd,
        "standard_error": se,
        "ci95_lower": mean - half,
        "ci95_upper": mean + half,
    }


def _coordinate_summaries(rows: list[list[float]]) -> dict[str, object]:
    return {
        label: _summary([row[index] for row in rows])
        for index, label in enumerate(LABELS)
    }


def _paired_difference(
    left: list[list[float]], right: list[list[float]]
) -> dict[str, object]:
    if len(left) != len(right):
        raise ValueError("paired difference requires equal row counts")
    return _coordinate_summaries(
        [[a - b for a, b in zip(left_row, right_row)] for left_row, right_row in zip(left, right)]
    )


def _independent_mean_difference(
    left: dict[str, float | int], right: dict[str, float | int]
) -> dict[str, float]:
    mean = float(left["mean"]) - float(right["mean"])
    se = math.sqrt(
        float(left["standard_error"]) ** 2
        + float(right["standard_error"]) ** 2
    )
    critical = T_CRITICAL_95[15]
    half = critical * se
    return {
        "mean": mean,
        "standard_error": se,
        "ci95_lower": mean - half,
        "ci95_upper": mean + half,
        "critical_value": critical,
        "conservative_df": 15,
    }


def _contains_zero(interval: dict[str, float | int]) -> bool:
    return float(interval["ci95_lower"]) <= 0.0 <= float(interval["ci95_upper"])


def _fd_audit(
    evaluate: Callable[..., tuple[tf.Tensor, tf.Tensor, dict[str, tf.Tensor]]],
    observations: tf.Tensor,
    design: tf.Tensor,
) -> dict[str, object]:
    initial, process = _particle_noise(
        TUNING_PARTICLE_SEEDS[0], TUNING_PARTICLE_COUNT
    )
    rows = []
    maximum_error = 0.0
    for offset in FD_OFFSETS:
        theta = tf.constant(
            [
                value + offset * shift
                for value, shift in zip(THETA, FD_OFFSET_VECTOR)
            ],
            tf.float32,
        )
        analytical = _evaluate_genut(
            evaluate, theta, observations, initial, process, design
        )["score"]
        for multiplier in (1.0, 2.0):
            fd_score = []
            relative_errors = []
            for index, base_step in enumerate(FD_STEPS):
                step = tf.constant(multiplier * base_step, tf.float32)
                direction = tf.one_hot(index, 6, dtype=tf.float32)
                plus, _, _ = evaluate(
                    theta + step * direction,
                    observations,
                    initial,
                    process,
                    design,
                )
                minus, _, _ = evaluate(
                    theta - step * direction,
                    observations,
                    initial,
                    process,
                    design,
                )
                fd_value = float(((plus - minus) / (2.0 * step)).numpy())
                error = abs(float(analytical[index]) - fd_value) / max(
                    1.0, abs(fd_value)
                )
                fd_score.append(fd_value)
                relative_errors.append(error)
                maximum_error = max(maximum_error, error)
            rows.append(
                {
                    "offset": offset,
                    "step_multiplier": multiplier,
                    "theta": [float(item) for item in theta.numpy()],
                    "analytical_score": analytical,
                    "fd_score": fd_score,
                    "relative_errors": relative_errors,
                }
            )
    return {
        "rows": rows,
        "maximum_relative_error": maximum_error,
        "diagnostic_only": True,
    }


def _tuning_reference() -> tuple[dict[int, dict[str, object]], Callable[..., Any]]:
    evaluate = _make_pf_evaluator(
        TUNING_REFERENCE_COUNT, len(TUNING_REFERENCE_SEEDS)
    )
    references = {}
    for partition in ("calibration", "validation"):
        for seed in DGP_SEEDS[partition]:
            observations = _dataset(seed)["observations"]
            row = _evaluate_pf(evaluate, observations, TUNING_REFERENCE_SEEDS)
            row["summary"] = _summary(row["values"])
            references[seed] = row
    return references, evaluate


def _tune() -> dict[str, object]:
    references, _ = _tuning_reference()
    design = _genut_design(TUNING_PARTICLE_COUNT)
    theta = tf.constant(THETA, tf.float32)
    candidates = []
    for controls in CONTROL_GRID:
        evaluate = _make_genut_evaluator(TUNING_PARTICLE_COUNT, controls)
        partition_objectives = {}
        maximum_residual = 0.0
        all_valid = True
        for partition in ("calibration", "validation"):
            value_errors = []
            variance_objectives = []
            for dataset_seed in DGP_SEEDS[partition]:
                observations = _dataset(dataset_seed)["observations"]
                values = []
                for particle_seed in TUNING_PARTICLE_SEEDS:
                    initial, process = _particle_noise(
                        particle_seed, TUNING_PARTICLE_COUNT
                    )
                    result = _evaluate_genut(
                        evaluate,
                        theta,
                        observations,
                        initial,
                        process,
                        design,
                    )
                    values.append([result["value"], *result["score"]])
                    maximum_residual = max(
                        maximum_residual, float(result["maximum_residual"])
                    )
                    all_valid = (
                        all_valid
                        and bool(result["finite"])
                        and "GPU" in str(result["device"]).upper()
                    )
                value_errors.append(
                    abs(
                        statistics.mean(row[0] for row in values)
                        - float(references[dataset_seed]["summary"]["mean"])
                    )
                )
                variance_objectives.append(
                    max(
                        statistics.variance(row[index] for row in values)
                        / (SCORE_SCALES[index] ** 2)
                        for index in range(len(LABELS))
                    )
                )
            partition_objectives[partition] = {
                "mean_absolute_value_error_to_pf": statistics.mean(value_errors),
                "mean_maximum_scaled_conditional_variance": statistics.mean(
                    variance_objectives
                ),
            }
        audit = _fd_audit(
            evaluate,
            _dataset(DGP_SEEDS["calibration"][0])["observations"],
            design,
        )
        eligible = (
            all_valid
            and maximum_residual < RESIDUAL_TOLERANCE
            and float(audit["maximum_relative_error"]) < FD_RELATIVE_TOLERANCE
        )
        candidates.append(
            {
                "controls": controls,
                "partition_objectives": partition_objectives,
                "maximum_residual": maximum_residual,
                "fd_audit": audit,
                "eligible": eligible,
            }
        )
    eligible = [row for row in candidates if row["eligible"]]
    if not eligible:
        raise RuntimeError("no eligible predator-prey GenUT tuning candidate")
    selected = min(
        eligible,
        key=lambda row: (
            row["partition_objectives"]["validation"][
                "mean_absolute_value_error_to_pf"
            ],
            row["partition_objectives"]["validation"][
                "mean_maximum_scaled_conditional_variance"
            ],
            row["partition_objectives"]["calibration"][
                "mean_absolute_value_error_to_pf"
            ],
            row["fd_audit"]["maximum_relative_error"],
            int(row["controls"]["sinkhorn_steps"]),
            -float(row["controls"]["ridge"]),
            float(row["controls"]["epsilon"]),
        ),
    )
    return {
        "scope": {
            "row_id": ROW_ID,
            "horizon": HORIZON,
            "particle_count": TUNING_PARTICLE_COUNT,
            "dtype": "float32",
            "tf32": True,
            "jit_compile": True,
            "design": "positive_gaussian_genut_dim2_equal_mass",
            "score": "recursive_forward_sensitivity_no_autodiff_no_fd",
        },
        "partitions": DGP_SEEDS,
        "particle_seeds": TUNING_PARTICLE_SEEDS,
        "reference": {
            "particle_count": TUNING_REFERENCE_COUNT,
            "seeds": TUNING_REFERENCE_SEEDS,
            "rows": references,
        },
        "control_grid": CONTROL_GRID,
        "candidates": candidates,
        "selected_controls": selected["controls"],
        "selected_candidate": selected,
        "claim_data_read_during_selection": False,
    }


def _route_identity(
    particle_count: int,
    controls: dict[str, float | int],
    observation_hash: str,
) -> dict[str, object]:
    from bayesfilter.highdim.cubature_genut_candidate import (
        CandidateRouteScope,
        issue_repository_candidate_route_identity,
        validate_repository_candidate_route_identity,
    )

    identity = issue_repository_candidate_route_identity(
        CandidateRouteScope(
            model_id="predator_prey_additive_gaussian",
            target_id=ROW_ID,
            horizon=HORIZON,
            particle_count=particle_count,
            state_dimension=2,
            parameter_count=6,
            dtype="float32",
            tf32_enabled=True,
            jit_compile=True,
            design_family="genut",
            control_family_id="predator_prey_genut_controls_v1",
        ),
        prepared_data_id=f"sha256:{observation_hash}",
        residual_design_id=(
            f"gaussian_genut_dim2_equal_mass_n{particle_count}_v1"
        ),
        controls={key: str(value) for key, value in controls.items()},
        adapter_id="predator_prey_additive_gaussian_v1",
    )
    validate_repository_candidate_route_identity(identity)
    return identity.to_dict()


def _deterministic_diagnostics(observations: tf.Tensor) -> dict[str, object]:
    from bayesfilter.testing.predator_prey_ukf_neutra_target_tf import (
        pp_ukf_likelihood_value_score_status,
    )

    observations64 = tf.cast(observations, tf.float64)
    physical_theta = tf.constant(THETA, tf.float64)

    source, dphysical_dsource = _physical_to_source(physical_theta)
    ukf_value, ukf_source_score, ukf_status = pp_ukf_likelihood_value_score_status(
        source[None, :], observations=observations64
    )
    ukf_physical_score = ukf_source_score[0] / dphysical_dsource
    return {
        "fixed_sgqf": {
            "role": "blocked_target_mismatch_not_evidence",
            "value": None,
            "physical_score": None,
            "route_id": "fixed_sgqf_direct_predator_prey_t20",
            "status": "blocked",
            "reason": (
                "generic SGQF transitions before y0; canonical predator-prey "
                "T20 assimilates y0 before the first transition"
            ),
        },
        "principal_sqrt_ukf": {
            "role": "same_target_analytical_approximation_diagnostic_not_oracle",
            "value": float(ukf_value[0].numpy()),
            "source_score": [float(item) for item in ukf_source_score[0].numpy()],
            "physical_score": [float(item) for item in ukf_physical_score.numpy()],
            "valid": bool(ukf_status["valid_pre_regularized_score"][0].numpy()),
            "route_id": "predator_prey_initial_observation_first_principal_sqrt_ukf",
        },
    }


def _claim(tuning: dict[str, object]) -> dict[str, object]:
    controls = dict(tuning["selected_controls"])
    dataset = _dataset(DGP_SEEDS["claim"][0])
    observations = dataset["observations"]
    observation_hash = _tensor_sha256(observations)
    theta = tf.constant(THETA, tf.float32)
    raw_genut = {}
    genut_summaries = {}
    identities = {}
    maximum_residual = 0.0
    for particle_count in PARTICLE_COUNTS:
        design = _genut_design(particle_count)
        evaluate = _make_genut_evaluator(particle_count, controls)
        rows = []
        raw_rows = []
        for particle_seed in CLAIM_PARTICLE_SEEDS:
            initial, process = _particle_noise(particle_seed, particle_count)
            result = _evaluate_genut(
                evaluate, theta, observations, initial, process, design
            )
            maximum_residual = max(
                maximum_residual, float(result["maximum_residual"])
            )
            if (
                not bool(result["finite"])
                or "GPU" not in str(result["device"]).upper()
                or float(result["maximum_residual"]) >= RESIDUAL_TOLERANCE
            ):
                raise RuntimeError(f"GenUT claim veto at N={particle_count}")
            rows.append([result["value"], *result["score"]])
            raw_rows.append({"particle_seed": particle_seed, **result})
        raw_genut[str(particle_count)] = raw_rows
        genut_summaries[str(particle_count)] = _coordinate_summaries(rows)
        identities[str(particle_count)] = _route_identity(
            particle_count, controls, observation_hash
        )

    pf_rows = {}
    pf_summaries = {}
    for particle_count in PF_REFERENCE_COUNTS:
        evaluate_pf = _make_pf_evaluator(
            particle_count, len(CLAIM_REFERENCE_SEEDS)
        )
        result = _evaluate_pf(evaluate_pf, observations, CLAIM_REFERENCE_SEEDS)
        if (
            not all(result["finite"])
            or "GPU" not in str(result["device"]).upper()
        ):
            raise RuntimeError(f"bootstrap PF reference veto at N={particle_count}")
        pf_rows[str(particle_count)] = result
        pf_summaries[str(particle_count)] = _summary(result["values"])

    genut_384 = [
        [row["value"], *row["score"]] for row in raw_genut["384"]
    ]
    genut_1002 = [
        [row["value"], *row["score"]] for row in raw_genut["1002"]
    ]
    score_stability = _paired_difference(genut_1002, genut_384)
    pf_low = [[value] + [0.0] * 6 for value in pf_rows[str(PF_REFERENCE_COUNTS[0])]["values"]]
    pf_high = [[value] + [0.0] * 6 for value in pf_rows[str(PF_REFERENCE_COUNTS[1])]["values"]]
    pf_refinement = _paired_difference(pf_high, pf_low)["value"]
    value_difference = _independent_mean_difference(
        genut_summaries["1002"]["value"],
        pf_summaries[str(PF_REFERENCE_COUNTS[1])],
    )
    score_stable = all(
        _contains_zero(score_stability[label]) for label in PARAMETER_LABELS
    )
    value_compatible = _contains_zero(value_difference) and _contains_zero(
        pf_refinement
    )
    return {
        "dataset_seed": DGP_SEEDS["claim"][0],
        "observation_sha256": observation_hash,
        "state_sha256": _tensor_sha256(dataset["states"]),
        "controls": controls,
        "route_identities": identities,
        "genut": {
            "particle_seeds": CLAIM_PARTICLE_SEEDS,
            "summaries": genut_summaries,
            "raw": raw_genut,
            "maximum_residual": maximum_residual,
        },
        "bootstrap_pf_reference": {
            "role": "independent_same_target_value_reference_not_exact_oracle",
            "particle_seeds": CLAIM_REFERENCE_SEEDS,
            "summaries": pf_summaries,
            "raw": pf_rows,
            "refinement_high_minus_low": pf_refinement,
            "refinement_compatible_at_95pct": _contains_zero(pf_refinement),
        },
        "genut_n1002_minus_pf_high_value": value_difference,
        "genut_n1002_minus_n384_paired": score_stability,
        "deterministic_diagnostics": _deterministic_diagnostics(observations),
        "criteria": {
            "value_compatible_at_95pct": value_compatible,
            "score_n_stable_all_coordinates_at_95pct": score_stable,
            "score_truth_established": False,
        },
    }


def _write_result_note(payload: dict[str, object]) -> None:
    claim = payload["claim"]
    lines = [
        "# GenUT Predator-Prey Leaderboard Continuation Result",
        "",
        "Date: 2026-07-22",
        "",
        f"Status: `{payload['decision']['status']}`",
        "",
        "## Value",
        "",
        "| Route | N | Mean | SD | 95% CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for particle_count in PARTICLE_COUNTS:
        row = claim["genut"]["summaries"][str(particle_count)]["value"]
        lines.append(
            f"| GenUT | {particle_count} | {row['mean']:.6f} | "
            f"{row['sample_sd']:.6f} | [{row['ci95_lower']:.6f}, {row['ci95_upper']:.6f}] |"
        )
    for particle_count in PF_REFERENCE_COUNTS:
        row = claim["bootstrap_pf_reference"]["summaries"][str(particle_count)]
        lines.append(
            f"| Bootstrap PF reference | {particle_count} | {row['mean']:.6f} | "
            f"{row['sample_sd']:.6f} | [{row['ci95_lower']:.6f}, {row['ci95_upper']:.6f}] |"
        )
    difference = claim["genut_n1002_minus_pf_high_value"]
    lines.extend(
        [
            "",
            "## Score Stability",
            "",
            "| Coordinate | GenUT N=1002 mean | 95% CI | N=1002 minus N=384 95% CI | Stable |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label in PARAMETER_LABELS:
        row = claim["genut"]["summaries"]["1002"][label]
        delta = claim["genut_n1002_minus_n384_paired"][label]
        lines.append(
            f"| {label} | {row['mean']:.6f} | "
            f"[{row['ci95_lower']:.6f}, {row['ci95_upper']:.6f}] | "
            f"[{delta['ci95_lower']:.6f}, {delta['ci95_upper']:.6f}] | "
            f"{_contains_zero(delta)} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"GenUT N=1002 minus refined PF value: `{difference['mean']:.6f}` "
            f"with 95% interval `[{difference['ci95_lower']:.6f}, "
            f"{difference['ci95_upper']:.6f}]`.",
            "",
            payload["decision"]["text"],
            "",
            "| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |",
            "|---|---|---|---|---|---|",
            f"| Predator-prey GenUT value | {claim['criteria']['value_compatible_at_95pct']} | Passed engineering/reference checks | Bootstrap PF is refined, not exact | Preserve as candidate value evidence | No exact likelihood or superiority |",
            f"| Predator-prey GenUT score | N-stable={claim['criteria']['score_n_stable_all_coordinates_at_95pct']} | Recursive same-scalar audit passed in tuning | No independent marginal-score oracle | Build an independent analytical score authority or stronger consistency ladder | No score truth or HMC readiness |",
            "| Leaderboard admission | Not admitted | Identity valid; evidence incomplete | Score truth and integration schema absent | Close cross-cutting leaderboard gaps | No default change |",
            "",
            "## Inference Status",
            "",
            "| Evidence class | Status |",
            "|---|---|",
            "| Hard veto screen | Passed |",
            "| Statistically supported ranking | None; this is compatibility and stability evidence |",
            "| Descriptive-only differences | SGQF/UKF comparisons, runtimes, and per-rung means outside declared intervals |",
            "| Default readiness | Failed; GenUT remains experimental |",
            "| Next evidence needed | Independent analytical score validation, leaderboard wiring, and high-dimensional memory repair |",
            "",
            "## Post-Run Red Team",
            "",
            "The strongest alternative explanation is that two biased approximations agree at one dataset and parameter point. The result would be overturned by PF refinement drift, fresh-DGP value disagreement, or persistent score movement at larger N. The weakest evidence remains score truth, not finite execution.",
        ]
    )
    (ROOT / RESULT_NOTE).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(output_root: Path) -> dict[str, object]:
    started = time.perf_counter()
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    output_root.mkdir(parents=True, exist_ok=False)
    from bayesfilter.runtime.gpu_memory_policy import (
        configure_tensorflow_gpu_memory_growth,
    )

    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    logical = tf.config.list_logical_devices("GPU")
    if not logical:
        raise RuntimeError("predator-prey GenUT campaign requires a logical GPU")

    tuning = _tune()
    (output_root / "tuning.json").write_text(
        json.dumps(tuning, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    claim = _claim(tuning)
    allocator = tf.config.experimental.get_memory_info("GPU:0")
    criteria = claim["criteria"]
    if criteria["value_compatible_at_95pct"] and criteria[
        "score_n_stable_all_coordinates_at_95pct"
    ]:
        status = "VALUE_COMPATIBLE_SCORE_N_STABLE_DIAGNOSTIC_ONLY"
        text = (
            "GenUT passed the declared value-compatibility and score-N-stability "
            "screens. The row remains unadmitted because no independent score "
            "truth authority or leaderboard integration exists."
        )
    elif criteria["value_compatible_at_95pct"]:
        status = "VALUE_COMPATIBLE_SCORE_N_UNSTABLE_DIAGNOSTIC_ONLY"
        text = (
            "GenUT passed the value-compatibility screen but not the score-N-stability "
            "screen. This is a score-repair trigger, not leaderboard evidence."
        )
    else:
        status = "VALUE_NOT_COMPATIBLE_DIAGNOSTIC_ONLY"
        text = (
            "GenUT did not pass the refined value-reference screen. The current "
            "candidate is not eligible for leaderboard admission."
        )
    source_paths = (
        Path(__file__).relative_to(ROOT),
        Path("bayesfilter/highdim/cubature_genut_candidate.py"),
        Path("bayesfilter/highdim/cubature_genut_adapters.py"),
        Path("bayesfilter/highdim/cubature_genut_filter.py"),
        Path("bayesfilter/testing/predator_prey_bootstrap_pf_reference_tf.py"),
        PLAN,
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": CAMPAIGN_ID,
        "started_utc": started_utc,
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_time_seconds": time.perf_counter() - started,
        "git_commit": _git_commit(),
        "host": platform.node(),
        "tensorflow_version": tf.__version__,
        "plan": PLAN.as_posix(),
        "device": {
            "logical_devices": [item.name for item in logical],
            "genut_dtype": "float32",
            "reference_dtype": "float64",
            "tf32_enabled": bool(
                tf.config.experimental.tensor_float_32_execution_enabled()
            ),
            "jit_compile": True,
            "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        },
        "memory_policy": memory_policy,
        "gpu_allocator": {key: int(value) for key, value in allocator.items()},
        "target": {
            "row_id": ROW_ID,
            "horizon": HORIZON,
            "theta_physical": THETA,
            "parameter_order": PARAMETER_LABELS,
            "timing": "initial_observation_first_then_19_transitions",
            "initial_law": "N([50,5],I)",
            "process_covariance": "4I",
            "observation_covariance": "4I",
            "rk4_delta": 2.0,
            "rk4_internal_step": 0.1,
        },
        "configuration": {
            "particle_counts": PARTICLE_COUNTS,
            "pf_reference_counts": PF_REFERENCE_COUNTS,
            "control_grid": CONTROL_GRID,
            "dgp_seeds": DGP_SEEDS,
            "tuning_particle_seeds": TUNING_PARTICLE_SEEDS,
            "claim_particle_seeds": CLAIM_PARTICLE_SEEDS,
            "claim_reference_seeds": CLAIM_REFERENCE_SEEDS,
            "runtime_score": "recursive_forward_sensitivity_no_autodiff_no_fd",
        },
        "source_sha256": {
            path.as_posix(): _sha256(ROOT / path) for path in source_paths
        },
        "tuning": {
            "selected_controls": tuning["selected_controls"],
            "tuning_artifact": (output_root / "tuning.json").as_posix(),
            "claim_data_read_during_selection": False,
        },
        "claim": claim,
        "engineering_ledger": {
            "finite": True,
            "gpu_xla_tf32": True,
            "memory_growth": True,
            "repository_identity": True,
            "maximum_residual": claim["genut"]["maximum_residual"],
        },
        "numerical_ledger": {
            "value_compatible_at_95pct": criteria["value_compatible_at_95pct"],
            "score_n_stable_all_coordinates_at_95pct": criteria[
                "score_n_stable_all_coordinates_at_95pct"
            ],
            "score_truth_established": False,
        },
        "scientific_ledger": {
            "leaderboard_admitted": False,
            "default_changed": False,
            "statistically_supported_ranking": False,
        },
        "decision": {
            "status": status,
            "text": text,
            "leaderboard_admitted": False,
            "default_changed": False,
        },
        "nonclaims": [
            "no exact nonlinear likelihood or score",
            "no unbiasedness or superiority",
            "no HMC readiness",
            "no high-dimensional feasibility",
            "no Zhao-Cui source-faithfulness conclusion",
            "no GenUT default promotion",
        ],
    }
    result_path = output_root / "result.json"
    result_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": "bayesfilter.serious_run_manifest.v1",
        "git_commit": payload["git_commit"],
        "command": " ".join(sys.argv),
        "environment": sys.executable,
        "cpu_gpu_status": payload["device"],
        "memory_policy": memory_policy,
        "data_version": "canonical_generated_predator_prey_seed_81104",
        "random_seeds": {
            "dgp": DGP_SEEDS,
            "genut": CLAIM_PARTICLE_SEEDS,
            "pf_reference": CLAIM_REFERENCE_SEEDS,
        },
        "wall_time_seconds": payload["wall_time_seconds"],
        "output_artifact_paths": [
            result_path.as_posix(),
            (output_root / "tuning.json").as_posix(),
        ],
        "plan_file": PLAN.as_posix(),
        "result_file": RESULT_NOTE.as_posix(),
        "result_sha256": _sha256(result_path),
    }
    (output_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_result_note(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    payload = run(args.output_root)
    print(
        json.dumps(
            {
                "status": payload["decision"]["status"],
                "wall_time_seconds": payload["wall_time_seconds"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
