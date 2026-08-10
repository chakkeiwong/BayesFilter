#!/usr/bin/env python3
"""Run an equal-cost antithetic GenUT ensemble study on frozen Austria SIR."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf


PLAN = Path(
    "docs/plans/bayesfilter-genut-austria-antithetic-ensemble-plan-2026-08-03.md"
)
PRIOR_TUNING_ARTIFACT = Path(
    "docs/benchmarks/artifacts/moment_retuned_genut_whole_leaderboard_20260723/"
    "attempt05_final/austria_sir_T20_checkpoint.json"
)
SGQF_ARTIFACT = Path(
    "docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p6/"
    "SIR-SGQF/r1b-identity/gpu-attempt-02/result.json"
)
SCHEMA = "bayesfilter.genut_austria_antithetic_ensemble.v1"
CAMPAIGN_ID = "genut-austria-antithetic-ensemble-20260803"
MODEL_ID = "austria_sir_T20"
EXPECTED_OBSERVATION_SHA256 = (
    "cd794ad6e90a74f7cf6dc06b33550bff4bef6fbf66bb0917846d0691b5910f07"
)
EXPECTED_CONTROLS = {
    "epsilon": 8.0,
    "sinkhorn_steps": 16,
    "balance_steps": 16,
    "ridge": 1.0e-5,
    "higher_moment_correction_steps": 4,
    "higher_moment_strength": 0.2,
    "higher_moment_floor": 1.0e-5,
}
ZERO_EXTENSION_CONTROLS = {
    "pairwise_moment_correction_steps": 0,
    "pairwise_moment_strength": 0.0,
    "pairwise_moment_floor": 1.0e-5,
    "projected_cumulant_correction_steps": 0,
    "projected_cumulant_strength": 0.0,
    "projected_cumulant_floor": 1.0e-5,
}
CURRENT_CONTROLS_GRID = tuple(
    {
        "epsilon": epsilon,
        "sinkhorn_steps": steps,
        "balance_steps": balance,
        "ridge": 1.0e-5,
        "higher_moment_correction_steps": higher_steps,
        "higher_moment_strength": higher_strength,
        "higher_moment_floor": 1.0e-5,
        **ZERO_EXTENSION_CONTROLS,
    }
    for epsilon in (4.0, 8.0)
    for steps, balance in ((8, 8), (16, 16))
    for higher_steps, higher_strength in ((0, 0.02), (4, 0.2))
)
TUNING_SEEDS = (98101, 98102)
LABELS = (
    "value",
    "log_kappa_scale",
    "log_nu_scale",
    "log_observation_noise_scale",
)
PARTICLE_COUNT = 1008
HORIZON = 20
STATE_DIMENSION = 18
PARAMETER_DIMENSION = 3
K_VALUES = (1, 2, 4)
MAX_K = max(K_VALUES)
RESIDUAL_TOLERANCE = 5.0e-4
DISPLACEMENT_VETO = 2.0
SEVERE_SCORE_THRESHOLD = 1000.0
VARIANCE_FLOOR = 1.0e-18
BOOTSTRAP_DRAWS = 20000
BOOTSTRAP_SEED = 20260803
FD_RELATIVE_STEPS = (0.004, 0.008)
FD_MINIMUM_STEPS = (0.0004, 0.0008)
FD_RELATIVE_TOLERANCE = 0.05


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_sha256(value: tf.Tensor) -> str:
    serialized = tf.io.serialize_tensor(tf.convert_to_tensor(value)).numpy()
    return hashlib.sha256(serialized).hexdigest()


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _json_safe(value: Any) -> Any:
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            _json_safe(payload), indent=2, sort_keys=True, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires at least one value")
    position = probability * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(
        sorted_values[lower]
        + fraction * (sorted_values[upper] - sorted_values[lower])
    )


def _summary(values: Sequence[float]) -> Mapping[str, Any]:
    sample = tuple(float(value) for value in values)
    if not sample:
        raise ValueError("summary requires at least one value")
    mean = statistics.fmean(sample)
    standard_deviation = statistics.stdev(sample) if len(sample) > 1 else None
    return {
        "count": len(sample),
        "mean": mean,
        "standard_deviation": standard_deviation,
        "mcse_of_mean": (
            standard_deviation / math.sqrt(len(sample))
            if standard_deviation is not None
            else None
        ),
        "minimum": min(sample),
        "maximum": max(sample),
    }


def _sample_variance(values: Sequence[float]) -> float:
    return statistics.variance(values) if len(values) > 1 else 0.0


def _correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_sd = statistics.stdev(left)
    right_sd = statistics.stdev(right)
    if left_sd == 0.0 or right_sd == 0.0:
        return None
    covariance = sum(
        (a - statistics.fmean(left)) * (b - statistics.fmean(right))
        for a, b in zip(left, right)
    ) / (len(left) - 1)
    return covariance / (left_sd * right_sd)


def _vector(row: Mapping[str, Any]) -> list[float]:
    return [float(row["value"]), *[float(value) for value in row["score"]]]


def _mean_vectors(rows: Sequence[Mapping[str, Any]]) -> list[float]:
    vectors = [_vector(row) for row in rows]
    return [statistics.fmean(vector[index] for vector in vectors) for index in range(4)]


def _noise(seed: int) -> tuple[tf.Tensor, tf.Tensor]:
    return (
        tf.random.stateless_normal(
            [PARTICLE_COUNT, STATE_DIMENSION], [seed, 101], dtype=tf.float32
        ),
        tf.random.stateless_normal(
            [HORIZON, PARTICLE_COUNT, STATE_DIMENSION],
            [seed, 102],
            dtype=tf.float32,
        ),
    )


def _make_current_evaluator(
    target: Mapping[str, Any], controls: Mapping[str, Any]
) -> Any:
    """Bind every current optional control, including structural zero arms."""

    from bayesfilter.highdim.cubature_genut_filter import finite_value_score

    @tf.function(jit_compile=True, reduce_retracing=True)
    def evaluate(theta, observations, initial_noise, process_noise, design):
        theta = tf.ensure_shape(theta, [PARAMETER_DIMENSION])
        observations = tf.ensure_shape(observations, [HORIZON, 9])
        initial_noise = tf.ensure_shape(
            initial_noise, [PARTICLE_COUNT, STATE_DIMENSION]
        )
        process_noise = tf.ensure_shape(
            process_noise, [HORIZON, PARTICLE_COUNT, STATE_DIMENSION]
        )
        design = tf.ensure_shape(design, [PARTICLE_COUNT, STATE_DIMENSION])
        with tf.device("/GPU:0"):
            return finite_value_score(
                target["adapter"],
                theta,
                observations,
                initial_noise,
                process_noise,
                design,
                epsilon=float(controls["epsilon"]),
                sinkhorn_steps=int(controls["sinkhorn_steps"]),
                balance_steps=int(controls["balance_steps"]),
                ridge=float(controls["ridge"]),
                transition_before_first_observation=True,
                higher_moment_correction_steps=int(
                    controls["higher_moment_correction_steps"]
                ),
                higher_moment_strength=float(controls["higher_moment_strength"]),
                higher_moment_floor=float(controls["higher_moment_floor"]),
                pairwise_moment_correction_steps=int(
                    controls["pairwise_moment_correction_steps"]
                ),
                pairwise_moment_strength=float(
                    controls["pairwise_moment_strength"]
                ),
                pairwise_moment_floor=float(controls["pairwise_moment_floor"]),
                projected_cumulant_correction_steps=int(
                    controls["projected_cumulant_correction_steps"]
                ),
                projected_cumulant_strength=float(
                    controls["projected_cumulant_strength"]
                ),
                projected_cumulant_floor=float(
                    controls["projected_cumulant_floor"]
                ),
            )

    return evaluate


def _constituent_valid(row: Mapping[str, Any]) -> bool:
    return bool(row["finite"]) and bool(row["program_valid"]) and (
        "GPU" in str(row["device"]).upper()
    ) and max(
        float(row["max_mean_residual"]),
        float(row["max_row_residual"]),
        float(row["max_col_residual"]),
        float(row["score_increment_sum_residual"]),
    ) < RESIDUAL_TOLERANCE and float(
        row["maximum_normalized_shape_displacement"]
    ) <= DISPLACEMENT_VETO


def _evaluate_constituent(
    evaluator: Any,
    theta: tf.Tensor,
    observations: tf.Tensor,
    initial_noise: tf.Tensor,
    process_noise: tf.Tensor,
    design: tf.Tensor,
    *,
    seed: int,
    arm: str,
    sign: int,
    replicate: int,
    slot: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    value, score, diagnostics = evaluator(
        theta, observations, initial_noise, process_noise, design
    )
    elapsed = time.perf_counter() - started
    score_sum_residual = tf.reduce_max(
        tf.abs(tf.reduce_sum(diagnostics["score_increments"], axis=0) - score)
    )
    finite = bool(diagnostics["program_valid"].numpy()) and bool(
        tf.math.is_finite(value).numpy()
    ) and bool(tf.reduce_all(tf.math.is_finite(score)).numpy())
    row = {
        "replicate": replicate,
        "arm": arm,
        "slot": slot,
        "root_seed": seed,
        "sign": sign,
        "value": float(value.numpy()) if finite else None,
        "score": [float(item) for item in score.numpy().tolist()] if finite else None,
        "finite": finite,
        "program_valid": bool(diagnostics["program_valid"].numpy()),
        "max_mean_residual": float(diagnostics["max_mean_residual"].numpy()),
        "max_row_residual": float(diagnostics["max_row_residual"].numpy()),
        "max_col_residual": float(diagnostics["max_col_residual"].numpy()),
        "score_increment_sum_residual": float(score_sum_residual.numpy()),
        "maximum_normalized_shape_displacement": float(
            diagnostics["maximum_normalized_shape_displacement"].numpy()
        ),
        "mean_normalized_shape_residual_objective": float(
            diagnostics["mean_normalized_shape_residual_objective"].numpy()
        ),
        "minimum_row_mass": float(diagnostics["minimum_row_mass"].numpy()),
        "device": str(value.device),
        "runtime_seconds": elapsed,
    }
    row["valid"] = _constituent_valid(row)
    row["severe_score_tail"] = bool(
        finite and max(abs(item) for item in row["score"]) > SEVERE_SCORE_THRESHOLD
    )
    return row


def _assemble_ensembles(
    constituents: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_replicate: dict[int, dict[str, list[Mapping[str, Any]]]] = {}
    for row in constituents:
        replicate = int(row["replicate"])
        arm = str(row["arm"])
        by_replicate.setdefault(replicate, {"antithetic": [], "independent": []})[
            arm
        ].append(row)
    ensembles: list[dict[str, Any]] = []
    for replicate in sorted(by_replicate):
        arms = by_replicate[replicate]
        antithetic = sorted(arms["antithetic"], key=lambda row: int(row["slot"]))
        independent = sorted(arms["independent"], key=lambda row: int(row["slot"]))
        if len(antithetic) != 2 * MAX_K or len(independent) != 2 * MAX_K:
            raise ValueError("each replicate must contain eight constituents per arm")
        for k in K_VALUES:
            anti_prefix = antithetic[: 2 * k]
            independent_prefix = independent[: 2 * k]
            for arm_name, rows in (
                ("antithetic", anti_prefix),
                ("independent", independent_prefix),
            ):
                valid = all(bool(row["valid"]) for row in rows)
                values = _mean_vectors(rows) if valid else None
                ensembles.append(
                    {
                        "replicate": replicate,
                        "arm": arm_name,
                        "K": k,
                        "complete_run_count": 2 * k,
                        "valid": valid,
                        "value": values[0] if values is not None else None,
                        "score": values[1:] if values is not None else None,
                        "runtime_seconds": sum(
                            float(row["runtime_seconds"]) for row in rows
                        ),
                        "severe_score_tail": bool(
                            values is not None
                            and max(abs(value) for value in values[1:])
                            > SEVERE_SCORE_THRESHOLD
                        ),
                        "constituent_severe_tail_count": sum(
                            bool(row["severe_score_tail"]) for row in rows
                        ),
                    }
                )
    return ensembles


def _bootstrap_log_ratio(
    antithetic: Sequence[Sequence[float]],
    independent: Sequence[Sequence[float]],
    anti_runtime: Sequence[float],
    independent_runtime: Sequence[float],
) -> list[dict[str, Any]]:
    count = len(antithetic)
    if count != len(independent) or count < 2:
        return []
    rng = random.Random(BOOTSTRAP_SEED)
    bootstrap_indices = [
        [rng.randrange(count) for _ in range(count)] for _ in range(BOOTSTRAP_DRAWS)
    ]
    output = []
    family_tail = 0.05 / (2.0 * len(LABELS))
    point_tail = 0.025
    for coordinate, label in enumerate(LABELS):
        anti_values = [row[coordinate] for row in antithetic]
        independent_values = [row[coordinate] for row in independent]
        observed_anti_variance = _sample_variance(anti_values)
        observed_independent_variance = _sample_variance(independent_values)
        log_ratios = []
        efficiency_log_ratios = []
        for indices in bootstrap_indices:
            anti_sample = [anti_values[index] for index in indices]
            independent_sample = [independent_values[index] for index in indices]
            anti_variance = _sample_variance(anti_sample)
            independent_variance = _sample_variance(independent_sample)
            log_ratios.append(
                math.log(
                    (anti_variance + VARIANCE_FLOOR)
                    / (independent_variance + VARIANCE_FLOOR)
                )
            )
            anti_time = statistics.fmean(anti_runtime[index] for index in indices)
            independent_time = statistics.fmean(
                independent_runtime[index] for index in indices
            )
            efficiency_log_ratios.append(
                math.log(
                    ((anti_variance + VARIANCE_FLOOR) * anti_time)
                    / ((independent_variance + VARIANCE_FLOOR) * independent_time)
                )
            )
        log_ratios.sort()
        efficiency_log_ratios.sort()
        observed_log_ratio = math.log(
            (observed_anti_variance + VARIANCE_FLOOR)
            / (observed_independent_variance + VARIANCE_FLOOR)
        )
        family_interval = {
            "lower": _percentile(log_ratios, family_tail),
            "upper": _percentile(log_ratios, 1.0 - family_tail),
        }
        point_interval = {
            "lower": _percentile(log_ratios, point_tail),
            "upper": _percentile(log_ratios, 1.0 - point_tail),
        }
        output.append(
            {
                "label": label,
                "antithetic_variance": observed_anti_variance,
                "independent_variance": observed_independent_variance,
                "variance_ratio_antithetic_over_independent": math.exp(
                    observed_log_ratio
                ),
                "log_variance_ratio": observed_log_ratio,
                "pointwise_95_log_ratio_interval": point_interval,
                "familywise_95_log_ratio_interval": family_interval,
                "coordinate_nominated": family_interval["upper"] < 0.0,
                "variance_times_runtime_ratio": math.exp(
                    statistics.fmean(efficiency_log_ratios)
                ),
                "familywise_95_efficiency_log_ratio_interval": {
                    "lower": _percentile(efficiency_log_ratios, family_tail),
                    "upper": _percentile(
                        efficiency_log_ratios, 1.0 - family_tail
                    ),
                },
            }
        )
    return output


def _analysis(
    ensembles: Sequence[Mapping[str, Any]],
    constituents: Sequence[Mapping[str, Any]],
    sgqf: Sequence[float],
) -> Mapping[str, Any]:
    by_k: dict[str, Any] = {}
    for k in K_VALUES:
        anti_rows = sorted(
            (
                row
                for row in ensembles
                if int(row["K"]) == k and row["arm"] == "antithetic"
            ),
            key=lambda row: int(row["replicate"]),
        )
        independent_rows = sorted(
            (
                row
                for row in ensembles
                if int(row["K"]) == k and row["arm"] == "independent"
            ),
            key=lambda row: int(row["replicate"]),
        )
        paired_valid = [
            (anti, independent)
            for anti, independent in zip(anti_rows, independent_rows)
            if bool(anti["valid"]) and bool(independent["valid"])
        ]
        anti_valid_rows = [row for row in anti_rows if bool(row["valid"])]
        independent_valid_rows = [
            row for row in independent_rows if bool(row["valid"])
        ]
        anti_vectors = [_vector(row) for row, _ in paired_valid]
        independent_vectors = [_vector(row) for _, row in paired_valid]
        summaries = {}
        for arm, rows in (
            ("antithetic", anti_valid_rows),
            ("independent", independent_valid_rows),
        ):
            vectors = [_vector(row) for row in rows]
            summaries[arm] = (
                {
                    label: _summary([vector[index] for vector in vectors])
                    for index, label in enumerate(LABELS)
                }
                if vectors
                else {label: None for label in LABELS}
            )
            summaries[arm]["runtime_seconds"] = _summary(
                [
                    float(row["runtime_seconds"])
                    for row in (anti_rows if arm == "antithetic" else independent_rows)
                ]
            )
            summaries[arm]["invalid_ensemble_count"] = sum(
                not bool(row["valid"])
                for row in (anti_rows if arm == "antithetic" else independent_rows)
            )
            summaries[arm]["ensemble_severe_tail_count"] = sum(
                bool(row["severe_score_tail"])
                for row in (anti_rows if arm == "antithetic" else independent_rows)
            )
            summaries[arm]["rmse_to_sgqf_approximation"] = (
                {
                    label: math.sqrt(
                        statistics.fmean(
                            (vector[index] - sgqf[index]) ** 2 for vector in vectors
                        )
                    )
                    for index, label in enumerate(LABELS)
                }
                if vectors
                else {label: None for label in LABELS}
            )
        by_k[str(k)] = {
            "complete_runs_per_estimator": 2 * k,
            "replicate_count": len(anti_rows),
            "paired_valid_replicate_count": len(paired_valid),
            "summaries": summaries,
            "variance_comparison": _bootstrap_log_ratio(
                anti_vectors,
                independent_vectors,
                [float(row["runtime_seconds"]) for row, _ in paired_valid],
                [float(row["runtime_seconds"]) for _, row in paired_valid],
            ),
            "paired_antithetic_minus_independent": (
                {
                    label: _summary(
                        [
                            anti[index] - independent[index]
                            for anti, independent in zip(
                                anti_vectors, independent_vectors
                            )
                        ]
                    )
                    for index, label in enumerate(LABELS)
                }
                if paired_valid
                else {label: None for label in LABELS}
            ),
        }

    positive: list[list[float]] = []
    negative: list[list[float]] = []
    grouped: dict[tuple[int, int], dict[int, Mapping[str, Any]]] = {}
    for row in constituents:
        if row["arm"] != "antithetic":
            continue
        pair_index = int(row["slot"]) // 2
        grouped.setdefault((int(row["replicate"]), pair_index), {})[
            int(row["sign"])
        ] = row
    for pair in grouped.values():
        if set(pair) != {-1, 1}:
            raise ValueError("antithetic pair is incomplete")
        if bool(pair[1]["valid"]) and bool(pair[-1]["valid"]):
            positive.append(_vector(pair[1]))
            negative.append(_vector(pair[-1]))
    pair_correlations = {
        label: _correlation(
            [row[index] for row in positive], [row[index] for row in negative]
        )
        for index, label in enumerate(LABELS)
    }
    primary = by_k[str(MAX_K)]["variance_comparison"]
    all_constituents_valid = all(bool(row["valid"]) for row in constituents)
    return {
        "by_K": by_k,
        "primary_K": MAX_K,
        "primary_coordinate_nominations": {
            row["label"]: bool(row["coordinate_nominated"]) for row in primary
        },
        "any_primary_coordinate_nominated": all_constituents_valid and any(
            bool(row["coordinate_nominated"]) for row in primary
        ),
        "all_constituents_valid": all_constituents_valid,
        "antithetic_pair_correlations": pair_correlations,
        "constituent_severe_tail_counts": {
            "antithetic": sum(
                bool(row["severe_score_tail"])
                for row in constituents
                if row["arm"] == "antithetic"
            ),
            "independent": sum(
                bool(row["severe_score_tail"])
                for row in constituents
                if row["arm"] == "independent"
            ),
        },
        "sgqf_role": "same_target_deterministic_approximation_explanatory_only",
        "sgqf_value_and_score": list(sgqf),
        "accuracy_ranking_supported": False,
    }


def _select_prior_genut_row(prior: Mapping[str, Any]) -> Mapping[str, Any]:
    if prior.get("row_id") != MODEL_ID:
        raise ValueError("prior Austria tuning artifact has the wrong row")
    genut_rows = [
        row for row in prior.get("rows", []) if row.get("method") == "genut"
    ]
    if len(genut_rows) != 1:
        raise ValueError("prior Austria tuning artifact must contain one GenUT row")
    return genut_rows[0]


def _load_scope() -> tuple[dict[str, Any], dict[str, Any], list[float]]:
    prior = json.loads((ROOT / PRIOR_TUNING_ARTIFACT).read_text(encoding="utf-8"))
    prior_row = _select_prior_genut_row(prior)
    controls = prior_row.get("controls")
    if controls != EXPECTED_CONTROLS:
        raise ValueError("prior Austria controls do not match the frozen scope")
    scope = prior_row.get("scope", {})
    if scope.get("source_observation_sha256") != EXPECTED_OBSERVATION_SHA256:
        raise ValueError("prior Austria observation hash mismatch")
    sgqf_payload = json.loads((ROOT / SGQF_ARTIFACT).read_text(encoding="utf-8"))
    if (
        sgqf_payload.get("dataset", {}).get("observation_sha256")
        != EXPECTED_OBSERVATION_SHA256
    ):
        raise ValueError("SGQF observation hash mismatch")
    sgqf = [
        float(sgqf_payload["compiled_value"][0]),
        *[float(value) for value in sgqf_payload["compiled_score"][0]],
    ]
    return prior_row, scope, sgqf


def _issue_current_identity(
    target: Mapping[str, Any], controls: Mapping[str, Any]
) -> Mapping[str, Any]:
    from bayesfilter.highdim.cubature_genut_candidate import (
        CandidateRouteScope,
        issue_repository_candidate_route_identity,
        validate_repository_candidate_route_identity,
    )

    identity = issue_repository_candidate_route_identity(
        CandidateRouteScope(
            model_id=MODEL_ID,
            target_id=MODEL_ID,
            horizon=HORIZON,
            particle_count=PARTICLE_COUNT,
            state_dimension=STATE_DIMENSION,
            parameter_count=PARAMETER_DIMENSION,
            dtype="float32",
            tf32_enabled=True,
            jit_compile=True,
            design_family="cubature",
            control_family_id="higher_moment_contract_e_candidate_v1",
        ),
        prepared_data_id=str(target["source_observation_sha256"]),
        residual_design_id=f"fixed_cubature_candidate_n{PARTICLE_COUNT}",
        controls={key: str(value) for key, value in controls.items()},
        adapter_id="parameterized_austria_sir_v1",
    )
    validate_repository_candidate_route_identity(identity)
    return identity.to_dict()


def _validate_tuning_payload(
    payload: Mapping[str, Any],
    *,
    expected_identity: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    if payload.get("schema") != f"{SCHEMA}.tuning.v1":
        raise ValueError("current-source tuning artifact has the wrong schema")
    scope = payload.get("scope", {})
    expected_scope = {
        "model_id": MODEL_ID,
        "source_observation_sha256": EXPECTED_OBSERVATION_SHA256,
        "particle_count": PARTICLE_COUNT,
        "horizon": HORIZON,
        "dtype": "float32",
        "tf32_enabled": True,
        "jit_compile": True,
    }
    for key, expected in expected_scope.items():
        if scope.get(key) != expected:
            raise ValueError(f"current-source tuning scope mismatch: {key}")
    selected = payload.get("selected_controls")
    if selected not in CURRENT_CONTROLS_GRID:
        raise ValueError("selected controls are outside the frozen tuning grid")
    for key, expected in ZERO_EXTENSION_CONTROLS.items():
        if selected.get(key) != expected:
            raise ValueError(f"optional correction was not frozen to zero: {key}")
    if payload.get("claim_data_read_during_selection") is not False:
        raise ValueError("tuning artifact does not establish claim-data separation")
    identity = payload.get("route_identity", {})
    if expected_identity is not None and (
        identity.get("identity_sha256")
        != expected_identity.get("identity_sha256")
    ):
        raise ValueError("tuning identity does not match the current callable closure")
    return selected


def _current_source_tuning(
    target: Mapping[str, Any],
) -> Mapping[str, Any]:
    from docs.benchmarks.run_moment_retuned_genut_whole_leaderboard import (
        _evaluate as leaderboard_evaluate,
        _valid as leaderboard_valid,
    )

    candidates = []
    all_rows = []
    for candidate_index, controls in enumerate(CURRENT_CONTROLS_GRID):
        evaluator = _make_current_evaluator(target, controls)
        objectives = {}
        variance_objectives = {}
        eligible = True
        candidate_rows = []
        for partition in ("calibration", "validation"):
            partition_rows = []
            dataset_variances = []
            for dataset_index, observations in enumerate(target[partition]):
                rows = []
                for seed in TUNING_SEEDS:
                    row = leaderboard_evaluate(
                        evaluator,
                        target["theta"],
                        tf.cast(observations, tf.float32),
                        seed,
                        target["design"],
                    )
                    tagged = {
                        **row,
                        "candidate_index": candidate_index,
                        "partition": partition,
                        "dataset_index": dataset_index,
                    }
                    rows.append(tagged)
                    candidate_rows.append(tagged)
                    partition_rows.append(tagged)
                    all_rows.append(tagged)
                eligible = eligible and all(leaderboard_valid(row) for row in rows)
                if all(row["finite"] for row in rows):
                    vectors = [
                        [
                            float(row["value"]) / HORIZON,
                            *[
                                float(value) / math.sqrt(HORIZON)
                                for value in row["score"]
                            ],
                        ]
                        for row in rows
                    ]
                    dataset_variances.append(
                        max(
                            statistics.variance(
                                vector[index] for vector in vectors
                            )
                            for index in range(len(vectors[0]))
                        )
                    )
            finite_rows = [row for row in partition_rows if row["finite"]]
            objectives[partition] = (
                statistics.fmean(
                    float(row["mean_normalized_shape_residual_objective"])
                    for row in finite_rows
                )
                if len(finite_rows) == len(partition_rows) and partition_rows
                else None
            )
            variance_objectives[partition] = (
                statistics.fmean(dataset_variances)
                if len(dataset_variances) == len(target[partition])
                else None
            )
        candidates.append(
            {
                "candidate_index": candidate_index,
                "controls": controls,
                "objectives": objectives,
                "variance_objectives": variance_objectives,
                "eligible": eligible,
                "row_count": len(candidate_rows),
            }
        )
    eligible_candidates = [row for row in candidates if row["eligible"]]
    if not eligible_candidates:
        raise RuntimeError("no eligible current-source Austria GenUT controls")
    selected = min(
        eligible_candidates,
        key=lambda row: (
            row["objectives"]["validation"],
            row["objectives"]["calibration"],
            row["variance_objectives"]["validation"],
        ),
    )
    return {
        "selected_controls": dict(selected["controls"]),
        "selected_candidate_index": selected["candidate_index"],
        "selection_objective": (
            "validation_diagonal_moment_objective_then_calibration_then_"
            "validation_scaled_conditional_variance"
        ),
        "candidates": candidates,
        "rows": all_rows,
        "claim_data_read_during_selection": False,
    }


def _finite_difference_audit(
    evaluator: Any,
    theta: tf.Tensor,
    observations: tf.Tensor,
    design: tf.Tensor,
    *,
    root_seed: int,
) -> Mapping[str, Any]:
    initial, process = _noise(root_seed)
    base_rows = []
    for sign in (1, -1):
        row = _evaluate_constituent(
            evaluator,
            theta,
            observations,
            sign * initial,
            sign * process,
            design,
            seed=root_seed,
            arm="antithetic",
            sign=sign,
            replicate=-1,
            slot=0 if sign == 1 else 1,
        )
        if not row["valid"]:
            return {
                "root_seed": root_seed,
                "steps": [],
                "maximum_relative_error": None,
                "tolerance": FD_RELATIVE_TOLERANCE,
                "pass": False,
                "failure_reason": "INVALID_BASE_CONSTITUENT",
                "semantics": (
                    "same_fixed_noise_antithetic_average_scalar_central_difference"
                ),
            }
        base_rows.append(row)
    base_score = _mean_vectors(base_rows)[1:]
    step_rows = []
    maximum_relative_error = 0.0
    for relative_step, minimum_step in zip(
        FD_RELATIVE_STEPS, FD_MINIMUM_STEPS
    ):
        finite_difference = []
        realized_steps = []
        for parameter in range(PARAMETER_DIMENSION):
            step = max(
                minimum_step,
                relative_step * abs(float(theta[parameter].numpy())),
            )
            realized_steps.append(step)
            direction = tf.one_hot(parameter, PARAMETER_DIMENSION, dtype=tf.float32)
            endpoint_values = []
            for endpoint in (1, -1):
                endpoint_theta = theta + endpoint * step * direction
                signed_values = []
                for sign in (1, -1):
                    value, _, _ = evaluator(
                        endpoint_theta,
                        observations,
                        sign * initial,
                        sign * process,
                        design,
                    )
                    signed_values.append(float(value.numpy()))
                endpoint_values.append(statistics.fmean(signed_values))
            finite_difference.append(
                (endpoint_values[0] - endpoint_values[1]) / (2.0 * step)
            )
        relative_errors = [
            abs(fd - score) / max(abs(fd), abs(score), 1.0e-2)
            for fd, score in zip(finite_difference, base_score)
        ]
        maximum_relative_error = max(maximum_relative_error, *relative_errors)
        step_rows.append(
            {
                "relative_step": relative_step,
                "minimum_step": minimum_step,
                "realized_steps": realized_steps,
                "finite_difference": finite_difference,
                "recursive_antithetic_score": base_score,
                "relative_errors": relative_errors,
                "maximum_relative_error": max(relative_errors),
            }
        )
    return {
        "root_seed": root_seed,
        "steps": step_rows,
        "maximum_relative_error": maximum_relative_error,
        "tolerance": FD_RELATIVE_TOLERANCE,
        "pass": maximum_relative_error < FD_RELATIVE_TOLERANCE,
        "semantics": "same_fixed_noise_antithetic_average_scalar_central_difference",
    }


def _render(payload: Mapping[str, Any]) -> str:
    analysis = payload["analysis"]
    lines = [
        "# GenUT Austria SIR Antithetic-Ensemble Result",
        "",
        f"Status: `{payload['decision']['status']}`",
        "",
        "The comparison uses the same number of complete GenUT evaluations in both",
        "arms. SGQF is an explanatory approximation, not truth.",
        "",
    ]
    for k in K_VALUES:
        row = analysis["by_K"][str(k)]
        lines.extend(
            [
                f"## K={k}",
                "",
                "| Coordinate | Anti variance | Independent variance | Ratio | Log-ratio interval | Nominated |",
                "|---|---:|---:|---:|---:|---|",
            ]
        )
        for comparison in row["variance_comparison"]:
            interval = (
                comparison["familywise_95_log_ratio_interval"]
                if k == MAX_K
                else comparison["pointwise_95_log_ratio_interval"]
            )
            lines.append(
                f"| {comparison['label']} | {comparison['antithetic_variance']:.6g} | "
                f"{comparison['independent_variance']:.6g} | "
                f"{comparison['variance_ratio_antithetic_over_independent']:.6g} | "
                f"[{interval['lower']:.4g}, {interval['upper']:.4g}] | "
                f"{comparison['coordinate_nominated'] if k == MAX_K else 'explanatory'} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Decision",
            "",
            payload["decision"]["text"],
            "",
            "No score-accuracy ranking is supported because Austria has no exact",
            "`T=20` likelihood/score oracle in this experiment.",
        ]
    )
    return "\n".join(lines) + "\n"


def _initialize_gpu() -> tuple[Mapping[str, Any], Sequence[Any]]:
    from bayesfilter.runtime.gpu_memory_policy import (
        configure_tensorflow_gpu_memory_growth,
    )

    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    logical_devices = tf.config.list_logical_devices("GPU")
    if not logical_devices:
        raise RuntimeError("Austria antithetic campaign requires a logical GPU")
    return memory_policy, logical_devices


def run_tuning(output_root: Path) -> Mapping[str, Any]:
    started = time.perf_counter()
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    output_root.mkdir(parents=True, exist_ok=False)
    memory_policy, logical_devices = _initialize_gpu()

    from docs.benchmarks.run_moment_retuned_genut_whole_leaderboard import (
        _build_targets,
    )

    target = _build_targets()[MODEL_ID]
    if target["source_observation_sha256"] != EXPECTED_OBSERVATION_SHA256:
        raise ValueError("runtime Austria observation hash mismatch")
    tuning = _current_source_tuning(target)
    rows = tuning.pop("rows")
    controls = tuning["selected_controls"]
    route_identity = _issue_current_identity(target, controls)
    allocator = tf.config.experimental.get_memory_info("GPU:0")
    source_paths = (
        Path(__file__).relative_to(ROOT),
        PLAN,
        Path("bayesfilter/highdim/cubature_genut_filter.py"),
        Path("bayesfilter/highdim/cubature_genut_adapters.py"),
        Path("bayesfilter/highdim/higher_moment_contract_e.py"),
        Path("bayesfilter/highdim/cubature_genut_candidate.py"),
    )
    raw_path = output_root / "tuning_rows.json"
    _write_json(raw_path, {"rows": rows})
    result_path = output_root / "result.json"
    payload: dict[str, Any] = {
        "schema": f"{SCHEMA}.tuning.v1",
        "campaign_id": CAMPAIGN_ID,
        "status": "CURRENT_SOURCE_AUSTRIA_TUNING_COMPLETE",
        "started_utc": started_utc,
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_time_seconds": time.perf_counter() - started,
        "git_commit": _git_commit(),
        "host": platform.node(),
        "tensorflow_version": tf.__version__,
        "plan": PLAN.as_posix(),
        "scope": {
            "model_id": MODEL_ID,
            "source_observation_sha256": EXPECTED_OBSERVATION_SHA256,
            "runtime_observation_sha256": _tensor_sha256(target["observations"]),
            "particle_count": PARTICLE_COUNT,
            "horizon": HORIZON,
            "state_dimension": STATE_DIMENSION,
            "parameter_dimension": PARAMETER_DIMENSION,
            "dtype": "float32",
            "tf32_enabled": True,
            "jit_compile": True,
            "event_order": target["event_order"],
        },
        **tuning,
        "tuning_seeds": TUNING_SEEDS,
        "calibration_dataset_count": len(target["calibration"]),
        "validation_dataset_count": len(target["validation"]),
        "route_identity": route_identity,
        "device": {
            "logical_devices": [device.name for device in logical_devices],
            "dtype": "float32",
            "tf32_enabled": bool(
                tf.config.experimental.tensor_float_32_execution_enabled()
            ),
            "jit_compile": True,
            "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        },
        "memory_policy": memory_policy,
        "gpu_allocator": {key: int(value) for key, value in allocator.items()},
        "source_sha256": {
            path.as_posix(): _sha256(ROOT / path) for path in source_paths
        },
        "artifact_paths": {
            "result": result_path.as_posix(),
            "tuning_rows": raw_path.as_posix(),
        },
        "nonclaims": [
            "scope-specific tuning is not an antithetic result",
            "selection does not establish exact likelihood or score accuracy",
            "pairwise and projected-cumulant corrections are frozen off",
        ],
    }
    _validate_tuning_payload(payload, expected_identity=route_identity)
    _write_json(result_path, payload)
    manifest = {
        "schema": "bayesfilter.serious_run_manifest.v1",
        "git_commit": payload["git_commit"],
        "command": " ".join(sys.argv),
        "environment": sys.executable,
        "cpu_gpu_status": payload["device"],
        "memory_policy": memory_policy,
        "data_version": EXPECTED_OBSERVATION_SHA256,
        "random_seeds": {"tuning": TUNING_SEEDS},
        "wall_time_seconds": payload["wall_time_seconds"],
        "output_artifact_paths": list(payload["artifact_paths"].values()),
        "plan_file": PLAN.as_posix(),
        "result_file": result_path.as_posix(),
        "artifact_sha256": {
            path.name: _sha256(path) for path in (raw_path, result_path)
        },
    }
    _write_json(output_root / "run_manifest.json", manifest)
    return payload


def run(
    output_root: Path, *, replicates: int, tuning_artifact: Path
) -> Mapping[str, Any]:
    if replicates < 1 or replicates > 16:
        raise ValueError("replicates must be between 1 and 16")
    started = time.perf_counter()
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    output_root.mkdir(parents=True, exist_ok=False)

    memory_policy, logical_devices = _initialize_gpu()
    _, _, sgqf = _load_scope()
    from docs.benchmarks.run_moment_retuned_genut_whole_leaderboard import (
        _build_targets,
    )

    target = _build_targets()[MODEL_ID]
    if target["source_observation_sha256"] != EXPECTED_OBSERVATION_SHA256:
        raise ValueError("runtime Austria observation hash mismatch")
    tuning_path = tuning_artifact if tuning_artifact.is_absolute() else ROOT / tuning_artifact
    tuning_payload = json.loads(tuning_path.read_text(encoding="utf-8"))
    preliminary_controls = tuning_payload.get("selected_controls", {})
    current_identity_payload = _issue_current_identity(
        target, preliminary_controls
    )
    controls = dict(
        _validate_tuning_payload(
            tuning_payload, expected_identity=current_identity_payload
        )
    )
    evaluator = _make_current_evaluator(target, controls)

    warm_initial, warm_process = _noise(139999)
    warm_value, warm_score, _ = evaluator(
        target["theta"],
        target["observations"],
        warm_initial,
        warm_process,
        target["design"],
    )
    if "GPU" not in str(warm_value.device).upper() or not bool(
        tf.reduce_all(tf.math.is_finite(warm_score)).numpy()
    ):
        raise RuntimeError("GPU/XLA warmup did not produce a finite GPU result")

    constituents: list[dict[str, Any]] = []
    reflection_audit = []
    for replicate in range(replicates):
        anti_seeds = [140000 + 100 * replicate + index for index in range(MAX_K)]
        independent_seeds = [
            240000 + 100 * replicate + index for index in range(2 * MAX_K)
        ]
        anti_noise = []
        for seed in anti_seeds:
            initial, process = _noise(seed)
            exact_reflection = bool(
                tf.reduce_all(tf.equal(-initial, tf.negative(initial))).numpy()
            ) and bool(tf.reduce_all(tf.equal(-process, tf.negative(process))).numpy())
            if not exact_reflection:
                raise RuntimeError("TensorFlow tensor-negation audit failed")
            reflection_audit.append(
                {
                    "replicate": replicate,
                    "root_seed": seed,
                    "initial_sha256": _tensor_sha256(initial),
                    "process_sha256": _tensor_sha256(process),
                    "negative_initial_sha256": _tensor_sha256(-initial),
                    "negative_process_sha256": _tensor_sha256(-process),
                    "exact_tensor_negation": exact_reflection,
                }
            )
            anti_noise.extend([(seed, 1, initial, process), (seed, -1, -initial, -process)])
        independent_noise = [
            (seed, *_noise(seed)) for seed in independent_seeds
        ]

        # Alternate arm order by replicate to avoid a systematic timing-order bias.
        arm_order = (
            ("antithetic", "independent")
            if replicate % 2 == 0
            else ("independent", "antithetic")
        )
        for slot in range(2 * MAX_K):
            for arm in arm_order:
                if arm == "antithetic":
                    seed, sign, initial, process = anti_noise[slot]
                else:
                    seed, initial, process = independent_noise[slot]
                    sign = 1
                row = _evaluate_constituent(
                    evaluator,
                    target["theta"],
                    target["observations"],
                    initial,
                    process,
                    target["design"],
                    seed=seed,
                    arm=arm,
                    sign=sign,
                    replicate=replicate,
                    slot=slot,
                )
                constituents.append(row)

    ensembles = _assemble_ensembles(constituents)
    finite_difference = _finite_difference_audit(
        evaluator,
        target["theta"],
        target["observations"],
        target["design"],
        root_seed=140000,
    )
    analysis = _analysis(ensembles, constituents, sgqf)
    allocator = tf.config.experimental.get_memory_info("GPU:0")
    hard_valid = bool(
        analysis["all_constituents_valid"] and finite_difference["pass"]
    )
    primary_available = replicates > 1 and hard_valid
    any_nomination = bool(
        primary_available and analysis["any_primary_coordinate_nominated"]
    )
    if not hard_valid:
        status = "ANTITHETIC_PROMOTION_VETO_INVALID_OR_DERIVATIVE_FAILURE"
        decision_text = (
            "At least one constituent or the same-scalar derivative audit failed; "
            "no variance promotion is permitted, but all completed rows are retained."
        )
    elif replicates == 1:
        status = "PASS_GPU_XLA_SMOKE_NO_STATISTICAL_CLAIM"
        decision_text = "The one-replicate smoke passed; it carries no variance comparison."
    elif any_nomination:
        status = "ANTITHETIC_PARTIAL_COORDINATE_NOMINATION_FEASIBILITY_ONLY"
        decision_text = (
            "At least one K=4 coordinate passed the equal-cost familywise variance "
            "screen. Accuracy and default/HMC promotion remain unsupported."
        )
    else:
        status = "ANTITHETIC_NOT_NOMINATED_AGAINST_EQUAL_COST_BASELINE"
        decision_text = (
            "No K=4 coordinate passed the equal-cost familywise variance screen; "
            "the antithetic ensemble is not nominated for Austria GenUT."
        )

    source_paths = (
        Path(__file__).relative_to(ROOT),
        PLAN,
        Path("bayesfilter/highdim/cubature_genut_filter.py"),
        Path("bayesfilter/highdim/cubature_genut_adapters.py"),
        Path("bayesfilter/highdim/higher_moment_contract_e.py"),
    )
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "status": status,
        "started_utc": started_utc,
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_time_seconds": time.perf_counter() - started,
        "git_commit": _git_commit(),
        "working_tree_note": "source hashes bind the dirty working-tree implementation",
        "host": platform.node(),
        "python": sys.version,
        "tensorflow_version": tf.__version__,
        "plan": PLAN.as_posix(),
        "source_sha256": {
            path.as_posix(): _sha256(ROOT / path) for path in source_paths
        },
        "device": {
            "logical_devices": [device.name for device in logical_devices],
            "warmup_output_device": str(warm_value.device),
            "dtype": "float32",
            "tf32_enabled": bool(
                tf.config.experimental.tensor_float_32_execution_enabled()
            ),
            "jit_compile": True,
            "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        },
        "memory_policy": memory_policy,
        "gpu_allocator": {key: int(value) for key, value in allocator.items()},
        "scope": {
            "model_id": MODEL_ID,
            "horizon": HORIZON,
            "particle_count": PARTICLE_COUNT,
            "state_dimension": STATE_DIMENSION,
            "parameter_dimension": PARAMETER_DIMENSION,
            "theta": [float(value) for value in target["theta"].numpy().tolist()],
            "source_observation_sha256": target["source_observation_sha256"],
            "runtime_observation_sha256": _tensor_sha256(target["observations"]),
            "event_order": target["event_order"],
            "controls": controls,
            "tuning_artifact": str(tuning_path.relative_to(ROOT)),
            "tuning_artifact_sha256": _sha256(tuning_path),
            "current_repository_issued_route_identity": current_identity_payload,
        },
        "configuration": {
            "replicates": replicates,
            "K_values": K_VALUES,
            "primary_K": MAX_K,
            "complete_runs_per_primary_estimator": 2 * MAX_K,
            "baseline": "mean_of_2K_mutually_independent_complete_GenUT_runs",
            "candidate": "mean_of_K_complete_GenUT_Z_minus_Z_pairs",
            "score": "exact_derivative_of_each_ensembles_own_fixed_noise_GenUT_scalar",
            "antithetic_seed_rule": "140000 + 100*replicate + pair_index",
            "independent_seed_rule": "240000 + 100*replicate + constituent_index",
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "primary_interval": "paired_percentile_bootstrap_Bonferroni_familywise_95",
            "severe_score_threshold": SEVERE_SCORE_THRESHOLD,
        },
        "reflection_audit": reflection_audit,
        "finite_difference_audit": finite_difference,
        "analysis": analysis,
        "hard_valid": hard_valid,
        "decision": {
            "status": status,
            "text": decision_text,
            "default_changed": False,
            "accuracy_ranking_supported": False,
        },
        "inference_status": {
            "hard_veto_screen": "pass" if hard_valid else "failed",
            "statistically_supported_ranking": (
                "coordinate_specific_K4_variance_nomination_only"
                if any_nomination
                else "none"
            ),
            "descriptive_only_differences": (
                "K1_K2_all_means_SGQF_gaps_pair_correlations_tails_runtime"
            ),
            "default_readiness": "not_established",
            "next_evidence_needed": (
                "independent_dataset_replication_and_valid_same_event_order_accuracy_reference"
            ),
        },
        "nonclaims": [
            "SGQF is not an exact Austria value or score oracle",
            "the event-order-mismatched online SIR teacher was not used",
            "no unbiasedness or physical-score agreement claim",
            "no HMC efficiency or posterior-correctness claim",
            "no broad nonlinear-model or default-promotion claim",
        ],
        "post_run_red_team": {
            "strongest_alternative_explanation": (
                "a favorable variance ratio on one frozen dataset may be a seed-specific "
                "coupling effect and can coexist with finite-filter bias"
            ),
            "result_that_would_overturn_decision": (
                "fresh independent Austria datasets with familywise variance benefit and "
                "agreement against a validated same-event-order accuracy reference"
            ),
            "weakest_evidence": "score_accuracy_without_an_exact_or_converged_reference",
        },
    }
    raw_path = output_root / "raw_constituents.json"
    ensemble_path = output_root / "ensembles.json"
    _write_json(raw_path, {"constituents": constituents})
    _write_json(ensemble_path, {"ensembles": ensembles})
    result_path = output_root / "result.json"
    payload["artifact_paths"] = {
        "result": result_path.as_posix(),
        "raw_constituents": raw_path.as_posix(),
        "ensembles": ensemble_path.as_posix(),
        "markdown": (output_root / "result.md").as_posix(),
    }
    _write_json(result_path, payload)
    (output_root / "result.md").write_text(_render(payload), encoding="utf-8")
    manifest = {
        "schema": "bayesfilter.serious_run_manifest.v1",
        "git_commit": payload["git_commit"],
        "command": " ".join(sys.argv),
        "environment": sys.executable,
        "cpu_gpu_status": payload["device"],
        "memory_policy": memory_policy,
        "data_version": EXPECTED_OBSERVATION_SHA256,
        "random_seeds": {
            "antithetic_rule": payload["configuration"]["antithetic_seed_rule"],
            "independent_rule": payload["configuration"]["independent_seed_rule"],
            "bootstrap": BOOTSTRAP_SEED,
        },
        "wall_time_seconds": payload["wall_time_seconds"],
        "output_artifact_paths": list(payload["artifact_paths"].values()),
        "plan_file": PLAN.as_posix(),
        "result_file": result_path.as_posix(),
        "artifact_sha256": {
            path.name: _sha256(path)
            for path in (raw_path, ensemble_path, result_path, output_root / "result.md")
        },
    }
    _write_json(output_root / "run_manifest.json", manifest)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=16)
    parser.add_argument("--tune-only", action="store_true")
    parser.add_argument("--tuning-artifact", type=Path)
    args = parser.parse_args()
    if not args.tune_only and args.tuning_artifact is None:
        parser.error("--tuning-artifact is required unless --tune-only is used")
    if args.tune_only and args.tuning_artifact is not None:
        parser.error("--tuning-artifact cannot be combined with --tune-only")
    output_existed_before_run = args.output_root.exists()
    try:
        result = (
            run_tuning(args.output_root)
            if args.tune_only
            else run(
                args.output_root,
                replicates=args.replicates,
                tuning_artifact=args.tuning_artifact,
            )
        )
    except Exception as exc:
        if not output_existed_before_run and args.output_root.exists():
            _write_json(
                args.output_root / "failure.json",
                {
                    "schema": "bayesfilter.genut_austria_antithetic_failure.v1",
                    "status": "CAMPAIGN_FAILED",
                    "exception_type": f"{type(exc).__module__}.{type(exc).__name__}",
                    "exception_message": str(exc),
                    "traceback": traceback.format_exc(),
                    "plan": PLAN.as_posix(),
                    "scientific_interpretation": "none_incomplete_or_vetoed_run",
                },
            )
        raise
    print(
        json.dumps(
            {
                "status": result["status"],
                "wall_time_seconds": result["wall_time_seconds"],
                "artifact": result["artifact_paths"]["result"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
