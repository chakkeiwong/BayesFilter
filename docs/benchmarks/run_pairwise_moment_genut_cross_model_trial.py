#!/usr/bin/env python3
"""Tune and test pairwise-moment GenUT on LGSSM, KSC-SV, and predator-prey."""

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
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
from docs.benchmarks import run_moment_retuned_genut_whole_leaderboard as base


PLAN = Path(
    "docs/plans/bayesfilter-pairwise-moment-genut-lgssm-ksc-predator-prey-"
    "trial-plan-2026-07-30.md"
)
PRIOR_ARTIFACT = Path(
    "docs/benchmarks/artifacts/moment_retuned_genut_whole_leaderboard_20260723/"
    "attempt05_final/result.json"
)
SCHEMA = "bayesfilter.pairwise_moment_genut_cross_model_trial.v1"
MODEL_IDS = ("lgssm_T50", "ksc_sv_T10", "predator_prey_T20")
CLAIM_SEEDS = tuple(range(98201, 98217))
TUNING_SEEDS = (98401, 98402)
PAIRWISE_ARMS = tuple(
    (steps, strength)
    for steps in (1, 2, 4)
    for strength in (0.005, 0.01, 0.02, 0.05)
)
PAIRWISE_FLOOR = 1.0e-5


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe(value: Any) -> Any:
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _prior_rows() -> list[dict[str, Any]]:
    return json.loads((ROOT / PRIOR_ARTIFACT).read_text(encoding="utf-8"))["rows"]


def _prior_row(row_id: str, method: str) -> dict[str, Any]:
    matches = [
        row
        for row in _prior_rows()
        if row["row_id"] == row_id and row["method"] == method
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one prior {row_id}/{method} row")
    return matches[0]


def _baseline_controls(row_id: str, target: dict[str, Any]) -> dict[str, Any]:
    row = _prior_row(row_id, "genut")
    if row["scope"]["source_observation_sha256"] != target["source_observation_sha256"]:
        raise RuntimeError(f"{row_id} prior target hash mismatch")
    if row["scope"]["event_order"] != target["event_order"]:
        raise RuntimeError(f"{row_id} prior event-order mismatch")
    if row["scope"]["particle_count"] != base.N:
        raise RuntimeError(f"{row_id} prior particle-count mismatch")
    controls = dict(row["tuning"]["selected_controls"])
    controls.update(
        {
            "pairwise_moment_correction_steps": 0,
            "pairwise_moment_strength": 0.0,
            "pairwise_moment_floor": PAIRWISE_FLOOR,
        }
    )
    return controls


def _candidate_controls(
    baseline: dict[str, Any], steps: int, strength: float
) -> dict[str, Any]:
    return {
        **baseline,
        "pairwise_moment_correction_steps": steps,
        "pairwise_moment_strength": strength,
        "pairwise_moment_floor": PAIRWISE_FLOOR,
    }


def _evaluator(target: dict[str, Any], controls: dict[str, Any]):
    return base._make_evaluator(
        adapter=target["adapter"],
        horizon=int(target["observations"].shape[0]),
        observation_dim=target["observation_dim"],
        state_dim=target["state_dim"],
        parameter_dim=target["parameter_dim"],
        transition_before_first_observation=target["transition_before"],
        controls=controls,
    )


def _row(
    evaluator: Any,
    target: dict[str, Any],
    observations: tf.Tensor,
    seed: int,
) -> dict[str, Any]:
    return base._evaluate(
        evaluator,
        target["theta"],
        tf.cast(observations, tf.float32),
        seed,
        target["design"],
    )


def _coordinate_variances(rows: list[dict[str, Any]]) -> list[float]:
    vectors = [[row["value"], *row["score"]] for row in rows]
    return [
        statistics.variance(vector[index] for vector in vectors)
        for index in range(len(vectors[0]))
    ]


def _partition(
    evaluator: Any,
    target: dict[str, Any],
    datasets: list[tf.Tensor],
) -> dict[str, Any]:
    all_rows: list[dict[str, Any]] = []
    per_dataset_variances: list[list[float]] = []
    for dataset in datasets:
        rows = [_row(evaluator, target, dataset, seed) for seed in TUNING_SEEDS]
        all_rows.extend(rows)
        if all(base._valid(row) for row in rows):
            per_dataset_variances.append(_coordinate_variances(rows))
    valid = bool(all_rows) and all(base._valid(row) for row in all_rows)
    variances = None
    if valid and len(per_dataset_variances) == len(datasets):
        variances = [
            statistics.mean(row[index] for row in per_dataset_variances)
            for index in range(len(per_dataset_variances[0]))
        ]
    return {
        "valid": valid,
        "rows": all_rows,
        "coordinate_variances": variances,
        "mean_pairwise_objective": (
            statistics.mean(
                row["mean_normalized_pairwise_shape_residual_objective"]
                for row in all_rows
            )
            if valid
            else None
        ),
        "mean_diagonal_objective": (
            statistics.mean(
                row["mean_normalized_shape_residual_objective"] for row in all_rows
            )
            if valid
            else None
        ),
        "maximum_displacement": (
            max(row["maximum_normalized_shape_displacement"] for row in all_rows)
            if valid
            else None
        ),
    }


def _variance_ratios(
    candidate: list[float], baseline: list[float]
) -> list[float]:
    tiny = 1.0e-20
    return [
        max(candidate[index], tiny) / max(baseline[index], tiny)
        for index in range(1, len(baseline))
    ]


def _tune(
    row_id: str, target: dict[str, Any], baseline_controls: dict[str, Any]
) -> dict[str, Any]:
    grid = [baseline_controls] + [
        _candidate_controls(baseline_controls, steps, strength)
        for steps, strength in PAIRWISE_ARMS
    ]
    candidates = []
    for controls in grid:
        evaluator = _evaluator(target, controls)
        candidates.append(
            {
                "controls": controls,
                "calibration": _partition(evaluator, target, target["calibration"]),
                "validation": _partition(evaluator, target, target["validation"]),
            }
        )
    baseline = candidates[0]
    baseline_validation = baseline["validation"]
    if not baseline_validation["valid"]:
        raise RuntimeError(f"{row_id} diagonal-only validation baseline invalid")
    baseline_variances = baseline_validation["coordinate_variances"]
    baseline_pairwise = baseline_validation["mean_pairwise_objective"]
    assert baseline_variances is not None and baseline_pairwise is not None
    baseline_value_sd = math.sqrt(baseline_variances[0])

    for candidate in candidates:
        validation = candidate["validation"]
        variances = validation["coordinate_variances"]
        ratios = (
            _variance_ratios(variances, baseline_variances)
            if variances is not None
            else None
        )
        residual_improved = bool(
            validation["valid"]
            and validation["mean_pairwise_objective"] is not None
            and validation["mean_pairwise_objective"] < baseline_pairwise
        )
        score_variance_veto = bool(
            ratios is None or any(ratio > 1.0 for ratio in ratios)
        )
        value_variance_veto = bool(
            variances is None
            or math.sqrt(variances[0]) > 1.25 * baseline_value_sd
        )
        candidate["selection_diagnostics"] = {
            "pairwise_residual_improved": residual_improved,
            "score_variance_ratios": ratios,
            "score_variance_veto": score_variance_veto,
            "value_variance_veto": value_variance_veto,
            "eligible": bool(
                validation["valid"]
                and residual_improved
                and not score_variance_veto
                and not value_variance_veto
            ),
        }

    eligible = [
        candidate
        for candidate in candidates[1:]
        if candidate["selection_diagnostics"]["eligible"]
    ]
    if eligible:
        selected = min(
            eligible,
            key=lambda candidate: (
                max(candidate["selection_diagnostics"]["score_variance_ratios"]),
                statistics.geometric_mean(
                    candidate["selection_diagnostics"]["score_variance_ratios"]
                ),
                candidate["validation"]["mean_pairwise_objective"],
                candidate["validation"]["maximum_displacement"],
                candidate["controls"]["pairwise_moment_strength"],
                candidate["controls"]["pairwise_moment_correction_steps"],
            ),
        )
        status = "PAIRWISE_CANDIDATE_SELECTED"
    else:
        selected = baseline
        status = "NO_PAIRWISE_CANDIDATE_PASSED_TUNING_VETOES"
    return {
        "scope": row_id,
        "selection_status": status,
        "baseline_controls": baseline_controls,
        "selected_controls": selected["controls"],
        "candidates": candidates,
        "claim_data_read_during_selection": False,
        "reference_or_comparator_read_during_selection": False,
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = ("value", *[f"score_{index}" for index in range(len(rows[0]["score"]))])
    vectors = [[row["value"], *row["score"]] for row in rows]
    output: dict[str, Any] = {
        "count": len(rows),
        "labels": labels,
        "all_valid": all(base._valid(row) for row in rows),
    }
    for index, label in enumerate(labels):
        sample = [vector[index] for vector in vectors]
        mean = statistics.mean(sample)
        sd = statistics.stdev(sample) if len(sample) > 1 else 0.0
        half = 2.131449545559323 * sd / math.sqrt(len(sample)) if len(sample) > 1 else 0.0
        output[label] = {
            "mean": mean,
            "sample_sd": sd,
            "ci95_lower": mean - half,
            "ci95_upper": mean + half,
        }
    output["mean_pairwise_objective"] = statistics.mean(
        row["mean_normalized_pairwise_shape_residual_objective"] for row in rows
    )
    output["mean_diagonal_objective"] = statistics.mean(
        row["mean_normalized_shape_residual_objective"] for row in rows
    )
    return output


def _paired_summary(
    baseline: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> dict[str, Any]:
    labels = ("value", *[f"score_{index}" for index in range(len(baseline[0]["score"]))])
    baseline_vectors = [[row["value"], *row["score"]] for row in baseline]
    candidate_vectors = [[row["value"], *row["score"]] for row in candidate]
    output = {}
    for index, label in enumerate(labels):
        sample = [
            candidate_vectors[row][index] - baseline_vectors[row][index]
            for row in range(len(baseline))
        ]
        mean = statistics.mean(sample)
        sd = statistics.stdev(sample)
        half = 2.131449545559323 * sd / math.sqrt(len(sample))
        output[label] = {
            "mean": mean,
            "sample_sd": sd,
            "ci95_lower": mean - half,
            "ci95_upper": mean + half,
        }
    return output


def _bootstrap_variance_ratio(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    seed: int,
    draws: int = 10000,
) -> dict[str, Any]:
    rng = random.Random(seed)
    baseline_scores = [row["score"] for row in baseline]
    candidate_scores = [row["score"] for row in candidate]

    def ratio(indices: list[int]) -> float:
        values = []
        for coordinate in range(len(baseline_scores[0])):
            baseline_sd = statistics.stdev(
                baseline_scores[index][coordinate] for index in indices
            )
            candidate_sd = statistics.stdev(
                candidate_scores[index][coordinate] for index in indices
            )
            if baseline_sd <= 0.0:
                if candidate_sd <= 0.0:
                    values.append(1.0)
                    continue
                return math.inf
            values.append((candidate_sd / baseline_sd) ** 2)
        return statistics.geometric_mean(values)

    point = ratio(list(range(len(baseline))))
    ratios = []
    while len(ratios) < draws:
        indices = [rng.randrange(len(baseline)) for _ in baseline]
        try:
            value = ratio(indices)
        except statistics.StatisticsError:
            continue
        if math.isfinite(value):
            ratios.append(value)
    ratios.sort()
    return {
        "aggregate_geometric_variance_ratio": point,
        "bootstrap_ci95_lower": ratios[int(0.025 * draws)],
        "bootstrap_ci95_upper": ratios[int(0.975 * draws) - 1],
        "bootstrap_draws": draws,
    }


def _absolute_error_summary(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    reference: dict[str, Any],
) -> dict[str, Any]:
    reference_vector = [reference["value"], *reference["score"]]
    labels = ("value", *[f"score_{index}" for index in range(len(reference["score"]))])
    baseline_vectors = [[row["value"], *row["score"]] for row in baseline]
    candidate_vectors = [[row["value"], *row["score"]] for row in candidate]
    output = {}
    for coordinate, label in enumerate(labels):
        baseline_errors = [
            abs(row[coordinate] - reference_vector[coordinate])
            for row in baseline_vectors
        ]
        candidate_errors = [
            abs(row[coordinate] - reference_vector[coordinate])
            for row in candidate_vectors
        ]
        changes = [
            candidate_errors[index] - baseline_errors[index]
            for index in range(len(baseline_errors))
        ]
        mean = statistics.mean(changes)
        sd = statistics.stdev(changes)
        half = 2.131449545559323 * sd / math.sqrt(len(changes))
        output[label] = {
            "baseline_mean_absolute_error": statistics.mean(baseline_errors),
            "candidate_mean_absolute_error": statistics.mean(candidate_errors),
            "paired_change_mean": mean,
            "paired_change_ci95_lower": mean - half,
            "paired_change_ci95_upper": mean + half,
            "supported_improvement": mean + half < 0.0,
            "supported_regression": mean - half > 0.0,
        }
    return output


def _lgssm_reference(target: dict[str, Any]) -> dict[str, Any]:
    from docs.benchmarks.run_lgssm_cubature_genut_fp32 import _kalman_value_score

    value, score = _kalman_value_score(target["theta"], target["observations"])
    return {
        "kind": "exact_affine_kalman_analytical_score",
        "value": float(value.numpy()),
        "score": [float(item) for item in score.numpy()],
    }


def _ksc_dense_reference(target: dict[str, Any], *, smoke_only: bool) -> dict[str, Any]:
    from bayesfilter.highdim.sv_mixture_cut4 import (
        StochasticVolatilitySSM,
        scalar_sv_mixture_dense_reference,
    )

    model = StochasticVolatilitySSM(sigma=1.0)
    theta = tf.cast(target["theta"], tf.float64)
    raw = tf.cast(target["raw"], tf.float64)

    def value(current: tf.Tensor, order: int) -> float:
        with tf.device("/CPU:0"):
            result = scalar_sv_mixture_dense_reference(
                model, current, raw, order=order, radius=8.0
            )
        return float(result.log_likelihood.numpy())

    if smoke_only:
        return {"kind": "omitted_in_smoke", "value": None, "score": None}
    value_401 = value(theta, 401)
    value_601 = value(theta, 601)

    def score(order: int, step: float) -> list[float]:
        output = []
        for coordinate in range(2):
            basis = tf.one_hot(coordinate, 2, dtype=tf.float64)
            plus = value(theta + tf.constant(step, tf.float64) * basis, order)
            minus = value(theta - tf.constant(step, tf.float64) * basis, order)
            output.append((plus - minus) / (2.0 * step))
        return output

    score_401_1 = score(401, 1.0e-5)
    score_401_3 = score(401, 3.0e-5)
    score_601_3 = score(601, 3.0e-5)
    return {
        "kind": "sequential_dense_transformed_mixture_value_converged_fd_score_diagnostic",
        "value": value_601,
        "score": score_601_3,
        "orders": [401, 601],
        "radius": 8.0,
        "fd_steps": [1.0e-5, 3.0e-5],
        "value_order_gap": abs(value_601 - value_401),
        "fd_step_gap": max(
            abs(score_401_1[index] - score_401_3[index]) for index in range(2)
        ),
        "fd_order_gap": max(
            abs(score_401_3[index] - score_601_3[index]) for index in range(2)
        ),
        "score_provenance": "diagnostic centered finite difference of converged dense value",
    }


def _diagnostic_comparators(row_id: str, target: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for method in ("sgqf", "zhao_cui"):
        row = _prior_row(row_id, method)
        if row["scope"]["source_observation_sha256"] != target["source_observation_sha256"]:
            raise RuntimeError(f"{row_id}/{method} diagnostic target hash mismatch")
        output.append(
            {
                "method": method,
                "value": row.get("value"),
                "score": row.get("score"),
                "role": "same_target_approximate_diagnostic",
                "source_artifact": PRIOR_ARTIFACT.as_posix(),
            }
        )
    return output


def _scope(row_id: str, target: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "horizon": int(target["observations"].shape[0]),
        "state_dimension": target["state_dim"],
        "observation_dimension": target["observation_dim"],
        "parameter_dimension": target["parameter_dim"],
        "particle_count": base.N,
        "event_order": target["event_order"],
        "source_observation_sha256": target["source_observation_sha256"],
        "runtime_fp32_observation_sha256": base._tensor_hash(
            target["observations"], tf.float32
        ),
        "theta": [float(item) for item in target["theta"].numpy()],
    }


def _render(payload: dict[str, Any]) -> str:
    lines = [
        "# Pairwise-Moment GenUT Cross-Model Trial",
        "",
        f"Status: `{payload['status']}`",
        "",
        "| Model | Arm | Value mean (SD) | Score means | Score SDs |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for model in payload["models"]:
        for arm in ("baseline", "candidate"):
            summary = model["claim"][arm]["summary"]
            score_means = [
                summary[f"score_{index}"]["mean"]
                for index in range(model["scope"]["parameter_dimension"])
            ]
            score_sds = [
                summary[f"score_{index}"]["sample_sd"]
                for index in range(model["scope"]["parameter_dimension"])
            ]
            lines.append(
                f"| {model['row_id']} | {arm} | "
                f"{summary['value']['mean']:.7g} ({summary['value']['sample_sd']:.4g}) | "
                f"`{score_means}` | `{score_sds}` |"
            )
    lines += [
        "",
        "LGSSM uses the exact Kalman oracle. KSC-SV is a scalar-state structural",
        "null for off-diagonal pairwise moments and uses a dense transformed-mixture",
        "reference. Predator-prey comparator gaps are diagnostic only.",
    ]
    return "\n".join(lines) + "\n"


def run(output_root: Path, *, smoke_only: bool) -> dict[str, Any]:
    started = time.perf_counter()
    output_root.mkdir(parents=True, exist_ok=False)
    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    logical = tf.config.list_logical_devices("GPU")
    if not logical:
        raise RuntimeError("cross-model pairwise trial requires a logical GPU")

    all_targets = base._build_targets()
    targets = {row_id: all_targets[row_id] for row_id in MODEL_IDS}
    models = []
    seeds = CLAIM_SEEDS[:1] if smoke_only else CLAIM_SEEDS
    for model_index, (row_id, target) in enumerate(targets.items()):
        baseline_controls = _baseline_controls(row_id, target)
        if row_id == "ksc_sv_T10":
            candidate_controls = _candidate_controls(baseline_controls, 4, 0.02)
            tuning = {
                "scope": row_id,
                "selection_status": "STRUCTURAL_NULL_D1_NO_OFF_DIAGONAL_PAIRS",
                "baseline_controls": baseline_controls,
                "selected_controls": candidate_controls,
                "claim_data_read_during_selection": False,
                "reference_or_comparator_read_during_selection": False,
            }
        elif smoke_only:
            candidate_controls = _candidate_controls(baseline_controls, 1, 0.005)
            tuning = {
                "scope": row_id,
                "selection_status": "SMOKE_FORCED_NONZERO_PAIRWISE_ARM",
                "baseline_controls": baseline_controls,
                "selected_controls": candidate_controls,
                "claim_data_read_during_selection": False,
                "reference_or_comparator_read_during_selection": False,
            }
        else:
            tuning = _tune(row_id, target, baseline_controls)
            candidate_controls = tuning["selected_controls"]

        baseline_evaluator = _evaluator(target, baseline_controls)
        candidate_evaluator = _evaluator(target, candidate_controls)
        baseline_rows = [
            _row(baseline_evaluator, target, target["observations"], seed)
            for seed in seeds
        ]
        candidate_rows = [
            _row(candidate_evaluator, target, target["observations"], seed)
            for seed in seeds
        ]
        if not all(base._valid(row) for row in baseline_rows + candidate_rows):
            raise RuntimeError(f"{row_id} claim row failed finite-program gates")

        if row_id == "lgssm_T50":
            reference = _lgssm_reference(target)
        elif row_id == "ksc_sv_T10":
            reference = _ksc_dense_reference(target, smoke_only=smoke_only)
        else:
            reference = None

        if smoke_only:
            claim = {
                "baseline": {"controls": baseline_controls, "rows": baseline_rows},
                "candidate": {"controls": candidate_controls, "rows": candidate_rows},
            }
        else:
            baseline_summary = _summary(baseline_rows)
            candidate_summary = _summary(candidate_rows)
            variance_ratio = _bootstrap_variance_ratio(
                baseline_rows,
                candidate_rows,
                seed=20260730 + model_index,
            )
            value_shift_ok = abs(
                candidate_summary["value"]["mean"]
                - baseline_summary["value"]["mean"]
            ) <= baseline_summary["value"]["sample_sd"] / math.sqrt(len(seeds))
            claim = {
                "baseline": {
                    "controls": baseline_controls,
                    "rows": baseline_rows,
                    "summary": baseline_summary,
                },
                "candidate": {
                    "controls": candidate_controls,
                    "rows": candidate_rows,
                    "summary": candidate_summary,
                },
                "paired_candidate_minus_baseline": _paired_summary(
                    baseline_rows, candidate_rows
                ),
                "variance_ratio": variance_ratio,
                "gates": {
                    "all_score_coordinate_sds_lower": all(
                        candidate_summary[f"score_{index}"]["sample_sd"]
                        < baseline_summary[f"score_{index}"]["sample_sd"]
                        for index in range(target["parameter_dim"])
                    ),
                    "value_sd_within_25_percent": (
                        candidate_summary["value"]["sample_sd"]
                        <= 1.25 * baseline_summary["value"]["sample_sd"]
                    ),
                    "value_mean_shift_within_baseline_se": value_shift_ok,
                    "aggregate_variance_ratio_ci_below_one": (
                        variance_ratio["bootstrap_ci95_upper"] < 1.0
                    ),
                },
            }
            if reference is not None and reference.get("score") is not None:
                claim["absolute_error_to_reference"] = _absolute_error_summary(
                    baseline_rows, candidate_rows, reference
                )

        models.append(
            {
                "row_id": row_id,
                "scope": _scope(row_id, target),
                "tuning": tuning,
                "claim": claim,
                "reference": reference,
                "diagnostic_comparators": _diagnostic_comparators(row_id, target),
                "pairwise_constraint_count": {
                    "ordered_co_skewness": target["state_dim"]
                    * (target["state_dim"] - 1),
                    "unordered_co_kurtosis": target["state_dim"]
                    * (target["state_dim"] - 1)
                    // 2,
                },
            }
        )

    if smoke_only:
        status = "PASS_GPU_XLA_SMOKE"
    else:
        status = "COMPLETE_CROSS_MODEL_PAIRWISE_FEASIBILITY"
    result_json = output_root / "result.json"
    payload = {
        "schema_version": SCHEMA,
        "status": status,
        "hard_valid": True,
        "smoke_only": smoke_only,
        "plan": PLAN.as_posix(),
        "prior_baseline_artifact": PRIOR_ARTIFACT.as_posix(),
        "models": models,
        "configuration": {
            "particle_count": base.N,
            "claim_seeds": seeds,
            "tuning_seeds": TUNING_SEEDS,
            "pairwise_arms": PAIRWISE_ARMS,
            "dtype": "float32",
            "tf32": True,
            "jit_compile": True,
            "score_policy": "manual recursive forward sensitivity",
        },
        "device": {
            "logical_devices": [device.name for device in logical],
            "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        },
        "memory_policy": dict(memory_policy),
        "gpu_allocator": {
            key: int(value)
            for key, value in tf.config.experimental.get_memory_info("GPU:0").items()
        },
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "wall_time_seconds": time.perf_counter() - started,
        "run_manifest": {
            "command": [sys.executable, *sys.argv],
            "environment": sys.prefix,
            "host": platform.node(),
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "source_sha256": {
                PLAN.as_posix(): _sha256(ROOT / PLAN),
                PRIOR_ARTIFACT.as_posix(): _sha256(ROOT / PRIOR_ARTIFACT),
                Path(__file__).relative_to(ROOT).as_posix(): _sha256(Path(__file__)),
                "bayesfilter/highdim/higher_moment_contract_e.py": _sha256(
                    ROOT / "bayesfilter/highdim/higher_moment_contract_e.py"
                ),
                "bayesfilter/highdim/cubature_genut_filter.py": _sha256(
                    ROOT / "bayesfilter/highdim/cubature_genut_filter.py"
                ),
            },
        },
        "nonclaims": [
            "lower score variance does not by itself prove lower score bias",
            "KSC dense finite-difference score is diagnostic reference, not runtime GenUT score",
            "SGQF and Zhao-Cui are not nonlinear truth oracles",
            "no default, HMC, broad superiority, or NAWM claim",
        ],
    }
    result_json.write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if not smoke_only:
        (output_root / "result.md").write_text(_render(payload), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    payload = run(args.output_root.resolve(), smoke_only=args.smoke_only)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": str(args.output_root.resolve()),
                "wall_time_seconds": payload["wall_time_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
