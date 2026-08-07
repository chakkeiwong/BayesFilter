#!/usr/bin/env python3
"""Tune and run the six-model dual-cap GenUT feasibility leaderboard."""

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
from typing import Any

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

from bayesfilter.runtime.gpu_memory_policy import (
    configure_tensorflow_gpu_memory_growth,
)


PLAN = Path(
    "docs/plans/"
    "bayesfilter-genut-dual-cap-production-leaderboard-plan-2026-08-07.md"
)
PRIOR = Path(
    "docs/benchmarks/artifacts/"
    "moment_retuned_genut_whole_leaderboard_20260723/"
    "attempt05_final/result.json"
)
ROW_IDS = (
    "lgssm_T50",
    "ksc_sv_T10",
    "exact_sv_T10",
    "generalized_sv_T10",
    "predator_prey_T20",
    "austria_sir_T20",
)
CLAIM_SEEDS = tuple(range(98201, 98217))
TUNING_SEEDS = (98501, 98502)
AUSTRIA_TUNING_SEEDS = (98301, 98302)
PARTICLE_COUNT = 1008
PAIRWISE_STRENGTHS = (0.02, 0.05)
FD_STEPS = (1.0e-3, 1.0e-2, 3.0e-2)
RESIDUAL_TOLERANCE = 5.0e-4
CAP_OUTPUT_TOLERANCE = 0.980001


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


def _prior_controls(row_id: str) -> dict[str, Any]:
    payload = json.loads((ROOT / PRIOR).read_text(encoding="utf-8"))
    matches = [
        row
        for row in payload["rows"]
        if row["row_id"] == row_id and row["method"] == "genut"
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one prior GenUT row for {row_id}")
    return dict(matches[0]["controls"])


def _prior_scope(row_id: str) -> dict[str, Any]:
    payload = json.loads((ROOT / PRIOR).read_text(encoding="utf-8"))
    return next(
        row["scope"]
        for row in payload["rows"]
        if row["row_id"] == row_id and row["method"] == "genut"
    )


def _prior_comparators(
    targets: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    payload = json.loads((ROOT / PRIOR).read_text(encoding="utf-8"))
    result: dict[str, list[dict[str, Any]]] = {}
    for row_id, target in targets.items():
        cells = []
        for row in payload["rows"]:
            if row["row_id"] != row_id or row["method"] == "genut":
                continue
            if row_id == "austria_sir_T20" and row["method"] == "zhao_cui":
                continue
            observed_hash = row.get("target_hash") or row.get("scope", {}).get(
                "source_observation_sha256"
            )
            if observed_hash != target["source_observation_sha256"]:
                raise RuntimeError(f"{row_id}/{row['method']} target hash mismatch")
            cells.append(
                {
                    key: value
                    for key, value in row.items()
                    if key not in {"scope", "row_id"}
                }
            )
        result[row_id] = cells
    return result


def _build_targets() -> dict[str, dict[str, Any]]:
    """Build the frozen rows without importing unrelated nonlinear targets."""

    from bayesfilter.highdim.cubature_genut_adapters import (
        diagonal_lgssm_candidate_adapter,
        exact_transformed_sv_candidate_adapter,
        generalized_sv_prior_mean_candidate_adapter,
        ksc_mixture_sv_candidate_adapter,
        parameterized_austria_sir_candidate_adapter,
        predator_prey_candidate_adapter,
    )
    from bayesfilter.highdim.models import (
        p30_predator_prey_fixture_model,
        zhao_cui_sir_austria_model,
    )
    from bayesfilter.highdim.sv_mixture_cut4 import (
        exact_transformed_sv_observations,
        transformed_sv_observations,
    )
    from docs.benchmarks import run_moment_retuned_genut_whole_leaderboard as base
    from scripts.filtering_value_gradient_benchmark_generate_p8_datasets import (
        _generalized_sv_prior_mean_dataset,
        _lgssm_dataset,
        _sv_dataset,
    )

    targets: dict[str, dict[str, Any]] = {}
    lg = _lgssm_dataset(81100)

    def lg_data(seed: int) -> tf.Tensor:
        return tf.cast(_lgssm_dataset(seed)["observations"][:50], tf.float32)

    targets["lgssm_T50"] = {
        "theta": tf.constant([0.72, 0.55, 0.35, 0.35, 0.45], tf.float32),
        "observations": tf.cast(lg["observations"][:50], tf.float32),
        "source_observation_sha256": base._tensor_hash(lg["observations"][:50]),
        "calibration": [lg_data(91101), lg_data(91102)],
        "validation": [lg_data(91201), lg_data(91202)],
        "adapter": diagonal_lgssm_candidate_adapter(
            observation_matrix=tf.constant(
                [[1.0, 0.25, -0.15], [0.2, 1.1, 0.3], [-0.1, 0.35, 0.9]],
                tf.float32,
            )
        ),
        "design": base._genut_design(3),
        "state_dim": 3,
        "parameter_dim": 5,
        "observation_dim": 3,
        "transition_before": False,
        "model_id": "lgssm_T50",
        "event_order": "stationary_initial_draw_then_observe_y0_then_transitions",
    }

    source_sv = _sv_dataset(81101)
    raw64 = tf.cast(source_sv["observations"][:10], tf.float64)
    for kind, row_id, adapter in (
        ("ksc", "ksc_sv_T10", ksc_mixture_sv_candidate_adapter()),
        ("exact", "exact_sv_T10", exact_transformed_sv_candidate_adapter()),
    ):
        if kind == "exact":
            source_observations = exact_transformed_sv_observations(raw64)
        else:
            source_observations = transformed_sv_observations(raw64, offset=1.0e-8)

        def sv_data(seed: int, selected_kind: str = kind) -> tf.Tensor:
            raw_data = tf.cast(_sv_dataset(seed)["observations"][:10], tf.float64)
            if selected_kind == "exact":
                return tf.cast(exact_transformed_sv_observations(raw_data), tf.float32)
            return tf.cast(
                transformed_sv_observations(raw_data, offset=1.0e-8), tf.float32
            )

        targets[row_id] = {
            "theta": tf.cast(source_sv["truth_theta"], tf.float32),
            "observations": tf.cast(source_observations, tf.float32),
            "source_observation_sha256": base._tensor_hash(source_observations),
            "calibration": [sv_data(91111), sv_data(91112)],
            "validation": [sv_data(91211), sv_data(91212)],
            "adapter": adapter,
            "design": base._genut_design(1),
            "state_dim": 1,
            "parameter_dim": 2,
            "observation_dim": 1,
            "transition_before": False,
            "model_id": row_id,
            "event_order": (
                "stationary_initial_draw_then_observe_y0_to_y9_before_transitions"
            ),
        }

    generalized = _generalized_sv_prior_mean_dataset(81105)

    def generalized_data(seed: int) -> tf.Tensor:
        return tf.cast(
            _generalized_sv_prior_mean_dataset(seed)["observations"][:10],
            tf.float32,
        )

    targets["generalized_sv_T10"] = {
        "theta": tf.cast(generalized["truth_theta"], tf.float32),
        "observations": tf.cast(generalized["observations"][:10], tf.float32),
        "source_observation_sha256": base._tensor_hash(
            generalized["observations"][:10]
        ),
        "calibration": [generalized_data(91121), generalized_data(91122)],
        "validation": [generalized_data(91221), generalized_data(91222)],
        "adapter": generalized_sv_prior_mean_candidate_adapter(),
        "design": base._genut_design(1),
        "state_dim": 1,
        "parameter_dim": 3,
        "observation_dim": 1,
        "transition_before": True,
        "model_id": "generalized_sv_T10",
        "event_order": (
            "stationary_initial_draw_then_transition_before_every_observation"
        ),
    }

    pp_model = p30_predator_prey_fixture_model()

    def pp_data(seed: int, *, return_states: bool = False):
        generator = tf.random.Generator.from_seed(seed)
        state = pp_model.initial_mean + tf.linalg.matvec(
            tf.linalg.cholesky(pp_model.initial_covariance),
            generator.normal([2], dtype=tf.float64),
        )
        states = [state]
        observations = []
        for _ in range(20):
            state = pp_model.transition_mean(pp_model.true_parameters(), state)[0]
            state += tf.linalg.matvec(
                tf.linalg.cholesky(pp_model.process_covariance),
                generator.normal([2], dtype=tf.float64),
            )
            states.append(state)
            observations.append(
                state
                + tf.linalg.matvec(
                    tf.linalg.cholesky(pp_model.observation_covariance),
                    generator.normal([2], dtype=tf.float64),
                )
            )
        if return_states:
            return tf.stack(states), tf.stack(observations)
        return tf.cast(tf.stack(observations), tf.float32)

    _pp_states, pp_observations = pp_data(81104, return_states=True)
    targets["predator_prey_T20"] = {
        "theta": tf.cast(pp_model.true_parameters(), tf.float32),
        "observations": tf.cast(pp_observations, tf.float32),
        "source_observation_sha256": base._tensor_hash(pp_observations),
        "calibration": [pp_data(91131), pp_data(91132)],
        "validation": [pp_data(91231), pp_data(91232)],
        "adapter": predator_prey_candidate_adapter(),
        "design": base._genut_design(2),
        "state_dim": 2,
        "parameter_dim": 6,
        "observation_dim": 2,
        "transition_before": True,
        "model_id": "predator_prey_T20",
        "event_order": "x0_then_transition_1_to_20_then_observe_y1_to_y20",
    }

    sir_model = zhao_cui_sir_austria_model()
    with tf.device("/CPU:0"):
        _sir_states, all_sir_observations = sir_model.simulate(
            final_time=20, seed=81120
        )
    sir_observations = tf.convert_to_tensor(all_sir_observations, tf.float64)[1:21]

    def sir_data(seed: int) -> tf.Tensor:
        generator = tf.random.Generator.from_seed(seed)
        state = sir_model.initial_mean + tf.linalg.matvec(
            tf.linalg.cholesky(sir_model.initial_covariance),
            generator.normal([18], dtype=tf.float64),
        )
        observations = []
        for _ in range(20):
            state = sir_model.transition_mean(state)[0] + tf.linalg.matvec(
                tf.linalg.cholesky(sir_model.process_covariance),
                generator.normal([18], dtype=tf.float64),
            )
            observations.append(
                sir_model.infectious_components(state)[0]
                + tf.linalg.matvec(
                    tf.linalg.cholesky(sir_model.observation_covariance),
                    generator.normal([9], dtype=tf.float64),
                )
            )
        return tf.cast(tf.stack(observations), tf.float32)

    targets["austria_sir_T20"] = {
        "theta": tf.zeros([3], tf.float32),
        "observations": tf.cast(sir_observations, tf.float32),
        "source_observation_sha256": base._tensor_hash(sir_observations),
        "calibration": [sir_data(91141), sir_data(91142)],
        "validation": [sir_data(91241), sir_data(91242)],
        "adapter": parameterized_austria_sir_candidate_adapter(),
        "design": base._genut_design(18),
        "state_dim": 18,
        "parameter_dim": 3,
        "observation_dim": 9,
        "transition_before": True,
        "model_id": "austria_sir_T20",
        "event_order": "x0_then_transition_before_y1_to_y20",
    }
    return targets


def _evaluator(target: dict[str, Any], controls: dict[str, Any]):
    from docs.benchmarks import run_moment_retuned_genut_whole_leaderboard as base

    return base._make_evaluator(
        adapter=target["adapter"],
        horizon=int(target["observations"].shape[0]),
        observation_dim=target["observation_dim"],
        state_dim=target["state_dim"],
        parameter_dim=target["parameter_dim"],
        transition_before_first_observation=target["transition_before"],
        controls=controls,
    )


def _evaluate(
    evaluator: Any,
    target: dict[str, Any],
    observations: tf.Tensor,
    seed: int,
) -> dict[str, Any]:
    from docs.benchmarks import run_moment_retuned_genut_whole_leaderboard as base

    return base._evaluate(
        evaluator,
        target["theta"],
        tf.cast(observations, tf.float32),
        seed,
        target["design"],
    )


def _valid(row: dict[str, Any], *, capped: bool) -> bool:
    from docs.benchmarks import run_moment_retuned_genut_whole_leaderboard as base

    residuals = (
        row["max_mean_residual"],
        row["max_row_residual"],
        row["max_col_residual"],
        row["score_increment_sum_residual"],
    )
    return (
        base._valid(row)
        and max(residuals) <= RESIDUAL_TOLERANCE
        and (
            not capped
            or row["maximum_coordinatewise_post_cap_absolute"]
            < CAP_OUTPUT_TOLERANCE
        )
    )


def _scaled_variance(
    rows_by_dataset: list[list[dict[str, Any]]], horizon: int
) -> float:
    maxima = []
    for rows in rows_by_dataset:
        vectors = [
            [
                row["value"] / horizon,
                *[score / math.sqrt(horizon) for score in row["score"]],
            ]
            for row in rows
        ]
        maxima.append(
            max(
                statistics.variance(vector[index] for vector in vectors)
                for index in range(len(vectors[0]))
            )
        )
    return statistics.mean(maxima)


def _partition(
    evaluator: Any,
    target: dict[str, Any],
    datasets: list[tf.Tensor],
    *,
    capped: bool,
    tuning_seeds: tuple[int, ...],
) -> dict[str, Any]:
    rows_by_dataset = [
        [
            _evaluate(evaluator, target, observations, seed)
            for seed in tuning_seeds
        ]
        for observations in datasets
    ]
    rows = [row for dataset_rows in rows_by_dataset for row in dataset_rows]
    hard_valid = all(_valid(row, capped=capped) for row in rows)
    return {
        "hard_valid": hard_valid,
        "rows": rows,
        "mean_pairwise_residual_objective": (
            statistics.mean(
                row["mean_normalized_pairwise_shape_residual_objective"]
                for row in rows
            )
            if rows
            else None
        ),
        "scaled_value_score_variance": (
            _scaled_variance(rows_by_dataset, int(target["observations"].shape[0]))
            if hard_valid
            else None
        ),
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = ("value", *[f"score_{index}" for index in range(len(rows[0]["score"]))])
    vectors = [[row["value"], *row["score"]] for row in rows]
    result: dict[str, Any] = {"count": len(rows), "labels": labels}
    for index, label in enumerate(labels):
        sample = [vector[index] for vector in vectors]
        sd = statistics.stdev(sample)
        mean = statistics.mean(sample)
        result[label] = {
            "mean": mean,
            "sample_sd": sd,
            "mcse": sd / math.sqrt(len(sample)),
        }
    result.update(
        {
            "maximum_cap_active_fraction": max(
                row["fraction_coordinatewise_cap_active"] for row in rows
            ),
            "mean_cap_displacement": statistics.mean(
                row["mean_coordinatewise_cap_displacement"] for row in rows
            ),
            "maximum_pairwise_pre_cap_rms": max(
                row["maximum_pairwise_pre_cap_particle_rms"] for row in rows
            ),
            "minimum_pairwise_cap_scale": min(
                row["minimum_pairwise_particle_cap_scale"] for row in rows
            ),
        }
    )
    return result


def _paired(
    baseline: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> dict[str, Any]:
    labels = (
        "value",
        *[f"score_{index}" for index in range(len(baseline[0]["score"]))],
    )
    result = {}
    for index, label in enumerate(labels):
        differences = [
            [right["value"], *right["score"]][index]
            - [left["value"], *left["score"]][index]
            for left, right in zip(baseline, candidate)
        ]
        sd = statistics.stdev(differences)
        mean = statistics.mean(differences)
        result[label] = {
            "mean_candidate_minus_baseline": mean,
            "sample_sd": sd,
            "mcse": sd / math.sqrt(len(differences)),
            "negative_count": sum(item < 0.0 for item in differences),
            "positive_count": sum(item > 0.0 for item in differences),
        }
    return result


def _fd_ladder(
    evaluator: Any, target: dict[str, Any], seed: int
) -> dict[str, Any]:
    from docs.benchmarks import run_moment_retuned_genut_whole_leaderboard as base

    initial, process = base._noise(
        seed,
        int(target["observations"].shape[0]),
        target["state_dim"],
    )
    theta = target["theta"]
    value, score, diagnostics = evaluator(
        theta,
        target["observations"],
        initial,
        process,
        target["design"],
    )
    steps = []
    for step in FD_STEPS:
        rows = []
        for index in range(target["parameter_dim"]):
            direction = tf.one_hot(index, target["parameter_dim"], dtype=tf.float32)
            plus = evaluator(
                theta + step * direction,
                target["observations"],
                initial,
                process,
                target["design"],
            )[0]
            minus = evaluator(
                theta - step * direction,
                target["observations"],
                initial,
                process,
                target["design"],
            )[0]
            finite_difference = (plus - minus) / (2.0 * step)
            absolute = abs(
                float(finite_difference.numpy()) - float(score[index].numpy())
            )
            rows.append(
                {
                    "parameter": index,
                    "manual_score": float(score[index].numpy()),
                    "finite_difference": float(finite_difference.numpy()),
                    "absolute_residual": absolute,
                    "normalized_residual": absolute
                    / max(abs(float(score[index].numpy())), 1.0),
                }
            )
        steps.append(
            {
                "step": step,
                "rows": rows,
                "maximum_absolute_residual": max(
                    row["absolute_residual"] for row in rows
                ),
                "maximum_normalized_residual": max(
                    row["normalized_residual"] for row in rows
                ),
            }
        )
    return {
        "value": float(value.numpy()),
        "program_valid": bool(diagnostics["program_valid"].numpy()),
        "steps": steps,
        "role": "same-program explanatory diagnostic",
    }


def _render(payload: dict[str, Any]) -> str:
    lines = [
        "# GenUT Dual-Cap Production Leaderboard",
        "",
        "The Austria SIR Zhao-Cui cell is excluded from this campaign.",
        "All cross-method differences are descriptive unless paired uncertainty is shown.",
        "",
        "| Model | Arm | Valid | Value mean (SD) | Score means | Score SDs |",
        "|---|---|---:|---:|---|---|",
    ]
    for model in payload["models"]:
        for arm_name in ("diagonal", "dual_cap"):
            arm = model[arm_name]
            summary = arm["summary"]
            means = [
                summary[f"score_{index}"]["mean"]
                for index in range(model["scope"]["parameter_dimension"])
            ]
            sds = [
                summary[f"score_{index}"]["sample_sd"]
                for index in range(model["scope"]["parameter_dimension"])
            ]
            lines.append(
                f"| {model['row_id']} | {arm_name} | {arm['hard_valid']} | "
                f"{summary['value']['mean']:.8g} "
                f"({summary['value']['sample_sd']:.5g}) | `{means}` | `{sds}` |"
            )
    lines.extend(
        [
            "",
            "## Inference Status",
            "",
            "| Item | Status |",
            "|---|---|",
            f"| Hard veto screen | {'Pass' if payload['hard_valid'] else 'Fail'} |",
            "| Statistically supported ranking | None across methods |",
            "| Descriptive-only differences | Values, scores, comparator gaps, cap activity, and runtimes |",
            f"| Default selector readiness | {payload['selector_status']} |",
            "| HMC/NeuTra readiness | Not established |",
            "| Next evidence | Target-specific posterior and HMC validation if required |",
            "",
            f"JSON: `{payload['run_manifest']['output_json']}`",
            "",
        ]
    )
    return "\n".join(lines)


def run(output_root: Path, *, algorithm: str = "default") -> dict[str, Any]:
    started = time.perf_counter()
    output_root.mkdir(parents=True, exist_ok=False)
    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    logical = tf.config.list_logical_devices("GPU")
    if not logical:
        raise RuntimeError("production leaderboard requires a logical GPU")
    from bayesfilter.highdim.genut_algorithm import resolve_genut_algorithm
    from docs.benchmarks import run_moment_retuned_genut_whole_leaderboard as base

    selection_check = resolve_genut_algorithm(algorithm)
    if selection_check.algorithm != "dual_cap":
        raise ValueError("this campaign admits only the default dual_cap family")

    targets = _build_targets()
    comparators = _prior_comparators(targets)
    models = []
    for row_id in ROW_IDS:
        target = targets[row_id]
        tuning_seeds = (
            AUSTRIA_TUNING_SEEDS
            if row_id == "austria_sir_T20"
            else TUNING_SEEDS
        )
        prior_scope = _prior_scope(row_id)
        if (
            prior_scope["source_observation_sha256"]
            != target["source_observation_sha256"]
            or prior_scope["event_order"] != target["event_order"]
            or prior_scope["particle_count"] != PARTICLE_COUNT
        ):
            raise RuntimeError(f"{row_id} prior baseline scope mismatch")

        baseline_controls = resolve_genut_algorithm("diagonal").apply(
            _prior_controls(row_id)
        )
        baseline_evaluator = _evaluator(target, baseline_controls)
        baseline_calibration = _partition(
            baseline_evaluator,
            target,
            target["calibration"],
            capped=False,
            tuning_seeds=tuning_seeds,
        )
        baseline_validation = _partition(
            baseline_evaluator,
            target,
            target["validation"],
            capped=False,
            tuning_seeds=tuning_seeds,
        )
        if not (
            baseline_calibration["hard_valid"]
            and baseline_validation["hard_valid"]
        ):
            raise RuntimeError(f"{row_id} diagonal baseline is invalid")

        candidates = []
        for strength in PAIRWISE_STRENGTHS:
            selection = resolve_genut_algorithm(
                algorithm, pairwise_strength=strength
            )
            controls = selection.apply(_prior_controls(row_id))
            evaluator = _evaluator(target, controls)
            calibration = _partition(
                evaluator,
                target,
                target["calibration"],
                capped=True,
                tuning_seeds=tuning_seeds,
            )
            validation = _partition(
                evaluator,
                target,
                target["validation"],
                capped=True,
                tuning_seeds=tuning_seeds,
            )
            candidates.append(
                {
                    "algorithm": selection.algorithm,
                    "controls": controls,
                    "calibration": calibration,
                    "validation": validation,
                    "eligible": calibration["hard_valid"]
                    and validation["hard_valid"],
                }
            )
        eligible = [candidate for candidate in candidates if candidate["eligible"]]
        if not eligible:
            raise RuntimeError(f"{row_id} has no valid dual-cap controls")
        selected = min(
            eligible,
            key=lambda item: (
                item["validation"]["mean_pairwise_residual_objective"],
                item["validation"]["scaled_value_score_variance"],
                item["calibration"]["mean_pairwise_residual_objective"],
                item["controls"]["pairwise_moment_strength"],
            ),
        )
        candidate_evaluator = _evaluator(target, selected["controls"])
        baseline_rows = [
            _evaluate(baseline_evaluator, target, target["observations"], seed)
            for seed in CLAIM_SEEDS
        ]
        candidate_rows = [
            _evaluate(candidate_evaluator, target, target["observations"], seed)
            for seed in CLAIM_SEEDS
        ]
        baseline_valid = all(_valid(row, capped=False) for row in baseline_rows)
        candidate_valid = all(_valid(row, capped=True) for row in candidate_rows)
        if not baseline_valid:
            raise RuntimeError(f"{row_id} diagonal claim baseline is invalid")

        scope = {
            "row_id": row_id,
            "model_id": target["model_id"],
            "horizon": int(target["observations"].shape[0]),
            "state_dimension": target["state_dim"],
            "observation_dimension": target["observation_dim"],
            "parameter_dimension": target["parameter_dim"],
            "particle_count": PARTICLE_COUNT,
            "event_order": target["event_order"],
            "source_observation_sha256": target["source_observation_sha256"],
            "runtime_fp32_observation_sha256": base._tensor_hash(
                target["observations"], tf.float32
            ),
        }
        comparator_cells = [
            cell
            for cell in comparators.get(row_id, [])
            if not (row_id == "austria_sir_T20" and cell["method"] == "zhao_cui")
        ]
        model = {
            "row_id": row_id,
            "scope": scope,
            "tuning": {
                "selection_objective": (
                    "validation mean normalized pairwise residual; variance, "
                    "calibration residual, then lower strength tie-breakers"
                ),
                "claim_data_read_during_selection": False,
                "tuning_seeds": tuning_seeds,
                "candidates": candidates,
                "selected_controls": selected["controls"],
            },
            "diagonal": {
                "controls": baseline_controls,
                "calibration": baseline_calibration,
                "validation": baseline_validation,
                "rows": baseline_rows,
                "summary": _summary(baseline_rows),
                "hard_valid": baseline_valid,
            },
            "dual_cap": {
                "controls": selected["controls"],
                "rows": candidate_rows,
                "summary": _summary(candidate_rows),
                "hard_valid": candidate_valid,
                "paired_vs_diagonal": _paired(baseline_rows, candidate_rows),
                "finite_difference_ladder": _fd_ladder(
                    candidate_evaluator, target, CLAIM_SEEDS[0]
                ),
            },
            "comparators": comparator_cells,
        }
        models.append(model)
        (output_root / f"{row_id}_checkpoint.json").write_text(
            json.dumps(_safe(model), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    hard_valid = all(model["dual_cap"]["hard_valid"] for model in models)
    result_json = output_root / "result.json"
    result_markdown = output_root / "result.md"
    payload = {
        "schema_version": "bayesfilter.genut_dual_cap_production_leaderboard.v1",
        "status": (
            "SIX_MODEL_DUAL_CAP_FEASIBILITY_PASS"
            if hard_valid
            else "DUAL_CAP_PROMOTION_VETO"
        ),
        "hard_valid": hard_valid,
        "selector_status": (
            "default resolves to dual_cap; explicit alternatives preserved"
        ),
        "excluded_cell": {
            "row_id": "austria_sir_T20",
            "method": "zhao_cui",
            "reason": "separate agent-owned implementation program",
        },
        "models": models,
        "configuration": {
            "particle_count": PARTICLE_COUNT,
            "claim_seeds": CLAIM_SEEDS,
            "tuning_seeds": TUNING_SEEDS,
            "austria_tuning_seeds": AUSTRIA_TUNING_SEEDS,
            "pairwise_strength_grid": PAIRWISE_STRENGTHS,
            "fd_steps": FD_STEPS,
            "algorithm_request": algorithm,
            "algorithm_resolved": selection_check.algorithm,
            "dtype": "float32",
            "tf32": True,
            "jit_compile": True,
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
            "plan": PLAN.as_posix(),
            "output_json": str(result_json.relative_to(ROOT)),
            "output_markdown": str(result_markdown.relative_to(ROOT)),
            "random_seeds": {
                "tuning": TUNING_SEEDS,
                "claim": CLAIM_SEEDS,
            },
            "source_sha256": {
                PLAN.as_posix(): _sha256(ROOT / PLAN),
                Path(__file__).relative_to(ROOT).as_posix(): _sha256(Path(__file__)),
                "bayesfilter/highdim/genut_algorithm.py": _sha256(
                    ROOT / "bayesfilter/highdim/genut_algorithm.py"
                ),
                "bayesfilter/highdim/cubature_genut_filter.py": _sha256(
                    ROOT / "bayesfilter/highdim/cubature_genut_filter.py"
                ),
                "bayesfilter/highdim/higher_moment_contract_e.py": _sha256(
                    ROOT / "bayesfilter/highdim/higher_moment_contract_e.py"
                ),
            },
        },
        "nonclaims": [
            "no exact nonlinear score theorem",
            "no statistically supported cross-method ranking",
            "no posterior, HMC, or NeuTra readiness",
            "no Zhao-Cui Austria completion",
        ],
    }
    result_json.write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result_markdown.write_text(_render(payload), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--algorithm", default="default")
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    try:
        payload = run(output_root, algorithm=args.algorithm)
    except Exception as error:
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "failure.json").write_text(
            json.dumps(
                {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "plan": PLAN.as_posix(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        raise
    print(
        json.dumps(
            {
                "status": payload["status"],
                "hard_valid": payload["hard_valid"],
                "output": str(output_root),
                "wall_time_seconds": payload["wall_time_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
