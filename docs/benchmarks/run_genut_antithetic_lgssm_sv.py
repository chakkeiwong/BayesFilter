#!/usr/bin/env python3
"""Paired equal-cost GenUT antithetic study for LGSSM and exact SV."""

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


PLAN = Path("docs/plans/bayesfilter-genut-antithetic-lgssm-sv-plan-2026-07-22.md")
RESULT_NOTE = Path("docs/plans/bayesfilter-genut-antithetic-lgssm-sv-result-2026-07-22.md")
SCHEMA_VERSION = "bayesfilter.genut_antithetic_lgssm_sv.v1"
CAMPAIGN_ID = "genut-antithetic-lgssm-sv-20260722"
MODELS = ("lgssm", "sv")
HORIZON = 50
PARTICLE_COUNTS = {"lgssm": 1008, "sv": 1002}
PARAMETER_COUNTS = {"lgssm": 5, "sv": 2}
LABELS = {
    "lgssm": ("value", "phi1", "phi2", "phi3", "q_scale", "r_scale"),
    "sv": ("value", "theta_gamma", "theta_log_beta"),
}
LGSSM_THETA = (0.72, 0.55, 0.35, 0.35, 0.45)
SV_THETA = (0.25, -0.15)
OBSERVATION_MATRIX_VALUES = (
    (1.0, 0.25, -0.15),
    (0.2, 1.1, 0.3),
    (-0.1, 0.35, 0.9),
)
CONTROL_GRID = tuple(
    {
        "epsilon": epsilon,
        "sinkhorn_steps": steps,
        "balance_steps": balance_steps,
        "ridge": ridge,
    }
    for epsilon in (2.0, 4.0)
    for steps in (4, 8)
    for balance_steps in (4, 8)
    for ridge in (1.0e-6, 1.0e-5)
)
DGP_SEEDS = {
    "lgssm": {
        "calibration": (91001, 91002),
        "validation": (91101, 91102),
        "claim": tuple(range(91201, 91209)),
    },
    "sv": {
        "calibration": (92001, 92002),
        "validation": (92101, 92102),
        "claim": tuple(range(92201, 92209)),
    },
}
TUNING_PARTICLE_SEEDS = (93101, 93102, 93103, 93104)
CLAIM_PARTICLE_SEEDS = tuple(range(93201, 93217))
FD_OFFSETS = (-0.35, 0.0, 0.35)
FD_RELATIVE_STEPS = (4.0e-3, 8.0e-3)
FD_MINIMUM_STEPS = (4.0e-4, 8.0e-4)
FD_RELATIVE_TOLERANCE = 5.0e-2
VARIANCE_SCALES = {"lgssm": (50.0, 10.0, 10.0, 10.0, 10.0, 10.0), "sv": (50.0, 10.0, 10.0)}
FAMILYWISE_CRITICAL = {"lgssm": 3.635807421953938, "sv": 3.127552274246371}
VARIANCE_FLOOR = 1.0e-18
RESIDUAL_TOLERANCE = 5.0e-4


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


def _theta(model: str) -> tf.Tensor:
    return tf.constant(LGSSM_THETA if model == "lgssm" else SV_THETA, tf.float32)


def _generate_dataset(model: str, seed: int) -> dict[str, tf.Tensor]:
    """Generate one genuine DGP sequence with the pre-observation transition."""

    if model == "lgssm":
        theta = _theta(model)
        phi, q_scale, r_scale = theta[:3], theta[3], theta[4]
        initial_noise = tf.random.stateless_normal([3], [seed, 1], dtype=tf.float32)
        transition_noise = tf.random.stateless_normal(
            [HORIZON, 3], [seed, 2], dtype=tf.float32
        )
        observation_noise = tf.random.stateless_normal(
            [HORIZON, 3], [seed, 3], dtype=tf.float32
        )
        state = initial_noise * q_scale / tf.sqrt(1.0 - tf.square(phi))
        matrix = tf.constant(OBSERVATION_MATRIX_VALUES, tf.float32)
        states = tf.TensorArray(tf.float32, size=HORIZON, element_shape=(3,))
        observations = tf.TensorArray(tf.float32, size=HORIZON, element_shape=(3,))

        def body(index, previous, states_array, observations_array):
            current = phi * previous + q_scale * transition_noise[index]
            observed = tf.linalg.matvec(matrix, current) + r_scale * observation_noise[index]
            return (
                index + 1,
                current,
                states_array.write(index, current),
                observations_array.write(index, observed),
            )

        _, _, states, observations = tf.while_loop(
            lambda index, *_: index < HORIZON,
            body,
            (tf.zeros([], tf.int32), state, states, observations),
            parallel_iterations=1,
        )
        return {
            "states": states.stack(),
            "observations": observations.stack(),
            "initial_noise": initial_noise,
            "transition_noise": transition_noise,
            "observation_noise": observation_noise,
        }
    if model != "sv":
        raise ValueError(f"unknown model: {model}")
    theta = _theta(model)
    gamma = 0.5 * (
        1.0 + tf.math.erf(theta[0] / tf.sqrt(tf.constant(2.0, tf.float32)))
    )
    beta = tf.exp(theta[1])
    initial_noise = tf.random.stateless_normal([], [seed, 1], dtype=tf.float32)
    transition_noise = tf.random.stateless_normal(
        [HORIZON], [seed, 2], dtype=tf.float32
    )
    observation_noise = tf.random.stateless_normal(
        [HORIZON], [seed, 3], dtype=tf.float32
    )
    state = initial_noise / tf.sqrt(1.0 - tf.square(gamma))
    states = tf.TensorArray(tf.float32, size=HORIZON, element_shape=())
    observations = tf.TensorArray(tf.float32, size=HORIZON, element_shape=())

    def sv_body(index, previous, states_array, observations_array):
        current = gamma * previous + transition_noise[index]
        observed = current + 2.0 * tf.math.log(beta) + tf.math.log(
            tf.square(observation_noise[index])
        )
        return (
            index + 1,
            current,
            states_array.write(index, current),
            observations_array.write(index, observed),
        )

    _, _, states, observations = tf.while_loop(
        lambda index, *_: index < HORIZON,
        sv_body,
        (tf.zeros([], tf.int32), state, states, observations),
        parallel_iterations=1,
    )
    return {
        "states": states.stack(),
        "observations": observations.stack()[:, None],
        "initial_noise": tf.reshape(initial_noise, [1]),
        "transition_noise": transition_noise,
        "observation_noise": observation_noise,
    }


def _genut_design(model: str) -> tf.Tensor:
    from bayesfilter.highdim.cubature_genut_candidate import (
        gaussian_genut_design,
        replicate_positive_genut,
    )

    dimension = 3 if model == "lgssm" else 1
    return replicate_positive_genut(
        gaussian_genut_design(dim=dimension),
        num_particles=PARTICLE_COUNTS[model],
    )


def _model_adapter(model: str):
    from bayesfilter.highdim.cubature_genut_adapters import (
        diagonal_lgssm_candidate_adapter,
        exact_transformed_sv_candidate_adapter,
    )

    if model == "lgssm":
        return diagonal_lgssm_candidate_adapter(
            observation_matrix=tf.constant(OBSERVATION_MATRIX_VALUES, tf.float32)
        )
    if model == "sv":
        return exact_transformed_sv_candidate_adapter()
    raise ValueError(f"unknown model: {model}")


def _particle_noise(model: str, seed: int) -> tuple[tf.Tensor, tf.Tensor]:
    dimension = 3 if model == "lgssm" else 1
    particle_count = PARTICLE_COUNTS[model]
    initial = tf.random.stateless_normal(
        [particle_count, dimension], [seed, 101], dtype=tf.float32
    )
    process = tf.random.stateless_normal(
        [HORIZON, particle_count, dimension], [seed, 102], dtype=tf.float32
    )
    return initial, process


def _make_evaluator(model: str, adapter: Any, controls: dict[str, object]):
    from bayesfilter.highdim.cubature_genut_filter import finite_value_score

    particle_count = PARTICLE_COUNTS[model]
    dimension = 3 if model == "lgssm" else 1
    parameter_count = PARAMETER_COUNTS[model]

    @tf.function(jit_compile=True, reduce_retracing=True)
    def evaluate(theta, observations, initial, process, design):
        theta = tf.ensure_shape(theta, [parameter_count])
        observations = tf.ensure_shape(observations, [HORIZON, dimension])
        initial = tf.ensure_shape(initial, [particle_count, dimension])
        process = tf.ensure_shape(process, [HORIZON, particle_count, dimension])
        design = tf.ensure_shape(design, [particle_count, dimension])
        with tf.device("/GPU:0"):
            return finite_value_score(
                adapter,
                theta,
                observations,
                initial,
                process,
                design,
                epsilon=float(controls["epsilon"]),
                sinkhorn_steps=int(controls["sinkhorn_steps"]),
                balance_steps=int(controls.get("balance_steps", 8)),
                ridge=float(controls["ridge"]),
            )

    return evaluate


def _evaluate_constituent(
    evaluate: Any,
    theta: tf.Tensor,
    observations: tf.Tensor,
    initial: tf.Tensor,
    process: tf.Tensor,
    design: tf.Tensor,
) -> dict[str, object]:
    started = time.perf_counter()
    value, score, diagnostics = evaluate(
        theta, observations, initial, process, design
    )
    return {
        "value": float(value.numpy()),
        "score": [float(item) for item in score.numpy().tolist()],
        "finite": bool(tf.math.is_finite(value).numpy())
        and bool(tf.reduce_all(tf.math.is_finite(score)).numpy()),
        "max_mean_residual": float(diagnostics["max_mean_residual"].numpy()),
        "max_row_residual": float(diagnostics["max_row_residual"].numpy()),
        "max_col_residual": float(diagnostics["max_col_residual"].numpy()),
        "score_increment_sum_residual": float(
            tf.reduce_max(
                tf.abs(tf.reduce_sum(diagnostics["score_increments"], axis=0) - score)
            ).numpy()
        ),
        "device": str(value.device),
        "elapsed_seconds": time.perf_counter() - started,
    }


def _pair_estimators(
    first: dict[str, object],
    second: dict[str, object],
    negative: dict[str, object],
) -> dict[str, dict[str, object]]:
    def average(left: dict[str, object], right: dict[str, object]) -> dict[str, object]:
        return {
            "value": 0.5 * (float(left["value"]) + float(right["value"])),
            "score": [
                0.5 * (float(a) + float(b))
                for a, b in zip(left["score"], right["score"])
            ],
            "complete_run_count": 2,
        }

    return {
        "standard": {
            "value": float(first["value"]),
            "score": [float(item) for item in first["score"]],
            "complete_run_count": 1,
        },
        "independent_pair": average(first, second),
        "antithetic_pair": average(first, negative),
    }


def _values(estimator: dict[str, object]) -> list[float]:
    return [float(estimator["value"]), *[float(item) for item in estimator["score"]]]


def _sample_variances(rows: list[list[float]]) -> list[float]:
    return [statistics.variance(row[index] for row in rows) for index in range(len(rows[0]))]


def _dataset_statistics(
    model: str,
    dataset_seed: int,
    independent_pair: list[list[float]],
    antithetic_pair: list[list[float]],
    standard: list[list[float]],
    oracle: list[float] | None = None,
) -> dict[str, object]:
    independent_variance = _sample_variances(independent_pair)
    antithetic_variance = _sample_variances(antithetic_pair)
    standard_variance = _sample_variances(standard)
    if oracle is None:
        oracle = [0.0] * len(independent_variance)

    def means(rows: list[list[float]]) -> list[float]:
        return [statistics.mean(row[index] for row in rows) for index in range(len(rows[0]))]

    def mse(rows: list[list[float]]) -> list[float]:
        return [
            statistics.mean((row[index] - oracle[index]) ** 2 for row in rows)
            for index in range(len(rows[0]))
        ]

    independent_mean = means(independent_pair)
    antithetic_mean = means(antithetic_pair)
    independent_mse = mse(independent_pair)
    antithetic_mse = mse(antithetic_pair)
    return {
        "model": model,
        "dataset_seed": dataset_seed,
        "oracle": oracle,
        "standard_variance": standard_variance,
        "independent_pair_variance": independent_variance,
        "antithetic_pair_variance": antithetic_variance,
        "log_variance_ratio_antithetic_over_independent_pair": [
            math.log((anti + VARIANCE_FLOOR) / (independent + VARIANCE_FLOOR))
            for independent, anti in zip(independent_variance, antithetic_variance)
        ],
        "independent_pair_mean_error": [
            value - truth for value, truth in zip(independent_mean, oracle)
        ],
        "antithetic_pair_mean_error": [
            value - truth for value, truth in zip(antithetic_mean, oracle)
        ],
        "independent_pair_mse": independent_mse,
        "antithetic_pair_mse": antithetic_mse,
        "mse_ratio_antithetic_over_independent_pair": [
            (anti + VARIANCE_FLOOR) / (independent + VARIANCE_FLOOR)
            for independent, anti in zip(independent_mse, antithetic_mse)
        ],
    }


def _interval(values: list[float], critical: float) -> dict[str, float]:
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    half_width = critical * sd / math.sqrt(len(values))
    return {
        "mean": mean,
        "sd": sd,
        "mcse": sd / math.sqrt(len(values)),
        "critical": critical,
        "lower": mean - half_width,
        "upper": mean + half_width,
    }


def _outer_summary(model: str, datasets: list[dict[str, object]]) -> dict[str, object]:
    critical = FAMILYWISE_CRITICAL[model]
    variance_rows = []
    bias_rows = []
    mse_rows = []
    for index, label in enumerate(LABELS[model]):
        log_ratios = [
            float(row["log_variance_ratio_antithetic_over_independent_pair"][index])
            for row in datasets
        ]
        interval = _interval(log_ratios, critical)
        lower_count = sum(value < 0.0 for value in log_ratios)
        tail_count = min(lower_count, len(log_ratios) - lower_count)
        exact_sign_p = min(
            1.0,
            2.0
            * sum(
                math.comb(len(log_ratios), count)
                for count in range(tail_count + 1)
            )
            / (2.0 ** len(log_ratios)),
        )
        variance_rows.append(
            {
                "label": label,
                "familywise_interval": interval,
                "geometric_ratio": math.exp(interval["mean"]),
                "datasets_with_lower_antithetic_variance": lower_count,
                "dataset_count": len(log_ratios),
                "exact_two_sided_sign_test_p": exact_sign_p,
                "coordinate_nomination": interval["upper"] < 0.0,
            }
        )
        bias_difference = [
            float(row["antithetic_pair_mean_error"][index])
            - float(row["independent_pair_mean_error"][index])
            for row in datasets
        ]
        bias_rows.append({"label": label, "pointwise_interval": _interval(bias_difference, 2.3646242515927844)})
        mse_ratios = [
            math.log(float(row["mse_ratio_antithetic_over_independent_pair"][index]))
            for row in datasets
        ]
        mse_interval = _interval(mse_ratios, critical)
        mse_rows.append(
            {
                "label": label,
                "familywise_log_interval": mse_interval,
                "geometric_ratio": math.exp(mse_interval["mean"]),
            }
        )
    return {
        "model": model,
        "outer_sampling_unit": "independent_dgp_dataset",
        "dataset_count": len(datasets),
        "variance_ratio_antithetic_over_independent_pair": variance_rows,
        "mean_error_difference_antithetic_minus_independent_pair": bias_rows,
        "mse_ratio_antithetic_over_independent_pair": mse_rows,
        "all_coordinates_nominated": all(row["coordinate_nomination"] for row in variance_rows),
        "any_coordinate_nominated": any(row["coordinate_nomination"] for row in variance_rows),
    }


def _lgssm_oracle(observations: tf.Tensor) -> list[float]:
    from docs.benchmarks.run_lgssm_cubature_genut_fp32 import _kalman_value_score

    value, score = _kalman_value_score(_theta("lgssm"), observations)
    return [float(value.numpy()), *[float(item) for item in score.numpy().tolist()]]


def _sv_dense_once(
    observations: tf.Tensor, order: int, radius: float
) -> tuple[float, list[float]]:
    from bayesfilter.highdim.sv_mixture_cut4 import (
        ExactTransformedSVSSM,
        exact_transformed_sv_scalar_dense_reference,
    )

    theta = tf.cast(_theta("sv"), tf.float64)
    raw = tf.exp(0.5 * tf.cast(observations, tf.float64))
    with tf.device("/CPU:0"):
        with tf.GradientTape() as tape:
            tape.watch(theta)
            result = exact_transformed_sv_scalar_dense_reference(
                ExactTransformedSVSSM(sigma=1.0), theta, raw, order=order, radius=radius
            )
        score = tape.gradient(result.log_likelihood, theta)
    if score is None:
        raise RuntimeError("SV dense diagnostic oracle score is unavailable")
    return float(result.log_likelihood.numpy()), [float(item) for item in score.numpy().tolist()]


def _sv_oracle(observations: tf.Tensor) -> tuple[list[float], dict[str, object]]:
    coarse = _sv_dense_once(observations, 257, 8.0)
    fine = _sv_dense_once(observations, 401, 10.0)
    differences = [fine[0] - coarse[0], *[b - a for a, b in zip(coarse[1], fine[1])]]
    if abs(differences[0]) > 5.0e-4 or max(abs(item) for item in differences[1:]) > 2.0e-3:
        raise RuntimeError("SV dense oracle refinement gate failed")
    return [fine[0], *fine[1]], {
        "coarse": [coarse[0], *coarse[1]],
        "fine": [fine[0], *fine[1]],
        "fine_minus_coarse": differences,
    }


def _representative_thetas(model: str) -> tuple[tf.Tensor, ...]:
    center = _theta(model)
    direction = (
        tf.constant([0.08, -0.06, 0.05, 0.03, -0.03], tf.float32)
        if model == "lgssm"
        else tf.constant([0.08, -0.06], tf.float32)
    )
    return tuple(center + offset * direction for offset in FD_OFFSETS)


def _finite_difference_error(
    evaluate: Any,
    theta: tf.Tensor,
    observations: tf.Tensor,
    initial: tf.Tensor,
    process: tf.Tensor,
    design: tf.Tensor,
) -> float:
    value, score, _ = evaluate(theta, observations, initial, process, design)
    del value
    relative_errors = []
    for relative_step, minimum_step in zip(
        FD_RELATIVE_STEPS, FD_MINIMUM_STEPS
    ):
        for index in range(int(theta.shape[0])):
            step = tf.maximum(
                tf.constant(minimum_step, tf.float32),
                tf.constant(relative_step, tf.float32) * tf.abs(theta[index]),
            )
            direction = tf.one_hot(index, int(theta.shape[0]), dtype=tf.float32)
            plus = evaluate(
                theta + step * direction, observations, initial, process, design
            )[0]
            minus = evaluate(
                theta - step * direction, observations, initial, process, design
            )[0]
            finite_difference = (plus - minus) / (2.0 * step)
            scale = tf.maximum(
                tf.maximum(tf.abs(finite_difference), tf.abs(score[index])),
                tf.constant(1.0e-2, tf.float32),
            )
            relative_errors.append(
                float(
                    (tf.abs(finite_difference - score[index]) / scale).numpy()
                )
            )
    return max(relative_errors)


def _tune_model(model: str) -> dict[str, object]:
    adapter = _model_adapter(model)
    design = _genut_design(model)
    partitions = ("calibration", "validation")
    candidates = []
    for controls in CONTROL_GRID:
        evaluate = _make_evaluator(model, adapter, controls)
        partition_objectives: dict[str, float] = {}
        maximum_residual = 0.0
        all_valid = True
        for partition in partitions:
            dataset_objectives = []
            for dataset_seed in DGP_SEEDS[model][partition]:
                observations = _generate_dataset(model, dataset_seed)["observations"]
                coordinate_samples = [[] for _ in LABELS[model]]
                for particle_seed in TUNING_PARTICLE_SEEDS:
                    initial, process = _particle_noise(model, particle_seed)
                    row = _evaluate_constituent(
                        evaluate, _theta(model), observations, initial, process, design
                    )
                    values = [float(row["value"]), *[float(item) for item in row["score"]]]
                    for index, item in enumerate(values):
                        coordinate_samples[index].append(item)
                    maximum_residual = max(
                        maximum_residual,
                        float(row["max_mean_residual"]),
                        float(row["max_row_residual"]),
                        float(row["max_col_residual"]),
                        float(row["score_increment_sum_residual"]),
                    )
                    all_valid = all_valid and bool(row["finite"]) and "GPU" in str(row["device"]).upper()
                dataset_objectives.append(
                    max(
                        statistics.variance(samples)
                        / (VARIANCE_SCALES[model][index] ** 2)
                        for index, samples in enumerate(coordinate_samples)
                    )
                )
            partition_objectives[partition] = statistics.mean(dataset_objectives)
        audit_observations = _generate_dataset(model, DGP_SEEDS[model]["calibration"][0])["observations"]
        audit_initial, audit_process = _particle_noise(model, TUNING_PARTICLE_SEEDS[0])
        fd_error = max(
            _finite_difference_error(
                evaluate,
                theta,
                audit_observations,
                audit_initial,
                audit_process,
                design,
            )
            for theta in _representative_thetas(model)
        )
        eligible = (
            all_valid
            and maximum_residual < RESIDUAL_TOLERANCE
            and fd_error < FD_RELATIVE_TOLERANCE
        )
        candidates.append(
            {
                "controls": controls,
                "partition_objectives": partition_objectives,
                "maximum_residual": maximum_residual,
                "maximum_representative_fd_relative_error": fd_error,
                "eligible": eligible,
            }
        )
    eligible = [row for row in candidates if row["eligible"]]
    if not eligible:
        raise RuntimeError(f"no eligible tuning candidate for {model}")
    selected = min(
        eligible,
        key=lambda row: (
            row["partition_objectives"]["validation"],
            row["partition_objectives"]["calibration"],
            row["maximum_representative_fd_relative_error"],
            int(row["controls"]["sinkhorn_steps"]),
            -float(row["controls"]["ridge"]),
            float(row["controls"]["epsilon"]),
        ),
    )
    return {
        "model": model,
        "scope": {
            "horizon": HORIZON,
            "particle_count": PARTICLE_COUNTS[model],
            "dtype": "float32",
            "tf32": True,
            "jit_compile": True,
            "design": "gaussian_genut",
            "score_route": "recursive_forward_sensitivity_no_autodiff",
        },
        "partitions": DGP_SEEDS[model],
        "particle_seeds": TUNING_PARTICLE_SEEDS,
        "selection_does_not_use_oracle_value_or_score": True,
        "candidates": candidates,
        "selected_controls": selected["controls"],
        "selected_candidate": selected,
    }


def _claim_model(model: str, tuning: dict[str, object]) -> dict[str, object]:
    controls = dict(tuning["selected_controls"])
    evaluate = _make_evaluator(model, _model_adapter(model), controls)
    design = _genut_design(model)
    datasets = []
    raw = []
    oracle_refinements: dict[str, object] = {}
    for dataset_seed in DGP_SEEDS[model]["claim"]:
        generated = _generate_dataset(model, dataset_seed)
        observations = generated["observations"]
        if model == "lgssm":
            oracle = _lgssm_oracle(observations)
        else:
            oracle, refinement = _sv_oracle(observations)
            oracle_refinements[str(dataset_seed)] = refinement
        standard_values = []
        independent_values = []
        antithetic_values = []
        for particle_seed in CLAIM_PARTICLE_SEEDS:
            initial, process = _particle_noise(model, particle_seed)
            second_initial, second_process = _particle_noise(model, particle_seed + 10000)
            first = _evaluate_constituent(
                evaluate, _theta(model), observations, initial, process, design
            )
            second = _evaluate_constituent(
                evaluate, _theta(model), observations, second_initial, second_process, design
            )
            negative = _evaluate_constituent(
                evaluate, _theta(model), observations, -initial, -process, design
            )
            constituents = {"z1": first, "z2": second, "negative_z1": negative}
            if not all(
                bool(item["finite"])
                and "GPU" in str(item["device"]).upper()
                and max(
                    float(item["max_mean_residual"]),
                    float(item["max_row_residual"]),
                    float(item["max_col_residual"]),
                    float(item["score_increment_sum_residual"]),
                ) < RESIDUAL_TOLERANCE
                for item in constituents.values()
            ):
                raise RuntimeError(f"claim constituent veto for {model}")
            estimators = _pair_estimators(first, second, negative)
            standard_values.append(_values(estimators["standard"]))
            independent_values.append(_values(estimators["independent_pair"]))
            antithetic_values.append(_values(estimators["antithetic_pair"]))
            raw.append(
                {
                    "model": model,
                    "dataset_seed": dataset_seed,
                    "particle_seed": particle_seed,
                    "independent_partner_seed": particle_seed + 10000,
                    "constituents": constituents,
                    "estimators": estimators,
                }
            )
        datasets.append(
            _dataset_statistics(
                model,
                dataset_seed,
                independent_values,
                antithetic_values,
                standard_values,
                oracle,
            )
        )
    return {
        "model": model,
        "controls": controls,
        "dgp_dataset_seeds": DGP_SEEDS[model]["claim"],
        "particle_seeds": CLAIM_PARTICLE_SEEDS,
        "dataset_statistics": datasets,
        "outer_summary": _outer_summary(model, datasets),
        "oracle_refinements": oracle_refinements,
        "raw": raw,
    }


def _dataset_manifest(model: str, seed: int) -> dict[str, object]:
    generated = _generate_dataset(model, seed)
    return {
        "model": model,
        "seed": seed,
        "observation_sha256": _tensor_sha256(generated["observations"]),
        "state_sha256": _tensor_sha256(generated["states"]),
        "initial_law": "stationary_gaussian_before_first_transition",
        "timing": "x_minus1_draw_then_transition_to_x0_then_observe_y0",
        "transition_seed": [seed, 2],
        "observation_seed": [seed, 3],
        "equations": (
            "x_t=diag(phi)x_(t-1)+q_scale*eta_t; y_t=H*x_t+r_scale*epsilon_t"
            if model == "lgssm"
            else "h_t=gamma*h_(t-1)+eta_t; z_t=h_t+2log(beta)+log(epsilon_t^2)"
        ),
    }


def _write_result_note(payload: dict[str, object]) -> None:
    lines = [
        "# GenUT Antithetic LGSSM And SV Result",
        "",
        "Date: 2026-07-22",
        "",
        f"Status: `{payload['decision']['status']}`",
        "",
        "The primary comparator is an equal-cost average of two independent complete",
        "GenUT runs. The single-cloud arm is descriptive only.",
        "",
    ]
    for model in MODELS:
        claim = payload["claims"][model]
        lines.extend(
            [
                f"## {model.upper()}",
                "",
                f"Selected controls: `{claim['controls']}`.",
                "",
                "| Coordinate | Geometric variance ratio | Familywise 95% log-ratio CI | Datasets lower | Nominated |",
                "|---|---:|---:|---:|---|",
            ]
        )
        for row in claim["outer_summary"]["variance_ratio_antithetic_over_independent_pair"]:
            interval = row["familywise_interval"]
            lines.append(
                f"| {row['label']} | {row['geometric_ratio']:.4f} | "
                f"[{interval['lower']:.4f}, {interval['upper']:.4f}] | "
                f"{row['datasets_with_lower_antithetic_variance']}/{row['dataset_count']} | "
                f"{row['coordinate_nomination']} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Decision",
            "",
            payload["decision"]["text"],
            "",
            "This feasibility campaign does not change the default. Dataset-level mean",
            "error and MSE diagnostics are retained in `result.json`.",
        ]
    )
    RESULT_NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
        raise RuntimeError("campaign requires a logical GPU")
    tuning = {model: _tune_model(model) for model in MODELS}
    (output_root / "tuning.json").write_text(
        json.dumps(tuning, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    claims = {model: _claim_model(model, tuning[model]) for model in MODELS}
    for model in MODELS:
        (output_root / f"raw_{model}.json").write_text(
            json.dumps(claims[model].pop("raw"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    allocator = tf.config.experimental.get_memory_info("GPU:0")
    any_nomination = any(
        claims[model]["outer_summary"]["any_coordinate_nominated"] for model in MODELS
    )
    all_nomination = all(
        claims[model]["outer_summary"]["all_coordinates_nominated"] for model in MODELS
    )
    if all_nomination:
        status = "ANTITHETIC_ALL_COORDINATES_NOMINATED_FEASIBILITY_ONLY"
        text = "Antithetic averaging reduced conditional variance on every coordinate under the equal-cost screen. Replication at larger dataset count is required before any default proposal."
    elif any_nomination:
        status = "ANTITHETIC_PARTIAL_COORDINATE_NOMINATION_FEASIBILITY_ONLY"
        text = "Antithetic averaging reduced conditional variance only for a subset of coordinates under the equal-cost screen. It remains an optional experimental coupling."
    else:
        status = "ANTITHETIC_NOT_NOMINATED_AGAINST_EQUAL_COST_BASELINE"
        text = "No coordinate passed the equal-cost dataset-level variance screen. Antithetic averaging is not nominated as the default."
    source_paths = (
        Path(__file__).relative_to(ROOT),
        Path("bayesfilter/highdim/cubature_genut_adapters.py"),
        Path("bayesfilter/highdim/cubature_genut_filter.py"),
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
        "device": {
            "logical_devices": [item.name for item in logical],
            "dtype": "float32",
            "tf32_enabled": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
            "jit_compile": True,
            "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        },
        "memory_policy": memory_policy,
        "gpu_allocator": {key: int(value) for key, value in allocator.items()},
        "plan": PLAN.as_posix(),
        "source_sha256": {path.as_posix(): _sha256(ROOT / path) for path in source_paths},
        "configuration": {
            "horizon": HORIZON,
            "particle_counts": PARTICLE_COUNTS,
            "parameter_points": {"lgssm": LGSSM_THETA, "sv": SV_THETA},
            "control_grid": CONTROL_GRID,
            "dgp_seeds": DGP_SEEDS,
            "tuning_particle_seeds": TUNING_PARTICLE_SEEDS,
            "claim_particle_seeds": CLAIM_PARTICLE_SEEDS,
            "equal_cost_primary_comparator": "independent_pair_average",
            "candidate": "antithetic_pair_average",
            "single_cloud_role": "explanatory_only",
            "runtime_score": "recursive_forward_sensitivity_no_autodiff_no_fd",
            "tuning_fd_audit_relative_steps": FD_RELATIVE_STEPS,
            "tuning_fd_audit_minimum_steps": FD_MINIMUM_STEPS,
        },
        "dataset_manifest": [
            _dataset_manifest(model, seed)
            for model in MODELS
            for partition in ("calibration", "validation", "claim")
            for seed in DGP_SEEDS[model][partition]
        ],
        "tuning": tuning,
        "claims": claims,
        "hard_valid": True,
        "decision": {"status": status, "text": text, "default_changed": False},
        "nonclaims": [
            "no default promotion",
            "no unbiasedness claim",
            "no MLE or HMC performance claim",
            "no NAWM or broad nonlinear-model claim",
            "LGSSM Gaussian GenUT equals cubature in dimension three",
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
        "data_version": "generated_dgp_seeded_and_hashed",
        "random_seeds": DGP_SEEDS | {"particle": CLAIM_PARTICLE_SEEDS},
        "wall_time_seconds": payload["wall_time_seconds"],
        "output_artifact_paths": [result_path.as_posix(), (output_root / "tuning.json").as_posix()],
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
    print(json.dumps({"status": payload["decision"]["status"], "wall_time_seconds": payload["wall_time_seconds"]}, sort_keys=True))


if __name__ == "__main__":
    main()
