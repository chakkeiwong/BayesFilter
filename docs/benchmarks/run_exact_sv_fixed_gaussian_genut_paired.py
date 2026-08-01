#!/usr/bin/env python3
"""Run the reviewed paired Cubature/fixed-Gaussian-GenUT exact-SV study."""

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


PLAN = Path(
    "docs/plans/bayesfilter-exact-sv-fixed-gaussian-genut-paired-comparison-plan-2026-07-21.md"
)
HORIZON = 50
PARTICLE_COUNTS = (1002, 1998)
SINKHORN_STEPS = (4, 8)
DESIGN_FAMILIES = ("cubature", "fixed_gaussian_genut")
CLAIM_SEEDS = tuple(range(3620, 3636))
THETA_VALUES = (0.25, -0.15)
EPSILON = 2.0
BALANCE_STEPS = 8
RIDGE = 1.0e-5
POINTWISE_T_95_DF15 = 2.131449545559323
SCORE_FAMILYWISE_T_95_DF15 = 2.4898797034798923
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260721
VALUE_HALF_WIDTH_BUDGET = 0.25


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_sha256(value: tf.Tensor) -> str:
    return hashlib.sha256(tf.io.serialize_tensor(value).numpy()).hexdigest()


def _historical_nondgp_observations() -> tf.Tensor:
    with tf.device("/CPU:0"):
        return tf.random.stateless_normal(
            [HORIZON, 1], [7000 + HORIZON, 17], dtype=tf.float32
        )


def _fresh_dgp_observations(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Generate one exact stationary transformed-SV sequence on CPU."""

    theta = tf.convert_to_tensor(theta, tf.float32)
    gamma = 0.5 * (
        1.0 + tf.math.erf(theta[0] / tf.sqrt(tf.constant(2.0, tf.float32)))
    )
    beta = tf.exp(theta[1])
    initial = tf.random.stateless_normal([], [20260721, 201], dtype=tf.float32)
    innovations = tf.random.stateless_normal(
        [HORIZON], [20260721, 202], dtype=tf.float32
    )
    observation_noise = tf.random.stateless_normal(
        [HORIZON], [20260721, 203], dtype=tf.float32
    )
    x0 = initial / tf.sqrt(1.0 - tf.square(gamma))
    states = tf.TensorArray(tf.float32, size=HORIZON, element_shape=())

    def body(index: tf.Tensor, previous: tf.Tensor, output: tf.TensorArray):
        state = tf.where(
            tf.equal(index, 0), x0, gamma * previous + innovations[index]
        )
        return index + 1, state, output.write(index, state)

    _, _, states = tf.while_loop(
        lambda index, *_: index < HORIZON,
        body,
        (tf.zeros([], tf.int32), tf.zeros([], tf.float32), states),
        parallel_iterations=1,
    )
    state_path = states.stack()
    transformed = (
        state_path
        + 2.0 * tf.math.log(beta)
        + tf.math.log(tf.square(observation_noise))
    )
    return transformed[:, None], state_path[:, None]


def _nested_noise(particle_count: int, seed: int) -> tuple[tf.Tensor, tf.Tensor]:
    maximum = max(PARTICLE_COUNTS)
    with tf.device("/CPU:0"):
        initial = tf.random.stateless_normal(
            [maximum, 1], [seed, 101], dtype=tf.float32
        )[:particle_count]
        process = tf.random.stateless_normal(
            [HORIZON, maximum, 1], [seed, 102], dtype=tf.float32
        )[:, :particle_count]
    return initial, process


def paired_rank_designs(particle_count: int, seed: int) -> tuple[tf.Tensor, tf.Tensor]:
    """Return exact equal-weight Cubature and Gaussian-GenUT rows by common rank."""

    if particle_count < 1 or particle_count % 6 != 0:
        raise ValueError("paired scalar designs require N divisible by six")
    ranks = tf.argsort(
        tf.random.stateless_uniform([particle_count], [seed, 103], dtype=tf.float32),
        stable=True,
    )
    inverse = tf.argsort(ranks, stable=True)
    cubature_sorted = tf.concat(
        [
            -tf.ones([particle_count // 2], tf.float32),
            tf.ones([particle_count // 2], tf.float32),
        ],
        axis=0,
    )
    tail = particle_count // 6
    genut_sorted = tf.concat(
        [
            -tf.sqrt(tf.constant(3.0, tf.float32)) * tf.ones([tail], tf.float32),
            tf.zeros([4 * tail], tf.float32),
            tf.sqrt(tf.constant(3.0, tf.float32)) * tf.ones([tail], tf.float32),
        ],
        axis=0,
    )
    return (
        tf.gather(cubature_sorted, inverse)[:, None],
        tf.gather(genut_sorted, inverse)[:, None],
    )


def alternating_cubature_design(particle_count: int) -> tf.Tensor:
    if particle_count < 1 or particle_count % 2 != 0:
        raise ValueError("alternating scalar Cubature requires even N")
    return tf.tile(tf.constant([[-1.0], [1.0]], tf.float32), [particle_count // 2, 1])


def design_moments(design: tf.Tensor) -> dict[str, float]:
    values = tf.reshape(tf.convert_to_tensor(design, tf.float32), [-1])
    return {
        f"moment_{order}": float(tf.reduce_mean(tf.pow(values, order)).numpy())
        for order in range(1, 5)
    }


def _make_candidate(adapter: Any, *, particle_count: int, sinkhorn_steps: int):
    from bayesfilter.highdim.cubature_genut_filter import finite_value_score

    @tf.function(jit_compile=True, reduce_retracing=True)
    def evaluate(theta, observations, initial, process, design):
        initial = tf.ensure_shape(initial, [particle_count, 1])
        process = tf.ensure_shape(process, [HORIZON, particle_count, 1])
        design = tf.ensure_shape(design, [particle_count, 1])
        with tf.device("/GPU:0"):
            return finite_value_score(
                adapter,
                theta,
                observations,
                initial,
                process,
                design,
                epsilon=EPSILON,
                sinkhorn_steps=sinkhorn_steps,
                balance_steps=BALANCE_STEPS,
                ridge=RIDGE,
            )

    return evaluate


def _dense_reference(
    observations: tf.Tensor, theta: tf.Tensor, *, order: int, radius: float
) -> dict[str, object]:
    from bayesfilter.highdim.sv_mixture_cut4 import (
        ExactTransformedSVSSM,
        exact_transformed_sv_scalar_dense_reference,
    )

    theta64 = tf.cast(theta, tf.float64)
    raw = tf.exp(0.5 * tf.cast(observations, tf.float64))
    with tf.device("/CPU:0"):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(theta64)
            result = exact_transformed_sv_scalar_dense_reference(
                ExactTransformedSVSSM(sigma=1.0),
                theta64,
                raw,
                order=order,
                radius=radius,
            )
        score = tape.gradient(result.log_likelihood, theta64)
        score_increments = tape.jacobian(result.log_normalizers, theta64)
        del tape
    if score is None or score_increments is None:
        raise RuntimeError("dense exact-SV diagnostic differentiation failed")
    return {
        "order": order,
        "radius": radius,
        "value": float(result.log_likelihood.numpy()),
        "score": [float(value) for value in score.numpy().tolist()],
        "value_increments": [float(value) for value in result.log_normalizers.numpy().tolist()],
        "score_increments": [
            [float(value) for value in row]
            for row in score_increments.numpy().tolist()
        ],
        "score_increment_sum_residual": float(
            tf.reduce_max(tf.abs(tf.reduce_sum(score_increments, axis=0) - score)).numpy()
        ),
    }


def _refined_dense(observations: tf.Tensor, theta: tf.Tensor) -> dict[str, object]:
    refinements = [
        _dense_reference(observations, theta, order=257, radius=8.0),
        _dense_reference(observations, theta, order=401, radius=8.0),
        _dense_reference(observations, theta, order=401, radius=10.0),
    ]
    reference = refinements[-1]
    comparisons = [
        {
            "order": item["order"],
            "radius": item["radius"],
            "value_difference": float(item["value"]) - float(reference["value"]),
            "score_difference": [
                float(item["score"][index]) - float(reference["score"][index])
                for index in range(2)
            ],
        }
        for item in refinements[:-1]
    ]
    valid = all(
        abs(float(item["value_difference"])) <= 5.0e-5
        and max(abs(float(value)) for value in item["score_difference"]) <= 2.0e-4
        for item in comparisons
    ) and all(float(item["score_increment_sum_residual"]) <= 1.0e-10 for item in refinements)
    if not valid:
        raise RuntimeError("dense exact-SV refinement gate failed")
    return {"refinements": refinements, "comparisons": comparisons, "valid": valid}


def _row(
    evaluate: Any,
    theta: tf.Tensor,
    observations: tf.Tensor,
    initial: tf.Tensor,
    process: tf.Tensor,
    design: tf.Tensor,
    dense: dict[str, object],
    *,
    dataset: str,
    design_family: str,
    particle_count: int,
    sinkhorn_steps: int,
    seed: int,
    include_fd: bool,
) -> dict[str, object]:
    try:
        tf.config.experimental.reset_memory_stats("GPU:0")
    except (AttributeError, RuntimeError, tf.errors.InvalidArgumentError):
        pass
    started = time.perf_counter()
    value, score, diagnostics = evaluate(theta, observations, initial, process, design)
    fd: list[float] = []
    if include_fd:
        step = tf.constant(2.0e-3, tf.float32)
        for index in range(2):
            plus = tf.tensor_scatter_nd_add(theta, [[index]], [step])
            minus = tf.tensor_scatter_nd_sub(theta, [[index]], [step])
            plus_value = evaluate(plus, observations, initial, process, design)[0]
            minus_value = evaluate(minus, observations, initial, process, design)[0]
            fd.append(float(((plus_value - minus_value) / (2.0 * step)).numpy()))
    memory = tf.config.experimental.get_memory_info("GPU:0")
    score_values = [float(item) for item in score.numpy().tolist()]
    score_increments = [
        [float(item) for item in row]
        for row in diagnostics["score_increments"].numpy().tolist()
    ]
    dense_score = list(dense["score"])
    return {
        "dataset": dataset,
        "design_family": design_family,
        "particle_count": particle_count,
        "sinkhorn_steps": sinkhorn_steps,
        "seed": seed,
        "value": float(value.numpy()),
        "score": score_values,
        "value_error_to_dense": float(value.numpy()) - float(dense["value"]),
        "score_error_to_dense": [
            score_values[index] - float(dense_score[index]) for index in range(2)
        ],
        "value_increments": [
            float(item) for item in diagnostics["value_increments"].numpy().tolist()
        ],
        "score_increments": score_increments,
        "score_increment_error_to_dense": [
            [
                score_increments[time_index][index]
                - float(dense["score_increments"][time_index][index])
                for index in range(2)
            ]
            for time_index in range(HORIZON)
        ],
        "score_increment_sum_residual": max(
            abs(sum(item[index] for item in score_increments) - score_values[index])
            for index in range(2)
        ),
        "fd_probe_score": fd,
        "fd_max_abs_error": (
            max(abs(score_values[index] - fd[index]) for index in range(2))
            if include_fd
            else 0.0
        ),
        "max_mean_residual": float(diagnostics["max_mean_residual"].numpy()),
        "max_row_residual": float(diagnostics["max_row_residual"].numpy()),
        "max_col_residual": float(diagnostics["max_col_residual"].numpy()),
        "finite": bool(tf.math.is_finite(value).numpy())
        and bool(tf.reduce_all(tf.math.is_finite(score)).numpy()),
        "device": str(value.device),
        "gpu_placement": "GPU" in str(value.device).upper(),
        "gpu_allocator": {key: int(item) for key, item in memory.items()},
        "elapsed_seconds": time.perf_counter() - started,
    }


def _interval(values: list[float], *, critical: float) -> dict[str, float]:
    if len(values) != len(CLAIM_SEEDS):
        raise ValueError("interval requires exactly 16 complete-run seeds")
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    mcse = sd / math.sqrt(len(values))
    half_width = critical * mcse
    return {
        "mean": mean,
        "sd": sd,
        "mcse": mcse,
        "half_width": half_width,
        "lower": mean - half_width,
        "upper": mean + half_width,
        "minimum": min(values),
        "maximum": max(values),
    }


def _opg_payload(rows: list[dict[str, object]], dense: dict[str, object]) -> dict[str, object]:
    from bayesfilter.score_diagnostics_tf import tf_score_comparison_diagnostics

    scores = tf.constant([row["score"] for row in rows], tf.float64)
    increments = tf.constant(dense["score_increments"], tf.float64)
    kwargs = {
        "reference_score_increments": increments,
        "diagonal_shrinkage": 0.0,
        "base_ridge": 1.0,
        "ridge_floor": 0.0,
        "ridge_scale_diagonal": tf.ones([2], tf.float64),
    }
    per_seed = tf_score_comparison_diagnostics(candidate_score=scores, **kwargs)
    mean = tf_score_comparison_diagnostics(
        candidate_score=tf.reduce_mean(scores, axis=0), **kwargs
    )
    return {
        "construction": "regularized_average_predictive_score_opg",
        "settings": {
            "diagonal_shrinkage": 0.0,
            "base_ridge": 1.0,
            "ridge_floor": 0.0,
            "ridge_scale_diagonal": [1.0, 1.0],
            "scientific_status": "predeclared_descriptive_only_not_exact_fisher",
        },
        "total_metric": mean.total_metric.numpy().tolist(),
        "total_metric_eigenvalues": mean.total_metric_eigenvalues.numpy().tolist(),
        "condition_proxy": float(mean.total_metric_condition_proxy.numpy()),
        "mean_score_rms_metric_error": float(mean.rms_total_metric_error.numpy()),
        "mean_score_max_diagonal_standardized_error": float(
            mean.maximum_diagonal_standardized_error.numpy()
        ),
        "per_seed_rms_metric_error": per_seed.rms_total_metric_error.numpy().tolist(),
        "mean_per_seed_rms_metric_error": float(
            tf.reduce_mean(per_seed.rms_total_metric_error).numpy()
        ),
    }


def _summarize(rows: list[dict[str, object]], dense: dict[str, object]) -> dict[str, object]:
    value_errors = [float(row["value_error_to_dense"]) for row in rows]
    score_errors = [
        [float(row["score_error_to_dense"][index]) for row in rows]
        for index in range(2)
    ]
    value_interval = _interval(value_errors, critical=POINTWISE_T_95_DF15)
    pointwise = [
        _interval(values, critical=POINTWISE_T_95_DF15) for values in score_errors
    ]
    familywise = [
        _interval(values, critical=SCORE_FAMILYWISE_T_95_DF15)
        for values in score_errors
    ]
    engineering_valid = all(
        bool(row["finite"])
        and bool(row["gpu_placement"])
        and float(row["max_row_residual"]) < 1.0e-2
        and float(row["max_col_residual"]) < 1.0e-2
        and float(row["max_mean_residual"]) < 1.0e-2
        and float(row["score_increment_sum_residual"]) < 1.0e-3
        for row in rows
    )
    return {
        "mean_value": float(dense["value"]) + value_interval["mean"],
        "mean_score": [
            float(dense["score"][index]) + pointwise[index]["mean"]
            for index in range(2)
        ],
        "value_error_interval": value_interval,
        "score_error_pointwise_intervals": pointwise,
        "score_error_familywise_intervals": familywise,
        "opg_diagnostic": _opg_payload(rows, dense),
        "engineering_valid": engineering_valid,
        "value_preservation_valid": (
            value_interval["lower"] <= 0.0 <= value_interval["upper"]
            and value_interval["half_width"] <= VALUE_HALF_WIDTH_BUDGET
        ),
        "maximum_fd_error": max(float(row["fd_max_abs_error"]) for row in rows),
        "maximum_row_residual": max(float(row["max_row_residual"]) for row in rows),
        "maximum_col_residual": max(float(row["max_col_residual"]) for row in rows),
        "maximum_allocator_peak_bytes": max(
            int(row["gpu_allocator"]["peak"]) for row in rows
        ),
        "mean_elapsed_seconds": statistics.mean(
            float(row["elapsed_seconds"]) for row in rows
        ),
    }


def _paired_bootstrap(
    cubature: list[float],
    genut: list[float],
    *,
    statistic: str,
    seed_offset: int,
) -> dict[str, object]:
    if len(cubature) != len(genut) or len(cubature) != len(CLAIM_SEEDS):
        raise ValueError("paired bootstrap requires 16 paired complete runs")
    if statistic == "absolute_mean_error":
        function = lambda a, b: abs(statistics.mean(b)) - abs(statistics.mean(a))
    elif statistic == "mean":
        function = lambda a, b: statistics.mean(b[index] - a[index] for index in range(len(a)))
    else:
        raise ValueError("unsupported paired statistic")
    observed = function(cubature, genut)
    generator = random.Random(BOOTSTRAP_SEED + seed_offset)
    values = []
    for _ in range(BOOTSTRAP_REPLICATES):
        indices = [generator.randrange(len(cubature)) for _ in range(len(cubature))]
        values.append(
            function(
                [cubature[index] for index in indices],
                [genut[index] for index in indices],
            )
        )
    values.sort()
    lower = values[int(0.025 * len(values))]
    upper = values[int(0.975 * len(values)) - 1]
    return {
        "statistic": (
            "abs_mean_genut_error_minus_abs_mean_cubature_error"
            if statistic == "absolute_mean_error"
            else "mean_genut_minus_cubature"
        ),
        "observed": observed,
        "lower": lower,
        "upper": upper,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED + seed_offset,
        "genut_reduction_supported": upper < 0.0,
        "genut_increase_supported": lower > 0.0,
    }


def _paired_comparison(
    cubature_rows: list[dict[str, object]],
    genut_rows: list[dict[str, object]],
    cubature_summary: dict[str, object],
    genut_summary: dict[str, object],
    *,
    seed_offset: int,
) -> dict[str, object]:
    value = _paired_bootstrap(
        [float(row["value_error_to_dense"]) for row in cubature_rows],
        [float(row["value_error_to_dense"]) for row in genut_rows],
        statistic="absolute_mean_error",
        seed_offset=seed_offset,
    )
    scores = [
        _paired_bootstrap(
            [float(row["score_error_to_dense"][index]) for row in cubature_rows],
            [float(row["score_error_to_dense"][index]) for row in genut_rows],
            statistic="absolute_mean_error",
            seed_offset=seed_offset + 10 + index,
        )
        for index in range(2)
    ]
    opg = _paired_bootstrap(
        list(cubature_summary["opg_diagnostic"]["per_seed_rms_metric_error"]),
        list(genut_summary["opg_diagnostic"]["per_seed_rms_metric_error"]),
        statistic="mean",
        seed_offset=seed_offset + 20,
    )
    return {"value": value, "scores": scores, "per_seed_opg": opg}


def _scope_key(dataset: str, particle_count: int, steps: int, design: str) -> str:
    return f"{dataset}_n{particle_count}_s{steps}_{design}"


def run(
    output_root: Path, *, include_historical_nondgp_engineering_only: bool = False
) -> dict[str, object]:
    started = time.perf_counter()
    output_root.mkdir(parents=True, exist_ok=False)
    physical = tf.config.list_physical_devices("GPU")
    if not physical:
        raise RuntimeError("paired exact-SV GenUT study requires a visible GPU")
    tf.config.set_soft_device_placement(False)
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    with tf.device("/CPU:0"):
        theta = tf.constant(THETA_VALUES, tf.float32)
        fresh, fresh_states = _fresh_dgp_observations(theta)
        historical = (
            _historical_nondgp_observations()
            if include_historical_nondgp_engineering_only
            else None
        )
    observations = {"fresh_dgp": fresh}
    if historical is not None:
        observations["historical_nondgp_engineering_only"] = historical
    observation_payload = {
        name: {
            "sha256": _tensor_sha256(value),
            "values": [float(item) for item in tf.reshape(value, [-1]).numpy().tolist()],
        }
        for name, value in observations.items()
    }
    observation_payload["fresh_dgp"]["latent_states"] = [
        float(item) for item in tf.reshape(fresh_states, [-1]).numpy().tolist()
    ]
    (output_root / "observations.json").write_text(
        json.dumps(observation_payload, indent=2, sort_keys=True) + "\n"
    )

    dense_payload = {
        name: _refined_dense(value, theta) for name, value in observations.items()
    }
    (output_root / "dense_references.json").write_text(
        json.dumps(dense_payload, indent=2, sort_keys=True) + "\n"
    )
    dense = {name: payload["refinements"][-1] for name, payload in dense_payload.items()}

    from bayesfilter.highdim.cubature_genut_adapters import exact_transformed_sv_candidate_adapter

    adapter = exact_transformed_sv_candidate_adapter()
    evaluators = {
        (particle_count, steps): _make_candidate(
            adapter, particle_count=particle_count, sinkhorn_steps=steps
        )
        for particle_count in PARTICLE_COUNTS
        for steps in SINKHORN_STEPS
    }
    rows: dict[str, list[dict[str, object]]] = {}
    summaries: dict[str, dict[str, object]] = {}
    designs_manifest: dict[str, object] = {}

    for dataset_index, (dataset, data) in enumerate(observations.items()):
        for particle_count in PARTICLE_COUNTS:
            for steps in SINKHORN_STEPS:
                scoped_rows = {family: [] for family in DESIGN_FAMILIES}
                for seed_index, seed in enumerate(CLAIM_SEEDS):
                    initial, process = _nested_noise(particle_count, seed)
                    cubature, genut = paired_rank_designs(particle_count, seed)
                    if seed_index == 0:
                        designs_manifest[f"n{particle_count}_seed{seed}"] = {
                            "coupling": "common_stateless_rank_quantiles",
                            "cubature": design_moments(cubature),
                            "fixed_gaussian_genut": design_moments(genut),
                        }
                    include_fd = particle_count == min(PARTICLE_COUNTS) and seed_index == 0
                    for family, design in (
                        ("cubature", cubature),
                        ("fixed_gaussian_genut", genut),
                    ):
                        scoped_rows[family].append(
                            _row(
                                evaluators[(particle_count, steps)],
                                theta,
                                data,
                                initial,
                                process,
                                design,
                                dense[dataset],
                                dataset=dataset,
                                design_family=family,
                                particle_count=particle_count,
                                sinkhorn_steps=steps,
                                seed=seed,
                                include_fd=include_fd,
                            )
                        )
                for family in DESIGN_FAMILIES:
                    key = _scope_key(dataset, particle_count, steps, family)
                    rows[key] = scoped_rows[family]
                    summaries[key] = _summarize(scoped_rows[family], dense[dataset])

    comparisons: dict[str, object] = {}
    for dataset_index, dataset in enumerate(observations):
        for particle_index, particle_count in enumerate(PARTICLE_COUNTS):
            for step_index, steps in enumerate(SINKHORN_STEPS):
                cubature_key = _scope_key(dataset, particle_count, steps, "cubature")
                genut_key = _scope_key(dataset, particle_count, steps, "fixed_gaussian_genut")
                comparison_key = f"{dataset}_n{particle_count}_s{steps}"
                comparisons[comparison_key] = _paired_comparison(
                    rows[cubature_key],
                    rows[genut_key],
                    summaries[cubature_key],
                    summaries[genut_key],
                    seed_offset=1000 * dataset_index + 100 * particle_index + 10 * step_index,
                )

    n_high = max(PARTICLE_COUNTS)
    eligible_datasets = ("fresh_dgp",)
    dgp_score_reduction_each_step = all(
        any(
            comparisons[f"{dataset}_n{n_high}_s{steps}"]["scores"][index][
                "genut_reduction_supported"
            ]
            for index in range(2)
        )
        for dataset in eligible_datasets
        for steps in SINKHORN_STEPS
    )
    value_preservation = all(
        summaries[_scope_key(dataset, n_high, steps, family)]["value_preservation_valid"]
        for dataset in eligible_datasets
        for steps in SINKHORN_STEPS
        for family in DESIGN_FAMILIES
    )
    engineering_valid = all(summary["engineering_valid"] for summary in summaries.values())
    fd_valid = all(summary["maximum_fd_error"] <= 0.05 for summary in summaries.values())

    manifest = {
        "schema_version": "bayesfilter.exact_sv_fixed_gaussian_genut_manifest.v1",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "command": list(sys.argv),
        "host": platform.node(),
        "python": sys.version,
        "tensorflow": tf.__version__,
        "physical_devices": [device.name for device in physical],
        "memory_policy": dict(memory_policy),
        "dtype": "float32",
        "tf32_enabled": True,
        "jit_compile": True,
        "horizon": HORIZON,
        "particle_counts": list(PARTICLE_COUNTS),
        "sinkhorn_steps": list(SINKHORN_STEPS),
        "epsilon": EPSILON,
        "ridge": RIDGE,
        "claim_seeds": list(CLAIM_SEEDS),
        "theta_realized": [float(item) for item in theta.numpy().tolist()],
        "plan": str((ROOT / PLAN).resolve()),
        "source_sha256": {
            "runner": _sha256(Path(__file__).resolve()),
            "filter": _sha256(ROOT / "bayesfilter/highdim/cubature_genut_filter.py"),
            "adapter": _sha256(ROOT / "bayesfilter/highdim/cubature_genut_adapters.py"),
            "design": _sha256(ROOT / "bayesfilter/highdim/cubature_genut_candidate.py"),
            "plan": _sha256(ROOT / PLAN),
        },
    }
    payload = {
        "schema_version": "bayesfilter.exact_sv_fixed_gaussian_genut_paired.v1",
        "campaign_id": "exact-sv-fixed-gaussian-genut-paired-20260721",
        "manifest": manifest,
        "observations": observation_payload,
        "dense_references": dense_payload,
        "designs": designs_manifest,
        "summaries": summaries,
        "paired_comparisons": comparisons,
        "mechanism_support": {
            "scientific_status": "single_dgp_nomination_only_replication_required",
            "dgp_score_reduction_each_step": dgp_score_reduction_each_step,
            "population_mechanism_supported": False,
        },
        "dataset_eligibility": {
            "fresh_dgp": "scientifically_eligible_single_dgp",
            **(
                {
                    "historical_nondgp_engineering_only": (
                        "scientifically_ineligible_historical_engineering_only"
                    )
                }
                if include_historical_nondgp_engineering_only
                else {}
            ),
        },
        "value_preservation_valid": value_preservation,
        "engineering_valid": engineering_valid and fd_valid,
        "hard_valid": engineering_valid and fd_valid,
        "nonclaims": [
            "fixed Gaussian GenUT only; no adaptive GenUT result",
            "single DGP sequence cannot establish a population method ranking",
            "OPG diagnostic is not exact Fisher information and has no pass threshold",
            "one fixed DGP sequence does not establish DGP-population generality",
            "no exact-posterior HMC, default, leaderboard, Contract E, or NAWM claim",
        ],
        "wall_seconds": time.perf_counter() - started,
    }
    for key, scoped_rows in rows.items():
        (output_root / f"rows_{key}.json").write_text(
            json.dumps(scoped_rows, indent=2, sort_keys=True) + "\n"
        )
    result_path = output_root / "result.json"
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    manifest.update(
        {
            "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "wall_seconds": payload["wall_seconds"],
            "result_sha256": _sha256(result_path),
            "status": "completed",
        }
    )
    (output_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "output": str(output_root),
                "hard_valid": payload["hard_valid"],
                "mechanism_support": payload["mechanism_support"],
                "value_preservation_valid": payload["value_preservation_valid"],
                "wall_seconds": payload["wall_seconds"],
            },
            indent=2,
        )
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--include-historical-nondgp-engineering-only", action="store_true"
    )
    args = parser.parse_args()
    run(
        args.output_root.resolve(),
        include_historical_nondgp_engineering_only=(
            args.include_historical_nondgp_engineering_only
        ),
    )


if __name__ == "__main__":
    main()
