#!/usr/bin/env python3
"""Focused same-target regressions for the repaired GenUT transport scalar."""

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

from bayesfilter.runtime.gpu_memory_policy import (
    configure_tensorflow_gpu_memory_growth,
)

_MEMORY_POLICY = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)

from bayesfilter.highdim.cubature_genut_adapters import (
    diagonal_lgssm_candidate_adapter,
    exact_transformed_sv_candidate_adapter,
    predator_prey_candidate_adapter,
)
from bayesfilter.highdim.cubature_genut_candidate import (
    gaussian_genut_design,
    replicate_positive_genut,
)
from bayesfilter.highdim.cubature_genut_filter import finite_value_score


PLAN = Path(
    "docs/plans/bayesfilter-genut-transport-repair-regression-integration-plan-2026-07-22.md"
)
CONTROL_GRID = tuple(
    {
        "epsilon": epsilon,
        "sinkhorn_steps": sinkhorn_steps,
        "balance_steps": balance_steps,
        "ridge": ridge,
    }
    for epsilon in (2.0, 4.0)
    for sinkhorn_steps in (4, 8)
    for balance_steps in (4, 8)
    for ridge in (1.0e-6, 1.0e-5)
)
RESIDUAL_TOLERANCE = 5.0e-4
T95_DF15 = 2.131449545559323
LGSSM_THETA = (0.72, 0.55, 0.35, 0.35, 0.45)
SV_THETA = (0.25, -0.15)
PREDATOR_PREY_THETA = (0.6, 114.0, 25.0, 0.3, 0.5, 0.5)
OBSERVATION_MATRIX = (
    (1.0, 0.25, -0.15),
    (0.2, 1.1, 0.3),
    (-0.1, 0.35, 0.9),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_sha256(value: tf.Tensor) -> str:
    return hashlib.sha256(tf.io.serialize_tensor(value).numpy()).hexdigest()


def _summary(values: list[float]) -> dict[str, float | int]:
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    half = T95_DF15 * sd / math.sqrt(len(values))
    return {
        "count": len(values),
        "mean": mean,
        "sample_sd": sd,
        "standard_error": sd / math.sqrt(len(values)),
        "ci95_lower": mean - half,
        "ci95_upper": mean + half,
    }


def _summaries(
    rows: list[dict[str, Any]], labels: tuple[str, ...], oracle: list[float] | None
) -> dict[str, Any]:
    vectors = [[float(row["value"]), *[float(v) for v in row["score"]]] for row in rows]
    result: dict[str, Any] = {}
    for index, label in enumerate(labels):
        estimate = _summary([row[index] for row in vectors])
        if oracle is not None:
            errors = _summary([row[index] - oracle[index] for row in vectors])
            estimate["oracle"] = oracle[index]
            estimate["error"] = errors
        result[label] = estimate
    return result


def _make_evaluator(
    adapter: Any,
    *,
    particle_count: int,
    state_dimension: int,
    parameter_count: int,
    horizon: int,
    controls: dict[str, float | int],
    transition_before_first_observation: bool = True,
) -> Callable[..., tuple[tf.Tensor, tf.Tensor, dict[str, tf.Tensor]]]:
    @tf.function(jit_compile=True, reduce_retracing=True)
    def evaluate(theta, observations, initial, process, design):
        theta = tf.ensure_shape(theta, [parameter_count])
        observations = tf.ensure_shape(observations, [horizon, state_dimension])
        initial = tf.ensure_shape(initial, [particle_count, state_dimension])
        process = tf.ensure_shape(process, [horizon, particle_count, state_dimension])
        design = tf.ensure_shape(design, [particle_count, state_dimension])
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
                balance_steps=int(controls["balance_steps"]),
                ridge=float(controls["ridge"]),
                transition_before_first_observation=transition_before_first_observation,
            )

    return evaluate


def _evaluate(evaluate: Callable[..., Any], *arguments: tf.Tensor) -> dict[str, Any]:
    value, score, diagnostics = evaluate(*arguments)
    score_residual = tf.reduce_max(
        tf.abs(tf.reduce_sum(diagnostics["score_increments"], axis=0) - score)
    )
    finite = (
        bool(diagnostics["program_valid"].numpy())
        and bool(tf.math.is_finite(value).numpy())
        and bool(tf.reduce_all(tf.math.is_finite(score)).numpy())
    )
    return {
        "value": float(value.numpy()) if finite else None,
        "score": [float(v) for v in score.numpy()] if finite else None,
        "finite": finite,
        "program_valid": bool(diagnostics["program_valid"].numpy()),
        "max_mean_residual": float(diagnostics["max_mean_residual"].numpy()),
        "max_row_residual": float(diagnostics["max_row_residual"].numpy()),
        "max_col_residual": float(diagnostics["max_col_residual"].numpy()),
        "max_transition_residual": float(
            diagnostics["max_transition_residual"].numpy()
        ),
        "score_increment_sum_residual": float(score_residual.numpy()),
        "minimum_row_mass": float(diagnostics["minimum_row_mass"].numpy()),
        "maximum_post_quotient_column_tv_error": float(
            diagnostics["maximum_post_quotient_column_tv_error"].numpy()
        ),
        "minimum_covariance_gap_eigenvalue": float(
            diagnostics["minimum_covariance_gap_eigenvalue"].numpy()
        ),
        "device": str(value.device),
    }


def _valid(row: dict[str, Any]) -> bool:
    return (
        bool(row["finite"])
        and "GPU" in str(row["device"]).upper()
        and max(
            float(row["max_mean_residual"]),
            float(row["max_row_residual"]),
            float(row["max_col_residual"]),
            float(row["max_transition_residual"]),
            float(row["score_increment_sum_residual"]),
        )
        < RESIDUAL_TOLERANCE
    )


def _tune(
    *,
    name: str,
    evaluator_factory: Callable[[dict[str, float | int]], Callable[..., Any]],
    datasets: dict[str, list[tf.Tensor]],
    particle_seeds: tuple[int, ...],
    arguments: Callable[[tf.Tensor, int], tuple[tf.Tensor, ...]],
    scales: tuple[float, ...],
) -> dict[str, Any]:
    candidates = []
    for controls in CONTROL_GRID:
        evaluate = evaluator_factory(controls)
        objectives: dict[str, float | None] = {}
        maximum_residual = 0.0
        eligible = True
        for partition, observations_set in datasets.items():
            dataset_objectives = []
            for observations in observations_set:
                rows = []
                for particle_seed in particle_seeds:
                    row = _evaluate(
                        evaluate, *arguments(observations, particle_seed)
                    )
                    rows.append(row)
                    maximum_residual = max(
                        maximum_residual,
                        float(row["max_mean_residual"]),
                        float(row["max_row_residual"]),
                        float(row["max_col_residual"]),
                        float(row["max_transition_residual"]),
                        float(row["score_increment_sum_residual"]),
                    )
                    eligible = eligible and _valid(row)
                if all(row["finite"] for row in rows):
                    vectors = [[row["value"], *row["score"]] for row in rows]
                    dataset_objectives.append(
                        max(
                            statistics.variance(vector[index] for vector in vectors)
                            / (scales[index] ** 2)
                            for index in range(len(scales))
                        )
                    )
                else:
                    eligible = False
            objectives[partition] = (
                statistics.mean(dataset_objectives)
                if len(dataset_objectives) == len(observations_set)
                else None
            )
        candidates.append(
            {
                "controls": controls,
                "objectives": objectives,
                "maximum_residual": maximum_residual,
                "eligible": eligible,
            }
        )
    eligible_rows = [row for row in candidates if row["eligible"]]
    if not eligible_rows:
        raise RuntimeError(f"no eligible repaired GenUT controls for {name}")
    selected = min(
        eligible_rows,
        key=lambda row: (
            row["objectives"]["validation"],
            row["objectives"]["calibration"],
            int(row["controls"]["sinkhorn_steps"]),
            int(row["controls"]["balance_steps"]),
            -float(row["controls"]["ridge"]),
        ),
    )
    return {
        "scope": name,
        "selection_objective": "scaled_conditional_value_and_recursive_score_variance_no_oracle",
        "particle_seeds": particle_seeds,
        "control_grid": CONTROL_GRID,
        "candidates": candidates,
        "selected_controls": selected["controls"],
        "claim_data_read_during_selection": False,
    }


def _lgssm_observations(horizon: int) -> tf.Tensor:
    from scripts.filtering_value_gradient_benchmark_generate_p8_datasets import (
        _lgssm_dataset,
    )

    return tf.cast(_lgssm_dataset(81100)["observations"][:horizon], tf.float32)


def _lgssm_dgp(seed: int, horizon: int) -> tf.Tensor:
    theta = tf.constant(LGSSM_THETA, tf.float32)
    phi, q_scale, r_scale = theta[:3], theta[3], theta[4]
    initial = tf.random.stateless_normal([3], [seed, 1], dtype=tf.float32)
    process = tf.random.stateless_normal([horizon, 3], [seed, 2], dtype=tf.float32)
    noise = tf.random.stateless_normal([horizon, 3], [seed, 3], dtype=tf.float32)
    matrix = tf.constant(OBSERVATION_MATRIX, tf.float32)
    state = initial * q_scale / tf.sqrt(1.0 - tf.square(phi))
    output = tf.TensorArray(tf.float32, size=horizon, element_shape=(3,))

    def body(index, previous, array):
        current = phi * previous + q_scale * process[index]
        observation = tf.linalg.matvec(matrix, current) + r_scale * noise[index]
        return index + 1, current, array.write(index, observation)

    _, _, output = tf.while_loop(
        lambda index, *_: index < horizon,
        body,
        (tf.zeros([], tf.int32), state, output),
        parallel_iterations=1,
    )
    return output.stack()


def _lgssm_noise(seed: int, horizon: int) -> tuple[tf.Tensor, tf.Tensor]:
    return (
        tf.random.stateless_normal([1008, 3], [seed, horizon], dtype=tf.float32),
        tf.random.stateless_normal(
            [horizon, 1008, 3], [seed, horizon + 100], dtype=tf.float32
        ),
    )


def _run_lgssm() -> dict[str, Any]:
    from docs.benchmarks.run_lgssm_cubature_genut_fp32 import _kalman_value_score

    theta = tf.constant(LGSSM_THETA, tf.float32)
    design = replicate_positive_genut(
        gaussian_genut_design(dim=3), num_particles=1008
    )
    adapter = diagonal_lgssm_candidate_adapter(
        observation_matrix=tf.constant(OBSERVATION_MATRIX, tf.float32)
    )
    results = {}
    for horizon in (2, 10, 50):
        evaluator_factory = lambda controls, h=horizon: _make_evaluator(
            adapter,
            particle_count=1008,
            state_dimension=3,
            parameter_count=5,
            horizon=h,
            controls=controls,
        )

        def arguments(observations, seed, h=horizon):
            initial, process = _lgssm_noise(seed, h)
            return theta, observations, initial, process, design

        tuning = _tune(
            name=f"lgssm_n1008_t{horizon}",
            evaluator_factory=evaluator_factory,
            datasets={
                "calibration": [_lgssm_dgp(91001, horizon), _lgssm_dgp(91002, horizon)],
                "validation": [_lgssm_dgp(91101, horizon), _lgssm_dgp(91102, horizon)],
            },
            particle_seeds=(93101, 93102),
            arguments=arguments,
            scales=(float(horizon), *((math.sqrt(horizon),) * 5)),
        )
        observations = _lgssm_observations(horizon)
        oracle_value, oracle_score = _kalman_value_score(theta, observations)
        oracle = [float(oracle_value.numpy()), *[float(v) for v in oracle_score.numpy()]]
        evaluate = evaluator_factory(dict(tuning["selected_controls"]))
        rows = []
        for seed in range(82320, 82336):
            row = _evaluate(evaluate, *arguments(observations, seed))
            row["particle_seed"] = seed
            if not _valid(row):
                raise RuntimeError(f"LGSSM repaired claim veto at T={horizon}, seed={seed}")
            rows.append(row)
        results[str(horizon)] = {
            "scope": {"particle_count": 1008, "horizon": horizon},
            "observation_sha256": _tensor_sha256(observations),
            "tuning": tuning,
            "oracle": oracle,
            "labels": ("value", "phi1", "phi2", "phi3", "q_scale", "r_scale"),
            "raw": rows,
            "summary": _summaries(
                rows,
                ("value", "phi1", "phi2", "phi3", "q_scale", "r_scale"),
                oracle,
            ),
        }
    return results


def run_lgssm_common_seed_claim(
    output_root: Path, *, tuning_result: Path
) -> dict[str, Any]:
    """Replay only LGSSM claims with the historical common particle seeds."""

    started = time.perf_counter()
    output_root.mkdir(parents=True, exist_ok=False)
    source = json.loads(tuning_result.read_text(encoding="utf-8"))
    if source.get("hard_valid") is not True:
        raise RuntimeError("LGSSM claim resume requires a hard-valid tuning result")
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    logical = tf.config.list_logical_devices("GPU")
    if not logical:
        raise RuntimeError("LGSSM common-seed claim requires a logical GPU")
    theta = tf.constant(LGSSM_THETA, tf.float32)
    design = replicate_positive_genut(
        gaussian_genut_design(dim=3), num_particles=1008
    )
    adapter = diagonal_lgssm_candidate_adapter(
        observation_matrix=tf.constant(OBSERVATION_MATRIX, tf.float32)
    )
    from docs.benchmarks.run_lgssm_cubature_genut_fp32 import _kalman_value_score

    results: dict[str, Any] = {}
    for horizon in (2, 10, 50):
        source_scope = source["lgssm"][str(horizon)]
        controls = dict(source_scope["tuning"]["selected_controls"])
        observations = _lgssm_observations(horizon)
        oracle_value, oracle_score = _kalman_value_score(theta, observations)
        oracle = [
            float(oracle_value.numpy()),
            *[float(value) for value in oracle_score.numpy()],
        ]
        evaluate = _make_evaluator(
            adapter,
            particle_count=1008,
            state_dimension=3,
            parameter_count=5,
            horizon=horizon,
            controls=controls,
        )
        rows = []
        for seed in range(82220, 82236):
            initial, process = _lgssm_noise(seed, horizon)
            row = _evaluate(
                evaluate, theta, observations, initial, process, design
            )
            row["particle_seed"] = seed
            if not _valid(row):
                raise RuntimeError(
                    f"LGSSM common-seed claim veto at T={horizon}, seed={seed}"
                )
            rows.append(row)
        labels = ("value", "phi1", "phi2", "phi3", "q_scale", "r_scale")
        results[str(horizon)] = {
            "scope": {"particle_count": 1008, "horizon": horizon},
            "controls": controls,
            "observation_sha256": _tensor_sha256(observations),
            "particle_seeds": tuple(range(82220, 82236)),
            "oracle": oracle,
            "raw": rows,
            "summary": _summaries(rows, labels, oracle),
        }
    payload = {
        "schema_version": "bayesfilter.genut_transport_repair_lgssm_common_seed_claim.v1",
        "plan": PLAN.as_posix(),
        "tuning_result": tuning_result.relative_to(ROOT).as_posix(),
        "tuning_result_sha256": _sha256(tuning_result),
        "lgssm": results,
        "device": {
            "logical_devices": [device.name for device in logical],
            "dtype": "float32",
            "tf32_enabled": True,
            "jit_compile": True,
        },
        "memory_policy": dict(_MEMORY_POLICY),
        "gpu_allocator": {
            key: int(value)
            for key, value in tf.config.experimental.get_memory_info("GPU:0").items()
        },
        "wall_time_seconds": time.perf_counter() - started,
        "hard_valid": True,
        "comparison_role": "paired_common_seed_claim_only_frozen_tuning",
        "nonclaims": [
            "claim replay does not retune or establish default readiness",
            "old and repaired scalars differ by construction",
        ],
    }
    (output_root / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "hard_valid": True,
                "wall_time_seconds": payload["wall_time_seconds"],
                "output": str(output_root),
            }
        )
    )
    return payload


def _sv_dgp(seed: int) -> tf.Tensor:
    theta = tf.constant(SV_THETA, tf.float32)
    gamma = 0.5 * (1.0 + tf.math.erf(theta[0] / tf.sqrt(2.0)))
    beta = tf.exp(theta[1])
    state = tf.random.stateless_normal([], [seed, 1], dtype=tf.float32) / tf.sqrt(
        1.0 - tf.square(gamma)
    )
    process = tf.random.stateless_normal([50], [seed, 2], dtype=tf.float32)
    noise = tf.random.stateless_normal([50], [seed, 3], dtype=tf.float32)
    output = tf.TensorArray(tf.float32, size=50, element_shape=(1,))

    def body(index, previous, array):
        current = gamma * previous + process[index]
        observation = current + 2.0 * tf.math.log(beta) + tf.math.log(tf.square(noise[index]))
        return index + 1, current, array.write(index, observation[None])

    _, _, output = tf.while_loop(
        lambda index, *_: index < 50,
        body,
        (tf.zeros([], tf.int32), state, output),
        parallel_iterations=1,
    )
    return output.stack()


def _run_sv() -> dict[str, Any]:
    from docs.benchmarks import run_exact_sv_fixed_gaussian_genut_paired as prior

    particle_count = 1998
    theta = tf.constant(SV_THETA, tf.float32)
    adapter = exact_transformed_sv_candidate_adapter()
    observation_artifact = ROOT / (
        "docs/benchmarks/artifacts/exact_sv_fixed_gaussian_genut_paired_20260721/"
        "attempt02/observations.json"
    )
    reference_artifact = observation_artifact.with_name("result.json")
    observations_payload = json.loads(observation_artifact.read_text(encoding="utf-8"))
    observations = tf.reshape(
        tf.constant(observations_payload["fresh_dgp"]["values"], tf.float32), [50, 1]
    )
    reference = json.loads(reference_artifact.read_text(encoding="utf-8"))
    dense = reference["dense_references"]["fresh_dgp"]["refinements"][-1]
    oracle = [float(dense["value"]), *[float(v) for v in dense["score"]]]

    def evaluator_factory(controls):
        return _make_evaluator(
            adapter,
            particle_count=particle_count,
            state_dimension=1,
            parameter_count=2,
            horizon=50,
            controls=controls,
        )

    def arguments(data, seed):
        initial = tf.random.stateless_normal(
            [particle_count, 1], [seed, 101], dtype=tf.float32
        )
        process = tf.random.stateless_normal(
            [50, particle_count, 1], [seed, 102], dtype=tf.float32
        )
        design = prior.paired_rank_designs(particle_count, seed)[1]
        return theta, data, initial, process, design

    tuning = _tune(
        name="fresh_exact_sv_n1998_t50",
        evaluator_factory=evaluator_factory,
        datasets={
            "calibration": [_sv_dgp(92001), _sv_dgp(92002)],
            "validation": [_sv_dgp(92101), _sv_dgp(92102)],
        },
        particle_seeds=(93101, 93102),
        arguments=arguments,
        scales=(50.0, 10.0, 10.0),
    )
    evaluate = evaluator_factory(dict(tuning["selected_controls"]))
    rows = []
    for seed in range(3620, 3636):
        initial, process = prior._nested_noise(particle_count, seed)  # noqa: SLF001
        design = prior.paired_rank_designs(particle_count, seed)[1]
        row = _evaluate(evaluate, theta, observations, initial, process, design)
        row["particle_seed"] = seed
        if not _valid(row):
            raise RuntimeError(f"fresh exact-SV repaired claim veto at seed={seed}")
        rows.append(row)
    return {
        "scope": {"particle_count": particle_count, "horizon": 50},
        "observation_sha256": _tensor_sha256(observations),
        "tuning": tuning,
        "oracle": oracle,
        "labels": ("value", "theta_gamma", "theta_log_beta"),
        "raw": rows,
        "summary": _summaries(
            rows, ("value", "theta_gamma", "theta_log_beta"), oracle
        ),
        "prior_artifact": reference_artifact.relative_to(ROOT).as_posix(),
    }


def _run_predator_prey() -> dict[str, Any]:
    from docs.benchmarks import run_genut_predator_prey_leaderboard_continuation as prior

    particle_count = 1002
    theta = tf.constant(PREDATOR_PREY_THETA, tf.float32)
    adapter = predator_prey_candidate_adapter()
    design = prior._genut_design(particle_count)  # noqa: SLF001

    def evaluator_factory(controls):
        return _make_evaluator(
            adapter,
            particle_count=particle_count,
            state_dimension=2,
            parameter_count=6,
            horizon=20,
            controls=controls,
            transition_before_first_observation=False,
        )

    def arguments(observations, seed):
        initial, process = prior._particle_noise(seed, particle_count)  # noqa: SLF001
        return theta, observations, initial, process, design

    tuning = _tune(
        name="predator_prey_n1002_t20",
        evaluator_factory=evaluator_factory,
        datasets={
            "calibration": [
                prior._dataset(95101)["observations"],  # noqa: SLF001
                prior._dataset(95102)["observations"],  # noqa: SLF001
            ],
            "validation": [
                prior._dataset(95201)["observations"],  # noqa: SLF001
                prior._dataset(95202)["observations"],  # noqa: SLF001
            ],
        },
        particle_seeds=(96101, 96102),
        arguments=arguments,
        scales=(50.0, 25.0, 2.0, 1.0, 25.0, 10.0, 10.0),
    )
    dataset = prior._dataset(81104)  # noqa: SLF001
    observations = dataset["observations"]
    evaluate = evaluator_factory(dict(tuning["selected_controls"]))
    rows = []
    for seed in range(97201, 97217):
        row = _evaluate(evaluate, *arguments(observations, seed))
        row["particle_seed"] = seed
        if not _valid(row):
            raise RuntimeError(f"predator-prey repaired claim veto at seed={seed}")
        rows.append(row)
    labels = ("value", "r", "K", "a", "s", "u", "v")
    return {
        "scope": {"particle_count": particle_count, "horizon": 20},
        "observation_sha256": _tensor_sha256(observations),
        "state_sha256": _tensor_sha256(dataset["states"]),
        "tuning": tuning,
        "labels": labels,
        "raw": rows,
        "summary": _summaries(rows, labels, None),
        "score_authority": "no_exact_oracle_descriptive_only",
    }


def run(output_root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    output_root.mkdir(parents=True, exist_ok=False)
    memory_policy = _MEMORY_POLICY
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    logical = tf.config.list_logical_devices("GPU")
    if not logical:
        raise RuntimeError("repaired GenUT regressions require a logical GPU")
    payload = {
        "schema_version": "bayesfilter.genut_transport_repair_regressions.v1",
        "plan": PLAN.as_posix(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "device": {
            "logical_devices": [device.name for device in logical],
            "dtype": "float32",
            "tf32_enabled": bool(
                tf.config.experimental.tensor_float_32_execution_enabled()
            ),
            "jit_compile": True,
            "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        },
        "memory_policy": dict(memory_policy),
        "configuration": {
            "control_grid": CONTROL_GRID,
            "runtime_score": "recursive_forward_sensitivity_no_autodiff_no_fd",
            "particle_minimum_policy": "N>1000",
        },
        "lgssm": _run_lgssm(),
        "fresh_exact_sv": _run_sv(),
        "predator_prey": _run_predator_prey(),
        "hard_valid": True,
        "nonclaims": [
            "no nonlinear exactness claim",
            "no statistically supported cross-method ranking",
            "predator-prey score has no exact oracle",
            "regression feasibility does not establish default or HMC readiness",
        ],
    }
    payload["gpu_allocator"] = {
        key: int(value)
        for key, value in tf.config.experimental.get_memory_info("GPU:0").items()
    }
    payload["wall_time_seconds"] = time.perf_counter() - started
    payload["run_manifest"] = {
        "command": [sys.executable, *sys.argv],
        "environment": sys.prefix,
        "host": platform.node(),
        "python": sys.version.split()[0],
        "tensorflow": tf.__version__,
        "source_sha256": {
            Path(__file__).relative_to(ROOT).as_posix(): _sha256(Path(__file__)),
            "bayesfilter/highdim/cubature_genut_filter.py": _sha256(
                ROOT / "bayesfilter/highdim/cubature_genut_filter.py"
            ),
            "bayesfilter/highdim/cubature_genut_adapters.py": _sha256(
                ROOT / "bayesfilter/highdim/cubature_genut_adapters.py"
            ),
            PLAN.as_posix(): _sha256(ROOT / PLAN),
        },
    }
    (output_root / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "hard_valid": payload["hard_valid"],
                "wall_time_seconds": payload["wall_time_seconds"],
                "output": str(output_root),
            }
        )
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--lgssm-claim-resume", type=Path, default=None)
    args = parser.parse_args()
    if args.lgssm_claim_resume is None:
        run(args.output_root.resolve())
    else:
        run_lgssm_common_seed_claim(
            args.output_root.resolve(),
            tuning_result=args.lgssm_claim_resume.resolve(),
        )


if __name__ == "__main__":
    main()
