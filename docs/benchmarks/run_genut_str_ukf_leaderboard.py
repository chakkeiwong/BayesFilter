#!/usr/bin/env python3
"""Bounded GenUT candidate run for the existing Chapter 18b STR-UKF target.

This runner does not create a new DGP.  It reuses the frozen Chapter 18b data,
the repository-owned source chart, and the shared structural transition and
residual primitives.  Its output is a candidate leaderboard extension; it is
not an automatic method-admission or default-promotion artifact.
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
from typing import Any, Callable

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
    MEMORY_POLICY = {
        "schema": "bayesfilter.tensorflow.gpu_memory_policy.v1",
        "mode": "cpu_hidden_import_only",
        "configured_before_logical_device_initialization": True,
        "tf_force_gpu_allow_growth": os.environ.get(
            "TF_FORCE_GPU_ALLOW_GROWTH", "unset"
        ),
    }
else:
    MEMORY_POLICY = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)


def _require_serious_gpu_policy() -> None:
    if MEMORY_POLICY.get("mode") != "memory_growth":
        raise RuntimeError(
            "serious STR-UKF GenUT execution requires verified GPU memory growth"
        )

from bayesfilter.highdim.cubature_genut_adapters import structural_ukf_candidate_adapter
from bayesfilter.highdim.cubature_genut_candidate import (
    CandidateRouteScope,
    gaussian_genut_design,
    issue_repository_candidate_route_identity,
    replicate_positive_genut,
    validate_repository_candidate_route_identity,
)
from bayesfilter.highdim.cubature_genut_filter import finite_value_score
from bayesfilter.testing.structural_ukf_neutra_target_design_tf import (
    STRUCTURAL_FINAL_OBSERVATION_SHA256,
    STRUCTURAL_FINAL_STATE_SHA256,
    STRUCTURAL_PARAMETER_NAMES,
    STRUCTURAL_TRUTH_PHYSICAL,
    STRUCTURAL_UKF_SCOPE,
    generate_frozen_structural_dataset_tf,
    simulate_structural_trajectories_tf,
    structural_source_chart,
    structural_truth_source,
    structural_ukf_likelihood_value_score_status,
)

PLAN = Path("docs/plans/bayesfilter-genut-str-ukf-leaderboard-plan-2026-07-22.md")
RESULT_NOTE = Path(
    "docs/plans/bayesfilter-genut-str-ukf-leaderboard-result-2026-07-22.md"
)
ARTIFACT_NAME = "genut_str_ukf_leaderboard_20260722"
CAMPAIGN_ID = "genut-str-ukf-leaderboard-20260722"
SCHEMA_VERSION = "bayesfilter.genut_str_ukf_leaderboard.v1"
N = 1002
HORIZON = 100
TUNING_PREFIX = 100
CONTROL_GRID = tuple(
    {
        "epsilon": epsilon,
        "sinkhorn_steps": steps,
        "balance_steps": balance_steps,
        "ridge": ridge,
    }
    for epsilon in (2.0, 4.0)
    for steps in (4, 8)
    for balance_steps in (4, 8, 16, 32)
    for ridge in (1.0e-6, 1.0e-5)
)
CALIBRATION_SEEDS = (2026072201, 2026072202)
VALIDATION_SEEDS = (2026072211, 2026072212)
TUNING_PARTICLE_SEEDS = (2026072231, 2026072232, 2026072233, 2026072234)
CONSUMED_CLAIM_PARTICLE_SEEDS = tuple(range(2026072251, 2026072259))
CONSUMED_RETRY_CLAIM_PARTICLE_SEEDS = tuple(range(2026072291, 2026072299))
# Kept only for the historical claim-resume entry point. New runs use the
# fresh CLAIM_PARTICLE_SEEDS below and must not enter the resume path.
RETRY_CLAIM_PARTICLE_SEEDS = CONSUMED_RETRY_CLAIM_PARTICLE_SEEDS
CLAIM_PARTICLE_SEEDS = tuple(range(2026072301, 2026072309))
FD_STEPS = tf.constant([2.0e-3] * 5, tf.float32)
FD_RELATIVE_TOLERANCE = 5.0e-2
TRANSITION_RESIDUAL_TOLERANCE = 2.0e-5
RESET_RESIDUAL_TOLERANCE = 5.0e-4
SCORE_SUM_RELATIVE_TOLERANCE = 2.0e-5
MAX_MEMORY_BYTES = 12 * 1024**3
T_CRITICAL_95_DF7 = 2.364624251
TUNING_SCALES = (float(HORIZON), *((math.sqrt(HORIZON),) * 5))


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


def _particle_noise(seed: int, horizon: int = HORIZON) -> tuple[tf.Tensor, tf.Tensor]:
    roots = tf.random.experimental.stateless_split(tf.constant([seed, seed + 17], tf.int32), 2)
    return (
        tf.random.stateless_normal([N, 2], roots[0], dtype=tf.float32),
        tf.random.stateless_normal([horizon, N, 1], roots[1], dtype=tf.float32),
    )


def _genut_design() -> tf.Tensor:
    return replicate_positive_genut(
        gaussian_genut_design(dim=2, dtype=tf.float32), num_particles=N
    )


def _synthetic_dataset(seed: int) -> dict[str, tf.Tensor]:
    """Generate a full-scope calibration/validation sequence from the shared DGP."""

    roots = tf.random.experimental.stateless_split(
        tf.constant([seed, seed + 19], tf.int32), 2
    )
    source = tf.random.stateless_normal([1, 5], roots[0], dtype=tf.float64)
    physical, _ = structural_source_chart(source)
    _states, observations, residuals = simulate_structural_trajectories_tf(
        physical, horizon=HORIZON, seed=roots[1]
    )
    tf.debugging.assert_near(residuals, tf.zeros_like(residuals), atol=2.0e-14)
    return {
        "source": tf.cast(source[0], tf.float32),
        "observations": tf.cast(observations[0], tf.float32),
    }


def _make_evaluator(
    controls: dict[str, float | int], horizon: int
) -> Callable[..., tuple[tf.Tensor, tf.Tensor, dict[str, tf.Tensor]]]:
    adapter = structural_ukf_candidate_adapter()

    @tf.function(jit_compile=True, reduce_retracing=True)
    def evaluate(
        theta: tf.Tensor,
        observations: tf.Tensor,
        initial_noise: tf.Tensor,
        process_noise: tf.Tensor,
        design: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor, dict[str, tf.Tensor]]:
        theta = tf.ensure_shape(theta, [5])
        observations = tf.ensure_shape(observations, [horizon, 1])
        initial_noise = tf.ensure_shape(initial_noise, [N, 2])
        process_noise = tf.ensure_shape(process_noise, [horizon, N, 1])
        design = tf.ensure_shape(design, [N, 2])
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
                balance_steps=int(controls.get("balance_steps", 8)),
                ridge=float(controls["ridge"]),
                transition_before_first_observation=False,
            )

    return evaluate


def _evaluate(
    evaluate: Callable[..., tuple[tf.Tensor, tf.Tensor, dict[str, tf.Tensor]]],
    theta: tf.Tensor,
    observations: tf.Tensor,
    initial_noise: tf.Tensor,
    process_noise: tf.Tensor,
    design: tf.Tensor,
) -> dict[str, Any]:
    value, score, diagnostics = evaluate(
        theta, observations, initial_noise, process_noise, design
    )
    score_sum_residual = tf.reduce_max(
        tf.abs(tf.reduce_sum(diagnostics["score_increments"], axis=0) - score)
    )
    score_increment_sum = tf.reduce_sum(diagnostics["score_increments"], axis=0)
    score_sum_scale = tf.maximum(
        tf.constant(1.0, score.dtype),
        tf.maximum(tf.reduce_max(tf.abs(score)), tf.reduce_max(tf.abs(score_increment_sum))),
    )
    score_sum_relative_residual = score_sum_residual / score_sum_scale
    max_reset_residual = tf.reduce_max(
        tf.stack(
            [
                diagnostics["max_mean_residual"],
                diagnostics["max_row_residual"],
                diagnostics["max_col_residual"],
            ]
        )
    )
    maximum_residual = tf.reduce_max(
        tf.stack(
            [
                diagnostics["max_transition_residual"],
                max_reset_residual,
                score_sum_relative_residual,
            ]
        )
    )
    value_number = float(value.numpy())
    score_numbers = [float(v) for v in score.numpy()]
    nonfinite_components = []
    if not math.isfinite(value_number):
        nonfinite_components.append("value")
    nonfinite_components.extend(
        f"score[{index}]"
        for index, item in enumerate(score_numbers)
        if not math.isfinite(item)
    )
    finite = math.isfinite(value_number) and all(
        math.isfinite(item) for item in score_numbers
    )
    return {
        "value": value_number if math.isfinite(value_number) else None,
        "score": [item if math.isfinite(item) else None for item in score_numbers],
        "finite": finite,
        "nonfinite_components": nonfinite_components,
        "max_transition_residual": float(
            diagnostics["max_transition_residual"].numpy()
        ),
        "max_mean_residual": float(diagnostics["max_mean_residual"].numpy()),
        "max_row_residual": float(diagnostics["max_row_residual"].numpy()),
        "max_col_residual": float(diagnostics["max_col_residual"].numpy()),
        "score_sum_residual": float(score_sum_residual.numpy()),
        "score_sum_relative_residual": float(score_sum_relative_residual.numpy()),
        "max_reset_residual": float(max_reset_residual.numpy()),
        "maximum_residual": float(maximum_residual.numpy()),
        "program_valid": bool(diagnostics["program_valid"].numpy()),
        "minimum_row_mass": float(diagnostics["minimum_row_mass"].numpy()),
        "maximum_post_quotient_column_tv_error": float(
            diagnostics["maximum_post_quotient_column_tv_error"].numpy()
        ),
        "minimum_covariance_gap_eigenvalue": float(
            diagnostics["minimum_covariance_gap_eigenvalue"].numpy()
        ),
        "device": str(value.device),
    }


def _summary(values: list[float]) -> dict[str, float | int]:
    if len(values) < 2:
        raise ValueError("at least two values are required for an interval")
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    se = sd / math.sqrt(len(values))
    critical = T_CRITICAL_95_DF7 if len(values) == 8 else 2.1314495456
    half = critical * se
    return {
        "count": len(values),
        "mean": mean,
        "sample_sd": sd,
        "standard_error": se,
        "critical_value": critical,
        "ci95_lower": mean - half,
        "ci95_upper": mean + half,
    }


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not all(row["finite"] for row in rows):
        raise ValueError("cannot summarize non-finite candidate rows")
    labels = ("value", *STRUCTURAL_PARAMETER_NAMES)
    vectors = [[row["value"], *row["score"]] for row in rows]
    return {
        label: _summary([vector[index] for vector in vectors])
        for index, label in enumerate(labels)
    }


def _fd_audit(
    controls: dict[str, float | int],
    observations: tf.Tensor,
    design: tf.Tensor,
    theta: tf.Tensor,
) -> dict[str, Any]:
    evaluate = _make_evaluator(controls, HORIZON)
    initial, process = _particle_noise(TUNING_PARTICLE_SEEDS[0])
    analytical = _evaluate(evaluate, theta, observations, initial, process, design)
    errors = []
    rows = []
    for index in range(5):
        direction = tf.one_hot(index, 5, dtype=tf.float32)
        plus = _evaluate(
            evaluate, theta + FD_STEPS[index] * direction, observations, initial, process, design
        )["value"]
        minus = _evaluate(
            evaluate, theta - FD_STEPS[index] * direction, observations, initial, process, design
        )["value"]
        fd = (plus - minus) / (2.0 * float(FD_STEPS[index].numpy()))
        error = abs(analytical["score"][index] - fd) / max(1.0, abs(fd))
        errors.append(error)
        rows.append(
            {
                "parameter": STRUCTURAL_PARAMETER_NAMES[index],
                "analytical_source_score": analytical["score"][index],
                "finite_difference_source_score": fd,
                "relative_error": error,
            }
        )
    return {
        "rows": rows,
        "maximum_relative_error": max(errors),
        "diagnostic_only": True,
    }


def _tune(theta: tf.Tensor, design: tf.Tensor) -> dict[str, Any]:
    del theta
    candidates = []
    for controls in CONTROL_GRID:
        evaluate = _make_evaluator(controls, HORIZON)
        partition_rows = {}
        all_valid = True
        maximum_residual = 0.0
        maximum_transition_residual = 0.0
        maximum_reset_residual = 0.0
        maximum_score_sum_relative_residual = 0.0
        for name, seeds in (("calibration", CALIBRATION_SEEDS), ("validation", VALIDATION_SEEDS)):
            rows = []
            dataset_variance_objectives = []
            for data_seed in seeds:
                dataset = _synthetic_dataset(data_seed)
                observations = dataset["observations"]
                dataset_theta = dataset["source"]
                dataset_rows = []
                for particle_seed in TUNING_PARTICLE_SEEDS:
                    initial, process = _particle_noise(particle_seed)
                    row = _evaluate(
                        evaluate,
                        dataset_theta,
                        observations,
                        initial,
                        process,
                        design,
                    )
                    rows.append(row)
                    dataset_rows.append(row)
                    all_valid = all_valid and row["finite"] and "GPU" in row["device"].upper()
                    maximum_residual = max(maximum_residual, row["maximum_residual"])
                    maximum_transition_residual = max(
                        maximum_transition_residual,
                        row["max_transition_residual"],
                    )
                    maximum_reset_residual = max(
                        maximum_reset_residual, row["max_reset_residual"]
                    )
                    maximum_score_sum_relative_residual = max(
                        maximum_score_sum_relative_residual,
                        row["score_sum_relative_residual"],
                    )
                if all(row["finite"] for row in dataset_rows):
                    vectors = [[row["value"], *row["score"]] for row in dataset_rows]
                    dataset_variance_objectives.append(
                        max(
                            statistics.variance(vector[index] for vector in vectors)
                            / (TUNING_SCALES[index] ** 2)
                            for index in range(6)
                        )
                    )
                else:
                    dataset_variance_objectives.append(None)
            finite_rows = [row for row in rows if row["finite"]]
            summary = _summarize_rows(finite_rows) if len(finite_rows) >= 2 else None
            finite_objectives = [
                value for value in dataset_variance_objectives if value is not None
            ]
            partition_rows[name] = {
                "rows": rows,
                "summary": summary,
                "per_dataset_scaled_conditional_variance": dataset_variance_objectives,
                "conditional_variance_objective": (
                    statistics.mean(finite_objectives)
                    if len(finite_objectives) == len(dataset_variance_objectives)
                    else None
                ),
            }
        candidates.append(
            {
                "controls": controls,
                "partitions": partition_rows,
                "maximum_residual": maximum_residual,
                "maximum_transition_residual": maximum_transition_residual,
                "maximum_reset_residual": maximum_reset_residual,
                "maximum_score_sum_relative_residual": (
                    maximum_score_sum_relative_residual
                ),
                "fd_audit": None,
                "engineering_eligible_before_fd": (
                    all_valid
                    and maximum_reset_residual <= RESET_RESIDUAL_TOLERANCE
                    and maximum_transition_residual
                    <= TRANSITION_RESIDUAL_TOLERANCE
                    and maximum_score_sum_relative_residual
                    <= SCORE_SUM_RELATIVE_TOLERANCE
                ),
                "eligible": False,
            }
        )
    ordered = sorted(
        [candidate for candidate in candidates if candidate["engineering_eligible_before_fd"]],
        key=lambda candidate: (
            candidate["partitions"]["validation"]["conditional_variance_objective"],
            int(candidate["controls"]["sinkhorn_steps"]),
            int(candidate["controls"]["balance_steps"]),
            float(candidate["controls"]["epsilon"]),
            -float(candidate["controls"]["ridge"]),
        ),
    )
    calibration = _synthetic_dataset(CALIBRATION_SEEDS[0])
    validation = _synthetic_dataset(VALIDATION_SEEDS[-1])
    selected = None
    for candidate in ordered:
        fd_audits = [
            _fd_audit(
                candidate["controls"],
                dataset["observations"],
                design,
                dataset["source"],
            )
            for dataset in (calibration, validation)
        ]
        fd = {
            "audits": fd_audits,
            "maximum_relative_error": max(
                audit["maximum_relative_error"] for audit in fd_audits
            ),
            "diagnostic_only": True,
        }
        candidate["fd_audit"] = fd
        candidate["eligible"] = (
            fd["maximum_relative_error"] <= FD_RELATIVE_TOLERANCE
        )
        if candidate["eligible"]:
            selected = candidate
            break
    result = {
        "scope": {
            "target_id": STRUCTURAL_UKF_SCOPE,
            "horizon": HORIZON,
            "particle_count": N,
            "dtype": "float32",
            "tf32": True,
            "jit_compile": True,
            "score": "recursive_forward_sensitivity_no_autodiff_no_fd",
        },
        "calibration_seeds": CALIBRATION_SEEDS,
        "validation_seeds": VALIDATION_SEEDS,
        "particle_seeds": TUNING_PARTICLE_SEEDS,
        "control_grid": CONTROL_GRID,
        "candidates": candidates,
        "selected_controls": None if selected is None else selected["controls"],
        "selected_candidate_fd_audit": (
            None if selected is None else selected["fd_audit"]
        ),
        "status": (
            "NO_ELIGIBLE_CANDIDATE" if selected is None else "TUNING_CANDIDATE_SELECTED"
        ),
        "claim_data_read_during_selection": False,
    }
    return result


def _route_identity(controls: dict[str, float | int], observation_hash: str) -> dict[str, Any]:
    identity = issue_repository_candidate_route_identity(
        CandidateRouteScope(
            model_id="chapter18b_quadratic_structural",
            target_id=STRUCTURAL_UKF_SCOPE,
            horizon=HORIZON,
            particle_count=N,
            state_dimension=2,
            parameter_count=5,
            dtype="float32",
            tf32_enabled=True,
            jit_compile=True,
            design_family="genut",
            control_family_id="str_ukf_genut_row_quotient_controls_v2",
        ),
        prepared_data_id=f"sha256:{observation_hash}",
        residual_design_id=f"gaussian_genut_dim2_equal_mass_n{N}_v1",
        controls={key: str(value) for key, value in controls.items()},
        adapter_id="chapter18b_structural_shared_primitives_v1",
    )
    validate_repository_candidate_route_identity(identity)
    return identity.to_dict()


def _claim(theta: tf.Tensor, controls: dict[str, float | int], design: tf.Tensor) -> dict[str, Any]:
    states, observations64 = generate_frozen_structural_dataset_tf()
    observations = tf.cast(observations64, tf.float32)
    observation_hash = _tensor_sha256(observations64)
    if observation_hash != STRUCTURAL_FINAL_OBSERVATION_SHA256:
        raise RuntimeError("frozen observation hash mismatch")
    if _tensor_sha256(states) != STRUCTURAL_FINAL_STATE_SHA256:
        raise RuntimeError("frozen state hash mismatch")
    evaluate = _make_evaluator(controls, HORIZON)
    rows = []
    maximum_residual = 0.0
    maximum_transition_residual = 0.0
    maximum_reset_residual = 0.0
    maximum_score_sum_relative_residual = 0.0
    for particle_seed in CLAIM_PARTICLE_SEEDS:
        initial, process = _particle_noise(particle_seed)
        row = _evaluate(evaluate, theta, observations, initial, process, design)
        row["particle_seed"] = particle_seed
        rows.append(row)
        maximum_residual = max(maximum_residual, row["maximum_residual"])
        maximum_transition_residual = max(
            maximum_transition_residual, row["max_transition_residual"]
        )
        maximum_reset_residual = max(
            maximum_reset_residual, row["max_reset_residual"]
        )
        maximum_score_sum_relative_residual = max(
            maximum_score_sum_relative_residual,
            row["score_sum_relative_residual"],
        )
    ukf_value, ukf_score, ukf_status = structural_ukf_likelihood_value_score_status(
        tf.cast(theta[None, :], tf.float64), observations=observations64
    )
    ukf_source_score = ukf_score[0]
    physical, derivative = structural_source_chart(tf.cast(theta[None, :], tf.float64))
    ukf_physical_score = ukf_source_score / derivative[0]
    all_finite = all(row["finite"] for row in rows)
    if all_finite:
        genut_value_score = _summarize_rows(rows)
        genut_source_mean = tf.constant(
            [genut_value_score[name]["mean"] for name in STRUCTURAL_PARAMETER_NAMES],
            tf.float64,
        )
        genut_physical_score = genut_source_mean / derivative[0]
        mean_physical_score = [float(v) for v in genut_physical_score.numpy()]
        comparison = {
            "value_difference_genut_minus_ukf": (
                genut_value_score["value"]["mean"] - float(ukf_value[0].numpy())
            ),
            "physical_score_difference_genut_minus_ukf": [
                float(v) for v in (genut_physical_score - ukf_physical_score).numpy()
            ],
        }
    else:
        genut_value_score = None
        mean_physical_score = None
        comparison = {
            "value_difference_genut_minus_ukf": None,
            "physical_score_difference_genut_minus_ukf": None,
            "reason": "full-horizon GenUT claim contains non-finite rows",
        }
    return {
        "dataset": {
            "observation_sha256": observation_hash,
            "state_sha256": _tensor_sha256(states),
            "timing": "initial_observation_first_then_transitions_y1_to_y99",
        },
        "controls": controls,
        "route_identity": _route_identity(controls, observation_hash),
        "genut": {
            "particle_seeds": CLAIM_PARTICLE_SEEDS,
            "raw": rows,
            "summary": genut_value_score,
            "all_finite": all_finite,
            "nonfinite_particle_seeds": [
                row["particle_seed"] for row in rows if not row["finite"]
            ],
            "maximum_residual": maximum_residual,
            "maximum_transition_residual": maximum_transition_residual,
            "maximum_reset_residual": maximum_reset_residual,
            "maximum_score_sum_relative_residual": (
                maximum_score_sum_relative_residual
            ),
            "mean_physical_score": mean_physical_score,
        },
        "principal_sqrt_ukf": {
            "role": "same_target_analytical_approximation_diagnostic_not_oracle",
            "value": float(ukf_value[0].numpy()),
            "source_score": [float(v) for v in ukf_source_score.numpy()],
            "physical_score": [float(v) for v in ukf_physical_score.numpy()],
            "valid": bool(ukf_status["valid_pre_regularized_score"][0].numpy()),
            "deterministic_residual": float(ukf_status["deterministic_residual"][0].numpy()),
        },
        "comparison": comparison,
    }


def run_claim_resume(output_root: Path, tuning_path: Path) -> dict[str, Any]:
    """Resume only the frozen-data claim with new seeds after artifact repair."""

    _require_serious_gpu_policy()
    global CLAIM_PARTICLE_SEEDS
    started = time.perf_counter()
    output_root.mkdir(parents=True, exist_ok=False)
    tuning = json.loads(tuning_path.read_text(encoding="utf-8"))
    expected_scope = tuning.get("scope", {})
    if (
        expected_scope.get("target_id") != STRUCTURAL_UKF_SCOPE
        or expected_scope.get("horizon") != HORIZON
        or expected_scope.get("particle_count") != N
        or expected_scope.get("dtype") != "float32"
        or expected_scope.get("tf32") is not True
        or expected_scope.get("jit_compile") is not True
        or tuning.get("claim_data_read_during_selection") is not False
        or tuning.get("selected_controls") is None
    ):
        raise RuntimeError("claim resume tuning artifact scope mismatch")
    fd = tuning.get("selected_candidate_fd_audit")
    if fd is None or float(fd["maximum_relative_error"]) > FD_RELATIVE_TOLERANCE:
        raise RuntimeError("claim resume tuning artifact lacks a passing score audit")
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    logical = tf.config.list_logical_devices("GPU")
    if not logical:
        raise RuntimeError("STR-UKF GenUT claim resume requires a logical GPU")
    design = _genut_design()
    theta = tf.cast(structural_truth_source(), tf.float32)
    original_claim_seeds = CLAIM_PARTICLE_SEEDS
    CLAIM_PARTICLE_SEEDS = RETRY_CLAIM_PARTICLE_SEEDS
    try:
        claim = _claim(theta, dict(tuning["selected_controls"]), design)
    finally:
        CLAIM_PARTICLE_SEEDS = original_claim_seeds
    allocator = tf.config.experimental.get_memory_info("GPU:0")
    peak_bytes = int(allocator.get("peak", 0))
    hard_pass = (
        claim["genut"]["all_finite"]
        and claim["genut"]["maximum_reset_residual"] <= RESET_RESIDUAL_TOLERANCE
        and claim["genut"]["maximum_transition_residual"]
        <= TRANSITION_RESIDUAL_TOLERANCE
        and claim["genut"]["maximum_score_sum_relative_residual"]
        <= SCORE_SUM_RELATIVE_TOLERANCE
        and claim["principal_sqrt_ukf"]["valid"]
        and peak_bytes <= MAX_MEMORY_BYTES
    )
    source_paths = (
        Path(__file__).relative_to(ROOT),
        Path("bayesfilter/highdim/cubature_genut_candidate.py"),
        Path("bayesfilter/highdim/cubature_genut_filter.py"),
        Path("bayesfilter/highdim/cubature_genut_adapters.py"),
        Path("bayesfilter/testing/structural_ukf_neutra_target_design_tf.py"),
        PLAN,
    )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": CAMPAIGN_ID,
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_time_seconds": time.perf_counter() - started,
        "git_commit": _git_commit(),
        "plan": PLAN.as_posix(),
        "target": {
            "target_scope": STRUCTURAL_UKF_SCOPE,
            "parameter_names": STRUCTURAL_PARAMETER_NAMES,
            "truth_physical": [float(v) for v in STRUCTURAL_TRUTH_PHYSICAL.numpy()],
            "horizon": HORIZON,
            "particle_count": N,
            "transition_restriction": "scalar innovation; k deterministic from previous k and current m",
        },
        "device": {
            "logical_devices": [item.name for item in logical],
            "dtype": "float32",
            "tf32_enabled": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
            "jit_compile": True,
            "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        },
        "memory_policy": MEMORY_POLICY,
        "gpu_allocator": {key: int(value) for key, value in allocator.items()},
        "source_sha256": {path.as_posix(): _sha256(ROOT / path) for path in source_paths},
        "tuning": {
            "selected_controls": tuning["selected_controls"],
            "artifact": tuning_path.as_posix(),
            "selected_candidate_fd_audit": fd,
            "claim_data_read_during_selection": False,
        },
        "claim": claim,
        "engineering_ledger": {
            "finite": all(row["finite"] for row in claim["genut"]["raw"]),
            "gpu_xla_tf32": True,
            "memory_growth": True,
            "reset_residual_gate": claim["genut"]["maximum_reset_residual"] <= RESET_RESIDUAL_TOLERANCE,
            "transition_residual_gate": claim["genut"]["maximum_transition_residual"] <= TRANSITION_RESIDUAL_TOLERANCE,
            "score_sum_relative_gate": claim["genut"]["maximum_score_sum_relative_residual"] <= SCORE_SUM_RELATIVE_TOLERANCE,
            "allocator_peak_bytes": peak_bytes,
        },
        "scientific_ledger": {
            "candidate_leaderboard_row_included": hard_pass,
            "leaderboard_admitted": False,
            "default_changed": False,
            "statistically_supported_ranking": False,
        },
        "decision": {
            "status": "CANDIDATE_LEADERBOARD_ROW_INCLUDED" if hard_pass else "CANDIDATE_BLOCKED_HARD_GATE",
            "leaderboard_row_status": "included_candidate_not_admitted" if hard_pass else "blocked",
            "leaderboard_admitted": False,
            "default_changed": False,
        },
        "resume": {
            "reason": "attempt02 completed tuning and numerical claim but failed unsupported route-identity label during result assembly",
            "original_claim_seeds_not_reused": True,
            "fresh_claim_seeds": RETRY_CLAIM_PARTICLE_SEEDS,
        },
        "nonclaims": [
            "no exact nonlinear likelihood or score",
            "no unbiasedness or superiority",
            "no HMC or default readiness",
            "no high-dimensional feasibility",
        ],
    }
    (output_root / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_root / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "bayesfilter.serious_run_manifest.v1",
                "git_commit": payload["git_commit"],
                "command": " ".join(sys.argv),
                "environment": sys.executable,
                "random_seeds": {"claim_particle": RETRY_CLAIM_PARTICLE_SEEDS},
                "plan": PLAN.as_posix(),
                "result": (output_root / "result.json").as_posix(),
                "tuning_artifact": tuning_path.as_posix(),
                "memory_policy": MEMORY_POLICY,
                "device": payload["device"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_result_note(payload)
    return payload


def _write_result_note(payload: dict[str, Any]) -> None:
    claim = payload["claim"]
    genut = claim["genut"]["summary"]
    ukf = claim["principal_sqrt_ukf"]
    lines = [
        "# GenUT Chapter 18b Structural Leaderboard Result",
        "",
        "Date: 2026-07-22",
        "",
        f"Status: `{payload['decision']['status']}`",
        "",
        "The row is included as a candidate GenUT extension under the existing",
        f"`{STRUCTURAL_UKF_SCOPE}` target. It is not automatic method admission or",
        "default promotion.",
        "",
        "## Value and Score",
        "",
        "| Quantity | GenUT mean | GenUT 95% CI | Existing structural UKF |",
        "|---|---:|---:|---:|",
    ]
    if genut is None:
        lines.append(
            f"| value | blocked: non-finite claim | N/A | {ukf['value']:.8f} |"
        )
        for index, name in enumerate(STRUCTURAL_PARAMETER_NAMES):
            lines.append(
                f"| source score `{name}` | blocked: non-finite claim | N/A | "
                f"{ukf['source_score'][index]:.8f} |"
            )
    else:
        lines.append(
            f"| value | {genut['value']['mean']:.8f} | "
            f"[{genut['value']['ci95_lower']:.8f}, {genut['value']['ci95_upper']:.8f}] | "
            f"{ukf['value']:.8f} |"
        )
        for index, name in enumerate(STRUCTURAL_PARAMETER_NAMES):
            lines.append(
                f"| source score `{name}` | {genut[name]['mean']:.8f} | "
                f"[{genut[name]['ci95_lower']:.8f}, {genut[name]['ci95_upper']:.8f}] | "
                f"{ukf['source_score'][index]:.8f} |"
            )
    lines.extend(
        [
            "",
            "## Structural and Numerical Gates",
            "",
            f"- Maximum pre-reset transition residual: `{claim['genut']['maximum_transition_residual']:.3e}`.",
            f"- Maximum aggregate reset/marginal/score-sum residual: `{claim['genut']['maximum_residual']:.3e}`.",
            f"- Maximum reset/marginal residual: `{claim['genut']['maximum_reset_residual']:.3e}`.",
            f"- Maximum relative score-increment accounting residual: `{claim['genut']['maximum_score_sum_relative_residual']:.3e}`.",
            "- Process noise dimension: one scalar innovation; no independent `k` shock.",
            "- Initial observation ordering: `y0` assimilated before the first transition.",
            "- Runtime score: recursive forward sensitivity; no autodiff or finite difference.",
            f"- Frozen data hashes: state `{claim['dataset']['state_sha256']}`, observation `{claim['dataset']['observation_sha256']}`.",
            "",
            "## Decision",
            "",
            "| Decision | Status | Interpretation |",
            "|---|---|---|",
            f"| Candidate leaderboard row | `{payload['decision']['leaderboard_row_status']}` | Existing STR-UKF target, GenUT method extension included with raw evidence |",
            "| Exact likelihood/score | `not established` | Existing UKF is not an oracle and GenUT has no exact nonlinear oracle here |",
            "| Default/HMC promotion | `not evaluated` | Requires the cross-model admission contract and stronger score evidence |",
            "",
            "## Inference Status",
            "",
            "| Evidence class | Status |",
            "|---|---|",
            "| Hard veto screen | Passed if result status is candidate-ready |",
            "| Statistically supported ranking | None; UKF comparison is deterministic diagnostic evidence |",
            "| Descriptive-only differences | GenUT-minus-UKF value and score differences |",
            "| Default readiness | Not established |",
            "| Next evidence needed | Cross-model candidate admission, independent score authority, and high-dimensional memory validation |",
        ]
    )
    (ROOT / RESULT_NOTE).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_capacity(output_root: Path) -> dict[str, Any]:
    """Run the predeclared N=1002 T={2,10} trusted-GPU capacity probe."""

    _require_serious_gpu_policy()
    started = time.perf_counter()
    output_root.mkdir(parents=True, exist_ok=False)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    logical = tf.config.list_logical_devices("GPU")
    if not logical:
        raise RuntimeError("STR-UKF GenUT capacity probe requires a logical GPU")
    theta = tf.cast(structural_truth_source(), tf.float32)
    _states, observations64 = generate_frozen_structural_dataset_tf()
    design = _genut_design()
    controls = {
        "epsilon": 2.0,
        "sinkhorn_steps": 4,
        "balance_steps": 8,
        "ridge": 1.0e-5,
    }
    rows = []
    for horizon in (2, 10):
        evaluate = _make_evaluator(controls, horizon)
        initial, process = _particle_noise(2026072241, horizon)
        before = time.perf_counter()
        row = _evaluate(
            evaluate,
            theta,
            tf.cast(observations64[:horizon], tf.float32),
            initial,
            process,
            design,
        )
        row.update(
            {
                "horizon": horizon,
                "wall_time_seconds_including_compile": time.perf_counter() - before,
                "allocator": {
                    key: int(value)
                    for key, value in tf.config.experimental.get_memory_info("GPU:0").items()
                },
            }
        )
        rows.append(row)
    peak = max(row["allocator"]["peak"] for row in rows)
    passed = (
        all(row["finite"] and "GPU" in row["device"].upper() for row in rows)
        and all(row["max_transition_residual"] <= TRANSITION_RESIDUAL_TOLERANCE for row in rows)
        and all(row["maximum_residual"] <= RESET_RESIDUAL_TOLERANCE for row in rows)
        and rows[-1]["wall_time_seconds_including_compile"] <= 300.0
        and peak <= MAX_MEMORY_BYTES
    )
    payload = {
        "schema_version": "bayesfilter.genut_str_ukf_capacity.v1",
        "campaign_id": CAMPAIGN_ID,
        "git_commit": _git_commit(),
        "plan": PLAN.as_posix(),
        "target_scope": STRUCTURAL_UKF_SCOPE,
        "particle_count": N,
        "controls": controls,
        "rows": rows,
        "memory_policy": MEMORY_POLICY,
        "device": {
            "logical_devices": [item.name for item in logical],
            "dtype": "float32",
            "tf32_enabled": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
            "jit_compile": True,
            "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        },
        "passed": passed,
        "status": "PASS_CAPACITY" if passed else "BLOCK_CAPACITY",
        "wall_time_seconds": time.perf_counter() - started,
    }
    (output_root / "capacity.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def run_tuning_diagnostic(output_root: Path) -> dict[str, Any]:
    """Replay only the frozen tuning grid and preserve every rejection metric."""

    _require_serious_gpu_policy()
    started = time.perf_counter()
    output_root.mkdir(parents=True, exist_ok=False)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    logical = tf.config.list_logical_devices("GPU")
    if not logical:
        raise RuntimeError("STR-UKF GenUT tuning diagnostic requires a logical GPU")
    design = _genut_design()
    theta = tf.cast(structural_truth_source(), tf.float32)
    tuning = _tune(theta, design)
    payload = {
        "schema_version": "bayesfilter.genut_str_ukf_tuning_diagnostic.v1",
        "campaign_id": CAMPAIGN_ID,
        "git_commit": _git_commit(),
        "plan": PLAN.as_posix(),
        "memory_policy": MEMORY_POLICY,
        "device": {
            "logical_devices": [item.name for item in logical],
            "dtype": "float32",
            "tf32_enabled": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
            "jit_compile": True,
        },
        "tuning": tuning,
        "wall_time_seconds": time.perf_counter() - started,
    }
    (output_root / "tuning_diagnostic.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def run(output_root: Path) -> dict[str, Any]:
    _require_serious_gpu_policy()
    started = time.perf_counter()
    output_root.mkdir(parents=True, exist_ok=False)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    logical = tf.config.list_logical_devices("GPU")
    if not logical:
        raise RuntimeError("STR-UKF GenUT campaign requires a logical GPU")
    design = _genut_design()
    theta = tf.cast(structural_truth_source(), tf.float32)
    tuning = _tune(theta, design)
    (output_root / "tuning.json").write_text(
        json.dumps(tuning, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if tuning["selected_controls"] is None:
        raise RuntimeError("no eligible STR-UKF GenUT tuning candidate")
    claim = _claim(theta, dict(tuning["selected_controls"]), design)
    allocator = tf.config.experimental.get_memory_info("GPU:0")
    peak_bytes = int(allocator.get("peak", 0))
    hard_pass = (
        claim["genut"]["all_finite"]
        and claim["genut"]["maximum_reset_residual"] <= RESET_RESIDUAL_TOLERANCE
        and claim["genut"]["maximum_transition_residual"]
        <= TRANSITION_RESIDUAL_TOLERANCE
        and claim["genut"]["maximum_score_sum_relative_residual"]
        <= SCORE_SUM_RELATIVE_TOLERANCE
        and claim["principal_sqrt_ukf"]["valid"]
        and peak_bytes <= MAX_MEMORY_BYTES
    )
    status = "CANDIDATE_LEADERBOARD_ROW_INCLUDED" if hard_pass else "CANDIDATE_BLOCKED_HARD_GATE"
    source_paths = (
        Path(__file__).relative_to(ROOT),
        Path("bayesfilter/highdim/cubature_genut_candidate.py"),
        Path("bayesfilter/highdim/cubature_genut_filter.py"),
        Path("bayesfilter/highdim/cubature_genut_adapters.py"),
        Path("bayesfilter/testing/structural_ukf_neutra_target_design_tf.py"),
        PLAN,
    )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": CAMPAIGN_ID,
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_time_seconds": time.perf_counter() - started,
        "git_commit": _git_commit(),
        "plan": PLAN.as_posix(),
        "target": {
            "target_scope": STRUCTURAL_UKF_SCOPE,
            "parameter_names": STRUCTURAL_PARAMETER_NAMES,
            "truth_physical": [float(v) for v in STRUCTURAL_TRUTH_PHYSICAL.numpy()],
            "horizon": HORIZON,
            "particle_count": N,
            "transition_restriction": "scalar innovation; k deterministic from previous k and current m",
        },
        "device": {
            "logical_devices": [item.name for item in logical],
            "dtype": "float32",
            "tf32_enabled": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
            "jit_compile": True,
            "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        },
        "memory_policy": MEMORY_POLICY,
        "gpu_allocator": {key: int(value) for key, value in allocator.items()},
        "source_sha256": {path.as_posix(): _sha256(ROOT / path) for path in source_paths},
        "tuning": {
            "selected_controls": tuning["selected_controls"],
            "artifact": (output_root / "tuning.json").as_posix(),
            "claim_data_read_during_selection": False,
        },
        "claim": claim,
        "engineering_ledger": {
            "finite": claim["genut"]["all_finite"],
            "gpu_xla_tf32": True,
            "memory_growth": True,
            "reset_residual_gate": (
                claim["genut"]["maximum_reset_residual"]
                <= RESET_RESIDUAL_TOLERANCE
            ),
            "max_transition_residual": claim["genut"]["maximum_transition_residual"],
            "transition_residual_gate": (
                claim["genut"]["maximum_transition_residual"]
                <= TRANSITION_RESIDUAL_TOLERANCE
            ),
            "score_sum_relative_gate": (
                claim["genut"]["maximum_score_sum_relative_residual"]
                <= SCORE_SUM_RELATIVE_TOLERANCE
            ),
            "allocator_peak_bytes": peak_bytes,
        },
        "scientific_ledger": {
            "candidate_leaderboard_row_included": hard_pass,
            "leaderboard_admitted": False,
            "default_changed": False,
            "statistically_supported_ranking": False,
        },
        "decision": {
            "status": status,
            "leaderboard_row_status": "included_candidate_not_admitted" if hard_pass else "blocked",
            "leaderboard_admitted": False,
            "default_changed": False,
        },
        "nonclaims": [
            "no exact nonlinear likelihood or score",
            "no unbiasedness or superiority",
            "no HMC or default readiness",
            "no high-dimensional feasibility",
            "no Zhao-Cui source-faithfulness conclusion",
        ],
    }
    (output_root / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_root / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "bayesfilter.serious_run_manifest.v1",
                "git_commit": payload["git_commit"],
                "command": " ".join(sys.argv),
                "environment": sys.executable,
                "random_seeds": {
                    "calibration": CALIBRATION_SEEDS,
                    "validation": VALIDATION_SEEDS,
                    "tuning_particle": TUNING_PARTICLE_SEEDS,
                    "claim_particle": CLAIM_PARTICLE_SEEDS,
                },
                "plan": PLAN.as_posix(),
                "result": (output_root / "result.json").as_posix(),
                "memory_policy": MEMORY_POLICY,
                "device": payload["device"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_result_note(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("capacity", "tuning", "claim-resume", "full"), default="full"
    )
    parser.add_argument("--tuning-artifact", type=Path)
    args = parser.parse_args()
    if args.phase == "capacity":
        payload = run_capacity(args.output_root)
        status = payload["status"]
    elif args.phase == "tuning":
        payload = run_tuning_diagnostic(args.output_root)
        status = payload["tuning"]["status"]
    elif args.phase == "claim-resume":
        if args.tuning_artifact is None:
            raise ValueError("claim-resume requires --tuning-artifact")
        payload = run_claim_resume(args.output_root, args.tuning_artifact)
        status = payload["decision"]["status"]
    else:
        payload = run(args.output_root)
        status = payload["decision"]["status"]
    print(json.dumps({"status": status, "output": str(args.output_root)}))


if __name__ == "__main__":
    main()
