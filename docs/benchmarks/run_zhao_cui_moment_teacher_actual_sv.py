#!/usr/bin/env python3
"""Run the reviewed exact transformed-SV moment-teacher campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time


os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf


MEMORY_POLICY = None
if "--reference-only" not in sys.argv:
    from bayesfilter.runtime.gpu_memory_policy import (
        configure_tensorflow_gpu_memory_growth,
    )

    MEMORY_POLICY = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(False)

from bayesfilter.highdim.cubature_genut_adapters import (
    exact_transformed_sv_candidate_adapter,
)
from bayesfilter.highdim.ledh_contract_e_identity import (
    issue_moment_teacher_actual_sv_contract_e_route_identity,
)
from bayesfilter.highdim.sv_mixture_cut4 import (
    ExactTransformedSVSSM,
    _legendre_interval_nodes_weights,
    _logsumexp_weighted,
    _normal_log_prob,
    exact_log_chi_square_log_density,
)
from bayesfilter.highdim.zhao_cui_moment_teacher_lgssm_tf import (
    MomentTeacherControls,
)
from bayesfilter.highdim.zhao_cui_moment_teacher_nonlinear_tf import (
    ACTUAL_SV_ROUTE_ID,
    EVENT_ORDER,
    freeze_nonlinear_teacher_scale_shift_indices,
    issue_nonlinear_moment_teacher_tuning_artifact,
    make_nonlinear_moment_teacher_value_and_score_tf,
    make_nonlinear_tuning_scope,
    prepare_nonlinear_teacher_inputs,
    route_identity_prepared_inputs,
)
from bayesfilter.testing.zhao_cui_actual_sv_target_tf import (
    ACTUAL_SV_DATASET_ID,
    actual_sv_unconstrained_theta_tf,
    generate_source_order_actual_sv_dataset_tf,
)


SCHEMA = "bayesfilter.zhao_cui_moment_teacher_actual_sv.v1"
PLAN = "docs/plans/bayesfilter-zhao-cui-moment-teacher-actual-sv-campaign-plan-2026-07-31.md"
PARTICLE_COUNT = 1024
CALIBRATION_SEED = 83900
VALIDATION_SEED = 83901
CLAIM_SEEDS = tuple(range(83910, 83916))
T_CRITICAL_DF5 = 2.570581835636314


def _json(value):
    if isinstance(value, tf.Tensor):
        return _json(value.numpy())
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(item) for item in value]
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)


def _tensor_hash(value: tf.Tensor) -> str:
    return hashlib.sha256(bytes(tf.io.serialize_tensor(value).numpy())).hexdigest()


def _source_hash(path: str) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def _controls(candidate_index: int, *, repair_ladder: bool) -> MomentTeacherControls:
    return MomentTeacherControls(
        sinkhorn_steps=20,
        balance_steps=(20, 32)[candidate_index] if repair_ladder else 8,
        correction_steps=1,
        correction_strength=(0.005, 0.01)[candidate_index],
        correction_floor=1.0e-6,
        pairwise_correction_steps=0,
        pairwise_strength=0.0,
        pairwise_floor=1.0e-6,
        tt_ridge=(1.0e-3, 1.0e-4)[candidate_index],
        column_scale_floor=1.0e-6,
        condition_number_veto=1.0e8,
        fit_residual_veto=2.0,
    )


def _particle_prepared(observations: tf.Tensor, seed: int) -> dict[str, tf.Tensor]:
    horizon = int(observations.shape[0])
    raw_design = tf.random.stateless_normal(
        [horizon, PARTICLE_COUNT, 1], [seed, 2001], dtype=tf.float64
    )
    centered_design = raw_design - tf.reduce_mean(raw_design, axis=1, keepdims=True)
    centered_design *= tf.sqrt(
        tf.cast(PARTICLE_COUNT, tf.float64)
        / tf.cast(PARTICLE_COUNT - 1, tf.float64)
    )
    return {
        "observations": observations,
        "initial_noise": tf.cast(
            tf.random.stateless_normal(
                [PARTICLE_COUNT, 1], [seed, 101], dtype=tf.float64
            ),
            tf.float32,
        ),
        "process_noise": tf.cast(
            tf.random.stateless_normal(
                [horizon, PARTICLE_COUNT, 1], [seed, 1001], dtype=tf.float64
            ),
            tf.float32,
        ),
        "residual_design": tf.cast(centered_design, tf.float32),
        "prepared_ridge": tf.fill([horizon], tf.constant(1.0e-5, tf.float32)),
        "epsilon": tf.constant(0.5, tf.float32),
        "scaling": tf.constant(0.9, tf.float32),
    }


def _dense_source_order_value(
    transformed_observations: tf.Tensor,
    theta: tf.Tensor,
    *,
    order: int,
    radius: float,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Return source-order `x0 -> x1 -> y1` dense value and increments."""

    model = ExactTransformedSVSSM(sigma=1.0)
    theta = tf.convert_to_tensor(theta, tf.float64)
    observations = tf.convert_to_tensor(transformed_observations, tf.float64)
    grid, weights = _legendre_interval_nodes_weights(
        order=order, left=-radius, right=radius
    )
    parameters = model.physical_parameters(theta)
    gamma = parameters["gamma"]
    sigma = parameters["sigma"]
    beta = parameters["beta"]
    prior_scale = sigma / tf.sqrt(1.0 - tf.square(gamma))
    previous_density = tf.exp(
        _normal_log_prob(grid, tf.constant(0.0, tf.float64), prior_scale)
    )
    increments = []
    for time_index in range(int(observations.shape[0])):
        transition_log = _normal_log_prob(
            grid[:, None], gamma * grid[None, :], sigma
        )
        predictive = tf.reduce_sum(
            weights[None, :] * previous_density[None, :] * tf.exp(transition_log),
            axis=1,
        )
        residual = observations[time_index, 0] - 2.0 * tf.math.log(beta) - grid
        log_density = tf.math.log(predictive) + exact_log_chi_square_log_density(
            residual
        )
        increment = _logsumexp_weighted(log_density, weights)
        previous_density = tf.exp(log_density - increment)
        increments.append(increment)
    stacked = tf.stack(increments)
    return tf.reduce_sum(stacked), stacked


def _dense_arm(
    observations: tf.Tensor, theta: tf.Tensor, *, order: int, radius: float
) -> dict[str, object]:
    with tf.device("/CPU:0"):
        theta64 = tf.cast(theta, tf.float64)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(theta64)
            value, increments = _dense_source_order_value(
                observations, theta64, order=order, radius=radius
            )
        score = tape.gradient(value, theta64)
        increment_scores = tape.jacobian(increments, theta64)
        del tape
    if score is None or increment_scores is None:
        raise RuntimeError("actual-SV dense reference differentiation failed")
    return {
        "order": order,
        "radius": radius,
        "value": value,
        "score": score,
        "increments": increments,
        "increment_scores": increment_scores,
        "score_increment_sum_residual": tf.reduce_max(
            tf.abs(tf.reduce_sum(increment_scores, axis=0) - score)
        ),
    }


def _dense_reference(observations: tf.Tensor, theta: tf.Tensor) -> dict[str, object]:
    arms = [
        _dense_arm(observations, theta, order=257, radius=8.0),
        _dense_arm(observations, theta, order=401, radius=8.0),
        _dense_arm(observations, theta, order=401, radius=10.0),
    ]
    reference = arms[-1]
    comparisons = []
    for arm in arms[:-1]:
        comparisons.append(
            {
                "order": arm["order"],
                "radius": arm["radius"],
                "value_difference": arm["value"] - reference["value"],
                "score_difference": arm["score"] - reference["score"],
            }
        )
    maximum_value_gap = tf.reduce_max(
        tf.abs(tf.stack([item["value_difference"] for item in comparisons]))
    )
    maximum_score_gap = tf.reduce_max(
        tf.abs(tf.stack([item["score_difference"] for item in comparisons]))
    )
    maximum_sum_residual = tf.reduce_max(
        tf.stack([item["score_increment_sum_residual"] for item in arms])
    )
    valid = bool(
        (
            (maximum_value_gap <= tf.constant(5.0e-5, tf.float64))
            & (maximum_score_gap <= tf.constant(2.0e-4, tf.float64))
            & (maximum_sum_residual <= tf.constant(1.0e-10, tf.float64))
        ).numpy()
    )
    return {
        "status": "pass" if valid else "hard_veto",
        "valid": valid,
        "target": "source_order_exact_transformed_sv_x0_to_x1_to_y1",
        "value": reference["value"],
        "score": reference["score"],
        "increments": reference["increments"],
        "increment_scores": reference["increment_scores"],
        "arms": arms,
        "comparisons": comparisons,
        "maximum_value_gap": maximum_value_gap,
        "maximum_score_gap": maximum_score_gap,
        "maximum_score_increment_sum_residual": maximum_sum_residual,
    }


def _finite(result: dict[str, tf.Tensor]) -> bool:
    return bool(
        tf.math.is_finite(result["objective"]).numpy()
        and tf.reduce_all(tf.math.is_finite(result["score"])).numpy()
        and result["valid_chart"].numpy()
        and result["teacher_valid"].numpy()
        and tf.reduce_all(result["shape_valid_history"]).numpy()
    )


def _evaluate(evaluate, baseline, theta: tf.Tensor) -> dict[str, object]:
    started = time.perf_counter()
    candidate = evaluate(theta)
    candidate = evaluate(theta)
    empirical = baseline(theta)
    if hasattr(tf.experimental, "sync_devices"):
        tf.experimental.sync_devices()
    return {
        "objective": candidate["objective"],
        "score": candidate["score"],
        "empirical_contract_e_objective": empirical["objective"],
        "empirical_contract_e_score": empirical["score"],
        "paired_objective_difference": candidate["objective"] - empirical["objective"],
        "paired_score_difference": candidate["score"] - empirical["score"],
        "finite_valid": _finite(candidate),
        "baseline_valid": _finite(empirical),
        "candidate_valid_chart": candidate["valid_chart"],
        "candidate_teacher_valid": candidate["teacher_valid"],
        "candidate_all_shapes_valid": tf.reduce_all(
            candidate["shape_valid_history"]
        ),
        "baseline_valid_chart": empirical["valid_chart"],
        "baseline_teacher_valid": empirical["teacher_valid"],
        "baseline_all_shapes_valid": tf.reduce_all(
            empirical["shape_valid_history"]
        ),
        "maximum_mean_residual": tf.reduce_max(candidate["mean_residual_history"]),
        "maximum_covariance_residual": tf.reduce_max(
            candidate["covariance_residual_history"]
        ),
        "maximum_skew_residual": tf.reduce_max(candidate["skew_residual_history"]),
        "wall_seconds": time.perf_counter() - started,
    }


def _paired_summary(rows: list[dict[str, object]], reference: dict[str, object]):
    labels = ("value", "z_gamma", "log_beta")
    reference_values = [float(reference["value"]), *[float(x) for x in reference["score"]]]
    output = {}
    for index, label in enumerate(labels):
        candidate = [
            float(row["objective"] if index == 0 else row["score"][index - 1])
            for row in rows
        ]
        baseline = [
            float(
                row["empirical_contract_e_objective"]
                if index == 0
                else row["empirical_contract_e_score"][index - 1]
            )
            for row in rows
        ]
        candidate_errors = [value - reference_values[index] for value in candidate]
        baseline_errors = [value - reference_values[index] for value in baseline]
        gains = [
            abs(baseline_errors[row]) - abs(candidate_errors[row])
            for row in range(len(rows))
        ]
        gain_mean = statistics.mean(gains)
        gain_se = statistics.stdev(gains) / math.sqrt(len(gains))
        candidate_mcse = statistics.stdev(candidate_errors) / math.sqrt(len(rows))
        baseline_mcse = statistics.stdev(baseline_errors) / math.sqrt(len(rows))
        lower = gain_mean - T_CRITICAL_DF5 * gain_se
        upper = gain_mean + T_CRITICAL_DF5 * gain_se
        output[label] = {
            "reference": reference_values[index],
            "candidate_mean": statistics.mean(candidate),
            "baseline_mean": statistics.mean(baseline),
            "candidate_mean_error": statistics.mean(candidate_errors),
            "baseline_mean_error": statistics.mean(baseline_errors),
            "candidate_mcse": candidate_mcse,
            "baseline_mcse": baseline_mcse,
            "candidate_absolute_mean_error_over_mcse": (
                abs(statistics.mean(candidate_errors)) / candidate_mcse
                if candidate_mcse > 0.0 else None
            ),
            "baseline_absolute_mean_error_over_mcse": (
                abs(statistics.mean(baseline_errors)) / baseline_mcse
                if baseline_mcse > 0.0 else None
            ),
            "candidate_mean_absolute_error": statistics.mean(
                abs(value) for value in candidate_errors
            ),
            "baseline_mean_absolute_error": statistics.mean(
                abs(value) for value in baseline_errors
            ),
            "absolute_error_gain_mean": gain_mean,
            "absolute_error_gain_se": gain_se,
            "absolute_error_gain_ci95_lower": lower,
            "absolute_error_gain_ci95_upper": upper,
            "candidate_closer_seed_count": sum(value > 0.0 for value in gains),
            "supported_improvement": lower > 0.0,
            "supported_regression": upper < 0.0,
            "candidate_error_signs": [
                "positive" if value > 0 else "negative" if value < 0 else "zero"
                for value in candidate_errors
            ],
            "baseline_error_signs": [
                "positive" if value > 0 else "negative" if value < 0 else "zero"
                for value in baseline_errors
            ],
        }
    return output


def _device_payload() -> dict[str, object]:
    return {
        "memory_policy": MEMORY_POLICY,
        "physical_gpus": [item.name for item in tf.config.list_physical_devices("GPU")],
        "logical_gpus": [item.name for item in tf.config.list_logical_devices("GPU")],
        "allocator": {
            name: int(value)
            for name, value in tf.config.experimental.get_memory_info("GPU:0").items()
        },
        "tf32_enabled": tf.config.experimental.tensor_float_32_execution_enabled(),
        "jit_compile": True,
        "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--repair-ladder", action="store_true")
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    started = time.perf_counter()

    _, raw_observations, transformed64 = generate_source_order_actual_sv_dataset_tf()
    observations = tf.cast(transformed64, tf.float32)
    theta = tf.cast(actual_sv_unconstrained_theta_tf(), tf.float32)
    adapter = exact_transformed_sv_candidate_adapter(sigma=1.0)
    dense_reference = _dense_reference(transformed64, theta)
    if arguments.reference_only:
        payload = {
            "schema": SCHEMA,
            "status": "pass" if dense_reference["valid"] else "hard_veto",
            "mode": "cpu_hidden_reference_only",
            "target": "source_order_exact_transformed_sv",
            "dataset_id": ACTUAL_SV_DATASET_ID,
            "raw_observation_sha256": _tensor_hash(raw_observations),
            "transformed_observation_sha256": _tensor_hash(transformed64),
            "theta": theta,
            "dense_reference": dense_reference,
            "device": {
                "gpu_intentionally_hidden": os.environ.get("CUDA_VISIBLE_DEVICES") == "-1",
                "physical_gpus": [
                    item.name for item in tf.config.list_physical_devices("GPU")
                ],
            },
            "command": " ".join(sys.argv),
            "plan": PLAN,
            "wall_time_seconds": time.perf_counter() - started,
        }
        result_path = output / "result.json"
        result_path.write_text(
            json.dumps(_json(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"status": payload["status"], "result": str(result_path)}))
        if payload["status"] != "pass":
            raise SystemExit(2)
        return
    if not dense_reference["valid"]:
        payload = {
            "schema": SCHEMA,
            "status": "dense_reference_hard_veto",
            "dense_reference": dense_reference,
            "command": " ".join(sys.argv),
            "plan": PLAN,
            "wall_time_seconds": time.perf_counter() - started,
        }
        (output / "result.json").write_text(
            json.dumps(_json(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise SystemExit(2)

    horizon = int(observations.shape[0])
    scope = make_nonlinear_tuning_scope(
        model_id="zhao_cui_actual_transformed_sv_T20",
        target_id=ACTUAL_SV_DATASET_ID,
        route_id=ACTUAL_SV_ROUTE_ID,
        horizon=horizon,
        prepared_data_id=f"{ACTUAL_SV_DATASET_ID}_fp32_{EVENT_ORDER}",
        particle_count=PARTICLE_COUNT,
        state_dimension=1,
        parameter_count=2,
        dtype=tf.float32,
        tf32_enabled=False,
        jit_compile=True,
    )
    charts = (
        (tf.constant([0.0], tf.float32), tf.constant([5.0], tf.float32)),
        (tf.constant([0.0], tf.float32), tf.constant([7.0], tf.float32)),
    )
    calibration = _particle_prepared(observations, CALIBRATION_SEED)
    validation = _particle_prepared(observations, VALIDATION_SEED)
    candidate_rows = []
    selected = None
    for candidate_index, (offset, scale) in enumerate(charts):
        controls = _controls(candidate_index, repair_ladder=arguments.repair_ladder)
        try:
            teacher = prepare_nonlinear_teacher_inputs(
                adapter=adapter,
                observations=observations,
                state_offset=offset,
                state_scale=scale,
                center_theta=theta,
                initial_standard_deviation=1.25,
                process_standard_deviation=1.0,
                fit_rows=96,
                basis_size=2,
                rank=1,
                sweeps=1,
                defensive_weight=0.0,
                pair_indices=tf.zeros([0, 2], tf.int32),
                root_seed=85000 + candidate_index,
            )
            teacher = freeze_nonlinear_teacher_scale_shift_indices(
                teacher,
                controls,
                adapter,
                initial_variance=1.0,
                process_variance=1.0,
            )
            artifact = issue_nonlinear_moment_teacher_tuning_artifact(
                scope=scope,
                controls=controls,
                calibration_data_id=f"particle_seed_{CALIBRATION_SEED}",
                validation_data_id=f"particle_seed_{VALIDATION_SEED}",
                selection_record_id=(
                    f"candidate_{candidate_index}_valid_then_validation_skew_residual"
                ),
                chart_id=(
                    f"candidate_{candidate_index}_{_tensor_hash(offset)}_"
                    f"{_tensor_hash(scale)}"
                ),
                pair_set_id="scalar_empty_pair_set_v1",
            )

            def evaluators(prepared):
                return (
                    make_nonlinear_moment_teacher_value_and_score_tf(
                        adapter=adapter,
                        particle_prepared=prepared,
                        teacher_prepared=teacher,
                        tuning_artifact=artifact,
                        expected_scope=scope,
                        initial_variance=1.0,
                        process_variance=1.0,
                        jit_compile=True,
                    ),
                    make_nonlinear_moment_teacher_value_and_score_tf(
                        adapter=adapter,
                        particle_prepared=prepared,
                        teacher_prepared=None,
                        tuning_artifact=artifact,
                        expected_scope=scope,
                        initial_variance=1.0,
                        process_variance=1.0,
                        jit_compile=True,
                    ),
                )

            calibration_eval, calibration_base = evaluators(calibration)
            validation_eval, validation_base = evaluators(validation)
            calibration_row = _evaluate(calibration_eval, calibration_base, theta)
            validation_row = _evaluate(validation_eval, validation_base, theta)
            row = {
                "candidate_index": candidate_index,
                "artifact": artifact.to_dict(),
                "chart_offset": offset,
                "chart_scale": scale,
                "calibration": calibration_row,
                "validation": validation_row,
            }
            candidate_rows.append(row)
            valid = bool(
                calibration_row["finite_valid"]
                and calibration_row["baseline_valid"]
                and validation_row["finite_valid"]
                and validation_row["baseline_valid"]
            )
            metric = (float(validation_row["maximum_skew_residual"]), candidate_index)
            if valid and (selected is None or metric < selected[0]):
                selected = (metric, teacher, artifact, row)
        except Exception as error:
            candidate_rows.append(
                {
                    "candidate_index": candidate_index,
                    "status": "infrastructure_or_tuning_hard_veto",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )

    common = {
        "schema": SCHEMA,
        "target": "source_order_exact_transformed_sv",
        "target_classification": "fixed_hmc_adaptation_plus_extension_or_invention",
        "event_order": EVENT_ORDER,
        "particle_count": PARTICLE_COUNT,
        "horizon": horizon,
        "theta": theta,
        "physical_parameters": {"gamma": 0.6, "beta": 0.4, "sigma": 1.0},
        "dataset_id": ACTUAL_SV_DATASET_ID,
        "raw_observation_sha256": _tensor_hash(raw_observations),
        "transformed_observation_sha256": _tensor_hash(transformed64),
        "scope": scope.as_dict(),
        "calibration_seed": CALIBRATION_SEED,
        "validation_seed": VALIDATION_SEED,
        "claim_seeds": CLAIM_SEEDS,
        "repair_ladder": arguments.repair_ladder,
        "dense_reference": dense_reference,
        "candidate_rows": candidate_rows,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_sha256": {
            path: _source_hash(path)
            for path in (
                "bayesfilter/highdim/cubature_genut_adapters.py",
                "bayesfilter/highdim/cubature_genut_filter.py",
                "bayesfilter/highdim/ledh_contract_e_identity.py",
                "bayesfilter/highdim/zhao_cui_moment_teacher_nonlinear_tf.py",
                "bayesfilter/highdim/zhao_cui_moment_teacher_xla.py",
                "bayesfilter/highdim/higher_moment_contract_e.py",
                "bayesfilter/highdim/ledh_contract_e_streaming_tf.py",
                "bayesfilter/highdim/ledh_contract_e_reset_tf.py",
                "bayesfilter/testing/zhao_cui_actual_sv_target_tf.py",
            )
        },
        "command": " ".join(sys.argv),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "tensorflow": tf.__version__,
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "unset"),
        },
        "plan": PLAN,
    }
    if selected is None:
        payload = {
            **common,
            "status": "tuning_hard_veto",
            "claim_seeds_evaluated": False,
            "device": _device_payload(),
            "wall_time_seconds": time.perf_counter() - started,
            "nonclaims": [
                "tuning failed before claim execution",
                "not evidence against the particle or Contract E implementation",
                "not HMC or posterior readiness",
                "not source-faithful Zhao-Cui filtering",
            ],
        }
        (output / "result.json").write_text(
            json.dumps(_json(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise SystemExit(3)

    _, teacher, artifact, selected_row = selected
    claim_rows = []
    identities = []
    graph_operations = set()
    for seed in CLAIM_SEEDS:
        prepared = _particle_prepared(observations, seed)
        candidate = make_nonlinear_moment_teacher_value_and_score_tf(
            adapter=adapter,
            particle_prepared=prepared,
            teacher_prepared=teacher,
            tuning_artifact=artifact,
            expected_scope=scope,
            initial_variance=1.0,
            process_variance=1.0,
            jit_compile=True,
        )
        baseline = make_nonlinear_moment_teacher_value_and_score_tf(
            adapter=adapter,
            particle_prepared=prepared,
            teacher_prepared=None,
            tuning_artifact=artifact,
            expected_scope=scope,
            initial_variance=1.0,
            process_variance=1.0,
            jit_compile=True,
        )
        row = _evaluate(candidate, baseline, theta)
        row["seed"] = seed
        identity = issue_moment_teacher_actual_sv_contract_e_route_identity(
            prepared_inputs=route_identity_prepared_inputs(prepared, teacher, artifact)
        ).to_dict()
        row["route_identity_sha256"] = identity["identity_sha256"]
        claim_rows.append(row)
        identities.append(identity)
        graph = candidate.get_concrete_function().graph.as_graph_def()
        graph_operations.update(node.op for node in graph.node)
        for function in graph.library.function:
            graph_operations.update(node.op for node in function.node_def)

    paired_summary = _paired_summary(claim_rows, dense_reference)
    hard_vetoes = {
        "dense_reference_invalid": not dense_reference["valid"],
        "claim_invalid": not all(row["finite_valid"] for row in claim_rows),
        "baseline_invalid": not all(row["baseline_valid"] for row in claim_rows),
        "mean_covariance_restoration": any(
            float(row["maximum_mean_residual"]) > 2.0e-5
            or float(row["maximum_covariance_residual"]) > 2.0e-4
            for row in claim_rows
        ),
        "route_identity_count_mismatch": len(identities) != len(CLAIM_SEEDS),
        "tf32_enabled": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
        "pyfunc": bool({"PyFunc", "EagerPyFunc"} & graph_operations),
        "missing_while": not bool({"While", "StatelessWhile"} & graph_operations),
    }
    payload = {
        **common,
        "status": "pass" if not any(hard_vetoes.values()) else "hard_veto",
        "selected_candidate_index": selected_row["candidate_index"],
        "selected_tuning_artifact": artifact.to_dict(),
        "claim_rows": claim_rows,
        "paired_accuracy_summary": paired_summary,
        "route_identities": identities,
        "hard_vetoes": hard_vetoes,
        "graph": {
            "has_while": bool({"While", "StatelessWhile"} & graph_operations),
            "pyfunc_count": sum(
                operation in {"PyFunc", "EagerPyFunc"}
                for operation in graph_operations
            ),
        },
        "device": _device_payload(),
        "wall_time_seconds": time.perf_counter() - started,
        "nonclaims": [
            "six seeds provide feasibility-level paired uncertainty only",
            "not KSC-SV evidence",
            "not HMC or posterior readiness",
            "not high-dimensional scalability evidence",
            "not source-faithful Zhao-Cui filtering",
        ],
    }
    result_path = output / "result.json"
    result_path.write_text(
        json.dumps(_json(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": payload["status"], "result": str(result_path)}))
    if payload["status"] != "pass":
        raise SystemExit(4)


if __name__ == "__main__":
    main()
