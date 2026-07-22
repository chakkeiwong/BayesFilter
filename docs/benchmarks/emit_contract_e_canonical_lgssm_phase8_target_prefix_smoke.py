#!/usr/bin/env python3
"""Emit the bounded CPU-hidden Phase 8 target-prefix wiring smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

from bayesfilter.highdim import ledh_contract_e_canonical_lgssm_tf as canonical
from bayesfilter.highdim import ledh_contract_e_lgssm_preparation_tf as preparation
from bayesfilter.highdim.transport_chunk_policy import select_transport_chunks
from bayesfilter.linear.kalman_tf import tf_kalman_log_likelihood
from scripts.filtering_value_gradient_benchmark_generate_p8_datasets import (
    _lgssm_dataset,
)


SCHEMA_VERSION = "bayesfilter.contract_e_lgssm_phase8_target_prefix_smoke.v1"
TELEMETRY_SCHEMA_VERSION = "contract_e_phase8_target_prefix_telemetry_v1"
DATASET_SEED = 81100
ESTIMATOR_SEED = 81120
EXPECTED_OBSERVATION_SHA256 = (
    "ded8c5326f970868dccebe2719af8302bbf9c2124bb5daf909c1956b24e6373f"
)
THETA = (0.72, 0.55, 0.35, 0.35, 0.45)
NUM_PARTICLES = 4
TIME_STEPS = 1
RIDGE = 4.0
EPSILON = 0.5
SCALING = 0.75
SINKHORN_STEPS = 2
CHUNKS = select_transport_chunks(NUM_PARTICLES)
HMC_COORDINATES = (
    "atanh(phi1)",
    "atanh(phi2)",
    "atanh(phi3)",
    "log(q_scale)",
    "log(r_scale)",
)

TELEMETRY_SHAPES = {
    "quotient_mass_history": (1, 1, 4),
    "quotient_row_residual_history": (1, 1),
    "target_mean_history": (1, 1, 3),
    "target_covariance_history": (1, 1, 3, 3),
    "output_mean_history": (1, 1, 3),
    "output_covariance_history": (1, 1, 3, 3),
    "injected_covariance_history": (1, 1, 3, 3),
    "reset_affine_history": (1, 1, 3, 3),
    "ridged_identity_residual_history": (1, 1, 3, 3),
    "ridged_identity_scale_history": (1, 1, 3, 3),
    "ridged_identity_residual_fro_history": (1, 1),
    "raw_covariance_residual_history": (1, 1, 3, 3),
    "predicted_raw_covariance_residual_history": (1, 1, 3, 3),
    "raw_covariance_prediction_error_history": (1, 1, 3, 3),
    "raw_covariance_residual_fro_history": (1, 1),
    "raw_covariance_prediction_error_fro_history": (1, 1),
    "mean_residual_history": (1, 1, 3),
    "mean_residual_infinity_history": (1, 1),
    "residual_design_sum_history": (1, 1, 3),
    "residual_design_absolute_scale_history": (1, 1, 3),
    "gap_chol_diagonal_history": (1, 1, 3),
    "target_chol_diagonal_history": (1, 1, 3),
    "injected_chol_diagonal_history": (1, 1, 3),
    "gap_condition_proxy_history": (1, 1),
    "target_condition_proxy_history": (1, 1),
    "injected_condition_proxy_history": (1, 1),
    "realized_ridge_history": (1, 1),
    "active_reset_history": (1, 1),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _serialized_tensor(value: Any) -> bytes:
    return tf.io.serialize_tensor(tf.convert_to_tensor(value)).numpy()


def _tensor_sha256(value: Any) -> str:
    return hashlib.sha256(_serialized_tensor(value)).hexdigest()


def _tensor_record(value: Any) -> dict[str, Any]:
    tensor = tf.convert_to_tensor(value)
    return {
        "dtype": tensor.dtype.name,
        "shape": tensor.shape.as_list(),
        "values": tensor.numpy().tolist(),
        "serialized_tensor_sha256": _tensor_sha256(tensor),
    }


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _target_observations() -> tf.Tensor:
    observations = tf.convert_to_tensor(
        _lgssm_dataset(DATASET_SEED)["observations"][:TIME_STEPS], tf.float64
    )
    if observations.shape != (TIME_STEPS, 3):
        raise ValueError(f"target observation shape drifted: {observations.shape}")
    realized_hash = _tensor_sha256(observations)
    if realized_hash != EXPECTED_OBSERVATION_SHA256:
        raise ValueError(
            "target observation hash drifted: "
            f"expected {EXPECTED_OBSERVATION_SHA256}, got {realized_hash}"
        )
    return observations


def _validate_environment() -> list[str]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("CUDA_VISIBLE_DEVICES=-1 is required before import")
    logical_devices = [device.name for device in tf.config.list_logical_devices()]
    if any("GPU" in device.upper() for device in logical_devices):
        raise RuntimeError(f"CPU-hidden run exposed a GPU: {logical_devices}")
    return logical_devices


def _kalman_value(theta: tf.Tensor, observations: tf.Tensor) -> tf.Tensor:
    phi = theta[:3]
    q_scale = theta[3]
    r_scale = theta[4]
    return tf_kalman_log_likelihood(
        observations=observations,
        transition_offset=tf.zeros([3], tf.float64),
        transition_matrix=tf.linalg.diag(phi),
        transition_covariance=tf.square(q_scale) * tf.eye(3, dtype=tf.float64),
        observation_offset=tf.zeros([3], tf.float64),
        observation_matrix=canonical._observation_matrix(tf.float64),
        observation_covariance=tf.square(r_scale) * tf.eye(3, dtype=tf.float64),
        initial_state_mean=tf.zeros([3], tf.float64),
        initial_state_covariance=tf.linalg.diag(
            tf.square(q_scale) / (1.0 - tf.square(phi))
        ),
    )


def _hmc_chain_factors(theta: tf.Tensor) -> tf.Tensor:
    return tf.concat([1.0 - tf.square(theta[:3]), theta[3:]], axis=0)


def _all_tensor_outputs_identical(
    first: Mapping[str, tf.Tensor], second: Mapping[str, tf.Tensor]
) -> bool:
    return set(first) == set(second) and all(
        _serialized_tensor(first[name]) == _serialized_tensor(second[name])
        for name in first
    )


def _all_finite(value: tf.Tensor) -> bool:
    tensor = tf.convert_to_tensor(value)
    if tensor.dtype == tf.bool:
        return True
    return bool(tf.reduce_all(tf.math.is_finite(tensor)).numpy())


def _validate_telemetry(
    result: Mapping[str, tf.Tensor],
) -> tuple[dict[str, dict[str, Any]], dict[str, bool]]:
    missing = sorted(set(TELEMETRY_SHAPES) - set(result))
    shape_valid = not missing and all(
        tuple(result[name].shape) == expected
        for name, expected in TELEMETRY_SHAPES.items()
    )
    finite = not missing and all(_all_finite(result[name]) for name in TELEMETRY_SHAPES)
    telemetry = {
        name: _tensor_record(result[name])
        for name in TELEMETRY_SHAPES
        if name in result
    }
    checks = {
        "required_field_set_complete": not missing,
        "all_static_shapes_match": shape_valid,
        "all_required_values_finite": finite,
    }
    return telemetry, checks


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")

    started = time.perf_counter()
    logical_devices = _validate_environment()
    observations = _target_observations()
    theta = tf.constant(THETA, tf.float64)
    if tuple(float(value) for value in theta.numpy()) != THETA:
        raise ValueError("physical parameter vector drifted before canonical execution")

    prepared_result = preparation.prepare_contract_e_lgssm_inputs(
        observations=observations,
        estimator_seeds=[ESTIMATOR_SEED],
        num_particles=NUM_PARTICLES,
        fixed_reset_mask=[[True]],
        prepared_ridge=[[RIDGE]],
        epsilon=EPSILON,
        scaling=SCALING,
        sinkhorn_steps=SINKHORN_STEPS,
        balance_steps=1,
        row_chunk_size=CHUNKS.row_chunk_size,
        col_chunk_size=CHUNKS.col_chunk_size,
        dtype=tf.float64,
    )
    callable_ = canonical.make_canonical_value_and_score_tf(
        preparation.prepared_values(prepared_result),
        steps=SINKHORN_STEPS,
        balance_steps=1,
        row_chunk_size=CHUNKS.row_chunk_size,
        col_chunk_size=CHUNKS.col_chunk_size,
        jit_compile=True,
        dtype=tf.float64,
    )
    first = callable_(theta)
    second = callable_(theta)

    with tf.GradientTape() as tape:
        tape.watch(theta)
        kalman_value = _kalman_value(theta, observations)
    kalman_score = tape.gradient(kalman_value, theta)
    chain_factors = _hmc_chain_factors(theta)
    canonical_score = first["score"]
    canonical_hmc_score = canonical_score * chain_factors
    kalman_hmc_score = kalman_score * chain_factors
    telemetry, telemetry_checks = _validate_telemetry(first)

    chart_checks = {
        "valid_chart": bool(tf.reduce_all(first["valid_chart"]).numpy()),
        "flow_valid_history": bool(
            tf.reduce_all(first["flow_valid_history"]).numpy()
        ),
        "geometry_valid_history": bool(
            tf.reduce_all(first["geometry_valid_history"]).numpy()
        ),
        "quotient_valid_history": bool(
            tf.reduce_all(first["quotient_valid_history"]).numpy()
        ),
        "reset_valid_history": bool(
            tf.reduce_all(first["reset_valid_history"]).numpy()
        ),
    }
    hard_checks = {
        "target_observation_hash_matches": (
            _tensor_sha256(observations) == EXPECTED_OBSERVATION_SHA256
        ),
        "target_observation_shape_matches": observations.shape == (1, 3),
        "physical_theta_matches": tuple(float(value) for value in theta.numpy())
        == THETA,
        "all_canonical_outputs_serialized_equal_across_two_calls": (
            _all_tensor_outputs_identical(first, second)
        ),
        "objective_serialized_equal_across_two_calls": (
            _serialized_tensor(first["objective"])
            == _serialized_tensor(second["objective"])
        ),
        "score_serialized_equal_across_two_calls": (
            _serialized_tensor(first["score"])
            == _serialized_tensor(second["score"])
        ),
        "all_chart_predicates_true": all(chart_checks.values()),
        "canonical_objective_finite": _all_finite(first["objective"]),
        "canonical_score_finite": _all_finite(canonical_score),
        "kalman_objective_finite": _all_finite(kalman_value),
        "kalman_score_finite": _all_finite(kalman_score),
        "one_concrete_canonical_callable": len(
            callable_._list_all_concrete_functions_for_serialization()
        )
        == 1,
        "cpu_hidden_no_logical_gpu": not any(
            "GPU" in device.upper() for device in logical_devices
        ),
        "jit_compile_true": True,
        **telemetry_checks,
    }
    invalid_chart = not all(chart_checks.values())
    all_hard_checks_pass = all(hard_checks.values())
    if all_hard_checks_pass:
        status = "TARGET_PREFIX_WIRING_SMOKE_PASSED_DESCRIPTIVE_ONLY"
    elif invalid_chart or not (
        hard_checks["canonical_objective_finite"]
        and hard_checks["canonical_score_finite"]
    ):
        status = "TRANSFERRED_FIXTURE_ARM_INVALID_FOR_TARGET_PREFIX"
    else:
        status = "TARGET_PREFIX_WIRING_SMOKE_HARD_VETO_FAILED"

    sources = [
        Path(__file__).resolve(),
        ROOT / "bayesfilter/highdim/ledh_contract_e_canonical_lgssm_tf.py",
        ROOT / "bayesfilter/highdim/ledh_contract_e_lgssm_preparation_tf.py",
        ROOT / "bayesfilter/highdim/ledh_contract_e_streaming_tf.py",
        ROOT / "bayesfilter/highdim/ledh_contract_e_reset_tf.py",
        ROOT / "bayesfilter/linear/kalman_tf.py",
        ROOT / "scripts/filtering_value_gradient_benchmark_generate_p8_datasets.py",
        ROOT
        / "experiments/dpf_implementation/tf_tfp/resampling/annealed_transport_tf.py",
    ]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "telemetry_schema_version": TELEMETRY_SCHEMA_VERSION,
        "program_id": "contract-e-canonical-gradient-migration-20260713",
        "phase": "8_rung1_target_prefix_smoke",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "configuration_classification": "fixture_transfer_harness_smoke_only",
        "source_sha256": {
            str(path.relative_to(ROOT)): _sha256(path) for path in sources
        },
        "environment": {
            "git_commit": _git_commit(),
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "tf_enable_onednn_opts": os.environ.get("TF_ENABLE_ONEDNN_OPTS"),
            "mplconfigdir": os.environ.get("MPLCONFIGDIR"),
            "logical_devices": logical_devices,
            "execution_role": "CPU_HIDDEN_REFERENCE_WIRING_SMOKE",
            "dtype": "float64",
            "jit_compile": True,
            "tf32_execution_enabled": (
                tf.config.experimental.tensor_float_32_execution_enabled()
            ),
            "command": [sys.executable, *sys.argv],
            "output": str(output.relative_to(ROOT)),
            "wall_time_seconds": time.perf_counter() - started,
        },
        "model_contract": {
            "dataset_seed": DATASET_SEED,
            "estimator_seed": ESTIMATOR_SEED,
            "time_steps": TIME_STEPS,
            "num_particles": NUM_PARTICLES,
            "timing": "transition_first_for_each_observation",
            "initial_law": "stationary_diagonal_q2_over_one_minus_phi2",
            "parameter_names": list(canonical.PARAMETER_NAMES),
            "physical_theta": list(THETA),
            "hmc_coordinates": list(HMC_COORDINATES),
            "observation": _tensor_record(observations),
            "expected_observation_serialized_tensor_sha256": (
                EXPECTED_OBSERVATION_SHA256
            ),
        },
        "transferred_configuration": {
            "ridge": RIDGE,
            "epsilon": EPSILON,
            "scaling": SCALING,
            "sinkhorn_steps": SINKHORN_STEPS,
            "row_chunk_size": CHUNKS.row_chunk_size,
            "col_chunk_size": CHUNKS.col_chunk_size,
            "fixed_reset_mask": [[True]],
            "classification": "NOT_A_TARGET_HYPOTHESIS_CANDIDATE_OR_DEFAULT",
        },
        "preparation_identity": prepared_result["identity"],
        "canonical": {
            "objective": _tensor_record(first["objective"]),
            "per_batch_log_likelihood": _tensor_record(
                first["per_batch_log_likelihood"]
            ),
            "physical_score": _tensor_record(canonical_score),
            "per_batch_physical_score": _tensor_record(first["per_batch_score"]),
            "hmc_score": _tensor_record(canonical_hmc_score),
            "valid_chart": _tensor_record(first["valid_chart"]),
            "minimum_mass": _tensor_record(first["minimum_mass"]),
        },
        "kalman_oracle": {
            "objective": _tensor_record(kalman_value),
            "physical_score": _tensor_record(kalman_score),
            "hmc_score": _tensor_record(kalman_hmc_score),
        },
        "paired_differences": {
            "value": _tensor_record(first["objective"] - kalman_value),
            "value_relative_to_abs_kalman": _tensor_record(
                (first["objective"] - kalman_value) / tf.abs(kalman_value)
            ),
            "physical_score": _tensor_record(canonical_score - kalman_score),
            "hmc_score": _tensor_record(canonical_hmc_score - kalman_hmc_score),
            "classification": "EXPLANATORY_ONLY_NO_EQUIVALENCE_MARGIN_APPLIED",
        },
        "telemetry": telemetry,
        "telemetry_expected_shapes": {
            name: list(shape) for name, shape in TELEMETRY_SHAPES.items()
        },
        "chart_checks": chart_checks,
        "hard_checks": hard_checks,
        "all_hard_checks_pass": all_hard_checks_pass,
        "gate_classification": {
            "wiring_smoke": status,
            "formal_phase1_fd": "INCONCLUSIVE_BLOCKED",
            "target_numerical_design": "NOT_EVALUATED",
            "kalman_equivalence": "NOT_EVALUATED_ALL_DIFFERENCES_EXPLANATORY",
            "primary_shape": "OWNER_STATISTICAL_AMENDMENT_BLOCKED",
        },
        "nonclaims": [
            "not a target ridge, reset, transport, residual, Sinkhorn, or chunk candidate",
            "not evidence for Kalman value or gradient equivalence",
            "not formal Phase 1 FD certification",
            "not T10 or T50_N10000 feasibility evidence",
            "not GPU, HMC, admission, leaderboard, default, release, or integrity readiness",
        ],
    }
    _write_json_exclusive(output, payload)
    print(json.dumps({"output": str(output), "status": status}, sort_keys=True))
    if not all_hard_checks_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
