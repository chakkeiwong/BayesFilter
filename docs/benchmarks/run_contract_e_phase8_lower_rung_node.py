#!/usr/bin/env python3
"""Run one isolated Contract E Phase 8 lower-rung node.

The node is deliberately diagnostic-only while the repository production
Contract E factory remains empty.  It emits a complete finite-program record
without changing the canonical implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

from bayesfilter.highdim import ledh_contract_e_canonical_lgssm_tf as canonical
from bayesfilter.highdim import ledh_contract_e_identity as identity
from bayesfilter.highdim import ledh_contract_e_lgssm_preparation_tf as preparation
from bayesfilter.highdim.transport_chunk_policy import select_transport_chunks
from bayesfilter.linear.kalman_tf import tf_kalman_log_likelihood
from scripts.filtering_value_gradient_benchmark_generate_p8_datasets import (
    _lgssm_dataset,
)


DATASET_SEED = 81100
ESTIMATOR_SEED = 80920
THETA = (0.72, 0.55, 0.35, 0.35, 0.45)
PARAMETER_NAMES = tuple(canonical.PARAMETER_NAMES)
TIME_STEPS = 2
EXPECTED_OBSERVATION_SHA256 = (
    "8b37260849e3b8e95e244dfe17109f2b22cde4abcf4779cd452a3438e341d635"
)
FD_THRESHOLD = 0.05 * math.sqrt(5.0)
FD_MULTIPLIERS = (8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.125)
FD_BASE = 2.0**-17
FLOAT64_EPSILON = 2.0**-52
EXPECTED_SOURCE_SHA256 = {
    "bayesfilter/highdim/ledh_contract_e_canonical_lgssm_tf.py": "6201d85642474a9819a1c8972e94bd49cd317cba9a5862145f90252ddcdd0d24",
    "bayesfilter/highdim/ledh_contract_e_streaming_tf.py": "b2208a9e9f65bceaa6a629e69fb8c0edcdeab39d79ba6f0e5c04c45a427ef34a",
    "bayesfilter/highdim/ledh_contract_e_reset_tf.py": "5a226b53f4a881a1b66cee00902dcd007c82de3c01e3440101c111c5095ee023",
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _tensor_bytes(value: Any) -> bytes:
    return tf.io.serialize_tensor(tf.convert_to_tensor(value)).numpy()


def _tensor_hash(value: Any) -> str:
    return _sha256_bytes(_tensor_bytes(value))


def _tensor_record(value: Any) -> dict[str, Any]:
    tensor = tf.convert_to_tensor(value)
    return {
        "dtype": tensor.dtype.name,
        "shape": tensor.shape.as_list(),
        "values": tensor.numpy().tolist(),
        "serialized_tensor_sha256": _tensor_hash(tensor),
    }


def _finite(value: Any) -> bool:
    tensor = tf.convert_to_tensor(value)
    if tensor.dtype == tf.bool:
        return True
    return bool(tf.reduce_all(tf.math.is_finite(tensor)).numpy())


def _scalar(value: Any) -> float:
    return float(tf.convert_to_tensor(value).numpy())


def _parse_float(value: str) -> float:
    return float.fromhex(value) if value.lower().startswith(("0x", "+0x", "-0x")) else float(value)


def _branch_hash(result: dict[str, tf.Tensor]) -> str:
    names = (
        "valid_chart",
        "flow_valid_history",
        "geometry_valid_history",
        "quotient_valid_history",
        "reset_valid_history",
        "epsilon0_floor_inactive",
        "sinkhorn_running_branch",
        "diameter_max_mask",
        "geometry_max_mask",
        "geometry_min_mask",
        "active_reset_history",
    )
    payload = b"".join(name.encode() + b"\0" + _tensor_bytes(result[name]) for name in names)
    return _sha256_bytes(payload)


def _record(result: dict[str, tf.Tensor]) -> dict[str, Any]:
    return {
        "objective": _scalar(result["objective"]),
        "objective_hex": _scalar(result["objective"]).hex(),
        "per_batch_log_likelihood": result["per_batch_log_likelihood"].numpy().tolist(),
        "physical_score": result["score"].numpy().tolist(),
        "hmc_score": (
            result["score"]
            * tf.concat([1.0 - tf.square(tf.constant(THETA[:3], tf.float64)), tf.constant(THETA[3:], tf.float64)], axis=0)
        ).numpy().tolist(),
        "valid_chart": result["valid_chart"].numpy().tolist(),
        "minimum_mass": result["minimum_mass"].numpy().tolist(),
        "branch_hash": _branch_hash(result),
        "output_hashes": {
            name: _tensor_hash(value)
            for name, value in sorted(result.items())
            if isinstance(value, tf.Tensor)
        },
    }


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


def _source_audit() -> dict[str, Any]:
    streaming_source = inspect.getsource(
        __import__(
            "bayesfilter.highdim.ledh_contract_e_streaming_tf",
            fromlist=["_streaming_column_mass_from_potentials_core"],
        )._streaming_column_mass_from_potentials_core
    )
    forbidden = (
        "num_particles, num_particles",
        "particle_count, particle_count",
        "_filterflow_exact_cost",
        "_filterflow_exact_transport_from_potentials",
    )
    return {
        "streaming_column_mass_forbidden_tokens_absent": not any(
            token in streaming_source for token in forbidden
        ),
        "canonical_factory_registry_empty": not bool(
            identity._PRODUCTION_FACTORY._route_specifications
        ),
        "route_status": "diagnostic_only_factory_empty",
    }


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _fd_summary(callable_: Any, theta: tf.Tensor, center: dict[str, Any]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for index, parameter in enumerate(PARAMETER_NAMES):
        direction = tf.one_hot(index, len(PARAMETER_NAMES), dtype=tf.float64)
        for multiplier in FD_MULTIPLIERS:
            nominal = multiplier * FD_BASE
            plus_theta = theta + tf.constant(nominal, tf.float64) * direction
            minus_theta = theta - tf.constant(nominal, tf.float64) * direction
            actual_plus = _scalar(plus_theta[index] - theta[index])
            actual_minus = _scalar(theta[index] - minus_theta[index])
            plus = _record(callable_(plus_theta))
            minus = _record(callable_(minus_theta))
            representable = (
                actual_plus.hex() == nominal.hex()
                and actual_minus.hex() == nominal.hex()
                and actual_plus.hex() == actual_minus.hex()
                and actual_plus > 0.0
            )
            branch_match = (
                plus["branch_hash"] == center["branch_hash"]
                and minus["branch_hash"] == center["branch_hash"]
            )
            endpoint_finite = all(
                math.isfinite(value)
                for value in (
                    plus["objective"],
                    minus["objective"],
                    *plus["physical_score"],
                    *minus["physical_score"],
                )
            )
            charts_valid = all(plus["valid_chart"]) and all(minus["valid_chart"])
            endpoint_valid = representable and branch_match and endpoint_finite and charts_valid
            estimate = (
                (plus["objective"] - minus["objective"]) / (2.0 * actual_plus)
                if endpoint_valid
                else None
            )
            cancellation_floor = (
                FLOAT64_EPSILON
                * (abs(plus["objective"]) + abs(minus["objective"]))
                / (2.0 * actual_plus)
                if endpoint_valid
                else None
            )
            eligible = (
                estimate is not None
                and cancellation_floor is not None
                and abs(estimate) > cancellation_floor
            )
            score = center["physical_score"][index]
            relative_error = abs(score - estimate) / abs(estimate) if eligible and estimate != 0.0 else None
            records.append(
                {
                    "parameter": parameter,
                    "multiplier": multiplier,
                    "nominal_step": nominal,
                    "actual_plus_step": actual_plus,
                    "actual_minus_step": actual_minus,
                    "nominal_step_hex": nominal.hex(),
                    "actual_plus_step_hex": actual_plus.hex(),
                    "actual_minus_step_hex": actual_minus.hex(),
                    "symmetric_representable": representable,
                    "branch_matches_center": branch_match,
                    "finite": endpoint_finite,
                    "charts_valid": charts_valid,
                    "endpoint_valid": endpoint_valid,
                    "fd_estimate": estimate,
                    "manual_score": score,
                    "cancellation_floor_diagnostic": cancellation_floor,
                    "relative_denominator_eligible": eligible,
                    "relative_error": relative_error,
                    "threshold": FD_THRESHOLD,
                    "pass": relative_error is not None and relative_error <= FD_THRESHOLD,
                }
            )
    return {
        "records": records,
        "all_endpoint_checks_valid": all(
            item["endpoint_valid"] and item["relative_denominator_eligible"]
            for item in records
        ),
        "all_relative_errors_pass": all(item["pass"] for item in records),
        "max_relative_error": max(
            (item["relative_error"] for item in records if item["relative_error"] is not None),
            default=None,
        ),
        "threshold": FD_THRESHOLD,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ridge", type=_parse_float, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--num-particles", type=int, default=32)
    parser.add_argument(
        "--reset-policy",
        choices=("all_active_contract_e", "no_reset_weighted"),
        default="all_active_contract_e",
    )
    parser.add_argument("--run-fd", action="store_true")
    args = parser.parse_args()
    chunks = select_transport_chunks(args.num_particles)
    args.row_chunk_size = chunks.row_chunk_size
    args.col_chunk_size = chunks.col_chunk_size
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("CUDA_VISIBLE_DEVICES=-1 is required before import")
    started = time.perf_counter()
    observations = tf.convert_to_tensor(
        _lgssm_dataset(DATASET_SEED)["observations"][:TIME_STEPS], tf.float64
    )
    if _tensor_hash(observations) != EXPECTED_OBSERVATION_SHA256:
        raise RuntimeError("observation identity mismatch")
    realized_source_hashes = {
        name: _sha256_path(ROOT / name) for name in EXPECTED_SOURCE_SHA256
    }
    if realized_source_hashes != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("frozen canonical source closure drifted")
    theta = tf.constant(THETA, tf.float64)
    reset_active = args.reset_policy == "all_active_contract_e"
    prepared_result = preparation.prepare_contract_e_lgssm_inputs(
        observations=observations,
        estimator_seeds=[ESTIMATOR_SEED],
        num_particles=args.num_particles,
        fixed_reset_mask=[[reset_active, reset_active]],
        prepared_ridge=[[args.ridge, args.ridge]],
        epsilon=0.5,
        scaling=0.9,
        sinkhorn_steps=args.steps,
        balance_steps=1,
        row_chunk_size=args.row_chunk_size,
        col_chunk_size=args.col_chunk_size,
        dtype=tf.float64,
    )
    callable_ = canonical.make_canonical_value_and_score_tf(
        preparation.prepared_values(prepared_result),
        steps=args.steps,
        balance_steps=1,
        row_chunk_size=args.row_chunk_size,
        col_chunk_size=args.col_chunk_size,
        jit_compile=True,
        dtype=tf.float64,
    )
    first_raw = callable_(theta)
    second_raw = callable_(theta)
    first = _record(first_raw)
    second = _record(second_raw)
    repeated = first == second
    kalman_value = _kalman_value(theta, observations)
    with tf.GradientTape() as tape:
        tape.watch(theta)
        kalman_taped = _kalman_value(theta, observations)
    kalman_score = tape.gradient(kalman_taped, theta)
    kalman_hmc = kalman_score * tf.concat(
        [1.0 - tf.square(theta[:3]), theta[3:]], axis=0
    )
    inactive_minimum_mass_sentinel = (
        args.reset_policy == "no_reset_weighted"
        and not _finite(first_raw["minimum_mass"])
    )
    finite_outputs = all(
        _finite(value)
        for name, value in first_raw.items()
        if name not in {"minimum_mass", "minimum_mass_history"}
    )
    telemetry_names = (
        "quotient_row_residual_history",
        "quotient_column_residual_history",
        "raw_covariance_residual_fro_history",
        "mean_residual_infinity_history",
        "gap_chol_diagonal_history",
        "target_chol_diagonal_history",
        "injected_chol_diagonal_history",
        "gap_condition_proxy_history",
        "target_condition_proxy_history",
        "injected_condition_proxy_history",
    )
    telemetry = {
        name: {
            "max_abs": float(tf.reduce_max(tf.abs(first_raw[name])).numpy()),
            "record": _tensor_record(first_raw[name]),
        }
        for name in telemetry_names
    }
    source_audit = _source_audit()
    hard_checks = {
        "observation_identity": True,
        "center_repeated_bitwise": repeated,
        "valid_chart": all(first["valid_chart"]),
        "all_executed_outputs_finite": finite_outputs,
        "inactive_minimum_mass_sentinel_allowed": (
            inactive_minimum_mass_sentinel or _finite(first_raw["minimum_mass"])
        ),
        "kalman_finite": _finite(kalman_value) and _finite(kalman_score),
        "all_cholesky_diagonals_positive": all(
            bool(tf.reduce_all(first_raw[name] > 0.0).numpy())
            for name in (
                "gap_chol_diagonal_history",
                "target_chol_diagonal_history",
                "injected_chol_diagonal_history",
            )
        ),
        "one_concrete_callable": len(callable_._list_all_concrete_functions_for_serialization()) == 1,
        "frozen_source_hashes_match": realized_source_hashes == EXPECTED_SOURCE_SHA256,
        "streaming_allocation_source_audit": source_audit[
            "streaming_column_mass_forbidden_tokens_absent"
        ],
        "factory_empty_as_expected": source_audit[
            "canonical_factory_registry_empty"
        ],
    }
    fd = _fd_summary(callable_, theta, first) if args.run_fd else None
    sources = [
        Path(__file__).resolve(),
        ROOT / "bayesfilter/highdim/ledh_contract_e_canonical_lgssm_tf.py",
        ROOT / "bayesfilter/highdim/ledh_contract_e_streaming_tf.py",
        ROOT / "bayesfilter/highdim/ledh_contract_e_reset_tf.py",
        ROOT / "bayesfilter/highdim/ledh_contract_e_lgssm_preparation_tf.py",
        ROOT / "experiments/dpf_implementation/tf_tfp/resampling/annealed_transport_tf.py",
    ]
    payload = {
        "schema_version": "bayesfilter.contract_e_phase8.lower_rung_node.v1",
        "program_id": "contract-e-canonical-gradient-migration-20260713",
        "phase": "8_lower_rung_continuation",
        "status": "DIAGNOSTIC_ONLY_FACTORY_EMPTY",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "dataset_seed": DATASET_SEED,
            "estimator_seed": ESTIMATOR_SEED,
            "time_steps": TIME_STEPS,
            "num_particles": args.num_particles,
            "reset_policy": args.reset_policy,
            "theta": list(THETA),
            "ridge": args.ridge,
            "epsilon": 0.5,
            "scaling": 0.9,
            "steps": args.steps,
            "row_chunk_size": args.row_chunk_size,
            "col_chunk_size": args.col_chunk_size,
            "delta_grad": 0.05,
            "fd_threshold": FD_THRESHOLD,
        },
        "environment": {
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip(),
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "dtype": "float64",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "logical_devices": [device.name for device in tf.config.list_logical_devices()],
            "jit_compile": True,
            "tf32_execution_enabled": tf.config.experimental.tensor_float_32_execution_enabled(),
            "wall_time_seconds": time.perf_counter() - started,
        },
        "source_sha256": {str(path.relative_to(ROOT)): _sha256_path(path) for path in sources},
        "preparation_identity": prepared_result["identity"],
        "route_identity": {
            "canonical_factory_registry_empty": identity._PRODUCTION_FACTORY._route_specifications == {},
            "route_status": "diagnostic_only_factory_empty",
            "reset_contract_id": identity.CONTRACT_E_RESET_CONTRACT_ID,
            "derivative_composition_id": identity.CONTRACT_E_DERIVATIVE_COMPOSITION_ID,
        },
        "hard_checks": hard_checks,
        "center": first,
        "center_repeat": second,
        "kalman": {
            "objective": _scalar(kalman_value),
            "physical_score": kalman_score.numpy().tolist(),
            "hmc_score": kalman_hmc.numpy().tolist(),
        },
        "paired_differences": {
            "value": first["objective"] - _scalar(kalman_value),
            "value_relative": (first["objective"] - _scalar(kalman_value)) / abs(_scalar(kalman_value)),
            "physical_score": [a - b for a, b in zip(first["physical_score"], kalman_score.numpy().tolist(), strict=True)],
            "hmc_score": [a - b for a, b in zip(first["hmc_score"], kalman_hmc.numpy().tolist(), strict=True)],
            "classification": "EXPLANATORY_UNTIL_EDGE_ORACLE_GATE",
        },
        "telemetry": telemetry,
        "causal_identification": {
            "first_reset_time_index": 0,
            "subsequent_likelihood_increment_time_index": 1,
            "first_reset_is_upstream_of_measured_later_increment": True,
            "final_reset_has_no_later_likelihood_increment": True,
            "first_reset_executed": reset_active,
            "no_reset_branch_carries_flow_particles_and_normalized_weights": (
                not reset_active
            ),
        },
        "source_audit": source_audit,
        "fd": fd,
        "nonclaims": [
            "factory remains empty; this is not canonical admission",
            "not full-box HMC readiness, primary-shape evidence, leaderboard release, or HMC execution",
            "audit count 16 has no role in this single-seed lower rung and gives no power claim",
        ],
    }
    if not all(hard_checks.values()):
        payload["status"] = "HARD_VETO"
    _write_exclusive(output, payload)
    print(json.dumps({"output": str(output), "status": payload["status"], "hard_checks": hard_checks}, sort_keys=True))
    if not all(hard_checks.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
