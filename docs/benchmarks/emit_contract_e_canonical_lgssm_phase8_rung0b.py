#!/usr/bin/env python3
"""Emit the CPU-hidden Phase 8 Rung 0B tiny-fixture Kalman comparison."""

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
from fractions import Fraction
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

from bayesfilter.highdim import ledh_contract_e_canonical_lgssm_tf as canonical
from bayesfilter.linear.kalman_tf import tf_kalman_log_likelihood


DEFAULT_FIXTURE = ROOT / "docs/plans" / (
    "bayesfilter-contract-e-canonical-gradient-migration-"
    "phase5-tiny-fixture-freeze-v2-2026-07-14.json"
)
UPSTREAM_DTYPE_RESULT = ROOT / "docs/plans" / (
    "bayesfilter-contract-e-canonical-gradient-migration-"
    "phase8-rung0a-dtype-repair-result-2026-07-14.md"
)


def _convert(value: Any) -> Any:
    if isinstance(value, list):
        return [_convert(item) for item in value]
    if isinstance(value, str):
        return float(Fraction(value))
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _prepared(fixture: dict[str, Any]) -> dict[str, Any]:
    return {
        "observations": _convert(fixture["observations"]),
        "initial_noise": _convert(fixture["initial_noise"]),
        "transition_noise": _convert(fixture["transition_noise"]),
        "fixed_reset_mask": fixture["fixed_reset_mask"],
        "residual_design": _convert(fixture["residual_design"]),
        "prepared_ridge": _convert(fixture["prepared_ridge"]),
        "epsilon": _convert(fixture["transport"]["epsilon"]),
        "scaling": _convert(fixture["transport"]["scaling"]),
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


def _direct_joint_gaussian_value(
    theta: tf.Tensor, observations: tf.Tensor
) -> tf.Tensor:
    """Independent stationary joint-Gaussian likelihood for the tiny oracle check."""

    phi = theta[:3]
    q_scale = theta[3]
    r_scale = theta[4]
    observation_matrix = canonical._observation_matrix(tf.float64)
    stationary_covariance = tf.linalg.diag(
        tf.square(q_scale) / (1.0 - tf.square(phi))
    )
    observation_covariance = tf.square(r_scale) * tf.eye(3, dtype=tf.float64)
    time_steps = int(observations.shape[0])
    block_rows = []
    for row_index in range(time_steps):
        blocks = []
        for column_index in range(time_steps):
            lag = abs(row_index - column_index)
            cross_covariance = (
                tf.linalg.diag(phi**lag) @ stationary_covariance
            )
            if row_index < column_index:
                cross_covariance = tf.linalg.matrix_transpose(cross_covariance)
            block = (
                observation_matrix
                @ cross_covariance
                @ tf.linalg.matrix_transpose(observation_matrix)
            )
            if row_index == column_index:
                block += observation_covariance
            blocks.append(block)
        block_rows.append(tf.concat(blocks, axis=1))
    joint_covariance = tf.concat(block_rows, axis=0)
    flattened = tf.reshape(observations, [-1])
    chol = tf.linalg.cholesky(joint_covariance)
    solved = tf.linalg.triangular_solve(chol, flattened[:, None])
    dimension = tf.cast(tf.size(flattened), tf.float64)
    return -0.5 * (
        dimension * tf.math.log(tf.constant(2.0 * 3.141592653589793, tf.float64))
        + 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)))
        + tf.reduce_sum(tf.square(solved))
    )


def _hmc_chain_factors(theta: tf.Tensor) -> tf.Tensor:
    return tf.concat([1.0 - tf.square(theta[:3]), theta[3:]], axis=0)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    raise RuntimeError("ARCHIVAL_WRONG_TRANSPORT_CHUNK_POLICY: this route is preserved only as provenance and cannot emit new evidence")
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-jit-compile", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    fixture_path = args.fixture.resolve()
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    prepared = _prepared(fixture)
    theta = tf.constant(_convert(fixture["center_theta"]), tf.float64)
    observations = tf.constant(prepared["observations"], tf.float64)
    kwargs = {
        "steps": int(fixture["transport"]["finite_sinkhorn_steps"]),
        "balance_steps": 1,
        "row_chunk_size": int(fixture["transport"]["row_chunk_size"]),
        "col_chunk_size": int(fixture["transport"]["col_chunk_size"]),
    }
    callable_ = canonical.make_canonical_value_and_score_tf(
        prepared,
        dtype=tf.float64,
        jit_compile=not args.no_jit_compile,
        **kwargs,
    )
    first = callable_(theta)
    second = callable_(theta)
    with tf.GradientTape() as tape:
        tape.watch(theta)
        kalman_value = _kalman_value(theta, observations)
    kalman_score = tape.gradient(kalman_value, theta)
    with tf.GradientTape() as tape:
        tape.watch(theta)
        direct_joint_value = _direct_joint_gaussian_value(theta, observations)
    direct_joint_score = tape.gradient(direct_joint_value, theta)
    chain_factors = _hmc_chain_factors(theta)
    canonical_score = first["score"]
    canonical_hmc_score = canonical_score * chain_factors
    kalman_hmc_score = kalman_score * chain_factors
    value_difference = first["objective"] - kalman_value
    score_difference = canonical_score - kalman_score
    hmc_score_difference = canonical_hmc_score - kalman_hmc_score
    center_repeat_equal = all(
        bool(tf.reduce_all(first[name] == second[name]).numpy())
        for name in (
            "objective",
            "per_batch_log_likelihood",
            "score",
            "per_batch_score",
            "valid_chart",
            "minimum_mass",
            "flow_valid_history",
            "geometry_valid_history",
            "quotient_valid_history",
            "reset_valid_history",
            "sinkhorn_running_branch",
        )
    )
    sources = [
        ROOT / "bayesfilter/highdim/ledh_contract_e_canonical_lgssm_tf.py",
        ROOT / "bayesfilter/highdim/ledh_contract_e_streaming_tf.py",
        ROOT / "bayesfilter/highdim/ledh_contract_e_reset_tf.py",
        ROOT / "bayesfilter/linear/kalman_tf.py",
        ROOT
        / "experiments/dpf_implementation/tf_tfp/resampling/annealed_transport_tf.py",
    ]
    payload = {
        "schema_version": "bayesfilter.contract_e_canonical_lgssm_phase8_rung0b.v1",
        "program_id": "contract-e-canonical-gradient-migration-20260713",
        "phase": "8_rung0b",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "TINY_FIXTURE_ORACLE_HARNESS_EXECUTED_DESCRIPTIVE_ONLY",
        "fixture": str(fixture_path.relative_to(ROOT)),
        "fixture_sha256": _sha256(fixture_path),
        "upstream_dtype_result": str(UPSTREAM_DTYPE_RESULT.relative_to(ROOT)),
        "source_sha256": {
            str(path.relative_to(ROOT)): _sha256(path) for path in sources
        },
        "environment": {
            "git_commit": _git_commit(),
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "logical_devices": [
                device.name for device in tf.config.list_logical_devices()
            ],
            "dtype": "float64",
            "jit_compile": not args.no_jit_compile,
            "wall_time_seconds": time.perf_counter() - started,
        },
        "model_contract": {
            "parameter_names": list(canonical.PARAMETER_NAMES),
            "physical_coordinates": list(canonical.PARAMETER_NAMES),
            "hmc_coordinates": [
                "atanh(phi1)",
                "atanh(phi2)",
                "atanh(phi3)",
                "log(q_scale)",
                "log(r_scale)",
            ],
            "timing": "transition_first_for_each_observation",
            "initial_law": "stationary_diagonal_q2_over_one_minus_phi2",
            "observations_are_fixed": True,
        },
        "theta": theta.numpy().tolist(),
        "hmc_chain_factors": chain_factors.numpy().tolist(),
        "canonical": {
            "objective": float(first["objective"]),
            "per_batch_log_likelihood": first[
                "per_batch_log_likelihood"
            ].numpy().tolist(),
            "physical_score": canonical_score.numpy().tolist(),
            "per_batch_physical_score": first["per_batch_score"].numpy().tolist(),
            "hmc_score": canonical_hmc_score.numpy().tolist(),
            "valid_chart": first["valid_chart"].numpy().tolist(),
            "minimum_mass": first["minimum_mass"].numpy().tolist(),
        },
        "kalman_oracle": {
            "objective": float(kalman_value),
            "physical_score": kalman_score.numpy().tolist(),
            "hmc_score": kalman_hmc_score.numpy().tolist(),
        },
        "independent_joint_gaussian_oracle": {
            "objective": float(direct_joint_value),
            "physical_score": direct_joint_score.numpy().tolist(),
            "difference_from_kalman": {
                "objective": float(direct_joint_value - kalman_value),
                "physical_score": (
                    direct_joint_score - kalman_score
                ).numpy().tolist(),
                "classification": "EXPLANATORY_CROSS_CHECK_NO_POST_RESULT_TOLERANCE",
            },
        },
        "paired_differences": {
            "value": float(value_difference),
            "value_relative_to_abs_kalman": float(
                value_difference / tf.abs(kalman_value)
            ),
            "physical_score": score_difference.numpy().tolist(),
            "hmc_score": hmc_score_difference.numpy().tolist(),
            "classification": "EXPLANATORY_ONLY_NO_EQUIVALENCE_MARGIN_APPLIED",
        },
        "hard_checks": {
            "center_repeat_bitwise_equal": center_repeat_equal,
            "canonical_chart_valid": bool(tf.reduce_all(first["valid_chart"]).numpy()),
            "canonical_value_finite": bool(tf.math.is_finite(first["objective"]).numpy()),
            "canonical_score_finite": bool(
                tf.reduce_all(tf.math.is_finite(canonical_score)).numpy()
            ),
            "kalman_value_finite": bool(tf.math.is_finite(kalman_value).numpy()),
            "kalman_score_finite": bool(
                tf.reduce_all(tf.math.is_finite(kalman_score)).numpy()
            ),
            "direct_joint_value_finite": bool(
                tf.math.is_finite(direct_joint_value).numpy()
            ),
            "direct_joint_score_finite": bool(
                tf.reduce_all(tf.math.is_finite(direct_joint_score)).numpy()
            ),
            "one_concrete_canonical_callable": len(
                callable_._list_all_concrete_functions_for_serialization()
            )
            == 1,
        },
        "gate_classification": {
            "same_program_derivative": "INHERITED_FROM_SOURCE_BOUND_RUNG0A_ZERO_ULP_CERTIFICATE",
            "formal_phase1_fd": "INCONCLUSIVE_BLOCKED",
            "kalman_equivalence": "NOT_EVALUATED_NO_MARGIN_FOR_TINY_FIXTURE",
            "target_scale_preparation": "NOT_EVALUATED_FROZEN_TINY_LITERALS_ONLY",
        },
        "nonclaims": [
            "not a target d3_T50_N10000 result",
            "not evidence for a Kalman-gradient equivalence margin",
            "not a residual-design or ridge default",
            "not formal Phase 1 FD certification",
            "not GPU, HMC, admission, leaderboard, or release readiness",
        ],
    }
    _write_json_atomic(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": payload["status"],
                "hard_checks": payload["hard_checks"],
                "paired_differences": payload["paired_differences"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
