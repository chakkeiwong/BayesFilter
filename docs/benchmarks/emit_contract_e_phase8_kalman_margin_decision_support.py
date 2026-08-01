#!/usr/bin/env python3
"""Emit Kalman-only decision support for the Phase 8 gradient margin."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
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

from bayesfilter.linear.kalman_tf import tf_kalman_log_likelihood
from scripts.filtering_value_gradient_benchmark_generate_p8_datasets import (
    _lgssm_dataset,
)


SCHEMA_VERSION = "bayesfilter.contract_e_phase8.kalman_margin_decision_support.v1"
DATASET_SEED = 81100
TIME_STEPS = 50
THETA = (0.72, 0.55, 0.35, 0.35, 0.45)
PHYSICAL_LOWER = (-0.95, -0.95, -0.95, 0.05, 0.05)
PHYSICAL_UPPER = (0.95, 0.95, 0.95, 2.0, 2.0)
PARAMETER_NAMES = ("phi1", "phi2", "phi3", "q_scale", "r_scale")
EXPECTED_OBSERVATION_SHA256 = (
    "8aa2e8102ef25d6accf5d30b9c341621af26fce151ac85133c5a0a6a44671e17"
)
FORBIDDEN_REPOSITORY_MODULES = (
    "bayesfilter.highdim.ledh_contract_e_canonical_lgssm_tf",
    "bayesfilter.highdim.ledh_contract_e_streaming_tf",
    "bayesfilter.highdim.ledh_contract_e_reset_tf",
    "bayesfilter.highdim.ledh_contract_e_lgssm_preparation_tf",
    "bayesfilter.highdim.ledh_contract_e_identity",
    "experiments.dpf_implementation.tf_tfp.resampling.annealed_transport_tf",
)
PLAN_PATH = (
    "docs/plans/bayesfilter-contract-e-canonical-gradient-migration-"
    "phase8-kalman-decision-support-subplan-2026-07-14.md"
)
RESULT_PATH = (
    "docs/plans/logs/contract-e-canonical-gradient-migration-2026-07-13/"
    "phase8/kalman-decision-support-attempt1/result.json"
)
EXACT_COMMAND = (
    "timeout 120s env CUDA_VISIBLE_DEVICES=-1 TF_ENABLE_ONEDNN_OPTS=0 "
    "MPLCONFIGDIR=/tmp python "
    "docs/benchmarks/emit_contract_e_phase8_kalman_margin_decision_support.py "
    f"--output {RESULT_PATH}"
)
CAMPAIGN_DEADLINE = "2026-07-14T09:32:19+08:00"
FLOAT64_UNIT_ROUNDOFF = 1.1102230246251565e-16
CHAIN_TOLERANCE_MULTIPLIER = 256.0


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


def _physical_to_hmc(theta: tf.Tensor) -> tf.Tensor:
    theta = tf.convert_to_tensor(theta, tf.float64)
    return tf.concat([tf.math.atanh(theta[:3]), tf.math.log(theta[3:])], axis=0)


def _hmc_to_physical(u: tf.Tensor) -> tf.Tensor:
    u = tf.convert_to_tensor(u, tf.float64)
    return tf.concat([tf.math.tanh(u[:3]), tf.math.exp(u[3:])], axis=0)


def _chain_factors(theta: tf.Tensor) -> tf.Tensor:
    theta = tf.convert_to_tensor(theta, tf.float64)
    return tf.concat([1.0 - tf.square(theta[:3]), theta[3:]], axis=0)


def _kalman_value(theta: tf.Tensor, observations: tf.Tensor) -> tf.Tensor:
    theta = tf.convert_to_tensor(theta, tf.float64)
    phi = theta[:3]
    q_scale = theta[3]
    r_scale = theta[4]
    observation_matrix = tf.constant(
        [
            [1.0, 0.25, -0.15],
            [0.2, 1.1, 0.3],
            [-0.1, 0.35, 0.9],
        ],
        tf.float64,
    )
    return tf_kalman_log_likelihood(
        observations=observations,
        transition_offset=tf.zeros([3], tf.float64),
        transition_matrix=tf.linalg.diag(phi),
        transition_covariance=tf.square(q_scale) * tf.eye(3, dtype=tf.float64),
        observation_offset=tf.zeros([3], tf.float64),
        observation_matrix=observation_matrix,
        observation_covariance=tf.square(r_scale) * tf.eye(3, dtype=tf.float64),
        initial_state_mean=tf.zeros([3], tf.float64),
        initial_state_covariance=tf.linalg.diag(
            tf.square(q_scale) / (1.0 - tf.square(phi))
        ),
    )


def _proposal_radii(theta: tf.Tensor) -> tf.Tensor:
    center = _physical_to_hmc(theta)
    lower = _physical_to_hmc(tf.constant(PHYSICAL_LOWER, tf.float64))
    upper = _physical_to_hmc(tf.constant(PHYSICAL_UPPER, tf.float64))
    return tf.maximum(center - lower, upper - center)


def _all_finite(*values: tf.Tensor) -> bool:
    return all(
        bool(tf.reduce_all(tf.math.is_finite(tf.convert_to_tensor(value))).numpy())
        for value in values
    )


def _chain_tolerance(direct: tf.Tensor, chained: tf.Tensor) -> tf.Tensor:
    direct = tf.convert_to_tensor(direct, tf.float64)
    chained = tf.convert_to_tensor(chained, tf.float64)
    return (
        tf.constant(CHAIN_TOLERANCE_MULTIPLIER * FLOAT64_UNIT_ROUNDOFF, tf.float64)
        * tf.maximum(
            tf.ones_like(direct),
            tf.maximum(tf.abs(direct), tf.abs(chained)),
        )
    )


def _forbidden_loaded_modules() -> list[str]:
    return sorted(
        name
        for name in sys.modules
        if any(
            name == forbidden or name.startswith(forbidden + ".")
            for forbidden in FORBIDDEN_REPOSITORY_MODULES
        )
    )


def _forbidden_source_imports() -> list[str]:
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    return sorted(
        name
        for name in imported
        if any(
            name == forbidden or name.startswith(forbidden + ".")
            for forbidden in FORBIDDEN_REPOSITORY_MODULES
        )
    )


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
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
        raise FileExistsError(f"refusing to overwrite {output}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("CUDA_VISIBLE_DEVICES=-1 is required before import")
    forbidden_source_imports = _forbidden_source_imports()
    forbidden_loaded_modules = _forbidden_loaded_modules()
    if forbidden_source_imports or forbidden_loaded_modules:
        raise RuntimeError(
            "forbidden canonical dependency detected: "
            f"source={forbidden_source_imports}, loaded={forbidden_loaded_modules}"
        )

    started = time.perf_counter()
    observations = tf.convert_to_tensor(
        _lgssm_dataset(DATASET_SEED)["observations"][:TIME_STEPS], tf.float64
    )
    observation_sha256 = _tensor_sha256(observations)
    if observation_sha256 != EXPECTED_OBSERVATION_SHA256:
        raise RuntimeError(
            f"observation identity mismatch: {observation_sha256}"
        )
    theta = tf.constant(THETA, tf.float64)
    with tf.GradientTape() as tape:
        tape.watch(theta)
        value_physical = _kalman_value(theta, observations)
    physical_score = tape.gradient(value_physical, theta)

    u = _physical_to_hmc(theta)
    with tf.GradientTape() as tape:
        tape.watch(u)
        value_hmc = _kalman_value(_hmc_to_physical(u), observations)
    direct_hmc_score = tape.gradient(value_hmc, u)
    chain_hmc_score = physical_score * _chain_factors(theta)
    chain_error = direct_hmc_score - chain_hmc_score
    radii = _proposal_radii(theta)
    weighted_oracle_contributions = radii * tf.abs(direct_hmc_score)
    s_oracle = tf.reduce_mean(weighted_oracle_contributions)
    absolute_budget_per_delta = s_oracle / radii
    contribution_share = weighted_oracle_contributions / tf.reduce_sum(
        weighted_oracle_contributions
    )

    chain_tolerance = _chain_tolerance(direct_hmc_score, chain_hmc_score)
    chain_pass = bool(
        tf.reduce_all(tf.abs(chain_error) <= chain_tolerance).numpy()
    )
    checks = {
        "forbidden_source_imports_absent": not forbidden_source_imports,
        "forbidden_canonical_modules_not_loaded": not forbidden_loaded_modules,
        "theta_roundtrip_within_float64_roundoff": bool(
            tf.reduce_all(
                tf.abs(_hmc_to_physical(u) - theta)
                <= tf.constant(8.0 * 2.220446049250313e-16, tf.float64)
                * tf.maximum(tf.ones_like(theta), tf.abs(theta))
            ).numpy()
        ),
        "value_parameterization_within_float64_roundoff": bool(
            (
                tf.abs(value_physical - value_hmc)
                <= tf.constant(64.0 * 2.220446049250313e-16, tf.float64)
                * tf.maximum(
                    tf.constant(1.0, tf.float64), tf.abs(value_physical)
                )
            ).numpy()
        ),
        "all_finite": _all_finite(
            value_physical,
            physical_score,
            direct_hmc_score,
            chain_hmc_score,
            radii,
            s_oracle,
            absolute_budget_per_delta,
        ),
        "s_oracle_positive": bool((s_oracle > 0.0).numpy()),
        "chain_rule_pass": chain_pass,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Kalman decision-support checks failed: {checks}")

    sources = [
        ROOT / "bayesfilter/linear/kalman_tf.py",
        ROOT / "scripts/filtering_value_gradient_benchmark_generate_p8_datasets.py",
        Path(__file__).resolve(),
    ]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "program_id": "contract-e-canonical-gradient-migration-20260713",
        "phase": "8_kalman_margin_decision_support",
        "status": "KALMAN_ONLY_DECISION_SUPPORT_PASSED",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "identity": {
            "dataset_seed": DATASET_SEED,
            "time_steps": TIME_STEPS,
            "physical_theta": list(THETA),
            "physical_lower": list(PHYSICAL_LOWER),
            "physical_upper": list(PHYSICAL_UPPER),
            "parameter_names": list(PARAMETER_NAMES),
            "observation_sha256": observation_sha256,
        },
        "oracle": {
            "log_likelihood": float(value_physical.numpy()),
            "physical_score": physical_score.numpy().tolist(),
            "hmc_score": direct_hmc_score.numpy().tolist(),
            "hmc_score_from_chain_rule": chain_hmc_score.numpy().tolist(),
            "chain_rule_error": chain_error.numpy().tolist(),
            "chain_rule_absolute_tolerance": chain_tolerance.numpy().tolist(),
            "chain_rule_tolerance_formula": (
                "256*float64_unit_roundoff*"
                "max(1,abs(direct_hmc_score_k),abs(chain_hmc_score_k))"
            ),
            "chain_rule_tolerance_role": (
                "engineering consistency allowance, not a formal forward-error bound"
            ),
        },
        "proposal_scale": {
            "hmc_center": u.numpy().tolist(),
            "box_radii": radii.numpy().tolist(),
            "weighted_oracle_contributions": (
                weighted_oracle_contributions.numpy().tolist()
            ),
            "oracle_contribution_share": contribution_share.numpy().tolist(),
            "s_oracle": float(s_oracle.numpy()),
            "absolute_hmc_gradient_error_budget_per_unit_delta_grad": (
                absolute_budget_per_delta.numpy().tolist()
            ),
            "formula": "abs_error_budget_k = delta_grad * s_oracle / r_k",
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
            "jit_compile": False,
            "jit_exception": (
                "explicit comparator-only eager GradientTape reference; not default "
                "GPU/XLA, production, HMC, or candidate evidence"
            ),
            "execution_role": "CPU_HIDDEN_COMPARATOR_ONLY_REFERENCE_EXCEPTION",
            "wall_time_seconds": time.perf_counter() - started,
        },
        "run_manifest": {
            "command": EXACT_COMMAND,
            "plan_file": PLAN_PATH,
            "result_file": RESULT_PATH,
            "campaign_deadline": CAMPAIGN_DEADLINE,
            "attempts_authorized": 1,
            "attempts_executed": 1,
            "retries_authorized": 0,
            "timeout_seconds": 120,
            "cpu_only_intentional": True,
            "gpu_run": False,
            "jit_compile": False,
            "forbidden_repository_modules": list(FORBIDDEN_REPOSITORY_MODULES),
            "forbidden_source_imports_observed": forbidden_source_imports,
            "forbidden_loaded_modules_observed": forbidden_loaded_modules,
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): _sha256(path) for path in sources
        },
        "nonclaims": [
            "no delta_grad selected",
            "no Contract E module imported or candidate output observed",
            "not Contract E accuracy, HMC readiness, Phase 9, or leaderboard evidence",
        ],
    }
    _write_exclusive(output, payload)


if __name__ == "__main__":
    main()
