#!/usr/bin/env python3
"""Emit the paired 16-seed small-shape Contract E reset audit."""

from __future__ import annotations

import argparse
import hashlib
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
import tensorflow_probability as tfp

from bayesfilter.highdim import ledh_contract_e_canonical_lgssm_tf as canonical
from bayesfilter.highdim import ledh_contract_e_identity as identity
from bayesfilter.highdim import ledh_contract_e_lgssm_preparation_tf as preparation
from bayesfilter.highdim.transport_chunk_policy import select_transport_chunks
from bayesfilter.linear.kalman_tf import tf_kalman_log_likelihood
from scripts.filtering_value_gradient_benchmark_generate_p8_datasets import _lgssm_dataset


DATASET_SEED = 81100
ESTIMATOR_SEEDS = tuple(range(81220, 81236))
TIME_STEPS = 2
NUM_PARTICLES = 128
THETA = (0.72, 0.55, 0.35, 0.35, 0.45)
RIDGE = 0.1225 * (2.0**-24)
STEPS = 20
FAMILYWISE_LEVEL = 0.95
FAMILY_SIZE = 6
VALUE_BOUNDARY = 0.001
GRADIENT_BOUNDARY = 0.05
EXPECTED_OBSERVATION_SHA256 = "8b37260849e3b8e95e244dfe17109f2b22cde4abcf4779cd452a3438e341d635"
EXPECTED_SOURCE_SHA256 = {
    "bayesfilter/highdim/ledh_contract_e_canonical_lgssm_tf.py": "6201d85642474a9819a1c8972e94bd49cd317cba9a5862145f90252ddcdd0d24",
    "bayesfilter/highdim/ledh_contract_e_streaming_tf.py": "b2208a9e9f65bceaa6a629e69fb8c0edcdeab39d79ba6f0e5c04c45a427ef34a",
    "bayesfilter/highdim/ledh_contract_e_reset_tf.py": "5a226b53f4a881a1b66cee00902dcd007c82de3c01e3440101c111c5095ee023",
}
QUANTITY_NAMES = ("value", "phi1", "phi2", "phi3", "q_scale", "r_scale")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_hash(value: Any) -> str:
    return hashlib.sha256(tf.io.serialize_tensor(tf.convert_to_tensor(value)).numpy()).hexdigest()


def _finite(value: Any) -> bool:
    tensor = tf.convert_to_tensor(value)
    if tensor.dtype == tf.bool:
        return True
    return bool(tf.reduce_all(tf.math.is_finite(tensor)).numpy())


def _kalman(theta: tf.Tensor, observations: tf.Tensor) -> tf.Tensor:
    phi = theta[:3]
    q = theta[3]
    r = theta[4]
    return tf_kalman_log_likelihood(
        observations=observations,
        transition_offset=tf.zeros([3], tf.float64),
        transition_matrix=tf.linalg.diag(phi),
        transition_covariance=tf.square(q) * tf.eye(3, dtype=tf.float64),
        observation_offset=tf.zeros([3], tf.float64),
        observation_matrix=canonical._observation_matrix(tf.float64),
        observation_covariance=tf.square(r) * tf.eye(3, dtype=tf.float64),
        initial_state_mean=tf.zeros([3], tf.float64),
        initial_state_covariance=tf.linalg.diag(tf.square(q) / (1.0 - tf.square(phi))),
    )


def _prepare(observations: tf.Tensor, active: bool) -> dict[str, Any]:
    batch = len(ESTIMATOR_SEEDS)
    chunks = select_transport_chunks(NUM_PARTICLES)
    return preparation.prepare_contract_e_lgssm_inputs(
        observations=observations,
        estimator_seeds=ESTIMATOR_SEEDS,
        num_particles=NUM_PARTICLES,
        fixed_reset_mask=[[active] * TIME_STEPS for _ in range(batch)],
        prepared_ridge=[[RIDGE] * TIME_STEPS for _ in range(batch)],
        epsilon=0.5,
        scaling=0.9,
        sinkhorn_steps=STEPS,
        balance_steps=1,
        row_chunk_size=chunks.row_chunk_size,
        col_chunk_size=chunks.col_chunk_size,
        dtype=tf.float64,
    )


def _arm(observations: tf.Tensor, active: bool) -> tuple[dict[str, Any], dict[str, tf.Tensor]]:
    prepared = _prepare(observations, active)
    chunks = select_transport_chunks(NUM_PARTICLES)
    callable_ = canonical.make_canonical_value_and_score_tf(
        preparation.prepared_values(prepared),
        steps=STEPS,
        balance_steps=1,
        row_chunk_size=chunks.row_chunk_size,
        col_chunk_size=chunks.col_chunk_size,
        jit_compile=True,
        dtype=tf.float64,
    )
    theta = tf.constant(THETA, tf.float64)
    first = callable_(theta)
    second = callable_(theta)
    repeated = all(
        tf.io.serialize_tensor(first[name]).numpy()
        == tf.io.serialize_tensor(second[name]).numpy()
        for name in first
    )
    executed_names = [
        name for name in first if name not in {"minimum_mass", "minimum_mass_history"}
    ]
    checks = {
        "repeated_bitwise": repeated,
        "valid_chart": bool(tf.reduce_all(first["valid_chart"]).numpy()),
        "executed_outputs_finite": all(_finite(first[name]) for name in executed_names),
        "inactive_mass_sentinel_valid": (
            _finite(first["minimum_mass"])
            if active
            else bool(tf.reduce_all(tf.math.is_inf(first["minimum_mass"])).numpy())
        ),
        "one_concrete_callable": len(callable_._list_all_concrete_functions_for_serialization()) == 1,
    }
    if not all(checks.values()):
        raise RuntimeError(f"arm checks failed: {checks}")
    chain = tf.concat([1.0 - tf.square(theta[:3]), theta[3:]], axis=0)
    return (
        {
            "policy": "all_active_contract_e" if active else "no_reset_weighted",
            "checks": checks,
            "preparation_identity": prepared["identity"],
            "objective": float(first["objective"].numpy()),
            "per_seed_value": first["per_batch_log_likelihood"].numpy().tolist(),
            "mean_physical_score": first["score"].numpy().tolist(),
            "per_seed_physical_score": first["per_batch_score"].numpy().tolist(),
            "mean_hmc_score": (first["score"] * chain).numpy().tolist(),
            "per_seed_hmc_score": (first["per_batch_score"] * chain[None, :]).numpy().tolist(),
            "branch_hashes": {
                name: _tensor_hash(first[name])
                for name in (
                    "flow_valid_history",
                    "geometry_valid_history",
                    "quotient_valid_history",
                    "reset_valid_history",
                    "active_reset_history",
                )
            },
        },
        first,
    )


def _interval(values: list[float], critical: float) -> dict[str, float]:
    n = float(len(values))
    if n <= 1.0 or any(not math.isfinite(value) for value in values):
        raise ValueError("Student interval requires at least two finite values")
    mean = math.fsum(values) / n
    sample_variance = math.fsum((value - mean) ** 2 for value in values) / (n - 1.0)
    sample_std = math.sqrt(sample_variance)
    standard_error = sample_std / math.sqrt(n)
    half_width = critical * standard_error
    return {
        "mean": mean,
        "sample_standard_deviation": sample_std,
        "standard_error": standard_error,
        "critical_value": critical,
        "half_width": half_width,
        "lower": mean - half_width,
        "upper": mean + half_width,
    }


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _static_context() -> tuple[tf.Tensor, dict[str, str]]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("CUDA_VISIBLE_DEVICES=-1 is required before TensorFlow import")
    observations = tf.convert_to_tensor(_lgssm_dataset(DATASET_SEED)["observations"][:TIME_STEPS], tf.float64)
    if _tensor_hash(observations) != EXPECTED_OBSERVATION_SHA256:
        raise RuntimeError("observation identity mismatch")
    realized_hashes = {name: _sha256(ROOT / name) for name in EXPECTED_SOURCE_SHA256}
    if realized_hashes != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("source closure drifted")
    return observations, realized_hashes


def _arm_child(output: Path, policy: str) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    started = time.perf_counter()
    observations, realized_hashes = _static_context()
    arm, _ = _arm(observations, policy == "all_active_contract_e")
    payload = {
        "schema_version": "bayesfilter.contract_e_phase8.paired_reset_audit16_arm.v1",
        "program_id": "contract-e-canonical-gradient-migration-20260713",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PAIRED_AUDIT16_ARM_COMPLETE",
        "arm": arm,
        "environment": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "logical_devices": [device.name for device in tf.config.list_logical_devices()],
            "dtype": "float64",
            "jit_compile": True,
            "wall_time_seconds": time.perf_counter() - started,
            "source_sha256": realized_hashes,
            "harness_sha256": _sha256(Path(__file__)),
        },
    }
    _write_exclusive(output, payload)
    print(json.dumps({"output": str(output), "status": payload["status"], "policy": policy}, sort_keys=True))


def _paired_preparation_identity_check(
    contract_e: dict[str, Any], no_reset: dict[str, Any]
) -> dict[str, Any]:
    left = contract_e["preparation_identity"]
    right = no_reset["preparation_identity"]
    left_hashes = dict(left["tensor_sha256"])
    right_hashes = dict(right["tensor_sha256"])
    left_mask = left_hashes.pop("fixed_reset_mask")
    right_mask = right_hashes.pop("fixed_reset_mask")
    fields = (
        "preparation_id",
        "residual_design_id",
        "rng_algorithm",
        "root_seeds_in_order",
        "num_particles",
        "time_steps",
        "sinkhorn_steps",
        "row_chunk_size",
        "col_chunk_size",
    )
    common_fields_equal = all(left[name] == right[name] for name in fields)
    checks = {
        "common_preparation_fields_equal": common_fields_equal,
        "all_non_mask_tensor_hashes_equal": left_hashes == right_hashes,
        "reset_mask_hashes_differ": left_mask != right_mask,
        "seed_order_exact": left["root_seeds_in_order"] == list(ESTIMATOR_SEEDS),
    }
    if not all(checks.values()):
        raise RuntimeError(f"paired preparation identity failed: {checks}")
    return checks


def _aggregate(output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    started = time.perf_counter()
    observations, realized_hashes = _static_context()
    arm_root = output.parent / f"{output.stem}-arms"
    arm_root.mkdir(parents=True, exist_ok=False)
    arm_records: dict[str, dict[str, Any]] = {}
    attempts = []
    for policy in ("all_active_contract_e", "no_reset_weighted"):
        arm_output = arm_root / f"{policy}.json"
        log_path = arm_root / f"{policy}.log"
        command = [
            sys.executable,
            str(Path(__file__).resolve().relative_to(ROOT)),
            "--output",
            str(arm_output.relative_to(ROOT)),
            "--arm",
            policy,
        ]
        arm_started = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env={
                    **os.environ,
                    "CUDA_VISIBLE_DEVICES": "-1",
                    "TF_ENABLE_ONEDNN_OPTS": "0",
                    "MPLCONFIGDIR": "/tmp",
                },
                capture_output=True,
                text=True,
                timeout=600,
                check=False,
            )
            log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
            exit_code = completed.returncode
            timed_out = False
        except subprocess.TimeoutExpired as error:
            stdout = error.stdout.decode() if isinstance(error.stdout, bytes) else (error.stdout or "")
            stderr = error.stderr.decode() if isinstance(error.stderr, bytes) else (error.stderr or "")
            log_path.write_text(stdout + stderr, encoding="utf-8")
            exit_code = None
            timed_out = True
        attempt = {
            "policy": policy,
            "command": command,
            "timeout_seconds": 600,
            "exit_code": exit_code,
            "timed_out": timed_out,
            "wall_time_seconds": time.perf_counter() - arm_started,
            "output": str(arm_output.relative_to(ROOT)),
            "log": str(log_path.relative_to(ROOT)),
        }
        attempts.append(attempt)
        if exit_code != 0 or not arm_output.is_file():
            raise RuntimeError(f"arm failed: {attempt}")
        arm_payload = json.loads(arm_output.read_text(encoding="utf-8"))
        if arm_payload.get("status") != "PAIRED_AUDIT16_ARM_COMPLETE":
            raise RuntimeError(f"arm status invalid: {policy}")
        arm_records[policy] = arm_payload["arm"]
    contract_e = arm_records["all_active_contract_e"]
    no_reset = arm_records["no_reset_weighted"]
    paired_identity_checks = _paired_preparation_identity_check(contract_e, no_reset)
    theta = tf.constant(THETA, tf.float64)
    with tf.GradientTape() as tape:
        tape.watch(theta)
        kalman_value_tensor = _kalman(theta, observations)
    kalman_physical = tape.gradient(kalman_value_tensor, theta)
    chain = tf.concat([1.0 - tf.square(theta[:3]), theta[3:]], axis=0)
    kalman_hmc = kalman_physical * chain
    scales = [abs(float(kalman_value_tensor.numpy())), *[abs(float(item)) for item in kalman_hmc.numpy()]]
    if any(scale == 0.0 or not math.isfinite(scale) for scale in scales):
        raise RuntimeError(f"invalid Kalman scale: {scales}")
    per_arm_z: dict[str, list[list[float]]] = {}
    for arm in (contract_e, no_reset):
        rows = []
        for value, gradient in zip(arm["per_seed_value"], arm["per_seed_hmc_score"], strict=True):
            rows.append([
                (float(value) - float(kalman_value_tensor.numpy())) / scales[0],
                *[
                    (float(candidate) - float(oracle)) / scale
                    for candidate, oracle, scale in zip(gradient, kalman_hmc.numpy(), scales[1:], strict=True)
                ],
            ])
        per_arm_z[arm["policy"]] = rows
    paired_loss = [
        [abs(left) - abs(right) for left, right in zip(ce_row, nr_row, strict=True)]
        for ce_row, nr_row in zip(per_arm_z["all_active_contract_e"], per_arm_z["no_reset_weighted"], strict=True)
    ]
    alpha_member = (1.0 - FAMILYWISE_LEVEL) / FAMILY_SIZE
    critical = float(
        tfp.distributions.StudentT(
            df=tf.constant(float(len(ESTIMATOR_SEEDS) - 1), tf.float64),
            loc=tf.constant(0.0, tf.float64),
            scale=tf.constant(1.0, tf.float64),
        ).quantile(tf.constant(1.0 - alpha_member / 2.0, tf.float64)).numpy()
    )
    loss_intervals = []
    directions = []
    for index, name in enumerate(QUANTITY_NAMES):
        interval = _interval([row[index] for row in paired_loss], critical)
        if interval["upper"] < 0.0:
            direction = "contract_e_lower_mean_absolute_error"
        elif interval["lower"] > 0.0:
            direction = "contract_e_higher_mean_absolute_error"
        else:
            direction = "inconclusive"
        directions.append(direction)
        loss_intervals.append({"quantity": name, "direction": direction, **interval})
    if len(set(directions)) == 1 and directions[0] != "inconclusive":
        overall = directions[0]
    else:
        overall = "mixed_or_inconclusive"
    contract_intervals = []
    for index, name in enumerate(QUANTITY_NAMES):
        interval = _interval([row[index] for row in per_arm_z["all_active_contract_e"]], critical)
        boundary = VALUE_BOUNDARY if index == 0 else GRADIENT_BOUNDARY
        equivalent = interval["lower"] > -boundary and interval["upper"] < boundary
        contract_intervals.append({"quantity": name, "boundary": boundary, "equivalent": equivalent, **interval})
    payload = {
        "schema_version": "bayesfilter.contract_e_phase8.paired_reset_audit16.v1",
        "program_id": "contract-e-canonical-gradient-migration-20260713",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PAIRED_AUDIT16_COMPLETE",
        "overall_paired_loss_classification": overall,
        "quantity_names": list(QUANTITY_NAMES),
        "estimator_seeds": list(ESTIMATOR_SEEDS),
        "kalman": {
            "value": float(kalman_value_tensor.numpy()),
            "physical_score": kalman_physical.numpy().tolist(),
            "hmc_score": kalman_hmc.numpy().tolist(),
            "scales": scales,
        },
        "arms": {contract_e["policy"]: contract_e, no_reset["policy"]: no_reset},
        "paired_preparation_identity_checks": paired_identity_checks,
        "arm_attempts": attempts,
        "per_arm_signed_normalized_error": per_arm_z,
        "paired_absolute_loss_difference": paired_loss,
        "paired_loss_intervals": loss_intervals,
        "contract_e_mean_error_intervals": contract_intervals,
        "interval_model": {
            "familywise_level": FAMILYWISE_LEVEL,
            "family_size": FAMILY_SIZE,
            "member_two_sided_alpha": alpha_member,
            "degrees_of_freedom": len(ESTIMATOR_SEEDS) - 1,
            "critical_value": critical,
            "model": "two_sided_Bonferroni_Student",
            "power_claim": False,
        },
        "route_identity": {
            "factory_registry_empty": not bool(identity._PRODUCTION_FACTORY._route_specifications),
            "reset_contract_id": identity.CONTRACT_E_RESET_CONTRACT_ID,
            "derivative_composition_id": identity.CONTRACT_E_DERIVATIVE_COMPOSITION_ID,
            "status": "diagnostic_only_factory_empty",
        },
        "run_manifest": {
            "command": [sys.executable, *sys.argv],
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip(),
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "tensorflow_probability": tfp.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "logical_devices": [device.name for device in tf.config.list_logical_devices()],
            "dtype": "float64",
            "jit_compile": True,
            "time_steps": TIME_STEPS,
            "num_particles": NUM_PARTICLES,
            "wall_time_seconds": time.perf_counter() - started,
            "source_sha256": realized_hashes,
            "harness_sha256": _sha256(Path(__file__)),
        },
        "nonclaims": [
            "small-shape diagnostic, not primary-shape equivalence",
            "Student/Bonferroni intervals are model-based with no power guarantee",
            "not canonical admission, full-box HMC readiness, leaderboard release, or program completion",
        ],
    }
    _write_exclusive(output, payload)
    print(json.dumps({"output": str(output), "status": payload["status"], "classification": overall}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--arm",
        choices=("all_active_contract_e", "no_reset_weighted"),
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if args.arm:
        _arm_child(output, args.arm)
    else:
        _aggregate(output)


if __name__ == "__main__":
    main()
