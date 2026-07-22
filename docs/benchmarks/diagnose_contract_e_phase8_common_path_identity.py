#!/usr/bin/env python3
"""Diagnose the exact LGSSM affine-proposal importance-weight identity."""

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

from bayesfilter.highdim import ledh_contract_e_canonical_lgssm_tf as canonical


THETA = (0.72, 0.55, 0.35, 0.35, 0.45)
ATOL = 2.0e-11
RTOL = 2.0e-11
EXPECTED_SOURCE_SHA256 = {
    "bayesfilter/highdim/ledh_contract_e_canonical_lgssm_tf.py":
        "33f37f6bfd156b82b3f66334545ce5c16ddb94a59040a2d434a36cec06ad8f0b",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _fixed_inputs() -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    initial_noise = tf.reshape(
        tf.linspace(tf.constant(-1.15, tf.float64), tf.constant(1.30, tf.float64), 24),
        [2, 4, 3],
    )
    transition_noise = tf.reshape(
        tf.linspace(tf.constant(0.85, tf.float64), tf.constant(-0.95, tf.float64), 24),
        [2, 4, 3],
    )
    observation = tf.constant([0.31, -0.27, 0.44], tf.float64)
    return initial_noise, transition_noise, observation


def _forward_expressions(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    initial_noise, transition_noise, observation = _fixed_inputs()
    components = canonical._lgssm_components(theta, batch_size=2)
    previous = initial_noise * components["initial_std"][None, None, :]
    prior_mean = tf.einsum("bnj,bdj->bnd", previous, components["transition_matrix"])
    pre_flow = prior_mean + components["q_scale"] * transition_noise
    flow = canonical._lgssm_flow_forward_core(
        prior_mean,
        pre_flow,
        observation,
        components["transition_covariance"],
        components["observation_covariance"],
    )
    transition_density = canonical._gaussian_log_density_forward_core(
        flow["particles"] - prior_mean, components["transition_covariance"]
    )["value"]
    proposal_density = canonical._gaussian_log_density_forward_core(
        pre_flow - prior_mean, components["transition_covariance"]
    )["value"]
    predicted_observation = tf.einsum(
        "md,bnd->bnm", components["observation_matrix"], flow["particles"]
    )
    observation_density = canonical._gaussian_log_density_forward_core(
        predicted_observation - observation[None, None, :],
        components["observation_covariance"],
    )["value"]
    correction = (
        transition_density
        + observation_density
        - proposal_density
        + flow["forward_log_abs_det"][:, None]
    )

    predictive_mean = tf.einsum(
        "md,bnd->bnm", components["observation_matrix"], prior_mean
    )
    predictive_covariance = (
        tf.einsum(
            "md,bdq,nq->bmn",
            components["observation_matrix"],
            components["transition_covariance"],
            components["observation_matrix"],
        )
        + components["observation_covariance"]
    )
    analytic = canonical._gaussian_log_density_forward_core(
        predictive_mean - observation[None, None, :], predictive_covariance
    )["value"]
    uniform_log_weight = -tf.math.log(tf.constant(4.0, tf.float64))
    computed_increment = tf.reduce_logsumexp(correction + uniform_log_weight, axis=1)
    analytic_increment = tf.reduce_logsumexp(analytic + uniform_log_weight, axis=1)
    return correction, analytic, computed_increment, analytic_increment


def _manual_correction_jvp(theta: tf.Tensor) -> tf.Tensor:
    initial_noise, transition_noise, observation = _fixed_inputs()
    components = canonical._lgssm_components(theta, batch_size=2)
    tangents = canonical._lgssm_component_tangents(theta, batch_size=2)
    previous = initial_noise * components["initial_std"][None, None, :]
    previous_tangent = (
        initial_noise[:, :, :, None] * tangents["d_initial_std"][None, None, :, :]
    )
    prior_mean = tf.einsum("bnj,bdj->bnd", previous, components["transition_matrix"])
    prior_mean_tangent = (
        tf.einsum(
            "bnjk,bdj->bndk", previous_tangent, components["transition_matrix"]
        )
        + tf.einsum(
            "bnj,bdjk->bndk", previous, tangents["d_transition_matrix"]
        )
    )
    pre_flow = prior_mean + components["q_scale"] * transition_noise
    pre_flow_tangent = prior_mean_tangent + (
        transition_noise[:, :, :, None]
        * tangents["d_transition_scale"][None, None, None, :]
    )
    flow = canonical._lgssm_flow_forward_core(
        prior_mean,
        pre_flow,
        observation,
        components["transition_covariance"],
        components["observation_covariance"],
    )
    flow_tangent = canonical._lgssm_flow_jvp_core(
        prior_mean,
        pre_flow,
        observation,
        components["transition_covariance"],
        components["observation_covariance"],
        prior_mean_tangent,
        pre_flow_tangent,
        tangents["d_transition_covariance"],
        tangents["d_observation_covariance"],
    )
    transition_density = canonical._gaussian_log_density_jvp_core(
        flow["particles"] - prior_mean,
        components["transition_covariance"],
        flow_tangent["particles"] - prior_mean_tangent,
        tangents["d_transition_covariance"],
    )
    proposal_density = canonical._gaussian_log_density_jvp_core(
        pre_flow - prior_mean,
        components["transition_covariance"],
        pre_flow_tangent - prior_mean_tangent,
        tangents["d_transition_covariance"],
    )
    predicted_observation = tf.einsum(
        "md,bnd->bnm", components["observation_matrix"], flow["particles"]
    )
    predicted_observation_tangent = tf.einsum(
        "md,bndk->bnmk", components["observation_matrix"], flow_tangent["particles"]
    )
    observation_density = canonical._gaussian_log_density_jvp_core(
        predicted_observation - observation[None, None, :],
        components["observation_covariance"],
        predicted_observation_tangent,
        tangents["d_observation_covariance"],
    )
    return (
        transition_density["tangent"]
        + observation_density["tangent"]
        - proposal_density["tangent"]
        + flow_tangent["forward_log_abs_det"][:, None, :]
    )


def _error(left: tf.Tensor, right: tf.Tensor) -> dict[str, float | bool]:
    left = tf.convert_to_tensor(left, tf.float64)
    right = tf.convert_to_tensor(right, tf.float64)
    absolute = tf.abs(left - right)
    scale = tf.maximum(tf.abs(left), tf.abs(right))
    relative = tf.math.divide_no_nan(absolute, scale)
    return {
        "maximum_absolute_error": float(tf.reduce_max(absolute).numpy()),
        "maximum_relative_error": float(tf.reduce_max(relative).numpy()),
        "all_close": bool(tf.reduce_all(absolute <= ATOL + RTOL * scale).numpy()),
    }


def _two_step_prepared() -> dict[str, tf.Tensor]:
    initial_noise, first_noise, first_observation = _fixed_inputs()
    second_noise = tf.reverse(first_noise, axis=[1]) * tf.constant(0.73, tf.float64)
    return {
        "observations": tf.stack(
            [first_observation, tf.constant([-0.18, 0.36, -0.22], tf.float64)]
        ),
        "initial_noise": initial_noise,
        "transition_noise": tf.stack([first_noise, second_noise], axis=1),
        "fixed_reset_mask": tf.zeros([2, 2], tf.bool),
        "residual_design": tf.reshape(
            tf.linspace(tf.constant(-0.7, tf.float64), tf.constant(0.8, tf.float64), 48),
            [2, 2, 4, 3],
        ),
        "prepared_ridge": tf.fill([2, 2], tf.constant(1.0e-6, tf.float64)),
        "epsilon": tf.constant(0.5, tf.float64),
        "scaling": tf.constant(0.9, tf.float64),
    }


def _independent_two_step_sis(theta: tf.Tensor) -> tf.Tensor:
    prepared = _two_step_prepared()
    phi = theta[:3]
    q_scale = theta[3]
    r_scale = theta[4]
    transition = tf.linalg.diag(phi)
    observation_matrix = tf.constant(
        [[1.0, 0.25, -0.15], [0.2, 1.1, 0.3], [-0.1, 0.35, 0.9]],
        tf.float64,
    )
    q_covariance = tf.square(q_scale) * tf.eye(3, dtype=tf.float64)
    r_covariance = tf.square(r_scale) * tf.eye(3, dtype=tf.float64)
    prior_chol = tf.linalg.cholesky(q_covariance)
    predictive_covariance = (
        observation_matrix @ q_covariance @ tf.transpose(observation_matrix)
        + r_covariance
    )
    predictive_chol = tf.linalg.cholesky(predictive_covariance)
    gain = tf.transpose(
        tf.linalg.cholesky_solve(
            predictive_chol,
            tf.transpose(q_covariance @ tf.transpose(observation_matrix)),
        )
    )
    post_covariance = q_covariance - gain @ observation_matrix @ q_covariance
    post_chol = tf.linalg.cholesky(
        0.5 * (post_covariance + tf.transpose(post_covariance))
    )
    affine = post_chol @ tf.linalg.triangular_solve(
        prior_chol, tf.eye(3, dtype=tf.float64)
    )
    initial_std = q_scale / tf.sqrt(1.0 - tf.square(phi))
    particles = prepared["initial_noise"] * initial_std[None, None, :]
    log_weights = tf.fill([2, 4], -tf.math.log(tf.constant(4.0, tf.float64)))
    total = tf.zeros([2], tf.float64)
    log_two_pi = tf.math.log(tf.constant(2.0 * math.pi, tf.float64))
    logdet = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(predictive_chol)))
    for time_index in range(2):
        prior_mean = tf.einsum("bnj,dj->bnd", particles, transition)
        predicted_observation = tf.einsum("md,bnd->bnm", observation_matrix, prior_mean)
        residual = prepared["observations"][time_index][None, None, :] - predicted_observation
        solved = tf.linalg.cholesky_solve(
            predictive_chol, tf.transpose(residual, [0, 2, 1])
        )
        quadratic = tf.reduce_sum(residual * tf.transpose(solved, [0, 2, 1]), axis=2)
        correction = -0.5 * (tf.constant(3.0, tf.float64) * log_two_pi + logdet + quadratic)
        logits = log_weights + correction
        increment = tf.reduce_logsumexp(logits, axis=1)
        total = total + increment
        log_weights = logits - increment[:, None]
        innovation = prepared["observations"][time_index][None, None, :] - predicted_observation
        post_mean = prior_mean + tf.einsum("dm,bnm->bnd", gain, innovation)
        base_noise = q_scale * prepared["transition_noise"][:, time_index]
        particles = post_mean + tf.linalg.matmul(base_noise, affine, transpose_b=True)
    return total


def compute_diagnostic() -> dict[str, Any]:
    theta = tf.constant(THETA, tf.float64)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(theta)
        correction, analytic, computed_increment, analytic_increment = _forward_expressions(theta)
    correction_autodiff = tape.jacobian(correction, theta)
    analytic_autodiff = tape.jacobian(analytic, theta)
    computed_increment_autodiff = tape.jacobian(computed_increment, theta)
    analytic_increment_autodiff = tape.jacobian(analytic_increment, theta)
    del tape
    manual = _manual_correction_jvp(theta)
    prepared = _two_step_prepared()
    with tf.GradientTape() as tape:
        tape.watch(theta)
        canonical_primal = canonical._canonical_primal_core(
            theta,
            prepared,
            steps=2,
            balance_steps=0,
            row_chunk_size=2,
            col_chunk_size=2,
        )["per_batch_log_likelihood"]
    canonical_primal_autodiff = tape.jacobian(canonical_primal, theta)
    canonical_manual = canonical._canonical_manual_jvp_core(
        theta,
        prepared,
        steps=2,
        balance_steps=0,
        row_chunk_size=2,
        col_chunk_size=2,
    )["per_batch_score"]
    with tf.GradientTape() as tape:
        tape.watch(theta)
        independent_primal = _independent_two_step_sis(theta)
    independent_autodiff = tape.jacobian(independent_primal, theta)
    finite_values = [
        correction,
        analytic,
        computed_increment,
        analytic_increment,
        correction_autodiff,
        analytic_autodiff,
        computed_increment_autodiff,
        analytic_increment_autodiff,
        manual,
        canonical_primal,
        canonical_primal_autodiff,
        canonical_manual,
        independent_primal,
        independent_autodiff,
    ]
    checks = {
        "all_finite": all(
            bool(tf.reduce_all(tf.math.is_finite(value)).numpy()) for value in finite_values
        ),
        "correction_value_identity": _error(correction, analytic),
        "correction_autodiff_identity": _error(correction_autodiff, analytic_autodiff),
        "manual_jvp_vs_code_autodiff": _error(manual, correction_autodiff),
        "manual_jvp_vs_analytic_autodiff": _error(manual, analytic_autodiff),
        "normalization_value_identity": _error(computed_increment, analytic_increment),
        "normalization_autodiff_identity": _error(
            computed_increment_autodiff, analytic_increment_autodiff
        ),
        "two_step_sis_value_identity": _error(canonical_primal, independent_primal),
        "two_step_sis_autodiff_identity": _error(
            canonical_primal_autodiff, independent_autodiff
        ),
        "two_step_sis_manual_jvp_identity": _error(canonical_manual, independent_autodiff),
    }
    identity_checks = [
        value["all_close"] for value in checks.values() if isinstance(value, dict)
    ]
    passed = bool(checks["all_finite"] and all(identity_checks))
    return {
        "status": "COMMON_PATH_IDENTITY_PASS" if passed else "COMMON_PATH_IDENTITY_FAIL",
        "passed": passed,
        "theta": list(THETA),
        "shape": {"batch_size": 2, "num_particles": 4, "state_dimension": 3},
        "tolerances": {"absolute": ATOL, "relative": RTOL},
        "checks": checks,
        "computed_correction": correction.numpy().tolist(),
        "analytic_predictive_log_density": analytic.numpy().tolist(),
        "computed_increment": computed_increment.numpy().tolist(),
        "analytic_increment": analytic_increment.numpy().tolist(),
        "manual_jvp": manual.numpy().tolist(),
        "code_autodiff_jacobian": correction_autodiff.numpy().tolist(),
        "analytic_autodiff_jacobian": analytic_autodiff.numpy().tolist(),
        "two_step_canonical_value": canonical_primal.numpy().tolist(),
        "two_step_independent_value": independent_primal.numpy().tolist(),
        "two_step_canonical_manual_jvp": canonical_manual.numpy().tolist(),
        "two_step_independent_autodiff": independent_autodiff.numpy().tolist(),
    }


def main() -> None:
    raise RuntimeError("ARCHIVAL_WRONG_TRANSPORT_CHUNK_POLICY: this route is preserved only as provenance and cannot emit new evidence")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("CUDA_VISIBLE_DEVICES=-1 is required before TensorFlow import")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    source_hashes = {name: _sha256(ROOT / name) for name in EXPECTED_SOURCE_SHA256}
    if source_hashes != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(f"source closure drifted: {source_hashes}")
    started = time.perf_counter()
    diagnostic = compute_diagnostic()
    if not diagnostic["passed"]:
        raise RuntimeError(json.dumps(diagnostic["checks"], sort_keys=True))
    payload = {
        "schema_version": "bayesfilter.contract_e_phase8.common_path_identity.v1",
        "program_id": "contract-e-canonical-gradient-migration-20260713",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **diagnostic,
        "evidence_role": "local_common_path_implementation_identity",
        "run_manifest": {
            "command": [sys.executable, *sys.argv],
            "git_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "logical_devices": [device.name for device in tf.config.list_logical_devices()],
            "dtype": "float64",
            "jit_compile": False,
            "wall_time_seconds": time.perf_counter() - started,
            "source_sha256": source_hashes,
            "harness_sha256": _sha256(Path(__file__)),
        },
        "nonclaims": [
            "does not establish finite-particle or Kalman equivalence",
            "does not establish reset correctness or primary-shape validity",
            "does not establish GPU/XLA, admission, HMC, leaderboard, default, or release readiness",
        ],
    }
    _write_exclusive(output, payload)
    print(json.dumps({"output": str(output), "status": payload["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
