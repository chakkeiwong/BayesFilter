"""Gradient scalar-contract audit for reusable annealed transport."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]

import argparse
import time
from typing import Any

import tensorflow as tf

from experiments.dpf_implementation.tf_tfp.resampling.annealed_transport_tf import (
    annealed_transport_resample_tf,
)
from experiments.dpf_implementation.tf_tfp.runners import (
    run_filterflow_final_gaps_closure_tf as final_gaps,
)
from experiments.dpf_implementation.tf_tfp.runners import (
    run_filterflow_lgssm_matched_cross_audit_tf as matched,
)
from experiments.dpf_implementation.tf_tfp.runners.common_tf import (
    OUTPUT_DIR,
    REPORT_DIR,
    environment_manifest,
    load_json,
    stable_digest,
    utc_now,
    write_json,
    write_text,
)


DTYPE = tf.float64
PLAN_PATH = "docs/plans/bayesfilter-dpf-annealed-transport-reference-alignment-plan-2026-05-31.md"
JSON_PATH = OUTPUT_DIR / "dpf_annealed_transport_gradient_contract_2026-05-31.json"
REPORT_PATH = REPORT_DIR / "dpf-annealed-transport-gradient-contract-2026-05-31.md"
THETA_GRID = (0.95, 0.9666666388511658, 0.9833333492279053, 1.0)
NUM_PARTICLES = 25


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only:
        _validate_payload(load_json(JSON_PATH))
        return 0
    start = time.perf_counter()
    payload = _run()
    payload["run_manifest"]["wall_time_seconds"] = time.perf_counter() - start
    payload["reproducibility_digest"] = _digest_payload(payload)
    write_json(JSON_PATH, payload)
    write_text(REPORT_PATH, _markdown(payload))
    _validate_payload(payload)
    print(payload["decision"])
    return 0


def _run() -> dict[str, Any]:
    smooth = final_gaps._run_smoothness_subprocess()
    filterflow = matched._run_filterflow_subprocess()
    observations = tf.constant(filterflow["observations"][:100], dtype=DTYPE)
    initial_particles = tf.constant(filterflow["initial_particles"][0], dtype=DTYPE)
    base_noises = tf.random.stateless_normal(
        [100, NUM_PARTICLES, 2],
        seed=tf.constant([2468, 1357], dtype=tf.int32),
        dtype=DTYPE,
    )
    bayesfilter_surface = []
    for theta_1 in THETA_GRID:
        for theta_2 in THETA_GRID:
            value_grad = _bayesfilter_scalar_and_grad(
                observations=observations,
                initial_particles=initial_particles,
                base_noises=base_noises,
                theta_1=float(theta_1),
                theta_2=float(theta_2),
            )
            bayesfilter_surface.append({"theta_1": float(theta_1), "theta_2": float(theta_2), **value_grad})
    scalar_ledger = _scalar_ledger(smooth, bayesfilter_surface)
    decision = "annealed_transport_gradient_contract_severe_unreconciled_magnitude_risk_recorded"
    if not all(row["finite_gradient"] and row["finite_value"] for row in bayesfilter_surface):
        decision = "annealed_transport_gradient_contract_nonfinite_veto"
    return {
        "decision": decision,
        "created_at_utc": utc_now(),
        "plan_path": PLAN_PATH,
        "question": "Same-scalar gradient contract audit for reusable annealed transport.",
        "filterflow_smoothness": smooth,
        "bayesfilter_gradient_surface": bayesfilter_surface,
        "scalar_contract_ledger": scalar_ledger,
        "gradient_claim_status": "finite_gradient_smoke_not_agreement",
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners.run_filterflow_annealed_transport_gradient_contract_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_implications": _non_implications(),
    }


def _bayesfilter_scalar_and_grad(
    *,
    observations: tf.Tensor,
    initial_particles: tf.Tensor,
    base_noises: tf.Tensor,
    theta_1: float,
    theta_2: float,
) -> dict[str, Any]:
    theta = tf.Variable([theta_1, theta_2], dtype=DTYPE)
    with tf.GradientTape() as tape:
        value = _bayesfilter_total_log_likelihood(
            observations=observations,
            initial_particles=initial_particles,
            base_noises=base_noises,
            theta=theta,
        )
    grad = tape.gradient(value, theta)
    return {
        "total_log_likelihood": float(value.numpy()),
        "negative_total_log_likelihood": float((-value).numpy()),
        "per_time_log_likelihood": float((value / tf.cast(tf.shape(observations)[0], DTYPE)).numpy()),
        "gradient_total_log_likelihood": tf.cast(grad, DTYPE).numpy().tolist(),
        "finite_value": bool(tf.math.is_finite(value).numpy()),
        "finite_gradient": bool(tf.reduce_all(tf.math.is_finite(grad)).numpy()),
    }


def _bayesfilter_total_log_likelihood(
    *,
    observations: tf.Tensor,
    initial_particles: tf.Tensor,
    base_noises: tf.Tensor,
    theta: tf.Tensor,
) -> tf.Tensor:
    particles = tf.cast(initial_particles, DTYPE)
    log_weights = tf.fill([NUM_PARTICLES], -tf.math.log(tf.cast(NUM_PARTICLES, DTYPE)))
    log_likelihood = tf.constant(0.0, DTYPE)
    observation_covariance = tf.eye(2, dtype=DTYPE) * tf.constant(0.1, DTYPE)
    for time_index, observation in enumerate(tf.unstack(tf.cast(observations, DTYPE), axis=0)):
        weights = tf.exp(log_weights)
        ess = 1.0 / tf.reduce_sum(weights * weights)
        do_resample = ess <= tf.constant(0.5 * NUM_PARTICLES, DTYPE)
        resampled = annealed_transport_resample_tf(
            particles,
            log_weights,
            epsilon=0.25,
            scaling=0.85,
            convergence_threshold=1e-6,
            max_iterations=80,
            ess_mask=tf.reshape(do_resample, [1]),
        )
        particles = tf.reshape(resampled.particles, [NUM_PARTICLES, 2])
        log_weights = tf.reshape(resampled.log_weights, [NUM_PARTICLES])
        particles = particles * theta[None, :] + tf.cast(base_noises[time_index], DTYPE)
        obs_logp = _observation_log_prob_single(particles, observation, observation_covariance)
        unnormalized = log_weights + obs_logp
        normalizer = tf.reduce_logsumexp(unnormalized)
        log_likelihood = log_likelihood + normalizer
        log_weights = unnormalized - normalizer
    return log_likelihood


def _observation_log_prob_single(particles: tf.Tensor, observation: tf.Tensor, covariance: tf.Tensor) -> tf.Tensor:
    residual = tf.cast(particles, DTYPE) - tf.cast(observation, DTYPE)[None, :]
    chol = tf.linalg.cholesky(covariance)
    solved = tf.linalg.cholesky_solve(chol, tf.transpose(residual))
    quad = tf.reduce_sum(tf.transpose(solved) * residual, axis=1)
    logdet = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)))
    return -0.5 * (
        2.0 * tf.math.log(tf.constant(2.0 * 3.141592653589793, DTYPE))
        + logdet
        + quad
    )


def _scalar_ledger(smooth: dict[str, Any], bayesfilter_surface: list[dict[str, Any]]) -> dict[str, Any]:
    status = "filterflow_smoothness_executed" if smooth.get("status") == "executed" else smooth.get("status")
    filterflow_gradients = smooth.get("dpf_gradients", [])
    bayes_gradients = [row["gradient_total_log_likelihood"] for row in bayesfilter_surface]
    comparable_count = min(len(filterflow_gradients), len(bayes_gradients))
    return {
        "status": "severe_unreconciled_gradient_magnitude_mismatch_risk_recorded",
        "filterflow_status": status,
        "filterflow_scalar": "total_log_likelihood_from_simple_linear_smoothness",
        "bayesfilter_scalar": "total_log_likelihood_common_observations_common_random_numbers",
        "normalization_variants_recorded": "total, negative total, per-time",
        "filterflow_gradient_count": len(filterflow_gradients),
        "bayesfilter_gradient_count": len(bayes_gradients),
        "comparable_gradient_count": comparable_count,
        "bayesfilter_all_finite": all(row["finite_gradient"] and row["finite_value"] for row in bayesfilter_surface),
        "interpretation": (
            "BayesFilter GradientTape surface is finite, but filterflow/BayesFilter same-randomness "
            "and gradient-magnitude agreement are severely unreconciled. Gradient agreement is not concluded."
        ),
    }


def _validate_payload(payload: dict[str, Any]) -> None:
    if payload["decision"] not in {
        "annealed_transport_gradient_contract_severe_unreconciled_magnitude_risk_recorded",
        "annealed_transport_gradient_contract_nonfinite_veto",
    }:
        raise RuntimeError(payload["decision"])
    if payload["gradient_claim_status"] != "finite_gradient_smoke_not_agreement":
        raise RuntimeError("gradient overclaim")
    if payload["run_manifest"]["pre_import_cuda_visible_devices"] != "-1":
        raise RuntimeError("missing CPU-only manifest")
    if "reproducibility_digest" not in payload:
        raise RuntimeError("missing digest")


def _markdown(payload: dict[str, Any]) -> str:
    return f"""# Annealed Transport Gradient Contract

## Decision

`{payload['decision']}`

## Scalar Contract Ledger

{_key_value_table(payload['scalar_contract_ledger'])}

## Gradient Claim Status

`{payload['gradient_claim_status']}`

## Non-Implications

{_bullets(payload['non_implications'])}
"""


def _key_value_table(values: dict[str, Any]) -> str:
    lines = ["| Key | Value |", "| --- | --- |"]
    for key, value in values.items():
        lines.append(f"| `{key}` | `{value}` |")
    return "\n".join(lines)


def _non_implications() -> list[str]:
    return [
        "No gradient agreement is concluded.",
        "No posterior correctness is concluded.",
        "No HMC readiness is concluded.",
        "No production readiness is concluded.",
        "No full supplement figure or table reproduction is concluded.",
    ]


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def _digest_payload(payload: dict[str, Any]) -> str:
    comparable = dict(payload)
    comparable["created_at_utc"] = "TIMESTAMP"
    comparable["run_manifest"] = dict(comparable["run_manifest"])
    comparable["run_manifest"]["wall_time_seconds"] = "WALL_TIME"
    comparable["run_manifest"]["dirty_state_summary"] = "DIRTY_STATE"
    return stable_digest(comparable)


if __name__ == "__main__":
    raise SystemExit(main())
