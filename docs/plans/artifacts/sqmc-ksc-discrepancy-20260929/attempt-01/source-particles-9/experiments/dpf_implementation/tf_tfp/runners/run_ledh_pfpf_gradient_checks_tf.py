"""Run TF/TFP LEDH-PF-PF-OT same-scalar GradientTape smoke check."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]

import argparse
import json
import sys
from pathlib import Path

import tensorflow as tf

from experiments.dpf_implementation.tf_tfp.filters.bootstrap_pf_tf import (
    normalize_log_weights_tf,
)
from experiments.dpf_implementation.tf_tfp.fixtures.lgssm_tf import (
    DTYPE,
    build_lgssm_fixture_tf,
    gaussian_logpdf_zero_mean_tf,
)
from experiments.dpf_implementation.tf_tfp.flows.jacobians_tf import (
    linear_observation_jacobian_tf,
)
from experiments.dpf_implementation.tf_tfp.flows.ledh_tf import ledh_flow_batch_tf
from experiments.dpf_implementation.tf_tfp.resampling.sinkhorn_tf import (
    sinkhorn_resample_tf,
)
from experiments.dpf_implementation.tf_tfp.runners.common_tf import (
    OUTPUT_DIR,
    REPORT_DIR,
    environment_manifest,
    scalar,
    stable_digest,
    utc_now,
    wall_time_call,
    write_json,
    write_text,
)


JSON_PATH = OUTPUT_DIR / "dpf_ledh_pfpf_ot_tf_tfp_gradient_check_2026-05-29.json"
REPORT_PATH = REPORT_DIR / "dpf-ledh-pfpf-ot-tf-tfp-gradient-check-result-2026-05-29.md"
SCALAR_ID = "lgssm_ledh_pfpf_ot_corrected_negative_log_normalizer_proxy_tf"
SEED = 464
NUM_PARTICLES = 32


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--check-reproducibility", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only:
        payload = _load_json_or_fail(JSON_PATH)
        _validate_payload(payload)
        return 0
    payload, runtime = wall_time_call(_run)
    payload["run_manifest"]["wall_time_seconds"] = runtime
    payload["reproducibility_digest"] = _digest_payload(payload)
    write_json(JSON_PATH, payload)
    write_text(REPORT_PATH, _markdown(payload))
    if args.check_reproducibility:
        second = _run()
        second["run_manifest"]["wall_time_seconds"] = payload["run_manifest"]["wall_time_seconds"]
        second["reproducibility_digest"] = _digest_payload(second)
        if second["reproducibility_digest"] != payload["reproducibility_digest"]:
            raise RuntimeError("LEDH-PF-PF-OT gradient reproducibility digest mismatch")
    _validate_payload(payload)
    print(payload["decision"])
    return 0


def _run() -> dict:
    fixture = build_lgssm_fixture_tf(horizon=6, fixture_generation_seed=2026052905)
    base_particles = tf.random.stateless_normal(
        [NUM_PARTICLES, fixture.state_dim],
        seed=tf.constant([SEED, 1], dtype=tf.int32),
        dtype=DTYPE,
    )
    transition_noise = tf.random.stateless_normal(
        [fixture.horizon, NUM_PARTICLES, fixture.state_dim],
        seed=tf.constant([SEED, 2], dtype=tf.int32),
        dtype=DTYPE,
    )
    theta = tf.Variable(1.0, dtype=DTYPE)
    with tf.GradientTape() as tape:
        value = _scalar_value(fixture, base_particles, transition_noise, theta)
    gradient = tape.gradient(value, theta)
    step = tf.constant(1e-3, dtype=DTYPE)
    plus = _scalar_value(fixture, base_particles, transition_noise, theta + step)
    minus = _scalar_value(fixture, base_particles, transition_noise, theta - step)
    finite_difference = (plus - minus) / (2.0 * step)
    abs_error = tf.abs(gradient - finite_difference)
    rel_error = abs_error / tf.maximum(tf.abs(finite_difference), tf.constant(1e-8, dtype=DTYPE))
    decision = "DPF_LEDH_PFPF_OT_TF_TFP_GRADIENT_CHECK_PASSED"
    if gradient is None or not bool(tf.math.is_finite(gradient).numpy()) or scalar(abs_error) > 1e-2:
        decision = "DPF_LEDH_PFPF_OT_TF_TFP_GRADIENT_CHECK_FAILED"
    return {
        "decision": decision,
        "question": "TF/TFP same-scalar GradientTape check for corrected LEDH-PF-PF-OT proxy",
        "created_at_utc": utc_now(),
        "backend": "tensorflow_tensorflow_probability",
        "scalar_id": SCALAR_ID,
        "same_scalar_contract": "GradientTape and finite difference both evaluate the LGSSM corrected LEDH-PF-PF-OT negative log-normalizer proxy with fixed observations and common random numbers.",
        "seed": SEED,
        "num_particles": NUM_PARTICLES,
        "theta": scalar(theta),
        "value": scalar(value),
        "gradient_tape": scalar(gradient),
        "finite_difference": scalar(finite_difference),
        "absolute_error": scalar(abs_error),
        "relative_error": scalar(rel_error),
        "tolerance": 1e-2,
        "run_manifest": environment_manifest(
            command="CUDA_VISIBLE_DEVICES=-1 python -m experiments.dpf_implementation.tf_tfp.runners.run_ledh_pfpf_gradient_checks_tf",
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "model_definition": fixture.model_definition(),
        "non_implications": _non_implications(),
    }


def _scalar_value(
    fixture,
    base_particles: tf.Tensor,
    transition_noise: tf.Tensor,
    theta: tf.Tensor,
) -> tf.Tensor:
    theta = tf.cast(theta, DTYPE)
    q_cov = theta * theta * fixture.Q
    particles = fixture.m0 + tf.linalg.matvec(tf.linalg.cholesky(fixture.P0), base_particles)
    log_weights = tf.fill([NUM_PARTICLES], -tf.math.log(tf.cast(NUM_PARTICLES, DTYPE)))
    log_normalizer = tf.constant(0.0, dtype=DTYPE)
    observation_jacobian = linear_observation_jacobian_tf(fixture.C)
    for t, observation in enumerate(tf.unstack(fixture.observations, axis=0)):
        ancestors = particles
        mean = tf.linalg.matmul(ancestors, fixture.A, transpose_b=True)
        noise = tf.linalg.matmul(transition_noise[t], tf.linalg.cholesky(fixture.Q), transpose_b=True)
        pre_flow = mean + theta * noise
        flow = ledh_flow_batch_tf(
            pre_flow_particles=pre_flow,
            ancestors=ancestors,
            observation=observation,
            transition_matrix=fixture.A,
            transition_covariance=q_cov,
            observation_covariance=fixture.R,
            observation_fn=lambda x: tf.linalg.matvec(fixture.C, x),
            observation_jacobian_fn=observation_jacobian,
            observation_residual_fn=lambda predicted, observed: tf.reshape(observed, [-1])
            - tf.reshape(predicted, [-1]),
        )
        post_flow = flow.post_flow_particles
        target_mean = tf.linalg.matmul(ancestors, fixture.A, transpose_b=True)
        target_transition = gaussian_logpdf_zero_mean_tf(post_flow - target_mean, q_cov)
        predicted = tf.linalg.matmul(post_flow, fixture.C, transpose_b=True)
        residual = tf.reshape(observation, [1, fixture.obs_dim]) - predicted
        target_observation = gaussian_logpdf_zero_mean_tf(residual, fixture.R)
        corrected = (
            log_weights
            + target_transition
            + target_observation
            - flow.pre_flow_log_density
            + flow.forward_log_det
        )
        weights, incremental = normalize_log_weights_tf(corrected)
        log_normalizer = log_normalizer + incremental
        resampled = sinkhorn_resample_tf(
            post_flow,
            weights,
            epsilon=0.75,
            max_iterations=60,
            tolerance=1e-7,
        )
        particles = resampled.particles
        log_weights = tf.fill([NUM_PARTICLES], -tf.math.log(tf.cast(NUM_PARTICLES, DTYPE)))
    return -log_normalizer


def _validate_payload(payload: dict) -> None:
    if payload["decision"] != "DPF_LEDH_PFPF_OT_TF_TFP_GRADIENT_CHECK_PASSED":
        raise RuntimeError(payload["decision"])
    if payload["run_manifest"]["pre_import_cuda_visible_devices"] != "-1":
        raise RuntimeError("missing CPU-only pre-import manifest")
    if payload["scalar_id"] != SCALAR_ID:
        raise RuntimeError("wrong scalar id")
    if "reproducibility_digest" not in payload:
        raise RuntimeError("missing reproducibility digest")


def _markdown(payload: dict) -> str:
    return f"""# TF/TFP LEDH-PF-PF-OT Gradient Check Result

## Decision

`{payload['decision']}`

## Decision Table

| Check | Status | Evidence |
| --- | --- | --- |
| scalar id | pass | `{payload['scalar_id']}` |
| same-scalar contract | pass | GradientTape and finite difference use fixed observations/common random numbers |
| GradientTape gradient | smoke | `{payload['gradient_tape']:.6f}` |
| finite-difference gradient | reference | `{payload['finite_difference']:.6f}` |
| absolute error | veto | `{payload['absolute_error']:.3e}` |

## Interpretation

The TF/TFP same-scalar GradientTape smoke check passed for the named corrected
LEDH-PF-PF-OT proxy scalar.  This is not posterior, HMC, or production
validation.

## Non-Implications

{_non_implications_markdown()}
"""


def _non_implications() -> list[str]:
    return [
        "No production readiness is concluded.",
        "No public API readiness is concluded.",
        "No HMC readiness is concluded.",
        "No posterior correctness is concluded.",
        "No likelihood-score validity is concluded beyond this named proxy scalar.",
        "No NAWM-scale readiness is concluded.",
        "No banking or model-risk claim is concluded.",
        "No monograph claim is concluded without separate review.",
    ]


def _non_implications_markdown() -> str:
    return "\n".join(f"- {item}" for item in _non_implications())


def _digest_payload(payload: dict) -> str:
    stable = {
        key: value
        for key, value in payload.items()
        if key not in {"created_at_utc", "run_manifest", "reproducibility_digest"}
    }
    return stable_digest(stable)


def _load_json_or_fail(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
