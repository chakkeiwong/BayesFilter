"""Run TF/TFP same-scalar smoke checks for retained-teacher Sinkhorn warm starts."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]

import argparse
import json
import sys
from pathlib import Path

import tensorflow as tf

from experiments.dpf_implementation.tf_tfp.filters.dpf_ot_tf import run_ot_dpf_tf
from experiments.dpf_implementation.tf_tfp.fixtures.lgssm_tf import (
    DTYPE,
    build_lgssm_fixture_tf,
    gaussian_logpdf_zero_mean_tf,
)
from experiments.dpf_implementation.tf_tfp.resampling.sinkhorn_tf import (
    SinkhornLogStateTF,
    canonicalize_sinkhorn_log_state_tf,
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


JSON_PATH = OUTPUT_DIR / "retained_teacher_sinkhorn_scalar_checks_2026-06-18.json"
REPORT_PATH = REPORT_DIR / "retained-teacher-sinkhorn-scalar-checks-2026-06-18.md"
SCALAR_ID = "lgssm_retained_teacher_sinkhorn_warmstart_negative_log_normalizer_proxy_tf"
SEED = 444
NUM_PARTICLES = 32
FIXTURE_HORIZON = 8
FIXTURE_GENERATION_SEED = 2026052805
THETA_VALUE = 1.0
FD_STEP = 1e-3
TOLERANCE = 5e-3
SINKHORN_EPSILON = 0.75
SINKHORN_ITERATIONS = 60
SINKHORN_TOLERANCE = 1e-7
ESS_THRESHOLD_RATIO = 1.1
EXPECTED_DECISION = "RETAINED_TEACHER_SINKHORN_SCALAR_CHECK_PASSED"


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
            raise RuntimeError("warm-start scalar reproducibility digest mismatch")
    _validate_payload(payload)
    print(payload["decision"])
    return 0


def _run() -> dict:
    fixture = build_lgssm_fixture_tf(
        horizon=FIXTURE_HORIZON,
        fixture_generation_seed=FIXTURE_GENERATION_SEED,
    )
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
    theta = tf.Variable(THETA_VALUE, dtype=DTYPE)
    with tf.GradientTape() as tape:
        value = _scalar_value(fixture, base_particles, transition_noise, theta)
    gradient = tape.gradient(value, theta)
    step = tf.constant(FD_STEP, dtype=DTYPE)
    plus = _scalar_value(fixture, base_particles, transition_noise, theta + step)
    minus = _scalar_value(fixture, base_particles, transition_noise, theta - step)
    finite_difference = (plus - minus) / (2.0 * step)
    abs_error = tf.abs(gradient - finite_difference)
    rel_error = abs_error / tf.maximum(tf.abs(finite_difference), tf.constant(1e-8, dtype=DTYPE))
    decision = EXPECTED_DECISION
    if not bool(tf.math.is_finite(gradient).numpy()) or scalar(abs_error) > TOLERANCE:
        decision = "RETAINED_TEACHER_SINKHORN_SCALAR_CHECK_FAILED"
    payload = {
        "decision": decision,
        "question": "Does the integrated retained-teacher Sinkhorn warm-start route satisfy a same-executed-scalar GradientTape smoke check on deterministic LGSSM data?",
        "created_at_utc": utc_now(),
        "backend": "tensorflow_tensorflow_probability",
        "scalar_id": SCALAR_ID,
        "same_scalar_contract": (
            "GradientTape and finite difference both evaluate the exact same "
            "negative log-normalizer proxy through run_ot_dpf_tf with the "
            "retained_teacher_sinkhorn_warmstart transport branch, fixed observations, "
            "common random numbers, and ESS threshold ratio 1.1 so the resampling branch is always taken."
        ),
        "seed": SEED,
        "num_particles": NUM_PARTICLES,
        "theta": scalar(theta),
        "value": scalar(value),
        "gradient_tape": scalar(gradient),
        "finite_difference": scalar(finite_difference),
        "absolute_error": scalar(abs_error),
        "relative_error": scalar(rel_error),
        "tolerance": TOLERANCE,
        "warmstart_policy": {
            "provider": "heuristic_weight_log_state_warmstart",
            "gauge_policy": "mean_log_u_zero",
            "depends_on_weights": True,
            "depends_on_particles": False,
            "learned_student_used": False,
        },
        "transport_settings": {
            "transport_method": "retained_teacher_sinkhorn_warmstart",
            "sinkhorn_epsilon": SINKHORN_EPSILON,
            "sinkhorn_iterations": SINKHORN_ITERATIONS,
            "sinkhorn_tolerance": SINKHORN_TOLERANCE,
            "ess_threshold_ratio": ESS_THRESHOLD_RATIO,
            "forced_resampling_policy": "ess_threshold_ratio_greater_than_one",
        },
        "run_manifest": environment_manifest(
            command="CUDA_VISIBLE_DEVICES=-1 python -m experiments.dpf_implementation.tf_tfp.runners.run_retained_teacher_sinkhorn_scalar_checks_tf",
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "model_definition": fixture.model_definition(),
        "non_implications": _non_implications(),
    }
    return payload


def _scalar_value(
    fixture,
    base_particles: tf.Tensor,
    transition_noise: tf.Tensor,
    theta: tf.Tensor,
) -> tf.Tensor:
    chol_p0 = tf.linalg.cholesky(fixture.P0)
    particles0 = fixture.m0 + tf.linalg.matvec(chol_p0, base_particles)

    def initial_sample(num_particles: int, seed: int) -> tf.Tensor:
        del num_particles, seed
        return particles0

    def transition_sample(previous_particles: tf.Tensor, seed: int, time_index: int) -> tf.Tensor:
        del seed
        mean = tf.linalg.matmul(previous_particles, fixture.A, transpose_b=True)
        noise = tf.linalg.matmul(transition_noise[time_index], tf.linalg.cholesky(fixture.Q), transpose_b=True)
        return mean + theta * noise

    def observation_log_density(particles: tf.Tensor, observation: tf.Tensor, time_index: int) -> tf.Tensor:
        del time_index
        predicted = tf.linalg.matmul(particles, fixture.C, transpose_b=True)
        residual = tf.reshape(observation, [1, fixture.obs_dim]) - predicted
        return gaussian_logpdf_zero_mean_tf(residual, fixture.R)

    def heuristic_weight_log_state_warmstart(
        particles: tf.Tensor,
        weights: tf.Tensor,
        epsilon: float,
        time_index: int,
    ) -> SinkhornLogStateTF:
        del particles, epsilon, time_index
        log_weights = tf.math.log(tf.maximum(tf.cast(weights, DTYPE), tf.constant(1e-300, dtype=DTYPE)))
        zeros = tf.zeros_like(log_weights)
        return canonicalize_sinkhorn_log_state_tf(log_weights, zeros)

    result = run_ot_dpf_tf(
        observations=fixture.observations,
        initial_sample=initial_sample,
        transition_sample=transition_sample,
        observation_log_density=observation_log_density,
        seed=SEED,
        num_particles=NUM_PARTICLES,
        ess_threshold_ratio=ESS_THRESHOLD_RATIO,
        sinkhorn_epsilon=SINKHORN_EPSILON,
        sinkhorn_iterations=SINKHORN_ITERATIONS,
        sinkhorn_tolerance=SINKHORN_TOLERANCE,
        transport_method="retained_teacher_sinkhorn_warmstart",
        retained_teacher_warmstart_fn=heuristic_weight_log_state_warmstart,
    )
    return -result.log_likelihood_estimate


def _validate_payload(payload: dict) -> None:
    if payload["decision"] != EXPECTED_DECISION:
        raise RuntimeError(payload["decision"])
    if payload["run_manifest"]["pre_import_cuda_visible_devices"] != "-1":
        raise RuntimeError("missing CPU-only pre-import manifest")
    if payload["scalar_id"] != SCALAR_ID:
        raise RuntimeError("wrong scalar id")
    if "reproducibility_digest" not in payload:
        raise RuntimeError("missing reproducibility digest")


def _markdown(payload: dict) -> str:
    return f"""# Retained-Teacher Sinkhorn Scalar Check Result

## Decision

`{payload['decision']}`

## Decision Table

| Check | Status | Evidence |
| --- | --- | --- |
| scalar id | pass | `{payload['scalar_id']}` |
| same-scalar contract | pass | integrated retained-teacher warm-start route |
| GradientTape gradient | smoke | `{payload['gradient_tape']:.6f}` |
| finite-difference gradient | reference | `{payload['finite_difference']:.6f}` |
| absolute error | veto | `{payload['absolute_error']:.3e}` |
| warm-start provider | pass | `{payload['warmstart_policy']['provider']}` |

## Interpretation

The integrated retained-teacher Sinkhorn warm-start route passed a same-executed-
scalar GradientTape smoke check on deterministic LGSSM data. This is local
numerical evidence about the executed scalar graph only. It is not posterior,
HMC, or production validation, and it does not constitute a learned-student
claim.

## Non-Implications

{_non_implications_markdown()}
"""


def _non_implications() -> list[str]:
    return [
        "No learned-student training claim is concluded.",
        "No posterior correctness is concluded.",
        "No HMC readiness is concluded.",
        "No production-readiness claim is concluded.",
        "No cross-model generalization claim is concluded.",
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
