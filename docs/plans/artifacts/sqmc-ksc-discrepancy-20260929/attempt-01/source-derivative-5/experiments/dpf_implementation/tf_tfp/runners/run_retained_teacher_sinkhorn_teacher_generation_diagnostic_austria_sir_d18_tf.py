"""Diagnose the first failing teacher-generation Sinkhorn event on Austria SIR d18 clouds."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import tensorflow as tf

from bayesfilter.highdim.models import zhao_cui_sir_austria_model
from experiments.dpf_implementation.tf_tfp.filters.bootstrap_pf_tf import normalize_log_weights_tf
from experiments.dpf_implementation.tf_tfp.resampling.sinkhorn_tf import sinkhorn_resample_tf
from experiments.dpf_implementation.tf_tfp.runners.common_tf import (
    OUTPUT_DIR,
    REPORT_DIR,
    REPO_ROOT,
    environment_manifest,
    scalar,
    stable_digest,
    tensor_to_json,
    utc_now,
    wall_time_call,
    write_json,
    write_text,
)


DTYPE = tf.float64
EMPTY_THETA = tf.zeros([0], dtype=DTYPE)
JSON_PATH = OUTPUT_DIR / "retained_teacher_sinkhorn_teacher_generation_diagnostic_austria_sir_d18_2026-07-03.json"
REPORT_PATH = REPORT_DIR / "retained-teacher-sinkhorn-teacher-generation-diagnostic-austria-sir-d18-2026-07-03.md"
RESULT_PATH = "docs/plans/bayesfilter-neural-ot-austria-sir-d18-retained-teacher-blocker-repair-result-2026-07-03.md"
PLAN_PATH = "docs/plans/bayesfilter-neural-ot-austria-sir-d18-retained-teacher-blocker-repair-amendment-2026-07-03.md"
SCALAR_ID = "retained_teacher_sinkhorn_teacher_generation_diagnostic_austria_sir_d18_tf"
TRAIN_SEEDS = (111, 222)
HELDOUT_SEEDS = (333, 444)
NUM_PARTICLES = 64
ESS_THRESHOLD_RATIO = 0.5
FINAL_TIME = 6
SIMULATION_SEED_OFFSET = 7100
NOMINAL_EPSILON = 0.75
NOMINAL_MAX_ITERATIONS = 90
NOMINAL_TOLERANCE = 1e-8
SCALE_ADAPTIVE_ITERATIONS = 500
SCALE_ADAPTIVE_TOLERANCE = 1e-6
EXPECTED_DECISION = "AUSTRIA_SIR_D18_TEACHER_GENERATION_SCALE_ADAPTIVE_EPSILON_REPAIR_CANDIDATE_IDENTIFIED"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only:
        payload = _load_json_or_fail(JSON_PATH)
        _validate_payload(payload)
        return 0
    payload, runtime = wall_time_call(_run)
    payload["run_manifest"]["wall_time_seconds"] = runtime
    payload["reproducibility_digest"] = _digest_payload(payload)
    markdown = _markdown(payload)
    write_json(JSON_PATH, payload)
    write_text(REPORT_PATH, markdown)
    write_text(REPO_ROOT / RESULT_PATH, markdown)
    _validate_payload(payload)
    print(payload["decision"])
    return 0


def _run() -> dict[str, Any]:
    model = zhao_cui_sir_austria_model()
    first_failure = _find_first_failure(model)
    if first_failure is None:
        decision = "AUSTRIA_SIR_D18_TEACHER_GENERATION_BLOCKER_NOT_REPRODUCED"
        payload = {
            "decision": decision,
            "question": "What is the first failing teacher-generation Sinkhorn event on Austria SIR d18, and does a bounded scale-adaptive epsilon probe rescue it?",
            "created_at_utc": utc_now(),
            "backend": "tensorflow_tensorflow_probability",
            "scalar_id": SCALAR_ID,
            "plan_path": PLAN_PATH,
            "model": model.manifest_payload(),
            "diagnostic": None,
            "run_manifest": environment_manifest(
                command="CUDA_VISIBLE_DEVICES=-1 python -m experiments.dpf_implementation.tf_tfp.runners.run_retained_teacher_sinkhorn_teacher_generation_diagnostic_austria_sir_d18_tf",
                pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
            ),
            "non_implications": _non_implications(),
        }
        return payload

    decision = EXPECTED_DECISION if first_failure["scale_adaptive_probe"]["finite"] else "AUSTRIA_SIR_D18_TEACHER_GENERATION_BLOCKER_PERSISTS_AFTER_SCALE_ADAPTIVE_PROBE"
    payload = {
        "decision": decision,
        "question": "What is the first failing teacher-generation Sinkhorn event on Austria SIR d18, and does a bounded scale-adaptive epsilon probe rescue it?",
        "created_at_utc": utc_now(),
        "backend": "tensorflow_tensorflow_probability",
        "scalar_id": SCALAR_ID,
        "plan_path": PLAN_PATH,
        "model": model.manifest_payload(),
        "diagnostic": first_failure,
        "run_manifest": environment_manifest(
            command="CUDA_VISIBLE_DEVICES=-1 python -m experiments.dpf_implementation.tf_tfp.runners.run_retained_teacher_sinkhorn_teacher_generation_diagnostic_austria_sir_d18_tf",
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_implications": _non_implications(),
    }
    return payload


def _find_first_failure(model) -> dict[str, Any] | None:
    for split, seeds in (("train", TRAIN_SEEDS), ("heldout", HELDOUT_SEEDS)):
        for seed in seeds:
            failure = _first_failure_for_seed(model, seed=seed, split=split)
            if failure is not None:
                return failure
    return None


def _first_failure_for_seed(model, *, seed: int, split: str) -> dict[str, Any] | None:
    _, observations = model.simulate(final_time=FINAL_TIME, seed=SIMULATION_SEED_OFFSET + int(seed))
    particles = _sample_initial_particles(model, seed=seed)
    log_weights = tf.fill([NUM_PARTICLES], -tf.math.log(tf.cast(NUM_PARTICLES, DTYPE)))
    for time_index, observation in enumerate(tf.unstack(observations, axis=0)):
        if time_index > 0:
            particles = _sample_transition_particles(model, particles, seed=seed, time_index=time_index)
        obs_log_weights = model.observation_log_density(EMPTY_THETA, particles, observation, time_index)
        weights, incremental = normalize_log_weights_tf(log_weights + obs_log_weights)
        ess = 1.0 / tf.reduce_sum(weights * weights)
        should_capture = bool((ess < ESS_THRESHOLD_RATIO * NUM_PARTICLES).numpy())
        if should_capture:
            pre_resampling_particles = tf.identity(particles)
            pre_resampling_weights = tf.identity(weights)
            try:
                sinkhorn_resample_tf(
                    pre_resampling_particles,
                    pre_resampling_weights,
                    epsilon=NOMINAL_EPSILON,
                    max_iterations=NOMINAL_MAX_ITERATIONS,
                    tolerance=NOMINAL_TOLERANCE,
                )
                particles = pre_resampling_particles
                log_weights = tf.fill([NUM_PARTICLES], -tf.math.log(tf.cast(NUM_PARTICLES, DTYPE)))
                continue
            except FloatingPointError as error:
                return _build_failure_payload(
                    model,
                    split=split,
                    seed=seed,
                    time_index=time_index,
                    observation=observation,
                    particles=pre_resampling_particles,
                    weights=pre_resampling_weights,
                    incremental=incremental,
                    ess=ess,
                    nominal_error=str(error),
                )
        log_weights = tf.math.log(tf.maximum(weights, tf.constant(1e-300, dtype=DTYPE)))
    return None


def _build_failure_payload(
    model,
    *,
    split: str,
    seed: int,
    time_index: int,
    observation: tf.Tensor,
    particles: tf.Tensor,
    weights: tf.Tensor,
    incremental: tf.Tensor,
    ess: tf.Tensor,
    nominal_error: str,
) -> dict[str, Any]:
    cost_matrix = _pairwise_squared_euclidean_cost(particles)
    cost_mean = scalar(tf.reduce_mean(cost_matrix))
    scale_adaptive_epsilon = cost_mean
    nominal = {
        "epsilon": NOMINAL_EPSILON,
        "max_iterations": NOMINAL_MAX_ITERATIONS,
        "tolerance": NOMINAL_TOLERANCE,
        "failure_message": nominal_error,
    }
    scale_adaptive_probe = {
        "epsilon": scale_adaptive_epsilon,
        "max_iterations": SCALE_ADAPTIVE_ITERATIONS,
        "tolerance": SCALE_ADAPTIVE_TOLERANCE,
        "finite": False,
    }
    try:
        repaired = sinkhorn_resample_tf(
            particles,
            weights,
            epsilon=scale_adaptive_epsilon,
            max_iterations=SCALE_ADAPTIVE_ITERATIONS,
            tolerance=SCALE_ADAPTIVE_TOLERANCE,
        )
        scale_adaptive_probe.update(
            {
                "finite": True,
                "max_row_residual": float(repaired.diagnostics["max_row_residual"]),
                "max_column_residual": float(repaired.diagnostics["max_column_residual"]),
                "total_mass_residual": float(repaired.diagnostics["total_mass_residual"]),
                "iterations_used": int(repaired.diagnostics["iterations_used"]),
            }
        )
    except FloatingPointError as error:
        scale_adaptive_probe["failure_message"] = str(error)

    return {
        "split": split,
        "seed": int(seed),
        "time_index": int(time_index),
        "incremental_log_normalizer": scalar(incremental),
        "ess": scalar(ess),
        "ess_ratio": scalar(ess / tf.cast(NUM_PARTICLES, DTYPE)),
        "source_weight_min": float(tf.reduce_min(weights).numpy()),
        "source_weight_max": float(tf.reduce_max(weights).numpy()),
        "source_weight_perplexity": _perplexity(weights),
        "state_min": float(tf.reduce_min(particles).numpy()),
        "state_max": float(tf.reduce_max(particles).numpy()),
        "state_rms": scalar(tf.sqrt(tf.reduce_mean(tf.square(particles)))),
        "cost_mean": cost_mean,
        "cost_max": float(tf.reduce_max(cost_matrix).numpy()),
        "cost_max_over_nominal_epsilon": float(tf.reduce_max(cost_matrix).numpy() / NOMINAL_EPSILON),
        "observation": tensor_to_json(observation),
        "particles_checksum": stable_digest(tensor_to_json(particles)),
        "weights_checksum": stable_digest(tensor_to_json(weights)),
        "nominal": nominal,
        "scale_adaptive_probe": scale_adaptive_probe,
    }


def _sample_initial_particles(model, *, seed: int) -> tf.Tensor:
    chol = tf.linalg.cholesky(model.initial_covariance)
    noise = tf.random.stateless_normal([NUM_PARTICLES, model.state_dim()], seed=_seed_pair(seed, 17), dtype=DTYPE)
    return model.initial_mean[tf.newaxis, :] + tf.linalg.matmul(noise, chol, transpose_b=True)


def _sample_transition_particles(model, particles: tf.Tensor, *, seed: int, time_index: int) -> tf.Tensor:
    noise = tf.random.stateless_normal(
        [tf.shape(particles)[0], model.state_dim()],
        seed=_seed_pair(seed, 2000 + int(time_index)),
        dtype=DTYPE,
    )
    return model.transition_push_from_standard_normal(EMPTY_THETA, particles, noise, time_index)


def _seed_pair(seed: int, salt: int) -> tf.Tensor:
    return tf.constant([int(seed) % 2147483647, int(salt) % 2147483647], dtype=tf.int32)


def _pairwise_squared_euclidean_cost(x: tf.Tensor) -> tf.Tensor:
    values = tf.cast(x, DTYPE)
    diff = values[:, None, :] - values[None, :, :]
    return tf.reduce_sum(diff * diff, axis=-1)


def _perplexity(weights: tf.Tensor) -> float:
    safe = tf.maximum(tf.cast(weights, DTYPE), tf.constant(1e-300, dtype=DTYPE))
    entropy = -tf.reduce_sum(safe * tf.math.log(safe))
    return float(tf.exp(entropy).numpy())


def _validate_payload(payload: dict[str, Any]) -> None:
    if payload["decision"] not in {
        EXPECTED_DECISION,
        "AUSTRIA_SIR_D18_TEACHER_GENERATION_BLOCKER_PERSISTS_AFTER_SCALE_ADAPTIVE_PROBE",
        "AUSTRIA_SIR_D18_TEACHER_GENERATION_BLOCKER_NOT_REPRODUCED",
    }:
        raise RuntimeError(payload["decision"])
    if payload["run_manifest"]["pre_import_cuda_visible_devices"] != "-1":
        raise RuntimeError("missing CPU-only pre-import manifest")
    if payload["scalar_id"] != SCALAR_ID:
        raise RuntimeError("wrong scalar id")
    if payload["decision"] == EXPECTED_DECISION:
        if payload["diagnostic"] is None or not payload["diagnostic"]["scale_adaptive_probe"]["finite"]:
            raise RuntimeError("missing successful scale-adaptive diagnostic")
    if "reproducibility_digest" not in payload:
        raise RuntimeError("missing reproducibility digest")


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Austria SIR d18 Retained-Teacher Blocker-Repair Diagnostic Result",
        "",
        "## Decision",
        "",
        f"`{payload['decision']}`",
    ]
    diagnostic = payload.get("diagnostic")
    if diagnostic is None:
        lines.extend(
            [
                "",
                "## Interpretation",
                "",
                "The nominal teacher-generation blocker did not reproduce during the diagnostic run. The next justified action is to rerun the teacher-data generator once under the same nominal settings.",
            ]
        )
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            "",
            "## First failing event",
            "",
            f"- Split / seed / time index: `{diagnostic['split']}` / `{diagnostic['seed']}` / `{diagnostic['time_index']}`",
            f"- ESS ratio: `{diagnostic['ess_ratio']:.6f}`",
            f"- Source weight min/max: `{diagnostic['source_weight_min']:.3e}` / `{diagnostic['source_weight_max']:.3e}`",
            f"- Source weight perplexity: `{diagnostic['source_weight_perplexity']:.6f}`",
            f"- State min/max/RMS: `{diagnostic['state_min']:.3e}` / `{diagnostic['state_max']:.3e}` / `{diagnostic['state_rms']:.3e}`",
            f"- Cost mean/max: `{diagnostic['cost_mean']:.6f}` / `{diagnostic['cost_max']:.6f}`",
            f"- Cost max / nominal epsilon: `{diagnostic['cost_max_over_nominal_epsilon']:.6f}`",
            f"- Nominal failure: `{diagnostic['nominal']['failure_message']}`",
            "",
            "## Bounded repair probe",
            "",
            f"- Scale-adaptive epsilon: `{diagnostic['scale_adaptive_probe']['epsilon']:.6f}`",
            f"- Iterations: `{diagnostic['scale_adaptive_probe']['max_iterations']}`",
            f"- Tolerance: `{diagnostic['scale_adaptive_probe']['tolerance']:.1e}`",
            f"- Finite: `{diagnostic['scale_adaptive_probe']['finite']}`",
        ]
    )
    if diagnostic["scale_adaptive_probe"]["finite"]:
        lines.extend(
            [
                f"- Max row residual: `{diagnostic['scale_adaptive_probe']['max_row_residual']:.3e}`",
                f"- Max column residual: `{diagnostic['scale_adaptive_probe']['max_column_residual']:.3e}`",
                f"- Total mass residual: `{diagnostic['scale_adaptive_probe']['total_mass_residual']:.3e}`",
                f"- Iterations used: `{diagnostic['scale_adaptive_probe']['iterations_used']}`",
            ]
        )
    else:
        lines.append(f"- Failure: `{diagnostic['scale_adaptive_probe'].get('failure_message', 'unknown')}``")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            _interpretation_text(payload),
            "",
            "## Non-Implications",
            "",
            _non_implications_markdown(),
        ]
    )
    return "\n".join(lines) + "\n"


def _interpretation_text(payload: dict[str, Any]) -> str:
    if payload["decision"] == EXPECTED_DECISION:
        return (
            "The nominal Austria SIR d18 teacher-generation blocker is reproducible, and a bounded scale-adaptive epsilon probe succeeds on the first failing event. The next justified action is to rerun the teacher-data generator under that exact reviewed repair and stop there."
        )
    if payload["decision"] == "AUSTRIA_SIR_D18_TEACHER_GENERATION_BLOCKER_PERSISTS_AFTER_SCALE_ADAPTIVE_PROBE":
        return (
            "The nominal Austria SIR d18 teacher-generation blocker is reproducible, and the first bounded scale-adaptive epsilon probe does not rescue it. The blocker should therefore be preserved until a further reviewed repair amendment selects the next single repair candidate."
        )
    return (
        "The nominal Austria SIR d18 teacher-generation blocker did not reproduce during the diagnostic pass, so the original teacher-data runner should be rerun once before any repair is adopted."
    )


def _non_implications() -> list[str]:
    return [
        "No donor-aligned student usefulness claim is concluded.",
        "No large-particle or N=10000 claim is concluded.",
        "No GPU scaling claim is concluded.",
        "No production-readiness claim is concluded.",
    ]


def _non_implications_markdown() -> str:
    return "\n".join(f"- {item}" for item in _non_implications())


def _digest_payload(payload: dict[str, Any]) -> str:
    stable = {
        key: value
        for key, value in payload.items()
        if key not in {"created_at_utc", "run_manifest", "reproducibility_digest"}
    }
    return stable_digest(stable)


def _load_json_or_fail(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
