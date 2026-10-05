"""Generate retained-teacher Sinkhorn teacher data on Austria SIR d18 clouds."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]

import argparse
import json
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
JSON_PATH = OUTPUT_DIR / "retained_teacher_sinkhorn_teacher_data_austria_sir_d18_2026-07-03.json"
REPORT_PATH = REPORT_DIR / "retained-teacher-sinkhorn-teacher-data-austria-sir-d18-2026-07-03.md"
PLAN_PATH = "docs/plans/bayesfilter-neural-ot-austria-sir-d18-retained-teacher-smoke-test-plan-2026-07-03.md"
RESULT_PATH = "docs/plans/bayesfilter-neural-ot-austria-sir-d18-retained-teacher-teacher-data-result-2026-07-03.md"
TRAIN_SEEDS = (111, 222)
HELDOUT_SEEDS = (333, 444)
NUM_PARTICLES = 64
ESS_THRESHOLD_RATIO = 0.5
TEACHER_EPSILON_POLICY = "cost_mean"
TEACHER_MAX_ITERATIONS = 500
TEACHER_TOLERANCE = 1e-6
FINAL_TIME = 6
SIMULATION_SEED_OFFSET = 7100
EXPECTED_DECISION = "RETAINED_TEACHER_SINKHORN_TEACHER_DATA_AUSTRIA_SIR_D18_READY"


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
    examples = []
    for seed in TRAIN_SEEDS:
        examples.extend(_capture_examples_for_seed(model, seed=seed, split="train"))
    for seed in HELDOUT_SEEDS:
        examples.extend(_capture_examples_for_seed(model, seed=seed, split="heldout"))

    decision = EXPECTED_DECISION if _all_examples_finite(examples) and examples else "RETAINED_TEACHER_SINKHORN_TEACHER_DATA_AUSTRIA_SIR_D18_FAILED"
    split_counts = _split_counts(examples)
    max_teacher_residual = _max_teacher_residual(examples)
    max_abs_canonical_mean = _max_abs_canonical_mean(examples)
    payload = {
        "decision": decision,
        "question": "Can BayesFilter generate reproducible retained-teacher Sinkhorn latent targets on Austria SIR d18 weighted clouds?",
        "created_at_utc": utc_now(),
        "backend": "tensorflow_tensorflow_probability",
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "artifact_paths": {
            "json": str(JSON_PATH.relative_to(REPO_ROOT)),
            "markdown_report": str(REPORT_PATH.relative_to(REPO_ROOT)),
            "phase_result": RESULT_PATH,
        },
        "teacher_generation_policy": {
            "capture_rule": "actual_resampling_trigger_clouds_only",
            "train_seeds": list(TRAIN_SEEDS),
            "heldout_seeds": list(HELDOUT_SEEDS),
            "num_particles": NUM_PARTICLES,
            "ess_threshold_ratio": ESS_THRESHOLD_RATIO,
            "teacher_epsilon_policy": TEACHER_EPSILON_POLICY,
            "teacher_max_iterations": TEACHER_MAX_ITERATIONS,
            "teacher_tolerance": TEACHER_TOLERANCE,
            "final_time": FINAL_TIME,
            "teacher_transport_method": "fixed_target_sinkhorn_local_comparator_tf",
            "latent_object": "canonicalized_log_domain_sinkhorn_state",
            "latent_gauge_policy": "mean_log_u_zero",
            "meta_ot_refit_target_half": "canonical_log_u",
            "meta_ot_refit_complementary_recovery": "teacher_side_sinkhorn_update",
            "route_family": "meta_ot_aligned_fixed_target_retained_sinkhorn_refit",
            "highdim_target_id": "zhao_cui_sir_austria_d18",
        },
        "model": model.manifest_payload(),
        "counts": {
            "total_examples": len(examples),
            **split_counts,
        },
        "summary": {
            "all_examples_finite": _all_examples_finite(examples),
            "max_teacher_residual": max_teacher_residual,
            "max_abs_canonical_mean_log_u": max_abs_canonical_mean,
            "dataset_checksum": stable_digest(examples),
        },
        "examples": examples,
        "run_manifest": environment_manifest(
            command="CUDA_VISIBLE_DEVICES=-1 python -m experiments.dpf_implementation.tf_tfp.runners.run_retained_teacher_sinkhorn_teacher_data_austria_sir_d18_tf",
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_implications": _non_implications(),
    }
    return payload


def _capture_examples_for_seed(model, *, seed: int, split: str) -> list[dict[str, Any]]:
    _, observations = model.simulate(final_time=FINAL_TIME, seed=SIMULATION_SEED_OFFSET + int(seed))
    particles = _sample_initial_particles(model, seed=seed)
    log_weights = tf.fill([NUM_PARTICLES], -tf.math.log(tf.cast(NUM_PARTICLES, DTYPE)))
    examples: list[dict[str, Any]] = []
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
            teacher_epsilon = _teacher_epsilon(pre_resampling_particles)
            teacher = sinkhorn_resample_tf(
                pre_resampling_particles,
                pre_resampling_weights,
                epsilon=teacher_epsilon,
                max_iterations=TEACHER_MAX_ITERATIONS,
                tolerance=TEACHER_TOLERANCE,
            )
            example = {
                "split": split,
                "seed": int(seed),
                "time_index": int(time_index),
                "incremental_log_normalizer": scalar(incremental),
                "ess": scalar(ess),
                "ess_ratio": scalar(ess / tf.cast(NUM_PARTICLES, DTYPE)),
                "observation": tensor_to_json(observation),
                "particles_checksum": stable_digest(tensor_to_json(pre_resampling_particles)),
                "weights_checksum": stable_digest(tensor_to_json(pre_resampling_weights)),
                "particles": tensor_to_json(pre_resampling_particles),
                "weights": tensor_to_json(pre_resampling_weights),
                "teacher": {
                    "teacher_epsilon": float(teacher_epsilon),
                    "teacher_max_iterations": int(TEACHER_MAX_ITERATIONS),
                    "teacher_tolerance": float(TEACHER_TOLERANCE),
                    "barycentric_checksum": stable_digest(tensor_to_json(teacher.particles)),
                    "barycentric_particles": tensor_to_json(teacher.particles),
                    "canonical_log_u": tensor_to_json(teacher.canonicalized_final_state.log_u),
                    "canonical_log_v": tensor_to_json(teacher.canonicalized_final_state.log_v),
                    "canonical_gauge_policy": teacher.canonicalized_final_state.gauge_policy,
                    "canonical_mean_log_u": scalar(tf.reduce_mean(teacher.canonicalized_final_state.log_u)),
                    "diagnostics": dict(teacher.diagnostics),
                },
            }
            examples.append(example)
            particles = teacher.particles
            log_weights = tf.fill([NUM_PARTICLES], -tf.math.log(tf.cast(NUM_PARTICLES, DTYPE)))
        else:
            log_weights = tf.math.log(tf.maximum(weights, tf.constant(1e-300, dtype=DTYPE)))
    return examples


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


def _teacher_epsilon(particles: tf.Tensor) -> float:
    values = tf.cast(particles, DTYPE)
    diff = values[:, None, :] - values[None, :, :]
    cost_matrix = tf.reduce_sum(diff * diff, axis=-1)
    return scalar(tf.reduce_mean(cost_matrix))


def _split_counts(examples: list[dict[str, Any]]) -> dict[str, int]:
    train_count = sum(1 for example in examples if example["split"] == "train")
    heldout_count = sum(1 for example in examples if example["split"] == "heldout")
    return {
        "train_examples": train_count,
        "heldout_examples": heldout_count,
    }


def _max_teacher_residual(examples: list[dict[str, Any]]) -> float:
    residuals = []
    for example in examples:
        diagnostics = example["teacher"]["diagnostics"]
        residuals.extend(
            [
                float(diagnostics["max_row_residual"]),
                float(diagnostics["max_column_residual"]),
                float(diagnostics["total_mass_residual"]),
            ]
        )
    return max(residuals) if residuals else 0.0


def _max_abs_canonical_mean(examples: list[dict[str, Any]]) -> float:
    means = [abs(float(example["teacher"]["canonical_mean_log_u"])) for example in examples]
    return max(means) if means else 0.0


def _all_examples_finite(examples: list[dict[str, Any]]) -> bool:
    for example in examples:
        diagnostics = example["teacher"]["diagnostics"]
        if not diagnostics["finite_coupling"] or not diagnostics["finite_particles"]:
            return False
    return True


def _validate_payload(payload: dict[str, Any]) -> None:
    if payload["decision"] != EXPECTED_DECISION:
        raise RuntimeError(payload["decision"])
    if payload["run_manifest"]["pre_import_cuda_visible_devices"] != "-1":
        raise RuntimeError("missing CPU-only pre-import manifest")
    if payload["model"]["state_dimension"] != 18:
        raise RuntimeError("Austria SIR smoke test must use d18 state")
    if payload["model"]["observation_dimension"] != 9:
        raise RuntimeError("Austria SIR smoke test must use 9-dimensional observations")
    if payload["counts"]["heldout_examples"] <= 0:
        raise RuntimeError("teacher-data payload captured no heldout examples")
    if payload["summary"]["max_abs_canonical_mean_log_u"] > 1e-10:
        raise RuntimeError("canonicalized log_u mean exceeded tolerance")
    if "reproducibility_digest" not in payload:
        raise RuntimeError("missing reproducibility digest")


def _markdown(payload: dict[str, Any]) -> str:
    return f"""# Austria SIR d18 Retained-Teacher Teacher-Data Result

## Decision

`{payload['decision']}`

## Decision Table

| Check | Status | Evidence |
| --- | --- | --- |
| CPU-only manifest | pass | `pre_import_cuda_visible_devices={payload['run_manifest']['pre_import_cuda_visible_devices']}` |
| state / observation dimensions | pass | `{payload['model']['state_dimension']}` / `{payload['model']['observation_dimension']}` |
| total examples | pass | `{payload['counts']['total_examples']}` |
| train / heldout split | pass | `{payload['counts']['train_examples']}` / `{payload['counts']['heldout_examples']}` |
| max teacher residual | pass | `{payload['summary']['max_teacher_residual']:.3e}` |
| dataset checksum | pass | `{payload['summary']['dataset_checksum']}` |

## Interpretation

This artifact establishes the first retained-teacher Sinkhorn teacher-data payload on the fixed Austria SIR d18 high-dimensional target. It is an interface artifact only; it does not yet conclude local usefulness or scaling benefit.

## Non-Implications

{_non_implications_markdown()}
"""


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
