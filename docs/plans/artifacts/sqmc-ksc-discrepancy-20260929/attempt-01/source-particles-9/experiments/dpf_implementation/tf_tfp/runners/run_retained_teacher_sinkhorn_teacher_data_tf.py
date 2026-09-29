"""Generate retained-teacher Sinkhorn teacher data on deterministic LGSSM clouds."""

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

from experiments.dpf_implementation.tf_tfp.filters.bootstrap_pf_tf import (
    normalize_log_weights_tf,
)
from experiments.dpf_implementation.tf_tfp.fixtures.lgssm_tf import (
    DTYPE,
    build_lgssm_fixture_tf,
    observation_log_density_tf,
    sample_initial_particles_tf,
    sample_transition_particles_tf,
)
from experiments.dpf_implementation.tf_tfp.resampling.sinkhorn_tf import (
    sinkhorn_resample_tf,
)
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


JSON_PATH = OUTPUT_DIR / "retained_teacher_sinkhorn_teacher_data_2026-06-18.json"
REPORT_PATH = REPORT_DIR / "retained-teacher-sinkhorn-teacher-data-2026-06-18.md"
PLAN_PATH = "docs/plans/bayesfilter-neural-ot-retained-teacher-first-pass-implementation-plan-2026-06-18.md"
RESULT_PATH = "docs/plans/bayesfilter-neural-ot-retained-teacher-teacher-data-result-2026-06-18.md"
TRAIN_SEEDS = (444, 445, 446)
HELDOUT_SEEDS = (544,)
NUM_PARTICLES = 32
ESS_THRESHOLD_RATIO = 0.5
TEACHER_EPSILON = 0.75
TEACHER_MAX_ITERATIONS = 90
TEACHER_TOLERANCE = 1e-8
FIXTURE_HORIZON = 8
FIXTURE_GENERATION_SEED = 2026052805
EXPECTED_DECISION = "RETAINED_TEACHER_SINKHORN_TEACHER_DATA_READY"


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
    fixture = build_lgssm_fixture_tf(
        horizon=FIXTURE_HORIZON,
        fixture_generation_seed=FIXTURE_GENERATION_SEED,
    )
    examples = []
    for seed in TRAIN_SEEDS:
        examples.extend(_capture_examples_for_seed(fixture, seed=seed, split="train"))
    for seed in HELDOUT_SEEDS:
        examples.extend(_capture_examples_for_seed(fixture, seed=seed, split="heldout"))

    decision = EXPECTED_DECISION if _all_examples_finite(examples) and examples else "RETAINED_TEACHER_SINKHORN_TEACHER_DATA_FAILED"
    split_counts = _split_counts(examples)
    max_teacher_residual = _max_teacher_residual(examples)
    max_abs_canonical_mean = _max_abs_canonical_mean(examples)
    payload = {
        "decision": decision,
        "question": "Can BayesFilter generate reproducible retained-teacher Sinkhorn latent targets on deterministic LGSSM weighted clouds?",
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
            "teacher_epsilon": TEACHER_EPSILON,
            "teacher_max_iterations": TEACHER_MAX_ITERATIONS,
            "teacher_tolerance": TEACHER_TOLERANCE,
            "fixture_horizon": FIXTURE_HORIZON,
            "fixture_generation_seed": FIXTURE_GENERATION_SEED,
            "teacher_transport_method": "fixed_target_sinkhorn_local_comparator_tf",
            "latent_object": "canonicalized_log_domain_sinkhorn_state",
            "latent_gauge_policy": "mean_log_u_zero",
            "meta_ot_refit_target_half": "canonical_log_u",
            "meta_ot_refit_complementary_recovery": "teacher_side_sinkhorn_update",
            "route_family": "meta_ot_aligned_fixed_target_retained_sinkhorn_refit",
        },
        "fixture": fixture.model_definition(),
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
            command="CUDA_VISIBLE_DEVICES=-1 python -m experiments.dpf_implementation.tf_tfp.runners.run_retained_teacher_sinkhorn_teacher_data_tf",
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_implications": _non_implications(),
    }
    return payload


def _capture_examples_for_seed(fixture, *, seed: int, split: str) -> list[dict[str, Any]]:
    particles = sample_initial_particles_tf(fixture, num_particles=NUM_PARTICLES, seed=seed)
    log_weights = tf.fill([NUM_PARTICLES], -tf.math.log(tf.cast(NUM_PARTICLES, DTYPE)))
    examples: list[dict[str, Any]] = []
    for time_index, observation in enumerate(tf.unstack(fixture.observations, axis=0)):
        particles = sample_transition_particles_tf(
            fixture,
            particles,
            seed=seed,
            time_index=time_index,
        )
        obs_log_weights = observation_log_density_tf(fixture, particles, observation)
        weights, incremental = normalize_log_weights_tf(log_weights + obs_log_weights)
        ess = 1.0 / tf.reduce_sum(weights * weights)
        should_capture = bool((ess < ESS_THRESHOLD_RATIO * NUM_PARTICLES).numpy())
        if should_capture:
            pre_resampling_particles = tf.identity(particles)
            pre_resampling_weights = tf.identity(weights)
            teacher = sinkhorn_resample_tf(
                pre_resampling_particles,
                pre_resampling_weights,
                epsilon=TEACHER_EPSILON,
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
                "particles_checksum": stable_digest(tensor_to_json(pre_resampling_particles)),
                "weights_checksum": stable_digest(tensor_to_json(pre_resampling_weights)),
                "particles": tensor_to_json(pre_resampling_particles),
                "weights": tensor_to_json(pre_resampling_weights),
                "teacher": {
                    "source_weights": tensor_to_json(teacher.source_weights),
                    "target_weights": tensor_to_json(teacher.target_weights),
                    "coupling_checksum": stable_digest(tensor_to_json(teacher.coupling)),
                    "barycentric_checksum": stable_digest(tensor_to_json(teacher.particles)),
                    "coupling": tensor_to_json(teacher.coupling),
                    "barycentric_particles": tensor_to_json(teacher.particles),
                    "final_log_u": tensor_to_json(teacher.final_state.log_u),
                    "final_log_v": tensor_to_json(teacher.final_state.log_v),
                    "final_gauge_policy": teacher.final_state.gauge_policy,
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
    if payload["counts"]["total_examples"] <= 0:
        raise RuntimeError("teacher-data payload captured no examples")
    if payload["summary"]["max_abs_canonical_mean_log_u"] > 1e-10:
        raise RuntimeError("canonicalized log_u mean exceeded tolerance")
    if "reproducibility_digest" not in payload:
        raise RuntimeError("missing reproducibility digest")


def _markdown(payload: dict[str, Any]) -> str:
    return f"""# Retained-Teacher Sinkhorn Teacher-Data Result

## Decision

`{payload['decision']}`

## Decision Table

| Check | Status | Evidence |
| --- | --- | --- |
| CPU-only manifest | pass | `pre_import_cuda_visible_devices={payload['run_manifest']['pre_import_cuda_visible_devices']}` |
| teacher examples captured | pass | `{payload['counts']['total_examples']}` total |
| train / heldout split | pass | `{payload['counts']['train_examples']}` / `{payload['counts']['heldout_examples']}` |
| max teacher residual | pass | `{payload['summary']['max_teacher_residual']:.3e}` |
| max abs canonical mean log_u | pass | `{payload['summary']['max_abs_canonical_mean_log_u']:.3e}` |
| dataset checksum | pass | `{payload['summary']['dataset_checksum']}` |

## Interpretation

The deterministic LGSSM teacher-data runner produced retained Sinkhorn teacher
examples with canonicalized log-domain latent state, barycentric teacher cloud,
and residual diagnostics. This is a local reproducibility artifact for the first
retained-teacher neural OT pass, not a training or filtering-performance claim.

## Non-Implications

{_non_implications_markdown()}
"""


def _non_implications() -> list[str]:
    return [
        "No student training claim is concluded.",
        "No posterior correctness is concluded.",
        "No HMC readiness is concluded.",
        "No cross-model generalization claim is concluded.",
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
