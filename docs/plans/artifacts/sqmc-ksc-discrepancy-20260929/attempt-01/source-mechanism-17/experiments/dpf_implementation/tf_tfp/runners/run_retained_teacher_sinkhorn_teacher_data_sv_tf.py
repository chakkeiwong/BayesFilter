"""Generate retained-teacher Sinkhorn teacher data on stochastic-volatility clouds."""

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

from experiments.dpf_implementation.tf_tfp.filters.bootstrap_pf_tf import normalize_log_weights_tf
from experiments.dpf_implementation.tf_tfp.fixtures.stochastic_volatility_tf import (
    DTYPE,
    build_stochastic_volatility_fixture_tf,
    sv_observation_log_density_tf,
    sv_transition_mean_tf,
)
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


JSON_PATH = OUTPUT_DIR / "retained_teacher_sinkhorn_teacher_data_sv_2026-06-18.json"
REPORT_PATH = REPORT_DIR / "retained-teacher-sinkhorn-teacher-data-sv-2026-06-18.md"
RESULT_PATH = "docs/plans/bayesfilter-neural-ot-retained-teacher-teacher-data-sv-result-2026-06-18.md"
PLAN_PATH = "docs/plans/bayesfilter-neural-ot-retained-teacher-sv-cross-envelope-plan-2026-06-18.md"
TRAIN_SEEDS = (111, 222, 333, 444)
HELDOUT_SEEDS = (555, 666)
NUM_PARTICLES = 24
ESS_THRESHOLD_RATIO = 0.5
TEACHER_EPSILON = 0.75
TEACHER_MAX_ITERATIONS = 90
TEACHER_TOLERANCE = 1e-8
FIXTURE_HORIZON = 18
FIXTURE_GENERATION_SEED = 2026052901
EXPECTED_DECISION = "RETAINED_TEACHER_SINKHORN_TEACHER_DATA_SV_READY"


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
    fixture = build_stochastic_volatility_fixture_tf(
        horizon=FIXTURE_HORIZON,
        fixture_generation_seed=FIXTURE_GENERATION_SEED,
    )
    examples = []
    for seed in TRAIN_SEEDS:
        examples.extend(_capture_examples_for_seed(fixture, seed=seed, split="train"))
    for seed in HELDOUT_SEEDS:
        examples.extend(_capture_examples_for_seed(fixture, seed=seed, split="heldout"))
    decision = EXPECTED_DECISION if _all_examples_finite(examples) and examples else "RETAINED_TEACHER_SINKHORN_TEACHER_DATA_SV_FAILED"
    payload = {
        "decision": decision,
        "question": "Can BayesFilter generate a retained-teacher Sinkhorn latent dataset on deterministic stochastic-volatility weighted clouds?",
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
            "train_examples": sum(1 for example in examples if example["split"] == "train"),
            "heldout_examples": sum(1 for example in examples if example["split"] == "heldout"),
        },
        "summary": {
            "all_examples_finite": _all_examples_finite(examples),
            "dataset_checksum": stable_digest(examples),
        },
        "examples": examples,
        "run_manifest": environment_manifest(
            command="CUDA_VISIBLE_DEVICES=-1 python -m experiments.dpf_implementation.tf_tfp.runners.run_retained_teacher_sinkhorn_teacher_data_sv_tf",
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_implications": _non_implications(),
    }
    return payload


def _capture_examples_for_seed(fixture, *, seed: int, split: str) -> list[dict[str, Any]]:
    base = tf.random.stateless_normal([NUM_PARTICLES], seed=tf.constant([seed, 1], dtype=tf.int32), dtype=DTYPE)
    noise = tf.random.stateless_normal([fixture.horizon, NUM_PARTICLES], seed=tf.constant([seed, 2], dtype=tf.int32), dtype=DTYPE)
    particles = fixture.h0_mean + tf.sqrt(fixture.h0_variance) * base
    log_weights = tf.fill([NUM_PARTICLES], -tf.math.log(tf.cast(NUM_PARTICLES, DTYPE)))
    examples: list[dict[str, Any]] = []
    for time_index, observation in enumerate(tf.unstack(fixture.observations, axis=0)):
        particles = sv_transition_mean_tf(particles, mu=fixture.mu, phi=fixture.phi) + fixture.sigma * noise[time_index]
        obs_log_weights = sv_observation_log_density_tf(particles, observation)
        weights, incremental = normalize_log_weights_tf(log_weights + obs_log_weights)
        ess = 1.0 / tf.reduce_sum(weights * weights)
        should_capture = bool((ess < ESS_THRESHOLD_RATIO * NUM_PARTICLES).numpy())
        if should_capture:
            pre_resampling_particles = tf.identity(particles)
            pre_resampling_weights = tf.identity(weights)
            teacher = sinkhorn_resample_tf(
                pre_resampling_particles[:, None],
                pre_resampling_weights,
                epsilon=TEACHER_EPSILON,
                max_iterations=TEACHER_MAX_ITERATIONS,
                tolerance=TEACHER_TOLERANCE,
            )
            examples.append(
                {
                    "split": split,
                    "seed": int(seed),
                    "time_index": int(time_index),
                    "incremental_log_normalizer": scalar(incremental),
                    "ess": scalar(ess),
                    "ess_ratio": scalar(ess / tf.cast(NUM_PARTICLES, DTYPE)),
                    "particles": tensor_to_json(pre_resampling_particles[:, None]),
                    "weights": tensor_to_json(pre_resampling_weights),
                    "teacher": {
                        "barycentric_particles": tensor_to_json(teacher.particles),
                        "canonical_log_u": tensor_to_json(teacher.canonicalized_final_state.log_u),
                        "canonical_log_v": tensor_to_json(teacher.canonicalized_final_state.log_v),
                        "canonical_gauge_policy": teacher.canonicalized_final_state.gauge_policy,
                        "diagnostics": dict(teacher.diagnostics),
                    },
                }
            )
            particles = tf.reshape(teacher.particles, [-1])
            log_weights = tf.fill([NUM_PARTICLES], -tf.math.log(tf.cast(NUM_PARTICLES, DTYPE)))
        else:
            log_weights = tf.math.log(tf.maximum(weights, tf.constant(1e-300, dtype=DTYPE)))
    return examples


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
    if payload["counts"]["heldout_examples"] <= 0:
        raise RuntimeError("SV artifact captured no heldout examples")
    if "reproducibility_digest" not in payload:
        raise RuntimeError("missing reproducibility digest")


def _markdown(payload: dict[str, Any]) -> str:
    return f"""# Retained-Teacher Sinkhorn SV Teacher-Data Result

## Decision

`{payload['decision']}`

## Decision Table

| Check | Status | Evidence |
| --- | --- | --- |
| CPU-only manifest | pass | `pre_import_cuda_visible_devices={payload['run_manifest']['pre_import_cuda_visible_devices']}` |
| total examples | pass | `{payload['counts']['total_examples']}` |
| train / heldout split | pass | `{payload['counts']['train_examples']}` / `{payload['counts']['heldout_examples']}` |
| dataset checksum | pass | `{payload['summary']['dataset_checksum']}` |

## Interpretation

This artifact creates the first retained-teacher Sinkhorn teacher-data dataset on
the stochastic-volatility fixture family, preserving the same BayesFilter teacher
object and latent-state contract used on the LGSSM envelope.

## Non-Implications

{_non_implications_markdown()}
"""


def _non_implications() -> list[str]:
    return [
        "No student training claim is concluded.",
        "No posterior correctness is concluded.",
        "No HMC readiness is concluded.",
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
