#!/usr/bin/env python3
"""Run one bounded T1 score-tangent pilot or an untouched score claim."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import tensorflow as tf  # noqa: E402

from bayesfilter.runtime.gpu_memory_policy import (  # noqa: E402
    configure_tensorflow_gpu_memory_growth,
)


MEMORY_POLICY = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)

from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_artifact_compat import (  # noqa: E402
    load_lane_b_t1_artifact_v1_compat,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_t1_score_tf import (  # noqa: E402
    EXPECTED_T1_PARENT_IDENTITY,
    LaneBT1TangentTrainer,
    LaneBT1TangentTrainingConfig,
    estimate_t1_fisher_score,
    generate_t1_score_batch,
    load_t1_score_artifact,
    make_compiled_tangent_train_step,
    make_compiled_child_origin_score,
    make_t1_score_artifact,
    save_t1_score_artifact,
    t1_score_source_closure,
    tangent_validation_metrics,
    tangent_workspace_estimate_bytes,
)


PLAN = Path(
    "docs/plans/bayesfilter-zhao-cui-austria-sir-lane-b-t1-score-plan-2026-07-31.md"
)
PARENT_DIR = ROOT / (
    "docs/plans/artifacts/zhao-cui-austria-sir-lane-b-t1-20260730/"
    "pilot-final-02/p05_r4_b5_lr3e4_l1_1e9/artifact"
)
TRAIN_SEED = 74101
VALIDATION_SEED = 74201
CALIBRATION_SEED = 74301
UNTOUCHED_SEED = 74401
TRAIN_COUNT = 8192
VALIDATION_COUNT = 16384
CALIBRATION_COUNT = 32768
UNTOUCHED_COUNT = 65536
MEMORY_CAP_BYTES = 6 * 1024**3

ARM_TABLE: Mapping[str, Mapping[str, Any]] = {
    "s01_lr3e4_l1_0": {"learning_rate": 3e-4, "l1_weight": 0.0},
    "s02_lr3e4_l1_1e9": {"learning_rate": 3e-4, "l1_weight": 1e-9},
    "s03_lr3e4_l1_1e8": {"learning_rate": 3e-4, "l1_weight": 1e-8},
    "s04_lr1e4_l1_1e9": {"learning_rate": 1e-4, "l1_weight": 1e-9},
    "s05_lr1e3_l1_1e9": {"learning_rate": 1e-3, "l1_weight": 1e-9},
    "s06_lr1e4_l1_1e8": {"learning_rate": 1e-4, "l1_weight": 1e-8},
}


def _config(arm_id: str, *, train_steps: int = 96) -> LaneBT1TangentTrainingConfig:
    if arm_id not in ARM_TABLE:
        raise ValueError(f"unknown T1 score arm: {arm_id}")
    row = ARM_TABLE[arm_id]
    return LaneBT1TangentTrainingConfig(
        arm_id=arm_id,
        learning_rate=float(row["learning_rate"]),
        l1_weight=float(row["l1_weight"]),
        l2_weight=1e-10,
        gradient_clip_norm=100.0,
        batch_size=512,
        train_steps=int(train_steps),
        seed=74501,
    )


def _jsonable(value: object) -> object:
    if isinstance(value, tf.Tensor):
        array = value.numpy()
        return array.item() if array.ndim == 0 else array.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(command: list[str]) -> str:
    return subprocess.run(
        ["git", *command], cwd=ROOT, check=False, capture_output=True, text=True
    ).stdout.strip()


def _run_manifest(argv: list[str], started: float) -> Mapping[str, object]:
    return {
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_status_short": _git(["status", "--short"]).splitlines(),
        "command": argv,
        "python": sys.version,
        "platform": platform.platform(),
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV", "N/A"),
        "tensorflow": tf.__version__,
        "devices": [device.name for device in tf.config.list_logical_devices()],
        "gpu_memory_policy": dict(MEMORY_POLICY),
        "tf32_enabled": True,
        "jit_compile": True,
        "dtype": "float64",
        "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        "plan": PLAN.as_posix(),
        "source_closure": dict(t1_score_source_closure()),
        "wall_time_seconds": time.monotonic() - started,
    }


def _make_optimizer(config: LaneBT1TangentTrainingConfig):
    return tf.keras.optimizers.Adam(learning_rate=config.learning_rate)


def _pilot(
    arm_id: str,
    output_dir: Path,
    *,
    train_count: int,
    validation_count: int,
    train_steps: int,
    started: float,
) -> Mapping[str, object]:
    parent = load_lane_b_t1_artifact_v1_compat(PARENT_DIR)
    if parent.identity.hash.value != EXPECTED_T1_PARENT_IDENTITY:
        raise ValueError("admitted T1 parent identity changed")
    config = _config(arm_id, train_steps=train_steps)
    training = generate_t1_score_batch(
        parent=parent,
        sample_count=int(train_count),
        seed=TRAIN_SEED,
        role="score_training",
    )
    validation = generate_t1_score_batch(
        parent=parent,
        sample_count=int(validation_count),
        seed=VALIDATION_SEED,
        role="score_validation",
        score_scale=training.score_scale,
    )
    trainer = LaneBT1TangentTrainer(parent)
    optimizer = _make_optimizer(config)
    train_step = make_compiled_tangent_train_step(trainer, optimizer, config)
    last_terms: tuple[tf.Tensor, ...] | None = None
    for step in range(config.train_steps):
        if step % max(training.sample_count // config.batch_size, 1) == 0:
            permutation = tf.random.experimental.stateless_shuffle(
                tf.range(training.sample_count),
                seed=tf.constant([config.seed, step], tf.int32),
            )
        start = (step * config.batch_size) % training.sample_count
        indices = permutation[start : start + config.batch_size]
        last_terms = train_step(
            tf.gather(training.local_points, indices),
            tf.gather(training.target_score, indices),
            tf.gather(training.target_weights, indices),
            training.score_scale,
            tf.constant(training.sample_count, tf.int32),
        )
    if last_terms is None:
        raise RuntimeError("T1 score training executed no steps")
    training_fisher = estimate_t1_fisher_score(training)
    _pre_calibration_value, pre_calibration_score = trainer.freeze_child().increment_and_score(
        tf.zeros([3], tf.float64)
    )
    calibration_alpha = trainer.calibrate_normalizer_score(training_fisher.score)
    _post_calibration_value, post_calibration_score = trainer.freeze_child().increment_and_score(
        tf.zeros([3], tf.float64)
    )
    metrics = tangent_validation_metrics(trainer, validation)
    metrics = {
        **metrics,
        "training_fisher_score": training_fisher.score,
        "training_fisher_standard_error": training_fisher.standard_error,
        "pre_calibration_child_score": pre_calibration_score,
        "calibration_alpha": calibration_alpha,
        "post_calibration_child_score": post_calibration_score,
        "calibration_score_residual": post_calibration_score - training_fisher.score,
    }
    artifact = make_t1_score_artifact(
        trainer=trainer,
        config=config,
        training_batch=training,
        validation_batch=validation,
        validation_metrics=metrics,
    )
    artifact_dir = output_dir / "artifact"
    manifest_path = save_t1_score_artifact(artifact, artifact_dir)
    reloaded = load_t1_score_artifact(artifact_dir, parent=parent)
    child_value, child_score = reloaded.child().increment_and_score(
        tf.zeros([3], tf.float64)
    )
    compiled_child_value, compiled_child_score = make_compiled_child_origin_score(
        reloaded.child()
    )()
    xla_tie_out = tf.reduce_max(
        tf.abs(
            tf.concat(
                [
                    tf.reshape(compiled_child_value - child_value, [1]),
                    compiled_child_score - child_score,
                ],
                axis=0,
            )
        )
    )
    memory = tf.config.experimental.get_memory_info("GPU:0")
    workspace = tangent_workspace_estimate_bytes(
        parent=parent, batch_size=config.batch_size
    )
    origin_residual = tf.abs(child_value - parent.value())
    finite_gate = bool(
        tf.reduce_all(
            tf.stack(
                [
                    tf.reduce_all(tf.math.is_finite(value))
                    for value in (*last_terms, *metrics.values(), child_score)
                ]
            )
        ).numpy()
    )
    passed = (
        reloaded.identity == artifact.identity
        and float(origin_residual) <= 2e-13
        and finite_gate
        and float(xla_tie_out) <= 3e-11
        and int(memory["peak"]) <= MEMORY_CAP_BYTES
        and workspace <= MEMORY_CAP_BYTES
    )
    return {
        "schema_version": "bayesfilter.zhao_cui_austria_sir_lane_b_t1_score_pilot.v1",
        "status": "PASS_T1_SCORE_PILOT_ARM" if passed else "BLOCK_T1_SCORE_PILOT_ARM",
        "arm_id": arm_id,
        "score_artifact_identity": artifact.identity.hash.value,
        "child_identity": artifact.child().identity.hash.value,
        "parent_identity": parent.identity.hash.value,
        "parent_value": parent.value(),
        "child_origin_value": child_value,
        "child_origin_score": child_score,
        "compiled_child_origin_value": compiled_child_value,
        "compiled_child_origin_score": compiled_child_score,
        "xla_eager_tie_out": xla_tie_out,
        "training_fisher_estimate": training_fisher.manifest_payload(),
        "origin_value_residual": origin_residual,
        "training_last_terms": {
            name: value
            for name, value in zip(
                ("loss", "data_loss", "regularization", "gradient_norm"),
                last_terms,
            )
        },
        "validation_metrics": metrics,
        "training_manifest": training.manifest_payload(),
        "validation_manifest": validation.manifest_payload(),
        "artifact_directory": artifact_dir.relative_to(ROOT).as_posix(),
        "artifact_manifest_sha256": _sha256(manifest_path),
        "workspace_estimate_bytes": workspace,
        "gpu_allocator": {key: int(value) for key, value in memory.items()},
        "gates": {
            "fresh_reload": reloaded.identity == artifact.identity,
            "origin_value": float(origin_residual) <= 2e-13,
            "finite": finite_gate,
            "xla_eager_tie_out": float(xla_tie_out) <= 3e-11,
            "memory": int(memory["peak"]) <= MEMORY_CAP_BYTES and workspace <= MEMORY_CAP_BYTES,
            "passed": passed,
        },
        "run_manifest": _run_manifest(sys.argv, started),
        "nonclaims": (
            "validation metrics are selection diagnostics only",
            "no observed-data score admission before untouched Fisher comparison",
            "no T2 or HMC readiness",
        ),
    }


def _claim(
    artifact_dir: Path,
    output_dir: Path,
    *,
    sample_count: int,
    seed: int,
    role: str,
    started: float,
) -> Mapping[str, object]:
    parent = load_lane_b_t1_artifact_v1_compat(PARENT_DIR)
    artifact = load_t1_score_artifact(artifact_dir, parent=parent)
    batch = generate_t1_score_batch(
        parent=parent,
        sample_count=int(sample_count),
        seed=int(seed),
        role=str(role),
        score_scale=artifact.score_scale,
    )
    estimate = estimate_t1_fisher_score(batch)
    child_value, child_score = artifact.child().increment_and_score(
        tf.zeros([3], tf.float64)
    )
    compiled_child_value, compiled_child_score = make_compiled_child_origin_score(
        artifact.child()
    )()
    xla_tie_out = tf.reduce_max(
        tf.abs(
            tf.concat(
                [
                    tf.reshape(compiled_child_value - child_value, [1]),
                    compiled_child_score - child_score,
                ],
                axis=0,
            )
        )
    )
    difference = tf.abs(child_score - estimate.score)
    tolerance = 3.0 * estimate.standard_error + tf.constant(1e-5, tf.float64)
    coordinate_gate = difference <= tolerance
    informative_gate = tf.reduce_all(estimate.standard_error <= tf.constant([2.0, 1.0, 0.5], tf.float64))
    memory = tf.config.experimental.get_memory_info("GPU:0")
    passed = (
        bool(tf.reduce_all(coordinate_gate).numpy())
        and bool(informative_gate.numpy())
        and float(xla_tie_out) <= 3e-11
        and int(memory["peak"]) <= MEMORY_CAP_BYTES
    )
    pass_status = (
        "PASS_T1_SCORE_CALIBRATION"
        if role == "score_calibration"
        else "PASS_LANE_B_T1_VALUE_AND_TOTAL_SCORE"
    )
    return {
        "schema_version": "bayesfilter.zhao_cui_austria_sir_lane_b_t1_score_claim.v1",
        "status": pass_status if passed else "BLOCK_T1_SCORE_CLAIM",
        "role": role,
        "score_artifact_identity": artifact.identity.hash.value,
        "child_identity": artifact.child().identity.hash.value,
        "parent_identity": parent.identity.hash.value,
        "child_origin_value": child_value,
        "child_origin_score": child_score,
        "compiled_child_origin_value": compiled_child_value,
        "compiled_child_origin_score": compiled_child_score,
        "xla_eager_tie_out": xla_tie_out,
        "fisher_estimate": estimate.manifest_payload(),
        "absolute_score_difference": difference,
        "score_tolerance": tolerance,
        "coordinate_gate": coordinate_gate,
        "informative_mcse_gate": informative_gate,
        "gpu_allocator": {key: int(value) for key, value in memory.items()},
        "gates": {
            "coordinate_score": bool(tf.reduce_all(coordinate_gate).numpy()),
            "informative_mcse": bool(informative_gate.numpy()),
            "xla_eager_tie_out": float(xla_tie_out) <= 3e-11,
            "memory": int(memory["peak"]) <= MEMORY_CAP_BYTES,
            "passed": passed,
        },
        "batch_manifest": batch.manifest_payload(),
        "artifact_directory": artifact_dir.relative_to(ROOT).as_posix(),
        "run_manifest": _run_manifest(sys.argv, started),
        "decision_table": {
            "decision": "open_T2_score_review" if passed else "reject_selected_T1_tangent_child",
            "primary_criterion_status": "passed" if passed else "failed",
            "veto_diagnostic_status": "passed" if int(memory["peak"]) <= MEMORY_CAP_BYTES else "failed_memory",
            "main_uncertainty": "iid ratio-estimator Monte Carlo standard error",
            "next_justified_action": "review T2 total previous-marginal derivative" if passed else "preserve result and classify fit versus capacity failure",
            "not_concluded": "no T2/T20 score, HMC, posterior, exact-likelihood theorem, or production readiness",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pilot-arm", choices=tuple(ARM_TABLE))
    group.add_argument("--claim-artifact", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=TRAIN_COUNT)
    parser.add_argument("--validation-count", type=int, default=VALIDATION_COUNT)
    parser.add_argument("--train-steps", type=int, default=96)
    parser.add_argument("--claim-count", type=int, default=CALIBRATION_COUNT)
    parser.add_argument("--claim-seed", type=int, default=CALIBRATION_SEED)
    parser.add_argument("--claim-role", choices=("score_calibration", "score_untouched"), default="score_calibration")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        if args.pilot_arm is not None:
            result = _pilot(
                args.pilot_arm,
                output_dir,
                train_count=args.train_count,
                validation_count=args.validation_count,
                train_steps=args.train_steps,
                started=started,
            )
        else:
            result = _claim(
                args.claim_artifact.resolve(),
                output_dir,
                sample_count=args.claim_count,
                seed=args.claim_seed,
                role=args.claim_role,
                started=started,
            )
        _write_json(output_dir / "result.json", result)
    except Exception as exc:
        _write_json(
            output_dir / "failure.json",
            {
                "schema_version": "bayesfilter.zhao_cui_austria_sir_lane_b_t1_score_failure.v1",
                "status": "INFRASTRUCTURE_OR_IMPLEMENTATION_FAILURE",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "command": sys.argv,
                "gpu_memory_policy": dict(MEMORY_POLICY),
            },
        )
        raise


if __name__ == "__main__":
    main()
