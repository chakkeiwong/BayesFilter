"""Train and evaluate a minimal retained-teacher Sinkhorn warm-start student smoke runner."""

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

from experiments.dpf_implementation.tf_tfp.resampling.sinkhorn_tf import (
    SinkhornLogStateTF,
    sinkhorn_resample_tf,
)
from experiments.dpf_implementation.tf_tfp.resampling.sinkhorn_warmstart_student_tf import (
    RetainedTeacherWarmStartConfigTF,
    SinkhornWarmStartStudentTF,
    predict_sinkhorn_initial_state_tf,
    teacher_state_loss_tf,
)
from experiments.dpf_implementation.tf_tfp.runners.common_tf import (
    OUTPUT_DIR,
    REPORT_DIR,
    REPO_ROOT,
    environment_manifest,
    load_json,
    rmse_tf,
    scalar,
    stable_digest,
    utc_now,
    wall_time_call,
    write_json,
    write_text,
)


INPUT_JSON_PATH = OUTPUT_DIR / "retained_teacher_sinkhorn_teacher_data_2026-06-18.json"
JSON_PATH = OUTPUT_DIR / "retained_teacher_sinkhorn_student_smoke_2026-06-18.json"
REPORT_PATH = REPORT_DIR / "retained-teacher-sinkhorn-student-smoke-2026-06-18.md"
RESULT_PATH = "docs/plans/bayesfilter-neural-ot-retained-teacher-student-smoke-result-2026-06-18.md"
SCALAR_ID = "retained_teacher_sinkhorn_student_latent_and_replay_smoke_tf"
TRAINING_SEED = 20260618
EPOCHS = 250
LEARNING_RATE = 1e-2
CORRECTIVE_ITERATIONS = 20
EXPECTED_DECISION = "RETAINED_TEACHER_SINKHORN_STUDENT_SMOKE_PASSED"


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
    markdown = _markdown(payload)
    write_json(JSON_PATH, payload)
    write_text(REPORT_PATH, markdown)
    write_text(REPO_ROOT / RESULT_PATH, markdown)
    if args.check_reproducibility:
        second = _run()
        second["run_manifest"]["wall_time_seconds"] = payload["run_manifest"]["wall_time_seconds"]
        second["reproducibility_digest"] = _digest_payload(second)
        if second["reproducibility_digest"] != payload["reproducibility_digest"]:
            raise RuntimeError("student smoke reproducibility digest mismatch")
    _validate_payload(payload)
    print(payload["decision"])
    return 0


def _run() -> dict[str, Any]:
    teacher_payload = load_json(INPUT_JSON_PATH)
    teacher_policy = dict(teacher_payload["teacher_generation_policy"])
    examples = [_tensorize_example(example, teacher_policy) for example in teacher_payload["examples"]]
    train_examples = [example for example in examples if example["split"] == "train"]
    heldout_examples = [example for example in examples if example["split"] == "heldout"]
    if not train_examples or not heldout_examples:
        raise RuntimeError("teacher-data artifact must include both train and heldout examples")

    tf.keras.utils.set_random_seed(TRAINING_SEED)
    config = RetainedTeacherWarmStartConfigTF()
    model = SinkhornWarmStartStudentTF(config)
    _ = model(train_examples[0]["particles"], train_examples[0]["weights"], epsilon=train_examples[0]["epsilon"])
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    initial_train_loss = _mean_latent_loss(model, train_examples)
    loss_trace = []
    for epoch in range(EPOCHS):
        with tf.GradientTape() as tape:
            loss = _mean_latent_loss(model, train_examples)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        if epoch in {0, 9, 49, 99, EPOCHS - 1}:
            loss_trace.append({"epoch": int(epoch + 1), "train_loss": scalar(loss)})

    final_train_loss = _mean_latent_loss(model, train_examples)
    heldout_latent_loss = _mean_latent_loss(model, heldout_examples)
    train_replay = _replay_metrics(model, train_examples, corrective_iterations=CORRECTIVE_ITERATIONS)
    heldout_replay = _replay_metrics(model, heldout_examples, corrective_iterations=CORRECTIVE_ITERATIONS)

    decision = EXPECTED_DECISION
    if not _all_metrics_finite(final_train_loss, heldout_latent_loss, train_replay, heldout_replay):
        decision = "RETAINED_TEACHER_SINKHORN_STUDENT_SMOKE_FAILED"
    elif scalar(final_train_loss) >= scalar(initial_train_loss):
        decision = "RETAINED_TEACHER_SINKHORN_STUDENT_SMOKE_FAILED"

    payload = {
        "decision": decision,
        "question": "Can the minimal retained-teacher Sinkhorn warm-start student fit latent teacher state on the small LGSSM teacher-data artifact and produce finite heldout replay diagnostics?",
        "created_at_utc": utc_now(),
        "backend": "tensorflow_tensorflow_probability",
        "scalar_id": SCALAR_ID,
        "input_teacher_data_json": str(INPUT_JSON_PATH.relative_to(REPO_ROOT)),
        "teacher_data_reproducibility_digest": teacher_payload.get("reproducibility_digest"),
        "training_contract": {
            "training_seed": TRAINING_SEED,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "optimizer": "Adam",
            "corrective_iterations": CORRECTIVE_ITERATIONS,
            "teacher_epsilon": teacher_policy["teacher_epsilon"],
            "teacher_tolerance": teacher_policy["teacher_tolerance"],
            "train_examples": len(train_examples),
            "heldout_examples": len(heldout_examples),
            "model_config": {
                "particle_hidden_dim": config.particle_hidden_dim,
                "particle_hidden_layers": config.particle_hidden_layers,
                "pooled_hidden_dim": config.pooled_hidden_dim,
                "pooled_hidden_layers": config.pooled_hidden_layers,
                "epsilon_feature_scale": config.epsilon_feature_scale,
            },
        },
        "losses": {
            "initial_train_latent_loss": scalar(initial_train_loss),
            "final_train_latent_loss": scalar(final_train_loss),
            "heldout_latent_loss": scalar(heldout_latent_loss),
            "loss_trace": loss_trace,
        },
        "replay_metrics": {
            "train": train_replay,
            "heldout": heldout_replay,
        },
        "run_manifest": environment_manifest(
            command="CUDA_VISIBLE_DEVICES=-1 python -m experiments.dpf_implementation.tf_tfp.runners.run_retained_teacher_sinkhorn_student_smoke_tf",
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_implications": _non_implications(),
    }
    return payload


def _tensorize_example(example: dict[str, Any], teacher_policy: dict[str, Any]) -> dict[str, Any]:
    teacher = example["teacher"]
    return {
        "split": example["split"],
        "seed": int(example["seed"]),
        "time_index": int(example["time_index"]),
        "epsilon": float(teacher_policy["teacher_epsilon"]),
        "tolerance": float(teacher_policy["teacher_tolerance"]),
        "particles": tf.convert_to_tensor(example["particles"], dtype=tf.float64),
        "weights": tf.convert_to_tensor(example["weights"], dtype=tf.float64),
        "teacher_state": SinkhornLogStateTF(
            log_u=tf.convert_to_tensor(teacher["canonical_log_u"], dtype=tf.float64),
            log_v=tf.convert_to_tensor(teacher["canonical_log_v"], dtype=tf.float64),
            gauge_policy=teacher["canonical_gauge_policy"],
        ),
        "teacher_barycentric": tf.convert_to_tensor(teacher["barycentric_particles"], dtype=tf.float64),
        "teacher_diagnostics": dict(teacher["diagnostics"]),
    }


def _mean_latent_loss(model: SinkhornWarmStartStudentTF, examples: list[dict[str, Any]]) -> tf.Tensor:
    losses = []
    for example in examples:
        predicted_state = predict_sinkhorn_initial_state_tf(
            model,
            example["particles"],
            example["weights"],
            example["epsilon"],
        )
        losses.append(teacher_state_loss_tf(predicted_state, example["teacher_state"]))
    return tf.add_n(losses) / tf.cast(len(losses), tf.float64)


def _replay_metrics(
    model: SinkhornWarmStartStudentTF,
    examples: list[dict[str, Any]],
    *,
    corrective_iterations: int,
) -> dict[str, Any]:
    student_rmses = []
    zero_rmses = []
    student_residuals = []
    zero_residuals = []
    student_better_or_equal_count = 0
    for example in examples:
        predicted_state = predict_sinkhorn_initial_state_tf(
            model,
            example["particles"],
            example["weights"],
            example["epsilon"],
        )
        student_result = sinkhorn_resample_tf(
            example["particles"],
            example["weights"],
            epsilon=example["epsilon"],
            max_iterations=corrective_iterations,
            tolerance=example["tolerance"],
            initial_state=predicted_state,
        )
        zero_result = sinkhorn_resample_tf(
            example["particles"],
            example["weights"],
            epsilon=example["epsilon"],
            max_iterations=corrective_iterations,
            tolerance=example["tolerance"],
        )
        student_rmse = rmse_tf(student_result.particles, example["teacher_barycentric"])
        zero_rmse = rmse_tf(zero_result.particles, example["teacher_barycentric"])
        student_rmses.append(student_rmse)
        zero_rmses.append(zero_rmse)
        student_residuals.append(
            max(
                float(student_result.diagnostics["max_row_residual"]),
                float(student_result.diagnostics["max_column_residual"]),
                float(student_result.diagnostics["total_mass_residual"]),
            )
        )
        zero_residuals.append(
            max(
                float(zero_result.diagnostics["max_row_residual"]),
                float(zero_result.diagnostics["max_column_residual"]),
                float(zero_result.diagnostics["total_mass_residual"]),
            )
        )
        if student_rmse <= zero_rmse:
            student_better_or_equal_count += 1
    return {
        "mean_student_teacher_cloud_rmse": float(sum(student_rmses) / len(student_rmses)),
        "mean_zero_teacher_cloud_rmse": float(sum(zero_rmses) / len(zero_rmses)),
        "max_student_teacher_cloud_rmse": float(max(student_rmses)),
        "max_zero_teacher_cloud_rmse": float(max(zero_rmses)),
        "max_student_residual": float(max(student_residuals)),
        "max_zero_residual": float(max(zero_residuals)),
        "student_better_or_equal_count": int(student_better_or_equal_count),
        "example_count": len(examples),
    }


def _all_metrics_finite(
    final_train_loss: tf.Tensor,
    heldout_latent_loss: tf.Tensor,
    train_replay: dict[str, Any],
    heldout_replay: dict[str, Any],
) -> bool:
    values = [
        scalar(final_train_loss),
        scalar(heldout_latent_loss),
        train_replay["mean_student_teacher_cloud_rmse"],
        train_replay["mean_zero_teacher_cloud_rmse"],
        heldout_replay["mean_student_teacher_cloud_rmse"],
        heldout_replay["mean_zero_teacher_cloud_rmse"],
        train_replay["max_student_residual"],
        train_replay["max_zero_residual"],
        heldout_replay["max_student_residual"],
        heldout_replay["max_zero_residual"],
    ]
    return all(tf.math.is_finite(tf.constant(value, dtype=tf.float64)).numpy() for value in values)


def _validate_payload(payload: dict[str, Any]) -> None:
    if payload["decision"] != EXPECTED_DECISION:
        raise RuntimeError(payload["decision"])
    if payload["run_manifest"]["pre_import_cuda_visible_devices"] != "-1":
        raise RuntimeError("missing CPU-only pre-import manifest")
    if payload["scalar_id"] != SCALAR_ID:
        raise RuntimeError("wrong scalar id")
    if payload["losses"]["final_train_latent_loss"] >= payload["losses"]["initial_train_latent_loss"]:
        raise RuntimeError("training did not improve train latent loss")
    if "reproducibility_digest" not in payload:
        raise RuntimeError("missing reproducibility digest")


def _markdown(payload: dict[str, Any]) -> str:
    losses = payload["losses"]
    heldout = payload["replay_metrics"]["heldout"]
    return f"""# Retained-Teacher Sinkhorn Student Smoke Result

## Decision

`{payload['decision']}`

## Decision Table

| Check | Status | Evidence |
| --- | --- | --- |
| CPU-only manifest | pass | `pre_import_cuda_visible_devices={payload['run_manifest']['pre_import_cuda_visible_devices']}` |
| train latent loss improved | pass | `{losses['initial_train_latent_loss']:.3e} -> {losses['final_train_latent_loss']:.3e}` |
| heldout latent loss finite | pass | `{losses['heldout_latent_loss']:.3e}` |
| heldout student replay RMSE | smoke | `{heldout['mean_student_teacher_cloud_rmse']:.3e}` |
| heldout zero-init replay RMSE | comparator | `{heldout['mean_zero_teacher_cloud_rmse']:.3e}` |
| heldout better-or-equal count | explanatory | `{heldout['student_better_or_equal_count']}/{heldout['example_count']}` |

## Interpretation

The minimal DeepSets-style retained-teacher Sinkhorn student trained on the
small LGSSM teacher-data artifact, reduced train latent loss, and produced finite
heldout replay diagnostics under a reduced corrective budget. This is a local
smoke result only; it does not establish posterior correctness, HMC validity,
or broad generalization.

## Non-Implications

{_non_implications_markdown()}
"""


def _non_implications() -> list[str]:
    return [
        "No posterior correctness is concluded.",
        "No HMC readiness is concluded.",
        "No production-readiness claim is concluded.",
        "No broad cross-model generalization claim is concluded.",
        "No promotion of the student over the retained teacher is concluded from this smoke artifact alone.",
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
