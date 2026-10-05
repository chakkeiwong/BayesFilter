"""Run low-budget heldout evaluation for retained-teacher Sinkhorn warm-start on stochastic-volatility teacher data."""

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


INPUT_JSON_PATH = OUTPUT_DIR / "retained_teacher_sinkhorn_teacher_data_sv_2026-06-18.json"
JSON_PATH = OUTPUT_DIR / "retained_teacher_sinkhorn_low_budget_eval_sv_2026-06-18.json"
REPORT_PATH = REPORT_DIR / "retained-teacher-sinkhorn-low-budget-eval-sv-2026-06-18.md"
RESULT_PATH = "docs/plans/bayesfilter-neural-ot-retained-teacher-low-budget-eval-sv-result-2026-06-18.md"
PLAN_PATH = "docs/plans/bayesfilter-neural-ot-retained-teacher-sv-cross-envelope-plan-2026-06-18.md"
SCALAR_ID = "retained_teacher_sinkhorn_low_budget_teacher_cloud_fidelity_sv_tf"
TRAINING_SEED = 20260618
EPOCHS = 250
LEARNING_RATE = 1e-2
PRIMARY_BUDGETS = (5, 10)
EXPLANATORY_BUDGETS = (20,)
BUDGET_TOLERANCE_FLOORS = {
    5: 1e-5,
    10: 1e-7,
    20: 1e-8,
}
EXPECTED_DECISION = "RETAINED_TEACHER_SINKHORN_LOW_BUDGET_EVAL_SV_PASSED"


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
    teacher_payload = load_json(INPUT_JSON_PATH)
    teacher_policy = dict(teacher_payload["teacher_generation_policy"])
    examples = [_tensorize_example(example, teacher_policy) for example in teacher_payload["examples"]]
    train_examples = [example for example in examples if example["split"] == "train"]
    heldout_examples = [example for example in examples if example["split"] == "heldout"]
    if not train_examples or not heldout_examples:
        raise RuntimeError("SV teacher-data artifact must include both train and heldout examples")

    tf.keras.utils.set_random_seed(TRAINING_SEED)
    model = SinkhornWarmStartStudentTF(RetainedTeacherWarmStartConfigTF())
    _ = model(train_examples[0]["particles"], train_examples[0]["weights"], epsilon=train_examples[0]["epsilon"])
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    initial_train_loss = _mean_latent_loss(model, train_examples)
    for _ in range(EPOCHS):
        with tf.GradientTape() as tape:
            loss = _mean_latent_loss(model, train_examples)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    final_train_loss = _mean_latent_loss(model, train_examples)
    heldout_latent_loss = _mean_latent_loss(model, heldout_examples)

    all_budgets = PRIMARY_BUDGETS + EXPLANATORY_BUDGETS
    budget_metrics = {
        str(budget): _budget_metrics(
            model,
            heldout_examples,
            budget,
            tolerance=max(float(teacher_policy["teacher_tolerance"]), float(BUDGET_TOLERANCE_FLOORS[budget])),
        )
        for budget in all_budgets
    }

    decision = EXPECTED_DECISION
    if scalar(final_train_loss) >= scalar(initial_train_loss):
        decision = "RETAINED_TEACHER_SINKHORN_LOW_BUDGET_EVAL_SV_FAILED"
    elif not _budgets_all_finite(budget_metrics):
        decision = "RETAINED_TEACHER_SINKHORN_LOW_BUDGET_EVAL_SV_FAILED"
    else:
        for budget in PRIMARY_BUDGETS:
            metrics = budget_metrics[str(budget)]
            if metrics["mean_student_teacher_cloud_rmse"] > metrics["mean_zero_teacher_cloud_rmse"]:
                decision = "RETAINED_TEACHER_SINKHORN_LOW_BUDGET_EVAL_SV_FAILED"
                break

    payload = {
        "decision": decision,
        "question": "On stochastic-volatility heldout teacher-data examples, does student-warm-started retained Sinkhorn match or improve teacher-cloud fidelity relative to zero-init at low corrective budgets?",
        "created_at_utc": utc_now(),
        "backend": "tensorflow_tensorflow_probability",
        "scalar_id": SCALAR_ID,
        "plan_path": PLAN_PATH,
        "input_teacher_data_json": str(INPUT_JSON_PATH.relative_to(REPO_ROOT)),
        "teacher_data_reproducibility_digest": teacher_payload.get("reproducibility_digest"),
        "training_contract": {
            "training_seed": TRAINING_SEED,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "optimizer": "Adam",
            "primary_budgets": list(PRIMARY_BUDGETS),
            "explanatory_budgets": list(EXPLANATORY_BUDGETS),
            "budget_tolerance_floors": {str(k): v for k, v in BUDGET_TOLERANCE_FLOORS.items()},
            "teacher_epsilon": teacher_policy["teacher_epsilon"],
            "teacher_tolerance": teacher_policy["teacher_tolerance"],
            "train_examples": len(train_examples),
            "heldout_examples": len(heldout_examples),
        },
        "losses": {
            "initial_train_latent_loss": scalar(initial_train_loss),
            "final_train_latent_loss": scalar(final_train_loss),
            "heldout_latent_loss": scalar(heldout_latent_loss),
        },
        "budget_metrics": budget_metrics,
        "run_manifest": environment_manifest(
            command="CUDA_VISIBLE_DEVICES=-1 python -m experiments.dpf_implementation.tf_tfp.runners.run_retained_teacher_sinkhorn_low_budget_eval_sv_tf",
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_implications": _non_implications(),
    }
    return payload


def _tensorize_example(example: dict[str, Any], teacher_policy: dict[str, Any]) -> dict[str, Any]:
    teacher = example["teacher"]
    return {
        "split": example["split"],
        "epsilon": float(teacher_policy["teacher_epsilon"]),
        "particles": tf.convert_to_tensor(example["particles"], dtype=tf.float64),
        "weights": tf.convert_to_tensor(example["weights"], dtype=tf.float64),
        "teacher_state": SinkhornLogStateTF(
            log_u=tf.convert_to_tensor(teacher["canonical_log_u"], dtype=tf.float64),
            log_v=tf.convert_to_tensor(teacher["canonical_log_v"], dtype=tf.float64),
            gauge_policy=teacher["canonical_gauge_policy"],
        ),
        "teacher_barycentric": tf.convert_to_tensor(teacher["barycentric_particles"], dtype=tf.float64),
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


def _budget_metrics(
    model: SinkhornWarmStartStudentTF,
    heldout_examples: list[dict[str, Any]],
    corrective_budget: int,
    *,
    tolerance: float,
) -> dict[str, Any]:
    student_rmses = []
    zero_rmses = []
    student_residuals = []
    zero_residuals = []
    better_or_equal = 0
    for example in heldout_examples:
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
            max_iterations=corrective_budget,
            tolerance=tolerance,
            initial_state=predicted_state,
        )
        zero_result = sinkhorn_resample_tf(
            example["particles"],
            example["weights"],
            epsilon=example["epsilon"],
            max_iterations=corrective_budget,
            tolerance=tolerance,
        )
        student_rmse = rmse_tf(student_result.particles, example["teacher_barycentric"])
        zero_rmse = rmse_tf(zero_result.particles, example["teacher_barycentric"])
        student_rmses.append(student_rmse)
        zero_rmses.append(zero_rmse)
        student_residuals.append(_max_residual(student_result.diagnostics))
        zero_residuals.append(_max_residual(zero_result.diagnostics))
        if student_rmse <= zero_rmse:
            better_or_equal += 1
    return {
        "mean_student_teacher_cloud_rmse": float(sum(student_rmses) / len(student_rmses)),
        "mean_zero_teacher_cloud_rmse": float(sum(zero_rmses) / len(zero_rmses)),
        "max_student_teacher_cloud_rmse": float(max(student_rmses)),
        "max_zero_teacher_cloud_rmse": float(max(zero_rmses)),
        "max_student_residual": float(max(student_residuals)),
        "max_zero_residual": float(max(zero_residuals)),
        "student_better_or_equal_count": int(better_or_equal),
        "heldout_example_count": len(heldout_examples),
        "evaluation_tolerance": float(tolerance),
    }


def _max_residual(diagnostics: dict[str, Any]) -> float:
    return max(
        float(diagnostics["max_row_residual"]),
        float(diagnostics["max_column_residual"]),
        float(diagnostics["total_mass_residual"]),
    )


def _budgets_all_finite(budget_metrics: dict[str, dict[str, Any]]) -> bool:
    values = []
    for metrics in budget_metrics.values():
        values.extend(
            [
                metrics["mean_student_teacher_cloud_rmse"],
                metrics["mean_zero_teacher_cloud_rmse"],
                metrics["max_student_teacher_cloud_rmse"],
                metrics["max_zero_teacher_cloud_rmse"],
                metrics["max_student_residual"],
                metrics["max_zero_residual"],
            ]
        )
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
    for budget in PRIMARY_BUDGETS:
        metrics = payload["budget_metrics"][str(budget)]
        if metrics["mean_student_teacher_cloud_rmse"] > metrics["mean_zero_teacher_cloud_rmse"]:
            raise RuntimeError(f"student is worse than zero-init at primary budget {budget}")
    if "reproducibility_digest" not in payload:
        raise RuntimeError("missing reproducibility digest")


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Retained-Teacher Sinkhorn SV Low-Budget Evaluation Result",
        "",
        "## Decision",
        "",
        f"`{payload['decision']}`",
        "",
        "## Decision Table",
        "",
        "| Budget | Student mean RMSE | Zero-init mean RMSE | Student max residual | Zero-init max residual | Student better-or-equal |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for budget in list(PRIMARY_BUDGETS) + list(EXPLANATORY_BUDGETS):
        metrics = payload["budget_metrics"][str(budget)]
        lines.append(
            f"| `{budget}` | `{metrics['mean_student_teacher_cloud_rmse']:.3e}` | `{metrics['mean_zero_teacher_cloud_rmse']:.3e}` | `{metrics['max_student_residual']:.3e}` | `{metrics['max_zero_residual']:.3e}` | `{metrics['student_better_or_equal_count']}/{metrics['heldout_example_count']}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This rung asks whether the retained-teacher low-budget warm-start effect survives the first family expansion from LGSSM to stochastic volatility. A pass remains local evidence only for this SV envelope.",
            "",
            "## Non-Implications",
            "",
            _non_implications_markdown(),
        ]
    )
    return "\n".join(lines) + "\n"


def _non_implications() -> list[str]:
    return [
        "No posterior correctness is concluded.",
        "No HMC readiness is concluded.",
        "No broad nonlinear generalization claim is concluded.",
        "No production-readiness claim is concluded.",
        "No success on range-bearing or structural fixtures is concluded from this SV rung.",
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
