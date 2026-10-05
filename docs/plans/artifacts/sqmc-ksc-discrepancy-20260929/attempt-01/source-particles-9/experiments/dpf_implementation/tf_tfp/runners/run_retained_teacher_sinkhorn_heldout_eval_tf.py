"""Run heldout evaluation for retained-teacher Sinkhorn warm-start versus zero-init."""

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
    meta_ot_dual_objective_loss_tf,
    predict_canonical_log_u_tf,
    predict_sinkhorn_initial_state_tf,
    teacher_log_u_loss_tf,
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
JSON_PATH = OUTPUT_DIR / "retained_teacher_sinkhorn_heldout_eval_better_contract_2026-06-28.json"
REPORT_PATH = REPORT_DIR / "retained-teacher-sinkhorn-heldout-eval-better-contract-2026-06-28.md"
RESULT_PATH = "docs/plans/bayesfilter-neural-ot-metaot-refit-better-evidence-contract-result-2026-06-28.md"
PLAN_PATH = "docs/plans/bayesfilter-neural-ot-metaot-refit-better-evidence-contract-plan-2026-06-28.md"
SCALAR_ID = "retained_teacher_sinkhorn_heldout_teacher_cloud_fidelity_tf"
TRAINING_SEED = 20260618
EPOCHS = 250
LEARNING_RATE = 1e-2
BUDGETS = (5, 10, 20)
EXPECTED_DECISION = "RETAINED_TEACHER_SINKHORN_HELDOUT_EVAL_LOCAL_USEFULNESS_ON_DISCRIMINATING_BUDGETS"
PRIMARY_BUDGET = 20
BUDGET_TOLERANCE_FLOORS = {
    5: 1e-5,
    10: 1e-7,
    20: 1e-8,
}


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
        raise RuntimeError("teacher-data artifact must include both train and heldout examples")

    tf.keras.utils.set_random_seed(TRAINING_SEED)
    model = SinkhornWarmStartStudentTF(
        RetainedTeacherWarmStartConfigTF(prediction_head="meta_ot_log_u")
    )
    _ = model(train_examples[0]["particles"], train_examples[0]["weights"], epsilon=train_examples[0]["epsilon"])
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    initial_train_loss = _mean_training_loss(model, train_examples)
    for _ in range(EPOCHS):
        with tf.GradientTape() as tape:
            loss = _mean_training_loss(model, train_examples)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    final_train_loss = _mean_training_loss(model, train_examples)
    heldout_log_u_loss = _mean_log_u_loss(model, heldout_examples)

    budget_metrics = {
        str(budget): _budget_metrics(
            model,
            heldout_examples,
            budget,
            tolerance=max(float(teacher_policy["teacher_tolerance"]), float(BUDGET_TOLERANCE_FLOORS[budget])),
        )
        for budget in BUDGETS
    }
    budget_metrics = _annotate_budget_regimes(budget_metrics)
    primary = budget_metrics[str(PRIMARY_BUDGET)]
    discriminating_budgets = [
        metrics for metrics in budget_metrics.values()
        if metrics["budget_regime"] == "discriminating"
    ]
    decision = EXPECTED_DECISION
    if scalar(final_train_loss) >= scalar(initial_train_loss):
        decision = "RETAINED_TEACHER_SINKHORN_HELDOUT_EVAL_OBJECTIVE_ROUTE_FAILED"
    elif not _budgets_all_finite(budget_metrics):
        decision = "RETAINED_TEACHER_SINKHORN_HELDOUT_EVAL_OBJECTIVE_ROUTE_FAILED"
    elif not discriminating_budgets:
        decision = "RETAINED_TEACHER_SINKHORN_HELDOUT_EVAL_NO_DISCRIMINATING_BUDGET"
    elif all(
        metrics["mean_student_teacher_cloud_rmse"] > metrics["mean_zero_teacher_cloud_rmse"]
        for metrics in discriminating_budgets
    ):
        decision = "RETAINED_TEACHER_SINKHORN_HELDOUT_EVAL_NON_PROMOTED_ON_DISCRIMINATING_BUDGETS"

    payload = {
        "decision": decision,
        "question": "On heldout teacher-data examples, does donor-aligned one-half student-warm-started retained Sinkhorn show local usefulness on discriminating corrective-budget rungs without being over-read as algorithm failure when zero-init saturates high-budget rungs?",
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
            "loss_route": "meta_ot_log_u_dual_objective_plus_teacher_log_u",
            "prediction_head": "meta_ot_log_u",
            "budgets": list(BUDGETS),
            "primary_budget": PRIMARY_BUDGET,
            "budget_tolerance_floors": {str(k): v for k, v in BUDGET_TOLERANCE_FLOORS.items()},
            "teacher_epsilon": teacher_policy["teacher_epsilon"],
            "teacher_tolerance": teacher_policy["teacher_tolerance"],
            "train_examples": len(train_examples),
            "heldout_examples": len(heldout_examples),
        },
        "losses": {
            "initial_train_loss": scalar(initial_train_loss),
            "final_train_loss": scalar(final_train_loss),
            "heldout_log_u_loss": scalar(heldout_log_u_loss),
        },
        "budget_metrics": budget_metrics,
        "run_manifest": environment_manifest(
            command="CUDA_VISIBLE_DEVICES=-1 python -m experiments.dpf_implementation.tf_tfp.runners.run_retained_teacher_sinkhorn_heldout_eval_tf",
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
        "tolerance": float(teacher_policy["teacher_tolerance"]),
        "particles": tf.convert_to_tensor(example["particles"], dtype=tf.float64),
        "weights": tf.convert_to_tensor(example["weights"], dtype=tf.float64),
        "teacher_state": SinkhornLogStateTF(
            log_u=tf.convert_to_tensor(teacher["canonical_log_u"], dtype=tf.float64),
            log_v=tf.convert_to_tensor(teacher["canonical_log_v"], dtype=tf.float64),
            gauge_policy=teacher["canonical_gauge_policy"],
        ),
        "teacher_barycentric": tf.convert_to_tensor(teacher["barycentric_particles"], dtype=tf.float64),
    }


def _mean_training_loss(model: SinkhornWarmStartStudentTF, examples: list[dict[str, Any]]) -> tf.Tensor:
    losses = []
    for example in examples:
        predicted_log_u = predict_canonical_log_u_tf(
            model,
            example["particles"],
            example["weights"],
            example["epsilon"],
        )
        losses.append(
            meta_ot_dual_objective_loss_tf(
                predicted_log_u,
                example["particles"],
                example["weights"],
                example["epsilon"],
            )
            + teacher_log_u_loss_tf(predicted_log_u, example["teacher_state"])
        )
    return tf.add_n(losses) / tf.cast(len(losses), tf.float64)


def _mean_log_u_loss(model: SinkhornWarmStartStudentTF, examples: list[dict[str, Any]]) -> tf.Tensor:
    losses = []
    for example in examples:
        predicted_log_u = predict_canonical_log_u_tf(
            model,
            example["particles"],
            example["weights"],
            example["epsilon"],
        )
        losses.append(teacher_log_u_loss_tf(predicted_log_u, example["teacher_state"]))
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


def _annotate_budget_regimes(budget_metrics: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    annotated = {}
    for budget, metrics in budget_metrics.items():
        row = dict(metrics)
        saturated = row["mean_zero_teacher_cloud_rmse"] <= 1e-12
        row["budget_regime"] = "saturated_zero_init" if saturated else "discriminating"
        annotated[budget] = row
    return annotated


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
    if payload["decision"] not in {
        EXPECTED_DECISION,
        "RETAINED_TEACHER_SINKHORN_HELDOUT_EVAL_NON_PROMOTED_ON_DISCRIMINATING_BUDGETS",
    }:
        raise RuntimeError(payload["decision"])
    if payload["run_manifest"]["pre_import_cuda_visible_devices"] != "-1":
        raise RuntimeError("missing CPU-only pre-import manifest")
    if payload["scalar_id"] != SCALAR_ID:
        raise RuntimeError("wrong scalar id")
    if payload["losses"]["final_train_loss"] >= payload["losses"]["initial_train_loss"]:
        raise RuntimeError("training did not improve train loss")
    if "reproducibility_digest" not in payload:
        raise RuntimeError("missing reproducibility digest")
    discriminating_budgets = [
        metrics for metrics in payload["budget_metrics"].values()
        if metrics["budget_regime"] == "discriminating"
    ]
    if not discriminating_budgets:
        raise RuntimeError("missing discriminating budget rung")


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Retained-Teacher Sinkhorn Better-Contract Heldout Evaluation Result",
        "",
        "## Decision",
        "",
        f"`{payload['decision']}`",
        "",
        "## Decision Table",
        "",
        "| Budget | Regime | Student mean RMSE | Zero-init mean RMSE | Student max residual | Zero-init max residual | Student better-or-equal |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for budget, metrics in payload["budget_metrics"].items():
        lines.append(
            f"| `{budget}` | `{metrics['budget_regime']}` | `{metrics['mean_student_teacher_cloud_rmse']:.3e}` | `{metrics['mean_zero_teacher_cloud_rmse']:.3e}` | `{metrics['max_student_residual']:.3e}` | `{metrics['max_zero_residual']:.3e}` | `{metrics['student_better_or_equal_count']}/{metrics['heldout_example_count']}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This better-contract rung separates discriminating budgets from saturated zero-init budgets. Failure to beat zero-init on a saturated high-budget rung is not treated as algorithm failure by itself; the result is instead interpreted through the discriminating budget rows and the residual contract.",
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
        "No production-readiness claim is concluded.",
        "No broad cross-model generalization claim is concluded.",
        "No promotion of the student over the retained teacher beyond this heldout rung is concluded.",
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
