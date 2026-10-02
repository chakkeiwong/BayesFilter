"""Run low-budget heldout evaluation for retained-teacher Sinkhorn warm-start on the expanded Austria SIR d18 teacher artifact."""

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


INPUT_JSON_PATH = OUTPUT_DIR / "retained_teacher_sinkhorn_teacher_data_austria_sir_d18_expanded_2026-07-05.json"
JSON_PATH = OUTPUT_DIR / "retained_teacher_sinkhorn_low_budget_eval_austria_sir_d18_expanded_2026-07-05.json"
REPORT_PATH = REPORT_DIR / "retained-teacher-sinkhorn-low-budget-eval-austria-sir-d18-expanded-2026-07-05.md"
RESULT_PATH = "docs/plans/bayesfilter-neural-ot-austria-sir-d18-retained-teacher-expanded-low-budget-eval-result-2026-07-05.md"
PLAN_PATH = "docs/plans/bayesfilter-neural-ot-austria-sir-d18-retained-teacher-nonpromotion-disambiguation-plan-2026-07-05.md"
SCALAR_ID = "retained_teacher_sinkhorn_low_budget_teacher_cloud_fidelity_austria_sir_d18_expanded_tf"
TRAINING_SEED = 20260705
EPOCHS = 250
LEARNING_RATE = 1e-2
PRIMARY_BUDGETS = (3, 5)
EXPLANATORY_BUDGETS = ()
BUDGET_TOLERANCE_FLOORS = {
    3: 1e-4,
    5: 1e-5,
}
EXPECTED_DECISION = "RETAINED_TEACHER_SINKHORN_AUSTRIA_SIR_D18_EXPANDED_LOCAL_USEFULNESS_ON_DISCRIMINATING_BUDGETS"


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
        raise RuntimeError("expanded Austria SIR d18 teacher-data artifact must include both train and heldout examples")

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
    budget_metrics = _annotate_budget_regimes(budget_metrics)

    decision = EXPECTED_DECISION
    if scalar(final_train_loss) >= scalar(initial_train_loss):
        decision = "RETAINED_TEACHER_SINKHORN_AUSTRIA_SIR_D18_EXPANDED_OBJECTIVE_ROUTE_FAILED"
    elif not _budgets_all_finite(budget_metrics):
        decision = "RETAINED_TEACHER_SINKHORN_AUSTRIA_SIR_D18_EXPANDED_OBJECTIVE_ROUTE_FAILED"
    else:
        for budget in PRIMARY_BUDGETS:
            metrics = budget_metrics[str(budget)]
            if metrics["mean_student_teacher_cloud_rmse"] > metrics["mean_zero_teacher_cloud_rmse"]:
                decision = "RETAINED_TEACHER_SINKHORN_AUSTRIA_SIR_D18_EXPANDED_NON_PROMOTED_ON_DISCRIMINATING_BUDGETS"
                break

    representative_epsilon = next(
        (
            float(example["teacher"]["teacher_epsilon"])
            for example in teacher_payload["examples"]
            if "teacher_epsilon" in example.get("teacher", {})
        ),
        float(teacher_policy.get("teacher_epsilon", 0.75)),
    )
    payload = {
        "decision": decision,
        "question": "On the expanded Austria SIR d18 heldout teacher-data artifact, does donor-aligned one-half retained Sinkhorn beat zero-init on the discriminating budgets?",
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
            "primary_budgets": list(PRIMARY_BUDGETS),
            "explanatory_budgets": list(EXPLANATORY_BUDGETS),
            "budget_tolerance_floors": {str(k): v for k, v in BUDGET_TOLERANCE_FLOORS.items()},
            "teacher_epsilon": representative_epsilon,
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
            command="CUDA_VISIBLE_DEVICES=-1 python -m experiments.dpf_implementation.tf_tfp.runners.run_retained_teacher_sinkhorn_low_budget_eval_austria_sir_d18_expanded_tf",
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_implications": _non_implications(),
    }
    return payload


def _tensorize_example(example: dict[str, Any], teacher_policy: dict[str, Any]) -> dict[str, Any]:
    teacher = example["teacher"]
    return {
        "split": example["split"],
        "epsilon": float(teacher.get("teacher_epsilon", teacher_policy.get("teacher_epsilon", 0.75))),
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
        "RETAINED_TEACHER_SINKHORN_AUSTRIA_SIR_D18_EXPANDED_NON_PROMOTED_ON_DISCRIMINATING_BUDGETS",
    }:
        raise RuntimeError(payload["decision"])
    if payload["run_manifest"]["pre_import_cuda_visible_devices"] != "-1":
        raise RuntimeError("missing CPU-only pre-import manifest")
    if payload["scalar_id"] != SCALAR_ID:
        raise RuntimeError("wrong scalar id")
    if payload["losses"]["final_train_loss"] >= payload["losses"]["initial_train_loss"]:
        raise RuntimeError("training did not improve train loss")
    discriminating_budgets = [
        payload["budget_metrics"][str(budget)] for budget in PRIMARY_BUDGETS
        if payload["budget_metrics"][str(budget)]["budget_regime"] == "discriminating"
    ]
    if not discriminating_budgets:
        raise RuntimeError("missing discriminating primary-budget rung")
    if "reproducibility_digest" not in payload:
        raise RuntimeError("missing reproducibility digest")


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Austria SIR d18 Expanded Retained-Teacher Low-Budget Evaluation Result",
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
    for budget in list(PRIMARY_BUDGETS) + list(EXPLANATORY_BUDGETS):
        metrics = payload["budget_metrics"][str(budget)]
        lines.append(
            f"| `{budget}` | `{metrics['budget_regime']}` | `{metrics['mean_student_teacher_cloud_rmse']:.3e}` | `{metrics['mean_zero_teacher_cloud_rmse']:.3e}` | `{metrics['max_student_residual']:.3e}` | `{metrics['max_zero_residual']:.3e}` | `{metrics['student_better_or_equal_count']}/{metrics['heldout_example_count']}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This expanded Austria SIR d18 run keeps the same model, repair, and replay semantics while modestly strengthening the artifact. Its role is to distinguish weak-artifact non-promotion from persistent local non-usefulness on the fixed family.",
            "",
            "## Non-Implications",
            "",
            _non_implications_markdown(),
        ]
    )
    return "\n".join(lines) + "\n"


def _non_implications() -> list[str]:
    return [
        "No large-particle or N=10000 claim is concluded.",
        "No GPU scaling claim is concluded.",
        "No production-readiness claim is concluded.",
        "No parameterized-SIR claim is concluded.",
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
