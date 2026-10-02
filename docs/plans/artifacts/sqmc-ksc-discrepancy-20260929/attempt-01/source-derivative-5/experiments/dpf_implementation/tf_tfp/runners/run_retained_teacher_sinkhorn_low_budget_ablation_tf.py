"""Run low-budget warm-start ablations on the expanded retained-teacher Sinkhorn artifact."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import tensorflow as tf

from experiments.dpf_implementation.tf_tfp.resampling.sinkhorn_tf import (
    SinkhornLogStateTF,
    canonicalize_sinkhorn_log_state_tf,
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


INPUT_JSON_PATH = OUTPUT_DIR / "retained_teacher_sinkhorn_teacher_data_expanded_2026-06-18.json"
JSON_PATH = OUTPUT_DIR / "retained_teacher_sinkhorn_low_budget_ablation_2026-06-18.json"
REPORT_PATH = REPORT_DIR / "retained-teacher-sinkhorn-low-budget-ablation-2026-06-18.md"
RESULT_PATH = "docs/plans/bayesfilter-neural-ot-retained-teacher-low-budget-ablation-result-2026-06-18.md"
PLAN_PATH = "docs/plans/bayesfilter-neural-ot-retained-teacher-low-budget-ablation-plan-2026-06-18.md"
SCALAR_ID = "retained_teacher_sinkhorn_low_budget_warmstart_ablation_tf"
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
EXPECTED_DECISION = "RETAINED_TEACHER_SINKHORN_LOW_BUDGET_ABLATION_PASSED"


class _LearnedVariant:
    def __init__(self, name: str, config: RetainedTeacherWarmStartConfigTF) -> None:
        self.name = name
        self.config = config
        self.model = SinkhornWarmStartStudentTF(config)
        self.initial_train_loss: float | None = None
        self.final_train_loss: float | None = None
        self.heldout_latent_loss: float | None = None


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
        raise RuntimeError("expanded teacher-data artifact must include both train and heldout examples")

    base_variant = _LearnedVariant("learned_base", RetainedTeacherWarmStartConfigTF())
    wide_variant = _LearnedVariant(
        "learned_wide",
        RetainedTeacherWarmStartConfigTF(
            particle_hidden_dim=64,
            particle_hidden_layers=3,
            pooled_hidden_dim=64,
            pooled_hidden_layers=2,
        ),
    )
    learned_variants = [base_variant, wide_variant]
    for variant in learned_variants:
        _train_variant(variant, train_examples, heldout_examples)

    all_budgets = PRIMARY_BUDGETS + EXPLANATORY_BUDGETS
    budget_metrics = {
        str(budget): _budget_metrics_for_budget(
            budget,
            heldout_examples,
            teacher_policy,
            learned_variants,
        )
        for budget in all_budgets
    }

    decision = EXPECTED_DECISION
    for variant in learned_variants:
        if variant.final_train_loss is None or variant.initial_train_loss is None:
            decision = "RETAINED_TEACHER_SINKHORN_LOW_BUDGET_ABLATION_FAILED"
        elif variant.final_train_loss >= variant.initial_train_loss:
            decision = "RETAINED_TEACHER_SINKHORN_LOW_BUDGET_ABLATION_FAILED"
    if not _budgets_all_finite(budget_metrics):
        decision = "RETAINED_TEACHER_SINKHORN_LOW_BUDGET_ABLATION_FAILED"
    else:
        for budget in PRIMARY_BUDGETS:
            metrics = budget_metrics[str(budget)]
            zero = metrics["zero_init"]["mean_teacher_cloud_rmse"]
            heuristic = metrics["heuristic"]["mean_teacher_cloud_rmse"]
            learned_base = metrics["learned_base"]["mean_teacher_cloud_rmse"]
            if learned_base > zero or learned_base > heuristic:
                decision = "RETAINED_TEACHER_SINKHORN_LOW_BUDGET_ABLATION_FAILED"
                break

    payload = {
        "decision": decision,
        "question": "Under the expanded LGSSM retained-teacher artifact, how do zero-init, heuristic, learned-base, and learned-wide compare at low corrective budgets?",
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
        "learned_variants": {
            variant.name: {
                "config": {
                    "particle_hidden_dim": variant.config.particle_hidden_dim,
                    "particle_hidden_layers": variant.config.particle_hidden_layers,
                    "pooled_hidden_dim": variant.config.pooled_hidden_dim,
                    "pooled_hidden_layers": variant.config.pooled_hidden_layers,
                    "epsilon_feature_scale": variant.config.epsilon_feature_scale,
                },
                "initial_train_latent_loss": variant.initial_train_loss,
                "final_train_latent_loss": variant.final_train_loss,
                "heldout_latent_loss": variant.heldout_latent_loss,
            }
            for variant in learned_variants
        },
        "budget_metrics": budget_metrics,
        "run_manifest": environment_manifest(
            command="CUDA_VISIBLE_DEVICES=-1 python -m experiments.dpf_implementation.tf_tfp.runners.run_retained_teacher_sinkhorn_low_budget_ablation_tf",
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_implications": _non_implications(),
    }
    return payload


def _train_variant(
    variant: _LearnedVariant,
    train_examples: list[dict[str, Any]],
    heldout_examples: list[dict[str, Any]],
) -> None:
    tf.keras.utils.set_random_seed(TRAINING_SEED)
    _ = variant.model(train_examples[0]["particles"], train_examples[0]["weights"], epsilon=train_examples[0]["epsilon"])
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    variant.initial_train_loss = scalar(_mean_latent_loss(variant.model, train_examples))
    for _ in range(EPOCHS):
        with tf.GradientTape() as tape:
            loss = _mean_latent_loss(variant.model, train_examples)
        grads = tape.gradient(loss, variant.model.trainable_variables)
        optimizer.apply_gradients(zip(grads, variant.model.trainable_variables))
    variant.final_train_loss = scalar(_mean_latent_loss(variant.model, train_examples))
    variant.heldout_latent_loss = scalar(_mean_latent_loss(variant.model, heldout_examples))


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


def _heuristic_state(example: dict[str, Any]) -> SinkhornLogStateTF:
    log_weights = tf.math.log(tf.maximum(example["weights"], tf.constant(1e-300, dtype=tf.float64)))
    zeros = tf.zeros_like(log_weights)
    return canonicalize_sinkhorn_log_state_tf(log_weights, zeros)


def _budget_metrics_for_budget(
    budget: int,
    heldout_examples: list[dict[str, Any]],
    teacher_policy: dict[str, Any],
    learned_variants: list[_LearnedVariant],
) -> dict[str, Any]:
    tolerance = max(float(teacher_policy["teacher_tolerance"]), float(BUDGET_TOLERANCE_FLOORS[budget]))
    methods: dict[str, Callable[[dict[str, Any]], SinkhornLogStateTF | None]] = {
        "zero_init": lambda example: None,
        "heuristic": _heuristic_state,
        "learned_base": lambda example: predict_sinkhorn_initial_state_tf(learned_variants[0].model, example["particles"], example["weights"], example["epsilon"]),
        "learned_wide": lambda example: predict_sinkhorn_initial_state_tf(learned_variants[1].model, example["particles"], example["weights"], example["epsilon"]),
    }
    return {
        name: _evaluate_method(method, heldout_examples, budget, tolerance)
        for name, method in methods.items()
    }


def _evaluate_method(
    state_builder: Callable[[dict[str, Any]], SinkhornLogStateTF | None],
    heldout_examples: list[dict[str, Any]],
    budget: int,
    tolerance: float,
) -> dict[str, Any]:
    rmses = []
    residuals = []
    for example in heldout_examples:
        state = state_builder(example)
        kwargs = {}
        if state is not None:
            kwargs["initial_state"] = state
        result = sinkhorn_resample_tf(
            example["particles"],
            example["weights"],
            epsilon=example["epsilon"],
            max_iterations=budget,
            tolerance=tolerance,
            **kwargs,
        )
        rmses.append(rmse_tf(result.particles, example["teacher_barycentric"]))
        residuals.append(
            max(
                float(result.diagnostics["max_row_residual"]),
                float(result.diagnostics["max_column_residual"]),
                float(result.diagnostics["total_mass_residual"]),
            )
        )
    return {
        "mean_teacher_cloud_rmse": float(sum(rmses) / len(rmses)),
        "max_teacher_cloud_rmse": float(max(rmses)),
        "max_residual": float(max(residuals)),
        "heldout_example_count": len(heldout_examples),
        "evaluation_tolerance": float(tolerance),
    }


def _budgets_all_finite(budget_metrics: dict[str, dict[str, Any]]) -> bool:
    values = []
    for method_table in budget_metrics.values():
        for metrics in method_table.values():
            values.extend(
                [
                    metrics["mean_teacher_cloud_rmse"],
                    metrics["max_teacher_cloud_rmse"],
                    metrics["max_residual"],
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
    for variant_name, metrics in payload["learned_variants"].items():
        if metrics["final_train_latent_loss"] >= metrics["initial_train_latent_loss"]:
            raise RuntimeError(f"{variant_name} did not improve train latent loss")
    for budget in PRIMARY_BUDGETS:
        metrics = payload["budget_metrics"][str(budget)]
        learned_base = metrics["learned_base"]["mean_teacher_cloud_rmse"]
        if learned_base > metrics["zero_init"]["mean_teacher_cloud_rmse"]:
            raise RuntimeError(f"learned_base is worse than zero_init at budget {budget}")
        if learned_base > metrics["heuristic"]["mean_teacher_cloud_rmse"]:
            raise RuntimeError(f"learned_base is worse than heuristic at budget {budget}")
    if "reproducibility_digest" not in payload:
        raise RuntimeError("missing reproducibility digest")


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Retained-Teacher Sinkhorn Low-Budget Warm-Start Ablation Result",
        "",
        "## Decision",
        "",
        f"`{payload['decision']}`",
        "",
        "## Decision Table",
        "",
        "| Budget | Method | Mean RMSE | Max RMSE | Max residual |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for budget, method_table in payload["budget_metrics"].items():
        for method, metrics in method_table.items():
            lines.append(
                f"| `{budget}` | `{method}` | `{metrics['mean_teacher_cloud_rmse']:.3e}` | `{metrics['max_teacher_cloud_rmse']:.3e}` | `{metrics['max_residual']:.3e}` |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This rung attributes the low-budget effect across zero-init, heuristic, learned-base, and learned-wide warm-start regimes on the expanded LGSSM artifact. Passing here means the learned-base model remains competitive with both zero-init and heuristic at the primary budgets.",
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
        "No universal architecture recommendation is concluded beyond this envelope.",
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
