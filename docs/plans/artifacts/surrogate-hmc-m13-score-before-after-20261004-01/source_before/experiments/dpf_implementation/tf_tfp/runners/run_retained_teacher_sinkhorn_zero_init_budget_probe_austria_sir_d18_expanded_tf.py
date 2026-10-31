"""Probe zero-init corrected retained-Sinkhorn budgets on the expanded Austria SIR d18 teacher artifact."""

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

from experiments.dpf_implementation.tf_tfp.resampling.sinkhorn_tf import sinkhorn_resample_tf
from experiments.dpf_implementation.tf_tfp.runners.common_tf import (
    OUTPUT_DIR,
    REPORT_DIR,
    REPO_ROOT,
    environment_manifest,
    load_json,
    rmse_tf,
    stable_digest,
    utc_now,
    wall_time_call,
    write_json,
    write_text,
)


INPUT_JSON_PATH = OUTPUT_DIR / "retained_teacher_sinkhorn_teacher_data_austria_sir_d18_expanded_2026-07-05.json"
JSON_PATH = OUTPUT_DIR / "retained_teacher_sinkhorn_zero_init_budget_probe_austria_sir_d18_expanded_2026-07-05.json"
REPORT_PATH = REPORT_DIR / "retained-teacher-sinkhorn-zero-init-budget-probe-austria-sir-d18-expanded-2026-07-05.md"
RESULT_PATH = "docs/plans/bayesfilter-neural-ot-austria-sir-d18-retained-teacher-expanded-zero-init-probe-result-2026-07-05.md"
PLAN_PATH = "docs/plans/bayesfilter-neural-ot-austria-sir-d18-retained-teacher-nonpromotion-disambiguation-plan-2026-07-05.md"
SCALAR_ID = "retained_teacher_sinkhorn_zero_init_budget_probe_austria_sir_d18_expanded_tf"
PROBE_BUDGETS = (1, 2, 3, 5)
BUDGET_TOLERANCE_FLOORS = {
    1: 1e-3,
    2: 3e-4,
    3: 1e-4,
    5: 1e-5,
}
SATURATION_THRESHOLD = 1e-12
NO_RUNG_DECISION = "NO_USABLE_AUSTRIA_SIR_D18_EXPANDED_RANKING_RUNG_UNDER_CURRENT_ARTIFACT"
DISCRIMINATING_DECISION = "AUSTRIA_SIR_D18_EXPANDED_ZERO_INIT_DISCRIMINATING_BUDGETS_IDENTIFIED"


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
    heldout_examples = [example for example in examples if example["split"] == "heldout"]
    if not heldout_examples:
        raise RuntimeError("expanded Austria SIR d18 teacher-data artifact must include heldout examples")

    budget_metrics = {
        str(budget): _budget_metrics(
            heldout_examples,
            budget,
            tolerance=max(float(teacher_policy["teacher_tolerance"]), float(BUDGET_TOLERANCE_FLOORS[budget])),
        )
        for budget in PROBE_BUDGETS
    }
    budget_metrics = _annotate_budget_regimes(budget_metrics)
    discriminating_budgets = [
        budget for budget in PROBE_BUDGETS if budget_metrics[str(budget)]["budget_regime"] == "discriminating"
    ]
    saturated_budgets = [
        budget for budget in PROBE_BUDGETS if budget_metrics[str(budget)]["budget_regime"] == "saturated_zero_init"
    ]
    recommended_primary_budgets = discriminating_budgets[-2:]
    recommended_explanatory_budgets = saturated_budgets[:1]
    decision = DISCRIMINATING_DECISION if discriminating_budgets else NO_RUNG_DECISION

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
        "question": "Does the expanded Austria SIR d18 retained-teacher artifact contain any discriminating zero-init corrective budget before saturation?",
        "created_at_utc": utc_now(),
        "backend": "tensorflow_tensorflow_probability",
        "scalar_id": SCALAR_ID,
        "plan_path": PLAN_PATH,
        "input_teacher_data_json": str(INPUT_JSON_PATH.relative_to(REPO_ROOT)),
        "teacher_data_reproducibility_digest": teacher_payload.get("reproducibility_digest"),
        "probe_contract": {
            "probe_budgets": list(PROBE_BUDGETS),
            "budget_tolerance_floors": {str(k): v for k, v in BUDGET_TOLERANCE_FLOORS.items()},
            "teacher_epsilon": representative_epsilon,
            "teacher_tolerance": teacher_policy["teacher_tolerance"],
            "heldout_examples": len(heldout_examples),
            "saturation_threshold": SATURATION_THRESHOLD,
        },
        "budget_metrics": budget_metrics,
        "probe_summary": {
            "discriminating_budgets": discriminating_budgets,
            "saturated_budgets": saturated_budgets,
            "recommended_primary_budgets": recommended_primary_budgets,
            "recommended_explanatory_budgets": recommended_explanatory_budgets,
            "first_saturated_budget": saturated_budgets[0] if saturated_budgets else None,
        },
        "run_manifest": environment_manifest(
            command="CUDA_VISIBLE_DEVICES=-1 python -m experiments.dpf_implementation.tf_tfp.runners.run_retained_teacher_sinkhorn_zero_init_budget_probe_austria_sir_d18_expanded_tf",
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
        "teacher_barycentric": tf.convert_to_tensor(teacher["barycentric_particles"], dtype=tf.float64),
    }


def _budget_metrics(
    heldout_examples: list[dict[str, Any]],
    corrective_budget: int,
    *,
    tolerance: float,
) -> dict[str, Any]:
    zero_rmses = []
    zero_residuals = []
    for example in heldout_examples:
        zero_result = sinkhorn_resample_tf(
            example["particles"],
            example["weights"],
            epsilon=example["epsilon"],
            max_iterations=corrective_budget,
            tolerance=tolerance,
        )
        zero_rmses.append(rmse_tf(zero_result.particles, example["teacher_barycentric"]))
        zero_residuals.append(_max_residual(zero_result.diagnostics))
    return {
        "mean_zero_teacher_cloud_rmse": float(sum(zero_rmses) / len(zero_rmses)),
        "max_zero_teacher_cloud_rmse": float(max(zero_rmses)),
        "max_zero_residual": float(max(zero_residuals)),
        "heldout_example_count": len(heldout_examples),
        "evaluation_tolerance": float(tolerance),
    }


def _annotate_budget_regimes(budget_metrics: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    annotated = {}
    for budget, metrics in budget_metrics.items():
        row = dict(metrics)
        saturated = row["mean_zero_teacher_cloud_rmse"] <= SATURATION_THRESHOLD
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
                metrics["mean_zero_teacher_cloud_rmse"],
                metrics["max_zero_teacher_cloud_rmse"],
                metrics["max_zero_residual"],
            ]
        )
    return all(tf.math.is_finite(tf.constant(value, dtype=tf.float64)).numpy() for value in values)


def _validate_payload(payload: dict[str, Any]) -> None:
    if payload["decision"] not in {DISCRIMINATING_DECISION, NO_RUNG_DECISION}:
        raise RuntimeError(payload["decision"])
    if payload["run_manifest"]["pre_import_cuda_visible_devices"] != "-1":
        raise RuntimeError("missing CPU-only pre-import manifest")
    if payload["scalar_id"] != SCALAR_ID:
        raise RuntimeError("wrong scalar id")
    if not _budgets_all_finite(payload["budget_metrics"]):
        raise RuntimeError("non-finite budget metrics")
    discriminating_budgets = payload["probe_summary"]["discriminating_budgets"]
    recommended_primary_budgets = payload["probe_summary"]["recommended_primary_budgets"]
    if payload["decision"] == DISCRIMINATING_DECISION:
        if not discriminating_budgets:
            raise RuntimeError("missing discriminating budgets")
        if not recommended_primary_budgets:
            raise RuntimeError("missing recommended primary budgets")
    if payload["decision"] == NO_RUNG_DECISION and discriminating_budgets:
        raise RuntimeError("no-rung decision inconsistent with discriminating budgets")
    if "reproducibility_digest" not in payload:
        raise RuntimeError("missing reproducibility digest")


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Austria SIR d18 Expanded Retained-Teacher Zero-Init Budget Probe Result",
        "",
        "## Decision",
        "",
        f"`{payload['decision']}`",
        "",
        "## Decision Table",
        "",
        "| Budget | Regime | Zero-init mean RMSE | Zero-init max RMSE | Zero-init max residual | Heldout examples |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for budget in PROBE_BUDGETS:
        metrics = payload["budget_metrics"][str(budget)]
        lines.append(
            f"| `{budget}` | `{metrics['budget_regime']}` | `{metrics['mean_zero_teacher_cloud_rmse']:.3e}` | `{metrics['max_zero_teacher_cloud_rmse']:.3e}` | `{metrics['max_zero_residual']:.3e}` | `{metrics['heldout_example_count']}` |"
        )
    lines.extend(
        [
            "",
            "## Probe Summary",
            "",
            f"- Discriminating budgets: `{payload['probe_summary']['discriminating_budgets']}`",
            f"- Saturated budgets: `{payload['probe_summary']['saturated_budgets']}`",
            f"- Recommended primary budgets: `{payload['probe_summary']['recommended_primary_budgets']}`",
            f"- Recommended explanatory budgets: `{payload['probe_summary']['recommended_explanatory_budgets']}`",
            "",
            "## Interpretation",
            "",
            "The expanded Austria SIR d18 artifact still furnishes a governed discriminating rung, so the non-promotion disambiguation step can remain on the same ladder family rather than drifting into a ladder-calibration problem.",
            "",
            "## Non-Implications",
            "",
            _non_implications_markdown(),
        ]
    )
    return "\n".join(lines) + "\n"


def _non_implications() -> list[str]:
    return [
        "No donor-aligned student win/loss claim is concluded by this zero-init-only probe.",
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
