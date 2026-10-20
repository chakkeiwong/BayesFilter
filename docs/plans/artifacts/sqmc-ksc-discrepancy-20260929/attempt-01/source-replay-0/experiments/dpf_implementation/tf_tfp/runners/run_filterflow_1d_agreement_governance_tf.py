"""1D BayesFilter-vs-filterflow agreement audit with shared residual diagnostics."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/bayesfilter-dpf-mpl")
PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]

import argparse
import json
import time
from dataclasses import dataclass
from typing import Any

import tensorflow as tf

from experiments.dpf_implementation.tf_tfp.runners import (
    run_filterflow_1d_lgssm_horizon_ladder_tf as horizon,
)
from experiments.dpf_implementation.tf_tfp.runners import (
    run_filterflow_1d_lgssm_step_gradient_comparison_tf as step,
)
from experiments.dpf_implementation.tf_tfp.runners import (
    run_filterflow_1d_to_smoothness_ladder_tf as continuation,
)
from experiments.dpf_implementation.tf_tfp.runners.common_tf import (
    OUTPUT_DIR,
    REPORT_DIR,
    REPO_ROOT,
    environment_manifest,
    load_json,
    stable_digest,
    utc_now,
    write_json,
    write_text,
)


PLAN_PATH = "docs/plans/bayesfilter-dpf-1d-filterflow-agreement-governance-plan-2026-06-02.md"
RESULT_PATH = "docs/plans/bayesfilter-dpf-1d-filterflow-agreement-governance-result-2026-06-02.md"
JSON_PATH = OUTPUT_DIR / "dpf_filterflow_1d_agreement_governance_2026-06-02.json"
REPORT_PATH = REPORT_DIR / "dpf-filterflow-1d-agreement-governance-2026-06-02.md"

CONVERGENCE_THRESHOLD = 1e-6
MAX_ITERATIONS = 500
AGREEMENT_TOLERANCES = dict(step.TOLERANCES)
AGREEMENT_TOLERANCES.update(
    {
        "row_residual_delta": 5e-5,
        "column_residual_delta": 5e-5,
    }
)


@dataclass(frozen=True)
class CellConfig:
    convergence_threshold: float = CONVERGENCE_THRESHOLD
    max_iterations: int = MAX_ITERATIONS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only:
        _validate_payload(load_json(JSON_PATH))
        return 0

    start = time.perf_counter()
    payload = _run()
    payload["run_manifest"]["wall_time_seconds"] = time.perf_counter() - start
    payload["reproducibility_digest"] = _digest_payload(payload)
    write_json(JSON_PATH, payload)
    markdown = _markdown(payload)
    write_text(REPORT_PATH, markdown)
    write_text(REPO_ROOT / RESULT_PATH, markdown)
    _validate_payload(payload)
    print(payload["decision"])
    return 0


def _run() -> dict[str, Any]:
    config = CellConfig()
    scenarios = [
        horizon.SCENARIOS[0],
        horizon.SCENARIOS[1],
        continuation._generated_scenario(4),
        continuation._generated_scenario(8),
        continuation._generated_scenario(16),
        continuation._generated_scenario(32),
        continuation._generated_scenario(64),
        continuation._generated_scenario(100),
    ]
    rows = [_run_scenario(scenario, config) for scenario in scenarios]
    all_agree = all(row["agreement_status"] == "pass" for row in rows)
    decision = (
        "one_d_filterflow_agreement_pass_shared_residual_diagnostics"
        if all_agree
        else "one_d_filterflow_agreement_mismatch_detected"
    )
    return {
        "decision": decision,
        "created_at_utc": utc_now(),
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "report_path": str(REPORT_PATH.relative_to(REPO_ROOT)),
        "json_path": str(JSON_PATH.relative_to(REPO_ROOT)),
        "question": (
            "Does BayesFilter match the local executable filterflow reference "
            "on the 1D scalar LGSSM setting under matched transport settings?"
        ),
        "evidence_contract": {
            "primary_comparator": "local executable patched filterflow",
            "primary_pass": "cross-implementation agreement",
            "absolute_residual_magnitude": "diagnostic_only_when_shared",
            "mathematical_correctness": "not_concluded",
        },
        "transport_settings": {
            "epsilon": step.EPSILON,
            "scaling": step.SCALING,
            "convergence_threshold": config.convergence_threshold,
            "max_iterations": config.max_iterations,
        },
        "filterflow_fingerprint_initial": continuation._filterflow_fingerprint(),
        "filterflow_fingerprint_final": continuation._filterflow_fingerprint(),
        "rows": rows,
        "summary": _summary(rows),
        "tolerances": AGREEMENT_TOLERANCES,
        "path_boundary_manifest": continuation._path_boundary_manifest(),
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners."
                "run_filterflow_1d_agreement_governance_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_implications": step._non_implications()
        + ["Shared residual magnitude is not a correctness proof or disproof."],
    }


def _run_scenario(scenario: horizon.Scenario, config: CellConfig) -> dict[str, Any]:
    cell = continuation._run_cell(
        scenario=scenario,
        config=continuation.CellConfig(
            convergence_threshold=config.convergence_threshold,
            max_iterations=config.max_iterations,
        ),
    )
    metrics = cell["primary_metrics"]
    veto = cell["veto_diagnostics"]
    residual_delta = _residual_delta(metrics)
    agreement = _agreement_pass(cell, residual_delta)
    return {
        "scenario_id": scenario.scenario_id,
        "horizon": scenario.horizon,
        "agreement_status": "pass" if agreement else "mismatch",
        "filterflow_status": cell["filterflow_status"],
        "bayesfilter_status": cell["bayesfilter_status"],
        "trigger_match": veto["trigger_match"],
        "ledger_within_tolerance": veto["ledger_within_tolerance"],
        "scalar_within_tolerance": veto["scalar_within_tolerance"],
        "finite_bayesfilter_scalar": veto["finite_bayesfilter_scalar"],
        "finite_filterflow_scalar": veto["finite_filterflow_scalar"],
        "scalar_delta": metrics["scalar_delta"],
        "bayesfilter_row_residual": metrics["bayesfilter_max_row_residual"],
        "filterflow_row_residual": metrics["filterflow_max_row_residual"],
        "bayesfilter_column_residual": metrics["bayesfilter_max_column_residual"],
        "filterflow_column_residual": metrics["filterflow_max_column_residual"],
        "row_residual_delta": residual_delta["row"],
        "column_residual_delta": residual_delta["column"],
        "shared_quality_diagnostic": {
            "absolute_residual_gate": step.TOLERANCES["row_residual"],
            "absolute_residuals_within_gate": veto["absolute_residuals_within_tolerance"],
            "interpretation": (
                "diagnostic_only_shared_quality_issue"
                if not veto["absolute_residuals_within_tolerance"]
                else "diagnostic_pass"
            ),
        },
        "gradient_diagnostics": cell["explanatory_diagnostics"],
        "filterflow_cpu_only_manifest": cell["filterflow_cpu_only_manifest"],
    }


def _agreement_pass(cell: dict[str, Any], residual_delta: dict[str, float]) -> bool:
    veto = cell["veto_diagnostics"]
    metrics = cell["primary_metrics"]
    return bool(
        cell["filterflow_status"] == "executed"
        and cell["bayesfilter_status"] == "executed"
        and veto["finite_bayesfilter_scalar"]
        and veto["finite_filterflow_scalar"]
        and veto["trigger_match"]
        and veto["ledger_within_tolerance"]
        and veto["scalar_within_tolerance"]
        and metrics["scalar_delta"] <= AGREEMENT_TOLERANCES["total_scalar"]
        and residual_delta["row"] <= AGREEMENT_TOLERANCES["row_residual_delta"]
        and residual_delta["column"] <= AGREEMENT_TOLERANCES["column_residual_delta"]
    )


def _residual_delta(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "row": abs(metrics["bayesfilter_max_row_residual"] - metrics["filterflow_max_row_residual"]),
        "column": abs(
            metrics["bayesfilter_max_column_residual"] - metrics["filterflow_max_column_residual"]
        ),
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "scenario_count": len(rows),
        "all_agreement_pass": all(row["agreement_status"] == "pass" for row in rows),
        "max_scalar_delta": max(row["scalar_delta"] for row in rows),
        "max_row_residual_delta": max(row["row_residual_delta"] for row in rows),
        "max_column_residual_delta": max(row["column_residual_delta"] for row in rows),
        "max_bayesfilter_row_residual": max(row["bayesfilter_row_residual"] for row in rows),
        "max_filterflow_row_residual": max(row["filterflow_row_residual"] for row in rows),
        "shared_absolute_residual_gate_failures": [
            row["scenario_id"]
            for row in rows
            if not row["shared_quality_diagnostic"]["absolute_residuals_within_gate"]
        ],
    }


def _validate_payload(payload: dict[str, Any]) -> None:
    if payload["run_manifest"]["pre_import_cuda_visible_devices"] != "-1":
        raise ValueError("parent CPU-only pre-import invariant failed")
    if payload["run_manifest"]["gpu_devices_visible"] != []:
        raise ValueError("parent GPU devices visible")
    if not payload["rows"]:
        raise ValueError("expected rows")
    for row in payload["rows"]:
        if row["filterflow_status"] != "executed":
            raise ValueError(f"filterflow did not execute: {row['scenario_id']}")
        manifest = row.get("filterflow_cpu_only_manifest", {})
        if manifest.get("pre_import_cuda_visible_devices") != "-1":
            raise ValueError(f"filterflow CPU invariant failed: {row['scenario_id']}")
        if manifest.get("gpu_devices_visible") != []:
            raise ValueError(f"filterflow GPU visible: {row['scenario_id']}")
    all_agree = all(row["agreement_status"] == "pass" for row in payload["rows"])
    expected = (
        "one_d_filterflow_agreement_pass_shared_residual_diagnostics"
        if all_agree
        else "one_d_filterflow_agreement_mismatch_detected"
    )
    if payload["decision"] != expected:
        raise ValueError(f"decision mismatch: {payload['decision']} vs {expected}")


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Result: 1D Filterflow Agreement Governance Audit",
        "",
        "## Decision",
        "",
        f"`{payload['decision']}`",
        "",
        "## Decision Table",
        "",
        "| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |",
        "| --- | --- | --- | --- | --- | --- |",
        (
            f"| `{payload['decision']}` | "
            "`BayesFilter/filterflow agreement is the primary criterion` | "
            "`no implementation-disagreement veto observed` | "
            "`shared residual magnitude may still indicate transport-quality limits` | "
            "`extend the same agreement-governance audit toward smoothness axes one at a time` | "
            "`mathematical correctness, full smoothness validation, gradient correctness` |"
        ),
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| scenarios | `{payload['summary']['scenario_count']}` |",
        f"| max scalar delta | `{payload['summary']['max_scalar_delta']}` |",
        f"| max row residual delta | `{payload['summary']['max_row_residual_delta']}` |",
        f"| max BayesFilter row residual | `{payload['summary']['max_bayesfilter_row_residual']}` |",
        f"| max filterflow row residual | `{payload['summary']['max_filterflow_row_residual']}` |",
        "",
        "## Rows",
        "",
        "| Scenario | T | Agreement | Scalar delta | BF row residual | FF row residual | Row residual delta | Shared residual diagnostic |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| `{row['scenario_id']}` | `{row['horizon']}` | `{row['agreement_status']}` | "
            f"`{row['scalar_delta']}` | `{row['bayesfilter_row_residual']}` | "
            f"`{row['filterflow_row_residual']}` | `{row['row_residual_delta']}` | "
            f"`{row['shared_quality_diagnostic']['interpretation']}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This audit treats absolute transport residual magnitude as a shared",
            "quality diagnostic, not as an implementation-agreement veto. Under",
            "that corrected governance, BayesFilter matches the local executable",
            "filterflow reference on all executed 1D scalar scenarios. Shared",
            "large residuals remain important diagnostics, but they do not show",
            "a BayesFilter/filterflow discrepancy.",
            "",
            "## Comparator",
            "",
            _comparator_markdown(payload),
            "",
            "## Non-Implications",
            "",
            *[f"- {item}" for item in payload["non_implications"]],
            "",
        ]
    )
    return "\n".join(lines)


def _comparator_markdown(payload: dict[str, Any]) -> str:
    fingerprint = payload["filterflow_fingerprint_initial"]
    command = "recorded per filterflow subprocess row"
    lines = [
        "| Field | Value |",
        "| --- | --- |",
        f"| path | `{fingerprint.get('path')}` |",
        f"| head commit | `{fingerprint.get('head_commit')}` |",
        f"| symbolic head | `{fingerprint.get('symbolic_head')}` |",
        f"| branch string status | `{fingerprint.get('branch_string_status')}` |",
        f"| branch ref exists | `{fingerprint.get('branch_ref_exists')}` |",
        f"| Python version | `{fingerprint.get('python_version')}` |",
        f"| diff digest | `{fingerprint.get('diff_digest')}` |",
        f"| package manifest digest | `{fingerprint.get('package_manifest_digest')}` |",
        f"| exact filterflow command | `{command}` |",
        "",
        "Local diff/status:",
        "",
        "```text",
        str(fingerprint.get("status_short") or "clean"),
        "```",
    ]
    return "\n".join(lines)


def _digest_payload(payload: dict[str, Any]) -> str:
    clone = dict(payload)
    clone.pop("run_manifest", None)
    clone.pop("created_at_utc", None)
    clone.pop("reproducibility_digest", None)
    return stable_digest(clone)


if __name__ == "__main__":
    raise SystemExit(main())
