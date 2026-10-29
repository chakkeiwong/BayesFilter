"""Full comparison audit for BayesFilter OT-DPF versus filterflow."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]

import argparse
import time
from typing import Any

from experiments.dpf_implementation.tf_tfp.runners import (
    run_filterflow_final_gaps_closure_tf as final_gaps,
)
from experiments.dpf_implementation.tf_tfp.runners import (
    run_filterflow_lgssm_matched_cross_audit_tf as matched,
)
from experiments.dpf_implementation.tf_tfp.runners.common_tf import (
    OUTPUT_DIR,
    REPORT_DIR,
    environment_manifest,
    load_json,
    stable_digest,
    utc_now,
    write_json,
    write_text,
)
from experiments.dpf_implementation.tf_tfp.runners.filterflow_reference_policy import (
    FILTERFLOW_REFERENCE_BRANCH,
    FILTERFLOW_REFERENCE_COMMIT,
    validate_filterflow_reference_status,
)


PLAN_PATH = "docs/plans/bayesfilter-dpf-filterflow-full-comparison-plan-2026-05-31.md"
JSON_PATH = OUTPUT_DIR / "dpf_filterflow_full_comparison_2026-05-31.json"
REPORT_PATH = REPORT_DIR / "dpf-filterflow-full-comparison-2026-05-31.md"
FILTERFLOW_BRANCH = FILTERFLOW_REFERENCE_BRANCH
FILTERFLOW_COMMIT = FILTERFLOW_REFERENCE_COMMIT


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
    write_text(REPORT_PATH, _markdown(payload))
    _validate_payload(payload)
    print(payload["decision"])
    return 0


def _run() -> dict[str, Any]:
    matched_payload = matched._run()
    final_payload = final_gaps._run()
    matrix = _comparison_matrix(matched_payload, final_payload)
    red_flags = _red_flags(matched_payload, final_payload)
    discrepancy_ledger = _discrepancy_ledger(matched_payload, final_payload)
    decision = _decision(matched_payload, final_payload, matrix, red_flags)
    return {
        "decision": decision,
        "created_at_utc": utc_now(),
        "plan_path": PLAN_PATH,
        "question": (
            "Full LGSSM comparison of BayesFilter experimental TF/TFP OT-DPF lanes "
            "against patched executable Corenflos/JTT94 filterflow."
        ),
        "filterflow_reference": {
            "branch": matched_payload["filterflow_status"]["branch"],
            "commit": matched_payload["filterflow_status"]["commit"],
            "upstream_base": matched_payload["filterflow_status"]["upstream_base"],
            "status": matched_payload["filterflow_status"]["status"],
            "diff_summary": matched_payload["filterflow_status"]["diff_summary"],
            "reference_status": (
                "local BayesFilter float64 filterflow reference branch, not pristine upstream source"
            ),
        },
        "paper_code_setting_ledger": final_payload["paper_code_ledger"],
        "source_support_ledger": final_payload["source_support_ledger"],
        "matched_lgssm_audit": {
            "decision": matched_payload["decision"],
            "settings": matched_payload["settings"],
            "kalman_alignment": matched_payload["kalman_alignment"],
            "comparison": matched_payload["comparison"],
            "scaling_and_epsilon_ledger": matched_payload["scaling_and_epsilon_ledger"],
            "discrepancy_ledger": matched_payload["discrepancy_ledger"],
        },
        "smoothness_gradient_diagnostics": final_payload["smoothness_gradient_replication"],
        "fixed_target_sinkhorn_diagnostics": final_payload["fixed_target_sinkhorn_diagnosis"],
        "comparison_matrix": matrix,
        "discrepancy_ledger": discrepancy_ledger,
        "red_flag_ledger": red_flags,
        "method_status_summary": _method_status_summary(matched_payload, final_payload),
        "claude_review": {
            "status": "prior_full_comparison_review_reconciled_no_pending_placeholder",
            "protocol": "claude -p --model claude-opus-4-7 --effort max",
            "max_iterations": 5,
            "reconciled_by": "annealed transport reference-alignment result review round 5",
        },
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners.run_filterflow_full_comparison_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_implications": _non_implications(),
    }


def _comparison_matrix(
    matched_payload: dict[str, Any],
    final_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    filterflow_style_status = _comparison_status(
        matched_payload["comparison"], "bayesfilter_filterflow_style_transport_ess"
    )
    fixed_status = _comparison_status(
        matched_payload["comparison"], "bayesfilter_scaled_fixed_sinkhorn_ess"
    )
    pf_status = _comparison_status(matched_payload["comparison"], "bayesfilter_pf")
    smooth = final_payload["smoothness_gradient_replication"]
    fixed_diag = final_payload["fixed_target_sinkhorn_diagnosis"]
    return [
        {
            "lane": "paper_table_1",
            "source": "Corenflos main paper",
            "status": "context_only",
            "primary_scalar": "per_time_log_likelihood_error",
            "interpretation": "published context, not sole authority for executable comparison",
        },
        {
            "lane": "exact_kalman",
            "source": "filterflow_and_bayesfilter",
            "status": (
                "pass"
                if matched_payload["kalman_alignment"]["all_within_tolerance"]
                else "veto"
            ),
            "primary_scalar": "lgssm_log_likelihood",
            "max_abs_delta": matched_payload["kalman_alignment"]["max_abs_delta"],
        },
        {
            "lane": "filterflow_pf",
            "source": "patched_filterflow",
            "status": "executed",
            "primary_scalar": "per_time_log_likelihood_error",
            "interpretation": "external classical baseline",
        },
        {
            "lane": "bayesfilter_pf",
            "source": "tf_tfp",
            "status": pf_status,
            "primary_scalar": "per_time_log_likelihood_error",
            "interpretation": "internal bootstrap PF calibration against filterflow PF",
        },
        {
            "lane": "filterflow_regularised_transform",
            "source": "patched_filterflow",
            "status": "executed",
            "primary_scalar": "per_time_log_likelihood_error",
            "interpretation": "external DPF reference",
        },
        {
            "lane": "bayesfilter_filterflow_style_transport",
            "source": "tf_tfp_audit_mirror",
            "status": filterflow_style_status,
            "primary_scalar": "per_time_log_likelihood_error",
            "interpretation": "matched paper-style annealed regularized transport semantics",
        },
        {
            "lane": "bayesfilter_fixed_target_sinkhorn",
            "source": "tf_tfp",
            "status": fixed_status,
            "primary_scalar": "per_time_log_likelihood_error_plus_sinkhorn_residuals",
            "diagnostic_status": fixed_diag["status"],
            "interpretation": "diagnostic branch, not filterflow-equivalent",
        },
        {
            "lane": "bayesfilter_ledh_pfpf_ot",
            "source": "tf_tfp",
            "status": "structured_not_run_as_matched_filterflow_table_lane",
            "primary_scalar": "not_available_under_matched_filterflow_protocol",
            "interpretation": (
                "Existing LEDH-PF-PF-OT LGSSM runner uses the repo fixture and proposal-correction "
                "diagnostics; it was not promoted into this filterflow table because this comparison "
                "requires the exact filterflow observation path, covariance convention, seed protocol, "
                "and per-time likelihood-error scalar."
            ),
        },
        {
            "lane": "smoothness_gradient_contract",
            "source": "patched_filterflow_and_kalman_fd_reference",
            "status": _smoothness_status(smooth),
            "primary_scalar": "bounded_gradient_smoke",
            "gradient_rmse": smooth.get("gradient_rmse"),
            "gradient_cosine_vs_kalman_fd": smooth.get("gradient_cosine_vs_kalman_fd"),
            "interpretation": (
                "finite-gradient smoke only; severe unresolved scalar/randomness/"
                "gradient-magnitude mismatch remains"
            ),
        },
    ]


def _method_status_summary(
    matched_payload: dict[str, Any],
    final_payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "matched_audit_decision": matched_payload["decision"],
        "final_gaps_decision": final_payload["decision"],
        "bayesfilter_pf_status": _comparison_status(matched_payload["comparison"], "bayesfilter_pf"),
        "bayesfilter_filterflow_style_transport_status": _comparison_status(
            matched_payload["comparison"], "bayesfilter_filterflow_style_transport_ess"
        ),
        "bayesfilter_fixed_target_sinkhorn_status": _comparison_status(
            matched_payload["comparison"], "bayesfilter_scaled_fixed_sinkhorn_ess"
        ),
        "fixed_target_sinkhorn_diagnostic_status": final_payload["fixed_target_sinkhorn_diagnosis"][
            "status"
        ],
        "smoothness_gradient_status": _smoothness_status(
            final_payload["smoothness_gradient_replication"]
        ),
    }


def _red_flags(
    matched_payload: dict[str, Any],
    final_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    fixed_status = _comparison_status(
        matched_payload["comparison"], "bayesfilter_scaled_fixed_sinkhorn_ess"
    )
    smooth = final_payload["smoothness_gradient_replication"]
    flags = [
        {
            "id": "paper_transition_covariance_ambiguity",
            "severity": "medium",
            "status": "recorded_and_controlled",
            "detail": (
                "Paper/supplement text says 0.5 I_2; executable filterflow uses I_2 and prior "
                "bounded rerun showed Table 1 scale matches executable I_2."
            ),
        },
        {
            "id": "patched_filterflow_not_pristine",
            "severity": "medium",
            "status": "recorded",
            "detail": "Local filterflow is a Python 3.11 compatibility branch.",
        },
        {
            "id": "fixed_target_sinkhorn_not_paper_equivalent",
            "severity": "high" if fixed_status != "within_filterflow_mc_band" else "medium",
            "status": final_payload["fixed_target_sinkhorn_diagnosis"]["status"],
            "detail": (
                "BayesFilter fixed-target Sinkhorn remains a diagnostic branch and must not be "
                "read as Corenflos/filterflow annealed regularized transport."
            ),
        },
        {
            "id": "smoothness_gradient_severe_unreconciled_magnitude_mismatch",
            "severity": "high",
            "status": _smoothness_status(smooth),
            "detail": (
                "Bounded smoothness run produced finite gradients, but severe scalar/randomness/"
                "gradient-magnitude reconciliation gaps against Kalman finite-difference "
                "diagnostics remain."
            ),
        },
    ]
    return flags


def _discrepancy_ledger(
    matched_payload: dict[str, Any],
    final_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    ledger = [
        {
            "id": "kalman_alignment",
            "status": (
                "pass"
                if matched_payload["kalman_alignment"]["all_within_tolerance"]
                else "veto"
            ),
            "detail": f"max_abs_delta={matched_payload['kalman_alignment']['max_abs_delta']:.6g}",
        },
        {
            "id": "pf_calibration",
            "status": _comparison_status(matched_payload["comparison"], "bayesfilter_pf"),
            "detail": "BayesFilter PF versus filterflow PF under matched LGSSM protocol.",
        },
        {
            "id": "paper_style_transport_match",
            "status": _comparison_status(
                matched_payload["comparison"], "bayesfilter_filterflow_style_transport_ess"
            ),
            "detail": "BayesFilter audit mirror versus filterflow RegularisedTransform.",
        },
        {
            "id": "fixed_target_sinkhorn_branch",
            "status": _comparison_status(
                matched_payload["comparison"], "bayesfilter_scaled_fixed_sinkhorn_ess"
            ),
            "detail": final_payload["fixed_target_sinkhorn_diagnosis"]["status"],
        },
        {
            "id": "gradient_smoothness",
            "status": _smoothness_status(final_payload["smoothness_gradient_replication"]),
            "detail": (
                "Finite-gradient smoke only; severe scalar/randomness/gradient-magnitude "
                "mismatch remains unreconciled."
            ),
        },
        {
            "id": "ledh_pfpf_ot_matched_table",
            "status": "not_run_structured_scope_limit",
            "detail": "No matched filterflow-table scalar was promoted for LEDH-PF-PF-OT in this audit.",
        },
    ]
    return ledger


def _decision(
    matched_payload: dict[str, Any],
    final_payload: dict[str, Any],
    matrix: list[dict[str, Any]],
    red_flags: list[dict[str, Any]],
) -> str:
    del matrix
    try:
        validate_filterflow_reference_status(matched_payload["filterflow_status"])
    except RuntimeError:
        return "blocked_filterflow_reference_mismatch"
    if not matched_payload["kalman_alignment"]["all_within_tolerance"]:
        return "blocked_kalman_mismatch"
    if matched_payload["decision"] != "filterflow_style_transport_matched":
        return "red_flag_filterflow_style_transport_not_matched"
    if final_payload["decision"] != "final_gaps_closed_unconditional_fixed_sinkhorn_compute_gap_identified":
        return "red_flag_final_gaps_not_closed"
    if any(flag["status"] == "veto" for flag in red_flags):
        return "red_flag_veto_present"
    return "full_comparison_filterflow_reference_matched_with_fixed_sinkhorn_diagnostic_gap"


def _comparison_status(comparison: list[dict[str, Any]], method_id: str) -> str:
    rows = [row for row in comparison if row["bayesfilter_method"] == method_id]
    if not rows:
        return "missing"
    if any(row["bayesfilter_status"] != "executed" for row in rows):
        return "bayesfilter_veto_or_missing"
    if all(row["within_one_filterflow_sd"] for row in rows):
        return "within_filterflow_mc_band"
    return "outside_filterflow_mc_band"


def _smoothness_status(smooth: dict[str, Any]) -> str:
    if smooth.get("status") != "executed":
        return smooth.get("status", "blocked")
    if smooth.get("finite_likelihoods") and smooth.get("finite_gradients"):
        return "finite_gradient_smoke_with_severe_unreconciled_magnitude_mismatch"
    return "veto_nonfinite_smoothness_gradient"


def _markdown(payload: dict[str, Any]) -> str:
    return f"""# Filterflow Full Comparison For BayesFilter OT-DPF

## Decision

`{payload['decision']}`

## Scope

This result consolidates the matched executable-filterflow LGSSM audit and the
remaining final-gaps diagnostics into one full comparison artifact.  It uses the
patched local filterflow branch as external reference code and keeps BayesFilter
implementation evidence under the experimental TF/TFP DPF lane.

## Filterflow Reference

- Branch: `{payload['filterflow_reference']['branch']}`
- Commit: `{payload['filterflow_reference']['commit']}`
- Upstream base: `{payload['filterflow_reference']['upstream_base']}`
- Status: `{payload['filterflow_reference']['status']}`
- Diff summary: `{payload['filterflow_reference']['diff_summary']}`
- Reference status: {payload['filterflow_reference']['reference_status']}

## Comparison Matrix

{_matrix_table(payload['comparison_matrix'])}

## Method Status Summary

{_key_value_table(payload['method_status_summary'])}

## Matched LGSSM Table Comparison

{matched._comparison_table(payload['matched_lgssm_audit']['comparison'])}

## Red Flags

{_red_flag_table(payload['red_flag_ledger'])}

## Discrepancy Ledger

{_ledger_table(payload['discrepancy_ledger'])}

## Smoothness Gradient Diagnostics

{_smoothness_markdown(payload['smoothness_gradient_diagnostics'])}

## Fixed-Target Sinkhorn Diagnostics

{_fixed_sinkhorn_markdown(payload['fixed_target_sinkhorn_diagnostics'])}

## Non-Implications

{_non_implications_markdown(payload['non_implications'])}
"""


def _matrix_table(rows: list[dict[str, Any]]) -> str:
    lines = ["| Lane | Source | Status | Primary scalar | Interpretation |", "| --- | --- | --- | --- | --- |"]
    for row in rows:
        lines.append(
            "| `{lane}` | {source} | `{status}` | {scalar} | {interp} |".format(
                lane=row["lane"],
                source=row["source"],
                status=row["status"],
                scalar=row["primary_scalar"],
                interp=row.get("interpretation", ""),
            )
        )
    return "\n".join(lines)


def _key_value_table(values: dict[str, Any]) -> str:
    lines = ["| Key | Value |", "| --- | --- |"]
    for key, value in values.items():
        lines.append(f"| `{key}` | `{value}` |")
    return "\n".join(lines)


def _red_flag_table(rows: list[dict[str, Any]]) -> str:
    lines = ["| ID | Severity | Status | Detail |", "| --- | --- | --- | --- |"]
    for row in rows:
        lines.append(
            f"| `{row['id']}` | `{row['severity']}` | `{row['status']}` | {row['detail']} |"
        )
    return "\n".join(lines)


def _ledger_table(rows: list[dict[str, Any]]) -> str:
    lines = ["| ID | Status | Detail |", "| --- | --- | --- |"]
    for row in rows:
        lines.append(f"| `{row['id']}` | `{row['status']}` | {row['detail']} |")
    return "\n".join(lines)


def _smoothness_markdown(smooth: dict[str, Any]) -> str:
    if smooth.get("status") != "executed":
        return f"- Status: `{smooth.get('status')}`"
    return "\n".join(
        [
            f"- Status: `{_smoothness_status(smooth)}`",
            f"- Finite likelihoods: `{smooth.get('finite_likelihoods')}`",
            f"- Finite gradients: `{smooth.get('finite_gradients')}`",
            f"- Likelihood RMSE: `{smooth.get('likelihood_rmse')}`",
            f"- Gradient RMSE: `{smooth.get('gradient_rmse')}`",
            f"- Gradient max absolute delta: `{smooth.get('gradient_max_abs_delta')}`",
            f"- Gradient cosine vs Kalman finite difference: `{smooth.get('gradient_cosine_vs_kalman_fd')}`",
            f"- Gradient sign agreement: `{smooth.get('gradient_sign_agreement')}`",
        ]
    )


def _fixed_sinkhorn_markdown(fixed: dict[str, Any]) -> str:
    rows = fixed.get("rows", [])
    eps025 = [row for row in rows if row.get("epsilon") == 0.25]
    budget100 = next((row for row in eps025 if row.get("budget") == 100), None)
    budget500 = next((row for row in eps025 if row.get("budget") == 500), None)
    details = [
        f"- Status: `{fixed.get('status')}`",
        f"- Diagnostic scope: {fixed.get('interpretation')}",
    ]
    if budget100:
        details.append(f"- Epsilon 0.25 budget 100 max residual: `{budget100['max_residual']}`")
    if budget500:
        details.append(f"- Epsilon 0.25 budget 500 max residual: `{budget500['max_residual']}`")
    return "\n".join(details)


def _validate_payload(payload: dict[str, Any]) -> None:
    if payload["plan_path"] != PLAN_PATH:
        raise RuntimeError("wrong plan path")
    validate_filterflow_reference_status(payload["filterflow_reference"])
    if payload["matched_lgssm_audit"]["decision"] != "filterflow_style_transport_matched":
        raise RuntimeError("matched audit did not match filterflow-style transport")
    if not payload["matched_lgssm_audit"]["kalman_alignment"]["all_within_tolerance"]:
        raise RuntimeError("Kalman alignment failed")
    if payload["run_manifest"]["pre_import_cuda_visible_devices"] != "-1":
        raise RuntimeError("missing CPU-only manifest")
    if "reproducibility_digest" not in payload:
        raise RuntimeError("missing reproducibility digest")
    if not any(row["lane"] == "bayesfilter_fixed_target_sinkhorn" for row in payload["comparison_matrix"]):
        raise RuntimeError("missing fixed-target Sinkhorn matrix row")
    if not any(row["lane"] == "bayesfilter_ledh_pfpf_ot" for row in payload["comparison_matrix"]):
        raise RuntimeError("missing LEDH-PF-PF-OT scope row")
    if payload["decision"] != "full_comparison_filterflow_reference_matched_with_fixed_sinkhorn_diagnostic_gap":
        raise RuntimeError(payload["decision"])


def _digest_payload(payload: dict[str, Any]) -> str:
    comparable = dict(payload)
    comparable["created_at_utc"] = "TIMESTAMP"
    comparable["run_manifest"] = dict(comparable["run_manifest"])
    comparable["run_manifest"]["wall_time_seconds"] = "WALL_TIME"
    comparable["run_manifest"]["dirty_state_summary"] = "DIRTY_STATE"
    comparable["claude_review"] = dict(comparable["claude_review"])
    comparable["claude_review"]["status"] = "REVIEW_STATUS"
    return stable_digest(comparable)


def _non_implications() -> list[str]:
    return [
        "No production readiness is concluded.",
        "No public API readiness is concluded.",
        "No posterior correctness is concluded.",
        "No HMC readiness is concluded.",
        "No general nonlinear-SSM validity is concluded.",
        "No external macro-model validation is concluded.",
        "No banking/model-risk claim is concluded.",
        "No monograph claim is concluded.",
        "No claim that patched filterflow is pristine upstream source is concluded.",
        "No claim that fixed-target Sinkhorn is filterflow-equivalent is concluded.",
        "No full supplement figure or learning-table reproduction is concluded.",
    ]


def _non_implications_markdown(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


if __name__ == "__main__":
    raise SystemExit(main())
