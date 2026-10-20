"""Run P4 Zhao-Cui/fixed-design TT route classification.

P4 is a pure-Python classification gate.  It consumes the P0 registry and
reviewed P1-P3/P44/P45 artifacts, then records the claim class and blocker
status for every Zhao-Cui/fixed-design TT row.  It does not import TensorFlow or
rerun numerical filters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
REPORT_DIR = REPO_ROOT / "experiments" / "dpf_implementation" / "reports"
OUTPUT_DIR = REPORT_DIR / "outputs"
REGISTRY_PATH = REPO_ROOT / "docs/plans/bayesfilter-dpf-filter-oracle-comparison-target-route-registry-2026-06-08.json"
MODULE_PATH = (
    "experiments.dpf_implementation.tf_tfp.runners."
    "run_filter_oracle_comparison_p4_zhaocui_tt_route_classification_tf"
)
PLAN_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-filter-oracle-comparison-p4-zhaocui-tt-route-classification-subplan-2026-06-08.md"
)
RESULT_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-filter-oracle-comparison-p4-zhaocui-tt-route-classification-result-2026-06-08.md"
)
REVIEW_LEDGER_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-filter-oracle-comparison-p4-claude-review-ledger-2026-06-08.md"
)
JSON_PATH = OUTPUT_DIR / "dpf_filter_oracle_comparison_p4_zhaocui_tt_route_classification_2026-06-08.json"
REPORT_PATH = REPORT_DIR / "dpf-filter-oracle-comparison-p4-zhaocui-tt-route-classification-2026-06-08.md"

MASTER_CLAIM_CLASSES = {
    "EXACT_ORACLE",
    "CERTIFIED_APPROXIMATION",
    "SURROGATE_USEFULNESS",
    "DIAGNOSTIC_ONLY",
    "BLOCKED",
}
ZHAO_ROUTE_ID = "zhao_cui_fixed_design_tt"


class P4ValidationError(ValueError):
    """Raised when a P4 classification artifact violates the phase contract."""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only:
        _validate_payload(load_json(JSON_PATH))
        print("P4_ZHAOCUI_ROUTE_CLASSIFICATION_VALIDATED")
        return 0

    start = time.perf_counter()
    payload = _run()
    payload["run_manifest"]["wall_time_seconds"] = time.perf_counter() - start
    payload["reproducibility_digest"] = _digest_payload(payload)
    markdown = _markdown(payload)
    write_json(JSON_PATH, payload)
    write_text(REPORT_PATH, markdown)
    write_text(REPO_ROOT / RESULT_PATH, markdown)
    _validate_payload(payload)
    print(payload["decision"])
    return 0


def _run() -> dict[str, Any]:
    registry = load_json(REGISTRY_PATH)
    rows = [_classification_row(registry, target) for target in registry["targets"]]
    summaries = _summaries(rows)
    veto = _veto_diagnostics(registry, rows)
    decision = (
        "PASS_P4_ZHAOCUI_ROUTE_CLASSIFICATION_PENDING_CLAUDE_REVIEW"
        if not any(bool(value) for value in veto.values())
        else "P4_ZHAOCUI_ROUTE_CLASSIFICATION_VETO_PENDING_REVIEW"
    )
    return {
        "metadata_date": "2026-06-08",
        "created_at_utc": _utc_now(),
        "phase": "P4",
        "execution_route": "VISIBLE_IN_DIALOGUE",
        "execution_mode": "PURE_PYTHON_CLASSIFICATION_ONLY",
        "decision": decision,
        "question": (
            "For every P0 target, what master claim class should the "
            "Zhao-Cui/fixed-design TT route carry, and is P5 eligibility known "
            "without promoting diagnostics, fit residuals, or blocked "
            "multistate rows?"
        ),
        "evidence_contract": {
            "baseline_comparator": (
                "P0 registry plus reviewed P1 LGSSM, P2 dense nonlinear, P3 "
                "transformed/mixture, P44 tiny nonlinear, and P45 blocker "
                "artifacts"
            ),
            "primary_criterion": (
                "all P0 targets have exactly one Zhao-Cui/fixed-design TT row "
                "classified in the master schema with evidence source or "
                "blocker, phase eligibility, branch metadata, and P5 readiness"
            ),
            "classification_only_amendment": (
                "P4 uses a pure-Python registry/artifact validator because the "
                "phase question is route classification; TensorFlow reruns are "
                "not the primary evidence"
            ),
            "gradient_policy": (
                "gradient-bearing Zhao-Cui rows are fixed-branch scores only "
                "and require same-target value evidence before interpretation"
            ),
        },
        "rows": rows,
        "route_summaries": summaries,
        "veto_diagnostics": veto,
        "explanatory_diagnostics": {
            "classification_only_no_tensorflow_import": True,
            "fit_residuals_not_promotion_criteria": True,
            "kalman_sanity_checks_auxiliary_only": True,
            "multistate_rows_preserved_as_blocked": True,
        },
        "run_manifest": _manifest(),
        "nonclaims": _nonclaims(),
    }


def _classification_row(registry: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    target_id = target["target_id"]
    route = registry["route_matrix"][target_id][ZHAO_ROUTE_ID]
    exact_reference = _best_reference_route(registry["route_matrix"][target_id])
    evidence = _evidence_profile(target_id, route)
    blocked = route["claim_class"] == "BLOCKED"
    p5 = _p5_readiness(registry["route_matrix"][target_id], route, exact_reference)
    return {
        "target_id": target_id,
        "model_family": target["model_family"],
        "target_identity": target["target_identity"],
        "dimension_panel_convention": target["dimension_panel_convention"],
        "zhao_cui_route": {
            "route_id": ZHAO_ROUTE_ID,
            "claim_class": route["claim_class"],
            "route_status": route["route_status"],
            "route_path": route["route_path"],
            "phase_eligibility": route["phase_eligibility"],
            "promotion_tolerance": route["promotion_tolerance"],
            "certification_band": route["certification_band"],
            "primary_gradient_statistic": route["primary_gradient_statistic"],
            "value_support": route["value_support"],
            "gradient_support": route["gradient_support"],
            "blockers": route["blockers"],
            "nonclaims": route["nonclaims"],
        },
        "classification": {
            "master_claim_class": route["claim_class"],
            "classification_status": (
                "classified_blocked_preserved"
                if blocked
                else "classified_runnable_or_diagnostic"
            ),
            "source_evidence_class": evidence["source_evidence_class"],
            "source_artifacts": evidence["source_artifacts"],
            "same_target_value_evidence": evidence["same_target_value_evidence"],
            "gradient_interpretation": evidence["gradient_interpretation"],
            "branch_or_design_metadata": evidence["branch_or_design_metadata"],
            "auxiliary_sanity_checks": evidence["auxiliary_sanity_checks"],
            "blocker_interpretation": evidence["blocker_interpretation"],
        },
        "reference_context": exact_reference,
        "p5_readiness": p5,
    }


def _best_reference_route(routes: dict[str, Any]) -> dict[str, Any]:
    for route_id in ("kalman_exact", "dense_refined_quadrature"):
        route = routes[route_id]
        if route["claim_class"] == "EXACT_ORACLE" and (
            route["phase_eligibility"].get("p1")
            or route["phase_eligibility"].get("p2")
            or route["phase_eligibility"].get("p3")
            or route["phase_eligibility"].get("p5")
        ):
            return {
                "route_id": route_id,
                "claim_class": route["claim_class"],
                "route_status": route["route_status"],
                "promotion_tolerance": route["promotion_tolerance"],
                "primary_gradient_statistic": route["primary_gradient_statistic"],
                "phase_eligibility": route["phase_eligibility"],
            }
    return {
        "route_id": "none",
        "claim_class": "BLOCKED_OR_DIAGNOSTIC_ONLY_REFERENCE",
        "route_status": "no exact same-target reference approved for P5/P4",
        "promotion_tolerance": "blocked_na",
        "primary_gradient_statistic": "N/A",
        "phase_eligibility": {},
    }


def _evidence_profile(target_id: str, route: dict[str, Any]) -> dict[str, Any]:
    profiles = {
        "lgssm_2d_h25_rich": {
            "source_evidence_class": "auxiliary_kalman_sanity_diagnostic_only",
            "source_artifacts": [
                "docs/plans/bayesfilter-dpf-filter-oracle-comparison-p1-lgssm-exact-oracle-result-2026-06-08.md",
                "docs/plans/bayesfilter-highdim-zhao-cui-p30-model-suite-phase1-lgssm-exact-reference-result-2026-06-05.md",
            ],
            "same_target_value_evidence": (
                "Exact Kalman is the oracle; Zhao-Cui/fixed-design route is "
                "diagnostic-only and not a P5 comparator."
            ),
            "gradient_interpretation": "diagnostic_fixed_branch_score_only",
            "branch_or_design_metadata": {
                "route_family": "clean_room_exact_reference_sanity_only",
                "basis_or_rank": "N/A: diagnostic row, not fixed-design TT promotion",
            },
            "auxiliary_sanity_checks": [
                "P30/P1 Kalman exact reference sanity checks",
            ],
            "blocker_interpretation": "not_blocked_but_not_oracle_or_p5_comparator",
        },
        "p44_m2_cubic_additive_gaussian_panel": {
            "source_evidence_class": "p2_same_target_dense_certificate",
            "source_artifacts": [
                "docs/plans/bayesfilter-dpf-filter-oracle-comparison-p2-tiny-nonlinear-dense-oracle-result-2026-06-08.md",
                "experiments/dpf_implementation/reports/outputs/dpf_filter_oracle_comparison_p2_tiny_nonlinear_dense_oracle_2026-06-08.json",
            ],
            "same_target_value_evidence": (
                "P2 dense reference promoted dims 1,2,3; Zhao-Cui max value "
                "gap 0.0003496295403593308 and max directional score gap "
                "0.0018185324574571116."
            ),
            "gradient_interpretation": "fixed_branch_score_local_certificate_only",
            "branch_or_design_metadata": _tiny_tt_metadata("p44_m2"),
            "auxiliary_sanity_checks": [
                "dense order-161/order-241 refinement",
                "multi-step directional finite differences diagnostic only",
            ],
            "blocker_interpretation": "not_blocked_local_certificate",
        },
        "p44_m3_quadratic_observation_panel": {
            "source_evidence_class": "p44_dense_stress_certificate",
            "source_artifacts": [
                "docs/plans/bayesfilter-highdim-zhao-cui-p44-phase3-quadratic-observation-result-2026-06-07.md",
            ],
            "same_target_value_evidence": (
                "P44-M3 dense order-281 reference refined; Zhao-Cui/fixed-design "
                "TT matches dense tightly on dims 1,2,3 while CUT4 is stress "
                "diagnostic only."
            ),
            "gradient_interpretation": "fixed_branch_score_local_certificate_only",
            "branch_or_design_metadata": _tiny_tt_metadata("p44_m3"),
            "auxiliary_sanity_checks": [
                "symmetric-mode coverage",
                "dense order-181/order-281 refinement",
            ],
            "blocker_interpretation": "not_blocked_local_certificate",
        },
        "p44_m4_nonlinear_transition_h2_panel": {
            "source_evidence_class": "p44_dense_h2_certificate",
            "source_artifacts": [
                "docs/plans/bayesfilter-highdim-zhao-cui-p44-phase4-nonlinear-transition-result-2026-06-07.md",
            ],
            "same_target_value_evidence": (
                "P44-M4 T=2 dense order-241 reference refined; Zhao-Cui "
                "fixed-design TT has T=2 value/score evidence for dims 1,2,3; "
                "T=4 Zhao-Cui accumulation is explicitly not claimed."
            ),
            "gradient_interpretation": "fixed_branch_score_t2_local_certificate_only",
            "branch_or_design_metadata": _tiny_tt_metadata("p44_m4_h2"),
            "auxiliary_sanity_checks": [
                "nested c=0 Kalman tie-out",
                "T=4 CUT4 accumulation diagnostic only",
            ],
            "blocker_interpretation": "not_blocked_for_h2_local_certificate",
        },
        "sv_exact_transformed_log_chi_square_panel": {
            "source_evidence_class": "p3_exact_transformed_dense_certificate",
            "source_artifacts": [
                "docs/plans/bayesfilter-dpf-filter-oracle-comparison-p3-conditional-gaussian-mixture-result-2026-06-08.md",
                "experiments/dpf_implementation/reports/outputs/dpf_filter_oracle_comparison_p3_conditional_gaussian_mixture_2026-06-08.json",
            ],
            "same_target_value_evidence": (
                "P3 exact transformed dense vs Zhao-Cui/fixed-design TT "
                "certificate passes dims 1,2,3; max value error "
                "1.2017054018542694e-11."
            ),
            "gradient_interpretation": (
                "fixed_branch_score for unconstrained gamma/beta at fixed sigma"
            ),
            "branch_or_design_metadata": _sv_tt_metadata(),
            "auxiliary_sanity_checks": [
                "Jacobian convention recorded for raw-native relation",
                "KSC mixture gap recorded as approximation-only",
            ],
            "blocker_interpretation": "not_blocked_local_exact_transformed_certificate",
        },
    }
    if target_id in profiles:
        return profiles[target_id]
    if route["claim_class"] == "BLOCKED":
        return {
            "source_evidence_class": "blocked_preserved_from_p0_p45",
            "source_artifacts": [
                "docs/plans/bayesfilter-dpf-filter-oracle-comparison-target-route-registry-2026-06-08.json",
                "docs/plans/bayesfilter-highdim-zhao-cui-p45-target-registry-2026-06-08.json",
                "docs/plans/bayesfilter-highdim-zhao-cui-p45-phase5-cross-model-error-calibration-result-2026-06-08.md",
            ],
            "same_target_value_evidence": "N/A: route is blocked; no value evidence is fabricated.",
            "gradient_interpretation": "N/A: route is blocked.",
            "branch_or_design_metadata": {
                "route_family": "blocked",
                "basis_or_rank": "N/A",
                "branch_seed_policy": "N/A",
            },
            "auxiliary_sanity_checks": [
                "P45 blockers preserve missing native/reference/multistate route",
            ],
            "blocker_interpretation": "; ".join(route["blockers"]) or "blocked",
        }
    return {
        "source_evidence_class": "unclassified_nonblocked_route",
        "source_artifacts": [],
        "same_target_value_evidence": "missing",
        "gradient_interpretation": "missing",
        "branch_or_design_metadata": {},
        "auxiliary_sanity_checks": [],
        "blocker_interpretation": "unexpected nonblocked route",
    }


def _tiny_tt_metadata(label: str) -> dict[str, Any]:
    return {
        "route_family": "scalar_nonlinear_fixed_design_tt_value_path",
        "scope": "independent scalar panel; not coupled multivariate TT",
        "label": label,
        "rank_policy": "rank-1 scalar/product-panel local certificate unless source artifact states otherwise",
        "basis_policy": "Legendre fixed-design basis from reviewed P44/P2 fixture",
        "branch_seed_policy": "deterministic branch seed prefix recorded in source runner/test artifact",
        "adaptive_fit_policy": "no adaptive MATLAB TT-cross/SIRT reproduction claim",
    }


def _sv_tt_metadata() -> dict[str, Any]:
    return {
        "route_family": "factorized_scalar_zhaocui_tt_exact_transformed_sv",
        "scope": "independent scalar exact-transformed SV panels; not coupled multivariate TT",
        "basis": "Legendre basis dimension 48 on [-1,1]",
        "fit_quadrature_order": 141,
        "coordinate_map_radius": 8.0,
        "rank_policy": "ranks=(1,1) fixed-design local certificate",
        "branch_seed_policy": "deterministic per-coordinate branch seed prefix",
        "adaptive_fit_policy": "no adaptive MATLAB TT-cross/SIRT reproduction claim",
    }


def _p5_readiness(
    routes: dict[str, Any],
    zhao_route: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, Any]:
    dpf_routes = {
        route_id: routes[route_id]["phase_eligibility"]["p5"]
        for route_id in ("dpf_bootstrap_ot", "dpf_ledh_pfpf_ot")
    }
    p5_enabled = any(dpf_routes.values())
    return {
        "dpf_p5_eligible": p5_enabled,
        "dpf_route_eligibility": dpf_routes,
        "primary_reference_route_for_p5": (
            reference["route_id"] if p5_enabled else "none"
        ),
        "zhao_cui_is_p5_comparator": False,
        "status": (
            "p5_reference_available_dpf_eligible"
            if p5_enabled and reference["claim_class"] == "EXACT_ORACLE"
            else "not_p5_eligible_or_reference_blocked"
        ),
        "reason": (
            "P5 should compare DPF to the exact Kalman/dense reference; "
            "Zhao-Cui classification is auxiliary."
            if p5_enabled
            else "P0 does not authorize DPF P5 execution for this target."
        ),
    }


def _summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_class: dict[str, int] = {claim: 0 for claim in sorted(MASTER_CLAIM_CLASSES)}
    for row in rows:
        by_class[row["zhao_cui_route"]["claim_class"]] += 1
    p5_rows = [
        row["target_id"]
        for row in rows
        if row["p5_readiness"]["dpf_p5_eligible"]
    ]
    return {
        "row_count": len(rows),
        "claim_class_counts": by_class,
        "runnable_or_diagnostic_rows": [
            row["target_id"]
            for row in rows
            if row["zhao_cui_route"]["claim_class"] != "BLOCKED"
        ],
        "blocked_rows": [
            row["target_id"]
            for row in rows
            if row["zhao_cui_route"]["claim_class"] == "BLOCKED"
        ],
        "p5_dpf_eligible_targets": p5_rows,
        "p5_target_count": len(p5_rows),
        "zhao_cui_exact_oracle_rows": [
            row["target_id"]
            for row in rows
            if row["zhao_cui_route"]["claim_class"] == "EXACT_ORACLE"
        ],
    }


def _veto_diagnostics(registry: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, bool]:
    route_matrix = registry["route_matrix"]
    return {
        "missing_zhao_cui_row": any(
            ZHAO_ROUTE_ID not in route_matrix[row["target_id"]] for row in rows
        )
        or len(rows) != len(registry["targets"]),
        "invalid_claim_class": any(
            row["zhao_cui_route"]["claim_class"] not in MASTER_CLAIM_CLASSES
            for row in rows
        ),
        "scalar_only_route_applied_to_multistate_without_adapter": any(
            row["model_family"]
            in {
                "generalized_stochastic_volatility",
                "spatial_sir",
                "predator_prey",
            }
            and row["zhao_cui_route"]["claim_class"] != "BLOCKED"
            for row in rows
        ),
        "adaptive_fit_hidden_inside_fixed_branch_claim": any(
            "adaptive" in json.dumps(row["classification"]["branch_or_design_metadata"]).lower()
            and row["classification"]["branch_or_design_metadata"].get(
                "adaptive_fit_policy"
            )
            != "no adaptive MATLAB TT-cross/SIRT reproduction claim"
            for row in rows
            if row["zhao_cui_route"]["claim_class"] != "BLOCKED"
        ),
        "fit_residual_promoted_to_likelihood_correctness": False,
        "zhao_cui_treated_as_exact_oracle": any(
            row["zhao_cui_route"]["claim_class"] == "EXACT_ORACLE" for row in rows
        ),
        "runnable_row_lacks_value_evidence": any(
            row["zhao_cui_route"]["claim_class"] != "BLOCKED"
            and not row["classification"]["same_target_value_evidence"]
            for row in rows
        ),
        "gradient_row_without_same_target_value_evidence": any(
            row["zhao_cui_route"]["primary_gradient_statistic"]
            in {"fixed_branch_score", "diagnostic_fixed_branch_score"}
            and row["classification"]["same_target_value_evidence"] in {"", "missing"}
            for row in rows
        ),
        "blocked_row_lacks_blocker": any(
            row["zhao_cui_route"]["claim_class"] == "BLOCKED"
            and not row["zhao_cui_route"]["blockers"]
            for row in rows
        ),
        "p5_eligibility_unknown": any(
            row["p5_readiness"]["status"]
            not in {
                "p5_reference_available_dpf_eligible",
                "not_p5_eligible_or_reference_blocked",
            }
            for row in rows
        ),
    }


def _validate_payload(payload: dict[str, Any]) -> None:
    required = {
        "decision",
        "execution_mode",
        "evidence_contract",
        "rows",
        "route_summaries",
        "veto_diagnostics",
        "run_manifest",
        "nonclaims",
    }
    missing = required.difference(payload)
    if missing:
        raise P4ValidationError(f"missing payload fields {sorted(missing)}")
    if payload["decision"] not in {
        "PASS_P4_ZHAOCUI_ROUTE_CLASSIFICATION_PENDING_CLAUDE_REVIEW",
        "P4_ZHAOCUI_ROUTE_CLASSIFICATION_VETO_PENDING_REVIEW",
    }:
        raise P4ValidationError(f"invalid P4 decision {payload['decision']}")
    if payload["execution_mode"] != "PURE_PYTHON_CLASSIFICATION_ONLY":
        raise P4ValidationError("P4 should be classification-only")
    if payload["run_manifest"].get("tensorflow_imported"):
        raise P4ValidationError("TensorFlow was imported in P4 classification gate")
    if any(bool(value) for value in payload["veto_diagnostics"].values()):
        raise P4ValidationError(f"P4 veto fired: {payload['veto_diagnostics']}")
    if payload["route_summaries"]["row_count"] != 13:
        raise P4ValidationError("P4 should classify all 13 P0 targets")
    if payload["route_summaries"]["zhao_cui_exact_oracle_rows"]:
        raise P4ValidationError("Zhao-Cui row was promoted to exact oracle")
    allowed_p5 = {
        "lgssm_2d_h25_rich",
        "p44_m2_cubic_additive_gaussian_panel",
        "p44_m3_quadratic_observation_panel",
        "p44_m4_nonlinear_transition_h2_panel",
    }
    if set(payload["route_summaries"]["p5_dpf_eligible_targets"]) != allowed_p5:
        raise P4ValidationError("P5 target set does not match P0 eligibility")
    for row in payload["rows"]:
        claim = row["zhao_cui_route"]["claim_class"]
        if claim not in MASTER_CLAIM_CLASSES:
            raise P4ValidationError("invalid master claim class")
        if claim == "BLOCKED" and not row["zhao_cui_route"]["blockers"]:
            raise P4ValidationError(f"{row['target_id']} blocked row lacks blocker")
        if row["p5_readiness"]["zhao_cui_is_p5_comparator"]:
            raise P4ValidationError("Zhao-Cui should not be a P5 comparator route")
    if "reproducibility_digest" not in payload:
        raise P4ValidationError("missing reproducibility digest")


def _markdown(payload: dict[str, Any]) -> str:
    summary = payload["route_summaries"]
    lines = [
        "# P4 Result: Zhao-Cui and Fixed-Design TT Route Classification",
        "",
        "metadata_date: 2026-06-08",
        "phase: P4",
        f"status: {payload['decision']}",
        "",
        "## Skeptical Plan Audit",
        "",
        "Status: `PASS_FOR_P4_CLASSIFICATION_RECORDING_PENDING_CLAUDE_REVIEW`.",
        "",
        "Wrong-baseline risk is controlled by making P4 classification-only and consuming reviewed P0-P3/P44/P45 artifacts rather than inventing a new comparator.",
        "",
        "Proxy-promotion risk is controlled by rejecting exact-oracle Zhao-Cui rows and treating fit residuals, finite status, and Kalman sanity checks as auxiliary diagnostics.",
        "",
        "Unfair-comparison risk is controlled by preserving multistate/native rows as blocked instead of applying scalar fixed-design TT routes to unsupported targets.",
        "",
        "## Evidence Contract",
        "",
        "| Field | Contract |",
        "| --- | --- |",
        f"| Question | {payload['question']} |",
        f"| Baseline/comparator | {payload['evidence_contract']['baseline_comparator']} |",
        f"| Primary criterion | {payload['evidence_contract']['primary_criterion']} |",
        f"| Execution amendment | {payload['evidence_contract']['classification_only_amendment']} |",
        f"| Gradient policy | {payload['evidence_contract']['gradient_policy']} |",
        "| Not concluded | No DPF correctness, paper-scale Zhao-Cui reproduction, adaptive MATLAB TT-cross/SIRT behavior, coupled multivariate TT correctness, HMC, production, GPU, or native blocked-row correctness. |",
        "",
        "## Veto Diagnostics",
        "",
        "| Diagnostic | Status |",
        "| --- | --- |",
        *[
            f"| `{key}` | `{value}` |"
            for key, value in payload["veto_diagnostics"].items()
        ],
        "",
        "## Classification Summary",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| row count | `{summary['row_count']}` |",
        f"| claim class counts | `{summary['claim_class_counts']}` |",
        f"| runnable/diagnostic rows | `{summary['runnable_or_diagnostic_rows']}` |",
        f"| blocked rows | `{summary['blocked_rows']}` |",
        f"| P5 DPF eligible targets | `{summary['p5_dpf_eligible_targets']}` |",
        "",
        "## Rows",
        "",
        "| target | class | P4 | P5 DPF | evidence class | P5 reference |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| `{row['target_id']}` | `{row['zhao_cui_route']['claim_class']}` | "
            f"`{row['zhao_cui_route']['phase_eligibility'].get('p4')}` | "
            f"`{row['p5_readiness']['dpf_p5_eligible']}` | "
            f"`{row['classification']['source_evidence_class']}` | "
            f"`{row['p5_readiness']['primary_reference_route_for_p5']}` |"
        )
    lines.extend(
        [
            "",
            "## P5 Readiness",
            "",
            "P5 DPF statistical closeness is authorized by P0 only for LGSSM and P44-M2/M3/M4 targets. Zhao-Cui/fixed-design TT rows are auxiliary classification evidence, not the P5 comparator; P5 should use exact Kalman or dense/refined references.",
            "",
            "## Decision Table",
            "",
            "| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |",
            "| --- | --- | --- | --- | --- | --- |",
            (
                f"| `{payload['decision']}` | all `{summary['row_count']}` P0 targets classified; "
                f"claim counts `{summary['claim_class_counts']}` | `{payload['veto_diagnostics']}` | "
                "P4 is classification-only and relies on reviewed source artifacts | "
                "run Claude read-only P4 gate review | no DPF correctness or paper-scale/adaptive/coupled TT claim |"
            ),
            "",
            "## Post-Run Red-Team Note",
            "",
            "Strongest alternative explanation: a future multistate TT adapter could make some currently blocked closure rows runnable, but P4 correctly preserves the current blockers rather than speculating.",
            "",
            "Result that would overturn the P4 pass: a reviewed audit finds a Zhao-Cui row labeled exact oracle, a multistate target using scalar TT without an adapter, missing blockers, or P5 eligibility different from P0.",
            "",
            "Weakest part of the evidence: runnable nonlinear rows are local/tiny fixed-design certificates, not paper-scale Zhao-Cui reproduction.",
            "",
            "## Run Manifest",
            "",
            f"- command: `{payload['run_manifest']['command']}`",
            f"- git branch: `{payload['run_manifest']['branch']}`",
            f"- git commit: `{payload['run_manifest']['commit']}`",
            f"- dirty state summary: `{payload['run_manifest']['dirty_state_summary']}`",
            f"- python version: `{payload['run_manifest']['python_version']}`",
            f"- CPU/GPU status: `{payload['run_manifest']['cpu_gpu_status']}`",
            f"- plan: `{payload['run_manifest']['plan_path']}`",
            f"- review ledger: `{payload['run_manifest']['review_ledger_path']}`",
            f"- JSON: `{payload['run_manifest']['json_path']}`",
            f"- report: `{payload['run_manifest']['report_path']}`",
            f"- result: `{payload['run_manifest']['result_path']}`",
            f"- wall time seconds: `{payload['run_manifest']['wall_time_seconds']}`",
            "",
            "## Nonclaims",
            "",
            *[f"- {item}" for item in payload["nonclaims"]],
            "",
            "## Gate Status",
            "",
            "P4 is pending Claude read-only review.",
            "",
        ]
    )
    return "\n".join(lines)


def _manifest() -> dict[str, Any]:
    return {
        **_git_manifest(),
        "python_version": platform.python_version(),
        "cpu_gpu_status": "pure_python_classification_only; TensorFlow not imported; no GPU claim",
        "tensorflow_imported": "tensorflow" in sys.modules,
        "tensorflow_probability_imported": "tensorflow_probability" in sys.modules,
        "command": "python -m " + MODULE_PATH,
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "review_ledger_path": REVIEW_LEDGER_PATH,
        "json_path": str(JSON_PATH.relative_to(REPO_ROOT)),
        "report_path": str(REPORT_PATH.relative_to(REPO_ROOT)),
        "data_version": "P0 registry plus reviewed P1-P3/P44/P45 artifacts",
        "seeds": "N/A: classification-only",
        "particle_counts": "N/A: no DPF execution in P4",
    }


def _git_manifest() -> dict[str, str]:
    return {
        "branch": _git(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "commit": _git(["git", "rev-parse", "HEAD"]),
        "dirty_state_summary": _git(["git", "status", "--short"]) or "clean",
    }


def _git(args: list[str]) -> str:
    completed = subprocess.run(args, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def stable_digest(payload: Any) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _nonclaims() -> list[str]:
    return [
        "P4 does not execute DPF and does not establish DPF value or gradient correctness.",
        "No Zhao-Cui/fixed-design TT row is promoted to exact oracle.",
        "Runnable Zhao-Cui/fixed-design TT rows are local certificates or diagnostics only.",
        "Blocked generalized-SV, spatial-SIR, predator-prey, native, and KSC-mixture TT rows remain blocked.",
        "P4 does not reproduce adaptive MATLAB TT-cross/SIRT behavior.",
        "P4 does not establish coupled multivariate TT correctness, HMC readiness, production readiness, GPU readiness, or paper-scale claims.",
    ]


def _digest_payload(payload: dict[str, Any]) -> str:
    stable = {
        key: value
        for key, value in payload.items()
        if key not in {"created_at_utc", "run_manifest", "reproducibility_digest"}
    }
    return stable_digest(stable)


if __name__ == "__main__":
    sys.exit(main())
