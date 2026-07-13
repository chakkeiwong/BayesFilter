#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.highdim.ledh_forward_contract import (
    FIXED_SIR_AUSTRIA_ROW_ID,
    LGSSM_M3_T50_ROW_ID,
    make_fixed_sir_logscale_forward_contract,
    make_lgssm_m3_t50_forward_contract,
)
from bayesfilter.highdim.ledh_score_contract import (
    LEDH_SCORE_ARTIFACT_SCHEMA_VERSION,
    validate_ledh_score_artifact,
)

DATE = "2026-07-06"
BASELINE_PATH = ROOT / "docs/plans/bayesfilter-two-lane-highdim-leaderboard-results-2026-07-03.json"
LEDGER_PATH = ROOT / "docs/plans/bayesfilter-ledh-inclusive-highdim-leaderboard-phase1-row-admission-ledger-2026-07-03.json"
LGSSM_LEDGER_PATH = ROOT / "docs/plans/bayesfilter-ledh-inclusive-highdim-leaderboard-phase4-lgssm-m3-t50-same-target-value-ladder-N10000-2026-07-03.json"
SIR_LEDGER_PATH = ROOT / "docs/plans/bayesfilter-ledh-inclusive-highdim-leaderboard-phase4-fixed-sir-value-ladder-N10000-2026-07-03.json"
LGSSM_SCORE_PATH = ROOT / "docs/plans/ledh-phase5-lgssm-score-memory-n10000-2026-07-06.json"
SIR_SCORE_PATH = ROOT / "docs/plans/ledh-phase5-fixed-sir-score-memory-n10000-2026-07-06.json"
DEFAULT_JSON = ROOT / "docs/plans/bayesfilter-two-lane-highdim-ledh-inclusive-leaderboard-results-2026-07-06.json"
DEFAULT_MD = ROOT / "docs/plans/bayesfilter-two-lane-highdim-ledh-inclusive-leaderboard-results-2026-07-06.md"

LEDH_ALGORITHM_ID = "ledh_pfpf_ot"
COMPARATOR_MODE = "frozen_non_ledh_baseline_plus_fresh_ledh"
HIGHDIM_ROWS = [
    "benchmark_lgssm_exact_oracle_m3_T50",
    "zhao_cui_sv_actual_nongaussian_T1000",
    "zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000",
    "zhao_cui_spatial_sir_austria_j9_T20",
    "zhao_cui_spatial_sir_austria_j9_T20_parameterized_logscale",
    "zhao_cui_predator_prey_T20",
    "zhao_cui_generalized_sv_synthetic_from_estimated_values",
]
COMPARISON_ALGORITHMS = [
    "fixed_sgqf",
    "ukf",
    "zhao_cui_scalar_or_multistate",
    LEDH_ALGORITHM_ID,
]


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rel(path: Path) -> str:
    resolved = path if path.is_absolute() else ROOT / path
    try:
        return str(resolved.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.6g}"
    return str(value)


def _sample_summary(values: list[float]) -> dict[str, float | None]:
    mean = statistics.fmean(values)
    if len(values) < 2:
        return {"mean": mean, "sample_sd": None, "mcse": None}
    sample_sd = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
    return {"mean": mean, "sample_sd": sample_sd, "mcse": sample_sd / math.sqrt(len(values))}


def _lgssm_forward_contract_from_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    target_identity = artifact.get("target_identity", {})
    forward_contract = target_identity.get("forward_contract")
    if forward_contract is not None:
        return dict(forward_contract)
    return make_lgssm_m3_t50_forward_contract(
        truth_theta=target_identity.get("truth_theta", [0.72, 0.55, 0.35, 0.35, 0.45]),
        time_steps=artifact.get("shape", {}).get("time_steps"),
        num_particles=artifact.get("shape", {}).get("num_particles"),
        batch_seeds=artifact.get("batch_seeds", ()),
        full_leaderboard_row=bool(target_identity.get("full_leaderboard_row", False)),
    ).to_manifest()


def _fixed_sir_forward_contract_from_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    return make_fixed_sir_logscale_forward_contract(
        time_steps=artifact.get("shape", {}).get("time_steps"),
        num_particles=artifact.get("shape", {}).get("num_particles"),
        batch_seeds=artifact.get("batch_seeds", ()),
        full_leaderboard_row=True,
    ).to_manifest()


def _copy_baseline_row(row: dict[str, Any], *, baseline_path: Path) -> dict[str, Any]:
    copied = dict(row)
    copied["comparator_provenance"] = "frozen_non_ledh_baseline"
    copied["source_baseline"] = _rel(baseline_path)
    copied["runtime_rankable_with_ledh"] = False
    copied["nonclaims"] = list(copied.get("nonclaims", [])) + [
        "frozen non-LEDH baseline row copied for status/value context",
        "not rerun under the LEDH GPU/XLA harness",
        "not runtime-rankable against fresh LEDH rows",
    ]
    return copied


def _phase7_status(row_scope: str, evidence_paths: list[str], gpu_xla_status: str) -> dict[str, Any]:
    return {
        "scope": row_scope,
        "batch_status": "fresh_ledh_phase4_batched_value_run",
        "cpu_timing_status": "not_run_cpu_not_default_ledh_target",
        "gpu_xla_status": gpu_xla_status,
        "timing_rank_status": "not_rankable_frozen_non_ledh_baseline",
        "evidence_paths": list(evidence_paths),
        "reason": (
            "Fresh LEDH GPU/XLA/TF32 value timing is recorded for diagnostics, "
            "but non-LEDH rows are frozen baseline rows and were not rerun in "
            "the same harness."
        ),
        "nonclaims": [
            "not runtime-rankable against frozen non-LEDH rows",
            "not score evidence",
            "not HMC readiness evidence",
        ],
    }


def _blocked_phase7_status(row_scope: str, reason: str) -> dict[str, Any]:
    return {
        "scope": row_scope,
        "batch_status": "not_run_blocked_or_scoped",
        "cpu_timing_status": "not_run",
        "gpu_xla_status": "not_run",
        "timing_rank_status": "not_rankable_no_fresh_value_row",
        "evidence_paths": [],
        "reason": reason,
        "nonclaims": [
            "not runtime evidence",
            "not value correctness evidence",
            "not score correctness evidence",
        ],
    }


def _score_artifact_candidate(raw_artifact: dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw_artifact.get("score_artifact"), dict):
        return dict(raw_artifact["score_artifact"])
    return dict(raw_artifact)


def _score_candidate_summary(candidate: dict[str, Any] | None) -> dict[str, Any]:
    if candidate is None:
        return {
            "score_candidate_derivative_provenance": None,
            "score_candidate_admission_status": None,
        }
    provenance = candidate.get("score_derivative_provenance", candidate.get("score_route"))
    return {
        "score_candidate_derivative_provenance": provenance,
        "score_candidate_admission_status": candidate.get("score_admission_status"),
    }


def _score_payload(
    score_artifact: dict[str, Any] | None,
    *,
    expected_row_id: str,
    source_value_artifact: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str | None]:
    if score_artifact is None:
        return None, None, "score artifact not provided"
    candidate = _score_artifact_candidate(score_artifact)
    if candidate.get("row_id") != expected_row_id:
        raise ValueError(f"score artifact row mismatch for {expected_row_id}")

    legacy_route = candidate.get("score_route")
    if legacy_route is not None and "manual_total_vjp" in str(legacy_route):
        return (
            None,
            {
                "score_derivative_provenance": str(legacy_route),
                "score_admission_status": "historical_diagnostic_not_admitted",
            },
            (
                "historical manual_total_vjp raw score-memory artifact is "
                "diagnostic-only and blocked from leaderboard score admission"
            ),
        )

    if candidate.get("schema_version") != LEDH_SCORE_ARTIFACT_SCHEMA_VERSION:
        route = candidate.get("score_route")
        summary = {
            "score_derivative_provenance": route,
            "score_admission_status": "legacy_raw_score_memory_not_admitted",
        }
        return (
            None,
            summary,
            (
                "legacy score-memory artifact lacks the Phase 1 score artifact "
                "schema and is not admitted"
            ),
        )

    try:
        admitted = validate_ledh_score_artifact(
            candidate,
            source_value_artifact=source_value_artifact,
            expected_row_id=expected_row_id,
            require_admitted=True,
        )
    except ValueError as admitted_error:
        try:
            normalized = validate_ledh_score_artifact(
                candidate,
                source_value_artifact=source_value_artifact,
                expected_row_id=expected_row_id,
                require_admitted=False,
            )
        except ValueError as validation_error:
            return None, candidate, f"score artifact rejected by Phase 1 contract: {validation_error}"
        return None, normalized, f"score artifact is not full admission: {admitted_error}"
    return admitted, admitted, None


def _score_row_fields(
    score_artifact: dict[str, Any] | None,
    *,
    coordinate_system: str,
    blocked_reason: str,
    score_path: Path | None,
    score_candidate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if score_artifact is None:
        return {
            "comparison_status": "executed_value_only_score_blocked",
            "row_admission_status": "executed_same_target_value_score_blocked",
            "score": None,
            "score_l2_norm": None,
            "score_status": "blocked_score_until_same_target_no_tape_gate",
            "score_status_reason": blocked_reason,
            "score_derivative_provenance": None,
            "score_coordinate_system": coordinate_system,
            "score_evidence_artifact": None,
            "score_memory_peak_mib": None,
            "score_fd_abs_error": None,
            "score_fd_rel_error": None,
            "score_candidate_artifact": _rel(score_path) if score_path is not None else None,
            **_score_candidate_summary(score_candidate),
        }
    score_value = score_artifact.get("score")
    score_l2 = None
    if score_value is not None:
        score_l2 = math.sqrt(sum(float(value) * float(value) for value in score_value))
    correctness = score_artifact.get("score_correctness", {})
    memory = score_artifact.get("memory_diagnostics", {})
    return {
        "comparison_status": "executed_value_score",
        "row_admission_status": "executed_same_target_value_score_compact_contract_pass",
        "score": score_value,
        "score_l2_norm": score_l2,
        "score_status": "admitted_same_target_compact_score_n10000",
        "score_status_reason": (
            "Phase 1 shared score artifact contract admitted compact N=10000 score"
        ),
        "score_derivative_provenance": score_artifact["score_derivative_provenance"],
        "score_coordinate_system": coordinate_system,
        "score_evidence_artifact": _rel(score_path) if score_path is not None else None,
        "score_memory_peak_mib": memory.get("peak_mib"),
        "score_fd_abs_error": correctness.get("max_abs_error", correctness.get("abs_error")),
        "score_fd_rel_error": correctness.get(
            "max_relative_error",
            correctness.get("max_rel_error", correctness.get("rel_error")),
        ),
        "score_candidate_artifact": _rel(score_path) if score_path is not None else None,
        "score_candidate_derivative_provenance": score_artifact["score_derivative_provenance"],
        "score_candidate_admission_status": score_artifact["score_admission_status"],
    }


def _lgssm_ledh_row(
    artifact: dict[str, Any],
    *,
    artifact_path: Path,
    score_artifact: dict[str, Any] | None = None,
    score_path: Path | None = None,
) -> dict[str, Any]:
    avg = artifact["average_log_likelihood_estimate"]
    total = artifact["total_log_likelihood_estimate"]
    evidence = [_rel(artifact_path)]
    forward_contract = _lgssm_forward_contract_from_artifact(artifact)
    score_artifact, score_candidate, score_blocker = _score_payload(
        score_artifact,
        expected_row_id=LGSSM_M3_T50_ROW_ID,
        source_value_artifact=artifact,
    )
    score_fields = _score_row_fields(
        score_artifact,
        coordinate_system="theta=(phi1,phi2,phi3,q_scale,r_scale)",
        blocked_reason=score_blocker
        or "same-target total derivative not implemented in Phase 4 value runner",
        score_path=score_path,
        score_candidate=score_candidate,
    )
    return {
        "algorithm_id": LEDH_ALGORITHM_ID,
        "row_id": LGSSM_M3_T50_ROW_ID,
        "comparison_status": score_fields["comparison_status"],
        "lane": "highdim_source_scope",
        "row_scope": "main_observed_data_filtering_row",
        "target_scope": "main_observed_data_filtering_row",
        "target_contract_status": "same_target_value_score",
        "target_match_status": (
            "same_target_value_score"
            if score_artifact is not None
            else artifact["target_identity"]["same_target_status"]
        ),
        "forward_contract": forward_contract,
        "target_scalar": forward_contract["target_scalar"],
        "target_output_tensor_field": forward_contract["output_tensor_field"],
        "target_density_fields": forward_contract["target_density_fields"],
        "proposal_flow_fields": forward_contract["proposal_flow_fields"],
        "correction_formula": forward_contract["correction_formula"],
        "row_admission_status": score_fields["row_admission_status"],
        "average_log_likelihood": avg["mean"],
        "log_likelihood": total["mean"],
        "mc_standard_error": avg["mcse"],
        "mc_standard_deviation": avg["sample_sd"],
        "total_log_likelihood_mc_standard_error": total["mcse"],
        "total_log_likelihood_mc_standard_deviation": total["sample_sd"],
        "score": score_fields["score"],
        "score_l2_norm": score_fields["score_l2_norm"],
        "score_status": score_fields["score_status"],
        "score_status_reason": score_fields["score_status_reason"],
        "score_derivative_provenance": score_fields["score_derivative_provenance"],
        "score_coordinate_system": score_fields["score_coordinate_system"],
        "score_evidence_artifact": score_fields["score_evidence_artifact"],
        "score_memory_peak_mib": score_fields["score_memory_peak_mib"],
        "score_fd_abs_error": score_fields["score_fd_abs_error"],
        "score_fd_rel_error": score_fields["score_fd_rel_error"],
        "score_candidate_artifact": score_fields["score_candidate_artifact"],
        "score_candidate_derivative_provenance": score_fields[
            "score_candidate_derivative_provenance"
        ],
        "score_candidate_admission_status": score_fields["score_candidate_admission_status"],
        "value_status": artifact["value_status"],
        "particle_count": artifact["shape"]["num_particles"],
        "seeds": artifact["batch_seeds"],
        "runtime_seconds": artifact["warm_call_timing_summary_seconds"].get("mean"),
        "compile_and_first_call_seconds": artifact["compile_and_first_call_seconds"],
        "runtime_rankable": False,
        "runtime_rankable_with_non_ledh": False,
        "device_provenance": "fresh_ledh_gpu_xla_tf32_phase4",
        "comparator_provenance": "fresh_ledh_phase4",
        "source_artifact": _rel(artifact_path),
        "exact_value_comparison": artifact["exact_value_comparison"],
        "ess_min_by_seed": artifact["ess_min_by_seed"],
        "phase7_batch_gpu_xla_status": _phase7_status(
            "main_observed_data_filtering_row",
            evidence,
            "executed_gpu_xla_tf32_value_only",
        ),
        "reason": (
            "same-target LEDH value and no-tape score executed at N=10000"
            if score_artifact is not None
            else "same-target LEDH value executed at N=10000; score remains blocked"
        ),
        "nonclaims": list(artifact["nonclaims"])
        + [
            "not runtime-rankable against frozen non-LEDH rows",
            "not HMC readiness evidence",
            "not posterior correctness evidence",
        ],
    }


def _sir_ledh_row(
    artifact: dict[str, Any],
    *,
    artifact_path: Path,
    score_artifact: dict[str, Any] | None = None,
    score_path: Path | None = None,
) -> dict[str, Any]:
    total_values = [float(value) for value in artifact["log_likelihood"]]
    time_steps = int(artifact["shape"]["time_steps"])
    total_summary = _sample_summary(total_values)
    average_values = [value / float(time_steps) for value in total_values]
    average_summary = _sample_summary(average_values)
    evidence = [_rel(artifact_path)]
    forward_contract = _fixed_sir_forward_contract_from_artifact(artifact)
    score_artifact, score_candidate, score_blocker = _score_payload(
        score_artifact,
        expected_row_id=FIXED_SIR_AUSTRIA_ROW_ID,
        source_value_artifact=artifact,
    )
    score_fields = _score_row_fields(
        score_artifact,
        coordinate_system="theta=(log_kappa_scale,log_nu_scale,log_obs_noise_scale)",
        blocked_reason=score_blocker
        or (
            "fixed SIR now has sir_log_scale_theta, but the full-row "
            "same-target total-derivative score is not implemented or checked"
        ),
        score_path=score_path,
        score_candidate=score_candidate,
    )
    return {
        "algorithm_id": LEDH_ALGORITHM_ID,
        "row_id": FIXED_SIR_AUSTRIA_ROW_ID,
        "comparison_status": score_fields["comparison_status"],
        "lane": "highdim_source_scope",
        "row_scope": "main_observed_data_filtering_row",
        "target_scope": "main_observed_data_filtering_row",
        "target_contract_status": "amended_sir_log_scale_theta_observed_data_value_score",
        "target_match_status": "sir_log_scale_theta_observed_data_value_score",
        "forward_contract": forward_contract,
        "target_scalar": forward_contract["target_scalar"],
        "target_output_tensor_field": forward_contract["output_tensor_field"],
        "target_density_fields": forward_contract["target_density_fields"],
        "proposal_flow_fields": forward_contract["proposal_flow_fields"],
        "correction_formula": forward_contract["correction_formula"],
        "row_admission_status": score_fields["row_admission_status"],
        "average_log_likelihood": average_summary["mean"],
        "log_likelihood": total_summary["mean"],
        "mc_standard_error": average_summary["mcse"],
        "mc_standard_deviation": average_summary["sample_sd"],
        "total_log_likelihood_mc_standard_error": total_summary["mcse"],
        "total_log_likelihood_mc_standard_deviation": total_summary["sample_sd"],
        "score": score_fields["score"],
        "score_l2_norm": score_fields["score_l2_norm"],
        "score_status": score_fields["score_status"],
        "score_status_reason": score_fields["score_status_reason"],
        "score_derivative_provenance": score_fields["score_derivative_provenance"],
        "score_coordinate_system": score_fields["score_coordinate_system"],
        "score_evidence_artifact": score_fields["score_evidence_artifact"],
        "score_memory_peak_mib": score_fields["score_memory_peak_mib"],
        "score_fd_abs_error": score_fields["score_fd_abs_error"],
        "score_fd_rel_error": score_fields["score_fd_rel_error"],
        "score_candidate_artifact": score_fields["score_candidate_artifact"],
        "score_candidate_derivative_provenance": score_fields[
            "score_candidate_derivative_provenance"
        ],
        "score_candidate_admission_status": score_fields["score_candidate_admission_status"],
        "value_status": "executed_fixed_sir_value_only",
        "particle_count": artifact["shape"]["num_particles"],
        "seeds": artifact["batch_seeds"],
        "runtime_seconds": artifact["warm_call_timing_summary_seconds"].get("mean"),
        "compile_and_first_call_seconds": artifact["compile_and_first_call_seconds"],
        "runtime_rankable": False,
        "runtime_rankable_with_non_ledh": False,
        "device_provenance": "fresh_ledh_gpu_xla_tf32_phase4",
        "comparator_provenance": "fresh_ledh_phase4",
        "source_artifact": _rel(artifact_path),
        "ess_min_by_seed": artifact["ess_min_by_seed"],
        "phase7_batch_gpu_xla_status": _phase7_status(
            "main_observed_data_filtering_row",
            evidence,
            "executed_gpu_xla_tf32_value_only",
        ),
        "reason": (
            "fixed spatial SIR LEDH value and no-tape score executed at N=10000 "
            "under the amended sir_log_scale_theta row"
            if score_artifact is not None
            else (
                "fixed spatial SIR LEDH value executed at N=10000 under the "
                "amended sir_log_scale_theta row; score remains blocked until "
                "same-target no-tape evidence passes"
            )
        ),
        "nonclaims": list(artifact["nonclaims"])
        + [
            "not exact nonlinear likelihood correctness evidence",
            "not runtime-rankable against frozen non-LEDH rows",
            "not Zhao-Cui TT/SIRT source-faithfulness evidence",
        ],
    }


def _ledger_blocked_ledh_row(row: dict[str, Any], *, ledger_path: Path) -> dict[str, Any]:
    row_scope = (
        "scoped_component_row"
        if row["ledh_row_scope_decision"] == "scoped_component_row"
        else "main_observed_data_filtering_row"
    )
    reason = "; ".join(
        item
        for item in [
            str(row.get("blocker") or ""),
            f"value_status={row['value_status']}",
            f"score_status={row['score_status']}",
        ]
        if item
    )
    return {
        "algorithm_id": LEDH_ALGORITHM_ID,
        "row_id": row["row_id"],
        "comparison_status": (
            "scoped_component_status_only"
            if row["ledh_row_scope_decision"] == "scoped_component_row"
            else "blocked"
        ),
        "lane": "highdim_source_scope",
        "row_scope": row_scope,
        "target_scope": row["requested_row_scope"],
        "target_contract_status": row["target_match_status"],
        "target_match_status": row["target_match_status"],
        "row_admission_status": row["ledh_row_scope_decision"],
        "average_log_likelihood": None,
        "log_likelihood": None,
        "mc_standard_error": None,
        "mc_standard_deviation": None,
        "score": None,
        "score_l2_norm": None,
        "score_status": row["score_status"],
        "score_status_reason": row["required_score_admission_artifact"],
        "score_derivative_provenance": None,
        "score_coordinate_system": None,
        "value_status": row["value_status"],
        "particle_count": None,
        "seeds": None,
        "runtime_seconds": None,
        "runtime_rankable": False,
        "runtime_rankable_with_non_ledh": False,
        "device_provenance": "not_run_blocked_or_scoped",
        "comparator_provenance": "fresh_ledh_status_from_phase1_ledger",
        "source_ledger": _rel(ledger_path),
        "phase7_batch_gpu_xla_status": _blocked_phase7_status(row_scope, reason),
        "reason": reason,
        "required_phase2_work": list(row.get("required_phase2_work", [])),
        "required_score_admission_artifact": row["required_score_admission_artifact"],
        "current_adapter_evidence": list(row.get("current_adapter_evidence", [])),
        "forbidden_claims": list(row.get("forbidden_claims", [])),
        "nonclaims": [
            "not LEDH value execution",
            "not LEDH score execution",
            "not HMC readiness evidence",
            "not runtime ranking evidence",
        ],
    }


def _ledh_row_for(
    ledger_row: dict[str, Any],
    *,
    ledger_path: Path,
    lgssm_artifact: dict[str, Any],
    lgssm_path: Path,
    lgssm_score_artifact: dict[str, Any] | None,
    lgssm_score_path: Path | None,
    sir_artifact: dict[str, Any],
    sir_path: Path,
    sir_score_artifact: dict[str, Any] | None,
    sir_score_path: Path | None,
) -> dict[str, Any]:
    if ledger_row["row_id"] == "benchmark_lgssm_exact_oracle_m3_T50":
        return _lgssm_ledh_row(
            lgssm_artifact,
            artifact_path=lgssm_path,
            score_artifact=lgssm_score_artifact,
            score_path=lgssm_score_path,
        )
    if ledger_row["row_id"] == "zhao_cui_spatial_sir_austria_j9_T20":
        return _sir_ledh_row(
            sir_artifact,
            artifact_path=sir_path,
            score_artifact=sir_score_artifact,
            score_path=sir_score_path,
        )
    return _ledger_blocked_ledh_row(ledger_row, ledger_path=ledger_path)


def _row_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row_id in HIGHDIM_ROWS:
        cells = [row for row in rows if row["row_id"] == row_id]
        ledh = next(row for row in cells if row["algorithm_id"] == LEDH_ALGORITHM_ID)
        executed = [
            row["algorithm_id"]
            for row in cells
            if str(row["comparison_status"]).startswith("executed")
        ]
        score_admitted = [
            row["algorithm_id"]
            for row in cells
            if str(row.get("comparison_status")) == "executed_value_score"
        ]
        output.append(
            {
                "row_id": row_id,
                "row_scope": ledh["row_scope"],
                "comparison_algorithm_count": len(cells),
                "comparison_algorithms": [row["algorithm_id"] for row in cells],
                "executed_algorithms": executed,
                "score_admitted_algorithms": score_admitted,
                "ledh_status": ledh["comparison_status"],
                "ledh_value_status": ledh["value_status"],
                "ledh_score_status": ledh["score_status"],
                "ledh_runtime_rankable": False,
                "full_four_way_value_ready": all(
                    str(row["comparison_status"]).startswith("executed") for row in cells
                ),
                "full_four_way_score_ready": len(score_admitted) == len(COMPARISON_ALGORITHMS),
                "blocked_or_missing_algorithms": [
                    row["algorithm_id"]
                    for row in cells
                    if not str(row["comparison_status"]).startswith("executed")
                ],
            }
        )
    return output


def _validate_payload(payload: dict[str, Any]) -> None:
    if payload["comparator_mode"] != COMPARATOR_MODE:
        raise ValueError("wrong comparator mode")
    if payload["runtime_cross_ranking_allowed"] is not False:
        raise ValueError("runtime cross-ranking must be disabled")
    rows = payload["rows"]
    for row_id in HIGHDIM_ROWS:
        cells = [row for row in rows if row.get("row_id") == row_id]
        algorithms = [row.get("algorithm_id") for row in cells]
        if set(algorithms) != set(COMPARISON_ALGORITHMS):
            raise ValueError(f"{row_id} algorithms mismatch: {algorithms}")
        ledh = [row for row in cells if row.get("algorithm_id") == LEDH_ALGORITHM_ID]
        if len(ledh) != 1:
            raise ValueError(f"{row_id} must have exactly one LEDH row")
        ledh_row = ledh[0]
        if ledh_row.get("runtime_rankable") is not False:
            raise ValueError(f"{row_id} LEDH row must not be runtime rankable")
        if not ledh_row.get("score_status"):
            raise ValueError(f"{row_id} LEDH score status must be explicit")
        if ledh_row.get("score_status") == "admitted_same_target_compact_score_n10000":
            if ledh_row.get("score_derivative_provenance") is None:
                raise ValueError(f"{row_id} admitted score must name provenance")
            provenance = str(ledh_row["score_derivative_provenance"])
            if not provenance.startswith("compact_forward_sensitivity_no_autodiff_same_scalar_"):
                raise ValueError(f"{row_id} admitted score must use compact provenance")
            if "manual_total_vjp" in provenance:
                raise ValueError(f"{row_id} historical route cannot be admitted")
        else:
            if ledh_row.get("score_derivative_provenance") is not None:
                raise ValueError(f"{row_id} blocked score must not expose admitted provenance")
            candidate_provenance = str(
                ledh_row.get("score_candidate_derivative_provenance") or ""
            )
            if "manual_total_vjp" in candidate_provenance and not str(
                ledh_row.get("score_status_reason") or ""
            ).startswith("historical manual_total_vjp"):
                raise ValueError(f"{row_id} historical candidate must be blocked explicitly")


def build_artifact(
    *,
    baseline_path: Path = BASELINE_PATH,
    ledger_path: Path = LEDGER_PATH,
    lgssm_path: Path = LGSSM_LEDGER_PATH,
    sir_path: Path = SIR_LEDGER_PATH,
    lgssm_score_path: Path = LGSSM_SCORE_PATH,
    sir_score_path: Path = SIR_SCORE_PATH,
) -> dict[str, Any]:
    baseline = _load(baseline_path)
    ledger = _load(ledger_path)
    lgssm_artifact = _load(lgssm_path)
    sir_artifact = _load(sir_path)
    lgssm_score_artifact = _load(lgssm_score_path)
    sir_score_artifact = _load(sir_score_path)
    ledger_by_row = {row["row_id"]: row for row in ledger["rows"]}
    if set(ledger_by_row) != set(HIGHDIM_ROWS):
        raise ValueError("ledger rows do not match highdim rows")

    rows: list[dict[str, Any]] = []
    for row_id in HIGHDIM_ROWS:
        baseline_rows = [
            row
            for row in baseline["rows"]
            if row.get("row_id") == row_id
            and row.get("algorithm_id") != LEDH_ALGORITHM_ID
        ]
        expected_non_ledh = set(COMPARISON_ALGORITHMS) - {LEDH_ALGORITHM_ID}
        baseline_algorithms = {row.get("algorithm_id") for row in baseline_rows}
        if baseline_algorithms != expected_non_ledh:
            raise ValueError(f"{row_id} baseline algorithms mismatch: {baseline_algorithms}")
        rows.extend(_copy_baseline_row(row, baseline_path=baseline_path) for row in baseline_rows)
        rows.append(
            _ledh_row_for(
                ledger_by_row[row_id],
                ledger_path=ledger_path,
                lgssm_artifact=lgssm_artifact,
                lgssm_path=lgssm_path,
                lgssm_score_artifact=lgssm_score_artifact,
                lgssm_score_path=lgssm_score_path,
                sir_artifact=sir_artifact,
                sir_path=sir_path,
                sir_score_artifact=sir_score_artifact,
                sir_score_path=sir_score_path,
            )
        )

    payload = {
        "benchmark": "bayesfilter_two_lane_highdim_ledh_inclusive_leaderboard_results",
        "metadata_date": DATE,
        "comparison_algorithm_ids": COMPARISON_ALGORITHMS,
        "comparator_mode": COMPARATOR_MODE,
        "runtime_cross_ranking_allowed": False,
        "manifest": {
            "baseline_artifact": _rel(baseline_path),
            "ledh_admission_ledger": _rel(ledger_path),
            "lgssm_ledh_value_artifact": _rel(lgssm_path),
            "sir_ledh_value_artifact": _rel(sir_path),
            "lgssm_ledh_score_artifact": _rel(lgssm_score_path),
            "sir_ledh_score_artifact": _rel(sir_score_path),
            "execution_mode": (
                "frozen_non_ledh_baseline_plus_fresh_ledh_value_with_phase8_score_contract_gate"
            ),
            "frozen_non_ledh_rows": True,
            "fresh_ledh_execution": True,
            "score_memory_candidate_rows": [
                LGSSM_M3_T50_ROW_ID,
                FIXED_SIR_AUSTRIA_ROW_ID,
            ],
            "score_admission_policy": (
                "phase1_validated_compact_score_artifact_required_for_admission"
            ),
        },
        "rows": rows,
        "row_summary": _row_summary(rows),
        "nonclaims": [
            "Frozen non-LEDH rows are copied from the July 3 baseline and were not rerun.",
            "Runtime cross-ranking between frozen non-LEDH rows and fresh LEDH rows is forbidden.",
            (
                "Raw score-memory artifacts are not admitted unless they pass "
                "the Phase 1 shared compact score artifact contract."
            ),
            (
                "Historical manual_total_vjp score-memory artifacts are diagnostic-only "
                "and blocked from leaderboard score admission."
            ),
            "HMC readiness, posterior correctness, and scientific superiority are not claimed.",
        ],
    }
    payload = _json_safe(payload)
    _validate_payload(payload)
    return payload


def _markdown(payload: dict[str, Any], json_path: Path) -> str:
    lines = [
        "# Two-Lane Highdim LEDH-Inclusive Leaderboard Results",
        "",
        f"Authoritative JSON artifact: `{_rel(json_path)}`.",
        "",
        f"Comparator mode: `{payload['comparator_mode']}`.",
        "",
        "Runtime cross-ranking allowed: `False`.",
        "",
        "## Rows",
        "",
        "| Row | Algorithm | Status | Avg loglik | MCSE | Score status | Runtime rankable | Reason |",
        "| --- | --- | --- | ---: | ---: | --- | --- | --- |",
    ]
    for row in payload["rows"]:
        runtime_rankable = row.get(
            "runtime_rankable",
            row.get("runtime_rankable_with_ledh", False),
        )
        lines.append(
            f"| {row['row_id']} | {row['algorithm_id']} | {row['comparison_status']} | "
            f"{_fmt(row.get('average_log_likelihood'))} | {_fmt(row.get('mc_standard_error'))} | "
            f"{row.get('score_status') or ''} | {runtime_rankable} | "
            f"{row.get('reason') or row.get('score_status_reason') or ''} |"
        )
    lines.extend(
        [
            "",
            "## Row Summary",
            "",
            "| Row | LEDH status | LEDH value status | LEDH score status | Four-way value ready | Four-way score ready |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in payload["row_summary"]:
        lines.append(
            f"| {row['row_id']} | {row['ledh_status']} | {row['ledh_value_status']} | "
            f"{row['ledh_score_status']} | {row['full_four_way_value_ready']} | "
            f"{row['full_four_way_score_ready']} |"
        )
    lines.extend(["", "## Nonclaims", ""])
    lines.extend(f"- {item}" for item in payload["nonclaims"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=BASELINE_PATH)
    parser.add_argument("--ledger", type=Path, default=LEDGER_PATH)
    parser.add_argument("--lgssm-ledh", type=Path, default=LGSSM_LEDGER_PATH)
    parser.add_argument("--sir-ledh", type=Path, default=SIR_LEDGER_PATH)
    parser.add_argument("--lgssm-score", type=Path, default=LGSSM_SCORE_PATH)
    parser.add_argument("--sir-score", type=Path, default=SIR_SCORE_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()
    payload = build_artifact(
        baseline_path=args.baseline,
        ledger_path=args.ledger,
        lgssm_path=args.lgssm_ledh,
        sir_path=args.sir_ledh,
        lgssm_score_path=args.lgssm_score,
        sir_score_path=args.sir_score,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(_markdown(payload, args.output) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
