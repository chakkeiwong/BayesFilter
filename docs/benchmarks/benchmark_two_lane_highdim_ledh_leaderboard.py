#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DATE = "2026-07-03"
DEFAULT_BASELINE = (
    ROOT / "docs/plans/bayesfilter-two-lane-highdim-leaderboard-results-2026-07-03.json"
)
DEFAULT_LEDGER = (
    ROOT
    / "docs/plans/bayesfilter-ledh-inclusive-highdim-leaderboard-phase1-row-admission-ledger-2026-07-03.json"
)
DEFAULT_JSON = (
    ROOT
    / "docs/plans/bayesfilter-two-lane-highdim-ledh-inclusive-leaderboard-dry-run-2026-07-03.json"
)
DEFAULT_MD = (
    ROOT
    / "docs/plans/bayesfilter-two-lane-highdim-ledh-inclusive-leaderboard-dry-run-2026-07-03.md"
)

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


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _display_path(path: Path) -> str:
    resolved = path if path.is_absolute() else ROOT / path
    try:
        return str(resolved.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.6g}"
    return str(value)


def _phase7_status_for_ledh(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "scope": row["row_scope"],
        "batch_status": "not_run_phase2_schema_dry_run",
        "cpu_timing_status": "not_run_phase2_schema_dry_run",
        "gpu_xla_status": "not_run_phase2_schema_dry_run",
        "timing_rank_status": "not_rankable_phase2_schema_only",
        "evidence_paths": [],
        "reason": (
            "Phase 2 emits schema and admission status only; LEDH GPU/XLA value "
            "execution begins no earlier than Phase 3/4 gates."
        ),
        "nonclaims": [
            "not runtime evidence",
            "not value correctness evidence",
            "not score correctness evidence",
        ],
    }


def _comparison_status_from_ledger(row: dict[str, Any]) -> str:
    decision = row["ledh_row_scope_decision"]
    if decision in {
        "in_scope_for_later_full_value_score_only_after_gates",
        "fixed_spatial_sir_value_arm_candidate_only",
    }:
        return "dry_run_value_candidate"
    if decision == "scoped_component_row":
        return "scoped_component_status_only"
    return "blocked"


def _row_scope_from_ledger(row: dict[str, Any]) -> str:
    decision = row["ledh_row_scope_decision"]
    if decision == "scoped_component_row":
        return "scoped_component_row"
    return "main_leaderboard_row"


def _blocked_reason_from_ledger(row: dict[str, Any]) -> str:
    pieces = []
    if row.get("blocker"):
        pieces.append(str(row["blocker"]))
    pieces.append(f"value_status={row['value_status']}")
    pieces.append(f"score_status={row['score_status']}")
    return "; ".join(pieces)


def _ledh_row_from_ledger(row: dict[str, Any], *, ledger_path: Path) -> dict[str, Any]:
    payload = {
        "algorithm_id": LEDH_ALGORITHM_ID,
        "row_id": row["row_id"],
        "comparison_status": _comparison_status_from_ledger(row),
        "row_scope": _row_scope_from_ledger(row),
        "row_admission_status": row["ledh_row_scope_decision"],
        "target_contract_status": row["target_match_status"],
        "target_match_status": row["target_match_status"],
        "target_scope": row["requested_row_scope"],
        "value_status": row["value_status"],
        "score_status": row["score_status"],
        "score_status_reason": row["required_score_admission_artifact"],
        "score_derivative_provenance": None,
        "score_coordinate_system": None,
        "score": None,
        "score_l2_norm": None,
        "average_log_likelihood": None,
        "runtime_seconds": None,
        "mc_standard_error": None,
        "mc_standard_deviation": None,
        "particle_count": None,
        "seeds": None,
        "device_provenance": "not_run_phase2_schema_dry_run",
        "comparator_provenance": "fresh_ledh_schema_dry_run",
        "source_ledger": _display_path(ledger_path),
        "runtime_rankable": False,
        "reason": _blocked_reason_from_ledger(row),
        "phase7_batch_gpu_xla_status": {},
        "required_phase2_work": list(row.get("required_phase2_work", [])),
        "required_score_admission_artifact": row["required_score_admission_artifact"],
        "current_adapter_evidence": list(row.get("current_adapter_evidence", [])),
        "forbidden_claims": list(row.get("forbidden_claims", [])),
        "nonclaims": [
            "phase2 schema dry-run only",
            "not LEDH value execution",
            "not LEDH score execution",
            "not HMC readiness evidence",
            "not runtime ranking evidence",
        ],
    }
    payload["phase7_batch_gpu_xla_status"] = _phase7_status_for_ledh(payload)
    return payload


def _copy_baseline_row(row: dict[str, Any], *, baseline_path: Path) -> dict[str, Any]:
    copied = dict(row)
    copied["comparator_provenance"] = "frozen_non_ledh_baseline"
    copied["source_baseline"] = _display_path(baseline_path)
    copied["runtime_rankable_with_ledh"] = False
    copied.setdefault("nonclaims", [])
    copied["nonclaims"] = list(copied["nonclaims"]) + [
        "frozen non-LEDH baseline row copied for status/value context",
        "not rerun under the LEDH GPU/XLA harness",
        "not runtime-rankable against fresh LEDH rows",
    ]
    if copied.get("algorithm_id") == "zhao_cui_scalar_or_multistate":
        copied["nonclaims"] += [
            "frozen Zhao-Cui admission retains the route-specific analytical-score status of its source artifact",
        ]
    return copied


def _row_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for row_id in HIGHDIM_ROWS:
        row_cells = [row for row in rows if row["row_id"] == row_id]
        ledh = next(row for row in row_cells if row["algorithm_id"] == LEDH_ALGORITHM_ID)
        executed = [
            row["algorithm_id"]
            for row in row_cells
            if str(row["comparison_status"]).startswith("executed")
        ]
        summaries.append(
            {
                "row_id": row_id,
                "row_scope": ledh["row_scope"],
                "comparison_algorithm_count": len(row_cells),
                "comparison_algorithms": [row["algorithm_id"] for row in row_cells],
                "executed_algorithms": executed,
                "ledh_status": ledh["comparison_status"],
                "ledh_value_status": ledh["value_status"],
                "ledh_score_status": ledh["score_status"],
                "ledh_runtime_rankable": False,
                "full_four_way_ready": False,
                "blocked_or_missing_algorithms": [
                    row["algorithm_id"]
                    for row in row_cells
                    if not str(row["comparison_status"]).startswith("executed")
                ],
            }
        )
    return summaries


def _validate_payload(payload: dict[str, Any]) -> None:
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
        if not ledh_row.get("reason"):
            raise ValueError(f"{row_id} LEDH row must have a reason")
        if ledh_row.get("runtime_rankable") is not False:
            raise ValueError(f"{row_id} LEDH row must not be runtime rankable")
        if ledh_row.get("score_status") and not ledh_row.get("score_status_reason"):
            raise ValueError(f"{row_id} score status must have a reason")


def build_artifact(
    *,
    baseline_path: Path = DEFAULT_BASELINE,
    ledger_path: Path = DEFAULT_LEDGER,
) -> dict[str, Any]:
    baseline = _load(baseline_path)
    ledger = _load(ledger_path)
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
        baseline_algorithms = {row.get("algorithm_id") for row in baseline_rows}
        expected_non_ledh = set(COMPARISON_ALGORITHMS) - {LEDH_ALGORITHM_ID}
        if baseline_algorithms != expected_non_ledh:
            raise ValueError(f"{row_id} baseline algorithms mismatch: {baseline_algorithms}")
        rows.extend(_copy_baseline_row(row, baseline_path=baseline_path) for row in baseline_rows)
        rows.append(_ledh_row_from_ledger(ledger_by_row[row_id], ledger_path=ledger_path))

    payload = {
        "benchmark": "bayesfilter_two_lane_highdim_ledh_inclusive_leaderboard_dry_run",
        "metadata_date": DATE,
        "comparison_algorithm_ids": COMPARISON_ALGORITHMS,
        "comparator_mode": COMPARATOR_MODE,
        "runtime_cross_ranking_allowed": False,
        "manifest": {
            "baseline_artifact": _display_path(baseline_path),
            "ledh_admission_ledger": _display_path(ledger_path),
            "execution_mode": "phase2_schema_dry_run_no_ledh_values_executed",
            "frozen_non_ledh_rows": True,
            "fresh_ledh_execution": False,
            "initial_executable_arms": ledger["phase2_handoff"]["initial_executable_arms"],
            "initially_blocked_arms": ledger["phase2_handoff"]["initially_blocked_arms"],
        },
        "rows": rows,
        "row_summary": _row_summary(rows),
        "nonclaims": [
            "This is a Phase 2 schema dry-run artifact.",
            "No LEDH value was executed by this artifact.",
            "No LEDH score was executed by this artifact.",
            "Frozen non-LEDH rows are not rerun under the LEDH GPU/XLA harness.",
            "Runtime cross-ranking between frozen non-LEDH rows and LEDH rows is forbidden.",
            "Rows marked blocked or scoped are not full LEDH observed-data filtering admissions.",
            "HMC readiness, posterior correctness, and scientific superiority are not claimed.",
        ],
    }
    payload = _json_safe(payload)
    _validate_payload(payload)
    return payload


def _markdown(payload: dict[str, Any], json_path: Path) -> str:
    lines = [
        "# Two-Lane Highdim LEDH-Inclusive Leaderboard Dry Run",
        "",
        f"Authoritative JSON artifact: `{_display_path(json_path)}`.",
        "",
        f"Comparator mode: `{payload['comparator_mode']}`.",
        "",
        "Runtime cross-ranking allowed: `False`.",
        "",
        "## Rows",
        "",
        "| Row | Algorithm | Status | Value status | Score status | Runtime rankable | Avg loglik | MC SE | Reason |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['row_id']} | {row['algorithm_id']} | {row['comparison_status']} | "
            f"{row.get('value_status') or ''} | {row.get('score_status') or ''} | "
            f"{row.get('runtime_rankable', row.get('runtime_rankable_with_ledh', False))} | "
            f"{_fmt(row.get('average_log_likelihood'))} | {_fmt(row.get('mc_standard_error'))} | "
            f"{row.get('reason') or row.get('score_status_reason') or ''} |"
        )
    lines.extend(
        [
            "",
            "## Row Summary",
            "",
            "| Row | LEDH status | LEDH value status | LEDH score status | Full four-way ready |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in payload["row_summary"]:
        lines.append(
            f"| {row['row_id']} | {row['ledh_status']} | {row['ledh_value_status']} | "
            f"{row['ledh_score_status']} | {row['full_four_way_ready']} |"
        )
    lines.extend(["", "## Nonclaims", ""])
    lines.extend(f"- {item}" for item in payload["nonclaims"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()
    payload = build_artifact(baseline_path=args.baseline, ledger_path=args.ledger)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_output.write_text(_markdown(payload, args.output) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
