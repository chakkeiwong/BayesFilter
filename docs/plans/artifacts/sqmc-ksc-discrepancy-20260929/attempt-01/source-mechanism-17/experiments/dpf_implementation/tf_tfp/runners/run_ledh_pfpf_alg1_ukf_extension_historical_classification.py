"""Classify old LEDH-PFPF extension and historical lanes for P8.

This P8 runner is intentionally pure Python.  It does not import TensorFlow,
FilterFlow, or old LEDH-PFPF-OT implementation modules.  The phase is a
classification gate: old annealed, FilterFlow-matched, and auxiliary-flow
repair artifacts are historical/scaffolding context only unless a separate
reviewed extension plan reruns them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
REPORT_DIR = REPO_ROOT / "experiments" / "dpf_implementation" / "reports"
OUTPUT_DIR = REPORT_DIR / "outputs"

MODULE_PATH = (
    "experiments.dpf_implementation.tf_tfp.runners."
    "run_ledh_pfpf_alg1_ukf_extension_historical_classification"
)
PLAN_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-ledh-pfpf-alg1-ukf-ledh-pfpf-ot-rerun-p8-filterflow-annealed-historical-regression-subplan-2026-06-10.md"
)
RESULT_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-ledh-pfpf-alg1-ukf-ledh-pfpf-ot-rerun-p8-filterflow-annealed-historical-regression-result-2026-06-10.md"
)
REVIEW_LEDGER_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-ledh-pfpf-alg1-ukf-ledh-pfpf-ot-rerun-claude-review-ledger-2026-06-10.md"
)
JSON_PATH = OUTPUT_DIR / "dpf_ledh_pfpf_alg1_ukf_extension_historical_classification_2026-06-10.json"
REPORT_PATH = REPORT_DIR / "dpf-ledh-pfpf-alg1-ukf-extension-historical-classification-2026-06-10.md"

LOCAL_PASS_DECISION = "LOCAL_PASS_P8_EXTENSION_HISTORICAL_CLASSIFICATION_PENDING_CLAUDE_REVIEW"
VETO_DECISION = "P8_EXTENSION_HISTORICAL_CLASSIFICATION_VETO_PENDING_REVIEW"
ALLOWED_DISPOSITIONS = {
    "ALG1_EXTENSION_RERUN",
    "HISTORICAL_ONLY_NOT_EVIDENCE",
    "SCAFFOLDING_ONLY",
    "BLOCKED_REQUIRES_SEPARATE_PLAN",
}

PREREQUISITE_RESULTS = [
    "docs/plans/bayesfilter-dpf-ledh-pfpf-alg1-ukf-ledh-pfpf-ot-rerun-p0-inventory-registry-result-2026-06-10.md",
    "docs/plans/bayesfilter-dpf-ledh-pfpf-alg1-ukf-ledh-pfpf-ot-rerun-p1-direct-lgssm-range-bearing-result-2026-06-10.md",
    "docs/plans/bayesfilter-dpf-ledh-pfpf-alg1-ukf-ledh-pfpf-ot-rerun-p2-v2-contracts-result-2026-06-10.md",
    "docs/plans/bayesfilter-dpf-ledh-pfpf-alg1-ukf-ledh-pfpf-ot-rerun-p3-v2-values-result-2026-06-10.md",
    "docs/plans/bayesfilter-dpf-ledh-pfpf-alg1-ukf-ledh-pfpf-ot-rerun-p4-v2-gradients-result-2026-06-10.md",
    "docs/plans/bayesfilter-dpf-ledh-pfpf-alg1-ukf-ledh-pfpf-ot-rerun-p5-filter-oracle-statistical-closeness-result-2026-06-10.md",
    "docs/plans/bayesfilter-dpf-ledh-pfpf-alg1-ukf-ledh-pfpf-ot-rerun-p6-cross-filter-calibration-result-2026-06-10.md",
    "docs/plans/bayesfilter-dpf-ledh-pfpf-alg1-ukf-ledh-pfpf-ot-rerun-p7-p44-blocker-closure-result-2026-06-10.md",
]

OLD_LANES = [
    {
        "lane_id": "annealed_transport_lgssm",
        "old_runner": "experiments/dpf_implementation/tf_tfp/runners/run_ledh_pfpf_annealed_transport_lgssm_tf.py",
        "json_prefix": "dpf_ledh_pfpf_annealed_transport",
        "report_prefix": "dpf-ledh-pfpf-annealed-transport",
        "disposition": "HISTORICAL_ONLY_NOT_EVIDENCE",
        "route_class": "BAYESFILTER_EXTENSION_NOT_SOURCE_CORE",
        "reason": (
            "Annealed transport is a BayesFilter extension diagnostic.  It is "
            "not the Li-Coates Algorithm 1 UKF covariance-lifecycle core."
        ),
        "future_action": "A future extension rerun requires a separate reviewed amendment.",
    },
    {
        "lane_id": "filterflow_matched_ledh_pfpf_ot",
        "old_runner": "experiments/dpf_implementation/tf_tfp/runners/run_filterflow_matched_ledh_pfpf_ot_tf.py",
        "json_prefix": "dpf_filterflow_matched_ledh_pfpf_ot",
        "report_prefix": "dpf-filterflow-matched-ledh-pfpf-ot",
        "disposition": "SCAFFOLDING_ONLY",
        "route_class": "SAME_CONTRACT_ADAPTER_SCAFFOLDING_NOT_EVIDENCE",
        "reason": (
            "Matched FilterFlow rows are same-contract adapter diagnostics.  "
            "They do not prove FilterFlow correctness or source Algorithm 1 "
            "UKF correctness."
        ),
        "future_action": "Use only as adapter/protocol scaffolding in a separate plan.",
    },
    {
        "lane_id": "auxiliary_flow_source_faithful_repair",
        "old_runner": "experiments/dpf_implementation/tf_tfp/runners/run_ledh_pfpf_source_faithful_repair_tf.py",
        "json_prefix": "dpf_ledh_pfpf_source_faithful_repair",
        "report_prefix": "dpf-ledh-pfpf-source-faithful-repair",
        "disposition": "HISTORICAL_ONLY_NOT_EVIDENCE",
        "route_class": "AUXILIARY_FLOW_LEDHPFPF_NOT_ALGORITHM1_UKF_CORE",
        "reason": (
            "The repair is source-faithful for the auxiliary-flow LEDH/PF-PF "
            "formulation, not for the current Algorithm 1 UKF covariance "
            "lifecycle replacement route."
        ),
        "future_action": (
            "Any use as extension or comparative evidence requires a separate "
            "reviewed plan and cannot supersede P1-P7 Algorithm 1 UKF rows."
        ),
    },
]


class P8ClassificationError(ValueError):
    """Raised when the P8 classification artifact violates its contract."""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only:
        _validate_payload(_load_json(JSON_PATH))
        print("P8_EXTENSION_HISTORICAL_CLASSIFICATION_VALIDATED")
        return 0

    start = time.perf_counter()
    payload = _run()
    payload["run_manifest"]["wall_time_seconds"] = time.perf_counter() - start
    payload["reproducibility_digest"] = _digest_payload(payload)
    markdown = _markdown(payload)
    _write_json(JSON_PATH, payload)
    _write_text(REPORT_PATH, markdown)
    _write_text(REPO_ROOT / RESULT_PATH, markdown)
    _validate_payload(payload)
    print(payload["decision"])
    return 0


def _run() -> dict[str, Any]:
    rows = [_classification_row(spec) for spec in OLD_LANES]
    veto = _veto_diagnostics(rows)
    decision = LOCAL_PASS_DECISION if not any(veto.values()) else VETO_DECISION
    return {
        "metadata_date": "2026-06-10",
        "created_at_utc": _utc_now(),
        "phase": "P8",
        "execution_route": "VISIBLE_IN_DIALOGUE",
        "execution_mode": "PURE_PYTHON_CLASSIFICATION_ONLY",
        "decision": decision,
        "question": (
            "Which old FilterFlow-matched, annealed-transport, and "
            "source-faithful-repair lanes should remain historical/scaffolding "
            "rather than source Algorithm 1 UKF evidence?"
        ),
        "evidence_contract": {
            "baseline_comparator": (
                "P1-P7 Algorithm 1 UKF replacement artifacts plus old "
                "FilterFlow/annealed/source-faithful artifacts as historical "
                "coverage definitions only."
            ),
            "primary_criterion": (
                "Every old extension or historical lane has exactly one "
                "reviewable disposition and cannot be mistaken for source "
                "Algorithm 1 UKF evidence."
            ),
            "promotion_policy": (
                "No P8 row is promoted.  No extension rerun is performed "
                "without a separate reviewed amendment."
            ),
        },
        "skeptical_plan_audit": {
            "status": "PASS_FOR_CLASSIFICATION_ONLY",
            "wrong_baseline_control": (
                "Old artifacts are coverage/history/scaffolding, not truth "
                "sources or Algorithm 1 UKF evidence."
            ),
            "proxy_promotion_control": (
                "Finite old rows and same-contract residuals are explanatory "
                "only and cannot promote correctness."
            ),
            "stop_condition_control": (
                "Missing old lanes, source-core labels on OT/annealed rows, "
                "FilterFlow mutation, or any extension rerun without a "
                "separate plan veto P8."
            ),
            "environment_control": "Pure-Python classification; TensorFlow, FilterFlow, and old DPF modules are not imported.",
        },
        "prerequisite_results": _prerequisite_rows(),
        "classification_rows": rows,
        "classification_summary": _classification_summary(rows),
        "veto_diagnostics": veto,
        "decision_table": _decision_table(decision, veto, rows),
        "historical_quarantine_policy": {
            "old_ledh_pfpf_ot_current_evidence": False,
            "ot_or_annealed_source_algorithm1_core": False,
            "filterflow_correctness_proof": False,
            "extension_rerun_performed": False,
        },
        "run_manifest": _manifest(),
        "nonclaims": _nonclaims(),
    }


def _classification_row(spec: dict[str, str]) -> dict[str, Any]:
    json_artifacts = _artifact_paths(OUTPUT_DIR, spec["json_prefix"], ".json")
    report_artifacts = _artifact_paths(REPORT_DIR, spec["report_prefix"], ".md")
    return {
        "lane_id": spec["lane_id"],
        "old_runner": spec["old_runner"],
        "old_runner_exists": (REPO_ROOT / spec["old_runner"]).exists(),
        "disposition": spec["disposition"],
        "route_class": spec["route_class"],
        "source_algorithm1_ukf_evidence": False,
        "algorithm1_extension_rerun_performed": False,
        "old_ledh_pfpf_ot_used_as_current_evidence": False,
        "mutates_filterflow": False,
        "same_contract_residual_used_as_correctness_proof": False,
        "separate_plan_required_for_future_rerun": spec["disposition"] != "ALG1_EXTENSION_RERUN",
        "reason": spec["reason"],
        "future_action": spec["future_action"],
        "json_artifacts": [_artifact_summary(path) for path in json_artifacts],
        "report_artifacts": [_file_summary(path) for path in report_artifacts],
    }


def _artifact_paths(root: Path, prefix: str, suffix: str) -> list[Path]:
    return sorted(path for path in root.glob(f"{prefix}*{suffix}") if path.is_file())


def _artifact_summary(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    summary = payload.get("summary")
    if summary is None:
        summary = {
            "row_count": len(payload.get("rows", [])),
            "row_summary_count": len(payload.get("row_summaries", [])),
            "extra_row_summary_count": len(payload.get("extra_row_summaries", [])),
            "true_veto_count": _true_veto_count(payload.get("veto_diagnostics", {})),
        }
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "sha256": _file_sha256(path),
        "decision": payload.get("decision"),
        "created_at_utc": payload.get("created_at_utc"),
        "summary": summary,
        "run_manifest_present": "run_manifest" in payload,
        "nonclaims_present": "nonclaims" in payload or "non_implications" in payload,
    }


def _file_summary(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "sha256": _file_sha256(path),
        "exists": path.exists(),
    }


def _prerequisite_rows() -> list[dict[str, Any]]:
    rows = []
    for path_text in PREREQUISITE_RESULTS:
        path = REPO_ROOT / path_text
        rows.append(
            {
                "path": path_text,
                "exists": path.exists(),
                "sha256": _file_sha256(path) if path.exists() else None,
            }
        )
    return rows


def _classification_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    dispositions: dict[str, int] = {}
    for row in rows:
        dispositions[row["disposition"]] = dispositions.get(row["disposition"], 0) + 1
    return {
        "lane_count": len(rows),
        "disposition_counts": dispositions,
        "source_algorithm1_ukf_evidence_rows": sum(
            1 for row in rows if row["source_algorithm1_ukf_evidence"]
        ),
        "extension_rerun_rows": sum(
            1 for row in rows if row["algorithm1_extension_rerun_performed"]
        ),
        "json_artifact_count": sum(len(row["json_artifacts"]) for row in rows),
        "report_artifact_count": sum(len(row["report_artifacts"]) for row in rows),
    }


def _veto_diagnostics(rows: list[dict[str, Any]]) -> dict[str, bool]:
    return {
        "p1_p7_prerequisite_missing": any(not row["exists"] for row in _prerequisite_rows()),
        "missing_required_old_lane": len(rows) != len(OLD_LANES)
        or any(not row["old_runner_exists"] for row in rows)
        or any(not row["json_artifacts"] for row in rows),
        "invalid_disposition": any(row["disposition"] not in ALLOWED_DISPOSITIONS for row in rows),
        "old_ledh_pfpf_ot_used_as_current_evidence": any(
            row["old_ledh_pfpf_ot_used_as_current_evidence"] for row in rows
        ),
        "ot_or_annealed_called_source_algorithm1_core": any(
            row["source_algorithm1_ukf_evidence"]
            and row["lane_id"] in {"annealed_transport_lgssm", "filterflow_matched_ledh_pfpf_ot"}
            for row in rows
        ),
        "auxiliary_flow_repair_used_as_current_algorithm1_ukf_evidence": any(
            row["lane_id"] == "auxiliary_flow_source_faithful_repair"
            and row["source_algorithm1_ukf_evidence"]
            for row in rows
        ),
        "filterflow_mutation_required_or_performed": any(row["mutates_filterflow"] for row in rows),
        "same_contract_residual_promoted_to_correctness_proof": any(
            row["same_contract_residual_used_as_correctness_proof"] for row in rows
        ),
        "extension_rerun_without_separate_plan": any(
            row["algorithm1_extension_rerun_performed"]
            and row["separate_plan_required_for_future_rerun"]
            for row in rows
        ),
        "classification_missing_run_manifest_status": any(
            not all(item["run_manifest_present"] for item in row["json_artifacts"])
            for row in rows
        ),
    }


def _decision_table(
    decision: str,
    veto: dict[str, bool],
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summary = _classification_summary(rows)
    return [
        {
            "decision": decision,
            "primary_criterion_status": (
                f"{summary['lane_count']} old extension/historical lanes classified; "
                f"{summary['source_algorithm1_ukf_evidence_rows']} source Algorithm 1 UKF evidence rows emitted"
            ),
            "veto_diagnostic_status": {key: value for key, value in veto.items() if value} or "no structural vetoes",
            "main_uncertainty": "future extension reruns require separate reviewed plans",
            "next_justified_action": "Claude read-only P8 review, then P9 closeout/supersession ledger",
            "not_concluded": "no FilterFlow correctness proof, no OT or annealed source-core claim, no production claim",
        }
    ]


def _validate_payload(payload: dict[str, Any]) -> None:
    required = {
        "decision",
        "classification_rows",
        "classification_summary",
        "veto_diagnostics",
        "decision_table",
        "run_manifest",
        "nonclaims",
        "reproducibility_digest",
    }
    missing = required.difference(payload)
    if missing:
        raise P8ClassificationError(f"missing payload fields {sorted(missing)}")
    if payload["decision"] != LOCAL_PASS_DECISION:
        raise P8ClassificationError(f"invalid P8 decision {payload['decision']}")
    if any(bool(value) for value in payload["veto_diagnostics"].values()):
        raise P8ClassificationError(f"true veto diagnostics: {payload['veto_diagnostics']}")
    rows = payload["classification_rows"]
    if len(rows) != len(OLD_LANES):
        raise P8ClassificationError("unexpected P8 lane count")
    for row in rows:
        if row["disposition"] not in ALLOWED_DISPOSITIONS:
            raise P8ClassificationError(f"invalid disposition {row['disposition']}")
        if row["source_algorithm1_ukf_evidence"]:
            raise P8ClassificationError("P8 must not emit source Algorithm 1 UKF evidence rows")
        if row["algorithm1_extension_rerun_performed"]:
            raise P8ClassificationError("P8 classification must not perform extension reruns")
        if not row["json_artifacts"]:
            raise P8ClassificationError(f"missing JSON artifacts for {row['lane_id']}")


def _markdown(payload: dict[str, Any]) -> str:
    summary = payload["classification_summary"]
    lines = [
        "# P8 Result: Extension And Historical Classification",
        "",
        "metadata_date: 2026-06-10",
        "phase: P8",
        f"status: {payload['decision']}",
        "",
        "## Evidence Contract",
        "",
        "| Field | Contract |",
        "| --- | --- |",
        f"| Question | {payload['question']} |",
        f"| Baseline/comparator | {payload['evidence_contract']['baseline_comparator']} |",
        f"| Primary criterion | {payload['evidence_contract']['primary_criterion']} |",
        f"| Promotion policy | {payload['evidence_contract']['promotion_policy']} |",
        "",
        "## Decision Table",
        "",
        "| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["decision_table"]:
        lines.append(
            f"| `{row['decision']}` | {row['primary_criterion_status']} | "
            f"`{row['veto_diagnostic_status']}` | {row['main_uncertainty']} | "
            f"{row['next_justified_action']} | {row['not_concluded']} |"
        )
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Classified lanes: `{summary['lane_count']}`.",
            f"- Disposition counts: `{summary['disposition_counts']}`.",
            f"- Source Algorithm 1 UKF evidence rows emitted: `{summary['source_algorithm1_ukf_evidence_rows']}`.",
            f"- Extension reruns performed: `{summary['extension_rerun_rows']}`.",
            "",
            "## Classification Rows",
            "",
            "| Lane | Disposition | Route class | Source Alg1 UKF evidence | Extension rerun | Reason |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in payload["classification_rows"]:
        lines.append(
            f"| `{row['lane_id']}` | `{row['disposition']}` | `{row['route_class']}` | "
            f"`{row['source_algorithm1_ukf_evidence']}` | "
            f"`{row['algorithm1_extension_rerun_performed']}` | {row['reason']} |"
        )
    lines.extend(
        [
            "",
            "## Veto Diagnostics",
            "",
            "| Diagnostic | Status |",
            "| --- | --- |",
            *[f"| `{key}` | `{value}` |" for key, value in payload["veto_diagnostics"].items()],
            "",
            "## Run Manifest",
            "",
            "```json",
            json.dumps(payload["run_manifest"], indent=2, sort_keys=True),
            "```",
            "",
            "## Nonclaims",
            "",
            *[f"- {item}" for item in payload["nonclaims"]],
            "",
        ]
    )
    return "\n".join(lines)


def _manifest() -> dict[str, Any]:
    return {
        **_git_manifest(),
        "python_version": platform.python_version(),
        "command": "python -m " + MODULE_PATH,
        "execution_mode": "PURE_PYTHON_CLASSIFICATION_ONLY",
        "cpu_gpu_status": "not_applicable_no_tensorflow_import",
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "review_ledger_path": REVIEW_LEDGER_PATH,
        "json_path": str(JSON_PATH.relative_to(REPO_ROOT)),
        "report_path": str(REPORT_PATH.relative_to(REPO_ROOT)),
        "old_lane_count": len(OLD_LANES),
    }


def _nonclaims() -> list[str]:
    return [
        "P8 does not rerun FilterFlow, annealed transport, or auxiliary-flow repair lanes.",
        "P8 does not treat OT or annealed transport as source Li-Coates Algorithm 1 core.",
        "P8 does not treat FilterFlow same-contract residuals as a correctness proof.",
        "P8 does not revive old dpf_ledh_pfpf_ot artifacts as current evidence.",
        "P8 does not establish production, HMC, GPU, public API, or default-policy readiness.",
    ]


def _true_veto_count(veto: dict[str, Any]) -> int:
    return sum(1 for value in veto.values() if bool(value))


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest_payload(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _git_manifest() -> dict[str, str]:
    return {
        "branch": _git(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "commit": _git(["git", "rev-parse", "HEAD"]),
        "dirty_state_summary": _git(["git", "status", "--short"]) or "clean",
    }


def _git(args: list[str]) -> str:
    completed = subprocess.run(args, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
