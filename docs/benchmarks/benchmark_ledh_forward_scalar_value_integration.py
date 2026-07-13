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
    ACTUAL_SV_ROW_ID,
    FIXED_SIR_AUSTRIA_ROW_ID,
    GENERALIZED_SV_ROW_ID,
    KSC_SV_ROW_ID,
    LEDH_FORWARD_ADMISSION_STATUS_ADMITTED,
    LEDH_OUTPUT_TENSOR_FIELD_LOG_LIKELIHOOD,
    LEDH_TARGET_SCALAR_OBSERVED_DATA_LOG_LIKELIHOOD,
    LGSSM_M3_T50_ROW_ID,
    PARAMETERIZED_SIR_DIAGNOSTIC_ROW_ID,
    PREDATOR_PREY_ROW_ID,
    validate_ledh_forward_scalar_artifact,
)


DATE = "2026-07-07"
SCHEMA_VERSION = "bayesfilter.highdim.ledh_forward_scalar_value_integration.v1"
LEDH_ALGORITHM_ID = "ledh_pfpf_ot"
SCORE_INTEGRATION_STATUS = "blocked_out_of_scope_forward_scalar_only"

DEFAULT_OUTPUT = (
    ROOT / "docs/plans/bayesfilter-ledh-forward-scalar-value-integration-results-2026-07-07.json"
)
DEFAULT_MARKDOWN_OUTPUT = (
    ROOT / "docs/plans/bayesfilter-ledh-forward-scalar-value-integration-results-2026-07-07.md"
)

MAIN_ROW_ARTIFACTS = (
    (
        LGSSM_M3_T50_ROW_ID,
        ROOT / "docs/plans/ledh-phase2-lgssm-forward-scalar-artifact-2026-07-07.json",
    ),
    (
        FIXED_SIR_AUSTRIA_ROW_ID,
        ROOT / "docs/plans/ledh-phase3-fixed-sir-forward-scalar-artifact-2026-07-07.json",
    ),
    (
        PREDATOR_PREY_ROW_ID,
        ROOT / "docs/plans/ledh-phase4-predator-prey-forward-scalar-artifact-2026-07-07.json",
    ),
    (
        ACTUAL_SV_ROW_ID,
        ROOT / "docs/plans/ledh-phase5-actual-sv-forward-scalar-artifact-2026-07-07.json",
    ),
    (
        GENERALIZED_SV_ROW_ID,
        ROOT / "docs/plans/ledh-phase6-generalized-sv-forward-scalar-artifact-2026-07-07.json",
    ),
    (
        KSC_SV_ROW_ID,
        ROOT / "docs/plans/ledh-phase7-ksc-sv-forward-scalar-artifact-2026-07-07.json",
    ),
)


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


def _sample_summary(values: list[float]) -> dict[str, float | None]:
    mean = statistics.fmean(values)
    if len(values) < 2:
        return {"mean": mean, "sample_sd": None, "mcse": None}
    sample_sd = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
    return {"mean": mean, "sample_sd": sample_sd, "mcse": sample_sd / math.sqrt(len(values))}


def _timing_diagnostics(artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        "compile_and_first_call_seconds": artifact.get("compile_and_first_call_seconds"),
        "warm_call_timing_summary_seconds": artifact.get("warm_call_timing_summary_seconds"),
        "output_devices": list(artifact.get("output_devices", [])),
        "device": artifact.get("device"),
        "device_scope": artifact.get("device_scope"),
        "runtime_rankable": False,
        "runtime_rankable_with_non_ledh": False,
        "runtime_cross_ranking_allowed": False,
        "diagnostic_only": True,
    }


def _row_semantics(row_id: str, artifact: dict[str, Any]) -> dict[str, Any]:
    if row_id == KSC_SV_ROW_ID:
        semantics = dict(artifact.get("ksc_sv_semantics", {}))
        return {
            "target_family": "ksc_finite_gaussian_mixture_surrogate",
            "target_policy_label_preserved": True,
            "target_observation_density": semantics.get("target_observation_density"),
            "target_transform": semantics.get("target_transform"),
            "transform_offset": semantics.get("transform_offset"),
            "ksc_mixture_is_target_likelihood": semantics.get(
                "ksc_mixture_is_target_likelihood"
            ),
            "actual_sv_exact_log_chi_square_target_used": semantics.get(
                "actual_sv_exact_log_chi_square_target_used"
            ),
            "generalized_sv_target_used": semantics.get("generalized_sv_target_used"),
            "legacy_raw_gaussian_callback_used": semantics.get(
                "legacy_raw_gaussian_callback_used"
            ),
        }
    if row_id == ACTUAL_SV_ROW_ID:
        semantics = dict(artifact.get("actual_sv_semantics", {}))
        return {
            "target_family": "exact_transformed_actual_sv_log_chi_square",
            "target_policy_label_preserved": True,
            "target_observation_density": semantics.get("target_observation_density")
            or artifact.get("target_observation_density"),
            "transform_offset": semantics.get("transform_offset")
            if "transform_offset" in semantics
            else artifact.get("transform_offset"),
        }
    if row_id == GENERALIZED_SV_ROW_ID:
        semantics = dict(artifact.get("generalized_sv_semantics", {}))
        return {
            "target_family": "source_route_prior_mean_generalized_sv",
            "target_policy_label_preserved": True,
            "target_observation_density": semantics.get("target_observation_density")
            or artifact.get("target_observation_density"),
        }
    if row_id == PREDATOR_PREY_ROW_ID:
        semantics = dict(artifact.get("predator_prey_semantics", {}))
        return {
            "target_family": "additive_gaussian_predator_prey",
            "target_policy_label_preserved": True,
            "flow_observation_contract": semantics.get("flow_observation_contract"),
        }
    if row_id == FIXED_SIR_AUSTRIA_ROW_ID:
        semantics = dict(artifact.get("sir_semantics", {}))
        return {
            "target_family": "fixed_sir_infectious_components_gaussian",
            "target_policy_label_preserved": True,
            "state_dimension": semantics.get("state_dimension"),
            "observation_dimension": semantics.get("observation_dimension"),
        }
    return {
        "target_family": "lgssm_gaussian_observation_density",
        "target_policy_label_preserved": True,
    }


def _integration_row(
    *,
    expected_row_id: str,
    source_path: Path,
) -> dict[str, Any]:
    artifact = _load(source_path)
    normalized = validate_ledh_forward_scalar_artifact(
        artifact,
        expected_row_id=expected_row_id,
        require_admitted=True,
    )
    log_values = [float(value) for value in normalized["log_likelihood_by_seed"]]
    avg_values = [float(value) for value in normalized["average_log_likelihood_by_seed"]]
    if any(not math.isfinite(value) for value in log_values + avg_values):
        raise ValueError(f"{expected_row_id} has nonfinite likelihood values")
    if normalized["admission_status"] != LEDH_FORWARD_ADMISSION_STATUS_ADMITTED:
        raise ValueError(f"{expected_row_id} is not admitted")
    row = {
        "algorithm_id": LEDH_ALGORITHM_ID,
        "row_id": expected_row_id,
        "row_scope": "main_observed_data_filtering_row",
        "source_artifact": _rel(source_path),
        "admission_status": normalized["admission_status"],
        "value_status": "admitted_forward_scalar_value_only",
        "score_integration_status": SCORE_INTEGRATION_STATUS,
        "target_scalar": normalized["target_scalar"],
        "target_output_tensor_field": normalized["target_output_tensor_field"],
        "target_observation_policy": normalized["target_observation_policy"],
        "flow_observation_policy": normalized["flow_observation_policy"],
        "target_density_used_for_correction": normalized[
            "target_density_used_for_correction"
        ],
        "target_density_fields": list(normalized["target_density_fields"]),
        "proposal_flow_fields": list(normalized["proposal_flow_fields"]),
        "correction_formula": normalized["correction_formula"],
        "theta_coordinate_system": normalized["theta_coordinate_system"],
        "theta_values": list(normalized["theta_values"]),
        "time_steps": normalized["time_steps"],
        "num_particles": normalized["num_particles"],
        "batch_seeds": list(normalized["batch_seeds"]),
        "log_likelihood_by_seed": log_values,
        "average_log_likelihood_by_seed": avg_values,
        "log_likelihood_summary": _sample_summary(log_values),
        "average_log_likelihood_summary": _sample_summary(avg_values),
        "runtime_diagnostics": _timing_diagnostics(artifact),
        "row_semantics": _row_semantics(expected_row_id, artifact),
        "runtime_rankable": False,
        "runtime_cross_ranking_allowed": False,
        "nonclaims": list(normalized["nonclaims"])
        + [
            "phase8 value-only integration row",
            "not score admission",
            "not score correctness",
            "not runtime-rankable against frozen non-LEDH rows",
            "not all-algorithm comparison evidence",
        ],
    }
    forbidden_score_keys = {
        "score",
        "score_l2_norm",
        "score_status",
        "score_derivative_provenance",
        "score_coordinate_system",
        "score_evidence_artifact",
    }
    leaked = forbidden_score_keys.intersection(row)
    if leaked:
        raise ValueError(f"{expected_row_id} row leaked score keys: {sorted(leaked)}")
    return row


def _validate_payload(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("wrong integration schema_version")
    if payload.get("target_scalar") != LEDH_TARGET_SCALAR_OBSERVED_DATA_LOG_LIKELIHOOD:
        raise ValueError("wrong target scalar")
    if payload.get("target_output_tensor_field") != LEDH_OUTPUT_TENSOR_FIELD_LOG_LIKELIHOOD:
        raise ValueError("wrong target output tensor field")
    if payload.get("score_integration_status") != SCORE_INTEGRATION_STATUS:
        raise ValueError("score integration must be blocked")
    if payload.get("runtime_cross_ranking_allowed") is not False:
        raise ValueError("runtime cross-ranking must be disabled")
    if payload.get("all_algorithm_comparison_allowed") is not False:
        raise ValueError("all-algorithm comparison must be disabled")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("rows must be a list")
    expected_ids = [row_id for row_id, _path in MAIN_ROW_ARTIFACTS]
    row_ids = [row.get("row_id") for row in rows]
    if row_ids != expected_ids:
        raise ValueError(f"main row ids mismatch: {row_ids}")
    if PARAMETERIZED_SIR_DIAGNOSTIC_ROW_ID in row_ids:
        raise ValueError("parameterized SIR diagnostic row must not be a main row")
    for row in rows:
        if row.get("target_scalar") != LEDH_TARGET_SCALAR_OBSERVED_DATA_LOG_LIKELIHOOD:
            raise ValueError(f"{row.get('row_id')} wrong row target scalar")
        if row.get("target_output_tensor_field") != LEDH_OUTPUT_TENSOR_FIELD_LOG_LIKELIHOOD:
            raise ValueError(f"{row.get('row_id')} wrong row output tensor")
        if row.get("score_integration_status") != SCORE_INTEGRATION_STATUS:
            raise ValueError(f"{row.get('row_id')} score status not blocked")
        if row.get("runtime_cross_ranking_allowed") is not False:
            raise ValueError(f"{row.get('row_id')} runtime ranking not disabled")
        if row.get("admission_status") != LEDH_FORWARD_ADMISSION_STATUS_ADMITTED:
            raise ValueError(f"{row.get('row_id')} not admitted")
        if not row.get("row_semantics", {}).get("target_policy_label_preserved"):
            raise ValueError(f"{row.get('row_id')} target policy label not preserved")
        score_keys = {
            "score",
            "score_l2_norm",
            "score_status",
            "score_derivative_provenance",
            "score_coordinate_system",
            "score_evidence_artifact",
        }
        leaked = score_keys.intersection(row)
        if leaked:
            raise ValueError(f"{row.get('row_id')} leaked score keys: {sorted(leaked)}")
    ksc = next(row for row in rows if row["row_id"] == KSC_SV_ROW_ID)
    semantics = ksc["row_semantics"]
    if semantics.get("target_family") != "ksc_finite_gaussian_mixture_surrogate":
        raise ValueError("KSC target family label was not preserved")
    if semantics.get("ksc_mixture_is_target_likelihood") is not True:
        raise ValueError("KSC mixture target-likelihood flag must be true")
    if semantics.get("actual_sv_exact_log_chi_square_target_used") is not False:
        raise ValueError("KSC row must not claim exact actual-SV target use")
    if semantics.get("transform_offset") != 1.0e-8:
        raise ValueError("KSC transform offset must be preserved")


def build_artifact() -> dict[str, Any]:
    rows = [
        _integration_row(expected_row_id=row_id, source_path=path)
        for row_id, path in MAIN_ROW_ARTIFACTS
    ]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "metadata_date": DATE,
        "phase": 8,
        "program": "bayesfilter-ledh-forward-scalar-per-model",
        "algorithm_id": LEDH_ALGORITHM_ID,
        "target_scalar": LEDH_TARGET_SCALAR_OBSERVED_DATA_LOG_LIKELIHOOD,
        "target_output_tensor_field": LEDH_OUTPUT_TENSOR_FIELD_LOG_LIKELIHOOD,
        "integration_scope": "ledh_value_only_admitted_forward_scalar_rows",
        "score_integration_status": SCORE_INTEGRATION_STATUS,
        "runtime_cross_ranking_allowed": False,
        "all_algorithm_comparison_allowed": False,
        "main_row_count": len(rows),
        "main_row_ids": [row["row_id"] for row in rows],
        "diagnostic_rows": [
            {
                "row_id": PARAMETERIZED_SIR_DIAGNOSTIC_ROW_ID,
                "status": "excluded_from_main_value_leaderboard",
                "reason": (
                    "legacy scoped parameterized SIR diagnostic row; no separate "
                    "Phase 8 main-row admission artifact"
                ),
            }
        ],
        "rows": rows,
        "nonclaims": [
            "value-only integration from admitted forward-scalar artifacts",
            "not score admission",
            "not score correctness",
            "not all-algorithm comparison evidence",
            "not runtime ranking evidence",
            "not HMC readiness evidence",
            "not posterior correctness evidence",
            "not scientific superiority evidence",
            "KSC row is finite-mixture surrogate target evidence, not exact native actual-SV likelihood",
        ],
    }
    _validate_payload(payload)
    return _json_safe(payload)


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.6g}"
    return str(value)


def _write_markdown(path: Path, payload: dict[str, Any], json_path: Path) -> None:
    lines = [
        "# LEDH Forward-Scalar Value Integration",
        "",
        f"- JSON artifact: `{_rel(json_path)}`",
        f"- Schema: `{payload['schema_version']}`",
        f"- Target scalar: `{payload['target_scalar']}`",
        f"- Output tensor field: `{payload['target_output_tensor_field']}`",
        f"- Score integration: `{payload['score_integration_status']}`",
        f"- Runtime cross-ranking allowed: `{payload['runtime_cross_ranking_allowed']}`",
        f"- Main row count: `{payload['main_row_count']}`",
        "",
        "| Row | Mean Log Likelihood | MCSE | Target Policy | Source Artifact |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for row in payload["rows"]:
        summary = row["log_likelihood_summary"]
        lines.append(
            "| "
            + " | ".join(
                [
                    row["row_id"],
                    _fmt(summary["mean"]),
                    _fmt(summary["mcse"]),
                    row["target_observation_policy"],
                    f"`{row['source_artifact']}`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Diagnostic Rows",
            "",
        ]
    )
    for row in payload["diagnostic_rows"]:
        lines.append(f"- `{row['row_id']}`: `{row['status']}`. {row['reason']}")
    lines.extend(
        [
            "",
            "## Nonclaims",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in payload["nonclaims"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = build_artifact()
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(args.markdown_output, payload, output_path)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
