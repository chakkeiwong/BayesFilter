"""Continue scalar 1D BayesFilter/filterflow debugging after exact arithmetic.

The only promotion question is cross-implementation agreement against the local
executable filterflow reference. This runner does not assert correctness of
either implementation.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/bayesfilter-dpf-mpl")
PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]

import argparse
import time
from typing import Any

import tensorflow as tf

from experiments.dpf_implementation.tf_tfp.runners import (
    run_filterflow_1d_lgssm_step_gradient_comparison_tf as step,
)
from experiments.dpf_implementation.tf_tfp.runners import (
    run_filterflow_1d_to_smoothness_agreement_ladder_tf as agreement,
)
from experiments.dpf_implementation.tf_tfp.runners import (
    run_filterflow_1d_to_smoothness_ladder_tf as continuation,
)
from experiments.dpf_implementation.tf_tfp.runners import (
    run_filterflow_r1_filterflow_exact_arithmetic_tf as exact,
)
from experiments.dpf_implementation.tf_tfp.runners import (
    run_filterflow_r1_observation_path_mismatch_localization_tf as localization,
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


PLAN_PATH = "docs/plans/bayesfilter-dpf-exact-arithmetic-continuation-debug-plan-2026-06-02.md"
RESULT_PATH = "docs/plans/bayesfilter-dpf-exact-arithmetic-continuation-debug-result-2026-06-02.md"
JSON_PATH = OUTPUT_DIR / "dpf_exact_arithmetic_continuation_debug_2026-06-02.json"
REPORT_PATH = REPORT_DIR / "dpf-exact-arithmetic-continuation-debug-2026-06-02.md"
RUNG_IDS = (
    "R1_filterflow_observation_path_exact_arithmetic",
    "R2_filterflow_initial_particles_exact_arithmetic",
    "R3_filterflow_transition_stream",
    "R4_resampling_policy_trigger_semantics",
    "R5_scalar_contract",
    "R6_theta_grid_surface",
    "R7_2d_constant_velocity_state",
    "R8_gradient_smoothness_surface",
)


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
    initial_fingerprint = continuation._filterflow_fingerprint()
    fixture = agreement._filterflow_fixture_subprocess()
    rung_ledger: list[dict[str, Any]] = []
    first_failing: dict[str, Any] | None = None
    first_blocked: dict[str, Any] | None = None

    if fixture.get("status") != "executed":
        first_blocked = _cell_id("R1_filterflow_observation_path_exact_arithmetic", "fixture_blocked")
        rung_ledger = _blocked_tail(
            start_index=0,
            blocker="blocked_by_filterflow_fixture_subprocess",
            first_failing=None,
            first_blocked=first_blocked,
        )
    else:
        control = localization._control_base_scenario()
        filterflow_observation = localization._r1_base_scenario(fixture, control)
        r1 = _run_evidence_rung(
            rung="R1_filterflow_observation_path_exact_arithmetic",
            cell_id="filterflow_observation_path_fixed_initial_particles",
            scenario=filterflow_observation,
            changed_axis="observations switched to executable filterflow smoothness fixture",
            fixed_variables={
                "initial_particles": list(control.initial_particles),
                "transition_noises_source": "controlled generated_T100 replay ledger",
            },
        )
        rung_ledger.append(r1)
        if r1["status"] == "mismatch":
            first_failing = r1["first_failing_cell"]
            first_blocked = _cell_id("R2_filterflow_initial_particles_exact_arithmetic", "blocked_after_R1")
            rung_ledger.extend(_blocked_tail(1, r1["blocker_reason"], first_failing, first_blocked))
        elif r1["status"] == "blocked":
            first_blocked = r1["first_blocked_cell"]
            rung_ledger.extend(_blocked_tail(1, r1["blocker_reason"], None, first_blocked))
        else:
            r2_scenario = localization.LocalizationScenario(
                scenario_id="filterflow_observation_and_initial_particles_exact_arithmetic",
                observations=filterflow_observation.observations,
                initial_particles=tuple(float(v) for v in fixture["initial_particles_scalar"]),
                transition_noises=filterflow_observation.transition_noises,
            )
            r2 = _run_evidence_rung(
                rung="R2_filterflow_initial_particles_exact_arithmetic",
                cell_id="filterflow_initial_particles_data_seed_123_after_data_generation",
                scenario=r2_scenario,
                changed_axis="initial particles switched to executable filterflow fixture draw",
                fixed_variables={
                    "observations_source": "executable filterflow smoothness fixture",
                    "transition_noises_source": "controlled generated_T100 replay ledger",
                    "initial_particles": list(r2_scenario.initial_particles),
                },
            )
            rung_ledger.append(r2)
            if r2["status"] == "mismatch":
                first_failing = r2["first_failing_cell"]
                first_blocked = _cell_id("R3_filterflow_transition_stream", "blocked_after_R2")
                rung_ledger.extend(_blocked_tail(2, r2["blocker_reason"], first_failing, first_blocked))
            elif r2["status"] == "blocked":
                first_blocked = r2["first_blocked_cell"]
                rung_ledger.extend(_blocked_tail(2, r2["blocker_reason"], None, first_blocked))
            else:
                r3 = _transition_stream_blocker(initial_fingerprint)
                rung_ledger.append(r3)
                first_blocked = r3["first_blocked_cell"]
                rung_ledger.extend(_blocked_tail(3, r3["blocker_reason"], None, first_blocked))

    final_fingerprint = continuation._filterflow_fingerprint()
    comparator_drift = continuation._fingerprints_drifted(initial_fingerprint, final_fingerprint)
    if comparator_drift:
        first_failing = None
        first_blocked = _cell_id("R1_filterflow_observation_path_exact_arithmetic", "comparator_drift")
        rung_ledger = _blocked_tail(
            start_index=0,
            blocker="blocked_by_comparator_drift",
            first_failing=None,
            first_blocked=first_blocked,
            comparator_before=initial_fingerprint,
            comparator_after=final_fingerprint,
        )

    first_failing_cell = first_failing or _none_cell()
    first_blocked_cell = first_blocked or _none_cell()
    decision = _decision(first_failing_cell, first_blocked_cell, rung_ledger)
    return {
        "decision": decision,
        "created_at_utc": utc_now(),
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "report_path": str(REPORT_PATH.relative_to(REPO_ROOT)),
        "json_path": str(JSON_PATH.relative_to(REPO_ROOT)),
        "question": (
            "After exact filterflow arithmetic is enforced, where is the next "
            "scalar 1D BayesFilter/filterflow discrepancy or blocker?"
        ),
        "evidence_contract": {
            "primary_comparator": "local executable patched filterflow checkout",
            "primary_question": "cross_implementation_difference_only",
            "primary_pass": "exact_arithmetic_ledger_agreement",
            "mathematical_correctness": "not_concluded",
        },
        "arithmetic_policy": {
            "bayesfilter_evidence_path": "filterflow_exact_float32",
            "fixed_target_sinkhorn": "not used",
            "bf64_contrast": "not rerun in this continuation artifact",
        },
        "first_failing_cell": first_failing_cell,
        "first_blocked_cell": first_blocked_cell,
        "rung_ledger": rung_ledger,
        "fixture_summary": agreement._fixture_summary(fixture),
        "filterflow_fingerprint_initial": initial_fingerprint,
        "filterflow_fingerprint_final": final_fingerprint,
        "comparator_drift": comparator_drift,
        "path_boundary_manifest": continuation._path_boundary_manifest(),
        "tolerances": agreement._agreement_tolerances(),
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners."
                "run_filterflow_exact_arithmetic_continuation_debug_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_implications": step._non_implications()
        + [
            "No correctness claim is made for either implementation.",
            "No smoothness-surface gradient correctness is concluded.",
            "No production dtype default is concluded.",
        ],
    }


def _run_evidence_rung(
    *,
    rung: str,
    cell_id: str,
    scenario: localization.LocalizationScenario,
    changed_axis: str,
    fixed_variables: dict[str, Any],
) -> dict[str, Any]:
    filterflow = localization._filterflow_reference(scenario)
    if filterflow.get("status") != "executed":
        cell = {
            "cell_id": cell_id,
            "status": "blocked",
            "filterflow_status": filterflow.get("status"),
            "filterflow_blocker": filterflow.get("blocker"),
        }
        return _rung_row(
            rung=rung,
            status="blocked",
            blocker_reason=filterflow.get("blocker", "unknown filterflow blocker"),
            first_failing_cell=None,
            first_blocked_cell=_cell_id(rung, cell_id),
            changed_axis=changed_axis,
            fixed_variables=fixed_variables,
            cells=[cell],
            primary_metrics={},
            explanatory_diagnostics={},
            filterflow_cpu_only_manifest=filterflow.get("cpu_only_manifest"),
        )
    bayesfilter = exact._bayesfilter_arithmetic_reference(
        scenario,
        dtype=tf.float32,
        arithmetic_label="filterflow_exact_float32",
    )
    comparison = exact._comparison_payload(bayesfilter, filterflow)
    status = "pass" if comparison.get("implementation_agreement") else "mismatch"
    first_failing = None
    blocker = None
    if status == "mismatch":
        first_failing = _cell_id(rung, cell_id)
        blocker = f"blocked_after_{rung}_mismatch"
    return _rung_row(
        rung=rung,
        status=status,
        blocker_reason=blocker,
        first_failing_cell=first_failing,
        first_blocked_cell=None,
        changed_axis=changed_axis,
        fixed_variables={
            **fixed_variables,
            "theta": step.THETA0,
            "Q": step.Q,
            "R": step.R,
            "num_particles": step.NUM_PARTICLES,
            "epsilon": step.EPSILON,
            "scaling": step.SCALING,
            "convergence_threshold": agreement.CONVERGENCE_THRESHOLD,
            "max_iterations": agreement.MAX_ITERATIONS,
        },
        cells=[
            {
                "cell_id": cell_id,
                "status": status,
                "scenario_id": scenario.scenario_id,
                "horizon": scenario.horizon,
                "comparison": comparison,
                "bayesfilter": exact._summary_reference(bayesfilter),
                "filterflow": exact._summary_reference(filterflow),
            }
        ],
        primary_metrics=_primary_metrics(comparison, bayesfilter, filterflow),
        explanatory_diagnostics={
            "gradient_promotion": "not_concluded",
            "bayesfilter_gradient_tape": bayesfilter.get("gradient_tape"),
            "filterflow_gradient_tape": filterflow.get("gradient_tape"),
            "bayesfilter_finite_difference_gradient": bayesfilter.get("finite_difference_gradient"),
            "filterflow_finite_difference_gradient": filterflow.get("finite_difference_gradient"),
        },
        filterflow_cpu_only_manifest=filterflow.get("cpu_only_manifest"),
    )


def _transition_stream_blocker(initial_fingerprint: dict[str, Any]) -> dict[str, Any]:
    blocker = (
        "blocked_by_uninstrumented_filterflow_transition_proposal_stream: "
        "the executable filterflow smoothness path samples proposal noises "
        "inside its SMC loop and the public script fixture does not serialize "
        "per-time proposal draws. Continuing the one-axis ladder requires a "
        "reviewed non-mutating trace wrapper or explicit authorization to "
        "instrument the local filterflow reference."
    )
    return _rung_row(
        rung="R3_filterflow_transition_stream",
        status="blocked",
        blocker_reason=blocker,
        first_failing_cell=None,
        first_blocked_cell=_cell_id("R3_filterflow_transition_stream", "proposal_stream_not_replayed"),
        changed_axis="transition proposal stream switched toward executable filterflow smoothness path",
        fixed_variables={"comparator_before": initial_fingerprint},
        cells=[],
        primary_metrics={},
        explanatory_diagnostics={
            "not_a_mismatch": True,
            "required_next_control": "trace or replay exact filterflow proposal random stream",
        },
        filterflow_cpu_only_manifest=None,
    )


def _blocked_tail(
    start_index: int,
    blocker: str | None,
    first_failing: dict[str, Any] | None,
    first_blocked: dict[str, Any] | None,
    comparator_before: dict[str, Any] | None = None,
    comparator_after: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    inherited = blocker or "blocked_after_prior_rung"
    rows = []
    for rung in RUNG_IDS[start_index:]:
        rows.append(
            _rung_row(
                rung=rung,
                status="blocked",
                blocker_reason=inherited,
                first_failing_cell=first_failing,
                first_blocked_cell=first_blocked,
                changed_axis="blocked_before_execution",
                fixed_variables={},
                cells=[],
                primary_metrics={},
                explanatory_diagnostics={
                    "blocked_rung_interpretation": (
                        "This rung is not evidence-bearing; the first failing "
                        "or blocked cell is reported separately."
                    )
                },
                filterflow_cpu_only_manifest=None,
                comparator_before=comparator_before,
                comparator_after=comparator_after,
            )
        )
    return rows


def _rung_row(
    *,
    rung: str,
    status: str,
    blocker_reason: str | None,
    first_failing_cell: dict[str, Any] | None,
    first_blocked_cell: dict[str, Any] | None,
    changed_axis: str,
    fixed_variables: dict[str, Any],
    cells: list[dict[str, Any]],
    primary_metrics: dict[str, Any],
    explanatory_diagnostics: dict[str, Any],
    filterflow_cpu_only_manifest: dict[str, Any] | None,
    comparator_before: dict[str, Any] | None = None,
    comparator_after: dict[str, Any] | None = None,
) -> dict[str, Any]:
    before = comparator_before or continuation._filterflow_fingerprint()
    return {
        "rung": rung,
        "status": status,
        "evidence_bearing": bool(cells and status in {"pass", "mismatch"}),
        "failure_observed_directly": status == "mismatch",
        "blocker_reason": blocker_reason,
        "first_failing_cell": first_failing_cell,
        "first_blocked_cell": first_blocked_cell,
        "changed_axis": changed_axis,
        "fixed_variables": fixed_variables,
        "cells": cells,
        "primary_metrics": primary_metrics,
        "explanatory_diagnostics": explanatory_diagnostics,
        "parent_cpu_only_manifest": _parent_cpu_manifest(),
        "filterflow_cpu_only_manifest": filterflow_cpu_only_manifest,
        "comparator_fingerprint_before": before,
        "comparator_fingerprint_after": comparator_after or before,
    }


def _primary_metrics(
    comparison: dict[str, Any],
    bayesfilter: dict[str, Any],
    filterflow: dict[str, Any],
) -> dict[str, Any]:
    residuals = comparison.get("absolute_residuals") or {}
    residual_deltas = comparison.get("residual_deltas") or {}
    return {
        "implementation_agreement": bool(comparison.get("implementation_agreement")),
        "scalar_delta": comparison.get("scalar_delta"),
        "max_field_delta": comparison.get("max_field_delta"),
        "first_failure": comparison.get("first_failure"),
        "trigger_match": comparison.get("trigger_match"),
        "ledger_within_tolerance": comparison.get("ledger_within_tolerance"),
        "scalar_within_tolerance": comparison.get("scalar_within_tolerance"),
        "row_residual_delta": residual_deltas.get("row"),
        "column_residual_delta": residual_deltas.get("column"),
        "bayesfilter_max_row_residual": residuals.get("bayesfilter_max_row_residual"),
        "filterflow_max_row_residual": residuals.get("filterflow_max_row_residual"),
        "bayesfilter_max_column_residual": residuals.get("bayesfilter_max_column_residual"),
        "filterflow_max_column_residual": residuals.get("filterflow_max_column_residual"),
        "bayesfilter_scalar": bayesfilter.get("scalar"),
        "filterflow_scalar": filterflow.get("scalar"),
    }


def _decision(
    first_failing: dict[str, Any],
    first_blocked: dict[str, Any],
    rows: list[dict[str, Any]],
) -> str:
    if any(row["blocker_reason"] == "blocked_by_comparator_drift" for row in rows):
        return "exact_arithmetic_continuation_blocked_by_comparator_drift"
    if first_failing["rung_id"] != "none_observed":
        return "exact_arithmetic_continuation_first_mismatch_detected"
    if first_blocked["rung_id"] != "none_observed":
        return "exact_arithmetic_continuation_blocked_after_partial_agreement"
    return "exact_arithmetic_continuation_no_mismatch_observed"


def _cell_id(rung: str, cell_id: str) -> dict[str, Any]:
    return {"rung_id": rung, "cell_index": 0, "cell_id": cell_id}


def _none_cell() -> dict[str, Any]:
    return {"rung_id": "none_observed", "cell_index": -1, "cell_id": "none_observed"}


def _parent_cpu_manifest() -> dict[str, Any]:
    return {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "pre_import_cuda_visible_devices": PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        "gpu_devices_visible": [str(device) for device in tf.config.list_physical_devices("GPU")],
    }


def _validate_payload(payload: dict[str, Any]) -> None:
    required = {
        "decision",
        "plan_path",
        "result_path",
        "first_failing_cell",
        "first_blocked_cell",
        "rung_ledger",
        "filterflow_fingerprint_initial",
        "filterflow_fingerprint_final",
        "path_boundary_manifest",
        "run_manifest",
        "non_implications",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"missing payload keys: {sorted(missing)}")
    if len(payload["rung_ledger"]) != len(RUNG_IDS):
        raise ValueError("rung ledger must contain all rungs")
    if payload["decision"] not in {
        "exact_arithmetic_continuation_blocked_by_comparator_drift",
        "exact_arithmetic_continuation_first_mismatch_detected",
        "exact_arithmetic_continuation_blocked_after_partial_agreement",
        "exact_arithmetic_continuation_no_mismatch_observed",
    }:
        raise ValueError(f"unexpected decision: {payload['decision']}")
    _validate_cpu(payload["run_manifest"], "parent run")
    for row in payload["rung_ledger"]:
        if row["rung"] not in RUNG_IDS:
            raise ValueError(f"unexpected rung: {row['rung']}")
        if row["status"] not in {"pass", "mismatch", "blocked"}:
            raise ValueError(f"bad rung status: {row['status']}")
        _validate_cpu(row["parent_cpu_only_manifest"], f"{row['rung']} parent")
        if row["filterflow_cpu_only_manifest"] is not None:
            _validate_cpu(row["filterflow_cpu_only_manifest"], f"{row['rung']} filterflow")
        if row["status"] == "pass" and not row["primary_metrics"].get("implementation_agreement"):
            raise ValueError(f"pass without agreement: {row['rung']}")
        if row["status"] == "mismatch" and not row["failure_observed_directly"]:
            raise ValueError(f"mismatch without direct failure: {row['rung']}")
    if any(bool(value) for value in payload["path_boundary_manifest"].values()):
        raise ValueError(f"path boundary violation: {payload['path_boundary_manifest']}")
    if payload["fixture_summary"].get("status") == "executed":
        _validate_cpu(payload["fixture_summary"]["cpu_only_manifest"], "fixture")


def _validate_cpu(manifest: dict[str, Any], label: str) -> None:
    if manifest.get("pre_import_cuda_visible_devices") != "-1":
        raise ValueError(f"{label}: pre-import CUDA_VISIBLE_DEVICES is not -1")
    if manifest.get("cuda_visible_devices") != "-1":
        raise ValueError(f"{label}: CUDA_VISIBLE_DEVICES is not -1")
    if manifest.get("gpu_devices_visible") != []:
        raise ValueError(f"{label}: GPU devices visible {manifest.get('gpu_devices_visible')}")


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Exact-Arithmetic Continuation Debug",
        "",
        "## Decision",
        "",
        f"`{payload['decision']}`",
        "",
        "## First Cells",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| first failing cell | `{payload['first_failing_cell']}` |",
        f"| first blocked cell | `{payload['first_blocked_cell']}` |",
        "",
        "## Rung Ledger",
        "",
        "| Rung | Status | Evidence-bearing | Direct failure | Blocker | Changed axis |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["rung_ledger"]:
        lines.append(
            f"| `{row['rung']}` | `{row['status']}` | "
            f"`{row['evidence_bearing']}` | `{row['failure_observed_directly']}` | "
            f"`{row['blocker_reason']}` | `{row['changed_axis']}` |"
        )
    lines.extend(["", "## Evidence Cells", ""])
    for row in payload["rung_ledger"]:
        if not row["cells"]:
            continue
        metrics = row["primary_metrics"]
        lines.extend(
            [
                f"### {row['rung']}",
                "",
                "| Metric | Value |",
                "| --- | ---: |",
                f"| implementation agreement | `{metrics['implementation_agreement']}` |",
                f"| scalar delta | `{metrics['scalar_delta']}` |",
                f"| max field delta | `{metrics['max_field_delta']}` |",
                f"| row residual delta | `{metrics['row_residual_delta']}` |",
                f"| column residual delta | `{metrics['column_residual_delta']}` |",
                f"| first failure | `{metrics['first_failure']}` |",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            _interpretation(payload),
            "",
            "## Non-Implications",
            "",
            *[f"- {item}" for item in payload["non_implications"]],
            "",
        ]
    )
    return "\n".join(lines)


def _interpretation(payload: dict[str, Any]) -> str:
    if payload["first_failing_cell"]["rung_id"] != "none_observed":
        return "A direct cross-implementation mismatch was found at the first failing cell."
    if payload["first_blocked_cell"]["rung_id"] != "none_observed":
        return (
            "No direct mismatch was found before the first blocker. The next "
            "debugging step is to obtain an exact replay or trace of the "
            "filterflow transition proposal random stream."
        )
    return "No direct mismatch or blocker was observed in the bounded ladder."


def _digest_payload(payload: dict[str, Any]) -> str:
    clone = dict(payload)
    clone.pop("run_manifest", None)
    clone.pop("created_at_utc", None)
    clone.pop("reproducibility_digest", None)
    return stable_digest(clone)


if __name__ == "__main__":
    raise SystemExit(main())
