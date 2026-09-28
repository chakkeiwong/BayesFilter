"""Localize the R1 observation-path BayesFilter/filterflow mismatch.

The question is agreement against the local executable filterflow checkout.
No correctness claim is made for either implementation.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/bayesfilter-dpf-mpl")
PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]

import argparse
import json
import math
import time
from dataclasses import dataclass
from typing import Any

import tensorflow as tf

from experiments.dpf_implementation.tf_tfp.resampling import annealed_transport_tf
from experiments.dpf_implementation.tf_tfp.runners import (
    run_filterflow_1d_lgssm_step_gradient_comparison_tf as step,
)
from experiments.dpf_implementation.tf_tfp.runners import (
    run_filterflow_1d_to_smoothness_agreement_ladder_tf as agreement,
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


PLAN_PATH = "docs/plans/bayesfilter-dpf-r1-observation-path-mismatch-hypothesis-plan-2026-06-02.md"
RESULT_PATH = "docs/plans/bayesfilter-dpf-r1-observation-path-mismatch-hypothesis-result-2026-06-02.md"
REVIEW_LOOP_PATH = "docs/plans/bayesfilter-dpf-r1-observation-path-mismatch-hypothesis-review-loop-2026-06-02.md"
JSON_PATH = OUTPUT_DIR / "dpf_filterflow_r1_observation_path_mismatch_localization_2026-06-02.json"
REPORT_PATH = REPORT_DIR / "dpf-filterflow-r1-observation-path-mismatch-localization-2026-06-02.md"

PREFIXES = (1, 2, 4, 8, 16, 32, 64, 100)
SCALES = (1.0, 0.1, 0.01)
DTYPES = (tf.float64, tf.float32)
PRIMARY_SCALE_DTYPE = "float64"
OBSERVATION_FIELD_TOLERANCE = 5e-5

FIELD_SPECS = (
    ("pre_particles", "pre_particles", "predicted_particles"),
    ("pre_log_weights", "pre_log_weights", "normalized_log_weights"),
    ("ess", "ess", "normalized_log_weights"),
    ("transport_cost_matrix", "transport_cost_matrix", "transport_cost_matrix"),
    ("transport_matrix", "transport_matrix", "transport_matrix"),
    ("post_transport_particles", "post_transport_particles", "post_transport_particles"),
    ("post_transport_log_weights", "post_transport_log_weights", "normalized_log_weights"),
    ("predicted_particles", "predicted_particles", "predicted_particles"),
    ("observation_log_likelihoods", "observation_log_likelihoods", "observation_log_likelihoods"),
    ("unnormalized_log_weights", "unnormalized_log_weights", "observation_log_likelihoods"),
    ("per_step_log_normalizer", "per_step_log_normalizer", "per_step_log_normalizer"),
    ("post_update_log_weights", "post_update_log_weights", "normalized_log_weights"),
    ("row_residual", "row_residual", "row_residual_delta"),
    ("column_residual", "column_residual", "column_residual_delta"),
)

ALLOWED_HYPOTHESIS_STATUSES = {
    "supported",
    "partially_supported",
    "weakened",
    "inconclusive",
    "not_tested_blocked",
    "localized_unexplained",
    "harness_control_failed",
    "audit_risk_identified",
    "audit_risk_not_found",
}


@dataclass(frozen=True)
class LocalizationScenario:
    scenario_id: str
    observations: tuple[float, ...]
    initial_particles: tuple[float, ...]
    transition_noises: tuple[tuple[float, ...], ...]

    @property
    def horizon(self) -> int:
        return len(self.observations)


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
    if fixture.get("status") != "executed":
        return _blocked_payload(
            decision="r1_hypothesis_filterflow_fixture_blocked",
            blocker=fixture.get("blocker", "unknown fixture blocker"),
            initial_fingerprint=initial_fingerprint,
            fixture=fixture,
        )

    control_base = _control_base_scenario()
    r1_base = _r1_base_scenario(fixture, control_base)
    control_case = _run_case(control_base, case_id="matched_control_generated_T100")
    r1_case = _run_case(r1_base, case_id="r1_filterflow_observation_path_unscaled")
    prefix_cases = [
        _run_case(_prefix_scenario(r1_base, prefix), case_id=f"r1_prefix_T{prefix}")
        for prefix in PREFIXES
    ]
    scale_cases = [
        _run_case(_scaled_scenario(r1_base, scale), case_id=f"r1_scale_{scale:g}")
        for scale in SCALES
    ]
    final_fingerprint = continuation._filterflow_fingerprint()
    comparator_drift = continuation._fingerprints_drifted(initial_fingerprint, final_fingerprint)
    hypothesis_statuses = _hypothesis_statuses(
        control_case=control_case,
        r1_case=r1_case,
        prefix_cases=prefix_cases,
        scale_cases=scale_cases,
        control_base=control_base,
        r1_base=r1_base,
    )
    decision = _decision(
        comparator_drift=comparator_drift,
        control_case=control_case,
        r1_case=r1_case,
    )
    return {
        "decision": decision,
        "created_at_utc": utc_now(),
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "review_loop_path": REVIEW_LOOP_PATH,
        "report_path": str(REPORT_PATH.relative_to(REPO_ROOT)),
        "json_path": str(JSON_PATH.relative_to(REPO_ROOT)),
        "question": (
            "Which mechanism explains the accepted R1 observation-path "
            "BayesFilter/filterflow mismatch?"
        ),
        "evidence_contract": {
            "primary_comparator": "local executable patched filterflow checkout",
            "primary_question": "cross_implementation_difference_only",
            "same_harness_control_required": True,
            "correctness_of_either_implementation": "not_concluded",
        },
        "plan_review_status": "ACCEPT_after_round_4",
        "hypothesis_statuses": hypothesis_statuses,
        "control_case": control_case,
        "r1_case": r1_case,
        "prefix_cases": prefix_cases,
        "scale_cases": scale_cases,
        "fixture_summary": agreement._fixture_summary(fixture),
        "observation_summaries": {
            "matched_control_generated_T100": _observation_summary(control_base.observations),
            "r1_filterflow_observation_path": _observation_summary(r1_base.observations),
        },
        "wrapper_audit": _wrapper_audit(),
        "filterflow_fingerprint_initial": initial_fingerprint,
        "filterflow_fingerprint_final": final_fingerprint,
        "comparator_drift": comparator_drift,
        "path_boundary_manifest": continuation._path_boundary_manifest(),
        "tolerances": _tolerances(),
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners."
                "run_filterflow_r1_observation_path_mismatch_localization_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_implications": step._non_implications()
        + [
            "No correctness claim is made for either implementation.",
            "No dtype default-policy change is concluded.",
            "No tolerance policy change is concluded.",
        ],
    }


def _blocked_payload(
    *,
    decision: str,
    blocker: str,
    initial_fingerprint: dict[str, Any],
    fixture: dict[str, Any],
) -> dict[str, Any]:
    final_fingerprint = continuation._filterflow_fingerprint()
    return {
        "decision": decision,
        "created_at_utc": utc_now(),
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "review_loop_path": REVIEW_LOOP_PATH,
        "question": "Which mechanism explains the accepted R1 observation-path mismatch?",
        "blocker": blocker,
        "plan_review_status": "ACCEPT_after_round_4",
        "hypothesis_statuses": {
            key: {"status": "not_tested_blocked", "evidence": blocker}
            for key in [f"H{index}" for index in range(1, 9)]
        },
        "control_case": None,
        "r1_case": None,
        "prefix_cases": [],
        "scale_cases": [],
        "fixture_summary": fixture,
        "filterflow_fingerprint_initial": initial_fingerprint,
        "filterflow_fingerprint_final": final_fingerprint,
        "comparator_drift": continuation._fingerprints_drifted(initial_fingerprint, final_fingerprint),
        "path_boundary_manifest": continuation._path_boundary_manifest(),
        "tolerances": _tolerances(),
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners."
                "run_filterflow_r1_observation_path_mismatch_localization_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_implications": step._non_implications(),
    }


def _control_base_scenario() -> LocalizationScenario:
    generated = continuation._generated_scenario(100)
    return LocalizationScenario(
        scenario_id="generated_T100_same_harness_control",
        observations=tuple(float(v) for v in generated.observations),
        initial_particles=tuple(float(v) for v in step.INITIAL_PARTICLES),
        transition_noises=tuple(
            tuple(float(v) for v in row) for row in generated.transition_noises
        ),
    )


def _r1_base_scenario(
    fixture: dict[str, Any],
    control: LocalizationScenario,
) -> LocalizationScenario:
    return LocalizationScenario(
        scenario_id="r1_filterflow_observation_path",
        observations=tuple(float(v) for v in fixture["observations_scalar"]),
        initial_particles=control.initial_particles,
        transition_noises=control.transition_noises,
    )


def _prefix_scenario(
    scenario: LocalizationScenario,
    horizon: int,
) -> LocalizationScenario:
    return LocalizationScenario(
        scenario_id=f"{scenario.scenario_id}_prefix_T{horizon}",
        observations=scenario.observations[:horizon],
        initial_particles=scenario.initial_particles,
        transition_noises=scenario.transition_noises[:horizon],
    )


def _scaled_scenario(
    scenario: LocalizationScenario,
    scale: float,
) -> LocalizationScenario:
    return LocalizationScenario(
        scenario_id=f"{scenario.scenario_id}_scale_{scale:g}",
        observations=tuple(float(scale) * value for value in scenario.observations),
        initial_particles=scenario.initial_particles,
        transition_noises=scenario.transition_noises,
    )


def _run_case(scenario: LocalizationScenario, *, case_id: str) -> dict[str, Any]:
    filterflow = _filterflow_reference(scenario)
    dtype_rows = {}
    if filterflow.get("status") == "executed":
        for dtype in DTYPES:
            label = _dtype_label(dtype)
            bayesfilter = _bayesfilter_reference(scenario, dtype=dtype)
            comparison = agreement._compare_runs(bayesfilter, filterflow)
            dtype_rows[label] = _dtype_result(
                bayesfilter=bayesfilter,
                filterflow=filterflow,
                comparison=comparison,
                dtype_label=label,
            )
    return {
        "case_id": case_id,
        "scenario_id": scenario.scenario_id,
        "horizon": scenario.horizon,
        "filterflow_status": filterflow.get("status"),
        "filterflow_blocker": filterflow.get("blocker"),
        "filterflow_cpu_only_manifest": filterflow.get("cpu_only_manifest"),
        "dtype_rows": dtype_rows,
    }


def _bayesfilter_reference(
    scenario: LocalizationScenario,
    *,
    dtype: tf.DType,
) -> dict[str, Any]:
    step_originals = _snapshot_step_globals()
    annealed_dtype = annealed_transport_tf.DTYPE
    try:
        step.DTYPE = dtype
        annealed_transport_tf.DTYPE = dtype
        step.HORIZON = scenario.horizon
        step.INITIAL_PARTICLES = list(scenario.initial_particles)
        step.OBSERVATIONS = list(scenario.observations)
        step.TRANSITION_NOISES = [list(row) for row in scenario.transition_noises]
        step.CONVERGENCE_THRESHOLD = agreement.CONVERGENCE_THRESHOLD
        step.MAX_ITERATIONS = agreement.MAX_ITERATIONS
        payload = step._bayesfilter_reference()
        payload["diagnostic_dtype"] = _dtype_label(dtype)
        return payload
    finally:
        _restore_step_globals(step_originals)
        annealed_transport_tf.DTYPE = annealed_dtype


def _filterflow_reference(scenario: LocalizationScenario) -> dict[str, Any]:
    agreement_scenario = agreement.AgreementScenario(
        scenario_id=scenario.scenario_id,
        observations=scenario.observations,
        initial_particles=scenario.initial_particles,
        transition_noises=scenario.transition_noises,
    )
    return agreement._filterflow_reference_subprocess(agreement_scenario)


def _snapshot_step_globals() -> dict[str, Any]:
    return {
        "DTYPE": step.DTYPE,
        "HORIZON": step.HORIZON,
        "INITIAL_PARTICLES": step.INITIAL_PARTICLES,
        "OBSERVATIONS": step.OBSERVATIONS,
        "TRANSITION_NOISES": step.TRANSITION_NOISES,
        "CONVERGENCE_THRESHOLD": step.CONVERGENCE_THRESHOLD,
        "MAX_ITERATIONS": step.MAX_ITERATIONS,
    }


def _restore_step_globals(values: dict[str, Any]) -> None:
    step.DTYPE = values["DTYPE"]
    step.HORIZON = values["HORIZON"]
    step.INITIAL_PARTICLES = values["INITIAL_PARTICLES"]
    step.OBSERVATIONS = values["OBSERVATIONS"]
    step.TRANSITION_NOISES = values["TRANSITION_NOISES"]
    step.CONVERGENCE_THRESHOLD = values["CONVERGENCE_THRESHOLD"]
    step.MAX_ITERATIONS = values["MAX_ITERATIONS"]


def _dtype_result(
    *,
    bayesfilter: dict[str, Any],
    filterflow: dict[str, Any],
    comparison: dict[str, Any],
    dtype_label: str,
) -> dict[str, Any]:
    detailed = _detailed_field_comparison(bayesfilter, filterflow)
    metrics = _primary_metrics(bayesfilter, filterflow, comparison, detailed)
    return {
        "dtype": dtype_label,
        "bayesfilter_status": bayesfilter.get("status"),
        "comparison_status": comparison.get("status"),
        "implementation_agreement": bool(comparison.get("implementation_agreement")),
        "primary_metrics": metrics,
        "first_failure": detailed["first_failure"],
        "per_time_field_deltas": detailed["per_time_field_deltas"],
        "cpu_only_manifest": _parent_cpu_manifest(),
    }


def _detailed_field_comparison(
    bayesfilter: dict[str, Any],
    filterflow: dict[str, Any],
) -> dict[str, Any]:
    per_time = []
    first_failure: dict[str, Any] | None = None
    if bayesfilter.get("status") != "executed" or filterflow.get("status") != "executed":
        return {
            "first_failure": {
                "status": "blocked",
                "time_index": None,
                "field_set": [],
                "reason": "missing executed payload",
            },
            "per_time_field_deltas": [],
            "max_field_delta": float("inf"),
        }

    max_field_delta = 0.0
    for bf_step, ff_step in zip(bayesfilter["ledger"], filterflow["ledger"], strict=True):
        deltas = {}
        failing_fields = []
        for label, key, tolerance_key in FIELD_SPECS:
            delta = _field_delta(bf_step[key], ff_step[key])
            tolerance = _field_tolerance(tolerance_key)
            deltas[label] = {
                "delta": delta,
                "tolerance": tolerance,
                "fails": delta > tolerance,
            }
            max_field_delta = max(max_field_delta, delta)
            if delta > tolerance:
                failing_fields.append(label)
        trigger_match = bf_step["resampling_flags"] == ff_step["resampling_flags"]
        if not trigger_match:
            failing_fields.append("resampling_flags")
        time_row = {
            "time_index": int(bf_step["time_index"]),
            "bayesfilter_resampling_flags": bf_step["resampling_flags"],
            "filterflow_resampling_flags": ff_step["resampling_flags"],
            "trigger_match": trigger_match,
            "field_deltas": deltas,
            "failing_field_set": sorted(set(failing_fields)),
        }
        if first_failure is None and failing_fields:
            first_failure = {
                "status": "failed",
                "time_index": int(bf_step["time_index"]),
                "field_set": sorted(set(failing_fields)),
                "triggered": bool(any(bf_step["resampling_flags"])),
            }
        per_time.append(time_row)
    if first_failure is None:
        first_failure = {
            "status": "no_field_failure",
            "time_index": None,
            "field_set": [],
            "triggered": False,
        }
    return {
        "first_failure": first_failure,
        "per_time_field_deltas": per_time,
        "max_field_delta": max_field_delta,
    }


def _primary_metrics(
    bayesfilter: dict[str, Any],
    filterflow: dict[str, Any],
    comparison: dict[str, Any],
    detailed: dict[str, Any],
) -> dict[str, Any]:
    if comparison.get("status") != "compared":
        return {
            "scalar_delta": float("inf"),
            "scalar_relative_delta": float("inf"),
            "max_field_delta": detailed["max_field_delta"],
            "implementation_agreement": False,
        }
    residual_deltas = comparison.get("residual_deltas", {"row": float("inf"), "column": float("inf")})
    scalar_delta = float(comparison["scalar_delta"])
    filterflow_scalar = float(filterflow["scalar"])
    return {
        "bayesfilter_scalar": bayesfilter.get("scalar"),
        "filterflow_scalar": filterflow.get("scalar"),
        "scalar_delta": scalar_delta,
        "scalar_relative_delta": scalar_delta / max(abs(filterflow_scalar), 1e-12),
        "max_field_delta": detailed["max_field_delta"],
        "trigger_match": comparison["trigger_match"],
        "ledger_within_tolerance": comparison["ledger_within_tolerance"],
        "scalar_within_tolerance": comparison["scalar_within_tolerance"],
        "row_residual_delta": residual_deltas["row"],
        "column_residual_delta": residual_deltas["column"],
        "implementation_agreement": comparison["implementation_agreement"],
        "absolute_residuals": comparison["absolute_residuals"],
        "residual_deltas": residual_deltas,
    }


def _hypothesis_statuses(
    *,
    control_case: dict[str, Any],
    r1_case: dict[str, Any],
    prefix_cases: list[dict[str, Any]],
    scale_cases: list[dict[str, Any]],
    control_base: LocalizationScenario,
    r1_base: LocalizationScenario,
) -> dict[str, dict[str, Any]]:
    if not _case_agrees(control_case, PRIMARY_SCALE_DTYPE):
        return {
            f"H{index}": {
                "status": "harness_control_failed",
                "evidence": "same-harness generated_T100 control did not pass BF64 agreement",
            }
            for index in range(1, 9)
        }
    if _case_agrees(r1_case, PRIMARY_SCALE_DTYPE):
        return {
            f"H{index}": {
                "status": "not_tested_blocked",
                "evidence": "unscaled R1 mismatch was not reproduced",
            }
            for index in range(1, 9)
        }

    h1 = _classify_h1(r1_case)
    h2 = _classify_h2(r1_case, scale_cases)
    h3 = _classify_h3(prefix_cases)
    h4 = _classify_h4(r1_case)
    h5 = _classify_h5(r1_case)
    h6 = _classify_h6(r1_case)
    h7 = _classify_h7(control_base, r1_base, scale_cases)
    h8 = _classify_h8()
    statuses = {
        "H1": h1,
        "H2": h2,
        "H3": h3,
        "H4": h4,
        "H5": h5,
        "H6": h6,
        "H7": h7,
        "H8": h8,
    }
    for key, value in statuses.items():
        if value["status"] not in ALLOWED_HYPOTHESIS_STATUSES:
            raise ValueError(f"{key} emitted invalid status {value['status']}")
    return statuses


def _classify_h1(r1_case: dict[str, Any]) -> dict[str, Any]:
    bf64 = _dtype_row(r1_case, "float64")
    bf32 = _dtype_row(r1_case, "float32")
    if bf64 is None or bf32 is None:
        return {"status": "inconclusive", "evidence": "missing dtype rows"}
    m64 = bf64["primary_metrics"]
    m32 = bf32["primary_metrics"]
    scalar_ratio = _reduction_ratio(m64["scalar_delta"], m32["scalar_delta"])
    ledger_ratio = _reduction_ratio(m64["max_field_delta"], m32["max_field_delta"])
    evidence = {
        "scalar_reduction_ratio": scalar_ratio,
        "max_field_reduction_ratio": ledger_ratio,
        "bf32_implementation_agreement": bf32["implementation_agreement"],
    }
    if scalar_ratio >= 10.0 and ledger_ratio >= 10.0 and bf32["implementation_agreement"]:
        return {"status": "supported", "evidence": evidence}
    if scalar_ratio >= 10.0 or ledger_ratio >= 10.0:
        return {"status": "partially_supported", "evidence": evidence}
    if scalar_ratio < 2.0 and ledger_ratio < 2.0:
        return {"status": "weakened", "evidence": evidence}
    return {"status": "inconclusive", "evidence": evidence}


def _classify_h2(
    r1_case: dict[str, Any],
    scale_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    base = _dtype_row(r1_case, PRIMARY_SCALE_DTYPE)
    rows = [_dtype_row(case, PRIMARY_SCALE_DTYPE) for case in scale_cases]
    if base is None or any(row is None for row in rows):
        return {"status": "inconclusive", "evidence": "missing BF64 scale rows"}
    base_delta = base["primary_metrics"]["scalar_delta"]
    deltas = {
        case["case_id"]: row["primary_metrics"]["scalar_delta"]
        for case, row in zip(scale_cases, rows, strict=True)
    }
    first_sets = {
        case["case_id"]: set(row["first_failure"]["field_set"])
        for case, row in zip(scale_cases, rows, strict=True)
    }
    base_set = set(base["first_failure"]["field_set"])
    scale01_ok = deltas["r1_scale_0.1"] <= base_delta * 0.1 * 0.1 * 5.0
    scale001_ok = deltas["r1_scale_0.01"] <= base_delta * 0.01 * 0.01 * 5.0
    monotone = deltas["r1_scale_1"] >= deltas["r1_scale_0.1"] >= deltas["r1_scale_0.01"]
    set_intersections = {
        key: sorted(value.intersection(base_set)) for key, value in first_sets.items()
    }
    both_intersect = bool(set_intersections["r1_scale_0.1"]) and bool(set_intersections["r1_scale_0.01"])
    evidence = {
        "base_scalar_delta": base_delta,
        "scale_scalar_deltas": deltas,
        "monotone": monotone,
        "field_set_intersections": set_intersections,
    }
    if scale01_ok and scale001_ok and both_intersect:
        return {"status": "supported", "evidence": evidence}
    if monotone or bool(set_intersections["r1_scale_0.1"]) or bool(set_intersections["r1_scale_0.01"]):
        return {"status": "partially_supported", "evidence": evidence}
    return {"status": "weakened", "evidence": evidence}


def _classify_h3(prefix_cases: list[dict[str, Any]]) -> dict[str, Any]:
    for case in prefix_cases:
        row = _dtype_row(case, PRIMARY_SCALE_DTYPE)
        if row is None:
            return {"status": "inconclusive", "evidence": f"missing row for {case['case_id']}"}
        if not row["implementation_agreement"]:
            first = row["first_failure"]
            evidence = {
                "first_failing_prefix": case["horizon"],
                "first_failing_time": first["time_index"],
                "first_failing_field_set": first["field_set"],
            }
            if first["status"] == "failed" and first["field_set"]:
                return {"status": "supported", "evidence": evidence}
            return {"status": "partially_supported", "evidence": evidence}
    return {
        "status": "weakened",
        "evidence": "no failing prefix reproduced in BF64 prefix ladder",
    }


def _classify_h4(r1_case: dict[str, Any]) -> dict[str, Any]:
    row = _dtype_row(r1_case, PRIMARY_SCALE_DTYPE)
    if row is None:
        return {"status": "inconclusive", "evidence": "missing BF64 R1 row"}
    first = row["first_failure"]
    field_set = set(first["field_set"])
    transport_fields = {
        "transport_matrix",
        "post_transport_particles",
        "row_residual",
        "column_residual",
    }
    intersect = field_set.intersection(transport_fields)
    evidence = {
        "first_failure": first,
        "transport_field_intersection": sorted(intersect),
        "row_residual_delta": row["primary_metrics"]["row_residual_delta"],
        "column_residual_delta": row["primary_metrics"]["column_residual_delta"],
    }
    if not first.get("triggered"):
        return {"status": "weakened", "evidence": evidence}
    if intersect and field_set.issubset(transport_fields):
        return {"status": "supported", "evidence": evidence}
    if intersect:
        return {"status": "partially_supported", "evidence": evidence}
    return {"status": "weakened", "evidence": evidence}


def _classify_h5(r1_case: dict[str, Any]) -> dict[str, Any]:
    row = _dtype_row(r1_case, PRIMARY_SCALE_DTYPE)
    if row is None:
        return {"status": "inconclusive", "evidence": "missing BF64 R1 row"}
    field_set = set(row["first_failure"]["field_set"])
    evidence = {"first_failure": row["first_failure"]}
    if "transport_cost_matrix" in field_set and not {
        "transport_matrix",
        "post_transport_particles",
    }.intersection(field_set):
        return {"status": "supported", "evidence": evidence}
    if "transport_cost_matrix" in field_set:
        return {"status": "partially_supported", "evidence": evidence}
    return {"status": "weakened", "evidence": evidence}


def _classify_h6(r1_case: dict[str, Any]) -> dict[str, Any]:
    row = _dtype_row(r1_case, PRIMARY_SCALE_DTYPE)
    if row is None:
        return {"status": "inconclusive", "evidence": "missing BF64 R1 row"}
    relative = row["primary_metrics"]["scalar_relative_delta"]
    field_set = set(row["first_failure"]["field_set"])
    residual_only = field_set.issubset({"row_residual", "column_residual"})
    evidence = {
        "scalar_relative_delta": relative,
        "first_failure": row["first_failure"],
        "residual_only_field_failure": residual_only,
    }
    if relative < 5e-7 and residual_only:
        return {"status": "supported", "evidence": evidence}
    if relative < 5e-7:
        return {"status": "partially_supported", "evidence": evidence}
    return {"status": "weakened", "evidence": evidence}


def _classify_h7(
    control_base: LocalizationScenario,
    r1_base: LocalizationScenario,
    scale_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    control_rms = _observation_summary(control_base.observations)["rms"]
    r1_rms = _observation_summary(r1_base.observations)["rms"]
    ratio = r1_rms / max(control_rms, 1e-12)
    rows = [_dtype_row(case, PRIMARY_SCALE_DTYPE) for case in scale_cases]
    if any(row is None for row in rows):
        return {"status": "inconclusive", "evidence": "missing BF64 scale rows"}
    deltas = [row["primary_metrics"]["scalar_delta"] for row in rows]
    monotone = deltas[0] >= deltas[1] >= deltas[2]
    evidence = {
        "control_observation_rms": control_rms,
        "r1_observation_rms": r1_rms,
        "rms_ratio": ratio,
        "scale_scalar_deltas": deltas,
        "monotone_scale_deltas": monotone,
    }
    if ratio >= 10.0 and monotone:
        return {"status": "supported", "evidence": evidence}
    if ratio >= 3.0 or monotone:
        return {"status": "partially_supported", "evidence": evidence}
    return {"status": "weakened", "evidence": evidence}


def _classify_h8() -> dict[str, Any]:
    audit = _wrapper_audit()
    status = "audit_risk_not_found" if audit["required_tokens_present"] else "audit_risk_identified"
    return {"status": status, "evidence": audit}


def _wrapper_audit() -> dict[str, Any]:
    source = step._filterflow_script()
    required_tokens = [
        "DTYPE = tf.float32",
        "from filterflow.resampling.differentiable.regularized_transport.plan import transport",
        "from filterflow.resampling.differentiable.regularized_transport.utils import cost, diameter",
        "transport(",
        "tf.constant(EPSILON, DTYPE)",
        "tf.constant(SCALING, DTYPE)",
        "tf.constant(CONVERGENCE_THRESHOLD, DTYPE)",
        "tf.constant(MAX_ITERATIONS, tf.int32)",
    ]
    missing = [token for token in required_tokens if token not in source]
    return {
        "status": "audit_localization_only",
        "source": "run_filterflow_1d_lgssm_step_gradient_comparison_tf._filterflow_script",
        "source_digest": stable_digest(source),
        "required_tokens_present": not missing,
        "missing_tokens": missing,
    }


def _case_agrees(case: dict[str, Any], dtype_label: str) -> bool:
    row = _dtype_row(case, dtype_label)
    return bool(row and row["implementation_agreement"])


def _dtype_row(case: dict[str, Any], dtype_label: str) -> dict[str, Any] | None:
    return case.get("dtype_rows", {}).get(dtype_label)


def _reduction_ratio(before: float, after: float) -> float:
    return float(before) / max(float(after), 1e-300)


def _decision(
    *,
    comparator_drift: bool,
    control_case: dict[str, Any],
    r1_case: dict[str, Any],
) -> str:
    if comparator_drift:
        return "r1_hypothesis_blocked_by_comparator_drift"
    if not _case_agrees(control_case, PRIMARY_SCALE_DTYPE):
        return "r1_hypothesis_harness_control_failed"
    if _case_agrees(r1_case, PRIMARY_SCALE_DTYPE):
        return "r1_hypothesis_r1_mismatch_not_reproduced"
    return "r1_hypothesis_localization_completed"


def _field_delta(left: Any, right: Any) -> float:
    left_tensor = tf.constant(left, tf.float64)
    right_tensor = tf.constant(right, tf.float64)
    return float(tf.reduce_max(tf.abs(left_tensor - right_tensor)).numpy())


def _field_tolerance(tolerance_key: str) -> float:
    tolerances = _tolerances()
    return float(tolerances.get(tolerance_key, OBSERVATION_FIELD_TOLERANCE))


def _tolerances() -> dict[str, float]:
    return agreement._agreement_tolerances()


def _observation_summary(values: tuple[float, ...]) -> dict[str, float]:
    count = len(values)
    if count == 0:
        return {"count": 0.0, "min": math.nan, "max": math.nan, "rms": math.nan}
    square_mean = sum(value * value for value in values) / count
    return {
        "count": float(count),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / count,
        "rms": math.sqrt(square_mean),
    }


def _parent_cpu_manifest() -> dict[str, Any]:
    return {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "pre_import_cuda_visible_devices": PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        "gpu_devices_visible": [str(device) for device in tf.config.list_physical_devices("GPU")],
    }


def _dtype_label(dtype: tf.DType) -> str:
    if dtype == tf.float32:
        return "float32"
    if dtype == tf.float64:
        return "float64"
    return dtype.name


def _validate_payload(payload: dict[str, Any]) -> None:
    required = {
        "decision",
        "plan_path",
        "result_path",
        "review_loop_path",
        "hypothesis_statuses",
        "filterflow_fingerprint_initial",
        "filterflow_fingerprint_final",
        "path_boundary_manifest",
        "run_manifest",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"missing payload keys: {sorted(missing)}")
    if payload["plan_review_status"] != "ACCEPT_after_round_4":
        raise ValueError("unexpected plan review status")
    if payload["run_manifest"].get("pre_import_cuda_visible_devices") != "-1":
        raise ValueError("parent pre-import CUDA_VISIBLE_DEVICES is not -1")
    if payload["run_manifest"].get("gpu_devices_visible") != []:
        raise ValueError("parent GPU devices visible")
    if any(bool(value) for value in payload["path_boundary_manifest"].values()):
        raise ValueError(f"path boundary violation: {payload['path_boundary_manifest']}")
    for key, value in payload["hypothesis_statuses"].items():
        if key not in {f"H{index}" for index in range(1, 9)}:
            raise ValueError(f"unexpected hypothesis key: {key}")
        if value["status"] not in ALLOWED_HYPOTHESIS_STATUSES:
            raise ValueError(f"invalid hypothesis status {key}: {value['status']}")
    for case_key in ("control_case", "r1_case"):
        case = payload.get(case_key)
        if case is not None:
            _validate_case(case)
    for case in payload.get("prefix_cases", []):
        _validate_case(case)
    for case in payload.get("scale_cases", []):
        _validate_case(case)


def _validate_case(case: dict[str, Any]) -> None:
    if case.get("filterflow_status") == "executed":
        manifest = case.get("filterflow_cpu_only_manifest", {})
        if manifest.get("pre_import_cuda_visible_devices") != "-1":
            raise ValueError(f"{case['case_id']} filterflow pre-import CUDA not -1")
        if manifest.get("gpu_devices_visible") != []:
            raise ValueError(f"{case['case_id']} filterflow GPU devices visible")
    for dtype_label, row in case.get("dtype_rows", {}).items():
        manifest = row.get("cpu_only_manifest", {})
        if manifest.get("pre_import_cuda_visible_devices") != "-1":
            raise ValueError(f"{case['case_id']} {dtype_label} CPU manifest failed")
        if manifest.get("gpu_devices_visible") != []:
            raise ValueError(f"{case['case_id']} {dtype_label} GPU devices visible")


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# R1 Observation-Path Mismatch Hypothesis Result",
        "",
        "## Decision",
        "",
        f"`{payload['decision']}`",
        "",
        "## Hypothesis Statuses",
        "",
        "| Hypothesis | Status | Key evidence |",
        "| --- | --- | --- |",
    ]
    for key in [f"H{index}" for index in range(1, 9)]:
        row = payload["hypothesis_statuses"][key]
        lines.append(f"| `{key}` | `{row['status']}` | `{_short_evidence(row['evidence'])}` |")
    lines.extend(
        [
            "",
            "## Control And R1",
            "",
            "| Case | BF64 agreement | BF32 agreement | BF64 scalar delta | BF32 scalar delta | BF64 first failure |",
            "| --- | --- | --- | ---: | ---: | --- |",
        ]
    )
    for case_key in ("control_case", "r1_case"):
        case = payload.get(case_key)
        if case is None:
            continue
        bf64 = _dtype_row(case, "float64")
        bf32 = _dtype_row(case, "float32")
        lines.append(
            f"| `{case['case_id']}` | `{_agreement_cell(bf64)}` | `{_agreement_cell(bf32)}` | "
            f"`{_scalar_delta_cell(bf64)}` | `{_scalar_delta_cell(bf32)}` | "
            f"`{_first_failure_cell(bf64)}` |"
        )
    lines.extend(
        [
            "",
            "## First Prefix Failures",
            "",
            "| Prefix | BF64 agreement | BF64 first failure | BF32 agreement |",
            "| ---: | --- | --- | --- |",
        ]
    )
    for case in payload.get("prefix_cases", []):
        bf64 = _dtype_row(case, "float64")
        bf32 = _dtype_row(case, "float32")
        lines.append(
            f"| `{case['horizon']}` | `{_agreement_cell(bf64)}` | "
            f"`{_first_failure_cell(bf64)}` | `{_agreement_cell(bf32)}` |"
        )
    lines.extend(
        [
            "",
            "## Scale Diagnostics",
            "",
            "| Scale case | BF64 scalar delta | BF64 first failure | BF32 scalar delta |",
            "| --- | ---: | --- | ---: |",
        ]
    )
    for case in payload.get("scale_cases", []):
        bf64 = _dtype_row(case, "float64")
        bf32 = _dtype_row(case, "float32")
        lines.append(
            f"| `{case['case_id']}` | `{_scalar_delta_cell(bf64)}` | "
            f"`{_first_failure_cell(bf64)}` | `{_scalar_delta_cell(bf32)}` |"
        )
    lines.extend(
        [
            "",
            "## Observation Summaries",
            "",
            "```json",
            json.dumps(payload.get("observation_summaries", {}), indent=2, sort_keys=True),
            "```",
            "",
            "## Non-Implications",
            "",
            *[f"- {item}" for item in payload["non_implications"]],
            "",
        ]
    )
    return "\n".join(lines)


def _agreement_cell(row: dict[str, Any] | None) -> str:
    if row is None:
        return "missing"
    return str(bool(row["implementation_agreement"]))


def _scalar_delta_cell(row: dict[str, Any] | None) -> str:
    if row is None:
        return "missing"
    return str(row["primary_metrics"]["scalar_delta"])


def _first_failure_cell(row: dict[str, Any] | None) -> str:
    if row is None:
        return "missing"
    first = row["first_failure"]
    return f"t={first['time_index']}, fields={first['field_set']}"


def _short_evidence(evidence: Any) -> str:
    text = json.dumps(evidence, sort_keys=True, default=str)
    if len(text) > 240:
        return text[:237] + "..."
    return text


def _digest_payload(payload: dict[str, Any]) -> str:
    clone = dict(payload)
    clone.pop("run_manifest", None)
    clone.pop("created_at_utc", None)
    clone.pop("reproducibility_digest", None)
    return stable_digest(clone)


if __name__ == "__main__":
    raise SystemExit(main())
