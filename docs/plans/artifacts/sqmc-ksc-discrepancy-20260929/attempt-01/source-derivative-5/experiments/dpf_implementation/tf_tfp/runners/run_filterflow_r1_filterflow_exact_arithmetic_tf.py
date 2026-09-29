"""R1 BayesFilter/filterflow comparison using filterflow-exact arithmetic.

This runner only audits cross-implementation agreement against the local
executable filterflow checkout. It does not assert correctness of either
implementation.
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


PLAN_PATH = "docs/plans/bayesfilter-dpf-r1-filterflow-exact-arithmetic-plan-2026-06-02.md"
RESULT_PATH = "docs/plans/bayesfilter-dpf-r1-filterflow-exact-arithmetic-result-2026-06-02.md"
JSON_PATH = OUTPUT_DIR / "dpf_filterflow_r1_filterflow_exact_arithmetic_2026-06-02.json"
REPORT_PATH = REPORT_DIR / "dpf-filterflow-r1-filterflow-exact-arithmetic-2026-06-02.md"
CASES = (
    ("matched_control_generated_T100", "control_full"),
    ("r1_prefix_T4", "r1_prefix_T4"),
    ("r1_filterflow_observation_path_T100", "r1_full"),
)
ARITHMETIC_TOKENS = (
    "DTYPE = tf.float32",
    "xx = tf.reduce_sum",
    "xy = tf.matmul",
    "tf.clip_by_value",
    "log_weight_normalization_residual",
    "epsilon_0 = particles_diameter ** 2",
    "_filterflow_exact_sinkhorn_loop",
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
    if fixture.get("status") != "executed":
        return _blocked_payload(
            "r1_filterflow_exact_arithmetic_fixture_blocked",
            fixture.get("blocker", "unknown fixture blocker"),
            initial_fingerprint,
        )

    control = localization._control_base_scenario()
    r1 = localization._r1_base_scenario(fixture, control)
    scenarios = {
        "control_full": control,
        "r1_prefix_T4": localization._prefix_scenario(r1, 4),
        "r1_full": r1,
    }
    cases = [
        _run_case(case_id=case_id, scenario=scenarios[scenario_key])
        for case_id, scenario_key in CASES
    ]
    final_fingerprint = continuation._filterflow_fingerprint()
    comparator_drift = continuation._fingerprints_drifted(initial_fingerprint, final_fingerprint)
    decision = _decision(cases, comparator_drift)
    return {
        "decision": decision,
        "created_at_utc": utc_now(),
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "report_path": str(REPORT_PATH.relative_to(REPO_ROOT)),
        "json_path": str(JSON_PATH.relative_to(REPO_ROOT)),
        "question": (
            "Does BayesFilter match local executable filterflow on the R1 "
            "observation-path comparison when BayesFilter uses filterflow's "
            "float32 arithmetic convention?"
        ),
        "evidence_contract": {
            "primary_comparator": "local executable patched filterflow checkout",
            "primary_question": "cross_implementation_difference_only",
            "primary_pass": "filterflow_exact_float32_agreement",
            "mathematical_correctness": "not_concluded",
        },
        "arithmetic_policy": {
            "exact_path_dtype": "tf.float32",
            "distance_formula": "filterflow xx - 2xy + yy with clipping",
            "log_weight_policy": "do not renormalize before transport; record residual",
            "annealing_start": "filterflow max_min(scaled_x, scaled_x)^2 without pre-clamp",
            "fixed_target_sinkhorn": "not used",
        },
        "case_results": cases,
        "summary": _summary(cases),
        "fixture_summary": agreement._fixture_summary(fixture),
        "arithmetic_source_audit": _arithmetic_source_audit(),
        "filterflow_fingerprint_initial": initial_fingerprint,
        "filterflow_fingerprint_final": final_fingerprint,
        "comparator_drift": comparator_drift,
        "path_boundary_manifest": continuation._path_boundary_manifest(),
        "tolerances": agreement._agreement_tolerances(),
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners."
                "run_filterflow_r1_filterflow_exact_arithmetic_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_implications": step._non_implications()
        + [
            "No correctness claim is made for either implementation.",
            "No production dtype default is concluded.",
            "No tolerance policy change is concluded.",
        ],
    }


def _blocked_payload(
    decision: str,
    blocker: str,
    initial_fingerprint: dict[str, Any],
) -> dict[str, Any]:
    final_fingerprint = continuation._filterflow_fingerprint()
    return {
        "decision": decision,
        "created_at_utc": utc_now(),
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "report_path": str(REPORT_PATH.relative_to(REPO_ROOT)),
        "json_path": str(JSON_PATH.relative_to(REPO_ROOT)),
        "blocker": blocker,
        "case_results": [],
        "summary": {"status": "blocked", "blocker": blocker},
        "arithmetic_source_audit": _arithmetic_source_audit(),
        "filterflow_fingerprint_initial": initial_fingerprint,
        "filterflow_fingerprint_final": final_fingerprint,
        "comparator_drift": continuation._fingerprints_drifted(initial_fingerprint, final_fingerprint),
        "path_boundary_manifest": continuation._path_boundary_manifest(),
        "tolerances": agreement._agreement_tolerances(),
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners."
                "run_filterflow_r1_filterflow_exact_arithmetic_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_implications": step._non_implications(),
    }


def _run_case(
    *,
    case_id: str,
    scenario: localization.LocalizationScenario,
) -> dict[str, Any]:
    filterflow = localization._filterflow_reference(scenario)
    if filterflow.get("status") != "executed":
        return {
            "case_id": case_id,
            "scenario_id": scenario.scenario_id,
            "horizon": scenario.horizon,
            "status": "blocked",
            "blocker": filterflow.get("blocker", "unknown filterflow blocker"),
            "filterflow": filterflow,
        }
    bf64 = _bayesfilter_arithmetic_reference(scenario, dtype=tf.float64, arithmetic_label="bayesfilter_float64")
    bf32 = _bayesfilter_arithmetic_reference(scenario, dtype=tf.float32, arithmetic_label="filterflow_exact_float32")
    comparisons = {
        "bayesfilter_float64": _comparison_payload(bf64, filterflow),
        "filterflow_exact_float32": _comparison_payload(bf32, filterflow),
    }
    return {
        "case_id": case_id,
        "scenario_id": scenario.scenario_id,
        "horizon": scenario.horizon,
        "status": _case_status(comparisons["filterflow_exact_float32"]),
        "filterflow_status": filterflow.get("status"),
        "bayesfilter_float64": _summary_reference(bf64),
        "filterflow_exact_float32": _summary_reference(bf32),
        "filterflow": _summary_reference(filterflow),
        "comparisons": comparisons,
    }


def _bayesfilter_arithmetic_reference(
    scenario: localization.LocalizationScenario,
    *,
    dtype: tf.DType,
    arithmetic_label: str,
) -> dict[str, Any]:
    step_originals = localization._snapshot_step_globals()
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
        theta = tf.Variable(step.THETA0, dtype=dtype)
        with tf.GradientTape() as tape:
            scalar_value, ledger = step._run_bayesfilter_scalar(theta)
        gradient = tape.gradient(scalar_value, theta)
        plus, _ = step._run_bayesfilter_scalar(
            tf.constant(step.THETA0 + step.FINITE_DIFF_STEP, dtype)
        )
        minus, _ = step._run_bayesfilter_scalar(
            tf.constant(step.THETA0 - step.FINITE_DIFF_STEP, dtype)
        )
        finite_difference = (plus - minus) / tf.constant(
            2.0 * step.FINITE_DIFF_STEP,
            dtype,
        )
        return {
            "status": "executed",
            "backend": "tensorflow_tensorflow_probability",
            "arithmetic_label": arithmetic_label,
            "diagnostic_dtype": _dtype_label(dtype),
            "scalar": step._float(scalar_value),
            "gradient_tape": step._maybe_float(gradient),
            "finite_difference_gradient": step._float(finite_difference),
            "gradient_delta_vs_finite_difference": step._maybe_float(
                gradient - finite_difference if gradient is not None else None
            ),
            "finite_scalar": step._finite(scalar_value),
            "finite_gradient": gradient is not None and step._finite(gradient),
            "ledger": ledger,
            "cpu_only_manifest": _parent_cpu_manifest(),
        }
    finally:
        localization._restore_step_globals(step_originals)
        annealed_transport_tf.DTYPE = annealed_dtype


def _comparison_payload(
    bayesfilter: dict[str, Any],
    filterflow: dict[str, Any],
) -> dict[str, Any]:
    comparison = agreement._compare_runs(bayesfilter, filterflow)
    detailed = localization._detailed_field_comparison(bayesfilter, filterflow)
    first_failure = detailed["first_failure"]
    return {
        "status": comparison.get("status"),
        "implementation_agreement": bool(comparison.get("implementation_agreement")),
        "scalar_delta": comparison.get("scalar_delta"),
        "scalar_within_tolerance": comparison.get("scalar_within_tolerance"),
        "ledger_within_tolerance": comparison.get("ledger_within_tolerance"),
        "trigger_match": comparison.get("trigger_match"),
        "max_field_delta": detailed["max_field_delta"],
        "first_failure": first_failure,
        "residual_deltas": comparison.get("residual_deltas"),
        "absolute_residuals": comparison.get("absolute_residuals"),
        "per_time_field_deltas": detailed["per_time_field_deltas"],
    }


def _summary_reference(reference: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": reference.get("status"),
        "arithmetic_label": reference.get("arithmetic_label"),
        "diagnostic_dtype": reference.get("diagnostic_dtype"),
        "scalar": reference.get("scalar"),
        "gradient_tape": reference.get("gradient_tape"),
        "finite_difference_gradient": reference.get("finite_difference_gradient"),
        "finite_scalar": reference.get("finite_scalar"),
        "finite_gradient": reference.get("finite_gradient"),
        "ledger_length": len(reference.get("ledger", [])),
        "cpu_only_manifest": reference.get("cpu_only_manifest"),
    }


def _case_status(exact_comparison: dict[str, Any]) -> str:
    if exact_comparison.get("status") != "compared":
        return "blocked"
    return "pass" if exact_comparison.get("implementation_agreement") else "mismatch"


def _summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    exact_passes = [
        case.get("status") == "pass"
        for case in cases
        if case.get("status") in {"pass", "mismatch"}
    ]
    failing = [
        {
            "case_id": case["case_id"],
            "first_failure": case["comparisons"]["filterflow_exact_float32"]["first_failure"],
            "scalar_delta": case["comparisons"]["filterflow_exact_float32"]["scalar_delta"],
            "max_field_delta": case["comparisons"]["filterflow_exact_float32"]["max_field_delta"],
        }
        for case in cases
        if case.get("status") == "mismatch"
    ]
    return {
        "case_count": len(cases),
        "filterflow_exact_all_pass": bool(exact_passes) and all(exact_passes),
        "blocked_cases": [case["case_id"] for case in cases if case.get("status") == "blocked"],
        "failing_cases": failing,
        "cross_implementation_question_only": True,
    }


def _decision(cases: list[dict[str, Any]], comparator_drift: bool) -> str:
    if comparator_drift:
        return "r1_filterflow_exact_arithmetic_blocked_by_comparator_drift"
    if any(case.get("status") == "blocked" for case in cases):
        return "r1_filterflow_exact_arithmetic_filterflow_blocker"
    control = next(case for case in cases if case["case_id"] == "matched_control_generated_T100")
    if control.get("status") != "pass":
        return "r1_filterflow_exact_arithmetic_control_failed"
    r1_prefix = next(case for case in cases if case["case_id"] == "r1_prefix_T4")
    r1_full = next(case for case in cases if case["case_id"] == "r1_filterflow_observation_path_T100")
    if r1_prefix.get("status") == "pass" and r1_full.get("status") == "pass":
        return "r1_filterflow_exact_arithmetic_r1_match"
    if r1_prefix.get("status") == "pass":
        return "r1_filterflow_exact_arithmetic_prefix_match_full_mismatch"
    return "r1_filterflow_exact_arithmetic_mismatch"


def _arithmetic_source_audit() -> dict[str, Any]:
    sources = {
        "filterflow_script": step._filterflow_script(),
        "annealed_transport_tf": (
            REPO_ROOT
            / "experiments/dpf_implementation/tf_tfp/resampling/annealed_transport_tf.py"
        ).read_text(encoding="utf-8"),
        "step_comparison_runner": (
            REPO_ROOT
            / "experiments/dpf_implementation/tf_tfp/runners/"
            / "run_filterflow_1d_lgssm_step_gradient_comparison_tf.py"
        ).read_text(encoding="utf-8"),
    }
    token_locations = {
        token: [name for name, source in sources.items() if token in source]
        for token in ARITHMETIC_TOKENS
    }
    return {
        "status": "passed" if all(token_locations.values()) else "missing_token",
        "token_locations": token_locations,
        "source_digests": {name: stable_digest(source) for name, source in sources.items()},
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
        "case_results",
        "summary",
        "arithmetic_source_audit",
        "filterflow_fingerprint_initial",
        "filterflow_fingerprint_final",
        "comparator_drift",
        "path_boundary_manifest",
        "run_manifest",
        "non_implications",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"missing payload keys: {sorted(missing)}")
    allowed = {
        "r1_filterflow_exact_arithmetic_fixture_blocked",
        "r1_filterflow_exact_arithmetic_blocked_by_comparator_drift",
        "r1_filterflow_exact_arithmetic_filterflow_blocker",
        "r1_filterflow_exact_arithmetic_control_failed",
        "r1_filterflow_exact_arithmetic_r1_match",
        "r1_filterflow_exact_arithmetic_prefix_match_full_mismatch",
        "r1_filterflow_exact_arithmetic_mismatch",
    }
    if payload["decision"] not in allowed:
        raise ValueError(f"unexpected decision: {payload['decision']}")
    manifest = payload["run_manifest"]
    if manifest.get("pre_import_cuda_visible_devices") != "-1":
        raise ValueError("parent pre-import CUDA_VISIBLE_DEVICES is not -1")
    if manifest.get("cuda_visible_devices") != "-1":
        raise ValueError("parent CUDA_VISIBLE_DEVICES is not -1")
    if manifest.get("gpu_devices_visible") != []:
        raise ValueError(f"parent GPU devices visible: {manifest.get('gpu_devices_visible')}")
    if any(bool(value) for value in payload["path_boundary_manifest"].values()):
        raise ValueError(f"path boundary violation: {payload['path_boundary_manifest']}")
    if payload["arithmetic_source_audit"]["status"] != "passed":
        raise ValueError(f"arithmetic source audit failed: {payload['arithmetic_source_audit']}")
    if payload["decision"] == "r1_filterflow_exact_arithmetic_fixture_blocked":
        return
    if payload["comparator_drift"]:
        raise ValueError("comparator drift detected")
    for case in payload["case_results"]:
        if case.get("status") == "blocked":
            if payload["decision"] != "r1_filterflow_exact_arithmetic_filterflow_blocker":
                raise ValueError("blocked case with non-blocker decision")
            continue
        exact = case["comparisons"]["filterflow_exact_float32"]
        if exact["status"] != "compared":
            raise ValueError(f"case comparison did not run: {case['case_id']}")
        if exact["trigger_match"] is not True:
            raise ValueError(f"trigger mismatch in {case['case_id']}")
        if case.get("status") == "pass" and not exact["implementation_agreement"]:
            raise ValueError(f"pass case without implementation agreement: {case['case_id']}")
    if payload["decision"] == "r1_filterflow_exact_arithmetic_r1_match":
        if not payload["summary"]["filterflow_exact_all_pass"]:
            raise ValueError("match decision without all exact cases passing")


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# R1 Filterflow-Exact Arithmetic",
        "",
        "## Decision",
        "",
        f"`{payload['decision']}`",
        "",
        "## Contract",
        "",
        "This artifact compares BayesFilter against the local executable filterflow checkout only.",
        "It does not assert correctness of either implementation.",
        "",
        "## Arithmetic Policy",
        "",
        "| Control | Value |",
        "| --- | --- |",
    ]
    for key, value in payload.get("arithmetic_policy", {}).items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(["", "## Cases", "", "| Case | Status | Exact scalar delta | Exact max field delta | First failure |", "| --- | --- | ---: | ---: | --- |"])
    for case in payload["case_results"]:
        if case.get("status") == "blocked":
            lines.append(f"| `{case['case_id']}` | `blocked` | N/A | N/A | `{case.get('blocker')}` |")
            continue
        exact = case["comparisons"]["filterflow_exact_float32"]
        first = exact["first_failure"]
        lines.append(
            f"| `{case['case_id']}` | `{case['status']}` | "
            f"`{exact['scalar_delta']}` | `{exact['max_field_delta']}` | "
            f"`{first['status']}@{first['time_index']}:{first['field_set']}` |"
        )
    lines.extend(
        [
            "",
            "## BF64 Contrast",
            "",
            "| Case | BF64 scalar delta | BF64 max field delta | BF64 first failure |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for case in payload["case_results"]:
        if case.get("status") == "blocked":
            continue
        bf64 = case["comparisons"]["bayesfilter_float64"]
        first = bf64["first_failure"]
        lines.append(
            f"| `{case['case_id']}` | `{bf64['scalar_delta']}` | "
            f"`{bf64['max_field_delta']}` | "
            f"`{first['status']}@{first['time_index']}:{first['field_set']}` |"
        )
    lines.extend(
        [
            "",
            "## Source Audit",
            "",
            f"`{payload['arithmetic_source_audit']['status']}`",
            "",
            "## Non-Implications",
            "",
            *[f"- {item}" for item in payload["non_implications"]],
            "",
        ]
    )
    return "\n".join(lines)


def _digest_payload(payload: dict[str, Any]) -> str:
    clone = dict(payload)
    clone.pop("run_manifest", None)
    clone.pop("created_at_utc", None)
    clone.pop("reproducibility_digest", None)
    return stable_digest(clone)


if __name__ == "__main__":
    raise SystemExit(main())
