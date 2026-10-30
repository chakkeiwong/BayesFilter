"""1D-to-smoothness LGSSM continuation ladder against executable filterflow."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/bayesfilter-dpf-mpl")
PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]

import argparse
import json
import subprocess
import textwrap
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


PLAN_PATH = "docs/plans/bayesfilter-dpf-1d-to-smoothness-lgssm-continuation-ladder-plan-2026-06-02.md"
RESULT_PATH = "docs/plans/bayesfilter-dpf-1d-to-smoothness-lgssm-continuation-ladder-result-2026-06-02.md"
REVIEW_LOOP_PATH = "docs/plans/bayesfilter-dpf-1d-to-smoothness-lgssm-continuation-ladder-review-loop-2026-06-02.md"
JSON_PATH = OUTPUT_DIR / "dpf_filterflow_1d_to_smoothness_ladder_2026-06-02.json"
REPORT_PATH = REPORT_DIR / "dpf-filterflow-1d-to-smoothness-ladder-2026-06-02.md"
FILTERFLOW_ENV_PYTHON = REPO_ROOT / ".localenv" / "filterflow-py311" / "bin" / "python"
FILTERFLOW_PATH = REPO_ROOT / ".localsource" / "filterflow"

R1_THRESHOLDS = (1e-6, 1e-7, 1e-8)
R1_MAX_ITERATIONS = (200, 500, 1000)
RUNG_IDS = (
    "R1_T4_residual_ladder",
    "R2_1d_horizon_ladder",
    "R3_1d_particle_count_ladder",
    "R4_1d_random_stream_alignment",
    "R5_1d_resampling_policy_match",
    "R6_1d_parameter_grid_surface",
    "R7_1d_smoothness_scalar_contract",
    "R8_2d_constant_velocity_bridge",
)


@dataclass(frozen=True)
class CellConfig:
    convergence_threshold: float
    max_iterations: int


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
    initial_fingerprint = _filterflow_fingerprint()
    rung_ledger: list[dict[str, Any]] = []
    r1 = _run_r1(initial_fingerprint)
    rung_ledger.append(r1)
    first_failing = r1["rung"] if r1["status"] == "veto" else None
    first_blocked = None

    if r1["status"] != "pass":
        inherited = r1["blocker_reason"] or "blocked_by_R1_residual_veto"
        for rung in RUNG_IDS[1:]:
            if first_blocked is None:
                first_blocked = rung
            rung_ledger.append(_blocked_rung(rung, inherited, first_failing, first_blocked))
    else:
        selected = r1["primary_metrics"]["selected_r1_cell"]
        r2 = _run_r2(initial_fingerprint, selected)
        rung_ledger.append(r2)
        if r2["status"] == "veto" and first_failing is None:
            first_failing = r2["rung"]
        if r2["status"] != "pass":
            inherited = r2["blocker_reason"] or "blocked_by_R2_veto"
            for rung in RUNG_IDS[2:]:
                if first_blocked is None:
                    first_blocked = rung
                rung_ledger.append(_blocked_rung(rung, inherited, first_failing, first_blocked))
        else:
            inherited = "blocked_by_unimplemented_post_R2_continuation_after_R1_R2_pass"
            first_blocked = RUNG_IDS[2]
            for rung in RUNG_IDS[2:]:
                rung_ledger.append(_blocked_rung(rung, inherited, first_failing, first_blocked))

    final_fingerprint = _filterflow_fingerprint()
    if _fingerprints_drifted(initial_fingerprint, final_fingerprint):
        rung_ledger = _mark_comparator_drift(rung_ledger, initial_fingerprint, final_fingerprint)
        first_failing = None
        first_blocked = next((row["rung"] for row in rung_ledger if row["status"] == "blocked"), None)

    first_failing_field = first_failing or "none_observed_in_bounded_ladder"
    first_blocked_field = first_blocked or "none_observed_in_bounded_ladder"
    technical_decision = _decision(first_failing_field, first_blocked_field, rung_ledger)
    return {
        "decision": "one_d_to_smoothness_ladder_protocol_blocked_inspection_only",
        "technical_observation_decision": technical_decision,
        "protocol_status": "blocked_inspection_only_nonaccepted_plan",
        "downstream_use": (
            "blocked until the plan is accepted in a reviewed loop or the human "
            "explicitly authorizes this inspection-only artifact as sufficient"
        ),
        "created_at_utc": utc_now(),
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "review_loop_path": REVIEW_LOOP_PATH,
        "report_path": str(REPORT_PATH.relative_to(REPO_ROOT)),
        "json_path": str(JSON_PATH.relative_to(REPO_ROOT)),
        "question": (
            "Within the executed bounded scalar-prefix of the "
            "1D-to-smoothness ladder, which rung first fails?"
        ),
        "answer_scope": (
            "This run localizes the first evidence-bearing failure within the "
            "executed R1/R2 scalar-state prefix. It does not answer the full "
            "smoothness-alignment question because R3-R8 are blocked after the "
            "R2 veto."
        ),
        "plan_review_status": "REJECT_patched_for_user_inspection_after_round_5",
        "reference_policy": {
            "canonical_executable_reference": "current local patched .localsource/filterflow",
            "authoritative_identity": "HEAD commit plus local-diff/status fingerprint",
            "branch_string_status": "descriptive_only",
            "filterflow_source_mutation": "runner does not mutate filterflow source",
            "fixed_target_sinkhorn": "not used",
        },
        "first_failing_rung": first_failing_field,
        "first_blocked_rung": first_blocked_field,
        "selected_inherited_setting": _selected_setting(rung_ledger),
        "filterflow_fingerprint_initial": initial_fingerprint,
        "filterflow_fingerprint_final": final_fingerprint,
        "rung_ledger": rung_ledger,
        "path_boundary_manifest": _path_boundary_manifest(),
        "tolerances": step.TOLERANCES,
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners."
                "run_filterflow_1d_to_smoothness_ladder_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_implications": step._non_implications(),
    }


def _run_r1(initial_fingerprint: dict[str, Any]) -> dict[str, Any]:
    cells = []
    scenario = horizon.SCENARIOS[1]
    for max_iterations in R1_MAX_ITERATIONS:
        for threshold in R1_THRESHOLDS:
            before = _filterflow_fingerprint()
            cell = _run_cell(
                scenario=scenario,
                config=CellConfig(threshold, max_iterations),
            )
            after = _filterflow_fingerprint()
            cell["comparator_fingerprint_before"] = before
            cell["comparator_fingerprint_after"] = after
            cells.append(cell)
    passing = [cell for cell in cells if cell["cell_status"] == "pass"]
    selected = _select_r1_cell(passing)
    status = "pass" if selected is not None else "veto"
    blocker = None if selected is not None else "blocked_by_R1_residual_veto"
    max_row = max(cell["primary_metrics"]["max_absolute_row_residual"] for cell in cells)
    max_col = max(cell["primary_metrics"]["max_absolute_column_residual"] for cell in cells)
    return _rung_row(
        rung="R1_T4_residual_ladder",
        status=status,
        evidence_bearing=True,
        blocker_reason=blocker,
        inherited_blocker=None,
        first_failing_rung="R1_T4_residual_ladder" if status == "veto" else None,
        first_blocked_rung=None,
        comparator_before=initial_fingerprint,
        comparator_after=cells[-1]["comparator_fingerprint_after"],
        fixed_variables={
            "scenario_id": scenario.scenario_id,
            "horizon": scenario.horizon,
            "theta": step.THETA0,
            "num_particles": step.NUM_PARTICLES,
            "epsilon": step.EPSILON,
            "scaling": step.SCALING,
            "observations": list(scenario.observations),
            "transition_noises": [list(row) for row in scenario.transition_noises],
        },
        varied_variables={
            "convergence_threshold": list(R1_THRESHOLDS),
            "max_iterations": list(R1_MAX_ITERATIONS),
        },
        primary_metrics={
            "selected_r1_cell": selected,
            "passing_cell_count": len(passing),
            "evaluated_cell_count": len(cells),
            "max_absolute_row_residual": max_row,
            "max_absolute_column_residual": max_col,
            "cell_summaries": cells,
        },
        veto_diagnostics={
            "residual_tolerance": step.TOLERANCES["row_residual"],
            "r2_unblocked": selected is not None,
        },
        explanatory_diagnostics={
            "known_prior_T4_row_residual": 0.0005233019270021178,
            "sweep_contract": (
                "threshold in {1e-6,1e-7,1e-8}; max_iterations in "
                "{200,500,1000}"
            ),
        },
        filterflow_cpu_only_manifest=_first_filterflow_cpu_manifest(cells),
    )


def _run_r2(initial_fingerprint: dict[str, Any], selected: dict[str, Any]) -> dict[str, Any]:
    config = CellConfig(
        convergence_threshold=float(selected["convergence_threshold"]),
        max_iterations=int(selected["max_iterations"]),
    )
    scenarios = [_generated_scenario(t) for t in (4, 8, 16, 32, 100)]
    rows = []
    status = "pass"
    blocker = None
    for scenario in scenarios:
        cell = _run_cell(scenario=scenario, config=config)
        rows.append(cell)
        if cell["cell_status"] != "pass":
            status = "veto"
            blocker = f"blocked_by_R2_horizon_{scenario.horizon}_veto"
            break
    max_row = max(row["primary_metrics"]["max_absolute_row_residual"] for row in rows)
    max_col = max(row["primary_metrics"]["max_absolute_column_residual"] for row in rows)
    return _rung_row(
        rung="R2_1d_horizon_ladder",
        status=status,
        evidence_bearing=True,
        blocker_reason=blocker,
        inherited_blocker=None,
        first_failing_rung="R2_1d_horizon_ladder" if status == "veto" else None,
        first_blocked_rung=None,
        comparator_before=initial_fingerprint,
        comparator_after=_filterflow_fingerprint(),
        fixed_variables={
            "theta": step.THETA0,
            "num_particles": step.NUM_PARTICLES,
            "epsilon": step.EPSILON,
            "scaling": step.SCALING,
            "selected_r1_cell": selected,
        },
        varied_variables={"horizon": [scenario.horizon for scenario in scenarios]},
        primary_metrics={
            "evaluated_horizons": [row["scenario"]["horizon"] for row in rows],
            "max_absolute_row_residual": max_row,
            "max_absolute_column_residual": max_col,
            "cell_summaries": rows,
        },
        veto_diagnostics={"residual_tolerance": step.TOLERANCES["row_residual"]},
        explanatory_diagnostics={
            "generated_fixture": "tf_stateless_normal_fixed_seed",
            "note": "R2 is reached only if R1 selects a passing residual setting.",
        },
        filterflow_cpu_only_manifest=_first_filterflow_cpu_manifest(rows),
    )


def _run_cell(*, scenario: horizon.Scenario, config: CellConfig) -> dict[str, Any]:
    bayesfilter = _bayesfilter_reference(scenario, config)
    filterflow = _run_filterflow_subprocess(scenario, config)
    comparison = horizon._compare_runs(bayesfilter, filterflow)
    cell_pass = _cell_pass(bayesfilter, filterflow, comparison)
    metrics = _cell_metrics(bayesfilter, filterflow, comparison)
    return {
        "cell_status": "pass" if cell_pass else "veto",
        "scenario": {
            "scenario_id": scenario.scenario_id,
            "horizon": scenario.horizon,
        },
        "convergence_threshold": config.convergence_threshold,
        "max_iterations": config.max_iterations,
        "bayesfilter_status": bayesfilter.get("status"),
        "filterflow_status": filterflow.get("status"),
        "comparison_status": comparison.get("status"),
        "filterflow_command": filterflow.get("command"),
        "primary_metrics": metrics,
        "veto_diagnostics": {
            "trigger_match": comparison.get("trigger_match"),
            "ledger_within_tolerance": comparison.get("ledger_within_tolerance"),
            "scalar_within_tolerance": comparison.get("scalar_within_tolerance"),
            "absolute_residuals_within_tolerance": comparison.get(
                "absolute_residuals_within_tolerance"
            ),
            "finite_bayesfilter_scalar": bayesfilter.get("finite_scalar"),
            "finite_filterflow_scalar": filterflow.get("finite_scalar"),
            "filterflow_blocker": filterflow.get("blocker"),
        },
        "explanatory_diagnostics": {
            "bayesfilter_gradient_tape": bayesfilter.get("gradient_tape"),
            "filterflow_gradient_tape": filterflow.get("gradient_tape"),
            "bayesfilter_finite_difference_gradient": bayesfilter.get(
                "finite_difference_gradient"
            ),
            "filterflow_finite_difference_gradient": filterflow.get(
                "finite_difference_gradient"
            ),
            "gradient_promotion": "not_concluded",
        },
        "filterflow_cpu_only_manifest": filterflow.get("cpu_only_manifest"),
    }


def _bayesfilter_reference(scenario: horizon.Scenario, config: CellConfig) -> dict[str, Any]:
    originals = _snapshot_step_globals()
    try:
        _apply_step_globals(scenario, config)
        return horizon._bayesfilter_reference(scenario)
    finally:
        _restore_step_globals(originals)


def _run_filterflow_subprocess(
    scenario: horizon.Scenario,
    config: CellConfig,
) -> dict[str, Any]:
    if not FILTERFLOW_ENV_PYTHON.exists():
        return {
            "status": "blocked",
            "blocker": f"missing filterflow env python: {FILTERFLOW_ENV_PYTHON}",
        }
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env["PYTHONPATH"] = str(FILTERFLOW_PATH)
    env["MPLCONFIGDIR"] = str(REPO_ROOT / ".cache" / "filterflow-mpl")
    completed = subprocess.run(
        [str(FILTERFLOW_ENV_PYTHON), "-c", _filterflow_outer_script(scenario, config)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    if completed.returncode != 0:
        return {
            "status": "blocked",
            "blocker": "filterflow subprocess failed",
            "stdout_excerpt": completed.stdout[-4000:],
            "stderr_excerpt": completed.stderr[-4000:],
        }
    stdout = completed.stdout
    start = stdout.rfind("FILTERFLOW_CONTINUATION_JSON_BEGIN")
    end = stdout.rfind("FILTERFLOW_CONTINUATION_JSON_END")
    if start < 0 or end < 0 or end <= start:
        return {
            "status": "blocked",
            "blocker": "filterflow continuation JSON sentinels missing",
            "stdout_excerpt": stdout[-4000:],
            "stderr_excerpt": completed.stderr[-4000:],
        }
    payload = json.loads(stdout[start + len("FILTERFLOW_CONTINUATION_JSON_BEGIN"):end].strip())
    payload["command"] = (
        f"CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=.localsource/filterflow "
        f"{FILTERFLOW_ENV_PYTHON} -c <1D continuation cell script>"
    )
    payload["stderr_excerpt"] = completed.stderr[-2000:]
    return payload


def _filterflow_outer_script(scenario: horizon.Scenario, config: CellConfig) -> str:
    return textwrap.dedent(
        f"""
        import json
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/bayesfilter-dpf-mpl")

        import experiments.dpf_implementation.tf_tfp.runners.run_filterflow_1d_lgssm_step_gradient_comparison_tf as step

        step.HORIZON = {scenario.horizon}
        step.OBSERVATIONS = {list(scenario.observations)!r}
        step.TRANSITION_NOISES = {[list(row) for row in scenario.transition_noises]!r}
        step.CONVERGENCE_THRESHOLD = {config.convergence_threshold!r}
        step.MAX_ITERATIONS = {config.max_iterations}

        payload = step._run_filterflow_subprocess()
        print("FILTERFLOW_CONTINUATION_JSON_BEGIN")
        print(json.dumps(payload, sort_keys=True))
        print("FILTERFLOW_CONTINUATION_JSON_END")
        """
    )


def _snapshot_step_globals() -> dict[str, Any]:
    return {
        "HORIZON": step.HORIZON,
        "OBSERVATIONS": step.OBSERVATIONS,
        "TRANSITION_NOISES": step.TRANSITION_NOISES,
        "CONVERGENCE_THRESHOLD": step.CONVERGENCE_THRESHOLD,
        "MAX_ITERATIONS": step.MAX_ITERATIONS,
    }


def _apply_step_globals(scenario: horizon.Scenario, config: CellConfig) -> None:
    step.HORIZON = scenario.horizon
    step.OBSERVATIONS = list(scenario.observations)
    step.TRANSITION_NOISES = [list(row) for row in scenario.transition_noises]
    step.CONVERGENCE_THRESHOLD = config.convergence_threshold
    step.MAX_ITERATIONS = config.max_iterations


def _restore_step_globals(values: dict[str, Any]) -> None:
    step.HORIZON = values["HORIZON"]
    step.OBSERVATIONS = values["OBSERVATIONS"]
    step.TRANSITION_NOISES = values["TRANSITION_NOISES"]
    step.CONVERGENCE_THRESHOLD = values["CONVERGENCE_THRESHOLD"]
    step.MAX_ITERATIONS = values["MAX_ITERATIONS"]


def _generated_scenario(horizon_value: int) -> horizon.Scenario:
    obs = tf.random.stateless_normal(
        [horizon_value],
        seed=[20260602, horizon_value],
        dtype=step.DTYPE,
    ) * tf.constant(0.1, step.DTYPE)
    noises = tf.random.stateless_normal(
        [horizon_value, step.NUM_PARTICLES],
        seed=[20260603, horizon_value],
        dtype=step.DTYPE,
    )
    return horizon.Scenario(
        scenario_id=f"generated_T{horizon_value}",
        observations=tuple(float(v) for v in obs.numpy().tolist()),
        transition_noises=tuple(tuple(float(v) for v in row) for row in noises.numpy().tolist()),
    )


def _cell_pass(
    bayesfilter: dict[str, Any],
    filterflow: dict[str, Any],
    comparison: dict[str, Any],
) -> bool:
    return bool(
        bayesfilter.get("finite_scalar")
        and filterflow.get("finite_scalar")
        and horizon._forward_match(comparison)
    )


def _cell_metrics(
    bayesfilter: dict[str, Any],
    filterflow: dict[str, Any],
    comparison: dict[str, Any],
) -> dict[str, Any]:
    if comparison.get("status") != "compared":
        return {
            "max_absolute_row_residual": float("inf"),
            "max_absolute_column_residual": float("inf"),
            "scalar_delta": float("inf"),
            "ledger_within_tolerance": False,
            "absolute_residuals_within_tolerance": False,
        }
    residuals = comparison["absolute_residuals"]
    return {
        "max_absolute_row_residual": max(
            residuals["bayesfilter_max_row_residual"],
            residuals["filterflow_max_row_residual"],
        ),
        "max_absolute_column_residual": max(
            residuals["bayesfilter_max_column_residual"],
            residuals["filterflow_max_column_residual"],
        ),
        "bayesfilter_max_row_residual": residuals["bayesfilter_max_row_residual"],
        "filterflow_max_row_residual": residuals["filterflow_max_row_residual"],
        "bayesfilter_max_column_residual": residuals["bayesfilter_max_column_residual"],
        "filterflow_max_column_residual": residuals["filterflow_max_column_residual"],
        "scalar_delta": comparison["scalar_delta"],
        "ledger_within_tolerance": comparison["ledger_within_tolerance"],
        "scalar_within_tolerance": comparison["scalar_within_tolerance"],
        "absolute_residuals_within_tolerance": comparison[
            "absolute_residuals_within_tolerance"
        ],
        "trigger_match": comparison["trigger_match"],
        "bayesfilter_scalar": bayesfilter.get("scalar"),
        "filterflow_scalar": filterflow.get("scalar"),
    }


def _select_r1_cell(cells: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not cells:
        return None
    selected = sorted(
        cells,
        key=lambda cell: (
            int(cell["max_iterations"]),
            -float(cell["convergence_threshold"]),
            float(cell["primary_metrics"]["max_absolute_row_residual"]),
        ),
    )[0]
    return {
        "convergence_threshold": selected["convergence_threshold"],
        "max_iterations": selected["max_iterations"],
        "max_absolute_row_residual": selected["primary_metrics"]["max_absolute_row_residual"],
        "max_absolute_column_residual": selected["primary_metrics"][
            "max_absolute_column_residual"
        ],
    }


def _rung_row(
    *,
    rung: str,
    status: str,
    evidence_bearing: bool,
    blocker_reason: str | None,
    inherited_blocker: str | None,
    first_failing_rung: str | None,
    first_blocked_rung: str | None,
    comparator_before: dict[str, Any],
    comparator_after: dict[str, Any],
    fixed_variables: dict[str, Any],
    varied_variables: dict[str, Any],
    primary_metrics: dict[str, Any],
    veto_diagnostics: dict[str, Any],
    explanatory_diagnostics: dict[str, Any],
    filterflow_cpu_only_manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "rung": rung,
        "status": status,
        "evidence_bearing": evidence_bearing,
        "failure_observed_directly": status == "veto" and inherited_blocker is None,
        "blocker_reason": blocker_reason,
        "inherited_blocker": inherited_blocker,
        "first_failing_rung": first_failing_rung,
        "first_blocked_rung": first_blocked_rung,
        "comparator_fingerprint_before": comparator_before,
        "comparator_fingerprint_after": comparator_after,
        "fixed_variables": fixed_variables,
        "varied_variables": varied_variables,
        "primary_metrics": primary_metrics,
        "veto_diagnostics": veto_diagnostics,
        "explanatory_diagnostics": explanatory_diagnostics,
        "parent_cpu_only_manifest": _parent_cpu_manifest(),
        "filterflow_cpu_only_manifest": filterflow_cpu_only_manifest,
    }


def _blocked_rung(
    rung: str,
    inherited: str,
    first_failing: str | None,
    first_blocked: str | None,
) -> dict[str, Any]:
    fingerprint = _filterflow_fingerprint()
    return _rung_row(
        rung=rung,
        status="blocked",
        evidence_bearing=False,
        blocker_reason=inherited,
        inherited_blocker=inherited,
        first_failing_rung=first_failing,
        first_blocked_rung=first_blocked,
        comparator_before=fingerprint,
        comparator_after=fingerprint,
        fixed_variables={},
        varied_variables={},
        primary_metrics={},
        veto_diagnostics={"blocked_before_execution": True},
        explanatory_diagnostics={
            "blocked_rung_interpretation": (
                "This rung was not evidence-bearing; the first failing rung is "
                "reported separately."
            )
        },
        filterflow_cpu_only_manifest=None,
    )


def _mark_comparator_drift(
    rows: list[dict[str, Any]],
    initial: dict[str, Any],
    final: dict[str, Any],
) -> list[dict[str, Any]]:
    marked = []
    for row in rows:
        new_row = dict(row)
        new_row["status"] = "blocked"
        new_row["evidence_bearing"] = False
        new_row["failure_observed_directly"] = False
        new_row["blocker_reason"] = "blocked_by_comparator_drift"
        new_row["inherited_blocker"] = "blocked_by_comparator_drift"
        new_row["comparator_fingerprint_before"] = initial
        new_row["comparator_fingerprint_after"] = final
        marked.append(new_row)
    return marked


def _parent_cpu_manifest() -> dict[str, Any]:
    return {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "pre_import_cuda_visible_devices": PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        "gpu_devices_visible": [str(device) for device in tf.config.list_physical_devices("GPU")],
    }


def _first_filterflow_cpu_manifest(cells: list[dict[str, Any]]) -> dict[str, Any] | None:
    for cell in cells:
        manifest = cell.get("filterflow_cpu_only_manifest")
        if manifest is not None:
            return manifest
    return None


def _filterflow_fingerprint() -> dict[str, Any]:
    if not FILTERFLOW_PATH.exists():
        return {"path": str(FILTERFLOW_PATH), "status": "missing"}
    branch = _git_filterflow(["rev-parse", "--abbrev-ref", "HEAD"])
    diff_raw = _git_filterflow(["diff", "--no-ext-diff"])
    pip_freeze = _filterflow_python(["-m", "pip", "freeze"])
    return {
        "path": str(FILTERFLOW_PATH),
        "status": "current_local_patched_checkout",
        "head_commit": _git_filterflow(["rev-parse", "HEAD"]),
        "symbolic_head": branch,
        "branch_string_status": "descriptive_only",
        "branch_ref_exists": _branch_ref_exists(branch),
        "status_short": _git_filterflow(["status", "--short"]),
        "status_branch": _git_filterflow(["status", "--short", "--branch"]),
        "diff_digest": stable_digest(diff_raw),
        "python_version": _filterflow_python(["--version"]),
        "package_manifest_digest": stable_digest(pip_freeze),
        "provenance_note": (
            "Comparator identity is HEAD commit plus local-diff/status "
            "fingerprint; branch string is descriptive only."
        ),
    }


def _git_filterflow(args: list[str]) -> str:
    completed = subprocess.run(
        ["git", "-C", str(FILTERFLOW_PATH), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _branch_ref_exists(branch: str) -> bool:
    completed = subprocess.run(
        ["git", "-C", str(FILTERFLOW_PATH), "show-ref", "--verify", f"refs/heads/{branch}"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def _filterflow_python(args: list[str]) -> str:
    if not FILTERFLOW_ENV_PYTHON.exists():
        return f"missing:{FILTERFLOW_ENV_PYTHON}"
    completed = subprocess.run(
        [str(FILTERFLOW_ENV_PYTHON), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return (completed.stdout or completed.stderr).strip()


def _fingerprints_drifted(left: dict[str, Any], right: dict[str, Any]) -> bool:
    keys = (
        "head_commit",
        "symbolic_head",
        "status_short",
        "diff_digest",
        "python_version",
        "package_manifest_digest",
    )
    return any(left.get(key) != right.get(key) for key in keys)


def _selected_setting(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    for row in rows:
        selected = row.get("primary_metrics", {}).get("selected_r1_cell")
        if selected is not None:
            return selected
    return None


def _path_boundary_manifest() -> dict[str, Any]:
    return {
        "production_bayesfilter_code_used": False,
        "tests_code_used": False,
        "monograph_chapters_used_or_edited": False,
        "highdim_lane_used_or_edited": False,
        "student_or_vendored_code_used_as_authority": False,
        "dsge_nawm_code_used": False,
        "filterflow_source_mutated": False,
    }


def _decision(
    first_failing: str,
    first_blocked: str,
    rows: list[dict[str, Any]],
) -> str:
    if any(row["blocker_reason"] == "blocked_by_comparator_drift" for row in rows):
        return "one_d_to_smoothness_ladder_blocked_by_comparator_drift"
    if first_failing == "R1_T4_residual_ladder":
        return "one_d_to_smoothness_ladder_R1_residual_veto"
    if first_failing == "R2_1d_horizon_ladder":
        return "one_d_to_smoothness_ladder_R2_veto"
    if first_blocked != "none_observed_in_bounded_ladder":
        return "one_d_to_smoothness_ladder_blocked_after_partial_pass"
    return "one_d_to_smoothness_ladder_no_failure_observed"


def _validate_payload(payload: dict[str, Any]) -> None:
    required = {
        "decision",
        "technical_observation_decision",
        "protocol_status",
        "downstream_use",
        "first_failing_rung",
        "first_blocked_rung",
        "rung_ledger",
        "run_manifest",
        "filterflow_fingerprint_initial",
        "filterflow_fingerprint_final",
        "path_boundary_manifest",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"missing payload keys: {sorted(missing)}")
    if len(payload["rung_ledger"]) != len(RUNG_IDS):
        raise ValueError("rung ledger must contain all eight rungs")
    manifest = payload["run_manifest"]
    if manifest.get("pre_import_cuda_visible_devices") != "-1":
        raise ValueError("parent pre-import CUDA_VISIBLE_DEVICES is not -1")
    if manifest.get("gpu_devices_visible") != []:
        raise ValueError("parent GPU devices visible")
    for row in payload["rung_ledger"]:
        if row["rung"] not in RUNG_IDS:
            raise ValueError(f"unknown rung: {row['rung']}")
        if row["status"] not in {"pass", "veto", "blocked", "diagnostic_only"}:
            raise ValueError(f"bad status for {row['rung']}: {row['status']}")
        parent_manifest = row.get("parent_cpu_only_manifest", {})
        if parent_manifest.get("pre_import_cuda_visible_devices") != "-1":
            raise ValueError(f"rung parent CPU pre-import invariant failed: {row['rung']}")
        if parent_manifest.get("gpu_devices_visible") != []:
            raise ValueError(f"rung parent GPU visible: {row['rung']}")
        ff_manifest = row.get("filterflow_cpu_only_manifest")
        if ff_manifest is not None:
            if ff_manifest.get("pre_import_cuda_visible_devices") != "-1":
                raise ValueError(f"filterflow CPU pre-import invariant failed: {row['rung']}")
            if ff_manifest.get("gpu_devices_visible") != []:
                raise ValueError(f"filterflow GPU visible: {row['rung']}")
    first_direct = next(
        (row["rung"] for row in payload["rung_ledger"] if row["failure_observed_directly"]),
        None,
    )
    expected_first = first_direct or "none_observed_in_bounded_ladder"
    if payload["first_failing_rung"] != expected_first:
        raise ValueError(
            f"first failing rung mismatch: {payload['first_failing_rung']} vs {expected_first}"
        )
    first_blocked = next(
        (row["rung"] for row in payload["rung_ledger"] if row["status"] == "blocked"),
        None,
    )
    expected_blocked = first_blocked or "none_observed_in_bounded_ladder"
    if payload["first_blocked_rung"] != expected_blocked:
        raise ValueError(
            f"first blocked rung mismatch: {payload['first_blocked_rung']} vs {expected_blocked}"
        )
    if not payload["path_boundary_manifest"]:
        raise ValueError("missing path boundary manifest")


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Result: 1D-to-Smoothness LGSSM Continuation Ladder",
        "",
        "## Decision",
        "",
        f"`{payload['decision']}`",
        "",
        f"Technical observation: `{payload['technical_observation_decision']}`",
        "",
        f"Protocol status: `{payload['protocol_status']}`",
        "",
        f"Downstream use: {payload['downstream_use']}",
        "",
        f"Plan review status: `{payload['plan_review_status']}`",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| first failing rung | `{payload['first_failing_rung']}` |",
        f"| first blocked rung | `{payload['first_blocked_rung']}` |",
        f"| selected inherited setting | `{payload['selected_inherited_setting']}` |",
        "",
        "## Decision Table",
        "",
        "| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |",
        "| --- | --- | --- | --- | --- | --- |",
        (
            f"| `{payload['decision']}` | "
            "`protocol blocked; technical R1/R2 prefix executed` | "
            "`R2 T=32 residual veto; plan review non-acceptance veto` | "
            "`whether R2 failure is fixture-specific, horizon-specific, or transport convergence/annealing-specific` | "
            "`reopen plan review or obtain explicit human authorization, then isolate R2 T=32 residual with horizon/cost/iteration diagnostics` | "
            "`full smoothness alignment, production readiness, gradient correctness` |"
        ),
        "",
        "## Rung Ledger",
        "",
        "| Rung | Status | Evidence-bearing | Direct failure | Veto/block reason | Inherited blocker |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["rung_ledger"]:
        lines.append(
            f"| `{row['rung']}` | `{row['status']}` | "
            f"`{row['evidence_bearing']}` | `{row['failure_observed_directly']}` | "
            f"`{row['blocker_reason']}` | `{row['inherited_blocker']}` |"
        )
    lines.extend(["", "## Canonical Per-Rung Ledger", ""])
    for row in payload["rung_ledger"]:
        lines.extend(_canonical_rung_markdown(row))
    lines.extend(["", "## R1 Cell Summary", ""])
    r1 = payload["rung_ledger"][0]
    cells = r1.get("primary_metrics", {}).get("cell_summaries", [])
    if cells:
        lines.extend(
            [
                "| threshold | max iterations | status | scalar delta | max row residual | max column residual |",
                "| ---: | ---: | --- | ---: | ---: | ---: |",
            ]
        )
        for cell in cells:
            metrics = cell["primary_metrics"]
            lines.append(
                f"| `{cell['convergence_threshold']}` | `{cell['max_iterations']}` | "
                f"`{cell['cell_status']}` | `{metrics['scalar_delta']}` | "
                f"`{metrics['max_absolute_row_residual']}` | "
                f"`{metrics['max_absolute_column_residual']}` |"
            )
    lines.extend(["", "## R2 Horizon Summary", ""])
    r2_cells = []
    if len(payload["rung_ledger"]) > 1:
        r2_cells = payload["rung_ledger"][1].get("primary_metrics", {}).get("cell_summaries", [])
    if r2_cells:
        lines.extend(
            [
                "| horizon | status | scalar delta | max row residual | max column residual | residual pass |",
                "| ---: | --- | ---: | ---: | ---: | --- |",
            ]
        )
        for cell in r2_cells:
            metrics = cell["primary_metrics"]
            veto = cell["veto_diagnostics"]
            lines.append(
                f"| `{cell['scenario']['horizon']}` | `{cell['cell_status']}` | "
                f"`{metrics['scalar_delta']}` | `{metrics['max_absolute_row_residual']}` | "
                f"`{metrics['max_absolute_column_residual']}` | "
                f"`{veto['absolute_residuals_within_tolerance']}` |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            _interpretation(payload),
            "",
            "## Scope Caveat",
            "",
            payload["answer_scope"],
            "",
            "## Governance Blocker",
            "",
            "Plan review reached round 5 without Claude `ACCEPT`. The final",
            "plan-review finding was patched, and the run is preserved as an",
            "inspection-only diagnostic. This is not a protocol-clean reviewed",
            "result and must not be promoted as downstream evidence unless the",
            "human explicitly authorizes that use or plan review is reopened and",
            "accepted.",
            "",
            "## Comparator",
            "",
            _comparator_markdown(payload),
            "",
            "## Verification Notes",
            "",
            "The runner validates parent and filterflow-subprocess CPU-only manifests,",
            "rung ledger shape, first-failure/first-blocked consistency, and JSON",
            "schema invariants during normal and `--validate-only` execution.",
            "",
            "## Non-Implications",
            "",
            *[f"- {item}" for item in payload["non_implications"]],
            "",
        ]
    )
    return "\n".join(lines)


def _canonical_rung_markdown(row: dict[str, Any]) -> list[str]:
    lines = [
        f"### {row['rung']}",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| status | `{row['status']}` |",
        f"| evidence bearing | `{row['evidence_bearing']}` |",
        f"| failure observed directly | `{row['failure_observed_directly']}` |",
        f"| blocker reason | `{row['blocker_reason']}` |",
        f"| inherited blocker | `{row['inherited_blocker']}` |",
        f"| first failing rung | `{row['first_failing_rung']}` |",
        f"| first blocked rung | `{row['first_blocked_rung']}` |",
        "",
        "Comparator fingerprint before:",
        "",
        _json_block(_compact_fingerprint(row["comparator_fingerprint_before"])),
        "",
        "Comparator fingerprint after:",
        "",
        _json_block(_compact_fingerprint(row["comparator_fingerprint_after"])),
        "",
        "Fixed variables:",
        "",
        _json_block(row["fixed_variables"]),
        "",
        "Varied variables:",
        "",
        _json_block(row["varied_variables"]),
        "",
        "Primary metrics:",
        "",
        _json_block(_compact_primary_metrics(row["primary_metrics"])),
        "",
        "Veto diagnostics:",
        "",
        _json_block(row["veto_diagnostics"]),
        "",
        "Explanatory diagnostics:",
        "",
        _json_block(row["explanatory_diagnostics"]),
        "",
    ]
    return lines


def _compact_fingerprint(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "head_commit": value.get("head_commit"),
        "symbolic_head": value.get("symbolic_head"),
        "branch_string_status": value.get("branch_string_status"),
        "branch_ref_exists": value.get("branch_ref_exists"),
        "status_short": value.get("status_short"),
        "diff_digest": value.get("diff_digest"),
        "python_version": value.get("python_version"),
        "package_manifest_digest": value.get("package_manifest_digest"),
    }


def _compact_primary_metrics(value: dict[str, Any]) -> dict[str, Any]:
    compact = dict(value)
    if "cell_summaries" in compact:
        compact["cell_summaries"] = [
            {
                "scenario": cell.get("scenario"),
                "convergence_threshold": cell.get("convergence_threshold"),
                "max_iterations": cell.get("max_iterations"),
                "cell_status": cell.get("cell_status"),
                "filterflow_command": cell.get("filterflow_command"),
                "primary_metrics": cell.get("primary_metrics"),
                "veto_diagnostics": cell.get("veto_diagnostics"),
                "explanatory_diagnostics": cell.get("explanatory_diagnostics"),
            }
            for cell in compact["cell_summaries"]
        ]
    return compact


def _json_block(value: Any) -> str:
    return "```json\n" + json.dumps(value, indent=2, sort_keys=True, default=str) + "\n```"


def _comparator_markdown(payload: dict[str, Any]) -> str:
    fingerprint = payload["filterflow_fingerprint_initial"]
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
        f"| exact filterflow command | `{_first_filterflow_command(payload)}` |",
        "",
        "Local diff/status:",
        "",
        "```text",
        str(fingerprint.get("status_short") or "clean"),
        "```",
    ]
    return "\n".join(lines)


def _first_filterflow_command(payload: dict[str, Any]) -> str:
    for row in payload["rung_ledger"]:
        for cell in row.get("primary_metrics", {}).get("cell_summaries", []):
            command = cell.get("filterflow_command")
            if command:
                return command
    return "recorded in JSON cell summaries"


def _interpretation(payload: dict[str, Any]) -> str:
    if payload["first_failing_rung"] == "R1_T4_residual_ladder":
        return (
            "The first evidence-bearing failure is R1: the fixed T=4 scalar "
            "fixture still violates the predeclared absolute transport "
            "residual tolerance after the bounded convergence/iteration sweep. "
            "R2-R8 are blocked and should not be interpreted."
        )
    if payload["first_failing_rung"] == "R2_1d_horizon_ladder":
        return (
            "Within the executed scalar-state prefix, R1 found a viable "
            "residual setting, but the R2 horizon extension produced the first "
            "direct evidence-bearing failure at T=32. This is not a claim that "
            "the full smoothness-alignment ladder has been exercised."
        )
    return (
        "No evidence-bearing failure was observed before the first blocked rung; "
        "blocked rungs are not claims about later smoothness behavior."
    )


def _digest_payload(payload: dict[str, Any]) -> str:
    clone = dict(payload)
    clone.pop("run_manifest", None)
    clone.pop("created_at_utc", None)
    clone.pop("reproducibility_digest", None)
    return stable_digest(clone)


if __name__ == "__main__":
    raise SystemExit(main())
