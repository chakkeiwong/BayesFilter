"""Float64 continuation debug against the executable filterflow reference.

This is a difference audit only. It compares BayesFilter TF/TFP experimental
paths to the local float64 filterflow reference branch and does not assert
correctness of either implementation.
"""

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

from experiments.dpf_implementation.tf_tfp.resampling import annealed_transport_tf
from experiments.dpf_implementation.tf_tfp.runners import (
    run_filterflow_1d_lgssm_horizon_ladder_tf as horizon,
)
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
from experiments.dpf_implementation.tf_tfp.runners import (
    run_filterflow_r3_proposal_trace_replay_tf as r3,
)
from experiments.dpf_implementation.tf_tfp.runners import (
    run_filterflow_r3_float64_trace_replay_tf as r3_float64,
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
from experiments.dpf_implementation.tf_tfp.runners.filterflow_reference_policy import (
    FILTERFLOW_BRANCH_MARKER,
    FILTERFLOW_REFERENCE_COMMIT,
    FILTERFLOW_REFERENCE_DTYPE,
    reference_policy,
    validate_filterflow_reference_status,
)


DTYPE = tf.float64
PLAN_PATH = "docs/plans/bayesfilter-dpf-filterflow-float64-continuation-debug-plan-2026-06-03.md"
RESULT_PATH = "docs/plans/bayesfilter-dpf-filterflow-float64-continuation-debug-result-2026-06-03.md"
JSON_PATH = OUTPUT_DIR / "dpf_filterflow_float64_continuation_debug_2026-06-03.json"
REPORT_PATH = REPORT_DIR / "dpf-filterflow-float64-continuation-debug-2026-06-03.md"
FILTERFLOW_PATH = REPO_ROOT / ".localsource" / "filterflow"
FILTERFLOW_MARKER_PATH = FILTERFLOW_PATH / FILTERFLOW_BRANCH_MARKER
RUNG_IDS = (
    "R0_controlled_1d_T2",
    "R1_controlled_1d_T4",
    "R2_filterflow_observation_path_T100",
    "R3_filterflow_initial_particles_T100",
    "R4_filterflow_2d_trace_replay",
)
TRACE_REPLAY_DTYPES = ("tf.float32",)
TRACE_REPLAY_FLOAT32_STATUS = (
    "The current reusable R3 trace/replay helper is still float32-specific; "
    "this runner records that as a remaining instrumentation gap rather than "
    "promoting old float32 evidence to the float64 reference lane."
)
TRACE_REPLAY_FLOAT64_STATUS = (
    "The continuation runner consumes the no-runtime-shim float64 R3 "
    "trace/replay helper as the evidence-bearing R4 rung."
)


@dataclass(frozen=True)
class Float64Scenario:
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
    reference_status = _filterflow_status()
    validate_filterflow_reference_status(reference_status, marker_path=FILTERFLOW_MARKER_PATH)
    initial_fingerprint = continuation._filterflow_fingerprint()
    fixture = _filterflow_float64_fixture_subprocess()
    rung_ledger: list[dict[str, Any]] = []
    first_failing: dict[str, Any] | None = None
    first_blocked: dict[str, Any] | None = None

    if fixture.get("status") != "executed":
        first_blocked = _cell_id("R2_filterflow_observation_path_T100", "fixture_blocked")
        rung_ledger = _blocked_tail(
            0,
            "blocked_by_filterflow_fixture_subprocess",
            first_failing=None,
            first_blocked=first_blocked,
            reference_status=reference_status,
        )
    else:
        scenarios = _scenario_sequence(fixture)
        for index, (rung, cell_id, scenario, changed_axis) in enumerate(scenarios):
            row = _run_scalar_rung(
                rung=rung,
                cell_id=cell_id,
                scenario=scenario,
                changed_axis=changed_axis,
                reference_status=reference_status,
            )
            rung_ledger.append(row)
            if row["status"] == "mismatch":
                first_failing = row["first_failing_cell"]
                if index + 1 < len(RUNG_IDS):
                    first_blocked = _cell_id(RUNG_IDS[index + 1], "blocked_after_prior_mismatch")
                    rung_ledger.extend(
                        _blocked_tail(
                            index + 1,
                            row["blocker_reason"],
                            first_failing=first_failing,
                            first_blocked=first_blocked,
                            reference_status=reference_status,
                        )
                    )
                break
            if row["status"] == "blocked":
                first_blocked = row["first_blocked_cell"]
                rung_ledger.extend(
                    _blocked_tail(
                        index + 1,
                        row["blocker_reason"],
                        first_failing=None,
                        first_blocked=first_blocked,
                        reference_status=reference_status,
                    )
                )
                break
        else:
            r4 = _run_r3_trace_rung(reference_status)
            rung_ledger.append(r4)
            if r4["status"] == "mismatch":
                first_failing = r4["first_failing_cell"]
            elif r4["status"] == "blocked":
                first_blocked = r4["first_blocked_cell"]

    final_fingerprint = continuation._filterflow_fingerprint()
    comparator_drift = continuation._fingerprints_drifted(initial_fingerprint, final_fingerprint)
    if comparator_drift:
        first_failing = None
        first_blocked = _cell_id("R0_controlled_1d_T2", "comparator_drift")
        rung_ledger = _blocked_tail(
            0,
            "blocked_by_comparator_drift",
            first_failing=None,
            first_blocked=first_blocked,
            reference_status=reference_status,
            comparator_before=initial_fingerprint,
            comparator_after=final_fingerprint,
        )

    if len(rung_ledger) != len(RUNG_IDS):
        rung_ledger.extend(
            _blocked_tail(
                len(rung_ledger),
                "blocked_after_prior_rung",
                first_failing=first_failing,
                first_blocked=first_blocked,
                reference_status=reference_status,
            )
        )
    first_failing_cell = first_failing or _none_cell()
    first_blocked_cell = first_blocked or _none_cell()
    decision = _decision(first_failing_cell, first_blocked_cell, rung_ledger, comparator_drift)
    return {
        "decision": decision,
        "created_at_utc": utc_now(),
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "report_path": str(REPORT_PATH.relative_to(REPO_ROOT)),
        "json_path": str(JSON_PATH.relative_to(REPO_ROOT)),
        "question": (
            "Where does BayesFilter first diverge from the local float64 "
            "filterflow executable reference in the bounded continuation ladder?"
        ),
        "evidence_contract": {
            "primary_comparator": "local float64 filterflow reference branch",
            "primary_question": "cross_implementation_difference_only",
            "primary_pass": "ledger_and_trigger_agreement_with_float64_reference",
            "mathematical_correctness": "not_concluded",
            "gradient_correctness": "not_concluded",
        },
        "reference_policy": reference_policy(),
        "filterflow_status": reference_status,
        "filterflow_fingerprint_initial": initial_fingerprint,
        "filterflow_fingerprint_final": final_fingerprint,
        "comparator_drift": comparator_drift,
        "first_failing_cell": first_failing_cell,
        "first_blocked_cell": first_blocked_cell,
        "rung_ledger": rung_ledger,
        "fixture_summary": agreement._fixture_summary(fixture),
        "tolerances": {
            **agreement._agreement_tolerances(),
            "trace_replay": r3_float64.TRACE_TOLERANCE,
        },
        "path_boundary_manifest": continuation._path_boundary_manifest(),
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners."
                "run_filterflow_float64_continuation_debug_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_implications": _non_implications(),
    }


def _scenario_sequence(
    fixture: dict[str, Any],
) -> list[tuple[str, str, Float64Scenario, str]]:
    controlled_t2 = _from_horizon_scenario(
        horizon.SCENARIOS[0],
        initial_particles=tuple(float(v) for v in step.INITIAL_PARTICLES),
    )
    controlled_t4 = _from_horizon_scenario(
        horizon.SCENARIOS[1],
        initial_particles=tuple(float(v) for v in step.INITIAL_PARTICLES),
    )
    control = localization._control_base_scenario()
    r1_base = localization._r1_base_scenario(fixture, control)
    r1 = Float64Scenario(
        scenario_id="T100_filterflow_observation_path_fixed_initial_particles_float64",
        observations=tuple(float(v) for v in r1_base.observations),
        initial_particles=tuple(float(v) for v in control.initial_particles),
        transition_noises=tuple(tuple(float(v) for v in row) for row in control.transition_noises),
    )
    r2 = Float64Scenario(
        scenario_id="T100_filterflow_observation_and_initial_particles_float64",
        observations=r1.observations,
        initial_particles=tuple(float(v) for v in fixture["initial_particles_scalar"]),
        transition_noises=r1.transition_noises,
    )
    return [
        (
            "R0_controlled_1d_T2",
            "controlled_T2_anchor",
            controlled_t2,
            "baseline controlled scalar-state T=2",
        ),
        (
            "R1_controlled_1d_T4",
            "controlled_T4_extension",
            controlled_t4,
            "controlled scalar-state horizon extended to T=4",
        ),
        (
            "R2_filterflow_observation_path_T100",
            "filterflow_observation_path_fixed_initial_particles",
            r1,
            "observations switched to executable float64 filterflow fixture path",
        ),
        (
            "R3_filterflow_initial_particles_T100",
            "filterflow_initial_particles_after_data_generation",
            r2,
            "initial particles switched to executable float64 filterflow fixture draw",
        ),
    ]


def _from_horizon_scenario(
    scenario: horizon.Scenario,
    *,
    initial_particles: tuple[float, ...],
) -> Float64Scenario:
    return Float64Scenario(
        scenario_id=scenario.scenario_id,
        observations=tuple(float(v) for v in scenario.observations),
        initial_particles=initial_particles,
        transition_noises=tuple(tuple(float(v) for v in row) for row in scenario.transition_noises),
    )


def _run_scalar_rung(
    *,
    rung: str,
    cell_id: str,
    scenario: Float64Scenario,
    changed_axis: str,
    reference_status: dict[str, Any],
) -> dict[str, Any]:
    bayesfilter = _bayesfilter_scalar_reference(scenario)
    filterflow = _filterflow_scalar_reference(scenario)
    comparison = _compare_scalar(bayesfilter, filterflow)
    status = _scalar_status(bayesfilter, filterflow, comparison)
    first_failing = None
    first_blocked = None
    blocker = None
    if status == "mismatch":
        first_failing = _cell_id(rung, cell_id)
        blocker = f"blocked_after_{rung}_mismatch"
    elif status == "blocked":
        first_blocked = _cell_id(rung, cell_id)
        blocker = filterflow.get("blocker") or comparison.get("blocker") or f"blocked_in_{rung}"
    return _rung_row(
        rung=rung,
        status=status,
        evidence_bearing=True,
        failure_observed_directly=status == "mismatch",
        blocker_reason=blocker,
        first_failing_cell=first_failing,
        first_blocked_cell=first_blocked,
        changed_axis=changed_axis,
        fixed_variables={
            "scenario_id": scenario.scenario_id,
            "horizon": scenario.horizon,
            "theta": step.THETA0,
            "Q": step.Q,
            "R": step.R,
            "num_particles": step.NUM_PARTICLES,
            "epsilon": step.EPSILON,
            "scaling": step.SCALING,
            "convergence_threshold": step.CONVERGENCE_THRESHOLD,
            "max_iterations": step.MAX_ITERATIONS,
            "dtype": FILTERFLOW_REFERENCE_DTYPE,
        },
        primary_metrics=_scalar_primary_metrics(bayesfilter, filterflow, comparison),
        explanatory_diagnostics={
            "gradient_promotion": "not_concluded",
            "bayesfilter_gradient_tape": bayesfilter.get("gradient_tape"),
            "filterflow_gradient_tape": filterflow.get("gradient_tape"),
            "bayesfilter_finite_difference_gradient": bayesfilter.get("finite_difference_gradient"),
            "filterflow_finite_difference_gradient": filterflow.get("finite_difference_gradient"),
            "first_failure": comparison.get("first_failure"),
        },
        cells=[
            {
                "cell_id": cell_id,
                "status": status,
                "scenario_id": scenario.scenario_id,
                "horizon": scenario.horizon,
                "comparison": comparison,
                "bayesfilter": _compact_reference(bayesfilter),
                "filterflow": _compact_reference(filterflow),
            }
        ],
        filterflow_cpu_only_manifest=filterflow.get("cpu_only_manifest"),
        reference_status=reference_status,
    )


def _bayesfilter_scalar_reference(scenario: Float64Scenario) -> dict[str, Any]:
    step_originals = _snapshot_step_globals()
    annealed_dtype = annealed_transport_tf.DTYPE
    try:
        step.DTYPE = DTYPE
        annealed_transport_tf.DTYPE = DTYPE
        _apply_step_scenario(scenario)
        return step._bayesfilter_reference()
    finally:
        _restore_step_globals(step_originals)
        annealed_transport_tf.DTYPE = annealed_dtype


def _filterflow_scalar_reference(scenario: Float64Scenario) -> dict[str, Any]:
    if not agreement.FILTERFLOW_ENV_PYTHON.exists():
        return {
            "status": "blocked",
            "blocker": f"missing filterflow env python: {agreement.FILTERFLOW_ENV_PYTHON}",
        }
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env["PYTHONPATH"] = str(FILTERFLOW_PATH)
    env["MPLCONFIGDIR"] = str(REPO_ROOT / ".cache" / "filterflow-mpl")
    completed = subprocess.run(
        [str(agreement.FILTERFLOW_ENV_PYTHON), "-c", _filterflow_scalar_script(scenario)],
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
            "blocker": "filterflow scalar float64 subprocess failed",
            "stdout_excerpt": completed.stdout[-4000:],
            "stderr_excerpt": completed.stderr[-4000:],
        }
    payload = agreement._extract_json_payload(
        completed.stdout,
        "FILTERFLOW_FLOAT64_SCALAR_JSON_BEGIN",
        "FILTERFLOW_FLOAT64_SCALAR_JSON_END",
        "filterflow float64 scalar JSON sentinels missing",
        completed.stderr,
    )
    payload["command"] = (
        f"CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=.localsource/filterflow "
        f"{agreement.FILTERFLOW_ENV_PYTHON} -c <float64 scalar continuation script>"
    )
    return payload


def _filterflow_scalar_script(scenario: Float64Scenario) -> str:
    return f"""
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/bayesfilter-dpf-mpl")

import experiments.dpf_implementation.tf_tfp.runners.run_filterflow_1d_lgssm_step_gradient_comparison_tf as step

step.DTYPE = step.tf.float64
step.HORIZON = {scenario.horizon}
step.INITIAL_PARTICLES = {list(scenario.initial_particles)!r}
step.OBSERVATIONS = {list(scenario.observations)!r}
step.TRANSITION_NOISES = {[list(row) for row in scenario.transition_noises]!r}
script = step._filterflow_script()
script = script.replace("DTYPE = tf.float32", "DTYPE = tf.float64")
script = script.replace("FILTERFLOW_1D_JSON_BEGIN", "FILTERFLOW_FLOAT64_SCALAR_JSON_BEGIN")
script = script.replace("FILTERFLOW_1D_JSON_END", "FILTERFLOW_FLOAT64_SCALAR_JSON_END")
namespace = {{}}
exec(script, namespace, namespace)
"""


def _filterflow_float64_fixture_subprocess() -> dict[str, Any]:
    if not agreement.FILTERFLOW_ENV_PYTHON.exists():
        return {
            "status": "blocked",
            "blocker": f"missing filterflow env python: {agreement.FILTERFLOW_ENV_PYTHON}",
        }
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env["PYTHONPATH"] = str(FILTERFLOW_PATH)
    env["MPLCONFIGDIR"] = str(REPO_ROOT / ".cache" / "filterflow-mpl")
    completed = subprocess.run(
        [str(agreement.FILTERFLOW_ENV_PYTHON), "-c", _filterflow_float64_fixture_script()],
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
            "blocker": "filterflow float64 fixture subprocess failed",
            "stdout_excerpt": completed.stdout[-4000:],
            "stderr_excerpt": completed.stderr[-4000:],
        }
    return agreement._extract_json_payload(
        completed.stdout,
        "FILTERFLOW_FLOAT64_FIXTURE_JSON_BEGIN",
        "FILTERFLOW_FLOAT64_FIXTURE_JSON_END",
        "filterflow float64 fixture JSON sentinels missing",
        completed.stderr,
    )


def _filterflow_float64_fixture_script() -> str:
    return textwrap.dedent(
        f"""
        import inspect
        import json
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES")

        if not hasattr(inspect, "getargspec"):
            inspect.getargspec = inspect.getfullargspec

        import numpy as np
        import tensorflow as tf

        from scripts.simple_linear_smoothness import get_data

        DATA_SEED = {agreement.DATA_SEED}
        T = 100
        N = {step.NUM_PARTICLES}
        NP_DTYPE = np.float64
        np_random_state = np.random.RandomState(seed=DATA_SEED)
        transition_matrix = (
            np.diag([1.0, 1.0]).astype(NP_DTYPE)
            + np.array([[0.0, 1.0], [0.0, 0.0]], dtype=NP_DTYPE)
        )
        observation_matrix = np.array([[1.0, 0.0]], dtype=NP_DTYPE)
        transition_covariance = np.array([[1.0 / 3.0, 0.5], [0.5, 1.0]], dtype=NP_DTYPE)
        observation_covariance = np.array([[0.01]], dtype=NP_DTYPE)
        observations_2d, _kf = get_data(
            transition_matrix,
            observation_matrix,
            transition_covariance,
            observation_covariance,
            T=T,
            random_state=np_random_state,
        )
        initial_particles_2d = np_random_state.normal(0.0, 0.01, [1, N, 2]).astype(NP_DTYPE)
        scalar_observations = observations_2d.reshape(-1).astype(NP_DTYPE)
        payload = {{
            "status": "executed",
            "fixture_contract": "filterflow_float64_observations_and_initial_particles_replayed_as_explicit_tensors",
            "data_seed": DATA_SEED,
            "filter_seed": {agreement.FILTER_SEED},
            "dtype": "float64",
            "observation_model_source": "scripts.simple_linear_smoothness.get_data",
            "initial_particle_source": "same NumPy RandomState after get_data",
            "observations_scalar": scalar_observations.astype(float).tolist(),
            "initial_particles_scalar": initial_particles_2d[0, :, 0].astype(float).tolist(),
            "initial_particles_2d": initial_particles_2d.astype(float).tolist(),
            "cpu_only_manifest": {{
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "pre_import_cuda_visible_devices": PRE_IMPORT_CUDA_VISIBLE_DEVICES,
                "gpu_devices_visible": [str(device) for device in tf.config.list_physical_devices("GPU")],
            }},
        }}
        print("FILTERFLOW_FLOAT64_FIXTURE_JSON_BEGIN")
        print(json.dumps(payload, sort_keys=True))
        print("FILTERFLOW_FLOAT64_FIXTURE_JSON_END")
        """
    )


def _run_r3_trace_rung(reference_status: dict[str, Any]) -> dict[str, Any]:
    source_audit = _r3_float64_source_audit()
    if not source_audit["float64_ready"]:
        blocker = "blocked_by_float32_specific_r3_trace_replay_helper"
        return _rung_row(
            rung="R4_filterflow_2d_trace_replay",
            status="blocked",
            evidence_bearing=False,
            failure_observed_directly=False,
            blocker_reason=blocker,
            first_failing_cell=None,
            first_blocked_cell=_cell_id("R4_filterflow_2d_trace_replay", "r3_trace_replay_float64_instrumentation_gap"),
            changed_axis="2D filterflow transition proposal trace replay",
            fixed_variables={
                "required_dtype": FILTERFLOW_REFERENCE_DTYPE,
                "existing_helper_dtype_tokens": list(TRACE_REPLAY_DTYPES),
            },
            primary_metrics={
                "implementation_agreement": False,
                "trace_executed": False,
                "float64_ready": False,
            },
            explanatory_diagnostics={
                "not_a_direct_mismatch": True,
                "blocker_detail": TRACE_REPLAY_FLOAT32_STATUS,
                "source_audit": source_audit,
                "required_next_control": "create a non-mutating float64 R3 trace/replay helper",
            },
            cells=[],
            filterflow_cpu_only_manifest=None,
            reference_status=reference_status,
        )
    trace = r3_float64._run()
    comparison = trace.get("computed_resampling_comparison") or trace.get("comparison", {})
    status = "pass" if comparison.get("implementation_agreement") else "mismatch"
    return _rung_row(
        rung="R4_filterflow_2d_trace_replay",
        status=status,
        evidence_bearing=True,
        failure_observed_directly=status == "mismatch",
        blocker_reason=None if status == "pass" else "blocked_after_R4_trace_mismatch",
        first_failing_cell=None if status == "pass" else _cell_id("R4_filterflow_2d_trace_replay", "trace_replay"),
        first_blocked_cell=None,
        changed_axis="2D filterflow transition proposal trace replay",
        fixed_variables={"required_dtype": FILTERFLOW_REFERENCE_DTYPE},
        primary_metrics={
            "implementation_agreement": comparison.get("implementation_agreement"),
            "decision": trace.get("decision"),
            "first_failure": comparison.get("first_failure"),
            "series_deltas": comparison.get("series_deltas"),
            "trace_contract": trace.get("filterflow_trace", {}).get("trace_contract"),
            "runtime_shims": trace.get("filterflow_trace", {}).get(
                "non_mutating_filterflow_runtime_shims",
            ),
        },
        explanatory_diagnostics={
            "source_audit": source_audit,
            "float64_trace_replay_status": TRACE_REPLAY_FLOAT64_STATUS,
            "traced_transport_matrix_comparison": trace.get("traced_transport_matrix_comparison"),
        },
        cells=[],
        filterflow_cpu_only_manifest=trace.get("filterflow_trace", {}).get("cpu_only_manifest"),
        reference_status=reference_status,
    )


def _r3_float64_source_audit() -> dict[str, Any]:
    source = (
        REPO_ROOT
        / "experiments/dpf_implementation/tf_tfp/runners/run_filterflow_r3_float64_trace_replay_tf.py"
    ).read_text(encoding="utf-8")
    float32_count = source.count("tf.float32") + source.count("np.float32")
    float64_ready = float32_count == 0 and "annealed_transport_tf.DTYPE = tf.float32" not in source
    return {
        "float64_ready": float64_ready,
        "float32_token_count": float32_count,
        "source_digest": stable_digest(source),
        "interpretation": (
            "R4 is not rerun as float64 while the trace/replay helper still "
            "hard-codes float32 arithmetic."
        )
        if not float64_ready
        else "Float64 R3 helper has no obvious float32 tokens.",
    }


def _compare_scalar(
    bayesfilter: dict[str, Any],
    filterflow: dict[str, Any],
) -> dict[str, Any]:
    comparison = agreement._compare_runs(bayesfilter, filterflow)
    if comparison.get("status") != "compared":
        return comparison
    detailed = localization._detailed_field_comparison(bayesfilter, filterflow)
    comparison["first_failure"] = detailed["first_failure"]
    comparison["per_time_field_deltas"] = detailed["per_time_field_deltas"]
    comparison["max_field_delta"] = detailed["max_field_delta"]
    return comparison


def _scalar_status(
    bayesfilter: dict[str, Any],
    filterflow: dict[str, Any],
    comparison: dict[str, Any],
) -> str:
    if filterflow.get("status") != "executed" or comparison.get("status") != "compared":
        return "blocked"
    if not bayesfilter.get("finite_scalar") or not filterflow.get("finite_scalar"):
        return "mismatch"
    return "pass" if comparison.get("implementation_agreement") else "mismatch"


def _scalar_primary_metrics(
    bayesfilter: dict[str, Any],
    filterflow: dict[str, Any],
    comparison: dict[str, Any],
) -> dict[str, Any]:
    if comparison.get("status") != "compared":
        return {
            "implementation_agreement": False,
            "scalar_delta": float("inf"),
            "max_field_delta": float("inf"),
        }
    residuals = comparison["absolute_residuals"]
    residual_deltas = comparison.get("residual_deltas", {})
    return {
        "implementation_agreement": bool(comparison.get("implementation_agreement")),
        "bayesfilter_scalar": bayesfilter.get("scalar"),
        "filterflow_scalar": filterflow.get("scalar"),
        "scalar_delta": comparison.get("scalar_delta"),
        "scalar_within_tolerance": comparison.get("scalar_within_tolerance"),
        "ledger_within_tolerance": comparison.get("ledger_within_tolerance"),
        "trigger_match": comparison.get("trigger_match"),
        "max_field_delta": comparison.get("max_field_delta"),
        "row_residual_delta": residual_deltas.get("row"),
        "column_residual_delta": residual_deltas.get("column"),
        "bayesfilter_max_row_residual": residuals.get("bayesfilter_max_row_residual"),
        "filterflow_max_row_residual": residuals.get("filterflow_max_row_residual"),
        "bayesfilter_max_column_residual": residuals.get("bayesfilter_max_column_residual"),
        "filterflow_max_column_residual": residuals.get("filterflow_max_column_residual"),
    }


def _compact_reference(reference: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": reference.get("status"),
        "backend": reference.get("backend"),
        "scalar": reference.get("scalar"),
        "gradient_tape": reference.get("gradient_tape"),
        "finite_difference_gradient": reference.get("finite_difference_gradient"),
        "finite_scalar": reference.get("finite_scalar"),
        "finite_gradient": reference.get("finite_gradient"),
        "ledger_length": len(reference.get("ledger", [])),
        "cpu_only_manifest": reference.get("cpu_only_manifest"),
    }


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


def _apply_step_scenario(scenario: Float64Scenario) -> None:
    step.HORIZON = scenario.horizon
    step.INITIAL_PARTICLES = list(scenario.initial_particles)
    step.OBSERVATIONS = list(scenario.observations)
    step.TRANSITION_NOISES = [list(row) for row in scenario.transition_noises]
    step.CONVERGENCE_THRESHOLD = step.CONVERGENCE_THRESHOLD
    step.MAX_ITERATIONS = step.MAX_ITERATIONS


def _restore_step_globals(values: dict[str, Any]) -> None:
    step.DTYPE = values["DTYPE"]
    step.HORIZON = values["HORIZON"]
    step.INITIAL_PARTICLES = values["INITIAL_PARTICLES"]
    step.OBSERVATIONS = values["OBSERVATIONS"]
    step.TRANSITION_NOISES = values["TRANSITION_NOISES"]
    step.CONVERGENCE_THRESHOLD = values["CONVERGENCE_THRESHOLD"]
    step.MAX_ITERATIONS = values["MAX_ITERATIONS"]


def _rung_row(
    *,
    rung: str,
    status: str,
    evidence_bearing: bool,
    failure_observed_directly: bool,
    blocker_reason: str | None,
    first_failing_cell: dict[str, Any] | None,
    first_blocked_cell: dict[str, Any] | None,
    changed_axis: str,
    fixed_variables: dict[str, Any],
    primary_metrics: dict[str, Any],
    explanatory_diagnostics: dict[str, Any],
    cells: list[dict[str, Any]],
    filterflow_cpu_only_manifest: dict[str, Any] | None,
    reference_status: dict[str, Any],
) -> dict[str, Any]:
    return {
        "rung": rung,
        "status": status,
        "evidence_bearing": evidence_bearing,
        "failure_observed_directly": failure_observed_directly,
        "blocker_reason": blocker_reason,
        "first_failing_cell": first_failing_cell,
        "first_blocked_cell": first_blocked_cell,
        "changed_axis": changed_axis,
        "fixed_variables": fixed_variables,
        "primary_metrics": primary_metrics,
        "explanatory_diagnostics": explanatory_diagnostics,
        "cells": cells,
        "parent_cpu_only_manifest": _parent_cpu_manifest(),
        "filterflow_cpu_only_manifest": filterflow_cpu_only_manifest,
        "reference_status": reference_status,
    }


def _blocked_tail(
    start_index: int,
    blocker: str | None,
    *,
    first_failing: dict[str, Any] | None,
    first_blocked: dict[str, Any] | None,
    reference_status: dict[str, Any],
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
                evidence_bearing=False,
                failure_observed_directly=False,
                blocker_reason=inherited,
                first_failing_cell=first_failing,
                first_blocked_cell=first_blocked,
                changed_axis="blocked_before_execution",
                fixed_variables={
                    "comparator_before": comparator_before,
                    "comparator_after": comparator_after,
                },
                primary_metrics={},
                explanatory_diagnostics={
                    "blocked_rung_interpretation": (
                        "This rung is not evidence-bearing; the first direct "
                        "mismatch or first blocker is reported separately."
                    )
                },
                cells=[],
                filterflow_cpu_only_manifest=None,
                reference_status=reference_status,
            )
        )
    return rows


def _cell_id(rung: str, cell_id: str) -> dict[str, Any]:
    return {"rung_id": rung, "cell_index": 0, "cell_id": cell_id}


def _none_cell() -> dict[str, Any]:
    return {"rung_id": "none_observed", "cell_index": -1, "cell_id": "none_observed"}


def _decision(
    first_failing: dict[str, Any],
    first_blocked: dict[str, Any],
    rows: list[dict[str, Any]],
    comparator_drift: bool,
) -> str:
    if comparator_drift:
        return "filterflow_float64_continuation_blocked_by_comparator_drift"
    if first_failing["rung_id"] != "none_observed":
        return "filterflow_float64_continuation_first_mismatch_detected"
    if first_blocked["rung_id"] != "none_observed":
        if all(row["status"] in {"pass", "blocked"} for row in rows):
            return "filterflow_float64_continuation_blocked_after_partial_agreement"
        return "filterflow_float64_continuation_blocked"
    return "filterflow_float64_continuation_no_mismatch_observed"


def _filterflow_status() -> dict[str, Any]:
    return {
        "path": str(FILTERFLOW_PATH),
        "branch": _git_filterflow(["rev-parse", "--abbrev-ref", "HEAD"]),
        "commit": _git_filterflow(["rev-parse", "HEAD"]),
        "status_short": _git_filterflow(["status", "--short"]),
        "status_branch": _git_filterflow(["status", "--short", "--branch"]),
        "dtype": FILTERFLOW_REFERENCE_DTYPE,
        "expected_commit": FILTERFLOW_REFERENCE_COMMIT,
        "marker_path": str(FILTERFLOW_MARKER_PATH),
    }


def _git_filterflow(args: list[str]) -> str:
    completed = subprocess.run(
        ["git", "-C", str(FILTERFLOW_PATH), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _parent_cpu_manifest() -> dict[str, Any]:
    return {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "pre_import_cuda_visible_devices": PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        "gpu_devices_visible": [str(device) for device in tf.config.list_physical_devices("GPU")],
    }


def _non_implications() -> list[str]:
    return [
        "No mathematical correctness claim for either implementation.",
        "No production readiness claim.",
        "No public API readiness claim.",
        "No posterior correctness claim.",
        "No gradient correctness claim.",
        "No general nonlinear-SSM validation claim.",
        "No DSGE/NAWM validation claim.",
        "No monograph claim.",
        "Fixed-target Sinkhorn remains a local BayesFilter diagnostic only.",
    ]


def _validate_payload(payload: dict[str, Any]) -> None:
    required = {
        "decision",
        "plan_path",
        "result_path",
        "filterflow_status",
        "first_failing_cell",
        "first_blocked_cell",
        "rung_ledger",
        "path_boundary_manifest",
        "run_manifest",
        "non_implications",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"missing payload keys: {sorted(missing)}")
    validate_filterflow_reference_status(payload["filterflow_status"], marker_path=FILTERFLOW_MARKER_PATH)
    allowed = {
        "filterflow_float64_continuation_blocked_by_comparator_drift",
        "filterflow_float64_continuation_first_mismatch_detected",
        "filterflow_float64_continuation_blocked_after_partial_agreement",
        "filterflow_float64_continuation_blocked",
        "filterflow_float64_continuation_no_mismatch_observed",
    }
    if payload["decision"] not in allowed:
        raise ValueError(f"unexpected decision: {payload['decision']}")
    if len(payload["rung_ledger"]) != len(RUNG_IDS):
        raise ValueError(f"expected {len(RUNG_IDS)} rungs")
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


def _validate_cpu(manifest: dict[str, Any], label: str) -> None:
    if manifest.get("pre_import_cuda_visible_devices") != "-1":
        raise ValueError(f"{label}: pre-import CUDA_VISIBLE_DEVICES is not -1")
    if manifest.get("cuda_visible_devices") != "-1":
        raise ValueError(f"{label}: CUDA_VISIBLE_DEVICES is not -1")
    if manifest.get("gpu_devices_visible") != []:
        raise ValueError(f"{label}: GPU devices visible {manifest.get('gpu_devices_visible')}")


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Filterflow Float64 Continuation Debug",
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
        "## Reference",
        "",
        "| Key | Value |",
        "| --- | --- |",
    ]
    for key, value in payload["reference_policy"].items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(
        [
            "",
            "## Rung Ledger",
            "",
            "| Rung | Status | Evidence-bearing | Direct failure | Scalar delta | Max field delta | Blocker |",
            "| --- | --- | --- | --- | ---: | ---: | --- |",
        ]
    )
    for row in payload["rung_ledger"]:
        metrics = row.get("primary_metrics", {})
        lines.append(
            f"| `{row['rung']}` | `{row['status']}` | "
            f"`{row['evidence_bearing']}` | `{row['failure_observed_directly']}` | "
            f"`{metrics.get('scalar_delta', 'N/A')}` | "
            f"`{metrics.get('max_field_delta', 'N/A')}` | "
            f"`{row['blocker_reason']}` |"
        )
    lines.extend(["", "## Evidence Details", ""])
    for row in payload["rung_ledger"]:
        lines.extend(
            [
                f"### {row['rung']}",
                "",
                f"Changed axis: {row['changed_axis']}",
                "",
                _json_block(
                    {
                        "primary_metrics": row["primary_metrics"],
                        "explanatory_diagnostics": row["explanatory_diagnostics"],
                    }
                ),
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
    first_fail = payload["first_failing_cell"]
    first_block = payload["first_blocked_cell"]
    if first_fail["rung_id"] != "none_observed":
        return (
            "A direct BayesFilter/filterflow difference was observed at "
            f"`{first_fail['rung_id']}` / `{first_fail['cell_id']}`."
        )
    if first_block["rung_id"] != "none_observed":
        return (
            "No direct difference was observed before the first blocker. The "
            f"next required control is `{first_block['cell_id']}`."
        )
    return "No direct difference or blocker was observed in the bounded ladder."


def _json_block(payload: dict[str, Any]) -> str:
    return "```json\n" + json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n```"


def _digest_payload(payload: dict[str, Any]) -> str:
    clone = dict(payload)
    clone.pop("run_manifest", None)
    clone.pop("created_at_utc", None)
    clone.pop("reproducibility_digest", None)
    return stable_digest(clone)


if __name__ == "__main__":
    raise SystemExit(main())
