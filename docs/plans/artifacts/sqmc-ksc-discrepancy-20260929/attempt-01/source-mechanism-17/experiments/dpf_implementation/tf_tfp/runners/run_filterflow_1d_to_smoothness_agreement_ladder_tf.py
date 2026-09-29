"""1D-to-smoothness BayesFilter/filterflow agreement ladder.

The promotion question is cross-implementation agreement against the local
executable filterflow checkout. Shared transport-quality oddities are recorded
as diagnostics, not treated as evidence that one implementation differs from
the other.
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

from experiments.dpf_implementation.tf_tfp.runners import (
    run_filterflow_1d_lgssm_horizon_ladder_tf as horizon,
)
from experiments.dpf_implementation.tf_tfp.runners import (
    run_filterflow_1d_lgssm_step_gradient_comparison_tf as step,
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


DTYPE = tf.float64
PLAN_PATH = "docs/plans/bayesfilter-dpf-1d-to-smoothness-filterflow-agreement-ladder-plan-2026-06-02.md"
RESULT_PATH = "docs/plans/bayesfilter-dpf-1d-to-smoothness-filterflow-agreement-ladder-result-2026-06-02.md"
REVIEW_LOOP_PATH = "docs/plans/bayesfilter-dpf-1d-to-smoothness-filterflow-agreement-ladder-review-loop-2026-06-02.md"
JSON_PATH = OUTPUT_DIR / "dpf_filterflow_1d_to_smoothness_agreement_ladder_2026-06-02.json"
REPORT_PATH = REPORT_DIR / "dpf-filterflow-1d-to-smoothness-agreement-ladder-2026-06-02.md"
FILTERFLOW_ENV_PYTHON = REPO_ROOT / ".localenv" / "filterflow-py311" / "bin" / "python"
FILTERFLOW_PATH = REPO_ROOT / ".localsource" / "filterflow"

THETA0 = step.THETA0
Q = step.Q
R = step.R
NUM_PARTICLES = step.NUM_PARTICLES
EPSILON = step.EPSILON
SCALING = step.SCALING
CONVERGENCE_THRESHOLD = step.CONVERGENCE_THRESHOLD
MAX_ITERATIONS = step.MAX_ITERATIONS
DATA_SEED = 123
FILTER_SEED = 1234
OBSERVATION_SCALE = 0.1

RUNG_IDS = (
    "R1_1d_T100_filterflow_observation_path",
    "R2_1d_initial_particle_generation",
    "R3_1d_transition_random_stream",
    "R4_resampling_policy_trigger_semantics",
    "R5_scalar_contract",
    "R6_theta_grid_surface",
    "R7_2d_constant_velocity_state",
    "R8_gradient_smoothness_surface",
)


@dataclass(frozen=True)
class AgreementScenario:
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
    fixture = _filterflow_fixture_subprocess()
    rung_ledger: list[dict[str, Any]] = []
    first_failing: dict[str, Any] | None = None
    first_blocked: dict[str, Any] | None = None

    if fixture.get("status") != "executed":
        first_blocked = _first_cell("R1_1d_T100_filterflow_observation_path", "fixture_blocked")
        rung_ledger = [
            _blocked_rung(
                rung,
                blocker="blocked_by_filterflow_fixture_subprocess",
                first_failing=None,
                first_blocked=first_blocked,
            )
            for rung in RUNG_IDS
        ]
    else:
        r1 = _run_r1(initial_fingerprint, fixture)
        rung_ledger.append(r1)
        if r1["status"] == "mismatch":
            first_failing = r1["first_failing_cell"]
            first_blocked = _first_cell("R2_1d_initial_particle_generation", "blocked_after_R1")
            _append_blocked_after(rung_ledger, start_index=1, blocker=r1["blocker_reason"], first_failing=first_failing, first_blocked=first_blocked)
        elif r1["status"] == "blocked":
            first_blocked = r1["first_blocked_cell"]
            _append_blocked_after(rung_ledger, start_index=1, blocker=r1["blocker_reason"], first_failing=None, first_blocked=first_blocked)
        else:
            r2 = _run_r2(initial_fingerprint, fixture)
            rung_ledger.append(r2)
            if r2["status"] == "mismatch":
                first_failing = r2["first_failing_cell"]
                first_blocked = _first_cell("R3_1d_transition_random_stream", "blocked_after_R2")
                _append_blocked_after(rung_ledger, start_index=2, blocker=r2["blocker_reason"], first_failing=first_failing, first_blocked=first_blocked)
            elif r2["status"] == "blocked":
                first_blocked = r2["first_blocked_cell"]
                _append_blocked_after(rung_ledger, start_index=2, blocker=r2["blocker_reason"], first_failing=None, first_blocked=first_blocked)
            else:
                r3 = _run_r3_structured_blocker(initial_fingerprint)
                rung_ledger.append(r3)
                first_blocked = r3["first_blocked_cell"]
                _append_blocked_after(rung_ledger, start_index=3, blocker=r3["blocker_reason"], first_failing=None, first_blocked=first_blocked)

    final_fingerprint = continuation._filterflow_fingerprint()
    if continuation._fingerprints_drifted(initial_fingerprint, final_fingerprint):
        first_failing = None
        first_blocked = _first_cell("R1_1d_T100_filterflow_observation_path", "comparator_drift")
        rung_ledger = [
            _blocked_rung(
                rung,
                blocker="blocked_by_comparator_drift",
                first_failing=None,
                first_blocked=first_blocked,
                comparator_before=initial_fingerprint,
                comparator_after=final_fingerprint,
            )
            for rung in RUNG_IDS
        ]

    first_failing_field = first_failing or _none_cell()
    first_blocked_field = first_blocked or _none_cell()
    decision = _decision(first_failing_field, first_blocked_field, rung_ledger)
    return {
        "decision": decision,
        "created_at_utc": utc_now(),
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "review_loop_path": REVIEW_LOOP_PATH,
        "report_path": str(REPORT_PATH.relative_to(REPO_ROOT)),
        "json_path": str(JSON_PATH.relative_to(REPO_ROOT)),
        "question": (
            "At which one-axis transition, if any, does BayesFilter first "
            "diverge from the local executable filterflow reference?"
        ),
        "evidence_contract": {
            "primary_comparator": "local executable patched filterflow checkout",
            "primary_pass": "cross_implementation_agreement",
            "shared_residual_magnitude": "diagnostic_only",
            "mathematical_correctness": "not_concluded",
        },
        "plan_review_status": "ACCEPT_after_round_3_seed_correction",
        "first_failing_cell": first_failing_field,
        "first_blocked_cell": first_blocked_field,
        "rung_ledger": rung_ledger,
        "fixture": _fixture_summary(fixture),
        "filterflow_fingerprint_initial": initial_fingerprint,
        "filterflow_fingerprint_final": final_fingerprint,
        "path_boundary_manifest": continuation._path_boundary_manifest(),
        "tolerances": _agreement_tolerances(),
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners."
                "run_filterflow_1d_to_smoothness_agreement_ladder_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_implications": step._non_implications()
        + [
            "No correctness claim is made for either implementation.",
            "No smoothness-surface gradient correctness is concluded.",
            "Shared residual magnitude is not a BayesFilter/filterflow discrepancy.",
        ],
    }


def _run_r1(initial_fingerprint: dict[str, Any], fixture: dict[str, Any]) -> dict[str, Any]:
    transition_scenario = continuation._generated_scenario(100)
    scenario = AgreementScenario(
        scenario_id="T100_filterflow_observation_path_fixed_initial_particles",
        observations=tuple(float(v) for v in fixture["observations_scalar"]),
        initial_particles=tuple(float(v) for v in step.INITIAL_PARTICLES),
        transition_noises=tuple(
            tuple(float(v) for v in row) for row in transition_scenario.transition_noises
        ),
    )
    cell = _run_agreement_cell(scenario, cell_id="T100_filterflow_observation_path")
    return _evidence_rung(
        rung="R1_1d_T100_filterflow_observation_path",
        cell=cell,
        comparator_before=initial_fingerprint,
        fixed_variables={
            "theta": THETA0,
            "Q": Q,
            "R": R,
            "num_particles": NUM_PARTICLES,
            "initial_particles": list(scenario.initial_particles),
            "transition_noises_source": "existing_controlled_generated_T100_replay_ledger",
            "epsilon": EPSILON,
            "scaling": SCALING,
            "convergence_threshold": CONVERGENCE_THRESHOLD,
            "max_iterations": MAX_ITERATIONS,
            "data_seed": DATA_SEED,
        },
        changed_axis="observations switched to local executable filterflow-generated T=100 path",
    )


def _run_r2(initial_fingerprint: dict[str, Any], fixture: dict[str, Any]) -> dict[str, Any]:
    transition_scenario = continuation._generated_scenario(100)
    scenario = AgreementScenario(
        scenario_id="T100_filterflow_observation_and_initial_particles",
        observations=tuple(float(v) for v in fixture["observations_scalar"]),
        initial_particles=tuple(float(v) for v in fixture["initial_particles_scalar"]),
        transition_noises=tuple(
            tuple(float(v) for v in row) for row in transition_scenario.transition_noises
        ),
    )
    cell = _run_agreement_cell(
        scenario,
        cell_id="filterflow_initial_particles_data_seed_123_after_data_generation",
    )
    return _evidence_rung(
        rung="R2_1d_initial_particle_generation",
        cell=cell,
        comparator_before=initial_fingerprint,
        fixed_variables={
            "inherited_from_R1": True,
            "initial_particles": list(scenario.initial_particles),
            "initial_particle_source": (
                "filterflow smoothness NumPy data_seed=123 state after data generation, "
                "dimension-collapsed to scalar first coordinate"
            ),
            "filter_seed": FILTER_SEED,
            "filter_seed_role": "not used for initial particles; reserved for SMC seed splitting",
        },
        changed_axis="initial particles switched to executable filterflow draw",
    )


def _run_r3_structured_blocker(initial_fingerprint: dict[str, Any]) -> dict[str, Any]:
    blocker = (
        "blocked_by_uninstrumented_filterflow_transition_proposal_stream: "
        "filterflow SMC._return splits a TensorFlow seed inside a tf.while_loop "
        "and proposal samples are not serialized by the public script path; "
        "capturing exact per-time proposal random draws would require source "
        "instrumentation or a separately reviewed tracing wrapper."
    )
    return _blocked_rung(
        "R3_1d_transition_random_stream",
        blocker=blocker,
        first_failing=None,
        first_blocked=_first_cell("R3_1d_transition_random_stream", "filterflow_transition_stream_seed_1234"),
        comparator_before=initial_fingerprint,
        explanatory={
            "source_evidence": [
                ".localsource/filterflow/filterflow/smc.py:132-136 splits seed inside update loop",
                ".localsource/filterflow/filterflow/proposal/bootstrap.py and optimal_proposal.py sample inside propose",
                ".localsource/filterflow/scripts/simple_linear_smoothness.py exposes final surfaces, not per-time proposal draws",
            ],
            "not_a_bayesfilter_filterflow_mismatch": True,
            "required_next_control": "reviewed non-mutating trace wrapper or explicit human authorization to instrument local filterflow reference",
        },
    )


def _run_agreement_cell(scenario: AgreementScenario, *, cell_id: str) -> dict[str, Any]:
    bayesfilter = _bayesfilter_reference(scenario)
    filterflow = _filterflow_reference_subprocess(scenario)
    comparison = _compare_runs(bayesfilter, filterflow)
    status = _cell_status(bayesfilter, filterflow, comparison)
    return {
        "cell_index": 0,
        "cell_id": cell_id,
        "status": status,
        "scenario_id": scenario.scenario_id,
        "horizon": scenario.horizon,
        "bayesfilter_status": bayesfilter.get("status"),
        "filterflow_status": filterflow.get("status"),
        "comparison_status": comparison.get("status"),
        "filterflow_blocker": filterflow.get("blocker"),
        "primary_metrics": _primary_metrics(bayesfilter, filterflow, comparison),
        "veto_diagnostics": _veto_diagnostics(bayesfilter, filterflow, comparison),
        "shared_quality_diagnostic": _shared_quality_diagnostic(comparison),
        "gradient_diagnostics": _gradient_diagnostics(bayesfilter, filterflow, comparison),
        "filterflow_cpu_only_manifest": filterflow.get("cpu_only_manifest"),
    }


def _bayesfilter_reference(scenario: AgreementScenario) -> dict[str, Any]:
    originals = _snapshot_step_globals()
    try:
        step.HORIZON = scenario.horizon
        step.INITIAL_PARTICLES = list(scenario.initial_particles)
        step.OBSERVATIONS = list(scenario.observations)
        step.TRANSITION_NOISES = [list(row) for row in scenario.transition_noises]
        return step._bayesfilter_reference()
    finally:
        _restore_step_globals(originals)


def _filterflow_reference_subprocess(scenario: AgreementScenario) -> dict[str, Any]:
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
        [str(FILTERFLOW_ENV_PYTHON), "-c", _filterflow_reference_script(scenario)],
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
            "blocker": "filterflow agreement subprocess failed",
            "stdout_excerpt": completed.stdout[-4000:],
            "stderr_excerpt": completed.stderr[-4000:],
        }
    return _extract_json_payload(
        completed.stdout,
        "FILTERFLOW_AGREEMENT_CELL_JSON_BEGIN",
        "FILTERFLOW_AGREEMENT_CELL_JSON_END",
        "filterflow agreement cell JSON sentinels missing",
        completed.stderr,
    )


def _filterflow_reference_script(scenario: AgreementScenario) -> str:
    return textwrap.dedent(
        f"""
        import json
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/bayesfilter-dpf-mpl")

        import experiments.dpf_implementation.tf_tfp.runners.run_filterflow_1d_lgssm_step_gradient_comparison_tf as step

        step.HORIZON = {scenario.horizon}
        step.INITIAL_PARTICLES = {list(scenario.initial_particles)!r}
        step.OBSERVATIONS = {list(scenario.observations)!r}
        step.TRANSITION_NOISES = {[list(row) for row in scenario.transition_noises]!r}
        step.CONVERGENCE_THRESHOLD = {CONVERGENCE_THRESHOLD!r}
        step.MAX_ITERATIONS = {MAX_ITERATIONS}

        payload = step._run_filterflow_subprocess()
        print("FILTERFLOW_AGREEMENT_CELL_JSON_BEGIN")
        print(json.dumps(payload, sort_keys=True))
        print("FILTERFLOW_AGREEMENT_CELL_JSON_END")
        """
    )


def _filterflow_fixture_subprocess() -> dict[str, Any]:
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
        [str(FILTERFLOW_ENV_PYTHON), "-c", _filterflow_fixture_script()],
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
            "blocker": "filterflow fixture subprocess failed",
            "stdout_excerpt": completed.stdout[-4000:],
            "stderr_excerpt": completed.stderr[-4000:],
        }
    return _extract_json_payload(
        completed.stdout,
        "FILTERFLOW_FIXTURE_JSON_BEGIN",
        "FILTERFLOW_FIXTURE_JSON_END",
        "filterflow fixture JSON sentinels missing",
        completed.stderr,
    )


def _filterflow_fixture_script() -> str:
    return textwrap.dedent(
        f"""
        import json
        import inspect
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES")

        if not hasattr(inspect, "getargspec"):
            inspect.getargspec = inspect.getfullargspec

        import numpy as np
        import tensorflow as tf

        from scripts.simple_linear_smoothness import get_data

        DATA_SEED = {DATA_SEED}
        T = 100
        N = {NUM_PARTICLES}
        np_random_state = np.random.RandomState(seed=DATA_SEED)
        transition_matrix = np.diag([1.0, 1.0]).astype(np.float32) + np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float32)
        observation_matrix = np.array([[1.0, 0.0]], dtype=np.float32)
        transition_covariance = np.array([[1.0 / 3.0, 0.5], [0.5, 1.0]], dtype=np.float32)
        observation_covariance = np.array([[0.01]], dtype=np.float32)
        observations_2d, _kf = get_data(
            transition_matrix,
            observation_matrix,
            transition_covariance,
            observation_covariance,
            T=T,
            random_state=np_random_state,
        )
        initial_particles_2d = np_random_state.normal(0.0, 0.01, [1, N, 2]).astype(np.float32)
        scalar_observations = observations_2d.reshape(-1).astype(np.float32)
        payload = {{
            "status": "executed",
            "fixture_contract": "filterflow_observations_and_initial_particles_replayed_as_explicit_tensors",
            "data_seed": DATA_SEED,
            "filter_seed": {FILTER_SEED},
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
        print("FILTERFLOW_FIXTURE_JSON_BEGIN")
        print(json.dumps(payload, sort_keys=True))
        print("FILTERFLOW_FIXTURE_JSON_END")
        """
    )


def _extract_json_payload(
    stdout: str,
    begin: str,
    end: str,
    blocker: str,
    stderr: str,
) -> dict[str, Any]:
    start = stdout.rfind(begin)
    finish = stdout.rfind(end)
    if start < 0 or finish < 0 or finish <= start:
        return {
            "status": "blocked",
            "blocker": blocker,
            "stdout_excerpt": stdout[-4000:],
            "stderr_excerpt": stderr[-4000:],
        }
    raw = stdout[start + len(begin):finish].strip()
    payload = json.loads(raw)
    payload["stderr_excerpt"] = stderr[-2000:]
    return payload


def _compare_runs(bayesfilter: dict[str, Any], filterflow: dict[str, Any]) -> dict[str, Any]:
    comparison = horizon._compare_runs(bayesfilter, filterflow)
    if comparison.get("status") != "compared":
        return comparison
    residuals = comparison["absolute_residuals"]
    comparison["residual_deltas"] = {
        "row": abs(
            residuals["bayesfilter_max_row_residual"]
            - residuals["filterflow_max_row_residual"]
        ),
        "column": abs(
            residuals["bayesfilter_max_column_residual"]
            - residuals["filterflow_max_column_residual"]
        ),
    }
    comparison["implementation_agreement"] = bool(
        comparison["trigger_match"]
        and comparison["ledger_within_tolerance"]
        and comparison["scalar_within_tolerance"]
        and comparison["residual_deltas"]["row"] <= _agreement_tolerances()["row_residual_delta"]
        and comparison["residual_deltas"]["column"] <= _agreement_tolerances()["column_residual_delta"]
    )
    return comparison


def _cell_status(
    bayesfilter: dict[str, Any],
    filterflow: dict[str, Any],
    comparison: dict[str, Any],
) -> str:
    if filterflow.get("status") != "executed" or comparison.get("status") != "compared":
        return "blocked"
    if not bayesfilter.get("finite_scalar") or not filterflow.get("finite_scalar"):
        return "mismatch"
    return "pass" if comparison.get("implementation_agreement") else "mismatch"


def _primary_metrics(
    bayesfilter: dict[str, Any],
    filterflow: dict[str, Any],
    comparison: dict[str, Any],
) -> dict[str, Any]:
    if comparison.get("status") != "compared":
        return {
            "scalar_delta": float("inf"),
            "implementation_agreement": False,
        }
    residuals = comparison["absolute_residuals"]
    return {
        "bayesfilter_scalar": bayesfilter.get("scalar"),
        "filterflow_scalar": filterflow.get("scalar"),
        "scalar_delta": comparison["scalar_delta"],
        "trigger_match": comparison["trigger_match"],
        "ledger_within_tolerance": comparison["ledger_within_tolerance"],
        "scalar_within_tolerance": comparison["scalar_within_tolerance"],
        "implementation_agreement": comparison["implementation_agreement"],
        "bayesfilter_max_row_residual": residuals["bayesfilter_max_row_residual"],
        "filterflow_max_row_residual": residuals["filterflow_max_row_residual"],
        "bayesfilter_max_column_residual": residuals["bayesfilter_max_column_residual"],
        "filterflow_max_column_residual": residuals["filterflow_max_column_residual"],
        "row_residual_delta": comparison["residual_deltas"]["row"],
        "column_residual_delta": comparison["residual_deltas"]["column"],
    }


def _veto_diagnostics(
    bayesfilter: dict[str, Any],
    filterflow: dict[str, Any],
    comparison: dict[str, Any],
) -> dict[str, Any]:
    if comparison.get("status") != "compared":
        return {
            "filterflow_blocker": filterflow.get("blocker"),
            "comparison_status": comparison.get("status"),
        }
    return {
        "filterflow_blocker": filterflow.get("blocker"),
        "finite_bayesfilter_scalar": bayesfilter.get("finite_scalar"),
        "finite_filterflow_scalar": filterflow.get("finite_scalar"),
        "trigger_match": comparison["trigger_match"],
        "ledger_within_tolerance": comparison["ledger_within_tolerance"],
        "scalar_within_tolerance": comparison["scalar_within_tolerance"],
        "row_residual_delta_within_tolerance": (
            comparison["residual_deltas"]["row"]
            <= _agreement_tolerances()["row_residual_delta"]
        ),
        "column_residual_delta_within_tolerance": (
            comparison["residual_deltas"]["column"]
            <= _agreement_tolerances()["column_residual_delta"]
        ),
        "absolute_residuals_within_quality_gate": comparison[
            "absolute_residuals_within_tolerance"
        ],
    }


def _shared_quality_diagnostic(comparison: dict[str, Any]) -> dict[str, Any]:
    if comparison.get("status") != "compared":
        return {"status": "not_available"}
    shared_fail = not comparison["absolute_residuals_within_tolerance"]
    agreement = bool(comparison.get("implementation_agreement"))
    if shared_fail and not agreement:
        status = "not_shared_residual_disagreement"
        interpretation = "residual_delta_part_of_cross_implementation_mismatch"
    elif shared_fail:
        status = "shared_quality_diagnostic"
        interpretation = "diagnostic_only_not_cross_implementation_failure"
    else:
        status = "diagnostic_pass"
        interpretation = "diagnostic_pass"
    return {
        "status": status,
        "absolute_residual_gate": step.TOLERANCES["row_residual"],
        "absolute_residuals_within_gate": comparison[
            "absolute_residuals_within_tolerance"
        ],
        "interpretation": interpretation,
    }


def _gradient_diagnostics(
    bayesfilter: dict[str, Any],
    filterflow: dict[str, Any],
    comparison: dict[str, Any],
) -> dict[str, Any]:
    if comparison.get("status") != "compared":
        return {"gradient_promotion": "not_concluded"}
    return {
        "bayesfilter_gradient_tape": bayesfilter.get("gradient_tape"),
        "filterflow_gradient_tape": filterflow.get("gradient_tape"),
        "bayesfilter_finite_difference_gradient": bayesfilter.get(
            "finite_difference_gradient"
        ),
        "filterflow_finite_difference_gradient": filterflow.get(
            "finite_difference_gradient"
        ),
        "gradient_delta": comparison.get("gradient_delta"),
        "finite_difference_delta": comparison.get("finite_difference_delta"),
        "scalar_gradient_contract_reconciled": False,
        "gradient_promotion": "not_concluded",
    }


def _evidence_rung(
    *,
    rung: str,
    cell: dict[str, Any],
    comparator_before: dict[str, Any],
    fixed_variables: dict[str, Any],
    changed_axis: str,
) -> dict[str, Any]:
    status = cell["status"]
    first_failing = None
    first_blocked = None
    blocker_reason = None
    if status == "mismatch":
        first_failing = _first_cell(rung, cell["cell_id"], cell["cell_index"])
        blocker_reason = f"blocked_after_{rung}_mismatch"
    elif status == "blocked":
        first_blocked = _first_cell(rung, cell["cell_id"], cell["cell_index"])
        blocker_reason = cell.get("filterflow_blocker") or f"blocked_in_{rung}"
    return _rung_row(
        rung=rung,
        status=status,
        evidence_bearing=True,
        failure_observed_directly=status == "mismatch",
        blocker_reason=blocker_reason,
        inherited_blocker=None,
        first_failing_cell=first_failing,
        first_blocked_cell=first_blocked,
        comparator_before=comparator_before,
        comparator_after=continuation._filterflow_fingerprint(),
        fixed_variables=fixed_variables,
        changed_axis=changed_axis,
        cells=[cell],
        primary_metrics=cell["primary_metrics"],
        veto_diagnostics=cell["veto_diagnostics"],
        explanatory_diagnostics={
            "shared_quality_diagnostic": cell["shared_quality_diagnostic"],
            "gradient_diagnostics": cell["gradient_diagnostics"],
        },
        filterflow_cpu_only_manifest=cell["filterflow_cpu_only_manifest"],
    )


def _blocked_rung(
    rung: str,
    *,
    blocker: str,
    first_failing: dict[str, Any] | None,
    first_blocked: dict[str, Any] | None,
    comparator_before: dict[str, Any] | None = None,
    comparator_after: dict[str, Any] | None = None,
    explanatory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fingerprint = comparator_before or continuation._filterflow_fingerprint()
    return _rung_row(
        rung=rung,
        status="blocked",
        evidence_bearing=False,
        failure_observed_directly=False,
        blocker_reason=blocker,
        inherited_blocker=blocker,
        first_failing_cell=first_failing,
        first_blocked_cell=first_blocked,
        comparator_before=fingerprint,
        comparator_after=comparator_after or fingerprint,
        fixed_variables={},
        changed_axis="blocked_before_execution",
        cells=[],
        primary_metrics={},
        veto_diagnostics={"blocked_before_execution": True},
        explanatory_diagnostics=explanatory or {
            "blocked_rung_interpretation": (
                "This rung is not evidence-bearing; the first failing and "
                "first blocked cells are reported separately."
            )
        },
        filterflow_cpu_only_manifest=None,
    )


def _append_blocked_after(
    rows: list[dict[str, Any]],
    *,
    start_index: int,
    blocker: str | None,
    first_failing: dict[str, Any] | None,
    first_blocked: dict[str, Any] | None,
) -> None:
    inherited = blocker or "blocked_after_prior_rung"
    for rung in RUNG_IDS[start_index:]:
        rows.append(
            _blocked_rung(
                rung,
                blocker=inherited,
                first_failing=first_failing,
                first_blocked=first_blocked,
            )
        )


def _rung_row(
    *,
    rung: str,
    status: str,
    evidence_bearing: bool,
    failure_observed_directly: bool,
    blocker_reason: str | None,
    inherited_blocker: str | None,
    first_failing_cell: dict[str, Any] | None,
    first_blocked_cell: dict[str, Any] | None,
    comparator_before: dict[str, Any],
    comparator_after: dict[str, Any],
    fixed_variables: dict[str, Any],
    changed_axis: str,
    cells: list[dict[str, Any]],
    primary_metrics: dict[str, Any],
    veto_diagnostics: dict[str, Any],
    explanatory_diagnostics: dict[str, Any],
    filterflow_cpu_only_manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "rung": rung,
        "status": status,
        "evidence_bearing": evidence_bearing,
        "failure_observed_directly": failure_observed_directly,
        "blocker_reason": blocker_reason,
        "inherited_blocker": inherited_blocker,
        "first_failing_cell": first_failing_cell,
        "first_blocked_cell": first_blocked_cell,
        "comparator_fingerprint_before": comparator_before,
        "comparator_fingerprint_after": comparator_after,
        "fixed_variables": fixed_variables,
        "changed_axis": changed_axis,
        "cells": cells,
        "primary_metrics": primary_metrics,
        "veto_diagnostics": veto_diagnostics,
        "explanatory_diagnostics": explanatory_diagnostics,
        "parent_cpu_only_manifest": _parent_cpu_manifest(),
        "filterflow_cpu_only_manifest": filterflow_cpu_only_manifest,
    }


def _first_cell(rung: str, cell_id: str, cell_index: int = 0) -> dict[str, Any]:
    return {"rung_id": rung, "cell_index": cell_index, "cell_id": cell_id}


def _none_cell() -> dict[str, Any]:
    return {"rung_id": "none_observed", "cell_index": -1, "cell_id": "none_observed"}


def _snapshot_step_globals() -> dict[str, Any]:
    return {
        "HORIZON": step.HORIZON,
        "INITIAL_PARTICLES": step.INITIAL_PARTICLES,
        "OBSERVATIONS": step.OBSERVATIONS,
        "TRANSITION_NOISES": step.TRANSITION_NOISES,
        "CONVERGENCE_THRESHOLD": step.CONVERGENCE_THRESHOLD,
        "MAX_ITERATIONS": step.MAX_ITERATIONS,
    }


def _restore_step_globals(values: dict[str, Any]) -> None:
    step.HORIZON = values["HORIZON"]
    step.INITIAL_PARTICLES = values["INITIAL_PARTICLES"]
    step.OBSERVATIONS = values["OBSERVATIONS"]
    step.TRANSITION_NOISES = values["TRANSITION_NOISES"]
    step.CONVERGENCE_THRESHOLD = values["CONVERGENCE_THRESHOLD"]
    step.MAX_ITERATIONS = values["MAX_ITERATIONS"]


def _parent_cpu_manifest() -> dict[str, Any]:
    return {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "pre_import_cuda_visible_devices": PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        "gpu_devices_visible": [str(device) for device in tf.config.list_physical_devices("GPU")],
    }


def _agreement_tolerances() -> dict[str, float]:
    return {
        **step.TOLERANCES,
        "row_residual_delta": 5e-5,
        "column_residual_delta": 5e-5,
    }


def _fixture_summary(fixture: dict[str, Any]) -> dict[str, Any]:
    if fixture.get("status") != "executed":
        return fixture
    return {
        "status": fixture["status"],
        "fixture_contract": fixture["fixture_contract"],
        "data_seed": fixture["data_seed"],
        "filter_seed": fixture["filter_seed"],
        "observation_count": len(fixture["observations_scalar"]),
        "initial_particles_scalar": fixture["initial_particles_scalar"],
        "transition_noises_source": "not_from_fixture; runner uses existing controlled generated_T100 ledger",
        "cpu_only_manifest": fixture["cpu_only_manifest"],
    }


def _decision(
    first_failing: dict[str, Any],
    first_blocked: dict[str, Any],
    rows: list[dict[str, Any]],
) -> str:
    if any(row["blocker_reason"] == "blocked_by_comparator_drift" for row in rows):
        return "one_d_to_smoothness_agreement_ladder_blocked_by_comparator_drift"
    if first_failing["rung_id"] != "none_observed":
        return "one_d_to_smoothness_agreement_ladder_first_mismatch_detected"
    if first_blocked["rung_id"] != "none_observed":
        return "one_d_to_smoothness_agreement_ladder_blocked_after_partial_agreement"
    return "one_d_to_smoothness_agreement_ladder_no_mismatch_observed"


def _validate_payload(payload: dict[str, Any]) -> None:
    required = {
        "decision",
        "plan_path",
        "result_path",
        "review_loop_path",
        "first_failing_cell",
        "first_blocked_cell",
        "rung_ledger",
        "filterflow_fingerprint_initial",
        "filterflow_fingerprint_final",
        "path_boundary_manifest",
        "run_manifest",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"missing payload keys: {sorted(missing)}")
    if len(payload["rung_ledger"]) != len(RUNG_IDS):
        raise ValueError("rung ledger must contain all eight rungs")
    if payload["plan_review_status"] != "ACCEPT_after_round_3_seed_correction":
        raise ValueError("unexpected plan review status")
    _validate_cpu_manifest(payload["run_manifest"], "parent run")
    for row in payload["rung_ledger"]:
        if row["rung"] not in RUNG_IDS:
            raise ValueError(f"unknown rung: {row['rung']}")
        if row["status"] not in {"pass", "mismatch", "blocked"}:
            raise ValueError(f"bad rung status {row['rung']}: {row['status']}")
        _validate_cpu_manifest(row["parent_cpu_only_manifest"], f"{row['rung']} parent")
        if row["filterflow_cpu_only_manifest"] is not None:
            _validate_cpu_manifest(
                row["filterflow_cpu_only_manifest"],
                f"{row['rung']} filterflow",
            )
        if row["status"] == "pass":
            if not row["primary_metrics"].get("implementation_agreement"):
                raise ValueError(f"pass without agreement: {row['rung']}")
        if row["status"] == "mismatch":
            if not row["failure_observed_directly"]:
                raise ValueError(f"mismatch without direct failure: {row['rung']}")
    first_direct = next(
        (row["first_failing_cell"] for row in payload["rung_ledger"] if row["failure_observed_directly"]),
        None,
    )
    expected_first = first_direct or _none_cell()
    if payload["first_failing_cell"] != expected_first:
        raise ValueError("first failing cell mismatch")
    first_blocked = next(
        (row["first_blocked_cell"] for row in payload["rung_ledger"] if row["status"] == "blocked"),
        None,
    )
    expected_blocked = first_blocked or _none_cell()
    if payload["first_blocked_cell"] != expected_blocked:
        raise ValueError("first blocked cell mismatch")
    boundaries = payload["path_boundary_manifest"]
    if any(bool(value) for value in boundaries.values()):
        raise ValueError(f"path boundary violation: {boundaries}")
    if payload["fixture"].get("status") == "executed":
        _validate_cpu_manifest(payload["fixture"]["cpu_only_manifest"], "fixture filterflow")


def _validate_cpu_manifest(manifest: dict[str, Any], label: str) -> None:
    if manifest.get("pre_import_cuda_visible_devices") != "-1":
        raise ValueError(f"{label}: pre-import CUDA_VISIBLE_DEVICES is not -1")
    if manifest.get("gpu_devices_visible") != []:
        raise ValueError(f"{label}: GPU devices visible {manifest.get('gpu_devices_visible')}")


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Result: 1D-to-Smoothness Filterflow Agreement Ladder",
        "",
        "## Decision",
        "",
        f"`{payload['decision']}`",
        "",
        "## Decision Table",
        "",
        "| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |",
        "| --- | --- | --- | --- | --- | --- |",
        (
            f"| `{payload['decision']}` | "
            f"`{_primary_status(payload)}` | "
            f"`{_veto_status(payload)}` | "
            f"`{_main_uncertainty(payload)}` | "
            f"`{_next_action(payload)}` | "
            "`correctness of either implementation, gradient correctness, full smoothness validation` |"
        ),
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
    lines.extend(["", "## Evidence-Bearing Cells", ""])
    for row in payload["rung_ledger"]:
        if not row["cells"]:
            continue
        cell = row["cells"][0]
        metrics = cell["primary_metrics"]
        shared = cell["shared_quality_diagnostic"]
        lines.extend(
            [
                f"### {row['rung']}",
                "",
                "| Metric | Value |",
                "| --- | ---: |",
                f"| scalar delta | `{metrics['scalar_delta']}` |",
                f"| row residual delta | `{metrics['row_residual_delta']}` |",
                f"| column residual delta | `{metrics['column_residual_delta']}` |",
                f"| BayesFilter row residual | `{metrics['bayesfilter_max_row_residual']}` |",
                f"| filterflow row residual | `{metrics['filterflow_max_row_residual']}` |",
                f"| implementation agreement | `{metrics['implementation_agreement']}` |",
                "",
                f"Shared quality diagnostic: `{shared['status']}`.",
                "",
            ]
        )
    lines.extend(
        [
            "## First Discrepancy Or Blocker",
            "",
            _first_discrepancy_markdown(payload),
            "",
            "## Fixture",
            "",
            _json_block(payload["fixture"]),
            "",
            "## Comparator",
            "",
            _json_block(_compact_fingerprint(payload["filterflow_fingerprint_initial"])),
            "",
            "## Claude Review",
            "",
            f"Plan review status: `{payload['plan_review_status']}`.",
            "Result review status is recorded in the review-loop artifact.",
            "",
            "## Non-Implications",
            "",
            *[f"- {item}" for item in payload["non_implications"]],
            "",
        ]
    )
    return "\n".join(lines)


def _primary_status(payload: dict[str, Any]) -> str:
    passing = [row["rung"] for row in payload["rung_ledger"] if row["status"] == "pass"]
    return f"passed agreement on {len(passing)} evidence-bearing rungs: {passing}"


def _veto_status(payload: dict[str, Any]) -> str:
    if payload["first_failing_cell"]["rung_id"] != "none_observed":
        return "implementation mismatch observed"
    if payload["first_blocked_cell"]["rung_id"] != "none_observed":
        return f"blocked at {payload['first_blocked_cell']['rung_id']}"
    return "no mismatch or blocker observed"


def _main_uncertainty(payload: dict[str, Any]) -> str:
    first_failing = payload["first_failing_cell"]
    if first_failing["rung_id"] != "none_observed":
        return (
            f"why {first_failing['rung_id']} differs after the changed axis; "
            "likely scalar/ledger scale, dtype, or transport residual sensitivity"
        )
    if payload["first_blocked_cell"]["rung_id"] == "R3_1d_transition_random_stream":
        return "transition/proposal stream capture remains the first blocker"
    if payload["first_blocked_cell"]["rung_id"] != "none_observed":
        return f"structured blocker at {payload['first_blocked_cell']['rung_id']}"
    return "none within executed bounded ladder"


def _next_action(payload: dict[str, Any]) -> str:
    first_failing = payload["first_failing_cell"]
    if first_failing["rung_id"] != "none_observed":
        return (
            "localize the first failing cell by per-time scalar increments, "
            "first bad ledger field, float32-vs-float64 stress, and shorter "
            "prefixes of the same observation path"
        )
    if payload["first_blocked_cell"]["rung_id"] == "R3_1d_transition_random_stream":
        return "build a reviewed non-mutating filterflow trace wrapper for R3"
    if payload["first_blocked_cell"]["rung_id"] != "none_observed":
        return f"resolve structured blocker at {payload['first_blocked_cell']['rung_id']}"
    return "extend to the next bounded rung or replicate across seeds"


def _first_discrepancy_markdown(payload: dict[str, Any]) -> str:
    first_failing = payload["first_failing_cell"]
    if first_failing["rung_id"] != "none_observed":
        return (
            f"The first observed BayesFilter/filterflow mismatch is "
            f"`{first_failing}`. Later rungs are blocked by this first "
            "mismatch and are not evidence-bearing."
        )
    first_blocked = payload["first_blocked_cell"]
    if first_blocked["rung_id"] != "none_observed":
        return (
            f"No implementation mismatch was observed before the first "
            f"structured blocker `{first_blocked}`."
        )
    return "No mismatch or blocker was observed in the executed bounded ladder."


def _compact_fingerprint(fingerprint: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "status",
        "head_commit",
        "symbolic_head",
        "branch_string_status",
        "status_short",
        "diff_digest",
        "python_version",
        "package_manifest_digest",
    )
    return {key: fingerprint.get(key) for key in keys if key in fingerprint}


def _json_block(value: Any) -> str:
    return "```json\n" + json.dumps(value, indent=2, sort_keys=True, default=str) + "\n```"


def _digest_payload(payload: dict[str, Any]) -> str:
    clone = dict(payload)
    clone.pop("created_at_utc", None)
    clone.pop("run_manifest", None)
    clone.pop("reproducibility_digest", None)
    return stable_digest(clone)


if __name__ == "__main__":
    raise SystemExit(main())
