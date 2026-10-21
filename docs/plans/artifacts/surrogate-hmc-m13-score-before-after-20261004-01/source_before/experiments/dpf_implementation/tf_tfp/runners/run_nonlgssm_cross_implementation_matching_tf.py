"""Non-LGSSM cross-implementation matching runner.

This runner compares only explicitly shared small computations.  It does not
use any implementation as an oracle and does not claim filtering correctness.
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
from pathlib import Path
from typing import Any

import tensorflow as tf

from bayesfilter import highdim
from experiments.dpf_implementation.tf_tfp.runners.common_tf import (
    OUTPUT_DIR,
    REPORT_DIR,
    REPO_ROOT,
    environment_manifest,
    load_json,
    scalar,
    stable_digest,
    utc_now,
    write_json,
    write_text,
)
from experiments.dpf_implementation.tf_tfp.runners.filterflow_reference_policy import (
    FILTERFLOW_BRANCH_MARKER,
    FILTERFLOW_REFERENCE_DTYPE,
    reference_policy,
    validate_filterflow_reference_status,
)


DTYPE = tf.float64
PLAN_PATH = "docs/plans/bayesfilter-dpf-nonlgssm-cross-implementation-matching-plan-2026-06-06.md"
RESULT_PATH = "docs/plans/bayesfilter-dpf-nonlgssm-cross-implementation-matching-result-2026-06-06.md"
JSON_PATH = OUTPUT_DIR / "dpf_nonlgssm_cross_implementation_matching_2026-06-06.json"
REPORT_PATH = REPORT_DIR / "dpf-nonlgssm-cross-implementation-matching-2026-06-06.md"
FILTERFLOW_ENV_PYTHON = REPO_ROOT / ".localenv" / "filterflow-py311" / "bin" / "python"
FILTERFLOW_PATH = REPO_ROOT / ".localsource" / "filterflow"
FILTERFLOW_MARKER_PATH = FILTERFLOW_PATH / FILTERFLOW_BRANCH_MARKER
FILTERFLOW_SV_MODEL = FILTERFLOW_PATH / "filterflow" / "models" / "stochastic_volatility.py"
STUDENT_NONLINEAR_PANEL = (
    REPO_ROOT / "experiments/student_dpf_baselines/reports/outputs/nonlinear_reference_panel_2026-05-10.json"
)
STUDENT_NONLINEAR_SUMMARY = (
    REPO_ROOT / "experiments/student_dpf_baselines/reports/outputs/nonlinear_reference_panel_summary_2026-05-10.json"
)
STUDENT_MLCOE_CLASSIC = (
    REPO_ROOT / "experiments/student_dpf_baselines/vendor/2026MLCOE/src/models/classic_ssm.py"
)
STUDENT_ADVANCED_SV = (
    REPO_ROOT / "experiments/student_dpf_baselines/vendor/advanced_particle_filter/tf_models/svssm.py"
)
VALUE_TOLERANCE = 5e-10
GRADIENT_TOLERANCE = 5e-8

SV_CONTRACT = {
    "gamma": 0.6,
    "beta": 0.4,
    "sigma": 1.0,
    "x_prev": [-1.1, -0.2, 0.35, 1.25],
    "x_next": [-0.75, 0.08, 0.42, 0.95],
    "x_obs": [-0.9, -0.1, 0.45, 1.3],
    "observation": 0.17,
    "scalar": "sum transition log densities plus sum observation log densities",
    "compared_gradient": "direct physical parameters (gamma, beta)",
}


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
    markdown = _markdown(payload)
    write_json(JSON_PATH, payload)
    write_text(REPORT_PATH, markdown)
    write_text(REPO_ROOT / RESULT_PATH, markdown)
    _validate_payload(payload)
    print(payload["decision"])
    return 0


def _run() -> dict[str, Any]:
    filterflow_status = _filterflow_checkout_manifest()
    cells = [
        _sv_density_value_gradient_cell(filterflow_status),
        _student_sv_inventory_cell(),
        _student_range_bearing_cell(),
        _spatial_sir_cell(),
        _predator_prey_cell(),
    ]
    decision = _decision(cells)
    return {
        "decision": decision,
        "created_at_utc": utc_now(),
        "question": "Non-LGSSM cross-implementation common-sense matching",
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "filterflow_reference_policy": reference_policy(),
        "filterflow_status": filterflow_status,
        "tolerances": {
            "value_abs": VALUE_TOLERANCE,
            "gradient_abs": GRADIENT_TOLERANCE,
        },
        "cells": cells,
        "summary": _summary(cells),
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners."
                "run_nonlgssm_cross_implementation_matching_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_claims": [
            "no filtering-algorithm correctness claim",
            "no implementation is treated as an oracle",
            "no TT-filter correctness claim",
            "no paper-scale validation claim",
            "no HMC/DSGE/GPU/production readiness claim",
            "interface-blocked models are not counted as failures",
            "student nonlinear panel is comparison-only evidence",
        ],
    }


def _sv_density_value_gradient_cell(filterflow_status: dict[str, Any]) -> dict[str, Any]:
    filterflow = _filterflow_sv_density_subprocess()
    bayesfilter = _bayesfilter_sv_density()
    if filterflow.get("status") != "executed":
        return {
            "model": "stochastic_volatility",
            "implementations": ["BayesFilter", "FilterFlow"],
            "cell_type": "density_value_gradient",
            "status": "INTERFACE_BLOCKED",
            "decision": "sv_filterflow_density_subprocess_blocked",
            "primary_criterion": "shared SV density value and physical-parameter gradient must execute in both implementations",
            "metrics": {},
            "mismatch_class": "filterflow_subprocess_blocked",
            "reason": filterflow.get("blocker", "unknown filterflow subprocess blocker"),
            "filterflow": filterflow,
            "bayesfilter": bayesfilter,
            "non_claim": "no SV equality claim when comparator subprocess is blocked",
        }
    comparison = _compare_sv_runs(bayesfilter, filterflow)
    matched = (
        comparison["value_within_tolerance"]
        and comparison["gradient_within_tolerance"]
        and bayesfilter["finite"]
        and filterflow["finite"]
    )
    return {
        "model": "stochastic_volatility",
        "implementations": ["BayesFilter", "FilterFlow"],
        "cell_type": "density_value_gradient",
        "status": "MATCHED" if matched else "EXPLAINED_MISMATCH",
        "decision": "sv_density_value_gradient_matched" if matched else "sv_density_value_gradient_mismatch",
        "primary_criterion": (
            "shared 1D SV transition plus observation density scalar and direct "
            "physical gradient (gamma,beta) match within tolerance"
        ),
        "metrics": comparison,
        "mismatch_class": None if matched else "sv_density_or_physical_gradient_delta",
        "contract": SV_CONTRACT,
        "bayesfilter": bayesfilter,
        "filterflow": filterflow,
        "filterflow_status": filterflow_status,
        "non_claim": "density agreement is not an end-to-end particle-filter correctness claim",
    }


def _bayesfilter_sv_density() -> dict[str, Any]:
    model = highdim.StochasticVolatilitySSM(sigma=SV_CONTRACT["sigma"])
    x_prev = _column_tensor(SV_CONTRACT["x_prev"])
    x_next = _column_tensor(SV_CONTRACT["x_next"])
    x_obs = _column_tensor(SV_CONTRACT["x_obs"])
    observation = tf.constant([SV_CONTRACT["observation"]], dtype=DTYPE)
    gamma = tf.Variable(SV_CONTRACT["gamma"], dtype=DTYPE)
    beta = tf.Variable(SV_CONTRACT["beta"], dtype=DTYPE)
    with tf.GradientTape() as tape:
        theta = model.unconstrained_from_physical(gamma, beta)
        transition = model.transition_log_density(theta, x_prev, x_next, t=1)
        observation_ll = model.observation_log_density(theta, x_obs, observation, t=1)
        value = tf.reduce_sum(transition) + tf.reduce_sum(observation_ll)
    gradient = tape.gradient(value, [gamma, beta])
    finite = (
        bool(tf.math.is_finite(value).numpy())
        and all(grad is not None and bool(tf.math.is_finite(grad).numpy()) for grad in gradient)
        and bool(tf.reduce_all(tf.math.is_finite(transition)).numpy())
        and bool(tf.reduce_all(tf.math.is_finite(observation_ll)).numpy())
    )
    return {
        "status": "executed",
        "backend": "bayesfilter.highdim.StochasticVolatilitySSM",
        "parameterization": "unconstrained theta created inside tape from physical gamma,beta",
        "scalar": scalar(value),
        "transition_log_density": _json_tensor(transition),
        "observation_log_density": _json_tensor(observation_ll),
        "physical_gradient": [_maybe_scalar(grad) for grad in gradient],
        "finite": finite,
    }


def _filterflow_sv_density_subprocess() -> dict[str, Any]:
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
        [str(FILTERFLOW_ENV_PYTHON), "-c", _filterflow_sv_script()],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    if completed.returncode != 0:
        return {
            "status": "blocked",
            "blocker": "filterflow SV density subprocess failed",
            "returncode": completed.returncode,
            "stdout_excerpt": completed.stdout[-4000:],
            "stderr_excerpt": completed.stderr[-4000:],
        }
    start = completed.stdout.rfind("FILTERFLOW_SV_DENSITY_JSON_BEGIN")
    end = completed.stdout.rfind("FILTERFLOW_SV_DENSITY_JSON_END")
    if start < 0 or end < 0 or end <= start:
        return {
            "status": "blocked",
            "blocker": "filterflow SV density JSON sentinels missing",
            "stdout_excerpt": completed.stdout[-4000:],
            "stderr_excerpt": completed.stderr[-4000:],
        }
    raw = completed.stdout[start + len("FILTERFLOW_SV_DENSITY_JSON_BEGIN") : end].strip()
    payload = json.loads(raw)
    payload["stderr_excerpt"] = completed.stderr[-2000:]
    return payload


def _filterflow_sv_script() -> str:
    return textwrap.dedent(
        f"""
        import inspect
        import json
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/bayesfilter-dpf-mpl")

        if not hasattr(inspect, "getargspec"):
            inspect.getargspec = inspect.getfullargspec

        import tensorflow as tf

        from filterflow.base import State
        from filterflow.models.stochastic_volatility import SVObservationModel, SVTransitionModel

        DTYPE = tf.float64
        GAMMA = {SV_CONTRACT["gamma"]!r}
        BETA = {SV_CONTRACT["beta"]!r}
        SIGMA = {SV_CONTRACT["sigma"]!r}
        X_PREV = {SV_CONTRACT["x_prev"]!r}
        X_NEXT = {SV_CONTRACT["x_next"]!r}
        X_OBS = {SV_CONTRACT["x_obs"]!r}
        OBSERVATION = {SV_CONTRACT["observation"]!r}

        def to_json(tensor):
            return tf.cast(tensor, tf.float64).numpy().tolist()

        def scalar(tensor):
            return float(tf.cast(tensor, tf.float64).numpy())

        gamma = tf.Variable([[GAMMA]], dtype=DTYPE)
        beta = tf.Variable([[BETA]], dtype=DTYPE)
        sigma_chol = tf.constant([[SIGMA]], dtype=DTYPE)
        mu = tf.constant([0.0], dtype=DTYPE)
        prior_state = State(tf.reshape(tf.constant(X_PREV, dtype=DTYPE), [1, len(X_PREV), 1]))
        proposed_state = State(tf.reshape(tf.constant(X_NEXT, dtype=DTYPE), [1, len(X_NEXT), 1]))
        observation_state = State(tf.reshape(tf.constant(X_OBS, dtype=DTYPE), [1, len(X_OBS), 1]))
        observation = tf.constant([OBSERVATION], dtype=DTYPE)

        with tf.GradientTape() as tape:
            transition_model = SVTransitionModel(mu, gamma, sigma_chol)
            observation_model = SVObservationModel(beta)
            transition = transition_model.loglikelihood(prior_state, proposed_state, tf.constant([], dtype=DTYPE))
            observation_ll = observation_model.loglikelihood(observation_state, observation)
            value = tf.reduce_sum(transition) + tf.reduce_sum(observation_ll)
        grad_gamma, grad_beta = tape.gradient(value, [gamma, beta])

        payload = {{
            "status": "executed",
            "backend": "filterflow.models.stochastic_volatility",
            "dtype": DTYPE.name,
            "pre_import_cuda_visible_devices": PRE_IMPORT_CUDA_VISIBLE_DEVICES,
            "gpu_devices_visible": [str(device) for device in tf.config.list_physical_devices("GPU")],
            "parameterization": "direct physical F=[[gamma]] and observation_chol=[[beta]]",
            "scalar": scalar(value),
            "transition_log_density": to_json(tf.reshape(transition, [-1])),
            "observation_log_density": to_json(tf.reshape(observation_ll, [-1])),
            "physical_gradient": [scalar(grad_gamma[0, 0]), scalar(grad_beta[0, 0])],
            "finite": bool(
                tf.math.is_finite(value).numpy()
                and tf.reduce_all(tf.math.is_finite(transition)).numpy()
                and tf.reduce_all(tf.math.is_finite(observation_ll)).numpy()
                and tf.reduce_all(tf.math.is_finite(grad_gamma)).numpy()
                and tf.reduce_all(tf.math.is_finite(grad_beta)).numpy()
            ),
        }}
        print("FILTERFLOW_SV_DENSITY_JSON_BEGIN")
        print(json.dumps(payload, sort_keys=True))
        print("FILTERFLOW_SV_DENSITY_JSON_END")
        """
    )


def _student_sv_inventory_cell() -> dict[str, Any]:
    mlcoe_text = STUDENT_MLCOE_CLASSIC.read_text(encoding="utf-8") if STUDENT_MLCOE_CLASSIC.exists() else ""
    advanced_text = STUDENT_ADVANCED_SV.read_text(encoding="utf-8") if STUDENT_ADVANCED_SV.exists() else ""
    mlcoe_has_sv = "class StochasticVolatilityModel" in mlcoe_text
    advanced_has_sv = "SVSSMParams" in advanced_text and "svssm_observation_log_prob" in advanced_text
    return {
        "model": "student_stochastic_volatility",
        "implementations": ["2026MLCOE", "advanced_particle_filter"],
        "cell_type": "interface_inventory",
        "status": "PREP_ONLY" if (mlcoe_has_sv or advanced_has_sv) else "INTERFACE_BLOCKED",
        "decision": "student_sv_surfaces_identified_not_yet_density_matched"
        if (mlcoe_has_sv or advanced_has_sv)
        else "student_sv_surface_missing",
        "primary_criterion": "identify whether student SV surfaces expose the same density/gradient object",
        "metrics": {
            "mlcoe_classic_sv_model_exists": mlcoe_has_sv,
            "advanced_tf_svssm_exists": advanced_has_sv,
        },
        "mismatch_class": "student_parameterization_adapter_needed",
        "artifacts": {
            "mlcoe_classic_model": _relative_or_none(STUDENT_MLCOE_CLASSIC),
            "advanced_tf_svssm": _relative_or_none(STUDENT_ADVANCED_SV),
        },
        "reason": (
            "student SV surfaces use implementation-specific parameterizations or "
            "filter APIs; no same scalar and gradient contract is promoted in this slice"
        ),
        "non_claim": "student SV existence is not BayesFilter-vs-student equality evidence",
    }


def _student_range_bearing_cell() -> dict[str, Any]:
    if not STUDENT_NONLINEAR_PANEL.exists() or not STUDENT_NONLINEAR_SUMMARY.exists():
        return {
            "model": "student_range_bearing_nonlinear",
            "implementations": ["2026MLCOE", "advanced_particle_filter"],
            "cell_type": "student_reference_panel",
            "status": "INTERFACE_BLOCKED",
            "decision": "student_range_bearing_panel_missing",
            "primary_criterion": "existing student range-bearing panel artifact is available",
            "metrics": {},
            "mismatch_class": "student_panel_artifact_missing",
            "reason": "student nonlinear panel or summary JSON missing",
            "non_claim": "missing student panel is not a BayesFilter model failure",
        }
    panel = json.loads(STUDENT_NONLINEAR_PANEL.read_text(encoding="utf-8"))
    summary = json.loads(STUDENT_NONLINEAR_SUMMARY.read_text(encoding="utf-8"))
    records = panel["records"]
    ok_records = [row for row in records if row["status"] == "ok"]
    failed_records = [row for row in records if row["status"] != "ok"]
    return {
        "model": "student_range_bearing_nonlinear",
        "implementations": ["2026MLCOE", "advanced_particle_filter"],
        "cell_type": "student_reference_panel",
        "status": "PREP_ONLY",
        "decision": "student_range_bearing_panel_available_comparison_only",
        "primary_criterion": "confirm the quarantined student nonlinear panel exists and is not misused as BayesFilter equality evidence",
        "metrics": {
            "fixture_count": len(panel["panel"]["fixtures"]),
            "record_count": len(records),
            "ok_record_count": len(ok_records),
            "failed_record_count": len(failed_records),
            "implementation_summary_keys": sorted(summary.get("implementation_summary", {}).keys()),
        },
        "mismatch_class": "not_same_as_bayesfilter_sir_or_predator_prey",
        "artifacts": {
            "panel": _relative_or_none(STUDENT_NONLINEAR_PANEL),
            "summary": _relative_or_none(STUDENT_NONLINEAR_SUMMARY),
        },
        "non_claim": "range-bearing RMSE/ESS panel is not BayesFilter-vs-FilterFlow value or gradient equality evidence",
    }


def _spatial_sir_cell() -> dict[str, Any]:
    model = highdim.p30_spatial_sir_fixture_model(compartments=2)
    smoke = _bayesfilter_model_contract_smoke(model, tf.zeros([0], dtype=DTYPE))
    return {
        "model": "spatial_sir",
        "implementations": ["BayesFilter", "FilterFlow", "student_repos"],
        "cell_type": "interface_inventory_with_bayesfilter_smoke",
        "status": "INTERFACE_BLOCKED",
        "decision": "spatial_sir_no_comparable_second_implementation_identified",
        "primary_criterion": "do not force BayesFilter first-gate SIR contract into a false cross-implementation tie-out",
        "metrics": smoke,
        "mismatch_class": "no_comparable_filterflow_or_student_sir_interface",
        "reason": "FilterFlow and student inventories expose no same spatial-SIR density/filter contract in this slice",
        "non_claim": "finite BayesFilter SIR density smoke is not a cross-implementation match",
    }


def _predator_prey_cell() -> dict[str, Any]:
    model = highdim.p30_predator_prey_fixture_model()
    smoke = _bayesfilter_model_contract_smoke(model, model.true_parameters())
    return {
        "model": "predator_prey",
        "implementations": ["BayesFilter", "FilterFlow", "student_repos"],
        "cell_type": "interface_inventory_with_bayesfilter_smoke",
        "status": "INTERFACE_BLOCKED",
        "decision": "predator_prey_no_comparable_second_implementation_identified",
        "primary_criterion": "do not force BayesFilter first-gate predator-prey contract into a false cross-implementation tie-out",
        "metrics": smoke,
        "mismatch_class": "no_comparable_filterflow_or_student_predator_prey_interface",
        "reason": "FilterFlow and student inventories expose no same predator-prey density/filter contract in this slice",
        "non_claim": "finite BayesFilter predator-prey density smoke is not a cross-implementation match",
    }


def _bayesfilter_model_contract_smoke(model: Any, theta: tf.Tensor) -> dict[str, Any]:
    if isinstance(model, highdim.SpatialSIRSSM):
        x_prev = model.initial_mean[tf.newaxis, :]
        x_next = model.transition_mean(x_prev)
        y_t = model.infectious_components(x_next)[0]
    elif isinstance(model, highdim.PredatorPreySSM):
        x_prev = model.initial_mean[tf.newaxis, :]
        x_next = model.transition_mean(theta, x_prev)
        y_t = x_next[0]
    else:
        raise TypeError(f"unsupported smoke model: {type(model)}")
    transition = model.transition_log_density(theta, x_prev, x_next, t=1)
    observation = model.observation_log_density(theta, x_next, y_t, t=1)
    return {
        "state_dim": model.state_dim(),
        "observation_dim": model.observation_dim(),
        "transition_log_density": _json_tensor(transition),
        "observation_log_density": _json_tensor(observation),
        "finite": bool(
            tf.reduce_all(tf.math.is_finite(transition)).numpy()
            and tf.reduce_all(tf.math.is_finite(observation)).numpy()
        ),
        "manifest_family": model.manifest_payload()["family"],
    }


def _compare_sv_runs(bayesfilter: dict[str, Any], filterflow: dict[str, Any]) -> dict[str, Any]:
    scalar_delta = bayesfilter["scalar"] - filterflow["scalar"]
    transition_delta = _max_abs_nested_delta(
        bayesfilter["transition_log_density"],
        filterflow["transition_log_density"],
    )
    observation_delta = _max_abs_nested_delta(
        bayesfilter["observation_log_density"],
        filterflow["observation_log_density"],
    )
    gradient_delta = [
        float(left) - float(right)
        for left, right in zip(bayesfilter["physical_gradient"], filterflow["physical_gradient"], strict=True)
    ]
    max_abs_gradient_delta = max(abs(value) for value in gradient_delta)
    max_abs_value_delta = max(abs(scalar_delta), transition_delta, observation_delta)
    return {
        "scalar_delta": scalar_delta,
        "max_abs_transition_delta": transition_delta,
        "max_abs_observation_delta": observation_delta,
        "max_abs_value_delta": max_abs_value_delta,
        "gradient_delta": gradient_delta,
        "max_abs_gradient_delta": max_abs_gradient_delta,
        "value_within_tolerance": max_abs_value_delta <= VALUE_TOLERANCE,
        "gradient_within_tolerance": max_abs_gradient_delta <= GRADIENT_TOLERANCE,
        "filterflow_cpu_only": (
            filterflow.get("pre_import_cuda_visible_devices") == "-1"
            and filterflow.get("gpu_devices_visible") == []
        ),
    }


def _decision(cells: list[dict[str, Any]]) -> str:
    if any(cell["status"] == "EXPLAINED_MISMATCH" and not cell.get("mismatch_class") for cell in cells):
        return "nonlgssm_matching_unclassified_mismatch_veto"
    if any(cell["status"] == "MATCHED" for cell in cells):
        return "nonlgssm_matching_sv_density_pass_with_interface_blockers"
    if any(cell["status"] == "EXPLAINED_MISMATCH" for cell in cells):
        return "nonlgssm_matching_explained_mismatch"
    return "nonlgssm_matching_no_executed_matches"


def _summary(cells: list[dict[str, Any]]) -> dict[str, Any]:
    statuses: dict[str, int] = {}
    for cell in cells:
        statuses[cell["status"]] = statuses.get(cell["status"], 0) + 1
    return {
        "num_cells": len(cells),
        "status_counts": statuses,
        "models": [cell["model"] for cell in cells],
    }


def _validate_payload(payload: dict[str, Any]) -> None:
    if payload["run_manifest"]["pre_import_cuda_visible_devices"] != "-1":
        raise RuntimeError("CPU-only pre-import manifest missing")
    if payload["run_manifest"]["gpu_devices_visible"] != []:
        raise RuntimeError("GPU visible in CPU-only run")
    if "reproducibility_digest" not in payload:
        raise RuntimeError("missing reproducibility digest")
    validate_filterflow_reference_status(payload["filterflow_status"], marker_path=FILTERFLOW_MARKER_PATH)
    if payload["filterflow_reference_policy"]["dtype"] != FILTERFLOW_REFERENCE_DTYPE:
        raise RuntimeError("wrong filterflow dtype policy")
    for cell in payload["cells"]:
        if cell["status"] not in {"MATCHED", "EXPLAINED_MISMATCH", "INTERFACE_BLOCKED", "PREP_ONLY"}:
            raise RuntimeError(f"unknown status: {cell['status']}")
        if cell["status"] == "EXPLAINED_MISMATCH" and not cell.get("mismatch_class"):
            raise RuntimeError(f"unclassified mismatch: {cell['model']}")
        if cell["status"] == "MATCHED" and not cell.get("metrics", {}).get("value_within_tolerance", True):
            raise RuntimeError(f"matched cell with failed value tolerance: {cell['model']}")
        if cell["status"] == "MATCHED" and not cell.get("metrics", {}).get("gradient_within_tolerance", True):
            raise RuntimeError(f"matched cell with failed gradient tolerance: {cell['model']}")
    sv_cells = [cell for cell in payload["cells"] if cell["model"] == "stochastic_volatility"]
    if not sv_cells or sv_cells[0]["status"] != "MATCHED":
        raise RuntimeError("SV density match did not execute as the primary non-LGSSM equality cell")


def _markdown(payload: dict[str, Any]) -> str:
    return f"""# DPF Non-LGSSM Cross-Implementation Matching Result

metadata_date: 2026-06-06

## Decision

`{payload['decision']}`

## Summary

- Cells: `{payload['summary']['num_cells']}`
- Status counts: `{payload['summary']['status_counts']}`

## Cell Table

{_cell_table(payload['cells'])}

## SV Match

The stochastic-volatility density cell compared BayesFilter and executable
float64 FilterFlow on the same 1D transition and observation density scalar,
using physical parameters `(gamma,beta)` for the gradient comparison.

- Max absolute value delta:
  `{_sv_metric(payload, 'max_abs_value_delta')}`
- Max absolute gradient delta:
  `{_sv_metric(payload, 'max_abs_gradient_delta')}`

## Interpretation

The non-LGSSM slice promotes one actual equality cell: stochastic-volatility
transition/observation density value and physical-parameter gradient match
between BayesFilter and float64 FilterFlow.  Student SV and range-bearing rows
remain preparation/comparison-only evidence.  SIR and predator-prey remain
interface-blocked for cross-implementation matching because no same-model
FilterFlow or student surface was identified.

## Non-Claims

{_bullet_list(payload['non_claims'])}
"""


def _cell_table(cells: list[dict[str, Any]]) -> str:
    lines = [
        "| model | cell type | status | decision | mismatch class |",
        "|---|---|---|---|---|",
    ]
    for cell in cells:
        lines.append(
            "| {model} | {cell_type} | `{status}` | `{decision}` | {mismatch} |".format(
                model=cell["model"],
                cell_type=cell["cell_type"],
                status=cell["status"],
                decision=cell["decision"],
                mismatch=cell.get("mismatch_class") or "N/A",
            )
        )
    return "\n".join(lines)


def _sv_metric(payload: dict[str, Any], name: str) -> str:
    for cell in payload["cells"]:
        if cell["model"] == "stochastic_volatility":
            return str(cell["metrics"].get(name))
    return "N/A"


def _filterflow_checkout_manifest() -> dict[str, Any]:
    if not FILTERFLOW_PATH.exists():
        return {
            "path": str(FILTERFLOW_PATH),
            "status": "missing",
            "commit": "N/A",
            "branch": "N/A",
            "status_short": "N/A",
            "sv_model_exists": False,
        }
    return {
        "path": str(FILTERFLOW_PATH),
        "status": "current_local_float64_reference_checkout",
        "commit": _git_filterflow(["rev-parse", "HEAD"]),
        "branch": _git_filterflow(["rev-parse", "--abbrev-ref", "HEAD"]),
        "status_short": _git_filterflow(["status", "--short"]),
        "sv_model_exists": FILTERFLOW_SV_MODEL.exists(),
        "marker_exists": FILTERFLOW_MARKER_PATH.exists(),
        "provenance_note": "local float64 FilterFlow comparator, not pristine upstream and not an oracle",
    }


def _git_filterflow(args: list[str]) -> str:
    completed = subprocess.run(
        ["git", "-C", str(FILTERFLOW_PATH), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _digest_payload(payload: dict[str, Any]) -> str:
    comparable = dict(payload)
    comparable["created_at_utc"] = "TIMESTAMP"
    comparable["run_manifest"] = dict(comparable["run_manifest"])
    comparable["run_manifest"]["wall_time_seconds"] = "WALL_TIME"
    comparable["run_manifest"]["dirty_state_summary"] = "DIRTY_STATE"
    comparable["filterflow_status"] = dict(comparable["filterflow_status"])
    comparable["filterflow_status"]["status_short"] = "FILTERFLOW_STATUS"
    return stable_digest(comparable)


def _column_tensor(values: list[float]) -> tf.Tensor:
    return tf.reshape(tf.constant(values, dtype=DTYPE), [-1, 1])


def _json_tensor(value: tf.Tensor) -> Any:
    return tf.cast(value, DTYPE).numpy().tolist()


def _maybe_scalar(value: tf.Tensor | None) -> float | None:
    if value is None:
        return None
    return scalar(value)


def _max_abs_nested_delta(left: Any, right: Any) -> float:
    left_tensor = tf.convert_to_tensor(left, dtype=DTYPE)
    right_tensor = tf.convert_to_tensor(right, dtype=DTYPE)
    return scalar(tf.reduce_max(tf.abs(left_tensor - right_tensor)))


def _relative_or_none(path: Path) -> str | None:
    if not path.exists():
        return None
    return str(path.relative_to(REPO_ROOT))


def _bullet_list(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


if __name__ == "__main__":
    raise SystemExit(main())
