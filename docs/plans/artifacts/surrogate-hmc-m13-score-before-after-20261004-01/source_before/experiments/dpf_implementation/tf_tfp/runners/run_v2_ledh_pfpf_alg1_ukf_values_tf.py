"""Run V2 Algorithm 1 UKF LEDH-PFPF diagnostic value replacements."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/bayesfilter-dpf-mpl")
PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]

import argparse
import math
import statistics
import sys
import time
import traceback
from typing import Any

import tensorflow as tf

from experiments.dpf_implementation.tf_tfp.filters.bootstrap_pf_tf import (
    run_bootstrap_particle_filter_tf,
)
from experiments.dpf_implementation.tf_tfp.filters.ledh_pfpf_alg1_ukf_tf import (
    METHOD_GENERATION,
    algorithm1_route_identifiers,
    run_ledh_pfpf_alg1_ukf_tf,
    validate_algorithm1_route_identifiers,
)
from experiments.dpf_implementation.tf_tfp.fixtures.common_model_suite_tf import (
    DTYPE,
    EXPECTED_V2_MODEL_IDS,
    CommonModelSpecV2,
    bayesfilter_model_for_spec_v2,
    common_model_specs_v2,
    range_bearing_observation_tf,
)
from experiments.dpf_implementation.tf_tfp.flows.jacobians_tf import (
    range_bearing_jacobian_tf,
)
from experiments.dpf_implementation.tf_tfp.runners.common_tf import (
    OUTPUT_DIR,
    REPORT_DIR,
    REPO_ROOT,
    environment_manifest,
    load_json,
    scalar,
    stable_digest,
    tensor_to_json,
    utc_now,
    write_json,
    write_text,
)


MODULE_PATH = (
    "experiments.dpf_implementation.tf_tfp.runners."
    "run_v2_ledh_pfpf_alg1_ukf_values_tf"
)
PLAN_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-ledh-pfpf-alg1-ukf-ledh-pfpf-ot-rerun-p3-v2-values-subplan-2026-06-10.md"
)
RESULT_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-ledh-pfpf-alg1-ukf-ledh-pfpf-ot-rerun-p3-v2-values-result-2026-06-10.md"
)
REVIEW_LEDGER_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-ledh-pfpf-alg1-ukf-ledh-pfpf-ot-rerun-claude-review-ledger-2026-06-10.md"
)
P2_JSON_PATH = OUTPUT_DIR / "dpf_v2_ledh_pfpf_alg1_ukf_contracts_2026-06-10.json"
JSON_PATH = OUTPUT_DIR / "dpf_v2_ledh_pfpf_alg1_ukf_values_2026-06-10.json"
REPORT_PATH = REPORT_DIR / "dpf-v2-ledh-pfpf-alg1-ukf-values-2026-06-10.md"

METHOD_IDS = (
    "bootstrap_pf_no_resampling_tf",
    "ledh_pfpf_alg1_ukf_no_resampling_tf",
)
LOCAL_PASS_DECISION = (
    "LOCAL_PASS_P3_V2_ALG1_UKF_VALUES_DIAGNOSTIC_ONLY_PENDING_CLAUDE_REVIEW"
)
VETO_DECISION = "P3_V2_ALG1_UKF_VALUES_VETO_PENDING_REVIEW"
VALUE_SEEDS = [101, 202, 303, 404, 505]
VALUE_PARTICLE_COUNTS = [8, 16, 32]
PSEUDO_TIME_STEPS = [0.5, 0.5]
UKF_ALPHA = 1.0
UKF_BETA = 2.0
UKF_KAPPA = 0.0
COVARIANCE_FLOOR = 1e-10
RANK_TOLERANCE = 1e-12


class P3ValidationError(ValueError):
    """Raised when the P3 payload violates the frozen value contract."""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only:
        _validate_payload(load_json(JSON_PATH))
        print("P3_V2_LEDHPFPF_ALG1_UKF_VALUES_VALIDATED")
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
    p2_payload = load_json(P2_JSON_PATH)
    _preflight(p2_payload)
    specs = common_model_specs_v2()
    spec_by_id = {spec.model_id: spec for spec in specs}
    cells = []
    value_rows = []
    for contract in p2_payload["contracts"]:
        spec = spec_by_id[str(contract["model_id"])]
        if contract["status"] != "RUNNABLE_ALG1":
            cells.append(_blocked_cell(spec, contract))
            continue
        rows = _run_contract_value_rows(spec, contract)
        value_rows.extend(rows)
        cells.append(_value_cell(spec, contract, rows))
    summaries = _value_summaries(value_rows)
    veto = _veto_diagnostics(p2_payload, cells, value_rows, summaries)
    decision = LOCAL_PASS_DECISION if not any(bool(value) for value in veto.values()) else VETO_DECISION
    manifest = environment_manifest(
        command="CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m " + MODULE_PATH,
        pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
    )
    manifest.update(
        {
            "plan_path": PLAN_PATH,
            "result_path": RESULT_PATH,
            "review_ledger_path": REVIEW_LEDGER_PATH,
            "p2_json_path": str(P2_JSON_PATH.relative_to(REPO_ROOT)),
            "json_path": str(JSON_PATH.relative_to(REPO_ROOT)),
            "report_path": str(REPORT_PATH.relative_to(REPO_ROOT)),
            "p2_contract_bundle_checksum": p2_payload["contract_bundle_checksum"],
            "value_seed_list": list(VALUE_SEEDS),
            "value_particle_counts": list(VALUE_PARTICLE_COUNTS),
            "pseudo_time_steps": list(PSEUDO_TIME_STEPS),
            "ukf_parameters": _ukf_parameters(),
            "covariance_floor": COVARIANCE_FLOOR,
            "rank_tolerance": RANK_TOLERANCE,
        }
    )
    return {
        "metadata_date": "2026-06-10",
        "created_at_utc": utc_now(),
        "phase": "P3",
        "execution_route": "VISIBLE_IN_DIALOGUE",
        "decision": decision,
        "skeptical_plan_audit": {
            "status": "PASS_FOR_DIAGNOSTIC_VALUE_RERUN",
            "wrong_baseline_control": (
                "P3 consumes the frozen P2 contracts.  LGSSM uses exact Kalman "
                "as a comparator; non-LGSSM rows are diagnostic-only."
            ),
            "proxy_metric_control": (
                "Finite values, ESS, particle-ladder trends, and non-LGSSM "
                "value magnitudes cannot promote correctness in P3."
            ),
            "stop_conditions": (
                "Non-runnable P2 rows remain blocked.  Runnable rows that fail "
                "execution must be classified instead of disappearing."
            ),
        },
        "evidence_contract": {
            "question": (
                "For V2 rows declared runnable in P2, do Algorithm 1 UKF value "
                "runs execute finitely and preserve Monte Carlo uncertainty?"
            ),
            "baseline_comparator": (
                "P2 frozen contracts; exact Kalman on LGSSM; bootstrap no-flow "
                "PF as a baseline comparator.  Other rows have no exact oracle "
                "in P3 and remain diagnostic-only."
            ),
            "primary_criterion": (
                "Every P2 row appears with a reviewed status; every P2 runnable "
                "row is executed or explicitly downgraded; P2-blocked rows may "
                "carry forward only with adapter reasons; finite runnable rows "
                "include uncertainty, route fields, and no promotion claim."
            ),
            "veto_diagnostics": list(veto.keys()),
            "not_concluded": _nonclaims(),
        },
        "gate_definition": {
            "local_decision_semantics": (
                "LOCAL_PASS means the locally generated P3 artifact satisfies "
                "the pre-Claude gate and remains pending read-only Claude "
                "review.  It is not an unconditional phase pass."
            ),
            "p2_blocked_carry_forward_allowed": True,
            "p2_blocked_carry_forward_rule": (
                "Rows blocked in P2 may remain BLOCKED_REQUIRES_ADAPTER in P3 "
                "with row visibility, zero value rows, N/A comparator route, "
                "missing adapter items, and status reason preserved.  P3 does "
                "not require those rows to execute values."
            ),
            "p2_runnable_rule": (
                "Rows marked RUNNABLE_ALG1 in P2 must execute value rows or "
                "be explicitly downgraded with failure diagnostics."
            ),
            "promotion_rule": (
                "P3 value evidence is diagnostic-only.  Finite values, ESS, "
                "and particle-ladder trends cannot promote correctness."
            ),
        },
        "threshold_policy": {
            "value_tolerance": "N/A_DIAGNOSTIC_ONLY_IN_P3",
            "certification_band": "N/A_DIAGNOSTIC_ONLY_IN_P3",
            "finite_only_promotion_allowed": False,
            "reason": (
                "P2 froze diagnostic-only thresholds because Monte Carlo error "
                "depends on model noise, horizon, dimension, nonlinearity, and "
                "particle count."
            ),
        },
        "required_v2_model_ids": list(EXPECTED_V2_MODEL_IDS),
        "methods": list(METHOD_IDS),
        "cells": cells,
        "value_rows": value_rows,
        "value_summaries": summaries,
        "veto_diagnostics": veto,
        "execution_diagnostics": {
            "old_ledh_pfpf_ot_imported": _old_runtime_module_loaded(),
            "algorithm1_value_rows_executed": len(
                [row for row in value_rows if row["method_id"] == "ledh_pfpf_alg1_ukf_no_resampling_tf"]
            ),
            "algorithm1_gradients_computed": False,
            "ot_or_annealed_transport_used": False,
            "filterflow_subprocess_run": False,
            "student_command_run": False,
        },
        "summary": _summary(cells, summaries),
        "artifact_paths": {
            "json": str(JSON_PATH.relative_to(REPO_ROOT)),
            "markdown_report": str(REPORT_PATH.relative_to(REPO_ROOT)),
            "phase_result": RESULT_PATH,
        },
        "run_manifest": manifest,
        "nonclaims": _nonclaims(),
    }


def _run_contract_value_rows(
    spec: CommonModelSpecV2,
    contract: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for num_particles in VALUE_PARTICLE_COUNTS:
        for seed in VALUE_SEEDS:
            for method_id in METHOD_IDS:
                rows.append(_run_one_value_row(spec, contract, method_id, seed, num_particles))
    return rows


def _run_one_value_row(
    spec: CommonModelSpecV2,
    contract: dict[str, Any],
    method_id: str,
    seed: int,
    num_particles: int,
) -> dict[str, Any]:
    try:
        callbacks = _callbacks_for_spec(spec, contract)
        if method_id == "bootstrap_pf_no_resampling_tf":
            result = run_bootstrap_particle_filter_tf(
                observations=callbacks["observations"],
                initial_sample=callbacks["initial_sample"],
                transition_sample=callbacks["transition_sample"],
                observation_log_density=callbacks["observation_log_density_fn"],
                seed=seed,
                num_particles=num_particles,
                ess_threshold_ratio=-1.0,
                method_id=method_id,
            )
            value = result.log_likelihood_estimate
            diagnostics = _bootstrap_diagnostics(result)
            route_fields = _baseline_route_fields()
        elif method_id == "ledh_pfpf_alg1_ukf_no_resampling_tf":
            result = run_ledh_pfpf_alg1_ukf_tf(
                observations=callbacks["observations"],
                initial_sample=callbacks["initial_sample"],
                initial_covariance=callbacks["initial_covariance"],
                transition_sample=callbacks["transition_sample"],
                transition_mean_fn=callbacks["transition_mean_fn"],
                transition_log_density_fn=callbacks["transition_log_density_fn"],
                observation_mean_fn=callbacks["observation_mean_fn"],
                observation_jacobian_fn=callbacks["observation_jacobian_fn"],
                observation_log_density_fn=callbacks["observation_log_density_fn"],
                process_noise_covariance_fn=callbacks["process_noise_covariance_fn"],
                observation_covariance_fn=callbacks["observation_covariance_fn"],
                seed=seed,
                num_particles=num_particles,
                pseudo_time_steps=tf.constant(PSEUDO_TIME_STEPS, DTYPE),
                resampling_route="none",
                alpha=UKF_ALPHA,
                beta=UKF_BETA,
                kappa=UKF_KAPPA,
                covariance_floor=COVARIANCE_FLOOR,
                rank_tolerance=RANK_TOLERANCE,
                method_id=method_id,
            )
            value = result.log_likelihood_estimate
            diagnostics = _alg1_diagnostics(result)
            route_fields = _augmented_algorithm1_route()
        else:
            raise ValueError(f"unknown method_id {method_id}")
        reference = _reference_value(spec, callbacks)
        value_error = None if reference is None else scalar(value - tf.constant(reference, DTYPE))
        return {
            "model_id": spec.model_id,
            "contract_id": contract["contract_id"],
            "contract_checksum": contract["contract_checksum"],
            "method_id": method_id,
            "seed": int(seed),
            "num_particles": int(num_particles),
            "value": scalar(value),
            "reference_value": reference,
            "comparator_route": contract["scalar_contract"]["comparator_route"],
            "value_error": value_error,
            "per_observation_value_error": (
                None
                if value_error is None
                else value_error / float(contract["horizon"])
            ),
            "finite": bool(tf.math.is_finite(value).numpy()) and bool(diagnostics["finite"]),
            "row_status": "RERUN_ALG1_DIAGNOSTIC_ONLY",
            "value_tolerance": "N/A_DIAGNOSTIC_ONLY_IN_P3",
            "primary_promote_statistic": "N/A_DIAGNOSTIC_ONLY_IN_P3",
            "stochastic_score_claim": "not_claimed",
            "route_fields": route_fields,
            "diagnostics": diagnostics,
        }
    except Exception as exc:  # noqa: BLE001 - row classification preserves the failure.
        return {
            "model_id": spec.model_id,
            "contract_id": contract["contract_id"],
            "contract_checksum": contract["contract_checksum"],
            "method_id": method_id,
            "seed": int(seed),
            "num_particles": int(num_particles),
            "value": None,
            "reference_value": None,
            "comparator_route": contract["scalar_contract"]["comparator_route"],
            "value_error": None,
            "per_observation_value_error": None,
            "finite": False,
            "row_status": "BLOCKED_REQUIRES_ADAPTER",
            "value_tolerance": "N/A_BLOCKED_REQUIRES_ADAPTER",
            "primary_promote_statistic": "N/A_BLOCKED_REQUIRES_ADAPTER",
            "stochastic_score_claim": "not_claimed",
            "route_fields": (
                _augmented_algorithm1_route()
                if method_id == "ledh_pfpf_alg1_ukf_no_resampling_tf"
                else _baseline_route_fields()
            ),
            "diagnostics": {
                "finite": False,
                "execution_error_type": type(exc).__name__,
                "execution_error": str(exc),
                "traceback_tail": traceback.format_exc(limit=3).splitlines()[-6:],
                "old_ledh_pfpf_ot_used": False,
            },
        }


def _callbacks_for_spec(spec: CommonModelSpecV2, contract: dict[str, Any]) -> dict[str, Any]:
    model = bayesfilter_model_for_spec_v2(spec)
    theta = tf.convert_to_tensor(contract["theta"], DTYPE)
    observations = tf.convert_to_tensor(contract["path_fixture"]["observations"], DTYPE)
    params = spec.parameters
    initial_mean = _initial_mean(spec)
    initial_covariance = _initial_covariance(spec)
    process_covariance = _process_covariance(spec)
    observation_covariance = _observation_covariance(spec)

    def initial_sample(num_particles: int, seed: int) -> tf.Tensor:
        return _mvn_sample(initial_mean, initial_covariance, seed, 11, (num_particles,))

    def transition_mean_fn(points: tf.Tensor, _time_index: int) -> tf.Tensor:
        points = tf.convert_to_tensor(points, DTYPE)
        if spec.model_id in {"lgssm_2d_h25_rich", "range_bearing_4d_h20_rich"}:
            return tf.linalg.matmul(points, tf.convert_to_tensor(params["A"], DTYPE), transpose_b=True)
        if spec.model_id == "spatial_sir_j3_rk4":
            return model.transition_mean(points)
        if spec.model_id == "predator_prey_rk4":
            return model.transition_mean(theta, points)
        raise ValueError(f"{spec.model_id}: no Algorithm 1 transition_mean_fn contract")

    def transition_sample(ancestors: tf.Tensor, seed: int, time_index: int) -> tf.Tensor:
        mean = transition_mean_fn(ancestors, time_index)
        noise = _mvn_sample(
            tf.zeros([spec.state_dim], DTYPE),
            process_covariance,
            seed,
            1000 + int(time_index),
            (int(ancestors.shape[0]),),
        )
        return mean + noise

    def transition_log_density_fn(current: tf.Tensor, previous: tf.Tensor, time_index: int) -> tf.Tensor:
        return tf.cast(model.transition_log_density(theta, previous, current, t=int(time_index) + 1), DTYPE)

    def observation_mean_fn(points: tf.Tensor, _time_index: int) -> tf.Tensor:
        points = tf.convert_to_tensor(points, DTYPE)
        if spec.model_id == "lgssm_2d_h25_rich":
            return tf.linalg.matmul(points, tf.convert_to_tensor(params["C"], DTYPE), transpose_b=True)
        if spec.model_id == "range_bearing_4d_h20_rich":
            return range_bearing_observation_tf(points)
        if spec.model_id == "spatial_sir_j3_rk4":
            return points[:, 1::2]
        if spec.model_id == "predator_prey_rk4":
            return points
        raise ValueError(f"{spec.model_id}: no Algorithm 1 observation_mean_fn contract")

    def observation_jacobian_fn(point: tf.Tensor, _time_index: int) -> tf.Tensor:
        point = tf.reshape(tf.convert_to_tensor(point, DTYPE), [-1])
        if spec.model_id == "lgssm_2d_h25_rich":
            return tf.convert_to_tensor(params["C"], DTYPE)
        if spec.model_id == "range_bearing_4d_h20_rich":
            return range_bearing_jacobian_tf(point)
        if spec.model_id == "spatial_sir_j3_rk4":
            return _sir_observation_jacobian(spec.state_dim, spec.observation_dim)
        if spec.model_id == "predator_prey_rk4":
            return tf.eye(spec.state_dim, dtype=DTYPE)
        raise ValueError(f"{spec.model_id}: no Algorithm 1 observation_jacobian_fn contract")

    def observation_log_density_fn(particles: tf.Tensor, observation: tf.Tensor, time_index: int) -> tf.Tensor:
        return tf.cast(model.observation_log_density(theta, particles, observation, t=int(time_index) + 1), DTYPE)

    return {
        "model": model,
        "theta": theta,
        "observations": observations,
        "initial_covariance": initial_covariance,
        "initial_sample": initial_sample,
        "transition_sample": transition_sample,
        "transition_mean_fn": transition_mean_fn,
        "transition_log_density_fn": transition_log_density_fn,
        "observation_mean_fn": observation_mean_fn,
        "observation_jacobian_fn": observation_jacobian_fn,
        "observation_log_density_fn": observation_log_density_fn,
        "process_noise_covariance_fn": lambda _x_prev, _t: process_covariance,
        "observation_covariance_fn": lambda _t: observation_covariance,
    }


def _reference_value(spec: CommonModelSpecV2, callbacks: dict[str, Any]) -> float | None:
    if spec.model_id != "lgssm_2d_h25_rich":
        return None
    params = spec.parameters
    mean = tf.convert_to_tensor(params["m0"], DTYPE)
    covariance = tf.convert_to_tensor(params["P0"], DTYPE)
    a = tf.convert_to_tensor(params["A"], DTYPE)
    c = tf.convert_to_tensor(params["C"], DTYPE)
    q = tf.convert_to_tensor(params["Q"], DTYPE)
    r = tf.convert_to_tensor(params["R"], DTYPE)
    value = tf.constant(0.0, DTYPE)
    for observation in tf.unstack(callbacks["observations"], axis=0):
        pred_mean = tf.linalg.matvec(a, mean)
        pred_cov = a @ covariance @ tf.transpose(a) + q
        obs_mean = tf.linalg.matvec(c, pred_mean)
        innovation_cov = c @ pred_cov @ tf.transpose(c) + r
        residual = tf.reshape(observation, [-1]) - obs_mean
        value = value + _gaussian_logpdf_zero_mean(tf.reshape(residual, [1, -1]), innovation_cov)[0]
        gain = tf.transpose(tf.linalg.solve(innovation_cov, c @ pred_cov))
        mean = pred_mean + tf.linalg.matvec(gain, residual)
        covariance = _symmetrize((tf.eye(int(a.shape[0]), dtype=DTYPE) - gain @ c) @ pred_cov)
    return scalar(value)


def _value_cell(
    spec: CommonModelSpecV2,
    contract: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    finite_rows = [row for row in rows if bool(row["finite"])]
    status = (
        "RERUN_ALG1_DIAGNOSTIC_ONLY"
        if len(finite_rows) == len(rows)
        else "BLOCKED_REQUIRES_ADAPTER"
    )
    return {
        "model_id": spec.model_id,
        "contract_id": contract["contract_id"],
        "contract_checksum": contract["contract_checksum"],
        "p2_status": contract["status"],
        "p3_status": status,
        "row_count": len(rows),
        "finite_row_count": len(finite_rows),
        "methods": list(METHOD_IDS),
        "seed_count": len(VALUE_SEEDS),
        "particle_ladder": list(VALUE_PARTICLE_COUNTS),
        "comparator_route": contract["scalar_contract"]["comparator_route"],
        "primary_promote_statistic": "N/A_DIAGNOSTIC_ONLY_IN_P3",
        "value_tolerance": "N/A_DIAGNOSTIC_ONLY_IN_P3",
        "reason": (
            "finite diagnostic value rows only; no calibrated promotion band"
            if status == "RERUN_ALG1_DIAGNOSTIC_ONLY"
            else "one or more value rows failed and are classified with diagnostics"
        ),
    }


def _blocked_cell(spec: CommonModelSpecV2, contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_id": spec.model_id,
        "contract_id": contract["contract_id"],
        "contract_checksum": contract["contract_checksum"],
        "p2_status": contract["status"],
        "p3_status": "BLOCKED_REQUIRES_ADAPTER",
        "row_count": 0,
        "finite_row_count": 0,
        "methods": [],
        "seed_count": "N/A",
        "particle_ladder": "N/A",
        "comparator_route": contract["scalar_contract"]["comparator_route"],
        "primary_promote_statistic": "N/A_BLOCKED_REQUIRES_ADAPTER",
        "value_tolerance": "N/A_BLOCKED_REQUIRES_ADAPTER",
        "missing_adapter_items": list(contract["missing_adapter_items"]),
        "reason": contract["status_reason"],
    }


def _value_summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for model_id in sorted({row["model_id"] for row in rows}):
        summaries[model_id] = {}
        for method_id in METHOD_IDS:
            summaries[model_id][method_id] = {}
            for count in VALUE_PARTICLE_COUNTS:
                subset = [
                    row for row in rows
                    if row["model_id"] == model_id
                    and row["method_id"] == method_id
                    and row["num_particles"] == count
                ]
                values = [float(row["value"]) for row in subset if row["value"] is not None]
                errors = [
                    float(row["value_error"]) for row in subset
                    if row["value_error"] is not None
                ]
                summaries[model_id][method_id][str(count)] = {
                    "seed_count": len(subset),
                    "finite_count": sum(1 for row in subset if bool(row["finite"])),
                    "mean_value": statistics.fmean(values) if values else None,
                    "value_standard_error": _standard_error(values),
                    "value_ci95": _ci95(values),
                    "value_rmse_vs_reference": _rmse(errors) if errors else None,
                    "mean_value_error": statistics.fmean(errors) if errors else None,
                    "value_error_standard_error": _standard_error(errors),
                    "min_ess": (
                        min(float(row["diagnostics"]["ess_min"]) for row in subset if row["finite"])
                        if any(row["finite"] for row in subset)
                        else None
                    ),
                    "row_status": "RERUN_ALG1_DIAGNOSTIC_ONLY",
                }
    return summaries


def _veto_diagnostics(
    p2_payload: dict[str, Any],
    cells: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    summaries: dict[str, Any],
) -> dict[str, bool]:
    del summaries
    ids = [cell["model_id"] for cell in cells]
    runnable_ids = {
        contract["model_id"] for contract in p2_payload["contracts"]
        if contract["status"] == "RUNNABLE_ALG1"
    }
    runnable_cells = [cell for cell in cells if cell["model_id"] in runnable_ids]
    expected_rows = len(runnable_ids) * len(METHOD_IDS) * len(VALUE_SEEDS) * len(VALUE_PARTICLE_COUNTS)
    return {
        "p2_contract_absent_or_not_passed": (
            p2_payload.get("decision") != "PASS_P2_V2_ALG1_UKF_CONTRACTS_PENDING_CLAUDE_REVIEW"
        ),
        "row_count_or_order_mismatch": tuple(ids) != EXPECTED_V2_MODEL_IDS,
        "runnable_row_missing_value_rows": len(rows) != expected_rows,
        "old_ledh_pfpf_ot_runtime_module_imported": _old_runtime_module_loaded(),
        "old_route_used_as_current_algorithm1_evidence": any(
            row["method_id"] == "ledh_pfpf_alg1_ukf_no_resampling_tf"
            and "ledh_pfpf_ot" in str(row["route_fields"]).lower().replace(
                "previous_ledh_pfpf_ot_evidence_status", ""
            )
            for row in rows
        ),
        "algorithm1_route_fields_missing": any(
            row["method_id"] == "ledh_pfpf_alg1_ukf_no_resampling_tf"
            and not _is_augmented_algorithm1_route(row["route_fields"])
            for row in rows
        ),
        "missing_monte_carlo_uncertainty": any(
            cell["model_id"] in runnable_ids
            and cell["seed_count"] != len(VALUE_SEEDS)
            for cell in runnable_cells
        ),
        "unclassified_execution_failure": any(
            not row["finite"] and row["row_status"] != "BLOCKED_REQUIRES_ADAPTER"
            for row in rows
        ),
        "unsupported_comparator_ranked": any(
            cell["comparator_route"] != "exact_kalman_for_lgssm"
            and cell["p3_status"] == "RERUN_ALG1"
            for cell in cells
        ),
        "finite_only_promoted": any(
            cell["p3_status"] == "RERUN_ALG1"
            for cell in cells
        ),
        "value_used_to_promote_gradient": False,
        "algorithm1_gradients_computed": False,
        "ot_or_annealed_transport_used": False,
    }


def _validate_payload(payload: dict[str, Any]) -> None:
    required = {
        "decision",
        "phase",
        "required_v2_model_ids",
        "cells",
        "value_rows",
        "value_summaries",
        "veto_diagnostics",
        "execution_diagnostics",
        "run_manifest",
        "reproducibility_digest",
    }
    missing = required.difference(payload)
    if missing:
        raise P3ValidationError(f"missing payload fields {sorted(missing)}")
    if payload["decision"] != LOCAL_PASS_DECISION:
        raise P3ValidationError(f"P3 decision is not local pass: {payload['decision']}")
    gate = payload.get("gate_definition", {})
    if not gate.get("p2_blocked_carry_forward_allowed"):
        raise P3ValidationError("P3 gate does not allow documented P2 blocked carry-forwards")
    ids = [cell["model_id"] for cell in payload["cells"]]
    if tuple(ids) != EXPECTED_V2_MODEL_IDS:
        raise P3ValidationError(f"cell id gate failed: {ids}")
    if payload["execution_diagnostics"]["old_ledh_pfpf_ot_imported"]:
        raise P3ValidationError("old LEDH-PFPF-OT runtime module imported")
    if payload["execution_diagnostics"]["algorithm1_gradients_computed"]:
        raise P3ValidationError("P3 computed gradients")
    if payload["execution_diagnostics"]["ot_or_annealed_transport_used"]:
        raise P3ValidationError("P3 used OT or annealed transport")
    if any(bool(value) for value in payload["veto_diagnostics"].values()):
        raise P3ValidationError(f"true veto diagnostics: {payload['veto_diagnostics']}")
    for row in payload["value_rows"]:
        if row["method_id"] == "ledh_pfpf_alg1_ukf_no_resampling_tf":
            if not _is_augmented_algorithm1_route(row["route_fields"]):
                raise P3ValidationError(f"{row['model_id']}: missing Algorithm 1 route fields")
            if row["stochastic_score_claim"] != "not_claimed":
                raise P3ValidationError("P3 value row made stochastic score claim")
        if row["row_status"] == "RERUN_ALG1":
            raise P3ValidationError("P3 promoted a finite value row")
    if payload["run_manifest"]["pre_import_cuda_visible_devices"] != "-1":
        raise P3ValidationError("TensorFlow was not forced CPU-only before import")


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# P3 Result: V2 Algorithm 1 UKF Value Replacement",
        "",
        "metadata_date: 2026-06-10",
        "phase: P3",
        f"status: {payload['decision']}",
        "",
        "## Evidence Contract",
        "",
        "| Field | Contract |",
        "| --- | --- |",
        f"| Question | {payload['evidence_contract']['question']} |",
        f"| Baseline/comparator | {payload['evidence_contract']['baseline_comparator']} |",
        f"| Primary criterion | {payload['evidence_contract']['primary_criterion']} |",
        f"| Threshold policy | {payload['threshold_policy']['reason']} |",
        f"| Not concluded | {'; '.join(payload['evidence_contract']['not_concluded'])} |",
        "",
        "## Cells",
        "",
        "| Model | P2 status | P3 status | Rows | Finite rows | Comparator | Reason |",
        "| --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    for cell in payload["cells"]:
        lines.append(
            f"| `{cell['model_id']}` | `{cell['p2_status']}` | `{cell['p3_status']}` | "
            f"`{cell['row_count']}` | `{cell['finite_row_count']}` | "
            f"`{cell['comparator_route']}` | {cell['reason']} |"
        )
    lines.extend(
        [
            "",
            "## Veto Diagnostics",
            "",
            "| Diagnostic | Status |",
            "| --- | --- |",
            *[
                f"| `{key}` | `{value}` |"
                for key, value in payload["veto_diagnostics"].items()
            ],
            "",
            "## Value Summaries",
            "",
        ]
    )
    for model_id, methods in payload["value_summaries"].items():
        lines.append(f"### {model_id}")
        lines.append("")
        lines.append("| Method | Particles | Seeds | Finite | Mean value | SE | RMSE vs reference | Min ESS |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for method_id, by_count in methods.items():
            for count in VALUE_PARTICLE_COUNTS:
                cell = by_count[str(count)]
                lines.append(
                    f"| `{method_id}` | {count} | `{cell['seed_count']}` | "
                    f"`{cell['finite_count']}` | `{cell['mean_value']}` | "
                    f"`{cell['value_standard_error']}` | "
                    f"`{cell['value_rmse_vs_reference']}` | `{cell['min_ess']}` |"
                )
        lines.append("")
    lines.extend(
        [
            "## Decision Table",
            "",
            "| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |",
            "| --- | --- | --- | --- | --- | --- |",
            (
                f"| `{payload['decision']}` | every P2 row appears; P2 runnable rows executed "
                "or explicitly downgraded; P2-blocked rows remain visible with adapter reasons; "
                "finite rows carry uncertainty | "
                f"`{payload['veto_diagnostics']}` | non-LGSSM rows have no exact oracle in P3 | "
                "Claude P3 read-only review, then P4 gradients consume frozen contracts | "
                "no gradient, stochastic-resampling, OT-extension, performance, or production claim |"
            ),
            "",
            "## Gate Definition",
            "",
            f"- Local decision semantics: {payload['gate_definition']['local_decision_semantics']}",
            f"- P2 blocked carry-forward allowed: `{payload['gate_definition']['p2_blocked_carry_forward_allowed']}`",
            f"- P2 blocked carry-forward rule: {payload['gate_definition']['p2_blocked_carry_forward_rule']}",
            f"- P2 runnable rule: {payload['gate_definition']['p2_runnable_rule']}",
            f"- Promotion rule: {payload['gate_definition']['promotion_rule']}",
            "",
            "## Post-Run Red-Team Note",
            "",
            "Strongest alternative explanation: finite non-LGSSM values demonstrate execution only, not correctness, because P3 lacks exact nonlinear oracles.",
            "",
            "Result that would overturn the local decision: Claude finds old-route leakage, missing P2 consumption, missing uncertainty, or a finite-only promotion.",
            "",
            "Weakest part of the evidence: P3 is diagnostic-only by design and does not calibrate thresholds.",
            "",
            "## Run Manifest",
            "",
            f"- command: `{payload['run_manifest']['command']}`",
            f"- git branch: `{payload['run_manifest']['branch']}`",
            f"- git commit: `{payload['run_manifest']['commit']}`",
            f"- CPU/GPU status: CPU-only, `CUDA_VISIBLE_DEVICES={payload['run_manifest']['pre_import_cuda_visible_devices']}` before TensorFlow import",
            f"- visible GPU devices: `{payload['run_manifest']['gpu_devices_visible']}`",
            f"- P2 JSON: `{payload['run_manifest']['p2_json_path']}`",
            f"- P2 contract checksum: `{payload['run_manifest']['p2_contract_bundle_checksum']}`",
            f"- value seeds: `{payload['run_manifest']['value_seed_list']}`",
            f"- value particle counts: `{payload['run_manifest']['value_particle_counts']}`",
            f"- pseudo-time steps: `{payload['run_manifest']['pseudo_time_steps']}`",
            f"- UKF parameters: `{payload['run_manifest']['ukf_parameters']}`",
            f"- JSON: `{payload['run_manifest']['json_path']}`",
            f"- report: `{payload['run_manifest']['report_path']}`",
            f"- result: `{payload['run_manifest']['result_path']}`",
            f"- wall time seconds: `{payload['run_manifest']['wall_time_seconds']}`",
            "",
            "## Nonclaims",
            "",
            *[f"- {item}" for item in payload["nonclaims"]],
            "",
            "## Gate Status",
            "",
            "P3 is pending Claude read-only review.",
            "",
        ]
    )
    return "\n".join(lines)


def _bootstrap_diagnostics(result: Any) -> dict[str, Any]:
    ess_values = [float(v) for v in tensor_to_json(result.ess_by_time)]
    return {
        "route_id": result.method_id,
        "resampling_route": "none",
        "resampling_count": int(result.resampling_count),
        "ess_min": min(ess_values),
        "ess_mean": statistics.fmean(ess_values),
        "finite": bool(result.finite),
        "old_ledh_pfpf_ot_used": False,
    }


def _alg1_diagnostics(result: Any) -> dict[str, Any]:
    validate_algorithm1_route_identifiers(result.route_identifiers)
    ess_values = [float(v) for v in tensor_to_json(result.ess_by_time)]
    step_diags = result.resampling_diagnostics
    return {
        "route_id": result.method_id,
        "route_identifiers": dict(result.route_identifiers),
        "resampling_route": result.route_identifiers["resampling_route"],
        "resampling_count": int(result.resampling_count),
        "ess_min": min(ess_values),
        "ess_mean": statistics.fmean(ess_values),
        "finite": bool(result.finite),
        "pseudo_time_step_count": len(PSEUDO_TIME_STEPS),
        "min_predicted_covariance_eigenvalue": min(
            float(diag["min_predicted_covariance_eigenvalue"]) for diag in step_diags
        ),
        "max_prediction_floor_count": max(
            int(diag["max_prediction_floor_count"]) for diag in step_diags
        ),
        "min_forward_log_det": min(float(diag["min_forward_log_det"]) for diag in step_diags),
        "max_forward_log_det": max(float(diag["max_forward_log_det"]) for diag in step_diags),
        "finite_forward_log_det": all(bool(diag["finite_forward_log_det"]) for diag in step_diags),
        "finite_updated_covariances": all(bool(diag["finite_updated_covariances"]) for diag in step_diags),
        "old_ledh_pfpf_ot_used": False,
    }


def _summary(cells: list[dict[str, Any]], summaries: dict[str, Any]) -> dict[str, Any]:
    return {
        "status_counts": {
            status: sum(1 for cell in cells if cell["p3_status"] == status)
            for status in sorted({cell["p3_status"] for cell in cells})
        },
        "models": [cell["model_id"] for cell in cells],
        "summary_models": sorted(summaries),
        "promoted_rows": [],
    }


def _preflight(p2_payload: dict[str, Any]) -> None:
    if p2_payload.get("decision") != "PASS_P2_V2_ALG1_UKF_CONTRACTS_PENDING_CLAUDE_REVIEW":
        raise P3ValidationError("P2 contract artifact has not passed local execution")
    ids = [contract.get("model_id") for contract in p2_payload.get("contracts", [])]
    if tuple(ids) != EXPECTED_V2_MODEL_IDS:
        raise P3ValidationError(f"P2 contract id gate failed: {ids}")
    if any(bool(value) for value in p2_payload.get("veto_diagnostics", {}).values()):
        raise P3ValidationError("P2 artifact contains true veto diagnostics")


def _initial_mean(spec: CommonModelSpecV2) -> tf.Tensor:
    p = spec.parameters
    if spec.model_id in {"lgssm_2d_h25_rich", "range_bearing_4d_h20_rich"}:
        return tf.convert_to_tensor(p["m0"], DTYPE)
    if spec.model_id in {"spatial_sir_j3_rk4", "predator_prey_rk4"}:
        return tf.convert_to_tensor(p["initial_mean"], DTYPE)
    raise ValueError(f"{spec.model_id}: no initial mean contract")


def _initial_covariance(spec: CommonModelSpecV2) -> tf.Tensor:
    p = spec.parameters
    if spec.model_id in {"lgssm_2d_h25_rich", "range_bearing_4d_h20_rich"}:
        return tf.convert_to_tensor(p["P0"], DTYPE)
    if spec.model_id in {"spatial_sir_j3_rk4", "predator_prey_rk4"}:
        return tf.convert_to_tensor(p["initial_covariance"], DTYPE)
    raise ValueError(f"{spec.model_id}: no initial covariance contract")


def _process_covariance(spec: CommonModelSpecV2) -> tf.Tensor:
    p = spec.parameters
    if spec.model_id in {"lgssm_2d_h25_rich", "range_bearing_4d_h20_rich"}:
        return tf.convert_to_tensor(p["Q"], DTYPE)
    if spec.model_id in {"spatial_sir_j3_rk4", "predator_prey_rk4"}:
        return tf.convert_to_tensor(p["process_covariance"], DTYPE)
    raise ValueError(f"{spec.model_id}: no process covariance contract")


def _observation_covariance(spec: CommonModelSpecV2) -> tf.Tensor:
    p = spec.parameters
    if spec.model_id == "lgssm_2d_h25_rich":
        return tf.convert_to_tensor(p["R"], DTYPE)
    if spec.model_id == "range_bearing_4d_h20_rich":
        return tf.convert_to_tensor(p["R"], DTYPE)
    if spec.model_id in {"spatial_sir_j3_rk4", "predator_prey_rk4"}:
        return tf.convert_to_tensor(p["observation_covariance"], DTYPE)
    raise ValueError(f"{spec.model_id}: no observation covariance contract")


def _sir_observation_jacobian(state_dim: int, obs_dim: int) -> tf.Tensor:
    rows = []
    for index in range(obs_dim):
        row = [0.0] * state_dim
        row[2 * index + 1] = 1.0
        rows.append(row)
    return tf.constant(rows, DTYPE)


def _mvn_sample(
    loc: tf.Tensor,
    covariance: tf.Tensor,
    seed: int,
    salt: int,
    sample_shape: tuple[int, ...],
) -> tf.Tensor:
    loc = tf.convert_to_tensor(loc, DTYPE)
    chol = tf.linalg.cholesky(tf.convert_to_tensor(covariance, DTYPE))
    shape = list(sample_shape) + [int(loc.shape[0])]
    normal = tf.random.stateless_normal(shape, seed=_seed_pair(seed, salt), dtype=DTYPE)
    if not sample_shape:
        return loc + tf.linalg.matvec(chol, normal)
    return loc + tf.linalg.matmul(normal, chol, transpose_b=True)


def _gaussian_logpdf_zero_mean(residuals: tf.Tensor, covariance: tf.Tensor) -> tf.Tensor:
    residuals = tf.convert_to_tensor(residuals, DTYPE)
    covariance = _symmetrize(tf.convert_to_tensor(covariance, DTYPE))
    dim = tf.cast(tf.shape(residuals)[-1], DTYPE)
    chol = tf.linalg.cholesky(covariance)
    solved = tf.linalg.triangular_solve(chol, tf.transpose(residuals), lower=True)
    quadratic = tf.reduce_sum(solved * solved, axis=0)
    log_det = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)))
    return -0.5 * (dim * tf.math.log(tf.constant(2.0 * math.pi, DTYPE)) + log_det + quadratic)


def _augmented_algorithm1_route() -> dict[str, str]:
    route = algorithm1_route_identifiers(resampling_route="none")
    validate_algorithm1_route_identifiers(route)
    route.update(
        {
            "prediction_covariance_route": "ukf_prediction_per_particle_covariance",
            "update_covariance_route": "ukf_update_per_particle_covariance",
            "core_resampling_route": "none",
            "extension_resampling_route": "none",
            "evidence_route_class": "SOURCE_ALGORITHM1_CORE",
        }
    )
    return route


def _baseline_route_fields() -> dict[str, str]:
    return {
        "method_generation": "N/A_BASELINE_NOT_ALGORITHM1",
        "flow_source_route": "N/A_BASELINE_NOT_ALGORITHM1",
        "covariance_route": "N/A_BASELINE_NOT_ALGORITHM1",
        "prediction_covariance_route": "N/A_BASELINE_NOT_ALGORITHM1",
        "update_covariance_route": "N/A_BASELINE_NOT_ALGORITHM1",
        "flow_anchor_route": "N/A_BASELINE_NOT_ALGORITHM1",
        "resampling_route": "none",
        "core_resampling_route": "none",
        "extension_resampling_route": "none",
        "evidence_route_class": "BASELINE_NOT_ALGORITHM1",
        "previous_ledh_pfpf_ot_evidence_status": "quarantined",
    }


def _is_augmented_algorithm1_route(route: dict[str, str]) -> bool:
    expected = _augmented_algorithm1_route()
    return all(route.get(key) == value for key, value in expected.items())


def _old_runtime_module_loaded() -> bool:
    return any(
        name.endswith(".ledh_pfpf_ot_tf")
        or name.endswith(".run_v2_ledh_pfpf_ot_values_tf")
        or name.endswith(".run_v2_ledh_pfpf_ot_contracts_tf")
        for name in sys.modules
    )


def _ukf_parameters() -> dict[str, float]:
    return {"alpha": UKF_ALPHA, "beta": UKF_BETA, "kappa": UKF_KAPPA}


def _standard_error(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return statistics.stdev(values) / math.sqrt(len(values))


def _ci95(values: list[float]) -> list[float] | None:
    if len(values) < 2:
        return None
    mean = statistics.fmean(values)
    se = _standard_error(values)
    assert se is not None
    return [mean - 1.96 * se, mean + 1.96 * se]


def _rmse(values: list[float]) -> float:
    return math.sqrt(statistics.fmean([value * value for value in values]))


def _symmetrize(matrix: tf.Tensor) -> tf.Tensor:
    return 0.5 * (matrix + tf.transpose(matrix))


def _seed_pair(seed: int, salt: int) -> tf.Tensor:
    return tf.constant([int(seed) % 2147483647, int(salt) % 2147483647], dtype=tf.int32)


def _digest_payload(payload: dict[str, Any]) -> str:
    stable = {
        key: value
        for key, value in payload.items()
        if key not in {"created_at_utc", "run_manifest", "reproducibility_digest"}
    }
    return stable_digest(stable)


def _nonclaims() -> list[str]:
    return [
        "P3 value rows are diagnostic-only and do not certify numerical closeness.",
        "P3 value rows do not imply gradient correctness.",
        "P3 does not use OT or annealed transport.",
        "P3 does not establish stochastic-resampling correctness.",
        "P3 blocked rows are adapter work items, not negative scientific evidence.",
        "P3 does not establish production readiness, HMC readiness, GPU performance, or broad model superiority.",
    ]


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
