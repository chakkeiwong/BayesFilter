"""Run V2 Algorithm 1 UKF LEDH-PFPF diagnostic gradient replacements."""

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
from collections.abc import Callable
from typing import Any

import tensorflow as tf

from bayesfilter.linear.kalman_qr_derivatives_tf import (
    tf_qr_linear_gaussian_score_hessian,
)
from bayesfilter.linear.types_tf import (
    TFLinearGaussianStateSpace,
    TFLinearGaussianStateSpaceDerivatives,
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
    common_model_specs_v2,
    gaussian_logpdf_zero_mean_tf,
    observation_residual_tf,
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
    "run_v2_ledh_pfpf_alg1_ukf_gradients_tf"
)
PLAN_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-ledh-pfpf-alg1-ukf-ledh-pfpf-ot-rerun-p4-v2-gradients-subplan-2026-06-10.md"
)
RESULT_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-ledh-pfpf-alg1-ukf-ledh-pfpf-ot-rerun-p4-v2-gradients-result-2026-06-10.md"
)
REVIEW_LEDGER_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-ledh-pfpf-alg1-ukf-ledh-pfpf-ot-rerun-claude-review-ledger-2026-06-10.md"
)
P2_JSON_PATH = OUTPUT_DIR / "dpf_v2_ledh_pfpf_alg1_ukf_contracts_2026-06-10.json"
P3_JSON_PATH = OUTPUT_DIR / "dpf_v2_ledh_pfpf_alg1_ukf_values_2026-06-10.json"
JSON_PATH = OUTPUT_DIR / "dpf_v2_ledh_pfpf_alg1_ukf_gradients_2026-06-10.json"
REPORT_PATH = REPORT_DIR / "dpf-v2-ledh-pfpf-alg1-ukf-gradients-2026-06-10.md"

METHOD_ID = "ledh_pfpf_alg1_ukf_no_resampling_tf"
LOCAL_PASS_DECISION = (
    "LOCAL_PASS_P4_V2_ALG1_UKF_GRADIENTS_DIAGNOSTIC_ONLY_PENDING_CLAUDE_REVIEW"
)
VETO_DECISION = "P4_V2_ALG1_UKF_GRADIENTS_VETO_PENDING_REVIEW"
GRADIENT_SEEDS = [101, 202, 303]
GRADIENT_PARTICLE_COUNTS = [4, 8, 16]
PSEUDO_TIME_STEPS = [0.5, 0.5]
UKF_ALPHA = 1.0
UKF_BETA = 2.0
UKF_KAPPA = 0.0
COVARIANCE_FLOOR = 1e-10
RANK_TOLERANCE = 1e-12
FD_STEP = 1e-5


class P4ValidationError(ValueError):
    """Raised when the P4 payload violates the frozen gradient contract."""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only:
        _validate_payload(load_json(JSON_PATH))
        print("P4_V2_LEDHPFPF_ALG1_UKF_GRADIENTS_VALIDATED")
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
    p3_payload = load_json(P3_JSON_PATH)
    _preflight(p2_payload, p3_payload)
    specs = common_model_specs_v2()
    spec_by_id = {spec.model_id: spec for spec in specs}
    cells = []
    gradient_rows = []
    fd_rows = []
    for contract in p2_payload["contracts"]:
        spec = spec_by_id[str(contract["model_id"])]
        gradient_contract = contract["p4_gradient_contract"]
        if contract["status"] != "RUNNABLE_ALG1":
            cells.append(_blocked_cell(spec, contract, "p2_blocked_carry_forward"))
            continue
        if not gradient_contract["runnable"]:
            cells.append(_blocked_cell(spec, contract, "no_reviewed_same_scalar_gradient_contract"))
            continue
        rows = _run_contract_gradient_rows(spec, contract)
        gradient_rows.extend(rows)
        fd_row = _finite_difference_row(spec, contract)
        fd_rows.append(fd_row)
        cells.append(_gradient_cell(spec, contract, rows, fd_row))
    summaries = _gradient_summaries(gradient_rows)
    veto = _veto_diagnostics(p2_payload, p3_payload, cells, gradient_rows, fd_rows)
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
            "p3_json_path": str(P3_JSON_PATH.relative_to(REPO_ROOT)),
            "json_path": str(JSON_PATH.relative_to(REPO_ROOT)),
            "report_path": str(REPORT_PATH.relative_to(REPO_ROOT)),
            "p2_contract_bundle_checksum": p2_payload["contract_bundle_checksum"],
            "gradient_seed_list": list(GRADIENT_SEEDS),
            "gradient_particle_counts": list(GRADIENT_PARTICLE_COUNTS),
            "pseudo_time_steps": list(PSEUDO_TIME_STEPS),
            "ukf_parameters": _ukf_parameters(),
            "covariance_floor": COVARIANCE_FLOOR,
            "rank_tolerance": RANK_TOLERANCE,
            "finite_difference_step": FD_STEP,
        }
    )
    return {
        "metadata_date": "2026-06-10",
        "created_at_utc": utc_now(),
        "phase": "P4",
        "execution_route": "VISIBLE_IN_DIALOGUE",
        "decision": decision,
        "skeptical_plan_audit": {
            "status": "PASS_FOR_DIAGNOSTIC_FIXED_BRANCH_GRADIENT_RERUN",
            "wrong_baseline_control": (
                "P4 consumes frozen P2/P3 artifacts.  Exact Kalman gradient is "
                "used only for the LGSSM row; other finite differences are "
                "diagnostic-only and cannot promote correctness."
            ),
            "proxy_metric_control": (
                "Finite AD gradients, finite differences, cosine similarity, "
                "and particle-ladder trends cannot promote stochastic score "
                "correctness or HMC readiness in P4."
            ),
            "stop_conditions": (
                "Rows without a reviewed same-scalar gradient contract remain "
                "blocked; execution failures are classified instead of dropped."
            ),
        },
        "evidence_contract": {
            "question": (
                "For V2 rows with valid gradient estimands, do Algorithm 1 UKF "
                "fixed-branch gradients execute finitely with uncertainty and "
                "without value-to-gradient promotion?"
            ),
            "baseline_comparator": (
                "P2 contracts, P3 value scalar, exact LGSSM Kalman score where "
                "available, and same-scalar finite differences as diagnostic-only "
                "checks on non-LGSSM gradient-runnable rows."
            ),
            "primary_criterion": (
                "Every P2 row appears; P2/P4 gradient-runnable rows are executed "
                "or explicitly downgraded; rows without a reviewed same-scalar "
                "gradient contract remain blocked with reasons; finite gradients "
                "include uncertainty, route fields, scalar and parameterization "
                "identifiers, and no promotion claim."
            ),
            "veto_diagnostics": list(veto.keys()),
            "not_concluded": _nonclaims(),
        },
        "gate_definition": {
            "local_decision_semantics": (
                "LOCAL_PASS means the P4 artifact satisfies the local pre-Claude "
                "diagnostic gradient gate.  It is not an unconditional pass, "
                "correctness promotion, stochastic-score proof, or HMC-readiness claim."
            ),
            "p2_blocked_carry_forward_allowed": True,
            "gradient_contract_block_allowed": True,
            "p2_runnable_gradient_rule": (
                "A P2 value-runnable row may still be gradient-blocked if "
                "P2 did not declare a reviewed same-scalar gradient contract."
            ),
            "promotion_rule": (
                "P4 gradient evidence is diagnostic-only.  Finite AD gradients "
                "and finite-difference residuals cannot promote correctness."
            ),
        },
        "required_v2_model_ids": list(EXPECTED_V2_MODEL_IDS),
        "method_id": METHOD_ID,
        "cells": cells,
        "gradient_rows": gradient_rows,
        "finite_difference_rows": fd_rows,
        "gradient_summaries": summaries,
        "veto_diagnostics": veto,
        "execution_diagnostics": {
            "old_ledh_pfpf_ot_imported": _old_runtime_module_loaded(),
            "algorithm1_gradient_rows_executed": len(gradient_rows),
            "value_rows_executed_in_p4": False,
            "algorithm1_values_reused_from_p3": True,
            "ot_or_annealed_transport_used": False,
            "stochastic_resampling_gradient_claim": "not_claimed",
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


def _run_contract_gradient_rows(
    spec: CommonModelSpecV2,
    contract: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for num_particles in GRADIENT_PARTICLE_COUNTS:
        for seed in GRADIENT_SEEDS:
            rows.append(_run_one_gradient_row(spec, contract, seed, num_particles))
    return rows


def _run_one_gradient_row(
    spec: CommonModelSpecV2,
    contract: dict[str, Any],
    seed: int,
    num_particles: int,
) -> dict[str, Any]:
    route_fields = _augmented_algorithm1_route()
    try:
        initial_theta = _initial_gradient_theta(spec)
        with tf.GradientTape() as tape:
            tape.watch(initial_theta)
            value, diagnostics = _algorithm1_value_for_gradient_theta(
                spec=spec,
                theta=initial_theta,
                seed=seed,
                num_particles=num_particles,
            )
        gradient = tape.gradient(value, initial_theta)
        if gradient is None:
            gradient = tf.fill(tf.shape(initial_theta), tf.constant(float("nan"), dtype=DTYPE))
        reference = _reference_gradient(spec, initial_theta)
        reference_tensor = None if reference is None else tf.constant(reference, dtype=DTYPE)
        gradient_error = None if reference_tensor is None else gradient - reference_tensor
        finite = bool(
            tf.math.is_finite(value).numpy()
            and tf.reduce_all(tf.math.is_finite(gradient)).numpy()
            and diagnostics["finite"]
        )
        return {
            "model_id": spec.model_id,
            "contract_id": contract["contract_id"],
            "contract_checksum": contract["contract_checksum"],
            "method_id": METHOD_ID,
            "seed": int(seed),
            "num_particles": int(num_particles),
            "horizon": int(contract["horizon"]),
            "value": scalar(value),
            "fixed_branch_gradient": tensor_to_json(gradient),
            "gradient_parameterization": _gradient_parameterization(spec),
            "gradient_scalar": contract["p4_gradient_contract"]["gradient_scalar"],
            "gradient_scope": contract["p4_gradient_contract"]["gradient_scope"],
            "gradient_reference": reference,
            "gradient_reference_route": _reference_route(spec),
            "gradient_error": None if gradient_error is None else tensor_to_json(gradient_error),
            "gradient_error_norm": None if gradient_error is None else scalar(tf.linalg.norm(gradient_error)),
            "relative_gradient_error": (
                None
                if gradient_error is None or reference_tensor is None
                else scalar(
                    tf.linalg.norm(gradient_error)
                    / tf.maximum(tf.constant(1.0, DTYPE), tf.linalg.norm(reference_tensor))
                )
            ),
            "finite": finite,
            "row_status": "RERUN_ALG1_DIAGNOSTIC_ONLY" if finite else "BLOCKED_REQUIRES_ADAPTER",
            "primary_promote_statistic": "N/A_DIAGNOSTIC_ONLY_IN_P4",
            "gradient_tolerance": contract["threshold_contract"]["gradient_tolerance"],
            "finite_difference_policy": "diagnostic_only_not_a_promotion_gate",
            "branch_freeze_policy": "fixed_seed_no_resampling_gradient_through_realized_algorithm1_value",
            "stochastic_score_claim": "not_claimed",
            "route_fields": route_fields,
            "diagnostics": diagnostics,
        }
    except Exception as exc:  # noqa: BLE001 - row classification preserves the failure.
        return {
            "model_id": spec.model_id,
            "contract_id": contract["contract_id"],
            "contract_checksum": contract["contract_checksum"],
            "method_id": METHOD_ID,
            "seed": int(seed),
            "num_particles": int(num_particles),
            "horizon": int(contract["horizon"]),
            "value": None,
            "fixed_branch_gradient": None,
            "gradient_parameterization": _gradient_parameterization(spec),
            "gradient_scalar": contract["p4_gradient_contract"]["gradient_scalar"],
            "gradient_scope": contract["p4_gradient_contract"]["gradient_scope"],
            "gradient_reference": None,
            "gradient_reference_route": _reference_route(spec),
            "gradient_error": None,
            "gradient_error_norm": None,
            "relative_gradient_error": None,
            "finite": False,
            "row_status": "BLOCKED_REQUIRES_ADAPTER",
            "primary_promote_statistic": "N/A_BLOCKED_REQUIRES_ADAPTER",
            "gradient_tolerance": "N/A_BLOCKED_REQUIRES_ADAPTER",
            "finite_difference_policy": "diagnostic_only_not_a_promotion_gate",
            "branch_freeze_policy": "fixed_seed_no_resampling_gradient_through_realized_algorithm1_value",
            "stochastic_score_claim": "not_claimed",
            "route_fields": route_fields,
            "diagnostics": {
                "finite": False,
                "execution_error_type": type(exc).__name__,
                "execution_error": str(exc),
                "traceback_tail": traceback.format_exc(limit=3).splitlines()[-6:],
                "old_ledh_pfpf_ot_used": False,
            },
        }


def _algorithm1_value_for_gradient_theta(
    *,
    spec: CommonModelSpecV2,
    theta: tf.Tensor,
    seed: int,
    num_particles: int,
) -> tuple[tf.Tensor, dict[str, Any]]:
    callbacks = _gradient_callbacks(spec, theta)
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
        method_id=METHOD_ID,
    )
    return result.log_likelihood_estimate, _alg1_diagnostics(result)


def _gradient_callbacks(spec: CommonModelSpecV2, theta: tf.Tensor) -> dict[str, Any]:
    params = spec.parameters
    observations = tf.convert_to_tensor(spec.path_contract["observations"], DTYPE)
    initial_mean = _initial_mean(spec)
    initial_covariance = _initial_covariance(spec)
    process_covariance = _process_covariance(spec)
    observation_covariance = _observation_covariance_for_theta(spec, theta)

    def initial_sample(num_particles: int, seed: int) -> tf.Tensor:
        return _mvn_sample(initial_mean, initial_covariance, seed, 11, (num_particles,))

    def transition_mean_fn(points: tf.Tensor, _time_index: int) -> tf.Tensor:
        points = tf.convert_to_tensor(points, DTYPE)
        if spec.model_id == "lgssm_2d_h25_rich":
            scale = tf.reshape(tf.cast(theta, DTYPE), [-1])[0]
            return tf.linalg.matmul(points, tf.convert_to_tensor(params["A"], DTYPE) * scale, transpose_b=True)
        if spec.model_id == "range_bearing_4d_h20_rich":
            return tf.linalg.matmul(points, tf.convert_to_tensor(params["A"], DTYPE), transpose_b=True)
        if spec.model_id == "predator_prey_rk4":
            return _predator_prey_model(spec).transition_mean(_predator_theta(spec, theta), points)
        raise ValueError(f"{spec.model_id}: no P4 transition_mean_fn contract")

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
        if spec.model_id == "lgssm_2d_h25_rich":
            scale = tf.reshape(tf.cast(theta, DTYPE), [-1])[0]
            mean = tf.linalg.matmul(previous, tf.convert_to_tensor(params["A"], DTYPE) * scale, transpose_b=True)
            return gaussian_logpdf_zero_mean_tf(current - mean, process_covariance)
        if spec.model_id == "range_bearing_4d_h20_rich":
            mean = tf.linalg.matmul(previous, tf.convert_to_tensor(params["A"], DTYPE), transpose_b=True)
            return gaussian_logpdf_zero_mean_tf(current - mean, process_covariance)
        if spec.model_id == "predator_prey_rk4":
            model = _predator_prey_model(spec)
            return tf.cast(
                model.transition_log_density(
                    _predator_theta(spec, theta), previous, current, t=int(time_index) + 1
                ),
                DTYPE,
            )
        raise ValueError(f"{spec.model_id}: no P4 transition_log_density_fn contract")

    def observation_mean_fn(points: tf.Tensor, _time_index: int) -> tf.Tensor:
        points = tf.convert_to_tensor(points, DTYPE)
        if spec.model_id == "lgssm_2d_h25_rich":
            return tf.linalg.matmul(points, tf.convert_to_tensor(params["C"], DTYPE), transpose_b=True)
        if spec.model_id == "range_bearing_4d_h20_rich":
            return range_bearing_observation_tf(points)
        if spec.model_id == "predator_prey_rk4":
            return points
        raise ValueError(f"{spec.model_id}: no P4 observation_mean_fn contract")

    def observation_jacobian_fn(point: tf.Tensor, _time_index: int) -> tf.Tensor:
        point = tf.reshape(tf.convert_to_tensor(point, DTYPE), [-1])
        if spec.model_id == "lgssm_2d_h25_rich":
            return tf.convert_to_tensor(params["C"], DTYPE)
        if spec.model_id == "range_bearing_4d_h20_rich":
            return range_bearing_jacobian_tf(point)
        if spec.model_id == "predator_prey_rk4":
            return tf.eye(spec.state_dim, dtype=DTYPE)
        raise ValueError(f"{spec.model_id}: no P4 observation_jacobian_fn contract")

    def observation_log_density_fn(particles: tf.Tensor, observation: tf.Tensor, _time_index: int) -> tf.Tensor:
        particles = tf.convert_to_tensor(particles, DTYPE)
        observation = tf.reshape(tf.convert_to_tensor(observation, DTYPE), [-1])
        if spec.model_id == "lgssm_2d_h25_rich":
            pred = tf.linalg.matmul(particles, tf.convert_to_tensor(params["C"], DTYPE), transpose_b=True)
            residual = pred - observation[tf.newaxis, :]
            return gaussian_logpdf_zero_mean_tf(residual, observation_covariance)
        if spec.model_id == "range_bearing_4d_h20_rich":
            pred = range_bearing_observation_tf(particles)
            residual = observation_residual_tf(pred, observation)
            return gaussian_logpdf_zero_mean_tf(residual, observation_covariance)
        if spec.model_id == "predator_prey_rk4":
            residual = particles - observation[tf.newaxis, :]
            return gaussian_logpdf_zero_mean_tf(residual, observation_covariance)
        raise ValueError(f"{spec.model_id}: no P4 observation_log_density_fn contract")

    return {
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


def _initial_gradient_theta(spec: CommonModelSpecV2) -> tf.Tensor:
    knobs = [knob for knob in spec.gradient_contract["knobs"] if bool(knob.get("include"))]
    if spec.model_id == "lgssm_2d_h25_rich":
        return tf.constant([1.0, 1.0], DTYPE)
    if spec.model_id == "range_bearing_4d_h20_rich":
        return tf.constant([0.12, 0.04], DTYPE)
    if spec.model_id == "predator_prey_rk4":
        return tf.constant([float(knobs[0]["initial_value"])], DTYPE)
    raise ValueError(f"{spec.model_id}: no P4 gradient theta contract")


def _gradient_parameterization(spec: CommonModelSpecV2) -> dict[str, Any]:
    knobs = [dict(knob) for knob in spec.gradient_contract["knobs"] if bool(knob.get("include"))]
    return {
        "knob_names": [knob["name"] for knob in knobs],
        "initial_values": [knob.get("initial_value") for knob in knobs],
        "parameterization": [knob.get("parameterization") for knob in knobs],
        "scalar": spec.gradient_contract["gradient_scalar"],
    }


def _finite_difference_row(spec: CommonModelSpecV2, contract: dict[str, Any]) -> dict[str, Any]:
    initial = _initial_gradient_theta(spec)
    try:
        fd = _finite_difference_gradient(
            lambda value: _algorithm1_value_for_gradient_theta(
                spec=spec,
                theta=value,
                seed=GRADIENT_SEEDS[0],
                num_particles=GRADIENT_PARTICLE_COUNTS[0],
            )[0],
            initial,
        )
        return {
            "model_id": spec.model_id,
            "contract_id": contract["contract_id"],
            "finite_difference_gradient": tensor_to_json(fd),
            "finite": bool(tf.reduce_all(tf.math.is_finite(fd)).numpy()),
            "finite_difference_step": FD_STEP,
            "policy": "diagnostic_only_not_a_promotion_gate",
        }
    except Exception as exc:  # noqa: BLE001 - diagnostic-only row preserves failure.
        return {
            "model_id": spec.model_id,
            "contract_id": contract["contract_id"],
            "finite_difference_gradient": None,
            "finite": False,
            "finite_difference_step": FD_STEP,
            "policy": "diagnostic_only_not_a_promotion_gate",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _finite_difference_gradient(
    value_fn: Callable[[tf.Tensor], tf.Tensor],
    theta: tf.Tensor,
) -> tf.Tensor:
    values = []
    for index in range(int(theta.shape[0])):
        basis = tf.one_hot(index, int(theta.shape[0]), dtype=DTYPE)
        plus = theta + tf.constant(FD_STEP, DTYPE) * basis
        minus = theta - tf.constant(FD_STEP, DTYPE) * basis
        values.append((value_fn(plus) - value_fn(minus)) / tf.constant(2.0 * FD_STEP, DTYPE))
    return tf.stack(values)


def _reference_gradient(spec: CommonModelSpecV2, theta: tf.Tensor) -> list[float] | None:
    if spec.model_id != "lgssm_2d_h25_rich":
        return None
    return _lgssm_kalman_reference(spec, theta)["score"]


def _reference_route(spec: CommonModelSpecV2) -> str:
    if spec.model_id == "lgssm_2d_h25_rich":
        return "exact_kalman_lgssm_gradient"
    return "diagnostic_finite_difference_only_no_exact_gradient_oracle"


def _lgssm_kalman_reference(spec: CommonModelSpecV2, theta: tf.Tensor) -> dict[str, Any]:
    transition_scale, observation_scale = tf.unstack(tf.cast(theta, DTYPE))
    params = spec.parameters
    a = tf.convert_to_tensor(params["A"], DTYPE) * transition_scale
    c = tf.convert_to_tensor(params["C"], DTYPE)
    q = tf.convert_to_tensor(params["Q"], DTYPE)
    r = tf.convert_to_tensor(params["R"], DTYPE) * observation_scale
    observations = tf.convert_to_tensor(spec.path_contract["observations"], DTYPE)
    model = TFLinearGaussianStateSpace(
        initial_mean=tf.convert_to_tensor(params["m0"], DTYPE),
        initial_covariance=tf.convert_to_tensor(params["P0"], DTYPE),
        transition_offset=tf.zeros([2], dtype=DTYPE),
        transition_matrix=a,
        transition_covariance=q,
        observation_offset=tf.zeros([1], dtype=DTYPE),
        observation_matrix=c,
        observation_covariance=r,
    )
    zero_n = tf.zeros([2, 2], dtype=DTYPE)
    zero_m = tf.zeros([2, 1], dtype=DTYPE)
    derivatives = TFLinearGaussianStateSpaceDerivatives(
        d_initial_mean=zero_n,
        d_initial_covariance=tf.zeros([2, 2, 2], dtype=DTYPE),
        d_transition_offset=zero_n,
        d_transition_matrix=tf.stack([a / theta[0], tf.zeros_like(a)], axis=0),
        d_transition_covariance=tf.zeros([2, 2, 2], dtype=DTYPE),
        d_observation_offset=zero_m,
        d_observation_matrix=tf.zeros([2, 1, 2], dtype=DTYPE),
        d_observation_covariance=tf.stack([tf.zeros_like(r), r / theta[1]], axis=0),
        d2_initial_mean=tf.zeros([2, 2, 2], dtype=DTYPE),
        d2_initial_covariance=tf.zeros([2, 2, 2, 2], dtype=DTYPE),
        d2_transition_offset=tf.zeros([2, 2, 2], dtype=DTYPE),
        d2_transition_matrix=tf.zeros([2, 2, 2, 2], dtype=DTYPE),
        d2_transition_covariance=tf.zeros([2, 2, 2, 2], dtype=DTYPE),
        d2_observation_offset=tf.zeros([2, 2, 1], dtype=DTYPE),
        d2_observation_matrix=tf.zeros([2, 2, 1, 2], dtype=DTYPE),
        d2_observation_covariance=tf.zeros([2, 2, 1, 1], dtype=DTYPE),
    )
    result = tf_qr_linear_gaussian_score_hessian(
        observations,
        model,
        derivatives,
        backend="tf_qr_sqrt",
        jitter=tf.constant(1e-9, dtype=DTYPE),
    )
    return {
        "reference_id": "tf_qr_sqrt_differentiated_kalman",
        "horizon": int(observations.shape[0]),
        "log_likelihood": scalar(result.log_likelihood),
        "score": tensor_to_json(tf.cast(result.score, DTYPE)),
        "finite": bool(
            tf.math.is_finite(result.log_likelihood).numpy()
            and tf.reduce_all(tf.math.is_finite(result.score)).numpy()
        ),
    }


def _gradient_cell(
    spec: CommonModelSpecV2,
    contract: dict[str, Any],
    rows: list[dict[str, Any]],
    fd_row: dict[str, Any],
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
        "p4_status": status,
        "row_count": len(rows),
        "finite_row_count": len(finite_rows),
        "seed_count": len(GRADIENT_SEEDS),
        "particle_ladder": list(GRADIENT_PARTICLE_COUNTS),
        "gradient_scalar": contract["p4_gradient_contract"]["gradient_scalar"],
        "gradient_reference_route": _reference_route(spec),
        "finite_difference_finite": bool(fd_row["finite"]),
        "gradient_tolerance": contract["threshold_contract"]["gradient_tolerance"],
        "reason": (
            "finite diagnostic fixed-branch gradient rows only; no calibrated promotion band"
            if status == "RERUN_ALG1_DIAGNOSTIC_ONLY"
            else "one or more gradient rows failed and are classified with diagnostics"
        ),
    }


def _blocked_cell(spec: CommonModelSpecV2, contract: dict[str, Any], blocked_route: str) -> dict[str, Any]:
    missing = list(contract.get("missing_adapter_items", []))
    if blocked_route == "no_reviewed_same_scalar_gradient_contract":
        missing = [
            "reviewed same-scalar physical-gradient parameterization",
            "gradient callback contract binding declared knobs to Algorithm 1 value scalar",
        ]
    return {
        "model_id": spec.model_id,
        "contract_id": contract["contract_id"],
        "contract_checksum": contract["contract_checksum"],
        "p2_status": contract["status"],
        "p4_status": "BLOCKED_REQUIRES_ADAPTER",
        "row_count": 0,
        "finite_row_count": 0,
        "seed_count": "N/A",
        "particle_ladder": "N/A",
        "gradient_scalar": contract["p4_gradient_contract"]["gradient_scalar"],
        "gradient_reference_route": "N/A_BLOCKED_REQUIRES_ADAPTER",
        "finite_difference_finite": "N/A",
        "gradient_tolerance": "N/A_BLOCKED_REQUIRES_ADAPTER",
        "blocked_route": blocked_route,
        "missing_adapter_items": missing,
        "reason": (
            contract["status_reason"]
            if blocked_route == "p2_blocked_carry_forward"
            else "P2 did not declare this row runnable for same-scalar P4 gradients."
        ),
    }


def _gradient_summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for model_id in sorted({row["model_id"] for row in rows}):
        summaries[model_id] = {}
        for count in GRADIENT_PARTICLE_COUNTS:
            subset = [
                row
                for row in rows
                if row["model_id"] == model_id and row["num_particles"] == count
            ]
            norms = [
                float(row["gradient_error_norm"])
                for row in subset
                if row["gradient_error_norm"] is not None
            ]
            gradients = [
                [float(v) for v in row["fixed_branch_gradient"]]
                for row in subset
                if row["fixed_branch_gradient"] is not None
            ]
            summaries[model_id][str(count)] = {
                "seed_count": len(subset),
                "finite_count": sum(1 for row in subset if bool(row["finite"])),
                "mean_gradient": _component_mean(gradients),
                "gradient_component_standard_error": _component_standard_error(gradients),
                "mean_gradient_error_norm": statistics.fmean(norms) if norms else None,
                "gradient_error_norm_standard_error": _standard_error(norms),
                "gradient_error_norm_ci95": _ci95(norms),
                "row_status": "RERUN_ALG1_DIAGNOSTIC_ONLY",
            }
    return summaries


def _veto_diagnostics(
    p2_payload: dict[str, Any],
    p3_payload: dict[str, Any],
    cells: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    fd_rows: list[dict[str, Any]],
) -> dict[str, bool]:
    ids = [cell["model_id"] for cell in cells]
    runnable_ids = {
        contract["model_id"]
        for contract in p2_payload["contracts"]
        if contract["status"] == "RUNNABLE_ALG1"
        and bool(contract["p4_gradient_contract"]["runnable"])
    }
    expected_rows = len(runnable_ids) * len(GRADIENT_SEEDS) * len(GRADIENT_PARTICLE_COUNTS)
    return {
        "p2_contract_absent_or_not_passed": (
            p2_payload.get("decision") != "PASS_P2_V2_ALG1_UKF_CONTRACTS_PENDING_CLAUDE_REVIEW"
        ),
        "p3_values_not_ready": not str(p3_payload.get("decision", "")).startswith("LOCAL_PASS_P3_"),
        "row_count_or_order_mismatch": tuple(ids) != EXPECTED_V2_MODEL_IDS,
        "runnable_gradient_row_missing_rows": len(rows) != expected_rows,
        "old_ledh_pfpf_ot_runtime_module_imported": _old_runtime_module_loaded(),
        "old_route_used_as_current_algorithm1_evidence": any(
            row["method_id"] == METHOD_ID
            and "ledh_pfpf_ot" in str(row["route_fields"]).lower().replace(
                "previous_ledh_pfpf_ot_evidence_status", ""
            )
            for row in rows
        ),
        "algorithm1_route_fields_missing": any(
            row["method_id"] == METHOD_ID
            and not _is_augmented_algorithm1_route(row["route_fields"])
            for row in rows
        ),
        "gradient_row_nonfinite": any(not bool(row["finite"]) for row in rows),
        "missing_gradient_monte_carlo_uncertainty": any(
            cell["model_id"] in runnable_ids
            and cell["seed_count"] != len(GRADIENT_SEEDS)
            for cell in cells
        ),
        "gradient_scalar_or_parameterization_missing": any(
            not row["gradient_scalar"]
            or not row["gradient_parameterization"]["knob_names"]
            for row in rows
        ),
        "finite_difference_as_promotion_gate": any(
            fd_row["policy"] != "diagnostic_only_not_a_promotion_gate"
            for fd_row in fd_rows
        ),
        "value_used_to_promote_gradient": False,
        "stochastic_resampling_gradient_claimed": any(
            row["stochastic_score_claim"] != "not_claimed" for row in rows
        ),
        "ot_or_annealed_transport_used": False,
    }


def _validate_payload(payload: dict[str, Any]) -> None:
    required = {
        "decision",
        "phase",
        "required_v2_model_ids",
        "cells",
        "gradient_rows",
        "finite_difference_rows",
        "gradient_summaries",
        "veto_diagnostics",
        "execution_diagnostics",
        "run_manifest",
        "reproducibility_digest",
    }
    missing = required.difference(payload)
    if missing:
        raise P4ValidationError(f"missing payload fields {sorted(missing)}")
    if payload["decision"] != LOCAL_PASS_DECISION:
        raise P4ValidationError(f"P4 decision is not local pass: {payload['decision']}")
    ids = [cell["model_id"] for cell in payload["cells"]]
    if tuple(ids) != EXPECTED_V2_MODEL_IDS:
        raise P4ValidationError(f"cell id gate failed: {ids}")
    if payload["execution_diagnostics"]["old_ledh_pfpf_ot_imported"]:
        raise P4ValidationError("old LEDH-PFPF-OT runtime module imported")
    if payload["execution_diagnostics"]["ot_or_annealed_transport_used"]:
        raise P4ValidationError("P4 used OT or annealed transport")
    if any(bool(value) for value in payload["veto_diagnostics"].values()):
        raise P4ValidationError(f"true veto diagnostics: {payload['veto_diagnostics']}")
    for row in payload["gradient_rows"]:
        if row["method_id"] != METHOD_ID:
            raise P4ValidationError("unexpected P4 method id")
        if not row["finite"]:
            raise P4ValidationError("nonfinite gradient row")
        if row["row_status"] == "RERUN_ALG1":
            raise P4ValidationError("P4 promoted a finite gradient row")
        if row["stochastic_score_claim"] != "not_claimed":
            raise P4ValidationError("P4 made stochastic score claim")
        if not _is_augmented_algorithm1_route(row["route_fields"]):
            raise P4ValidationError("P4 row missing Algorithm 1 route fields")
    if payload["run_manifest"]["pre_import_cuda_visible_devices"] != "-1":
        raise P4ValidationError("TensorFlow was not forced CPU-only before import")


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# P4 Result: V2 Algorithm 1 UKF Gradient Replacement",
        "",
        "metadata_date: 2026-06-10",
        "phase: P4",
        f"status: {payload['decision']}",
        "",
        "## Evidence Contract",
        "",
        "| Field | Contract |",
        "| --- | --- |",
        f"| Question | {payload['evidence_contract']['question']} |",
        f"| Baseline/comparator | {payload['evidence_contract']['baseline_comparator']} |",
        f"| Primary criterion | {payload['evidence_contract']['primary_criterion']} |",
        f"| Not concluded | {'; '.join(payload['evidence_contract']['not_concluded'])} |",
        "",
        "## Cells",
        "",
        "| Model | P2 status | P4 status | Rows | Finite rows | Gradient reference | Reason |",
        "| --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    for cell in payload["cells"]:
        lines.append(
            f"| `{cell['model_id']}` | `{cell['p2_status']}` | `{cell['p4_status']}` | "
            f"`{cell['row_count']}` | `{cell['finite_row_count']}` | "
            f"`{cell['gradient_reference_route']}` | {cell['reason']} |"
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
            "## Gradient Summaries",
            "",
        ]
    )
    for model_id, by_count in payload["gradient_summaries"].items():
        lines.append(f"### {model_id}")
        lines.append("")
        lines.append("| Particles | Seeds | Finite | Mean gradient | Component SE | Mean error norm | Error norm SE |")
        lines.append("| --- | ---: | ---: | --- | --- | ---: | ---: |")
        for count in GRADIENT_PARTICLE_COUNTS:
            summary = by_count[str(count)]
            lines.append(
                f"| {count} | `{summary['seed_count']}` | `{summary['finite_count']}` | "
                f"`{summary['mean_gradient']}` | `{summary['gradient_component_standard_error']}` | "
                f"`{summary['mean_gradient_error_norm']}` | "
                f"`{summary['gradient_error_norm_standard_error']}` |"
            )
        lines.append("")
    lines.extend(
        [
            "## Gate Definition",
            "",
            f"- Local decision semantics: {payload['gate_definition']['local_decision_semantics']}",
            f"- P2 blocked carry-forward allowed: `{payload['gate_definition']['p2_blocked_carry_forward_allowed']}`",
            f"- Gradient-contract block allowed: `{payload['gate_definition']['gradient_contract_block_allowed']}`",
            f"- P2 runnable gradient rule: {payload['gate_definition']['p2_runnable_gradient_rule']}",
            f"- Promotion rule: {payload['gate_definition']['promotion_rule']}",
            "",
            "## Decision Table",
            "",
            "| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |",
            "| --- | --- | --- | --- | --- | --- |",
            (
                f"| `{payload['decision']}` | every P2 row appears; P4-runnable rows executed "
                "or explicitly downgraded; blocked rows remain visible with reasons; "
                "finite gradients carry uncertainty | "
                f"`{payload['veto_diagnostics']}` | non-LGSSM gradients have no exact oracle in P4 | "
                "Claude P4 read-only review, then P5 consumes P2-P4 artifacts | "
                "no stochastic-score, HMC, OT-extension, performance, or production claim |"
            ),
            "",
            "## Post-Run Red-Team Note",
            "",
            "Strongest alternative explanation: finite non-LGSSM gradients demonstrate AD execution only, not correctness, because P4 lacks exact nonlinear gradient oracles.",
            "",
            "Result that would overturn the local decision: Claude finds scalar mismatch, missing route fields, value-to-gradient promotion, old-route leakage, or unsupported gradient rows treated as passes.",
            "",
            "Weakest part of the evidence: P4 is diagnostic-only by design and excludes stochastic resampling gradients.",
            "",
            "## Run Manifest",
            "",
            f"- command: `{payload['run_manifest']['command']}`",
            f"- git branch: `{payload['run_manifest']['branch']}`",
            f"- git commit: `{payload['run_manifest']['commit']}`",
            f"- CPU/GPU status: CPU-only, `CUDA_VISIBLE_DEVICES={payload['run_manifest']['pre_import_cuda_visible_devices']}` before TensorFlow import",
            f"- visible GPU devices: `{payload['run_manifest']['gpu_devices_visible']}`",
            f"- P2 JSON: `{payload['run_manifest']['p2_json_path']}`",
            f"- P3 JSON: `{payload['run_manifest']['p3_json_path']}`",
            f"- gradient seeds: `{payload['run_manifest']['gradient_seed_list']}`",
            f"- gradient particle counts: `{payload['run_manifest']['gradient_particle_counts']}`",
            f"- pseudo-time steps: `{payload['run_manifest']['pseudo_time_steps']}`",
            f"- UKF parameters: `{payload['run_manifest']['ukf_parameters']}`",
            f"- finite-difference step: `{payload['run_manifest']['finite_difference_step']}`",
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
            "P4 is pending Claude read-only review.",
            "",
        ]
    )
    return "\n".join(lines)


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


def _is_augmented_algorithm1_route(route: dict[str, str]) -> bool:
    expected = _augmented_algorithm1_route()
    return all(route.get(key) == value for key, value in expected.items())


def _old_runtime_module_loaded() -> bool:
    return any(
        name.endswith(".ledh_pfpf_ot_tf")
        or name.endswith(".run_v2_ledh_pfpf_ot_gradients_tf")
        or name.endswith(".run_v2_ledh_pfpf_ot_values_tf")
        for name in sys.modules
    )


def _initial_mean(spec: CommonModelSpecV2) -> tf.Tensor:
    p = spec.parameters
    if spec.model_id in {"lgssm_2d_h25_rich", "range_bearing_4d_h20_rich"}:
        return tf.convert_to_tensor(p["m0"], DTYPE)
    if spec.model_id == "predator_prey_rk4":
        return tf.convert_to_tensor(p["initial_mean"], DTYPE)
    raise ValueError(f"{spec.model_id}: no initial mean contract")


def _initial_covariance(spec: CommonModelSpecV2) -> tf.Tensor:
    p = spec.parameters
    if spec.model_id in {"lgssm_2d_h25_rich", "range_bearing_4d_h20_rich"}:
        return tf.convert_to_tensor(p["P0"], DTYPE)
    if spec.model_id == "predator_prey_rk4":
        return tf.convert_to_tensor(p["initial_covariance"], DTYPE)
    raise ValueError(f"{spec.model_id}: no initial covariance contract")


def _process_covariance(spec: CommonModelSpecV2) -> tf.Tensor:
    p = spec.parameters
    if spec.model_id in {"lgssm_2d_h25_rich", "range_bearing_4d_h20_rich"}:
        return tf.convert_to_tensor(p["Q"], DTYPE)
    if spec.model_id == "predator_prey_rk4":
        return tf.convert_to_tensor(p["process_covariance"], DTYPE)
    raise ValueError(f"{spec.model_id}: no process covariance contract")


def _observation_covariance_for_theta(spec: CommonModelSpecV2, theta: tf.Tensor) -> tf.Tensor:
    p = spec.parameters
    if spec.model_id == "lgssm_2d_h25_rich":
        observation_scale = tf.reshape(tf.cast(theta, DTYPE), [-1])[1]
        return tf.convert_to_tensor(p["R"], DTYPE) * observation_scale
    if spec.model_id == "range_bearing_4d_h20_rich":
        sigma_range, sigma_bearing = tf.unstack(tf.reshape(tf.cast(theta, DTYPE), [-1]))
        return tf.linalg.diag(tf.stack([sigma_range * sigma_range, sigma_bearing * sigma_bearing]))
    if spec.model_id == "predator_prey_rk4":
        return tf.convert_to_tensor(p["observation_covariance"], DTYPE)
    raise ValueError(f"{spec.model_id}: no observation covariance contract")


def _predator_prey_model(spec: CommonModelSpecV2) -> Any:
    from bayesfilter import highdim

    p = spec.parameters
    return highdim.PredatorPreySSM(
        initial_mean=p["initial_mean"],
        delta=p["delta"],
        rk4_internal_step=p["rk4_internal_step"],
        process_covariance=p["process_covariance"],
        observation_covariance=p["observation_covariance"],
        initial_covariance=p["initial_covariance"],
        domain_policy=p["domain_policy"],
    )


def _predator_theta(spec: CommonModelSpecV2, gradient_theta: tf.Tensor) -> tf.Tensor:
    base = tf.convert_to_tensor(spec.theta, DTYPE)
    r = tf.reshape(tf.cast(gradient_theta, DTYPE), [-1])[0]
    return tf.concat([r[tf.newaxis], base[1:]], axis=0)


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


def _seed_pair(seed: int, salt: int) -> tf.Tensor:
    return tf.constant([int(seed) % 2147483647, int(salt) % 2147483647], dtype=tf.int32)


def _ukf_parameters() -> dict[str, float]:
    return {"alpha": UKF_ALPHA, "beta": UKF_BETA, "kappa": UKF_KAPPA}


def _preflight(p2_payload: dict[str, Any], p3_payload: dict[str, Any]) -> None:
    if p2_payload.get("decision") != "PASS_P2_V2_ALG1_UKF_CONTRACTS_PENDING_CLAUDE_REVIEW":
        raise P4ValidationError("P2 contract artifact has not passed local execution")
    if not str(p3_payload.get("decision", "")).startswith("LOCAL_PASS_P3_"):
        raise P4ValidationError("P3 value artifact is not ready")
    ids = [contract.get("model_id") for contract in p2_payload.get("contracts", [])]
    if tuple(ids) != EXPECTED_V2_MODEL_IDS:
        raise P4ValidationError(f"P2 contract id gate failed: {ids}")
    if any(bool(value) for value in p2_payload.get("veto_diagnostics", {}).values()):
        raise P4ValidationError("P2 artifact contains true veto diagnostics")
    if any(bool(value) for value in p3_payload.get("veto_diagnostics", {}).values()):
        raise P4ValidationError("P3 artifact contains true veto diagnostics")


def _summary(cells: list[dict[str, Any]], summaries: dict[str, Any]) -> dict[str, Any]:
    return {
        "status_counts": {
            status: sum(1 for cell in cells if cell["p4_status"] == status)
            for status in sorted({cell["p4_status"] for cell in cells})
        },
        "models": [cell["model_id"] for cell in cells],
        "summary_models": sorted(summaries),
        "promoted_rows": [],
    }


def _component_mean(values: list[list[float]]) -> list[float] | None:
    if not values:
        return None
    dim = len(values[0])
    return [statistics.fmean(row[index] for row in values) for index in range(dim)]


def _component_standard_error(values: list[list[float]]) -> list[float] | None:
    if len(values) < 2:
        return None
    dim = len(values[0])
    return [_standard_error([row[index] for row in values]) for index in range(dim)]


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


def _digest_payload(payload: dict[str, Any]) -> str:
    stable = {
        key: value
        for key, value in payload.items()
        if key not in {"created_at_utc", "run_manifest", "reproducibility_digest"}
    }
    return stable_digest(stable)


def _nonclaims() -> list[str]:
    return [
        "P4 gradient rows are diagnostic-only and do not certify numerical gradient correctness.",
        "P4 finite differences are explanatory diagnostics only, not promotion gates.",
        "P4 value evidence from P3 does not imply gradient correctness.",
        "P4 does not use OT or annealed transport.",
        "P4 does not establish stochastic-resampling gradient correctness.",
        "P4 blocked rows are adapter work items, not negative scientific evidence.",
        "P4 does not establish production readiness, HMC readiness, GPU performance, or broad model superiority.",
    ]


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
