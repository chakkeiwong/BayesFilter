"""Run visible P4 bootstrap-OT fixed-branch AD-gradient comparisons."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/bayesfilter-dpf-mpl")
PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]

import argparse
import math
import time
from collections.abc import Callable
from typing import Any

import tensorflow as tf

from experiments.dpf_implementation.tf_tfp.fixtures.common_model_suite_tf import (
    EXPECTED_V2_MODEL_IDS,
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
from experiments.dpf_implementation.tf_tfp.runners import run_v2_bootstrap_ot_values_tf as p3


DTYPE = tf.float64
PLAN_PATH = "docs/plans/bayesfilter-dpf-v2-algorithm-full-bf-filterflow-comparison-p4-bootstrap-ot-gradients-subplan-2026-06-07.md"
RESULT_PATH = "docs/plans/bayesfilter-dpf-v2-algorithm-full-bf-filterflow-comparison-p4-bootstrap-ot-gradients-result-2026-06-07.md"
REPAIR_AMENDMENT_PATH = "docs/plans/bayesfilter-dpf-v2-algorithm-full-bf-filterflow-comparison-p4-inactive-zero-gradient-repair-amendment-2026-06-08.md"
P0_VISIBLE_JSON_PATH = OUTPUT_DIR / "dpf_v2_algorithm_full_comparison_p0_visible_governance_2026-06-08.json"
P1_JSON_PATH = OUTPUT_DIR / "dpf_v2_algorithm_full_comparison_p1_architecture_2026-06-07.json"
P2_JSON_PATH = OUTPUT_DIR / "dpf_v2_bootstrap_ot_contracts_2026-06-07.json"
P3_JSON_PATH = OUTPUT_DIR / "dpf_v2_bootstrap_ot_values_2026-06-07.json"
JSON_PATH = OUTPUT_DIR / "dpf_v2_bootstrap_ot_gradients_2026-06-07.json"
REPORT_PATH = REPORT_DIR / "dpf-v2-bootstrap-ot-gradients-2026-06-07.md"
P2_PASS_DECISION = "PASS_P2_BOOTSTRAP_OT_CONTRACTS_READY_FOR_P3"
P3_PASS_DECISION = "PASS_P3_BOOTSTRAP_OT_VALUES_READY_FOR_P4"
P2_BUNDLE_CHECKSUM = "53722eaea627646a85b99a2dde94d733b31e35e3a10886b05df20a733a1df11c"
P3_REPRODUCIBILITY_DIGEST = "3cf2161ff3ff470240f3a8c60f2405f6aff919769a013e68f56d19808905d521"
VALUE_TOLERANCE = 5e-10
GRADIENT_TOLERANCE = 5e-8
FD_STEP = 1e-5
INACTIVE_ZERO_GRADIENT_REASONS = {
    (
        "sv_1d_h18_rich",
        "sigma",
    ): (
        "Under the P4 fixed-additive-innovation scalar, sigma parameterizes "
        "the transition noise scale but the scalar uses transition mean plus "
        "frozen innovations and observation log density; derivative is zero "
        "by scalar derivation."
    ),
    (
        "structural_ar1_quadratic_h16",
        "sigma",
    ): (
        "Under the P4 fixed-additive-innovation scalar, sigma parameterizes "
        "the structural AR(1) transition scale but the scalar uses the "
        "deterministic mean/completion plus frozen innovations and observation "
        "log density; derivative is zero by scalar derivation."
    ),
}
EXPECTED_DECISIONS = {
    "PENDING_CLAUDE_REVIEW",
    "PASS_P4_BOOTSTRAP_OT_GRADIENTS_READY_FOR_P5",
    "P4_BOOTSTRAP_OT_GRADIENTS_CLASSIFIED_MISMATCH_PENDING_REVIEW",
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
    p0_payload = load_json(P0_VISIBLE_JSON_PATH)
    p1_payload = load_json(P1_JSON_PATH)
    p2_payload = load_json(P2_JSON_PATH)
    p3_payload = load_json(P3_JSON_PATH)
    _preflight_prior_artifacts(p0_payload, p1_payload, p2_payload, p3_payload)

    cells = []
    for contract in p2_payload["contracts"]:
        cells.append(_gradient_cell(contract))
    decision = _decision(cells)
    return {
        "metadata_date": "2026-06-07",
        "created_at_utc": utc_now(),
        "phase": "P4",
        "execution_route": "VISIBLE_IN_DIALOGUE",
        "decision": decision,
        "question": (
            "Do BayesFilter and BayesFilter-owned FilterFlow-side adapters match "
            "bootstrap-OT fixed-branch AD gradients for all P2-included physical "
            "knobs across all six V2 rows?"
        ),
        "plan_path": PLAN_PATH,
        "repair_amendment_path": REPAIR_AMENDMENT_PATH,
        "result_path": RESULT_PATH,
        "p0_visible_json_path": str(P0_VISIBLE_JSON_PATH.relative_to(REPO_ROOT)),
        "p1_architecture_json_path": str(P1_JSON_PATH.relative_to(REPO_ROOT)),
        "p2_contracts_json_path": str(P2_JSON_PATH.relative_to(REPO_ROOT)),
        "p3_values_json_path": str(P3_JSON_PATH.relative_to(REPO_ROOT)),
        "p2_contract_bundle_checksum": p2_payload["contract_bundle_checksum"],
        "p3_reproducibility_digest": p3_payload["reproducibility_digest"],
        "required_v2_model_ids": list(EXPECTED_V2_MODEL_IDS),
        "tolerances": {
            "value_abs": VALUE_TOLERANCE,
            "gradient_abs": GRADIENT_TOLERANCE,
            "finite_difference_step": FD_STEP,
            "finite_difference_policy": "diagnostic_only_not_a_promotion_gate",
        },
        "adapter_policy": {
            "bayesfilter_adapter": "P4 differentiable physical-knob layer over the P3 fixed-branch path",
            "filterflow_side_adapter": (
                "BayesFilter-owned contract-formula adapter plus shared P3 "
                "filterflow_style_annealed_transport_tf component"
            ),
            "filterflow_checkout_mutated": False,
            "filterflow_checkout_executed": False,
            "neither_side_is_oracle": True,
        },
        "primary_criterion_fields": {
            "included_gradient_knobs": _included_knob_summary(cells),
            "excluded_gradient_knobs": _excluded_knob_summary(cells),
            "all_included_knobs_executed": all(
                cell["status"] == "MATCHED" or cell["status"] == "PREDECLARED_EXCLUDED"
                for cell in cells
            ),
            "all_gradient_rows_matched_or_predeclared_excluded": all(
                cell["status"] in {"MATCHED", "PREDECLARED_EXCLUDED"} for cell in cells
            ),
            "inactive_zero_gradient_reasons": _inactive_zero_gradient_summary(cells),
            "finite_difference_promotion_gate": False,
        },
        "veto_diagnostics": _veto_diagnostics(cells, p2_payload, p3_payload),
        "explanatory_only_fields": {
            "status_counts": _status_counts(cells),
            "max_abs_scalar_delta": _max_metric(cells, "scalar_abs_delta"),
            "max_abs_gradient_delta": _max_metric(cells, "max_abs_gradient_delta"),
            "finite_difference_policy": "FD diagnostics are recorded per side but do not promote or fail P4.",
            "gradient_norm_policy": "Gradient norms explain scale only and are not a pass criterion.",
        },
        "cells": cells,
        "summary": _summary(cells),
        "review_round": 0,
        "open_material_blockers": [],
        "repair_amendment_required": False,
        "next_allowed_action": "run chunked Claude P4 read-only review before P5",
        "artifact_paths": {
            "json": str(JSON_PATH.relative_to(REPO_ROOT)),
            "markdown_report": str(REPORT_PATH.relative_to(REPO_ROOT)),
            "phase_result": RESULT_PATH,
        },
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners.run_v2_bootstrap_ot_gradients_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_claims": [
            "P4 validates deterministic fixed-branch bootstrap-OT AD-gradient agreement only.",
            "P4 does not establish stochastic resampling distribution correctness.",
            "P4 does not establish gradients through random or discrete branch decisions.",
            "P4 does not prove BayesFilter or FilterFlow correctness.",
            "P4 does not make a student implementation claim.",
            "P4 does not make a GPU, scalability, deployment, or production-readiness claim.",
        ],
    }


def _gradient_cell(contract: dict[str, Any]) -> dict[str, Any]:
    included = _included_knobs(contract)
    excluded = _excluded_knobs(contract)
    if not included:
        return {
            "model": contract["model_id"],
            "family": contract["family"],
            "implementations": ["BayesFilter", "FilterFlow-side adapter"],
            "cell_type": "v2_bootstrap_ot_fixed_branch_ad_gradient",
            "status": "PREDECLARED_EXCLUDED",
            "decision": f"{contract['model_id']}_bootstrap_ot_gradients_predeclared_excluded",
            "primary_criterion": "no P2-included physical gradient knob for this row",
            "metrics": {
                "included_knob_count": 0,
                "excluded_knobs": [knob["name"] for knob in excluded],
                "finite_difference_promotion_gate": False,
            },
            "mismatch_class": None,
            "contract": _contract_digest_view(contract),
            "bayesfilter": None,
            "filterflow": None,
            "non_claim": "predeclared gradient exclusion is not gradient evidence",
        }

    bayesfilter = _gradient_side(contract, "bayesfilter")
    filterflow = _gradient_side(contract, "filterflow_side")
    metrics = _compare_gradient_sides(bayesfilter, filterflow)
    matched = (
        metrics["scalar_within_tolerance"]
        and metrics["gradient_within_tolerance"]
        and bayesfilter["finite"]
        and filterflow["finite"]
        and bayesfilter["gradient_knob_names"] == filterflow["gradient_knob_names"]
        and bayesfilter["contract_checksum"] == filterflow["contract_checksum"] == contract["contract_checksum"]
    )
    return {
        "model": contract["model_id"],
        "family": contract["family"],
        "implementations": ["BayesFilter", "FilterFlow-side adapter"],
        "cell_type": "v2_bootstrap_ot_fixed_branch_ad_gradient",
        "status": "MATCHED" if matched else "EXPLAINED_MISMATCH",
        "decision": (
            f"{contract['model_id']}_bootstrap_ot_gradients_matched"
            if matched
            else f"{contract['model_id']}_bootstrap_ot_gradients_mismatch"
        ),
        "primary_criterion": "fixed-branch scalar and included physical-knob AD gradients match within tolerance",
        "metrics": metrics,
        "mismatch_class": None if matched else "v2_bootstrap_ot_scalar_or_gradient_delta",
        "contract": _contract_digest_view(contract),
        "bayesfilter": bayesfilter,
        "filterflow": filterflow,
        "non_claim": "fixed-branch AD-gradient agreement is not stochastic differentiable-resampling correctness",
    }


def _gradient_side(contract: dict[str, Any], side: str) -> dict[str, Any]:
    included = _included_knobs(contract)
    initial = [float(knob["initial_value"]) for knob in included]
    variables = [tf.Variable(value, dtype=DTYPE) for value in initial]
    value_fn = _value_function(contract, side)
    with tf.GradientTape() as tape:
        value = value_fn(variables)
    gradients = tape.gradient(value, variables)
    gradient_values, inactive_zero_reasons = _encode_gradients_by_contract(
        gradients,
        [knob["name"] for knob in included],
        contract["model_id"],
    )
    fd_values = _central_finite_difference(value_fn, initial)
    finite = (
        _finite_scalar(value)
        and all(
            (gradient is not None and _finite_scalar(gradient))
            or knob["name"] in inactive_zero_reasons
            for gradient, knob in zip(gradients, included, strict=True)
        )
    )
    return {
        "status": "executed",
        "backend": f"{side}_v2_bootstrap_ot_fixed_branch_ad_gradient",
        "model_id": contract["model_id"],
        "scalar": scalar(value),
        "gradient": gradient_values,
        "finite_difference_gradient": fd_values,
        "gradient_delta_vs_finite_difference": [
            None if grad is None else float(grad) - float(fd)
            for grad, fd in zip(gradient_values, fd_values, strict=True)
        ],
        "inactive_zero_gradient_knobs": list(inactive_zero_reasons),
        "inactive_zero_gradient_reasons": inactive_zero_reasons,
        "disconnected_zero_gradient_knobs": list(inactive_zero_reasons),
        "finite": bool(finite),
        "finite_difference_finite": all(math.isfinite(float(fd)) for fd in fd_values),
        "gradient_knob_names": [knob["name"] for knob in included],
        "gradient_parameterization": {
            knob["name"]: knob["parameterization"] for knob in included
        },
        "finite_difference_policy": "diagnostic_only_not_a_promotion_gate",
        "contract_checksum": contract["contract_checksum"],
        "adapter_input_checksum": stable_digest(contract),
    }


def _value_function(contract: dict[str, Any], side: str) -> Callable[[list[tf.Tensor]], tf.Tensor]:
    def value_fn(params: list[tf.Tensor]) -> tf.Tensor:
        parameterized = _parameterized_contract(contract, params)
        if side == "bayesfilter":
            return _contract_scalar(
                parameterized,
                backend="bayesfilter_v2_bootstrap_ot_fixed_branch_ad_gradient",
                transition_mean=lambda previous: _bayesfilter_transition_mean(parameterized, previous),
                transition_log_density=lambda previous, predicted, step: _bayesfilter_transition_log_density(
                    parameterized, previous, predicted, step
                ),
                observation_log_density=lambda predicted, observation, step: _bayesfilter_observation_log_density(
                    parameterized, predicted, observation, step
                ),
                complete_structural=lambda previous, predicted: (
                    _complete_structural(parameterized, previous, predicted)
                    if parameterized["model_id"] == "structural_ar1_quadratic_h16"
                    else predicted
                ),
            )
        if side == "filterflow_side":
            return _contract_scalar(
                parameterized,
                backend="filterflow_side_v2_bootstrap_ot_fixed_branch_ad_gradient",
                transition_mean=lambda previous: _filterflow_transition_mean(parameterized, previous),
                transition_log_density=lambda previous, predicted, step: _filterflow_transition_log_density(
                    parameterized, previous, predicted, step
                ),
                observation_log_density=lambda predicted, observation, step: _filterflow_observation_log_density(
                    parameterized, predicted, observation, step
                ),
                complete_structural=lambda previous, predicted: (
                    _complete_structural(parameterized, previous, predicted)
                    if parameterized["model_id"] == "structural_ar1_quadratic_h16"
                    else predicted
                ),
            )
        raise ValueError(f"unknown P4 gradient side: {side}")

    return value_fn


def _contract_scalar(
    contract: dict[str, Any],
    *,
    backend: str,
    transition_mean: Any,
    transition_log_density: Any,
    observation_log_density: Any,
    complete_structural: Any,
) -> tf.Tensor:
    del backend
    particles = tf.convert_to_tensor(contract["initial_particles"], DTYPE)
    innovations = tf.convert_to_tensor(contract["transition_innovations"], DTYPE)
    observations = tf.convert_to_tensor(contract["observations"], DTYPE)
    fixed_mask = [bool(flag) for flag in contract["fixed_ess_trigger_mask"]]
    log_weights = p3._uniform_log_weights(int(particles.shape[0]))
    total_scalar = tf.zeros([], DTYPE)

    for step in range(int(contract["horizon"])):
        predicted_particles = transition_mean(particles) + innovations[step]
        predicted_particles = complete_structural(particles, predicted_particles)
        _transition_density = transition_log_density(particles, predicted_particles, step)
        observation_density = observation_log_density(predicted_particles, observations[step], step)
        unnormalized = log_weights + observation_density
        increment = tf.reduce_logsumexp(unnormalized)
        total_scalar = total_scalar + increment
        log_weights = unnormalized - increment
        if bool(fixed_mask[step]):
            transport = p3.annealed_transport_resample_tf(
                predicted_particles,
                log_weights,
                epsilon=float(contract["ot_settings"]["sinkhorn_epsilon"]),
                scaling=float(contract["ot_settings"]["annealed_scaling"]),
                convergence_threshold=float(contract["ot_settings"]["annealed_convergence_threshold"]),
                max_iterations=int(contract["ot_settings"]["sinkhorn_iterations"]),
                ess_mask=tf.constant([True], dtype=tf.bool),
                transport_gradient_mode=str(contract["ot_settings"]["transport_gradient_mode"]),
                application_mode=str(contract["ot_settings"]["application_mode"]),
            )
            particles = tf.cast(transport.particles, DTYPE)
            log_weights = tf.cast(transport.log_weights, DTYPE)
        else:
            particles = predicted_particles
    return total_scalar


def _parameterized_contract(contract: dict[str, Any], params: list[tf.Tensor]) -> dict[str, Any]:
    clone = dict(contract)
    parameters = dict(contract["parameters"])
    theta = tf.convert_to_tensor(contract["theta"], DTYPE)
    for knob, value in zip(_included_knobs(contract), params, strict=True):
        name = str(knob["name"])
        value = tf.cast(value, DTYPE)
        if name == "transition_matrix_scale":
            parameters["A"] = p3._tensor(contract["parameters"]["A"]) * value
        elif name == "observation_noise_scale":
            parameters["R"] = p3._tensor(contract["parameters"]["R"]) * value
        elif name == "mu":
            parameters["mu"] = value
        elif name == "phi":
            parameters["phi"] = value
        elif name == "sigma":
            parameters["sigma"] = value
        elif name == "sigma_range":
            base = p3._tensor(parameters["R"])
            parameters["R"] = tf.linalg.diag(
                tf.stack([tf.square(value), base[1, 1]])
            )
        elif name == "sigma_bearing":
            base = p3._tensor(parameters["R"])
            parameters["R"] = tf.linalg.diag(
                tf.stack([base[0, 0], tf.square(value)])
            )
        elif name == "rho":
            parameters["rho"] = value
        elif name == "c":
            parameters["c"] = value
        elif name == "r":
            theta = tf.tensor_scatter_nd_update(theta, [[0]], [value])
        else:
            raise ValueError(f"unimplemented P4 gradient knob: {name}")
    clone["parameters"] = parameters
    clone["theta"] = theta
    return clone


def _bayesfilter_transition_mean(contract: dict[str, Any], particles: tf.Tensor) -> tf.Tensor:
    model_id = str(contract["model_id"])
    p = contract["parameters"]
    particles = tf.convert_to_tensor(particles, DTYPE)
    if model_id in {"lgssm_2d_h25_rich", "range_bearing_4d_h20_rich"}:
        return tf.linalg.matmul(particles, p3._tensor(p["A"]), transpose_b=True)
    if model_id == "sv_1d_h18_rich":
        mu = p3._tensor(p["mu"])
        phi = p3._tensor(p["phi"])
        return mu + phi * (particles - mu)
    if model_id == "structural_ar1_quadratic_h16":
        current_m = p3._tensor(p["rho"]) * particles[:, 0]
        current_k = p3._structural_complete_k(p, particles[:, 1], particles[:, 0], current_m)
        return tf.stack([current_m, current_k], axis=1)
    if model_id == "predator_prey_rk4":
        return p3._predator_transition_mean(p3._tensor(contract["theta"]), particles, p)
    raise ValueError(f"unsupported P4 included-gradient model: {model_id}")


def _filterflow_transition_mean(contract: dict[str, Any], particles: tf.Tensor) -> tf.Tensor:
    return _bayesfilter_transition_mean(contract, particles)


def _bayesfilter_transition_log_density(
    contract: dict[str, Any],
    previous: tf.Tensor,
    predicted: tf.Tensor,
    step: int,
) -> tf.Tensor:
    return p3._contract_transition_log_density(contract, previous, predicted, step)


def _filterflow_transition_log_density(
    contract: dict[str, Any],
    previous: tf.Tensor,
    predicted: tf.Tensor,
    step: int,
) -> tf.Tensor:
    return p3._contract_transition_log_density(contract, previous, predicted, step)


def _bayesfilter_observation_log_density(
    contract: dict[str, Any],
    particles: tf.Tensor,
    observation: tf.Tensor,
    step: int,
) -> tf.Tensor:
    return p3._contract_observation_log_density(contract, particles, observation, step)


def _filterflow_observation_log_density(
    contract: dict[str, Any],
    particles: tf.Tensor,
    observation: tf.Tensor,
    step: int,
) -> tf.Tensor:
    return p3._contract_observation_log_density(contract, particles, observation, step)


def _complete_structural(contract: dict[str, Any], previous: tf.Tensor, predicted: tf.Tensor) -> tf.Tensor:
    return p3._contract_complete_structural(contract, previous, predicted)


def _compare_gradient_sides(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    scalar_delta = abs(float(left["scalar"]) - float(right["scalar"]))
    gradient_delta = []
    disconnected = False
    for left_grad, right_grad in zip(left["gradient"], right["gradient"], strict=True):
        if left_grad is None or right_grad is None:
            gradient_delta.append(None)
            disconnected = True
        else:
            gradient_delta.append(float(left_grad) - float(right_grad))
    finite_gradient_deltas = [float(value) for value in gradient_delta if value is not None]
    max_abs_gradient_delta = max(abs(value) for value in finite_gradient_deltas) if finite_gradient_deltas else 0.0
    left_fd_delta = [
        None if grad is None else float(grad) - float(fd)
        for grad, fd in zip(left["gradient"], left["finite_difference_gradient"], strict=True)
    ]
    right_fd_delta = [
        None if grad is None else float(grad) - float(fd)
        for grad, fd in zip(right["gradient"], right["finite_difference_gradient"], strict=True)
    ]
    return {
        "scalar_abs_delta": scalar_delta,
        "gradient_delta": gradient_delta,
        "max_abs_gradient_delta": max_abs_gradient_delta,
        "bayesfilter_ad_vs_fd_delta": left_fd_delta,
        "filterflow_ad_vs_fd_delta": right_fd_delta,
        "bayesfilter_gradient_norm": _gradient_norm(left["gradient"]),
        "filterflow_gradient_norm": _gradient_norm(right["gradient"]),
        "scalar_within_tolerance": scalar_delta <= VALUE_TOLERANCE,
        "gradient_within_tolerance": (
            not disconnected and max_abs_gradient_delta <= GRADIENT_TOLERANCE
        ),
        "has_disconnected_gradient": disconnected,
        "finite_difference_promotion_gate": False,
        "value_tolerance": VALUE_TOLERANCE,
        "gradient_tolerance": GRADIENT_TOLERANCE,
        "finite_difference_step": FD_STEP,
    }


def _central_finite_difference(
    value_fn: Callable[[list[tf.Tensor]], tf.Tensor],
    params: list[float],
) -> list[float]:
    gradients = []
    for index in range(len(params)):
        plus = [tf.constant(value, DTYPE) for value in params]
        minus = [tf.constant(value, DTYPE) for value in params]
        plus[index] = plus[index] + tf.constant(FD_STEP, DTYPE)
        minus[index] = minus[index] - tf.constant(FD_STEP, DTYPE)
        gradients.append(
            scalar(
                (value_fn(plus) - value_fn(minus))
                / tf.constant(2.0 * FD_STEP, DTYPE)
            )
        )
    return gradients


def _encode_gradients_by_contract(
    gradients: list[tf.Tensor | None],
    knob_names: list[str],
    model_id: str,
) -> tuple[list[float | None], dict[str, str]]:
    values: list[float | None] = []
    inactive_zero_reasons: dict[str, str] = {}
    for gradient, knob_name in zip(gradients, knob_names, strict=True):
        if gradient is None:
            reason = INACTIVE_ZERO_GRADIENT_REASONS.get((model_id, knob_name))
            if reason is not None:
                values.append(0.0)
                inactive_zero_reasons[knob_name] = reason
            else:
                values.append(None)
            continue
        values.append(scalar(gradient))
    return values, inactive_zero_reasons


def _included_knobs(contract: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(knob)
        for knob in contract["gradient_contract"]["knobs"]
        if bool(knob.get("include"))
    ]


def _excluded_knobs(contract: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(knob)
        for knob in contract["gradient_contract"]["knobs"]
        if not bool(knob.get("include"))
    ]


def _included_knob_summary(cells: list[dict[str, Any]]) -> dict[str, list[str]]:
    summary = {}
    for cell in cells:
        if cell["status"] == "PREDECLARED_EXCLUDED":
            summary[cell["model"]] = []
        else:
            summary[cell["model"]] = list(cell["bayesfilter"]["gradient_knob_names"])
    return summary


def _excluded_knob_summary(cells: list[dict[str, Any]]) -> dict[str, list[str]]:
    return {
        cell["model"]: list(cell["metrics"].get("excluded_knobs", []))
        for cell in cells
        if cell["status"] == "PREDECLARED_EXCLUDED"
    }


def _inactive_zero_gradient_summary(cells: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    summary: dict[str, dict[str, str]] = {}
    for cell in cells:
        bayesfilter = cell.get("bayesfilter") or {}
        reasons = bayesfilter.get("inactive_zero_gradient_reasons", {})
        if reasons:
            summary[cell["model"]] = dict(reasons)
    return summary


def _veto_diagnostics(
    cells: list[dict[str, Any]],
    p2_payload: dict[str, Any],
    p3_payload: dict[str, Any],
) -> dict[str, bool]:
    ids = tuple(cell["model"] for cell in cells)
    included_cells = [cell for cell in cells if cell["status"] != "PREDECLARED_EXCLUDED"]
    return {
        "missing_v2_row": ids != EXPECTED_V2_MODEL_IDS,
        "row_order_mismatch": ids != EXPECTED_V2_MODEL_IDS,
        "p2_contract_checksum_changed": p2_payload["contract_bundle_checksum"] != P2_BUNDLE_CHECKSUM,
        "p3_value_pass_missing_or_changed": (
            p3_payload.get("decision") != P3_PASS_DECISION
            or p3_payload.get("reproducibility_digest") != P3_REPRODUCIBILITY_DIGEST
        ),
        "gradient_knob_changed_after_p2": _gradient_knob_changed_after_p2(cells, p2_payload),
        "scalar_mismatch": any(
            cell.get("metrics", {}).get("scalar_abs_delta", 0.0) > VALUE_TOLERANCE
            for cell in included_cells
        ),
        "ad_gradient_mismatch": any(
            cell.get("metrics", {}).get("max_abs_gradient_delta", 0.0) > GRADIENT_TOLERANCE
            for cell in included_cells
        ),
        "nonfinite_scalar_or_gradient": any(
            not bool(cell.get("bayesfilter", {}).get("finite", False))
            or not bool(cell.get("filterflow", {}).get("finite", False))
            for cell in included_cells
        ),
        "unclassified_gradient_mismatch": any(
            cell["status"] == "EXPLAINED_MISMATCH" and not cell.get("mismatch_class")
            for cell in included_cells
        ),
        "disconnected_gradient_posthoc_classification": any(
            None in cell.get("bayesfilter", {}).get("gradient", [])
            or None in cell.get("filterflow", {}).get("gradient", [])
            for cell in included_cells
        ),
        "finite_difference_promoted_to_gradient_gate": False,
        "localsource_filterflow_mutated": False,
        "student_command_or_metric": False,
        "oracle_framing": False,
    }


def _gradient_knob_changed_after_p2(cells: list[dict[str, Any]], p2_payload: dict[str, Any]) -> bool:
    p2_names = {
        contract["model_id"]: list(contract["gradient_contract"]["included_required_knob_names"])
        for contract in p2_payload["contracts"]
    }
    for cell in cells:
        if cell["status"] == "PREDECLARED_EXCLUDED":
            observed = []
        else:
            observed = list(cell["bayesfilter"]["gradient_knob_names"])
        if observed != p2_names.get(cell["model"], []):
            return True
    return False


def _contract_digest_view(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "contract_id": contract["contract_id"],
        "model_id": contract["model_id"],
        "algorithm": contract["algorithm"],
        "horizon": contract["horizon"],
        "fixed_ess_trigger_mask": contract["fixed_ess_trigger_mask"],
        "fixed_ess_trigger_count": contract["fixed_ess_trigger_count"],
        "ot_settings": contract["ot_settings"],
        "scalar_definition": contract["scalar_definition"],
        "gradient_contract": contract["gradient_contract"],
        "contract_checksum": contract["contract_checksum"],
        "component_checksums": contract["component_checksums"],
    }


def _preflight_prior_artifacts(
    p0_payload: dict[str, Any],
    p1_payload: dict[str, Any],
    p2_payload: dict[str, Any],
    p3_payload: dict[str, Any],
) -> None:
    if p0_payload.get("decision") != "PASS_P0_READY_FOR_P1":
        raise ValueError(f"P0 visible governance is not passed: {p0_payload.get('decision')}")
    if p1_payload.get("decision") != "PASS_P1_ARCHITECTURE_READY_FOR_P2":
        raise ValueError(f"P1 architecture is not passed: {p1_payload.get('decision')}")
    if p2_payload.get("decision") != P2_PASS_DECISION:
        raise ValueError(f"P2 bootstrap-OT contracts are not passed: {p2_payload.get('decision')}")
    if p2_payload.get("contract_bundle_checksum") != P2_BUNDLE_CHECKSUM:
        raise ValueError("P2 bootstrap-OT contract bundle checksum changed")
    if p3_payload.get("decision") != P3_PASS_DECISION:
        raise ValueError(f"P3 bootstrap-OT values are not passed: {p3_payload.get('decision')}")
    if p3_payload.get("reproducibility_digest") != P3_REPRODUCIBILITY_DIGEST:
        raise ValueError("P3 reproducibility digest changed")
    for phase, payload in (("P0", p0_payload), ("P1", p1_payload), ("P2", p2_payload), ("P3", p3_payload)):
        if tuple(payload.get("required_v2_model_ids", [])) != EXPECTED_V2_MODEL_IDS:
            raise ValueError(f"{phase} required model id gate failed")


def _validate_payload(payload: dict[str, Any]) -> None:
    required = {
        "phase",
        "decision",
        "primary_criterion_fields",
        "veto_diagnostics",
        "explanatory_only_fields",
        "review_round",
        "open_material_blockers",
        "repair_amendment_required",
        "next_allowed_action",
        "cells",
        "run_manifest",
        "artifact_paths",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"P4 payload missing required fields: {sorted(missing)}")
    if payload["phase"] != "P4":
        raise ValueError(f"unexpected phase: {payload['phase']}")
    if payload["decision"] not in EXPECTED_DECISIONS:
        raise ValueError(f"P4 payload decision not recognized: {payload['decision']}")
    ids = [cell["model"] for cell in payload["cells"]]
    if tuple(ids) != EXPECTED_V2_MODEL_IDS:
        raise ValueError(f"P4 cell id gate failed: {ids}")
    if payload["p2_contract_bundle_checksum"] != P2_BUNDLE_CHECKSUM:
        raise ValueError("P4 payload references changed P2 bundle checksum")
    if payload["p3_reproducibility_digest"] != P3_REPRODUCIBILITY_DIGEST:
        raise ValueError("P4 payload references changed P3 digest")
    if payload["artifact_paths"]["json"] != str(JSON_PATH.relative_to(REPO_ROOT)):
        raise ValueError("P4 JSON artifact path mismatch")
    if payload["run_manifest"].get("pre_import_cuda_visible_devices") != "-1":
        raise ValueError("P4 TensorFlow run was not CPU-only before import")
    for key in (
        "missing_v2_row",
        "row_order_mismatch",
        "p2_contract_checksum_changed",
        "p3_value_pass_missing_or_changed",
        "gradient_knob_changed_after_p2",
        "finite_difference_promoted_to_gradient_gate",
        "localsource_filterflow_mutated",
        "student_command_or_metric",
        "oracle_framing",
    ):
        if payload["veto_diagnostics"].get(key):
            raise ValueError(f"P4 veto diagnostic fired: {key}")
    if payload["decision"] in {"PENDING_CLAUDE_REVIEW", "PASS_P4_BOOTSTRAP_OT_GRADIENTS_READY_FOR_P5"}:
        if payload["veto_diagnostics"].get("scalar_mismatch"):
            raise ValueError("P4 scalar mismatch on passable decision")
        if payload["veto_diagnostics"].get("ad_gradient_mismatch"):
            raise ValueError("P4 AD-gradient mismatch on passable decision")
        if payload["veto_diagnostics"].get("nonfinite_scalar_or_gradient"):
            raise ValueError("P4 nonfinite scalar or gradient on passable decision")
        if payload["veto_diagnostics"].get("unclassified_gradient_mismatch"):
            raise ValueError("P4 unclassified gradient mismatch on passable decision")
        if payload["veto_diagnostics"].get("disconnected_gradient_posthoc_classification"):
            raise ValueError("P4 disconnected gradient on passable decision")
        for cell in payload["cells"]:
            if cell["status"] not in {"MATCHED", "PREDECLARED_EXCLUDED"}:
                raise ValueError(f"P4 row not matched or excluded on passable decision: {cell['model']}")
    elif payload["decision"] == "P4_BOOTSTRAP_OT_GRADIENTS_CLASSIFIED_MISMATCH_PENDING_REVIEW":
        if not any(cell["status"] == "EXPLAINED_MISMATCH" for cell in payload["cells"]):
            raise ValueError("P4 mismatch decision without classified mismatch cell")


def _decision(cells: list[dict[str, Any]]) -> str:
    if all(cell["status"] in {"MATCHED", "PREDECLARED_EXCLUDED"} for cell in cells):
        return "PENDING_CLAUDE_REVIEW"
    return "P4_BOOTSTRAP_OT_GRADIENTS_CLASSIFIED_MISMATCH_PENDING_REVIEW"


def _summary(cells: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_cells": len(cells),
        "models": [cell["model"] for cell in cells],
        "status_counts": _status_counts(cells),
        "max_abs_scalar_delta": _max_metric(cells, "scalar_abs_delta"),
        "max_abs_gradient_delta": _max_metric(cells, "max_abs_gradient_delta"),
        "included_knob_count": sum(
            len(cell.get("bayesfilter", {}).get("gradient_knob_names", []))
            for cell in cells
            if cell.get("bayesfilter")
        ),
        "predeclared_excluded_rows": [
            cell["model"] for cell in cells if cell["status"] == "PREDECLARED_EXCLUDED"
        ],
    }


def _status_counts(cells: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for cell in cells:
        counts[cell["status"]] = counts.get(cell["status"], 0) + 1
    return counts


def _max_metric(cells: list[dict[str, Any]], name: str) -> float:
    values = [
        float(cell.get("metrics", {}).get(name, 0.0))
        for cell in cells
        if name in cell.get("metrics", {}) and cell.get("metrics", {}).get(name) is not None
    ]
    return max(values) if values else 0.0


def _finite_scalar(value: tf.Tensor) -> bool:
    return bool(tf.math.is_finite(tf.cast(value, DTYPE)).numpy())


def _maybe_scalar(value: tf.Tensor | None) -> float | None:
    if value is None:
        return None
    return scalar(value)


def _gradient_norm(values: list[float | None]) -> float | None:
    if any(value is None for value in values):
        return None
    return math.sqrt(sum(float(value) * float(value) for value in values))


def _digest_payload(payload: dict[str, Any]) -> str:
    clone = dict(payload)
    clone.pop("reproducibility_digest", None)
    if "run_manifest" in clone:
        clone["run_manifest"] = dict(clone["run_manifest"])
        clone["run_manifest"].pop("wall_time_seconds", None)
    return stable_digest(clone)


def _markdown(payload: dict[str, Any]) -> str:
    run_manifest = payload["run_manifest"]
    if payload["decision"] == "PASS_P4_BOOTSTRAP_OT_GRADIENTS_READY_FOR_P5":
        review_state = "review_round: 5 Claude final synthesis returned VERDICT: AGREE"
        decision_uncertainty = "shared adapter defects remain possible outside the fixed-branch comparator"
        decision_next = "begin P5 PRECHECK visibly in the current dialogue"
    else:
        review_state = f"review_round: {payload['review_round']} pending chunked Claude P4 gradient review"
        decision_uncertainty = "Claude may find adapter or artifact adequacy gaps"
        decision_next = "run chunked Claude P4 read-only review"
    lines = [
        "# DPF V2 Algorithm Full Comparison P4 Bootstrap-OT Gradients Result",
        "",
        "metadata_date: 2026-06-07",
        "visible_execution_timestamp: `2026-06-08T10:36:23+08:00`",
        "phase: P4",
        "execution_route: `VISIBLE_IN_DIALOGUE`",
        f"status: `{payload['decision']}`",
        "",
        "## Question",
        "",
        payload["question"],
        "",
        "## Evidence Contract",
        "",
        "Primary criterion:",
        "",
        "- for every P2-included physical knob, BayesFilter and the BayesFilter-owned FilterFlow-side adapter consume the same frozen P2 bootstrap-OT contract;",
        "- scalar values match within declared tolerance on the deterministic fixed branch;",
        "- AD gradients match within declared tolerance for included knobs;",
        "- rows with no P2-included knobs remain predeclared exclusions, not post-result failures.",
        "",
        "Veto diagnostics:",
        "",
        "- nonfinite scalar or AD gradient for an included knob;",
        "- BF/FF scalar mismatch;",
        "- BF/FF AD-gradient mismatch;",
        "- gradient knob drift after P2;",
        "- finite differences promoted to a gradient gate;",
        "- disconnected gradient classified after result inspection except for the two reviewed derivation-inactive zero-gradient knobs;",
        "- `.localsource/filterflow` mutation, student command, or oracle framing.",
        "",
        "Non-claims:",
        "",
        "- P4 does not establish full stochastic-filter gradient correctness.",
        "- P4 does not establish gradients through random or discrete branch decisions.",
        "",
        "## Local Skeptical Phase Audit",
        "",
        "Audit status: `PASS_LOCAL_PHASE_AUDIT`.",
        "",
        "Wrong-baseline risk: controlled. P4 uses the reviewed P2 contract bundle checksum and reviewed P3 value digest.",
        "",
        "Proxy-metric risk: controlled. FD ladders, gradient norms, and transport residuals are explanatory only.",
        "",
        "Missing stop-condition risk: controlled. P4 stops on included-knob nonfinite or mismatched AD gradients and preserves predeclared excluded knobs.",
        "",
        "Inactive-gradient repair risk: controlled. The reviewed P4 repair amendment allows AD `None` to be encoded as explicit `0.0` only for `sv_1d_h18_rich:sigma` and `structural_ar1_quadratic_h16:sigma`; it does not waive scalar finiteness, connected-gradient finiteness, BF/FF agreement, row-order, checksum, or governance vetoes.",
        "",
        "Unfair-comparison risk: controlled. Both adapters consume identical contract bytes, branch masks, OT settings, scalar definition, and included knob list.",
        "",
        "Environment-mismatch risk: controlled. TensorFlow was run CPU-only with pre-import `CUDA_VISIBLE_DEVICES=-1`; no GPU claim is made.",
        "",
        "Audit decision: local pass pending Claude read-only review.",
        "",
        "## Result",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- JSON artifact: `{payload['artifact_paths']['json']}`",
        f"- Markdown report: `{payload['artifact_paths']['markdown_report']}`",
        f"- Phase result: `{payload['artifact_paths']['phase_result']}`",
        f"- Repair amendment: `{payload['repair_amendment_path']}`",
        f"- P2 contract bundle checksum: `{payload['p2_contract_bundle_checksum']}`",
        f"- P3 reproducibility digest: `{payload['p3_reproducibility_digest']}`",
        f"- Reproducibility digest: `{payload['reproducibility_digest']}`",
        "",
        "## Gradient Cells",
        "",
        "| Model id | Status | Knobs | Scalar delta | Max AD gradient delta |",
        "|---|---|---|---:|---:|",
    ]
    for cell in payload["cells"]:
        metrics = cell["metrics"]
        if cell["status"] == "PREDECLARED_EXCLUDED":
            knobs = f"excluded: {metrics['excluded_knobs']}"
        else:
            knobs = cell["bayesfilter"]["gradient_knob_names"]
        lines.append(
            f"| `{cell['model']}` | {cell['status']} | `{knobs}` | "
            f"{metrics.get('scalar_abs_delta', 'N/A')} | "
            f"{metrics.get('max_abs_gradient_delta', 'N/A')} |"
        )
    lines.extend(
        [
            "",
            "## Primary Criterion Fields",
            "",
        ]
    )
    lines.extend(f"- {key}: `{value}`" for key, value in payload["primary_criterion_fields"].items())
    lines.extend(
        [
            "",
            "## Veto Diagnostics",
            "",
        ]
    )
    lines.extend(f"- {key}: `{value}`" for key, value in payload["veto_diagnostics"].items())
    lines.extend(
        [
            "",
            "## Explanatory Only Fields",
            "",
        ]
    )
    lines.extend(f"- {key}: `{value}`" for key, value in payload["explanatory_only_fields"].items())
    lines.extend(
        [
            "",
            "## Command Manifest",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| git commit | `{run_manifest.get('commit')}` |",
            f"| git branch | `{run_manifest.get('branch')}` |",
            f"| dirty status | `{_single_line(run_manifest.get('dirty_state_summary'))}` |",
            f"| command | `{run_manifest.get('command')}` |",
            f"| validation commands | `python -m json.tool {payload['artifact_paths']['json']}`; `CUDA_VISIBLE_DEVICES=-1 python -m experiments.dpf_implementation.tf_tfp.runners.run_v2_bootstrap_ot_gradients_tf --validate-only`; `git diff --check` on P4 files |",
            f"| environment | `/home/chakwong/BayesFilter`; Python `{run_manifest.get('python_version')}`; TensorFlow `{run_manifest.get('package_versions', {}).get('tensorflow')}`; TFP `{run_manifest.get('package_versions', {}).get('tensorflow_probability')}` |",
            f"| CPU/GPU status | CPU-only TensorFlow run; pre-import `CUDA_VISIBLE_DEVICES={run_manifest.get('pre_import_cuda_visible_devices')}`; visible GPUs `{run_manifest.get('gpu_devices_visible')}` |",
            "| random seeds | no RNG consumed in P4; frozen particles, observations, transition innovations, and masks from P2 |",
            "| dtype | `tf.float64` / JSON dtype `float64` |",
            f"| output artifacts | `{payload['artifact_paths']['json']}`; `{payload['artifact_paths']['markdown_report']}`; `{payload['artifact_paths']['phase_result']}` |",
            "",
            "## Review State",
            "",
            review_state,
            "",
            f"open_material_blockers: {payload['open_material_blockers'] or 'none identified locally'}",
            "",
            f"repair_amendment_required: {str(payload['repair_amendment_required']).lower()}",
            "",
            f"next_allowed_action: {payload['next_allowed_action']}",
            "",
            "## Decision Table",
            "",
            "| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |",
            "|---|---|---|---|---|---|",
            f"| `{payload['decision']}` | all included bootstrap-OT AD gradients matched locally; excluded rows were predeclared | all local P4 veto diagnostics clear | {decision_uncertainty} | {decision_next} | no stochastic gradient correctness, implementation proof, student claim, GPU claim, or production readiness |",
            "",
            "## Post-Run Red Team",
            "",
            "Strongest alternative explanation: because both adapters are BayesFilter-owned, P4 can miss a shared defect in the parameterized contract-formula path.",
            "",
            "What would overturn the local decision: any reviewer finding that an included P2 knob was omitted, that a knob parameterization drifted, that FD diagnostics were used as a pass gate, or that P2/P3 checksums were not preserved.",
            "",
            "Weakest part of the evidence: P4 is fixed-branch AD-gradient evidence only and does not test stochastic resampling or gradients through discrete branch decisions.",
            "",
        ]
    )
    return "\n".join(lines)


def _single_line(value: Any) -> str:
    return str(value).replace("\n", " | ")


if __name__ == "__main__":
    raise SystemExit(main())
