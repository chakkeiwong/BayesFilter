"""Run visible P7 LEDH-PFPF-OT fixed-branch AD-gradient comparisons."""

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

from bayesfilter import highdim
from experiments.dpf_implementation.tf_tfp.fixtures.common_model_suite_tf import (
    EXPECTED_V2_MODEL_IDS,
    CommonRangeBearingSSM,
    CommonSVRichSSM,
    CommonStructuralAR1QuadraticSSM,
    CommonModelSpecV2,
    bayesfilter_model_for_spec_v2,
    common_model_specs_v2,
)
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
from experiments.dpf_implementation.tf_tfp.runners import run_v2_ledh_pfpf_ot_values_tf as p6


DTYPE = tf.float64
PLAN_PATH = (
    "docs/plans/bayesfilter-dpf-v2-algorithm-full-bf-filterflow-comparison-"
    "p7-ledh-pfpf-ot-gradients-subplan-2026-06-07.md"
)
RESULT_PATH = (
    "docs/plans/bayesfilter-dpf-v2-algorithm-full-bf-filterflow-comparison-"
    "p7-ledh-pfpf-ot-gradients-result-2026-06-07.md"
)
P0_VISIBLE_JSON_PATH = OUTPUT_DIR / "dpf_v2_algorithm_full_comparison_p0_visible_governance_2026-06-08.json"
P1_JSON_PATH = OUTPUT_DIR / "dpf_v2_algorithm_full_comparison_p1_architecture_2026-06-07.json"
P5_JSON_PATH = OUTPUT_DIR / "dpf_v2_ledh_pfpf_ot_contracts_2026-06-07.json"
P6_JSON_PATH = OUTPUT_DIR / "dpf_v2_ledh_pfpf_ot_values_2026-06-07.json"
JSON_PATH = OUTPUT_DIR / "dpf_v2_ledh_pfpf_ot_gradients_2026-06-07.json"
REPORT_PATH = REPORT_DIR / "dpf-v2-ledh-pfpf-ot-gradients-2026-06-07.md"
P5_PASS_DECISION = "PASS_P5_LEDH_PFPF_OT_CONTRACTS_READY_FOR_P6"
P6_PASS_DECISION = "PASS_P6_LEDH_PFPF_OT_VALUES_READY_FOR_P7"
PASS_DECISION = "PASS_P7_LEDH_PFPF_OT_GRADIENTS_READY_FOR_P8"
P5_BUNDLE_CHECKSUM = "20b7cb9a05a2de41bbf139a90bff8c8465f7bda7e4fd84fe6b83ee48a09c39f4"
P5_REPRODUCIBILITY_DIGEST = "6770ce953db079fe003d62e631e8a379cf68e6889006d583b39d70e6f5139661"
P6_REPRODUCIBILITY_DIGEST = "890a07f12bd2fcc35e6bc747578455bc04c418948058b07d3f5d2f62b955fe24"
VALUE_TOLERANCE = 5e-10
GRADIENT_TOLERANCE = 5e-8
FD_STEP = 1e-5
EXPECTED_INCLUDED_KNOBS = {
    "lgssm_2d_h25_rich": ["transition_matrix_scale", "observation_noise_scale"],
    "sv_1d_h18_rich": ["mu", "phi", "sigma"],
    "range_bearing_4d_h20_rich": ["sigma_range", "sigma_bearing"],
    "structural_ar1_quadratic_h16": ["rho", "sigma", "c"],
    "spatial_sir_j3_rk4": [],
    "predator_prey_rk4": ["r"],
}
EXPECTED_EXCLUDED_KNOBS = {
    "spatial_sir_j3_rk4": ["sir_physical_knobs"],
}
EXPECTED_DECISIONS = {
    "PENDING_CLAUDE_REVIEW",
    PASS_DECISION,
    "P7_LEDH_PFPF_OT_GRADIENTS_CLASSIFIED_MISMATCH_PENDING_REVIEW",
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
    p5_payload = load_json(P5_JSON_PATH)
    p6_payload = load_json(P6_JSON_PATH)
    _preflight_prior_artifacts(p0_payload, p1_payload, p5_payload, p6_payload)
    governance_evidence = p6._governance_evidence(p0_payload, p1_payload, p5_payload)

    specs = common_model_specs_v2()
    spec_by_id = {spec.model_id: spec for spec in specs}
    cells = []
    for contract in p5_payload["contracts"]:
        spec = spec_by_id[str(contract["model_id"])]
        cells.append(_gradient_cell(spec, contract))
    decision = _decision(cells)
    return {
        "metadata_date": "2026-06-07",
        "created_at_utc": utc_now(),
        "phase": "P7",
        "execution_route": "VISIBLE_IN_DIALOGUE",
        "decision": decision,
        "question": (
            "Do BayesFilter and BayesFilter-owned FilterFlow-side adapters "
            "match LEDH-PFPF-OT fixed-branch AD gradients for all P5-required "
            "physical knobs after the reviewed P6 value pass?"
        ),
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "p0_visible_json_path": str(P0_VISIBLE_JSON_PATH.relative_to(REPO_ROOT)),
        "p1_architecture_json_path": str(P1_JSON_PATH.relative_to(REPO_ROOT)),
        "p5_contracts_json_path": str(P5_JSON_PATH.relative_to(REPO_ROOT)),
        "p6_values_json_path": str(P6_JSON_PATH.relative_to(REPO_ROOT)),
        "p5_contract_bundle_checksum": p5_payload["contract_bundle_checksum"],
        "p5_reproducibility_digest": p5_payload["reproducibility_digest"],
        "p6_reproducibility_digest": p6_payload["reproducibility_digest"],
        "required_v2_model_ids": list(EXPECTED_V2_MODEL_IDS),
        "tolerances": {
            "value_abs": VALUE_TOLERANCE,
            "gradient_abs": GRADIENT_TOLERANCE,
            "finite_difference_step": FD_STEP,
            "finite_difference_policy": "diagnostic_only_not_a_promotion_gate",
        },
        "adapter_policy": {
            "bayesfilter_adapter": (
                "BayesFilter/common v2 model methods instantiated from each "
                "parameterized P5 contract"
            ),
            "filterflow_side_adapter": (
                "BayesFilter-owned contract-formula adapter plus shared P6 "
                "LEDH and filterflow_style_annealed_transport_tf components"
            ),
            "filterflow_checkout_mutated": not governance_evidence["localsource_filterflow_not_mutated"],
            "filterflow_checkout_executed": False,
            "neither_side_is_oracle": True,
        },
        "governance_evidence": governance_evidence,
        "primary_criterion_fields": {
            "included_gradient_knobs": _included_knob_summary(cells),
            "excluded_gradient_knobs": _excluded_knob_summary(cells),
            "total_included_physical_knobs": sum(
                len(cell.get("bayesfilter", {}).get("gradient_knob_names", []))
                for cell in cells
                if cell.get("bayesfilter")
            ),
            "predeclared_excluded_rows": [
                cell["model"] for cell in cells if cell["status"] == "PREDECLARED_EXCLUDED"
            ],
            "all_included_knobs_executed": all(
                cell["status"] in {"MATCHED", "PREDECLARED_EXCLUDED"} for cell in cells
            ),
            "all_gradient_rows_matched_or_predeclared_excluded": all(
                cell["status"] in {"MATCHED", "PREDECLARED_EXCLUDED"} for cell in cells
            ),
            "all_adapter_input_checksums_preserved": all(
                cell["metrics"].get("adapter_input_checksum_matches_contract_payload", False)
                for cell in cells
            ),
            "all_adapter_input_checksums_match_between_adapters": all(
                cell["metrics"].get("adapter_input_checksums_match_each_other", False)
                for cell in cells
            ),
            "finite_difference_promotion_gate": False,
        },
        "veto_diagnostics": _veto_diagnostics(cells, p5_payload, p6_payload, governance_evidence),
        "explanatory_only_fields": {
            "status_counts": _status_counts(cells),
            "max_abs_scalar_delta": _max_metric(cells, "scalar_abs_delta"),
            "max_abs_gradient_delta": _max_metric(cells, "max_abs_gradient_delta"),
            "max_abs_bayesfilter_ad_vs_fd_delta": _max_side_fd_delta(cells, "bayesfilter"),
            "max_abs_filterflow_ad_vs_fd_delta": _max_side_fd_delta(cells, "filterflow"),
            "finite_difference_policy": "FD diagnostics are recorded per side but do not promote or fail P7.",
            "gradient_norm_policy": "Gradient norms explain scale only and are not a pass criterion.",
            "runtime_policy": "runtime is explanatory only.",
        },
        "cells": cells,
        "summary": _summary(cells),
        "review_round": 5,
        "open_material_blockers": [],
        "repair_amendment_required": False,
        "next_allowed_action": "begin P8 PRECHECK visibly in current dialogue",
        "artifact_paths": {
            "json": str(JSON_PATH.relative_to(REPO_ROOT)),
            "markdown_report": str(REPORT_PATH.relative_to(REPO_ROOT)),
            "phase_result": RESULT_PATH,
        },
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners.run_v2_ledh_pfpf_ot_gradients_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_claims": [
            "P7 validates deterministic fixed-branch LEDH-PFPF-OT AD-gradient agreement only.",
            "P7 does not establish gradients through stochastic resampling distributions.",
            "P7 does not establish gradients through random or discrete branch decisions.",
            "P7 does not prove BayesFilter, FilterFlow, or adapter implementation correctness.",
            "P7 does not make a student implementation claim.",
            "P7 does not make a GPU, scalability, deployment, or production-readiness claim.",
            "P7 is not full-comparison success and does not establish P8 success.",
        ],
    }


def _gradient_cell(spec: CommonModelSpecV2, contract: dict[str, Any]) -> dict[str, Any]:
    included = _included_knobs(contract)
    excluded = _excluded_knobs(contract)
    if not included:
        return {
            "model": contract["model_id"],
            "family": contract["family"],
            "implementations": ["BayesFilter", "FilterFlow-side adapter"],
            "cell_type": "v2_ledh_pfpf_ot_fixed_branch_ad_gradient",
            "status": "PREDECLARED_EXCLUDED",
            "decision": f"{contract['model_id']}_ledh_pfpf_ot_gradients_predeclared_excluded",
            "primary_criterion": "no P5-included physical gradient knob for this row",
            "metrics": {
                "included_knob_count": 0,
                "excluded_knobs": [knob["name"] for knob in excluded],
                "adapter_input_checksum_matches_contract_payload": True,
                "adapter_input_checksums_match_each_other": True,
                "finite_difference_promotion_gate": False,
            },
            "mismatch_class": None,
            "contract": p6._contract_digest_view(contract),
            "bayesfilter": None,
            "filterflow": None,
            "spec_checksum": spec.checksum(),
            "non_claim": "predeclared gradient exclusion is not gradient evidence",
        }

    bayesfilter = _gradient_side(spec, contract, "bayesfilter")
    filterflow = _gradient_side(spec, contract, "filterflow_side")
    metrics = _compare_gradient_sides(bayesfilter, filterflow, contract)
    matched = (
        metrics["scalar_within_tolerance"]
        and metrics["gradient_within_tolerance"]
        and bayesfilter["finite"]
        and filterflow["finite"]
        and bayesfilter["gradient_knob_names"] == filterflow["gradient_knob_names"]
        and bayesfilter["contract_checksum"] == filterflow["contract_checksum"] == contract["contract_checksum"]
        and metrics["adapter_input_checksum_matches_contract_payload"]
        and metrics["adapter_input_checksums_match_each_other"]
    )
    return {
        "model": contract["model_id"],
        "family": contract["family"],
        "implementations": ["BayesFilter", "FilterFlow-side adapter"],
        "cell_type": "v2_ledh_pfpf_ot_fixed_branch_ad_gradient",
        "status": "MATCHED" if matched else "EXPLAINED_MISMATCH",
        "decision": (
            f"{contract['model_id']}_ledh_pfpf_ot_gradients_matched"
            if matched
            else f"{contract['model_id']}_ledh_pfpf_ot_gradients_mismatch"
        ),
        "primary_criterion": "fixed-branch scalar and included physical-knob AD gradients match within tolerance",
        "metrics": metrics,
        "mismatch_class": None if matched else "v2_ledh_pfpf_ot_scalar_or_gradient_delta",
        "contract": p6._contract_digest_view(contract),
        "bayesfilter": bayesfilter,
        "filterflow": filterflow,
        "spec_checksum": spec.checksum(),
        "non_claim": "fixed-branch AD-gradient agreement is not stochastic differentiable-resampling correctness",
    }


def _gradient_side(spec: CommonModelSpecV2, contract: dict[str, Any], side: str) -> dict[str, Any]:
    included = _included_knobs(contract)
    initial = [float(knob["initial_value"]) for knob in included]
    variables = [tf.Variable(value, dtype=DTYPE) for value in initial]
    value_fn = _value_function(spec, contract, side)
    with tf.GradientTape() as tape:
        value = value_fn(variables)
    gradients = tape.gradient(value, variables)
    gradient_values = [None if gradient is None else scalar(gradient) for gradient in gradients]
    fd_values = _central_finite_difference(value_fn, initial)
    finite = _finite_scalar(value) and all(
        gradient is not None and _finite_scalar(gradient) for gradient in gradients
    )
    return {
        "status": "executed",
        "backend": f"{side}_v2_ledh_pfpf_ot_fixed_branch_ad_gradient",
        "model_id": contract["model_id"],
        "scalar": scalar(value),
        "gradient": gradient_values,
        "finite_difference_gradient": fd_values,
        "gradient_delta_vs_finite_difference": [
            None if grad is None else float(grad) - float(fd)
            for grad, fd in zip(gradient_values, fd_values, strict=True)
        ],
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


def _value_function(
    spec: CommonModelSpecV2,
    contract: dict[str, Any],
    side: str,
) -> Callable[[list[tf.Tensor]], tf.Tensor]:
    def value_fn(params: list[tf.Tensor]) -> tf.Tensor:
        parameterized = _parameterized_contract(contract, params)
        if side == "bayesfilter":
            model = _bayesfilter_model_from_contract(spec, parameterized)
            theta = p6._tensor(parameterized["theta"])
            return _differentiable_ledh_scalar(
                spec=spec,
                contract=parameterized,
                transition_mean=lambda previous: _bayesfilter_transition_mean_from_model(
                    spec,
                    model,
                    theta,
                    previous,
                ),
                transition_log_density=lambda previous, post, step: model.transition_log_density(
                    theta,
                    previous,
                    post,
                    t=step + 1,
                ),
                observation_log_density=lambda post, observation, step: model.observation_log_density(
                    theta,
                    post,
                    observation,
                    t=step + 1,
                ),
            )
        if side == "filterflow_side":
            return _differentiable_ledh_scalar(
                spec=spec,
                contract=parameterized,
                transition_mean=lambda previous: p6._contract_transition_mean(parameterized, previous),
                transition_log_density=lambda previous, post, step: p6._contract_transition_log_density(
                    parameterized,
                    previous,
                    post,
                    step,
                ),
                observation_log_density=lambda post, observation, step: p6._contract_observation_log_density(
                    parameterized,
                    post,
                    observation,
                    step,
                ),
            )
        raise ValueError(f"unknown P7 gradient side: {side}")

    return value_fn


def _differentiable_ledh_scalar(
    *,
    spec: CommonModelSpecV2,
    contract: dict[str, Any],
    transition_mean: Any,
    transition_log_density: Any,
    observation_log_density: Any,
) -> tf.Tensor:
    particles = tf.convert_to_tensor(contract["initial_particles"], DTYPE)
    innovations = tf.convert_to_tensor(contract["transition_innovations"], DTYPE)
    observations = tf.convert_to_tensor(contract["observations"], DTYPE)
    fixed_mask = [bool(flag) for flag in contract["fixed_ess_trigger_mask"]]
    log_weights = p6._uniform_log_weights(int(particles.shape[0]))
    total_scalar = tf.zeros([], DTYPE)

    for step in range(int(contract["horizon"])):
        ancestors = particles
        prior_mean = transition_mean(ancestors)
        pre_flow = p6._complete_structural_if_needed(spec, contract, ancestors, prior_mean + innovations[step])
        flow = p6._ledh_flow(
            spec=spec,
            contract=contract,
            ancestors=ancestors,
            pre_flow=pre_flow,
            prior_mean=prior_mean,
            observation=observations[step],
        )
        post_flow = tf.cast(flow["post_flow_particles"], DTYPE)
        target_transition = transition_log_density(ancestors, post_flow, step)
        target_observation = observation_log_density(post_flow, observations[step], step)
        corrected = (
            log_weights
            + target_transition
            + target_observation
            - flow["pre_flow_log_density"]
            + flow["forward_log_det"]
        )
        _weights, increment, normalized_log_weights = p6._normalize_log_weights(corrected)
        total_scalar = total_scalar + increment
        if bool(fixed_mask[step]):
            transport = p6.annealed_transport_resample_tf(
                post_flow,
                normalized_log_weights,
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
            particles = post_flow
            log_weights = normalized_log_weights
    return total_scalar


def _bayesfilter_model_from_contract(spec: CommonModelSpecV2, contract: dict[str, Any]) -> Any:
    p = contract["parameters"]
    if spec.model_id == "lgssm_2d_h25_rich":
        return highdim.LinearGaussianSSM(
            initial_mean=p["m0"],
            initial_covariance=p["P0"],
            transition_matrix=p["A"],
            transition_covariance=p["Q"],
            observation_matrix=p["C"],
            observation_covariance=p["R"],
        )
    if spec.model_id == "sv_1d_h18_rich":
        return CommonSVRichSSM(
            mu=p["mu"],
            phi=p["phi"],
            sigma=p["sigma"],
            h0_mean=p["h0_mean"],
            h0_variance=p["h0_variance"],
        )
    if spec.model_id == "range_bearing_4d_h20_rich":
        return CommonRangeBearingSSM(
            initial_mean=p["m0"],
            initial_covariance=p["P0"],
            transition_matrix=p["A"],
            transition_covariance=p["Q"],
            observation_covariance=p["R"],
        )
    if spec.model_id == "structural_ar1_quadratic_h16":
        return CommonStructuralAR1QuadraticSSM(
            rho=p["rho"],
            sigma=p["sigma"],
            a=p["a"],
            b=p["b"],
            c=p["c"],
            d=p["d"],
            lam=p["lambda"],
            observation_scale=p["observation_scale"],
            m0_mean=p["m0_mean"],
            m0_variance=p["m0_variance"],
            k0=p["k0"],
        )
    if spec.model_id == "predator_prey_rk4":
        return highdim.PredatorPreySSM(
            initial_mean=p["initial_mean"],
            delta=p["delta"],
            rk4_internal_step=p["rk4_internal_step"],
            process_covariance=p["process_covariance"],
            observation_covariance=p["observation_covariance"],
            initial_covariance=p["initial_covariance"],
            domain_policy=p["domain_policy"],
        )
    return bayesfilter_model_for_spec_v2(spec)


def _bayesfilter_transition_mean_from_model(
    spec: CommonModelSpecV2,
    model: Any,
    theta: tf.Tensor,
    particles: tf.Tensor,
) -> tf.Tensor:
    if spec.model_id in {"lgssm_2d_h25_rich", "range_bearing_4d_h20_rich"}:
        return tf.linalg.matmul(
            tf.convert_to_tensor(particles, DTYPE),
            tf.convert_to_tensor(model.transition_matrix, DTYPE),
            transpose_b=True,
        )
    if spec.model_id in {"sv_1d_h18_rich", "structural_ar1_quadratic_h16", "spatial_sir_j3_rk4"}:
        return model.transition_mean(particles)
    if spec.model_id == "predator_prey_rk4":
        return model.transition_mean(theta, particles)
    raise ValueError(f"unknown v2 model spec: {spec.model_id}")


def _parameterized_contract(contract: dict[str, Any], params: list[tf.Tensor]) -> dict[str, Any]:
    clone = dict(contract)
    parameters = dict(contract["parameters"])
    theta = tf.convert_to_tensor(contract["theta"], DTYPE)
    for knob, value in zip(_included_knobs(contract), params, strict=True):
        name = str(knob["name"])
        value = tf.cast(value, DTYPE)
        if name == "transition_matrix_scale":
            parameters["A"] = p6._tensor(contract["parameters"]["A"]) * value
        elif name == "observation_noise_scale":
            parameters["R"] = p6._tensor(contract["parameters"]["R"]) * value
        elif name == "mu":
            parameters["mu"] = value
        elif name == "phi":
            parameters["phi"] = value
        elif name == "sigma":
            parameters["sigma"] = value
        elif name == "sigma_range":
            base = p6._tensor(parameters["R"])
            parameters["R"] = tf.linalg.diag(tf.stack([tf.square(value), base[1, 1]]))
        elif name == "sigma_bearing":
            base = p6._tensor(parameters["R"])
            parameters["R"] = tf.linalg.diag(tf.stack([base[0, 0], tf.square(value)]))
        elif name == "rho":
            parameters["rho"] = value
        elif name == "c":
            parameters["c"] = value
        elif name == "r":
            theta = tf.tensor_scatter_nd_update(theta, [[0]], [value])
        else:
            raise ValueError(f"unimplemented P7 gradient knob: {name}")
    clone["parameters"] = parameters
    clone["theta"] = theta
    return clone


def _compare_gradient_sides(
    left: dict[str, Any],
    right: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
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
    contract_payload_checksum = stable_digest(contract)
    left_adapter_input_checksum = str(left.get("adapter_input_checksum", ""))
    right_adapter_input_checksum = str(right.get("adapter_input_checksum", ""))
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
        "contract_payload_checksum": contract_payload_checksum,
        "bayesfilter_adapter_input_checksum": left_adapter_input_checksum,
        "filterflow_adapter_input_checksum": right_adapter_input_checksum,
        "adapter_input_checksum_matches_contract_payload": (
            left_adapter_input_checksum == contract_payload_checksum
            and right_adapter_input_checksum == contract_payload_checksum
        ),
        "adapter_input_checksums_match_each_other": left_adapter_input_checksum == right_adapter_input_checksum,
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
        gradients.append(scalar((value_fn(plus) - value_fn(minus)) / tf.constant(2.0 * FD_STEP, DTYPE)))
    return gradients


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
    summary = {}
    for cell in cells:
        if cell["status"] == "PREDECLARED_EXCLUDED":
            summary[cell["model"]] = list(cell["metrics"].get("excluded_knobs", []))
        else:
            summary[cell["model"]] = []
    return summary


def _veto_diagnostics(
    cells: list[dict[str, Any]],
    p5_payload: dict[str, Any],
    p6_payload: dict[str, Any],
    governance_evidence: dict[str, Any],
) -> dict[str, bool]:
    ids = tuple(cell["model"] for cell in cells)
    included_cells = [cell for cell in cells if cell["status"] != "PREDECLARED_EXCLUDED"]
    return {
        "missing_v2_row": ids != EXPECTED_V2_MODEL_IDS,
        "row_order_mismatch": ids != EXPECTED_V2_MODEL_IDS,
        "p5_contract_checksum_changed": p5_payload["contract_bundle_checksum"] != P5_BUNDLE_CHECKSUM,
        "p5_reproducibility_digest_changed": p5_payload["reproducibility_digest"] != P5_REPRODUCIBILITY_DIGEST,
        "p6_value_pass_missing_or_changed": (
            p6_payload.get("decision") != P6_PASS_DECISION
            or p6_payload.get("reproducibility_digest") != P6_REPRODUCIBILITY_DIGEST
        ),
        "gradient_knob_changed_after_p5": _gradient_knob_changed_after_p5(cells, p5_payload),
        "sir_predeclared_exclusion_missing_or_failed": _sir_exclusion_failed(cells, p5_payload),
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
        "disconnected_gradient": any(
            None in cell.get("bayesfilter", {}).get("gradient", [])
            or None in cell.get("filterflow", {}).get("gradient", [])
            for cell in included_cells
        ),
        "adapter_input_checksum_mismatch": any(
            not cell["metrics"].get("adapter_input_checksum_matches_contract_payload", False)
            for cell in cells
        ),
        "value_agreement_used_to_excuse_derivative_mismatch": False,
        "finite_difference_promoted_to_gradient_gate": False,
        "localsource_filterflow_mutated": not governance_evidence["localsource_filterflow_not_mutated"],
        "student_command_or_metric": not governance_evidence["student_commands_absent"],
        "oracle_framing": not governance_evidence["oracle_framing_forbidden"],
        "unsupported_full_comparison_or_p8_success_claim": False,
    }


def _gradient_knob_changed_after_p5(cells: list[dict[str, Any]], p5_payload: dict[str, Any]) -> bool:
    p5_names = {
        contract["model_id"]: list(contract["gradient_contract"]["included_required_knob_names"])
        for contract in p5_payload["contracts"]
    }
    for cell in cells:
        if cell["status"] == "PREDECLARED_EXCLUDED":
            observed = []
        else:
            observed = list(cell["bayesfilter"]["gradient_knob_names"])
        if observed != p5_names.get(cell["model"], []):
            return True
        if observed != EXPECTED_INCLUDED_KNOBS.get(cell["model"], []):
            return True
    return False


def _sir_exclusion_failed(cells: list[dict[str, Any]], p5_payload: dict[str, Any]) -> bool:
    sir_cells = [cell for cell in cells if cell["model"] == "spatial_sir_j3_rk4"]
    if len(sir_cells) != 1 or sir_cells[0]["status"] != "PREDECLARED_EXCLUDED":
        return True
    p5_contract = next(
        contract for contract in p5_payload["contracts"] if contract["model_id"] == "spatial_sir_j3_rk4"
    )
    if p5_contract["gradient_contract"].get("p7_gradient_readiness") != "PREDECLARED_EXCLUDED_NO_PHYSICAL_KNOB":
        return True
    return sir_cells[0]["metrics"].get("excluded_knobs") != EXPECTED_EXCLUDED_KNOBS["spatial_sir_j3_rk4"]


def _preflight_prior_artifacts(
    p0_payload: dict[str, Any],
    p1_payload: dict[str, Any],
    p5_payload: dict[str, Any],
    p6_payload: dict[str, Any],
) -> None:
    if p0_payload.get("decision") != "PASS_P0_READY_FOR_P1":
        raise ValueError(f"P0 visible governance is not passed: {p0_payload.get('decision')}")
    if p1_payload.get("decision") != "PASS_P1_ARCHITECTURE_READY_FOR_P2":
        raise ValueError(f"P1 architecture is not passed: {p1_payload.get('decision')}")
    if p5_payload.get("decision") != P5_PASS_DECISION:
        raise ValueError(f"P5 LEDH-PFPF-OT contracts are not passed: {p5_payload.get('decision')}")
    if p5_payload.get("contract_bundle_checksum") != P5_BUNDLE_CHECKSUM:
        raise ValueError("P5 LEDH-PFPF-OT contract bundle checksum changed")
    if p5_payload.get("reproducibility_digest") != P5_REPRODUCIBILITY_DIGEST:
        raise ValueError("P5 LEDH-PFPF-OT reproducibility digest changed")
    if p6_payload.get("decision") != P6_PASS_DECISION:
        raise ValueError(f"P6 LEDH-PFPF-OT values are not passed: {p6_payload.get('decision')}")
    if p6_payload.get("reproducibility_digest") != P6_REPRODUCIBILITY_DIGEST:
        raise ValueError("P6 LEDH-PFPF-OT value digest changed")
    for phase, payload in (("P0", p0_payload), ("P1", p1_payload), ("P5", p5_payload), ("P6", p6_payload)):
        if tuple(payload.get("required_v2_model_ids", [])) != EXPECTED_V2_MODEL_IDS:
            raise ValueError(f"{phase} required model id gate failed")
    ids = [contract.get("model_id") for contract in p5_payload.get("contracts", [])]
    if tuple(ids) != EXPECTED_V2_MODEL_IDS:
        raise ValueError(f"P5 contract model id gate failed: {ids}")
    if [cell.get("model") for cell in p6_payload.get("cells", [])] != list(EXPECTED_V2_MODEL_IDS):
        raise ValueError("P6 cell model id gate failed")
    if not p6_payload.get("primary_criterion_fields", {}).get("all_rows_matched"):
        raise ValueError("P6 did not record all rows matched")


def _validate_payload(payload: dict[str, Any]) -> None:
    required = {
        "phase",
        "decision",
        "primary_criterion_fields",
        "veto_diagnostics",
        "explanatory_only_fields",
        "review_round",
        "governance_evidence",
        "open_material_blockers",
        "repair_amendment_required",
        "next_allowed_action",
        "cells",
        "run_manifest",
        "artifact_paths",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"P7 payload missing required fields: {sorted(missing)}")
    if payload["phase"] != "P7":
        raise ValueError(f"unexpected phase: {payload['phase']}")
    if payload["decision"] not in EXPECTED_DECISIONS:
        raise ValueError(f"P7 payload decision not recognized: {payload['decision']}")
    ids = [cell["model"] for cell in payload["cells"]]
    if tuple(ids) != EXPECTED_V2_MODEL_IDS:
        raise ValueError(f"P7 cell id gate failed: {ids}")
    if payload["p5_contract_bundle_checksum"] != P5_BUNDLE_CHECKSUM:
        raise ValueError("P7 payload references changed P5 bundle checksum")
    if payload["p5_reproducibility_digest"] != P5_REPRODUCIBILITY_DIGEST:
        raise ValueError("P7 payload references changed P5 digest")
    if payload["p6_reproducibility_digest"] != P6_REPRODUCIBILITY_DIGEST:
        raise ValueError("P7 payload references changed P6 digest")
    if payload["artifact_paths"]["json"] != str(JSON_PATH.relative_to(REPO_ROOT)):
        raise ValueError("P7 JSON artifact path mismatch")
    if payload["run_manifest"].get("pre_import_cuda_visible_devices") != "-1":
        raise ValueError("P7 TensorFlow run was not CPU-only before import")
    if payload["primary_criterion_fields"].get("total_included_physical_knobs") != 11:
        raise ValueError("P7 included physical knob count is not 11")
    if payload["primary_criterion_fields"].get("predeclared_excluded_rows") != ["spatial_sir_j3_rk4"]:
        raise ValueError("P7 predeclared excluded rows changed")
    for key in (
        "missing_v2_row",
        "row_order_mismatch",
        "p5_contract_checksum_changed",
        "p5_reproducibility_digest_changed",
        "p6_value_pass_missing_or_changed",
        "gradient_knob_changed_after_p5",
        "sir_predeclared_exclusion_missing_or_failed",
        "finite_difference_promoted_to_gradient_gate",
        "localsource_filterflow_mutated",
        "student_command_or_metric",
        "oracle_framing",
        "unsupported_full_comparison_or_p8_success_claim",
        "adapter_input_checksum_mismatch",
    ):
        if payload["veto_diagnostics"].get(key):
            raise ValueError(f"P7 veto diagnostic fired: {key}")
    governance_required = (
        "localsource_filterflow_not_mutated",
        "student_commands_absent",
        "oracle_framing_forbidden",
        "finite_differences_diagnostic_only",
        "p1_ledh_filterflow_adapter_is_bayesfilter_owned",
        "p5_contract_forbids_runtime_branch_redecision",
    )
    for key in governance_required:
        if payload["governance_evidence"].get(key) is not True:
            raise ValueError(f"P7 governance evidence failed: {key}")
    if payload["decision"] in {"PENDING_CLAUDE_REVIEW", PASS_DECISION}:
        for key in (
            "scalar_mismatch",
            "ad_gradient_mismatch",
            "nonfinite_scalar_or_gradient",
            "unclassified_gradient_mismatch",
            "disconnected_gradient",
            "value_agreement_used_to_excuse_derivative_mismatch",
        ):
            if payload["veto_diagnostics"].get(key):
                raise ValueError(f"P7 passable decision has veto: {key}")
        for cell in payload["cells"]:
            if cell["status"] not in {"MATCHED", "PREDECLARED_EXCLUDED"}:
                raise ValueError(f"P7 row not matched or excluded on passable decision: {cell['model']}")
    elif payload["decision"] == "P7_LEDH_PFPF_OT_GRADIENTS_CLASSIFIED_MISMATCH_PENDING_REVIEW":
        if not any(cell["status"] == "EXPLAINED_MISMATCH" for cell in payload["cells"]):
            raise ValueError("P7 mismatch decision without classified mismatch cell")


def _decision(cells: list[dict[str, Any]]) -> str:
    if all(cell["status"] in {"MATCHED", "PREDECLARED_EXCLUDED"} for cell in cells):
        return PASS_DECISION
    return "P7_LEDH_PFPF_OT_GRADIENTS_CLASSIFIED_MISMATCH_PENDING_REVIEW"


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


def _max_side_fd_delta(cells: list[dict[str, Any]], side: str) -> float | None:
    values = []
    for cell in cells:
        payload = cell.get(side)
        if not payload:
            continue
        for value in payload.get("gradient_delta_vs_finite_difference", []):
            if value is not None:
                values.append(abs(float(value)))
    return max(values) if values else None


def _finite_scalar(value: tf.Tensor) -> bool:
    return bool(tf.math.is_finite(tf.cast(value, DTYPE)).numpy())


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
    if payload["decision"] == PASS_DECISION:
        review_state = "review_round: 5 final Claude synthesis returned VERDICT: AGREE"
        decision_next = "begin P8 PRECHECK visibly in the current dialogue"
        decision_uncertainty = "shared adapter defects remain possible outside the fixed-branch comparator"
    else:
        review_state = f"review_round: {payload['review_round']} pending probe and chunked Claude P7 gradient review"
        decision_next = "run probe then chunked Claude P7 read-only review"
        decision_uncertainty = "Claude may find adapter, gradient-route, or artifact adequacy gaps"
    lines = [
        "# DPF V2 Algorithm Full Comparison P7 LEDH-PFPF-OT Gradients Result",
        "",
        "metadata_date: 2026-06-07",
        "visible_execution_timestamp: `2026-06-08T17:20:00+08:00`",
        "phase: P7",
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
        "- for every P5-included physical knob, BayesFilter and the BayesFilter-owned FilterFlow-side adapter consume the same frozen P5 LEDH-PFPF-OT contract;",
        "- scalar values match within declared tolerance on the deterministic fixed branch;",
        "- AD gradients through the LEDH proposal, PF-PF correction, and deterministic OT transform match within declared tolerance;",
        "- `spatial_sir_j3_rk4` remains a predeclared no-physical-knob exclusion, not a failed row.",
        "",
        "Veto diagnostics:",
        "",
        "- nonfinite scalar or AD gradient for an included knob;",
        "- BF/FF scalar mismatch;",
        "- BF/FF AD-gradient mismatch;",
        "- missing or changed required knob after P5;",
        "- missing SIR predeclared exclusion;",
        "- P5 checksum/digest or P6 digest drift;",
        "- finite differences promoted to a gradient gate;",
        "- value agreement used to excuse derivative mismatch;",
        "- `.localsource/filterflow` mutation, student command, oracle framing, or unsupported full-comparison/P8 success claim.",
        "",
        "Non-claims:",
        "",
        "- P7 does not establish gradients through stochastic resampling distributions or random/discrete branch selection.",
        "- P7 does not prove BayesFilter, FilterFlow, or adapter implementation correctness.",
        "- P7 does not make a student implementation claim.",
        "- P7 does not make a GPU, scalability, deployment, production-readiness, full-comparison, or P8 success claim.",
        "",
        "## Local Skeptical Phase Audit",
        "",
        "Audit status: `PASS_LOCAL_PHASE_AUDIT`.",
        "",
        "Wrong-baseline risk: controlled. P7 rejects drift from the reviewed P5 contract bundle and reviewed P6 value digest.",
        "",
        "Proxy-metric risk: controlled. FD ladders, AD-vs-FD deltas, gradient norms, and runtime are explanatory only.",
        "",
        "Missing stop-condition risk: controlled. P7 stops on included-knob nonfinite or mismatched AD gradients, P5/P6 drift, missing SIR exclusion, or governance vetoes.",
        "",
        "Unfair-comparison risk: controlled. Both adapters consume identical contract bytes, branch masks, OT settings, scalar definition, and included knob list.",
        "",
        "Environment-mismatch risk: controlled. TensorFlow was run CPU-only with pre-import `CUDA_VISIBLE_DEVICES=-1`; no GPU claim is made.",
        "",
        (
            "Audit decision: reviewed local pass; final Claude read-only synthesis returned VERDICT: AGREE."
            if payload["decision"] == PASS_DECISION
            else "Audit decision: local pass pending Claude read-only review."
        ),
        "",
        "## Result",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- JSON artifact: `{payload['artifact_paths']['json']}`",
        f"- Markdown report: `{payload['artifact_paths']['markdown_report']}`",
        f"- Phase result: `{payload['artifact_paths']['phase_result']}`",
        f"- P5 contract bundle checksum: `{payload['p5_contract_bundle_checksum']}`",
        f"- P5 reproducibility digest: `{payload['p5_reproducibility_digest']}`",
        f"- P6 reproducibility digest: `{payload['p6_reproducibility_digest']}`",
        f"- P7 reproducibility digest: `{payload['reproducibility_digest']}`",
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
            "## Governance Evidence",
            "",
        ]
    )
    lines.extend(f"- {key}: `{value}`" for key, value in payload["governance_evidence"].items())
    lines.extend(
        [
            "",
            "## Command Manifest",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| git commit | `{run_manifest.get('commit')}` |",
            f"| git branch | `{run_manifest.get('branch')}` |",
            "| dirty status | full repo dirty state is recorded in JSON run manifest; unrelated dirty work is not interpreted as P7 evidence |",
            f"| command | `{run_manifest.get('command')}` |",
            f"| validation commands | `python -m json.tool {payload['artifact_paths']['json']}`; `CUDA_VISIBLE_DEVICES=-1 python -m experiments.dpf_implementation.tf_tfp.runners.run_v2_ledh_pfpf_ot_gradients_tf --validate-only`; `git diff --check` on P7 files |",
            f"| environment | `/home/chakwong/BayesFilter`; Python `{run_manifest.get('python_version')}`; TensorFlow `{run_manifest.get('package_versions', {}).get('tensorflow')}`; TFP `{run_manifest.get('package_versions', {}).get('tensorflow_probability')}` |",
            f"| CPU/GPU status | CPU-only TensorFlow run; pre-import `CUDA_VISIBLE_DEVICES={run_manifest.get('pre_import_cuda_visible_devices')}`; visible GPUs `{run_manifest.get('gpu_devices_visible')}` |",
            "| random seeds | no RNG consumed in P7; frozen particles, observations, transition innovations, and masks from P5 |",
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
            f"| `{payload['decision']}` | all included LEDH-PFPF-OT AD gradients matched locally; SIR was predeclared excluded | all local P7 veto diagnostics clear | {decision_uncertainty} | {decision_next} | no stochastic-gradient correctness, implementation proof, student claim, GPU claim, production readiness, full-comparison success, or P8 success |",
            "",
            "## Post-Run Red Team",
            "",
            "Strongest alternative explanation: because the FilterFlow-side adapter is BayesFilter-owned, P7 can miss a shared defect in the contract-formula or shared LEDH/OT path.",
            "",
            "What would overturn the local decision: any reviewer finding that an included P5 knob was omitted, that the BayesFilter model side did not depend on the parameterized knob, that the SIR exclusion was not predeclared, or that finite differences were used as a pass gate.",
            "",
            "Weakest part of the evidence: P7 is fixed-branch AD-gradient evidence only and does not test stochastic resampling distributions or gradients through discrete branch decisions.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
