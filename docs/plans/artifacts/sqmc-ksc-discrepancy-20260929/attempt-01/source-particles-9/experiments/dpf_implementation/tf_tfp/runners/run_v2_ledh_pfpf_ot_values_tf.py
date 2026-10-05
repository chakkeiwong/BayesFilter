"""Run visible P6 LEDH-PFPF-OT fixed-contract value comparisons."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/bayesfilter-dpf-mpl")
PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]

import argparse
import math
import time
from typing import Any

import tensorflow as tf
import tensorflow_probability as tfp

from experiments.dpf_implementation.tf_tfp.fixtures.common_model_suite_tf import (
    EXPECTED_V2_MODEL_IDS,
    CommonModelSpecV2,
    bayesfilter_model_for_spec_v2,
    common_model_specs_v2,
    complete_k_tf,
    observation_residual_tf,
    range_bearing_observation_tf,
    structural_observation_mean_tf,
)
from experiments.dpf_implementation.tf_tfp.flows.jacobians_tf import (
    linear_observation_jacobian_tf,
    range_bearing_jacobian_tf,
)
from experiments.dpf_implementation.tf_tfp.flows.ledh_tf import ledh_flow_batch_tf
from experiments.dpf_implementation.tf_tfp.resampling.annealed_transport_tf import (
    annealed_transport_resample_tf,
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
from experiments.dpf_implementation.tf_tfp.runners.run_v2_bootstrap_ot_values_tf import (
    _contract_observation_log_density,
    _contract_transition_log_density,
    _contract_transition_mean,
    _ess_from_log_weights,
    _field_tolerance as _bootstrap_field_tolerance,
    _json_finite,
    _mvn_log_prob,
    _no_transport_diagnostics,
    _predator_transition_mean,
    _sir_transition_mean,
    _tensor,
    _transport_summary,
    _uniform_log_weights,
    _weighted_mean_variance,
)


tfd = tfp.distributions
DTYPE = tf.float64
PLAN_PATH = (
    "docs/plans/bayesfilter-dpf-v2-algorithm-full-bf-filterflow-comparison-"
    "p6-ledh-pfpf-ot-values-subplan-2026-06-07.md"
)
RESULT_PATH = (
    "docs/plans/bayesfilter-dpf-v2-algorithm-full-bf-filterflow-comparison-"
    "p6-ledh-pfpf-ot-values-result-2026-06-07.md"
)
P0_VISIBLE_JSON_PATH = OUTPUT_DIR / "dpf_v2_algorithm_full_comparison_p0_visible_governance_2026-06-08.json"
P1_JSON_PATH = OUTPUT_DIR / "dpf_v2_algorithm_full_comparison_p1_architecture_2026-06-07.json"
P5_JSON_PATH = OUTPUT_DIR / "dpf_v2_ledh_pfpf_ot_contracts_2026-06-07.json"
JSON_PATH = OUTPUT_DIR / "dpf_v2_ledh_pfpf_ot_values_2026-06-07.json"
REPORT_PATH = REPORT_DIR / "dpf-v2-ledh-pfpf-ot-values-2026-06-07.md"
P5_PASS_DECISION = "PASS_P5_LEDH_PFPF_OT_CONTRACTS_READY_FOR_P6"
PASS_DECISION = "PASS_P6_LEDH_PFPF_OT_VALUES_READY_FOR_P7"
P5_BUNDLE_CHECKSUM = "20b7cb9a05a2de41bbf139a90bff8c8465f7bda7e4fd84fe6b83ee48a09c39f4"
P5_REPRODUCIBILITY_DIGEST = "6770ce953db079fe003d62e631e8a379cf68e6889006d583b39d70e6f5139661"
VALUE_TOLERANCE = 5e-10
LEDGER_TOLERANCE = 5e-10
TRANSPORT_TOLERANCE = 5e-10
EXPECTED_DECISIONS = {
    "PENDING_CLAUDE_REVIEW",
    PASS_DECISION,
    "P6_LEDH_PFPF_OT_VALUES_CLASSIFIED_MISMATCH_PENDING_REVIEW",
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
    _preflight_prior_artifacts(p0_payload, p1_payload, p5_payload)
    governance_evidence = _governance_evidence(p0_payload, p1_payload, p5_payload)

    specs = common_model_specs_v2()
    contracts = list(p5_payload["contracts"])
    spec_by_id = {spec.model_id: spec for spec in specs}
    cells = []
    for contract in contracts:
        spec = spec_by_id[str(contract["model_id"])]
        bayesfilter = _run_bayesfilter_adapter(spec, contract)
        filterflow = _run_filterflow_side_adapter(contract)
        cells.append(_cell(spec, contract, bayesfilter, filterflow))

    decision = _decision(cells)
    return {
        "metadata_date": "2026-06-07",
        "created_at_utc": utc_now(),
        "phase": "P6",
        "execution_route": "VISIBLE_IN_DIALOGUE",
        "decision": decision,
        "question": (
            "Do BayesFilter and BayesFilter-owned FilterFlow-side adapters "
            "match LEDH-PFPF-OT fixed-contract values and ledgers for all six V2 rows?"
        ),
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "p0_visible_json_path": str(P0_VISIBLE_JSON_PATH.relative_to(REPO_ROOT)),
        "p1_architecture_json_path": str(P1_JSON_PATH.relative_to(REPO_ROOT)),
        "p5_contracts_json_path": str(P5_JSON_PATH.relative_to(REPO_ROOT)),
        "p5_contract_bundle_checksum": p5_payload["contract_bundle_checksum"],
        "p5_reproducibility_digest": p5_payload["reproducibility_digest"],
        "required_v2_model_ids": list(EXPECTED_V2_MODEL_IDS),
        "tolerances": {
            "value_abs": VALUE_TOLERANCE,
            "ledger_abs": LEDGER_TOLERANCE,
            "transport_abs": TRANSPORT_TOLERANCE,
        },
        "adapter_policy": {
            "bayesfilter_adapter": "BayesFilter model methods plus P5 LEDH flow contract",
            "filterflow_side_adapter": (
                "BayesFilter-owned contract-formula adapter plus shared "
                "filterflow_style_annealed_transport_tf component"
            ),
            "filterflow_checkout_mutated": not governance_evidence["localsource_filterflow_not_mutated"],
            "filterflow_checkout_executed": False,
            "neither_side_is_oracle": True,
        },
        "governance_evidence": governance_evidence,
        "primary_criterion_fields": {
            "primary_ledger_fields": _primary_fields(),
            "transport_fields": _transport_fields(),
            "all_rows_matched": all(cell["status"] == "MATCHED" for cell in cells),
            "all_contract_checksums_preserved": all(
                cell["contract"]["contract_checksum"] == cell["bayesfilter"]["contract_checksum"]
                == cell["filterflow"]["contract_checksum"]
                for cell in cells
            ),
            "all_adapter_input_checksums_preserved": all(
                cell["metrics"].get("adapter_input_checksum_matches_contract_payload", False)
                for cell in cells
            ),
            "all_adapter_input_checksums_match_between_adapters": all(
                cell["metrics"].get("adapter_input_checksums_match_each_other", False)
                for cell in cells
            ),
            "all_fixed_masks_preserved": all(
                cell["metrics"].get("fixed_mask_matches_contract", False)
                for cell in cells
            ),
        },
        "veto_diagnostics": {
            "missing_v2_row": tuple(cell["model"] for cell in cells) != EXPECTED_V2_MODEL_IDS,
            "row_order_mismatch": tuple(cell["model"] for cell in cells) != EXPECTED_V2_MODEL_IDS,
            "p5_contract_checksum_changed": p5_payload["contract_bundle_checksum"] != P5_BUNDLE_CHECKSUM,
            "p5_reproducibility_digest_changed": p5_payload["reproducibility_digest"] != P5_REPRODUCIBILITY_DIGEST,
            "runtime_branch_mask_differs_from_p5": any(
                not cell["metrics"].get("fixed_mask_matches_contract", False)
                for cell in cells
            ),
            "transport_settings_differ_from_p5": any(
                not cell["metrics"].get("transport_settings_match_contract", False)
                for cell in cells
            ),
            "pfpf_correction_equation_mismatch": any(
                not cell["metrics"].get("pfpf_correction_matches", False)
                for cell in cells
            ),
            "value_delta_exceeds_tolerance": any(
                cell.get("metrics", {}).get("scalar_abs_delta", math.inf) > VALUE_TOLERANCE
                for cell in cells
            ),
            "ledger_delta_exceeds_tolerance": any(
                not cell.get("metrics", {}).get("all_primary_fields_within_tolerance", False)
                for cell in cells
            ),
            "nonfinite_scalar_or_ledger": any(_cell_nonfinite(cell) for cell in cells),
            "adapter_input_checksum_mismatch": any(
                not cell["metrics"].get("adapter_input_checksum_matches_contract_payload", False)
                for cell in cells
            ),
            "unclassified_ledger_mismatch": any(
                cell["status"] not in {"MATCHED", "EXPLAINED_MISMATCH"} for cell in cells
            ),
            "localsource_filterflow_mutated": not governance_evidence["localsource_filterflow_not_mutated"],
            "student_command_or_metric": not governance_evidence["student_commands_absent"],
            "oracle_framing": not governance_evidence["oracle_framing_forbidden"],
            "finite_difference_promoted_to_gradient_gate": not governance_evidence[
                "finite_differences_diagnostic_only"
            ],
        },
        "explanatory_only_fields": {
            "status_counts": _status_counts(cells),
            "max_abs_delta": _max_abs_delta(cells),
            "ess_policy": "ESS is reported as ledger context only; fixed P5 masks are the branch source.",
            "runtime_policy": "runtime is explanatory only",
            "transport_residual_policy": "transport residuals can veto nonfinite or mismatched ledger fields but do not promote correctness",
        },
        "cells": cells,
        "summary": _summary(cells),
        "review_round": 5,
        "open_material_blockers": [],
        "repair_amendment_required": False,
        "next_allowed_action": "begin P7 PRECHECK visibly in current dialogue",
        "artifact_paths": {
            "json": str(JSON_PATH.relative_to(REPO_ROOT)),
            "markdown_report": str(REPORT_PATH.relative_to(REPO_ROOT)),
            "phase_result": RESULT_PATH,
        },
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners.run_v2_ledh_pfpf_ot_values_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_claims": [
            "P6 validates deterministic fixed-contract LEDH-PFPF-OT value agreement only.",
            "P6 does not establish LEDH-PFPF-OT gradient agreement.",
            "P6 does not establish stochastic resampling distribution correctness.",
            "P6 does not prove BayesFilter or FilterFlow correctness.",
            "P6 does not make a student implementation claim.",
            "P6 does not make a GPU, scalability, deployment, or production-readiness claim.",
            "P6 is not full-comparison success and does not establish P7 or P8 success.",
        ],
    }


def _run_bayesfilter_adapter(spec: CommonModelSpecV2, contract: dict[str, Any]) -> dict[str, Any]:
    model = bayesfilter_model_for_spec_v2(spec)
    return _run_contract(
        spec=spec,
        contract=contract,
        backend="bayesfilter_v2_ledh_pfpf_ot_fixed_contract",
        transition_mean=lambda previous: _bayesfilter_transition_mean(spec, model, _tensor(contract["theta"]), previous),
        transition_log_density=lambda previous, post, step: model.transition_log_density(
            _tensor(contract["theta"]),
            previous,
            post,
            t=step + 1,
        ),
        observation_log_density=lambda post, observation, step: model.observation_log_density(
            _tensor(contract["theta"]),
            post,
            observation,
            t=step + 1,
        ),
    )


def _run_filterflow_side_adapter(contract: dict[str, Any]) -> dict[str, Any]:
    spec = next(spec for spec in common_model_specs_v2() if spec.model_id == contract["model_id"])
    return _run_contract(
        spec=spec,
        contract=contract,
        backend="filterflow_side_v2_ledh_pfpf_ot_contract_formula_adapter",
        transition_mean=lambda previous: _contract_transition_mean(contract, previous),
        transition_log_density=lambda previous, post, step: _contract_transition_log_density(
            contract,
            previous,
            post,
            step,
        ),
        observation_log_density=lambda post, observation, step: _contract_observation_log_density(
            contract,
            post,
            observation,
            step,
        ),
    )


def _run_contract(
    *,
    spec: CommonModelSpecV2,
    contract: dict[str, Any],
    backend: str,
    transition_mean: Any,
    transition_log_density: Any,
    observation_log_density: Any,
) -> dict[str, Any]:
    particles = tf.convert_to_tensor(contract["initial_particles"], DTYPE)
    innovations = tf.convert_to_tensor(contract["transition_innovations"], DTYPE)
    observations = tf.convert_to_tensor(contract["observations"], DTYPE)
    fixed_mask = [bool(flag) for flag in contract["fixed_ess_trigger_mask"]]
    n_particles = int(particles.shape[0])
    log_weights = _uniform_log_weights(n_particles)
    total_scalar = tf.zeros([], DTYPE)
    resampling_count = 0
    ledger = []

    for step in range(int(contract["horizon"])):
        ancestors = particles
        previous_log_weights = log_weights
        prior_mean = transition_mean(ancestors)
        pre_flow = _complete_structural_if_needed(spec, contract, ancestors, prior_mean + innovations[step])
        flow = _ledh_flow(
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
        weights, increment, normalized_log_weights = _normalize_log_weights(corrected)
        total_scalar = total_scalar + increment
        ess = _ess_from_log_weights(normalized_log_weights)
        filtered_mean, filtered_variance = _weighted_mean_variance(post_flow, weights)
        resampling_applied = bool(fixed_mask[step])
        if resampling_applied:
            transport = annealed_transport_resample_tf(
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
            post_transport_particles = tf.cast(transport.particles, DTYPE)
            post_transport_log_weights = tf.cast(transport.log_weights, DTYPE)
            transport_matrix = tf.cast(transport.transport_matrix, DTYPE)
            transport_diagnostics = dict(transport.diagnostics)
            resampling_count += 1
        else:
            post_transport_particles = post_flow
            post_transport_log_weights = normalized_log_weights
            transport_matrix = tf.zeros([n_particles, n_particles], DTYPE)
            transport_diagnostics = _no_transport_diagnostics(contract)
        ledger.append(
            _step_ledger(
                step=step,
                fixed_ess_trigger=resampling_applied,
                ancestors=ancestors,
                previous_log_weights=previous_log_weights,
                prior_mean=prior_mean,
                transition_innovations=innovations[step],
                observation=observations[step],
                flow=flow,
                target_transition_log_density=target_transition,
                target_observation_log_density=target_observation,
                corrected_log_weights=corrected,
                normalized_log_weights=normalized_log_weights,
                weights=weights,
                incremental_log_normalizer=increment,
                ess=ess,
                filtered_mean=filtered_mean,
                filtered_variance=filtered_variance,
                resampling_applied=resampling_applied,
                transport_matrix=transport_matrix,
                transport_diagnostics=transport_diagnostics,
                post_transport_particles=post_transport_particles,
                post_transport_log_weights=post_transport_log_weights,
            )
        )
        particles = post_transport_particles
        log_weights = post_transport_log_weights

    return {
        "status": "executed",
        "backend": backend,
        "model_id": contract["model_id"],
        "scalar": scalar(total_scalar),
        "finite": _ledger_finite(ledger) and bool(tf.math.is_finite(total_scalar).numpy()),
        "resampling_count": int(resampling_count),
        "fixed_ess_trigger_mask": fixed_mask,
        "ledger": ledger,
        "contract_checksum": contract["contract_checksum"],
        "adapter_input_checksum": stable_digest(contract),
    }


def _ledh_flow(
    *,
    spec: CommonModelSpecV2,
    contract: dict[str, Any],
    ancestors: tf.Tensor,
    pre_flow: tf.Tensor,
    prior_mean: tf.Tensor,
    observation: tf.Tensor,
) -> dict[str, Any]:
    model_id = spec.model_id
    p = contract["parameters"]
    if model_id == "lgssm_2d_h25_rich":
        observation_jacobian = linear_observation_jacobian_tf(_tensor(p["C"]))
        result = ledh_flow_batch_tf(
            pre_flow_particles=pre_flow,
            ancestors=ancestors,
            observation=observation,
            transition_matrix=_tensor(p["A"]),
            transition_covariance=_tensor(p["Q"]),
            observation_covariance=_tensor(p["R"]),
            observation_fn=lambda x: tf.linalg.matvec(_tensor(p["C"]), x),
            observation_jacobian_fn=observation_jacobian,
            observation_residual_fn=lambda predicted, observed: tf.reshape(tf.cast(observed, DTYPE), [-1])
            - tf.reshape(tf.cast(predicted, DTYPE), [-1]),
        )
        return _flow_result_dict(result)
    if model_id == "range_bearing_4d_h20_rich":
        result = ledh_flow_batch_tf(
            pre_flow_particles=pre_flow,
            ancestors=ancestors,
            observation=observation,
            transition_matrix=_tensor(p["A"]),
            transition_covariance=_tensor(p["Q"]),
            observation_covariance=_tensor(p["R"]),
            observation_fn=range_bearing_observation_tf,
            observation_jacobian_fn=range_bearing_jacobian_tf,
            observation_residual_fn=observation_residual_tf,
        )
        return _flow_result_dict(result)
    if model_id == "sv_1d_h18_rich":
        prior_var = _tensor(p["sigma"]) * _tensor(p["sigma"])
        z = tf.math.log(tf.reshape(tf.cast(observation, DTYPE), [1])[0] ** 2 + tf.constant(1e-6, DTYPE))
        post, logdet, q0_log, diagnostics = _scalar_ledh_flow(
            pre=tf.reshape(pre_flow[:, 0], [-1]),
            prior_mean=tf.reshape(prior_mean[:, 0], [-1]),
            prior_var=prior_var,
            observation_surrogate=z,
            obs_var=tf.constant(2.0, DTYPE),
            component_id="v2_scalar_sv_ledh_flow",
        )
        return {
            "pre_flow_particles": pre_flow,
            "post_flow_particles": post[:, None],
            "pre_flow_log_density": q0_log,
            "forward_log_det": logdet,
            "local_posterior_means": post[:, None],
            "local_posterior_covariances": tf.reshape(prior_var * tf.exp(2.0 * logdet), [-1, 1, 1]),
            "diagnostics": diagnostics,
        }
    if model_id == "structural_ar1_quadratic_h16":
        post_m, logdet, q0_log, diagnostics = _structural_ledh_flow(
            pre_m=pre_flow[:, 0],
            prior_mean=prior_mean[:, 0],
            previous_m=ancestors[:, 0],
            previous_k=ancestors[:, 1],
            observation=tf.reshape(tf.cast(observation, DTYPE), [1])[0],
            parameters=p,
        )
        post_k = _structural_complete_k(p, ancestors[:, 1], ancestors[:, 0], post_m)
        post = tf.stack([post_m, post_k], axis=1)
        return {
            "pre_flow_particles": pre_flow,
            "post_flow_particles": post,
            "pre_flow_log_density": q0_log,
            "forward_log_det": logdet,
            "local_posterior_means": post,
            "local_posterior_covariances": tf.zeros([int(pre_flow.shape[0]), 2, 2], DTYPE),
            "diagnostics": diagnostics,
        }
    if model_id == "spatial_sir_j3_rk4":
        return _autodiff_local_affine_flow(
            pre_flow=pre_flow,
            prior_mean=prior_mean,
            transition_covariance=_tensor(p["process_covariance"]),
            observation_covariance=_tensor(p["observation_covariance"]),
            observation=observation,
            observation_fn=lambda x: x[1::2],
            observation_jacobian_fn=lambda _x: _sir_observation_jacobian(int(pre_flow.shape[1])),
            residual_fn=lambda predicted, observed: tf.reshape(tf.cast(observed, DTYPE), [-1])
            - tf.reshape(tf.cast(predicted, DTYPE), [-1]),
            component_id="v2_spatial_sir_autodiff_ledh_flow",
        )
    if model_id == "predator_prey_rk4":
        dim = int(pre_flow.shape[1])
        return _autodiff_local_affine_flow(
            pre_flow=pre_flow,
            prior_mean=prior_mean,
            transition_covariance=_tensor(p["process_covariance"]),
            observation_covariance=_tensor(p["observation_covariance"]),
            observation=observation,
            observation_fn=lambda x: tf.cast(x, DTYPE),
            observation_jacobian_fn=lambda _x: tf.eye(dim, dtype=DTYPE),
            residual_fn=lambda predicted, observed: tf.reshape(tf.cast(observed, DTYPE), [-1])
            - tf.reshape(tf.cast(predicted, DTYPE), [-1]),
            component_id="v2_predator_prey_autodiff_ledh_flow",
        )
    raise ValueError(f"unknown model id: {model_id}")


def _flow_result_dict(result: Any) -> dict[str, Any]:
    return {
        "pre_flow_particles": tf.cast(result.pre_flow_particles, DTYPE),
        "post_flow_particles": tf.cast(result.post_flow_particles, DTYPE),
        "pre_flow_log_density": tf.cast(result.pre_flow_log_density, DTYPE),
        "forward_log_det": tf.cast(result.forward_log_det, DTYPE),
        "local_posterior_means": tf.cast(result.local_posterior_means, DTYPE),
        "local_posterior_covariances": tf.cast(result.local_posterior_covariances, DTYPE),
        "diagnostics": dict(result.diagnostics),
    }


def _scalar_ledh_flow(
    *,
    pre: tf.Tensor,
    prior_mean: tf.Tensor,
    prior_var: tf.Tensor,
    observation_surrogate: tf.Tensor,
    obs_var: tf.Tensor,
    component_id: str,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, dict[str, Any]]:
    post_var = 1.0 / (1.0 / prior_var + 1.0 / obs_var)
    post_mean = post_var * (prior_mean / prior_var + observation_surrogate / obs_var)
    scale = tf.sqrt(post_var / prior_var)
    post = post_mean + scale * (pre - prior_mean)
    logdet = tf.math.log(scale) * tf.ones_like(pre)
    q0_log = _normal_logpdf(pre - prior_mean, tf.sqrt(prior_var))
    diagnostics = _scalar_flow_diagnostics(
        component_id=component_id,
        pre=pre,
        post=post,
        logdet=logdet,
        q0_log=q0_log,
        min_jacobian=tf.reduce_min(tf.abs(scale)),
    )
    return post, logdet, q0_log, diagnostics


def _structural_ledh_flow(
    *,
    pre_m: tf.Tensor,
    prior_mean: tf.Tensor,
    previous_m: tf.Tensor,
    previous_k: tf.Tensor,
    observation: tf.Tensor,
    parameters: dict[str, Any],
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, dict[str, Any]]:
    prior_var = _tensor(parameters["sigma"]) * _tensor(parameters["sigma"])
    obs_var = _tensor(parameters["observation_scale"]) * _tensor(parameters["observation_scale"])
    pre_k = _structural_complete_k(parameters, previous_k, previous_m, pre_m)
    g_pre = structural_observation_mean_tf(tf.stack([pre_m, pre_k], axis=1), _tensor(parameters["lambda"]))
    jacobian = (
        _tensor(parameters["b"])
        + 2.0 * _tensor(parameters["c"]) * pre_m
        + _tensor(parameters["d"]) * previous_m
        + _tensor(parameters["lambda"])
    )
    intercept = g_pre - jacobian * pre_m
    post_var = 1.0 / (1.0 / prior_var + jacobian * jacobian / obs_var)
    post_mean = post_var * (prior_mean / prior_var + jacobian * (observation - intercept) / obs_var)
    scale = tf.sqrt(tf.maximum(post_var, tf.constant(1e-12, DTYPE)) / prior_var)
    post_m = post_mean + scale * (pre_m - prior_mean)
    logdet = tf.math.log(scale)
    q0_log = _normal_logpdf(pre_m - prior_mean, tf.sqrt(prior_var))
    diagnostics = _scalar_flow_diagnostics(
        component_id="v2_structural_split_scalar_ledh_flow",
        pre=pre_m,
        post=post_m,
        logdet=logdet,
        q0_log=q0_log,
        min_jacobian=tf.reduce_min(tf.abs(scale)),
    )
    diagnostics["singular_completion_policy"] = "density_on_m_recomplete_k"
    diagnostics["max_abs_structural_jacobian"] = scalar(tf.reduce_max(tf.abs(jacobian)))
    return post_m, logdet, q0_log, diagnostics


def _autodiff_local_affine_flow(
    *,
    pre_flow: tf.Tensor,
    prior_mean: tf.Tensor,
    transition_covariance: tf.Tensor,
    observation_covariance: tf.Tensor,
    observation: tf.Tensor,
    observation_fn: Any,
    observation_jacobian_fn: Any,
    residual_fn: Any,
    component_id: str,
) -> dict[str, Any]:
    x0_batch = tf.cast(pre_flow, DTYPE)
    prior_mean = tf.cast(prior_mean, DTYPE)
    q = _stabilize_covariance(transition_covariance)
    r = _stabilize_covariance(observation_covariance)
    q_chol = tf.linalg.cholesky(q)
    q_precision = tf.linalg.cholesky_solve(q_chol, tf.eye(int(q.shape[0]), dtype=DTYPE))
    r_chol = tf.linalg.cholesky(r)
    r_precision = tf.linalg.cholesky_solve(r_chol, tf.eye(int(r.shape[0]), dtype=DTYPE))
    mapped = []
    logdets = []
    post_means = []
    post_covariances = []
    min_singulars = []
    max_singulars = []
    for x0, mean in zip(tf.unstack(x0_batch, axis=0), tf.unstack(prior_mean, axis=0), strict=True):
        h_ref = tf.cast(observation_fn(x0), DTYPE)
        h_jac = tf.cast(observation_jacobian_fn(x0), DTYPE)
        residual = tf.reshape(residual_fn(h_ref, observation), [-1])
        pseudo_observation = tf.linalg.matvec(h_jac, x0) + residual
        post_precision = q_precision + tf.transpose(h_jac) @ r_precision @ h_jac
        post_cov = tf.linalg.inv(_stabilize_covariance(post_precision))
        post_cov = _stabilize_covariance(post_cov)
        info = (
            tf.linalg.matvec(q_precision, mean)
            + tf.linalg.matvec(tf.transpose(h_jac) @ r_precision, pseudo_observation)
        )
        post_mean = tf.linalg.matvec(post_cov, info)
        post_chol = tf.linalg.cholesky(post_cov)
        q_inv = tf.linalg.triangular_solve(q_chol, tf.eye(int(q_chol.shape[0]), dtype=DTYPE))
        affine = post_chol @ q_inv
        x1 = post_mean + tf.linalg.matvec(affine, x0 - mean)
        logdet = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(post_chol))) - tf.reduce_sum(
            tf.math.log(tf.linalg.diag_part(q_chol))
        )
        singulars = tf.linalg.svd(affine, compute_uv=False)
        mapped.append(x1)
        logdets.append(logdet)
        post_means.append(post_mean)
        post_covariances.append(post_cov)
        min_singulars.append(tf.reduce_min(singulars))
        max_singulars.append(tf.reduce_max(singulars))
    post_flow = tf.stack(mapped, axis=0)
    forward_log_det = tf.stack(logdets, axis=0)
    q0_log = _mvn_log_prob(x0_batch, prior_mean, q)
    diagnostics = {
        "component_id": component_id,
        "map_convention": "x1 = local_posterior_mean + L_post L_prior^{-1}(x0 - prior_mean)",
        "forward_log_det": "frozen_local_affine_log_abs_det",
        "forward_log_det_scope": "local_observation_jacobian_held_fixed_per_particle",
        "finite_pre_flow": _finite_tensor(x0_batch),
        "finite_post_flow": _finite_tensor(post_flow),
        "finite_forward_log_det": _finite_tensor(forward_log_det),
        "finite_pre_flow_log_density": _finite_tensor(q0_log),
        "min_jacobian_singular_value": scalar(tf.reduce_min(tf.stack(min_singulars))),
        "max_jacobian_singular_value": scalar(tf.reduce_max(tf.stack(max_singulars))),
        "backend": "tensorflow",
    }
    return {
        "pre_flow_particles": x0_batch,
        "post_flow_particles": post_flow,
        "pre_flow_log_density": q0_log,
        "forward_log_det": forward_log_det,
        "local_posterior_means": tf.stack(post_means, axis=0),
        "local_posterior_covariances": tf.stack(post_covariances, axis=0),
        "diagnostics": diagnostics,
    }


def _step_ledger(
    *,
    step: int,
    fixed_ess_trigger: bool,
    ancestors: tf.Tensor,
    previous_log_weights: tf.Tensor,
    prior_mean: tf.Tensor,
    transition_innovations: tf.Tensor,
    observation: tf.Tensor,
    flow: dict[str, Any],
    target_transition_log_density: tf.Tensor,
    target_observation_log_density: tf.Tensor,
    corrected_log_weights: tf.Tensor,
    normalized_log_weights: tf.Tensor,
    weights: tf.Tensor,
    incremental_log_normalizer: tf.Tensor,
    ess: tf.Tensor,
    filtered_mean: tf.Tensor,
    filtered_variance: tf.Tensor,
    resampling_applied: bool,
    transport_matrix: tf.Tensor,
    transport_diagnostics: dict[str, Any],
    post_transport_particles: tf.Tensor,
    post_transport_log_weights: tf.Tensor,
) -> dict[str, Any]:
    transport_summary = _transport_summary(transport_matrix)
    pfpf_reconstructed = (
        previous_log_weights
        + target_transition_log_density
        + target_observation_log_density
        - flow["pre_flow_log_density"]
        + flow["forward_log_det"]
    )
    return {
        "step": int(step),
        "fixed_ess_trigger": bool(fixed_ess_trigger),
        "ancestors": tensor_to_json(ancestors),
        "previous_log_weights": tensor_to_json(previous_log_weights),
        "previous_weights": tensor_to_json(tf.exp(previous_log_weights)),
        "prior_mean": tensor_to_json(prior_mean),
        "transition_innovations": tensor_to_json(transition_innovations),
        "observation": tensor_to_json(observation),
        "pre_flow_particles": tensor_to_json(flow["pre_flow_particles"]),
        "ledh_local_posterior_means": tensor_to_json(flow["local_posterior_means"]),
        "ledh_local_posterior_covariances": tensor_to_json(flow["local_posterior_covariances"]),
        "post_flow_particles": tensor_to_json(flow["post_flow_particles"]),
        "pre_flow_log_density": tensor_to_json(flow["pre_flow_log_density"]),
        "forward_log_det": tensor_to_json(flow["forward_log_det"]),
        "target_transition_log_density": tensor_to_json(target_transition_log_density),
        "target_observation_log_density": tensor_to_json(target_observation_log_density),
        "pfpf_corrected_log_weights": tensor_to_json(corrected_log_weights),
        "pfpf_reconstructed_log_weights": tensor_to_json(pfpf_reconstructed),
        "pfpf_correction_max_abs_residual": scalar(tf.reduce_max(tf.abs(corrected_log_weights - pfpf_reconstructed))),
        "normalized_log_weights": tensor_to_json(normalized_log_weights),
        "weights": tensor_to_json(weights),
        "incremental_log_normalizer": scalar(incremental_log_normalizer),
        "ess": scalar(ess),
        "filtered_mean": tensor_to_json(filtered_mean),
        "filtered_variance": tensor_to_json(filtered_variance),
        "resampling_applied": bool(resampling_applied),
        "transport_matrix": tensor_to_json(transport_matrix),
        "transport_matrix_summary": transport_summary,
        "transport_matrix_checksum": stable_digest(tensor_to_json(transport_matrix)),
        "transport_diagnostics": _jsonable_for_runner(transport_diagnostics),
        "post_transport_particles": tensor_to_json(post_transport_particles),
        "post_transport_log_weights": tensor_to_json(post_transport_log_weights),
        "post_transport_weights": tensor_to_json(tf.exp(post_transport_log_weights)),
        "ledh_diagnostics": _jsonable_for_runner(flow["diagnostics"]),
    }


def _normalize_log_weights(log_weights: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    log_weights = tf.cast(log_weights, DTYPE)
    increment = tf.reduce_logsumexp(log_weights)
    normalized_log_weights = log_weights - increment
    return tf.exp(normalized_log_weights), increment, normalized_log_weights


def _bayesfilter_transition_mean(
    spec: CommonModelSpecV2,
    model: Any,
    theta: tf.Tensor,
    particles: tf.Tensor,
) -> tf.Tensor:
    particles = tf.convert_to_tensor(particles, DTYPE)
    if spec.model_id in {"lgssm_2d_h25_rich", "range_bearing_4d_h20_rich"}:
        return tf.linalg.matmul(
            particles,
            tf.convert_to_tensor(spec.parameters["A"], DTYPE),
            transpose_b=True,
        )
    if spec.model_id == "sv_1d_h18_rich":
        mu = tf.convert_to_tensor(spec.parameters["mu"], DTYPE)
        phi = tf.convert_to_tensor(spec.parameters["phi"], DTYPE)
        return mu + phi * (particles - mu)
    if spec.model_id == "structural_ar1_quadratic_h16":
        return model.transition_mean(particles)
    if spec.model_id == "spatial_sir_j3_rk4":
        return model.transition_mean(particles)
    if spec.model_id == "predator_prey_rk4":
        return model.transition_mean(theta, particles)
    raise ValueError(f"unknown v2 model spec: {spec.model_id}")


def _complete_structural_if_needed(
    spec: CommonModelSpecV2,
    contract: dict[str, Any],
    ancestors: tf.Tensor,
    predicted: tf.Tensor,
) -> tf.Tensor:
    if spec.model_id != "structural_ar1_quadratic_h16":
        return predicted
    p = contract["parameters"]
    current_m = predicted[:, 0]
    current_k = _structural_complete_k(p, ancestors[:, 1], ancestors[:, 0], current_m)
    return tf.stack([current_m, current_k], axis=1)


def _structural_complete_k(
    parameters: dict[str, Any],
    previous_k: tf.Tensor,
    previous_m: tf.Tensor,
    current_m: tf.Tensor,
) -> tf.Tensor:
    return complete_k_tf(
        previous_k=previous_k,
        previous_m=previous_m,
        current_m=current_m,
        a=_tensor(parameters["a"]),
        b=_tensor(parameters["b"]),
        c=_tensor(parameters["c"]),
        d=_tensor(parameters["d"]),
    )


def _normal_logpdf(residual: tf.Tensor, scale: tf.Tensor) -> tf.Tensor:
    variance = tf.cast(scale, DTYPE) * tf.cast(scale, DTYPE)
    return -0.5 * (
        tf.math.log(tf.constant(2.0 * math.pi, DTYPE) * variance)
        + tf.cast(residual, DTYPE) * tf.cast(residual, DTYPE) / variance
    )


def _scalar_flow_diagnostics(
    *,
    component_id: str,
    pre: tf.Tensor,
    post: tf.Tensor,
    logdet: tf.Tensor,
    q0_log: tf.Tensor,
    min_jacobian: tf.Tensor,
) -> dict[str, Any]:
    return {
        "component_id": component_id,
        "map_convention": "scalar local affine LEDH map",
        "forward_log_det": "scalar_log_abs_det",
        "finite_pre_flow": _finite_tensor(pre),
        "finite_post_flow": _finite_tensor(post),
        "finite_forward_log_det": _finite_tensor(logdet),
        "finite_pre_flow_log_density": _finite_tensor(q0_log),
        "min_jacobian_singular_value": scalar(min_jacobian),
        "max_abs_forward_log_det": scalar(tf.reduce_max(tf.abs(logdet))),
        "backend": "tensorflow",
    }


def _sir_observation_jacobian(state_dim: int) -> tf.Tensor:
    obs_dim = state_dim // 2
    rows = []
    for obs_index in range(obs_dim):
        row = [0.0] * state_dim
        row[2 * obs_index + 1] = 1.0
        rows.append(row)
    return tf.constant(rows, DTYPE)


def _stabilize_covariance(covariance: tf.Tensor, jitter: float = 1e-9) -> tf.Tensor:
    covariance = tf.cast(covariance, DTYPE)
    sym = 0.5 * (covariance + tf.transpose(covariance))
    eigvals = tf.linalg.eigvalsh(sym)
    min_eig = tf.reduce_min(eigvals)
    needed = tf.maximum(tf.constant(jitter, dtype=DTYPE) - min_eig, 0.0)
    return sym + needed * tf.eye(int(sym.shape[0]), dtype=DTYPE)


def _finite_tensor(value: tf.Tensor) -> bool:
    return bool(tf.reduce_all(tf.math.is_finite(tf.cast(value, DTYPE))).numpy())


def _cell(
    spec: CommonModelSpecV2,
    contract: dict[str, Any],
    bayesfilter: dict[str, Any],
    filterflow: dict[str, Any],
) -> dict[str, Any]:
    comparison = _compare_path_payloads(bayesfilter, filterflow, contract)
    matched = (
        comparison["all_primary_fields_within_tolerance"]
        and comparison["adapter_input_checksum_matches_contract_payload"]
        and comparison["adapter_input_checksums_match_each_other"]
        and comparison["fixed_mask_matches_contract"]
        and comparison["transport_settings_match_contract"]
        and comparison["pfpf_correction_matches"]
        and bayesfilter["finite"]
        and filterflow["finite"]
        and bayesfilter["resampling_count"] == int(contract["fixed_ess_trigger_count"])
        and filterflow["resampling_count"] == int(contract["fixed_ess_trigger_count"])
    )
    return {
        "model": spec.model_id,
        "family": spec.family,
        "implementations": ["BayesFilter", "FilterFlow-side adapter"],
        "cell_type": "v2_ledh_pfpf_ot_fixed_contract_value_path",
        "status": "MATCHED" if matched else "EXPLAINED_MISMATCH",
        "decision": f"{spec.model_id}_ledh_pfpf_ot_values_matched"
        if matched
        else f"{spec.model_id}_ledh_pfpf_ot_values_mismatch",
        "primary_criterion": "scalar and required LEDH-PFPF-OT ledger fields match within tolerance",
        "metrics": comparison,
        "mismatch_class": None if matched else "v2_ledh_pfpf_ot_value_or_ledger_delta",
        "contract": _contract_digest_view(contract),
        "bayesfilter": bayesfilter,
        "filterflow": filterflow,
        "spec_checksum": spec.checksum(),
        "non_claim": "fixed-contract LEDH-PFPF-OT value agreement is not gradient or stochastic resampling correctness",
    }


def _compare_path_payloads(left: dict[str, Any], right: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    fields = _primary_fields()
    ledger_deltas = []
    max_abs_delta = abs(float(left["scalar"]) - float(right["scalar"]))
    all_within = max_abs_delta <= VALUE_TOLERANCE
    contract_payload_checksum = stable_digest(contract)
    left_adapter_input_checksum = str(left.get("adapter_input_checksum", ""))
    right_adapter_input_checksum = str(right.get("adapter_input_checksum", ""))
    masks_left = []
    masks_right = []
    transport_settings_match = True
    pfpf_matches = True
    for left_step, right_step in zip(left["ledger"], right["ledger"], strict=True):
        masks_left.append(bool(left_step["fixed_ess_trigger"]))
        masks_right.append(bool(right_step["fixed_ess_trigger"]))
        pfpf_matches = (
            pfpf_matches
            and float(left_step["pfpf_correction_max_abs_residual"]) <= LEDGER_TOLERANCE
            and float(right_step["pfpf_correction_max_abs_residual"]) <= LEDGER_TOLERANCE
        )
        step_metrics = {"step": left_step["step"], "fields": {}}
        for field in fields:
            if field in {"fixed_ess_trigger", "resampling_applied"}:
                field_max = 0.0 if bool(left_step[field]) == bool(right_step[field]) else math.inf
            elif field == "transport_matrix_checksum":
                field_max = 0.0 if str(left_step[field]) == str(right_step[field]) else math.inf
            else:
                left_value = tf.reshape(tf.convert_to_tensor(left_step[field], DTYPE), [-1])
                right_value = tf.reshape(tf.convert_to_tensor(right_step[field], DTYPE), [-1])
                delta = tf.abs(left_value - right_value)
                field_max = float(tf.reduce_max(delta).numpy()) if int(tf.size(delta).numpy()) else 0.0
            tolerance = _field_tolerance(field)
            within = field_max <= tolerance
            all_within = all_within and within
            max_abs_delta = max(max_abs_delta, field_max)
            step_metrics["fields"][field] = {"max_abs_delta": field_max, "within_tolerance": within}
        for side in (left_step, right_step):
            diag = side["transport_diagnostics"]
            transport_settings_match = transport_settings_match and (
                float(diag["epsilon"]) == float(contract["ot_settings"]["sinkhorn_epsilon"])
                and float(diag["scaling"]) == float(contract["ot_settings"]["annealed_scaling"])
                and float(diag["convergence_threshold"]) == float(contract["ot_settings"]["annealed_convergence_threshold"])
                and int(diag["max_iterations"]) == int(contract["ot_settings"]["sinkhorn_iterations"])
                and str(diag["transport_gradient_mode"]) == str(contract["ot_settings"]["transport_gradient_mode"])
                and str(diag["application_mode"]) == str(contract["ot_settings"]["application_mode"])
            )
        ledger_deltas.append(step_metrics)
    contract_mask = [bool(flag) for flag in contract["fixed_ess_trigger_mask"]]
    return {
        "scalar_abs_delta": abs(float(left["scalar"]) - float(right["scalar"])),
        "ledger_deltas": ledger_deltas,
        "max_abs_delta": max_abs_delta,
        "all_primary_fields_within_tolerance": all_within,
        "contract_payload_checksum": contract_payload_checksum,
        "bayesfilter_adapter_input_checksum": left_adapter_input_checksum,
        "filterflow_adapter_input_checksum": right_adapter_input_checksum,
        "adapter_input_checksum_matches_contract_payload": (
            left_adapter_input_checksum == contract_payload_checksum
            and right_adapter_input_checksum == contract_payload_checksum
        ),
        "adapter_input_checksums_match_each_other": left_adapter_input_checksum == right_adapter_input_checksum,
        "fixed_mask_matches_contract": masks_left == contract_mask and masks_right == contract_mask,
        "bayesfilter_fixed_mask": masks_left,
        "filterflow_fixed_mask": masks_right,
        "contract_fixed_mask": contract_mask,
        "transport_settings_match_contract": bool(transport_settings_match),
        "pfpf_correction_matches": bool(pfpf_matches),
        "value_tolerance": VALUE_TOLERANCE,
        "ledger_tolerance": LEDGER_TOLERANCE,
        "transport_tolerance": TRANSPORT_TOLERANCE,
        "explanatory_fields": [
            "ess",
            "filtered_mean",
            "filtered_variance",
            "ledh_diagnostics",
            "transport_residuals",
        ],
    }


def _primary_fields() -> list[str]:
    return [
        "ancestors",
        "prior_mean",
        "pre_flow_particles",
        "ledh_local_posterior_means",
        "ledh_local_posterior_covariances",
        "post_flow_particles",
        "pre_flow_log_density",
        "forward_log_det",
        "target_transition_log_density",
        "target_observation_log_density",
        "pfpf_corrected_log_weights",
        "normalized_log_weights",
        "incremental_log_normalizer",
        "fixed_ess_trigger",
        "resampling_applied",
        "transport_matrix",
        "transport_matrix_checksum",
        "post_transport_particles",
        "post_transport_log_weights",
    ]


def _transport_fields() -> list[str]:
    return [
        "transport_matrix",
        "transport_matrix_summary",
        "transport_matrix_checksum",
        "transport_diagnostics",
        "post_transport_particles",
        "post_transport_log_weights",
    ]


def _field_tolerance(field: str) -> float:
    if field in {"incremental_log_normalizer"}:
        return VALUE_TOLERANCE
    if field in {"transport_matrix", "post_transport_particles", "post_transport_log_weights"}:
        return TRANSPORT_TOLERANCE
    return _bootstrap_field_tolerance(field) if field in {"transport_matrix_checksum"} else LEDGER_TOLERANCE


def _contract_digest_view(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "contract_id": contract["contract_id"],
        "model_id": contract["model_id"],
        "algorithm": contract["algorithm"],
        "horizon": contract["horizon"],
        "num_particles": contract["num_particles"],
        "fixed_ess_trigger_mask": contract["fixed_ess_trigger_mask"],
        "fixed_ess_trigger_count": contract["fixed_ess_trigger_count"],
        "ledh_affine_map": contract["ledh_affine_map"],
        "forward_log_det_convention": contract["forward_log_det_convention"],
        "ot_settings": contract["ot_settings"],
        "scalar_definition": contract["scalar_definition"],
        "contract_checksum": contract["contract_checksum"],
        "component_checksums": contract["component_checksums"],
    }


def _preflight_prior_artifacts(
    p0_payload: dict[str, Any],
    p1_payload: dict[str, Any],
    p5_payload: dict[str, Any],
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
    if tuple(p0_payload.get("required_v2_model_ids", [])) != EXPECTED_V2_MODEL_IDS:
        raise ValueError("P0 model id gate failed")
    if tuple(p1_payload.get("required_v2_model_ids", [])) != EXPECTED_V2_MODEL_IDS:
        raise ValueError("P1 model id gate failed")
    if tuple(p5_payload.get("required_v2_model_ids", [])) != EXPECTED_V2_MODEL_IDS:
        raise ValueError("P5 required model id gate failed")
    ids = [contract.get("model_id") for contract in p5_payload.get("contracts", [])]
    if tuple(ids) != EXPECTED_V2_MODEL_IDS:
        raise ValueError(f"P5 contract model id gate failed: {ids}")
    diagnostics = p5_payload.get("execution_diagnostics", {})
    if diagnostics.get("ledh_pfpf_ot_values_computed") or diagnostics.get("ledh_pfpf_ot_gradients_computed"):
        raise ValueError("P5 artifact claims LEDH value or gradient execution")


def _governance_evidence(
    p0_payload: dict[str, Any],
    p1_payload: dict[str, Any],
    p5_payload: dict[str, Any],
) -> dict[str, Any]:
    p0_vetoes = p0_payload.get("veto_diagnostics", {})
    p0_explanatory = p0_payload.get("explanatory_only_diagnostics", {})
    p1_primary = p1_payload.get("primary_criterion_status", {})
    p1_vetoes = p1_payload.get("veto_diagnostics", {})
    p1_execution = p1_payload.get("execution_diagnostics", {})
    p5_vetoes = p5_payload.get("veto_diagnostics", {})
    return {
        "source_artifacts": {
            "p0_visible_governance_json": str(P0_VISIBLE_JSON_PATH.relative_to(REPO_ROOT)),
            "p1_architecture_json": str(P1_JSON_PATH.relative_to(REPO_ROOT)),
            "p5_ledh_contract_json": str(P5_JSON_PATH.relative_to(REPO_ROOT)),
        },
        "localsource_filterflow_not_mutated": (
            p0_explanatory.get("filterflow_status") == "clean"
            and p0_vetoes.get("filterflow_mutation") == "PASS"
            and p1_primary.get("filterflow_mutation_required") is False
            and p1_vetoes.get("planned_adapter_mutates_filterflow") == "PASS"
            and p5_vetoes.get("localsource_filterflow_mutated") == "PASS"
        ),
        "student_commands_absent": (
            p0_explanatory.get("student_implementation_commands_run") is False
            and p0_vetoes.get("student_command_or_student_metric") == "PASS"
            and p1_execution.get("student_implementation_commands_run") is False
            and p5_vetoes.get("student_command_or_metric") == "PASS"
        ),
        "oracle_framing_forbidden": (
            p0_vetoes.get("oracle_treatment_allowed") == "PASS"
            and p1_primary.get("filterflow_adapters_bayesfilter_owned") == "PASS"
            and p5_vetoes.get("oracle_framing") == "PASS"
        ),
        "finite_differences_diagnostic_only": (
            p0_vetoes.get("finite_difference_gradient_promotion_allowed") == "PASS"
            and p5_vetoes.get("finite_difference_promoted_to_gradient_gate") == "PASS"
        ),
        "p1_ledh_filterflow_adapter_is_bayesfilter_owned": (
            p1_primary.get("filterflow_adapters_bayesfilter_owned") == "PASS"
            and p1_payload.get("adapter_semantics", {})
            .get("ledh_pfpf_ot", {})
            .get("filterflow_support")
            == "BayesFilter-owned adapter-hosted support; not native FilterFlow LEDH"
        ),
        "p5_contract_forbids_runtime_branch_redecision": p5_vetoes.get(
            "runtime_boolean_ess_trigger_in_primary_contract"
        )
        == "PASS",
    }


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
        raise ValueError(f"P6 payload missing required fields: {sorted(missing)}")
    if payload["phase"] != "P6":
        raise ValueError(f"unexpected phase: {payload['phase']}")
    if payload["decision"] not in EXPECTED_DECISIONS:
        raise ValueError(f"P6 payload decision not passable: {payload['decision']}")
    ids = [cell["model"] for cell in payload["cells"]]
    if tuple(ids) != EXPECTED_V2_MODEL_IDS:
        raise ValueError(f"P6 cell id gate failed: {ids}")
    if payload["p5_contract_bundle_checksum"] != P5_BUNDLE_CHECKSUM:
        raise ValueError("P6 payload references changed P5 bundle checksum")
    if payload["p5_reproducibility_digest"] != P5_REPRODUCIBILITY_DIGEST:
        raise ValueError("P6 payload references changed P5 digest")
    if payload["artifact_paths"]["json"] != str(JSON_PATH.relative_to(REPO_ROOT)):
        raise ValueError("P6 JSON artifact path mismatch")
    if payload["run_manifest"].get("pre_import_cuda_visible_devices") != "-1":
        raise ValueError("P6 TensorFlow run was not CPU-only before import")
    for key in (
        "missing_v2_row",
        "row_order_mismatch",
        "p5_contract_checksum_changed",
        "p5_reproducibility_digest_changed",
        "runtime_branch_mask_differs_from_p5",
        "transport_settings_differ_from_p5",
        "pfpf_correction_equation_mismatch",
        "nonfinite_scalar_or_ledger",
        "adapter_input_checksum_mismatch",
        "unclassified_ledger_mismatch",
        "localsource_filterflow_mutated",
        "student_command_or_metric",
        "oracle_framing",
        "finite_difference_promoted_to_gradient_gate",
    ):
        if payload["veto_diagnostics"].get(key):
            raise ValueError(f"P6 veto diagnostic fired: {key}")
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
            raise ValueError(f"P6 governance evidence failed: {key}")
    if not payload["primary_criterion_fields"].get("all_adapter_input_checksums_preserved"):
        raise ValueError("P6 adapter input checksums do not match contract payload")
    if not payload["primary_criterion_fields"].get("all_adapter_input_checksums_match_between_adapters"):
        raise ValueError("P6 adapter input checksums differ between adapters")
    if "ledh_local_posterior_covariances" not in payload["primary_criterion_fields"].get(
        "primary_ledger_fields", []
    ):
        raise ValueError("P6 primary ledger fields omit LEDH local posterior covariances")
    for cell in payload["cells"]:
        if not cell.get("bayesfilter", {}).get("finite", False):
            raise ValueError(f"P6 BayesFilter nonfinite row: {cell['model']}")
        if not cell.get("filterflow", {}).get("finite", False):
            raise ValueError(f"P6 FilterFlow adapter nonfinite row: {cell['model']}")
        expected_checksum = cell["metrics"].get("contract_payload_checksum")
        if cell.get("bayesfilter", {}).get("adapter_input_checksum") != expected_checksum:
            raise ValueError(f"P6 BayesFilter adapter input checksum mismatch: {cell['model']}")
        if cell.get("filterflow", {}).get("adapter_input_checksum") != expected_checksum:
            raise ValueError(f"P6 FilterFlow adapter input checksum mismatch: {cell['model']}")
    if payload["decision"] in {"PENDING_CLAUDE_REVIEW", PASS_DECISION}:
        if payload["veto_diagnostics"].get("value_delta_exceeds_tolerance"):
            raise ValueError("P6 value delta exceeds tolerance on passable decision")
        if payload["veto_diagnostics"].get("ledger_delta_exceeds_tolerance"):
            raise ValueError("P6 ledger delta exceeds tolerance on passable decision")
        if not payload["primary_criterion_fields"].get("all_rows_matched"):
            raise ValueError("P6 not all rows matched on passable decision")
        for cell in payload["cells"]:
            if cell["status"] != "MATCHED":
                raise ValueError(f"P6 row did not match on passable decision: {cell['model']} {cell['status']}")
    elif payload["decision"] == "P6_LEDH_PFPF_OT_VALUES_CLASSIFIED_MISMATCH_PENDING_REVIEW":
        if not any(cell["status"] == "EXPLAINED_MISMATCH" for cell in payload["cells"]):
            raise ValueError("P6 mismatch decision without classified mismatch cell")


def _decision(cells: list[dict[str, Any]]) -> str:
    if all(cell["status"] == "MATCHED" for cell in cells):
        return PASS_DECISION
    return "P6_LEDH_PFPF_OT_VALUES_CLASSIFIED_MISMATCH_PENDING_REVIEW"


def _review_state_line(payload: dict[str, Any]) -> str:
    if payload["decision"] == PASS_DECISION:
        return f"review_round: {payload['review_round']} final Claude synthesis returned VERDICT: AGREE"
    if payload["decision"] == "PENDING_CLAUDE_REVIEW":
        return f"review_round: {payload['review_round']} pending chunked Claude P6 value review"
    return f"review_round: {payload['review_round']} classified mismatch pending Claude review"


def _ledger_finite(ledger: list[dict[str, Any]]) -> bool:
    return all(_json_finite(row) for row in ledger)


def _cell_nonfinite(cell: dict[str, Any]) -> bool:
    return (
        cell.get("bayesfilter", {}).get("finite") is False
        or cell.get("filterflow", {}).get("finite") is False
    )


def _status_counts(cells: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for cell in cells:
        counts[cell["status"]] = counts.get(cell["status"], 0) + 1
    return counts


def _max_abs_delta(cells: list[dict[str, Any]]) -> float | None:
    values = [
        float(cell.get("metrics", {}).get("max_abs_delta", 0.0))
        for cell in cells
        if "max_abs_delta" in cell.get("metrics", {})
    ]
    return max(values) if values else None


def _summary(cells: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_cells": len(cells),
        "models": [cell["model"] for cell in cells],
        "status_counts": _status_counts(cells),
        "max_abs_delta": _max_abs_delta(cells),
    }


def _digest_payload(payload: dict[str, Any]) -> str:
    clone = dict(payload)
    clone.pop("reproducibility_digest", None)
    if "run_manifest" in clone:
        clone["run_manifest"] = dict(clone["run_manifest"])
        clone["run_manifest"].pop("wall_time_seconds", None)
    return stable_digest(clone)


def _markdown(payload: dict[str, Any]) -> str:
    run_manifest = payload["run_manifest"]
    lines = [
        "# DPF V2 Algorithm Full Comparison P6 LEDH-PFPF-OT Values Result",
        "",
        "metadata_date: 2026-06-07",
        "visible_execution_timestamp: `2026-06-08T16:04:00+08:00`",
        "phase: P6",
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
        "- for every V2 row, BayesFilter and the BayesFilter-owned FilterFlow-side adapter consume the same frozen P5 LEDH-PFPF-OT contract;",
        "- scalar values match within declared tolerance;",
        "- required ledgers match, including pre-flow proposals, LEDH affine parameters, post-flow particles, pre-flow proposal log density, forward logdet, target transition density, observation density, PF-PF corrected log weights, frozen ESS trigger masks, OT transport matrix checksums, post-transport particles, incremental log normalizers, and final scalar.",
        "",
        "Veto diagnostics:",
        "",
        "- nonfinite proposal, logdet, corrected weight, scalar, or transport field;",
        "- runtime branch mask differs from frozen P5 mask;",
        "- transport settings differ from P5;",
        "- PF-PF correction equation mismatch;",
        "- BF/FF value delta exceeds tolerance;",
        "- unclassified ledger mismatch;",
        "- `.localsource/filterflow` mutation, student command, oracle framing, or finite differences promoted to a gradient gate.",
        "",
        "Non-claims:",
        "",
        "- P6 does not establish LEDH-PFPF-OT gradient agreement, LEDH proposal optimality, or stochastic resampling distribution correctness.",
        "- P6 does not prove BayesFilter, FilterFlow, or adapter implementation correctness.",
        "- P6 does not make a student implementation claim.",
        "- P6 does not make a GPU, scalability, deployment, or production-readiness claim.",
        "- P6 is not full-comparison success and does not establish P7 or P8 success.",
        "",
        "## Local Skeptical Phase Audit",
        "",
        "Audit status: `PASS_LOCAL_PHASE_AUDIT`.",
        "",
        "Wrong-baseline risk: controlled. P6 uses the reviewed P5 contract bundle checksum and digest as the baseline and rejects drift.",
        "",
        "Proxy-metric risk: controlled. ESS, runtime, LEDH diagnostics, and transport residuals are explanatory unless they trigger explicit finite or ledger-veto checks.",
        "",
        "Missing stop-condition risk: controlled. P6 stops on row disappearance, mask drift, transport setting drift, PF-PF mismatch, value/ledger mismatch, nonfinite values, or stale P5 checksum/digest.",
        "",
        "Unfair-comparison risk: controlled. Both adapters consume identical P5 contract bytes and the same frozen mask.",
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
        f"- P5 contract bundle checksum: `{payload['p5_contract_bundle_checksum']}`",
        f"- P5 reproducibility digest: `{payload['p5_reproducibility_digest']}`",
        f"- P6 reproducibility digest: `{payload['reproducibility_digest']}`",
        "",
        "## Path Cells",
        "",
        "| Model id | Status | Scalar delta | Max ledger delta | Fixed mask | Resampling count |",
        "|---|---|---:|---:|---|---:|",
    ]
    for cell in payload["cells"]:
        metrics = cell["metrics"]
        lines.append(
            f"| `{cell['model']}` | {cell['status']} | {metrics['scalar_abs_delta']} | "
            f"{metrics['max_abs_delta']} | `{metrics['contract_fixed_mask']}` | "
            f"{cell['bayesfilter']['resampling_count']} |"
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
            f"- status counts: `{payload['explanatory_only_fields']['status_counts']}`",
            f"- max abs delta: `{payload['explanatory_only_fields']['max_abs_delta']}`",
            f"- ESS policy: `{payload['explanatory_only_fields']['ess_policy']}`",
            f"- runtime policy: `{payload['explanatory_only_fields']['runtime_policy']}`",
            f"- transport residual policy: `{payload['explanatory_only_fields']['transport_residual_policy']}`",
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
            "| dirty status | full repo dirty state is recorded in JSON run manifest; unrelated dirty work is not interpreted as P6 evidence |",
            f"| command | `{run_manifest.get('command')}` |",
            f"| validation commands | `python -m json.tool {payload['artifact_paths']['json']}`; `CUDA_VISIBLE_DEVICES=-1 python -m experiments.dpf_implementation.tf_tfp.runners.run_v2_ledh_pfpf_ot_values_tf --validate-only`; `git diff --check` on P6 files |",
            f"| environment | `/home/chakwong/BayesFilter`; Python `{run_manifest.get('python_version')}`; TensorFlow `{run_manifest.get('package_versions', {}).get('tensorflow')}`; TFP `{run_manifest.get('package_versions', {}).get('tensorflow_probability')}` |",
            f"| CPU/GPU status | CPU-only TensorFlow run; pre-import `CUDA_VISIBLE_DEVICES={run_manifest.get('pre_import_cuda_visible_devices')}`; visible GPUs `{run_manifest.get('gpu_devices_visible')}` |",
            "| random seeds | no RNG consumed in P6; frozen particles, observations, transition innovations, and masks from P5 |",
            "| dtype | `tf.float64` / JSON dtype `float64` |",
            f"| output artifacts | `{payload['artifact_paths']['json']}`; `{payload['artifact_paths']['markdown_report']}`; `{payload['artifact_paths']['phase_result']}` |",
            "",
            "## Review State",
            "",
            _review_state_line(payload),
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
            f"| `{payload['decision']}` | all six LEDH-PFPF-OT value ledgers matched locally | all local P6 veto diagnostics clear | Claude may find adapter or artifact adequacy gaps | run chunked Claude P6 read-only review | no gradient agreement, stochastic resampling correctness, implementation proof, student claim, GPU claim, scalability, deployment, production readiness, full-comparison success, or P7/P8 success |",
            "",
            "## Post-Run Red Team",
            "",
            "Strongest alternative explanation: because both adapters are BayesFilter-owned, P6 can miss a defect shared by the contract-formula adapter and the BayesFilter model surface.",
            "",
            "What would overturn the local decision: any reviewer finding that the FilterFlow-side adapter is not the P1/P5-frozen adapter surface, that a required LEDH ledger field is omitted, that P5 masks/settings changed, or that the same-contract identity is broken.",
            "",
            "Weakest part of the evidence: P6 is fixed-contract value evidence only and does not test gradient correctness or stochastic resampling distribution behavior.",
            "",
        ]
    )
    return "\n".join(lines)


def _jsonable_for_runner(value: Any) -> Any:
    if isinstance(value, tf.Tensor):
        return tensor_to_json(value)
    if isinstance(value, dict):
        return {str(k): _jsonable_for_runner(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_jsonable_for_runner(v) for v in value]
    if isinstance(value, list):
        return [_jsonable_for_runner(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
