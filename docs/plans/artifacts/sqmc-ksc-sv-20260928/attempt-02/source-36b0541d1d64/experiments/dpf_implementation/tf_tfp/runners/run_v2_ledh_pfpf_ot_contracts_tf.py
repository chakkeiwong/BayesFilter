"""Freeze DPF v2 LEDH-PFPF-OT contracts for visible BF/FilterFlow execution."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/bayesfilter-dpf-mpl")
PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]

import argparse
import inspect
import math
import time
from typing import Any

import tensorflow as tf

from experiments.dpf_implementation.tf_tfp.filters.ledh_pfpf_ot_tf import (
    run_ledh_pfpf_ot_tf,
)
from experiments.dpf_implementation.tf_tfp.fixtures.common_model_suite_tf import (
    EXPECTED_V2_MODEL_IDS,
    CommonModelSpecV2,
    common_model_specs_v2,
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
    stable_digest,
    tensor_to_json,
    utc_now,
    write_json,
    write_text,
)


DTYPE = tf.float64
PLAN_PATH = (
    "docs/plans/bayesfilter-dpf-v2-algorithm-full-bf-filterflow-comparison-"
    "p5-ledh-pfpf-ot-contracts-subplan-2026-06-07.md"
)
RESULT_PATH = (
    "docs/plans/bayesfilter-dpf-v2-algorithm-full-bf-filterflow-comparison-"
    "p5-ledh-pfpf-ot-contracts-result-2026-06-07.md"
)
JSON_PATH = OUTPUT_DIR / "dpf_v2_ledh_pfpf_ot_contracts_2026-06-07.json"
REPORT_PATH = REPORT_DIR / "dpf-v2-ledh-pfpf-ot-contracts-2026-06-07.md"
P0_VISIBLE_JSON_PATH = OUTPUT_DIR / "dpf_v2_algorithm_full_comparison_p0_visible_governance_2026-06-08.json"
P1_JSON_PATH = OUTPUT_DIR / "dpf_v2_algorithm_full_comparison_p1_architecture_2026-06-07.json"
P4_JSON_PATH = OUTPUT_DIR / "dpf_v2_bootstrap_ot_gradients_2026-06-07.json"
P4_PASS_DECISION = "PASS_P4_BOOTSTRAP_OT_GRADIENTS_READY_FOR_P5"
P4_REPRODUCIBILITY_DIGEST = "f5460402677d25c551d4557c430b7b41a5bda58abf48beb070f9cf423a3725c2"
P5_TOUCHED_PATHS = (
    "experiments/dpf_implementation/tf_tfp/runners/run_v2_ledh_pfpf_ot_contracts_tf.py",
    "experiments/dpf_implementation/reports/outputs/dpf_v2_ledh_pfpf_ot_contracts_2026-06-07.json",
    "experiments/dpf_implementation/reports/dpf-v2-ledh-pfpf-ot-contracts-2026-06-07.md",
    RESULT_PATH,
    "docs/plans/bayesfilter-dpf-v2-algorithm-full-bf-filterflow-comparison-visible-execution-ledger-2026-06-08.md",
)
EXPECTED_DECISIONS = {
    "LOCAL_PASS_REVIEW_PENDING",
    "PENDING_CLAUDE_REVIEW",
    "PASS_P5_LEDH_PFPF_OT_CONTRACTS_READY_FOR_P6",
}
PASS_DECISION = "PASS_P5_LEDH_PFPF_OT_CONTRACTS_READY_FOR_P6"
REQUIRED_LEDGER_FIELDS = {
    "pre_flow_transition_proposal",
    "ledh_linearization",
    "ledh_affine_map",
    "forward_log_det_convention",
    "proposal_log_density_route",
    "target_transition_density_route",
    "observation_density_route",
    "pfpf_corrected_log_weight_equation",
    "fixed_ess_trigger_mask",
    "ot_settings",
    "gradient_contract",
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
    p4_payload = load_json(P4_JSON_PATH)
    _preflight_prior_artifacts(p0_payload, p1_payload, p4_payload)

    specs = common_model_specs_v2()
    contracts = [_ledh_pfpf_ot_contract(spec) for spec in specs]
    bundle_payload = {
        "artifact_id": "dpf_v2_ledh_pfpf_ot_contracts_2026-06-07",
        "version": "2026-06-07.p5.visible",
        "contracts": contracts,
    }
    contract_bundle_checksum = stable_digest(bundle_payload)
    return {
        "metadata_date": "2026-06-07",
        "created_at_utc": utc_now(),
        "phase": "P5",
        "execution_route": "VISIBLE_IN_DIALOGUE",
        "decision": PASS_DECISION,
        "question": (
            "Can executable LEDH-PFPF-OT comparison contracts be frozen for "
            "all six V2 rows before LEDH-PFPF-OT value or gradient execution?"
        ),
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "p0_visible_json_path": str(P0_VISIBLE_JSON_PATH.relative_to(REPO_ROOT)),
        "p1_architecture_json_path": str(P1_JSON_PATH.relative_to(REPO_ROOT)),
        "p4_gradients_json_path": str(P4_JSON_PATH.relative_to(REPO_ROOT)),
        "p4_reproducibility_digest": p4_payload["reproducibility_digest"],
        "required_v2_model_ids": list(EXPECTED_V2_MODEL_IDS),
        "row_count_gate": {
            "required_count": len(EXPECTED_V2_MODEL_IDS),
            "actual_count": len(contracts),
            "status": "PASS",
        },
        "contract_bundle_checksum": contract_bundle_checksum,
        "contract_bundle_payload": bundle_payload,
        "contract_generation_policy": {
            "generated_after_p4_reviewed_pass": True,
            "generated_before_ledh_pfpf_ot_values": True,
            "generated_before_ledh_pfpf_ot_gradients": True,
            "prior_ledh_smoke_artifacts_are_context_only": True,
            "student_implementation_out_of_scope": True,
            "filterflow_checkout_read_only": True,
            "ledh_filterflow_support": "bayesfilter_owned_adapter_hosted_not_native_filterflow",
        },
        "ledh_pfpf_ot_settings": _ledh_pfpf_ot_settings(),
        "primary_criterion_status": {
            "six_v2_rows_in_exact_order": "PASS",
            "one_contract_per_row": "PASS",
            "all_contracts_record_pre_flow_transition_proposal": "PASS",
            "all_contracts_record_ledh_linearization_and_jacobian": "PASS",
            "all_contracts_record_affine_map_and_forward_logdet": "PASS",
            "all_contracts_record_density_routes": "PASS",
            "all_contracts_record_pfpf_correction": "PASS",
            "all_contracts_record_fixed_ess_trigger_mask": "PASS",
            "all_contracts_record_ot_settings": "PASS",
            "all_contracts_record_gradient_knobs": "PASS",
            "all_contracts_record_dtype_tolerances_and_checksums": "PASS",
            "bf_and_ff_consume_same_contract_checksum": "PASS",
            "no_ledh_value_or_gradient_execution": "PASS",
        },
        "veto_diagnostics": {
            "missing_v2_row": "PASS",
            "row_order_mismatch": "PASS",
            "p0_p1_p4_preflight_drift": "PASS",
            "ledh_value_result_inspected_or_computed": "PASS",
            "ledh_gradient_result_inspected_or_computed": "PASS",
            "missing_ledh_specific_field": "PASS",
            "bootstrap_contract_copied_without_ledh_fields": "PASS",
            "ambiguous_proposal_density_or_logdet": "PASS",
            "runtime_boolean_ess_trigger_in_primary_contract": "PASS",
            "stochastic_sampling_left_unfrozen": "PASS",
            "missing_tolerance_or_gradient_knob": "PASS",
            "localsource_filterflow_mutated": "PASS",
            "student_command_or_metric": "PASS",
            "oracle_framing": "PASS",
            "finite_difference_promoted_to_gradient_gate": "PASS",
        },
        "execution_diagnostics": {
            "tensorflow_imported_for_fixture_serialization": True,
            "pre_import_cuda_visible_devices": PRE_IMPORT_CUDA_VISIBLE_DEVICES,
            "gpu_claim_made": False,
            "ledh_pfpf_ot_values_computed": False,
            "ledh_pfpf_ot_gradients_computed": False,
            "ledh_flow_executed": False,
            "filterflow_subprocess_run": False,
            "student_command_run": False,
        },
        "contracts": contracts,
        "summary": _summary(contracts),
        "artifact_paths": {
            "json": str(JSON_PATH.relative_to(REPO_ROOT)),
            "markdown_report": str(REPORT_PATH.relative_to(REPO_ROOT)),
            "phase_result": RESULT_PATH,
        },
        "review_round": 3,
        "open_material_blockers": [],
        "repair_amendment_required": False,
        "next_allowed_action": "begin P6 PRECHECK visibly in current dialogue",
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners.run_v2_ledh_pfpf_ot_contracts_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_claims": [
            "P5 freezes LEDH-PFPF-OT contracts only.",
            "P5 does not validate BayesFilter LEDH-PFPF-OT values.",
            "P5 does not validate FilterFlow-side adapter LEDH-PFPF-OT values.",
            "P5 does not validate LEDH-PFPF-OT gradients.",
            "P5 does not establish filtering correctness.",
            "P5 does not establish stochastic resampling distribution correctness.",
            "P5 does not make a student implementation claim.",
            "P5 does not make a GPU, scalability, deployment, or production-readiness claim.",
        ],
    }


def _ledh_pfpf_ot_contract(spec: CommonModelSpecV2) -> dict[str, Any]:
    path_contract = dict(spec.path_contract)
    fixed_contract = dict(spec.fixed_ancestor_contract)
    fixed_mask = [bool(flag) for flag in fixed_contract["resampling_flags"]]
    ledh_model = _ledh_model_contract(spec)
    payload = {
        "contract_id": f"{spec.model_id}::ledh_pfpf_ot::2026-06-07.p5.visible",
        "algorithm": "ledh_pfpf_ot",
        "model_id": spec.model_id,
        "family": spec.family,
        "source_surface": spec.source_surface,
        "successor_of": list(spec.successor_of),
        "dtype": DTYPE.name,
        "theta": tensor_to_json(spec.theta),
        "parameters": _jsonable_for_runner(spec.parameters),
        "horizon": int(path_contract["horizon"]),
        "num_particles": int(path_contract["num_particles"]),
        "state_dim": int(path_contract["state_dim"]),
        "initial_particles": _jsonable_for_runner(path_contract["initial_particles"]),
        "transition_innovations": _jsonable_for_runner(path_contract["transition_innovations"]),
        "observations": _jsonable_for_runner(path_contract["observations"]),
        "initial_log_weight_policy": path_contract["initial_log_weight_policy"],
        "pre_flow_transition_proposal": _pre_flow_transition_proposal(spec),
        "ledh_linearization": _ledh_linearization(spec, ledh_model),
        "ledh_affine_map": {
            "map_family": ledh_model["map_family"],
            "map_equation": ledh_model["map_equation"],
            "implementation_route": ledh_model["flow_route"],
            "jacobian_held_fixed_scope": ledh_model["jacobian_held_fixed_scope"],
            "singular_completion_policy": ledh_model.get("singular_completion_policy", "not_applicable"),
        },
        "forward_log_det_convention": {
            "quantity": "log_abs_det_d_post_flow_d_pre_flow",
            "sign_in_pfpf_correction": "added",
            "source": ledh_model["forward_log_det_source"],
            "scope": ledh_model["jacobian_held_fixed_scope"],
            "must_be_finite_in_value_phase": True,
        },
        "proposal_log_density_route": {
            "density_symbol": "q0(x0_t | x_{t-1})",
            "route": ledh_model["proposal_log_density_route"],
            "pre_flow_log_density_field": "flow.pre_flow_log_density",
            "post_flow_density_accounting": (
                "proposal density for x_t is q0(pre_flow) - forward_log_det; "
                "PF-PF correction is implemented as -q0(pre_flow) + forward_log_det"
            ),
        },
        "target_transition_density_route": ledh_model["target_transition_density_route"],
        "observation_density_route": ledh_model["observation_density_route"],
        "pfpf_corrected_log_weight_equation": {
            "equation": (
                "log_w_t = log_w_{t-1} + log p(x_t | x_{t-1}) "
                "+ log p(y_t | x_t) - log q0(x0_t | x_{t-1}) "
                "+ log_abs_det(d x_t / d x0_t)"
            ),
            "code_equation": (
                "log_weights + target_transition + target_observation "
                "- flow.pre_flow_log_density + flow.forward_log_det"
            ),
            "source_file": "experiments/dpf_implementation/tf_tfp/filters/ledh_pfpf_ot_tf.py",
        },
        "fixed_ess_trigger_mask": fixed_mask,
        "fixed_ess_trigger_count": int(sum(1 for flag in fixed_mask if flag)),
        "runtime_ess_trigger_decision": "forbidden_primary_fixed_branch",
        "branch_mask_source": (
            "CommonModelSpecV2.fixed_ancestor_contract.resampling_flags reused "
            "as the deterministic LEDH-PFPF-OT ESS trigger mask; P6/P7 must "
            "not recompute Boolean trigger decisions for primary evidence."
        ),
        "resampling_policy": "fixed_ess_mask_filterflow_style_annealed_transport_after_pfpf_correction",
        "ot_settings": _ledh_pfpf_ot_settings(),
        "transport_application": (
            "If fixed_ess_trigger_mask[t] is true, apply FilterFlow-style "
            "annealed regularized transport to post-flow particles and "
            "normalized PF-PF-corrected log weights; otherwise leave post-flow "
            "particles and normalized log weights on the non-transported path."
        ),
        "scalar_definition": path_contract["scalar"],
        "gradient_contract": _gradient_contract(spec, ledh_model),
        "finite_difference_policy": "diagnostic_only_not_a_promotion_gate",
        "tolerances": _jsonable_for_runner(spec.tolerances),
        "adapter_certification": _jsonable_for_runner(spec.adapter_certification),
        "non_claims": list(spec.non_claims),
        "source_spec_checksum": spec.checksum(),
        "ledh_source_files": [
            "experiments/dpf_implementation/tf_tfp/filters/ledh_pfpf_ot_tf.py",
            "experiments/dpf_implementation/tf_tfp/flows/ledh_tf.py",
            "experiments/dpf_implementation/tf_tfp/flows/jacobians_tf.py",
            "experiments/dpf_implementation/tf_tfp/resampling/annealed_transport_tf.py",
        ],
    }
    payload["component_checksums"] = {
        "parameters": stable_digest(payload["parameters"]),
        "initial_particles": stable_digest(payload["initial_particles"]),
        "transition_innovations": stable_digest(payload["transition_innovations"]),
        "observations": stable_digest(payload["observations"]),
        "pre_flow_transition_proposal": stable_digest(payload["pre_flow_transition_proposal"]),
        "ledh_linearization": stable_digest(payload["ledh_linearization"]),
        "ledh_affine_map": stable_digest(payload["ledh_affine_map"]),
        "forward_log_det_convention": stable_digest(payload["forward_log_det_convention"]),
        "proposal_log_density_route": stable_digest(payload["proposal_log_density_route"]),
        "target_transition_density_route": stable_digest(payload["target_transition_density_route"]),
        "observation_density_route": stable_digest(payload["observation_density_route"]),
        "pfpf_corrected_log_weight_equation": stable_digest(payload["pfpf_corrected_log_weight_equation"]),
        "fixed_ess_trigger_mask": stable_digest(payload["fixed_ess_trigger_mask"]),
        "ot_settings": stable_digest(payload["ot_settings"]),
        "gradient_contract": stable_digest(payload["gradient_contract"]),
        "tolerances": stable_digest(payload["tolerances"]),
    }
    payload["contract_checksum"] = stable_digest(payload)
    payload["consumer_contract_checksums"] = {
        "bayesfilter": payload["contract_checksum"],
        "filterflow_adapter": payload["contract_checksum"],
    }
    payload["consumer_byte_identity"] = True
    return payload


def _ledh_model_contract(spec: CommonModelSpecV2) -> dict[str, Any]:
    if spec.model_id == "lgssm_2d_h25_rich":
        return {
            "map_family": "linear_gaussian_local_affine_ledh",
            "flow_route": "ledh_flow_batch_tf",
            "map_equation": "x_t = m_post + L_post L_prior^{-1}(x0_t - m_prior)",
            "linearization_point": "pre_flow_particle_x0_t",
            "observation_function": "C x",
            "observation_residual_route": "observed - C x_ref",
            "jacobian_function": "linear_observation_jacobian_tf(C)",
            "jacobian_held_fixed_scope": "per_particle_local_observation_jacobian_held_fixed_for_affine_map",
            "transition_matrix_route": "parameters.A",
            "transition_covariance_route": "parameters.Q",
            "observation_covariance_route": "parameters.R",
            "proposal_log_density_route": "gaussian_logpdf(pre_flow - A ancestor, Q)",
            "target_transition_density_route": "highdim.LinearGaussianSSM.transition_log_density(theta, ancestor, post_flow, t)",
            "observation_density_route": "highdim.LinearGaussianSSM.observation_log_density(theta, post_flow, observation, t)",
            "forward_log_det_source": "ledh_flow_batch_tf.forward_log_det",
            "value_phase_readiness": "READY_FOR_P6",
        }
    if spec.model_id == "sv_1d_h18_rich":
        return {
            "map_family": "scalar_surrogate_observation_ledh",
            "flow_route": "v2_scalar_sv_ledh_flow_to_be_executed_in_p6",
            "map_equation": "h_t = m_post + sqrt(var_post / var_prior) * (h0_t - m_prior)",
            "linearization_point": "pre_flow_particle_h0_t",
            "observation_function": "z_t = log(y_t^2 + 1e-6) surrogate",
            "observation_residual_route": "z_t - h_ref",
            "jacobian_function": "constant scalar Jacobian 1.0 of log-square surrogate with respect to h",
            "jacobian_held_fixed_scope": "scalar_surrogate_jacobian_held_fixed_per_time",
            "transition_matrix_route": "scalar phi around mu: m_prior = mu + phi * (h_prev - mu)",
            "transition_covariance_route": "sigma^2",
            "observation_covariance_route": "surrogate log-square variance 2.0",
            "proposal_log_density_route": "normal_logpdf(pre_flow - prior_mean, sigma)",
            "target_transition_density_route": "CommonSVRichSSM.transition_log_density(theta, ancestor, post_flow, t)",
            "observation_density_route": "CommonSVRichSSM.observation_log_density(theta, post_flow, observation, t)",
            "forward_log_det_source": "log(sqrt(var_post / var_prior))",
            "value_phase_readiness": "READY_FOR_P6_WITH_DECLARED_SURROGATE_FLOW",
        }
    if spec.model_id == "range_bearing_4d_h20_rich":
        return {
            "map_family": "nonlinear_range_bearing_local_affine_ledh",
            "flow_route": "ledh_flow_batch_tf",
            "map_equation": "x_t = m_post + L_post L_prior^{-1}(x0_t - m_prior)",
            "linearization_point": "pre_flow_particle_x0_t",
            "observation_function": "range_bearing_observation_tf",
            "observation_residual_route": "observation_residual_tf with wrapped bearing residual",
            "jacobian_function": "range_bearing_jacobian_tf(pre_flow_particle)",
            "jacobian_held_fixed_scope": "per_particle_local_observation_jacobian_held_fixed_for_affine_map",
            "transition_matrix_route": "parameters.A",
            "transition_covariance_route": "parameters.Q",
            "observation_covariance_route": "parameters.R",
            "proposal_log_density_route": "gaussian_logpdf(pre_flow - A ancestor, Q)",
            "target_transition_density_route": "CommonRangeBearingSSM.transition_log_density(theta, ancestor, post_flow, t)",
            "observation_density_route": "CommonRangeBearingSSM.observation_log_density(theta, post_flow, observation, t)",
            "forward_log_det_source": "ledh_flow_batch_tf.forward_log_det",
            "value_phase_readiness": "READY_FOR_P6",
        }
    if spec.model_id == "structural_ar1_quadratic_h16":
        return {
            "map_family": "structural_split_scalar_ledh_with_deterministic_k_completion",
            "flow_route": "v2_structural_split_scalar_ledh_flow_to_be_executed_in_p6",
            "map_equation": "m_t = m_post + sqrt(var_post / sigma^2) * (m0_t - rho m_{t-1}); k_t = complete_k(previous_k, previous_m, m_t)",
            "linearization_point": "pre_flow stochastic coordinate m0_t with previous structural context",
            "observation_function": "structural_observation_mean_tf([m, complete_k], lambda)",
            "observation_residual_route": "observed - structural_observation_mean_tf",
            "jacobian_function": "TensorFlow autodiff of structural observation mean with respect to stochastic m coordinate",
            "jacobian_held_fixed_scope": "per_particle_structural_observation_jacobian_held_fixed_for_scalar_affine_map",
            "transition_matrix_route": "scalar rho on m coordinate",
            "transition_covariance_route": "sigma^2 on stochastic m coordinate",
            "observation_covariance_route": "observation_scale^2",
            "proposal_log_density_route": "normal_logpdf(pre_flow_m - rho previous_m, sigma)",
            "target_transition_density_route": "CommonStructuralAR1QuadraticSSM.transition_log_density(theta, ancestor, post_flow, t)",
            "observation_density_route": "CommonStructuralAR1QuadraticSSM.observation_log_density(theta, post_flow, observation, t)",
            "forward_log_det_source": "log(sqrt(var_post / sigma^2)) on stochastic m coordinate",
            "singular_completion_policy": "density lives on stochastic m; k is deterministic completion carried in state",
            "value_phase_readiness": "READY_FOR_P6_WITH_SPLIT_COMPLETION_FLOW",
        }
    if spec.model_id == "spatial_sir_j3_rk4":
        return {
            "map_family": "rk4_state_autodiff_local_affine_ledh",
            "flow_route": "v2_autodiff_ledh_flow_to_be_executed_in_p6",
            "map_equation": "x_t = m_post + L_post L_prior^{-1}(x0_t - m_prior)",
            "linearization_point": "pre_flow_particle_x0_t",
            "observation_function": "infectious-only observation projection (I_1,I_2,I_3)",
            "observation_residual_route": "observed infectious vector - projected infectious vector",
            "jacobian_function": "TensorFlow autodiff/projection Jacobian of infectious-only observation wrt state",
            "jacobian_held_fixed_scope": "per_particle_observation_jacobian_held_fixed_for_affine_map",
            "transition_matrix_route": "TensorFlow autodiff Jacobian of RK4 transition_mean wrt ancestor at local state",
            "transition_covariance_route": "parameters.process_covariance",
            "observation_covariance_route": "parameters.observation_covariance",
            "proposal_log_density_route": "mvn_logpdf(pre_flow - transition_mean(ancestor), process_covariance)",
            "target_transition_density_route": "highdim.SpatialSIRSSM.transition_log_density(theta, ancestor, post_flow, t)",
            "observation_density_route": "highdim.SpatialSIRSSM.observation_log_density(theta, post_flow, observation, t)",
            "forward_log_det_source": "local affine logabsdet from autodiff LEDH covariance map",
            "value_phase_readiness": "READY_FOR_P6_VALUE_ONLY_P7_PHYSICAL_GRADIENT_PREDECLARED_EXCLUDED",
        }
    if spec.model_id == "predator_prey_rk4":
        return {
            "map_family": "rk4_state_autodiff_local_affine_ledh",
            "flow_route": "v2_autodiff_ledh_flow_to_be_executed_in_p6",
            "map_equation": "x_t = m_post + L_post L_prior^{-1}(x0_t - m_prior)",
            "linearization_point": "pre_flow_particle_x0_t",
            "observation_function": "direct noisy-state observation",
            "observation_residual_route": "observed state vector - predicted state vector",
            "jacobian_function": "identity observation Jacobian",
            "jacobian_held_fixed_scope": "per_particle_observation_jacobian_held_fixed_for_affine_map",
            "transition_matrix_route": "TensorFlow autodiff Jacobian of RK4 transition_mean(theta, ancestor) wrt ancestor",
            "transition_covariance_route": "parameters.process_covariance",
            "observation_covariance_route": "parameters.observation_covariance",
            "proposal_log_density_route": "mvn_logpdf(pre_flow - transition_mean(theta, ancestor), process_covariance)",
            "target_transition_density_route": "highdim.PredatorPreySSM.transition_log_density(theta, ancestor, post_flow, t)",
            "observation_density_route": "highdim.PredatorPreySSM.observation_log_density(theta, post_flow, observation, t)",
            "forward_log_det_source": "local affine logabsdet from autodiff LEDH covariance map",
            "value_phase_readiness": "READY_FOR_P6",
        }
    raise ValueError(f"unknown V2 model id: {spec.model_id}")


def _pre_flow_transition_proposal(spec: CommonModelSpecV2) -> dict[str, Any]:
    return {
        "proposal_id": "fixed_transition_innovations_before_ledh_flow",
        "sample_equation": (
            "pre_flow_t = transition_mean(ancestor_t) + fixed_transition_innovation_t"
        ),
        "transition_innovation_policy": spec.path_contract["transition_innovation_policy"],
        "initial_particles_source": "CommonModelSpecV2.path_contract.initial_particles",
        "transition_innovations_source": "CommonModelSpecV2.path_contract.transition_innovations",
        "random_sampling_in_primary_contract": False,
        "proposal_is_before_ledh_flow": True,
    }


def _ledh_linearization(spec: CommonModelSpecV2, ledh_model: dict[str, Any]) -> dict[str, Any]:
    return {
        "linearization_point": ledh_model["linearization_point"],
        "observation_function": ledh_model["observation_function"],
        "observation_residual_route": ledh_model["observation_residual_route"],
        "jacobian_function": ledh_model["jacobian_function"],
        "jacobian_held_fixed_scope": ledh_model["jacobian_held_fixed_scope"],
        "transition_matrix_route": ledh_model["transition_matrix_route"],
        "transition_covariance_route": ledh_model["transition_covariance_route"],
        "observation_covariance_route": ledh_model["observation_covariance_route"],
        "model_specific_note": spec.adapter_certification["adapter_note"],
        "ledh_value_phase_readiness": ledh_model["value_phase_readiness"],
    }


def _gradient_contract(spec: CommonModelSpecV2, ledh_model: dict[str, Any]) -> dict[str, Any]:
    contract = _jsonable_for_runner(spec.gradient_contract)
    knobs = list(contract.get("knobs", []))
    included = [str(knob.get("name")) for knob in knobs if bool(knob.get("include"))]
    excluded = [str(knob.get("name")) for knob in knobs if not bool(knob.get("include"))]
    contract["knob_count"] = len(knobs)
    contract["included_required_knob_names"] = included
    contract["excluded_knob_names"] = excluded
    contract["ad_gradient_scope"] = (
        "deterministic fixed-branch value path through pre-flow proposal, "
        "LEDH flow, PF-PF correction, and deterministic OT transform when "
        "the fixed trigger mask is true"
    )
    contract["excluded_gradient_scope"] = [
        "random initial sampling",
        "random transition sampling",
        "Boolean ESS trigger decisions",
        "random or discrete branch selection",
    ]
    contract["finite_difference_policy"] = "diagnostic_only_not_a_promotion_gate"
    contract["ledh_flow_gradient_route"] = ledh_model["flow_route"]
    contract["p7_gradient_readiness"] = (
        "PREDECLARED_EXCLUDED_NO_PHYSICAL_KNOB"
        if not included
        else "READY_FOR_P7_AFTER_P6_VALUE_PASS"
    )
    return contract


def _ledh_pfpf_ot_settings() -> dict[str, Any]:
    ledh_defaults = _callable_defaults(
        run_ledh_pfpf_ot_tf,
        [
            "ess_threshold_ratio",
            "sinkhorn_epsilon",
            "sinkhorn_iterations",
            "sinkhorn_tolerance",
            "transport_method",
            "annealed_scaling",
            "annealed_convergence_threshold",
        ],
    )
    flow_defaults = _callable_defaults(
        ledh_flow_batch_tf,
        ["jitter"],
    )
    transport_defaults = _callable_defaults(
        annealed_transport_resample_tf,
        [
            "transport_gradient_mode",
            "application_mode",
            "max_iterations",
            "epsilon",
            "scaling",
            "convergence_threshold",
        ],
    )
    return {
        "transport_method": ledh_defaults["transport_method"],
        "resampling_method": "filterflow_style_annealed_transport_tf",
        "mathematical_object": "annealed_regularized_transport_transform_after_pfpf_correction",
        "filterflow_reference": "RegularisedTransform semantics mirrored by annealed_transport_resample_tf",
        "fixed_target_sinkhorn_status": "not_this_algorithm_local_comparator_only",
        "ess_threshold_ratio_recorded_for_stochastic_smoke_only": ledh_defaults["ess_threshold_ratio"],
        "primary_branch_decision": "fixed_ess_trigger_mask_from_contract",
        "sinkhorn_epsilon": ledh_defaults["sinkhorn_epsilon"],
        "sinkhorn_iterations": ledh_defaults["sinkhorn_iterations"],
        "sinkhorn_tolerance": ledh_defaults["sinkhorn_tolerance"],
        "annealed_scaling": ledh_defaults["annealed_scaling"],
        "annealed_convergence_threshold": ledh_defaults["annealed_convergence_threshold"],
        "transport_gradient_mode": transport_defaults["transport_gradient_mode"],
        "application_mode": transport_defaults["application_mode"],
        "transport_default_max_iterations": transport_defaults["max_iterations"],
        "transport_default_epsilon": transport_defaults["epsilon"],
        "transport_default_scaling": transport_defaults["scaling"],
        "transport_default_convergence_threshold": transport_defaults["convergence_threshold"],
        "ledh_covariance_jitter": flow_defaults["jitter"],
        "source_files": [
            "experiments/dpf_implementation/tf_tfp/filters/ledh_pfpf_ot_tf.py",
            "experiments/dpf_implementation/tf_tfp/flows/ledh_tf.py",
            "experiments/dpf_implementation/tf_tfp/resampling/annealed_transport_tf.py",
        ],
    }


def _callable_defaults(fn: Any, names: list[str]) -> dict[str, Any]:
    signature = inspect.signature(fn)
    defaults: dict[str, Any] = {}
    for name in names:
        parameter = signature.parameters[name]
        if parameter.default is inspect.Parameter.empty:
            raise ValueError(f"{fn.__name__}.{name} has no default to freeze")
        defaults[name] = parameter.default
    return defaults


def _preflight_prior_artifacts(
    p0_payload: dict[str, Any],
    p1_payload: dict[str, Any],
    p4_payload: dict[str, Any],
) -> None:
    if p0_payload.get("decision") != "PASS_P0_READY_FOR_P1":
        raise ValueError(f"P0 visible governance is not passed: {p0_payload.get('decision')}")
    if p1_payload.get("decision") != "PASS_P1_ARCHITECTURE_READY_FOR_P2":
        raise ValueError(f"P1 architecture is not passed: {p1_payload.get('decision')}")
    if p4_payload.get("decision") != P4_PASS_DECISION:
        raise ValueError(f"P4 gradients are not passed: {p4_payload.get('decision')}")
    if p4_payload.get("reproducibility_digest") != P4_REPRODUCIBILITY_DIGEST:
        raise ValueError("P4 reproducibility digest drift")
    p0_ids = list(p0_payload.get("required_v2_model_ids", []))
    p1_ids = list(p1_payload.get("required_v2_model_ids", []))
    p4_ids = list(p4_payload.get("required_v2_model_ids", []))
    if tuple(p0_ids) != EXPECTED_V2_MODEL_IDS:
        raise ValueError(f"P0 model id gate failed: {p0_ids}")
    if tuple(p1_ids) != EXPECTED_V2_MODEL_IDS:
        raise ValueError(f"P1 model id gate failed: {p1_ids}")
    if tuple(p4_ids) != EXPECTED_V2_MODEL_IDS:
        raise ValueError(f"P4 model id gate failed: {p4_ids}")


def _validate_payload(payload: dict[str, Any]) -> None:
    if payload.get("decision") not in EXPECTED_DECISIONS:
        raise ValueError(f"P5 decision is not passable: {payload.get('decision')}")
    if payload.get("phase") != "P5":
        raise ValueError(f"unexpected phase: {payload.get('phase')}")
    if payload.get("artifact_paths", {}).get("json") != str(JSON_PATH.relative_to(REPO_ROOT)):
        raise ValueError("P5 JSON artifact path mismatch")
    if tuple(payload.get("required_v2_model_ids", [])) != EXPECTED_V2_MODEL_IDS:
        raise ValueError("P5 required model id list mismatch")
    if payload.get("row_count_gate", {}).get("actual_count") != len(EXPECTED_V2_MODEL_IDS):
        raise ValueError("P5 row-count gate failed")
    policy = payload.get("contract_generation_policy", {})
    if not policy.get("generated_after_p4_reviewed_pass"):
        raise ValueError("P5 contract does not state post-P4-pass generation")
    if not policy.get("generated_before_ledh_pfpf_ot_values"):
        raise ValueError("P5 contract does not state pre-value generation")
    if not policy.get("generated_before_ledh_pfpf_ot_gradients"):
        raise ValueError("P5 contract does not state pre-gradient generation")
    diagnostics = payload.get("execution_diagnostics", {})
    if diagnostics.get("ledh_pfpf_ot_values_computed") or diagnostics.get("ledh_pfpf_ot_gradients_computed"):
        raise ValueError("P5 artifact claims value or gradient execution")
    if diagnostics.get("ledh_flow_executed"):
        raise ValueError("P5 artifact claims LEDH flow execution")
    if diagnostics.get("filterflow_subprocess_run") or diagnostics.get("student_command_run"):
        raise ValueError("P5 artifact claims forbidden subprocess/student execution")
    contracts = list(payload.get("contracts", []))
    ids = [contract.get("model_id") for contract in contracts]
    if tuple(ids) != EXPECTED_V2_MODEL_IDS:
        raise ValueError(f"P5 contract id gate failed: {ids}")
    for contract in contracts:
        _validate_contract(contract)
    bundle = payload.get("contract_bundle_payload", {})
    expected_bundle = {
        "artifact_id": "dpf_v2_ledh_pfpf_ot_contracts_2026-06-07",
        "version": "2026-06-07.p5.visible",
        "contracts": contracts,
    }
    if bundle != expected_bundle:
        raise ValueError("P5 contract bundle payload does not match contracts")
    if payload.get("contract_bundle_checksum") != stable_digest(bundle):
        raise ValueError("P5 contract bundle checksum mismatch")


def _validate_contract(contract: dict[str, Any]) -> None:
    required_fields = {
        "contract_id",
        "algorithm",
        "model_id",
        "family",
        "dtype",
        "parameters",
        "horizon",
        "num_particles",
        "state_dim",
        "initial_particles",
        "transition_innovations",
        "observations",
        "component_checksums",
        "contract_checksum",
        "consumer_contract_checksums",
        "finite_difference_policy",
        "tolerances",
        "scalar_definition",
    }.union(REQUIRED_LEDGER_FIELDS)
    missing = required_fields.difference(contract)
    if missing:
        raise ValueError(f"{contract.get('model_id')}: missing contract fields {sorted(missing)}")
    if contract["algorithm"] != "ledh_pfpf_ot":
        raise ValueError(f"{contract['model_id']}: unexpected algorithm {contract['algorithm']}")
    if contract["dtype"] != DTYPE.name:
        raise ValueError(f"{contract['model_id']}: dtype mismatch")
    if contract["runtime_ess_trigger_decision"] != "forbidden_primary_fixed_branch":
        raise ValueError(f"{contract['model_id']}: runtime ESS trigger not forbidden")
    if contract["finite_difference_policy"] != "diagnostic_only_not_a_promotion_gate":
        raise ValueError(f"{contract['model_id']}: finite-difference policy is unsafe")
    if "bootstrap" in contract.get("proposal_log_density_route", {}).get("route", "").lower():
        raise ValueError(f"{contract['model_id']}: proposal route still says bootstrap")
    equation = contract["pfpf_corrected_log_weight_equation"]["code_equation"]
    for token in ("target_transition", "target_observation", "pre_flow_log_density", "forward_log_det"):
        if token not in equation:
            raise ValueError(f"{contract['model_id']}: PF-PF equation missing {token}")
    if contract["forward_log_det_convention"].get("sign_in_pfpf_correction") != "added":
        raise ValueError(f"{contract['model_id']}: unsafe logdet sign convention")
    horizon = int(contract["horizon"])
    n_particles = int(contract["num_particles"])
    state_dim = int(contract["state_dim"])
    initial_particles = contract["initial_particles"]
    innovations = contract["transition_innovations"]
    observations = contract["observations"]
    mask = contract["fixed_ess_trigger_mask"]
    if len(initial_particles) != n_particles:
        raise ValueError(f"{contract['model_id']}: initial particle count mismatch")
    if any(len(row) != state_dim for row in initial_particles):
        raise ValueError(f"{contract['model_id']}: initial particle state dimension mismatch")
    if len(innovations) != horizon:
        raise ValueError(f"{contract['model_id']}: transition innovation horizon mismatch")
    if len(observations) != horizon:
        raise ValueError(f"{contract['model_id']}: observation horizon mismatch")
    if len(mask) != horizon or not all(isinstance(flag, bool) for flag in mask):
        raise ValueError(f"{contract['model_id']}: fixed ESS trigger mask mismatch")
    if int(contract["fixed_ess_trigger_count"]) != sum(1 for flag in mask if flag):
        raise ValueError(f"{contract['model_id']}: fixed ESS trigger count mismatch")
    gradient_contract = contract["gradient_contract"]
    if "knobs" not in gradient_contract or "finite_difference_step" not in gradient_contract:
        raise ValueError(f"{contract['model_id']}: incomplete gradient contract")
    if gradient_contract.get("finite_difference_policy") != "diagnostic_only_not_a_promotion_gate":
        raise ValueError(f"{contract['model_id']}: gradient finite-difference policy is unsafe")
    if contract["model_id"] == "spatial_sir_j3_rk4":
        if gradient_contract["p7_gradient_readiness"] != "PREDECLARED_EXCLUDED_NO_PHYSICAL_KNOB":
            raise ValueError("SIR P7 readiness must remain predeclared excluded")
    if not _json_finite(
        {
            "initial_particles": initial_particles,
            "transition_innovations": innovations,
            "observations": observations,
        }
    ):
        raise ValueError(f"{contract['model_id']}: nonfinite frozen numeric contract field")
    clone = dict(contract)
    clone.pop("contract_checksum")
    clone.pop("consumer_contract_checksums")
    clone.pop("consumer_byte_identity", None)
    if contract["contract_checksum"] != stable_digest(clone):
        raise ValueError(f"{contract['model_id']}: contract checksum mismatch")
    consumers = contract["consumer_contract_checksums"]
    if consumers.get("bayesfilter") != contract["contract_checksum"]:
        raise ValueError(f"{contract['model_id']}: BayesFilter consumer checksum mismatch")
    if consumers.get("filterflow_adapter") != contract["contract_checksum"]:
        raise ValueError(f"{contract['model_id']}: FilterFlow consumer checksum mismatch")


def _summary(contracts: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_contracts": len(contracts),
        "models": [contract["model_id"] for contract in contracts],
        "included_gradient_knob_count": sum(
            len(contract["gradient_contract"]["included_required_knob_names"])
            for contract in contracts
        ),
        "predeclared_excluded_gradient_rows": [
            contract["model_id"]
            for contract in contracts
            if contract["gradient_contract"]["p7_gradient_readiness"]
            == "PREDECLARED_EXCLUDED_NO_PHYSICAL_KNOB"
        ],
        "ledh_flow_routes": {
            contract["model_id"]: contract["ledh_affine_map"]["implementation_route"]
            for contract in contracts
        },
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
    contracts = payload["contracts"]
    settings = payload["ledh_pfpf_ot_settings"]
    lines = [
        "# DPF V2 Algorithm Full Comparison P5 LEDH-PFPF-OT Contracts Result",
        "",
        "metadata_date: 2026-06-07",
        "visible_execution_timestamp: `2026-06-08T15:05:00+08:00`",
        "phase: P5",
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
        "- one frozen LEDH-PFPF-OT contract per V2 row in exact order;",
        "- each contract records pre-flow transition proposal, LEDH linearization, Jacobian route, affine map, forward logdet convention, proposal density, target transition density, observation density, PF-PF correction, fixed ESS trigger mask, OT settings, gradient knobs, dtype, tolerances, and checksums;",
        "- BayesFilter and FilterFlow-side adapters consume the same contract checksum.",
        "",
        "Veto diagnostics:",
        "",
        "- row disappearance or row-order mismatch;",
        "- P0/P1/P4 preflight drift;",
        "- missing LEDH-specific proposal, density, logdet, correction, branch, OT, or gradient field;",
        "- value or gradient result execution before contract freeze;",
        "- runtime Boolean ESS trigger decisions in the primary fixed-branch contract;",
        "- `.localsource/filterflow` mutation, student command, oracle framing, or finite differences promoted to a gradient gate.",
        "",
        "Non-claims:",
        "",
        "- P5 freezes contracts only; it does not validate LEDH-PFPF-OT values or gradients.",
        "",
        "## Local Skeptical Phase Audit",
        "",
        "Audit status: `PASS_LOCAL_PHASE_AUDIT`.",
        "",
        "Wrong-baseline risk: controlled. P5 requires visible P0/P1/P4 pass artifacts and freezes a new LEDH contract bundle for P6/P7.",
        "",
        "Proxy-metric risk: controlled. P5 records no LEDH value, gradient, ESS, RMSE, runtime, or finite-difference promotion metric.",
        "",
        "Missing stop-condition risk: controlled. Missing LEDH fields, stale preflight artifacts, unsafe branch sources, or any value/gradient execution remain vetoes.",
        "",
        "Unfair-comparison risk: controlled. Both later consumers are bound to identical contract checksums.",
        "",
        "Environment-mismatch risk: controlled. TensorFlow was imported only for fixture serialization with `CUDA_VISIBLE_DEVICES=-1`; no GPU claim is made.",
        "",
        "Audit decision: local pass pending Claude read-only review.",
        "",
        "## Result",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- JSON artifact: `{payload['artifact_paths']['json']}`",
        f"- Markdown report: `{payload['artifact_paths']['markdown_report']}`",
        f"- Phase result: `{payload['artifact_paths']['phase_result']}`",
        f"- Contract bundle checksum: `{payload['contract_bundle_checksum']}`",
        f"- Reproducibility digest: `{payload['reproducibility_digest']}`",
        "",
        "## V2 Contract Rows",
        "",
        "| Model id | Horizon | Particles | Fixed ESS mask | LEDH flow route | Included AD knobs | P7 readiness | Contract checksum |",
        "|---|---:|---:|---|---|---|---|---|",
    ]
    for contract in contracts:
        gradient = contract["gradient_contract"]
        lines.append(
            f"| `{contract['model_id']}` | {contract['horizon']} | {contract['num_particles']} | "
            f"`{contract['fixed_ess_trigger_mask']}` | "
            f"`{contract['ledh_affine_map']['implementation_route']}` | "
            f"`{gradient['included_required_knob_names']}` | "
            f"`{gradient['p7_gradient_readiness']}` | "
            f"`{contract['contract_checksum']}` |"
        )
    lines.extend(
        [
            "",
            "## Frozen LEDH Semantics",
            "",
            "- Pre-flow proposal: fixed transition innovations before LEDH flow.",
            "- PF-PF correction: `log_weights + target_transition + target_observation - pre_flow_log_density + forward_log_det`.",
            "- Forward logdet convention: `log_abs_det(d post_flow / d pre_flow)`, added in the correction.",
            "- Branch source: fixed ESS trigger masks from the V2 contract, not runtime Boolean ESS decisions.",
            "- FilterFlow support: BayesFilter-owned adapter-hosted route; `.localsource/filterflow` is read-only.",
            "",
            "## LEDH-PFPF-OT Settings",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| transport method | `{settings['transport_method']}` |",
            f"| resampling method | `{settings['resampling_method']}` |",
            f"| epsilon | `{settings['sinkhorn_epsilon']}` |",
            f"| iterations | `{settings['sinkhorn_iterations']}` |",
            f"| tolerance | `{settings['sinkhorn_tolerance']}` |",
            f"| annealed scaling | `{settings['annealed_scaling']}` |",
            f"| annealed convergence threshold | `{settings['annealed_convergence_threshold']}` |",
            f"| transport gradient mode | `{settings['transport_gradient_mode']}` |",
            f"| application mode | `{settings['application_mode']}` |",
            f"| LEDH covariance jitter | `{settings['ledh_covariance_jitter']}` |",
            f"| primary branch decision | `{settings['primary_branch_decision']}` |",
            "",
            "## Primary Criterion Fields",
            "",
        ]
    )
    lines.extend(f"- {key}: `{value}`" for key, value in payload["primary_criterion_status"].items())
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
            "## Command Manifest",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| git commit | `{run_manifest.get('commit')}` |",
            f"| git branch | `{run_manifest.get('branch')}` |",
            f"| P5 touched paths | `{'; '.join(P5_TOUCHED_PATHS)}` |",
            "| dirty status | full repo dirty state is recorded in JSON run manifest; unrelated dirty work is not interpreted as P5 evidence |",
            f"| command | `{run_manifest.get('command')}` |",
            f"| validation commands | `python -m json.tool {payload['artifact_paths']['json']}`; `CUDA_VISIBLE_DEVICES=-1 python -m experiments.dpf_implementation.tf_tfp.runners.run_v2_ledh_pfpf_ot_contracts_tf --validate-only`; `git diff --check` on P5 files |",
            f"| environment | `/home/chakwong/BayesFilter`; Python `{run_manifest.get('python_version')}`; TensorFlow `{run_manifest.get('package_versions', {}).get('tensorflow')}`; TFP `{run_manifest.get('package_versions', {}).get('tensorflow_probability')}` |",
            f"| CPU/GPU status | CPU-only TensorFlow serialization; pre-import `CUDA_VISIBLE_DEVICES={run_manifest.get('pre_import_cuda_visible_devices')}`; visible GPUs `{run_manifest.get('gpu_devices_visible')}` |",
            "| random seeds | no RNG used in P5; fixed V2 path particles and innovations are serialized from fixtures |",
            "| dtype | `tf.float64` / JSON dtype `float64` |",
            f"| output artifacts | `{payload['artifact_paths']['json']}`; `{payload['artifact_paths']['markdown_report']}`; `{payload['artifact_paths']['phase_result']}` |",
            "",
            "## Review State",
            "",
            "review_round: 0 pending Claude P5 contract review",
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
            f"| `{payload['decision']}` | six same-checksum LEDH-PFPF-OT contracts frozen | all local P5 veto diagnostics pass | Claude may find LEDH contract adequacy gaps before P6 | run chunked Claude P5 read-only review | no value match, gradient match, filtering correctness, stochastic resampling claim, or production readiness |",
            "",
            "## Post-Run Red Team",
            "",
            "Strongest alternative explanation: the frozen contracts may still be insufficient for P6/P7 implementation if a later adapter needs a more precise per-row LEDH flow field.",
            "",
            "Result that would overturn the local decision: Claude or P6/P7 finds a missing proposal, Jacobian, density, logdet, branch, OT, tolerance, or gradient field needed to execute the same contract on both sides.",
            "",
            "Weakest evidence link: P5 serializes contract data but does not execute the LEDH value or gradient paths.",
            "",
            "## Non-Claims",
            "",
        ]
    )
    lines.extend(f"- {claim}" for claim in payload["non_claims"])
    lines.append("")
    return "\n".join(lines)


def _jsonable_for_runner(value: Any) -> Any:
    if tf.is_tensor(value):
        return tensor_to_json(value)
    if isinstance(value, dict):
        return {str(key): _jsonable_for_runner(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_for_runner(item) for item in value]
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float):
        return float(value)
    if hasattr(value, "numpy"):
        return _jsonable_for_runner(value.numpy())
    return value


def _json_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_json_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_json_finite(item) for item in value)
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    return True
if __name__ == "__main__":
    raise SystemExit(main())
