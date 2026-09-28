"""Freeze V2 Algorithm 1 UKF LEDH-PFPF contracts.

P2 is a contract phase only.  It serializes model-row applicability,
Algorithm 1 route identifiers, callback obligations, scalar definitions, and
diagnostic-only thresholds before any P3 value or P4 gradient execution.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/bayesfilter-dpf-mpl")
PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]

import argparse
import math
import sys
import time
from typing import Any

import tensorflow as tf

from experiments.dpf_implementation.tf_tfp.filters.ledh_pfpf_alg1_ukf_tf import (
    COVARIANCE_ROUTE,
    FLOW_ANCHOR_ROUTE,
    FLOW_SOURCE_ROUTE,
    METHOD_GENERATION,
    PREVIOUS_LEDHPFPF_OT_EVIDENCE_STATUS,
    algorithm1_route_identifiers,
    validate_algorithm1_route_identifiers,
)
from experiments.dpf_implementation.tf_tfp.fixtures.common_model_suite_tf import (
    DTYPE,
    EXPECTED_V2_MODEL_IDS,
    CommonModelSpecV2,
    common_model_specs_v2,
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
from experiments.dpf_implementation.tf_tfp.runners.run_ledh_pfpf_alg1_ukf_p5_lgssm_comparison_tf import (
    COVARIANCE_FLOOR,
    PSEUDO_TIME_STEPS,
    RANK_TOLERANCE,
    _ukf_parameters,
)


MODULE_PATH = (
    "experiments.dpf_implementation.tf_tfp.runners."
    "run_v2_ledh_pfpf_alg1_ukf_contracts_tf"
)
PLAN_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-ledh-pfpf-alg1-ukf-ledh-pfpf-ot-rerun-p2-v2-contracts-subplan-2026-06-10.md"
)
RESULT_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-ledh-pfpf-alg1-ukf-ledh-pfpf-ot-rerun-p2-v2-contracts-result-2026-06-10.md"
)
REVIEW_LEDGER_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-ledh-pfpf-alg1-ukf-ledh-pfpf-ot-rerun-claude-review-ledger-2026-06-10.md"
)
REGISTRY_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-ledh-pfpf-alg1-ukf-ledh-pfpf-ot-rerun-registry-2026-06-10.json"
)
P1_JSON_PATH = OUTPUT_DIR / "dpf_ledh_pfpf_alg1_ukf_direct_replacements_2026-06-10.json"
JSON_PATH = OUTPUT_DIR / "dpf_v2_ledh_pfpf_alg1_ukf_contracts_2026-06-10.json"
REPORT_PATH = REPORT_DIR / "dpf-v2-ledh-pfpf-alg1-ukf-contracts-2026-06-10.md"

VALUE_SEEDS = [101, 202, 303, 404, 505]
VALUE_PARTICLE_COUNTS = [8, 16, 32]
GRADIENT_SEEDS = [101, 202, 303]
GRADIENT_PARTICLE_COUNTS = [4, 8, 16]
ALLOWED_CONTRACT_STATUSES = {
    "RUNNABLE_ALG1",
    "N_A_NOT_APPLICABLE",
    "BLOCKED_REQUIRES_ADAPTER",
}
RUNNABLE_MODEL_IDS = {
    "lgssm_2d_h25_rich",
    "range_bearing_4d_h20_rich",
    "spatial_sir_j3_rk4",
    "predator_prey_rk4",
}


class P2ValidationError(ValueError):
    """Raised when the P2 contract artifact violates the freeze contract."""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only:
        _validate_payload(load_json(JSON_PATH))
        print("P2_V2_LEDHPFPF_ALG1_UKF_CONTRACTS_VALIDATED")
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
    registry = load_json(REPO_ROOT / REGISTRY_PATH)
    p1_payload = load_json(P1_JSON_PATH)
    _preflight(registry, p1_payload)
    specs = common_model_specs_v2()
    contracts = [_contract_for_spec(spec) for spec in specs]
    bundle = {
        "artifact_id": "dpf_v2_ledh_pfpf_alg1_ukf_contracts_2026-06-10",
        "version": "2026-06-10.p2.visible",
        "contracts": contracts,
    }
    veto = _veto_diagnostics(contracts)
    decision = (
        "PASS_P2_V2_ALG1_UKF_CONTRACTS_PENDING_CLAUDE_REVIEW"
        if not any(bool(value) for value in veto.values())
        else "P2_V2_ALG1_UKF_CONTRACTS_VETO_PENDING_REVIEW"
    )
    manifest = environment_manifest(
        command="CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m " + MODULE_PATH,
        pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
    )
    manifest.update(
        {
            "plan_path": PLAN_PATH,
            "result_path": RESULT_PATH,
            "review_ledger_path": REVIEW_LEDGER_PATH,
            "registry_path": REGISTRY_PATH,
            "p1_json_path": str(P1_JSON_PATH.relative_to(REPO_ROOT)),
            "json_path": str(JSON_PATH.relative_to(REPO_ROOT)),
            "report_path": str(REPORT_PATH.relative_to(REPO_ROOT)),
            "required_v2_model_ids": list(EXPECTED_V2_MODEL_IDS),
            "value_seed_list": list(VALUE_SEEDS),
            "value_particle_counts": list(VALUE_PARTICLE_COUNTS),
            "gradient_seed_list": list(GRADIENT_SEEDS),
            "gradient_particle_counts": list(GRADIENT_PARTICLE_COUNTS),
            "pseudo_time_steps": list(PSEUDO_TIME_STEPS),
            "ukf_parameters": _ukf_parameters(),
            "covariance_floor": COVARIANCE_FLOOR,
            "rank_tolerance": RANK_TOLERANCE,
        }
    )
    return {
        "metadata_date": "2026-06-10",
        "created_at_utc": utc_now(),
        "phase": "P2",
        "execution_route": "VISIBLE_IN_DIALOGUE",
        "decision": decision,
        "skeptical_plan_audit": {
            "status": "PASS_FOR_CONTRACT_FREEZE_ONLY",
            "wrong_baseline_control": (
                "Old V2 LEDH-PFPF-OT contracts define coverage only.  P2 does "
                "not use old values, gradients, or OT route identifiers as "
                "Algorithm 1 evidence."
            ),
            "proxy_metric_control": (
                "P2 records no value, gradient, ESS, or runtime performance "
                "metrics.  Threshold fields are frozen as diagnostic-only or "
                "adapter-blocked before execution."
            ),
            "stop_conditions": (
                "P3/P4 must consume this frozen contract.  Rows with missing "
                "callback contracts are blocked rather than run."
            ),
            "environment_control": (
                "TensorFlow is forced CPU-only before import and recorded in manifest."
            ),
        },
        "evidence_contract": {
            "question": (
                "Can the old V2 LEDH-PFPF-OT contract lane be replaced with "
                "Algorithm 1 UKF contracts for every V2 model row?"
            ),
            "baseline_comparator": (
                "Old V2 contracts define coverage.  Current contract rows bind "
                "Algorithm 1 route fields, callbacks, scalar definitions, seeds, "
                "particle ladders, pseudo-time schedule, UKF parameters, and "
                "diagnostic-only threshold policy."
            ),
            "primary_criterion": (
                "Exactly six V2 rows are frozen in order with status RUNNABLE_ALG1, "
                "N_A_NOT_APPLICABLE, or BLOCKED_REQUIRES_ADAPTER."
            ),
            "veto_diagnostics": list(veto.keys()),
            "not_concluded": _nonclaims(),
        },
        "threshold_policy": {
            "p2_numerical_execution": False,
            "value_tolerance_policy": (
                "N/A_DIAGNOSTIC_ONLY_IN_P3 unless a later exact-oracle phase "
                "imports a calibrated threshold before execution."
            ),
            "gradient_tolerance_policy": (
                "N/A_DIAGNOSTIC_ONLY_IN_P4 unless the row has a same-scalar "
                "gradient contract and a later phase freezes a calibrated band "
                "before execution."
            ),
            "finite_only_promotion_allowed": False,
            "reason": (
                "Noise scale, horizon, nonlinearity, and particle count alter "
                "Monte Carlo error.  P2 therefore freezes diagnostic execution "
                "contracts instead of inventing global error thresholds."
            ),
        },
        "required_v2_model_ids": list(EXPECTED_V2_MODEL_IDS),
        "row_count_gate": {
            "required_count": len(EXPECTED_V2_MODEL_IDS),
            "actual_count": len(contracts),
            "status": "PASS",
        },
        "contract_bundle_payload": bundle,
        "contract_bundle_checksum": stable_digest(bundle),
        "contracts": contracts,
        "summary": _summary(contracts),
        "veto_diagnostics": veto,
        "execution_diagnostics": {
            "tensorflow_imported_for_contract_serialization": True,
            "pre_import_cuda_visible_devices": PRE_IMPORT_CUDA_VISIBLE_DEVICES,
            "gpu_claim_made": False,
            "algorithm1_values_computed": False,
            "algorithm1_gradients_computed": False,
            "algorithm1_flow_executed": False,
            "old_ledh_pfpf_ot_imported": _old_runtime_module_loaded(),
            "filterflow_subprocess_run": False,
            "student_command_run": False,
        },
        "artifact_paths": {
            "json": str(JSON_PATH.relative_to(REPO_ROOT)),
            "markdown_report": str(REPORT_PATH.relative_to(REPO_ROOT)),
            "phase_result": RESULT_PATH,
        },
        "run_manifest": manifest,
        "nonclaims": _nonclaims(),
    }


def _contract_for_spec(spec: CommonModelSpecV2) -> dict[str, Any]:
    status = _contract_status(spec)
    route_fields = _augmented_algorithm1_route(resampling_route="none")
    callbacks = _callback_contract(spec, status)
    scalar = _scalar_contract(spec, status)
    thresholds = _threshold_contract(spec, status)
    gradient = _gradient_contract(spec, status)
    payload = {
        "contract_id": f"{spec.model_id}::ledh_pfpf_alg1_ukf::2026-06-10.p2.visible",
        "model_id": spec.model_id,
        "family": spec.family,
        "status": status["status"],
        "status_reason": status["reason"],
        "missing_adapter_items": status["missing_adapter_items"],
        "algorithm": "ledh_pfpf_alg1_ukf",
        "method_id": "ledh_pfpf_alg1_ukf_no_resampling_tf",
        "dtype": DTYPE.name,
        "theta": tensor_to_json(spec.theta),
        "state_dim": spec.state_dim,
        "observation_dim": spec.observation_dim,
        "horizon": int(spec.path_contract["horizon"]),
        "num_particles_in_contract_fixture": int(spec.path_contract["num_particles"]),
        "source_surface": spec.source_surface,
        "source_spec_checksum": spec.checksum(),
        "parameters": _jsonable(spec.parameters),
        "path_fixture": {
            "initial_particles": _jsonable(spec.path_contract["initial_particles"]),
            "transition_innovations": _jsonable(spec.path_contract["transition_innovations"]),
            "observations": _jsonable(spec.path_contract["observations"]),
            "path_contract_checksum": stable_digest(_jsonable(spec.path_contract)),
        },
        "algorithm1_route_fields": route_fields,
        "callback_contract": callbacks,
        "scalar_contract": scalar,
        "threshold_contract": thresholds,
        "stochastic_contract": {
            "value_seed_list": list(VALUE_SEEDS),
            "value_particle_counts": list(VALUE_PARTICLE_COUNTS),
            "minimum_value_seed_count": len(VALUE_SEEDS),
            "gradient_seed_list": list(GRADIENT_SEEDS),
            "gradient_particle_counts": list(GRADIENT_PARTICLE_COUNTS),
            "minimum_gradient_seed_count": len(GRADIENT_SEEDS),
            "confidence_interval": "normal 95% CI over seeds for diagnostic rows",
            "one_seed_ranking_allowed": False,
        },
        "algorithm1_parameters": {
            "pseudo_time_steps": list(PSEUDO_TIME_STEPS),
            "ukf_parameters": _ukf_parameters(),
            "covariance_floor": COVARIANCE_FLOOR,
            "rank_tolerance": RANK_TOLERANCE,
            "core_resampling_route": "none",
            "extension_resampling_route": "none",
        },
        "p3_value_contract": {
            "consumer_phase": "P3",
            "allowed_final_statuses": [
                "RERUN_ALG1_DIAGNOSTIC_ONLY",
                "N_A_NOT_APPLICABLE",
                "BLOCKED_REQUIRES_ADAPTER",
            ],
            "runnable": status["status"] == "RUNNABLE_ALG1",
            "value_scalar": scalar["value_scalar"],
            "value_tolerance": thresholds["value_tolerance"],
            "primary_promote_statistic": thresholds["primary_promote_statistic"],
        },
        "p4_gradient_contract": {
            "consumer_phase": "P4",
            "allowed_final_statuses": [
                "RERUN_ALG1_DIAGNOSTIC_ONLY",
                "N_A_NOT_APPLICABLE",
                "BLOCKED_REQUIRES_ADAPTER",
            ],
            "runnable": status["status"] == "RUNNABLE_ALG1" and gradient["same_scalar_gradient_declared"],
            "gradient_scalar": gradient["gradient_scalar"],
            "gradient_tolerance": thresholds["gradient_tolerance"],
            "gradient_scope": gradient["gradient_scope"],
        },
        "core_extension_split": {
            "source_algorithm1_core": status["status"] == "RUNNABLE_ALG1",
            "ot_extension_included": False,
            "extension_resampling_route": "none",
            "ot_or_annealed_transport_status": "not_part_of_source_algorithm1_p2_contract",
        },
        "old_evidence_policy": {
            "previous_ledh_pfpf_ot_evidence_status": PREVIOUS_LEDHPFPF_OT_EVIDENCE_STATUS,
            "old_v2_contract_used_as_current_evidence": False,
            "old_v2_contract_used_as_coverage_inventory_only": True,
        },
        "non_claims": list(spec.non_claims) + _nonclaims(),
    }
    payload["component_checksums"] = {
        "parameters": stable_digest(payload["parameters"]),
        "path_fixture": stable_digest(payload["path_fixture"]),
        "algorithm1_route_fields": stable_digest(payload["algorithm1_route_fields"]),
        "callback_contract": stable_digest(payload["callback_contract"]),
        "scalar_contract": stable_digest(payload["scalar_contract"]),
        "threshold_contract": stable_digest(payload["threshold_contract"]),
        "stochastic_contract": stable_digest(payload["stochastic_contract"]),
        "algorithm1_parameters": stable_digest(payload["algorithm1_parameters"]),
    }
    payload["contract_checksum"] = stable_digest(
        {key: value for key, value in payload.items() if key != "contract_checksum"}
    )
    return payload


def _contract_status(spec: CommonModelSpecV2) -> dict[str, Any]:
    if spec.model_id in RUNNABLE_MODEL_IDS:
        return {
            "status": "RUNNABLE_ALG1",
            "reason": "Algorithm 1 additive-noise callback contract is declared for P3/P4 diagnostic execution.",
            "missing_adapter_items": [],
        }
    if spec.model_id == "sv_1d_h18_rich":
        return {
            "status": "BLOCKED_REQUIRES_ADAPTER",
            "reason": (
                "The current V2 stochastic-volatility row has a non-Gaussian "
                "observation likelihood.  A log-square Gaussian surrogate may be "
                "a BayesFilter extension, but it is not frozen here as source "
                "Algorithm 1 evidence."
            ),
            "missing_adapter_items": [
                "reviewed observation_mean_fn and observation_covariance_fn for true or surrogate SV likelihood",
                "explicit source-core versus extension classification for log-square surrogate",
                "target observation_log_density_fn matching the chosen scalar",
                "same-scalar value and gradient comparator policy",
            ],
        }
    if spec.model_id == "structural_ar1_quadratic_h16":
        return {
            "status": "BLOCKED_REQUIRES_ADAPTER",
            "reason": (
                "The structural row uses stochastic m dynamics plus deterministic "
                "k completion.  A reviewed singular-completion Algorithm 1 adapter "
                "is required before source-core execution."
            ),
            "missing_adapter_items": [
                "reviewed stochastic-coordinate transition_sample and transition_log_density",
                "deterministic k-completion carried consistently through post-flow state",
                "PSD covariance lifecycle for singular completion without false full-state density",
                "same-scalar gradient parameterization for structural knobs",
            ],
        }
    return {
        "status": "N_A_NOT_APPLICABLE",
        "reason": "No P2 Algorithm 1 contract classification exists for this model id.",
        "missing_adapter_items": ["unexpected V2 model id"],
    }


def _callback_contract(
    spec: CommonModelSpecV2,
    status: dict[str, Any],
) -> dict[str, Any]:
    base = {
        "implementation_status": (
            "DECLARED_FOR_P3_P4_IMPLEMENTATION"
            if status["status"] == "RUNNABLE_ALG1"
            else "BLOCKED_REQUIRES_ADAPTER"
        ),
        "initial_sample_route": "fixed/stateless sampling from model initial distribution using P2 seeds",
        "initial_covariance_route": "model initial covariance",
        "transition_sample_route": "transition_mean_fn plus additive process noise or fixed transition innovations",
        "transition_mean_fn": "P2 model-specific route",
        "transition_log_density_fn": "P2 model-specific route",
        "observation_mean_fn": "P2 model-specific route",
        "observation_jacobian_fn": "P2 model-specific route",
        "observation_log_density_fn": "P2 model-specific route",
        "process_noise_covariance_fn": "P2 model-specific route",
        "observation_covariance_fn": "P2 model-specific route",
        "all_callbacks_required_before_p3_execution": True,
    }
    model_routes = {
        "lgssm_2d_h25_rich": {
            "initial_covariance_route": "parameters.P0",
            "transition_mean_fn": "points @ parameters.A.T",
            "transition_log_density_fn": "Normal(points_next; points_prev @ A.T, Q)",
            "observation_mean_fn": "points @ parameters.C.T",
            "observation_jacobian_fn": "constant parameters.C",
            "observation_log_density_fn": "Normal(y; points @ C.T, R)",
            "process_noise_covariance_fn": "constant parameters.Q",
            "observation_covariance_fn": "constant parameters.R",
            "comparator_route": "exact_kalman_for_lgssm",
        },
        "range_bearing_4d_h20_rich": {
            "initial_covariance_route": "parameters.P0",
            "transition_mean_fn": "points @ parameters.A.T",
            "transition_log_density_fn": "Normal(points_next; points_prev @ A.T, Q)",
            "observation_mean_fn": "range_bearing_observation_tf(points)",
            "observation_jacobian_fn": "per-particle range-bearing Jacobian with angle residual convention",
            "observation_log_density_fn": "wrapped-bearing Gaussian observation density",
            "process_noise_covariance_fn": "constant parameters.Q",
            "observation_covariance_fn": "constant parameters.R",
            "comparator_route": "diagnostic_only_no_exact_oracle_in_P3",
        },
        "spatial_sir_j3_rk4": {
            "initial_covariance_route": "parameters.initial_covariance",
            "transition_mean_fn": "SpatialSIRSSM.transition_mean local RK4 route",
            "transition_log_density_fn": "SpatialSIRSSM.transition_log_density",
            "observation_mean_fn": "infectious-coordinate projection",
            "observation_jacobian_fn": "constant infectious-coordinate projection Jacobian",
            "observation_log_density_fn": "SpatialSIRSSM.observation_log_density",
            "process_noise_covariance_fn": "constant parameters.process_covariance",
            "observation_covariance_fn": "constant parameters.observation_covariance",
            "comparator_route": "diagnostic_only_no_exact_oracle_in_P3",
        },
        "predator_prey_rk4": {
            "initial_covariance_route": "parameters.initial_covariance",
            "transition_mean_fn": "PredatorPreySSM.transition_mean(theta, points) local RK4 route",
            "transition_log_density_fn": "PredatorPreySSM.transition_log_density",
            "observation_mean_fn": "identity direct-state observation",
            "observation_jacobian_fn": "identity observation Jacobian",
            "observation_log_density_fn": "PredatorPreySSM.observation_log_density",
            "process_noise_covariance_fn": "constant parameters.process_covariance",
            "observation_covariance_fn": "constant parameters.observation_covariance",
            "comparator_route": "diagnostic_only_no_exact_oracle_in_P3",
        },
    }
    if spec.model_id in model_routes:
        base.update(model_routes[spec.model_id])
    return base


def _scalar_contract(
    spec: CommonModelSpecV2,
    status: dict[str, Any],
) -> dict[str, Any]:
    return {
        "value_scalar": "sum of per-step predictive log normalizers",
        "gradient_scalar": "fixed-branch value scalar when same-scalar gradient is enabled",
        "normalization": "absolute and per-observation log-likelihood error when comparator exists",
        "comparator_route": _callback_contract(spec, status).get(
            "comparator_route",
            "N/A_BLOCKED_REQUIRES_ADAPTER",
        ),
        "comparator_status": (
            "exact_oracle"
            if spec.model_id == "lgssm_2d_h25_rich" and status["status"] == "RUNNABLE_ALG1"
            else "diagnostic_or_blocked"
        ),
        "value_execution_in_p2": False,
        "gradient_execution_in_p2": False,
    }


def _threshold_contract(
    spec: CommonModelSpecV2,
    status: dict[str, Any],
) -> dict[str, Any]:
    if status["status"] != "RUNNABLE_ALG1":
        reason = "N/A_BLOCKED_REQUIRES_ADAPTER"
    else:
        reason = "N/A_DIAGNOSTIC_ONLY_IN_P3_P4"
    return {
        "value_tolerance": reason,
        "gradient_tolerance": reason,
        "certification_band": reason,
        "primary_promote_statistic": reason,
        "finite_only_promotion_allowed": False,
        "threshold_source": (
            "P2 intentionally freezes diagnostic-only status; calibrated "
            "thresholds require a later exact-oracle/calibration phase before "
            "execution."
        ),
        "model_noise_horizon_particle_policy": (
            "No global threshold is set because noise scale, horizon length, "
            "dimension, nonlinearity, and particle count change Monte Carlo "
            "uncertainty."
        ),
        "p3_value_row_can_promote": False,
        "p4_gradient_row_can_promote": False,
    }


def _gradient_contract(
    spec: CommonModelSpecV2,
    status: dict[str, Any],
) -> dict[str, Any]:
    knobs = _jsonable(spec.gradient_contract.get("knobs", []))
    included = [knob["name"] for knob in knobs if bool(knob.get("include"))]
    return {
        "gradient_scalar": "fixed-branch sum of predictive log normalizers",
        "gradient_scope": (
            "diagnostic fixed-branch AD through Algorithm 1 value path only; "
            "random sampling and Boolean branch decisions excluded"
        ),
        "same_scalar_gradient_declared": bool(included) and status["status"] == "RUNNABLE_ALG1",
        "knobs": knobs,
        "included_required_knob_names": included,
        "finite_difference_step": spec.gradient_contract.get("finite_difference_step"),
        "finite_difference_policy": "diagnostic_only_not_a_promotion_gate",
        "stochastic_resampling_gradient_claim": "not_claimed",
    }


def _veto_diagnostics(contracts: list[dict[str, Any]]) -> dict[str, bool]:
    ids = [contract["model_id"] for contract in contracts]
    statuses = [contract["status"] for contract in contracts]
    return {
        "row_count_mismatch": tuple(ids) != EXPECTED_V2_MODEL_IDS,
        "contract_status_invalid": any(status not in ALLOWED_CONTRACT_STATUSES for status in statuses),
        "old_ledh_pfpf_ot_runtime_module_imported": _old_runtime_module_loaded(),
        "old_route_used_as_current_algorithm1_evidence": any(
            _old_route_value_leaked(contract["algorithm1_route_fields"])
            or contract["old_evidence_policy"]["old_v2_contract_used_as_current_evidence"]
            for contract in contracts
        ),
        "missing_mandatory_algorithm1_route_field": any(
            not _is_augmented_algorithm1_route(contract["algorithm1_route_fields"])
            for contract in contracts
        ),
        "ot_labelled_as_algorithm1_core": any(
            contract["core_extension_split"]["ot_extension_included"]
            or contract["algorithm1_route_fields"]["extension_resampling_route"] != "none"
            for contract in contracts
        ),
        "blocked_row_missing_adapter_items": any(
            contract["status"] == "BLOCKED_REQUIRES_ADAPTER"
            and not contract["missing_adapter_items"]
            for contract in contracts
        ),
        "runnable_row_has_missing_adapter_items": any(
            contract["status"] == "RUNNABLE_ALG1"
            and bool(contract["missing_adapter_items"])
            for contract in contracts
        ),
        "scalar_or_gradient_object_unspecified": any(
            not contract["scalar_contract"]["value_scalar"]
            or not contract["p4_gradient_contract"]["gradient_scalar"]
            for contract in contracts
        ),
        "threshold_missing_without_reason": any(
            not str(contract["threshold_contract"]["value_tolerance"]).startswith("N/A")
            or not str(contract["threshold_contract"]["gradient_tolerance"]).startswith("N/A")
            for contract in contracts
        ),
        "value_or_gradient_executed_in_p2": False,
        "finite_only_promotion_allowed": any(
            contract["threshold_contract"]["finite_only_promotion_allowed"]
            for contract in contracts
        ),
    }


def _validate_payload(payload: dict[str, Any]) -> None:
    required = {
        "decision",
        "phase",
        "evidence_contract",
        "threshold_policy",
        "required_v2_model_ids",
        "row_count_gate",
        "contract_bundle_payload",
        "contract_bundle_checksum",
        "contracts",
        "summary",
        "veto_diagnostics",
        "execution_diagnostics",
        "artifact_paths",
        "run_manifest",
        "nonclaims",
        "reproducibility_digest",
    }
    missing = required.difference(payload)
    if missing:
        raise P2ValidationError(f"missing payload fields {sorted(missing)}")
    if payload["decision"] != "PASS_P2_V2_ALG1_UKF_CONTRACTS_PENDING_CLAUDE_REVIEW":
        raise P2ValidationError(f"P2 decision is not pass: {payload['decision']}")
    if payload["phase"] != "P2":
        raise P2ValidationError(f"unexpected phase {payload['phase']}")
    if tuple(payload["required_v2_model_ids"]) != EXPECTED_V2_MODEL_IDS:
        raise P2ValidationError("required V2 model ids drifted")
    contracts = list(payload["contracts"])
    ids = [contract["model_id"] for contract in contracts]
    if tuple(ids) != EXPECTED_V2_MODEL_IDS:
        raise P2ValidationError(f"contract id gate failed: {ids}")
    if payload["row_count_gate"]["actual_count"] != len(EXPECTED_V2_MODEL_IDS):
        raise P2ValidationError("row-count gate failed")
    if payload["execution_diagnostics"]["algorithm1_values_computed"]:
        raise P2ValidationError("P2 computed values")
    if payload["execution_diagnostics"]["algorithm1_gradients_computed"]:
        raise P2ValidationError("P2 computed gradients")
    if payload["execution_diagnostics"]["algorithm1_flow_executed"]:
        raise P2ValidationError("P2 executed Algorithm 1 flow")
    if payload["execution_diagnostics"]["old_ledh_pfpf_ot_imported"]:
        raise P2ValidationError("old LEDH-PFPF-OT module was imported")
    for contract in contracts:
        _validate_contract(contract)
    if any(bool(value) for value in payload["veto_diagnostics"].values()):
        raise P2ValidationError(f"true veto diagnostics: {payload['veto_diagnostics']}")
    bundle = payload["contract_bundle_payload"]
    expected_bundle = {
        "artifact_id": "dpf_v2_ledh_pfpf_alg1_ukf_contracts_2026-06-10",
        "version": "2026-06-10.p2.visible",
        "contracts": contracts,
    }
    if bundle != expected_bundle:
        raise P2ValidationError("contract bundle payload does not match contracts")
    if payload["contract_bundle_checksum"] != stable_digest(bundle):
        raise P2ValidationError("contract bundle checksum mismatch")
    manifest = payload["run_manifest"]
    if manifest["pre_import_cuda_visible_devices"] != "-1":
        raise P2ValidationError("TensorFlow was not forced CPU-only before import")


def _validate_contract(contract: dict[str, Any]) -> None:
    required = {
        "contract_id",
        "model_id",
        "status",
        "status_reason",
        "missing_adapter_items",
        "algorithm",
        "method_id",
        "dtype",
        "state_dim",
        "observation_dim",
        "horizon",
        "parameters",
        "path_fixture",
        "algorithm1_route_fields",
        "callback_contract",
        "scalar_contract",
        "threshold_contract",
        "stochastic_contract",
        "algorithm1_parameters",
        "p3_value_contract",
        "p4_gradient_contract",
        "core_extension_split",
        "old_evidence_policy",
        "component_checksums",
        "contract_checksum",
    }
    missing = required.difference(contract)
    if missing:
        raise P2ValidationError(f"{contract.get('model_id')}: missing fields {sorted(missing)}")
    if contract["status"] not in ALLOWED_CONTRACT_STATUSES:
        raise P2ValidationError(f"{contract['model_id']}: invalid status {contract['status']}")
    if contract["algorithm"] != "ledh_pfpf_alg1_ukf":
        raise P2ValidationError(f"{contract['model_id']}: wrong algorithm")
    if contract["method_id"] != "ledh_pfpf_alg1_ukf_no_resampling_tf":
        raise P2ValidationError(f"{contract['model_id']}: wrong method id")
    if contract["dtype"] != DTYPE.name:
        raise P2ValidationError(f"{contract['model_id']}: dtype mismatch")
    if not _is_augmented_algorithm1_route(contract["algorithm1_route_fields"]):
        raise P2ValidationError(f"{contract['model_id']}: route fields invalid")
    if _old_route_value_leaked(contract["algorithm1_route_fields"]):
        raise P2ValidationError(f"{contract['model_id']}: old route leaked into route values")
    if contract["old_evidence_policy"]["old_v2_contract_used_as_current_evidence"]:
        raise P2ValidationError(f"{contract['model_id']}: old evidence used")
    if contract["status"] == "BLOCKED_REQUIRES_ADAPTER" and not contract["missing_adapter_items"]:
        raise P2ValidationError(f"{contract['model_id']}: blocked row missing adapter items")
    if contract["status"] == "RUNNABLE_ALG1" and contract["missing_adapter_items"]:
        raise P2ValidationError(f"{contract['model_id']}: runnable row has adapter blockers")
    if contract["threshold_contract"]["finite_only_promotion_allowed"]:
        raise P2ValidationError(f"{contract['model_id']}: finite-only promotion allowed")
    if not str(contract["threshold_contract"]["value_tolerance"]).startswith("N/A"):
        raise P2ValidationError(f"{contract['model_id']}: value threshold lacks N/A diagnostic policy")
    if not str(contract["threshold_contract"]["gradient_tolerance"]).startswith("N/A"):
        raise P2ValidationError(f"{contract['model_id']}: gradient threshold lacks N/A diagnostic policy")
    if contract["core_extension_split"]["ot_extension_included"]:
        raise P2ValidationError(f"{contract['model_id']}: OT included in source core")
    if not _json_finite(
        {
            "path_fixture": contract["path_fixture"],
            "theta": contract["theta"],
        }
    ):
        raise P2ValidationError(f"{contract['model_id']}: nonfinite contract fixture field")
    clone = dict(contract)
    checksum = clone.pop("contract_checksum")
    if checksum != stable_digest(clone):
        raise P2ValidationError(f"{contract['model_id']}: contract checksum mismatch")


def _preflight(registry: dict[str, Any], p1_payload: dict[str, Any]) -> None:
    if registry.get("schema_version") != "dpf_ledh_pfpf_alg1_ukf_rerun_registry.v1":
        raise P2ValidationError("P0 registry schema mismatch")
    p2_rows = [
        row for row in registry.get("registry_rows", [])
        if row.get("replacement_phase") == "P2"
    ]
    if [row.get("old_lane_id") for row in p2_rows] != ["v2_contracts"]:
        raise P2ValidationError("P0 registry does not expose exactly the P2 v2_contracts row")
    if p1_payload.get("decision") != "PASS_P1_DIRECT_REPLACEMENTS_DIAGNOSTIC_ONLY_PENDING_CLAUDE_REVIEW":
        raise P2ValidationError("P1 direct replacement artifact has not passed local execution")
    if any(bool(value) for value in p1_payload.get("veto_diagnostics", {}).values()):
        raise P2ValidationError("P1 artifact contains true veto diagnostics")


def _summary(contracts: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_contracts": len(contracts),
        "models": [contract["model_id"] for contract in contracts],
        "status_counts": {
            status: sum(1 for contract in contracts if contract["status"] == status)
            for status in sorted(ALLOWED_CONTRACT_STATUSES)
        },
        "runnable_models": [
            contract["model_id"] for contract in contracts
            if contract["status"] == "RUNNABLE_ALG1"
        ],
        "blocked_models": [
            contract["model_id"] for contract in contracts
            if contract["status"] == "BLOCKED_REQUIRES_ADAPTER"
        ],
        "extension_rows": [
            contract["model_id"] for contract in contracts
            if contract["algorithm1_route_fields"]["evidence_route_class"]
            == "BAYESFILTER_EXTENSION_NOT_SOURCE_CORE"
        ],
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# P2 Result: V2 Algorithm 1 UKF Contract Replacement",
        "",
        "metadata_date: 2026-06-10",
        "phase: P2",
        f"status: {payload['decision']}",
        "",
        "## Skeptical Plan Audit",
        "",
        f"Status: `{payload['skeptical_plan_audit']['status']}`.",
        "",
        payload["skeptical_plan_audit"]["wrong_baseline_control"],
        "",
        payload["skeptical_plan_audit"]["proxy_metric_control"],
        "",
        payload["skeptical_plan_audit"]["stop_conditions"],
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
        "## Contract Rows",
        "",
        "| Model | Status | Comparator | Value tolerance | Gradient tolerance | Missing adapter items |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for contract in payload["contracts"]:
        missing = "; ".join(contract["missing_adapter_items"]) or "none"
        lines.append(
            f"| `{contract['model_id']}` | `{contract['status']}` | "
            f"`{contract['scalar_contract']['comparator_route']}` | "
            f"`{contract['threshold_contract']['value_tolerance']}` | "
            f"`{contract['threshold_contract']['gradient_tolerance']}` | {missing} |"
        )
    lines.extend(
        [
            "",
            "## Route Fields",
            "",
            "| Field | Value |",
            "| --- | --- |",
        ]
    )
    route = payload["contracts"][0]["algorithm1_route_fields"]
    for key, value in route.items():
        lines.append(f"| `{key}` | `{value}` |")
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
            "## Summary",
            "",
            f"- contract bundle checksum: `{payload['contract_bundle_checksum']}`",
            f"- status counts: `{payload['summary']['status_counts']}`",
            f"- runnable models: `{payload['summary']['runnable_models']}`",
            f"- blocked models: `{payload['summary']['blocked_models']}`",
            "",
            "## Decision Table",
            "",
            "| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |",
            "| --- | --- | --- | --- | --- | --- |",
            (
                f"| `{payload['decision']}` | six V2 contracts frozen in order with "
                "diagnostic-only thresholds | "
                f"`{payload['veto_diagnostics']}` | P3/P4 still must implement and validate callback adapters | "
                "Claude P2 read-only review, then P3 values consume frozen contracts | "
                "no value, gradient, performance, OT-extension, or production claim |"
            ),
            "",
            "## Post-Run Red-Team Note",
            "",
            "Strongest alternative explanation: a row marked runnable may still fail in P3 because the declared callback route has not yet been executed.",
            "",
            "Result that would overturn the local decision: P3/P4 or Claude finds a missing callback, an old-route leak, a row-order mismatch, or an implicit numerical threshold.",
            "",
            "Weakest part of the evidence: P2 is declarative contract freeze only; it deliberately does not execute Algorithm 1 values or gradients.",
            "",
            "## Run Manifest",
            "",
            f"- command: `{payload['run_manifest']['command']}`",
            f"- git branch: `{payload['run_manifest']['branch']}`",
            f"- git commit: `{payload['run_manifest']['commit']}`",
            f"- CPU/GPU status: CPU-only, `CUDA_VISIBLE_DEVICES={payload['run_manifest']['pre_import_cuda_visible_devices']}` before TensorFlow import",
            f"- visible GPU devices: `{payload['run_manifest']['gpu_devices_visible']}`",
            f"- value seeds: `{payload['run_manifest']['value_seed_list']}`",
            f"- value particle counts: `{payload['run_manifest']['value_particle_counts']}`",
            f"- gradient seeds: `{payload['run_manifest']['gradient_seed_list']}`",
            f"- gradient particle counts: `{payload['run_manifest']['gradient_particle_counts']}`",
            f"- pseudo-time steps: `{payload['run_manifest']['pseudo_time_steps']}`",
            f"- UKF parameters: `{payload['run_manifest']['ukf_parameters']}`",
            f"- plan: `{payload['run_manifest']['plan_path']}`",
            f"- registry: `{payload['run_manifest']['registry_path']}`",
            f"- P1 JSON: `{payload['run_manifest']['p1_json_path']}`",
            f"- review ledger: `{payload['run_manifest']['review_ledger_path']}`",
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
            "P2 is pending Claude read-only review.",
            "",
        ]
    )
    return "\n".join(lines)


def _augmented_algorithm1_route(*, resampling_route: str) -> dict[str, str]:
    route = algorithm1_route_identifiers(resampling_route=resampling_route)
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
    expected = {
        "method_generation": METHOD_GENERATION,
        "flow_source_route": FLOW_SOURCE_ROUTE,
        "covariance_route": COVARIANCE_ROUTE,
        "prediction_covariance_route": "ukf_prediction_per_particle_covariance",
        "update_covariance_route": "ukf_update_per_particle_covariance",
        "flow_anchor_route": FLOW_ANCHOR_ROUTE,
        "resampling_route": "none",
        "core_resampling_route": "none",
        "extension_resampling_route": "none",
        "evidence_route_class": "SOURCE_ALGORITHM1_CORE",
        "previous_ledh_pfpf_ot_evidence_status": PREVIOUS_LEDHPFPF_OT_EVIDENCE_STATUS,
    }
    return all(route.get(key) == value for key, value in expected.items())


def _old_route_value_leaked(route: dict[str, str]) -> bool:
    for key, value in route.items():
        if key == "previous_ledh_pfpf_ot_evidence_status":
            continue
        if "ledh_pfpf_ot" in str(value).lower():
            return True
    return False


def _old_runtime_module_loaded() -> bool:
    return any(
        name.endswith(".ledh_pfpf_ot_tf")
        or name.endswith(".run_v2_ledh_pfpf_ot_contracts_tf")
        or name.endswith(".run_v2_ledh_pfpf_ot_values_tf")
        or name.endswith(".run_v2_ledh_pfpf_ot_gradients_tf")
        for name in sys.modules
    )


def _digest_payload(payload: dict[str, Any]) -> str:
    stable = {
        key: value
        for key, value in payload.items()
        if key not in {"created_at_utc", "run_manifest", "reproducibility_digest"}
    }
    return stable_digest(stable)


def _jsonable(value: Any) -> Any:
    if tf.is_tensor(value):
        return tensor_to_json(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float):
        return float(value)
    if hasattr(value, "numpy"):
        return _jsonable(value.numpy())
    return value


def _json_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_json_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_json_finite(item) for item in value)
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


def _nonclaims() -> list[str]:
    return [
        "P2 freezes contracts only; it does not execute Algorithm 1 values.",
        "P2 does not execute gradients or validate gradient correctness.",
        "P2 does not set calibrated numerical value or gradient thresholds.",
        "P2 does not rank filters or compare performance.",
        "OT and annealed transport are not part of the source Algorithm 1 core.",
        "Blocked rows are adapter work items, not negative scientific evidence.",
        "P2 does not establish production readiness, HMC readiness, GPU performance, or broad model superiority.",
    ]


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
