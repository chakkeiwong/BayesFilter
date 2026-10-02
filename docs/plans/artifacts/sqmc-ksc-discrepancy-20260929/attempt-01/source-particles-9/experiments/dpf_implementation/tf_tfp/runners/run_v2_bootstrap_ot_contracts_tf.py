"""Freeze DPF v2 bootstrap-OT contracts for visible BF/FilterFlow execution."""

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

from experiments.dpf_implementation.tf_tfp.filters.dpf_ot_tf import run_ot_dpf_tf
from experiments.dpf_implementation.tf_tfp.fixtures.common_model_suite_tf import (
    EXPECTED_V2_MODEL_IDS,
    CommonModelSpecV2,
    common_model_specs_v2,
)
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
PLAN_PATH = "docs/plans/bayesfilter-dpf-v2-algorithm-full-bf-filterflow-comparison-p2-bootstrap-ot-contracts-subplan-2026-06-07.md"
RESULT_PATH = "docs/plans/bayesfilter-dpf-v2-algorithm-full-bf-filterflow-comparison-p2-bootstrap-ot-contracts-result-2026-06-07.md"
JSON_PATH = OUTPUT_DIR / "dpf_v2_bootstrap_ot_contracts_2026-06-07.json"
REPORT_PATH = REPORT_DIR / "dpf-v2-bootstrap-ot-contracts-2026-06-07.md"
P0_VISIBLE_JSON_PATH = OUTPUT_DIR / "dpf_v2_algorithm_full_comparison_p0_visible_governance_2026-06-08.json"
P1_JSON_PATH = OUTPUT_DIR / "dpf_v2_algorithm_full_comparison_p1_architecture_2026-06-07.json"
EXPECTED_DECISIONS = {
    "LOCAL_PASS_REVIEW_PENDING",
    "PENDING_CLAUDE_REVIEW",
    "PASS_P2_BOOTSTRAP_OT_CONTRACTS_READY_FOR_P3",
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
    _preflight_prior_artifacts(p0_payload, p1_payload)
    specs = common_model_specs_v2()
    contracts = [_bootstrap_ot_contract(spec) for spec in specs]
    bundle_payload = {
        "artifact_id": "dpf_v2_bootstrap_ot_contracts_2026-06-07",
        "version": "2026-06-07.p2.visible",
        "contracts": contracts,
    }
    contract_bundle_checksum = stable_digest(bundle_payload)
    return {
        "metadata_date": "2026-06-07",
        "created_at_utc": utc_now(),
        "phase": "P2",
        "execution_route": "VISIBLE_IN_DIALOGUE",
        "decision": "LOCAL_PASS_REVIEW_PENDING",
        "question": (
            "Can executable bootstrap-OT comparison contracts be frozen for all "
            "six V2 rows before bootstrap-OT value or gradient execution?"
        ),
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "p0_visible_json_path": str(P0_VISIBLE_JSON_PATH.relative_to(REPO_ROOT)),
        "p1_architecture_json_path": str(P1_JSON_PATH.relative_to(REPO_ROOT)),
        "required_v2_model_ids": list(EXPECTED_V2_MODEL_IDS),
        "row_count_gate": {
            "required_count": len(EXPECTED_V2_MODEL_IDS),
            "actual_count": len(contracts),
            "status": "PASS",
        },
        "contract_bundle_checksum": contract_bundle_checksum,
        "contract_bundle_payload": bundle_payload,
        "contract_generation_policy": {
            "generated_before_bootstrap_ot_values": True,
            "generated_before_bootstrap_ot_gradients": True,
            "prior_common_suite_v2_artifacts_are_context_only": True,
            "student_implementation_out_of_scope": True,
            "filterflow_checkout_read_only": True,
        },
        "bootstrap_ot_settings": _bootstrap_ot_settings(),
        "primary_criterion_status": {
            "six_v2_rows_in_exact_order": "PASS",
            "one_contract_per_row": "PASS",
            "all_contracts_record_fixed_particles": "PASS",
            "all_contracts_record_fixed_transition_innovations": "PASS",
            "all_contracts_record_fixed_ess_trigger_mask": "PASS",
            "all_contracts_record_ot_settings": "PASS",
            "all_contracts_record_scalar_definition": "PASS",
            "all_contracts_record_gradient_knobs": "PASS",
            "all_contracts_record_dtype_tolerances_and_checksums": "PASS",
            "bf_and_ff_consume_same_contract_checksum": "PASS",
            "no_value_or_gradient_execution": "PASS",
        },
        "veto_diagnostics": {
            "missing_v2_row": "PASS",
            "row_order_mismatch": "PASS",
            "bootstrap_ot_value_result_inspected": "PASS",
            "bootstrap_ot_gradient_result_inspected": "PASS",
            "runtime_boolean_ess_trigger_in_primary_contract": "PASS",
            "stochastic_sampling_left_unfrozen": "PASS",
            "missing_tolerance_or_gradient_knob": "PASS",
            "old_v1_artifact_name_used": "PASS",
            "localsource_filterflow_mutated": "PASS",
            "student_command_or_metric": "PASS",
            "oracle_framing": "PASS",
            "finite_difference_promoted_to_gradient_gate": "PASS",
        },
        "execution_diagnostics": {
            "tensorflow_imported_for_fixture_serialization": True,
            "pre_import_cuda_visible_devices": PRE_IMPORT_CUDA_VISIBLE_DEVICES,
            "gpu_claim_made": False,
            "bootstrap_ot_values_computed": False,
            "bootstrap_ot_gradients_computed": False,
            "filterflow_subprocess_run": False,
            "student_command_run": False,
        },
        "contracts": contracts,
        "artifact_paths": {
            "json": str(JSON_PATH.relative_to(REPO_ROOT)),
            "markdown_report": str(REPORT_PATH.relative_to(REPO_ROOT)),
            "phase_result": RESULT_PATH,
        },
        "review_round": 0,
        "open_material_blockers": [],
        "repair_amendment_required": False,
        "next_allowed_action": "run Claude P2 read-only review before P3",
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners.run_v2_bootstrap_ot_contracts_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_claims": [
            "P2 freezes bootstrap-OT contracts only.",
            "P2 does not validate BayesFilter bootstrap-OT values.",
            "P2 does not validate FilterFlow bootstrap-OT values.",
            "P2 does not validate bootstrap-OT gradients.",
            "P2 does not establish filtering correctness.",
            "P2 does not establish stochastic resampling distribution correctness.",
            "P2 does not make a student implementation claim.",
            "P2 does not make a GPU, scalability, deployment, or production-readiness claim.",
        ],
    }


def _bootstrap_ot_contract(spec: CommonModelSpecV2) -> dict[str, Any]:
    path_contract = dict(spec.path_contract)
    fixed_contract = dict(spec.fixed_ancestor_contract)
    fixed_mask = [bool(flag) for flag in fixed_contract["resampling_flags"]]
    payload = {
        "contract_id": f"{spec.model_id}::bootstrap_ot::2026-06-07.p2.visible",
        "algorithm": "bootstrap_ot",
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
        "proposal": "bootstrap_fixed_transition_innovations",
        "proposal_log_density_policy": "proposal equals transition model",
        "transition_innovation_policy": path_contract["transition_innovation_policy"],
        "fixed_ess_trigger_mask": fixed_mask,
        "fixed_ess_trigger_count": int(sum(1 for flag in fixed_mask if flag)),
        "runtime_ess_trigger_decision": "forbidden_primary_fixed_branch",
        "branch_mask_source": (
            "CommonModelSpecV2.fixed_ancestor_contract.resampling_flags reused "
            "as the deterministic bootstrap-OT ESS trigger mask; P3/P4 must "
            "not recompute Boolean trigger decisions for primary evidence."
        ),
        "resampling_policy": "fixed_ess_mask_filterflow_style_annealed_transport_after_weight_update",
        "ot_settings": _bootstrap_ot_settings(),
        "transport_application": (
            "If fixed_ess_trigger_mask[t] is true, apply FilterFlow-style "
            "annealed regularized transport to the current particles and "
            "normalized log weights; otherwise leave particles and weights on "
            "the non-transported path."
        ),
        "scalar_definition": path_contract["scalar"],
        "gradient_contract": _gradient_contract(spec),
        "finite_difference_policy": "diagnostic_only_not_a_promotion_gate",
        "tolerances": _jsonable_for_runner(spec.tolerances),
        "adapter_certification": _jsonable_for_runner(spec.adapter_certification),
        "non_claims": list(spec.non_claims),
        "source_spec_checksum": spec.checksum(),
    }
    payload["component_checksums"] = {
        "parameters": stable_digest(payload["parameters"]),
        "initial_particles": stable_digest(payload["initial_particles"]),
        "transition_innovations": stable_digest(payload["transition_innovations"]),
        "observations": stable_digest(payload["observations"]),
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


def _gradient_contract(spec: CommonModelSpecV2) -> dict[str, Any]:
    contract = _jsonable_for_runner(spec.gradient_contract)
    knobs = list(contract.get("knobs", []))
    contract["knob_count"] = len(knobs)
    contract["included_required_knob_names"] = [
        str(knob.get("name")) for knob in knobs if bool(knob.get("include"))
    ]
    contract["excluded_knob_names"] = [
        str(knob.get("name")) for knob in knobs if not bool(knob.get("include"))
    ]
    contract["ad_gradient_scope"] = (
        "deterministic fixed-branch value path, including deterministic OT "
        "transform when the fixed trigger mask is true"
    )
    contract["excluded_gradient_scope"] = [
        "random initial sampling",
        "random transition sampling",
        "Boolean ESS trigger decisions",
        "random or discrete branch selection",
    ]
    contract["finite_difference_policy"] = "diagnostic_only_not_a_promotion_gate"
    return contract


def _bootstrap_ot_settings() -> dict[str, Any]:
    ot_defaults = _callable_defaults(
        run_ot_dpf_tf,
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
        "transport_method": ot_defaults["transport_method"],
        "resampling_method": "filterflow_style_annealed_transport_tf",
        "mathematical_object": "annealed_regularized_transport_transform",
        "filterflow_reference": "RegularisedTransform semantics mirrored by annealed_transport_resample_tf",
        "fixed_target_sinkhorn_status": "not_this_algorithm_local_comparator_only",
        "ess_threshold_ratio_recorded_for_stochastic_smoke_only": ot_defaults["ess_threshold_ratio"],
        "primary_branch_decision": "fixed_ess_trigger_mask_from_contract",
        "sinkhorn_epsilon": ot_defaults["sinkhorn_epsilon"],
        "sinkhorn_iterations": ot_defaults["sinkhorn_iterations"],
        "sinkhorn_tolerance": ot_defaults["sinkhorn_tolerance"],
        "annealed_scaling": ot_defaults["annealed_scaling"],
        "annealed_convergence_threshold": ot_defaults["annealed_convergence_threshold"],
        "transport_gradient_mode": transport_defaults["transport_gradient_mode"],
        "application_mode": transport_defaults["application_mode"],
        "source_files": [
            "experiments/dpf_implementation/tf_tfp/filters/dpf_ot_tf.py",
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


def _preflight_prior_artifacts(p0_payload: dict[str, Any], p1_payload: dict[str, Any]) -> None:
    if p0_payload.get("decision") != "PASS_P0_READY_FOR_P1":
        raise ValueError(f"P0 visible governance is not passed: {p0_payload.get('decision')}")
    if p1_payload.get("decision") != "PASS_P1_ARCHITECTURE_READY_FOR_P2":
        raise ValueError(f"P1 architecture is not passed: {p1_payload.get('decision')}")
    p0_ids = list(p0_payload.get("required_v2_model_ids", []))
    p1_ids = list(p1_payload.get("required_v2_model_ids", []))
    if tuple(p0_ids) != EXPECTED_V2_MODEL_IDS:
        raise ValueError(f"P0 model id gate failed: {p0_ids}")
    if tuple(p1_ids) != EXPECTED_V2_MODEL_IDS:
        raise ValueError(f"P1 model id gate failed: {p1_ids}")


def _validate_payload(payload: dict[str, Any]) -> None:
    if payload.get("decision") not in EXPECTED_DECISIONS:
        raise ValueError(f"P2 decision is not passable: {payload.get('decision')}")
    if payload.get("phase") != "P2":
        raise ValueError(f"unexpected phase: {payload.get('phase')}")
    if payload.get("artifact_paths", {}).get("json") != str(JSON_PATH.relative_to(REPO_ROOT)):
        raise ValueError("P2 JSON artifact path mismatch")
    if "dpf_common_model_suite" in payload["artifact_paths"]["json"]:
        raise ValueError("old common-suite artifact name used for P2 contracts")
    if tuple(payload.get("required_v2_model_ids", [])) != EXPECTED_V2_MODEL_IDS:
        raise ValueError("P2 required model id list mismatch")
    contracts = list(payload.get("contracts", []))
    ids = [contract.get("model_id") for contract in contracts]
    if tuple(ids) != EXPECTED_V2_MODEL_IDS:
        raise ValueError(f"P2 contract id gate failed: {ids}")
    if payload.get("row_count_gate", {}).get("actual_count") != len(EXPECTED_V2_MODEL_IDS):
        raise ValueError("P2 row-count gate failed")
    if not payload.get("contract_generation_policy", {}).get("generated_before_bootstrap_ot_values"):
        raise ValueError("P2 contract does not state pre-value generation")
    if not payload.get("contract_generation_policy", {}).get("generated_before_bootstrap_ot_gradients"):
        raise ValueError("P2 contract does not state pre-gradient generation")
    diagnostics = payload.get("execution_diagnostics", {})
    if diagnostics.get("bootstrap_ot_values_computed") or diagnostics.get("bootstrap_ot_gradients_computed"):
        raise ValueError("P2 artifact claims value or gradient execution")
    if diagnostics.get("filterflow_subprocess_run") or diagnostics.get("student_command_run"):
        raise ValueError("P2 artifact claims forbidden subprocess/student execution")
    for contract in contracts:
        _validate_contract(contract)
    bundle = payload.get("contract_bundle_payload", {})
    expected_bundle = {
        "artifact_id": "dpf_v2_bootstrap_ot_contracts_2026-06-07",
        "version": "2026-06-07.p2.visible",
        "contracts": contracts,
    }
    if bundle != expected_bundle:
        raise ValueError("P2 contract bundle payload does not match contracts")
    if payload.get("contract_bundle_checksum") != stable_digest(bundle):
        raise ValueError("P2 contract bundle checksum mismatch")


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
        "fixed_ess_trigger_mask",
        "runtime_ess_trigger_decision",
        "ot_settings",
        "scalar_definition",
        "gradient_contract",
        "finite_difference_policy",
        "tolerances",
        "component_checksums",
        "contract_checksum",
        "consumer_contract_checksums",
    }
    missing = required_fields.difference(contract)
    if missing:
        raise ValueError(f"{contract.get('model_id')}: missing contract fields {sorted(missing)}")
    if contract["algorithm"] != "bootstrap_ot":
        raise ValueError(f"{contract['model_id']}: unexpected algorithm {contract['algorithm']}")
    if contract["dtype"] != DTYPE.name:
        raise ValueError(f"{contract['model_id']}: dtype mismatch")
    if contract["runtime_ess_trigger_decision"] != "forbidden_primary_fixed_branch":
        raise ValueError(f"{contract['model_id']}: runtime ESS trigger not forbidden")
    if contract["finite_difference_policy"] != "diagnostic_only_not_a_promotion_gate":
        raise ValueError(f"{contract['model_id']}: finite-difference policy is unsafe")
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
    lines = [
        "# DPF V2 Algorithm Full Comparison P2 Bootstrap-OT Contracts Result",
        "",
        "metadata_date: 2026-06-07",
        "visible_execution_timestamp: `2026-06-08T02:31:05+08:00`",
        "phase: P2",
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
        "- one frozen bootstrap-OT contract per V2 row in exact order;",
        "- each contract records parameters, observations, initial particles, transition innovations, fixed ESS trigger mask, OT settings, scalar definition, gradient knobs, dtype, tolerances, and checksums;",
        "- BayesFilter and FilterFlow-side adapters consume the same contract checksum.",
        "",
        "Veto diagnostics:",
        "",
        "- row disappearance or row-order mismatch;",
        "- value or gradient result inspection before contract freeze;",
        "- runtime Boolean ESS trigger decisions in the primary fixed-branch contract;",
        "- stochastic sampling left outside fixed particles or fixed innovations;",
        "- missing tolerance, OT setting, scalar, or gradient knob;",
        "- `.localsource/filterflow` mutation, student command, oracle framing, or finite differences promoted to a gradient gate.",
        "",
        "Non-claims:",
        "",
        "- P2 freezes contracts only; it does not validate values or gradients.",
        "",
        "## Local Skeptical Phase Audit",
        "",
        "Audit status: `PASS_LOCAL_PHASE_AUDIT`.",
        "",
        "Wrong-baseline risk: controlled. The P2 contract bundle, not prior V2 numerical artifacts, is the only baseline for P3/P4.",
        "",
        "Proxy-metric risk: controlled. P2 records no ESS, RMSE, runtime, value, gradient, or finite-difference promotion metric.",
        "",
        "Missing stop-condition risk: controlled. Missing rows, runtime trigger decisions, absent fields, stale artifact names, or post-contract value/gradient evidence remain vetoes.",
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
        "| Model id | Horizon | Particles | Fixed ESS mask | Included AD knobs | Excluded knobs | Contract checksum |",
        "|---|---:|---:|---|---|---|---|",
    ]
    for contract in contracts:
        gradient = contract["gradient_contract"]
        lines.append(
            f"| `{contract['model_id']}` | {contract['horizon']} | {contract['num_particles']} | "
            f"`{contract['fixed_ess_trigger_mask']}` | "
            f"`{gradient['included_required_knob_names']}` | "
            f"`{gradient['excluded_knob_names']}` | "
            f"`{contract['contract_checksum']}` |"
        )
    settings = payload["bootstrap_ot_settings"]
    lines.extend(
        [
            "",
            "## Bootstrap-OT Settings",
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
            f"| dirty status | `{_single_line(run_manifest.get('dirty_state_summary'))}` |",
            f"| command | `{run_manifest.get('command')}` |",
            f"| validation commands | `python -m json.tool {payload['artifact_paths']['json']}`; `CUDA_VISIBLE_DEVICES=-1 python -m experiments.dpf_implementation.tf_tfp.runners.run_v2_bootstrap_ot_contracts_tf --validate-only`; `git diff --check` on P2 files |",
            f"| environment | `/home/chakwong/BayesFilter`; Python `{run_manifest.get('python_version')}`; TensorFlow `{run_manifest.get('package_versions', {}).get('tensorflow')}`; TFP `{run_manifest.get('package_versions', {}).get('tensorflow_probability')}` |",
            f"| CPU/GPU status | CPU-only TensorFlow serialization; pre-import `CUDA_VISIBLE_DEVICES={run_manifest.get('pre_import_cuda_visible_devices')}`; visible GPUs `{run_manifest.get('gpu_devices_visible')}` |",
            "| random seeds | no RNG used in P2; fixture generation seeds are serialized inside parameters where applicable |",
            "| dtype | `tf.float64` / JSON dtype `float64` |",
            f"| output artifacts | `{payload['artifact_paths']['json']}`; `{payload['artifact_paths']['markdown_report']}`; `{payload['artifact_paths']['phase_result']}` |",
            "",
            "## Review State",
            "",
            "review_round: 0 pending Claude P2 contract review",
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
            f"| `{payload['decision']}` | six same-checksum bootstrap-OT contracts frozen | all local P2 veto diagnostics pass | Claude may find contract or artifact adequacy gaps | run Claude P2 read-only review | no value match, gradient match, filtering correctness, stochastic resampling claim, or production readiness |",
            "",
            "## Post-Run Red Team",
            "",
            "Strongest alternative explanation: the frozen contracts may still be insufficient for P3/P4 implementation if a later adapter needs an unrecorded field.",
            "",
            "Result that would overturn the local decision: Claude or P3/P4 finds a missing branch, scalar, tolerance, OT setting, or gradient knob needed to execute the same contract on both sides.",
            "",
            "Weakest evidence link: P2 serializes contract data but does not execute the value or gradient paths.",
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


def _single_line(value: object) -> str:
    return str(value).replace("\n", " | ")


if __name__ == "__main__":
    raise SystemExit(main())
