"""Run visible P3 bootstrap-OT fixed-branch value comparisons."""

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
    scalar,
    stable_digest,
    tensor_to_json,
    utc_now,
    write_json,
    write_text,
)


tfd = tfp.distributions
DTYPE = tf.float64
PLAN_PATH = "docs/plans/bayesfilter-dpf-v2-algorithm-full-bf-filterflow-comparison-p3-bootstrap-ot-values-subplan-2026-06-07.md"
RESULT_PATH = "docs/plans/bayesfilter-dpf-v2-algorithm-full-bf-filterflow-comparison-p3-bootstrap-ot-values-result-2026-06-07.md"
P0_VISIBLE_JSON_PATH = OUTPUT_DIR / "dpf_v2_algorithm_full_comparison_p0_visible_governance_2026-06-08.json"
P1_JSON_PATH = OUTPUT_DIR / "dpf_v2_algorithm_full_comparison_p1_architecture_2026-06-07.json"
P2_JSON_PATH = OUTPUT_DIR / "dpf_v2_bootstrap_ot_contracts_2026-06-07.json"
JSON_PATH = OUTPUT_DIR / "dpf_v2_bootstrap_ot_values_2026-06-07.json"
REPORT_PATH = REPORT_DIR / "dpf-v2-bootstrap-ot-values-2026-06-07.md"
P2_PASS_DECISION = "PASS_P2_BOOTSTRAP_OT_CONTRACTS_READY_FOR_P3"
P2_BUNDLE_CHECKSUM = "53722eaea627646a85b99a2dde94d733b31e35e3a10886b05df20a733a1df11c"
VALUE_TOLERANCE = 5e-10
LEDGER_TOLERANCE = 5e-10
TRANSPORT_TOLERANCE = 5e-10
EXPECTED_DECISIONS = {
    "PENDING_CLAUDE_REVIEW",
    "PASS_P3_BOOTSTRAP_OT_VALUES_READY_FOR_P4",
    "P3_BOOTSTRAP_OT_VALUES_CLASSIFIED_MISMATCH_PENDING_REVIEW",
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
    _preflight_prior_artifacts(p0_payload, p1_payload, p2_payload)

    specs = common_model_specs_v2()
    contracts = list(p2_payload["contracts"])
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
        "phase": "P3",
        "execution_route": "VISIBLE_IN_DIALOGUE",
        "decision": decision,
        "question": (
            "Do BayesFilter and BayesFilter-owned FilterFlow-side adapters "
            "match bootstrap-OT fixed-branch values and ledgers for all six V2 rows?"
        ),
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "p0_visible_json_path": str(P0_VISIBLE_JSON_PATH.relative_to(REPO_ROOT)),
        "p1_architecture_json_path": str(P1_JSON_PATH.relative_to(REPO_ROOT)),
        "p2_contracts_json_path": str(P2_JSON_PATH.relative_to(REPO_ROOT)),
        "p2_contract_bundle_checksum": p2_payload["contract_bundle_checksum"],
        "required_v2_model_ids": list(EXPECTED_V2_MODEL_IDS),
        "tolerances": {
            "value_abs": VALUE_TOLERANCE,
            "ledger_abs": LEDGER_TOLERANCE,
            "transport_abs": TRANSPORT_TOLERANCE,
        },
        "adapter_policy": {
            "bayesfilter_adapter": "BayesFilter model methods from bayesfilter_model_for_spec_v2",
            "filterflow_side_adapter": (
                "BayesFilter-owned contract-formula adapter plus shared "
                "filterflow_style_annealed_transport_tf component"
            ),
            "filterflow_checkout_mutated": False,
            "filterflow_checkout_executed": False,
            "neither_side_is_oracle": True,
        },
        "primary_criterion_fields": {
            "primary_ledger_fields": _primary_fields(),
            "transport_fields": _transport_fields(),
            "all_rows_matched": all(cell["status"] == "MATCHED" for cell in cells),
            "all_contract_checksums_preserved": all(
                cell["contract"]["contract_checksum"] == cell["bayesfilter"]["contract_checksum"]
                == cell["filterflow"]["contract_checksum"]
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
            "p2_contract_checksum_changed": p2_payload["contract_bundle_checksum"] != P2_BUNDLE_CHECKSUM,
            "runtime_branch_mask_differs_from_p2": any(
                not cell["metrics"].get("fixed_mask_matches_contract", False)
                for cell in cells
            ),
            "transport_settings_differ_from_p2": any(
                not cell["metrics"].get("transport_settings_match_contract", False)
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
            "unclassified_ledger_mismatch": any(
                cell["status"] not in {"MATCHED", "EXPLAINED_MISMATCH"} for cell in cells
            ),
            "localsource_filterflow_mutated": False,
            "student_command_or_metric": False,
            "oracle_framing": False,
            "finite_difference_promoted_to_gradient_gate": False,
        },
        "explanatory_only_fields": {
            "status_counts": _status_counts(cells),
            "max_abs_delta": _max_abs_delta(cells),
            "ess_policy": "ESS is reported as ledger context only; fixed P2 masks are the branch source.",
            "runtime_policy": "runtime is explanatory only",
            "transport_residual_policy": "transport residuals can veto nonfinite or mismatched ledger fields but do not promote correctness",
        },
        "cells": cells,
        "summary": _summary(cells),
        "review_round": 0,
        "open_material_blockers": [],
        "repair_amendment_required": False,
        "next_allowed_action": "run chunked Claude P3 read-only review before P4",
        "artifact_paths": {
            "json": str(JSON_PATH.relative_to(REPO_ROOT)),
            "markdown_report": str(REPORT_PATH.relative_to(REPO_ROOT)),
            "phase_result": RESULT_PATH,
        },
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners.run_v2_bootstrap_ot_values_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_claims": [
            "P3 validates deterministic fixed-branch bootstrap-OT value agreement only.",
            "P3 does not establish bootstrap-OT gradient agreement.",
            "P3 does not establish stochastic resampling distribution correctness.",
            "P3 does not prove BayesFilter or FilterFlow correctness.",
            "P3 does not make a student implementation claim.",
            "P3 does not make a GPU, scalability, deployment, or production-readiness claim.",
        ],
    }


def _run_bayesfilter_adapter(spec: CommonModelSpecV2, contract: dict[str, Any]) -> dict[str, Any]:
    model = bayesfilter_model_for_spec_v2(spec)

    def transition_mean(previous: tf.Tensor) -> tf.Tensor:
        return _bayesfilter_transition_mean(spec, model, tf.convert_to_tensor(contract["theta"], DTYPE), previous)

    def transition_log_density(previous: tf.Tensor, predicted: tf.Tensor, step: int) -> tf.Tensor:
        return model.transition_log_density(
            tf.convert_to_tensor(contract["theta"], DTYPE),
            previous,
            predicted,
            t=step + 1,
        )

    def observation_log_density(predicted: tf.Tensor, observation: tf.Tensor, step: int) -> tf.Tensor:
        return model.observation_log_density(
            tf.convert_to_tensor(contract["theta"], DTYPE),
            predicted,
            observation,
            t=step + 1,
        )

    return _run_contract(
        contract,
        backend="bayesfilter_v2_bootstrap_ot_fixed_branch",
        transition_mean=transition_mean,
        transition_log_density=transition_log_density,
        observation_log_density=observation_log_density,
        complete_structural=(
            lambda previous, predicted: model.complete_next_state(previous, predicted[:, 0])
            if spec.model_id == "structural_ar1_quadratic_h16"
            else predicted
        ),
    )


def _run_filterflow_side_adapter(contract: dict[str, Any]) -> dict[str, Any]:
    return _run_contract(
        contract,
        backend="filterflow_side_v2_bootstrap_ot_contract_formula_adapter",
        transition_mean=lambda previous: _contract_transition_mean(contract, previous),
        transition_log_density=lambda previous, predicted, step: _contract_transition_log_density(
            contract, previous, predicted, step
        ),
        observation_log_density=lambda predicted, observation, step: _contract_observation_log_density(
            contract, predicted, observation, step
        ),
        complete_structural=lambda previous, predicted: (
            _contract_complete_structural(contract, previous, predicted)
            if contract["model_id"] == "structural_ar1_quadratic_h16"
            else predicted
        ),
    )


def _run_contract(
    contract: dict[str, Any],
    *,
    backend: str,
    transition_mean: Any,
    transition_log_density: Any,
    observation_log_density: Any,
    complete_structural: Any,
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
        pre_update_particles = particles
        pre_update_log_weights = log_weights
        predicted_particles = transition_mean(pre_update_particles) + innovations[step]
        predicted_particles = complete_structural(pre_update_particles, predicted_particles)
        transition_density = transition_log_density(pre_update_particles, predicted_particles, step)
        observation_density = observation_log_density(predicted_particles, observations[step], step)
        unnormalized = log_weights + observation_density
        increment = tf.reduce_logsumexp(unnormalized)
        total_scalar = total_scalar + increment
        log_weights = unnormalized - increment
        weights = tf.exp(log_weights)
        ess = _ess_from_log_weights(log_weights)
        filtered_mean, filtered_variance = _weighted_mean_variance(predicted_particles, weights)
        resampling_applied = bool(fixed_mask[step])
        if resampling_applied:
            transport = annealed_transport_resample_tf(
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
            post_particles = tf.cast(transport.particles, DTYPE)
            post_log_weights = tf.cast(transport.log_weights, DTYPE)
            transport_matrix = tf.cast(transport.transport_matrix, DTYPE)
            transport_diagnostics = dict(transport.diagnostics)
            resampling_count += 1
        else:
            post_particles = predicted_particles
            post_log_weights = log_weights
            transport_matrix = tf.zeros([n_particles, n_particles], DTYPE)
            transport_diagnostics = _no_transport_diagnostics(contract)
        ledger.append(
            _step_ledger(
                step=step,
                fixed_ess_trigger=resampling_applied,
                pre_update_particles=pre_update_particles,
                pre_update_log_weights=pre_update_log_weights,
                predicted_particles=predicted_particles,
                transition_innovations=innovations[step],
                observation=observations[step],
                transition_log_density=transition_density,
                observation_log_density=observation_density,
                unnormalized_log_weights=unnormalized,
                incremental_log_normalizer=increment,
                normalized_log_weights=log_weights,
                weights=weights,
                ess=ess,
                filtered_mean=filtered_mean,
                filtered_variance=filtered_variance,
                resampling_applied=resampling_applied,
                transport_matrix=transport_matrix,
                transport_diagnostics=transport_diagnostics,
                post_transport_particles=post_particles,
                post_transport_log_weights=post_log_weights,
            )
        )
        particles = post_particles
        log_weights = post_log_weights

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


def _contract_transition_mean(contract: dict[str, Any], particles: tf.Tensor) -> tf.Tensor:
    model_id = str(contract["model_id"])
    p = contract["parameters"]
    particles = tf.convert_to_tensor(particles, DTYPE)
    if model_id in {"lgssm_2d_h25_rich", "range_bearing_4d_h20_rich"}:
        return tf.linalg.matmul(particles, _tensor(p["A"]), transpose_b=True)
    if model_id == "sv_1d_h18_rich":
        mu = _tensor(p["mu"])
        phi = _tensor(p["phi"])
        return mu + phi * (particles - mu)
    if model_id == "structural_ar1_quadratic_h16":
        current_m = _tensor(p["rho"]) * particles[:, 0]
        current_k = _structural_complete_k(p, particles[:, 1], particles[:, 0], current_m)
        return tf.stack([current_m, current_k], axis=1)
    if model_id == "spatial_sir_j3_rk4":
        return _sir_transition_mean(particles, p)
    if model_id == "predator_prey_rk4":
        return _predator_transition_mean(_tensor(contract["theta"]), particles, p)
    raise ValueError(f"unknown model id {model_id}")


def _contract_complete_structural(
    contract: dict[str, Any],
    previous: tf.Tensor,
    predicted: tf.Tensor,
) -> tf.Tensor:
    p = contract["parameters"]
    previous = tf.convert_to_tensor(previous, DTYPE)
    predicted = tf.convert_to_tensor(predicted, DTYPE)
    current_m = predicted[:, 0]
    current_k = _structural_complete_k(p, previous[:, 1], previous[:, 0], current_m)
    return tf.stack([current_m, current_k], axis=1)


def _contract_transition_log_density(
    contract: dict[str, Any],
    previous: tf.Tensor,
    predicted: tf.Tensor,
    step: int,
) -> tf.Tensor:
    del step
    model_id = str(contract["model_id"])
    p = contract["parameters"]
    previous = tf.convert_to_tensor(previous, DTYPE)
    predicted = tf.convert_to_tensor(predicted, DTYPE)
    if model_id == "lgssm_2d_h25_rich":
        return _mvn_log_prob(predicted, _contract_transition_mean(contract, previous), _tensor(p["Q"]))
    if model_id == "sv_1d_h18_rich":
        return _normal_log_prob(predicted[:, 0], _contract_transition_mean(contract, previous)[:, 0], _tensor(p["sigma"]))
    if model_id == "range_bearing_4d_h20_rich":
        return _mvn_log_prob(predicted, _contract_transition_mean(contract, previous), _tensor(p["Q"]))
    if model_id == "structural_ar1_quadratic_h16":
        return _normal_log_prob(predicted[:, 0], _tensor(p["rho"]) * previous[:, 0], _tensor(p["sigma"]))
    if model_id == "spatial_sir_j3_rk4":
        return _mvn_log_prob(predicted, _sir_transition_mean(previous, p), _tensor(p["process_covariance"]))
    if model_id == "predator_prey_rk4":
        return _mvn_log_prob(
            predicted,
            _predator_transition_mean(_tensor(contract["theta"]), previous, p),
            _tensor(p["process_covariance"]),
        )
    raise ValueError(f"unknown model id {model_id}")


def _contract_observation_log_density(
    contract: dict[str, Any],
    particles: tf.Tensor,
    observation: tf.Tensor,
    step: int,
) -> tf.Tensor:
    del step
    model_id = str(contract["model_id"])
    p = contract["parameters"]
    particles = tf.convert_to_tensor(particles, DTYPE)
    observation = tf.reshape(tf.convert_to_tensor(observation, DTYPE), [-1])
    if model_id == "lgssm_2d_h25_rich":
        loc = tf.linalg.matmul(particles, _tensor(p["C"]), transpose_b=True)
        return _mvn_log_prob(tf.broadcast_to(observation, tf.shape(loc)), loc, _tensor(p["R"]))
    if model_id == "sv_1d_h18_rich":
        scale = tf.exp(0.5 * particles[:, 0])
        return _normal_log_prob(tf.broadcast_to(observation[0], tf.shape(scale)), 0.0, scale)
    if model_id == "range_bearing_4d_h20_rich":
        predicted = _range_bearing_observation(particles)
        residual = observation[tf.newaxis, :] - predicted
        residual = tf.concat([residual[..., :1], _wrap_angle(residual[..., 1:2])], axis=-1)
        return _gaussian_logpdf_zero_mean(residual, _tensor(p["R"]))
    if model_id == "structural_ar1_quadratic_h16":
        mean = particles[:, 1] + _tensor(p["lambda"]) * particles[:, 0]
        return _normal_log_prob(tf.broadcast_to(observation[0], tf.shape(mean)), mean, _tensor(p["observation_scale"]))
    if model_id == "spatial_sir_j3_rk4":
        infectious = particles[:, 1::2]
        return _mvn_log_prob(
            tf.broadcast_to(observation, tf.shape(infectious)),
            infectious,
            _tensor(p["observation_covariance"]),
        )
    if model_id == "predator_prey_rk4":
        return _mvn_log_prob(
            tf.broadcast_to(observation, tf.shape(particles)),
            particles,
            _tensor(p["observation_covariance"]),
        )
    raise ValueError(f"unknown model id {model_id}")


def _sir_transition_mean(x_prev: tf.Tensor, parameters: dict[str, Any]) -> tf.Tensor:
    state = tf.convert_to_tensor(x_prev, DTYPE)
    kappa = _tensor(parameters["kappa"])
    nu = _tensor(parameters["nu"])
    neighbor_sets = [tuple(row) for row in parameters["neighbor_sets"]]
    adjacency = tf.constant(
        [[1.0 if j in row else 0.0 for j in range(len(neighbor_sets))] for row in neighbor_sets],
        dtype=DTYPE,
    )
    degree = tf.reduce_sum(adjacency, axis=1)
    substeps = int(parameters["rk4_substeps"])
    step = _tensor(parameters["delta"]) / tf.cast(substeps, DTYPE)

    def rhs(values: tf.Tensor) -> tf.Tensor:
        s = values[:, 0::2]
        i = values[:, 1::2]
        s_neighbor = tf.linalg.matmul(s, adjacency, transpose_b=True) - s * degree[tf.newaxis, :]
        i_neighbor = tf.linalg.matmul(i, adjacency, transpose_b=True) - i * degree[tf.newaxis, :]
        infection = kappa[tf.newaxis, :] * s * i
        ds = -infection + 0.5 * s_neighbor
        di = infection - nu[tf.newaxis, :] * i + 0.5 * i_neighbor
        return tf.reshape(tf.stack([ds, di], axis=2), [tf.shape(values)[0], tf.shape(values)[1]])

    for _ in range(substeps):
        k1 = rhs(state)
        k2 = rhs(state + 0.5 * step * k1)
        k3 = rhs(state + 0.5 * step * k2)
        k4 = rhs(state + step * k3)
        state = state + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return state


def _predator_transition_mean(theta: tf.Tensor, x_prev: tf.Tensor, parameters: dict[str, Any]) -> tf.Tensor:
    state = tf.convert_to_tensor(x_prev, DTYPE)
    r, k_capacity, a_half, s_rate, u_rate, v_rate = tf.unstack(tf.convert_to_tensor(theta, DTYPE))
    substeps = int(parameters["rk4_substeps"])
    step = _tensor(parameters["delta"]) / tf.cast(substeps, DTYPE)

    def rhs(values: tf.Tensor) -> tf.Tensor:
        prey = values[:, 0]
        predator = values[:, 1]
        interaction = prey * predator / (a_half + prey)
        d_prey = r * prey * (1.0 - prey / k_capacity) - s_rate * interaction
        d_predator = u_rate * interaction - v_rate * predator
        return tf.stack([d_prey, d_predator], axis=1)

    for _ in range(substeps):
        k1 = rhs(state)
        k2 = rhs(state + 0.5 * step * k1)
        k3 = rhs(state + 0.5 * step * k2)
        k4 = rhs(state + step * k3)
        state = state + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return state


def _structural_complete_k(
    parameters: dict[str, Any],
    previous_k: tf.Tensor,
    previous_m: tf.Tensor,
    current_m: tf.Tensor,
) -> tf.Tensor:
    return (
        _tensor(parameters["a"]) * previous_k
        + _tensor(parameters["b"]) * current_m
        + _tensor(parameters["c"]) * current_m * current_m
        + _tensor(parameters["d"]) * previous_m * current_m
    )


def _mvn_log_prob(values: tf.Tensor, loc: tf.Tensor, covariance: tf.Tensor) -> tf.Tensor:
    return tfd.MultivariateNormalTriL(
        loc=tf.convert_to_tensor(loc, DTYPE),
        scale_tril=tf.linalg.cholesky(tf.convert_to_tensor(covariance, DTYPE)),
    ).log_prob(tf.convert_to_tensor(values, DTYPE))


def _normal_log_prob(value: tf.Tensor, loc: tf.Tensor | float, scale: tf.Tensor | float) -> tf.Tensor:
    return tfd.Normal(loc=tf.convert_to_tensor(loc, DTYPE), scale=tf.convert_to_tensor(scale, DTYPE)).log_prob(
        tf.convert_to_tensor(value, DTYPE)
    )


def _range_bearing_observation(x: tf.Tensor) -> tf.Tensor:
    values = tf.convert_to_tensor(x, DTYPE)
    px = values[..., 0]
    py = values[..., 1]
    rng = tf.sqrt(px * px + py * py + tf.constant(1e-12, DTYPE))
    bearing = tf.atan2(py, px)
    return tf.stack([rng, bearing], axis=-1)


def _wrap_angle(value: tf.Tensor) -> tf.Tensor:
    pi = tf.constant(math.pi, DTYPE)
    return tf.math.floormod(tf.convert_to_tensor(value, DTYPE) + pi, 2.0 * pi) - pi


def _gaussian_logpdf_zero_mean(residuals: tf.Tensor, covariance: tf.Tensor) -> tf.Tensor:
    residuals = tf.convert_to_tensor(residuals, DTYPE)
    covariance = tf.convert_to_tensor(covariance, DTYPE)
    chol = tf.linalg.cholesky(covariance)
    solved = tf.linalg.cholesky_solve(chol, tf.transpose(residuals))
    quad = tf.reduce_sum(tf.transpose(solved) * residuals, axis=1)
    dim = tf.cast(tf.shape(covariance)[0], DTYPE)
    logdet = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)))
    return -0.5 * (dim * tf.math.log(tf.constant(2.0 * math.pi, DTYPE)) + logdet + quad)


def _step_ledger(
    *,
    step: int,
    fixed_ess_trigger: bool,
    pre_update_particles: tf.Tensor,
    pre_update_log_weights: tf.Tensor,
    predicted_particles: tf.Tensor,
    transition_innovations: tf.Tensor,
    observation: tf.Tensor,
    transition_log_density: tf.Tensor,
    observation_log_density: tf.Tensor,
    unnormalized_log_weights: tf.Tensor,
    incremental_log_normalizer: tf.Tensor,
    normalized_log_weights: tf.Tensor,
    weights: tf.Tensor,
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
    return {
        "step": int(step),
        "fixed_ess_trigger": bool(fixed_ess_trigger),
        "pre_update_particles": tensor_to_json(pre_update_particles),
        "pre_update_log_weights": tensor_to_json(pre_update_log_weights),
        "pre_update_weights": tensor_to_json(tf.exp(pre_update_log_weights)),
        "predicted_particles": tensor_to_json(predicted_particles),
        "transition_innovations": tensor_to_json(transition_innovations),
        "observation": tensor_to_json(observation),
        "transition_log_density": tensor_to_json(transition_log_density),
        "observation_log_density": tensor_to_json(observation_log_density),
        "unnormalized_log_weights": tensor_to_json(unnormalized_log_weights),
        "incremental_log_normalizer": scalar(incremental_log_normalizer),
        "normalized_log_weights": tensor_to_json(normalized_log_weights),
        "weights": tensor_to_json(weights),
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
    }


def _transport_summary(matrix: tf.Tensor) -> dict[str, Any]:
    matrix = tf.convert_to_tensor(matrix, DTYPE)
    if int(tf.size(matrix).numpy()) == 0:
        return {
            "shape": [int(dim) for dim in matrix.shape],
            "max_abs": 0.0,
            "row_sum": [],
            "column_sum": [],
        }
    return {
        "shape": [int(dim) for dim in matrix.shape],
        "max_abs": scalar(tf.reduce_max(tf.abs(matrix))),
        "row_sum": tensor_to_json(tf.reduce_sum(matrix, axis=1)),
        "column_sum": tensor_to_json(tf.reduce_sum(matrix, axis=0)),
    }


def _no_transport_diagnostics(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "component_id": "filterflow_style_annealed_transport_tf",
        "reference_algorithm": "canonical_executable_filterflow_regularised_transform",
        "resampling_status": "not_triggered_by_fixed_p2_mask",
        "epsilon": float(contract["ot_settings"]["sinkhorn_epsilon"]),
        "scaling": float(contract["ot_settings"]["annealed_scaling"]),
        "convergence_threshold": float(contract["ot_settings"]["annealed_convergence_threshold"]),
        "max_iterations": int(contract["ot_settings"]["sinkhorn_iterations"]),
        "transport_gradient_mode": str(contract["ot_settings"]["transport_gradient_mode"]),
        "application_mode": str(contract["ot_settings"]["application_mode"]),
        "triggered_rows": 0.0,
        "skipped_rows": 1.0,
        "max_row_residual": 0.0,
        "max_column_residual": 0.0,
        "finite_transport": True,
        "finite_particles": True,
        "backend": "tensorflow",
    }


def _cell(
    spec: CommonModelSpecV2,
    contract: dict[str, Any],
    bayesfilter: dict[str, Any],
    filterflow: dict[str, Any],
) -> dict[str, Any]:
    comparison = _compare_path_payloads(bayesfilter, filterflow, contract)
    matched = (
        comparison["all_primary_fields_within_tolerance"]
        and comparison["fixed_mask_matches_contract"]
        and comparison["transport_settings_match_contract"]
        and bayesfilter["finite"]
        and filterflow["finite"]
        and bayesfilter["resampling_count"] == int(contract["fixed_ess_trigger_count"])
        and filterflow["resampling_count"] == int(contract["fixed_ess_trigger_count"])
    )
    return {
        "model": spec.model_id,
        "family": spec.family,
        "implementations": ["BayesFilter", "FilterFlow-side adapter"],
        "cell_type": "v2_bootstrap_ot_fixed_branch_value_path",
        "status": "MATCHED" if matched else "EXPLAINED_MISMATCH",
        "decision": f"{spec.model_id}_bootstrap_ot_values_matched" if matched else f"{spec.model_id}_bootstrap_ot_values_mismatch",
        "primary_criterion": "scalar and required bootstrap-OT ledger fields match within tolerance",
        "metrics": comparison,
        "mismatch_class": None if matched else "v2_bootstrap_ot_value_or_ledger_delta",
        "contract": _contract_digest_view(contract),
        "bayesfilter": bayesfilter,
        "filterflow": filterflow,
        "spec_checksum": spec.checksum(),
        "non_claim": "fixed-branch bootstrap-OT value agreement is not gradient or stochastic resampling correctness",
    }


def _compare_path_payloads(left: dict[str, Any], right: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    fields = _primary_fields()
    ledger_deltas = []
    max_abs_delta = abs(float(left["scalar"]) - float(right["scalar"]))
    all_within = max_abs_delta <= VALUE_TOLERANCE
    masks_left = []
    masks_right = []
    transport_settings_match = True
    for left_step, right_step in zip(left["ledger"], right["ledger"], strict=True):
        masks_left.append(bool(left_step["fixed_ess_trigger"]))
        masks_right.append(bool(right_step["fixed_ess_trigger"]))
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
        "fixed_mask_matches_contract": masks_left == contract_mask and masks_right == contract_mask,
        "bayesfilter_fixed_mask": masks_left,
        "filterflow_fixed_mask": masks_right,
        "contract_fixed_mask": contract_mask,
        "transport_settings_match_contract": bool(transport_settings_match),
        "value_tolerance": VALUE_TOLERANCE,
        "ledger_tolerance": LEDGER_TOLERANCE,
        "transport_tolerance": TRANSPORT_TOLERANCE,
        "explanatory_fields": ["ess", "filtered_mean", "filtered_variance", "transport_residuals"],
    }


def _primary_fields() -> list[str]:
    return [
        "pre_update_particles",
        "predicted_particles",
        "observation_log_density",
        "unnormalized_log_weights",
        "incremental_log_normalizer",
        "normalized_log_weights",
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
    return LEDGER_TOLERANCE


def _contract_digest_view(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "contract_id": contract["contract_id"],
        "model_id": contract["model_id"],
        "algorithm": contract["algorithm"],
        "horizon": contract["horizon"],
        "num_particles": contract["num_particles"],
        "fixed_ess_trigger_mask": contract["fixed_ess_trigger_mask"],
        "fixed_ess_trigger_count": contract["fixed_ess_trigger_count"],
        "ot_settings": contract["ot_settings"],
        "scalar_definition": contract["scalar_definition"],
        "contract_checksum": contract["contract_checksum"],
        "component_checksums": contract["component_checksums"],
    }


def _preflight_prior_artifacts(
    p0_payload: dict[str, Any],
    p1_payload: dict[str, Any],
    p2_payload: dict[str, Any],
) -> None:
    if p0_payload.get("decision") != "PASS_P0_READY_FOR_P1":
        raise ValueError(f"P0 visible governance is not passed: {p0_payload.get('decision')}")
    if p1_payload.get("decision") != "PASS_P1_ARCHITECTURE_READY_FOR_P2":
        raise ValueError(f"P1 architecture is not passed: {p1_payload.get('decision')}")
    if p2_payload.get("decision") != P2_PASS_DECISION:
        raise ValueError(f"P2 bootstrap-OT contracts are not passed: {p2_payload.get('decision')}")
    if p2_payload.get("contract_bundle_checksum") != P2_BUNDLE_CHECKSUM:
        raise ValueError("P2 bootstrap-OT contract bundle checksum changed")
    if tuple(p0_payload.get("required_v2_model_ids", [])) != EXPECTED_V2_MODEL_IDS:
        raise ValueError("P0 model id gate failed")
    if tuple(p1_payload.get("required_v2_model_ids", [])) != EXPECTED_V2_MODEL_IDS:
        raise ValueError("P1 model id gate failed")
    if tuple(p2_payload.get("required_v2_model_ids", [])) != EXPECTED_V2_MODEL_IDS:
        raise ValueError("P2 required model id gate failed")
    ids = [contract.get("model_id") for contract in p2_payload.get("contracts", [])]
    if tuple(ids) != EXPECTED_V2_MODEL_IDS:
        raise ValueError(f"P2 contract model id gate failed: {ids}")
    if p2_payload.get("execution_diagnostics", {}).get("bootstrap_ot_values_computed"):
        raise ValueError("P2 artifact claims bootstrap-OT value execution")


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
        raise ValueError(f"P3 payload missing required fields: {sorted(missing)}")
    if payload["phase"] != "P3":
        raise ValueError(f"unexpected phase: {payload['phase']}")
    if payload["decision"] not in EXPECTED_DECISIONS:
        raise ValueError(f"P3 payload decision not passable: {payload['decision']}")
    ids = [cell["model"] for cell in payload["cells"]]
    if tuple(ids) != EXPECTED_V2_MODEL_IDS:
        raise ValueError(f"P3 cell id gate failed: {ids}")
    if payload["p2_contract_bundle_checksum"] != P2_BUNDLE_CHECKSUM:
        raise ValueError("P3 payload references changed P2 bundle checksum")
    if payload["artifact_paths"]["json"] != str(JSON_PATH.relative_to(REPO_ROOT)):
        raise ValueError("P3 JSON artifact path mismatch")
    if payload["run_manifest"].get("pre_import_cuda_visible_devices") != "-1":
        raise ValueError("P3 TensorFlow run was not CPU-only before import")
    for key in (
        "missing_v2_row",
        "row_order_mismatch",
        "p2_contract_checksum_changed",
        "runtime_branch_mask_differs_from_p2",
        "transport_settings_differ_from_p2",
        "nonfinite_scalar_or_ledger",
        "unclassified_ledger_mismatch",
        "localsource_filterflow_mutated",
        "student_command_or_metric",
        "oracle_framing",
        "finite_difference_promoted_to_gradient_gate",
    ):
        if payload["veto_diagnostics"].get(key):
            raise ValueError(f"P3 veto diagnostic fired: {key}")
    for cell in payload["cells"]:
        if not cell.get("bayesfilter", {}).get("finite", False):
            raise ValueError(f"P3 BayesFilter nonfinite row: {cell['model']}")
        if not cell.get("filterflow", {}).get("finite", False):
            raise ValueError(f"P3 FilterFlow adapter nonfinite row: {cell['model']}")
    if payload["decision"] in {"PENDING_CLAUDE_REVIEW", "PASS_P3_BOOTSTRAP_OT_VALUES_READY_FOR_P4"}:
        if payload["veto_diagnostics"].get("value_delta_exceeds_tolerance"):
            raise ValueError("P3 value delta exceeds tolerance on passable decision")
        if payload["veto_diagnostics"].get("ledger_delta_exceeds_tolerance"):
            raise ValueError("P3 ledger delta exceeds tolerance on passable decision")
        if not payload["primary_criterion_fields"].get("all_rows_matched"):
            raise ValueError("P3 not all rows matched on passable decision")
        for cell in payload["cells"]:
            if cell["status"] != "MATCHED":
                raise ValueError(f"P3 row did not match on passable decision: {cell['model']} {cell['status']}")
    elif payload["decision"] == "P3_BOOTSTRAP_OT_VALUES_CLASSIFIED_MISMATCH_PENDING_REVIEW":
        if not any(
            cell["status"] == "EXPLAINED_MISMATCH"
            for cell in payload["cells"]
        ):
            raise ValueError("P3 mismatch decision without classified mismatch cell")


def _decision(cells: list[dict[str, Any]]) -> str:
    if all(cell["status"] == "MATCHED" for cell in cells):
        return "PENDING_CLAUDE_REVIEW"
    return "P3_BOOTSTRAP_OT_VALUES_CLASSIFIED_MISMATCH_PENDING_REVIEW"


def _weighted_mean_variance(particles: tf.Tensor, weights: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    particles = tf.convert_to_tensor(particles, DTYPE)
    weights = tf.convert_to_tensor(weights, DTYPE)
    mean = tf.reduce_sum(weights[:, tf.newaxis] * particles, axis=0)
    centered = particles - mean
    variance = tf.reduce_sum(weights[:, tf.newaxis] * centered * centered, axis=0)
    return mean, variance


def _ess_from_log_weights(log_weights: tf.Tensor) -> tf.Tensor:
    weights = tf.exp(tf.convert_to_tensor(log_weights, DTYPE))
    return 1.0 / tf.reduce_sum(tf.square(weights))


def _uniform_log_weights(n_particles: int) -> tf.Tensor:
    return tf.fill([n_particles], -tf.math.log(tf.cast(n_particles, DTYPE)))


def _tensor(value: Any) -> tf.Tensor:
    return tf.convert_to_tensor(value, DTYPE)


def _ledger_finite(ledger: list[dict[str, Any]]) -> bool:
    return all(_json_finite(row) for row in ledger)


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
        "# DPF V2 Algorithm Full Comparison P3 Bootstrap-OT Values Result",
        "",
        "metadata_date: 2026-06-07",
        "visible_execution_timestamp: `2026-06-08T03:26:47+08:00`",
        "phase: P3",
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
        "- for every V2 row, BayesFilter and the BayesFilter-owned FilterFlow-side adapter consume the same frozen P2 bootstrap-OT contract;",
        "- scalar values match within declared tolerance;",
        "- required ledgers match, including initial/pre-update particles, predicted particles, observation log densities, unnormalized and normalized log weights, frozen ESS trigger masks, OT transport matrix checksums, post-transport particles, incremental log normalizers, and final scalar.",
        "",
        "Veto diagnostics:",
        "",
        "- nonfinite scalar or ledger field;",
        "- runtime branch mask differs from frozen P2 mask;",
        "- transport settings differ from P2;",
        "- BF/FF value delta exceeds tolerance;",
        "- unclassified ledger mismatch;",
        "- `.localsource/filterflow` mutation, student command, oracle framing, or finite differences promoted to a gradient gate.",
        "",
        "Non-claims:",
        "",
        "- P3 does not establish bootstrap-OT gradient agreement or stochastic resampling distribution correctness.",
        "",
        "## Local Skeptical Phase Audit",
        "",
        "Audit status: `PASS_LOCAL_PHASE_AUDIT`.",
        "",
        "Wrong-baseline risk: controlled. P3 uses the reviewed P2 contract bundle checksum as the baseline and rejects checksum drift.",
        "",
        "Proxy-metric risk: controlled. ESS, runtime, filtered moments, and transport residuals are explanatory unless they trigger explicit finite or ledger-veto checks.",
        "",
        "Missing stop-condition risk: controlled. P3 stops on row disappearance, mask drift, transport setting drift, value/ledger mismatch, nonfinite values, or stale P2 checksum.",
        "",
        "Unfair-comparison risk: controlled. Both adapters consume identical contract bytes and the same frozen mask.",
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
        f"- P2 contract bundle checksum: `{payload['p2_contract_bundle_checksum']}`",
        f"- Reproducibility digest: `{payload['reproducibility_digest']}`",
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
            "## Command Manifest",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| git commit | `{run_manifest.get('commit')}` |",
            f"| git branch | `{run_manifest.get('branch')}` |",
            f"| dirty status | `{_single_line(run_manifest.get('dirty_state_summary'))}` |",
            f"| command | `{run_manifest.get('command')}` |",
            f"| validation commands | `python -m json.tool {payload['artifact_paths']['json']}`; `CUDA_VISIBLE_DEVICES=-1 python -m experiments.dpf_implementation.tf_tfp.runners.run_v2_bootstrap_ot_values_tf --validate-only`; `git diff --check` on P3 files |",
            f"| environment | `/home/chakwong/BayesFilter`; Python `{run_manifest.get('python_version')}`; TensorFlow `{run_manifest.get('package_versions', {}).get('tensorflow')}`; TFP `{run_manifest.get('package_versions', {}).get('tensorflow_probability')}` |",
            f"| CPU/GPU status | CPU-only TensorFlow run; pre-import `CUDA_VISIBLE_DEVICES={run_manifest.get('pre_import_cuda_visible_devices')}`; visible GPUs `{run_manifest.get('gpu_devices_visible')}` |",
            "| random seeds | no RNG consumed in P3; frozen particles, observations, transition innovations, and masks from P2 |",
            "| dtype | `tf.float64` / JSON dtype `float64` |",
            f"| output artifacts | `{payload['artifact_paths']['json']}`; `{payload['artifact_paths']['markdown_report']}`; `{payload['artifact_paths']['phase_result']}` |",
            "",
            "## Review State",
            "",
            "review_round: 0 pending chunked Claude P3 value review",
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
            f"| `{payload['decision']}` | all six bootstrap-OT value ledgers matched locally | all local P3 veto diagnostics clear | Claude may find adapter or artifact adequacy gaps | run chunked Claude P3 read-only review | no gradient agreement, stochastic resampling correctness, implementation proof, student claim, GPU claim, or production readiness |",
            "",
            "## Post-Run Red Team",
            "",
            "Strongest alternative explanation: because both adapters are BayesFilter-owned, P3 can miss a defect shared by the contract-formula adapter and the BayesFilter model surface.",
            "",
            "What would overturn the local decision: any reviewer finding that the FilterFlow-side adapter is not the P1-frozen adapter surface, that a required ledger field is omitted, that P2 masks/settings changed, or that the same-contract identity is broken.",
            "",
            "Weakest part of the evidence: P3 is fixed-branch value evidence only and does not test gradient correctness or stochastic resampling distribution behavior.",
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


def _single_line(value: object) -> str:
    return str(value).replace("\n", " | ")


if __name__ == "__main__":
    raise SystemExit(main())
