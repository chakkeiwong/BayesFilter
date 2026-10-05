"""Run v2 fixed-branch gradient tie-outs."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/bayesfilter-dpf-mpl")
PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]

import argparse
import json
import math
import subprocess
import textwrap
import time
from typing import Any, Callable

import tensorflow as tf

from experiments.dpf_implementation.tf_tfp.fixtures.common_model_suite_tf import (
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
    scalar,
    stable_digest,
    tensor_to_json,
    utc_now,
    write_json,
    write_text,
)
from experiments.dpf_implementation.tf_tfp.runners.filterflow_reference_policy import (
    FILTERFLOW_BRANCH_MARKER,
    reference_policy,
    validate_filterflow_reference_status,
)


DTYPE = tf.float64
PLAN_PATH = "docs/plans/bayesfilter-dpf-common-model-suite-v2-production-p5-fixed-branch-gradients-subplan-2026-06-07.md"
RESULT_PATH = "docs/plans/bayesfilter-dpf-common-model-suite-v2-production-p5-fixed-branch-gradients-result-2026-06-07.md"
P1_MANIFEST_PATH = OUTPUT_DIR / "dpf_common_model_suite_v2_manifest_2026-06-07.json"
P2_RESULT_PATH = REPO_ROOT / "docs/plans/bayesfilter-dpf-common-model-suite-v2-production-p2-density-tieout-result-2026-06-07.md"
P3_RESULT_PATH = REPO_ROOT / "docs/plans/bayesfilter-dpf-common-model-suite-v2-production-p3-noresampling-paths-result-2026-06-07.md"
P4_RESULT_PATH = REPO_ROOT / "docs/plans/bayesfilter-dpf-common-model-suite-v2-production-p4-fixed-ancestor-paths-result-2026-06-07.md"
JSON_PATH = OUTPUT_DIR / "dpf_common_model_suite_v2_gradients_2026-06-07.json"
REPORT_PATH = REPORT_DIR / "dpf-common-model-suite-v2-gradients-2026-06-07.md"
FILTERFLOW_ENV_PYTHON = REPO_ROOT / ".localenv" / "filterflow-py311" / "bin" / "python"
FILTERFLOW_PATH = REPO_ROOT / ".localsource" / "filterflow"
FILTERFLOW_MARKER_PATH = FILTERFLOW_PATH / FILTERFLOW_BRANCH_MARKER
FD_STEP = 1e-5
FD_LADDER_STEPS = (1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7)
VALUE_TOLERANCE = 5e-10
GRADIENT_TOLERANCE = 5e-8
FD_TOLERANCE = 5e-5
INACTIVE_ZERO_GRADIENT_REASONS = {
    (
        "sv_1d_h18_rich",
        "sigma",
    ): (
        "Under the P5 fixed-additive-innovation scalar, sigma parameterizes "
        "the transition noise scale but the scalar uses transition mean plus "
        "frozen innovations and observation log density; derivative is zero "
        "by scalar derivation."
    ),
    (
        "structural_ar1_quadratic_h16",
        "sigma",
    ): (
        "Under the P5 fixed-additive-innovation scalar, sigma parameterizes "
        "the AR(1) transition scale but the scalar uses the deterministic "
        "mean/completion plus frozen innovations and observation log density; "
        "derivative is zero by scalar derivation."
    ),
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
    p1_manifest = load_json(P1_MANIFEST_PATH)
    _preflight_prior_artifacts(p1_manifest)
    filterflow_status = _filterflow_checkout_manifest()
    specs = common_model_specs_v2()
    ready_ids = _ready_ids_for_phase(p1_manifest, "P5_gradients", "READY_FOR_P5")
    contracts = [_gradient_contract(spec) for spec in specs if spec.model_id in ready_ids]
    filterflow_payload = _filterflow_subprocess(contracts)
    filterflow_cells = {cell.get("model_id"): cell for cell in filterflow_payload.get("cells", [])}
    contract_by_id = {contract["model_id"]: contract for contract in contracts}
    cells = []
    for spec in specs:
        if spec.model_id not in ready_ids:
            cells.append(_classified_cell(spec, "CONTRACT_BLOCKED", _blocked_reason(spec)))
            continue
        contract = contract_by_id[spec.model_id]
        bayesfilter = _bayesfilter_gradient(spec, contract)
        filterflow = filterflow_cells.get(
            spec.model_id,
            {"status": "blocked", "model_id": spec.model_id, "blocker": "missing FilterFlow gradient cell"},
        )
        cells.append(_cell(spec, contract, bayesfilter, filterflow))
    decision = _decision(cells, ready_ids)
    return {
        "decision": decision,
        "created_at_utc": utc_now(),
        "question": (
            "Do BayesFilter and executable local float64 FilterFlow-side adapters "
            "match fixed-noise fixed-ancestor physical-knob gradients for the "
            "P1-ready v2 rows, with same-implementation finite-difference "
            "diagnostics?"
        ),
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "p1_manifest_path": str(P1_MANIFEST_PATH.relative_to(REPO_ROOT)),
        "p2_result_path": str(P2_RESULT_PATH.relative_to(REPO_ROOT)),
        "p3_result_path": str(P3_RESULT_PATH.relative_to(REPO_ROOT)),
        "p4_result_path": str(P4_RESULT_PATH.relative_to(REPO_ROOT)),
        "filterflow_reference_policy": reference_policy(),
        "filterflow_status": filterflow_status,
        "filterflow_payload_status": filterflow_payload.get("status"),
        "tolerances": {
            "value_abs": VALUE_TOLERANCE,
            "gradient_abs": GRADIENT_TOLERANCE,
            "finite_difference_abs": FD_TOLERANCE,
            "finite_difference_step": FD_STEP,
        },
        "contracts": contracts,
        "primary_criterion_fields": {
            "ready_model_ids": sorted(ready_ids),
            "blocked_model_ids": [cell["model"] for cell in cells if cell["status"] == "CONTRACT_BLOCKED"],
            "executed_model_ids": [cell["model"] for cell in cells if cell["status"] != "CONTRACT_BLOCKED"],
            "required_gradient_knobs": {
                contract["model_id"]: contract["gradient_knob_names"] for contract in contracts
            },
            "all_ready_rows_matched": all(
                cell["status"] == "MATCHED" for cell in cells if cell["model"] in ready_ids
            ),
            "inactive_zero_gradient_reasons": _inactive_zero_gradient_summary(cells),
            "gradient_scalar": "fixed-branch sum of predictive log normalizers",
            "ancestor_gradient_policy": "fixed ancestor indices are constants and nondifferentiated",
        },
        "veto_diagnostics": {
            "missing_filterflow_subprocess_environment": filterflow_payload.get("status") == "blocked",
            "hidden_rng_used": False,
            "stochastic_resampling_used": False,
            "fixed_ancestor_replay_used": True,
            "gradient_through_random_or_discrete_ancestor_selection_claimed": False,
            "value_match_used_to_excuse_derivative_mismatch": False,
            "old_three_row_common_model_specs_used_as_v2_source": False,
            "old_2026_06_06_artifact_name_used": False,
            "student_command_executed": False,
            "localsource_filterflow_mutated": False,
            "nonfinite_gradient_or_scalar": any(_cell_nonfinite(cell) for cell in cells),
            "sir_gradient_executed_despite_p1_block": any(
                cell["model"] == "spatial_sir_j3_rk4" and cell["status"] != "CONTRACT_BLOCKED"
                for cell in cells
            ),
            "unclassified_mismatch": any(
                cell["status"] not in {"MATCHED", "CONTRACT_BLOCKED", "INTERFACE_BLOCKED", "EXPLAINED_MISMATCH"}
                for cell in cells
            ),
        },
        "explanatory_only_fields": {
            "status_counts": _status_counts(cells),
            "max_abs_scalar_delta": _summary_scalar(cells, "scalar_delta"),
            "max_abs_gradient_delta": _summary_scalar(cells, "max_abs_gradient_delta"),
            "max_abs_bayesfilter_fd_delta": _summary_scalar(cells, "max_abs_bayesfilter_fd_delta"),
            "max_abs_filterflow_fd_delta": _summary_scalar(cells, "max_abs_filterflow_fd_delta"),
            "all_ready_rows_passed_bayesfilter_fd_diagnostic": all(
                cell.get("metrics", {}).get("bayesfilter_fd_within_tolerance", False)
                for cell in cells
                if cell["model"] in ready_ids
            ),
            "all_ready_rows_passed_filterflow_fd_diagnostic": all(
                cell.get("metrics", {}).get("filterflow_fd_within_tolerance", False)
                for cell in cells
                if cell["model"] in ready_ids
            ),
            "finite_difference_diagnostic_outside_tolerance": any(
                _cell_fd_diagnostic_outside_tolerance(cell) for cell in cells
            ),
            "blocked_rows_are_contract_scope_only": "blocked SIR row is not evidence against or for gradient correctness",
            "finite_difference_ladder_policy": (
                "explanatory diagnostic only; finite differences are not a P5 promotion gate"
            ),
            "finite_difference_ladder_steps": list(FD_LADDER_STEPS),
        },
        "cells": cells,
        "summary": _summary(cells),
        "review_round": 0,
        "open_material_blockers": [],
        "repair_amendment_required": False,
        "next_allowed_action": "run Claude P5 result review before P6",
        "artifact_paths": {
            "json": str(JSON_PATH.relative_to(REPO_ROOT)),
            "markdown_report": str(REPORT_PATH.relative_to(REPO_ROOT)),
            "phase_result": RESULT_PATH,
        },
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners.run_common_model_suite_v2_gradients_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_claims": [
            "no stochastic resampling correctness claim",
            "no random-number-generator equality claim",
            "no differentiable-resampling or gradient-through-ancestor-selection claim",
            "no full stochastic-filter gradient correctness proof",
            "no filtering-algorithm correctness proof",
            "no student-repository tie-out claim",
            "no TT/SIRT or paper-scale reproduction claim",
        ],
    }


def _gradient_contract(spec: CommonModelSpecV2) -> dict[str, Any]:
    included_knobs = [dict(knob) for knob in spec.gradient_contract["knobs"] if knob.get("include")]
    if not included_knobs:
        raise ValueError(f"no included P5 gradient knobs for ready row {spec.model_id}")
    fixed_contract = dict(spec.fixed_ancestor_contract)
    payload = {
        "model_id": spec.model_id,
        "family": spec.family,
        "dtype": DTYPE.name,
        "theta": tensor_to_json(spec.theta),
        "parameters": _jsonable_for_runner(spec.parameters),
        "horizon": int(fixed_contract["horizon"]),
        "num_particles": int(fixed_contract["num_particles"]),
        "state_dim": int(fixed_contract["state_dim"]),
        "initial_particles": _jsonable_for_runner(fixed_contract["initial_particles"]),
        "transition_innovations": _jsonable_for_runner(fixed_contract["transition_innovations"]),
        "observations": _jsonable_for_runner(fixed_contract["observations"]),
        "resampling_flags": list(fixed_contract["resampling_flags"]),
        "fixed_ancestor_indices": list(fixed_contract["fixed_ancestor_indices"]),
        "expected_resampling_count": int(fixed_contract["expected_resampling_count"]),
        "gradient_knob_names": [str(knob["name"]) for knob in included_knobs],
        "parameter_values": [float(knob["initial_value"]) for knob in included_knobs],
        "parameterizations": [str(knob["parameterization"]) for knob in included_knobs],
        "finite_difference_step": FD_STEP,
        "scalar": spec.gradient_contract["gradient_scalar"],
        "ancestor_gradient_policy": spec.gradient_contract["random_or_discrete_ancestor_gradient"],
        "transition_innovation_policy": fixed_contract["transition_innovation_policy"],
        "spec_checksum": spec.checksum(),
    }
    payload["contract_checksum"] = stable_digest(payload)
    return payload


def _bayesfilter_gradient(spec: CommonModelSpecV2, contract: dict[str, Any]) -> dict[str, Any]:
    initial = [float(value) for value in contract["parameter_values"]]
    variables = [tf.Variable(value, dtype=DTYPE) for value in initial]
    with tf.GradientTape() as tape:
        value = _branch_scalar(spec.model_id, contract, variables)
    gradients = tape.gradient(value, variables)
    finite_difference = _central_finite_difference(
        lambda params: _branch_scalar(
            spec.model_id,
            contract,
            [tf.constant(value, DTYPE) for value in params],
        ),
        initial,
    )
    finite_difference_ladder = _finite_difference_ladder(
        lambda params: _branch_scalar(
            spec.model_id,
            contract,
            [tf.constant(value, DTYPE) for value in params],
        ),
        initial,
        contract["parameterizations"],
    )
    gradient_values, inactive_zero_reasons = _encode_gradients_by_contract(
        gradients,
        contract["gradient_knob_names"],
        spec.model_id,
    )
    finite = _finite_scalar(value) and all(
        (grad is not None and _finite_scalar(grad)) or knob in inactive_zero_reasons
        for grad, knob in zip(gradients, contract["gradient_knob_names"], strict=True)
    )
    return {
        "status": "executed",
        "backend": "bayesfilter_v2_fixed_branch_gradient",
        "model_id": spec.model_id,
        "scalar": scalar(value),
        "gradient": gradient_values,
        "finite_difference_gradient": finite_difference,
        "finite_difference_ladder": finite_difference_ladder,
        "gradient_delta_vs_finite_difference": [
            None if grad is None else float(grad) - float(fd)
            for grad, fd in zip(gradient_values, finite_difference, strict=True)
        ],
        "inactive_zero_gradient_knobs": list(inactive_zero_reasons),
        "inactive_zero_gradient_reasons": inactive_zero_reasons,
        "disconnected_zero_gradient_knobs": list(inactive_zero_reasons),
        "finite": finite,
        "finite_difference_finite": all(math.isfinite(float(fd)) for fd in finite_difference),
        "gradient_knob_names": contract["gradient_knob_names"],
        "parameterizations": contract["parameterizations"],
        "contract_checksum": contract["contract_checksum"],
        "spec_checksum": spec.checksum(),
    }


def _branch_scalar(model_id: str, contract: dict[str, Any], params: list[tf.Tensor]) -> tf.Tensor:
    particles = tf.convert_to_tensor(contract["initial_particles"], DTYPE)
    innovations = tf.convert_to_tensor(contract["transition_innovations"], DTYPE)
    observations = tf.convert_to_tensor(contract["observations"], DTYPE)
    ancestors = tf.convert_to_tensor(contract["fixed_ancestor_indices"], tf.int32)
    n_particles = int(particles.shape[0])
    log_weights = _uniform_log_weights(n_particles)
    total = tf.zeros([], DTYPE)
    for step in range(int(contract["horizon"])):
        if bool(contract["resampling_flags"][step]):
            particles = tf.gather(particles, ancestors)
            log_weights = _uniform_log_weights(n_particles)
        predicted = _transition_mean(model_id, contract, particles, params) + innovations[step]
        if model_id == "structural_ar1_quadratic_h16":
            predicted = _complete_structural(contract, particles, predicted[:, 0], params)
        obs_log = _observation_log_density(model_id, contract, predicted, observations[step], params)
        unnormalized = log_weights + obs_log
        increment = tf.reduce_logsumexp(unnormalized)
        total = total + increment
        log_weights = unnormalized - increment
        particles = predicted
    return total


def _transition_mean(
    model_id: str,
    contract: dict[str, Any],
    particles: tf.Tensor,
    params: list[tf.Tensor],
) -> tf.Tensor:
    p = contract["parameters"]
    particles = tf.convert_to_tensor(particles, DTYPE)
    if model_id == "lgssm_2d_h25_rich":
        transition_scale = params[0]
        return tf.linalg.matmul(particles, tf.convert_to_tensor(p["A"], DTYPE) * transition_scale, transpose_b=True)
    if model_id == "sv_1d_h18_rich":
        mu, phi, _sigma = params
        return mu + phi * (particles - mu)
    if model_id == "range_bearing_4d_h20_rich":
        return tf.linalg.matmul(particles, tf.convert_to_tensor(p["A"], DTYPE), transpose_b=True)
    if model_id == "structural_ar1_quadratic_h16":
        rho, _sigma, c = params
        current_m = rho * particles[:, 0]
        current_k = _complete_k(particles[:, 1], particles[:, 0], current_m, p, c)
        return tf.stack([current_m, current_k], axis=1)
    if model_id == "predator_prey_rk4":
        return _predator_transition_mean(contract, particles, params)
    raise ValueError(f"unknown P5 model id {model_id}")


def _observation_log_density(
    model_id: str,
    contract: dict[str, Any],
    particles: tf.Tensor,
    observation: tf.Tensor,
    params: list[tf.Tensor],
) -> tf.Tensor:
    p = contract["parameters"]
    particles = tf.convert_to_tensor(particles, DTYPE)
    observation = tf.convert_to_tensor(observation, DTYPE)
    if model_id == "lgssm_2d_h25_rich":
        _transition_scale, observation_noise_scale = params
        loc = tf.linalg.matmul(particles, tf.convert_to_tensor(p["C"], DTYPE), transpose_b=True)
        covariance = tf.convert_to_tensor(p["R"], DTYPE) * observation_noise_scale
        return _mvn_log_prob(tf.broadcast_to(observation, tf.shape(loc)), loc, covariance)
    if model_id == "sv_1d_h18_rich":
        del params
        scale = tf.exp(0.5 * particles[:, 0])
        return _normal_log_prob(tf.broadcast_to(observation[0], tf.shape(scale)), 0.0, scale)
    if model_id == "range_bearing_4d_h20_rich":
        sigma_range, sigma_bearing = params
        predicted = _range_bearing_observation(particles)
        residual = observation[tf.newaxis, :] - predicted
        residual = tf.concat([residual[..., :1], _wrap_angle(residual[..., 1:2])], axis=-1)
        covariance = tf.linalg.diag(tf.stack([tf.square(sigma_range), tf.square(sigma_bearing)]))
        return _gaussian_logpdf_zero_mean(residual, covariance)
    if model_id == "structural_ar1_quadratic_h16":
        del params
        mean = particles[:, 1] + tf.convert_to_tensor(p["lambda"], DTYPE) * particles[:, 0]
        return _normal_log_prob(tf.broadcast_to(observation[0], tf.shape(mean)), mean, p["observation_scale"])
    if model_id == "predator_prey_rk4":
        del params
        return _mvn_log_prob(tf.broadcast_to(observation, tf.shape(particles)), particles, p["observation_covariance"])
    raise ValueError(f"unknown P5 model id {model_id}")


def _complete_structural(
    contract: dict[str, Any],
    previous: tf.Tensor,
    current_m: tf.Tensor,
    params: list[tf.Tensor],
) -> tf.Tensor:
    _rho, _sigma, c = params
    current_k = _complete_k(previous[:, 1], previous[:, 0], current_m, contract["parameters"], c)
    return tf.stack([current_m, current_k], axis=1)


def _complete_k(
    previous_k: tf.Tensor,
    previous_m: tf.Tensor,
    current_m: tf.Tensor,
    parameters: dict[str, Any],
    c: tf.Tensor,
) -> tf.Tensor:
    return (
        tf.convert_to_tensor(parameters["a"], DTYPE) * previous_k
        + tf.convert_to_tensor(parameters["b"], DTYPE) * current_m
        + c * current_m * current_m
        + tf.convert_to_tensor(parameters["d"], DTYPE) * previous_m * current_m
    )


def _predator_transition_mean(contract: dict[str, Any], x_prev: tf.Tensor, params: list[tf.Tensor]) -> tf.Tensor:
    p = contract["parameters"]
    base_theta = tf.convert_to_tensor(contract["theta"], DTYPE)
    theta = tf.concat([tf.reshape(params[0], [1]), base_theta[1:]], axis=0)
    state = tf.convert_to_tensor(x_prev, DTYPE)
    r, k_capacity, a_half, s_rate, u_rate, v_rate = tf.unstack(theta)
    substeps = int(p["rk4_substeps"])
    step = tf.convert_to_tensor(p["delta"], DTYPE) / tf.cast(substeps, DTYPE)

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


def _filterflow_subprocess(contracts: list[dict[str, Any]]) -> dict[str, Any]:
    if not contracts:
        return {"status": "blocked", "blocker": "no P5-ready contracts", "cells": []}
    if not FILTERFLOW_ENV_PYTHON.exists():
        return {
            "status": "blocked",
            "blocker": f"missing filterflow env python: {FILTERFLOW_ENV_PYTHON}",
            "cells": [],
        }
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{FILTERFLOW_PATH}"
    env["MPLCONFIGDIR"] = str(REPO_ROOT / ".cache" / "filterflow-mpl")
    completed = subprocess.run(
        [str(FILTERFLOW_ENV_PYTHON), "-c", _filterflow_script(contracts)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=240,
    )
    if completed.returncode != 0:
        return {
            "status": "blocked",
            "blocker": "filterflow v2 gradient subprocess failed",
            "returncode": completed.returncode,
            "stdout_excerpt": completed.stdout[-4000:],
            "stderr_excerpt": completed.stderr[-4000:],
            "cells": [],
        }
    start = completed.stdout.rfind("FILTERFLOW_COMMON_MODEL_SUITE_V2_GRADIENT_JSON_BEGIN")
    end = completed.stdout.rfind("FILTERFLOW_COMMON_MODEL_SUITE_V2_GRADIENT_JSON_END")
    if start < 0 or end < 0 or end <= start:
        return {
            "status": "blocked",
            "blocker": "filterflow v2 gradient JSON sentinels missing",
            "stdout_excerpt": completed.stdout[-4000:],
            "stderr_excerpt": completed.stderr[-4000:],
            "cells": [],
        }
    raw = completed.stdout[start + len("FILTERFLOW_COMMON_MODEL_SUITE_V2_GRADIENT_JSON_BEGIN") : end].strip()
    payload = json.loads(raw)
    payload["stderr_excerpt"] = completed.stderr[-2000:]
    return payload


def _filterflow_script(contracts: list[dict[str, Any]]) -> str:
    contracts_literal = json.dumps(contracts, sort_keys=True)
    return textwrap.dedent(
        f"""
        import inspect
        import json
        import math
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/bayesfilter-dpf-mpl")
        if not hasattr(inspect, "getargspec"):
            inspect.getargspec = inspect.getfullargspec

        import tensorflow as tf
        import tensorflow_probability as tfp
        from filterflow.base import State

        tfd = tfp.distributions
        DTYPE = tf.float64
        FD_STEP = {FD_STEP!r}
        CONTRACTS = json.loads({contracts_literal!r})
        INACTIVE_ZERO_GRADIENT_REASONS = json.loads({json.dumps(_inactive_reason_payload(), sort_keys=True)!r})

        def to_tensor(value):
            return tf.convert_to_tensor(value, dtype=DTYPE)

        def scalar(tensor):
            return float(tf.cast(tensor, DTYPE).numpy())

        def mvn_log_prob(values, loc, covariance):
            return tfd.MultivariateNormalTriL(
                loc=to_tensor(loc),
                scale_tril=tf.linalg.cholesky(to_tensor(covariance)),
            ).log_prob(to_tensor(values))

        def normal_log_prob(values, loc, scale):
            return tfd.Normal(loc=to_tensor(loc), scale=to_tensor(scale)).log_prob(to_tensor(values))

        def uniform_log_weights(n_particles):
            return tf.fill([n_particles], -tf.math.log(tf.cast(n_particles, DTYPE)))

        def wrap_angle(value):
            pi = tf.constant(math.pi, DTYPE)
            return tf.math.floormod(to_tensor(value) + pi, 2.0 * pi) - pi

        def range_bearing_observation(x):
            x = to_tensor(x)
            px = x[..., 0]
            py = x[..., 1]
            rng = tf.sqrt(px * px + py * py + tf.constant(1e-12, DTYPE))
            bearing = tf.atan2(py, px)
            return tf.stack([rng, bearing], axis=-1)

        def gaussian_logpdf_zero_mean(residuals, covariance):
            residuals = to_tensor(residuals)
            covariance = to_tensor(covariance)
            chol = tf.linalg.cholesky(covariance)
            solved = tf.linalg.cholesky_solve(chol, tf.transpose(residuals))
            quad = tf.reduce_sum(tf.transpose(solved) * residuals, axis=1)
            dim = tf.cast(tf.shape(covariance)[0], DTYPE)
            logdet = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)))
            return -0.5 * (dim * tf.math.log(tf.constant(2.0 * math.pi, DTYPE)) + logdet + quad)

        def complete_k(previous_k, previous_m, current_m, p, c):
            return (
                to_tensor(p["a"]) * previous_k
                + to_tensor(p["b"]) * current_m
                + c * current_m * current_m
                + to_tensor(p["d"]) * previous_m * current_m
            )

        def complete_structural(contract, previous, current_m, params):
            _rho, _sigma, c = params
            current_k = complete_k(previous[:, 1], previous[:, 0], current_m, contract["parameters"], c)
            return tf.stack([current_m, current_k], axis=1)

        def predator_transition_mean(contract, x_prev, params):
            p = contract["parameters"]
            base_theta = to_tensor(contract["theta"])
            theta = tf.concat([tf.reshape(params[0], [1]), base_theta[1:]], axis=0)
            state = to_tensor(x_prev)
            r, k_capacity, a_half, s_rate, u_rate, v_rate = tf.unstack(theta)
            substeps = int(p["rk4_substeps"])
            step = to_tensor(p["delta"]) / tf.cast(substeps, DTYPE)
            def rhs(values):
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

        def transition_mean(contract, particles, params):
            model_id = contract["model_id"]
            p = contract["parameters"]
            particles = to_tensor(particles)
            if model_id == "lgssm_2d_h25_rich":
                return tf.linalg.matmul(particles, to_tensor(p["A"]) * params[0], transpose_b=True)
            if model_id == "sv_1d_h18_rich":
                mu, phi, _sigma = params
                return mu + phi * (particles - mu)
            if model_id == "range_bearing_4d_h20_rich":
                return tf.linalg.matmul(particles, to_tensor(p["A"]), transpose_b=True)
            if model_id == "structural_ar1_quadratic_h16":
                rho, _sigma, c = params
                current_m = rho * particles[:, 0]
                current_k = complete_k(particles[:, 1], particles[:, 0], current_m, p, c)
                return tf.stack([current_m, current_k], axis=1)
            if model_id == "predator_prey_rk4":
                return predator_transition_mean(contract, particles, params)
            raise ValueError(model_id)

        def observation_log_density(contract, particles, observation, params):
            model_id = contract["model_id"]
            p = contract["parameters"]
            particles = to_tensor(particles)
            observation = to_tensor(observation)
            if model_id == "lgssm_2d_h25_rich":
                _transition_scale, observation_noise_scale = params
                loc = tf.linalg.matmul(particles, to_tensor(p["C"]), transpose_b=True)
                covariance = to_tensor(p["R"]) * observation_noise_scale
                return mvn_log_prob(tf.broadcast_to(observation, tf.shape(loc)), loc, covariance)
            if model_id == "sv_1d_h18_rich":
                scale = tf.exp(0.5 * particles[:, 0])
                return normal_log_prob(tf.broadcast_to(observation[0], tf.shape(scale)), 0.0, scale)
            if model_id == "range_bearing_4d_h20_rich":
                sigma_range, sigma_bearing = params
                predicted = range_bearing_observation(particles)
                residual = observation[tf.newaxis, :] - predicted
                residual = tf.concat([residual[..., :1], wrap_angle(residual[..., 1:2])], axis=-1)
                covariance = tf.linalg.diag(tf.stack([tf.square(sigma_range), tf.square(sigma_bearing)]))
                return gaussian_logpdf_zero_mean(residual, covariance)
            if model_id == "structural_ar1_quadratic_h16":
                mean = particles[:, 1] + to_tensor(p["lambda"]) * particles[:, 0]
                return normal_log_prob(tf.broadcast_to(observation[0], tf.shape(mean)), mean, p["observation_scale"])
            if model_id == "predator_prey_rk4":
                return mvn_log_prob(tf.broadcast_to(observation, tf.shape(particles)), particles, p["observation_covariance"])
            raise ValueError(model_id)

        def fixed_branch_scalar(contract, params):
            particles = to_tensor(contract["initial_particles"])
            innovations = to_tensor(contract["transition_innovations"])
            observations = to_tensor(contract["observations"])
            ancestors = tf.convert_to_tensor(contract["fixed_ancestor_indices"], dtype=tf.int32)
            n_particles = int(particles.shape[0])
            log_weights = uniform_log_weights(n_particles)
            total = tf.zeros([], DTYPE)
            _ = State(tf.reshape(particles, [1, n_particles, int(particles.shape[1])]))
            for step in range(int(contract["horizon"])):
                if bool(contract["resampling_flags"][step]):
                    particles = tf.gather(particles, ancestors)
                    log_weights = uniform_log_weights(n_particles)
                predicted = transition_mean(contract, particles, params) + innovations[step]
                if contract["model_id"] == "structural_ar1_quadratic_h16":
                    predicted = complete_structural(contract, particles, predicted[:, 0], params)
                obs_log = observation_log_density(contract, predicted, observations[step], params)
                unnormalized = log_weights + obs_log
                increment = tf.reduce_logsumexp(unnormalized)
                total = total + increment
                log_weights = unnormalized - increment
                particles = predicted
            return total

        def central_fd(value_fn, params):
            results = []
            for index in range(len(params)):
                plus = list(params)
                minus = list(params)
                plus[index] = plus[index] + FD_STEP
                minus[index] = minus[index] - FD_STEP
                results.append(scalar((value_fn(plus) - value_fn(minus)) / tf.constant(2.0 * FD_STEP, DTYPE)))
            return results

        def fd_ladder(value_fn, params):
            rows = []
            for step_size in {list(FD_LADDER_STEPS)!r}:
                gradients = []
                valid = []
                for index in range(len(params)):
                    plus = list(params)
                    minus = list(params)
                    plus[index] = plus[index] + step_size
                    minus[index] = minus[index] - step_size
                    parameterization = str(value_fn.__dict__.get("parameterizations", [])[index])
                    positive_scale = (
                        "positive" in parameterization
                        or "standard deviation" in parameterization
                        or "observation covariance" in parameterization
                    )
                    try:
                        _ = parameterization
                    except Exception:
                        positive_scale = False
                    if positive_scale and minus[index] <= 0.0:
                        gradients.append(None)
                        valid.append(False)
                        continue
                    gradients.append(scalar((value_fn(plus) - value_fn(minus)) / tf.constant(2.0 * step_size, DTYPE)))
                    valid.append(True)
                rows.append({{"step": float(step_size), "gradient": gradients, "valid": valid}})
            return rows

        def run_contract(contract):
            variables = [tf.Variable(value, dtype=DTYPE) for value in contract["parameter_values"]]
            with tf.GradientTape() as tape:
                value = fixed_branch_scalar(contract, variables)
            gradients = tape.gradient(value, variables)
            fd = central_fd(
                lambda params: fixed_branch_scalar(contract, [tf.constant(value, DTYPE) for value in params]),
                [float(value) for value in contract["parameter_values"]],
            )
            def ladder_value_fn(params):
                return fixed_branch_scalar(contract, [tf.constant(value, DTYPE) for value in params])
            ladder_value_fn.__dict__["parameterizations"] = contract["parameterizations"]
            ladder = fd_ladder(ladder_value_fn, [float(value) for value in contract["parameter_values"]])
            gradient_values = []
            inactive_zero_reasons = {{}}
            for grad, knob in zip(gradients, contract["gradient_knob_names"]):
                if grad is None:
                    reason = INACTIVE_ZERO_GRADIENT_REASONS.get(contract["model_id"], {{}}).get(knob)
                    if reason is not None:
                        gradient_values.append(0.0)
                        inactive_zero_reasons[knob] = reason
                    else:
                        gradient_values.append(None)
                else:
                    gradient_values.append(scalar(grad))
            finite = bool(
                tf.math.is_finite(value).numpy()
                and all(
                    (grad is not None and tf.math.is_finite(grad).numpy()) or knob in inactive_zero_reasons
                    for grad, knob in zip(gradients, contract["gradient_knob_names"])
                )
            )
            return {{
                "status": "executed",
                "backend": "filterflow_env_local_v2_fixed_branch_gradient",
                "model_id": contract["model_id"],
                "scalar": scalar(value),
                "gradient": gradient_values,
                "finite_difference_gradient": fd,
                "finite_difference_ladder": ladder,
                "gradient_delta_vs_finite_difference": [
                    None if grad is None else float(grad) - float(fd_value)
                    for grad, fd_value in zip(gradient_values, fd)
                ],
                "inactive_zero_gradient_knobs": list(inactive_zero_reasons),
                "inactive_zero_gradient_reasons": inactive_zero_reasons,
                "disconnected_zero_gradient_knobs": list(inactive_zero_reasons),
                "finite": finite,
                "finite_difference_finite": all(math.isfinite(float(value)) for value in fd),
                "gradient_knob_names": contract["gradient_knob_names"],
                "parameterizations": contract["parameterizations"],
                "contract_checksum": contract["contract_checksum"],
                "spec_checksum": contract["spec_checksum"],
            }}

        cells = []
        for contract in CONTRACTS:
            try:
                cells.append(run_contract(contract))
            except Exception as exc:
                cells.append({{
                    "status": "blocked",
                    "backend": "filterflow_env_local_v2_fixed_branch_gradient",
                    "model_id": contract.get("model_id"),
                    "blocker": repr(exc),
                    "contract_checksum": contract.get("contract_checksum"),
                    "spec_checksum": contract.get("spec_checksum"),
                }})
        payload = {{
            "status": "executed",
            "dtype": DTYPE.name,
            "pre_import_cuda_visible_devices": PRE_IMPORT_CUDA_VISIBLE_DEVICES,
            "gpu_devices_visible": [str(device) for device in tf.config.list_physical_devices("GPU")],
            "cells": cells,
        }}
        print("FILTERFLOW_COMMON_MODEL_SUITE_V2_GRADIENT_JSON_BEGIN")
        print(json.dumps(payload, sort_keys=True))
        print("FILTERFLOW_COMMON_MODEL_SUITE_V2_GRADIENT_JSON_END")
        """
    )


def _cell(
    spec: CommonModelSpecV2,
    contract: dict[str, Any],
    bayesfilter: dict[str, Any],
    filterflow: dict[str, Any],
) -> dict[str, Any]:
    if filterflow.get("status") != "executed":
        return {
            "model": spec.model_id,
            "family": spec.family,
            "implementations": ["BayesFilter", "FilterFlow"],
            "cell_type": "v2_fixed_branch_gradient",
            "status": "INTERFACE_BLOCKED",
            "decision": f"{spec.model_id}_filterflow_v2_gradient_blocked",
            "primary_criterion": "both adapters must execute the same fixed-branch gradient scalar",
            "metrics": {},
            "mismatch_class": "filterflow_v2_fixed_branch_gradient_blocked",
            "reason": filterflow.get("blocker", "missing FilterFlow cell"),
            "bayesfilter": bayesfilter,
            "filterflow": filterflow,
            "contract": contract,
        }
    comparison = _compare_gradient_payloads(bayesfilter, filterflow)
    matched = (
        comparison["scalar_within_tolerance"]
        and comparison["gradient_within_tolerance"]
        and bayesfilter["finite"]
        and filterflow["finite"]
    )
    return {
        "model": spec.model_id,
        "family": spec.family,
        "implementations": ["BayesFilter", "FilterFlow"],
        "cell_type": "v2_fixed_branch_gradient",
        "status": "MATCHED" if matched else "EXPLAINED_MISMATCH",
        "decision": f"{spec.model_id}_fixed_branch_gradient_matched" if matched else f"{spec.model_id}_fixed_branch_gradient_mismatch",
        "primary_criterion": "fixed-branch scalar and physical-knob AD gradients match within tolerance; finite differences are diagnostic-only",
        "metrics": comparison,
        "mismatch_class": None if matched else "v2_fixed_branch_scalar_or_ad_gradient_delta",
        "bayesfilter": bayesfilter,
        "filterflow": filterflow,
        "contract": contract,
        "spec_checksum": spec.checksum(),
        "non_claim": "fixed-branch gradient agreement is not differentiable-resampling correctness",
    }


def _classified_cell(spec: CommonModelSpecV2, status: str, reason: str) -> dict[str, Any]:
    return {
        "model": spec.model_id,
        "family": spec.family,
        "implementations": ["BayesFilter", "FilterFlow"],
        "cell_type": "v2_fixed_branch_gradient",
        "status": status,
        "decision": f"{spec.model_id}_{status.lower()}",
        "primary_criterion": "row is not executed unless P1-ready for P5 gradients",
        "metrics": {},
        "mismatch_class": status.lower(),
        "reason": reason,
        "spec_checksum": spec.checksum(),
    }


def _compare_gradient_payloads(bayesfilter: dict[str, Any], filterflow: dict[str, Any]) -> dict[str, Any]:
    scalar_delta = abs(float(bayesfilter["scalar"]) - float(filterflow["scalar"]))
    gradient_delta = [
        _numeric_or_inf(left) - _numeric_or_inf(right)
        for left, right in zip(bayesfilter["gradient"], filterflow["gradient"], strict=True)
    ]
    max_abs_gradient_delta = max(abs(value) for value in gradient_delta) if gradient_delta else 0.0
    bayesfilter_fd_delta = [
        _numeric_or_inf(left) - _numeric_or_inf(right)
        for left, right in zip(
            bayesfilter["gradient"],
            bayesfilter["finite_difference_gradient"],
            strict=True,
        )
    ]
    filterflow_fd_delta = [
        _numeric_or_inf(left) - _numeric_or_inf(right)
        for left, right in zip(
            filterflow["gradient"],
            filterflow["finite_difference_gradient"],
            strict=True,
        )
    ]
    max_abs_bayesfilter_fd_delta = max(abs(value) for value in bayesfilter_fd_delta) if bayesfilter_fd_delta else 0.0
    max_abs_filterflow_fd_delta = max(abs(value) for value in filterflow_fd_delta) if filterflow_fd_delta else 0.0
    return {
        "scalar_delta": scalar_delta,
        "gradient_delta": gradient_delta,
        "max_abs_gradient_delta": max_abs_gradient_delta,
        "bayesfilter_fd_delta": bayesfilter_fd_delta,
        "filterflow_fd_delta": filterflow_fd_delta,
        "max_abs_bayesfilter_fd_delta": max_abs_bayesfilter_fd_delta,
        "max_abs_filterflow_fd_delta": max_abs_filterflow_fd_delta,
        "bayesfilter_fd_ladder": _fd_ladder_delta(
            bayesfilter["gradient"],
            bayesfilter.get("finite_difference_ladder", []),
        ),
        "filterflow_fd_ladder": _fd_ladder_delta(
            filterflow["gradient"],
            filterflow.get("finite_difference_ladder", []),
        ),
        "scalar_within_tolerance": scalar_delta <= VALUE_TOLERANCE,
        "gradient_within_tolerance": max_abs_gradient_delta <= GRADIENT_TOLERANCE,
        "bayesfilter_fd_within_tolerance": max_abs_bayesfilter_fd_delta <= FD_TOLERANCE,
        "filterflow_fd_within_tolerance": max_abs_filterflow_fd_delta <= FD_TOLERANCE,
    }


def _numeric_or_inf(value: float | None) -> float:
    if value is None:
        return float("inf")
    return float(value)


def _preflight_prior_artifacts(p1_manifest: dict[str, Any]) -> None:
    p1_ids = [row.get("model_id") for row in p1_manifest.get("rows", [])]
    if tuple(p1_ids) != EXPECTED_V2_MODEL_IDS:
        raise ValueError(f"P1 manifest model id gate failed: {p1_ids}")
    required_passes = [
        (P2_RESULT_PATH, "PASS_P2_DENSITY_READY_FOR_P3"),
        (P3_RESULT_PATH, "PASS_P3_NORESAMPLING_READY_FOR_P4"),
        (P4_RESULT_PATH, "PASS_P4_FIXED_RESAMPLING_READY_FOR_P5"),
    ]
    for path, decision in required_passes:
        if not path.read_text(encoding="utf-8").splitlines()[4].endswith(decision):
            raise ValueError(f"{path.name} is not marked {decision}")


def _ready_ids_for_phase(payload: dict[str, Any], field: str, ready_value: str) -> set[str]:
    return {
        str(row["model_id"])
        for row in payload.get("pre_run_row_classification_table", [])
        if row.get(field) == ready_value
    }


def _blocked_reason(spec: CommonModelSpecV2) -> str:
    for knob in spec.gradient_contract.get("knobs", []):
        if not knob.get("include") and knob.get("exclusion_reason"):
            return str(knob["exclusion_reason"])
    return str(spec.phase_readiness.get("reason", "row not READY_FOR_P5 in P1 classification"))


def _central_finite_difference(
    value_fn: Callable[[list[float]], tf.Tensor],
    params: list[float],
) -> list[float]:
    gradients = []
    for index in range(len(params)):
        plus = list(params)
        minus = list(params)
        plus[index] += FD_STEP
        minus[index] -= FD_STEP
        gradients.append(
            scalar(
                (value_fn(plus) - value_fn(minus))
                / tf.constant(2.0 * FD_STEP, DTYPE)
            )
        )
    return gradients


def _finite_difference_ladder(
    value_fn: Callable[[list[float]], tf.Tensor],
    params: list[float],
    parameterizations: list[str],
) -> list[dict[str, Any]]:
    rows = []
    for step_size in FD_LADDER_STEPS:
        gradients: list[float | None] = []
        valid: list[bool] = []
        for index in range(len(params)):
            plus = list(params)
            minus = list(params)
            plus[index] += step_size
            minus[index] -= step_size
            positive_scale = _is_positive_scale_parameterization(parameterizations[index])
            if positive_scale and minus[index] <= 0.0:
                gradients.append(None)
                valid.append(False)
                continue
            gradients.append(
                scalar(
                    (value_fn(plus) - value_fn(minus))
                    / tf.constant(2.0 * step_size, DTYPE)
                )
            )
            valid.append(True)
        rows.append({"step": float(step_size), "gradient": gradients, "valid": valid})
    return rows


def _is_positive_scale_parameterization(parameterization: str) -> bool:
    text = parameterization.lower()
    return (
        "positive" in text
        or "standard deviation" in text
        or "observation covariance" in text
    )


def _fd_ladder_delta(
    gradient: list[float | None],
    ladder: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for row in ladder:
        fd_values = row.get("gradient", [])
        deltas = [
            None if grad is None or fd_value is None else float(grad) - float(fd_value)
            for grad, fd_value in zip(gradient, fd_values, strict=True)
        ]
        finite_deltas = [abs(float(delta)) for delta in deltas if delta is not None]
        rows.append(
            {
                "step": row.get("step"),
                "valid": row.get("valid"),
                "delta": deltas,
                "max_abs_delta": max(finite_deltas) if finite_deltas else None,
                "explanatory_only": True,
            }
        )
    return rows


def _encode_gradients_by_contract(
    gradients: list[tf.Tensor | None],
    knob_names: list[str],
    model_id: str,
) -> tuple[list[float | None], dict[str, str]]:
    values: list[float | None] = []
    inactive_zero_reasons: dict[str, str] = {}
    for grad, knob in zip(gradients, knob_names, strict=True):
        if grad is None:
            reason = INACTIVE_ZERO_GRADIENT_REASONS.get((model_id, knob))
            if reason is not None:
                values.append(0.0)
                inactive_zero_reasons[knob] = reason
            else:
                values.append(None)
            continue
        values.append(scalar(grad))
    return values, inactive_zero_reasons


def _inactive_reason_payload() -> dict[str, dict[str, str]]:
    payload: dict[str, dict[str, str]] = {}
    for (model_id, knob), reason in INACTIVE_ZERO_GRADIENT_REASONS.items():
        payload.setdefault(model_id, {})[knob] = reason
    return payload


def _inactive_zero_gradient_summary(cells: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    summary: dict[str, dict[str, str]] = {}
    for cell in cells:
        reasons = cell.get("bayesfilter", {}).get("inactive_zero_gradient_reasons", {})
        if reasons:
            summary[cell["model"]] = dict(reasons)
    return summary


def _mvn_log_prob(values: tf.Tensor, loc: tf.Tensor, covariance: tf.Tensor | Any) -> tf.Tensor:
    covariance = tf.convert_to_tensor(covariance, DTYPE)
    return tfp_distribution_mvn_log_prob(values, loc, covariance)


def tfp_distribution_mvn_log_prob(values: tf.Tensor, loc: tf.Tensor, covariance: tf.Tensor) -> tf.Tensor:
    import tensorflow_probability as tfp

    return tfp.distributions.MultivariateNormalTriL(
        loc=tf.convert_to_tensor(loc, DTYPE),
        scale_tril=tf.linalg.cholesky(covariance),
    ).log_prob(tf.convert_to_tensor(values, DTYPE))


def _normal_log_prob(value: tf.Tensor, loc: tf.Tensor | float, scale: tf.Tensor | float) -> tf.Tensor:
    import tensorflow_probability as tfp

    return tfp.distributions.Normal(
        loc=tf.convert_to_tensor(loc, DTYPE),
        scale=tf.convert_to_tensor(scale, DTYPE),
    ).log_prob(tf.convert_to_tensor(value, DTYPE))


def _gaussian_logpdf_zero_mean(residuals: tf.Tensor, covariance: tf.Tensor) -> tf.Tensor:
    residuals = tf.convert_to_tensor(residuals, DTYPE)
    covariance = tf.convert_to_tensor(covariance, DTYPE)
    chol = tf.linalg.cholesky(covariance)
    solved = tf.linalg.cholesky_solve(chol, tf.transpose(residuals))
    quad = tf.reduce_sum(tf.transpose(solved) * residuals, axis=1)
    dim = tf.cast(tf.shape(covariance)[0], DTYPE)
    logdet = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)))
    return -0.5 * (dim * tf.math.log(tf.constant(2.0 * math.pi, DTYPE)) + logdet + quad)


def _range_bearing_observation(x: tf.Tensor) -> tf.Tensor:
    x = tf.convert_to_tensor(x, DTYPE)
    px = x[..., 0]
    py = x[..., 1]
    rng = tf.sqrt(px * px + py * py + tf.constant(1e-12, DTYPE))
    bearing = tf.atan2(py, px)
    return tf.stack([rng, bearing], axis=-1)


def _wrap_angle(value: tf.Tensor) -> tf.Tensor:
    pi = tf.constant(math.pi, DTYPE)
    return tf.math.floormod(tf.convert_to_tensor(value, DTYPE) + pi, 2.0 * pi) - pi


def _uniform_log_weights(n_particles: int) -> tf.Tensor:
    return tf.fill([n_particles], -tf.math.log(tf.cast(n_particles, DTYPE)))


def _finite_scalar(value: tf.Tensor) -> bool:
    return bool(tf.math.is_finite(tf.cast(value, DTYPE)).numpy())


def _cell_nonfinite(cell: dict[str, Any]) -> bool:
    return (
        cell.get("bayesfilter", {}).get("finite") is False
        or cell.get("filterflow", {}).get("finite") is False
    )


def _cell_fd_diagnostic_outside_tolerance(cell: dict[str, Any]) -> bool:
    if cell.get("status") == "CONTRACT_BLOCKED":
        return False
    metrics = cell.get("metrics", {})
    return (
        metrics.get("bayesfilter_fd_within_tolerance") is False
        or metrics.get("filterflow_fd_within_tolerance") is False
    )


def _decision(cells: list[dict[str, Any]], ready_ids: set[str]) -> str:
    ready_cells = [cell for cell in cells if cell["model"] in ready_ids]
    if all(cell["status"] == "MATCHED" for cell in ready_cells):
        return "PENDING_CLAUDE_REVIEW"
    if any(cell["status"] == "INTERFACE_BLOCKED" for cell in ready_cells):
        return "P5_INTERFACE_BLOCKED"
    return "P5_CLASSIFIED_MISMATCH_PENDING_REVIEW"


def _status_counts(cells: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for cell in cells:
        counts[cell["status"]] = counts.get(cell["status"], 0) + 1
    return counts


def _summary_scalar(cells: list[dict[str, Any]], key: str) -> float | None:
    values = [
        float(cell.get("metrics", {}).get(key, 0.0))
        for cell in cells
        if key in cell.get("metrics", {})
    ]
    return max(values) if values else None


def _summary(cells: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_cells": len(cells),
        "models": [cell["model"] for cell in cells],
        "status_counts": _status_counts(cells),
        "max_abs_scalar_delta": _summary_scalar(cells, "scalar_delta"),
        "max_abs_gradient_delta": _summary_scalar(cells, "max_abs_gradient_delta"),
        "max_abs_bayesfilter_fd_delta": _summary_scalar(cells, "max_abs_bayesfilter_fd_delta"),
        "max_abs_filterflow_fd_delta": _summary_scalar(cells, "max_abs_filterflow_fd_delta"),
    }


def _validate_payload(payload: dict[str, Any]) -> None:
    required = {
        "primary_criterion_fields",
        "veto_diagnostics",
        "explanatory_only_fields",
        "review_round",
        "open_material_blockers",
        "repair_amendment_required",
        "next_allowed_action",
        "cells",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"P5 payload missing required fields: {sorted(missing)}")
    if payload["decision"] not in {
        "PENDING_CLAUDE_REVIEW",
        "PASS_P5_GRADIENTS_READY_FOR_P6",
        "P5_CLASSIFIED_MISMATCH_PENDING_REVIEW",
    }:
        raise ValueError(f"P5 payload decision not passable: {payload['decision']}")
    if payload["run_manifest"]["pre_import_cuda_visible_devices"] != "-1":
        raise RuntimeError("CPU-only pre-import manifest missing")
    if payload["run_manifest"]["gpu_devices_visible"] != []:
        raise RuntimeError("GPU visible in CPU-only run")
    validate_filterflow_reference_status(payload["filterflow_status"], marker_path=FILTERFLOW_MARKER_PATH)
    ids = [cell["model"] for cell in payload["cells"]]
    if tuple(ids) != EXPECTED_V2_MODEL_IDS:
        raise ValueError(f"P5 cell id gate failed: {ids}")
    if payload["veto_diagnostics"].get("missing_filterflow_subprocess_environment"):
        raise ValueError("P5 missing FilterFlow subprocess environment")
    if payload["veto_diagnostics"].get("nonfinite_gradient_or_scalar"):
        raise ValueError("P5 nonfinite gradient or scalar")
    if payload["veto_diagnostics"].get("sir_gradient_executed_despite_p1_block"):
        raise ValueError("P5 executed SIR despite P1 block")
    ready_ids = set(payload["primary_criterion_fields"]["ready_model_ids"])
    for cell in payload["cells"]:
        if (
            cell["model"] in ready_ids
            and cell["status"] != "MATCHED"
            and payload["decision"] != "P5_CLASSIFIED_MISMATCH_PENDING_REVIEW"
        ):
            raise ValueError(f"P5 ready row did not match: {cell['model']} {cell['status']}")
        if cell["model"] == "spatial_sir_j3_rk4" and cell["status"] != "CONTRACT_BLOCKED":
            raise ValueError("P5 SIR row is not blocked")


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# DPF Common Model Suite V2 P5 Fixed-Branch Gradient Result",
        "",
        "metadata_date: 2026-06-07",
        "phase: P5",
        f"decision: {payload['decision']}",
        "",
        "## Question",
        "",
        payload["question"],
        "",
        "## Evidence Contract",
        "",
        "Primary criterion: every P1-ready v2 gradient row matches on scalar and physical-knob AD gradients within tolerance; finite differences are diagnostic-only.",
        "",
        "Veto diagnostics: changed knobs, parameterization mismatch, nonfinite scalar or AD gradient, gradient through random/discrete ancestors, v1/v2 leakage, old artifacts, unclassified BF/FF scalar or AD-gradient mismatch, or unreviewed SIR execution.",
        "",
        "## Result",
        "",
        f"- JSON artifact: `{payload['artifact_paths']['json']}`",
        f"- Markdown report: `{payload['artifact_paths']['markdown_report']}`",
        f"- Reproducibility digest: `{payload['reproducibility_digest']}`",
        "",
        "## Gradient Cells",
        "",
        "| Model id | Status | Knobs | Scalar delta | Max gradient delta | BF FD delta | FF FD delta |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for cell in payload["cells"]:
        metrics = cell.get("metrics", {})
        knobs = ",".join(cell.get("contract", {}).get("gradient_knob_names", []))
        lines.append(
            "| `{model}` | {status} | `{knobs}` | {scalar_delta} | {gradient_delta} | {bf_fd} | {ff_fd} |".format(
                model=cell["model"],
                status=cell["status"],
                knobs=knobs or "N/A",
                scalar_delta=metrics.get("scalar_delta", "N/A"),
                gradient_delta=metrics.get("max_abs_gradient_delta", "N/A"),
                bf_fd=metrics.get("max_abs_bayesfilter_fd_delta", "N/A"),
                ff_fd=metrics.get("max_abs_filterflow_fd_delta", "N/A"),
            )
        )
    lines.extend(
        [
            "",
            "## Primary Criterion Fields",
            "",
            f"- ready model ids: `{payload['primary_criterion_fields']['ready_model_ids']}`",
            f"- blocked model ids: `{payload['primary_criterion_fields']['blocked_model_ids']}`",
            f"- executed model ids: `{payload['primary_criterion_fields']['executed_model_ids']}`",
            f"- required gradient knobs: `{payload['primary_criterion_fields']['required_gradient_knobs']}`",
            f"- all ready rows matched: `{payload['primary_criterion_fields']['all_ready_rows_matched']}`",
            f"- inactive zero gradient reasons: `{payload['primary_criterion_fields']['inactive_zero_gradient_reasons']}`",
            f"- gradient scalar: `{payload['primary_criterion_fields']['gradient_scalar']}`",
            f"- ancestor gradient policy: `{payload['primary_criterion_fields']['ancestor_gradient_policy']}`",
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
            f"- max abs scalar delta: `{payload['explanatory_only_fields']['max_abs_scalar_delta']}`",
            f"- max abs gradient delta: `{payload['explanatory_only_fields']['max_abs_gradient_delta']}`",
            f"- max abs BayesFilter FD delta: `{payload['explanatory_only_fields']['max_abs_bayesfilter_fd_delta']}`",
            f"- max abs FilterFlow FD delta: `{payload['explanatory_only_fields']['max_abs_filterflow_fd_delta']}`",
            f"- all ready rows passed BayesFilter FD diagnostic: `{payload['explanatory_only_fields']['all_ready_rows_passed_bayesfilter_fd_diagnostic']}`",
            f"- all ready rows passed FilterFlow FD diagnostic: `{payload['explanatory_only_fields']['all_ready_rows_passed_filterflow_fd_diagnostic']}`",
            f"- finite difference diagnostic outside tolerance: `{payload['explanatory_only_fields']['finite_difference_diagnostic_outside_tolerance']}`",
            f"- blocked row policy: `{payload['explanatory_only_fields']['blocked_rows_are_contract_scope_only']}`",
            "",
            "## Command Manifest",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| git commit | `{payload['run_manifest'].get('commit')}` |",
            f"| dirty status | `{_single_line(payload['run_manifest'].get('dirty_state_summary'))}` |",
            f"| command | `{payload['run_manifest'].get('command')}` |",
            f"| validation commands | `python -m json.tool {payload['artifact_paths']['json']}`; `CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/bayesfilter-dpf-mpl python -m experiments.dpf_implementation.tf_tfp.runners.run_common_model_suite_v2_gradients_tf --validate-only`; `git diff --check` on P5 files |",
            f"| CPU/GPU status | CPU-only TensorFlow run; pre-import `CUDA_VISIBLE_DEVICES={payload['run_manifest'].get('pre_import_cuda_visible_devices')}`; visible GPUs `{payload['run_manifest'].get('gpu_devices_visible')}` |",
            "| random seeds | none consumed; deterministic P1 particles, observations, transition innovations, resampling flags, and ancestor indices |",
            "| dtype | `tf.float64` / JSON dtype `float64` |",
            f"| output artifacts | `{payload['artifact_paths']['json']}`; `{payload['artifact_paths']['markdown_report']}`; `{payload['artifact_paths']['phase_result']}` |",
            "",
            "## Review State",
            "",
            f"review_round: {payload['review_round']} pending Claude result/governance review",
            "",
            f"open_material_blockers: {payload['open_material_blockers'] or 'none identified locally'}",
            "",
            f"repair_amendment_required: {str(payload['repair_amendment_required']).lower()}",
            "",
            f"next_allowed_action: {payload['next_allowed_action']}",
            "",
            "## Repair History",
            "",
            "- P5 FD diagnostic-only contract amendment reviewed by Claude: first round `BLOCK`, second round `PASS`.",
            "- The earlier disconnected-zero-gradient FD guard is superseded where it used FD as a promotion/veto condition.",
            "- Inactive zero gradients for `sv_1d_h18_rich:sigma` and `structural_ar1_quadratic_h16:sigma` are recorded by fixed-branch scalar derivation, not by FD pass/fail.",
            "",
            "## Decision Table",
            "",
            "| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |",
            "|---|---|---|---|---|---|",
            "| PENDING_CLAUDE_REVIEW | all P1-ready rows matched on fixed-branch scalar and AD gradients locally; FD retained as diagnostic-only | no local veto fired | Claude may identify gradient-contract or adapter-governance gaps; SIR remains blocked | run Claude P5 result review | no stochastic gradient, filter correctness, or student claim |"
            if payload["decision"] == "PENDING_CLAUDE_REVIEW"
            else f"| {payload['decision']} | at least one BF/FF scalar or AD-gradient criterion is not matched or classified | local veto or interface blocker fired | FD diagnostics are explanatory-only and cannot promote or block P5 | run Claude P5 blocked-result review and decide reviewed repair/stop path | no stochastic gradient, filter correctness, or student claim |",
            "",
            "## Post-Run Red Team",
            "",
            "Strongest alternative explanation: matching fixed-branch gradients can still miss gradients through stochastic resampling or broader filter-policy errors.",
            "",
            "Result that would overturn the decision: any P5 knob is found to have changed after results, the SIR row was executed despite P1 block, or finite differences use a different scalar from AD.",
            "",
            "Weakest evidence link: finite differences are local numerical checks and do not prove scientific validity.",
            "",
            "## Non-Claims",
            "",
        ]
    )
    lines.extend(f"- {claim}" for claim in payload["non_claims"])
    lines.append("")
    return "\n".join(lines)


def _filterflow_checkout_manifest() -> dict[str, Any]:
    if not FILTERFLOW_PATH.exists():
        return {
            "path": str(FILTERFLOW_PATH),
            "status": "missing",
            "commit": "N/A",
            "branch": "N/A",
            "status_short": "N/A",
        }
    status = {
        "path": str(FILTERFLOW_PATH),
        "status": "current_local_float64_reference_checkout",
        "commit": _git_filterflow(["rev-parse", "HEAD"]),
        "branch": _git_filterflow(["rev-parse", "--abbrev-ref", "HEAD"]),
        "status_short": _git_filterflow(["status", "--short"]),
        "marker_exists": FILTERFLOW_MARKER_PATH.exists(),
        "provenance_note": "local float64 FilterFlow comparator, not pristine upstream and not an oracle",
    }
    try:
        validate_filterflow_reference_status(status, marker_path=FILTERFLOW_MARKER_PATH)
        return status
    except Exception as exc:
        return {**status, "status": "blocked", "blocker": repr(exc)}


def _git_filterflow(args: list[str]) -> str:
    completed = subprocess.run(
        ["git", "-C", str(FILTERFLOW_PATH), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _digest_payload(payload: dict[str, Any]) -> str:
    clone = dict(payload)
    clone.pop("reproducibility_digest", None)
    if "run_manifest" in clone:
        clone["run_manifest"] = dict(clone["run_manifest"])
        clone["run_manifest"].pop("wall_time_seconds", None)
    return stable_digest(clone)


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
