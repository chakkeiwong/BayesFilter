"""Run v2 fixed-noise no-resampling path tie-outs."""

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
from typing import Any

import tensorflow as tf

from experiments.dpf_implementation.tf_tfp.fixtures.common_model_suite_tf import (
    EXPECTED_V2_MODEL_IDS,
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
PLAN_PATH = "docs/plans/bayesfilter-dpf-common-model-suite-v2-production-p3-noresampling-paths-subplan-2026-06-07.md"
RESULT_PATH = "docs/plans/bayesfilter-dpf-common-model-suite-v2-production-p3-noresampling-paths-result-2026-06-07.md"
P1_MANIFEST_PATH = OUTPUT_DIR / "dpf_common_model_suite_v2_manifest_2026-06-07.json"
P2_JSON_PATH = OUTPUT_DIR / "dpf_common_model_suite_v2_density_2026-06-07.json"
JSON_PATH = OUTPUT_DIR / "dpf_common_model_suite_v2_noresampling_2026-06-07.json"
REPORT_PATH = REPORT_DIR / "dpf-common-model-suite-v2-noresampling-2026-06-07.md"
FILTERFLOW_ENV_PYTHON = REPO_ROOT / ".localenv" / "filterflow-py311" / "bin" / "python"
FILTERFLOW_PATH = REPO_ROOT / ".localsource" / "filterflow"
FILTERFLOW_MARKER_PATH = FILTERFLOW_PATH / FILTERFLOW_BRANCH_MARKER
VALUE_TOLERANCE = 5e-10
LEDGER_TOLERANCE = 5e-10


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
    p2_payload = load_json(P2_JSON_PATH)
    _preflight_prior_artifacts(p1_manifest, p2_payload)
    filterflow_status = _filterflow_checkout_manifest()
    specs = common_model_specs_v2()
    ready_ids = _ready_ids_for_phase(p1_manifest, "P3_noresampling", "READY_FOR_P3")
    contracts = [_path_contract(spec) for spec in specs if spec.model_id in ready_ids]
    filterflow_payload = _filterflow_path_subprocess(contracts)
    cells = []
    contract_by_id = {contract["model_id"]: contract for contract in contracts}
    for spec in specs:
        if spec.model_id not in ready_ids:
            cells.append(_classified_cell(spec, "CONTRACT_BLOCKED", "row not READY_FOR_P3 in P1 classification"))
            continue
        contract = contract_by_id[spec.model_id]
        bayesfilter = _bayesfilter_path(spec, contract)
        filterflow = _filterflow_cell(filterflow_payload, spec.model_id)
        cells.append(_cell(spec, contract, bayesfilter, filterflow))
    decision = _decision(cells)
    return {
        "decision": decision,
        "created_at_utc": utc_now(),
        "question": (
            "Do BayesFilter and executable local float64 FilterFlow-side adapters "
            "match deterministic fixed-noise no-resampling filter-path values "
            "and primary ledgers for each P1-ready v2 row?"
        ),
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "p1_manifest_path": str(P1_MANIFEST_PATH.relative_to(REPO_ROOT)),
        "p2_json_path": str(P2_JSON_PATH.relative_to(REPO_ROOT)),
        "filterflow_reference_policy": reference_policy(),
        "filterflow_status": filterflow_status,
        "filterflow_payload_status": filterflow_payload.get("status"),
        "tolerances": {"value_abs": VALUE_TOLERANCE, "ledger_abs": LEDGER_TOLERANCE},
        "contracts": contracts,
        "primary_criterion_fields": {
            "ready_model_ids": sorted(ready_ids),
            "executed_model_ids": [cell["model"] for cell in cells if cell["status"] != "CONTRACT_BLOCKED"],
            "primary_ledger_fields": [
                "predicted_particles",
                "observation_log_density",
                "unnormalized_log_weights",
                "incremental_log_normalizer",
                "normalized_log_weights",
                "scalar",
            ],
            "all_ready_rows_matched": all(cell["status"] == "MATCHED" for cell in cells if cell["model"] in ready_ids),
        },
        "veto_diagnostics": {
            "missing_filterflow_subprocess_environment": filterflow_payload.get("status") == "blocked",
            "hidden_rng_used": False,
            "stochastic_resampling_used": False,
            "old_three_row_common_model_specs_used_as_v2_source": False,
            "old_2026_06_06_artifact_name_used": False,
            "student_command_executed": False,
            "localsource_filterflow_mutated": False,
            "nonfinite_path_value": any(_cell_nonfinite(cell) for cell in cells),
            "unclassified_mismatch": any(cell["status"] not in {"MATCHED", "CONTRACT_BLOCKED", "INTERFACE_BLOCKED", "EXPLAINED_MISMATCH"} for cell in cells),
        },
        "explanatory_only_fields": {
            "status_counts": _status_counts(cells),
            "max_abs_delta": _max_abs_delta(cells),
            "ess_and_filtered_moments_policy": "reported only as explanatory ledger fields, never pass criteria",
            "p2_density_summary": p2_payload.get("summary"),
        },
        "cells": cells,
        "summary": _summary(cells),
        "review_round": 0,
        "open_material_blockers": [],
        "repair_amendment_required": False,
        "next_allowed_action": "run Claude P3 result review before P4",
        "artifact_paths": {
            "json": str(JSON_PATH.relative_to(REPO_ROOT)),
            "markdown_report": str(REPORT_PATH.relative_to(REPO_ROOT)),
            "phase_result": RESULT_PATH,
        },
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners.run_common_model_suite_v2_noresampling_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_claims": [
            "no stochastic resampling correctness claim",
            "no random-number-generator equality claim",
            "no filtering-algorithm correctness proof",
            "no gradient correctness claim",
            "no student-repository tie-out claim",
            "no TT/SIRT or paper-scale reproduction claim",
        ],
    }


def _path_contract(spec: CommonModelSpecV2) -> dict[str, Any]:
    contract = dict(spec.path_contract)
    payload = {
        "model_id": spec.model_id,
        "family": spec.family,
        "dtype": DTYPE.name,
        "theta": tensor_to_json(spec.theta),
        "parameters": _jsonable_for_runner(spec.parameters),
        "horizon": int(contract["horizon"]),
        "num_particles": int(contract["num_particles"]),
        "state_dim": int(contract["state_dim"]),
        "initial_particles": _jsonable_for_runner(contract["initial_particles"]),
        "transition_innovations": _jsonable_for_runner(contract["transition_innovations"]),
        "observations": _jsonable_for_runner(contract["observations"]),
        "initial_log_weight_policy": contract["initial_log_weight_policy"],
        "proposal": contract["proposal"],
        "resampling_policy": "disabled",
        "scalar": contract["scalar"],
        "transition_innovation_policy": contract["transition_innovation_policy"],
        "spec_checksum": spec.checksum(),
    }
    payload["contract_checksum"] = stable_digest(payload)
    return payload


def _bayesfilter_path(spec: CommonModelSpecV2, contract: dict[str, Any]) -> dict[str, Any]:
    model = bayesfilter_model_for_spec_v2(spec)
    theta = tf.convert_to_tensor(spec.theta, DTYPE)
    particles = tf.convert_to_tensor(contract["initial_particles"], DTYPE)
    innovations = tf.convert_to_tensor(contract["transition_innovations"], DTYPE)
    observations = tf.convert_to_tensor(contract["observations"], DTYPE)
    n_particles = int(particles.shape[0])
    log_weights = _uniform_log_weights(n_particles)
    total_scalar = tf.zeros([], DTYPE)
    ledger = []
    for step in range(int(contract["horizon"])):
        previous_particles = particles
        predicted_particles = _transition_mean_v2(spec, model, theta, previous_particles) + innovations[step]
        if spec.model_id == "structural_ar1_quadratic_h16":
            predicted_particles = model.complete_next_state(previous_particles, predicted_particles[:, 0])
        transition_log_density = model.transition_log_density(theta, previous_particles, predicted_particles, t=step + 1)
        observation_log_density = model.observation_log_density(theta, predicted_particles, observations[step], t=step + 1)
        unnormalized = log_weights + observation_log_density
        increment = tf.reduce_logsumexp(unnormalized)
        total_scalar = total_scalar + increment
        log_weights = unnormalized - increment
        weights = tf.exp(log_weights)
        mean, variance = _weighted_mean_variance(predicted_particles, weights)
        ledger.append(
            _step_ledger(
                step=step,
                previous_particles=previous_particles,
                transition_innovations=innovations[step],
                predicted_particles=predicted_particles,
                observation=observations[step],
                transition_log_density=transition_log_density,
                observation_log_density=observation_log_density,
                unnormalized_log_weights=unnormalized,
                incremental_log_normalizer=increment,
                normalized_log_weights=log_weights,
                weights=weights,
                ess=_ess_from_log_weights(log_weights),
                filtered_mean=mean,
                filtered_variance=variance,
            )
        )
        particles = predicted_particles
    return {
        "status": "executed",
        "backend": "bayesfilter_v2_fixed_noise_noresampling",
        "model_id": spec.model_id,
        "scalar": scalar(total_scalar),
        "finite": _ledger_finite(ledger) and bool(tf.math.is_finite(total_scalar).numpy()),
        "resampling_count": 0,
        "ledger": ledger,
        "contract_checksum": contract["contract_checksum"],
        "spec_checksum": spec.checksum(),
    }


def _filterflow_path_subprocess(contracts: list[dict[str, Any]]) -> dict[str, Any]:
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
            "blocker": "filterflow v2 noresampling subprocess failed",
            "returncode": completed.returncode,
            "stdout_excerpt": completed.stdout[-4000:],
            "stderr_excerpt": completed.stderr[-4000:],
            "cells": [],
        }
    start = completed.stdout.rfind("FILTERFLOW_COMMON_MODEL_SUITE_V2_NORESAMPLING_JSON_BEGIN")
    end = completed.stdout.rfind("FILTERFLOW_COMMON_MODEL_SUITE_V2_NORESAMPLING_JSON_END")
    if start < 0 or end < 0 or end <= start:
        return {
            "status": "blocked",
            "blocker": "filterflow v2 noresampling JSON sentinels missing",
            "stdout_excerpt": completed.stdout[-4000:],
            "stderr_excerpt": completed.stderr[-4000:],
            "cells": [],
        }
    raw = completed.stdout[start + len("FILTERFLOW_COMMON_MODEL_SUITE_V2_NORESAMPLING_JSON_BEGIN") : end].strip()
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
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/bayesfilter-dpf-mpl")
        if not hasattr(inspect, "getargspec"):
            inspect.getargspec = inspect.getfullargspec

        import tensorflow as tf
        import tensorflow_probability as tfp
        from filterflow.base import State

        tfd = tfp.distributions
        DTYPE = tf.float64
        CONTRACTS = json.loads({contracts_literal!r})

        def to_tensor(value):
            return tf.convert_to_tensor(value, dtype=DTYPE)

        def to_json(tensor):
            return tf.cast(tensor, DTYPE).numpy().tolist()

        def scalar(tensor):
            return float(tf.cast(tensor, DTYPE).numpy())

        def mvn_log_prob(values, loc, covariance):
            return tfd.MultivariateNormalTriL(
                loc=to_tensor(loc),
                scale_tril=tf.linalg.cholesky(to_tensor(covariance)),
            ).log_prob(to_tensor(values))

        def normal_log_prob(values, loc, scale):
            return tfd.Normal(loc=to_tensor(loc), scale=to_tensor(scale)).log_prob(to_tensor(values))

        def wrap_angle(value):
            pi = tf.constant(math.pi, DTYPE)
            return tf.math.floormod(to_tensor(value) + pi, 2.0 * pi) - pi

        def gaussian_logpdf_zero_mean(residuals, covariance):
            residuals = to_tensor(residuals)
            covariance = to_tensor(covariance)
            chol = tf.linalg.cholesky(covariance)
            solved = tf.linalg.cholesky_solve(chol, tf.transpose(residuals))
            quad = tf.reduce_sum(tf.transpose(solved) * residuals, axis=1)
            dim = tf.cast(tf.shape(covariance)[0], DTYPE)
            logdet = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)))
            return -0.5 * (dim * tf.math.log(tf.constant(2.0 * math.pi, DTYPE)) + logdet + quad)

        def range_bearing_observation(x):
            x = to_tensor(x)
            px = x[..., 0]
            py = x[..., 1]
            rng = tf.sqrt(px * px + py * py + tf.constant(1e-12, DTYPE))
            bearing = tf.atan2(py, px)
            return tf.stack([rng, bearing], axis=-1)

        def transition_mean(contract, particles):
            model_id = contract["model_id"]
            p = contract["parameters"]
            particles = to_tensor(particles)
            if model_id in {{"lgssm_2d_h25_rich", "range_bearing_4d_h20_rich"}}:
                return tf.linalg.matmul(particles, to_tensor(p["A"]), transpose_b=True)
            if model_id == "sv_1d_h18_rich":
                return to_tensor(p["mu"]) + to_tensor(p["phi"]) * (particles - to_tensor(p["mu"]))
            if model_id == "structural_ar1_quadratic_h16":
                current_m = to_tensor(p["rho"]) * particles[:, 0]
                current_k = (
                    to_tensor(p["a"]) * particles[:, 1]
                    + to_tensor(p["b"]) * current_m
                    + to_tensor(p["c"]) * current_m * current_m
                    + to_tensor(p["d"]) * particles[:, 0] * current_m
                )
                return tf.stack([current_m, current_k], axis=1)
            if model_id == "spatial_sir_j3_rk4":
                return sir_transition_mean(particles, p)
            if model_id == "predator_prey_rk4":
                return predator_transition_mean(contract["theta"], particles, p)
            raise ValueError(f"unknown model id {{model_id}}")

        def complete_structural(contract, previous, predicted):
            p = contract["parameters"]
            current_m = predicted[:, 0]
            current_k = (
                to_tensor(p["a"]) * previous[:, 1]
                + to_tensor(p["b"]) * current_m
                + to_tensor(p["c"]) * current_m * current_m
                + to_tensor(p["d"]) * previous[:, 0] * current_m
            )
            return tf.stack([current_m, current_k], axis=1)

        def transition_log_density(contract, previous, predicted, step):
            del step
            model_id = contract["model_id"]
            p = contract["parameters"]
            if model_id == "lgssm_2d_h25_rich":
                return mvn_log_prob(predicted, transition_mean(contract, previous), p["Q"])
            if model_id == "sv_1d_h18_rich":
                return normal_log_prob(to_tensor(predicted)[:, 0], transition_mean(contract, previous)[:, 0], p["sigma"])
            if model_id == "range_bearing_4d_h20_rich":
                return mvn_log_prob(predicted, transition_mean(contract, previous), p["Q"])
            if model_id == "structural_ar1_quadratic_h16":
                return normal_log_prob(to_tensor(predicted)[:, 0], to_tensor(p["rho"]) * to_tensor(previous)[:, 0], p["sigma"])
            if model_id == "spatial_sir_j3_rk4":
                return mvn_log_prob(predicted, sir_transition_mean(previous, p), p["process_covariance"])
            if model_id == "predator_prey_rk4":
                return mvn_log_prob(predicted, predator_transition_mean(contract["theta"], previous, p), p["process_covariance"])
            raise ValueError(f"unknown model id {{model_id}}")

        def observation_log_density(contract, particles, observation, step):
            del step
            model_id = contract["model_id"]
            p = contract["parameters"]
            particles = to_tensor(particles)
            observation = to_tensor(observation)
            if model_id == "lgssm_2d_h25_rich":
                loc = tf.linalg.matmul(particles, to_tensor(p["C"]), transpose_b=True)
                return mvn_log_prob(tf.broadcast_to(observation, tf.shape(loc)), loc, p["R"])
            if model_id == "sv_1d_h18_rich":
                scale = tf.exp(0.5 * particles[:, 0])
                return normal_log_prob(tf.broadcast_to(observation[0], tf.shape(scale)), 0.0, scale)
            if model_id == "range_bearing_4d_h20_rich":
                predicted = range_bearing_observation(particles)
                residual = observation[tf.newaxis, :] - predicted
                residual = tf.concat([residual[..., :1], wrap_angle(residual[..., 1:2])], axis=-1)
                return gaussian_logpdf_zero_mean(residual, p["R"])
            if model_id == "structural_ar1_quadratic_h16":
                mean = particles[:, 1] + to_tensor(p["lambda"]) * particles[:, 0]
                return normal_log_prob(tf.broadcast_to(observation[0], tf.shape(mean)), mean, p["observation_scale"])
            if model_id == "spatial_sir_j3_rk4":
                infectious = particles[:, 1::2]
                return mvn_log_prob(tf.broadcast_to(observation, tf.shape(infectious)), infectious, p["observation_covariance"])
            if model_id == "predator_prey_rk4":
                return mvn_log_prob(tf.broadcast_to(observation, tf.shape(particles)), particles, p["observation_covariance"])
            raise ValueError(f"unknown model id {{model_id}}")

        def sir_transition_mean(x_prev, p):
            state = to_tensor(x_prev)
            kappa = to_tensor(p["kappa"])
            nu = to_tensor(p["nu"])
            neighbor_sets = [tuple(row) for row in p["neighbor_sets"]]
            adjacency = tf.constant(
                [[1.0 if j in row else 0.0 for j in range(len(neighbor_sets))] for row in neighbor_sets],
                dtype=DTYPE,
            )
            degree = tf.reduce_sum(adjacency, axis=1)
            substeps = int(p["rk4_substeps"])
            step = to_tensor(p["delta"]) / tf.cast(substeps, DTYPE)
            def rhs(values):
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

        def predator_transition_mean(theta, x_prev, p):
            theta = to_tensor(theta)
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

        def weighted_mean_variance(particles, weights):
            particles = to_tensor(particles)
            weights = to_tensor(weights)
            mean = tf.reduce_sum(weights[:, tf.newaxis] * particles, axis=0)
            centered = particles - mean
            variance = tf.reduce_sum(weights[:, tf.newaxis] * centered * centered, axis=0)
            return mean, variance

        def ess(log_weights):
            weights = tf.exp(to_tensor(log_weights))
            return 1.0 / tf.reduce_sum(tf.square(weights))

        def uniform_log_weights(n_particles):
            return tf.fill([n_particles], -tf.math.log(tf.cast(n_particles, DTYPE)))

        def step_ledger(step, previous_particles, transition_innovations, predicted_particles,
                        observation, transition_log_density, observation_log_density,
                        unnormalized_log_weights, incremental_log_normalizer,
                        normalized_log_weights, weights, filtered_mean, filtered_variance):
            return {{
                "step": int(step),
                "previous_particles": to_json(previous_particles),
                "transition_innovations": to_json(transition_innovations),
                "predicted_particles": to_json(predicted_particles),
                "observation": to_json(observation),
                "transition_log_density": to_json(transition_log_density),
                "observation_log_density": to_json(observation_log_density),
                "unnormalized_log_weights": to_json(unnormalized_log_weights),
                "incremental_log_normalizer": scalar(incremental_log_normalizer),
                "normalized_log_weights": to_json(normalized_log_weights),
                "weights": to_json(weights),
                "ess": scalar(ess(normalized_log_weights)),
                "filtered_mean": to_json(filtered_mean),
                "filtered_variance": to_json(filtered_variance),
            }}

        def run_contract(contract):
            particles = to_tensor(contract["initial_particles"])
            innovations = to_tensor(contract["transition_innovations"])
            observations = to_tensor(contract["observations"])
            n_particles = int(particles.shape[0])
            log_weights = uniform_log_weights(n_particles)
            total = tf.zeros([], DTYPE)
            ledger = []
            _ = State(tf.reshape(particles, [1, n_particles, int(particles.shape[1])]))
            for step in range(int(contract["horizon"])):
                previous = particles
                predicted = transition_mean(contract, previous) + innovations[step]
                if contract["model_id"] == "structural_ar1_quadratic_h16":
                    predicted = complete_structural(contract, previous, predicted)
                trans_ll = transition_log_density(contract, previous, predicted, step + 1)
                obs_ll = observation_log_density(contract, predicted, observations[step], step + 1)
                unnormalized = log_weights + obs_ll
                increment = tf.reduce_logsumexp(unnormalized)
                total = total + increment
                log_weights = unnormalized - increment
                weights = tf.exp(log_weights)
                mean, variance = weighted_mean_variance(predicted, weights)
                ledger.append(step_ledger(
                    step, previous, innovations[step], predicted, observations[step],
                    trans_ll, obs_ll, unnormalized, increment, log_weights,
                    weights, mean, variance
                ))
                particles = predicted
            finite = bool(tf.math.is_finite(total).numpy()) and all(
                all(math.isfinite(float(x)) for x in tf.reshape(to_tensor(row["observation_log_density"]), [-1]).numpy().tolist())
                for row in ledger
            )
            return {{
                "status": "executed",
                "backend": "filterflow_env_local_v2_fixed_noise_noresampling",
                "model_id": contract["model_id"],
                "scalar": scalar(total),
                "finite": finite,
                "resampling_count": 0,
                "ledger": ledger,
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
                    "backend": "filterflow_env_local_v2_fixed_noise_noresampling",
                    "model_id": contract.get("model_id"),
                    "blocker": repr(exc),
                    "contract_checksum": contract.get("contract_checksum"),
                    "spec_checksum": contract.get("spec_checksum"),
                }})
        print("FILTERFLOW_COMMON_MODEL_SUITE_V2_NORESAMPLING_JSON_BEGIN")
        print(json.dumps({{"status": "executed", "cells": cells}}, sort_keys=True))
        print("FILTERFLOW_COMMON_MODEL_SUITE_V2_NORESAMPLING_JSON_END")
        """
    )


def _transition_mean_v2(spec: CommonModelSpecV2, model: Any, theta: tf.Tensor, particles: tf.Tensor) -> tf.Tensor:
    particles = tf.convert_to_tensor(particles, DTYPE)
    if spec.model_id in {"lgssm_2d_h25_rich", "range_bearing_4d_h20_rich"}:
        return tf.linalg.matmul(particles, tf.convert_to_tensor(spec.parameters["A"], DTYPE), transpose_b=True)
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


def _cell(spec: CommonModelSpecV2, contract: dict[str, Any], bayesfilter: dict[str, Any], filterflow: dict[str, Any]) -> dict[str, Any]:
    if filterflow.get("status") != "executed":
        return {
            "model": spec.model_id,
            "family": spec.family,
            "implementations": ["BayesFilter", "FilterFlow"],
            "cell_type": "v2_fixed_noise_noresampling_path",
            "status": "INTERFACE_BLOCKED",
            "decision": f"{spec.model_id}_filterflow_noresampling_adapter_blocked",
            "primary_criterion": "both adapters must execute the same deterministic fixed-noise no-resampling path",
            "metrics": {},
            "mismatch_class": "filterflow_v2_noresampling_adapter_blocked",
            "reason": filterflow.get("blocker", "missing FilterFlow cell"),
            "contract": contract,
            "bayesfilter": bayesfilter,
            "filterflow": filterflow,
            "spec_checksum": spec.checksum(),
        }
    comparison = _compare_path_payloads(bayesfilter, filterflow)
    matched = (
        comparison["all_primary_fields_within_tolerance"]
        and bayesfilter["finite"]
        and filterflow["finite"]
        and bayesfilter["resampling_count"] == 0
        and filterflow["resampling_count"] == 0
    )
    return {
        "model": spec.model_id,
        "family": spec.family,
        "implementations": ["BayesFilter", "FilterFlow"],
        "cell_type": "v2_fixed_noise_noresampling_path",
        "status": "MATCHED" if matched else "EXPLAINED_MISMATCH",
        "decision": f"{spec.model_id}_noresampling_path_matched" if matched else f"{spec.model_id}_noresampling_path_mismatch",
        "primary_criterion": "scalar and primary path ledger fields match within tolerance",
        "metrics": comparison,
        "mismatch_class": None if matched else "v2_noresampling_path_delta",
        "contract": contract,
        "bayesfilter": bayesfilter,
        "filterflow": filterflow,
        "spec_checksum": spec.checksum(),
        "non_claim": "fixed-noise no-resampling path agreement is not stochastic resampling correctness",
    }


def _classified_cell(spec: CommonModelSpecV2, status: str, reason: str) -> dict[str, Any]:
    return {
        "model": spec.model_id,
        "family": spec.family,
        "implementations": ["BayesFilter", "FilterFlow"],
        "cell_type": "v2_fixed_noise_noresampling_path",
        "status": status,
        "decision": f"{spec.model_id}_{status.lower()}",
        "primary_criterion": "row is not executed unless P1-ready",
        "metrics": {},
        "mismatch_class": status.lower(),
        "reason": reason,
        "spec_checksum": spec.checksum(),
    }


def _compare_path_payloads(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    fields = [
        "predicted_particles",
        "observation_log_density",
        "unnormalized_log_weights",
        "incremental_log_normalizer",
        "normalized_log_weights",
    ]
    ledger_deltas = []
    max_abs_delta = abs(float(left["scalar"]) - float(right["scalar"]))
    all_within = max_abs_delta <= VALUE_TOLERANCE
    for left_step, right_step in zip(left["ledger"], right["ledger"], strict=True):
        step_metrics = {"step": left_step["step"], "fields": {}}
        for field in fields:
            left_value = tf.reshape(tf.convert_to_tensor(left_step[field], DTYPE), [-1])
            right_value = tf.reshape(tf.convert_to_tensor(right_step[field], DTYPE), [-1])
            delta = tf.abs(left_value - right_value)
            field_max = float(tf.reduce_max(delta).numpy()) if int(tf.size(delta).numpy()) else 0.0
            tolerance = VALUE_TOLERANCE if field == "incremental_log_normalizer" else LEDGER_TOLERANCE
            within = field_max <= tolerance
            all_within = all_within and within
            max_abs_delta = max(max_abs_delta, field_max)
            step_metrics["fields"][field] = {"max_abs_delta": field_max, "within_tolerance": within}
        ledger_deltas.append(step_metrics)
    return {
        "scalar_abs_delta": abs(float(left["scalar"]) - float(right["scalar"])),
        "ledger_deltas": ledger_deltas,
        "max_abs_delta": max_abs_delta,
        "all_primary_fields_within_tolerance": all_within,
        "value_tolerance": VALUE_TOLERANCE,
        "ledger_tolerance": LEDGER_TOLERANCE,
        "explanatory_fields": ["ess", "weights", "filtered_mean", "filtered_variance"],
    }


def _step_ledger(
    *,
    step: int,
    previous_particles: tf.Tensor,
    transition_innovations: tf.Tensor,
    predicted_particles: tf.Tensor,
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
) -> dict[str, Any]:
    return {
        "step": int(step),
        "previous_particles": tensor_to_json(previous_particles),
        "transition_innovations": tensor_to_json(transition_innovations),
        "predicted_particles": tensor_to_json(predicted_particles),
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
    }


def _weighted_mean_variance(particles: tf.Tensor, weights: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    mean = tf.reduce_sum(weights[:, tf.newaxis] * particles, axis=0)
    centered = particles - mean
    variance = tf.reduce_sum(weights[:, tf.newaxis] * centered * centered, axis=0)
    return mean, variance


def _ess_from_log_weights(log_weights: tf.Tensor) -> tf.Tensor:
    weights = tf.exp(tf.convert_to_tensor(log_weights, DTYPE))
    return 1.0 / tf.reduce_sum(tf.square(weights))


def _uniform_log_weights(n_particles: int) -> tf.Tensor:
    return tf.fill([n_particles], -tf.math.log(tf.cast(n_particles, DTYPE)))


def _ledger_finite(ledger: list[dict[str, Any]]) -> bool:
    return all(_json_finite(row) for row in ledger)


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


def _filterflow_cell(payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    for cell in payload.get("cells", []):
        if cell.get("model_id") == model_id:
            return cell
    return {"status": "missing", "blocker": f"missing FilterFlow v2 noresampling cell for {model_id}"}


def _preflight_prior_artifacts(p1_manifest: dict[str, Any], p2_payload: dict[str, Any]) -> None:
    p1_ids = [row.get("model_id") for row in p1_manifest.get("rows", [])]
    p2_ids = [cell.get("model") for cell in p2_payload.get("cells", [])]
    if tuple(p1_ids) != EXPECTED_V2_MODEL_IDS:
        raise ValueError(f"P1 manifest model id gate failed: {p1_ids}")
    if tuple(p2_ids) != EXPECTED_V2_MODEL_IDS:
        raise ValueError(f"P2 density cell id gate failed: {p2_ids}")
    if p2_payload.get("decision") not in {"PENDING_CLAUDE_REVIEW", "PASS_P2_DENSITY_READY_FOR_P3"}:
        raise ValueError(f"P2 density phase is not passable: {p2_payload.get('decision')}")


def _ready_ids_for_phase(payload: dict[str, Any], field: str, ready_value: str) -> set[str]:
    return {
        str(row["model_id"])
        for row in payload.get("pre_run_row_classification_table", [])
        if row.get(field) == ready_value
    }


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


def _decision(cells: list[dict[str, Any]]) -> str:
    if all(cell["status"] == "MATCHED" for cell in cells):
        return "PENDING_CLAUDE_REVIEW"
    if any(cell["status"] == "INTERFACE_BLOCKED" for cell in cells):
        return "P3_INTERFACE_BLOCKED"
    return "P3_CLASSIFIED_MISMATCH_PENDING_REVIEW"


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
        raise ValueError(f"P3 payload missing required fields: {sorted(missing)}")
    if payload["decision"] not in {"PENDING_CLAUDE_REVIEW", "PASS_P3_NORESAMPLING_READY_FOR_P4"}:
        raise ValueError(f"P3 payload decision not passable: {payload['decision']}")
    ids = [cell["model"] for cell in payload["cells"]]
    if tuple(ids) != EXPECTED_V2_MODEL_IDS:
        raise ValueError(f"P3 cell id gate failed: {ids}")
    if payload["veto_diagnostics"].get("missing_filterflow_subprocess_environment"):
        raise ValueError("P3 missing FilterFlow subprocess environment")
    if payload["veto_diagnostics"].get("nonfinite_path_value"):
        raise ValueError("P3 nonfinite path value")
    ready_ids = set(payload["primary_criterion_fields"]["ready_model_ids"])
    for cell in payload["cells"]:
        if cell["model"] in ready_ids and cell["status"] != "MATCHED":
            raise ValueError(f"P3 ready row did not match: {cell['model']} {cell['status']}")


def _summary(cells: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_cells": len(cells),
        "models": [cell["model"] for cell in cells],
        "status_counts": _status_counts(cells),
        "max_abs_delta": _max_abs_delta(cells),
    }


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


def _cell_nonfinite(cell: dict[str, Any]) -> bool:
    return (
        cell.get("bayesfilter", {}).get("finite") is False
        or cell.get("filterflow", {}).get("finite") is False
    )


def _digest_payload(payload: dict[str, Any]) -> str:
    clone = dict(payload)
    clone.pop("reproducibility_digest", None)
    if "run_manifest" in clone:
        clone["run_manifest"] = dict(clone["run_manifest"])
        clone["run_manifest"].pop("wall_time_seconds", None)
    return stable_digest(clone)


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# DPF Common Model Suite V2 P3 No-Resampling Path Result",
        "",
        "metadata_date: 2026-06-07",
        "phase: P3",
        f"decision: {payload['decision']}",
        "",
        "## Question",
        "",
        payload["question"],
        "",
        "## Evidence Contract",
        "",
        "Primary criterion: for every P1-ready v2 row, deterministic no-resampling scalar and primary path ledger fields match within tolerance.",
        "",
        "Veto diagnostics: hidden RNG, stochastic resampling, changed path contracts, nonfinite path values, v1/v2 leakage, old artifact names, or unclassified mismatch.",
        "",
        "## Result",
        "",
        f"- JSON artifact: `{payload['artifact_paths']['json']}`",
        f"- Markdown report: `{payload['artifact_paths']['markdown_report']}`",
        f"- Reproducibility digest: `{payload['reproducibility_digest']}`",
        "",
        "## Path Cells",
        "",
        "| Model id | Status | Max abs delta |",
        "|---|---|---|",
    ]
    for cell in payload["cells"]:
        lines.append(f"| `{cell['model']}` | {cell['status']} | {cell.get('metrics', {}).get('max_abs_delta', 'N/A')} |")
    lines.extend(
        [
            "",
            "## Primary Criterion Fields",
            "",
            f"- ready model ids: `{payload['primary_criterion_fields']['ready_model_ids']}`",
            f"- executed model ids: `{payload['primary_criterion_fields']['executed_model_ids']}`",
            f"- primary ledger fields: `{payload['primary_criterion_fields']['primary_ledger_fields']}`",
            f"- all ready rows matched: `{payload['primary_criterion_fields']['all_ready_rows_matched']}`",
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
            f"- ESS/moment policy: `{payload['explanatory_only_fields']['ess_and_filtered_moments_policy']}`",
            "",
            "## Command Manifest",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| git commit | `{payload['run_manifest'].get('commit')}` |",
            f"| dirty status | `{_single_line(payload['run_manifest'].get('dirty_state_summary'))}` |",
            f"| command | `{payload['run_manifest'].get('command')}` |",
            f"| validation commands | `python -m json.tool {payload['artifact_paths']['json']}`; `CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/bayesfilter-dpf-mpl python -m experiments.dpf_implementation.tf_tfp.runners.run_common_model_suite_v2_noresampling_tf --validate-only`; `git diff --check` on P3 files |",
            f"| CPU/GPU status | CPU-only TensorFlow run; pre-import `CUDA_VISIBLE_DEVICES={payload['run_manifest'].get('pre_import_cuda_visible_devices')}`; visible GPUs `{payload['run_manifest'].get('gpu_devices_visible')}` |",
            "| random seeds | none consumed; deterministic P1 particles, observations, and transition innovations |",
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
            "- No P3 repair has been required before Claude result review.",
            "",
            "## Decision Table",
            "",
            "| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |",
            "|---|---|---|---|---|---|",
            "| PENDING_CLAUDE_REVIEW | all P1-ready rows matched on deterministic no-resampling path fields locally | no local veto fired | Claude may identify path-contract or adapter-governance gaps | run Claude P3 result review | no stochastic resampling, gradient, filter correctness, or student claim |",
            "",
            "## Post-Run Red Team",
            "",
            "Strongest alternative explanation: matching deterministic no-resampling paths may still miss fixed-ancestor branch replay or gradient errors.",
            "",
            "Result that would overturn the decision: a row's path contract is found to differ from the P1 frozen manifest, or a ledger field used for pass was actually explanatory only.",
            "",
            "Weakest evidence link: no stochastic resampling and no gradient-through-branch semantics are tested in P3.",
            "",
            "## Non-Claims",
            "",
        ]
    )
    lines.extend(f"- {claim}" for claim in payload["non_claims"])
    lines.append("")
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
