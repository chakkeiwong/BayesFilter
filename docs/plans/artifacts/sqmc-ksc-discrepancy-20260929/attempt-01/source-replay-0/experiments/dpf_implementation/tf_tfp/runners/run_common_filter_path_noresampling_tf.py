"""Run fixed-noise no-resampling filter-path tie-outs for common models."""

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
    CommonModelSpec,
    bayesfilter_model_for_spec,
    common_model_specs,
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
    FILTERFLOW_REFERENCE_DTYPE,
    reference_policy,
    validate_filterflow_reference_status,
)


DTYPE = tf.float64
PLAN_PATH = "docs/plans/bayesfilter-dpf-common-filter-path-noresampling-plan-2026-06-06.md"
RESULT_PATH = "docs/plans/bayesfilter-dpf-common-filter-path-noresampling-result-2026-06-06.md"
JSON_PATH = OUTPUT_DIR / "dpf_common_filter_path_noresampling_2026-06-06.json"
REPORT_PATH = REPORT_DIR / "dpf-common-filter-path-noresampling-2026-06-06.md"
FILTERFLOW_ENV_PYTHON = REPO_ROOT / ".localenv" / "filterflow-py311" / "bin" / "python"
FILTERFLOW_PATH = REPO_ROOT / ".localsource" / "filterflow"
FILTERFLOW_MARKER_PATH = FILTERFLOW_PATH / FILTERFLOW_BRANCH_MARKER
HORIZON = 3
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
    filterflow_status = _filterflow_checkout_manifest()
    specs = common_model_specs()
    contracts = [_filter_path_contract(spec) for spec in specs]
    filterflow_payload = _filterflow_subprocess(contracts)
    cells = []
    for spec, contract in zip(specs, contracts, strict=True):
        bayesfilter = _bayesfilter_filter_path(spec, contract)
        filterflow = _filterflow_cell(filterflow_payload, spec.model_id)
        cells.append(_cell(spec, contract, bayesfilter, filterflow))
    decision = _decision(cells)
    return {
        "decision": decision,
        "created_at_utc": utc_now(),
        "question": (
            "Fixed-noise no-resampling bootstrap filter-path tie-out across "
            "BayesFilter and executable float64 FilterFlow for the common model suite."
        ),
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "filterflow_reference_policy": reference_policy(),
        "filterflow_status": filterflow_status,
        "filterflow_payload_status": filterflow_payload.get("status"),
        "tolerances": {"value_abs": VALUE_TOLERANCE, "ledger_abs": LEDGER_TOLERANCE},
        "contracts": contracts,
        "cells": cells,
        "summary": _summary(cells),
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners.run_common_filter_path_noresampling_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_claims": [
            "no filtering-algorithm correctness claim",
            "no implementation is treated as an oracle",
            "no resampling correctness claim",
            "no gradient correctness claim",
            "range-bearing FilterFlow coverage uses a local subprocess adapter",
            "no student-repository tie-out claim",
            "no TT-filter correctness claim",
            "no paper-scale, HMC, DSGE, GPU, or production-readiness claim",
        ],
    }


def _filter_path_contract(spec: CommonModelSpec) -> dict[str, Any]:
    innovations = _fixed_transition_innovations(spec)
    observations = _fixed_observations(spec)
    payload = {
        "model_id": spec.model_id,
        "family": spec.family,
        "dtype": DTYPE.name,
        "horizon": HORIZON,
        "num_particles": int(spec.x0.shape[0]),
        "state_dim": int(spec.x0.shape[1]),
        "initial_particles": tensor_to_json(spec.x0),
        "transition_innovations": tensor_to_json(innovations),
        "observations": tensor_to_json(observations),
        "initial_log_weight_policy": "uniform_normalized_log_weights",
        "proposal": "bootstrap_fixed_transition_innovations",
        "resampling_policy": "disabled",
        "scalar": "sum of per-step predictive log normalizers",
        "spec": spec.payload(),
        "spec_checksum": spec.checksum(),
    }
    payload["contract_checksum"] = stable_digest(payload)
    return payload


def _fixed_transition_innovations(spec: CommonModelSpec) -> tf.Tensor:
    if spec.model_id == "lgssm_2d_linear":
        values = [
            [[0.020, -0.030], [-0.015, 0.040], [0.035, -0.020]],
            [[-0.010, 0.015], [0.025, -0.035], [-0.030, 0.020]],
            [[0.005, 0.025], [-0.020, 0.010], [0.015, -0.030]],
        ]
    elif spec.model_id == "sv_1d_synthetic":
        values = [
            [[0.10], [-0.05], [0.08]],
            [[-0.07], [0.12], [-0.04]],
            [[0.03], [-0.09], [0.11]],
        ]
    elif spec.model_id == "range_bearing_2d_cv":
        values = [
            [[0.006, -0.004, 0.002, -0.001], [-0.005, 0.003, -0.001, 0.002], [0.004, 0.002, 0.001, -0.002]],
            [[-0.003, 0.005, 0.001, 0.001], [0.002, -0.006, -0.002, 0.001], [0.005, -0.001, 0.002, -0.001]],
            [[0.004, 0.001, -0.001, 0.002], [-0.002, 0.004, 0.001, -0.002], [0.003, -0.003, -0.001, 0.001]],
        ]
    else:
        raise ValueError(f"unknown common model spec: {spec.model_id}")
    innovations = tf.constant(values, dtype=DTYPE)
    expected_shape = (HORIZON, int(spec.x0.shape[0]), int(spec.x0.shape[1]))
    if innovations.shape != expected_shape:
        raise ValueError(f"{spec.model_id}: invalid innovation shape {innovations.shape}")
    return innovations


def _fixed_observations(spec: CommonModelSpec) -> tf.Tensor:
    if spec.model_id == "lgssm_2d_linear":
        values = [[0.17], [0.06], [0.14]]
    elif spec.model_id == "sv_1d_synthetic":
        values = [[0.17], [-0.11], [0.21]]
    elif spec.model_id == "range_bearing_2d_cv":
        values = [[1.29, 0.50], [1.32, 0.47], [1.27, 0.52]]
    else:
        raise ValueError(f"unknown common model spec: {spec.model_id}")
    observations = tf.constant(values, dtype=DTYPE)
    if observations.shape[0] != HORIZON:
        raise ValueError(f"{spec.model_id}: invalid observation horizon {observations.shape}")
    return observations


def _bayesfilter_filter_path(spec: CommonModelSpec, contract: dict[str, Any]) -> dict[str, Any]:
    model = bayesfilter_model_for_spec(spec)
    theta = tf.convert_to_tensor(spec.theta, DTYPE)
    particles = tf.convert_to_tensor(contract["initial_particles"], DTYPE)
    innovations = tf.convert_to_tensor(contract["transition_innovations"], DTYPE)
    observations = tf.convert_to_tensor(contract["observations"], DTYPE)
    n_particles = int(particles.shape[0])
    log_weights = tf.fill([n_particles], -tf.math.log(tf.cast(n_particles, DTYPE)))
    total_scalar = tf.zeros([], DTYPE)
    ledger = []
    for step in range(HORIZON):
        previous_particles = particles
        predicted_particles = _transition_mean(spec, previous_particles) + innovations[step]
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
    finite = _ledger_finite(ledger) and bool(tf.math.is_finite(total_scalar).numpy())
    return {
        "status": "executed",
        "backend": "bayesfilter_highdim_fixed_noise_noresampling",
        "model_id": spec.model_id,
        "scalar": scalar(total_scalar),
        "finite": finite,
        "resampling_count": 0,
        "ledger": ledger,
        "contract_checksum": contract["contract_checksum"],
        "spec_checksum": spec.checksum(),
    }


def _transition_mean(spec: CommonModelSpec, particles: tf.Tensor) -> tf.Tensor:
    particles = tf.convert_to_tensor(particles, DTYPE)
    if spec.model_id in {"lgssm_2d_linear", "range_bearing_2d_cv"}:
        return tf.linalg.matmul(particles, tf.convert_to_tensor(spec.parameters["A"], DTYPE), transpose_b=True)
    if spec.model_id == "sv_1d_synthetic":
        gamma = tf.constant(float(spec.parameters["gamma"]), DTYPE)
        mu = tf.constant(float(spec.parameters.get("mu", 0.0)), DTYPE)
        return mu + gamma * (particles - mu)
    raise ValueError(f"unknown common model spec: {spec.model_id}")


def _filterflow_subprocess(contracts: list[dict[str, Any]]) -> dict[str, Any]:
    if not FILTERFLOW_ENV_PYTHON.exists():
        return {
            "status": "blocked",
            "blocker": f"missing filterflow env python: {FILTERFLOW_ENV_PYTHON}",
            "cells": [],
        }
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env["PYTHONPATH"] = str(FILTERFLOW_PATH)
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
            "blocker": "filterflow common filter-path subprocess failed",
            "returncode": completed.returncode,
            "stdout_excerpt": completed.stdout[-4000:],
            "stderr_excerpt": completed.stderr[-4000:],
            "cells": [],
        }
    start = completed.stdout.rfind("FILTERFLOW_COMMON_FILTER_PATH_JSON_BEGIN")
    end = completed.stdout.rfind("FILTERFLOW_COMMON_FILTER_PATH_JSON_END")
    if start < 0 or end < 0 or end <= start:
        return {
            "status": "blocked",
            "blocker": "filterflow common filter-path JSON sentinels missing",
            "stdout_excerpt": completed.stdout[-4000:],
            "stderr_excerpt": completed.stderr[-4000:],
            "cells": [],
        }
    raw = completed.stdout[start + len("FILTERFLOW_COMMON_FILTER_PATH_JSON_BEGIN") : end].strip()
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
        from filterflow.models.stochastic_volatility import SVObservationModel, SVTransitionModel
        from filterflow.observation.linear import LinearObservationModel
        from filterflow.transition.random_walk import RandomWalkModel

        tfd = tfp.distributions
        DTYPE = tf.float64
        HORIZON = {HORIZON}
        CONTRACTS = json.loads({contracts_literal!r})

        def to_tensor(value):
            return tf.convert_to_tensor(value, dtype=DTYPE)

        def to_json(tensor):
            return tf.cast(tensor, DTYPE).numpy().tolist()

        def scalar(tensor):
            return float(tf.cast(tensor, DTYPE).numpy())

        def state(rows, log_weights=None):
            tensor = to_tensor(rows)
            particles = tf.reshape(tensor, [1, int(tensor.shape[0]), int(tensor.shape[1])])
            if log_weights is None:
                return State(particles)
            return State(particles, log_weights=tf.reshape(to_tensor(log_weights), [1, int(tensor.shape[0])]))

        def mvn_log_prob(values, loc, covariance):
            return tfd.MultivariateNormalTriL(
                loc=to_tensor(loc),
                scale_tril=tf.linalg.cholesky(to_tensor(covariance)),
            ).log_prob(to_tensor(values))

        def gaussian_logpdf_zero_mean(residuals, covariance):
            residuals = to_tensor(residuals)
            covariance = to_tensor(covariance)
            chol = tf.linalg.cholesky(covariance)
            solved = tf.linalg.cholesky_solve(chol, tf.transpose(residuals))
            quad = tf.reduce_sum(tf.transpose(solved) * residuals, axis=1)
            dim = tf.cast(tf.shape(covariance)[0], DTYPE)
            logdet = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)))
            return -0.5 * (dim * tf.math.log(tf.constant(2.0 * math.pi, DTYPE)) + logdet + quad)

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

        def range_bearing_obs_ll(x_obs, observation, covariance):
            predicted = range_bearing_observation(x_obs)
            residual = to_tensor(observation)[tf.newaxis, :] - predicted
            residual = tf.concat([residual[..., :1], wrap_angle(residual[..., 1:2])], axis=-1)
            return gaussian_logpdf_zero_mean(residual, covariance)

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
                "resampling_applied": False,
            }}

        def common_loop(contract, transition_mean_fn, transition_ll_fn, observation_ll_fn, backend):
            particles = to_tensor(contract["initial_particles"])
            innovations = to_tensor(contract["transition_innovations"])
            observations = to_tensor(contract["observations"])
            n_particles = int(particles.shape[0])
            log_weights = tf.fill([n_particles], -tf.math.log(tf.cast(n_particles, DTYPE)))
            total_scalar = tf.zeros([], DTYPE)
            ledger = []
            for step in range(HORIZON):
                previous_particles = particles
                predicted_particles = transition_mean_fn(previous_particles) + innovations[step]
                transition_log_density = transition_ll_fn(previous_particles, predicted_particles)
                observation_log_density = observation_ll_fn(predicted_particles, observations[step])
                unnormalized = log_weights + observation_log_density
                increment = tf.reduce_logsumexp(unnormalized)
                total_scalar = total_scalar + increment
                log_weights = unnormalized - increment
                weights = tf.exp(log_weights)
                mean, variance = weighted_mean_variance(predicted_particles, weights)
                ledger.append(step_ledger(
                    step, previous_particles, innovations[step], predicted_particles,
                    observations[step], transition_log_density, observation_log_density,
                    unnormalized, increment, log_weights, weights, mean, variance,
                ))
                particles = predicted_particles
            finite = bool(
                tf.math.is_finite(total_scalar).numpy()
                and all(
                    all(
                        tf.reduce_all(tf.math.is_finite(to_tensor(step[key]))).numpy()
                        for key in [
                            "previous_particles",
                            "transition_innovations",
                            "predicted_particles",
                            "observation",
                            "transition_log_density",
                            "observation_log_density",
                            "unnormalized_log_weights",
                            "normalized_log_weights",
                            "weights",
                            "filtered_mean",
                            "filtered_variance",
                        ]
                    )
                    and math.isfinite(float(step["incremental_log_normalizer"]))
                    and math.isfinite(float(step["ess"]))
                    for step in ledger
                )
            )
            return {{
                "status": "executed",
                "backend": backend,
                "model_id": contract["model_id"],
                "scalar": scalar(total_scalar),
                "finite": finite,
                "resampling_count": 0,
                "ledger": ledger,
                "contract_checksum": contract["contract_checksum"],
                "spec_checksum": contract["spec_checksum"],
            }}

        def eval_lgssm(contract):
            p = contract["spec"]["parameters"]
            transition_model = RandomWalkModel(
                to_tensor(p["A"]),
                tfd.MultivariateNormalTriL(
                    loc=tf.zeros([len(p["m0"])], dtype=DTYPE),
                    scale_tril=tf.linalg.cholesky(to_tensor(p["Q"])),
                ),
            )
            observation_model = LinearObservationModel(
                to_tensor(p["C"]),
                tfd.MultivariateNormalTriL(
                    loc=tf.zeros([len(p["R"])], dtype=DTYPE),
                    scale_tril=tf.linalg.cholesky(to_tensor(p["R"])),
                ),
            )
            def transition_mean(particles):
                return tf.linalg.matmul(to_tensor(particles), to_tensor(p["A"]), transpose_b=True)
            def transition_ll(previous_particles, predicted_particles):
                return tf.reshape(
                    transition_model.loglikelihood(
                        state(previous_particles),
                        state(predicted_particles),
                        tf.constant([], dtype=DTYPE),
                    ),
                    [-1],
                )
            def observation_ll(predicted_particles, observation):
                return tf.reshape(
                    observation_model.loglikelihood(state(predicted_particles), to_tensor(observation)),
                    [-1],
                )
            return common_loop(contract, transition_mean, transition_ll, observation_ll, "filterflow_builtin_linear_gaussian_noresampling")

        def eval_sv(contract):
            p = contract["spec"]["parameters"]
            gamma = tf.constant([[p["gamma"]]], dtype=DTYPE)
            sigma = tf.constant([[p["sigma"]]], dtype=DTYPE)
            beta = tf.constant([[p["beta"]]], dtype=DTYPE)
            mu = tf.constant([p["mu"]], dtype=DTYPE)
            transition_model = SVTransitionModel(mu, gamma, sigma)
            observation_model = SVObservationModel(beta)
            def transition_mean(particles):
                return mu[tf.newaxis, :] + tf.reshape(tf.constant(p["gamma"], DTYPE), [1, 1]) * (to_tensor(particles) - mu[tf.newaxis, :])
            def transition_ll(previous_particles, predicted_particles):
                return tf.reshape(
                    transition_model.loglikelihood(
                        state(previous_particles),
                        state(predicted_particles),
                        tf.constant([], dtype=DTYPE),
                    ),
                    [-1],
                )
            def observation_ll(predicted_particles, observation):
                return tf.reshape(
                    observation_model.loglikelihood(state(predicted_particles), to_tensor(observation)),
                    [-1],
                )
            return common_loop(contract, transition_mean, transition_ll, observation_ll, "filterflow_builtin_sv_noresampling")

        def eval_range_bearing(contract):
            p = contract["spec"]["parameters"]
            transition_model = RandomWalkModel(
                to_tensor(p["A"]),
                tfd.MultivariateNormalTriL(
                    loc=tf.zeros([len(p["m0"])], dtype=DTYPE),
                    scale_tril=tf.linalg.cholesky(to_tensor(p["Q"])),
                ),
            )
            def transition_mean(particles):
                return tf.linalg.matmul(to_tensor(particles), to_tensor(p["A"]), transpose_b=True)
            def transition_ll(previous_particles, predicted_particles):
                return tf.reshape(
                    transition_model.loglikelihood(
                        state(previous_particles),
                        state(predicted_particles),
                        tf.constant([], dtype=DTYPE),
                    ),
                    [-1],
                )
            def observation_ll(predicted_particles, observation):
                return range_bearing_obs_ll(predicted_particles, observation, p["R"])
            return common_loop(contract, transition_mean, transition_ll, observation_ll, "filterflow_local_range_bearing_noresampling")

        cells = []
        for contract in CONTRACTS:
            if contract["model_id"] == "lgssm_2d_linear":
                cells.append(eval_lgssm(contract))
            elif contract["model_id"] == "sv_1d_synthetic":
                cells.append(eval_sv(contract))
            elif contract["model_id"] == "range_bearing_2d_cv":
                cells.append(eval_range_bearing(contract))
            else:
                cells.append({{"status": "blocked", "model_id": contract["model_id"], "blocker": "unknown common contract"}})

        payload = {{
            "status": "executed",
            "dtype": DTYPE.name,
            "pre_import_cuda_visible_devices": PRE_IMPORT_CUDA_VISIBLE_DEVICES,
            "gpu_devices_visible": [str(device) for device in tf.config.list_physical_devices("GPU")],
            "cells": cells,
        }}
        print("FILTERFLOW_COMMON_FILTER_PATH_JSON_BEGIN")
        print(json.dumps(payload, sort_keys=True))
        print("FILTERFLOW_COMMON_FILTER_PATH_JSON_END")
        """
    )


def _cell(
    spec: CommonModelSpec,
    contract: dict[str, Any],
    bayesfilter: dict[str, Any],
    filterflow: dict[str, Any],
) -> dict[str, Any]:
    if filterflow.get("status") != "executed":
        return {
            "model": spec.model_id,
            "family": spec.family,
            "implementations": ["BayesFilter", "FilterFlow"],
            "cell_type": "fixed_noise_noresampling_filter_path",
            "status": "INTERFACE_BLOCKED",
            "decision": f"{spec.model_id}_filterflow_filter_path_blocked",
            "primary_criterion": "both adapters must execute the same fixed-noise no-resampling filter path",
            "metrics": {},
            "mismatch_class": "filterflow_common_filter_path_blocked",
            "reason": filterflow.get("blocker", "missing FilterFlow cell"),
            "bayesfilter": bayesfilter,
            "filterflow": filterflow,
            "contract_checksum": contract["contract_checksum"],
            "spec_checksum": spec.checksum(),
        }
    comparison = _compare_filter_paths(bayesfilter, filterflow)
    matched = (
        comparison["all_fields_within_tolerance"]
        and bayesfilter["finite"]
        and filterflow["finite"]
        and bayesfilter["resampling_count"] == 0
        and filterflow["resampling_count"] == 0
    )
    return {
        "model": spec.model_id,
        "family": spec.family,
        "implementations": ["BayesFilter", "FilterFlow"],
        "cell_type": "fixed_noise_noresampling_filter_path",
        "status": "MATCHED" if matched else "EXPLAINED_MISMATCH",
        "decision": f"{spec.model_id}_filter_path_matched" if matched else f"{spec.model_id}_filter_path_mismatch",
        "primary_criterion": "fixed-noise no-resampling value path fields match within tolerance",
        "metrics": comparison,
        "mismatch_class": None if matched else "common_filter_path_delta_or_veto",
        "bayesfilter": bayesfilter,
        "filterflow": filterflow,
        "contract": contract,
        "contract_checksum": contract["contract_checksum"],
        "spec_checksum": spec.checksum(),
        "non_claim": "no-resampling value-path agreement is not a resampling or gradient equality claim",
    }


def _filterflow_cell(payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    if payload.get("status") != "executed":
        return {
            "status": "blocked",
            "model_id": model_id,
            "blocker": payload.get("blocker", "filterflow subprocess did not execute"),
        }
    for cell in payload.get("cells", []):
        if cell.get("model_id") == model_id:
            return cell
    return {"status": "blocked", "model_id": model_id, "blocker": "missing model cell in filterflow payload"}


def _compare_filter_paths(bayesfilter: dict[str, Any], filterflow: dict[str, Any]) -> dict[str, Any]:
    fields = [
        "previous_particles",
        "transition_innovations",
        "predicted_particles",
        "observation",
        "transition_log_density",
        "observation_log_density",
        "unnormalized_log_weights",
        "normalized_log_weights",
        "weights",
        "filtered_mean",
        "filtered_variance",
    ]
    step_deltas = []
    field_maxima: dict[str, float] = {field: 0.0 for field in fields}
    field_maxima["incremental_log_normalizer"] = 0.0
    field_maxima["ess"] = 0.0
    for left, right in zip(bayesfilter["ledger"], filterflow["ledger"], strict=True):
        deltas = {field: _max_abs_delta(left[field], right[field]) for field in fields}
        deltas["incremental_log_normalizer"] = abs(
            float(left["incremental_log_normalizer"]) - float(right["incremental_log_normalizer"])
        )
        deltas["ess"] = abs(float(left["ess"]) - float(right["ess"]))
        for field, value in deltas.items():
            field_maxima[field] = max(field_maxima[field], value)
        step_deltas.append({"step": left["step"], "field_max_abs_delta": deltas, "max_abs_delta": max(deltas.values())})
    scalar_delta = abs(float(bayesfilter["scalar"]) - float(filterflow["scalar"]))
    max_abs_delta = max([scalar_delta, *field_maxima.values()])
    return {
        "scalar_abs_delta": scalar_delta,
        "field_max_abs_delta": field_maxima,
        "step_deltas": step_deltas,
        "max_abs_delta": max_abs_delta,
        "all_fields_within_tolerance": max_abs_delta <= LEDGER_TOLERANCE,
        "filterflow_finite": bool(filterflow.get("finite")),
        "bayesfilter_finite": bool(bayesfilter.get("finite")),
        "resampling_counts": {
            "bayesfilter": int(bayesfilter.get("resampling_count", -1)),
            "filterflow": int(filterflow.get("resampling_count", -1)),
        },
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
        "resampling_applied": False,
    }


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


def _ledger_finite(ledger: list[dict[str, Any]]) -> bool:
    tensor_keys = [
        "previous_particles",
        "transition_innovations",
        "predicted_particles",
        "observation",
        "transition_log_density",
        "observation_log_density",
        "unnormalized_log_weights",
        "normalized_log_weights",
        "weights",
        "filtered_mean",
        "filtered_variance",
    ]
    scalar_keys = ["incremental_log_normalizer", "ess"]
    for step in ledger:
        for key in tensor_keys:
            if not bool(tf.reduce_all(tf.math.is_finite(tf.convert_to_tensor(step[key], DTYPE))).numpy()):
                return False
        for key in scalar_keys:
            if not math.isfinite(float(step[key])):
                return False
    return True


def _decision(cells: list[dict[str, Any]]) -> str:
    if any(cell["status"] == "EXPLAINED_MISMATCH" and not cell.get("mismatch_class") for cell in cells):
        return "common_filter_path_unclassified_mismatch_veto"
    if all(cell["status"] == "MATCHED" for cell in cells):
        return "common_filter_path_noresampling_all_matched"
    if any(cell["status"] == "EXPLAINED_MISMATCH" for cell in cells):
        return "common_filter_path_noresampling_explained_mismatch"
    return "common_filter_path_noresampling_blocked"


def _summary(cells: list[dict[str, Any]]) -> dict[str, Any]:
    statuses: dict[str, int] = {}
    for cell in cells:
        statuses[cell["status"]] = statuses.get(cell["status"], 0) + 1
    return {
        "num_cells": len(cells),
        "status_counts": statuses,
        "models": [cell["model"] for cell in cells],
        "max_abs_delta": max(float(cell.get("metrics", {}).get("max_abs_delta", 0.0)) for cell in cells),
        "scalar_abs_deltas": {
            cell["model"]: cell.get("metrics", {}).get("scalar_abs_delta", "N/A") for cell in cells
        },
    }


def _validate_payload(payload: dict[str, Any]) -> None:
    if payload["run_manifest"]["pre_import_cuda_visible_devices"] != "-1":
        raise RuntimeError("CPU-only pre-import manifest missing")
    if payload["run_manifest"]["gpu_devices_visible"] != []:
        raise RuntimeError("GPU visible in CPU-only run")
    if "reproducibility_digest" not in payload:
        raise RuntimeError("missing reproducibility digest")
    validate_filterflow_reference_status(payload["filterflow_status"], marker_path=FILTERFLOW_MARKER_PATH)
    if payload["filterflow_reference_policy"]["dtype"] != FILTERFLOW_REFERENCE_DTYPE:
        raise RuntimeError("wrong FilterFlow dtype policy")
    if payload["decision"] != "common_filter_path_noresampling_all_matched":
        raise RuntimeError(payload["decision"])
    if len(payload["cells"]) != 3:
        raise RuntimeError("expected three common model cells")
    if len(payload["contracts"]) != 3:
        raise RuntimeError("expected three common model contracts")
    for cell in payload["cells"]:
        if cell["status"] != "MATCHED":
            raise RuntimeError(f"non-matched common filter-path cell: {cell['model']}")
        if cell["metrics"]["max_abs_delta"] > LEDGER_TOLERANCE:
            raise RuntimeError(f"common filter-path tolerance failed: {cell['model']}")
        counts = cell["metrics"]["resampling_counts"]
        if counts["bayesfilter"] != 0 or counts["filterflow"] != 0:
            raise RuntimeError(f"unexpected resampling count: {cell['model']}")
        if len(cell["bayesfilter"]["ledger"]) != HORIZON or len(cell["filterflow"]["ledger"]) != HORIZON:
            raise RuntimeError(f"wrong horizon length: {cell['model']}")


def _markdown(payload: dict[str, Any]) -> str:
    return f"""# DPF Common Filter-Path No-Resampling Result

metadata_date: 2026-06-06

## Decision

`{payload['decision']}`

## Summary

- Cells: `{payload['summary']['num_cells']}`
- Status counts: `{payload['summary']['status_counts']}`
- Max absolute filter-path delta: `{payload['summary']['max_abs_delta']}`
- Scalar absolute deltas: `{payload['summary']['scalar_abs_deltas']}`

## Cell Table

{_cell_table(payload['cells'])}

## Interpretation

BayesFilter and executable float64 FilterFlow agree on the deterministic
fixed-noise no-resampling bootstrap value path for the common LGSSM,
stochastic-volatility, and range-bearing model suite.  This is the next
stronger common-sense tie-out after density-component equality: the two sides
now match the propagated particles, likelihood ledgers, normalized weights,
predictive log-normalizer scalar, ESS, and weighted moments under a shared
contract.

The result is intentionally scoped.  It does not test resampling, pathwise
gradients, random-number generator equivalence, student repositories, TT
filters, or scientific correctness of the filtering algorithm.

## Non-Claims

{_bullet_list(payload['non_claims'])}
"""


def _cell_table(cells: list[dict[str, Any]]) -> str:
    lines = [
        "| model | family | status | scalar delta | max path delta | backend note |",
        "|---|---|---:|---:|---:|---|",
    ]
    for cell in cells:
        backend = cell.get("filterflow", {}).get("backend", "N/A")
        metrics = cell.get("metrics", {})
        lines.append(
            "| {model} | {family} | `{status}` | `{scalar_delta}` | `{path_delta}` | {backend} |".format(
                model=cell["model"],
                family=cell["family"],
                status=cell["status"],
                scalar_delta=metrics.get("scalar_abs_delta", "N/A"),
                path_delta=metrics.get("max_abs_delta", "N/A"),
                backend=backend,
            )
        )
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
    return {
        "path": str(FILTERFLOW_PATH),
        "status": "current_local_float64_reference_checkout",
        "commit": _git_filterflow(["rev-parse", "HEAD"]),
        "branch": _git_filterflow(["rev-parse", "--abbrev-ref", "HEAD"]),
        "status_short": _git_filterflow(["status", "--short"]),
        "marker_exists": FILTERFLOW_MARKER_PATH.exists(),
        "provenance_note": "local float64 FilterFlow comparator, not pristine upstream and not an oracle",
    }


def _git_filterflow(args: list[str]) -> str:
    completed = subprocess.run(
        ["git", "-C", str(FILTERFLOW_PATH), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _digest_payload(payload: dict[str, Any]) -> str:
    comparable = dict(payload)
    comparable["created_at_utc"] = "TIMESTAMP"
    comparable["run_manifest"] = dict(comparable["run_manifest"])
    comparable["run_manifest"]["wall_time_seconds"] = "WALL_TIME"
    comparable["run_manifest"]["dirty_state_summary"] = "DIRTY_STATE"
    comparable["filterflow_status"] = dict(comparable["filterflow_status"])
    comparable["filterflow_status"]["status_short"] = "FILTERFLOW_STATUS"
    return stable_digest(comparable)


def _max_abs_delta(left: Any, right: Any) -> float:
    left_tensor = tf.convert_to_tensor(left, dtype=DTYPE)
    right_tensor = tf.convert_to_tensor(right, dtype=DTYPE)
    return scalar(tf.reduce_max(tf.abs(left_tensor - right_tensor)))


def _bullet_list(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


if __name__ == "__main__":
    raise SystemExit(main())
