"""Run fixed-branch gradient tie-outs for common BayesFilter/FilterFlow models."""

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

from bayesfilter import highdim
from experiments.dpf_implementation.tf_tfp.fixtures.common_model_suite_tf import (
    CommonModelSpec,
    CommonRangeBearingSSM,
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
from experiments.dpf_implementation.tf_tfp.runners.run_common_filter_path_noresampling_tf import (
    _fixed_observations,
    _fixed_transition_innovations,
)


DTYPE = tf.float64
PLAN_PATH = "docs/plans/bayesfilter-dpf-common-fixed-branch-gradient-plan-2026-06-06.md"
RESULT_PATH = "docs/plans/bayesfilter-dpf-common-fixed-branch-gradient-result-2026-06-06.md"
JSON_PATH = OUTPUT_DIR / "dpf_common_fixed_branch_gradient_2026-06-06.json"
REPORT_PATH = REPORT_DIR / "dpf-common-fixed-branch-gradient-2026-06-06.md"
FILTERFLOW_ENV_PYTHON = REPO_ROOT / ".localenv" / "filterflow-py311" / "bin" / "python"
FILTERFLOW_PATH = REPO_ROOT / ".localsource" / "filterflow"
FILTERFLOW_MARKER_PATH = FILTERFLOW_PATH / FILTERFLOW_BRANCH_MARKER
HORIZON = 3
RESAMPLING_FLAGS = (False, True, False)
FIXED_ANCESTORS = (0, 0, 2)
FD_STEP = 1e-5
VALUE_TOLERANCE = 5e-10
GRADIENT_TOLERANCE = 5e-8
FD_TOLERANCE = 5e-5


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
    contracts = [_gradient_contract(spec) for spec in specs]
    filterflow_payload = _filterflow_subprocess(contracts)
    cells = []
    for spec, contract in zip(specs, contracts, strict=True):
        bayesfilter = _bayesfilter_gradient(spec, contract)
        filterflow = _filterflow_cell(filterflow_payload, spec.model_id)
        cells.append(_cell(spec, contract, bayesfilter, filterflow))
    decision = _decision(cells)
    return {
        "decision": decision,
        "created_at_utc": utc_now(),
        "question": (
            "Fixed-noise fixed-ancestor bootstrap filter scalar and physical-knob "
            "gradient tie-out across BayesFilter and executable float64 FilterFlow."
        ),
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
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
        "cells": cells,
        "summary": _summary(cells),
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners.run_common_fixed_branch_gradient_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_claims": [
            "no filtering-algorithm correctness claim",
            "no implementation is treated as an oracle",
            "no random-number-generator equality claim",
            "no stochastic-resampler or differentiable-resampler correctness claim",
            "no gradient through random or discrete ancestor selection",
            "no student-repository tie-out claim",
            "no TT-filter correctness claim",
            "no paper-scale, HMC, DSGE, GPU, or production-readiness claim",
        ],
    }


def _gradient_contract(spec: CommonModelSpec) -> dict[str, Any]:
    if spec.model_id == "lgssm_2d_linear":
        knob_names = ["transition_matrix_scale"]
        parameter_values = [1.0]
        parameterization = "physical scalar multiplying transition matrix A"
    elif spec.model_id == "sv_1d_synthetic":
        knob_names = ["gamma", "beta"]
        parameter_values = [float(spec.parameters["gamma"]), float(spec.parameters["beta"])]
        parameterization = "physical stochastic-volatility parameters"
    elif spec.model_id == "range_bearing_2d_cv":
        knob_names = ["sigma_range"]
        parameter_values = [0.12]
        parameterization = "physical range observation standard deviation"
    else:
        raise ValueError(f"unknown common model spec: {spec.model_id}")
    payload = {
        "model_id": spec.model_id,
        "family": spec.family,
        "dtype": DTYPE.name,
        "horizon": HORIZON,
        "num_particles": int(spec.x0.shape[0]),
        "initial_particles": tensor_to_json(spec.x0),
        "transition_innovations": tensor_to_json(_fixed_transition_innovations(spec)),
        "observations": tensor_to_json(_fixed_observations(spec)),
        "resampling_flags": list(RESAMPLING_FLAGS),
        "fixed_ancestor_indices": list(FIXED_ANCESTORS),
        "scalar": "sum of per-step predictive log normalizers after fixed branch replay",
        "gradient_knob_names": knob_names,
        "parameter_values": parameter_values,
        "parameterization": parameterization,
        "finite_difference_step": FD_STEP,
        "spec": spec.payload(),
        "spec_checksum": spec.checksum(),
    }
    payload["contract_checksum"] = stable_digest(payload)
    return payload


def _bayesfilter_gradient(spec: CommonModelSpec, contract: dict[str, Any]) -> dict[str, Any]:
    initial = [float(value) for value in contract["parameter_values"]]
    variables = [tf.Variable(value, dtype=DTYPE) for value in initial]
    with tf.GradientTape() as tape:
        value = _bayesfilter_scalar(spec, contract, variables)
    gradients = tape.gradient(value, variables)
    finite_difference = _central_finite_difference(
        lambda params: _bayesfilter_scalar(
            spec,
            contract,
            [tf.constant(value, DTYPE) for value in params],
        ),
        initial,
    )
    gradient_values = [_maybe_scalar(grad) for grad in gradients]
    finite = _finite_scalar(value) and all(
        grad is not None and _finite_scalar(grad) for grad in gradients
    )
    return {
        "status": "executed",
        "backend": "bayesfilter_common_fixed_branch_gradient",
        "model_id": spec.model_id,
        "scalar": scalar(value),
        "gradient": gradient_values,
        "finite_difference_gradient": finite_difference,
        "gradient_delta_vs_finite_difference": [
            None if grad is None else float(grad) - float(fd)
            for grad, fd in zip(gradient_values, finite_difference, strict=True)
        ],
        "finite": finite and all(math.isfinite(float(fd)) for fd in finite_difference),
        "parameterization": contract["parameterization"],
        "gradient_knob_names": contract["gradient_knob_names"],
        "contract_checksum": contract["contract_checksum"],
        "spec_checksum": spec.checksum(),
    }


def _bayesfilter_scalar(
    spec: CommonModelSpec,
    contract: dict[str, Any],
    params: list[tf.Tensor],
) -> tf.Tensor:
    particles = tf.convert_to_tensor(contract["initial_particles"], DTYPE)
    innovations = tf.convert_to_tensor(contract["transition_innovations"], DTYPE)
    observations = tf.convert_to_tensor(contract["observations"], DTYPE)
    ancestors = tf.convert_to_tensor(contract["fixed_ancestor_indices"], tf.int32)
    n_particles = int(particles.shape[0])
    log_weights = _uniform_log_weights(n_particles)
    total = tf.zeros([], DTYPE)
    for step in range(HORIZON):
        if bool(contract["resampling_flags"][step]):
            particles = tf.gather(particles, ancestors)
            log_weights = _uniform_log_weights(n_particles)
        predicted = _bayesfilter_transition_mean(spec, particles, params) + innovations[step]
        obs_log = _bayesfilter_observation_log_density(spec, predicted, observations[step], params, step + 1)
        unnormalized = log_weights + obs_log
        increment = tf.reduce_logsumexp(unnormalized)
        total = total + increment
        log_weights = unnormalized - increment
        particles = predicted
    return total


def _bayesfilter_transition_mean(
    spec: CommonModelSpec,
    particles: tf.Tensor,
    params: list[tf.Tensor],
) -> tf.Tensor:
    particles = tf.convert_to_tensor(particles, DTYPE)
    if spec.model_id == "lgssm_2d_linear":
        transition_matrix = tf.convert_to_tensor(spec.parameters["A"], DTYPE) * params[0]
        return tf.linalg.matmul(particles, transition_matrix, transpose_b=True)
    if spec.model_id == "sv_1d_synthetic":
        gamma = tf.convert_to_tensor(params[0], DTYPE)
        mu = tf.constant(float(spec.parameters.get("mu", 0.0)), DTYPE)
        return mu + gamma * (particles - mu)
    if spec.model_id == "range_bearing_2d_cv":
        transition_matrix = tf.convert_to_tensor(spec.parameters["A"], DTYPE)
        return tf.linalg.matmul(particles, transition_matrix, transpose_b=True)
    raise ValueError(f"unknown common model spec: {spec.model_id}")


def _bayesfilter_observation_log_density(
    spec: CommonModelSpec,
    predicted: tf.Tensor,
    observation: tf.Tensor,
    params: list[tf.Tensor],
    step: int,
) -> tf.Tensor:
    if spec.model_id == "lgssm_2d_linear":
        transition_matrix = tf.convert_to_tensor(spec.parameters["A"], DTYPE) * params[0]
        model = highdim.LinearGaussianSSM(
            initial_mean=spec.parameters["m0"],
            initial_covariance=spec.parameters["P0"],
            transition_matrix=transition_matrix,
            transition_covariance=spec.parameters["Q"],
            observation_matrix=spec.parameters["C"],
            observation_covariance=spec.parameters["R"],
        )
        return model.observation_log_density(spec.theta, predicted, observation, t=step)
    if spec.model_id == "sv_1d_synthetic":
        model = highdim.StochasticVolatilitySSM(sigma=spec.parameters["sigma"])
        theta = model.unconstrained_from_physical(params[0], params[1])
        return model.observation_log_density(theta, predicted, observation, t=step)
    if spec.model_id == "range_bearing_2d_cv":
        sigma_range = tf.convert_to_tensor(params[0], DTYPE)
        sigma_bearing = tf.sqrt(tf.convert_to_tensor(spec.parameters["R"], DTYPE)[1, 1])
        observation_covariance = tf.linalg.diag(
            tf.stack([tf.square(sigma_range), tf.square(sigma_bearing)])
        )
        model = CommonRangeBearingSSM(
            initial_mean=spec.parameters["m0"],
            initial_covariance=spec.parameters["P0"],
            transition_matrix=spec.parameters["A"],
            transition_covariance=spec.parameters["Q"],
            observation_covariance=observation_covariance,
        )
        return model.observation_log_density(spec.theta, predicted, observation, t=step)
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
            "blocker": "filterflow common fixed-branch gradient subprocess failed",
            "returncode": completed.returncode,
            "stdout_excerpt": completed.stdout[-4000:],
            "stderr_excerpt": completed.stderr[-4000:],
            "cells": [],
        }
    start = completed.stdout.rfind("FILTERFLOW_COMMON_FIXED_BRANCH_GRADIENT_JSON_BEGIN")
    end = completed.stdout.rfind("FILTERFLOW_COMMON_FIXED_BRANCH_GRADIENT_JSON_END")
    if start < 0 or end < 0 or end <= start:
        return {
            "status": "blocked",
            "blocker": "filterflow common fixed-branch gradient JSON sentinels missing",
            "stdout_excerpt": completed.stdout[-4000:],
            "stderr_excerpt": completed.stderr[-4000:],
            "cells": [],
        }
    raw = completed.stdout[
        start + len("FILTERFLOW_COMMON_FIXED_BRANCH_GRADIENT_JSON_BEGIN") : end
    ].strip()
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
        from filterflow.models.stochastic_volatility import SVObservationModel
        from filterflow.observation.linear import LinearObservationModel

        tfd = tfp.distributions
        DTYPE = tf.float64
        HORIZON = {HORIZON}
        FD_STEP = {FD_STEP!r}
        CONTRACTS = json.loads({contracts_literal!r})

        def to_tensor(value):
            return tf.convert_to_tensor(value, dtype=DTYPE)

        def scalar(tensor):
            return float(tf.cast(tensor, DTYPE).numpy())

        def state(rows):
            tensor = to_tensor(rows)
            return State(tf.reshape(tensor, [1, int(tensor.shape[0]), int(tensor.shape[1])]))

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

        def range_bearing_obs_ll(x_obs, observation, covariance):
            predicted = range_bearing_observation(x_obs)
            residual = to_tensor(observation)[tf.newaxis, :] - predicted
            residual = tf.concat([residual[..., :1], wrap_angle(residual[..., 1:2])], axis=-1)
            return gaussian_logpdf_zero_mean(residual, covariance)

        def central_fd(value_fn, params):
            results = []
            for index in range(len(params)):
                plus = list(params)
                minus = list(params)
                plus[index] = plus[index] + FD_STEP
                minus[index] = minus[index] - FD_STEP
                results.append(scalar((value_fn(plus) - value_fn(minus)) / tf.constant(2.0 * FD_STEP, DTYPE)))
            return results

        def fixed_branch_scalar(contract, params):
            particles = to_tensor(contract["initial_particles"])
            innovations = to_tensor(contract["transition_innovations"])
            observations = to_tensor(contract["observations"])
            ancestors = tf.convert_to_tensor(contract["fixed_ancestor_indices"], dtype=tf.int32)
            n_particles = int(particles.shape[0])
            log_weights = uniform_log_weights(n_particles)
            total = tf.zeros([], DTYPE)
            for step in range(HORIZON):
                if bool(contract["resampling_flags"][step]):
                    particles = tf.gather(particles, ancestors)
                    log_weights = uniform_log_weights(n_particles)
                predicted = transition_mean(contract, particles, params) + innovations[step]
                obs_log = observation_log_density(contract, predicted, observations[step], params)
                unnormalized = log_weights + obs_log
                increment = tf.reduce_logsumexp(unnormalized)
                total = total + increment
                log_weights = unnormalized - increment
                particles = predicted
            return total

        def transition_mean(contract, particles, params):
            p = contract["spec"]["parameters"]
            particles = to_tensor(particles)
            if contract["model_id"] == "lgssm_2d_linear":
                return tf.linalg.matmul(particles, to_tensor(p["A"]) * params[0], transpose_b=True)
            if contract["model_id"] == "sv_1d_synthetic":
                gamma = tf.reshape(params[0], [])
                mu = tf.constant(p["mu"], DTYPE)
                return mu + gamma * (particles - mu)
            if contract["model_id"] == "range_bearing_2d_cv":
                return tf.linalg.matmul(particles, to_tensor(p["A"]), transpose_b=True)
            raise ValueError(contract["model_id"])

        def observation_log_density(contract, predicted, observation, params):
            p = contract["spec"]["parameters"]
            if contract["model_id"] == "lgssm_2d_linear":
                model = LinearObservationModel(
                    to_tensor(p["C"]),
                    tfd.MultivariateNormalTriL(
                        loc=tf.zeros([len(p["R"])], dtype=DTYPE),
                        scale_tril=tf.linalg.cholesky(to_tensor(p["R"])),
                    ),
                )
                return tf.reshape(model.loglikelihood(state(predicted), to_tensor(observation)), [-1])
            if contract["model_id"] == "sv_1d_synthetic":
                beta = tf.reshape(params[1], [1, 1])
                model = SVObservationModel(beta)
                return tf.reshape(model.loglikelihood(state(predicted), to_tensor(observation)), [-1])
            if contract["model_id"] == "range_bearing_2d_cv":
                sigma_range = tf.reshape(params[0], [])
                sigma_bearing = tf.sqrt(to_tensor(p["R"])[1, 1])
                covariance = tf.linalg.diag(tf.stack([tf.square(sigma_range), tf.square(sigma_bearing)]))
                return range_bearing_obs_ll(predicted, observation, covariance)
            raise ValueError(contract["model_id"])

        def run_contract(contract):
            variables = [tf.Variable(value, dtype=DTYPE) for value in contract["parameter_values"]]
            with tf.GradientTape() as tape:
                value = fixed_branch_scalar(contract, variables)
            gradients = tape.gradient(value, variables)
            gradient_values = [None if grad is None else scalar(grad) for grad in gradients]
            fd = central_fd(
                lambda params: fixed_branch_scalar(contract, [tf.constant(value, DTYPE) for value in params]),
                [float(value) for value in contract["parameter_values"]],
            )
            finite = bool(
                tf.math.is_finite(value).numpy()
                and all(grad is not None and tf.math.is_finite(grad).numpy() for grad in gradients)
                and all(math.isfinite(float(value)) for value in fd)
            )
            return {{
                "status": "executed",
                "backend": "filterflow_common_fixed_branch_gradient",
                "model_id": contract["model_id"],
                "scalar": scalar(value),
                "gradient": gradient_values,
                "finite_difference_gradient": fd,
                "gradient_delta_vs_finite_difference": [
                    None if grad is None else float(grad) - float(fd_value)
                    for grad, fd_value in zip(gradient_values, fd)
                ],
                "finite": finite,
                "parameterization": contract["parameterization"],
                "gradient_knob_names": contract["gradient_knob_names"],
                "contract_checksum": contract["contract_checksum"],
                "spec_checksum": contract["spec_checksum"],
            }}

        payload = {{
            "status": "executed",
            "dtype": DTYPE.name,
            "pre_import_cuda_visible_devices": PRE_IMPORT_CUDA_VISIBLE_DEVICES,
            "gpu_devices_visible": [str(device) for device in tf.config.list_physical_devices("GPU")],
            "cells": [run_contract(contract) for contract in CONTRACTS],
        }}
        print("FILTERFLOW_COMMON_FIXED_BRANCH_GRADIENT_JSON_BEGIN")
        print(json.dumps(payload, sort_keys=True))
        print("FILTERFLOW_COMMON_FIXED_BRANCH_GRADIENT_JSON_END")
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
            "cell_type": "fixed_branch_gradient",
            "status": "INTERFACE_BLOCKED",
            "decision": f"{spec.model_id}_filterflow_gradient_blocked",
            "primary_criterion": "both adapters must execute the same fixed-branch gradient scalar",
            "metrics": {},
            "mismatch_class": "filterflow_common_fixed_branch_gradient_blocked",
            "reason": filterflow.get("blocker", "missing FilterFlow cell"),
            "bayesfilter": bayesfilter,
            "filterflow": filterflow,
            "contract": contract,
        }
    comparison = _compare_gradient_payloads(bayesfilter, filterflow)
    matched = (
        comparison["scalar_within_tolerance"]
        and comparison["gradient_within_tolerance"]
        and comparison["bayesfilter_fd_within_tolerance"]
        and comparison["filterflow_fd_within_tolerance"]
        and bayesfilter["finite"]
        and filterflow["finite"]
    )
    return {
        "model": spec.model_id,
        "family": spec.family,
        "implementations": ["BayesFilter", "FilterFlow"],
        "cell_type": "fixed_branch_gradient",
        "status": "MATCHED" if matched else "EXPLAINED_MISMATCH",
        "decision": f"{spec.model_id}_fixed_branch_gradient_matched" if matched else f"{spec.model_id}_fixed_branch_gradient_mismatch",
        "primary_criterion": "fixed-branch scalar and physical-knob gradients match within tolerance",
        "metrics": comparison,
        "mismatch_class": None if matched else "common_fixed_branch_gradient_delta_or_fd_veto",
        "bayesfilter": bayesfilter,
        "filterflow": filterflow,
        "contract": contract,
        "non_claim": "fixed-branch gradient agreement is not differentiable-resampling correctness",
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


def _compare_gradient_payloads(bayesfilter: dict[str, Any], filterflow: dict[str, Any]) -> dict[str, Any]:
    scalar_delta = abs(float(bayesfilter["scalar"]) - float(filterflow["scalar"]))
    gradient_delta = [
        float(left) - float(right)
        for left, right in zip(bayesfilter["gradient"], filterflow["gradient"], strict=True)
    ]
    max_abs_gradient_delta = max(abs(value) for value in gradient_delta)
    bayesfilter_fd_delta = [
        float(left) - float(right)
        for left, right in zip(
            bayesfilter["gradient"],
            bayesfilter["finite_difference_gradient"],
            strict=True,
        )
    ]
    filterflow_fd_delta = [
        float(left) - float(right)
        for left, right in zip(
            filterflow["gradient"],
            filterflow["finite_difference_gradient"],
            strict=True,
        )
    ]
    max_abs_bayesfilter_fd_delta = max(abs(value) for value in bayesfilter_fd_delta)
    max_abs_filterflow_fd_delta = max(abs(value) for value in filterflow_fd_delta)
    return {
        "scalar_delta": scalar_delta,
        "gradient_delta": gradient_delta,
        "max_abs_gradient_delta": max_abs_gradient_delta,
        "bayesfilter_fd_delta": bayesfilter_fd_delta,
        "filterflow_fd_delta": filterflow_fd_delta,
        "max_abs_bayesfilter_fd_delta": max_abs_bayesfilter_fd_delta,
        "max_abs_filterflow_fd_delta": max_abs_filterflow_fd_delta,
        "scalar_within_tolerance": scalar_delta <= VALUE_TOLERANCE,
        "gradient_within_tolerance": max_abs_gradient_delta <= GRADIENT_TOLERANCE,
        "bayesfilter_fd_within_tolerance": max_abs_bayesfilter_fd_delta <= FD_TOLERANCE,
        "filterflow_fd_within_tolerance": max_abs_filterflow_fd_delta <= FD_TOLERANCE,
        "filterflow_cpu_only": (
            filterflow.get("pre_import_cuda_visible_devices", "-1") == "-1"
        ),
    }


def _fixed_branch_scalar_common(
    initial_particles: tf.Tensor,
    transition_innovations: tf.Tensor,
    observations: tf.Tensor,
    observation_log_density: Callable[[tf.Tensor, tf.Tensor, int], tf.Tensor],
    transition_mean: Callable[[tf.Tensor], tf.Tensor],
) -> tf.Tensor:
    particles = tf.convert_to_tensor(initial_particles, DTYPE)
    innovations = tf.convert_to_tensor(transition_innovations, DTYPE)
    observations = tf.convert_to_tensor(observations, DTYPE)
    ancestors = tf.constant(FIXED_ANCESTORS, dtype=tf.int32)
    n_particles = int(particles.shape[0])
    log_weights = _uniform_log_weights(n_particles)
    total = tf.zeros([], DTYPE)
    for step in range(HORIZON):
        if RESAMPLING_FLAGS[step]:
            particles = tf.gather(particles, ancestors)
            log_weights = _uniform_log_weights(n_particles)
        predicted = transition_mean(particles) + innovations[step]
        obs_log = observation_log_density(predicted, observations[step], step + 1)
        unnormalized = log_weights + obs_log
        increment = tf.reduce_logsumexp(unnormalized)
        total = total + increment
        log_weights = unnormalized - increment
        particles = predicted
    return total


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


def _uniform_log_weights(n_particles: int) -> tf.Tensor:
    return tf.fill([n_particles], -tf.math.log(tf.cast(n_particles, DTYPE)))


def _finite_scalar(value: tf.Tensor) -> bool:
    return bool(tf.math.is_finite(tf.cast(value, DTYPE)).numpy())


def _maybe_scalar(value: tf.Tensor | None) -> float | None:
    if value is None:
        return None
    return scalar(value)


def _decision(cells: list[dict[str, Any]]) -> str:
    if any(cell["status"] == "EXPLAINED_MISMATCH" and not cell.get("mismatch_class") for cell in cells):
        return "common_fixed_branch_gradient_unclassified_mismatch_veto"
    if all(cell["status"] == "MATCHED" for cell in cells):
        return "common_fixed_branch_gradient_all_matched"
    if any(cell["status"] == "EXPLAINED_MISMATCH" for cell in cells):
        return "common_fixed_branch_gradient_explained_mismatch"
    return "common_fixed_branch_gradient_blocked"


def _summary(cells: list[dict[str, Any]]) -> dict[str, Any]:
    statuses: dict[str, int] = {}
    for cell in cells:
        statuses[cell["status"]] = statuses.get(cell["status"], 0) + 1
    return {
        "num_cells": len(cells),
        "status_counts": statuses,
        "models": [cell["model"] for cell in cells],
        "max_abs_scalar_delta": max(float(cell.get("metrics", {}).get("scalar_delta", 0.0)) for cell in cells),
        "max_abs_gradient_delta": max(float(cell.get("metrics", {}).get("max_abs_gradient_delta", 0.0)) for cell in cells),
        "max_abs_bayesfilter_fd_delta": max(float(cell.get("metrics", {}).get("max_abs_bayesfilter_fd_delta", 0.0)) for cell in cells),
        "max_abs_filterflow_fd_delta": max(float(cell.get("metrics", {}).get("max_abs_filterflow_fd_delta", 0.0)) for cell in cells),
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
    if payload["decision"] != "common_fixed_branch_gradient_all_matched":
        raise RuntimeError(payload["decision"])
    if len(payload["cells"]) != 3:
        raise RuntimeError("expected three common model cells")
    for cell in payload["cells"]:
        if cell["status"] != "MATCHED":
            raise RuntimeError(f"non-matched common fixed-branch gradient cell: {cell['model']}")
        metrics = cell["metrics"]
        if metrics["scalar_delta"] > VALUE_TOLERANCE:
            raise RuntimeError(f"scalar tolerance failed: {cell['model']}")
        if metrics["max_abs_gradient_delta"] > GRADIENT_TOLERANCE:
            raise RuntimeError(f"gradient tolerance failed: {cell['model']}")
        if metrics["max_abs_bayesfilter_fd_delta"] > FD_TOLERANCE:
            raise RuntimeError(f"BayesFilter finite-difference tolerance failed: {cell['model']}")
        if metrics["max_abs_filterflow_fd_delta"] > FD_TOLERANCE:
            raise RuntimeError(f"FilterFlow finite-difference tolerance failed: {cell['model']}")


def _markdown(payload: dict[str, Any]) -> str:
    return f"""# DPF Common Fixed-Branch Gradient Result

metadata_date: 2026-06-06

## Decision

`{payload['decision']}`

## Summary

- Cells: `{payload['summary']['num_cells']}`
- Status counts: `{payload['summary']['status_counts']}`
- Max scalar delta: `{payload['summary']['max_abs_scalar_delta']}`
- Max gradient delta: `{payload['summary']['max_abs_gradient_delta']}`
- Max BayesFilter AD-vs-FD delta: `{payload['summary']['max_abs_bayesfilter_fd_delta']}`
- Max FilterFlow AD-vs-FD delta: `{payload['summary']['max_abs_filterflow_fd_delta']}`

## Cell Table

{_cell_table(payload['cells'])}

## Interpretation

BayesFilter and executable float64 FilterFlow agree on gradients of the same
fixed-noise, fixed-ancestor bootstrap scalar for the common LGSSM,
stochastic-volatility, and range-bearing model suite.  The gradient knobs are
explicit physical parameters: a transition-matrix scale for LGSSM,
`(gamma,beta)` for stochastic volatility, and range observation noise scale for
range-bearing.

This result is a branch replay gradient check.  Ancestor indices are fixed and
nondifferentiated, so the result does not test gradients through random
resampling, differentiable resampling, or any student repository.

## Non-Claims

{_bullet_list(payload['non_claims'])}
"""


def _cell_table(cells: list[dict[str, Any]]) -> str:
    lines = [
        "| model | status | knobs | scalar delta | max gradient delta | BF FD delta | FF FD delta |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for cell in cells:
        metrics = cell.get("metrics", {})
        knobs = ",".join(cell.get("contract", {}).get("gradient_knob_names", []))
        lines.append(
            "| {model} | `{status}` | `{knobs}` | `{scalar}` | `{grad}` | `{bf_fd}` | `{ff_fd}` |".format(
                model=cell["model"],
                status=cell["status"],
                knobs=knobs,
                scalar=metrics.get("scalar_delta", "N/A"),
                grad=metrics.get("max_abs_gradient_delta", "N/A"),
                bf_fd=metrics.get("max_abs_bayesfilter_fd_delta", "N/A"),
                ff_fd=metrics.get("max_abs_filterflow_fd_delta", "N/A"),
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


def _bullet_list(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


if __name__ == "__main__":
    raise SystemExit(main())
