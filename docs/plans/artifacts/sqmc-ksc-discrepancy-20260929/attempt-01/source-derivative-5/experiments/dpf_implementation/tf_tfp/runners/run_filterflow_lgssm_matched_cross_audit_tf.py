"""Matched LGSSM audit against patched external filterflow."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]

import argparse
import json
import math
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tensorflow as tf

from experiments.dpf_implementation.tf_tfp.runners.common_tf import (
    OUTPUT_DIR,
    REPORT_DIR,
    REPO_ROOT,
    environment_manifest,
    load_json,
    stable_digest,
    utc_now,
    write_json,
    write_text,
)
from experiments.dpf_implementation.tf_tfp.runners.filterflow_reference_policy import (
    FILTERFLOW_REFERENCE_BRANCH,
    FILTERFLOW_REFERENCE_COMMIT,
    FILTERFLOW_REFERENCE_DTYPE,
    FILTERFLOW_UPSTREAM_BASE_COMMIT,
    validate_filterflow_reference_status,
)


DTYPE = tf.float64
THETAS = (0.25, 0.5, 0.75)
EPSILONS = (0.25, 0.5, 0.75)
HORIZON = 150
NUM_REALIZATIONS = 100
NUM_PARTICLES = 25
DATA_SEED = 111
FILTER_SEED = 555
JSON_PATH = OUTPUT_DIR / "dpf_filterflow_lgssm_matched_cross_audit_2026-05-30.json"
REPORT_PATH = REPORT_DIR / "dpf-filterflow-lgssm-matched-cross-audit-2026-05-30.md"
FILTERFLOW_ENV_PYTHON = REPO_ROOT / ".localenv" / "filterflow-py311" / "bin" / "python"
FILTERFLOW_PATH = REPO_ROOT / ".localsource" / "filterflow"
UPSTREAM_FILTERFLOW_COMMIT = FILTERFLOW_UPSTREAM_BASE_COMMIT


@dataclass(frozen=True)
class MatchedSpec:
    transition_covariance_scale: float = 1.0
    observation_covariance_scale: float = 0.1
    initial_covariance_scale: float = 1.0


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
    write_json(JSON_PATH, payload)
    write_text(REPORT_PATH, _markdown(payload))
    _validate_payload(payload)
    print(payload["decision"])
    return 0


def _run() -> dict[str, Any]:
    filterflow_status = _filterflow_status()
    filterflow = _run_filterflow_subprocess()
    spec = MatchedSpec()
    observations = tf.constant(filterflow["observations"], dtype=DTYPE)
    initial_particles = tf.constant(filterflow["initial_particles"], dtype=DTYPE)
    kalman_tf_rows = _kalman_rows_tf(observations, spec)
    kalman_check = _kalman_alignment(filterflow["kalman"], kalman_tf_rows)
    bayesfilter_rows = _run_bayesfilter_rows(observations, initial_particles, spec)
    comparison = _compare_rows(filterflow["runs"], bayesfilter_rows)
    decision = _decision(filterflow, kalman_check, comparison)
    payload = {
        "decision": decision,
        "created_at_utc": utc_now(),
        "question": "Matched filterflow/BayesFilter LGSSM Section-5.1-style audit on the same observation path",
        "plan_path": "docs/plans/bayesfilter-dpf-filterflow-lgssm-matched-cross-audit-plan-2026-05-30.md",
        "filterflow_status": filterflow_status,
        "filterflow_command": filterflow["command"],
        "filterflow_runs": filterflow["runs"],
        "filterflow_kalman": filterflow["kalman"],
        "bayesfilter_kalman": kalman_tf_rows,
        "kalman_alignment": kalman_check,
        "bayesfilter_runs": bayesfilter_rows,
        "comparison": comparison,
        "settings": {
            "horizon": HORIZON,
            "num_particles": NUM_PARTICLES,
            "num_realizations": NUM_REALIZATIONS,
            "theta_grid": list(THETAS),
            "epsilon_grid": list(EPSILONS),
            "data_seed": DATA_SEED,
            "filter_seed": FILTER_SEED,
            "transition_covariance": "I_2",
            "observation_covariance": "0.1 I_2",
            "resampling_threshold": "relative Neff/ESS 0.5",
            "same_observation_path": True,
            "algorithmic_randomness": "same fixed observation path; independent algorithmic random streams, not bitwise matched",
            "filterflow_reference_branch": FILTERFLOW_REFERENCE_BRANCH,
            "filterflow_dtype": FILTERFLOW_REFERENCE_DTYPE,
        },
        "scaling_and_epsilon_ledger": _scaling_ledger(),
        "discrepancy_ledger": _discrepancy_ledger(comparison, kalman_check),
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners.run_filterflow_lgssm_matched_cross_audit_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_implications": _non_implications(),
    }
    return payload


def _run_filterflow_subprocess() -> dict[str, Any]:
    if not FILTERFLOW_ENV_PYTHON.exists():
        raise RuntimeError(f"missing filterflow env python: {FILTERFLOW_ENV_PYTHON}")
    script = _filterflow_subprocess_script()
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env["MPLCONFIGDIR"] = str(REPO_ROOT / ".cache" / "filterflow-mpl")
    env["PYTHONPATH"] = str(FILTERFLOW_PATH)
    completed = subprocess.run(
        [str(FILTERFLOW_ENV_PYTHON), "-c", script],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "filterflow subprocess failed\n"
            f"stdout:\n{completed.stdout[-4000:]}\n"
            f"stderr:\n{completed.stderr[-4000:]}"
        )
    stdout = completed.stdout
    start = stdout.rfind("FILTERFLOW_JSON_BEGIN")
    end = stdout.rfind("FILTERFLOW_JSON_END")
    if start < 0 or end < 0 or end <= start:
        raise RuntimeError(f"filterflow JSON sentinels missing from stdout:\n{stdout[-4000:]}")
    raw = stdout[start + len("FILTERFLOW_JSON_BEGIN"):end].strip()
    payload = json.loads(raw)
    payload["command"] = (
        f"CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=.cache/filterflow-mpl PYTHONPATH=.localsource/filterflow "
        f"{FILTERFLOW_ENV_PYTHON} -c <structured filterflow matched LGSSM script>"
    )
    payload["stderr_excerpt"] = completed.stderr[-2000:]
    return payload


def _filterflow_subprocess_script() -> str:
    return textwrap.dedent(
        f"""
        import json
        import os
        import tensorflow as tf
        np = __import__("numpy")

        from filterflow.base import State
        from filterflow.models.simple_linear_gaussian import make_filter
        from filterflow.resampling import MultinomialResampler, RegularisedTransform
        from filterflow.resampling.criterion import NeffCriterion
        from scripts.simple_linear_common import get_data, kf_loglikelihood

        THETAS = {list(THETAS)!r}
        EPSILONS = {list(EPSILONS)!r}
        T = {HORIZON}
        BATCH_SIZE = {NUM_REALIZATIONS}
        N = {NUM_PARTICLES}
        DATA_SEED = {DATA_SEED}
        FILTER_SEED = {FILTER_SEED}

        transition_matrix = 0.5 * np.eye(2, dtype=np.float64)
        transition_covariance = np.eye(2, dtype=np.float64)
        observation_matrix = np.eye(2, dtype=np.float64)
        observation_covariance = 0.1 * np.eye(2, dtype=np.float64)
        values = np.array(list(zip(THETAS, THETAS)), dtype=np.float64)
        rng = np.random.RandomState(seed=DATA_SEED)
        data, kf = get_data(
            transition_matrix,
            observation_matrix,
            transition_covariance,
            observation_covariance,
            T,
            rng,
            dtype=np.float64,
        )
        initial_particles = rng.normal(0., 1., [BATCH_SIZE, N, 2]).astype(np.float64)

        def kalman_rows():
            rows = []
            for theta in THETAS:
                kf.transition_matrices = np.diag([theta, theta]).astype(np.float64)
                ll = float(kf_loglikelihood(kf, data))
                rows.append({{"theta": float(theta), "log_likelihood": ll, "per_time": ll / T}})
            return rows

        def run_method(method, epsilon=None):
            if method == "pf":
                resampling_method = MultinomialResampler()
                kwargs = {{}}
            elif method == "regularized":
                kwargs = {{"epsilon": float(epsilon), "scaling": 0.9, "convergence_threshold": 1e-3}}
                resampling_method = RegularisedTransform(**kwargs)
            else:
                raise ValueError(method)
            criterion = NeffCriterion(0.5, True)
            observation_dataset = tf.data.Dataset.from_tensor_slices(data)
            transition_covariance_chol = tf.linalg.cholesky(transition_covariance)
            observation_covariance_chol = tf.linalg.cholesky(observation_covariance)
            rows = []
            for theta in THETAS:
                transition_var = tf.Variable(np.diag([theta, theta]).astype(np.float64), trainable=True, dtype=tf.float64)
                smc = make_filter(
                    tf.convert_to_tensor(observation_matrix, dtype=tf.float64),
                    transition_var,
                    observation_covariance_chol,
                    transition_covariance_chol,
                    resampling_method,
                    criterion,
                )
                final_state = smc(
                    State(tf.constant(initial_particles)),
                    observation_dataset,
                    n_observations=tf.constant(T),
                    return_final=True,
                    seed=tf.constant(FILTER_SEED),
                )
                values = final_state.log_likelihoods.numpy().astype(float)
                rows.append({{
                    "method_id": "filterflow_" + method,
                    "theta": float(theta),
                    "epsilon": None if epsilon is None else float(epsilon),
                    "mean_log_likelihood": float(np.mean(values)),
                    "std_log_likelihood": float(np.std(values, ddof=1)),
                    "mean_per_time": float(np.mean(values) / T),
                    "std_per_time": float(np.std(values, ddof=1) / T),
                    "finite": bool(np.all(np.isfinite(values))),
                    "raw_log_likelihoods": values.tolist(),
                    "kwargs": kwargs,
                }})
            return rows

        kalman = kalman_rows()
        runs = []
        pf_rows = run_method("pf")
        for row in pf_rows:
            kalman_value = next(k["log_likelihood"] for k in kalman if k["theta"] == row["theta"])
            row["mean_error_per_time"] = (row["mean_log_likelihood"] - kalman_value) / T
            row["std_error_per_time"] = row["std_log_likelihood"] / T
            runs.append(row)
        for eps in EPSILONS:
            for row in run_method("regularized", eps):
                kalman_value = next(k["log_likelihood"] for k in kalman if k["theta"] == row["theta"])
                row["mean_error_per_time"] = (row["mean_log_likelihood"] - kalman_value) / T
                row["std_error_per_time"] = row["std_log_likelihood"] / T
                runs.append(row)

        payload = {{
            "status": "executed",
            "python": os.sys.version.split()[0],
            "tensorflow": tf.__version__,
            "numpy": np.__version__,
            "dtype": "float64",
            "kalman": kalman,
            "runs": runs,
            "observations": data.astype(float).tolist(),
            "initial_particles": initial_particles.astype(float).tolist(),
            "settings": {{
                "transition_covariance": "I_2",
                "observation_covariance": "0.1 I_2",
                "resampling": "NeffCriterion(0.5, True)",
                "regularized_kwargs": {{"scaling": 0.9, "convergence_threshold": 1e-3}},
            }},
        }}
        print("FILTERFLOW_JSON_BEGIN")
        print(json.dumps(payload, sort_keys=True))
        print("FILTERFLOW_JSON_END")
        """
    )


def _run_bayesfilter_rows(
    observations: tf.Tensor,
    initial_particles: tf.Tensor,
    spec: MatchedSpec,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    kalman_by_theta = {theta: _kalman_log_likelihood(observations, theta, spec) for theta in THETAS}
    for method_id in (
        "bayesfilter_pf",
        "bayesfilter_scaled_fixed_sinkhorn_ess",
        "bayesfilter_filterflow_style_transport_ess",
    ):
        epsilons = (None,) if method_id == "bayesfilter_pf" else EPSILONS
        for epsilon in epsilons:
            for theta in THETAS:
                try:
                    result = _run_bayesfilter_method(
                        observations,
                        initial_particles,
                        spec,
                        theta,
                        method_id,
                        epsilon,
                    )
                    errors = (result["log_likelihoods"] - kalman_by_theta[theta]) / tf.cast(HORIZON, DTYPE)
                    row = {
                        "method_id": method_id,
                        "theta": theta,
                        "epsilon": epsilon,
                        "row_status": "executed",
                        "mean_error_per_time": _float(tf.reduce_mean(errors)),
                        "std_error_per_time": _sample_std(errors),
                        "mean_log_likelihood": _float(tf.reduce_mean(result["log_likelihoods"])),
                        "std_log_likelihood": _sample_std(result["log_likelihoods"]),
                        "finite": result["finite"],
                        "median_resampling_count": _float(_median_tensor(result["resampling_counts"])),
                        "median_min_ess": _float(_median_tensor(result["min_ess"])),
                        "max_residual": result["max_residual"],
                        "fixed_sinkhorn_computed_rows": result["fixed_sinkhorn_computed_rows"],
                        "fixed_sinkhorn_skipped_rows": result["fixed_sinkhorn_skipped_rows"],
                        "median_cost_scale": (
                            None
                            if result["cost_scales"].shape[0] == 0
                            else _float(_median_tensor(result["cost_scales"]))
                        ),
                        "diagnostic": result["diagnostic"],
                    }
                except FloatingPointError as exc:
                    row = {
                        "method_id": method_id,
                        "theta": theta,
                        "epsilon": epsilon,
                        "row_status": "veto",
                        "mean_error_per_time": None,
                        "std_error_per_time": None,
                        "mean_log_likelihood": None,
                        "std_log_likelihood": None,
                        "finite": False,
                        "median_resampling_count": None,
                        "median_min_ess": None,
                        "max_residual": _extract_residual(str(exc)),
                        "fixed_sinkhorn_computed_rows": None,
                        "fixed_sinkhorn_skipped_rows": None,
                        "median_cost_scale": None,
                        "diagnostic": str(exc),
                    }
                rows.append(row)
    return rows


def _run_bayesfilter_method(
    observations: tf.Tensor,
    initial_particles: tf.Tensor,
    spec: MatchedSpec,
    theta: float,
    method_id: str,
    epsilon: float | None,
) -> dict[str, Any]:
    particles = tf.cast(initial_particles, DTYPE)
    batch_size = tf.shape(particles)[0]
    log_weights = tf.fill([batch_size, NUM_PARTICLES], -tf.math.log(tf.cast(NUM_PARTICLES, DTYPE)))
    log_likelihoods = tf.zeros([batch_size], DTYPE)
    resampling_counts = tf.zeros([batch_size], DTYPE)
    min_ess = tf.ones([batch_size], DTYPE) * tf.cast(NUM_PARTICLES, DTYPE)
    max_residual = tf.constant(0.0, DTYPE)
    fixed_sinkhorn_computed_rows = tf.constant(0.0, DTYPE)
    fixed_sinkhorn_skipped_rows = tf.constant(0.0, DTYPE)
    cost_scales: list[tf.Tensor] = []
    finite = True
    for t in range(HORIZON):
        weights = tf.exp(log_weights)
        ess_before = _ess_batched(weights)
        do_resample = ess_before <= tf.constant(0.5 * NUM_PARTICLES, DTYPE)
        if method_id == "bayesfilter_pf":
            indices = tf.random.stateless_categorical(
                log_weights,
                NUM_PARTICLES,
                seed=_seed_pair(FILTER_SEED, 7000 + t),
                dtype=tf.int32,
            )
            gathered = tf.gather(particles, indices, batch_dims=1)
            particles = tf.where(do_resample[:, None, None], gathered, particles)
            log_weights = tf.where(do_resample[:, None], _uniform_log_weights(batch_size), log_weights)
            resampling_counts += tf.cast(do_resample, DTYPE)
        elif method_id == "bayesfilter_scaled_fixed_sinkhorn_ess":
            triggered_count = tf.reduce_sum(tf.cast(do_resample, DTYPE))
            skipped_count = tf.cast(batch_size, DTYPE) - triggered_count
            fixed_sinkhorn_computed_rows += triggered_count
            fixed_sinkhorn_skipped_rows += skipped_count
            if bool(tf.reduce_any(do_resample).numpy()):
                active_particles = tf.boolean_mask(particles, do_resample)
                active_weights = tf.boolean_mask(weights, do_resample)
                relaxed_active, diag = _fixed_sinkhorn_resample(active_particles, active_weights, float(epsilon))
                particles = tf.tensor_scatter_nd_update(particles, tf.where(do_resample), relaxed_active)
                max_residual = tf.maximum(max_residual, diag["max_residual"])
                cost_scales.append(diag["cost_scale"])
            log_weights = tf.where(do_resample[:, None], _uniform_log_weights(batch_size), log_weights)
            resampling_counts += tf.cast(do_resample, DTYPE)
        elif method_id == "bayesfilter_filterflow_style_transport_ess":
            transported, diag = _filterflow_style_transport_resample(particles, log_weights, float(epsilon))
            particles = tf.where(do_resample[:, None, None], transported, particles)
            log_weights = tf.where(do_resample[:, None], _uniform_log_weights(batch_size), log_weights)
            resampling_counts += tf.cast(do_resample, DTYPE)
            max_residual = tf.maximum(max_residual, diag["max_transport_column_residual"])
            if bool(tf.reduce_any(do_resample).numpy()):
                cost_scales.append(tf.boolean_mask(diag["cost_scale"], do_resample))
        else:
            raise ValueError(method_id)
        particles = _transition_particles(particles, theta, spec, method_id, t)
        obs_logp = _observation_log_prob(particles, observations[t], spec)
        unnormalized = log_weights + obs_logp
        normalizer = tf.reduce_logsumexp(unnormalized, axis=1)
        log_likelihoods += normalizer
        log_weights = unnormalized - normalizer[:, None]
        weights = tf.exp(log_weights)
        ess_after = _ess_batched(weights)
        min_ess = tf.minimum(min_ess, ess_after)
        finite = finite and bool(tf.reduce_all(tf.math.is_finite(particles)).numpy())
        finite = finite and bool(tf.reduce_all(tf.math.is_finite(log_weights)).numpy())
        finite = finite and bool(tf.reduce_all(tf.math.is_finite(log_likelihoods)).numpy())
    return {
        "log_likelihoods": log_likelihoods,
        "resampling_counts": resampling_counts,
        "min_ess": min_ess,
        "finite": finite,
        "max_residual": _float(max_residual),
        "fixed_sinkhorn_computed_rows": _float(fixed_sinkhorn_computed_rows),
        "fixed_sinkhorn_skipped_rows": _float(fixed_sinkhorn_skipped_rows),
        "cost_scales": tf.concat(cost_scales, axis=0) if cost_scales else tf.zeros([0], DTYPE),
        "diagnostic": "executed",
    }


def _fixed_sinkhorn_resample(
    particles: tf.Tensor,
    weights: tf.Tensor,
    epsilon: float,
) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
    x = tf.cast(particles, DTYPE)
    centered = x - tf.stop_gradient(tf.reduce_mean(x, axis=1, keepdims=True))
    scale = _filterflow_scale(x)
    x_cost = centered / tf.stop_gradient(scale[:, None, None])
    cost = 0.5 * _pairwise_squared(x_cost)
    coupling, residual = _log_domain_sinkhorn_coupling(cost, weights, epsilon)
    column_mass = tf.reduce_sum(coupling, axis=1)
    relaxed = tf.linalg.matmul(coupling, x, transpose_a=True) / tf.maximum(column_mass[:, :, None], 1e-300)
    return relaxed, {"max_residual": residual, "cost_scale": scale}


def _filterflow_style_transport_resample(
    particles: tf.Tensor,
    log_weights: tf.Tensor,
    epsilon: float,
) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
    x = tf.cast(particles, DTYPE)
    batch_size = tf.shape(x)[0]
    dimension = tf.cast(tf.shape(x)[2], DTYPE)
    centered = x - tf.stop_gradient(tf.reduce_mean(x, axis=1, keepdims=True))
    scale = _filterflow_scale(x)
    scaled_x = centered / tf.stop_gradient(scale[:, None, None])
    cost = 0.5 * _pairwise_squared(scaled_x)
    uniform_log = tf.fill([batch_size, NUM_PARTICLES], -tf.math.log(tf.cast(NUM_PARTICLES, DTYPE)))
    potentials = _filterflow_style_potentials(
        log_weights,
        uniform_log,
        cost,
        scaled_x,
        epsilon,
        scaling=0.9,
        threshold=1e-3,
        max_iter=100,
    )
    f, g = potentials["alpha"], potentials["beta"]
    temp = (f[:, :, None] + g[:, None, :] - cost) / tf.constant(epsilon, DTYPE)
    temp = temp - tf.reduce_logsumexp(temp, axis=1, keepdims=True) + tf.math.log(tf.cast(NUM_PARTICLES, DTYPE))
    temp = temp + log_weights[:, None, :]
    transport = tf.exp(temp)
    transported = tf.linalg.matmul(transport, x)
    row_residual = tf.reduce_max(tf.abs(tf.reduce_sum(transport, axis=2) - 1.0))
    column_mass = tf.reduce_sum(transport, axis=1)
    source_weights = tf.exp(log_weights)
    column_residual = tf.reduce_max(tf.abs(column_mass - source_weights * tf.cast(NUM_PARTICLES, DTYPE)))
    if bool(tf.reduce_any(~tf.math.is_finite(transported)).numpy()):
        raise FloatingPointError("filterflow-style transport emitted non-finite particles")
    return transported, {
        "max_transport_column_residual": tf.cast(column_residual, DTYPE),
        "max_transport_row_residual": tf.cast(row_residual, DTYPE),
        "cost_scale": scale,
        "potential_iterations": tf.cast(potentials["iterations"], DTYPE),
        "dimension": dimension,
    }


def _filterflow_style_potentials(
    log_alpha: tf.Tensor,
    log_beta: tf.Tensor,
    cost: tf.Tensor,
    scaled_x: tf.Tensor,
    epsilon: float,
    *,
    scaling: float,
    threshold: float,
    max_iter: int,
) -> dict[str, tf.Tensor]:
    batch_size = tf.shape(cost)[0]
    epsilon_target = tf.ones([batch_size], DTYPE) * tf.constant(epsilon, DTYPE)
    epsilon_running = _filterflow_epsilon_start(scaled_x)
    epsilon_running = tf.maximum(epsilon_running, epsilon_target)
    scaling_factor = tf.constant(scaling * scaling, DTYPE)
    threshold_tensor = tf.constant(threshold, DTYPE)
    a = _softmin(epsilon_running, cost, log_alpha)
    b = _softmin(epsilon_running, cost, log_beta)
    continue_mask = tf.ones([batch_size], dtype=tf.bool)
    iterations = 0
    for iteration in range(max_iter - 1):
        eps_col = epsilon_running[:, None]
        continue_col = continue_mask[:, None]
        at = tf.where(continue_col, _softmin(epsilon_running, cost, log_alpha + b / eps_col), a)
        bt = tf.where(continue_col, _softmin(epsilon_running, cost, log_beta + a / eps_col), b)
        a_new = 0.5 * (a + at)
        b_new = 0.5 * (b + bt)
        update = tf.maximum(
            tf.reduce_max(tf.abs(a_new - a), axis=1),
            tf.reduce_max(tf.abs(b_new - b), axis=1),
        )
        a, b = a_new, b_new
        new_epsilon = tf.maximum(epsilon_running * scaling_factor, epsilon_target)
        continue_mask = tf.logical_or(new_epsilon < epsilon_running, update > threshold_tensor)
        epsilon_running = new_epsilon
        iterations = iteration + 1
        if not bool(tf.reduce_all(continue_mask).numpy()):
            break
    eps_target_col = epsilon_target[:, None]
    final_a = _softmin(epsilon_target, cost, log_alpha + tf.stop_gradient(b) / eps_target_col)
    final_b = _softmin(epsilon_target, cost, log_beta + tf.stop_gradient(a) / eps_target_col)
    return {"alpha": final_a, "beta": final_b, "iterations": tf.ones([batch_size], DTYPE) * iterations}


def _log_domain_sinkhorn_coupling(
    cost: tf.Tensor,
    weights: tf.Tensor,
    epsilon: float,
) -> tuple[tf.Tensor, tf.Tensor]:
    weights = weights / tf.reduce_sum(weights, axis=1, keepdims=True)
    batch_size = tf.shape(cost)[0]
    target = tf.ones([batch_size, NUM_PARTICLES], DTYPE) / tf.cast(NUM_PARTICLES, DTYPE)
    log_source = tf.math.log(tf.maximum(weights, tf.constant(1e-300, DTYPE)))
    log_target = tf.math.log(target)
    log_kernel = -cost / tf.constant(epsilon, DTYPE)
    log_u = tf.zeros([batch_size, NUM_PARTICLES], DTYPE)
    log_v = tf.zeros([batch_size, NUM_PARTICLES], DTYPE)
    residual = tf.constant(float("inf"), DTYPE)
    for iteration in range(1, 101):
        log_u = log_source - tf.reduce_logsumexp(log_kernel + log_v[:, None, :], axis=2)
        log_v = log_target - tf.reduce_logsumexp(log_kernel + log_u[:, :, None], axis=1)
        if iteration % 10 == 0 or iteration == 100:
            coupling = tf.exp(log_u[:, :, None] + log_kernel + log_v[:, None, :])
            row_residual = tf.reduce_max(tf.abs(tf.reduce_sum(coupling, axis=2) - weights))
            column_residual = tf.reduce_max(tf.abs(tf.reduce_sum(coupling, axis=1) - target))
            residual = tf.maximum(row_residual, column_residual)
            if bool((residual <= 1e-7).numpy()):
                break
    coupling = tf.exp(log_u[:, :, None] + log_kernel + log_v[:, None, :])
    row_residual = tf.reduce_max(tf.abs(tf.reduce_sum(coupling, axis=2) - weights))
    column_residual = tf.reduce_max(tf.abs(tf.reduce_sum(coupling, axis=1) - target))
    residual = tf.maximum(row_residual, column_residual)
    if _float(residual) > 1e-5:
        raise FloatingPointError(f"fixed Sinkhorn residual too large: {_float(residual)}")
    return coupling, residual


def _softmin(epsilon: tf.Tensor, cost: tf.Tensor, f: tf.Tensor) -> tf.Tensor:
    eps = epsilon[:, None, None]
    temp = f[:, None, :] - cost / eps
    return -epsilon[:, None] * tf.reduce_logsumexp(temp, axis=2)


def _filterflow_epsilon_start(scaled_x: tf.Tensor) -> tf.Tensor:
    coordinate_range = tf.reduce_max(scaled_x, axis=[1, 2]) - tf.reduce_min(scaled_x, axis=[1, 2])
    return tf.maximum(coordinate_range * coordinate_range, tf.constant(1e-6, DTYPE))


def _transition_particles(
    particles: tf.Tensor,
    theta: float,
    spec: MatchedSpec,
    method_id: str,
    time_index: int,
) -> tf.Tensor:
    method_salt = {
        "bayesfilter_pf": 0,
        "bayesfilter_scaled_fixed_sinkhorn_ess": 100000,
        "bayesfilter_filterflow_style_transport_ess": 200000,
    }[method_id]
    noise = (
        tf.random.stateless_normal(
            tf.shape(particles),
            seed=_seed_pair(FILTER_SEED, method_salt + 2000 + time_index),
            dtype=DTYPE,
        )
        * math.sqrt(spec.transition_covariance_scale)
    )
    return tf.constant(theta, DTYPE) * particles + noise


def _observation_log_prob(particles: tf.Tensor, observation: tf.Tensor, spec: MatchedSpec) -> tf.Tensor:
    residual = particles - tf.cast(observation, DTYPE)[None, None, :]
    covariance = tf.eye(2, dtype=DTYPE) * spec.observation_covariance_scale
    chol = tf.linalg.cholesky(covariance)
    flat = tf.reshape(residual, [-1, 2])
    solved = tf.linalg.cholesky_solve(chol, tf.transpose(flat))
    quad = tf.reduce_sum(tf.transpose(solved) * flat, axis=1)
    logdet = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)))
    logp = -0.5 * (2.0 * tf.math.log(tf.constant(2.0 * math.pi, DTYPE)) + logdet + quad)
    return tf.reshape(logp, [tf.shape(particles)[0], NUM_PARTICLES])


def _kalman_rows_tf(observations: tf.Tensor, spec: MatchedSpec) -> list[dict[str, Any]]:
    return [
        {
            "theta": theta,
            "log_likelihood": _float(_kalman_log_likelihood(observations, theta, spec)),
            "per_time": _float(_kalman_log_likelihood(observations, theta, spec) / tf.cast(HORIZON, DTYPE)),
        }
        for theta in THETAS
    ]


def _kalman_log_likelihood(observations: tf.Tensor, theta: float, spec: MatchedSpec) -> tf.Tensor:
    a = tf.eye(2, dtype=DTYPE) * theta
    q = tf.eye(2, dtype=DTYPE) * spec.transition_covariance_scale
    r = tf.eye(2, dtype=DTYPE) * spec.observation_covariance_scale
    mean = tf.zeros([2], DTYPE)
    cov = tf.eye(2, dtype=DTYPE) * spec.initial_covariance_scale
    eye = tf.eye(2, dtype=DTYPE)
    ll = tf.constant(0.0, DTYPE)
    obs_list = tf.unstack(tf.cast(observations, DTYPE), axis=0)
    for index, observation in enumerate(obs_list):
        residual = observation - mean
        s_mat = cov + r
        ll += _gaussian_logpdf(residual, s_mat)
        gain = tf.linalg.matmul(cov, tf.linalg.inv(s_mat))
        mean = mean + tf.linalg.matvec(gain, residual)
        cov = (eye - gain) @ cov
        cov = 0.5 * (cov + tf.transpose(cov))
        if index != len(obs_list) - 1:
            mean = tf.linalg.matvec(a, mean)
            cov = a @ cov @ tf.transpose(a) + q
            cov = 0.5 * (cov + tf.transpose(cov))
    return ll


def _gaussian_logpdf(residual: tf.Tensor, covariance: tf.Tensor) -> tf.Tensor:
    chol = tf.linalg.cholesky(covariance)
    solved = tf.linalg.cholesky_solve(chol, residual[:, None])[:, 0]
    quad = tf.reduce_sum(solved * residual)
    logdet = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)))
    return -0.5 * (2.0 * tf.math.log(tf.constant(2.0 * math.pi, DTYPE)) + logdet + quad)


def _filterflow_scale(x: tf.Tensor) -> tf.Tensor:
    std = tf.math.reduce_std(tf.cast(x, DTYPE), axis=1)
    diameter = tf.reduce_max(std, axis=1)
    diameter = tf.where(diameter == 0.0, tf.ones_like(diameter), diameter)
    return diameter * tf.sqrt(tf.constant(2.0, DTYPE))


def _pairwise_squared(x: tf.Tensor) -> tf.Tensor:
    diff = x[:, :, None, :] - x[:, None, :, :]
    return tf.reduce_sum(diff * diff, axis=3)


def _ess_batched(weights: tf.Tensor) -> tf.Tensor:
    return 1.0 / tf.reduce_sum(tf.cast(weights, DTYPE) * tf.cast(weights, DTYPE), axis=1)


def _uniform_log_weights(batch_size: tf.Tensor) -> tf.Tensor:
    return tf.fill([batch_size, NUM_PARTICLES], -tf.math.log(tf.cast(NUM_PARTICLES, DTYPE)))


def _compare_rows(filterflow_rows: list[dict[str, Any]], bayesfilter_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    comparisons = []
    pairs = [
        ("filterflow_pf", "bayesfilter_pf"),
        ("filterflow_regularized", "bayesfilter_scaled_fixed_sinkhorn_ess"),
        ("filterflow_regularized", "bayesfilter_filterflow_style_transport_ess"),
    ]
    for external, internal in pairs:
        for epsilon in ((None,) if external == "filterflow_pf" else EPSILONS):
            for theta in THETAS:
                frow = _find_row(filterflow_rows, external, theta, epsilon)
                brow = _find_row(bayesfilter_rows, internal, theta, epsilon)
                delta = None
                within = None
                if frow and brow and brow["mean_error_per_time"] is not None:
                    delta = brow["mean_error_per_time"] - frow["mean_error_per_time"]
                    within = abs(delta) <= frow["std_error_per_time"]
                comparisons.append(
                    {
                        "external_method": external,
                        "bayesfilter_method": internal,
                        "theta": theta,
                        "epsilon": epsilon,
                        "filterflow_mean_error": None if not frow else frow["mean_error_per_time"],
                        "filterflow_std_error": None if not frow else frow["std_error_per_time"],
                        "bayesfilter_mean_error": None if not brow else brow["mean_error_per_time"],
                        "bayesfilter_std_error": None if not brow else brow["std_error_per_time"],
                        "delta": delta,
                        "within_one_filterflow_sd": within,
                        "bayesfilter_status": None if not brow else brow["row_status"],
                    }
                )
    return comparisons


def _decision(filterflow: dict[str, Any], kalman_check: dict[str, Any], comparison: list[dict[str, Any]]) -> str:
    if filterflow["status"] != "executed":
        return "blocked_filterflow_not_executed"
    if not kalman_check["all_within_tolerance"]:
        return "blocked_kalman_mismatch"
    ff_style = [
        row for row in comparison if row["bayesfilter_method"] == "bayesfilter_filterflow_style_transport_ess"
    ]
    fixed = [
        row for row in comparison if row["bayesfilter_method"] == "bayesfilter_scaled_fixed_sinkhorn_ess"
    ]
    if all(row["within_one_filterflow_sd"] for row in ff_style) and len(ff_style) == 9:
        return "filterflow_style_transport_matched"
    if any(row["bayesfilter_status"] != "executed" for row in fixed):
        return "current_fixed_sinkhorn_red_flag_filterflow_style_not_matched"
    return "red_flag"


def _kalman_alignment(filterflow_rows: list[dict[str, Any]], tf_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for frow in filterflow_rows:
        trow = next(row for row in tf_rows if row["theta"] == frow["theta"])
        delta = trow["log_likelihood"] - frow["log_likelihood"]
        rows.append({"theta": frow["theta"], "delta": delta, "abs_delta": abs(delta)})
    return {
        "rows": rows,
        "max_abs_delta": max(row["abs_delta"] for row in rows),
        "all_within_tolerance": all(row["abs_delta"] < 1e-6 for row in rows),
    }


def _find_row(rows: list[dict[str, Any]], method_id: str, theta: float, epsilon: float | None) -> dict[str, Any] | None:
    for row in rows:
        if row["method_id"] == method_id and row["theta"] == theta and row.get("epsilon") == epsilon:
            return row
    return None


def _filterflow_status() -> dict[str, Any]:
    return {
        "branch": _git(["git", "-C", str(FILTERFLOW_PATH), "rev-parse", "--abbrev-ref", "HEAD"]),
        "commit": _git(["git", "-C", str(FILTERFLOW_PATH), "rev-parse", "HEAD"]),
        "status": _git(["git", "-C", str(FILTERFLOW_PATH), "status", "--short", "--branch"]),
        "diff_summary": _git(["git", "-C", str(FILTERFLOW_PATH), "diff", "--stat"]) or "clean",
        "upstream_base": UPSTREAM_FILTERFLOW_COMMIT,
    }


def _scaling_ledger() -> dict[str, Any]:
    return {
        "filterflow_regularized": {
            "centering": "x - stop_gradient(mean(x, axis=particles))",
            "scale": "diameter(x, x) * sqrt(dimension)",
            "cost": "squared distance / 2 on scaled particles",
            "epsilon_schedule": "epsilon_0=max_min(scaled_x)^2, geometric decrease by scaling^2 to target",
            "convergence_threshold": 1e-3,
        },
        "bayesfilter_scaled_fixed_sinkhorn_ess": {
            "centering_scale_cost": "matched",
            "epsilon_schedule": "fixed target epsilon",
            "match_status": "similar_not_matched",
        },
        "bayesfilter_filterflow_style_transport_ess": {
            "centering_scale_cost": "matched_in_audit_runner",
            "epsilon_schedule": "filterflow-style annealing in audit runner",
            "match_status": "audit_runner_matched_attempt_not_production",
        },
    }


def _discrepancy_ledger(comparison: list[dict[str, Any]], kalman_check: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": "kalman_alignment",
            "status": "pass" if kalman_check["all_within_tolerance"] else "veto",
            "max_abs_delta": kalman_check["max_abs_delta"],
        },
        {
            "id": "fixed_sinkhorn_match",
            "status": _comparison_status(comparison, "bayesfilter_scaled_fixed_sinkhorn_ess"),
        },
        {
            "id": "filterflow_style_transport_match",
            "status": _comparison_status(comparison, "bayesfilter_filterflow_style_transport_ess"),
        },
        {
            "id": "random_stream",
            "status": "not_bitwise_matched",
            "detail": "same observations and initial cloud; algorithmic random streams are fixed but independent",
        },
    ]


def _comparison_status(comparison: list[dict[str, Any]], method_id: str) -> str:
    rows = [row for row in comparison if row["bayesfilter_method"] == method_id]
    if any(row["bayesfilter_status"] != "executed" for row in rows):
        return "bayesfilter_veto_or_missing"
    if all(row["within_one_filterflow_sd"] for row in rows):
        return "within_filterflow_mc_band"
    return "outside_filterflow_mc_band"


def _markdown(payload: dict[str, Any]) -> str:
    return f"""# Filterflow LGSSM Matched Cross-Audit

## Decision

`{payload['decision']}`

## Summary

This rerun uses the patched external filterflow branch and the same fixed
observation path for filterflow and BayesFilter.  Algorithmic random streams
are fixed but not bitwise matched.

## Filterflow

- Branch: `{payload['filterflow_status']['branch']}`
- Commit: `{payload['filterflow_status']['commit']}`
- Status: `{payload['filterflow_status']['status']}`
- Diff summary: `{payload['filterflow_status']['diff_summary']}`

## Kalman Alignment

- Max absolute log-likelihood delta: `{payload['kalman_alignment']['max_abs_delta']:.3e}`
- Within tolerance: `{payload['kalman_alignment']['all_within_tolerance']}`

## Comparison

{_comparison_table(payload['comparison'])}

## Discrepancies

{_discrepancy_table(payload['discrepancy_ledger'])}

## Non-Implications

{_non_implications_markdown()}
"""


def _comparison_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| External | BayesFilter | eps | theta | filterflow mean | BayesFilter mean | delta | within 1 sd | status |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {external} | {internal} | {eps} | {theta} | {fm} | {bm} | {delta} | {within} | `{status}` |".format(
                external=row["external_method"],
                internal=row["bayesfilter_method"],
                eps=_fmt(row["epsilon"]),
                theta=row["theta"],
                fm=_fmt(row["filterflow_mean_error"]),
                bm=_fmt(row["bayesfilter_mean_error"]),
                delta=_fmt(row["delta"]),
                within=row["within_one_filterflow_sd"],
                status=row["bayesfilter_status"],
            )
        )
    return "\n".join(lines)


def _discrepancy_table(rows: list[dict[str, Any]]) -> str:
    lines = ["| ID | Status | Detail |", "| --- | --- | --- |"]
    for row in rows:
        detail = row.get("detail", row.get("max_abs_delta", ""))
        lines.append(f"| `{row['id']}` | `{row['status']}` | {detail} |")
    return "\n".join(lines)


def _validate_payload(payload: dict[str, Any]) -> None:
    validate_filterflow_reference_status(payload["filterflow_status"])
    if payload["filterflow_runs"][0]["method_id"] not in {"filterflow_pf", "filterflow_regularized"}:
        raise RuntimeError("filterflow rows missing")
    if not payload["kalman_alignment"]["all_within_tolerance"]:
        raise RuntimeError("Kalman alignment failed")
    if payload["run_manifest"]["pre_import_cuda_visible_devices"] != "-1":
        raise RuntimeError("missing CPU-only manifest")
    if "reproducibility_digest" not in payload:
        raise RuntimeError("missing digest")


def _digest_payload(payload: dict[str, Any]) -> str:
    comparable = dict(payload)
    comparable["created_at_utc"] = "TIMESTAMP"
    comparable["run_manifest"] = dict(comparable["run_manifest"])
    comparable["run_manifest"]["wall_time_seconds"] = "WALL_TIME"
    comparable["run_manifest"]["dirty_state_summary"] = "DIRTY_STATE"
    return stable_digest(comparable)


def _git(args: list[str]) -> str:
    completed = subprocess.run(args, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _sample_std(values: tf.Tensor) -> float:
    values = tf.reshape(tf.cast(values, DTYPE), [-1])
    count = tf.cast(tf.shape(values)[0], DTYPE)
    mean = tf.reduce_mean(values)
    variance = tf.reduce_sum(tf.square(values - mean)) / tf.maximum(count - 1.0, 1.0)
    return _float(tf.sqrt(variance))


def _median_tensor(values: tf.Tensor) -> tf.Tensor:
    values = tf.sort(tf.reshape(tf.cast(values, DTYPE), [-1]))
    count = tf.shape(values)[0]
    mid = count // 2
    return tf.cond(
        tf.equal(count % 2, 1),
        lambda: values[mid],
        lambda: 0.5 * (values[mid - 1] + values[mid]),
    )


def _extract_residual(message: str) -> float | None:
    try:
        return float(message.rsplit(":", maxsplit=1)[-1].strip())
    except ValueError:
        return None


def _float(value: tf.Tensor) -> float:
    return float(tf.cast(value, DTYPE).numpy())


def _seed_pair(seed: int, salt: int) -> tf.Tensor:
    return tf.constant([int(seed) % 2147483647, int(salt) % 2147483647], dtype=tf.int32)


def _non_implications() -> list[str]:
    return [
        "No production readiness is concluded.",
        "No public API readiness is concluded.",
        "No HMC readiness is concluded.",
        "No posterior correctness is concluded.",
        "No general nonlinear-SSM validity is concluded.",
        "No claim that finite relaxed OT is categorical PF is concluded.",
        "No claim that patched filterflow is untouched upstream code is concluded.",
    ]


def _non_implications_markdown() -> str:
    return "\n".join(f"- {item}" for item in _non_implications())


if __name__ == "__main__":
    raise SystemExit(main())
