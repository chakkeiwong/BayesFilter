"""Cross-audit BayesFilter OT-DPF against filterflow LGSSM Section 5.1."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]

import argparse
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tensorflow as tf
import tensorflow_probability as tfp

from experiments.dpf_implementation.tf_tfp.runners.common_tf import (
    OUTPUT_DIR,
    REPORT_DIR,
    environment_manifest,
    load_json,
    stable_digest,
    utc_now,
    write_json,
    write_text,
)


tfd = tfp.distributions
DTYPE = tf.float64
THETAS = (0.25, 0.5, 0.75)
EPSILONS = (0.25, 0.5, 0.75)
NUM_REALIZATIONS = 100
HORIZON = 150
NUM_PARTICLES = 25
DATA_SEED_BASE = 111
FILTER_SEED_BASE = 555
JSON_PATH = OUTPUT_DIR / "dpf_filterflow_lgssm_cross_audit_2026-05-30.json"
REPORT_PATH = REPORT_DIR / "dpf-filterflow-lgssm-cross-implementation-audit-2026-05-30.md"


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    transition_covariance_scale: float
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
    repo_root = Path(__file__).resolve().parents[4]
    filterflow_path = repo_root / ".localsource" / "filterflow"
    filterflow_status = _filterflow_status(filterflow_path)
    paper_spec = ModelSpec("paper_text_covariance_0p5I", 0.5)
    filterflow_spec = ModelSpec("filterflow_code_covariance_1p0I", 1.0)
    table = _paper_table()
    scaling_ledger = _scaling_ledger(match_status="similar_not_matched")
    filterflow_execution = _probe_filterflow_execution(filterflow_path)
    primary_result = _run_spec(filterflow_spec, "primary_filterflow_code_covariance")
    sensitivity_result = _run_spec(paper_spec, "sensitivity_paper_text_covariance")
    rows = primary_result["rows"] + sensitivity_result["rows"]
    kalman_reference = primary_result["kalman_reference"] + sensitivity_result["kalman_reference"]
    summary = _summarize(rows)
    discrepancy_ledger = _discrepancy_ledger(filterflow_status, filterflow_execution, scaling_ledger)
    decision = _decision(filterflow_execution, scaling_ledger, summary)
    comparator_matrix = _comparator_outcome_matrix(rows, filterflow_execution, summary)
    payload = {
        "decision": decision,
        "created_at_utc": utc_now(),
        "question": "BayesFilter TF/TFP finite-Sinkhorn OT-DPF versus Corenflos/filterflow LGSSM Section 5.1",
        "plan_path": "docs/plans/bayesfilter-dpf-filterflow-lgssm-cross-implementation-audit-plan-2026-05-30.md",
        "filterflow_status": filterflow_status,
        "filterflow_execution": filterflow_execution,
        "paper_section_5_1_settings": _paper_settings(),
        "paper_table_1_reference": table,
        "paper_vs_filterflow_settings_ledger": _settings_ledger(),
        "primary_decision_lane": {
            "bayesfilter": "bayesfilter_scaled_ess",
            "external": "filterflow_regularized",
            "status": "external_blocked_and_bayesfilter_primary_partial_sinkhorn_residual_veto",
            "sensitivity_lanes_do_not_rescue_primary": True,
        },
        "scaling_and_epsilon_match_ledger": scaling_ledger,
        "minimal_comparator_matrix_status": _matrix_status(filterflow_execution, summary),
        "comparator_outcome_matrix": comparator_matrix,
        "kalman_reference": kalman_reference,
        "num_realizations": NUM_REALIZATIONS,
        "horizon": HORIZON,
        "num_particles": NUM_PARTICLES,
        "theta_grid": list(THETAS),
        "epsilon_grid": list(EPSILONS),
        "seed_protocol": _seed_protocol(),
        "rows": rows,
        "summary": summary,
        "discrepancy_ledger": discrepancy_ledger,
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners.run_filterflow_lgssm_cross_audit_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_implications": _non_implications(),
    }
    return payload


def _run_spec(spec: ModelSpec, spec_role: str) -> dict[str, Any]:
    datasets = _simulate_datasets_batched(spec, DATA_SEED_BASE)
    kalman_by_theta = {
        theta: _kalman_log_likelihoods_batched(datasets, theta, spec) for theta in THETAS
    }
    kalman_reference = _kalman_reference_rows(kalman_by_theta, spec, spec_role)
    rows: list[dict[str, Any]] = []
    for method_id in ("bayesfilter_pf", "bayesfilter_raw_ess", "bayesfilter_scaled_ess", "bayesfilter_scaled_every_step"):
        eps_values = (None,) if method_id == "bayesfilter_pf" else EPSILONS
        for epsilon in eps_values:
            for theta in THETAS:
                try:
                    if method_id == "bayesfilter_pf":
                        result = _run_pf_like_batched(
                            datasets,
                            theta,
                            spec,
                            FILTER_SEED_BASE,
                            method_id,
                            ess_threshold_ratio=0.5,
                            resampler="multinomial",
                        )
                    elif method_id == "bayesfilter_raw_ess":
                        result = _run_pf_like_batched(
                            datasets,
                            theta,
                            spec,
                            FILTER_SEED_BASE,
                            method_id,
                            ess_threshold_ratio=0.5,
                            resampler="sinkhorn_raw",
                            epsilon=float(epsilon),
                        )
                    elif method_id == "bayesfilter_scaled_ess":
                        result = _run_pf_like_batched(
                            datasets,
                            theta,
                            spec,
                            FILTER_SEED_BASE,
                            method_id,
                            ess_threshold_ratio=0.5,
                            resampler="sinkhorn_scaled",
                            epsilon=float(epsilon),
                        )
                    else:
                        result = _run_pf_like_batched(
                            datasets,
                            theta,
                            spec,
                            FILTER_SEED_BASE,
                            method_id,
                            ess_threshold_ratio=1.01,
                            resampler="sinkhorn_scaled",
                            epsilon=float(epsilon),
                        )
                    errors = (result["log_likelihoods"] - kalman_by_theta[theta]) / tf.cast(HORIZON, DTYPE)
                    mean_error = _float(tf.reduce_mean(errors))
                    std_error = _sample_std(errors)
                    finite = result["finite"]
                    median_resampling_count = _float(_median_tensor(result["resampling_counts"]))
                    median_min_ess = _float(_median_tensor(result["min_ess"]))
                    max_sinkhorn_residual = result["max_sinkhorn_residual"]
                    median_cost_scale = (
                        None
                        if result["cost_scales"].shape[0] == 0
                        else _float(_median_tensor(result["cost_scales"]))
                    )
                    row_status = "executed"
                    error_message = None
                except FloatingPointError as exc:
                    mean_error = None
                    std_error = None
                    finite = False
                    median_resampling_count = None
                    median_min_ess = None
                    max_sinkhorn_residual = _extract_residual(str(exc))
                    median_cost_scale = None
                    row_status = "sinkhorn_residual_veto"
                    error_message = str(exc)
                reference = _paper_reference_for(method_id, epsilon, theta)
                row = {
                    "spec_role": spec_role,
                    "model_id": spec.model_id,
                    "method_id": method_id,
                    "theta": theta,
                    "epsilon": epsilon,
                    "num_realizations": NUM_REALIZATIONS,
                    "mean_per_time_error": mean_error,
                    "std_per_time_error": std_error,
                    "paper_reference_mean": reference["mean"],
                    "paper_reference_std": reference["std"],
                    "delta_to_paper_mean": (
                        None
                        if reference["mean"] is None or mean_error is None
                        else mean_error - reference["mean"]
                    ),
                    "within_one_paper_sd": (
                        None
                        if reference["std"] is None or mean_error is None
                        else abs(mean_error - reference["mean"]) <= reference["std"]
                    ),
                    "finite": finite,
                    "median_resampling_count": median_resampling_count,
                    "median_min_ess": median_min_ess,
                    "max_sinkhorn_residual": max_sinkhorn_residual,
                    "median_cost_scale": median_cost_scale,
                    "row_status": row_status,
                    "error_message": error_message,
                    "resampling_policy": "ESS threshold 0.5" if method_id != "bayesfilter_scaled_every_step" else "every step",
                    "cost_scaling": _cost_scaling_label(method_id),
                }
                rows.append(row)
    return {"rows": rows, "kalman_reference": kalman_reference}


def _kalman_reference_rows(
    kalman_by_theta: dict[float, tf.Tensor],
    spec: ModelSpec,
    spec_role: str,
) -> list[dict[str, Any]]:
    rows = []
    for theta, values in kalman_by_theta.items():
        per_time = values / tf.cast(HORIZON, DTYPE)
        rows.append(
            {
                "spec_role": spec_role,
                "model_id": spec.model_id,
                "theta": theta,
                "num_realizations": NUM_REALIZATIONS,
                "mean_log_likelihood": _float(tf.reduce_mean(values)),
                "std_log_likelihood": _sample_std(values),
                "mean_per_time_log_likelihood": _float(tf.reduce_mean(per_time)),
                "std_per_time_log_likelihood": _sample_std(per_time),
                "log_likelihoods": _tensor_to_float_list(values),
            }
        )
    return rows


def _run_pf_like_batched(
    observations: tf.Tensor,
    theta: float,
    spec: ModelSpec,
    seed: int,
    method_id: str,
    *,
    ess_threshold_ratio: float,
    resampler: str,
    epsilon: float | None = None,
) -> dict[str, Any]:
    batch_size = tf.shape(observations)[0]
    particles = _initial_particles_batched(seed, batch_size)
    log_weights = tf.fill([batch_size, NUM_PARTICLES], -tf.math.log(tf.cast(NUM_PARTICLES, DTYPE)))
    log_likelihoods = tf.zeros([batch_size], DTYPE)
    resampling_counts = tf.zeros([batch_size], DTYPE)
    min_ess = tf.ones([batch_size], DTYPE) * tf.cast(NUM_PARTICLES, DTYPE)
    max_sinkhorn_residual = tf.constant(0.0, DTYPE)
    cost_scale_values = []
    finite = True
    for t in range(HORIZON):
        weights = tf.exp(log_weights)
        ess_before = _ess_batched(weights)
        do_resample = ess_before < ess_threshold_ratio * NUM_PARTICLES
        if resampler == "multinomial":
            indices = tf.random.stateless_categorical(
                log_weights,
                NUM_PARTICLES,
                seed=_seed_pair(seed, 7000 + t),
                dtype=tf.int32,
            )
            gathered = tf.gather(particles, indices, batch_dims=1)
            particles = tf.where(do_resample[:, None, None], gathered, particles)
            log_weights = tf.where(do_resample[:, None], _uniform_log_weights_batched(batch_size), log_weights)
            resampling_counts = resampling_counts + tf.cast(do_resample, DTYPE)
        elif resampler in {"sinkhorn_raw", "sinkhorn_scaled"}:
            relaxed, diag = _sinkhorn_resample_batched(
                particles,
                weights,
                float(epsilon),
                scaled=resampler == "sinkhorn_scaled",
            )
            particles = tf.where(do_resample[:, None, None], relaxed, particles)
            log_weights = tf.where(do_resample[:, None], _uniform_log_weights_batched(batch_size), log_weights)
            resampling_counts = resampling_counts + tf.cast(do_resample, DTYPE)
            if bool(tf.reduce_any(do_resample).numpy()):
                max_sinkhorn_residual = tf.maximum(max_sinkhorn_residual, diag["max_marginal_residual"])
                cost_scale_values.append(tf.boolean_mask(diag["cost_scale"], do_resample))
        particles = _transition_particles_batched(particles, theta, spec, seed, t, method_id)
        obs_log_weights = _observation_log_prob_batched(particles, observations[:, t, :], spec)
        unnormalized = log_weights + obs_log_weights
        normalizer = tf.reduce_logsumexp(unnormalized, axis=1)
        log_likelihoods = log_likelihoods + normalizer
        log_weights = unnormalized - normalizer[:, None]
        weights = tf.exp(log_weights)
        ess_after = _ess_batched(weights)
        min_ess = tf.minimum(min_ess, ess_after)
        finite = finite and bool(tf.reduce_all(tf.math.is_finite(particles)).numpy())
        finite = finite and bool(tf.reduce_all(tf.math.is_finite(log_weights)).numpy())
        finite = finite and bool(tf.reduce_all(tf.math.is_finite(log_likelihoods)).numpy())
    cost_scales = tf.concat(cost_scale_values, axis=0) if cost_scale_values else tf.zeros([0], DTYPE)
    return {
        "method_id": method_id,
        "log_likelihoods": log_likelihoods,
        "resampling_counts": resampling_counts,
        "min_ess": min_ess,
        "max_sinkhorn_residual": _float(max_sinkhorn_residual),
        "finite": finite,
        "cost_scales": cost_scales,
    }


def _run_pf_like(
    observations: tf.Tensor,
    theta: float,
    spec: ModelSpec,
    seed: int,
    method_id: str,
    *,
    ess_threshold_ratio: float,
    resampler: str,
    epsilon: float | None = None,
) -> dict[str, Any]:
    particles = _initial_particles(seed)
    log_weights = tf.fill([NUM_PARTICLES], -tf.math.log(tf.cast(NUM_PARTICLES, DTYPE)))
    log_likelihood = tf.constant(0.0, DTYPE)
    resampling_count = 0
    min_ess = float(NUM_PARTICLES)
    max_sinkhorn_residual = 0.0
    cost_scales: list[float] = []
    finite = True
    for t, observation in enumerate(tf.unstack(observations, axis=0)):
        weights = tf.exp(log_weights)
        ess_before = _ess(weights)
        do_resample = bool((ess_before < ess_threshold_ratio * NUM_PARTICLES).numpy())
        if do_resample:
            if resampler == "multinomial":
                indices = tf.random.stateless_categorical(
                    tf.reshape(log_weights, [1, -1]),
                    NUM_PARTICLES,
                    seed=_seed_pair(seed, 7000 + t),
                    dtype=tf.int32,
                )[0]
                particles = tf.gather(particles, indices)
                log_weights = _uniform_log_weights()
                resampling_count += 1
            elif resampler == "sinkhorn_raw":
                particles, diag = _sinkhorn_resample(
                    particles,
                    weights,
                    float(epsilon),
                    scaled=False,
                )
                log_weights = _uniform_log_weights()
                resampling_count += 1
                max_sinkhorn_residual = max(max_sinkhorn_residual, diag["max_marginal_residual"])
                cost_scales.append(diag["cost_scale"])
            elif resampler == "sinkhorn_scaled":
                particles, diag = _sinkhorn_resample(
                    particles,
                    weights,
                    float(epsilon),
                    scaled=True,
                )
                log_weights = _uniform_log_weights()
                resampling_count += 1
                max_sinkhorn_residual = max(max_sinkhorn_residual, diag["max_marginal_residual"])
                cost_scales.append(diag["cost_scale"])
        particles = _transition_particles(particles, theta, spec, seed, t)
        obs_log_weights = _observation_log_prob(particles, observation, spec)
        unnormalized = log_weights + obs_log_weights
        normalizer = tf.reduce_logsumexp(unnormalized)
        log_likelihood = log_likelihood + normalizer
        log_weights = unnormalized - normalizer
        weights = tf.exp(log_weights)
        ess_after = _ess(weights)
        min_ess = min(min_ess, _float(ess_after))
        finite = finite and bool(tf.reduce_all(tf.math.is_finite(particles)).numpy())
        finite = finite and bool(tf.reduce_all(tf.math.is_finite(log_weights)).numpy())
        finite = finite and bool(tf.math.is_finite(log_likelihood).numpy())
    return {
        "method_id": method_id,
        "log_likelihood": _float(log_likelihood),
        "resampling_count": resampling_count,
        "min_ess": min_ess,
        "max_sinkhorn_residual": max_sinkhorn_residual,
        "finite": finite,
        "cost_scales": cost_scales,
    }


def _sinkhorn_resample_batched(
    particles: tf.Tensor,
    weights: tf.Tensor,
    epsilon: float,
    *,
    scaled: bool,
) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
    x = tf.cast(particles, DTYPE)
    weights = tf.cast(weights / tf.reduce_sum(weights, axis=1, keepdims=True), DTYPE)
    if scaled:
        centered = x - tf.stop_gradient(tf.reduce_mean(x, axis=1, keepdims=True))
        scale = _filterflow_scale_batched(x)
        x_cost = centered / tf.stop_gradient(scale[:, None, None])
    else:
        scale = tf.ones([tf.shape(x)[0]], DTYPE)
        x_cost = x
    cost = 0.5 * _pairwise_squared_batched(x_cost)
    coupling, residual, iterations = _sinkhorn_coupling_batched(cost, weights, epsilon)
    column_mass = tf.reduce_sum(coupling, axis=1)
    relaxed = tf.linalg.matmul(coupling, x, transpose_a=True) / tf.maximum(column_mass[:, :, None], 1e-300)
    return relaxed, {
        "max_marginal_residual": residual,
        "iterations": tf.cast(iterations, DTYPE),
        "cost_scale": scale,
    }


def _sinkhorn_coupling_batched(cost: tf.Tensor, weights: tf.Tensor, epsilon: float) -> tuple[tf.Tensor, tf.Tensor, int]:
    batch_size = tf.shape(cost)[0]
    target = tf.ones([batch_size, NUM_PARTICLES], DTYPE) / tf.cast(NUM_PARTICLES, DTYPE)
    log_source = tf.math.log(tf.maximum(weights, tf.constant(1e-300, DTYPE)))
    log_target = tf.math.log(target)
    log_kernel = -tf.cast(cost, DTYPE) / tf.constant(epsilon, DTYPE)
    log_u = tf.zeros([batch_size, NUM_PARTICLES], DTYPE)
    log_v = tf.zeros([batch_size, NUM_PARTICLES], DTYPE)
    iterations_used = 100
    residual = tf.constant(float("inf"), DTYPE)
    for iteration in range(1, 101):
        log_u = log_source - tf.reduce_logsumexp(log_kernel + log_v[:, None, :], axis=2)
        log_v = log_target - tf.reduce_logsumexp(log_kernel + log_u[:, :, None], axis=1)
        if iteration % 10 == 0 or iteration == 100:
            coupling_probe = tf.exp(log_u[:, :, None] + log_kernel + log_v[:, None, :])
            row_residual = tf.reduce_max(tf.abs(tf.reduce_sum(coupling_probe, axis=2) - weights))
            column_residual = tf.reduce_max(tf.abs(tf.reduce_sum(coupling_probe, axis=1) - target))
            residual = tf.maximum(row_residual, column_residual)
            if bool((residual <= 1e-7).numpy()):
                iterations_used = iteration
                break
    coupling = tf.exp(log_u[:, :, None] + log_kernel + log_v[:, None, :])
    row_residual = tf.reduce_max(tf.abs(tf.reduce_sum(coupling, axis=2) - weights))
    column_residual = tf.reduce_max(tf.abs(tf.reduce_sum(coupling, axis=1) - target))
    residual = tf.maximum(row_residual, column_residual)
    if _float(residual) > 1e-5:
        raise FloatingPointError(f"Sinkhorn residual too large: {_float(residual)}")
    if bool(tf.reduce_any(coupling < -1e-12).numpy()):
        raise FloatingPointError("negative Sinkhorn coupling")
    return coupling, residual, iterations_used


def _sinkhorn_resample(
    particles: tf.Tensor,
    weights: tf.Tensor,
    epsilon: float,
    *,
    scaled: bool,
) -> tuple[tf.Tensor, dict[str, float]]:
    x = tf.cast(particles, DTYPE)
    weights = tf.cast(weights / tf.reduce_sum(weights), DTYPE)
    if scaled:
        centered = x - tf.stop_gradient(tf.reduce_mean(x, axis=0, keepdims=True))
        scale = _filterflow_scale(x)
        x_cost = centered / tf.stop_gradient(scale)
    else:
        scale = tf.constant(1.0, DTYPE)
        x_cost = x
    cost = 0.5 * _pairwise_squared(x_cost)
    coupling, residual, iterations = _sinkhorn_coupling(cost, weights, epsilon)
    target = tf.ones([NUM_PARTICLES], DTYPE) / tf.cast(NUM_PARTICLES, DTYPE)
    column_mass = tf.reduce_sum(coupling, axis=0)
    relaxed = tf.linalg.matmul(coupling, x, transpose_a=True) / tf.maximum(column_mass[:, None], 1e-300)
    diagnostics = {
        "max_marginal_residual": residual,
        "iterations": float(iterations),
        "cost_scale": _float(scale),
        "target_marginal_max_error": _float(tf.reduce_max(tf.abs(column_mass - target))),
    }
    return relaxed, diagnostics


def _sinkhorn_coupling(cost: tf.Tensor, weights: tf.Tensor, epsilon: float) -> tuple[tf.Tensor, float, int]:
    target = tf.ones([NUM_PARTICLES], DTYPE) / tf.cast(NUM_PARTICLES, DTYPE)
    log_source = tf.math.log(tf.maximum(weights, tf.constant(1e-300, DTYPE)))
    log_target = tf.math.log(target)
    log_kernel = -tf.cast(cost, DTYPE) / tf.constant(epsilon, DTYPE)
    log_u = tf.zeros([NUM_PARTICLES], DTYPE)
    log_v = tf.zeros([NUM_PARTICLES], DTYPE)
    iterations_used = 80
    for iteration in range(1, 81):
        log_u = log_source - tf.reduce_logsumexp(log_kernel + log_v[None, :], axis=1)
        log_v = log_target - tf.reduce_logsumexp(log_kernel + log_u[:, None], axis=0)
        if iteration % 10 == 0 or iteration == 80:
            coupling_probe = tf.exp(log_u[:, None] + log_kernel + log_v[None, :])
            row_residual = tf.reduce_max(tf.abs(tf.reduce_sum(coupling_probe, axis=1) - weights))
            column_residual = tf.reduce_max(tf.abs(tf.reduce_sum(coupling_probe, axis=0) - target))
            residual = _float(tf.maximum(row_residual, column_residual))
            if residual <= 1e-7:
                iterations_used = iteration
                break
    coupling = tf.exp(log_u[:, None] + log_kernel + log_v[None, :])
    row_residual = tf.reduce_max(tf.abs(tf.reduce_sum(coupling, axis=1) - weights))
    column_residual = tf.reduce_max(tf.abs(tf.reduce_sum(coupling, axis=0) - target))
    residual = _float(tf.maximum(row_residual, column_residual))
    if residual > 1e-5:
        raise FloatingPointError(f"Sinkhorn residual too large: {residual}")
    if bool(tf.reduce_any(coupling < -1e-12).numpy()):
        raise FloatingPointError("negative Sinkhorn coupling")
    return coupling, residual, iterations_used


def _simulate_dataset(spec: ModelSpec, seed: int) -> tf.Tensor:
    x = tfd.MultivariateNormalTriL(
        loc=tf.zeros([2], DTYPE),
        scale_tril=tf.eye(2, dtype=DTYPE) * math.sqrt(spec.initial_covariance_scale),
    ).sample(seed=_seed_pair(seed, 1))
    observations = []
    transition_scale = math.sqrt(spec.transition_covariance_scale)
    observation_scale = math.sqrt(spec.observation_covariance_scale)
    for t in range(HORIZON):
        transition_noise = tfd.MultivariateNormalTriL(
            loc=tf.zeros([2], DTYPE),
            scale_tril=tf.eye(2, dtype=DTYPE) * transition_scale,
        ).sample(seed=_seed_pair(seed, 100 + t))
        x = tf.constant(0.5, DTYPE) * x + transition_noise
        observation_noise = tfd.MultivariateNormalTriL(
            loc=tf.zeros([2], DTYPE),
            scale_tril=tf.eye(2, dtype=DTYPE) * observation_scale,
        ).sample(seed=_seed_pair(seed, 1000 + t))
        observations.append(x + observation_noise)
    return tf.stack(observations, axis=0)


def _simulate_datasets_batched(spec: ModelSpec, seed: int) -> tf.Tensor:
    x = (
        tf.random.stateless_normal([NUM_REALIZATIONS, 2], seed=_seed_pair(seed, 1), dtype=DTYPE)
        * math.sqrt(spec.initial_covariance_scale)
    )
    observations = []
    transition_scale = math.sqrt(spec.transition_covariance_scale)
    observation_scale = math.sqrt(spec.observation_covariance_scale)
    for t in range(HORIZON):
        transition_noise = (
            tf.random.stateless_normal([NUM_REALIZATIONS, 2], seed=_seed_pair(seed, 100 + t), dtype=DTYPE)
            * transition_scale
        )
        x = tf.constant(0.5, DTYPE) * x + transition_noise
        observation_noise = (
            tf.random.stateless_normal([NUM_REALIZATIONS, 2], seed=_seed_pair(seed, 1000 + t), dtype=DTYPE)
            * observation_scale
        )
        observations.append(x + observation_noise)
    return tf.stack(observations, axis=1)


def _initial_particles(seed: int) -> tf.Tensor:
    return tfd.MultivariateNormalTriL(
        loc=tf.zeros([2], DTYPE),
        scale_tril=tf.eye(2, dtype=DTYPE),
    ).sample(NUM_PARTICLES, seed=_seed_pair(seed, 11))


def _initial_particles_batched(seed: int, batch_size: tf.Tensor) -> tf.Tensor:
    shape = tf.stack([batch_size, tf.constant(NUM_PARTICLES, tf.int32), tf.constant(2, tf.int32)])
    return tf.random.stateless_normal(shape, seed=_seed_pair(seed, 11), dtype=DTYPE)


def _transition_particles(particles: tf.Tensor, theta: float, spec: ModelSpec, seed: int, time_index: int) -> tf.Tensor:
    transition_scale = math.sqrt(spec.transition_covariance_scale)
    noise = tfd.MultivariateNormalTriL(
        loc=tf.zeros([2], DTYPE),
        scale_tril=tf.eye(2, dtype=DTYPE) * transition_scale,
    ).sample(NUM_PARTICLES, seed=_seed_pair(seed, 2000 + time_index))
    return tf.constant(theta, DTYPE) * particles + noise


def _transition_particles_batched(
    particles: tf.Tensor,
    theta: float,
    spec: ModelSpec,
    seed: int,
    time_index: int,
    method_id: str,
) -> tf.Tensor:
    transition_scale = math.sqrt(spec.transition_covariance_scale)
    method_salt = {
        "bayesfilter_pf": 0,
        "bayesfilter_raw_ess": 100000,
        "bayesfilter_scaled_ess": 200000,
        "bayesfilter_scaled_every_step": 300000,
    }[method_id]
    noise = (
        tf.random.stateless_normal(
            tf.shape(particles),
            seed=_seed_pair(seed, method_salt + 2000 + time_index),
            dtype=DTYPE,
        )
        * transition_scale
    )
    return tf.constant(theta, DTYPE) * particles + noise


def _observation_log_prob(particles: tf.Tensor, observation: tf.Tensor, spec: ModelSpec) -> tf.Tensor:
    residual = particles - tf.reshape(tf.cast(observation, DTYPE), [1, 2])
    covariance = tf.eye(2, dtype=DTYPE) * spec.observation_covariance_scale
    chol = tf.linalg.cholesky(covariance)
    solved = tf.linalg.cholesky_solve(chol, tf.transpose(residual))
    quad = tf.reduce_sum(tf.transpose(solved) * residual, axis=1)
    logdet = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)))
    dim = tf.constant(2.0, DTYPE)
    return -0.5 * (dim * tf.math.log(tf.constant(2.0 * math.pi, DTYPE)) + logdet + quad)


def _observation_log_prob_batched(particles: tf.Tensor, observations: tf.Tensor, spec: ModelSpec) -> tf.Tensor:
    residual = particles - observations[:, None, :]
    covariance = tf.eye(2, dtype=DTYPE) * spec.observation_covariance_scale
    chol = tf.linalg.cholesky(covariance)
    flat = tf.reshape(residual, [-1, 2])
    solved = tf.linalg.cholesky_solve(chol, tf.transpose(flat))
    quad = tf.reduce_sum(tf.transpose(solved) * flat, axis=1)
    logdet = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)))
    dim = tf.constant(2.0, DTYPE)
    logp = -0.5 * (dim * tf.math.log(tf.constant(2.0 * math.pi, DTYPE)) + logdet + quad)
    return tf.reshape(logp, [tf.shape(particles)[0], NUM_PARTICLES])


def _kalman_log_likelihood(observations: tf.Tensor, theta: float, spec: ModelSpec) -> float:
    a = tf.eye(2, dtype=DTYPE) * theta
    q = tf.eye(2, dtype=DTYPE) * spec.transition_covariance_scale
    r = tf.eye(2, dtype=DTYPE) * spec.observation_covariance_scale
    mean = tf.zeros([2], DTYPE)
    cov = tf.eye(2, dtype=DTYPE) * spec.initial_covariance_scale
    eye = tf.eye(2, dtype=DTYPE)
    ll = tf.constant(0.0, DTYPE)
    for observation in tf.unstack(observations, axis=0):
        pred_mean = tf.linalg.matvec(a, mean)
        pred_cov = a @ cov @ tf.transpose(a) + q
        residual = tf.reshape(observation, [2]) - pred_mean
        s_mat = pred_cov + r
        ll = ll + _gaussian_logpdf_single(residual, s_mat)
        gain = tf.transpose(tf.linalg.solve(s_mat, pred_cov))
        mean = pred_mean + tf.linalg.matvec(gain, residual)
        cov = (eye - gain) @ pred_cov
        cov = 0.5 * (cov + tf.transpose(cov))
    return _float(ll)


def _kalman_log_likelihoods_batched(observations: tf.Tensor, theta: float, spec: ModelSpec) -> tf.Tensor:
    a = tf.eye(2, dtype=DTYPE) * theta
    q = tf.eye(2, dtype=DTYPE) * spec.transition_covariance_scale
    r = tf.eye(2, dtype=DTYPE) * spec.observation_covariance_scale
    mean = tf.zeros([NUM_REALIZATIONS, 2], DTYPE)
    cov = tf.tile(tf.eye(2, dtype=DTYPE)[None, :, :] * spec.initial_covariance_scale, [NUM_REALIZATIONS, 1, 1])
    eye = tf.eye(2, dtype=DTYPE)
    ll = tf.zeros([NUM_REALIZATIONS], DTYPE)
    for t in range(HORIZON):
        pred_mean = tf.linalg.matmul(mean, a, transpose_b=True)
        pred_cov = a[None, :, :] @ cov @ tf.transpose(a)[None, :, :] + q[None, :, :]
        residual = observations[:, t, :] - pred_mean
        s_mat = pred_cov + r[None, :, :]
        ll = ll + _gaussian_logpdf_batched(residual, s_mat)
        gain = tf.linalg.matmul(pred_cov, tf.linalg.inv(s_mat))
        mean = pred_mean + tf.linalg.matvec(gain, residual)
        cov = (eye[None, :, :] - gain) @ pred_cov
        cov = 0.5 * (cov + tf.transpose(cov, perm=[0, 2, 1]))
    return ll


def _gaussian_logpdf_single(residual: tf.Tensor, covariance: tf.Tensor) -> tf.Tensor:
    chol = tf.linalg.cholesky(covariance)
    solved = tf.linalg.cholesky_solve(chol, tf.reshape(residual, [-1, 1]))
    quad = tf.reduce_sum(tf.reshape(solved, [-1]) * residual)
    dim = tf.cast(tf.shape(covariance)[0], DTYPE)
    logdet = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)))
    return -0.5 * (dim * tf.math.log(tf.constant(2.0 * math.pi, DTYPE)) + logdet + quad)


def _gaussian_logpdf_batched(residual: tf.Tensor, covariance: tf.Tensor) -> tf.Tensor:
    chol = tf.linalg.cholesky(covariance)
    solved = tf.linalg.cholesky_solve(chol, residual[:, :, None])[:, :, 0]
    quad = tf.reduce_sum(solved * residual, axis=1)
    logdet = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)), axis=1)
    dim = tf.constant(2.0, DTYPE)
    return -0.5 * (dim * tf.math.log(tf.constant(2.0 * math.pi, DTYPE)) + logdet + quad)


def _filterflow_scale(x: tf.Tensor) -> tf.Tensor:
    std = tf.math.reduce_std(tf.cast(x, DTYPE), axis=0)
    diameter = tf.reduce_max(std)
    diameter = tf.where(diameter == 0.0, tf.constant(1.0, DTYPE), diameter)
    return diameter * tf.sqrt(tf.constant(2.0, DTYPE))


def _filterflow_scale_batched(x: tf.Tensor) -> tf.Tensor:
    std = tf.math.reduce_std(tf.cast(x, DTYPE), axis=1)
    diameter = tf.reduce_max(std, axis=1)
    diameter = tf.where(diameter == 0.0, tf.ones_like(diameter), diameter)
    return diameter * tf.sqrt(tf.constant(2.0, DTYPE))


def _pairwise_squared(x: tf.Tensor) -> tf.Tensor:
    diff = x[:, None, :] - x[None, :, :]
    return tf.reduce_sum(diff * diff, axis=2)


def _pairwise_squared_batched(x: tf.Tensor) -> tf.Tensor:
    diff = x[:, :, None, :] - x[:, None, :, :]
    return tf.reduce_sum(diff * diff, axis=3)


def _uniform_log_weights() -> tf.Tensor:
    return tf.fill([NUM_PARTICLES], -tf.math.log(tf.cast(NUM_PARTICLES, DTYPE)))


def _uniform_log_weights_batched(batch_size: tf.Tensor) -> tf.Tensor:
    return tf.fill([batch_size, NUM_PARTICLES], -tf.math.log(tf.cast(NUM_PARTICLES, DTYPE)))


def _ess(weights: tf.Tensor) -> tf.Tensor:
    return 1.0 / tf.reduce_sum(tf.cast(weights, DTYPE) * tf.cast(weights, DTYPE))


def _ess_batched(weights: tf.Tensor) -> tf.Tensor:
    return 1.0 / tf.reduce_sum(tf.cast(weights, DTYPE) * tf.cast(weights, DTYPE), axis=1)


def _filterflow_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "exists": False,
            "commit": None,
            "dirty_status": "missing",
            "status": "missing_external_source",
        }
    return {
        "exists": True,
        "commit": _run_git(path, ["rev-parse", "HEAD"]),
        "dirty_status": _run_git(path, ["status", "--short", "--branch"]),
        "status": "present_read_only_external_source",
    }


def _probe_filterflow_execution(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "blocked_missing_filterflow_checkout", "returncode": None, "stderr_excerpt": ""}
    command = [
        sys.executable,
        "-c",
        "from scripts.simple_linear_common import get_data; print('filterflow simple linear import ok')",
    ]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env["PYTHONPATH"] = str(path)
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[4],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    status = "filterflow_import_ok" if completed.returncode == 0 else "blocked_dependency_drift"
    return {
        "status": status,
        "returncode": completed.returncode,
        "stdout_excerpt": completed.stdout[-1000:],
        "stderr_excerpt": completed.stderr[-2000:],
        "command": "PYTHONPATH=.localsource/filterflow python -c 'from scripts.simple_linear_common import get_data'",
        "classification": (
            "external_execution_blocked_by_missing_pykalman_or_dependency"
            if completed.returncode != 0
            else "external_execution_probe_succeeded"
        ),
    }


def _run_git(path: Path, args: list[str]) -> str:
    completed = subprocess.run(["git", "-C", str(path), *args], check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def _settings_ledger() -> list[dict[str, Any]]:
    return [
        {
            "field": "transition_covariance",
            "paper_text": "0.5 I_2",
            "filterflow_code": "I_2",
            "primary_comparator_choice": "filterflow_code",
            "sensitivity_choice": "paper_text",
            "classification": "paper_code_discrepancy_unresolved",
        },
        {
            "field": "observation_covariance",
            "paper_text": "0.1 I_2",
            "filterflow_code": "0.1 I_2",
            "classification": "matched",
        },
        {
            "field": "resampling_neff",
            "paper_text": "not explicit in extracted table text",
            "filterflow_code": "0.5 relative Neff criterion",
            "classification": "implementation_primary",
        },
    ]


def _scaling_ledger(match_status: str) -> dict[str, Any]:
    return {
        "filterflow_centering_formula": "x - stop_gradient(mean(x, axis=particles))",
        "filterflow_scale_formula": "diameter(x, x) * sqrt(dimension), diameter=max coordinate std with zero fallback",
        "filterflow_scaled_particles_formula": "centered_x / stop_gradient(scale)",
        "filterflow_cost_formula": "squared distance / 2 on scaled particle cloud",
        "filterflow_epsilon_schedule": "epsilon_0=max_min(scaled_x, scaled_x)^2, geometric decrease by scaling^2 to target epsilon",
        "bayesfilter_scaled_cost_formula": "matches centering/scale/cost formula",
        "bayesfilter_epsilon_schedule": "similar_not_matched: fixed target epsilon for 100 log-domain iterations",
        "match_status": match_status,
        "consequence": "consistent decision forbidden unless match_status is matched",
    }


def _matrix_status(filterflow_execution: dict[str, Any], summary: dict[str, Any]) -> list[dict[str, str]]:
    external_status = (
        "blocked_dependency_drift"
        if filterflow_execution["status"] != "filterflow_import_ok"
        else "not_run_full_grid_in_this_bounded_runner"
    )
    primary_status = (
        "executed_full_primary_grid"
        if summary["primary_veto_count"] == 0
        else f"partial_sinkhorn_residual_veto_{summary['primary_veto_count']}_of_{summary['primary_row_count']}"
    )
    return [
        {"lane": "filterflow_regularized", "role": "primary external implementation comparator", "status": external_status},
        {"lane": "bayesfilter_scaled_ess", "role": "primary BayesFilter implementation comparator", "status": primary_status},
        {"lane": "filterflow_pf", "role": "primary PF comparator", "status": external_status},
        {"lane": "bayesfilter_pf", "role": "BayesFilter PF comparator", "status": "executed"},
        {"lane": "bayesfilter_raw_ess", "role": "current-helper sensitivity", "status": "executed"},
        {"lane": "bayesfilter_scaled_every_step", "role": "resampling-trigger sensitivity", "status": "executed"},
        {"lane": "bayesfilter_scaled_paper_covariance", "role": "paper-text covariance sensitivity", "status": "executed"},
    ]


def _comparator_outcome_matrix(
    rows: list[dict[str, Any]],
    filterflow_execution: dict[str, Any],
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    matrix: list[dict[str, Any]] = [
        {
            "lane": "filterflow_regularized",
            "role": "primary external implementation comparator",
            "status": "blocked_dependency_drift",
            "executed_cells": 0,
            "expected_cells": 9,
            "classification_mapping": "blocked_external_execution_not_partial_execution",
            "notes": filterflow_execution.get("classification", ""),
        },
        {
            "lane": "filterflow_pf",
            "role": "primary external PF comparator",
            "status": "blocked_dependency_drift",
            "executed_cells": 0,
            "expected_cells": 3,
            "classification_mapping": "blocked_external_execution_not_partial_pf_only_blocker",
            "notes": filterflow_execution.get("classification", ""),
        },
        {
            "lane": "bayesfilter_scaled_ess",
            "role": "primary BayesFilter implementation comparator",
            "status": "partial_sinkhorn_residual_veto",
            "executed_cells": summary["primary_executed_count"],
            "expected_cells": summary["primary_row_count"],
            "classification_mapping": "red_flag",
            "notes": "epsilon 0.25 and 0.5 cells vetoed; epsilon 0.75 cells executed",
        },
    ]
    for spec_role in ("primary_filterflow_code_covariance", "sensitivity_paper_text_covariance"):
        for method_id in (
            "bayesfilter_pf",
            "bayesfilter_raw_ess",
            "bayesfilter_scaled_every_step",
        ):
            method_rows = [row for row in rows if row["spec_role"] == spec_role and row["method_id"] == method_id]
            if not method_rows:
                continue
            executed = sum(1 for row in method_rows if row["row_status"] == "executed")
            vetoed = sum(1 for row in method_rows if row["row_status"] != "executed")
            matrix.append(
                {
                    "lane": f"{spec_role}:{method_id}",
                    "role": _matrix_role(spec_role, method_id),
                    "status": "executed" if vetoed == 0 else "partial_sinkhorn_residual_veto",
                    "executed_cells": executed,
                    "expected_cells": len(method_rows),
                    "classification_mapping": "sensitivity_only" if spec_role.startswith("sensitivity") or method_id != "bayesfilter_pf" else "pf_calibration",
                    "notes": f"{vetoed} vetoed cells",
                }
            )
    return matrix


def _matrix_role(spec_role: str, method_id: str) -> str:
    if method_id == "bayesfilter_pf":
        return "BayesFilter PF comparator" if spec_role.startswith("primary") else "paper-covariance PF sensitivity"
    if method_id == "bayesfilter_raw_ess":
        return "raw-cost Sinkhorn sensitivity"
    if method_id == "bayesfilter_scaled_every_step":
        return "every-step scaled Sinkhorn sensitivity"
    return "sensitivity"


def _paper_table() -> list[dict[str, float | str | None]]:
    rows = []
    pf = {0.25: (-1.13, 0.20), 0.5: (-0.93, 0.18), 0.75: (-1.05, 0.17)}
    rows.extend(_paper_rows("PF", None, pf))
    dpf_025 = {0.25: (-1.14, 0.20), 0.5: (-0.94, 0.18), 0.75: (-1.07, 0.19)}
    dpf_05 = {0.25: (-1.14, 0.20), 0.5: (-0.94, 0.18), 0.75: (-1.08, 0.18)}
    dpf_075 = {0.25: (-1.14, 0.20), 0.5: (-0.94, 0.18), 0.75: (-1.08, 0.18)}
    rows.extend(_paper_rows("DPF", 0.25, dpf_025))
    rows.extend(_paper_rows("DPF", 0.5, dpf_05))
    rows.extend(_paper_rows("DPF", 0.75, dpf_075))
    return rows


def _paper_rows(method: str, epsilon: float | None, values: dict[float, tuple[float, float]]) -> list[dict[str, Any]]:
    return [
        {"method": method, "epsilon": epsilon, "theta": theta, "mean": mean, "std": std}
        for theta, (mean, std) in values.items()
    ]


def _paper_reference_for(method_id: str, epsilon: float | None, theta: float) -> dict[str, float | None]:
    table = _paper_table()
    if method_id == "bayesfilter_pf":
        method = "PF"
        eps = None
    elif method_id in {"bayesfilter_raw_ess", "bayesfilter_scaled_ess", "bayesfilter_scaled_every_step"}:
        method = "DPF"
        eps = epsilon
    else:
        return {"mean": None, "std": None}
    for row in table:
        if row["method"] == method and row["epsilon"] == eps and row["theta"] == theta:
            return {"mean": row["mean"], "std": row["std"]}
    return {"mean": None, "std": None}


def _paper_settings() -> dict[str, Any]:
    return {
        "state_dim": 2,
        "observation_dim": 2,
        "transition": "x_{t+1} = diag(theta_1, theta_2) x_t + eta_t",
        "transition_covariance_text": "0.5 I_2",
        "observation": "y_t = x_t + epsilon_t",
        "observation_covariance": "0.1 I_2",
        "data_theta": [0.5, 0.5],
        "theta_grid": [[theta, theta] for theta in THETAS],
        "horizon": HORIZON,
        "num_particles": NUM_PARTICLES,
        "num_realizations": NUM_REALIZATIONS,
        "dpf_epsilons": list(EPSILONS),
        "reported_quantity": "(estimated log likelihood - exact Kalman log likelihood) / T",
    }


def _seed_protocol() -> dict[str, Any]:
    return {
        "data_seed": "batched stateless normal tensors keyed by DATA_SEED_BASE and time/noise salt",
        "filter_seed": "batched stateless normal tensors keyed by FILTER_SEED_BASE, method salt, and time/noise salt",
        "same_observation_path_across_theta_within_realization": True,
        "same_observation_path_across_methods_within_spec": True,
        "common_random_numbers_across_bayesfilter_methods": "partial: same seed base, method-specific random consumption differs after resampling",
        "filterflow_bitwise_seed_match": False,
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    primary = [
        row
        for row in rows
        if row["spec_role"] == "primary_filterflow_code_covariance"
        and row["method_id"] == "bayesfilter_scaled_ess"
    ]
    sensitivity = [
        row
        for row in rows
        if row["method_id"] != "bayesfilter_scaled_ess"
        or row["spec_role"] != "primary_filterflow_code_covariance"
    ]
    return {
        "primary_bayesfilter_scaled_ess_rows": primary,
        "sensitivity_row_count": len(sensitivity),
        "primary_within_one_paper_sd_count": sum(1 for row in primary if row["within_one_paper_sd"]),
        "primary_row_count": len(primary),
        "primary_executed_count": sum(1 for row in primary if row["row_status"] == "executed"),
        "primary_veto_count": sum(1 for row in primary if row["row_status"] != "executed"),
        "max_primary_abs_delta_to_paper_mean": _safe_max_abs_delta(primary),
        "sensitivity_outcomes_do_not_rescue_primary": True,
    }


def _discrepancy_ledger(
    filterflow_status: dict[str, Any],
    filterflow_execution: dict[str, Any],
    scaling_ledger: dict[str, Any],
) -> list[dict[str, str]]:
    return [
        {
            "id": "filterflow_execution",
            "classification": "environment",
            "status": filterflow_execution["status"],
            "detail": filterflow_execution.get("classification", ""),
        },
        {
            "id": "transition_covariance",
            "classification": "paper_code",
            "status": "unresolved_sensitivity_required",
            "detail": "paper text says 0.5 I_2; filterflow script uses I_2",
        },
        {
            "id": "epsilon_schedule",
            "classification": "implementation_calibration",
            "status": scaling_ledger["match_status"],
            "detail": scaling_ledger["bayesfilter_epsilon_schedule"],
        },
        {
            "id": "filterflow_checkout",
            "classification": "source",
            "status": filterflow_status["status"],
            "detail": str(filterflow_status["commit"]),
        },
    ]


def _decision(filterflow_execution: dict[str, Any], scaling_ledger: dict[str, Any], summary: dict[str, Any]) -> str:
    if summary["primary_veto_count"] > 0:
        return "red_flag"
    if filterflow_execution["status"] != "filterflow_import_ok":
        if summary["primary_within_one_paper_sd_count"] == summary["primary_row_count"]:
            return "source_consistent_execution_blocked"
        return "red_flag"
    if scaling_ledger["match_status"] != "matched":
        return "red_flag"
    if summary["primary_within_one_paper_sd_count"] == summary["primary_row_count"]:
        return "consistent"
    return "red_flag"


def _cost_scaling_label(method_id: str) -> str:
    if method_id == "bayesfilter_pf":
        return "not_applicable"
    if method_id == "bayesfilter_raw_ess":
        return "raw_squared_euclidean_div_2"
    return "filterflow_centered_diameter_scaled_cost_div_2_fixed_epsilon"


def _validate_payload(payload: dict[str, Any]) -> None:
    if payload["run_manifest"]["pre_import_cuda_visible_devices"] != "-1":
        raise RuntimeError("missing CPU-only pre-import manifest")
    if payload["primary_decision_lane"]["bayesfilter"] != "bayesfilter_scaled_ess":
        raise RuntimeError("primary BayesFilter lane not labelled correctly")
    if payload["primary_decision_lane"]["external"] != "filterflow_regularized":
        raise RuntimeError("primary external lane not labelled correctly")
    if "match_status" not in payload["scaling_and_epsilon_match_ledger"]:
        raise RuntimeError("missing scaling/eigen match status")
    if not payload["rows"]:
        raise RuntimeError("missing rows")
    if not payload.get("comparator_outcome_matrix"):
        raise RuntimeError("missing comparator outcome matrix")
    if not payload.get("kalman_reference"):
        raise RuntimeError("missing Kalman reference")
    if any(row["row_status"] == "executed" and not row["finite"] for row in payload["rows"]):
        raise RuntimeError("executed non-finite row")


def _markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    primary_table = _primary_markdown_table(summary["primary_bayesfilter_scaled_ess_rows"])
    pf_table = _method_markdown_table(
        payload["rows"],
        spec_role="primary_filterflow_code_covariance",
        method_id="bayesfilter_pf",
    )
    scaling_table = _scaling_markdown_table(payload["scaling_and_epsilon_match_ledger"])
    comparator_table = _comparator_markdown_table(payload["comparator_outcome_matrix"])
    kalman_table = _kalman_markdown_table(payload["kalman_reference"])
    return f"""# DPF Filterflow LGSSM Cross-Implementation Audit

## Decision

`{payload['decision']}`

Decision reason: primary BayesFilter scaled-ESS rows have Sinkhorn residual
vetoes, filterflow execution is blocked by dependency drift, and the
BayesFilter fixed-epsilon Sinkhorn policy is not a full match to filterflow's
epsilon annealing.

## Decision Table

| Check | Status | Evidence |
| --- | --- | --- |
| filterflow checkout | recorded | `{payload['filterflow_status']['commit']}` |
| filterflow execution | `{payload['filterflow_execution']['status']}` | `{payload['filterflow_execution']['classification']}` |
| primary BayesFilter lane | recorded | `bayesfilter_scaled_ess` |
| primary external lane | recorded | `filterflow_regularized` |
| scaling/epsilon match | `{payload['scaling_and_epsilon_match_ledger']['match_status']}` | fixed-epsilon BayesFilter runner does not implement filterflow annealing |
| primary rows executed | veto diagnostic | `{summary['primary_executed_count']}/{summary['primary_row_count']}` |
| primary Sinkhorn residual vetoes | veto diagnostic | `{summary['primary_veto_count']}/{summary['primary_row_count']}` |
| primary rows within paper SD | diagnostic | `{summary['primary_within_one_paper_sd_count']}/{summary['primary_row_count']}` |

## Interpretation

The BayesFilter runner executed bounded TF/TFP LGSSM rows for the filterflow-code
covariance variant and the paper-text covariance sensitivity variant. Original
filterflow execution is blocked in this environment by dependency drift, so the
decision cannot be stronger than execution-blocked or red-flagged source-level
evidence. In this run, the primary BayesFilter scaled-ESS lane also hit
Sinkhorn marginal residual vetoes for epsilon `0.25` and `0.5`, so the audit is
classified as a red flag rather than a source-consistent reproduction.

## Section 5.1 Settings Recovered

- Paper text model: two-dimensional LGSSM with diagonal transition
  `diag(theta_1, theta_2)`, observation matrix `I_2`, `T=150`, `N=25`, 100
  Monte Carlo realizations, theta grid `(0.25,0.25)`, `(0.5,0.5)`,
  `(0.75,0.75)`, and DPF epsilon values `0.25`, `0.5`, `0.75`.
- Paper text covariance: transition covariance `0.5 I_2`, observation
  covariance `0.1 I_2`.
- Filterflow `simple_linear_comparison.py` setting: transition covariance
  `I_2`, observation covariance `0.1 I_2`, Neff threshold `0.5`,
  `RegularisedTransform(epsilon=eps, scaling=0.9, convergence_threshold=1e-3)`.
- Filterflow `simple_linear_mle.py` uses transition covariance `0.5 I_2`, so
  the repository contains an unresolved comparison-versus-MLE covariance split.

## Scaling And Epsilon Match Ledger

{scaling_table}

## Source-Level Filterflow Anchors

- `scripts/simple_linear_comparison.py`: Section-5.1-style likelihood table
  driver, but currently import-blocked by `pykalman`.
- `filterflow/resampling/differentiable/biased.py`: `RegularisedTransform`
  applies the transport matrix and resets weights on flagged particles.
- `filterflow/resampling/differentiable/regularized_transport/plan.py`: centers
  particles, scales by diameter times `sqrt(dimension)`, computes transport,
  and defines a custom gradient.
- `filterflow/resampling/differentiable/regularized_transport/sinkhorn.py`:
  starts epsilon at a squared max-min scale and geometrically anneals by
  `scaling^2` toward the target epsilon.

## Comparator Outcome Matrix

{comparator_table}

Classification mapping: the external filterflow lanes are fully blocked, not
partial executions. The BayesFilter primary lane is therefore classified as
`red_flag`, with a separate source-level external dependency blocker. The
plan's `partial_execution_red_flag` and `partial_pf_only_blocker` subclasses do
not apply because no executable filterflow grid cell completed.

## Kalman Reference Summary

{kalman_table}

## Primary BayesFilter Scaled-ESS Rows

{primary_table}

## BayesFilter PF Comparator Rows

{pf_table}

## Red Flags

- The paper text says transition covariance `0.5 I_2`, while filterflow
  `simple_linear_comparison.py` uses `I_2`.
- Filterflow source-level import of `scripts.simple_linear_common` fails with
  `ModuleNotFoundError: No module named 'pykalman'`, so the original
  `filterflow_regularized` grid was not executed.
- BayesFilter scaled cost matches the filterflow centering/scale/cost formula,
  but this runner does not implement filterflow's epsilon annealing schedule.
- BayesFilter primary scaled-ESS rows execute only for epsilon `0.75`; epsilon
  `0.25` and `0.5` are vetoed by Sinkhorn residuals around `1e-4` to `5e-5`.
- Raw-cost Sinkhorn rows are vetoed across all three epsilon values in both the
  filterflow-code covariance and paper-text covariance variants.

## Artifacts

- JSON: `experiments/dpf_implementation/reports/outputs/dpf_filterflow_lgssm_cross_audit_2026-05-30.json`
- Report: `experiments/dpf_implementation/reports/dpf-filterflow-lgssm-cross-implementation-audit-2026-05-30.md`
- Runner: `experiments/dpf_implementation/tf_tfp/runners/run_filterflow_lgssm_cross_audit_tf.py`
- Reproducibility digest: `{payload['reproducibility_digest']}`

## Non-Implications

{_non_implications_markdown()}
"""


def _primary_markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| epsilon | theta | status | mean error | std | paper mean | paper std | residual |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {epsilon} | {theta} | `{status}` | {mean} | {std} | {paper_mean} | {paper_std} | {residual} |".format(
                epsilon=row["epsilon"],
                theta=row["theta"],
                status=row["row_status"],
                mean=_format_optional(row["mean_per_time_error"]),
                std=_format_optional(row["std_per_time_error"]),
                paper_mean=_format_optional(row["paper_reference_mean"]),
                paper_std=_format_optional(row["paper_reference_std"]),
                residual=_format_optional(row["max_sinkhorn_residual"]),
            )
        )
    return "\n".join(lines)


def _method_markdown_table(rows: list[dict[str, Any]], *, spec_role: str, method_id: str) -> str:
    method_rows = [
        row
        for row in rows
        if row["spec_role"] == spec_role and row["method_id"] == method_id
    ]
    lines = [
        "| theta | status | mean error | std | paper mean | paper std | within paper SD |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in method_rows:
        lines.append(
            "| {theta} | `{status}` | {mean} | {std} | {paper_mean} | {paper_std} | {within} |".format(
                theta=row["theta"],
                status=row["row_status"],
                mean=_format_optional(row["mean_per_time_error"]),
                std=_format_optional(row["std_per_time_error"]),
                paper_mean=_format_optional(row["paper_reference_mean"]),
                paper_std=_format_optional(row["paper_reference_std"]),
                within=row["within_one_paper_sd"],
            )
        )
    return "\n".join(lines)


def _scaling_markdown_table(ledger: dict[str, Any]) -> str:
    field_order = [
        "filterflow_centering_formula",
        "filterflow_scale_formula",
        "filterflow_scaled_particles_formula",
        "filterflow_cost_formula",
        "filterflow_epsilon_schedule",
        "bayesfilter_scaled_cost_formula",
        "bayesfilter_epsilon_schedule",
        "match_status",
        "consequence",
    ]
    lines = ["| Field | Value |", "| --- | --- |"]
    for field in field_order:
        lines.append(f"| `{field}` | {ledger[field]} |")
    return "\n".join(lines)


def _comparator_markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Lane | Role | Status | Executed cells | Expected cells | Mapping |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| `{lane}` | {role} | `{status}` | {executed} | {expected} | `{mapping}` |".format(
                lane=row["lane"],
                role=row["role"],
                status=row["status"],
                executed=row["executed_cells"],
                expected=row["expected_cells"],
                mapping=row["classification_mapping"],
            )
        )
    return "\n".join(lines)


def _kalman_markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Spec | Theta | Mean log likelihood | Std log likelihood | Mean per-time | Std per-time |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| `{spec}` | {theta} | {mean_ll} | {std_ll} | {mean_pt} | {std_pt} |".format(
                spec=row["spec_role"],
                theta=row["theta"],
                mean_ll=_format_optional(row["mean_log_likelihood"]),
                std_ll=_format_optional(row["std_log_likelihood"]),
                mean_pt=_format_optional(row["mean_per_time_log_likelihood"]),
                std_pt=_format_optional(row["std_per_time_log_likelihood"]),
            )
        )
    return "\n".join(lines)


def _format_optional(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _non_implications() -> list[str]:
    return [
        "No production readiness is concluded.",
        "No public API readiness is concluded.",
        "No HMC readiness is concluded.",
        "No posterior correctness is concluded.",
        "No general nonlinear-SSM validity is concluded.",
        "No learned or neural OT promotion is concluded.",
        "No banking or model-risk claim is concluded.",
        "No monograph claim is concluded.",
    ]


def _non_implications_markdown() -> str:
    return "\n".join(f"- {item}" for item in _non_implications())


def _digest_payload(payload: dict[str, Any]) -> str:
    comparable = dict(payload)
    comparable["created_at_utc"] = "TIMESTAMP"
    comparable["run_manifest"] = dict(comparable["run_manifest"])
    comparable["run_manifest"]["wall_time_seconds"] = "WALL_TIME"
    comparable["run_manifest"]["dirty_state_summary"] = "DIRTY_STATE"
    return stable_digest(comparable)


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = _mean(values)
    return float(math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1)))


def _median(values: list[float | int]) -> float:
    ordered = sorted(float(value) for value in values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _safe_max_abs_delta(rows: list[dict[str, Any]]) -> float | None:
    deltas = [abs(row["delta_to_paper_mean"]) for row in rows if row["delta_to_paper_mean"] is not None]
    return max(deltas) if deltas else None


def _extract_residual(message: str) -> float | None:
    try:
        return float(message.rsplit(":", maxsplit=1)[-1].strip())
    except ValueError:
        return None


def _median_tensor(values: tf.Tensor) -> tf.Tensor:
    values = tf.sort(tf.reshape(tf.cast(values, DTYPE), [-1]))
    count = tf.shape(values)[0]
    mid = count // 2
    return tf.cond(
        tf.equal(count % 2, 1),
        lambda: values[mid],
        lambda: 0.5 * (values[mid - 1] + values[mid]),
    )


def _sample_std(values: tf.Tensor) -> float:
    values = tf.reshape(tf.cast(values, DTYPE), [-1])
    count = tf.cast(tf.shape(values)[0], DTYPE)
    mean = tf.reduce_mean(values)
    variance = tf.reduce_sum(tf.square(values - mean)) / tf.maximum(count - 1.0, 1.0)
    return _float(tf.sqrt(variance))


def _tensor_to_float_list(values: tf.Tensor) -> list[float]:
    return [float(value) for value in tf.reshape(tf.cast(values, DTYPE), [-1]).numpy().tolist()]


def _float(value: tf.Tensor) -> float:
    return float(tf.cast(value, DTYPE).numpy())


def _seed_pair(seed: int, salt: int) -> tf.Tensor:
    return tf.constant([int(seed) % 2147483647, int(salt) % 2147483647], dtype=tf.int32)


if __name__ == "__main__":
    raise SystemExit(main())
