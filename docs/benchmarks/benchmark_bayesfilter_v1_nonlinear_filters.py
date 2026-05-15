"""NP1 CPU-only benchmark harness for BayesFilter v1 nonlinear filters.

This harness emits explicit NP1 row/manifest schemas for tiny CPU-only smoke
rows. It is benchmark-only and does not modify production filter semantics.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_PRE_PARSER = argparse.ArgumentParser(add_help=False)
_PRE_PARSER.add_argument(
    "--requested-device",
    choices=("cpu",),
    default="cpu",
    help="NP1 worker is CPU-only; GPU is intentionally hidden before TensorFlow import.",
)
_PRE_ARGS, _ = _PRE_PARSER.parse_known_args()
if _PRE_ARGS.requested_device == "cpu":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-bayesfilter")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf  # noqa: E402
import tensorflow_probability as tfp  # noqa: E402

from bayesfilter import (  # noqa: E402
    StatePartition,
    TFStructuralFirstDerivatives,
    affine_structural_to_linear_gaussian_tf,
    make_affine_structural_tf,
)
from bayesfilter.linear.kalman_tf import tf_linear_gaussian_log_likelihood  # noqa: E402
from bayesfilter.nonlinear.cut_tf import tf_cut4g_sigma_point_rule  # noqa: E402
from bayesfilter.testing import (  # noqa: E402
    dense_projection_first_step,
    make_affine_gaussian_structural_oracle_tf,
    make_nonlinear_accumulation_first_derivatives_tf,
    make_nonlinear_accumulation_model_tf,
    make_univariate_nonlinear_growth_first_derivatives_tf,
    make_univariate_nonlinear_growth_model_tf,
    model_a_observations_tf,
    model_b_observations_tf,
    model_c_observations_tf,
    nonlinear_sigma_point_diagnostic_snapshot,
    nonlinear_sigma_point_score_branch_summary,
    nonlinear_sigma_point_value_branch_summary,
    sigma_point_projection_first_step,
    tf_nonlinear_sigma_point_score,
    tf_nonlinear_sigma_point_value_filter,
)

BACKENDS = ("tf_svd_cubature", "tf_svd_ukf", "tf_svd_cut4")
VALUE_PATH = "value"
SCORE_PATH = "score"
ALLOWED_ROW_ROLES = {"value_timing", "score_timing", "branch_precheck", "skipped"}
NON_IMPLICATION_TEXT = (
    "NP1 CPU-only smoke rows do not certify broad speedups, default backend policy, "
    "GPU/XLA support, exact nonlinear likelihood quality for Models B-C, or HMC/Hessian readiness."
)


def _max_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _safe_float(value: Any) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _safe_tensor_float(value: Any) -> float | None:
    return _safe_float(tf.convert_to_tensor(value, dtype=tf.float64).numpy())


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        status = subprocess.check_output(
            ["git", "status", "--short"],
            cwd=ROOT,
            text=True,
        )
        return bool(status.strip())
    except Exception:
        return True


def _logical_devices() -> list[dict[str, str]]:
    return [
        {"name": device.name, "device_type": device.device_type}
        for device in tf.config.list_logical_devices()
    ]


def _model_a_builder(params: tf.Tensor):
    phi = params[0]
    sigma = params[1]
    obs_scale = params[2]
    partition = StatePartition(
        state_names=("m", "lag_m"),
        stochastic_indices=(0,),
        deterministic_indices=(1,),
        innovation_dim=1,
    )
    return make_affine_structural_tf(
        partition=partition,
        initial_mean=tf.zeros([2], dtype=tf.float64),
        initial_covariance=tf.linalg.diag(tf.constant([1.2, 0.7], dtype=tf.float64)),
        transition_offset=tf.zeros([2], dtype=tf.float64),
        transition_matrix=tf.stack(
            [
                tf.stack([phi, tf.constant(-0.10, dtype=tf.float64)]),
                tf.constant([1.0, 0.0], dtype=tf.float64),
            ]
        ),
        innovation_matrix=tf.reshape(
            tf.stack([sigma, tf.constant(0.0, dtype=tf.float64)]),
            [2, 1],
        ),
        innovation_covariance=tf.constant([[1.0]], dtype=tf.float64),
        observation_offset=tf.zeros([1], dtype=tf.float64),
        observation_matrix=tf.reshape(tf.stack([obs_scale, 0.0]), [1, 2]),
        observation_covariance=tf.constant([[0.15**2]], dtype=tf.float64),
    )


def _model_a_derivative_builder(params: tf.Tensor):
    phi = params[0]
    sigma = params[1]
    obs_scale = params[2]
    transition_matrix = tf.stack(
        [
            tf.stack([phi, tf.constant(-0.10, dtype=tf.float64)]),
            tf.constant([1.0, 0.0], dtype=tf.float64),
        ]
    )
    innovation_matrix = tf.reshape(
        tf.stack([sigma, tf.constant(0.0, dtype=tf.float64)]),
        [2, 1],
    )
    observation_matrix = tf.reshape(tf.stack([obs_scale, 0.0]), [1, 2])
    p = 3
    state_dim = 2
    innovation_dim = 1
    observation_dim = 1
    d_transition_matrix = tf.zeros([p, state_dim, state_dim], dtype=tf.float64)
    d_transition_matrix = tf.tensor_scatter_nd_update(
        d_transition_matrix,
        [[0, 0, 0]],
        [tf.constant(1.0, dtype=tf.float64)],
    )
    d_innovation_matrix = tf.zeros([p, state_dim, innovation_dim], dtype=tf.float64)
    d_innovation_matrix = tf.tensor_scatter_nd_update(
        d_innovation_matrix,
        [[1, 0, 0]],
        [tf.constant(1.0, dtype=tf.float64)],
    )
    d_observation_matrix = tf.zeros([p, observation_dim, state_dim], dtype=tf.float64)
    d_observation_matrix = tf.tensor_scatter_nd_update(
        d_observation_matrix,
        [[2, 0, 0]],
        [tf.constant(1.0, dtype=tf.float64)],
    )

    def transition_state_jacobian(previous: tf.Tensor, innovation: tf.Tensor) -> tf.Tensor:
        del innovation
        point_count = tf.shape(previous)[0]
        return tf.broadcast_to(transition_matrix[tf.newaxis, :, :], [point_count, 2, 2])

    def transition_innovation_jacobian(
        previous: tf.Tensor,
        innovation: tf.Tensor,
    ) -> tf.Tensor:
        del previous
        point_count = tf.shape(innovation)[0]
        return tf.broadcast_to(innovation_matrix[tf.newaxis, :, :], [point_count, 2, 1])

    def d_transition(previous: tf.Tensor, innovation: tf.Tensor) -> tf.Tensor:
        return (
            tf.einsum("pij,rj->pri", d_transition_matrix, previous)
            + tf.einsum("piq,rq->pri", d_innovation_matrix, innovation)
        )

    def observation_state_jacobian(states: tf.Tensor) -> tf.Tensor:
        point_count = tf.shape(states)[0]
        return tf.broadcast_to(observation_matrix[tf.newaxis, :, :], [point_count, 1, 2])

    def d_observation(states: tf.Tensor) -> tf.Tensor:
        return tf.einsum("pmj,rj->prm", d_observation_matrix, states)

    return TFStructuralFirstDerivatives(
        d_initial_mean=tf.zeros([p, state_dim], dtype=tf.float64),
        d_initial_covariance=tf.zeros([p, state_dim, state_dim], dtype=tf.float64),
        d_innovation_covariance=tf.zeros([p, innovation_dim, innovation_dim], dtype=tf.float64),
        d_observation_covariance=tf.zeros([p, observation_dim, observation_dim], dtype=tf.float64),
        transition_state_jacobian_fn=transition_state_jacobian,
        transition_innovation_jacobian_fn=transition_innovation_jacobian,
        d_transition_fn=d_transition,
        observation_state_jacobian_fn=observation_state_jacobian,
        d_observation_fn=d_observation,
        name="model_a_affine_benchmark_first_derivatives",
    )


def _model_b_builder(params: tf.Tensor):
    return make_nonlinear_accumulation_model_tf(
        rho=params[0],
        sigma=params[1],
        beta=params[2],
    )


def _model_b_derivative_builder(params: tf.Tensor):
    return make_nonlinear_accumulation_first_derivatives_tf(
        rho=params[0],
        sigma=params[1],
        beta=params[2],
    )


def _model_c_builder(params: tf.Tensor):
    return make_univariate_nonlinear_growth_model_tf(
        process_sigma=params[0],
        observation_sigma=params[1],
        initial_variance=params[2],
    )


def _model_c_derivative_builder(params: tf.Tensor):
    return make_univariate_nonlinear_growth_first_derivatives_tf(
        process_sigma=params[0],
        observation_sigma=params[1],
    )


def _model_cases() -> tuple[dict[str, Any], ...]:
    return (
        {
            "name": "model_a_affine_gaussian_structural_oracle",
            "model": make_affine_gaussian_structural_oracle_tf(),
            "observations": model_a_observations_tf()[:2],
            "reference_kind": "exact_linear_gaussian_kalman",
            "branch_grid": tf.constant(
                [[0.32, 0.22, 1.0], [0.35, 0.25, 1.0], [0.38, 0.28, 1.0]],
                dtype=tf.float64,
            ),
            "builder": _model_a_builder,
            "derivative_builder": _model_a_derivative_builder,
            "score_allow_fixed_null_support": False,
            "parameter_dim": 3,
            "parity_tolerance": 1e-10,
        },
        {
            "name": "model_b_nonlinear_accumulation",
            "model": make_nonlinear_accumulation_model_tf(),
            "observations": model_b_observations_tf()[:2],
            "reference_kind": "dense_one_step_projection_only",
            "branch_grid": tf.constant(
                [[0.66, 0.23, 0.75], [0.70, 0.25, 0.80], [0.74, 0.27, 0.85]],
                dtype=tf.float64,
            ),
            "builder": _model_b_builder,
            "derivative_builder": _model_b_derivative_builder,
            "score_allow_fixed_null_support": False,
            "parameter_dim": 3,
            "parity_tolerance": 1e-7,
        },
        {
            "name": "model_c_autonomous_nonlinear_growth",
            "model": make_univariate_nonlinear_growth_model_tf(),
            "observations": model_c_observations_tf()[:2],
            "reference_kind": "dense_one_step_projection_only",
            "branch_grid": tf.constant(
                [[0.90, 1.00, 0.20], [1.00, 1.00, 0.20], [1.10, 1.10, 0.25]],
                dtype=tf.float64,
            ),
            "builder": _model_c_builder,
            "derivative_builder": _model_c_derivative_builder,
            "score_allow_fixed_null_support": True,
            "parameter_dim": 2,
            "parity_tolerance": 1e-7,
        },
    )


def _exact_reference(case: dict[str, Any]):
    if case["reference_kind"] != "exact_linear_gaussian_kalman":
        return None
    linear = affine_structural_to_linear_gaussian_tf(case["model"])
    return tf_linear_gaussian_log_likelihood(
        case["observations"],
        linear,
        backend="tf_cholesky",
        jitter=tf.constant(0.0, dtype=tf.float64),
        return_filtered=True,
    )


def _first_step_projection_errors(case: dict[str, Any], backend: str) -> dict[str, float | None]:
    model = case["model"]
    observations = case["observations"]
    dim = model.partition.state_dim + model.partition.innovation_dim
    dense = dense_projection_first_step(model, observations[0], nodes_per_dim=9)
    if backend == "tf_svd_cut4":
        sigma_rule = tf_cut4g_sigma_point_rule(dim)
    else:
        from bayesfilter.nonlinear.sigma_points_tf import tf_unit_sigma_point_rule as _unit_rule

        sigma_rule = _unit_rule(dim, rule="cubature" if backend == "tf_svd_cubature" else "unscented")
    sigma = sigma_point_projection_first_step(
        model,
        observations[0],
        sigma_rule=sigma_rule,
    )
    return {
        "first_step_abs_log_likelihood_error": abs(
            float(sigma.log_likelihood.numpy()) - float(dense.log_likelihood.numpy())
        ),
        "first_step_filtered_mean_l2_error": _safe_tensor_float(
            tf.linalg.norm(sigma.filtered_mean - dense.filtered_mean)
        ),
        "first_step_filtered_covariance_fro_error": _safe_tensor_float(
            tf.linalg.norm(sigma.filtered_covariance - dense.filtered_covariance)
        ),
    }


def _exact_filtered_errors(result, reference) -> tuple[float | None, float | None]:
    if reference is None or result.filtered_means is None or result.filtered_covariances is None:
        return None, None
    mean_errors = tf.linalg.norm(result.filtered_means - reference.filtered_means, axis=1)
    cov_errors = tf.linalg.norm(
        result.filtered_covariances - reference.filtered_covariances,
        axis=[1, 2],
    )
    return _safe_tensor_float(tf.reduce_max(mean_errors)), _safe_tensor_float(tf.reduce_max(cov_errors))


def _time_call(fn, repeats: int) -> tuple[Any, float, float]:
    start = time.perf_counter()
    result = fn()
    first = time.perf_counter() - start
    steady_times = []
    steady_result = result
    for _ in range(repeats):
        start = time.perf_counter()
        steady_result = fn()
        steady_times.append(time.perf_counter() - start)
    mean_steady = sum(steady_times) / len(steady_times) if steady_times else first
    return steady_result, first, mean_steady


def _base_row(
    *,
    row_id: str,
    row_role: str,
    case: dict[str, Any],
    backend: str,
    path: str,
    point_count: int,
    polynomial_degree: int,
    return_filtered: bool,
    branch: str,
    parity: str,
    command: str,
    environment_id: str,
    artifact_path: str,
) -> dict[str, Any]:
    model = case["model"]
    observations = case["observations"]
    actual_device = "cpu" if tf.config.list_logical_devices("CPU") else "unavailable"
    row = {
        "row_id": row_id,
        "row_role": row_role,
        "model": case["name"],
        "backend": backend,
        "path": path,
        "dtype": "tf.float64",
        "T": int(observations.shape[0]),
        "timesteps": int(observations.shape[0]),
        "state_dim": int(model.partition.state_dim),
        "innovation_dim": int(model.partition.innovation_dim),
        "observation_dim": int(model.observation_dim),
        "dims": {
            "state": int(model.partition.state_dim),
            "innovation": int(model.partition.innovation_dim),
            "observation": int(model.observation_dim),
        },
        "parameter_dim": int(case["parameter_dim"]),
        "point_count": int(point_count),
        "polynomial_degree": int(polynomial_degree),
        "return_filtered": bool(return_filtered),
        "mode": "eager",
        "requested_device": "cpu",
        "actual_device": actual_device,
        "device_trust_label": "cpu_hidden_gpu",
        "gpu_intentionally_hidden": True,
        "compile_warmup_policy": "first_call_then_mean_of_repeats_no_tf_function_compile",
        "first_call": None,
        "steady": None,
        "memory": None,
        "branch": branch,
        "parity": parity,
        "tolerance": case["parity_tolerance"],
        "parity_tolerance": case["parity_tolerance"],
        "command": command,
        "environment_id": environment_id,
        "artifact_path": artifact_path,
        "skip_category": None,
        "skip_reason": None,
        "non_implication_text": NON_IMPLICATION_TEXT,
        "branch_precheck_id": None,
        "branch_precheck_status": None,
        "reference_kind": case["reference_kind"],
    }
    if row_role not in ALLOWED_ROW_ROLES:
        raise ValueError(f"invalid row_role: {row_role}")
    return row


def _branch_row(
    *,
    row_id: str,
    case: dict[str, Any],
    backend: str,
    command: str,
    environment_id: str,
    artifact_path: str,
) -> dict[str, Any]:
    summary = nonlinear_sigma_point_score_branch_summary(
        case["observations"],
        case["branch_grid"],
        case["builder"],
        case["derivative_builder"],
        backend=backend,
        allow_fixed_null_support=case["score_allow_fixed_null_support"],
    )
    point_count = int(summary.max_point_count)
    row = _base_row(
        row_id=row_id,
        row_role="branch_precheck",
        case=case,
        backend=backend,
        path=SCORE_PATH,
        point_count=point_count,
        polynomial_degree=0,
        return_filtered=False,
        branch="precheck_only",
        parity="not_applicable",
        command=command,
        environment_id=environment_id,
        artifact_path=artifact_path,
    )
    row.update(
        {
            "branch_precheck_status": "pass" if summary.ok_count == summary.total_count else "blocked",
            "score_branch_label": "structural_fixed_support_no_active_floor"
            if case["score_allow_fixed_null_support"]
            else "smooth_simple_spectrum_no_active_floor",
            "score_branch_ok_count": int(summary.ok_count),
            "score_branch_total_count": int(summary.total_count),
            "score_branch_ok_fraction": _safe_float(summary.ok_fraction),
            "score_branch_active_floor_count": int(summary.active_floor_count),
            "score_branch_weak_spectral_gap_count": int(summary.weak_spectral_gap_count),
            "score_branch_nonfinite_count": int(summary.nonfinite_count),
            "score_branch_failure_labels": list(summary.failure_labels),
            "score_branch_structural_null_count": int(summary.max_structural_null_count),
            "score_branch_structural_null_covariance_residual": _safe_float(
                summary.max_structural_null_covariance_residual
            ),
            "score_branch_fixed_null_derivative_residual": _safe_float(
                summary.max_fixed_null_derivative_residual
            ),
            "value_branch_ok_count": None,
            "value_branch_total_count": None,
        }
    )
    return row


def _skipped_cut4_row(
    *,
    row_id: str,
    case: dict[str, Any],
    command: str,
    environment_id: str,
    artifact_path: str,
) -> dict[str, Any]:
    augmented_dim = int(case["model"].partition.state_dim + case["model"].partition.innovation_dim)
    point_count = int(2 * augmented_dim + 2**augmented_dim)
    row = _base_row(
        row_id=row_id,
        row_role="skipped",
        case=case,
        backend="tf_svd_cut4",
        path=VALUE_PATH,
        point_count=point_count,
        polynomial_degree=4,
        return_filtered=False,
        branch="not_run",
        parity="not_run",
        command=command,
        environment_id=environment_id,
        artifact_path=artifact_path,
    )
    row.update(
        {
            "skip_category": "cut4_point_cap",
            "skip_reason": f"CUT4 point_count={point_count} exceeds NP1 default cap 512 for augmented_dim={augmented_dim}.",
            "branch_precheck_status": "not_applicable",
        }
    )
    return row


def _value_row(
    *,
    row_id: str,
    case: dict[str, Any],
    backend: str,
    return_filtered: bool,
    repeats: int,
    command: str,
    environment_id: str,
    artifact_path: str,
) -> dict[str, Any]:
    rss_before = _max_rss_mb()
    result, first_seconds, steady_seconds = _time_call(
        lambda: tf_nonlinear_sigma_point_value_filter(
            case["observations"],
            case["model"],
            backend=backend,
            return_filtered=return_filtered,
        ),
        repeats,
    )
    rss_after = _max_rss_mb()
    snapshot = nonlinear_sigma_point_diagnostic_snapshot(result, mode="value")
    branch = nonlinear_sigma_point_value_branch_summary(
        case["observations"],
        case["branch_grid"],
        case["builder"],
        backend=backend,
    )
    reference = _exact_reference(case)
    if reference is None:
        reference_log_likelihood = None
        abs_error = None
    else:
        reference_log_likelihood = float(reference.log_likelihood.numpy())
        abs_error = abs(float(result.log_likelihood.numpy()) - reference_log_likelihood)
    first_step_errors = _first_step_projection_errors(case, backend)
    exact_mean_error, exact_cov_error = _exact_filtered_errors(result, reference)
    row = _base_row(
        row_id=row_id,
        row_role="value_timing",
        case=case,
        backend=backend,
        path=VALUE_PATH,
        point_count=int(snapshot.point_count),
        polynomial_degree=int(snapshot.polynomial_degree),
        return_filtered=return_filtered,
        branch="finite_implemented_filter",
        parity="exact_linear_gaussian_reference" if reference is not None else "dense_one_step_projection_only",
        command=command,
        environment_id=environment_id,
        artifact_path=artifact_path,
    )
    row.update(
        {
            "max_integration_rank": int(snapshot.max_integration_rank),
            "first_call": {"seconds": _safe_float(first_seconds)},
            "steady": {"mean_seconds": _safe_float(steady_seconds), "repeats": repeats},
            "memory": {
                "max_rss_before_mb": _safe_float(rss_before),
                "max_rss_after_mb": _safe_float(rss_after),
                "max_rss_delta_mb": _safe_float(rss_after - rss_before),
            },
            "parity_status": "measured" if row["parity"] != "not_run" else "not_run",
            "value_branch_ok_count": int(branch.ok_count),
            "value_branch_total_count": int(branch.total_count),
            "value_branch_ok_fraction": _safe_float(branch.ok_fraction),
            "value_branch_active_floor_count": int(branch.active_floor_count),
            "value_branch_weak_spectral_gap_count": int(branch.weak_spectral_gap_count),
            "value_branch_nonfinite_count": int(branch.nonfinite_count),
            "value_branch_failure_labels": list(branch.failure_labels),
            "value_branch_structural_null_count": int(branch.max_structural_null_count),
            "value_branch_structural_null_covariance_residual": _safe_float(
                branch.max_structural_null_covariance_residual
            ),
            "value_branch_fixed_null_derivative_residual": _safe_float(
                branch.max_fixed_null_derivative_residual
            ),
            "score_branch_ok_count": None,
            "score_branch_total_count": None,
            "log_likelihood": _safe_float(result.log_likelihood.numpy()),
            "reference_log_likelihood": _safe_float(reference_log_likelihood) if reference_log_likelihood is not None else None,
            "abs_log_likelihood_error": _safe_float(abs_error) if abs_error is not None else None,
            "first_step_reference_kind": "dense_one_step_gaussian_projection",
            "first_step_abs_log_likelihood_error": first_step_errors["first_step_abs_log_likelihood_error"],
            "first_step_filtered_mean_l2_error": first_step_errors["first_step_filtered_mean_l2_error"],
            "first_step_filtered_covariance_fro_error": first_step_errors[
                "first_step_filtered_covariance_fro_error"
            ],
            "exact_filtered_mean_max_l2_error": exact_mean_error,
            "exact_filtered_covariance_max_fro_error": exact_cov_error,
        }
    )
    return row


def _score_row(
    *,
    row_id: str,
    case: dict[str, Any],
    backend: str,
    repeats: int,
    branch_precheck_row: dict[str, Any],
    command: str,
    environment_id: str,
    artifact_path: str,
) -> dict[str, Any]:
    rss_before = _max_rss_mb()
    score_params = case["branch_grid"][1]
    score_model = case["builder"](score_params)
    score_derivatives = case["derivative_builder"](score_params)
    try:
        result, first_seconds, steady_seconds = _time_call(
            lambda: tf_nonlinear_sigma_point_score(
                case["observations"],
                score_model,
                score_derivatives,
                backend=backend,
                allow_fixed_null_support=case["score_allow_fixed_null_support"],
            ),
            repeats,
        )
        rss_after = _max_rss_mb()
        snapshot = nonlinear_sigma_point_diagnostic_snapshot(result, mode="score")
        row = _base_row(
            row_id=row_id,
            row_role="score_timing",
            case=case,
            backend=backend,
            path=SCORE_PATH,
            point_count=int(snapshot.point_count),
            polynomial_degree=int(snapshot.polynomial_degree),
            return_filtered=False,
            branch=str(snapshot.derivative_branch),
            parity="legacy_score_impl_reference",
            command=command,
            environment_id=environment_id,
            artifact_path=artifact_path,
        )
        row.update(
            {
                "branch_precheck_id": branch_precheck_row["row_id"],
                "branch_precheck_status": branch_precheck_row["branch_precheck_status"],
                "max_integration_rank": int(snapshot.max_integration_rank),
                "first_call": {"seconds": _safe_float(first_seconds)},
                "steady": {"mean_seconds": _safe_float(steady_seconds), "repeats": repeats},
                "memory": {
                    "max_rss_before_mb": _safe_float(rss_before),
                    "max_rss_after_mb": _safe_float(rss_after),
                    "max_rss_delta_mb": _safe_float(rss_after - rss_before),
                },
                "parity_status": "measured_against_branch_precheck_only",
                "score_value": _safe_float(result.score.numpy()[0]),
                "derivative_method": str(snapshot.derivative_method),
                "derivative_branch": str(snapshot.derivative_branch),
                "structural_null_count": int(snapshot.structural_null_count),
                "structural_null_covariance_residual": _safe_float(
                    snapshot.structural_null_covariance_residual
                ),
                "fixed_null_derivative_residual": _safe_float(snapshot.fixed_null_derivative_residual),
            }
        )
        return row
    except Exception as exc:
        rss_after = _max_rss_mb()
        row = _base_row(
            row_id=row_id,
            row_role="skipped",
            case=case,
            backend=backend,
            path=SCORE_PATH,
            point_count=int(branch_precheck_row["point_count"]),
            polynomial_degree=0,
            return_filtered=False,
            branch="blocked_during_score_timing",
            parity="not_run",
            command=command,
            environment_id=environment_id,
            artifact_path=artifact_path,
        )
        row.update(
            {
                "branch_precheck_id": branch_precheck_row["row_id"],
                "branch_precheck_status": branch_precheck_row["branch_precheck_status"],
                "skip_category": "score_execution_blocked",
                "skip_reason": repr(exc),
                "memory": {
                    "max_rss_before_mb": _safe_float(rss_before),
                    "max_rss_after_mb": _safe_float(rss_after),
                    "max_rss_delta_mb": _safe_float(rss_after - rss_before),
                },
                "parity_status": "not_run",
            }
        )
        return row


def _manifest(
    *,
    command: str,
    output_json: Path,
    output_md: Path,
    plan_path: str,
    result_path: str,
) -> tuple[str, dict[str, Any]]:
    environment_id = "env-001"
    manifest = {
        "environment_id": environment_id,
        "command": command,
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "tensorflow_probability_version": tfp.__version__,
        "platform": platform.platform(),
        "cpu_gpu_visibility_policy": "cpu_only_hidden_gpu_before_tensorflow_import",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_intentionally_hidden": True,
        "requested_device": "cpu",
        "actual_visible_logical_devices": _logical_devices(),
        "artifact_paths": [str(output_json), str(output_md)],
        "governing_plan_path": plan_path,
        "governing_result_path": result_path,
    }
    return environment_id, manifest


def _markdown(payload: dict[str, Any], json_path: Path) -> str:
    lines = [
        "# BayesFilter V1 Nonlinear Performance NP1 CPU-only Smoke Artifact",
        "",
        f"Authoritative JSON artifact: `{json_path}`.",
        "",
        "## Manifest",
        "",
        f"- Command: `{payload['manifest']['command']}`",
        f"- Git commit: `{payload['manifest']['git_commit']}`",
        f"- Dirty worktree: `{payload['manifest']['git_dirty']}`",
        f"- Python / TF / TFP: `{payload['manifest']['python_version']}` / `{payload['manifest']['tensorflow_version']}` / `{payload['manifest']['tensorflow_probability_version']}`",
        f"- Visibility policy: `{payload['manifest']['cpu_gpu_visibility_policy']}` with `CUDA_VISIBLE_DEVICES={payload['manifest']['cuda_visible_devices']}`",
        "",
        "## Rows",
        "",
        "| Row ID | Role | Model | Backend | Path | T | Points | Branch precheck | First s | Steady s | Skip |",
        "| --- | --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for row in payload["rows"]:
        first_seconds = "n/a" if row["first_call"] is None else f"{row['first_call']['seconds']:.6f}"
        steady_seconds = "n/a" if row["steady"] is None else f"{row['steady']['mean_seconds']:.6f}"
        lines.append(
            "| {row_id} | {row_role} | {model} | {backend} | {path} | {T} | {point_count} | {branch_precheck} | {first} | {steady} | {skip} |".format(
                row_id=row["row_id"],
                row_role=row["row_role"],
                model=row["model"],
                backend=row["backend"],
                path=row["path"],
                T=row["T"],
                point_count=row["point_count"],
                branch_precheck=row.get("branch_precheck_status") or "n/a",
                first=first_seconds,
                steady=steady_seconds,
                skip=row.get("skip_category") or "",
            )
        )
    lines.extend(
        [
            "",
            "## Scope boundary",
            "",
            NON_IMPLICATION_TEXT,
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(parents=[_PRE_PARSER])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs/benchmarks/bayesfilter-v1-nonlinear-performance-np1-smoke-2026-05-16.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=ROOT / "docs/benchmarks/bayesfilter-v1-nonlinear-performance-np1-smoke-2026-05-16.md",
    )
    parser.add_argument(
        "--plan-path",
        type=str,
        default="docs/plans/bayesfilter-v1-nonlinear-performance-np1-benchmark-harness-plan-2026-05-15.md",
    )
    parser.add_argument(
        "--result-path",
        type=str,
        default="docs/plans/bayesfilter-v1-nonlinear-performance-np1-benchmark-harness-result-2026-05-16.md",
    )
    args = parser.parse_args()

    command = " ".join(sys.argv)
    environment_id, manifest = _manifest(
        command=command,
        output_json=args.output,
        output_md=args.markdown_output,
        plan_path=args.plan_path,
        result_path=args.result_path,
    )
    artifact_path = str(args.output)
    rows: list[dict[str, Any]] = []

    case_a = _model_cases()[0]
    for backend in BACKENDS:
        branch_row = _branch_row(
            row_id=f"branch-precheck-{backend}-model-a-tiny",
            case=case_a,
            backend=backend,
            command=command,
            environment_id=environment_id,
            artifact_path=artifact_path,
        )
        rows.append(branch_row)
        rows.append(
            _value_row(
                row_id=f"value-timing-{backend}-model-a-tiny-return-filtered-false",
                case=case_a,
                backend=backend,
                return_filtered=False,
                repeats=args.repeats,
                command=command,
                environment_id=environment_id,
                artifact_path=artifact_path,
            )
        )
        rows.append(
            _value_row(
                row_id=f"value-timing-{backend}-model-a-tiny-return-filtered-true",
                case=case_a,
                backend=backend,
                return_filtered=True,
                repeats=args.repeats,
                command=command,
                environment_id=environment_id,
                artifact_path=artifact_path,
            )
        )
        if branch_row["branch_precheck_status"] == "pass":
            rows.append(
                _score_row(
                    row_id=f"score-timing-{backend}-model-a-tiny",
                    case=case_a,
                    backend=backend,
                    repeats=args.repeats,
                    branch_precheck_row=branch_row,
                    command=command,
                    environment_id=environment_id,
                    artifact_path=artifact_path,
                )
            )

    rows.append(
        _skipped_cut4_row(
            row_id="skipped-cut4-point-cap-synthetic",
            case={
                **case_a,
                "model": _model_a_builder(tf.constant([0.35, 0.25, 1.0], dtype=tf.float64)),
            },
            command=command,
            environment_id=environment_id,
            artifact_path=artifact_path,
        )
    )
    rows[-1]["dims"] = {"state": 6, "innovation": 4, "observation": 1}
    rows[-1]["state_dim"] = 6
    rows[-1]["innovation_dim"] = 4
    rows[-1]["observation_dim"] = 1
    rows[-1]["parameter_dim"] = 3
    rows[-1]["T"] = 2
    rows[-1]["timesteps"] = 2
    rows[-1]["point_count"] = 1044
    rows[-1]["skip_reason"] = "CUT4 point_count=1044 exceeds NP1 default cap 512 for augmented_dim=10."

    payload = _json_safe(
        {
            "benchmark": "bayesfilter_v1_nonlinear_performance_np1_cpu_smoke",
            "claim_scope": "np1_cpu_only_schema_smoke",
            "manifest": manifest,
            "rows": rows,
            "blocked_claims": [
                "broad_gpu_speedup",
                "default_backend_policy",
                "exact_model_b_c_nonlinear_likelihood",
                "nonlinear_hmc_or_hessian_readiness",
            ],
        }
    )
    args.output.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.markdown_output.write_text(_markdown(payload, args.output) + "\n", encoding="utf-8")
    print(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
