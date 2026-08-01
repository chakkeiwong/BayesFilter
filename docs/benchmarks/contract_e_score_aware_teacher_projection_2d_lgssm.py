#!/usr/bin/env python3
"""CPU reference witness for score-aware finite-teacher projection.

This is an independent float64 reference experiment, not a production LEDH
implementation.  NumPy is used only to select and report a fixed active set;
the evaluated finite value and all derivatives use TensorFlow.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import tensorflow as tf


DTYPE = tf.float64
THETA0 = tf.constant([0.72, -1.05, -0.70], dtype=DTYPE)
FD_STEP = 1.0e-5
GAUSS_HERMITE_ORDER = 9
ACTIVE_INDICES = (108, 221, 2317, 2402, 2474, 3942, 4001)
FEATURE_NAMES = (
    "mass",
    "x1",
    "x2",
    "x1_sq",
    "x1_x2",
    "x2_sq",
    "next_predictive_density",
)
OUTPUT_PATH = Path(
    "docs/benchmarks/artifacts/"
    "contract_e_score_aware_teacher_projection_2d_lgssm_2026_07_15.json"
)


def _normal_logpdf(value: tf.Tensor, mean: tf.Tensor, variance: tf.Tensor) -> tf.Tensor:
    return -0.5 * (
        tf.math.log(tf.constant(2.0 * np.pi, dtype=DTYPE) * variance)
        + tf.square(value - mean) / variance
    )


def _model(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    rho, log_q, log_r = tf.unstack(theta)
    transition = tf.stack(
        [
            tf.stack([rho, tf.constant(0.18, DTYPE)]),
            tf.stack([tf.constant(-0.12, DTYPE), tf.constant(0.83, DTYPE)]),
        ]
    )
    q_scale = tf.exp(log_q)
    process_covariance = tf.square(q_scale) * tf.constant(
        [[1.0, 0.22], [0.22, 0.65]], dtype=DTYPE
    )
    observation = tf.constant([1.0, 0.35], dtype=DTYPE)
    observation_variance = tf.square(tf.exp(log_r))
    return transition, process_covariance, observation, observation_variance


def _gauss_hermite_rule(
    mean: tf.Tensor, covariance: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor]:
    nodes_1d, weights_1d = np.polynomial.hermite.hermgauss(GAUSS_HERMITE_ORDER)
    nodes_1d = np.sqrt(2.0) * nodes_1d
    weights_1d = weights_1d / np.sqrt(np.pi)
    standardized = tf.constant(
        [[x1, x2] for x1 in nodes_1d for x2 in nodes_1d], dtype=DTYPE
    )
    weights = tf.constant(
        [w1 * w2 for w1 in weights_1d for w2 in weights_1d], dtype=DTYPE
    )
    chol = tf.linalg.cholesky(covariance)
    points = mean[tf.newaxis, :] + standardized @ tf.transpose(chol)
    return points, weights


def _teacher(theta: tf.Tensor) -> dict[str, tf.Tensor]:
    transition, process_covariance, observation, observation_variance = _model(theta)
    prior_mean = tf.constant([0.25, -0.30], dtype=DTYPE)
    prior_covariance = tf.constant([[0.70, 0.16], [0.16, 0.45]], dtype=DTYPE)
    y_current = tf.constant(0.40, dtype=DTYPE)
    y_next = tf.constant(-0.15, dtype=DTYPE)

    parents, parent_weights = _gauss_hermite_rule(prior_mean, prior_covariance)
    innovations, innovation_weights = _gauss_hermite_rule(
        tf.zeros([2], DTYPE), process_covariance
    )
    candidates = tf.reshape(
        tf.einsum("ab,ib->ia", transition, parents)[:, tf.newaxis, :]
        + innovations[tf.newaxis, :, :],
        [GAUSS_HERMITE_ORDER**4, 2],
    )
    base_weights = tf.reshape(
        parent_weights[:, tf.newaxis] * innovation_weights[tf.newaxis, :], [-1]
    )

    current_means = tf.linalg.matvec(candidates, observation)
    current_log_terms = _normal_logpdf(
        y_current, current_means, observation_variance
    )
    current_log_unnormalized = tf.math.log(base_weights) + current_log_terms
    current_increment = tf.reduce_logsumexp(current_log_unnormalized)
    weights = tf.nn.softmax(current_log_unnormalized)

    next_means = tf.linalg.matvec(
        tf.einsum("ab,ib->ia", transition, candidates), observation
    )
    next_variance = (
        tf.einsum("i,ij,j->", observation, process_covariance, observation)
        + observation_variance
    )
    next_density = tf.exp(_normal_logpdf(y_next, next_means, next_variance))

    x1, x2 = tf.unstack(candidates, axis=1)
    features = tf.stack(
        [
            tf.ones_like(x1),
            x1,
            x2,
            tf.square(x1),
            x1 * x2,
            tf.square(x2),
            next_density,
        ],
        axis=0,
    )
    targets = tf.linalg.matvec(features, weights)
    next_increment = tf.math.log(targets[-1])
    return {
        "parents": parents,
        "parent_weights": parent_weights,
        "innovations": innovations,
        "innovation_weights": innovation_weights,
        "candidates": candidates,
        "weights": weights,
        "features": features,
        "targets": targets,
        "current_increment": current_increment,
        "next_increment": next_increment,
        "total_increment": current_increment + next_increment,
    }


def _fixed_row_scale(center_features: np.ndarray, center_targets: np.ndarray) -> np.ndarray:
    return np.maximum(
        1.0e-8,
        np.maximum(np.max(np.abs(center_features), axis=1), np.abs(center_targets)),
    )


def _select_positive_active_set(
    center_features: np.ndarray,
    center_targets: np.ndarray,
    row_scale: np.ndarray,
) -> tuple[tuple[int, ...], np.ndarray, float]:
    scaled_targets = center_targets / row_scale
    matrix = center_features[:, ACTIVE_INDICES] / row_scale[:, np.newaxis]
    weights = np.linalg.solve(matrix, scaled_targets)
    residual = np.max(np.abs(matrix @ weights - scaled_targets))
    if residual > 1.0e-11 or np.min(weights) <= 1.0e-7:
        raise RuntimeError("the prepared seven-point active set is not positive and feasible")
    return ACTIVE_INDICES, weights, float(np.linalg.cond(matrix))


def _student(
    theta: tf.Tensor,
    active_indices: tuple[int, ...],
    row_scale: tf.Tensor,
) -> dict[str, tf.Tensor]:
    teacher = _teacher(theta)
    active_features = tf.gather(teacher["features"], active_indices, axis=1)
    scaled_matrix = active_features / row_scale[:, tf.newaxis]
    scaled_targets = teacher["targets"] / row_scale
    weights = tf.linalg.solve(scaled_matrix, scaled_targets[:, tf.newaxis])[:, 0]
    matched_targets = tf.linalg.matvec(active_features, weights)
    next_increment = tf.math.log(matched_targets[-1])
    return {
        **teacher,
        "student_weights": weights,
        "student_targets": matched_targets,
        "student_next_increment": next_increment,
        "student_total_increment": teacher["current_increment"] + next_increment,
    }


def _jacobian(function: Any, theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    with tf.GradientTape() as tape:
        tape.watch(theta)
        value = function(theta)
    return value, tape.jacobian(value, theta)


def _central_difference(function: Any, theta: tf.Tensor) -> np.ndarray:
    center = theta.numpy()
    columns = []
    for index in range(center.size):
        plus = center.copy()
        minus = center.copy()
        plus[index] += FD_STEP
        minus[index] -= FD_STEP
        difference = (
            function(tf.constant(plus, DTYPE)).numpy()
            - function(tf.constant(minus, DTYPE)).numpy()
        ) / (2.0 * FD_STEP)
        columns.append(difference)
    return np.stack(columns, axis=-1)


def _exact_kalman(theta: tf.Tensor) -> tf.Tensor:
    transition, process_covariance, observation, observation_variance = _model(theta)
    mean = tf.constant([0.25, -0.30], dtype=DTYPE)
    covariance = tf.constant([[0.70, 0.16], [0.16, 0.45]], dtype=DTYPE)
    increments = []
    for observed in (0.40, -0.15):
        mean = tf.linalg.matvec(transition, mean)
        covariance = transition @ covariance @ tf.transpose(transition) + process_covariance
        predicted_observation = tf.tensordot(observation, mean, axes=1)
        innovation_variance = (
            tf.einsum("i,ij,j->", observation, covariance, observation)
            + observation_variance
        )
        innovation = tf.constant(observed, DTYPE) - predicted_observation
        increments.append(_normal_logpdf(tf.constant(observed, DTYPE), predicted_observation, innovation_variance))
        gain = tf.linalg.matvec(covariance, observation) / innovation_variance
        mean = mean + gain * innovation
        covariance = covariance - tf.tensordot(gain, tf.linalg.matvec(covariance, observation), axes=0)
        covariance = 0.5 * (covariance + tf.transpose(covariance))
    return tf.stack(increments)


def _to_list(value: tf.Tensor | np.ndarray) -> Any:
    array = value.numpy() if isinstance(value, tf.Tensor) else np.asarray(value)
    return array.tolist()


def main() -> None:
    started = time.perf_counter()
    center_teacher = _teacher(THETA0)
    center_features = center_teacher["features"].numpy()
    center_targets = center_teacher["targets"].numpy()
    row_scale_np = _fixed_row_scale(center_features, center_targets)
    active_indices, selected_weights, condition = _select_positive_active_set(
        center_features, center_targets, row_scale_np
    )
    row_scale = tf.constant(row_scale_np, DTYPE)

    teacher_targets, teacher_target_jacobian = _jacobian(
        lambda value: _teacher(value)["targets"], THETA0
    )
    student_targets, student_target_jacobian = _jacobian(
        lambda value: _student(value, active_indices, row_scale)["student_targets"],
        THETA0,
    )
    teacher_total, teacher_total_score = _jacobian(
        lambda value: _teacher(value)["total_increment"], THETA0
    )
    student_total, student_total_score = _jacobian(
        lambda value: _student(value, active_indices, row_scale)["student_total_increment"],
        THETA0,
    )
    teacher_next, teacher_next_score = _jacobian(
        lambda value: _teacher(value)["next_increment"], THETA0
    )
    student_next, student_next_score = _jacobian(
        lambda value: _student(value, active_indices, row_scale)["student_next_increment"],
        THETA0,
    )
    kalman_increments, kalman_increment_scores = _jacobian(_exact_kalman, THETA0)
    evaluated = _student(THETA0, active_indices, row_scale)

    teacher_feature_fd = _central_difference(lambda value: _teacher(value)["targets"], THETA0)
    student_feature_fd = _central_difference(
        lambda value: _student(value, active_indices, row_scale)["student_targets"], THETA0
    )
    teacher_total_fd = _central_difference(
        lambda value: _teacher(value)["total_increment"], THETA0
    )
    student_total_fd = _central_difference(
        lambda value: _student(value, active_indices, row_scale)["student_total_increment"], THETA0
    )

    feature_residual = float(tf.reduce_max(tf.abs(student_targets - teacher_targets)).numpy())
    tangent_residual = float(
        tf.reduce_max(tf.abs(student_target_jacobian - teacher_target_jacobian)).numpy()
    )
    next_value_residual = float(tf.abs(student_next - teacher_next).numpy())
    next_score_residual = float(
        tf.reduce_max(tf.abs(student_next_score - teacher_next_score)).numpy()
    )
    total_value_residual = float(tf.abs(student_total - teacher_total).numpy())
    total_score_residual = float(
        tf.reduce_max(tf.abs(student_total_score - teacher_total_score)).numpy()
    )
    autodiff_fd_residual = float(
        max(
            np.max(np.abs(teacher_target_jacobian.numpy() - teacher_feature_fd)),
            np.max(np.abs(student_target_jacobian.numpy() - student_feature_fd)),
            np.max(np.abs(teacher_total_score.numpy() - teacher_total_fd)),
            np.max(np.abs(student_total_score.numpy() - student_total_fd)),
        )
    )
    minimum_weight = float(tf.reduce_min(evaluated["student_weights"]).numpy())
    passed = bool(
        minimum_weight > 0.0
        and feature_residual <= 1.0e-10
        and tangent_residual <= 1.0e-8
        and next_value_residual <= 1.0e-10
        and next_score_residual <= 1.0e-8
        and total_value_residual <= 1.0e-10
        and total_score_residual <= 1.0e-8
        and autodiff_fd_residual <= 1.0e-7
    )

    payload = {
        "schema": "contract_e_score_aware_teacher_projection_2d_lgssm_v1",
        "status": "PASS" if passed else "FAIL",
        "execution": {
            "backend": "TensorFlow float64 CPU reference",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "jit_compile": False,
            "reason": "small independent reference and finite-difference witness",
            "command": (
                "CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/matplotlib-contract-e-score-aware "
                "python "
                "docs/benchmarks/contract_e_score_aware_teacher_projection_2d_lgssm.py"
            ),
            "git_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "source_worktree_note": (
                "dirty shared research worktree; evidence is bound to the script and JSON paths"
            ),
        },
        "model": {
            "theta_names": ["rho", "log_q_scale", "log_r_scale"],
            "theta": _to_list(THETA0),
            "prior_mean": [0.25, -0.30],
            "prior_covariance": [[0.70, 0.16], [0.16, 0.45]],
            "observations": [0.40, -0.15],
            "observation_vector": [1.0, 0.35],
            "transition_at_theta": _to_list(_model(THETA0)[0]),
            "process_covariance_at_theta": _to_list(_model(THETA0)[1]),
            "observation_variance_at_theta": float(_model(THETA0)[3].numpy()),
        },
        "teacher": {
            "construction": (
                "81-point tensor Gauss-Hermite parent rule by 81-point tensor "
                "Gauss-Hermite innovation rule"
            ),
            "one_dimensional_gauss_hermite_order": GAUSS_HERMITE_ORDER,
            "parent_count": GAUSS_HERMITE_ORDER**2,
            "innovation_count": GAUSS_HERMITE_ORDER**2,
            "candidate_count": GAUSS_HERMITE_ORDER**4,
            "parent_rule_weight_sum": float(tf.reduce_sum(evaluated["parent_weights"]).numpy()),
            "innovation_rule_weight_sum": float(
                tf.reduce_sum(evaluated["innovation_weights"]).numpy()
            ),
            "posterior_weight_sum": float(tf.reduce_sum(evaluated["weights"]).numpy()),
            "feature_names": list(FEATURE_NAMES),
            "feature_targets": _to_list(teacher_targets),
            "feature_target_jacobian": _to_list(teacher_target_jacobian),
        },
        "student": {
            "candidate_count": len(active_indices),
            "active_indices_zero_based": list(active_indices),
            "active_set_preparation": (
                "positive basic feasible set selected once at theta0, then frozen "
                "for the differentiated finite program"
            ),
            "weights": _to_list(evaluated["student_weights"]),
            "points": _to_list(tf.gather(evaluated["candidates"], active_indices)),
            "selection_center_weights_from_numpy": selected_weights.tolist(),
            "minimum_weight": minimum_weight,
            "scaled_feature_matrix_condition_number": condition,
            "fixed_row_scale": row_scale_np.tolist(),
            "feature_targets": _to_list(student_targets),
            "feature_target_jacobian": _to_list(student_target_jacobian),
        },
        "likelihood_and_score": {
            "teacher_current_increment": float(evaluated["current_increment"].numpy()),
            "teacher_next_increment": float(teacher_next.numpy()),
            "student_next_increment": float(student_next.numpy()),
            "teacher_total_two_observation_value": float(teacher_total.numpy()),
            "student_total_two_observation_value": float(student_total.numpy()),
            "teacher_next_score": _to_list(teacher_next_score),
            "student_next_score": _to_list(student_next_score),
            "teacher_total_score": _to_list(teacher_total_score),
            "student_total_score": _to_list(student_total_score),
            "kalman_increment_values": _to_list(kalman_increments),
            "kalman_total_value": float(tf.reduce_sum(kalman_increments).numpy()),
            "kalman_increment_score_jacobian": _to_list(kalman_increment_scores),
            "kalman_total_score": _to_list(tf.reduce_sum(kalman_increment_scores, axis=0)),
        },
        "residuals": {
            "feature_value_max_abs": feature_residual,
            "feature_tangent_max_abs": tangent_residual,
            "next_increment_abs": next_value_residual,
            "next_score_max_abs": next_score_residual,
            "total_value_abs": total_value_residual,
            "total_score_max_abs": total_score_residual,
            "autodiff_central_fd_max_abs": autodiff_fd_residual,
            "fd_step": FD_STEP,
        },
        "decision": {
            "finite_teacher_selected_feature_and_score_witness": "PASS" if passed else "FAIL",
            "hard_vetoes_supported": [] if passed else ["one or more declared numerical gates failed"],
            "what_is_not_concluded": [
                "exactness outside the selected feature span",
                "universal nonlinear or NAWM validity",
                "HMC readiness",
                "GPU/XLA or production readiness",
                "canonical status for contract_e_chol_v1",
            ],
        },
    }
    payload["execution"]["wall_time_seconds"] = time.perf_counter() - started
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
