#!/usr/bin/env python3
"""Evaluate a prepared predator--prey Contract E--TP short prefix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import numpy as np
import tensorflow as tf
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.highdim import ledh_contract_e_tp_predator_prey_tf as model
from bayesfilter.highdim.models import p30_predator_prey_fixture_model
from bayesfilter.ledh_fd_policy import evaluate_ledh_fd_policy
from bayesfilter.nonlinear.fixed_sgqf_structural_adapter_tf import (
    tf_predator_prey_to_fixed_sgqf_model,
)
from bayesfilter.nonlinear.fixed_sgqf_tf import (
    TFFixedSGQFBranchConfig,
    TFFixedSGQFNonlinearModel,
    tf_fixed_sgqf_cloud,
    tf_fixed_sgqf_filter,
)
from scripts.filtering_value_gradient_benchmark_generate_p8_datasets import (
    _predator_prey_dataset,
)


DTYPE = tf.float64
PARAMETER_NAMES = ("r", "K", "a", "s", "u", "v")
FD_STEP = 1.0e-5


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--reference-orders", default="9,11")
    parser.add_argument("--reference-prey-bounds", required=True)
    parser.add_argument("--reference-predator-bounds", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _bounds(value: str) -> tuple[float, float]:
    left, right = (float(item) for item in value.split(","))
    if not np.isfinite(left + right) or not left < right:
        raise ValueError("bounds must be finite and increasing")
    return left, right


def _grid(order: int, prey_bounds: tuple[float, float], predator_bounds: tuple[float, float]):
    nodes, weights = leggauss(order)
    axis_nodes = []
    axis_weights = []
    for left, right in (prey_bounds, predator_bounds):
        axis_nodes.append(0.5 * (left + right) + 0.5 * (right - left) * nodes)
        axis_weights.append(0.5 * (right - left) * weights)
    first, second = np.meshgrid(*axis_nodes, indexing="ij")
    first_weight, second_weight = np.meshgrid(*axis_weights, indexing="ij")
    return (
        tf.constant(np.stack([first.ravel(), second.ravel()], axis=1), DTYPE),
        tf.constant((first_weight * second_weight).ravel(), DTYPE),
    )


def _dense_reference(
    theta: tf.Tensor,
    observations: tf.Tensor,
    grid: tf.Tensor,
    weights: tf.Tensor,
) -> tf.Tensor:
    fixture = p30_predator_prey_fixture_model()
    log_weights = tf.math.log(weights)
    log_posterior = None
    total = tf.constant(0.0, DTYPE)
    for time_index, observation in enumerate(tf.unstack(observations, axis=0)):
        if time_index == 0:
            log_predictive = fixture.initial_log_density(theta, grid)
        else:
            transition = model._pairwise_transition(
                fixture, theta, grid, grid, time_index
            )
            log_predictive = tf.reduce_logsumexp(
                log_weights[:, None] + log_posterior[:, None] + transition,
                axis=0,
            )
        log_unnormalized = log_predictive + fixture.observation_log_density(
            theta, grid, observation, t=time_index
        )
        increment = tf.reduce_logsumexp(log_weights + log_unnormalized)
        total += increment
        log_posterior = log_unnormalized - increment
    return total


def _t2_semianalytic_reference(
    theta: tf.Tensor, observations: tf.Tensor, order: int
) -> tf.Tensor:
    fixture = p30_predator_prey_fixture_model()
    nodes, weights = hermgauss(order)
    nodes = tf.constant(np.sqrt(2.0) * nodes, DTYPE)
    weights = tf.constant(weights / np.sqrt(np.pi), DTYPE)
    standard_points, product_weights = model._product_rule(nodes, weights)
    initial_chol = tf.linalg.cholesky(fixture.initial_covariance)
    initial_points = fixture.initial_mean[None, :] + tf.linalg.matmul(
        standard_points, initial_chol, transpose_b=True
    )
    log_terms = fixture.observation_log_density(
        theta, initial_points, observations[0], t=0
    ) + model.one_step_target_continuation_log_likelihood(
        fixture, theta, initial_points, observations[1]
    )
    return tf.reduce_logsumexp(tf.math.log(product_weights) + log_terms)


def _corrected_time_order_sgqf_reference(
    theta: tf.Tensor, observations: tf.Tensor, sparse_level: int
) -> tf.Tensor:
    fixture = p30_predator_prey_fixture_model()
    initial_precision = tf.linalg.inv(fixture.initial_covariance)
    observation_precision = tf.linalg.inv(fixture.observation_covariance)
    covariance = tf.linalg.inv(initial_precision + observation_precision)
    mean = tf.linalg.matvec(
        covariance,
        tf.linalg.matvec(initial_precision, fixture.initial_mean)
        + tf.linalg.matvec(observation_precision, observations[0]),
    )
    initial_increment = model._gaussian_log_density(
        observations[0][None, :] - fixture.initial_mean[None, :],
        fixture.initial_covariance + fixture.observation_covariance,
    )[0]
    adapted = tf_predator_prey_to_fixed_sgqf_model(fixture, theta)
    if not adapted.eligible or adapted.model is None:
        raise RuntimeError("predator fixed-SGQF adapter unavailable")
    corrected_model = TFFixedSGQFNonlinearModel(
        initial_mean=mean,
        initial_covariance=covariance,
        process_covariance=fixture.process_covariance,
        observation_covariance=fixture.observation_covariance,
        transition_fn=adapted.model.transition_fn,
        observation_fn=adapted.model.observation_fn,
        name="corrected_time_order_predator_prey_fixed_sgqf",
    )
    result = tf_fixed_sgqf_filter(
        observations[1:],
        corrected_model,
        cloud=tf_fixed_sgqf_cloud(dim=2, sparse_level=sparse_level),
        branch_config=TFFixedSGQFBranchConfig(
            predictive_epsilon=1.0e-10, innovation_epsilon=1.0e-10
        ),
        return_filtered=False,
    )
    if result.failure is not None or result.log_likelihood is None:
        stage = result.failure.stage if result.failure is not None else "unknown"
        raise RuntimeError(f"corrected-time-order fixed-SGQF failed at {stage}")
    return initial_increment + result.log_likelihood


def _value_and_score(callable_, theta: tf.Tensor):
    with tf.GradientTape() as tape:
        tape.watch(theta)
        value = callable_(theta)
    return value, tape.gradient(value, theta)


def _central_fd(callable_, theta: tf.Tensor) -> list[float]:
    output = []
    for index in range(6):
        step = FD_STEP * max(1.0, abs(float(theta[index].numpy())))
        direction = tf.one_hot(index, 6, dtype=DTYPE)
        output.append(
            float(
                ((callable_(theta + step * direction) - callable_(theta - step * direction)) / (2.0 * step)).numpy()
            )
        )
    return output


def main() -> None:
    args = _parse()
    started = time.perf_counter()
    if args.output.exists():
        raise FileExistsError(args.output)
    path = args.preparation if args.preparation.is_absolute() else ROOT / args.preparation
    preparation = json.loads(path.read_text(encoding="utf-8"))
    time_steps = int(preparation["target"]["time_steps"])
    dataset = _predator_prey_dataset(81104)
    theta = tf.constant(dataset["truth_theta"], DTYPE)
    observations = tf.convert_to_tensor(dataset["observations"][:time_steps], DTYPE)
    nodes = tf.constant(preparation["teacher_quadrature"]["nodes"], DTYPE)
    weights = tf.constant(preparation["teacher_quadrature"]["weights"], DTYPE)
    continuation_grid = tf.constant(preparation["continuation_quadrature"]["points"], DTYPE)
    continuation_weights = tf.constant(preparation["continuation_quadrature"]["weights"], DTYPE)
    active_indices = tf.reshape(tf.constant(preparation["active_indices"], tf.int32), [time_steps - 1, model.FEATURE_COUNT])
    row_scales = tf.reshape(tf.constant(preparation["row_scales"], DTYPE), [time_steps - 1, model.FEATURE_COUNT])
    lookahead_steps = int(preparation["feature_contract"]["lookahead_steps"])

    def candidate(value: tf.Tensor) -> tf.Tensor:
        return model.contract_e_tp_predator_prey_recursive_core(
            value,
            observations,
            nodes,
            weights,
            active_indices,
            row_scales,
            continuation_grid,
            continuation_weights,
            lookahead_steps=lookahead_steps,
        )["objective"]

    candidate_value, candidate_score = _value_and_score(candidate, theta)
    candidate_result = model.contract_e_tp_predator_prey_recursive_core(
        theta,
        observations,
        nodes,
        weights,
        active_indices,
        row_scales,
        continuation_grid,
        continuation_weights,
        lookahead_steps=lookahead_steps,
    )
    finite_difference = _central_fd(candidate, theta)
    fd_policy = evaluate_ledh_fd_policy(
        candidate_score.numpy().tolist(), finite_difference, PARAMETER_NAMES
    )
    reference_orders = tuple(int(item) for item in args.reference_orders.split(","))
    prey_bounds = _bounds(args.reference_prey_bounds)
    predator_bounds = _bounds(args.reference_predator_bounds)
    references = []
    for order in reference_orders:
        if time_steps == 2:
            value, score = _value_and_score(
                lambda parameter: _t2_semianalytic_reference(
                    parameter, observations, order
                ),
                theta,
            )
        else:
            value, score = _value_and_score(
                lambda parameter: _corrected_time_order_sgqf_reference(
                    parameter, observations, order
                ),
                theta,
            )
        references.append({"order": order, "value": float(value.numpy()), "score": score.numpy().tolist()})
    finest = references[-1]
    finest_score = np.asarray(finest["score"])
    candidate_score_array = candidate_score.numpy()
    relative_error = np.abs(candidate_score_array - finest_score) / np.maximum(
        np.maximum(np.abs(candidate_score_array), np.abs(finest_score)), 1.0e-12
    )
    chart_pass = bool(tf.reduce_all(candidate_result["valid_history"]).numpy())
    payload = {
        "schema": "bayesfilter.contract_e_tp.predator_prey_prefix_result.v1",
        "status": "PASS_ENGINEERING" if chart_pass and fd_policy["status"] == "pass" else "FAIL_ENGINEERING",
        "algorithm_id": model.ALGORITHM_ID,
        "row_id": "zhao_cui_predator_prey_T20",
        "scope": "center_only_short_prefix_diagnostic",
        "target": {
            "time_steps": time_steps,
            "theta": theta.numpy().tolist(),
            "parameter_names": list(PARAMETER_NAMES),
            "time_order": preparation["target"]["time_order"],
            "support": preparation["target"]["support"],
        },
        "preparation": {
            "path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "teacher_order": preparation["teacher_quadrature"]["order"],
            "continuation_order": preparation["continuation_quadrature"]["order_per_axis"],
            "lookahead_steps": lookahead_steps,
        },
        "candidate": {
            "value": float(candidate_value.numpy()),
            "score": candidate_score_array.tolist(),
            "same_scalar_finite_difference": finite_difference,
            "same_scalar_fd_policy": fd_policy,
            "increment_history": candidate_result["increment_history"].numpy().tolist(),
        },
        "dense_references": references,
        "reference_refinement": {
            "finest_two_value_difference": references[-1]["value"] - references[-2]["value"],
            "finest_two_score_difference": (np.asarray(references[-1]["score"]) - np.asarray(references[-2]["score"])).tolist(),
            "box": {"prey": list(prey_bounds), "predator": list(predator_bounds)},
            "classification": (
                "semianalytic_initial_gauss_hermite_refinement"
                if time_steps == 2
                else "corrected_time_order_fixed_sgqf_sparse_level_refinement_approximate"
            ),
        },
        "candidate_vs_finest_reference": {
            "value_difference": float(candidate_value.numpy() - finest["value"]),
            "score_difference": (candidate_score_array - finest_score).tolist(),
            "componentwise_relative_error": relative_error.tolist(),
            "sign_reversal": (np.sign(candidate_score_array) != np.sign(finest_score)).tolist(),
            "equivalence_classification": "descriptive_only_margin_unavailable",
        },
        "chart": {
            "valid_history": candidate_result["valid_history"].numpy().tolist(),
            "minimum_weight_history": candidate_result["minimum_weight_history"].numpy().tolist(),
            "condition_number_history": candidate_result["condition_number_history"].numpy().tolist(),
            "maximum_feature_residual_abs": float(tf.reduce_max(tf.abs(candidate_result["feature_residual_history"])).numpy()) if time_steps > 1 else 0.0,
            "chart_pass": chart_pass,
        },
        "decision": {
            "hard_veto_screen_pass": chart_pass and fd_policy["status"] == "pass",
            "candidate_remains_viable": chart_pass and fd_policy["status"] == "pass",
            "statistically_supported_ranking": False,
            "default_readiness": False,
            "next_evidence": "grid-box/order refinement and next declared prefix",
        },
        "execution": {
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "backend": "TensorFlow_float64_CPU_reference_exception",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "wall_time_seconds": time.perf_counter() - started,
            "command": " ".join(sys.argv),
        },
        "nonclaims": [
            "no exact nonlinear filtering claim",
            "no cross-method equivalence margin",
            "not Zhao-Cui production comparator evidence",
            "not HMC or default readiness",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "time_steps": time_steps,
        "candidate_value": payload["candidate"]["value"],
        "reference_value": finest["value"],
        "value_difference": payload["candidate_vs_finest_reference"]["value_difference"],
        "candidate_score": payload["candidate"]["score"],
        "reference_score": finest["score"],
        "relative_score_error": payload["candidate_vs_finest_reference"]["componentwise_relative_error"],
        "fd_status": fd_policy["status"],
        "chart_pass": chart_pass,
    }, indent=2))
    if not payload["decision"]["hard_veto_screen_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
