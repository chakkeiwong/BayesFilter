#!/usr/bin/env python3
"""Prepare center-only predator--prey Contract E--TP charts."""

from __future__ import annotations

import argparse
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
from scipy.optimize import linprog


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.highdim import ledh_contract_e_tp_predator_prey_tf as model
from bayesfilter.highdim import ledh_contract_e_tp_tf as tp
from bayesfilter.highdim.models import p30_predator_prey_fixture_model
from scripts.filtering_value_gradient_benchmark_generate_p8_datasets import (
    _predator_prey_dataset,
)


DTYPE = tf.float64


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-steps", type=int, choices=(1, 2, 5, 20), required=True)
    parser.add_argument("--teacher-order", type=int, choices=(3, 5, 7, 9), required=True)
    parser.add_argument("--continuation-order", type=int, choices=(5, 7, 9, 11), required=True)
    parser.add_argument("--prey-bounds", default="-10,140")
    parser.add_argument("--predator-bounds", default="-20,25")
    parser.add_argument("--lookahead-steps", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _bounds(value: str) -> tuple[float, float]:
    left, right = (float(item) for item in value.split(","))
    if not np.isfinite(left + right) or not left < right:
        raise ValueError("bounds must be finite and increasing")
    return left, right


def _hermite(order: int) -> tuple[tf.Tensor, tf.Tensor]:
    nodes, weights = hermgauss(order)
    return tf.constant(np.sqrt(2.0) * nodes, DTYPE), tf.constant(weights / np.sqrt(np.pi), DTYPE)


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


def main() -> None:
    args = _parse()
    started = time.perf_counter()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.lookahead_steps < 1:
        raise ValueError("lookahead steps must be positive")
    prey_bounds = _bounds(args.prey_bounds)
    predator_bounds = _bounds(args.predator_bounds)
    fixture = p30_predator_prey_fixture_model()
    theta = fixture.true_parameters()
    observations = tf.convert_to_tensor(
        _predator_prey_dataset(81104)["observations"][: args.time_steps], DTYPE
    )
    nodes, weights = _hermite(args.teacher_order)
    continuation_nodes, continuation_axis_weights = _hermite(
        args.continuation_order
    )
    grid, grid_weights = model._product_rule(
        continuation_nodes, continuation_axis_weights
    )
    parents, parent_log_weights, standard_points, standard_log_weights = model.initial_rule(
        fixture, nodes, weights
    )
    active_indices = []
    row_scales = []
    charts = []
    for time_index in range(args.time_steps - 1):
        teacher = model._teacher_step(
            fixture,
            theta,
            parents,
            parent_log_weights,
            standard_points,
            standard_log_weights,
            observations[time_index],
            time_index,
        )
        stop = min(args.time_steps, time_index + 1 + args.lookahead_steps)
        features_tensor = model._features(
            fixture,
            theta,
            teacher["particles"],
            observations[time_index + 1 : stop],
            grid,
            grid_weights,
            first_future_time_index=time_index + 1,
        )
        features = features_tensor.numpy()
        normalized = tf.nn.softmax(teacher["log_unnormalized_weights"]).numpy()
        target = features @ normalized
        scale = np.maximum(1.0e-12, np.maximum(np.max(np.abs(features), axis=1), np.abs(target)))
        scaled = features / scale[:, None]
        scaled_target = target / scale
        rank = int(np.linalg.matrix_rank(scaled))
        if rank != model.FEATURE_COUNT:
            raise RuntimeError(f"time {time_index}: feature rank {rank}")
        solution = linprog(
            np.zeros(features.shape[1]),
            A_eq=scaled,
            b_eq=scaled_target,
            bounds=(0.0, None),
            method="highs",
        )
        indices = np.flatnonzero(solution.x > 1.0e-10) if solution.success else np.array([], int)
        if not solution.success or indices.size != model.FEATURE_COUNT:
            raise RuntimeError(
                f"time {time_index}: no positive square chart; status={solution.message!r}, active={indices.size}"
            )
        projection = tp._contract_e_tp_dense_square_forward_core(
            teacher["particles"],
            teacher["log_unnormalized_weights"],
            features_tensor,
            tf.constant(indices, tf.int32),
            tf.constant(scale, DTYPE),
        )
        parents = projection["student_points"]
        parent_log_weights = tf.math.log(projection["student_weights"])
        active_indices.append(indices.tolist())
        row_scales.append(scale.tolist())
        charts.append(
            {
                "time_index_zero_based": time_index,
                "teacher_count": int(features.shape[1]),
                "feature_rank": rank,
                "minimum_weight": float(projection["minimum_weight"].numpy()),
                "condition_number": float(projection["condition_number"].numpy()),
                "maximum_feature_residual_abs": float(tf.reduce_max(tf.abs(projection["feature_residual"])).numpy()),
                "minimum_teacher_coordinate": tf.reduce_min(teacher["particles"], axis=0).numpy().tolist(),
                "maximum_teacher_coordinate": tf.reduce_max(teacher["particles"], axis=0).numpy().tolist(),
            }
        )
    payload = {
        "schema": "bayesfilter.contract_e_tp.predator_prey_preparation.v1",
        "status": "PASS_CENTER_ONLY_PREPARATION",
        "algorithm_id": model.ALGORITHM_ID,
        "row_id": "zhao_cui_predator_prey_T20",
        "target": {
            "theta": theta.numpy().tolist(),
            "parameter_names": ["r", "K", "a", "s", "u", "v"],
            "time_steps": args.time_steps,
            "time_order": "initial_law_then_y0; transition_then_yt_for_t_positive",
            "support": "real_plane_additive_gaussian_no_clipping",
        },
        "feature_contract": {
            "feature_names": list(model.FEATURE_NAMES),
            "lookahead_steps": args.lookahead_steps,
            "continuation_measure": "target_rk4_gaussian_transition_times_target_gaussian_likelihood",
        },
        "teacher_quadrature": {
            "family": "tensor_gauss_hermite_standard_normal",
            "order": args.teacher_order,
            "nodes": nodes.numpy().tolist(),
            "weights": weights.numpy().tolist(),
        },
        "continuation_quadrature": {
            "family": "tensor_gauss_hermite_standard_normal_gaussian_closure",
            "order_per_axis": args.continuation_order,
            "state_box_used": False,
            "prey_bounds_argument_historical_not_used": list(prey_bounds),
            "predator_bounds_argument_historical_not_used": list(predator_bounds),
            "points": grid.numpy().tolist(),
            "weights": grid_weights.numpy().tolist(),
            "status": "fixed_gaussian_closure_quadrature_requires_order_refinement",
        },
        "active_indices": active_indices,
        "row_scales": row_scales,
        "charts": charts,
        "summary": {
            "chart_count": len(charts),
            "minimum_weight": min((row["minimum_weight"] for row in charts), default=None),
            "maximum_condition_number": max((row["condition_number"] for row in charts), default=None),
            "maximum_feature_residual_abs": max((row["maximum_feature_residual_abs"] for row in charts), default=0.0),
        },
        "execution": {
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "backend": "TensorFlow_float64_CPU_reference_exception_plus_SciPy_chart_selection",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "wall_time_seconds": time.perf_counter() - started,
            "command": " ".join(sys.argv),
        },
        "nonclaims": [
            "not support-box convergence",
            "not parameter-region chart validity",
            "not Zhao-Cui comparator certification",
            "not HMC or default readiness",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": payload["summary"]}, indent=2))


if __name__ == "__main__":
    main()
