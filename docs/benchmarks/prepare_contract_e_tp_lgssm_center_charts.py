#!/usr/bin/env python3
"""Prepare deterministic center-only Contract E--TP LGSSM charts."""

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

import numpy as np
import tensorflow as tf
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import linprog


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.highdim import ledh_contract_e_tp_lgssm_tf as model
from bayesfilter.highdim import ledh_contract_e_tp_tf as tp
from scripts.filtering_value_gradient_benchmark_generate_p8_datasets import (
    _lgssm_dataset,
)


DTYPE = tf.float64
DATASET_SEED = 81100
THETA = tf.constant([0.72, 0.55, 0.35, 0.35, 0.45], DTYPE)
def _tensor_sha256(value: tf.Tensor) -> str:
    return hashlib.sha256(tf.io.serialize_tensor(value).numpy()).hexdigest()


def _rule(order: int) -> tuple[tf.Tensor, tf.Tensor]:
    nodes, weights = hermgauss(order)
    return (
        tf.constant(np.sqrt(2.0) * nodes, DTYPE),
        tf.constant(weights / np.sqrt(np.pi), DTYPE),
    )


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", type=int, choices=(3, 5, 7), required=True)
    parser.add_argument("--time-steps", type=int, choices=(2, 5, 50), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse()
    started = time.perf_counter()
    if args.output.exists():
        raise FileExistsError(args.output)
    observations = tf.convert_to_tensor(
        _lgssm_dataset(DATASET_SEED)["observations"][: args.time_steps], DTYPE
    )
    nodes, weights = _rule(args.order)
    parents, parent_log_weights, innovations, innovation_log_weights = (
        model.initial_parents(THETA, nodes, weights)
    )
    active_indices = []
    row_scales = []
    chart_records = []
    for time_index in range(args.time_steps - 1):
        teacher = model._teacher_step(
            THETA,
            parents,
            parent_log_weights,
            innovations,
            innovation_log_weights,
            observations[time_index],
            observations[time_index + 1],
        )
        features = teacher["features"].numpy()
        log_weights = teacher["log_unnormalized_weights"].numpy()
        normalized = np.exp(log_weights - np.logaddexp.reduce(log_weights))
        target = features @ normalized
        scale = np.maximum(
            1.0e-8,
            np.maximum(np.max(np.abs(features), axis=1), np.abs(target)),
        )
        scaled_features = features / scale[:, None]
        scaled_target = target / scale
        solution = linprog(
            np.zeros(features.shape[1]),
            A_eq=scaled_features,
            b_eq=scaled_target,
            bounds=(0.0, None),
            method="highs",
        )
        indices = np.flatnonzero(solution.x > 1.0e-10) if solution.success else np.array([], int)
        if not solution.success or indices.size != model.FEATURE_COUNT:
            raise RuntimeError(
                f"time {time_index}: no positive square basic feasible chart; "
                f"status={solution.message!r}, active={indices.size}"
            )
        matrix = scaled_features[:, indices]
        selected_weights = np.linalg.solve(matrix, scaled_target)
        if np.min(selected_weights) <= 0.0 or np.linalg.matrix_rank(matrix) != model.FEATURE_COUNT:
            raise RuntimeError(f"time {time_index}: selected chart is not strictly positive/full rank")
        projection = tp._contract_e_tp_dense_square_forward_core(
            teacher["particles"],
            teacher["log_unnormalized_weights"],
            teacher["features"],
            tf.constant(indices, tf.int32),
            tf.constant(scale, DTYPE),
        )
        parents = projection["student_points"]
        parent_log_weights = tf.math.log(projection["student_weights"])
        active_indices.append(indices.tolist())
        row_scales.append(scale.tolist())
        chart_records.append(
            {
                "time_index_zero_based": time_index,
                "teacher_count": int(features.shape[1]),
                "active_indices": indices.tolist(),
                "row_scale": scale.tolist(),
                "weights": projection["student_weights"].numpy().tolist(),
                "minimum_weight": float(projection["minimum_weight"].numpy()),
                "scaled_condition_number": float(projection["condition_number"].numpy()),
                "feature_residual_max_abs": float(
                    tf.reduce_max(tf.abs(projection["feature_residual"])).numpy()
                ),
                "incoming_weight_sum": float(
                    tf.reduce_sum(tf.exp(parent_log_weights)).numpy()
                ),
            }
        )
    payload = {
        "schema": "bayesfilter.contract_e_tp.lgssm_center_chart_preparation.v1",
        "status": "PASS_CENTER_ONLY_PREPARATION",
        "algorithm_id": model.ALGORITHM_ID,
        "route": "corrected_ledh_parent_by_innovation_teacher",
        "scope": "center_only_not_parameter_region_certificate",
        "feature_names": list(model.FEATURE_NAMES),
        "target": {
            "row_id": "benchmark_lgssm_exact_oracle_m3_T50",
            "dataset_seed": DATASET_SEED,
            "observations_sha256": _tensor_sha256(observations),
            "theta": THETA.numpy().tolist(),
            "theta_coordinate_system": "physical_benchmark_exact_oracle",
            "time_steps": args.time_steps,
            "time_order": "transition_then_corrected_ledh_observation_increment_then_projection",
        },
        "quadrature": {
            "family": "tensor_gauss_hermite_standard_normal",
            "one_dimensional_order": args.order,
            "parent_count_initial": args.order**3,
            "innovation_count": args.order**3,
            "status": "capacity_baseline_hypothesis_not_promoted_default",
            "nodes": nodes.numpy().tolist(),
            "weights": weights.numpy().tolist(),
        },
        "selection": {
            "method": "scipy.optimize.linprog_highs_zero_objective_then_frozen_basic_support",
            "positive_support_cutoff": 1.0e-10,
            "runtime_active_set_selection": False,
            "clipping": False,
            "role": "offline_preparation_only",
        },
        "active_indices": active_indices,
        "row_scales": row_scales,
        "charts": chart_records,
        "summary": {
            "chart_count": len(chart_records),
            "minimum_weight": min(row["minimum_weight"] for row in chart_records),
            "maximum_condition_number": max(
                row["scaled_condition_number"] for row in chart_records
            ),
            "maximum_feature_residual_abs": max(
                row["feature_residual_max_abs"] for row in chart_records
            ),
        },
        "execution": {
            "backend": "TensorFlow float64 finite program plus SciPy offline support selection",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "jit_compile": False,
            "git_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
            ).stdout.strip(),
            "wall_time_seconds": time.perf_counter() - started,
            "command": (
                "CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/matplotlib-contract-e-tp "
                "python " + " ".join(sys.argv)
            ),
        },
        "nonclaims": [
            "no off-center or HMC parameter-region chart certificate",
            "no quadrature convergence or Kalman equivalence",
            "no canonical, leaderboard, default, or HMC admission",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
