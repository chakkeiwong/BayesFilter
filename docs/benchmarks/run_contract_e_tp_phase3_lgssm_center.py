#!/usr/bin/env python3
"""Evaluate the prepared center-only recursive LGSSM Contract E--TP rung."""

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


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.highdim import ledh_contract_e_tp_lgssm_tf as model
from scripts.filtering_value_gradient_benchmark_generate_p8_datasets import (
    _lgssm_dataset,
)


DTYPE = tf.float64
THETA = tf.constant([0.72, 0.55, 0.35, 0.35, 0.45], DTYPE)
PARAMETER_NAMES = ("phi1", "phi2", "phi3", "q_scale", "r_scale")
DEFAULT_PREPARATION = ROOT / (
    "docs/benchmarks/artifacts/contract_e_tp_all_models_2026_07_15/"
    "phase3_lgssm_order3_center_preparation_20260715/charts.json"
)
DELTA_GRAD = 0.05
VALUE_BOUNDARY = 0.001
FD_STEP = 1.0e-5


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-steps", type=int, choices=(1, 2, 5, 50), required=True)
    parser.add_argument("--preparation", type=Path, default=DEFAULT_PREPARATION)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse()
    started = time.perf_counter()
    if args.output.exists():
        raise FileExistsError(args.output)
    preparation_path = args.preparation
    if not preparation_path.is_absolute():
        preparation_path = ROOT / preparation_path
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    if preparation["target"]["time_steps"] < args.time_steps:
        raise ValueError("preparation horizon is shorter than the requested evaluation")
    observations = tf.convert_to_tensor(
        _lgssm_dataset(81100)["observations"][: args.time_steps], DTYPE
    )
    nodes = tf.constant(preparation["quadrature"]["nodes"], DTYPE)
    weights = tf.constant(preparation["quadrature"]["weights"], DTYPE)
    active_indices = tf.constant(
        preparation["active_indices"][: args.time_steps - 1], tf.int32
    )
    row_scales = tf.constant(
        preparation["row_scales"][: args.time_steps - 1], DTYPE
    )
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(THETA)
        result = model.contract_e_tp_lgssm_recursive_core(
            THETA,
            observations,
            nodes,
            weights,
            active_indices,
            row_scales,
        )
        oracle_value = model.exact_kalman_value(THETA, observations)
    score = tape.gradient(result["objective"], THETA)
    oracle_score = tape.gradient(oracle_value, THETA)
    finite_difference = []
    for index in range(model.lgssm.PARAMETER_COUNT):
        direction = np.zeros(model.lgssm.PARAMETER_COUNT)
        direction[index] = FD_STEP
        plus = model.contract_e_tp_lgssm_recursive_core(
            tf.constant(THETA.numpy() + direction, DTYPE),
            observations,
            nodes,
            weights,
            active_indices,
            row_scales,
        )["objective"]
        minus = model.contract_e_tp_lgssm_recursive_core(
            tf.constant(THETA.numpy() - direction, DTYPE),
            observations,
            nodes,
            weights,
            active_indices,
            row_scales,
        )["objective"]
        finite_difference.append(float((plus - minus).numpy() / (2.0 * FD_STEP)))
    finite_difference_array = np.asarray(finite_difference)
    score_array = score.numpy()
    oracle_score_array = oracle_score.numpy()
    fd_relative = np.abs(score_array - finite_difference_array) / np.maximum(
        np.maximum(np.abs(score_array), np.abs(finite_difference_array)), 1.0e-12
    )
    oracle_relative = np.abs(score_array - oracle_score_array) / np.abs(
        oracle_score_array
    )
    sign_reversal = np.sign(score_array) != np.sign(oracle_score_array)
    same_scalar_fd_pass = bool(np.max(fd_relative) <= 0.05 * np.sqrt(5.0))
    center_gradient_pass = bool(
        np.max(oracle_relative) <= DELTA_GRAD and not np.any(sign_reversal)
    )
    center_value_pass = bool(
        abs(float(result["objective"].numpy() - oracle_value.numpy())) <= VALUE_BOUNDARY
    )
    chart_pass = bool(tf.reduce_all(result["valid_history"]).numpy())
    payload = {
        "schema": "bayesfilter.contract_e_tp.phase3_lgssm_center.v1",
        "status": "PASS_ENGINEERING" if chart_pass and same_scalar_fd_pass else "FAIL_ENGINEERING",
        "algorithm_id": model.ALGORITHM_ID,
        "scope": "center_only_not_parameter_region_certificate",
        "target": {
            "row_id": "benchmark_lgssm_exact_oracle_m3_T50",
            "dataset_seed": 81100,
            "time_steps": args.time_steps,
            "theta": THETA.numpy().tolist(),
            "parameter_names": list(PARAMETER_NAMES),
            "route": "corrected_ledh_parent_by_innovation_teacher",
        },
        "preparation": {
            "path": str(preparation_path.relative_to(ROOT)),
            "sha256": _sha256(preparation_path),
            "scope": preparation["scope"],
            "quadrature_order": preparation["quadrature"]["one_dimensional_order"],
            "feature_names": preparation["feature_names"],
        },
        "value": {
            "contract_e_tp": float(result["objective"].numpy()),
            "kalman": float(oracle_value.numpy()),
            "difference": float(result["objective"].numpy() - oracle_value.numpy()),
            "center_boundary": VALUE_BOUNDARY,
            "center_screen_pass": center_value_pass,
            "increment_history": result["increment_history"].numpy().tolist(),
        },
        "score": {
            "contract_e_tp": score_array.tolist(),
            "kalman": oracle_score_array.tolist(),
            "difference": (score_array - oracle_score_array).tolist(),
            "componentwise_relative_error": oracle_relative.tolist(),
            "sign_reversal": sign_reversal.tolist(),
            "center_delta_grad": DELTA_GRAD,
            "center_screen_pass": center_gradient_pass,
            "same_scalar_finite_difference": finite_difference,
            "same_scalar_fd_relative_error": fd_relative.tolist(),
            "same_scalar_fd_threshold": float(0.05 * np.sqrt(5.0)),
            "same_scalar_fd_pass": same_scalar_fd_pass,
        },
        "chart": {
            "valid_history": result["valid_history"].numpy().tolist(),
            "minimum_weight_history": result["minimum_weight_history"].numpy().tolist(),
            "condition_number_history": result["condition_number_history"].numpy().tolist(),
            "feature_residual_max_abs": float(
                tf.reduce_max(tf.abs(result["feature_residual_history"])).numpy()
            )
            if args.time_steps > 1
            else 0.0,
            "chart_pass": chart_pass,
        },
        "decision": {
            "engineering_pass": chart_pass and same_scalar_fd_pass,
            "center_value_screen_pass": center_value_pass,
            "center_gradient_screen_pass": center_gradient_pass,
            "next_action": (
                "advance_to_order_refinement_or_later_phase"
                if center_value_pass and center_gradient_pass
                else "refine_quadrature_before_interpreting_recursive_accuracy"
            ),
            "ranking_supported": False,
            "differences_are_descriptive": True,
        },
        "execution": {
            "git_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
            ).stdout.strip(),
            "backend": "TensorFlow float64 CPU-hidden reference/diagnostic",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "jit_compile": False,
            "wall_time_seconds": time.perf_counter() - started,
            "command": " ".join(sys.argv),
        },
        "nonclaims": [
            "no off-center or HMC trajectory validity",
            "no stochastic ranking or cross-method equivalence",
            "no canonical, leaderboard, default, or HMC admission",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"value": payload["value"], "score": payload["score"], "decision": payload["decision"]}, indent=2))
    if not chart_pass or not same_scalar_fd_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
