#!/usr/bin/env python3
"""Run one bounded-memory LGSSM score direction for a prepared chart."""

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
FD_STEP = 1.0e-5


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--direction-index", type=int, choices=range(5), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    if args.output.exists():
        raise FileExistsError(args.output)
    preparation_path = args.preparation
    if not preparation_path.is_absolute():
        preparation_path = ROOT / preparation_path
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    time_steps = int(preparation["target"]["time_steps"])
    observations = tf.convert_to_tensor(
        _lgssm_dataset(81100)["observations"][:time_steps], DTYPE
    )
    nodes = tf.constant(preparation["quadrature"]["nodes"], DTYPE)
    weights = tf.constant(preparation["quadrature"]["weights"], DTYPE)
    active_indices = tf.constant(preparation["active_indices"], tf.int32)
    row_scales = tf.constant(preparation["row_scales"], DTYPE)
    mode = preparation["feature_mode"]
    lookahead_steps = preparation.get("lookahead_steps")
    tangent = tf.one_hot(args.direction_index, 5, dtype=DTYPE)
    with tf.autodiff.ForwardAccumulator(THETA, tangent) as accumulator:
        result = model.contract_e_tp_lgssm_score_informed_recursive_core(
            THETA,
            observations,
            nodes,
            weights,
            active_indices,
            row_scales,
            feature_mode=mode,
            lookahead_steps=lookahead_steps,
        )
    score = accumulator.jvp(result["objective"])
    direction = np.zeros(5)
    direction[args.direction_index] = FD_STEP
    plus = model.contract_e_tp_lgssm_score_informed_recursive_core(
        tf.constant(THETA.numpy() + direction, DTYPE),
        observations,
        nodes,
        weights,
        active_indices,
        row_scales,
        feature_mode=mode,
        lookahead_steps=lookahead_steps,
    )
    minus = model.contract_e_tp_lgssm_score_informed_recursive_core(
        tf.constant(THETA.numpy() - direction, DTYPE),
        observations,
        nodes,
        weights,
        active_indices,
        row_scales,
        feature_mode=mode,
        lookahead_steps=lookahead_steps,
    )
    finite_difference = (plus["objective"] - minus["objective"]) / (2.0 * FD_STEP)
    increment_fd = (
        plus["increment_history"] - minus["increment_history"]
    ) / (2.0 * FD_STEP)
    with tf.autodiff.ForwardAccumulator(THETA, tangent) as oracle_accumulator:
        oracle_value = model.exact_kalman_value(THETA, observations)
    oracle_score = oracle_accumulator.jvp(oracle_value)
    relative_fd = tf.abs(score - finite_difference) / tf.maximum(
        tf.maximum(tf.abs(score), tf.abs(finite_difference)), 1.0e-12
    )
    payload = {
        "schema": "bayesfilter.contract_e_tp.phase8_lgssm_direction_shard.v1",
        "status": "PASS_SAME_SCALAR_DIRECTION" if float(relative_fd.numpy()) <= 0.05 * np.sqrt(5.0) else "FAIL_SAME_SCALAR_DIRECTION",
        "preparation": {
            "path": str(preparation_path.relative_to(ROOT)),
            "sha256": _sha256(preparation_path),
        },
        "feature_mode": mode,
        "lookahead_steps": lookahead_steps,
        "time_steps": time_steps,
        "direction_index": args.direction_index,
        "objective": float(result["objective"].numpy()),
        "increment_history": result["increment_history"].numpy().tolist(),
        "score": float(score.numpy()),
        "finite_difference": float(finite_difference.numpy()),
        "finite_difference_relative_error": float(relative_fd.numpy()),
        "increment_finite_difference_history": increment_fd.numpy().tolist(),
        "kalman_value": float(oracle_value.numpy()),
        "kalman_score": float(oracle_score.numpy()),
        "chart": {
            "valid": bool(tf.reduce_all(result["valid_history"]).numpy()),
            "minimum_weight": float(tf.reduce_min(result["minimum_weight_history"]).numpy()),
            "maximum_condition_number": float(tf.reduce_max(result["condition_number_history"]).numpy()),
            "maximum_feature_residual_abs": float(tf.reduce_max(tf.abs(result["feature_residual_history"])).numpy()),
        },
        "execution": {
            "git_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
            ).stdout.strip(),
            "backend": "TensorFlow float64 CPU-hidden forward scalar JVP shard",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "wall_time_seconds": time.perf_counter() - started,
            "command": " ".join(sys.argv),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("status", "direction_index", "objective", "score", "finite_difference", "kalman_score")}, indent=2))
    if not payload["chart"]["valid"] or payload["status"].startswith("FAIL"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
