#!/usr/bin/env python3
"""Evaluate a prepared score-informed recursive LGSSM Contract E--TP rung."""

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
DELTA_GRAD = 0.05
VALUE_BOUNDARY = 0.001
FD_STEP = 1.0e-5


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--derivative-mode", choices=("reverse", "forward_scalar"), default="reverse"
    )
    parser.add_argument("--omit-kalman-prefix-scores", action="store_true")
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
    time_steps = int(preparation["target"]["time_steps"])
    feature_mode = preparation["feature_mode"]
    lookahead_steps = preparation.get("lookahead_steps")
    observations = tf.convert_to_tensor(
        _lgssm_dataset(81100)["observations"][:time_steps], DTYPE
    )
    nodes = tf.constant(preparation["quadrature"]["nodes"], DTYPE)
    weights = tf.constant(preparation["quadrature"]["weights"], DTYPE)
    active_indices = tf.constant(preparation["active_indices"], tf.int32)
    row_scales = tf.constant(preparation["row_scales"], DTYPE)
    if args.derivative_mode == "reverse":
        with tf.GradientTape() as tape:
            tape.watch(THETA)
            result = model.contract_e_tp_lgssm_score_informed_recursive_core(
                THETA,
                observations,
                nodes,
                weights,
                active_indices,
                row_scales,
                feature_mode=feature_mode,
                lookahead_steps=lookahead_steps,
            )
        score = tape.gradient(result["objective"], THETA)
    else:
        result = model.contract_e_tp_lgssm_score_informed_recursive_core(
            THETA,
            observations,
            nodes,
            weights,
            active_indices,
            row_scales,
            feature_mode=feature_mode,
            lookahead_steps=lookahead_steps,
        )
        score_columns = []
        for index in range(model.lgssm.PARAMETER_COUNT):
            tangent = tf.one_hot(index, model.lgssm.PARAMETER_COUNT, dtype=DTYPE)
            with tf.autodiff.ForwardAccumulator(THETA, tangent) as accumulator:
                directional_objective = (
                    model.contract_e_tp_lgssm_score_informed_recursive_core(
                        THETA,
                        observations,
                        nodes,
                        weights,
                        active_indices,
                        row_scales,
                        feature_mode=feature_mode,
                        lookahead_steps=lookahead_steps,
                    )["objective"]
                )
            score_columns.append(accumulator.jvp(directional_objective))
        score = tf.stack(score_columns)
    oracle_prefix_values = tf.stack(
        [
            model.exact_kalman_value(THETA, observations[: prefix + 1])
            for prefix in range(time_steps)
        ]
    )
    with tf.GradientTape() as oracle_tape:
        oracle_tape.watch(THETA)
        oracle_total_value = model.exact_kalman_value(THETA, observations)
    oracle_score = oracle_tape.gradient(oracle_total_value, THETA)
    oracle_value = oracle_prefix_values[-1]
    finite_difference = []
    finite_difference_increment_columns = []
    oracle_finite_difference_prefix_columns = []
    for index in range(model.lgssm.PARAMETER_COUNT):
        direction = np.zeros(model.lgssm.PARAMETER_COUNT)
        direction[index] = FD_STEP
        plus_result = model.contract_e_tp_lgssm_score_informed_recursive_core(
            tf.constant(THETA.numpy() + direction, DTYPE),
            observations,
            nodes,
            weights,
            active_indices,
            row_scales,
            feature_mode=feature_mode,
            lookahead_steps=lookahead_steps,
        )
        minus_result = model.contract_e_tp_lgssm_score_informed_recursive_core(
            tf.constant(THETA.numpy() - direction, DTYPE),
            observations,
            nodes,
            weights,
            active_indices,
            row_scales,
            feature_mode=feature_mode,
            lookahead_steps=lookahead_steps,
        )
        finite_difference.append(
            float(
                (plus_result["objective"] - minus_result["objective"]).numpy()
                / (2.0 * FD_STEP)
            )
        )
        finite_difference_increment_columns.append(
            (
                plus_result["increment_history"]
                - minus_result["increment_history"]
            )
            / (2.0 * FD_STEP)
        )
        if not args.omit_kalman_prefix_scores:
            plus_oracle_prefix = tf.stack(
                [
                    model.exact_kalman_value(
                        tf.constant(THETA.numpy() + direction, DTYPE),
                        observations[: prefix + 1],
                    )
                    for prefix in range(time_steps)
                ]
            )
            minus_oracle_prefix = tf.stack(
                [
                    model.exact_kalman_value(
                        tf.constant(THETA.numpy() - direction, DTYPE),
                        observations[: prefix + 1],
                    )
                    for prefix in range(time_steps)
                ]
            )
            oracle_finite_difference_prefix_columns.append(
                (plus_oracle_prefix - minus_oracle_prefix) / (2.0 * FD_STEP)
            )
    increment_score = tf.stack(finite_difference_increment_columns, axis=1)
    cumulative_score = tf.cumsum(increment_score, axis=0)
    oracle_prefix_score = (
        tf.stack(oracle_finite_difference_prefix_columns, axis=1)
        if oracle_finite_difference_prefix_columns
        else None
    )
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
        abs(float(result["objective"].numpy() - oracle_value.numpy()))
        <= VALUE_BOUNDARY
    )
    chart_pass = bool(tf.reduce_all(result["valid_history"]).numpy())
    payload = {
        "schema": "bayesfilter.contract_e_tp.phase8_lgssm_score_informed.v1",
        "status": (
            "PASS_ENGINEERING" if chart_pass and same_scalar_fd_pass else "FAIL_ENGINEERING"
        ),
        "algorithm_id": model.ALGORITHM_ID,
        "feature_mode": feature_mode,
        "lookahead_steps": lookahead_steps,
        "feature_role": preparation["feature_role"],
        "scope": "center_only_not_parameter_region_certificate",
        "target": {
            "row_id": "benchmark_lgssm_exact_oracle_m3_T50",
            "dataset_seed": 81100,
            "time_steps": time_steps,
            "theta": THETA.numpy().tolist(),
            "parameter_names": list(PARAMETER_NAMES),
            "route": "corrected_ledh_parent_by_innovation_teacher",
        },
        "preparation": {
            "path": str(preparation_path.relative_to(ROOT)),
            "sha256": _sha256(preparation_path),
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
            "cumulative_history": np.cumsum(
                result["increment_history"].numpy()
            ).tolist(),
            "kalman_prefix_history": oracle_prefix_values.numpy().tolist(),
        },
        "score": {
            "contract_e_tp": score_array.tolist(),
            "kalman": oracle_score_array.tolist(),
            "difference": (score_array - oracle_score_array).tolist(),
            "componentwise_relative_error": oracle_relative.tolist(),
            "sign_reversal": sign_reversal.tolist(),
            "center_delta_grad": DELTA_GRAD,
            "center_screen_pass": center_gradient_pass,
            "increment_history": increment_score.numpy().tolist(),
            "cumulative_history": cumulative_score.numpy().tolist(),
            "kalman_prefix_history": (
                oracle_prefix_score.numpy().tolist()
                if oracle_prefix_score is not None
                else None
            ),
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
            ),
            "score_mark_mean_history": result["score_mark_mean_history"].numpy().tolist(),
            "score_mark_center_residual_max_abs": float(
                tf.reduce_max(
                    tf.abs(result["score_mark_center_residual_history"])
                ).numpy()
            )
            if feature_mode == "progressive_target_model_score"
            else None,
            "chart_pass": chart_pass,
        },
        "decision": {
            "engineering_pass": chart_pass and same_scalar_fd_pass,
            "center_value_screen_pass": center_value_pass,
            "center_gradient_screen_pass": center_gradient_pass,
            "lgssm_candidate_pass": (
                chart_pass
                and same_scalar_fd_pass
                and center_value_pass
                and center_gradient_pass
            ),
            "ranking_supported": False,
            "differences_are_descriptive": True,
        },
        "execution": {
            "git_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "backend": "TensorFlow float64 CPU-hidden reference/diagnostic",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "jit_compile": False,
            "derivative_mode": args.derivative_mode,
            "per_time_score_history_role": "centered_fd_explanatory_localization",
            "kalman_prefix_score_omitted": args.omit_kalman_prefix_scores,
            "wall_time_seconds": time.perf_counter() - started,
            "command": " ".join(sys.argv),
        },
        "nonclaims": preparation["nonclaims"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "value": payload["value"],
                "score": {key: payload["score"][key] for key in (
                    "contract_e_tp",
                    "kalman",
                    "componentwise_relative_error",
                    "sign_reversal",
                    "same_scalar_fd_relative_error",
                )},
                "decision": payload["decision"],
            },
            indent=2,
        )
    )
    if not chart_pass or not same_scalar_fd_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
