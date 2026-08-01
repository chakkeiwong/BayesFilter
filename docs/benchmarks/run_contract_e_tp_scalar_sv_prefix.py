#!/usr/bin/env python3
"""Evaluate one prepared scalar-SV Contract E--TP prefix against dense target filters."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import numpy as np
import tensorflow as tf
from numpy.polynomial.legendre import leggauss


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.highdim import ledh_contract_e_tp_scalar_sv_tf as model
from bayesfilter.highdim.ledh_forward_contract import (
    ACTUAL_SV_ROW_ID,
    GENERALIZED_SV_ROW_ID,
    KSC_SV_ROW_ID,
)
from bayesfilter.ledh_fd_policy import evaluate_ledh_fd_policy
from scripts.filtering_value_gradient_benchmark_generate_p8_datasets import (
    _generalized_sv_prior_mean_dataset,
    _sv_dataset,
)


DTYPE = tf.float64
ROW_IDS = (ACTUAL_SV_ROW_ID, KSC_SV_ROW_ID, GENERALIZED_SV_ROW_ID)
PARAMETER_NAMES = {
    ACTUAL_SV_ROW_ID: ("gamma_unconstrained", "log_beta"),
    KSC_SV_ROW_ID: ("gamma_unconstrained", "log_beta"),
    GENERALIZED_SV_ROW_ID: ("gamma_unconstrained", "log_tau", "mu_over_tau"),
}
FD_STEP = 1.0e-5


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--reference-orders", default="129,257")
    parser.add_argument("--reference-radius", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_sha256(value: tf.Tensor) -> str:
    return hashlib.sha256(tf.io.serialize_tensor(value).numpy()).hexdigest()


def _row_data(row_id: str, time_steps: int) -> tuple[tf.Tensor, tf.Tensor]:
    if row_id in (ACTUAL_SV_ROW_ID, KSC_SV_ROW_ID):
        dataset = _sv_dataset(81101)
    else:
        dataset = _generalized_sv_prior_mean_dataset(81105)
    return (
        tf.constant(dataset["truth_theta"], DTYPE),
        tf.convert_to_tensor(dataset["observations"][:time_steps], DTYPE),
    )


def _legendre_rule(order: int, radius: float) -> tuple[tf.Tensor, tf.Tensor]:
    nodes, weights = leggauss(order)
    return (
        tf.constant(radius * nodes, DTYPE),
        tf.constant(radius * weights, DTYPE),
    )


def _value_and_score(callable_, theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    with tf.GradientTape() as tape:
        tape.watch(theta)
        result = callable_(theta)
    return result, tape.gradient(result, theta)


def _result_value_score_and_increment_score(
    callable_, theta: tf.Tensor
) -> tuple[dict[str, object], tf.Tensor, tf.Tensor]:
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(theta)
        result = callable_(theta)
        objective = result["objective"]
        increments = result["increment_history"]
    score = tape.gradient(objective, theta)
    increment_score = tape.jacobian(increments, theta)
    del tape
    return result, score, increment_score


def _central_fd(callable_, theta: tf.Tensor) -> list[float]:
    values = []
    for index in range(int(theta.shape[0])):
        direction = tf.one_hot(index, int(theta.shape[0]), dtype=DTYPE)
        values.append(
            float(
                (
                    callable_(theta + FD_STEP * direction)
                    - callable_(theta - FD_STEP * direction)
                ).numpy()
                / (2.0 * FD_STEP)
            )
        )
    return values


def main() -> None:
    args = _parse()
    started = time.perf_counter()
    if args.output.exists():
        raise FileExistsError(args.output)
    path = args.preparation if args.preparation.is_absolute() else ROOT / args.preparation
    preparation = json.loads(path.read_text(encoding="utf-8"))
    row_id = preparation["row_id"]
    if row_id not in ROW_IDS:
        raise ValueError(f"unsupported row in preparation: {row_id}")
    time_steps = int(preparation["target"]["time_steps"])
    spec = model.make_scalar_sv_spec(row_id)
    generated_theta, generated_raw_observations = _row_data(row_id, time_steps)
    theta = tf.constant(preparation["target"]["theta"], DTYPE)
    tf.debugging.assert_equal(theta, generated_theta)
    if "raw_observations" in preparation["target"]:
        raw_observations = tf.constant(
            preparation["target"]["raw_observations"], DTYPE
        )
        target_observations = tf.constant(
            preparation["target"]["target_observations"], DTYPE
        )
        flow_observations = tf.constant(
            preparation["target"]["flow_observations"], DTYPE
        )
        if _tensor_sha256(raw_observations) != preparation["target"]["raw_observations_sha256"]:
            raise ValueError("prepared raw-observation hash mismatch")
        if _tensor_sha256(target_observations) != preparation["target"]["target_observations_sha256"]:
            raise ValueError("prepared target-observation hash mismatch")
        if _tensor_sha256(flow_observations) != preparation["target"]["flow_observations_sha256"]:
            raise ValueError("prepared flow-observation hash mismatch")
    else:
        raw_observations = generated_raw_observations
        target_observations, flow_observations = model.target_and_flow_observations(
            spec, raw_observations
        )
    nodes = tf.constant(preparation["teacher_quadrature"]["nodes"], DTYPE)
    weights = tf.constant(preparation["teacher_quadrature"]["weights"], DTYPE)
    grid = tf.constant(preparation["continuation_quadrature"]["points"], DTYPE)
    grid_weights = tf.constant(
        preparation["continuation_quadrature"]["weights"], DTYPE
    )
    chart_mode = preparation.get("chart_contract", {}).get("mode", "fixed_square")
    lookahead_steps = int(preparation["feature_contract"]["lookahead_steps"])
    requested_lookaheads = tuple(
        int(value)
        for value in preparation["feature_contract"].get(
            "requested_progressive_lookaheads", [lookahead_steps]
        )
    )
    if chart_mode == "fixed_overcomplete_kkt":
        active_indices = tuple(
            tf.constant(value, tf.int32) for value in preparation["active_indices"]
        )
        row_scales = tuple(
            tf.constant(value, DTYPE) for value in preparation["row_scales"]
        )
        reference_weights = tuple(
            tf.constant(value, DTYPE) for value in preparation["reference_weights"]
        )
        precisions = tuple(
            tf.constant(value, DTYPE) for value in preparation["precisions"]
        )
    else:
        active_indices = tf.reshape(
            tf.constant(preparation["active_indices"], tf.int32),
            [time_steps - 1, model.FEATURE_COUNT],
        )
        row_scales = tf.reshape(
            tf.constant(preparation["row_scales"], DTYPE),
            [time_steps - 1, model.FEATURE_COUNT],
        )

    def candidate_result_fn(value: tf.Tensor) -> dict[str, object]:
        if chart_mode == "fixed_overcomplete_kkt":
            return model.contract_e_tp_scalar_sv_recursive_kkt_core(
                spec,
                value,
                target_observations,
                flow_observations,
                nodes,
                weights,
                active_indices,
                row_scales,
                reference_weights,
                precisions,
                grid,
                grid_weights,
                requested_lookaheads=requested_lookaheads,
            )
        return model.contract_e_tp_scalar_sv_recursive_core(
            spec,
            value,
            target_observations,
            flow_observations,
            nodes,
            weights,
            active_indices,
            row_scales,
            grid,
            grid_weights,
            lookahead_steps=lookahead_steps,
        )

    def candidate(value: tf.Tensor) -> tf.Tensor:
        return candidate_result_fn(value)["objective"]

    candidate_result, candidate_score, candidate_increment_score = (
        _result_value_score_and_increment_score(candidate_result_fn, theta)
    )
    candidate_value = candidate_result["objective"]
    finite_difference = _central_fd(candidate, theta)
    fd_policy = evaluate_ledh_fd_policy(
        candidate_score.numpy().tolist(),
        finite_difference,
        PARAMETER_NAMES[row_id],
    )

    reference_orders = tuple(int(item) for item in args.reference_orders.split(","))
    if len(reference_orders) < 2 or sorted(reference_orders) != list(reference_orders):
        raise ValueError("reference orders must contain at least two increasing integers")
    references = []
    for order in reference_orders:
        reference_grid, reference_weights = _legendre_rule(order, args.reference_radius)

        def reference_result_fn(value: tf.Tensor) -> dict[str, tf.Tensor]:
            return model.scalar_sv_dense_reference_value(
                spec,
                value,
                target_observations,
                reference_grid,
                reference_weights,
            )

        reference_result, reference_score, reference_increment_score = (
            _result_value_score_and_increment_score(reference_result_fn, theta)
        )
        reference_value = reference_result["objective"]
        references.append(
            {
                "order": order,
                "radius": args.reference_radius,
                "value": float(reference_value.numpy()),
                "score": reference_score.numpy().tolist(),
                "increment_history": reference_result["increment_history"].numpy().tolist(),
                "increment_score_history": reference_increment_score.numpy().tolist(),
            }
        )
    finest = references[-1]
    candidate_score_array = candidate_score.numpy()
    finest_score = np.asarray(finest["score"])
    score_scale = np.maximum(
        np.maximum(np.abs(candidate_score_array), np.abs(finest_score)), 1.0e-12
    )
    target_relative_error = np.abs(candidate_score_array - finest_score) / score_scale
    reference_value_gap = references[-1]["value"] - references[-2]["value"]
    reference_score_gap = (
        np.asarray(references[-1]["score"]) - np.asarray(references[-2]["score"])
    )
    chart_pass = bool(tf.reduce_all(candidate_result["valid_history"]).numpy())
    payload = {
        "schema": "bayesfilter.contract_e_tp.scalar_sv_prefix_result.v1",
        "status": "PASS_ENGINEERING" if chart_pass and fd_policy["status"] == "pass" else "FAIL_ENGINEERING",
        "algorithm_id": model.ALGORITHM_ID,
        "row_id": row_id,
        "scope": "center_only_prefix_diagnostic",
        "target": {
            "time_steps": time_steps,
            "theta": theta.numpy().tolist(),
            "parameter_names": list(PARAMETER_NAMES[row_id]),
            "target_observation_policy": spec.target_observation_policy,
            "flow_observation_policy": spec.flow_observation_policy,
            "transition_before_first_observation": spec.transition_before_first_observation,
        },
        "preparation": {
            "path": str(path.relative_to(ROOT)),
            "sha256": _sha256(path),
            "teacher_order": preparation["teacher_quadrature"]["order"],
            "continuation_order": preparation["continuation_quadrature"]["order"],
            "continuation_radius": preparation["continuation_quadrature"]["radius"],
            "lookahead_steps": lookahead_steps,
            "requested_progressive_lookaheads": list(requested_lookaheads),
            "chart_mode": chart_mode,
        },
        "candidate": {
            "value": float(candidate_value.numpy()),
            "score": candidate_score_array.tolist(),
            "increment_history": candidate_result["increment_history"].numpy().tolist(),
            "increment_score_history": candidate_increment_score.numpy().tolist(),
            "same_scalar_finite_difference": finite_difference,
            "same_scalar_fd_policy": fd_policy,
        },
        "dense_references": references,
        "reference_refinement": {
            "finest_two_value_difference": float(reference_value_gap),
            "finest_two_score_difference": reference_score_gap.tolist(),
            "classification": "descriptive_numerical_refinement_not_exact_error_bound",
        },
        "candidate_vs_finest_reference": {
            "value_difference": float(candidate_value.numpy() - finest["value"]),
            "score_difference": (candidate_score_array - finest_score).tolist(),
            "componentwise_relative_error": target_relative_error.tolist(),
            "sign_reversal": (np.sign(candidate_score_array) != np.sign(finest_score)).tolist(),
            "increment_score_difference_history": (
                candidate_increment_score.numpy()
                - np.asarray(finest["increment_score_history"])
            ).tolist(),
            "equivalence_classification": "descriptive_only_margin_unavailable",
        },
        "chart": {
            "valid_history": candidate_result["valid_history"].numpy().tolist(),
            "minimum_weight_history": candidate_result["minimum_weight_history"].numpy().tolist(),
            "condition_number_history": candidate_result["condition_number_history"].numpy().tolist(),
            "maximum_feature_residual_abs": float(
                max(
                    (
                        tf.reduce_max(tf.abs(value)).numpy()
                        for value in candidate_result["feature_residual_history"]
                    ),
                    default=0.0,
                )
                if chart_mode == "fixed_overcomplete_kkt"
                else tf.reduce_max(
                    tf.abs(candidate_result["feature_residual_history"])
                ).numpy()
            )
            if time_steps > 1
            else 0.0,
            "chart_pass": chart_pass,
        },
        "decision": {
            "hard_veto_screen_pass": chart_pass and fd_policy["status"] == "pass",
            "candidate_remains_viable": chart_pass and fd_policy["status"] == "pass",
            "statistically_supported_ranking": False,
            "default_readiness": False,
            "next_evidence": "teacher/continuation refinement or next declared prefix",
        },
        "execution": {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "backend": "TensorFlow_float64_CPU_reference_exception",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "jit_compile": False,
            "wall_time_seconds": time.perf_counter() - started,
            "command": " ".join(sys.argv),
        },
        "nonclaims": [
            "no unsupported cross-method equivalence threshold",
            "not full-horizon or parameter-region evidence",
            "not HMC, canonical, or default readiness",
            "not a statistically supported method ranking",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "row_id": row_id,
                "time_steps": time_steps,
                "candidate_value": payload["candidate"]["value"],
                "reference_value": finest["value"],
                "value_difference": payload["candidate_vs_finest_reference"]["value_difference"],
                "candidate_score": payload["candidate"]["score"],
                "reference_score": finest["score"],
                "relative_score_error": payload["candidate_vs_finest_reference"]["componentwise_relative_error"],
                "fd_status": fd_policy["status"],
                "chart_pass": chart_pass,
            },
            indent=2,
        )
    )
    if not payload["decision"]["hard_veto_screen_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
