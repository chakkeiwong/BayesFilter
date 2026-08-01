#!/usr/bin/env python3
"""Corroborate one deterministically weakest overcomplete solve at high precision."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
    raise RuntimeError("high-precision audit requires deliberate CPU-only hiding")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mpmath as mp  # noqa: E402
import tensorflow as tf  # noqa: E402

from bayesfilter.highdim import ledh_contract_e_tp_scalar_sv_tf as model  # noqa: E402


DTYPE = tf.float64


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--point-kind", choices=("design", "held-out"), required=True)
    parser.add_argument("--point-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mp_matrix(rows: list[list[float]]) -> mp.matrix:
    return mp.matrix([[mp.mpf(repr(value)) for value in row] for row in rows])


def _mp_vector(values: list[float]) -> mp.matrix:
    return mp.matrix([mp.mpf(repr(value)) for value in values])


def _solve_at_precision(
    scaled_matrix: list[list[float]],
    scaled_target: list[float],
    reference: list[float],
    digits: int,
) -> dict[str, Any]:
    mp.mp.dps = digits
    matrix = _mp_matrix(scaled_matrix)
    target = _mp_vector(scaled_target)
    r = _mp_vector(reference)
    inverse_precision = mp.diag(r)
    gram = matrix * inverse_precision * matrix.T
    residual = target - matrix * r
    multiplier = mp.lu_solve(gram, residual)
    weights = r + inverse_precision * matrix.T * multiplier
    equality = matrix * weights - target
    minimum = min(weights)
    return {
        "decimal_digits": digits,
        "minimum_weight": mp.nstr(minimum, digits),
        "minimum_index": min(range(len(weights)), key=lambda index: weights[index]),
        "all_weights_positive": all(value > 0 for value in weights),
        "maximum_scaled_equality_residual_abs": mp.nstr(
            max(abs(value) for value in equality), digits
        ),
        "weights": [mp.nstr(value, digits) for value in weights],
    }


def main() -> int:
    args = _parse()
    output = _path(args.output)
    if output.exists():
        raise FileExistsError(output)
    preparation_path = _path(args.preparation)
    preparation = _load(preparation_path)
    specification = _load(_path(preparation["specification"]["path"]))
    point_key = (
        "design_points_normalized_ordered"
        if args.point_kind == "design"
        else "held_out_points_normalized_ordered"
    )
    points = specification["parameter_geometry"][point_key]
    if not 0 <= args.point_index < len(points):
        raise ValueError("point index is outside the frozen ordered point set")
    center = tf.constant(specification["target"]["center_theta"], DTYPE)
    scale = tf.constant(
        specification["parameter_geometry"]["scale_diagonal"], DTYPE
    )
    theta = center + scale * tf.constant(points[args.point_index], DTYPE)
    spec = model.make_scalar_sv_spec(preparation["row_id"])
    target = tf.constant(preparation["target"]["target_observations"], DTYPE)
    flow = tf.constant(preparation["target"]["flow_observations"], DTYPE)
    nodes = tf.constant(preparation["teacher_quadrature"]["nodes"], DTYPE)
    weights = tf.constant(preparation["teacher_quadrature"]["weights"], DTYPE)
    grid = tf.constant(preparation["continuation_quadrature"]["points"], DTYPE)
    grid_weights = tf.constant(
        preparation["continuation_quadrature"]["weights"], DTYPE
    )
    lookahead = int(preparation["feature_contract"]["lookahead_steps"])
    parents, parent_log_weights, standard_nodes, standard_log_weights = model.initial_rule(
        spec, theta, nodes, weights
    )
    weakest: dict[str, Any] | None = None
    time_steps = int(preparation["target"]["time_steps"])
    for time_index in range(time_steps - 1):
        teacher = model._teacher_step(
            spec,
            theta,
            parents,
            parent_log_weights,
            standard_nodes,
            standard_log_weights,
            target[time_index],
            flow[time_index],
            time_index,
        )
        features = model._features(
            spec,
            theta,
            teacher["particles"],
            target[time_index + 1 : time_index + 1 + lookahead],
            grid,
            grid_weights,
            first_future_time_index=time_index + 1,
        )
        active = tf.constant(preparation["active_indices"][time_index], tf.int32)
        row_scale = tf.constant(preparation["row_scales"][time_index], DTYPE)
        reference = tf.constant(preparation["reference_weights"][time_index], DTYPE)
        projection = model.tp._contract_e_tp_diagonal_kkt_forward_core(
            teacher["particles"],
            teacher["log_unnormalized_weights"],
            features,
            active,
            row_scale,
            reference,
        )
        if not bool(projection["valid_chart"].numpy()):
            raise RuntimeError(f"float64 reconstruction invalid at time {time_index}")
        minimum = float(projection["minimum_weight"].numpy())
        candidate = {
            "time_index_zero_based": time_index,
            "minimum_weight_float64": minimum,
            "minimum_index": int(tf.argmin(projection["student_weights"]).numpy()),
            "scaled_matrix": projection["scaled_matrix"].numpy().tolist(),
            "scaled_target": projection["scaled_target"].numpy().tolist(),
            "reference_weights": reference.numpy().tolist(),
            "student_weights_float64": projection["student_weights"].numpy().tolist(),
            "scaled_relative_residual_float64": float(
                projection["scaled_relative_residual"].numpy()
            ),
        }
        if weakest is None or (minimum, time_index) < (
            weakest["minimum_weight_float64"],
            weakest["time_index_zero_based"],
        ):
            weakest = candidate
        parents = tf.reshape(projection["student_points"], [-1])
        parent_log_weights = tf.math.log(projection["student_weights"])
    if weakest is None:
        raise RuntimeError("audit requires at least one projection")
    precision_rows = [
        _solve_at_precision(
            weakest["scaled_matrix"],
            weakest["scaled_target"],
            weakest["reference_weights"],
            digits,
        )
        for digits in (50, 100, 200)
    ]
    sign_stable = all(row["all_weights_positive"] for row in precision_rows)
    minimum_indices_stable = all(
        row["minimum_index"] == weakest["minimum_index"] for row in precision_rows
    )
    payload = {
        "schema": "bayesfilter.contract_e_tp.actual_sv_overcomplete_high_precision_audit.v1",
        "status": "PASS_SIGN_STABLE" if sign_stable else "FAIL_SIGN_UNSTABLE",
        "preparation": {"path": str(args.preparation), "sha256": _sha256(preparation_path)},
        "point_kind": args.point_kind,
        "point_index": args.point_index,
        "normalized_point": points[args.point_index],
        "theta": theta.numpy().tolist(),
        "weakest_float64_solve": weakest,
        "precision_ladder": precision_rows,
        "sign_stable": sign_stable,
        "minimum_indices_stable": minimum_indices_stable,
        "role": "corroborating_recomputation_of_saved_float64_inputs_not_interval_proof",
        "backend": "mpmath_independent_high_precision_audit_exception",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "weakest": weakest, "precision_ladder": precision_rows}, indent=2))
    return 0 if sign_stable and minimum_indices_stable else 2


if __name__ == "__main__":
    raise SystemExit(main())
