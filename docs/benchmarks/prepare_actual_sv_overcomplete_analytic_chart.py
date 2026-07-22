#!/usr/bin/env python3
"""Prepare one TensorFlow-only Actual-SV overcomplete Pearson chart."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
    raise RuntimeError("preparation requires deliberate CPU-only CUDA hiding")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf  # noqa: E402

from bayesfilter.highdim import ledh_contract_e_tp_scalar_sv_tf as model  # noqa: E402
from bayesfilter.highdim.ledh_forward_contract import ACTUAL_SV_ROW_ID  # noqa: E402


DTYPE = tf.float64
SCHEMA = "bayesfilter.contract_e_tp.scalar_sv_overcomplete_preparation.v3"
PLAN = "docs/plans/bayesfilter-actual-sv-overcomplete-analytic-chart-repair-plan-2026-07-17.md"
SPECIFICATION = (
    "docs/benchmarks/artifacts/actual_sv_overcomplete_analytic_chart_repair_20260717/"
    "phase-01-specification/design_specification.json"
)
PREDECESSOR = (
    "docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/"
    "phase-04/scalar-sv/attempt-01-preparation-20260715/actual_t1000_preparation.json"
)


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-steps", type=int, required=True)
    parser.add_argument("--anchor-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_sha256(value: tf.Tensor) -> str:
    return hashlib.sha256(tf.io.serialize_tensor(value).numpy()).hexdigest()


def _write_fresh(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse()
    output = _path(args.output)
    specification_path = _path(SPECIFICATION)
    predecessor_path = _path(PREDECESSOR)
    specification = _load(specification_path)
    predecessor = _load(predecessor_path)
    frozen = specification["finite_program"]
    if args.time_steps not in (1, 2, 10, 100, 1000):
        raise ValueError("time_steps must be one of 1,2,10,100,1000")
    ladder = frozen["capacity_ladder"]
    if not ladder["minimum"] <= args.anchor_count <= ladder["maximum"]:
        raise ValueError("anchor_count is outside the frozen K ladder")
    if predecessor["row_id"] != ACTUAL_SV_ROW_ID:
        raise ValueError("predecessor is not the frozen Actual-SV row")
    if _sha256(predecessor_path) != specification["predecessor"]["preparation_sha256"]:
        raise ValueError("predecessor preparation hash differs from Phase 1")
    if predecessor["target"]["time_steps"] < args.time_steps:
        raise ValueError("predecessor horizon is shorter than requested")

    started = time.perf_counter()
    spec = model.make_scalar_sv_spec(ACTUAL_SV_ROW_ID)
    theta = tf.constant(specification["target"]["center_theta"], DTYPE)
    target = tf.constant(
        predecessor["target"]["target_observations"][: args.time_steps], DTYPE
    )
    flow = tf.constant(
        predecessor["target"]["flow_observations"][: args.time_steps], DTYPE
    )
    nodes = tf.constant(predecessor["teacher_quadrature"]["nodes"], DTYPE)
    weights = tf.constant(predecessor["teacher_quadrature"]["weights"], DTYPE)
    grid = tf.constant(predecessor["continuation_quadrature"]["points"], DTYPE)
    grid_weights = tf.constant(
        predecessor["continuation_quadrature"]["weights"], DTYPE
    )
    if int(nodes.shape[0]) != frozen["teacher_quadrature"]["order"]:
        raise ValueError("predecessor teacher order differs from Phase 1")
    if int(grid.shape[0]) != frozen["continuation_quadrature"]["order"]:
        raise ValueError("predecessor continuation order differs from Phase 1")

    parents, parent_log_weights, standard_nodes, standard_log_weights = model.initial_rule(
        spec, theta, nodes, weights
    )
    active_rows: list[list[int]] = []
    scale_rows: list[list[float]] = []
    reference_rows: list[list[float]] = []
    voronoi_rows: list[list[float]] = []
    chart_rows: list[dict[str, Any]] = []
    for time_index in range(args.time_steps - 1):
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
        window = target[time_index + 1 : time_index + 1 + frozen["requested_lookaheads"][0]]
        features = model._features(
            spec,
            theta,
            teacher["particles"],
            window,
            grid,
            grid_weights,
            first_future_time_index=time_index + 1,
        )
        prepared = model.prepare_actual_sv_overcomplete_chart_step(
            teacher["particles"],
            teacher["log_unnormalized_weights"],
            features,
            args.anchor_count,
        )
        valid = bool(prepared["preparation_valid"].numpy())
        active = prepared["active_indices"].numpy().tolist()
        reference = prepared["reference_weights"].numpy().tolist()
        scale = prepared["row_scale"].numpy().tolist()
        voronoi = prepared["voronoi_weights"].numpy().tolist()
        active_rows.append(active)
        reference_rows.append(reference)
        scale_rows.append(scale)
        voronoi_rows.append(voronoi)
        chart_rows.append(
            {
                "time_index_zero_based": time_index,
                "teacher_count": int(teacher["particles"].shape[0]),
                "active_indices": active,
                "voronoi_weights": voronoi,
                "reference_weights": reference,
                "preparation_valid": valid,
                "minimum_voronoi_weight": float(
                    tf.reduce_min(prepared["voronoi_weights"]).numpy()
                ),
                "minimum_reference_weight": float(
                    tf.reduce_min(prepared["reference_weights"]).numpy()
                ),
                "minimum_projected_weight": float(prepared["minimum_weight"].numpy()),
                "matrix_condition_number": float(
                    prepared["matrix_condition_number"].numpy()
                ),
                "gram_condition_number": float(
                    prepared["gram_condition_number"].numpy()
                ),
                "scaled_relative_residual": float(
                    prepared["scaled_relative_residual"].numpy()
                ),
                "maximum_feature_residual_abs": float(
                    tf.reduce_max(tf.abs(prepared["feature_residual"])).numpy()
                ),
            }
        )
        if not valid:
            break
        parents = tf.reshape(prepared["student_points"], [args.anchor_count])
        parent_log_weights = tf.math.log(prepared["student_weights"])

    all_valid = len(chart_rows) == args.time_steps - 1 and all(
        row["preparation_valid"] for row in chart_rows
    )
    payload = {
        "schema": SCHEMA,
        "status": "PASS_CENTER_PREPARATION" if all_valid else "FAIL_CENTER_PREPARATION",
        "algorithm_id": model.ALGORITHM_ID,
        "route_id": "actual_sv_overcomplete_pearson_fixed_reference_v1",
        "row_id": ACTUAL_SV_ROW_ID,
        "plan": PLAN,
        "specification": {
            "path": SPECIFICATION,
            "sha256": _sha256(specification_path),
        },
        "predecessor": {"path": PREDECESSOR, "sha256": _sha256(predecessor_path)},
        "target": {
            "theta": theta.numpy().tolist(),
            "parameter_names": specification["target"]["parameter_names"],
            "time_steps": args.time_steps,
            "target_observations": target.numpy().tolist(),
            "flow_observations": flow.numpy().tolist(),
            "target_observations_sha256": _tensor_sha256(target),
            "flow_observations_sha256": _tensor_sha256(flow),
            "transition_before_first_observation": False,
        },
        "feature_contract": {
            "names": frozen["features"],
            "lookahead_steps": frozen["requested_lookaheads"][0],
            "feature_count": frozen["feature_count"],
        },
        "chart_contract": {
            "anchor_count": args.anchor_count,
            "constant_across_time": True,
            "anchor_strategy": "tensorflow_weighted_quantile_voronoi",
            "reference_strategy": "tensorflow_pearson_equality_projection",
            "precision": "diagonal_inverse_reference_not_materialized",
            "runtime_selection": False,
        },
        "teacher_quadrature": {
            "family": "gauss_hermite_standard_normal",
            "order": int(nodes.shape[0]),
            "nodes": nodes.numpy().tolist(),
            "weights": weights.numpy().tolist(),
        },
        "continuation_quadrature": {
            "family": "legendre_fixed_interval",
            "order": int(grid.shape[0]),
            "radius": frozen["continuation_quadrature"]["radius"],
            "points": grid.numpy().tolist(),
            "weights": grid_weights.numpy().tolist(),
        },
        "active_indices": active_rows,
        "row_scales": scale_rows,
        "reference_weights": reference_rows,
        "voronoi_weights": voronoi_rows,
        "charts": chart_rows,
        "summary": {
            "prepared_chart_count": len(chart_rows),
            "all_preparation_valid": all_valid,
            "first_invalid_time": next(
                (row["time_index_zero_based"] for row in chart_rows if not row["preparation_valid"]),
                None,
            ),
            "minimum_reference_weight": min(
                (row["minimum_reference_weight"] for row in chart_rows), default=None
            ),
            "minimum_projected_weight": min(
                (row["minimum_projected_weight"] for row in chart_rows), default=None
            ),
            "maximum_gram_condition_number": max(
                (row["gram_condition_number"] for row in chart_rows), default=None
            ),
        },
        "execution": {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "git_dirty": bool(
                subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True)
            ),
            "backend": "TensorFlow_float64_CPU_candidate_preparation",
            "tensorflow_version": tf.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "command": " ".join(sys.argv),
            "wall_time_seconds": time.perf_counter() - started,
        },
        "nonclaims": [
            "not off-center chart validity",
            "not derivative correctness",
            "not scientific score equivalence",
            "not GPU or HMC readiness",
        ],
    }
    _write_fresh(output, payload)
    print(json.dumps({"status": payload["status"], "summary": payload["summary"]}, indent=2))
    return 0 if all_valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
