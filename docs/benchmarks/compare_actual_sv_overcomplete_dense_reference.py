#!/usr/bin/env python3
"""Compare a Phase 5 Actual-SV result with adjacent dense references."""

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
from typing import Any


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import numpy as np
import tensorflow as tf
from numpy.polynomial.legendre import leggauss


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.highdim import ledh_contract_e_tp_scalar_sv_tf as model
from bayesfilter.highdim.ledh_forward_contract import ACTUAL_SV_ROW_ID


DTYPE = tf.float64
PREPARATION_SCHEMA = "bayesfilter.contract_e_tp.scalar_sv_overcomplete_preparation.v3"
CANDIDATE_SCHEMA = "bayesfilter.contract_e_tp.actual_sv_overcomplete_result.v1"
RESULT_SCHEMA = "bayesfilter.contract_e_tp.actual_sv_overcomplete_dense_comparison.v1"
PARAMETER_NAMES = ("gamma_unconstrained", "log_beta")


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--candidate-result", type=Path, required=True)
    parser.add_argument("--reference-orders", default="129,257")
    parser.add_argument("--reference-radius", type=float, default=10.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _path(value: Path | str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_sha256(value: tf.Tensor) -> str:
    return hashlib.sha256(tf.io.serialize_tensor(value).numpy()).hexdigest()


def _legendre_rule(order: int, radius: float) -> tuple[tf.Tensor, tf.Tensor]:
    nodes, weights = leggauss(order)
    return (
        tf.constant(radius * nodes, DTYPE),
        tf.constant(radius * weights, DTYPE),
    )


def _value_and_score(
    callable_: Any, theta: tf.Tensor
) -> tuple[dict[str, tf.Tensor], tf.Tensor]:
    with tf.GradientTape() as tape:
        tape.watch(theta)
        result = callable_(theta)
        objective = result["objective"]
    return result, tape.gradient(objective, theta)


def _symmetric_relative(left: float, right: float) -> float:
    denominator = abs(left) + abs(right)
    if denominator == 0.0:
        return 0.0
    return 2.0 * abs(left - right) / denominator


def _componentwise_comparison(
    candidate: list[float], reference: list[float]
) -> dict[str, Any]:
    absolute = [abs(left - right) for left, right in zip(candidate, reference)]
    symmetric = [
        _symmetric_relative(left, right)
        for left, right in zip(candidate, reference)
    ]
    candidate_norm = math.sqrt(sum(value * value for value in candidate))
    reference_norm = math.sqrt(sum(value * value for value in reference))
    difference_norm = math.sqrt(
        sum((left - right) ** 2 for left, right in zip(candidate, reference))
    )
    norm_denominator = candidate_norm + reference_norm
    return {
        "difference": [
            left - right for left, right in zip(candidate, reference)
        ],
        "absolute_difference": absolute,
        "componentwise_symmetric_relative_difference": symmetric,
        "vector_norm_symmetric_relative_difference": (
            0.0 if norm_denominator == 0.0 else 2.0 * difference_norm / norm_denominator
        ),
        "sign_reversal": [
            math.copysign(1.0, left) != math.copysign(1.0, right)
            if left != 0.0 and right != 0.0
            else left != right
            for left, right in zip(candidate, reference)
        ],
    }


def _center_row(candidate: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in candidate.get("rows", []) if row.get("name") == "center"]
    if len(rows) != 1:
        raise ValueError("candidate result must contain exactly one center row")
    return rows[0]


def _validate_sources(
    preparation_path: Path,
    preparation: dict[str, Any],
    candidate_path: Path,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    if preparation.get("schema") != PREPARATION_SCHEMA:
        raise ValueError("comparison requires a v3 overcomplete preparation")
    if preparation.get("row_id") != ACTUAL_SV_ROW_ID:
        raise ValueError("comparison is restricted to the Actual-SV row")
    if preparation.get("route_id") != candidate.get("route_id"):
        raise ValueError("candidate and preparation route identities differ")
    if not preparation.get("summary", {}).get("all_preparation_valid", False):
        raise ValueError("preparation is not valid")
    if candidate.get("schema") != CANDIDATE_SCHEMA:
        raise ValueError("candidate result schema mismatch")
    if candidate.get("status") != "PASS_FINITE_PROGRAM":
        raise ValueError("candidate result did not pass its finite program")
    if candidate.get("evaluation_mode") != "score":
        raise ValueError("candidate result must contain a manual score")
    if candidate.get("point_set") != "fd":
        raise ValueError("candidate result must be the Phase 5 FD artifact")
    if int(candidate.get("time_steps", -1)) != int(
        preparation["target"]["time_steps"]
    ):
        raise ValueError("candidate and preparation horizons differ")
    if int(candidate.get("capacity", -1)) != int(
        preparation["chart_contract"]["anchor_count"]
    ):
        raise ValueError("candidate and preparation capacities differ")
    if candidate.get("preparation", {}).get("sha256") != _sha256(preparation_path):
        raise ValueError("candidate does not bind the supplied preparation bytes")
    center = _center_row(candidate)
    if not center.get("valid", False):
        raise ValueError("candidate center row is invalid")
    if not candidate.get("finite_difference", {}).get("pass", False):
        raise ValueError("candidate Phase 5 finite-difference gate did not pass")
    return {
        "preparation_path": str(preparation_path.relative_to(ROOT)),
        "preparation_sha256": _sha256(preparation_path),
        "candidate_result_path": str(candidate_path.relative_to(ROOT)),
        "candidate_result_sha256": _sha256(candidate_path),
        "route_id": preparation["route_id"],
        "candidate_finite_program_status": candidate["status"],
        "candidate_fd_gate_pass": True,
    }


def main() -> None:
    args = _parse()
    started = time.perf_counter()
    output = _path(args.output)
    if output.exists():
        raise FileExistsError(output)
    preparation_path = _path(args.preparation)
    candidate_path = _path(args.candidate_result)
    preparation = _load(preparation_path)
    candidate = _load(candidate_path)
    source_validation = _validate_sources(
        preparation_path, preparation, candidate_path, candidate
    )

    reference_orders = tuple(
        int(value.strip()) for value in args.reference_orders.split(",")
    )
    if len(reference_orders) != 2 or reference_orders[0] >= reference_orders[1]:
        raise ValueError("Phase 6 requires exactly two increasing reference orders")
    if args.reference_radius <= 0.0 or not math.isfinite(args.reference_radius):
        raise ValueError("reference radius must be positive and finite")

    theta = tf.constant(preparation["target"]["theta"], DTYPE)
    observations = tf.constant(preparation["target"]["target_observations"], DTYPE)
    spec = model.make_scalar_sv_spec(ACTUAL_SV_ROW_ID)
    references: list[dict[str, Any]] = []
    for order in reference_orders:
        grid, grid_weights = _legendre_rule(order, args.reference_radius)

        def dense_reference(value: tf.Tensor) -> dict[str, tf.Tensor]:
            return model.scalar_sv_dense_reference_value(
                spec, value, observations, grid, grid_weights
            )

        reference_result, reference_score = _value_and_score(dense_reference, theta)
        value = float(reference_result["objective"].numpy())
        score = [float(item) for item in reference_score.numpy().tolist()]
        if not math.isfinite(value) or not all(math.isfinite(item) for item in score):
            raise FloatingPointError(f"dense reference order {order} is nonfinite")
        references.append(
            {
                "order": order,
                "radius": args.reference_radius,
                "value": value,
                "score": score,
                "target_observations_sha256": _tensor_sha256(observations),
                "role": "same_target_dense_filtering_reference",
            }
        )

    center = _center_row(candidate)
    candidate_value = float(center["objective"]["values"])
    candidate_score = [float(value) for value in center["score_manual"]["values"]]
    coarse, fine = references
    candidate_comparison = _componentwise_comparison(candidate_score, fine["score"])
    refinement_comparison = _componentwise_comparison(coarse["score"], fine["score"])

    git_status = subprocess.check_output(
        ["git", "status", "--short"], cwd=ROOT, text=True
    )
    payload = {
        "schema": RESULT_SCHEMA,
        "status": "COMPLETE_DESCRIPTIVE_DIAGNOSTIC",
        "plan": preparation["plan"],
        "phase": 6,
        "scope": "actual_sv_center_same_target_dense_reference_comparison",
        "source_validation": source_validation,
        "target": {
            "time_steps": int(preparation["target"]["time_steps"]),
            "theta": preparation["target"]["theta"],
            "parameter_names": list(PARAMETER_NAMES),
            "target_observations_sha256": _tensor_sha256(observations),
        },
        "candidate": {
            "value": candidate_value,
            "manual_total_score": candidate_score,
            "score_source": "Phase 5 explicit total manual JVP of the same finite scalar",
        },
        "dense_references": references,
        "adjacent_reference_refinement": {
            "coarse_order": coarse["order"],
            "fine_order": fine["order"],
            "value_difference_fine_minus_coarse": fine["value"] - coarse["value"],
            "value_absolute_difference": abs(fine["value"] - coarse["value"]),
            "value_symmetric_relative_difference": _symmetric_relative(
                fine["value"], coarse["value"]
            ),
            "score": refinement_comparison,
            "classification": "descriptive_numerical_refinement_not_error_bound",
        },
        "candidate_vs_finest_reference": {
            "reference_order": fine["order"],
            "value_difference_candidate_minus_reference": candidate_value - fine["value"],
            "value_absolute_difference": abs(candidate_value - fine["value"]),
            "value_symmetric_relative_difference": _symmetric_relative(
                candidate_value, fine["value"]
            ),
            "score": candidate_comparison,
            "equivalence_classification": "descriptive_only_no_justified_margin",
        },
        "decision": {
            "candidate_engineering_gate_source_pass": True,
            "dense_reference_finite": True,
            "scientific_equivalence_pass": None,
            "statistically_supported_ranking": False,
            "default_readiness": False,
            "next_action": "preserve the observed differences and proceed to the independent GPU/XLA engineering gate",
        },
        "execution": {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "git_dirty": bool(git_status),
            "git_status_sha256": hashlib.sha256(git_status.encode()).hexdigest(),
            "command": " ".join(sys.argv),
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "backend": "TensorFlow_float64_CPU_dense_reference_exception",
            "tensorflow_version": tf.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "physical_gpus": [device.name for device in tf.config.list_physical_devices("GPU")],
            "jit_compile": False,
            "wall_time_seconds": time.perf_counter() - started,
            "output": str(output.relative_to(ROOT)),
        },
        "nonclaims": [
            "no post-hoc score-equivalence threshold",
            "adjacent quadrature refinement is not an exact error bound",
            "no scientific equivalence or method ranking",
            "not HMC, canonical Contract E--Chol, default, or leaderboard readiness",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "time_steps": payload["target"]["time_steps"],
                "candidate_value": candidate_value,
                "reference_value": fine["value"],
                "value_difference": payload["candidate_vs_finest_reference"][
                    "value_difference_candidate_minus_reference"
                ],
                "candidate_score": candidate_score,
                "reference_score": fine["score"],
                "score_absolute_difference": candidate_comparison[
                    "absolute_difference"
                ],
                "score_symmetric_relative_difference": candidate_comparison[
                    "componentwise_symmetric_relative_difference"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
