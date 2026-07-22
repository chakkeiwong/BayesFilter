#!/usr/bin/env python3
"""Compare the XLA manual score with eager TensorFlow forward autodiff."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
    raise RuntimeError("autodiff audit requires deliberate CPU-only hiding")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf  # noqa: E402

from bayesfilter.highdim import ledh_contract_e_tp_scalar_sv_tf as model  # noqa: E402


DTYPE = tf.float64


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--manual-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    args = _parse()
    output = _path(args.output)
    if output.exists():
        raise FileExistsError(output)
    preparation_path = _path(args.preparation)
    manual_path = _path(args.manual_result)
    preparation = _load(preparation_path)
    manual = _load(manual_path)
    center = next(row for row in manual["rows"] if row["name"] == "center")
    if manual["preparation"]["sha256"] != _sha256(preparation_path):
        raise ValueError("manual result and preparation identities differ")
    time_steps = int(preparation["target"]["time_steps"])
    capacity = int(preparation["chart_contract"]["anchor_count"])
    theta = tf.constant(preparation["target"]["theta"], DTYPE)
    core_arguments = (
        model.make_scalar_sv_spec(preparation["row_id"]),
        theta,
        tf.constant(preparation["target"]["target_observations"], DTYPE),
        tf.constant(preparation["target"]["flow_observations"], DTYPE),
        tf.constant(preparation["teacher_quadrature"]["nodes"], DTYPE),
        tf.constant(preparation["teacher_quadrature"]["weights"], DTYPE),
        tf.reshape(
            tf.constant(preparation["active_indices"], tf.int32),
            [time_steps - 1, capacity],
        ),
        tf.reshape(
            tf.constant(preparation["row_scales"], DTYPE),
            [time_steps - 1, model.FEATURE_COUNT],
        ),
        tf.reshape(
            tf.constant(preparation["reference_weights"], DTYPE),
            [time_steps - 1, capacity],
        ),
        tf.constant(preparation["continuation_quadrature"]["points"], DTYPE),
        tf.constant(preparation["continuation_quadrature"]["weights"], DTYPE),
    )
    started = time.perf_counter()
    forward_score = []
    objective = None
    for direction_index in range(2):
        direction = tf.one_hot(direction_index, 2, dtype=DTYPE)
        with tf.autodiff.ForwardAccumulator(theta, direction) as accumulator:
            result = model.contract_e_tp_actual_sv_overcomplete_loop_core(
                *core_arguments,
                lookahead_steps=int(
                    preparation["feature_contract"]["lookahead_steps"]
                ),
            )
        objective = result["objective"]
        forward_score.append(accumulator.jvp(objective))
    score = tf.stack(forward_score)
    manual_score = tf.constant(center["score_manual"]["values"], DTYPE)
    absolute = tf.abs(score - manual_score)
    relative = tf.linalg.norm(absolute) / tf.maximum(
        tf.linalg.norm(manual_score), tf.constant(1.0, DTYPE)
    )
    relative_tolerance = float(
        tf.sqrt(tf.constant(tf.experimental.numpy.finfo(tf.float64.as_numpy_dtype).eps, DTYPE)).numpy()
    )
    objective_matches = float(objective.numpy()) == float(
        center["objective"]["values"]
    )
    objective_absolute_difference = abs(
        float(objective.numpy()) - float(center["objective"]["values"])
    )
    all_finite = bool(tf.reduce_all(tf.math.is_finite(score)).numpy())
    score_agrees = float(relative.numpy()) <= relative_tolerance
    payload = {
        "schema": "bayesfilter.contract_e_tp.actual_sv_overcomplete_eager_forward_ad_audit.v1",
        "status": (
            "PASS_FORWARD_AD" if all_finite and score_agrees else "FAIL_FORWARD_AD"
        ),
        "preparation": {"path": str(args.preparation), "sha256": _sha256(preparation_path)},
        "manual_result": {"path": str(args.manual_result), "sha256": _sha256(manual_path)},
        "time_steps": time_steps,
        "theta": theta.numpy().tolist(),
        "objective_eager": float(objective.numpy()),
        "objective_xla_manual_result": center["objective"]["values"],
        "objective_bitwise_equal_as_float": objective_matches,
        "objective_absolute_difference": objective_absolute_difference,
        "objective_comparison_role": "descriptive_cross_eager_xla_roundoff_only",
        "score_forward_ad": score.numpy().tolist(),
        "score_manual": manual_score.numpy().tolist(),
        "maximum_absolute_difference": float(tf.reduce_max(absolute).numpy()),
        "relative_difference": float(relative.numpy()),
        "relative_tolerance": relative_tolerance,
        "relative_tolerance_provenance": "sqrt(float64 machine epsilon)",
        "score_agrees": score_agrees,
        "all_forward_ad_finite": all_finite,
        "wall_time_seconds": time.perf_counter() - started,
        "execution": {
            "backend": "TensorFlow_eager_ForwardAccumulator_CPU_reference_exception",
            "jit_compile": False,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "oracle_limitations": {
            "reverse_mode": "NaN from extreme tiny-weight adjoint path at T>=100",
            "forward_ad_tf_function_xla": "nonconstant transpose permutation in transformed nested loop",
            "forward_ad_tf_function_nonjit": "TensorList shape invariant failure in transformed nested loop",
        },
        "nonclaims": [
            "eager CPU reference is not the default production execution path",
            "autodiff agreement does not establish scientific target equivalence"
        ]
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["status"] == "PASS_FORWARD_AD" else 2


if __name__ == "__main__":
    raise SystemExit(main())
