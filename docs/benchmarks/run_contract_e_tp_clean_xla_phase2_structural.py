#!/usr/bin/env python3
"""Run the clean-XLA structural fixture on CPU-hidden or trusted GPU/XLA."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--time-steps", type=int, choices=(1, 2, 5), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


ARGS = _parse()
if ARGS.device == "cpu":
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("CPU mode requires CUDA_VISIBLE_DEVICES=-1 before Python")

import tensorflow as tf


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.highdim import ledh_contract_e_tp_structural_tf as structural
from bayesfilter.testing.contract_e_tp_clean_xla_guardrails import inventory_graph_def


DTYPE = tf.float64
PLAN = (
    "docs/plans/bayesfilter-contract-e-tp-all-model-clean-xla-validation-"
    "phase2-structural-support-regression-subplan-2026-07-15.md"
)
RESULT = (
    "docs/plans/bayesfilter-contract-e-tp-all-model-clean-xla-validation-"
    "phase2-structural-support-regression-result-2026-07-15.md"
)


def _path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_value(value: tf.Tensor) -> Any:
    result = value.numpy()
    return result.tolist() if hasattr(result, "tolist") else result


def _summary(result: dict[str, tf.Tensor]) -> dict[str, Any]:
    return {
        "valid": bool(result["valid"].numpy()),
        "objective": float(result["objective"].numpy()),
        "score": _json_value(result["score"]),
        "objective_all_finite": bool(tf.math.is_finite(result["objective"]).numpy()),
        "score_all_finite": bool(tf.reduce_all(tf.math.is_finite(result["score"])).numpy()),
        "final_parents_all_finite": bool(tf.reduce_all(tf.math.is_finite(result["final_parents"])).numpy()),
        "final_weights_all_finite": bool(tf.reduce_all(tf.math.is_finite(result["final_weights"])).numpy()),
        "max_abs_residual": float(tf.reduce_max(tf.abs(result["residual_history"])).numpy()),
        "max_value_bound": float(tf.reduce_max(result["value_bound_history"]).numpy()),
        "max_abs_residual_tangent": float(tf.reduce_max(tf.abs(result["residual_jacobian"])).numpy()),
        "max_tangent_bound": float(tf.reduce_max(result["tangent_bound"]).numpy()),
        "minimum_student_weight": float(tf.reduce_min(result["minimum_weight_history"]).numpy()),
        "maximum_feature_residual": float(tf.reduce_max(tf.abs(result["feature_residual_history"])).numpy()),
        "kernel_match": bool(tf.reduce_all(result["kernel_match_history"]).numpy()),
        "tangent_valid": bool(result["tangent_valid"].numpy()),
        "expansion_valid": bool(result["expansion_valid"].numpy()),
    }


def main() -> int:
    preparation_path = _path(ARGS.preparation)
    output = _path(ARGS.output)
    if output.exists():
        raise FileExistsError(output)
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    fixture = preparation["fixture"]
    steps = preparation["steps"][: ARGS.time_steps]
    theta = tf.constant(fixture["theta"], DTYPE)
    evaluate = structural.make_structural_fixture_recursive_tf(
        tf.constant(fixture["initial_parents"], DTYPE),
        tf.constant(fixture["initial_weights"], DTYPE),
        tf.constant(fixture["innovations"], DTYPE),
        tf.constant(fixture["innovation_weights"], DTYPE),
        tf.constant([step["active_indices"] for step in steps], tf.int32),
        tf.constant([step["row_scales"] for step in steps], DTYPE),
        jit_compile=True,
    )
    graph = inventory_graph_def(evaluate.get_concrete_function().graph.as_graph_def())
    started = time.perf_counter()
    valid = evaluate(theta, tf.constant(0.0, DTYPE))
    compile_and_first = time.perf_counter() - started
    started = time.perf_counter()
    valid_warm = evaluate(theta, tf.constant(0.0, DTYPE))
    warm_seconds = time.perf_counter() - started
    started = time.perf_counter()
    invalid = evaluate(theta, tf.constant(fixture["support_perturbation"], DTYPE))
    invalid_seconds = time.perf_counter() - started
    valid_summary = _summary(valid)
    invalid_summary = _summary(invalid)
    gates = {
        "valid_happy_path": valid_summary["valid"] and valid_summary["objective_all_finite"] and valid_summary["score_all_finite"],
        "support_value_guard": valid_summary["max_abs_residual"] <= valid_summary["max_value_bound"],
        "support_tangent_guard": valid_summary["max_abs_residual_tangent"] <= valid_summary["max_tangent_bound"],
        "fixed_positive_chart": valid_summary["minimum_student_weight"] > 0.0,
        "functional_loop_present": graph["functional_loop_count"] >= 1,
        "same_factory_invalid_false": not invalid_summary["valid"],
        "same_factory_invalid_poisoned": not any(
            invalid_summary[key]
            for key in (
                "objective_all_finite",
                "score_all_finite",
                "final_parents_all_finite",
                "final_weights_all_finite",
            )
        ),
        "off_support_separation": float(fixture["support_perturbation"]) >= 1.0e6 * valid_summary["max_value_bound"],
        "warm_replay_equal": all(
            bool(tf.reduce_all(tf.equal(valid[name], valid_warm[name])).numpy())
            for name in ("objective", "score", "valid", "final_parents", "final_weights")
        ),
    }
    if ARGS.device == "gpu":
        logical_gpus = tf.config.list_logical_devices("GPU")
        gates["gpu_visible"] = bool(logical_gpus)
    if not all(gates.values()):
        raise RuntimeError(f"structural fixture gate failed: {gates}")
    payload = {
        "schema": "contract_e_tp.clean_xla_phase2_structural_run.v1",
        "status": "PASS_STRUCTURAL_FIXTURE_COMPILED_SUPPORT_AND_TANGENT",
        "device": ARGS.device,
        "time_steps": ARGS.time_steps,
        "same_concrete_factory_for_valid_and_invalid": True,
        "valid": valid_summary,
        "invalid": invalid_summary,
        "graph": graph,
        "timing_seconds": {
            "compile_and_first": compile_and_first,
            "warm": warm_seconds,
            "same_factory_invalid": invalid_seconds,
        },
        "gates": gates,
        "run_manifest": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "git_status_short": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
            "command": " ".join(sys.argv),
            "python": sys.version,
            "platform": platform.platform(),
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "N/A"),
            "tensorflow_version": tf.__version__,
            "physical_devices": [str(device) for device in tf.config.list_physical_devices()],
            "visible_gpus": [str(device) for device in tf.config.list_logical_devices("GPU")],
            "device_trust_basis": "owner_designated_managed_session_visible_gpu_trusted" if ARGS.device == "gpu" else "cpu_hidden_reference_debug",
            "xla": True,
            "tf32": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
            "dtype": "float64",
            "seed": "N/A deterministic fixture",
            "preparation_path": str(preparation_path.relative_to(ROOT)),
            "preparation_sha256": _sha256(preparation_path),
            "output_path": str(output.relative_to(ROOT)),
            "plan": PLAN,
            "result": RESULT,
        },
        "nonclaims": [
            "not a DSGE, NAWM, or SIR result",
            "not general structural filtering accuracy",
            "not canonical, default, HMC, or leaderboard readiness",
            "not TensorFlow tanh kernel accuracy relative to mathematical tanh",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "device": ARGS.device, "gates": gates, "timing": payload["timing_seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
