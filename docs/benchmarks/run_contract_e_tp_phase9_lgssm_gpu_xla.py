#!/usr/bin/env python3
"""Run one trusted-GPU XLA LGSSM Contract E--TP scaling rung."""

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


os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

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


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--cpu-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(values: tf.Tensor) -> bool:
    return bool(tf.reduce_all(tf.math.is_finite(values)).numpy())


def main() -> None:
    args = _parse()
    output = _path(args.output)
    if output.exists():
        raise FileExistsError(output)
    preparation_path = _path(args.preparation)
    cpu_path = _path(args.cpu_result)
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    cpu = json.loads(cpu_path.read_text(encoding="utf-8"))
    if preparation["algorithm_id"] != model.ALGORITHM_ID:
        raise ValueError("wrong preparation algorithm")
    if preparation["feature_mode"] != "finite_lookahead":
        raise ValueError("Phase 9 controls only finite_lookahead LGSSM")
    if int(preparation["lookahead_steps"]) != 8:
        raise ValueError("Phase 9 controls only lookahead 8")
    if cpu["preparation"]["sha256"] != _sha256(preparation_path):
        raise ValueError("CPU result/preparation hash mismatch")
    time_steps = int(preparation["target"]["time_steps"])
    observations = tf.convert_to_tensor(
        _lgssm_dataset(81100)["observations"][:time_steps], DTYPE
    )
    nodes = tf.constant(preparation["quadrature"]["nodes"], DTYPE)
    weights = tf.constant(preparation["quadrature"]["weights"], DTYPE)
    active_indices = tf.constant(preparation["active_indices"], tf.int32)
    row_scales = tf.constant(preparation["row_scales"], DTYPE)

    tf.config.experimental.enable_tensor_float_32_execution(True)
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("Phase 9 requires a visible trusted GPU")
    tf.config.experimental.reset_memory_stats("GPU:0")
    evaluate = model.make_contract_e_tp_lgssm_score_informed_recursive_tf(
        observations,
        nodes,
        weights,
        active_indices,
        row_scales,
        feature_mode="finite_lookahead",
        lookahead_steps=8,
        jit_compile=True,
    )
    compile_started = time.perf_counter()
    first = evaluate(THETA)
    _ = first["score"].numpy()
    compile_seconds = time.perf_counter() - compile_started
    warm_started = time.perf_counter()
    result = evaluate(THETA)
    score = result["score"]
    _ = score.numpy()
    warm_seconds = time.perf_counter() - warm_started
    device = result["objective"].backing_device
    if "GPU" not in device.upper():
        raise RuntimeError(f"compiled output is not GPU-backed: {device}")
    chart_pass = bool(tf.reduce_all(result["valid_history"]).numpy())
    finite = _finite(result["objective"]) and _finite(score)
    if not chart_pass or not finite:
        raise RuntimeError("GPU/XLA result failed finite/chart gate")
    cpu_value = float(cpu["value"]["contract_e_tp"])
    cpu_score = [float(item) for item in cpu["score"]["contract_e_tp"]]
    value = float(result["objective"].numpy())
    score_values = [float(item) for item in score.numpy()]
    value_difference = value - cpu_value
    score_difference = [left - right for left, right in zip(score_values, cpu_score)]
    if not math.isfinite(value_difference) or not all(
        math.isfinite(item) for item in score_difference
    ):
        raise RuntimeError("CPU/GPU differences are nonfinite")
    memory = tf.config.experimental.get_memory_info("GPU:0")
    concrete = evaluate.get_concrete_function(THETA)
    operations = [node.op for node in concrete.graph.as_graph_def().node]
    payload = {
        "schema": "bayesfilter.contract_e_tp.phase9_lgssm_gpu_xla.v1",
        "status": "PASS_GPU_XLA_FLOAT64",
        "algorithm_id": model.ALGORITHM_ID,
        "target": {
            "row_id": "benchmark_lgssm_exact_oracle_m3_T50",
            "time_steps": time_steps,
            "theta": THETA.numpy().tolist(),
            "feature_mode": "finite_lookahead",
            "lookahead_steps": 8,
        },
        "preparation": {
            "path": str(preparation_path.relative_to(ROOT)),
            "sha256": _sha256(preparation_path),
        },
        "cpu_reference": {
            "path": str(cpu_path.relative_to(ROOT)),
            "sha256": _sha256(cpu_path),
            "value": cpu_value,
            "score": cpu_score,
        },
        "gpu": {
            "value": value,
            "score": score_values,
            "value_difference_from_cpu": value_difference,
            "score_difference_from_cpu": score_difference,
            "chart_valid": chart_pass,
            "minimum_weight": float(tf.reduce_min(result["minimum_weight_history"]).numpy()),
            "maximum_condition_number": float(tf.reduce_max(result["condition_number_history"]).numpy()),
            "maximum_feature_residual_abs": float(tf.reduce_max(tf.abs(result["feature_residual_history"])).numpy()),
        },
        "execution": {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "tensorflow_version": tf.__version__,
            "visible_gpu": gpus[0].name,
            "output_device": device,
            "dtype": DTYPE.name,
            "tf32_enabled": tf.config.experimental.tensor_float_32_execution_enabled(),
            "tf32_relevant_to_dtype": False,
            "jit_compile": True,
            "compile_plus_first_execution_seconds": compile_seconds,
            "warmed_execution_seconds": warm_seconds,
            "gpu_allocator_current_bytes": int(memory["current"]),
            "gpu_allocator_peak_bytes": int(memory["peak"]),
            "graph_operation_count": len(operations),
            "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
            "command": " ".join(sys.argv),
        },
        "capability_gaps": {
            "float32_tf32": "blocked_dtype_generic_refactor_required",
            "scalar_sv_recursive_xla": "blocked_xla_factory_not_implemented",
            "predator_prey_recursive_xla": "blocked_xla_factory_not_implemented",
        },
        "nonclaims": [
            "float64 GPU/XLA engineering evidence only",
            "not float32/TF32 production-target readiness",
            "not nonlinear GPU readiness",
            "not HMC, canonical, default, or leaderboard admission",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "target": payload["target"], "execution": payload["execution"]}, indent=2))


if __name__ == "__main__":
    main()
