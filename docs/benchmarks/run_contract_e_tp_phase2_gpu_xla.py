#!/usr/bin/env python3
"""Trusted GPU/XLA smoke for Contract E--TP streaming composition."""

from __future__ import annotations

import json
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

from bayesfilter.highdim import ledh_contract_e_tp_tf as tp


DTYPE = tf.float64
BLOCK_SIZE = 4
OUTPUT = Path(
    "docs/benchmarks/artifacts/contract_e_tp_all_models_2026_07_15/"
    "phase2_gpu_xla_smoke_repair1_20260715/phase2_gpu_xla_smoke.json"
)


def _block_program(
    sources: tuple[tf.Tensor, ...], start: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    parents, innovations, theta = sources
    flat_indices = tf.minimum(
        start + tf.range(BLOCK_SIZE, dtype=tf.int32),
        tf.size(parents) * tf.size(innovations) - 1,
    )
    parent_indices = flat_indices // tf.size(innovations)
    innovation_indices = flat_indices % tf.size(innovations)
    candidate = (
        tf.gather(parents, parent_indices)
        + tf.exp(theta[0]) * tf.gather(innovations, innovation_indices)
    )
    points = tf.stack([candidate, 0.2 * candidate + 0.05 * tf.square(candidate)], axis=1)
    log_weights = -0.5 * tf.square(candidate - theta[1]) + 0.1 * candidate
    features = tf.stack(
        [tf.ones_like(candidate), candidate, tf.square(candidate)], axis=0
    )
    return points, log_weights, features


def _invalid_mass_block_program(
    sources: tuple[tf.Tensor, ...], start: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    points, log_weights, features = _block_program(sources, start)
    return points, log_weights, tf.concat([2.0 * features[:1], features[1:]], axis=0)


def _tensor_list(value: tf.Tensor) -> list[float] | float:
    array = value.numpy()
    return float(array) if array.ndim == 0 else array.tolist()


def main() -> None:
    started = time.perf_counter()
    tf.config.experimental.enable_tensor_float_32_execution(True)
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("trusted GPU/XLA smoke requires one visible GPU")
    parents = tf.constant([-1.0, -0.5, 0.0, 0.5, 1.0], DTYPE)
    innovations = tf.constant([-0.3, 0.0, 0.3], DTYPE)
    theta = tf.constant([0.0, 0.08], DTYPE)
    sources = (parents, innovations, theta)
    source_tangents = (
        tf.linspace(tf.constant(-0.02, DTYPE), tf.constant(0.03, DTYPE), 5),
        tf.constant([0.01, -0.015, 0.02], DTYPE),
        tf.constant([0.04, -0.03], DTYPE),
    )
    active_indices = tf.constant([0, 7, 14], tf.int32)
    row_scale = tf.constant([1.0, 1.3, 1.69], DTYPE)
    teacher_count = tf.constant(15, tf.int32)
    forward = tp.make_contract_e_tp_streaming_square_forward_tf(
        block_size=BLOCK_SIZE, block_program=_block_program
    )
    jvp = tp.make_contract_e_tp_streaming_square_jvp_tf(
        block_size=BLOCK_SIZE, block_program=_block_program
    )
    vjp = tp.make_contract_e_tp_streaming_square_vjp_tf(
        block_size=BLOCK_SIZE, block_program=_block_program
    )
    tf.config.experimental.reset_memory_stats("GPU:0")
    forward_result = forward(sources, teacher_count, active_indices, row_scale)
    jvp_result = jvp(
        sources, source_tangents, teacher_count, active_indices, row_scale
    )
    vjp_result = vjp(
        sources,
        teacher_count,
        active_indices,
        row_scale,
        tf.constant([[0.2, -0.1], [0.05, 0.3], [-0.2, 0.15]], DTYPE),
        tf.constant([0.1, -0.4, 0.25], DTYPE),
        tf.constant([-0.2, 0.15, 0.05], DTYPE),
        tf.constant(0.31, DTYPE),
    )
    _ = [value.numpy() for value in vjp_result["source_bars"]]
    if not bool(forward_result["valid_chart"].numpy()):
        raise RuntimeError("valid Phase 2 fixture was rejected by the compiled chart")
    invalid_sources = (parents, innovations, theta)
    invalid_features_program = _invalid_mass_block_program
    invalid_forward = tp.make_contract_e_tp_streaming_square_forward_tf(
        block_size=BLOCK_SIZE, block_program=invalid_features_program
    )
    invalid_result = invalid_forward(
        invalid_sources, teacher_count, active_indices, row_scale
    )
    invalid_rejected = (
        not bool(invalid_result["valid_chart"].numpy())
        and bool(tf.reduce_all(tf.math.is_nan(invalid_result["student_weights"])).numpy())
        and bool(tf.reduce_all(tf.math.is_nan(invalid_result["student_points"])).numpy())
    )
    if not invalid_rejected:
        raise RuntimeError("compiled invalid chart did not fail closed")
    memory = tf.config.experimental.get_memory_info("GPU:0")
    concrete = forward.get_concrete_function(
        sources, teacher_count, active_indices, row_scale
    )
    operations = [node.op for node in concrete.graph.as_graph_def().node]
    payload = {
        "schema": "bayesfilter.contract_e_tp.phase2_gpu_xla_smoke.v1",
        "status": "PASS",
        "algorithm_id": tp.ALGORITHM_ID,
        "execution": {
            "git_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
            ).stdout.strip(),
            "tensorflow_version": tf.__version__,
            "device": gpus[0].name,
            "dtype": DTYPE.name,
            "tf32_enabled": tf.config.experimental.tensor_float_32_execution_enabled(),
            "jit_compile": True,
            "block_size": BLOCK_SIZE,
            "teacher_count": 15,
            "parent_count": 5,
            "innovation_count": 3,
            "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
            "command": (
                "/home/chakwong/anaconda3/envs/tf-gpu/bin/python "
                "docs/benchmarks/run_contract_e_tp_phase2_gpu_xla.py"
            ),
            "wall_time_seconds": time.perf_counter() - started,
            "gpu_allocator_current_bytes": int(memory["current"]),
            "gpu_allocator_peak_bytes": int(memory["peak"]),
        },
        "graph": {
            "operation_count": len(operations),
            "while_operation_count": sum(op in {"While", "StatelessWhile"} for op in operations),
            "tensor_list_operation_count": sum("TensorList" in op for op in operations),
            "dense_teacher_tensor_output": False,
            "retained_outputs": [
                "feature target",
                "fixed anchors",
                "student weights",
                "diagnostics",
            ],
        },
        "forward": {
            "valid_chart": bool(forward_result["valid_chart"].numpy()),
            "log_normalizer": _tensor_list(forward_result["log_normalizer"]),
            "target": _tensor_list(forward_result["target"]),
            "student_weights": _tensor_list(forward_result["student_weights"]),
            "minimum_weight": _tensor_list(forward_result["minimum_weight"]),
            "feature_residual": _tensor_list(forward_result["feature_residual"]),
        },
        "compiled_fail_closed_check": {
            "fixture": "mass feature is two instead of one",
            "valid_chart": bool(invalid_result["valid_chart"].numpy()),
            "student_outputs_nan_poisoned": invalid_rejected,
            "reason": "XLA ignores Assert operations; explicit predicates bind the veto",
        },
        "jvp": {
            "log_normalizer_tangent": _tensor_list(jvp_result["log_normalizer_tangent"]),
            "target_tangent": _tensor_list(jvp_result["target_tangent"]),
            "student_weights_tangent": _tensor_list(jvp_result["student_weights_tangent"]),
        },
        "vjp": {
            "source_bar_shapes": [value.shape.as_list() for value in vjp_result["source_bars"]],
            "source_bars_finite": all(
                bool(tf.reduce_all(tf.math.is_finite(value)).numpy())
                for value in vjp_result["source_bars"]
            ),
        },
        "nonclaims": [
            "tiny XLA composition smoke only",
            "not a full-horizon or scaling benchmark",
            "not evidence for canonical, leaderboard, default, or HMC admission",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
