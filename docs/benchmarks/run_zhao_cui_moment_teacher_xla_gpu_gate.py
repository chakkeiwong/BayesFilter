"""Bounded GPU/TF32/XLA gate for the graph-native moment teacher candidate."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth


MEMORY_POLICY = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)

from bayesfilter.highdim.zhao_cui_moment_teacher_xla import (
    padded_fixed_teacher_recursion_shape_xla,
    padded_squared_tt_shape_targets_jvp_xla,
)


TRUST_BASIS = "owner_designated_managed_session_visible_gpu_trusted"
SCHEMA = "bayesfilter.zhao_cui_moment_teacher_xla_gpu_gate.v1"
OUTPUT_NAMES = (
    "final_cores",
    "final_core_tangents",
    "marginal_values",
    "marginal_tangents",
    "normalizers",
    "skew",
    "kurtosis",
    "co_skew",
    "co_kurtosis",
    "skew_tangents",
    "kurtosis_tangents",
    "co_skew_tangents",
    "co_kurtosis_tangents",
)


def _json(value):
    if isinstance(value, tf.Tensor):
        return _json(value.numpy())
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(item) for item in value]
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    return str(value)


def _gpu_memory():
    try:
        info = tf.config.experimental.get_memory_info("GPU:0")
        return {key: int(value) for key, value in info.items()}
    except Exception as error:  # diagnostic field only
        return {"status": "unavailable", "error": str(error)}


def _sha256(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def _graph_ops(function, *args):
    concrete = function.get_concrete_function(*args)
    graph = concrete.graph.as_graph_def()
    counts = {}
    for node in graph.node:
        counts[node.op] = counts.get(node.op, 0) + 1
    for fn in graph.library.function:
        for node in fn.node_def:
            counts[node.op] = counts.get(node.op, 0) + 1
    return {
        "operation_counts": counts,
        "has_while": bool({"While", "StatelessWhile"} & set(counts)),
        "pyfunc_count": counts.get("PyFunc", 0) + counts.get("EagerPyFunc", 0),
    }


def _cast_float_args(args, dtype):
    return tuple(
        tf.cast(value, dtype) if value.dtype.is_floating else value for value in args
    )


def _parity_rows(fp32_result, fp64_result):
    rows = []
    for index, (fp32_value, fp64_value) in enumerate(
        zip(fp32_result[:-1], fp64_result[:-1])
    ):
        left = tf.cast(fp32_value, tf.float64)
        right = tf.cast(fp64_value, tf.float64)
        difference = tf.abs(left - right)
        denominator = tf.maximum(tf.abs(right), tf.constant(1e-8, tf.float64))
        relative_difference = difference / denominator
        relative_flat_index = tf.argmax(
            tf.reshape(relative_difference, [-1]), output_type=tf.int32
        )
        left_flat = tf.reshape(left, [-1])
        right_flat = tf.reshape(right, [-1])
        difference_flat = tf.reshape(difference, [-1])
        rows.append(
            {
                "output_index": index,
                "output_name": OUTPUT_NAMES[index],
                "candidate_max_abs": float(tf.reduce_max(tf.abs(left)).numpy()),
                "reference_max_abs": float(tf.reduce_max(tf.abs(right)).numpy()),
                "max_abs_error": float(tf.reduce_max(difference).numpy()),
                "max_rel_error": float(
                    tf.reduce_max(relative_difference).numpy()
                ),
                "max_rel_error_element": {
                    "flat_index": int(relative_flat_index.numpy()),
                    "candidate": float(left_flat[relative_flat_index].numpy()),
                    "reference": float(right_flat[relative_flat_index].numpy()),
                    "absolute_error": float(
                        difference_flat[relative_flat_index].numpy()
                    ),
                },
            }
        )
    return rows


def _basis_values(points):
    x = points
    return tf.stack([tf.ones_like(x), tf.sqrt(tf.constant(3.0, tf.float32)) * x,
                     tf.sqrt(tf.constant(5.0, tf.float32)) * (1.5 * tf.square(x) - 0.5)], axis=-1)


def _build_inputs():
    rows = 256
    points = tf.linspace(tf.constant(-0.9, tf.float32), tf.constant(0.9, tf.float32), rows)
    basis_values = tf.stack([_basis_values(points), _basis_values(tf.reverse(points, [0]))])
    mask = tf.constant(
        [
            [[[1, 0], [1, 0], [1, 0]], [[0, 0], [0, 0], [0, 0]]],
            [[[1, 1], [0, 0], [0, 0]], [[1, 1], [0, 0], [0, 0]]],
        ], tf.float32
    )
    initial = tf.constant(
        [
            [[[1.0, 0.0], [0.1, 0.0], [-0.1, 0.0]], [[0, 0], [0, 0], [0, 0]]],
            [[[0.9, 0.2], [0.0, 0.0], [0.0, 0.0]], [[0.1, 0.6], [0, 0], [0, 0]]],
        ], tf.float32
    ) * mask
    initial_dot = tf.fill(tf.shape(initial), tf.constant(0.001, tf.float32)) * mask
    target = tf.stack([
        tf.exp(-0.5 * tf.square(points - 0.1)),
        tf.exp(-0.4 * tf.square(tf.reverse(points, [0]) + 0.2)),
    ])
    dot_log_target = 0.1 * tf.stack([points, tf.reverse(points, [0])])
    weights = tf.fill([rows], tf.constant(1.0 / rows, tf.float32))
    dot_weights = tf.zeros_like(weights)
    operator_powers = tf.stack([
        tf.stack([
            tf.eye(3, dtype=tf.float32),
            tf.constant([[0.0, 0.5773503, 0.0], [0.5773503, 0.0, 0.5163978], [0.0, 0.5163978, 0.0]], tf.float32),
            tf.eye(3, dtype=tf.float32),
            tf.eye(3, dtype=tf.float32),
            tf.eye(3, dtype=tf.float32),
        ]),
        tf.stack([
            tf.eye(3, dtype=tf.float32),
            tf.constant([[0.0, -0.5773503, 0.0], [-0.5773503, 0.0, 0.5163978], [0.0, 0.5163978, 0.0]], tf.float32),
            tf.eye(3, dtype=tf.float32),
            tf.eye(3, dtype=tf.float32),
            tf.eye(3, dtype=tf.float32),
        ]),
    ])
    defensive_moments = tf.ones([2, 5], tf.float32)
    query_basis = basis_values
    return {
        "basis_values": basis_values,
        "active_mask": mask,
        "schedule": tf.constant([0, 1, 0, 1], tf.int32),
        "base_log_targets": tf.math.log(target),
        "dot_base_log_targets": dot_log_target,
        "weights": weights,
        "dot_weights": dot_weights,
        "initial_cores": initial,
        "initial_dot_cores": initial_dot,
        "scale_shift_indices": tf.constant([0, 0], tf.int32),
        "defensive_weights": tf.constant([0.05, 0.05], tf.float32),
        "dot_defensive_weights": tf.zeros([2], tf.float32),
        "query_basis_values": query_basis,
        "keep_mask": tf.constant([True, False]),
        "mass_operators": operator_powers[:, 0],
        "defensive_marginal_values": tf.ones([2, rows], tf.float32),
        "dot_defensive_marginal_values": tf.zeros([2, rows], tf.float32),
        "defensive_mass": tf.constant(1.0, tf.float32),
        "dot_defensive_mass": tf.constant(0.0, tf.float32),
        "operator_powers": operator_powers,
        "defensive_power_moments": defensive_moments,
    }


def main():
    parser = __import__("argparse").ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--no-tf32",
        action="store_true",
        help="Run the reviewed moment-teacher FP32-no-TF32 execution route.",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    campaign_started = time.perf_counter()
    tf32_enabled = not args.no_tf32
    tf.config.experimental.enable_tensor_float_32_execution(tf32_enabled)
    if not tf.config.list_physical_devices("GPU"):
        raise RuntimeError("GPU gate requires a visible GPU")
    inputs = _build_inputs()
    ridge = tf.constant(1e-5, tf.float32)
    floor = tf.constant(1e-6, tf.float32)
    veto = tf.constant(1e8, tf.float32)
    residual = tf.constant(2e-3, tf.float32)
    recursion_args = [
        inputs[key] for key in (
            "basis_values", "active_mask", "schedule", "base_log_targets",
            "dot_base_log_targets", "weights", "dot_weights", "initial_cores",
            "initial_dot_cores", "scale_shift_indices", "defensive_weights",
            "dot_defensive_weights", "query_basis_values", "keep_mask",
            "mass_operators", "defensive_marginal_values",
            "dot_defensive_marginal_values", "defensive_mass", "dot_defensive_mass",
        )
    ] + [ridge, floor, veto, residual]
    shape_args = [
        inputs[key] for key in (
            "initial_cores", "initial_dot_cores", "operator_powers",
            "defensive_power_moments",
        )
    ] + [
        tf.zeros([2], tf.float32), tf.zeros([2], tf.float32),
        tf.eye(2, dtype=tf.float32), tf.zeros([2, 2], tf.float32),
        tf.constant([[0, 1]], tf.int32), tf.constant(0.2, tf.float32),
        tf.constant(-0.01, tf.float32), tf.constant(1.0, tf.float32),
    ]
    fused_args = recursion_args[:-4] + [
        inputs["operator_powers"],
        inputs["defensive_power_moments"],
        tf.zeros([2], tf.float32),
        tf.zeros([2], tf.float32),
        tf.eye(2, dtype=tf.float32),
        tf.zeros([2, 2], tf.float32),
        tf.constant([[0, 1]], tf.int32),
    ] + recursion_args[-4:]
    started = time.perf_counter()
    tf.config.experimental.reset_memory_stats("GPU:0")
    warmup = padded_fixed_teacher_recursion_shape_xla(*fused_args)
    warmup[0].device
    tf.experimental.sync_devices() if hasattr(tf.experimental, "sync_devices") else None
    warmup_seconds = time.perf_counter() - started
    before = _gpu_memory()
    started = time.perf_counter()
    result = padded_fixed_teacher_recursion_shape_xla(*fused_args)
    shape_result = result[5:13]
    fp64_result = padded_fixed_teacher_recursion_shape_xla(
        *_cast_float_args(fused_args, tf.float64)
    )
    tf.experimental.sync_devices() if hasattr(tf.experimental, "sync_devices") else None
    elapsed = time.perf_counter() - started
    after = _gpu_memory()
    concrete_ops = _graph_ops(padded_fixed_teacher_recursion_shape_xla, *fused_args)
    shape_ops = _graph_ops(padded_squared_tt_shape_targets_jvp_xla, *shape_args)
    parity = _parity_rows(result, fp64_result)
    parity_max_abs = max(row["max_abs_error"] for row in parity)
    parity_max_rel = max(row["max_rel_error"] for row in parity)
    parity_abs_veto = 2e-3
    parity_rel_veto = 5e-3
    recursion_valid = bool(result[-1].numpy())
    shape_finite = bool(
        tf.reduce_all(
            tf.math.is_finite(
                tf.concat([tf.reshape(item, [-1]) for item in shape_result], axis=0)
            )
        ).numpy()
    )
    hard_vetoes = {
        "pyfunc": concrete_ops["pyfunc_count"] + shape_ops["pyfunc_count"],
        "missing_while": not (
            concrete_ops["has_while"] and shape_ops["has_while"]
        ),
        "invalid_recursion": not recursion_valid,
        "nonfinite_shape": not shape_finite,
        "fp32_fp64_parity": not (
            parity_max_abs <= parity_abs_veto
            and parity_max_rel <= parity_rel_veto
        ),
    }
    payload = {
        "schema": SCHEMA,
        "route_id": "zhao_cui_fixed_als_padded_xla_value_jvp_v1",
        "route_classification": "extension_or_invention",
        "plan": "docs/plans/bayesfilter-zhao-cui-moment-teacher-plan-2026-07-30.md",
        "result_note": "docs/plans/bayesfilter-zhao-cui-moment-teacher-result-2026-07-30.md",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "command": " ".join(sys.argv),
        "environment": {"python": sys.version, "platform": platform.platform(), "tensorflow": tf.__version__, "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "unset"), "tf32": tf32_enabled, "jit_compile": True},
        "device": {"physical_gpus": [device.name for device in tf.config.list_physical_devices("GPU")], "logical_gpus": [device.name for device in tf.config.list_logical_devices("GPU")], "memory_policy": MEMORY_POLICY, "trust_basis": TRUST_BASIS},
        "scope": {"dtype": "float32", "rows": 256, "dimension": 2, "padded_rank": 2, "basis": 3, "schedule": [0, 1, 0, 1], "ridge": 1e-5, "data_version": "deterministic_synthetic_prepared_fixture_v1", "seeds": "N/A_no_random_draws"},
        "source_sha256": {
            path: _sha256(path)
            for path in (
                "bayesfilter/highdim/zhao_cui_moment_teacher_xla.py",
                "docs/benchmarks/run_zhao_cui_moment_teacher_xla_gpu_gate.py",
                "docs/chapters/ch32c_entropic_ot_sinkhorn.tex",
                "docs/plans/bayesfilter-zhao-cui-moment-teacher-plan-2026-07-30.md",
            )
        },
        "warmup_seconds": warmup_seconds,
        "measured_seconds": elapsed,
        "gpu_allocator_before": before,
        "gpu_allocator_after": after,
        "candidate_kernel_executed": True,
        "recursion_valid": recursion_valid,
        "shape_finite": shape_finite,
        "fp32_fp64_parity": {
            "rows": parity,
            "max_abs_error": parity_max_abs,
            "max_rel_error": parity_max_rel,
            "absolute_veto": parity_abs_veto,
            "relative_veto": parity_rel_veto,
            "passed": parity_max_abs <= parity_abs_veto and parity_max_rel <= parity_rel_veto,
            "role": "fixture_mechanics_promotion_veto_not_model_accuracy",
        },
        "recursion_graph": concrete_ops,
        "shape_graph": shape_ops,
        "hard_vetoes": hard_vetoes,
        "passed": not any(bool(value) for value in hard_vetoes.values()),
        "execution_classification": (
            "historical_tf32_candidate_mechanics_gate"
            if tf32_enabled
            else "selected_moment_teacher_fp32_no_tf32_mechanics_gate"
        ),
        "execution_route_id": (
            "zhao_cui_moment_teacher_gpu_fp32_tf32_xla_candidate_v1"
            if tf32_enabled
            else "zhao_cui_moment_teacher_gpu_fp32_no_tf32_xla_v1"
        ),
        "nonclaims": ["no nonlinear filtering result", "no leaderboard result", "no repository-wide default change", "no HMC readiness"],
    }
    payload["wall_time_seconds"] = time.perf_counter() - campaign_started
    payload["output_artifact"] = str(output.relative_to(ROOT))
    result_path = output / "result.json"
    result_path.write_text(json.dumps(_json(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema": "bayesfilter.run_manifest.v1",
        "result": str(result_path.relative_to(ROOT)),
        "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        "plan": payload["plan"],
        "result_note": payload["result_note"],
        "git_commit": payload["git_commit"],
        "command": payload["command"],
        "environment": payload["environment"],
        "device": payload["device"],
        "data_version": payload["scope"]["data_version"],
        "seeds": payload["scope"]["seeds"],
        "wall_time_seconds": payload["wall_time_seconds"],
        "output_artifact": payload["output_artifact"],
        "trust_basis": TRUST_BASIS,
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(_json(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
