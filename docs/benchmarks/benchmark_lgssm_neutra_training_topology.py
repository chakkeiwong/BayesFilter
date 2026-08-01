#!/usr/bin/env python3
"""Matched diagnostic benchmark for LGSSM NeuTra training control topology."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PLAN_PATH = (
    "docs/plans/"
    "bayesfilter-lgssm-neutra-training-topology-performance-benchmark-plan-"
    "2026-07-14.md"
)
TARGET_SIGNATURE = (
    "f47619320ded5f70259c6932eb2436642a02834c7a0249c7c52c20a5a2302f30"
)
ADAPTER_SIGNATURE = (
    "42dc7bad0137fd9c31aa1d618bb4e560f68d1bbe3a7ab4f5ef95e458b2abc985"
)
DIMENSION = 18
BATCH_SIZE = 128
SEED = (20260714, 1411)
HIDDEN_LAYERS = (18, 18)
STAGE_COUNT = 3
LEARNING_RATE = 5.0e-3
CLIP_NORM = 10.0
PARITY_TOLERANCE = 1.0e-12
DEVICE = "/GPU:0"


class BenchmarkError(RuntimeError):
    """Raised when a benchmark validity condition fails."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)

    cell = subparsers.add_parser("cell")
    cell.add_argument("--mode", required=True, choices=("host_stepped", "graph_native"))
    cell.add_argument("--steps", required=True, type=int, choices=(5, 20, 100))
    cell.add_argument("--output", required=True)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--host", required=True)
    compare.add_argument("--graph", required=True)
    compare.add_argument("--output", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.action == "cell":
        result = run_cell(
            mode=args.mode, steps=args.steps, output=_repository_path(args.output)
        )
    else:
        result = compare_cells(
            host_path=_repository_path(args.host),
            graph_path=_repository_path(args.graph),
            output=_repository_path(args.output),
        )
    print(json.dumps(_brief(result), sort_keys=True))
    return 0


def run_cell(*, mode: str, steps: int, output: Path) -> Mapping[str, Any]:
    raise BenchmarkError(
        "historical diagnostic only: new optimizer-cell launches are forbidden "
        "because this comparator uses the retired row-mapped scalar LGSSM target"
    )


def _historical_run_cell(*, mode: str, steps: int, output: Path) -> Mapping[str, Any]:
    """Preserve the 2026-07-14 source as unreachable migration evidence."""

    raise BenchmarkError(
        "historical optimizer cell is non-executable migration evidence"
    )
    if output.exists():
        raise BenchmarkError(f"refusing to overwrite benchmark artifact: {output}")
    if steps <= 0:
        raise BenchmarkError("steps must be positive")
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
        raise BenchmarkError("GPU benchmark cannot run with CUDA hidden")

    import tensorflow as tf

    physical = tf.config.list_physical_devices("GPU")
    if not physical:
        raise BenchmarkError("trusted TensorFlow GPU is unavailable")
    for physical_device in physical:
        try:
            tf.config.experimental.set_memory_growth(physical_device, True)
        except RuntimeError:
            pass
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(True)

    from bayesfilter.inference.neutra_training import (
        PlainDenseIAFTrainingConfig,
        PlainDenseIAFTransport,
        _reviewed_value_score_status_target_fn,
    )
    from bayesfilter.testing.deterministic_lgssm_exact_target_tf import (
        load_deterministic_lgssm_exact_target,
    )
    from bayesfilter.testing.lgssm_neutra_strict_training_tf import (
        _load_affine_geometry,
    )

    bundle = load_deterministic_lgssm_exact_target(
        expected_target_signature=TARGET_SIGNATURE
    )
    if bundle.adapter.adapter_signature() != ADAPTER_SIGNATURE:
        raise BenchmarkError("exact-target adapter signature mismatch")
    center, factor = _load_affine_geometry(tf)
    config = PlainDenseIAFTrainingConfig(
        target_signature=bundle.target_signature,
        dimension=DIMENSION,
        affine_center=center,
        affine_factor=factor,
        output_dir=output.parent / "diagnostic_only_no_training_artifact",
        seed=SEED,
        hidden_layers=HIDDEN_LAYERS,
        stage_count=STAGE_COUNT,
        activation="elu",
        s_max=1.0,
        init_scale=0.02,
        steps=steps,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        final_learning_rate_fraction=1.0,
        clip_norm=CLIP_NORM,
        checkpoint_every=steps,
        heartbeat_every=steps,
        jit_compile=True,
        device=DEVICE,
        require_gpu=True,
    )
    with tf.device(DEVICE):
        flow = PlainDenseIAFTransport(config)
        variables = flow.trainable_variables
        first_moments = tuple(
            tf.Variable(tf.zeros_like(value), trainable=False) for value in variables
        )
        second_moments = tuple(
            tf.Variable(tf.zeros_like(value), trainable=False) for value in variables
        )
    state_variables = (*variables, *first_moments, *second_moments)
    if not all("GPU" in str(value.device).upper() for value in state_variables):
        raise BenchmarkError("all trainable and optimizer state must be on GPU")
    initial_state = tuple(value.numpy() for value in state_variables)

    target_value_status = _reviewed_value_score_status_target_fn(bundle.adapter)
    if target_value_status is None:
        raise BenchmarkError("exact target status telemetry is required")

    diagnostic_specs = (
        ("loss", tf.float64),
        ("raw_gradient_norm", tf.float64),
        ("clipped_gradient_norm", tf.float64),
        ("learning_rate", tf.float64),
        ("mean_log_abs_det_jacobian", tf.float64),
        ("target_values_finite", tf.bool),
        ("target_status_all_valid", tf.bool),
        ("target_status_nonvalid_count", tf.int32),
        ("target_floor_count_total", tf.int32),
        ("target_min_innovation_eigenvalue", tf.float64),
        ("target_max_innovation_condition_estimate", tf.float64),
    )

    def step_fn(step_index):
        step_i32 = tf.cast(step_index, tf.int32)
        seed = tf.stack(
            (
                tf.cast(config.seed[0], tf.int32),
                tf.cast(config.seed[1], tf.int32) + step_i32,
            )
        )
        z = tf.random.stateless_normal(
            (config.batch_size, config.dimension), seed=seed, dtype=tf.float64
        )
        with tf.GradientTape() as tape:
            theta, logdet = flow.forward_and_logdet(z)
            log_prob, status = target_value_status(theta)
            status_code = tf.convert_to_tensor(status["status_code"], tf.int32)
            status_valid = tf.convert_to_tensor(
                status["valid_pre_regularized_score"], tf.bool
            )
            status_nonvalid = tf.logical_or(
                tf.not_equal(status_code, tf.zeros_like(status_code)),
                tf.logical_not(status_valid),
            )
            loss = -tf.reduce_mean(log_prob + logdet)
        gradients = tape.gradient(loss, variables)
        if any(gradient is None for gradient in gradients):
            raise BenchmarkError("training gradient is missing")
        raw_norm = tf.linalg.global_norm(gradients)
        clipped, _ = tf.clip_by_global_norm(gradients, config.clip_norm)
        clipped_norm = tf.linalg.global_norm(clipped)
        progress = tf.cast(step_i32 + 1, tf.float64) / tf.cast(
            config.steps, tf.float64
        )
        lr = tf.cast(config.learning_rate, tf.float64) * (
            1.0
            - progress
            * (1.0 - tf.cast(config.final_learning_rate_fraction, tf.float64))
        )
        t = tf.cast(step_i32 + 1, tf.float64)
        beta1 = tf.cast(config.beta1, tf.float64)
        beta2 = tf.cast(config.beta2, tf.float64)
        for variable, gradient, first, second in zip(
            variables, clipped, first_moments, second_moments
        ):
            first.assign(beta1 * first + (1.0 - beta1) * gradient)
            second.assign(beta2 * second + (1.0 - beta2) * tf.square(gradient))
            first_hat = first / (1.0 - tf.pow(beta1, t))
            second_hat = second / (1.0 - tf.pow(beta2, t))
            variable.assign_sub(
                lr
                * first_hat
                / (tf.sqrt(second_hat) + tf.cast(config.epsilon, tf.float64))
            )
        return (
            loss,
            raw_norm,
            clipped_norm,
            lr,
            tf.reduce_mean(logdet),
            tf.reduce_all(tf.math.is_finite(log_prob)),
            tf.reduce_all(tf.logical_not(status_nonvalid)),
            tf.reduce_sum(tf.cast(status_nonvalid, tf.int32)),
            tf.reduce_sum(tf.convert_to_tensor(status["floor_count_value"], tf.int32)),
            tf.reduce_min(
                tf.convert_to_tensor(status["min_innovation_eigenvalue"], tf.float64)
            ),
            tf.reduce_max(
                tf.convert_to_tensor(status["innovation_condition_estimate"], tf.float64)
            ),
        )

    if mode == "host_stepped":
        compiled_step = tf.function(step_fn, jit_compile=True, reduce_retracing=True)
        trace_start = time.monotonic()
        concrete = compiled_step.get_concrete_function(
            tf.TensorSpec((), dtype=tf.int32)
        )
        trace_seconds = time.monotonic() - trace_start

        def run_once():
            # Historical diagnostic comparator only: one synchronized host call per step.
            rows = []
            for step in range(steps):
                outputs = compiled_step(tf.constant(step, tf.int32))
                rows.append(tuple(_materialize(value) for value in outputs))
            return tuple(rows)

    else:
        def graph_program():
            arrays = tuple(
                tf.TensorArray(
                    dtype=dtype,
                    size=steps,
                    clear_after_read=False,
                    element_shape=(),
                )
                for _name, dtype in diagnostic_specs
            )

            def condition(step_index, *_arrays):
                return step_index < tf.constant(steps, tf.int32)

            def body(step_index, *current_arrays):
                outputs = step_fn(step_index)
                updated = tuple(
                    array.write(step_index, value)
                    for array, value in zip(current_arrays, outputs)
                )
                return (step_index + tf.constant(1, tf.int32), *updated)

            final = tf.while_loop(
                condition,
                body,
                (tf.constant(0, tf.int32), *arrays),
                parallel_iterations=1,
            )
            return tuple(array.stack() for array in final[1:])

        compiled_graph = tf.function(
            graph_program, jit_compile=True, reduce_retracing=True
        )
        trace_start = time.monotonic()
        concrete = compiled_graph.get_concrete_function()
        trace_seconds = time.monotonic() - trace_start

        def run_once():
            outputs = compiled_graph()
            columns = tuple(_materialize(value) for value in outputs)
            return tuple(
                tuple(column[step] for column in columns) for step in range(steps)
            )

    graph_operation_types = tuple(
        sorted({operation.type for operation in concrete.graph.get_operations()})
    )
    has_while = any("While" in value for value in graph_operation_types)
    if mode == "graph_native" and not has_while:
        raise BenchmarkError("graph-native program lacks TensorFlow control flow")

    _restore_state(state_variables, initial_state)
    cold_start = time.monotonic()
    cold_diagnostics = run_once()
    cold_seconds = time.monotonic() - cold_start
    cold_state = _state_payload(state_variables)

    _restore_state(state_variables, initial_state)
    warm_start = time.monotonic()
    warm_diagnostics = run_once()
    warm_seconds = time.monotonic() - warm_start
    warm_state = _state_payload(state_variables)

    within_state_delta = _max_abs_difference(cold_state, warm_state)
    within_diagnostic_delta = _max_abs_difference(
        cold_diagnostics, warm_diagnostics
    )
    valid = _validate_diagnostics(warm_diagnostics, diagnostic_specs)
    if within_state_delta > PARITY_TOLERANCE:
        raise BenchmarkError("cold/warm final-state parity failed")
    if within_diagnostic_delta > PARITY_TOLERANCE:
        raise BenchmarkError("cold/warm diagnostic parity failed")
    if not valid:
        raise BenchmarkError("training diagnostics failed validity checks")

    result = {
        "schema": "bayesfilter.lgssm_neutra_training_topology_cell.v1",
        "passed": True,
        "mode": mode,
        "steps": steps,
        "question": "matched_host_stepped_vs_graph_native_training_time",
        "timing": {
            "trace_seconds": trace_seconds,
            "cold_compile_plus_execution_seconds": cold_seconds,
            "warm_execution_seconds": warm_seconds,
            "cold_seconds_per_step": cold_seconds / steps,
            "warm_seconds_per_step": warm_seconds / steps,
            "synchronization": (
                "all_returned_diagnostics_materialized_inside_timed_interval"
            ),
            "repetitions": {"cold": 1, "warm": 1},
            "evidence_class": "descriptive_single_repetition",
        },
        "topology": {
            "jit_compile": True,
            "compiled_function_invocations_per_repetition": (
                steps if mode == "host_stepped" else 1
            ),
            "python_step_loop": mode == "host_stepped",
            "tf_while_loop": mode == "graph_native",
            "outer_training_control_flow": (
                "python_for_over_compiled_step"
                if mode == "host_stepped"
                else "tf_while_loop_inside_single_compiled_program"
            ),
            "operation_inventory_includes_shared_target_internals": True,
            "graph_operation_types": graph_operation_types,
        },
        "parity": {
            "cold_warm_state_max_abs": within_state_delta,
            "cold_warm_diagnostic_max_abs": within_diagnostic_delta,
            "cold_state_hash": _stable_hash(cold_state),
            "warm_state_hash": _stable_hash(warm_state),
            "cold_diagnostic_hash": _stable_hash(cold_diagnostics),
            "warm_diagnostic_hash": _stable_hash(warm_diagnostics),
            "tolerance": PARITY_TOLERANCE,
        },
        "final_state": warm_state,
        "diagnostics": _diagnostic_rows(warm_diagnostics, diagnostic_specs),
        "configuration": {
            "target_signature": TARGET_SIGNATURE,
            "adapter_signature": ADAPTER_SIGNATURE,
            "dimension": DIMENSION,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "hidden_layers": HIDDEN_LAYERS,
            "stage_count": STAGE_COUNT,
            "learning_rate": LEARNING_RATE,
            "final_learning_rate_fraction": 1.0,
            "optimizer": "manual_adam",
            "clip_norm": CLIP_NORM,
            "dtype": "float64",
            "device": DEVICE,
        },
        "run_manifest": _run_manifest(tf, output),
        "plan": PLAN_PATH,
        "output": str(output.relative_to(ROOT)),
        "evidence_role": "diagnostic_performance_comparison_only",
        "nonclaims": (
            "no transport quality or posterior correctness claim",
            "no HMC convergence claim",
            "no recipe ranking or default-readiness claim",
            "no broad GPU speedup claim",
        ),
    }
    result = {**result, "artifact_hash": f"sha256:{_stable_hash(result)}"}
    _write_new_json(output, result)
    return result


def compare_cells(
    *, host_path: Path, graph_path: Path, output: Path
) -> Mapping[str, Any]:
    if output.exists():
        raise BenchmarkError(f"refusing to overwrite comparison: {output}")
    host = _read_mapping(host_path)
    graph = _read_mapping(graph_path)
    if host.get("mode") != "host_stepped" or graph.get("mode") != "graph_native":
        raise BenchmarkError("comparison modes are invalid")
    if host.get("steps") != graph.get("steps"):
        raise BenchmarkError("comparison step counts differ")
    if host.get("configuration") != graph.get("configuration"):
        raise BenchmarkError("comparison configurations differ")
    state_delta = _max_abs_difference(host["final_state"], graph["final_state"])
    diagnostic_delta = _max_abs_difference(
        _numeric_diagnostics(host["diagnostics"]),
        _numeric_diagnostics(graph["diagnostics"]),
    )
    parity_passed = bool(
        state_delta <= PARITY_TOLERANCE and diagnostic_delta <= PARITY_TOLERANCE
    )
    if not parity_passed:
        raise BenchmarkError("cross-mode update parity failed")

    host_cold = float(host["timing"]["cold_compile_plus_execution_seconds"])
    graph_cold = float(graph["timing"]["cold_compile_plus_execution_seconds"])
    host_warm = float(host["timing"]["warm_execution_seconds"])
    graph_warm = float(graph["timing"]["warm_execution_seconds"])
    warm_speedup = host_warm / graph_warm
    cold_speedup = host_cold / graph_cold
    if warm_speedup > 1.0:
        verdict = "GRAPH_NATIVE_FASTER_ON_MATCHED_WARM_RUNG"
    elif warm_speedup < 1.0:
        verdict = "GRAPH_NATIVE_SLOWER_ON_MATCHED_WARM_RUNG"
    else:
        verdict = "MATCHED_WARM_TIMES_EQUAL"
    result = {
        "schema": "bayesfilter.lgssm_neutra_training_topology_comparison.v1",
        "passed": True,
        "verdict": verdict,
        "steps": int(host["steps"]),
        "primary_criterion": "synchronized_warm_wall_time",
        "timing": {
            "host_cold_seconds": host_cold,
            "graph_cold_seconds": graph_cold,
            "cold_host_over_graph_ratio": cold_speedup,
            "host_warm_seconds": host_warm,
            "graph_warm_seconds": graph_warm,
            "warm_host_over_graph_speedup": warm_speedup,
            "warm_graph_time_reduction_fraction": 1.0 - graph_warm / host_warm,
            "evidence_class": "descriptive_single_repetition_per_cell",
        },
        "parity": {
            "passed": parity_passed,
            "final_state_max_abs": state_delta,
            "diagnostic_max_abs": diagnostic_delta,
            "tolerance": PARITY_TOLERANCE,
        },
        "inputs": {
            "host": _file_reference(host_path),
            "graph": _file_reference(graph_path),
        },
        "plan": PLAN_PATH,
        "output": str(output.relative_to(ROOT)),
        "interpretation_limit": (
            "one matched rung on one GPU with one warm repetition; no uncertainty interval"
        ),
        "nonclaims": (
            "no broad speedup claim",
            "no 100-step timing claim unless this is the 100-step rung",
            "no transport quality, HMC, or posterior claim",
        ),
    }
    result = {**result, "artifact_hash": f"sha256:{_stable_hash(result)}"}
    _write_new_json(output, result)
    return result


def _restore_state(variables: Sequence[Any], values: Sequence[Any]) -> None:
    for variable, value in zip(variables, values):
        variable.assign(value)
    # Force completion before a timed repetition begins.
    for variable in variables:
        variable.read_value().numpy()


def _state_payload(variables: Sequence[Any]) -> tuple[Any, ...]:
    return tuple(variable.numpy().tolist() for variable in variables)


def _materialize(value: Any) -> Any:
    materialized = value.numpy().tolist()
    return materialized


def _validate_diagnostics(
    diagnostics: Sequence[Sequence[Any]], specs: Sequence[tuple[str, Any]]
) -> bool:
    names = tuple(name for name, _dtype in specs)
    finite_index = names.index("target_values_finite")
    valid_index = names.index("target_status_all_valid")
    nonvalid_index = names.index("target_status_nonvalid_count")
    floor_index = names.index("target_floor_count_total")
    numeric_indices = tuple(
        names.index(name)
        for name in (
            "loss",
            "raw_gradient_norm",
            "clipped_gradient_norm",
            "learning_rate",
            "mean_log_abs_det_jacobian",
            "target_min_innovation_eigenvalue",
            "target_max_innovation_condition_estimate",
        )
    )
    return all(
        bool(row[finite_index])
        and bool(row[valid_index])
        and int(row[nonvalid_index]) == 0
        and int(row[floor_index]) == 0
        and all(math.isfinite(float(row[index])) for index in numeric_indices)
        for row in diagnostics
    )


def _diagnostic_rows(
    diagnostics: Sequence[Sequence[Any]], specs: Sequence[tuple[str, Any]]
) -> tuple[Mapping[str, Any], ...]:
    names = tuple(name for name, _dtype in specs)
    return tuple(
        {"step": step + 1, **dict(zip(names, row))}
        for step, row in enumerate(diagnostics)
    )


def _numeric_diagnostics(rows: Sequence[Mapping[str, Any]]) -> tuple[Any, ...]:
    return tuple(
        tuple(value for key, value in row.items() if key != "step") for row in rows
    )


def _max_abs_difference(left: Any, right: Any) -> float:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            return math.inf
        return max(
            (_max_abs_difference(left[key], right[key]) for key in left),
            default=0.0,
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return math.inf
        return max(
            (_max_abs_difference(a, b) for a, b in zip(left, right)), default=0.0
        )
    if isinstance(left, bool) or isinstance(right, bool):
        return 0.0 if bool(left) == bool(right) else math.inf
    try:
        return abs(float(left) - float(right))
    except (TypeError, ValueError):
        return 0.0 if left == right else math.inf


def _run_manifest(tf: Any, output: Path) -> Mapping[str, Any]:
    return {
        "git_commit": _command_output(("git", "rev-parse", "HEAD")),
        "git_dirty": bool(_command_output(("git", "status", "--porcelain"))),
        "command": tuple(sys.argv),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "tensorflow_version": tf.__version__,
        "physical_gpus": tuple(str(value) for value in tf.config.list_physical_devices("GPU")),
        "logical_gpus": tuple(str(value) for value in tf.config.list_logical_devices("GPU")),
        "tf32_enabled": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
        "xla_jit_compile": True,
        "device": DEVICE,
        "dtype": "float64",
        "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        "output": str(output.relative_to(ROOT)),
        "plan": PLAN_PATH,
    }


def _command_output(command: Sequence[str]) -> str:
    completed = subprocess.run(
        tuple(command), cwd=ROOT, check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def _repository_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _read_mapping(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise BenchmarkError(f"expected JSON mapping: {path}")
    return value


def _file_reference(path: Path) -> Mapping[str, Any]:
    return {
        "path": str(path.relative_to(ROOT)),
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "byte_count": path.stat().st_size,
    }


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def _brief(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        key: value[key]
        for key in ("mode", "steps", "passed", "verdict", "timing", "artifact_hash")
        if key in value
    }


if __name__ == "__main__":
    raise SystemExit(main())
