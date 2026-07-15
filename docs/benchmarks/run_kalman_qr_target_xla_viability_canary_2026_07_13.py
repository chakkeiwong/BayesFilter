#!/usr/bin/env python
"""Run a fresh-process target-horizon XLA canary for the true-batched QR pair."""

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
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = Path(__file__).resolve()
BENCHMARK_PATH = REPO_ROOT / "scripts/benchmark_kalman_qr_parameter_count_scaling.py"
PLAN_PATH = REPO_ROOT / (
    "docs/plans/bayesfilter-kalman-qr-target-xla-viability-live-plan-2026-07-13.md"
)
RESULT_PATH = REPO_ROOT / (
    "docs/plans/bayesfilter-kalman-qr-target-xla-viability-result-2026-07-13.md"
)
DEFAULT_OUTPUT = REPO_ROOT / (
    "docs/benchmarks/"
    "kalman_qr_target_xla_viability_cpu_d10_t120_p50_b1_2026-07-13.json"
)
WORK_ROOT = Path("/tmp/kalman_qr_target_xla_viability_2026_07_13")
SCHEMA = "bayesfilter.kalman_qr.target_xla_viability.v2"
METHODS = (
    "batch_native_analytical_qr_score",
    "batch_native_autodiff_qr_score",
)
RTOL = 2.0e-4
ATOL = 2.0e-4
KILL_GRACE_SECONDS = 10


def _strict_json(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        indent=indent,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(_strict_json(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    return completed.stdout.strip()


def _progress(path: Path, method: str, stage: str, started: float) -> None:
    _write_json(
        path,
        {
            "method": method,
            "stage": stage,
            "elapsed_seconds": time.perf_counter() - started,
        },
    )


def _flatten(value: Any) -> list[float]:
    if isinstance(value, list):
        flattened: list[float] = []
        for item in value:
            flattened.extend(_flatten(item))
        return flattened
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("numeric output must contain only finite numbers")
    return [float(value)]


def _comparison(candidate: Any, reference: Any) -> dict[str, Any]:
    candidate_flat = _flatten(candidate)
    reference_flat = _flatten(reference)
    if len(candidate_flat) != len(reference_flat):
        return {
            "passed": False,
            "candidate_count": len(candidate_flat),
            "reference_count": len(reference_flat),
            "max_abs_residual": None,
            "rtol": RTOL,
            "atol": ATOL,
        }
    residuals = [
        abs(candidate_value - reference_value)
        for candidate_value, reference_value in zip(
            candidate_flat, reference_flat, strict=True
        )
    ]
    passed = all(
        math.isfinite(candidate_value)
        and math.isfinite(reference_value)
        and residual <= ATOL + RTOL * abs(reference_value)
        for candidate_value, reference_value, residual in zip(
            candidate_flat, reference_flat, residuals, strict=True
        )
    )
    return {
        "passed": passed,
        "candidate_count": len(candidate_flat),
        "reference_count": len(reference_flat),
        "max_abs_residual": max(residuals, default=0.0),
        "rtol": RTOL,
        "atol": ATOL,
    }


def _graph_record(graph_def: Any) -> dict[str, Any]:
    top_level_nodes = len(graph_def.node)
    function_nodes = sum(
        len(function.node_def) for function in graph_def.library.function
    )
    op_histogram: dict[str, int] = {}
    for node in graph_def.node:
        op_histogram[node.op] = op_histogram.get(node.op, 0) + 1
    for function in graph_def.library.function:
        for node in function.node_def:
            op_histogram[node.op] = op_histogram.get(node.op, 0) + 1
    raw = graph_def.SerializeToString(deterministic=True)
    return {
        "serialized_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "top_level_node_count": top_level_nodes,
        "function_node_count": function_nodes,
        "total_node_count": top_level_nodes + function_nodes,
        "function_count": len(graph_def.library.function),
        "selected_op_counts": {
            name: op_histogram.get(name, 0)
            for name in (
                "EmptyTensorList",
                "StatelessWhile",
                "TensorListGetItem",
                "TensorListPopBack",
                "TensorListPushBack",
                "While",
            )
        },
    }


def _tensor_output(value: Any, score: Any, tf: Any) -> dict[str, Any]:
    value_values = value.numpy().tolist()
    score_values = score.numpy().tolist()
    all_finite = bool(
        tf.reduce_all(tf.math.is_finite(value)).numpy()
        and tf.reduce_all(tf.math.is_finite(score)).numpy()
    )
    return {
        "value": value_values,
        "score": score_values,
        "value_shape": value.shape.as_list(),
        "score_shape": score.shape.as_list(),
        "value_dtype": value.dtype.name,
        "score_dtype": score.dtype.name,
        "devices": [value.device, score.device],
        "all_finite": all_finite,
    }


def _child(args: argparse.Namespace) -> int:
    expected_cuda = "-1" if args.device == "cpu" else args.cuda_visible_devices
    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_cuda:
        raise RuntimeError("XLA child CUDA visibility does not match its CLI identity")
    started = time.perf_counter()
    _progress(args.progress, args.method, "before_tensorflow_import", started)

    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    selected_device = "/CPU:0"
    physical_gpus = tf.config.list_physical_devices("GPU")
    if args.device == "cpu":
        if physical_gpus:
            raise RuntimeError("CPU/XLA child unexpectedly sees a GPU")
        tf32_enabled = bool(
            tf.config.experimental.tensor_float_32_execution_enabled()
        )
        trust_basis = "cpu_hidden_debug_reference_exception"
    else:
        if len(physical_gpus) != 1:
            raise RuntimeError(
                f"GPU/XLA child requires exactly one visible GPU, found {len(physical_gpus)}"
            )
        tf.config.experimental.set_memory_growth(physical_gpus[0], True)
        tf.config.experimental.enable_tensor_float_32_execution(True)
        tf32_enabled = bool(
            tf.config.experimental.tensor_float_32_execution_enabled()
        )
        if not tf32_enabled:
            raise RuntimeError("GPU/XLA child failed to enable TF32")
        selected_device = "/GPU:0"
        trust_basis = "owner_designated_managed_session_visible_gpu_trusted"
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from scripts import benchmark_kalman_qr_parameter_count_scaling as benchmark

    _progress(args.progress, args.method, "fixture", started)
    fixture = benchmark.make_fixture(
        args.dimension,
        args.parameter_count,
        args.timesteps,
        dtype=tf.float32,
    )
    parameters = benchmark._make_parameter_batch(fixture, args.batch_size)
    builder = {
        "batch_native_analytical_qr_score": benchmark.build_batch_native_analytic_fn,
        "batch_native_autodiff_qr_score": (
            benchmark.build_batch_native_autodiff_fn
            if args.autodiff_variant == "full_helper"
            else benchmark.build_batch_native_autodiff_value_only_explicit_fn
        ),
    }[args.method]
    selected = builder(
        fixture,
        batch_size=args.batch_size,
        jit_compile=True,
    )

    _progress(args.progress, args.method, "trace", started)
    trace_started = time.perf_counter()
    with tf.device(selected_device):
        concrete = selected.get_concrete_function(parameters)
    trace_seconds = time.perf_counter() - trace_started
    graph = _graph_record(concrete.graph.as_graph_def(add_shapes=True))

    _progress(args.progress, args.method, "first_xla_call", started)
    first_started = time.perf_counter()
    with tf.device(selected_device):
        first_value, first_score = concrete(parameters)
    first_output = _tensor_output(first_value, first_score, tf)
    first_seconds = time.perf_counter() - first_started

    _progress(args.progress, args.method, "warm_xla_call", started)
    warm_started = time.perf_counter()
    with tf.device(selected_device):
        warm_value, warm_score = concrete(tf.identity(parameters))
    warm_output = _tensor_output(warm_value, warm_score, tf)
    warm_seconds = time.perf_counter() - warm_started

    logical_gpus = tf.config.list_logical_devices("GPU")
    gpu_memory_info = None
    if args.device == "gpu":
        gpu_memory_info = {
            key: int(value)
            for key, value in tf.config.experimental.get_memory_info("GPU:0").items()
        }

    record = {
        "state": "passed",
        "method": args.method,
        "autodiff_variant": args.autodiff_variant,
        "stage": "complete",
        "cell": {
            "dimension": args.dimension,
            "timesteps": args.timesteps,
            "parameter_count": args.parameter_count,
            "batch_size": args.batch_size,
            "dtype": "float32",
        },
        "jit_compile": True,
        "requested_device": args.device,
        "device": selected_device,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "physical_gpu_count": len(physical_gpus),
        "physical_gpu_names": [device.name for device in physical_gpus],
        "logical_gpu_count": len(logical_gpus),
        "logical_gpu_names": [device.name for device in logical_gpus],
        "tf32_enabled": tf32_enabled,
        "trust_basis": trust_basis,
        "gpu_memory_info_bytes": gpu_memory_info,
        "tensorflow_version": tf.__version__,
        "graphdef": graph,
        "trace_seconds": trace_seconds,
        "first_xla_call_seconds": first_seconds,
        "warm_xla_call_seconds": warm_seconds,
        "first_output": first_output,
        "warm_output": warm_output,
        "concrete_function_count": len(
            selected._list_all_concrete_functions_for_serialization()
        ),
        "child_wall_seconds": time.perf_counter() - started,
    }
    _write_json(args.child_output, record)
    _progress(args.progress, args.method, "complete", started)
    print(_strict_json({"state": "passed", "method": args.method}))
    return 0


def _parse_time(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    values: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        name, raw = line.split("=", 1)
        values[name] = int(raw) if name in {"max_rss_kb", "exit_status"} else float(raw)
    return values


def _run_method(args: argparse.Namespace, method: str) -> dict[str, Any]:
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{args.device}-d{args.dimension}-t{args.timesteps}-p{args.parameter_count}-"
        f"b{args.batch_size}-{method}"
    )
    child_output = WORK_ROOT / f"{stem}.json"
    progress = WORK_ROOT / f"{stem}.progress.json"
    time_output = WORK_ROOT / f"{stem}.time.txt"
    for path in (child_output, progress, time_output):
        path.unlink(missing_ok=True)

    child_command = [
        sys.executable,
        str(SCRIPT_PATH.relative_to(REPO_ROOT)),
        "--child",
        "--method",
        method,
        "--dimension",
        str(args.dimension),
        "--timesteps",
        str(args.timesteps),
        "--parameter-count",
        str(args.parameter_count),
        "--batch-size",
        str(args.batch_size),
        "--device",
        args.device,
        "--cuda-visible-devices",
        args.cuda_visible_devices,
        "--autodiff-variant",
        args.autodiff_variant,
        "--child-output",
        str(child_output),
        "--progress",
        str(progress),
    ]
    command = [
        "/usr/bin/time",
        "-f",
        "max_rss_kb=%M\nuser_seconds=%U\nsystem_seconds=%S\n"
        "elapsed_seconds=%e\nexit_status=%x",
        "-o",
        str(time_output),
        "timeout",
        "--signal=TERM",
        f"--kill-after={KILL_GRACE_SECONDS}s",
        f"{args.method_timeout_seconds}s",
        *child_command,
    ]
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": (
                "-1" if args.device == "cpu" else args.cuda_visible_devices
            ),
            "OMP_NUM_THREADS": "1",
            "TF_NUM_INTRAOP_THREADS": "1",
            "TF_NUM_INTEROP_THREADS": "1",
            "TF_CPP_MIN_LOG_LEVEL": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=args.method_timeout_seconds + KILL_GRACE_SECONDS + 20,
    )
    child = (
        json.loads(child_output.read_text(encoding="utf-8"))
        if child_output.is_file()
        else None
    )
    last_progress = (
        json.loads(progress.read_text(encoding="utf-8"))
        if progress.is_file()
        else None
    )
    state = "passed" if completed.returncode == 0 and child is not None else "failed"
    if completed.returncode == 124:
        state = "timeout"
    return {
        "state": state,
        "method": method,
        "returncode": completed.returncode,
        "timed_out": completed.returncode == 124,
        "command": command,
        "wall_seconds": time.perf_counter() - started,
        "resource_usage": _parse_time(time_output),
        "last_progress": last_progress,
        "child": child,
        "stdout_total_bytes": len(completed.stdout.encode("utf-8")),
        "stdout_sha256": hashlib.sha256(completed.stdout.encode("utf-8")).hexdigest(),
        "stdout_tail": completed.stdout[-4000:],
        "stderr_total_bytes": len(completed.stderr.encode("utf-8")),
        "stderr_sha256": hashlib.sha256(completed.stderr.encode("utf-8")).hexdigest(),
        "stderr_tail": completed.stderr[-32000:],
    }


def _output_checks(
    records: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    parameter_count: int,
    device: str,
    cuda_visible_devices: str,
    autodiff_variant: str,
) -> dict[str, Any]:
    passed = {
        record["method"]: record
        for record in records
        if record.get("state") == "passed" and isinstance(record.get("child"), Mapping)
    }
    all_methods_completed = set(passed) == set(METHODS) and len(records) == len(METHODS)
    finite_dtype_shape = all_methods_completed and all(
        child_output.get("all_finite") is True
        and child_output.get("value_shape") == [batch_size]
        and child_output.get("score_shape") == [batch_size, parameter_count]
        and child_output.get("value_dtype") == "float32"
        and child_output.get("score_dtype") == "float32"
        and record["child"].get("concrete_function_count") == 1
        for record in passed.values()
        for child_output in (
            record["child"].get("first_output", {}),
            record["child"].get("warm_output", {}),
        )
    )
    expected_device_suffix = "device:CPU:0" if device == "cpu" else "device:GPU:0"
    expected_cuda = "-1" if device == "cpu" else cuda_visible_devices
    device_provenance = all_methods_completed and all(
        record["child"].get("requested_device") == device
        and record["child"].get("autodiff_variant") == autodiff_variant
        and record["child"].get("device") == ("/CPU:0" if device == "cpu" else "/GPU:0")
        and record["child"].get("cuda_visible_devices") == expected_cuda
        and record["child"].get("physical_gpu_count") == (0 if device == "cpu" else 1)
        and record["child"].get("logical_gpu_count") == (0 if device == "cpu" else 1)
        and record["child"].get("trust_basis")
        == (
            "cpu_hidden_debug_reference_exception"
            if device == "cpu"
            else "owner_designated_managed_session_visible_gpu_trusted"
        )
        and (
            device == "cpu"
            or (
                record["child"].get("tf32_enabled") is True
                and isinstance(record["child"].get("gpu_memory_info_bytes"), Mapping)
                and type(record["child"]["gpu_memory_info_bytes"].get("peak")) is int
                and record["child"]["gpu_memory_info_bytes"]["peak"] > 0
            )
        )
        and all(
            isinstance(output_device, str)
            and output_device.endswith(expected_device_suffix)
            for child_output in (
                record["child"].get("first_output", {}),
                record["child"].get("warm_output", {}),
            )
            for output_device in child_output.get("devices", [])
        )
        and all(
            len(child_output.get("devices", [])) == 2
            for child_output in (
                record["child"].get("first_output", {}),
                record["child"].get("warm_output", {}),
            )
        )
        for record in passed.values()
    )
    warm_comparisons: dict[str, Any] = {}
    if all_methods_completed:
        for method, record in passed.items():
            warm_comparisons[method] = {
                "value": _comparison(
                    record["child"]["warm_output"]["value"],
                    record["child"]["first_output"]["value"],
                ),
                "score": _comparison(
                    record["child"]["warm_output"]["score"],
                    record["child"]["first_output"]["score"],
                ),
            }
    first_warm_parity = all_methods_completed and all(
        comparison["value"]["passed"] and comparison["score"]["passed"]
        for comparison in warm_comparisons.values()
    )
    cross_method = None
    if all_methods_completed:
        analytical = passed["batch_native_analytical_qr_score"]["child"]["first_output"]
        autodiff = passed["batch_native_autodiff_qr_score"]["child"]["first_output"]
        cross_method = {
            "value": _comparison(autodiff["value"], analytical["value"]),
            "score": _comparison(autodiff["score"], analytical["score"]),
        }
    cross_method_parity = bool(
        cross_method
        and cross_method["value"]["passed"]
        and cross_method["score"]["passed"]
    )
    return {
        "all_methods_completed": all_methods_completed,
        "finite_dtype_shape_and_single_trace": finite_dtype_shape,
        "device_and_runtime_provenance": device_provenance,
        "first_warm_parity": first_warm_parity,
        "cross_method_parity": cross_method_parity,
        "warm_comparisons": warm_comparisons,
        "cross_method_comparison": cross_method,
    }


def _decision(
    records: Sequence[Mapping[str, Any]], checks: Mapping[str, Any], *, device: str
) -> str:
    if all(checks.get(name) is True for name in (
        "all_methods_completed",
        "finite_dtype_shape_and_single_trace",
        "device_and_runtime_provenance",
        "first_warm_parity",
        "cross_method_parity",
    )):
        return f"{device}_xla_cell_passed"
    by_method = {record.get("method"): record for record in records}
    analytical = by_method.get("batch_native_analytical_qr_score", {})
    autodiff = by_method.get("batch_native_autodiff_qr_score", {})
    if analytical.get("state") != "passed":
        return f"stop_analytical_or_common_{device}_xla_failure"
    if autodiff.get("state") != "passed":
        return "run_autodiff_timestep_localization_ladder"
    return "stop_numerical_or_artifact_validity_failure"


def _parent(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    records = [_run_method(args, method) for method in METHODS]
    checks = _output_checks(
        records,
        batch_size=args.batch_size,
        parameter_count=args.parameter_count,
        device=args.device,
        cuda_visible_devices=args.cuda_visible_devices,
        autodiff_variant=args.autodiff_variant,
    )
    passed = all(
        checks[name] is True
        for name in (
            "all_methods_completed",
            "finite_dtype_shape_and_single_trace",
            "device_and_runtime_provenance",
            "first_warm_parity",
            "cross_method_parity",
        )
    )
    output = args.output.resolve()
    artifact = {
        "schema": SCHEMA,
        "state": "passed" if passed else "failed",
        "decision": _decision(records, checks, device=args.device),
        "question": (
            "Do the unchanged repaired true-batched analytical and autodiff QR "
            f"methods compile and execute under {args.device.upper()}/XLA at this "
            "target-horizon cell?"
        ),
        "cell": {
            "dimension": args.dimension,
            "timesteps": args.timesteps,
            "parameter_count": args.parameter_count,
            "batch_size": args.batch_size,
            "dtype": "float32",
        },
        "checks": checks,
        "records": records,
        "run_manifest": {
            "git_commit": _git_commit(),
            "command": list(getattr(sys, "orig_argv", sys.argv)),
            "python": sys.version,
            "platform": platform.platform(),
            "environment": {
                name: os.environ.get(name)
                for name in (
                    "CUDA_VISIBLE_DEVICES",
                    "OMP_NUM_THREADS",
                    "TF_NUM_INTRAOP_THREADS",
                    "TF_NUM_INTEROP_THREADS",
                    "XLA_FLAGS",
                    "CONDA_DEFAULT_ENV",
                    "CONDA_PREFIX",
                )
            },
            "requested_device": args.device,
            "requested_cuda_visible_devices": (
                "-1" if args.device == "cpu" else args.cuda_visible_devices
            ),
            "autodiff_variant": args.autodiff_variant,
            "device_status": (
                "CPU-only XLA diagnostic; GPU deliberately hidden"
                if args.device == "cpu"
                else "managed-session visible GPU/XLA diagnostic"
            ),
            "trust_basis": (
                "cpu_hidden_debug_reference_exception"
                if args.device == "cpu"
                else "owner_designated_managed_session_visible_gpu_trusted"
            ),
            "jit_compile": True,
            "tf32_status": (
                "recorded by each child; not applicable to CPU arithmetic"
                if args.device == "cpu"
                else "enabled and recorded by each child"
            ),
            "random_seeds": "N/A; deterministic fixture and parameter cloud",
            "data_version": "synthetic nested Kalman QR fixture in benchmark source",
            "wall_time_seconds": time.perf_counter() - started,
            "output_artifact": str(output.relative_to(REPO_ROOT)),
            "plan_file": str(PLAN_PATH.relative_to(REPO_ROOT)),
            "result_file": str(RESULT_PATH.relative_to(REPO_ROOT)),
            "source_sha256": {
                str(SCRIPT_PATH.relative_to(REPO_ROOT)): _sha256(SCRIPT_PATH),
                str(BENCHMARK_PATH.relative_to(REPO_ROOT)): _sha256(BENCHMARK_PATH),
                str(PLAN_PATH.relative_to(REPO_ROOT)): _sha256(PLAN_PATH),
            },
        },
        "evidence_roles": {
            "method_completion_and_numerical_checks": "continuation_gate",
            "timeout_crash_or_nonfinite": "continuation_or_repair_trigger_by_method",
            "graphdef_structure": "explanatory_only",
            "peak_rss_and_elapsed_seconds": "explanatory_only",
        },
        "nonclaims": [
            (
                "no GPU viability conclusion"
                if args.device == "cpu"
                else "no full GPU lattice viability conclusion"
            ),
            "no full dimension/parameter/batch lattice readiness",
            "no speed or memory superiority ranking",
            "no HMC or posterior correctness",
            "no production/default readiness or scientific-validity claim",
        ],
    }
    _write_json(output, artifact)
    print(
        _strict_json(
            {
                "state": artifact["state"],
                "decision": artifact["decision"],
                "cell": artifact["cell"],
                "checks": checks,
                "output": str(output),
            },
            indent=2,
        )
    )
    return 0 if passed else 1


def _fake_record(
    method: str,
    value: float,
    score: list[float],
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    device_name = "/CPU:0" if device == "cpu" else "/GPU:0"
    full_device_name = (
        f"/job:localhost/replica:0/task:0{device_name.replace('/', '/device:')}"
    )
    output = {
        "value": [value],
        "score": [score],
        "value_shape": [1],
        "score_shape": [1, len(score)],
        "value_dtype": "float32",
        "score_dtype": "float32",
        "devices": [full_device_name, full_device_name],
        "all_finite": True,
    }
    return {
        "state": "passed",
        "method": method,
        "child": {
            "first_output": output,
            "warm_output": dict(output),
            "concrete_function_count": 1,
            "requested_device": device,
            "device": device_name,
            "cuda_visible_devices": "-1" if device == "cpu" else "1",
            "physical_gpu_count": 0 if device == "cpu" else 1,
            "logical_gpu_count": 0 if device == "cpu" else 1,
            "tf32_enabled": device == "gpu",
            "trust_basis": (
                "cpu_hidden_debug_reference_exception"
                if device == "cpu"
                else "owner_designated_managed_session_visible_gpu_trusted"
            ),
            "gpu_memory_info_bytes": None if device == "cpu" else {"peak": 1024},
            "autodiff_variant": "full_helper",
        },
    }


def _self_check() -> int:
    valid = [
        _fake_record("batch_native_analytical_qr_score", -2.0, [0.1, 0.2]),
        _fake_record("batch_native_autodiff_qr_score", -2.0, [0.1, 0.2]),
    ]
    valid_checks = _output_checks(
        valid,
        batch_size=1,
        parameter_count=2,
        device="cpu",
        cuda_visible_devices="-1",
        autodiff_variant="full_helper",
    )
    failed_method = json.loads(_strict_json(valid))
    failed_method[1]["state"] = "timeout"
    failed_checks = _output_checks(
        failed_method,
        batch_size=1,
        parameter_count=2,
        device="cpu",
        cuda_visible_devices="-1",
        autodiff_variant="full_helper",
    )
    parity_failure = json.loads(_strict_json(valid))
    parity_failure[1]["child"]["first_output"]["score"][0][0] = 1.0
    parity_failure[1]["child"]["warm_output"]["score"][0][0] = 1.0
    parity_checks = _output_checks(
        parity_failure,
        batch_size=1,
        parameter_count=2,
        device="cpu",
        cuda_visible_devices="-1",
        autodiff_variant="full_helper",
    )
    gpu_valid = [
        _fake_record(
            "batch_native_analytical_qr_score", -2.0, [0.1, 0.2], device="gpu"
        ),
        _fake_record(
            "batch_native_autodiff_qr_score", -2.0, [0.1, 0.2], device="gpu"
        ),
    ]
    gpu_checks = _output_checks(
        gpu_valid,
        batch_size=1,
        parameter_count=2,
        device="gpu",
        cuda_visible_devices="1",
        autodiff_variant="full_helper",
    )
    gpu_fallback = json.loads(_strict_json(gpu_valid))
    gpu_fallback[1]["child"]["first_output"]["devices"] = [
        "/job:localhost/replica:0/task:0/device:CPU:0",
        "/job:localhost/replica:0/task:0/device:CPU:0",
    ]
    gpu_fallback_checks = _output_checks(
        gpu_fallback,
        batch_size=1,
        parameter_count=2,
        device="gpu",
        cuda_visible_devices="1",
        autodiff_variant="full_helper",
    )
    checks = {
        "valid_pair_passes": all(
            valid_checks[name] is True
            for name in (
                "all_methods_completed",
                "finite_dtype_shape_and_single_trace",
                "device_and_runtime_provenance",
                "first_warm_parity",
                "cross_method_parity",
            )
        ),
        "failed_method_fails_closed": failed_checks["all_methods_completed"] is False,
        "parity_failure_fails_closed": parity_checks["cross_method_parity"] is False,
        "gpu_provenance_passes": gpu_checks["device_and_runtime_provenance"] is True,
        "gpu_cpu_fallback_fails_closed": gpu_fallback_checks[
            "device_and_runtime_provenance"
        ]
        is False,
        "failure_decision_localizes_autodiff": _decision(
            failed_method, failed_checks, device="cpu"
        )
        == "run_autodiff_timestep_localization_ladder",
    }
    print(_strict_json(checks, indent=2))
    return 0 if all(checks.values()) else 1


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=int, default=10)
    parser.add_argument("--timesteps", type=int, default=120)
    parser.add_argument("--parameter-count", type=int, default=50)
    parser.add_argument("--batch-size", type=int, choices=(1, 4, 16), default=1)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--cuda-visible-devices", default="-1")
    parser.add_argument(
        "--autodiff-variant",
        choices=("full_helper", "value_only_explicit"),
        default="full_helper",
    )
    parser.add_argument("--method-timeout-seconds", type=int, default=300)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--child-output", type=Path)
    parser.add_argument("--progress", type=Path)
    args = parser.parse_args(argv)
    for name in ("dimension", "timesteps", "parameter_count", "method_timeout_seconds"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.child and None in (args.method, args.child_output, args.progress):
        parser.error("--child requires --method, --child-output, and --progress")
    if args.child and args.self_check:
        parser.error("--child and --self-check are mutually exclusive")
    if args.device == "cpu" and args.cuda_visible_devices != "-1":
        parser.error("CPU mode requires --cuda-visible-devices=-1")
    if args.device == "gpu" and args.cuda_visible_devices == "-1":
        parser.error("GPU mode requires an explicit visible GPU")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.self_check:
        return _self_check()
    if args.child:
        return _child(args)
    return _parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
