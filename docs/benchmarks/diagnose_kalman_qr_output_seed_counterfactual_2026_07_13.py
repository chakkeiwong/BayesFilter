#!/usr/bin/env python
"""Compare explicit all-ones and reduce-sum VJP construction for Kalman QR."""

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
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = Path(__file__).resolve()
BENCHMARK_PATH = REPO_ROOT / "scripts/benchmark_kalman_qr_parameter_count_scaling.py"
PLAN_PATH = REPO_ROOT / (
    "docs/plans/bayesfilter-kalman-qr-batched-xla-lean-repair-plan-2026-07-13.md"
)
RESULT_PATH = REPO_ROOT / (
    "docs/plans/bayesfilter-kalman-qr-batched-xla-lean-repair-result-2026-07-13.md"
)
DEFAULT_OUTPUT = REPO_ROOT / (
    "docs/benchmarks/kalman_qr_output_seed_counterfactual_2026-07-13.json"
)
SCHEMA = "bayesfilter.kalman_qr.output_seed_counterfactual.v1"
DIMENSION = 10
TIMESTEPS = 8
PARAMETER_COUNTS = (50, 150)
BATCH_SIZES = (1, 4)
METHODS = ("explicit_ones_seed", "reduce_sum")
RTOL = 2.0e-4
ATOL = 2.0e-4


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strict_json(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        indent=indent,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
    )


def _flatten(value: Any) -> list[float]:
    if isinstance(value, list):
        result: list[float] = []
        for item in value:
            result.extend(_flatten(item))
        return result
    return [float(value)]


def _comparison(actual: Any, expected: Any) -> dict[str, Any]:
    actual_flat = _flatten(actual)
    expected_flat = _flatten(expected)
    if len(actual_flat) != len(expected_flat):
        return {
            "passed": False,
            "actual_count": len(actual_flat),
            "expected_count": len(expected_flat),
            "max_abs_residual": None,
        }
    residuals = [abs(left - right) for left, right in zip(actual_flat, expected_flat)]
    limits = [ATOL + RTOL * abs(reference) for reference in expected_flat]
    return {
        "passed": all(
            math.isfinite(left)
            and math.isfinite(right)
            and residual <= limit
            for left, right, residual, limit in zip(
                actual_flat, expected_flat, residuals, limits
            )
        ),
        "actual_count": len(actual_flat),
        "expected_count": len(expected_flat),
        "max_abs_residual": max(residuals, default=0.0),
        "rtol": RTOL,
        "atol": ATOL,
    }


def _output_digest(output: dict[str, Any]) -> str:
    return hashlib.sha256(_strict_json(output).encode("utf-8")).hexdigest()


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
                "BroadcastGradientArgs",
                "Fill",
                "OnesLike",
                "StatelessWhile",
                "Sum",
                "While",
            )
        },
    }


def _tensor_output(value: Any, score: Any) -> dict[str, Any]:
    return {
        "value": value.numpy().tolist(),
        "score": score.numpy().tolist(),
        "value_shape": value.shape.as_list(),
        "score_shape": score.shape.as_list(),
        "value_dtype": value.dtype.name,
        "score_dtype": score.dtype.name,
        "all_finite": bool(
            value.dtype.is_floating
            and score.dtype.is_floating
            and __import__("tensorflow").reduce_all(
                __import__("tensorflow").math.is_finite(value)
            )
            and __import__("tensorflow").reduce_all(
                __import__("tensorflow").math.is_finite(score)
            )
        ),
    }


def _child(method: str, parameter_count: int, batch_size: int) -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("child requires CUDA_VISIBLE_DEVICES=-1")

    import tensorflow as tf

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from scripts import benchmark_kalman_qr_parameter_count_scaling as benchmark

    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    fixture = benchmark.make_fixture(
        DIMENSION,
        parameter_count,
        TIMESTEPS,
        dtype=tf.float32,
    )
    params = benchmark._make_parameter_batch(fixture, batch_size)
    builder = (
        benchmark.build_batch_native_autodiff_fn
        if method == "explicit_ones_seed"
        else benchmark.build_batch_native_autodiff_reduce_sum_fn
    )
    selected = builder(fixture, batch_size=batch_size, jit_compile=False)
    trace_started = time.perf_counter()
    concrete = selected.get_concrete_function()
    trace_seconds = time.perf_counter() - trace_started
    graph_def = concrete.graph.as_graph_def(add_shapes=True)
    value, score = selected(params)
    output = _tensor_output(value, score)

    perturbed_output = None
    if batch_size == 4:
        axis = tf.cast(tf.range(parameter_count) + 1, tf.float32)
        perturbation = tf.constant(1.0e-3, tf.float32) * tf.math.sin(
            tf.constant(0.17, tf.float32) * axis
        )
        perturbed_params = tf.tensor_scatter_nd_add(params, [[2]], [perturbation])
        perturbed_value, perturbed_score = selected(perturbed_params)
        perturbed_output = _tensor_output(perturbed_value, perturbed_score)

    analytical_output = None
    analytical_perturbed_output = None
    if method == "reduce_sum":
        analytical = benchmark.build_batch_native_analytic_fn(
            fixture,
            batch_size=batch_size,
            jit_compile=False,
        )
        analytical_value, analytical_score = analytical(params)
        analytical_output = _tensor_output(analytical_value, analytical_score)
        if batch_size == 4:
            analytical_value, analytical_score = analytical(perturbed_params)
            analytical_perturbed_output = _tensor_output(
                analytical_value, analytical_score
            )

    payload = {
        "method": method,
        "parameter_count": parameter_count,
        "batch_size": batch_size,
        "dtype": "float32",
        "dimension": DIMENSION,
        "timesteps": TIMESTEPS,
        "tensorflow_version": tf.__version__,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "jit_compile": False,
        "trace_seconds": trace_seconds,
        "graphdef": _graph_record(graph_def),
        "output": output,
        "perturbed_output": perturbed_output,
        "analytical_output": analytical_output,
        "analytical_perturbed_output": analytical_perturbed_output,
    }
    print(_strict_json(payload))
    return 0


def _run_child(method: str, parameter_count: int, batch_size: int) -> dict[str, Any]:
    command = [
        sys.executable,
        str(SCRIPT_PATH.relative_to(REPO_ROOT)),
        "--child",
        "--method",
        method,
        "--parameter-count",
        str(parameter_count),
        "--batch-size",
        str(batch_size),
    ]
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "-1",
            "OMP_NUM_THREADS": "1",
            "TF_NUM_INTRAOP_THREADS": "1",
            "TF_NUM_INTEROP_THREADS": "1",
            "TF_CPP_MIN_LOG_LEVEL": "2",
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
        timeout=180,
    )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"child failed for method={method}, P={parameter_count}, B={batch_size}: "
            + completed.stderr[-4000:]
        )
    row = json.loads(completed.stdout)
    row["child_command"] = command
    row["child_wall_seconds"] = elapsed
    row["stderr_tail"] = completed.stderr[-1000:]
    return row


def _strip_outputs(child: dict[str, Any]) -> dict[str, Any]:
    result = {
        key: value
        for key, value in child.items()
        if key
        not in {
            "output",
            "perturbed_output",
            "analytical_output",
            "analytical_perturbed_output",
        }
    }
    for name in (
        "output",
        "perturbed_output",
        "analytical_output",
        "analytical_perturbed_output",
    ):
        if child[name] is not None:
            result[f"{name}_digest"] = _output_digest(child[name])
    return result


def _reduction_percent(baseline: int, candidate: int) -> float:
    return 100.0 * (baseline - candidate) / baseline


def _cell_record(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    baseline_output = baseline["output"]
    candidate_output = candidate["output"]
    analytical_output = candidate["analytical_output"]
    checks: dict[str, Any] = {
        "baseline_metadata": (
            baseline_output["value_shape"] == [baseline["batch_size"]]
            and baseline_output["score_shape"]
            == [baseline["batch_size"], baseline["parameter_count"]]
            and baseline_output["value_dtype"] == "float32"
            and baseline_output["score_dtype"] == "float32"
            and baseline_output["all_finite"]
        ),
        "candidate_metadata": (
            candidate_output["value_shape"] == [candidate["batch_size"]]
            and candidate_output["score_shape"]
            == [candidate["batch_size"], candidate["parameter_count"]]
            and candidate_output["value_dtype"] == "float32"
            and candidate_output["score_dtype"] == "float32"
            and candidate_output["all_finite"]
        ),
        "value_candidate_vs_baseline": _comparison(
            candidate_output["value"], baseline_output["value"]
        ),
        "score_candidate_vs_baseline": _comparison(
            candidate_output["score"], baseline_output["score"]
        ),
        "value_candidate_vs_analytical": _comparison(
            candidate_output["value"], analytical_output["value"]
        ),
        "score_candidate_vs_analytical": _comparison(
            candidate_output["score"], analytical_output["score"]
        ),
    }
    if candidate["batch_size"] == 4:
        baseline_perturbed = baseline["perturbed_output"]
        candidate_perturbed = candidate["perturbed_output"]
        analytical_perturbed = candidate["analytical_perturbed_output"]
        unaffected = (0, 1, 3)
        candidate_unaffected_values = [
            candidate_perturbed["value"][index] for index in unaffected
        ]
        candidate_unaffected_scores = [
            candidate_perturbed["score"][index] for index in unaffected
        ]
        original_unaffected_values = [
            candidate_output["value"][index] for index in unaffected
        ]
        original_unaffected_scores = [
            candidate_output["score"][index] for index in unaffected
        ]
        checks.update(
            {
                "perturbed_value_candidate_vs_baseline": _comparison(
                    candidate_perturbed["value"], baseline_perturbed["value"]
                ),
                "perturbed_score_candidate_vs_baseline": _comparison(
                    candidate_perturbed["score"], baseline_perturbed["score"]
                ),
                "perturbed_value_candidate_vs_analytical": _comparison(
                    candidate_perturbed["value"], analytical_perturbed["value"]
                ),
                "perturbed_score_candidate_vs_analytical": _comparison(
                    candidate_perturbed["score"], analytical_perturbed["score"]
                ),
                "unaffected_value_row_independence": _comparison(
                    candidate_unaffected_values, original_unaffected_values
                ),
                "unaffected_score_row_independence": _comparison(
                    candidate_unaffected_scores, original_unaffected_scores
                ),
                "perturbed_row_changed": (
                    candidate_perturbed["value"][2] != candidate_output["value"][2]
                    or candidate_perturbed["score"][2]
                    != candidate_output["score"][2]
                ),
            }
        )
    correctness_passed = all(
        value if isinstance(value, bool) else value["passed"]
        for value in checks.values()
    )
    baseline_graph = baseline["graphdef"]
    candidate_graph = candidate["graphdef"]
    structural = {
        "serialized_bytes_reduction_percent": _reduction_percent(
            baseline_graph["serialized_bytes"], candidate_graph["serialized_bytes"]
        ),
        "total_node_count_reduction_percent": _reduction_percent(
            baseline_graph["total_node_count"], candidate_graph["total_node_count"]
        ),
    }
    return {
        "parameter_count": baseline["parameter_count"],
        "batch_size": baseline["batch_size"],
        "correctness_passed": correctness_passed,
        "checks": checks,
        "structural_delta": structural,
        "baseline": _strip_outputs(baseline),
        "candidate": _strip_outputs(candidate),
    }


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def _parent(output: Path) -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("parent requires CUDA_VISIBLE_DEVICES=-1")
    started = time.perf_counter()
    children: dict[tuple[int, int, str], dict[str, Any]] = {}
    for parameter_count in PARAMETER_COUNTS:
        for batch_size in BATCH_SIZES:
            for method in METHODS:
                children[(parameter_count, batch_size, method)] = _run_child(
                    method, parameter_count, batch_size
                )
    cells = [
        _cell_record(
            children[(parameter_count, batch_size, "explicit_ones_seed")],
            children[(parameter_count, batch_size, "reduce_sum")],
        )
        for parameter_count in PARAMETER_COUNTS
        for batch_size in BATCH_SIZES
    ]
    all_correct = all(cell["correctness_passed"] for cell in cells)
    no_regression = all(
        cell["structural_delta"][metric] >= -1.0
        for cell in cells
        for metric in (
            "serialized_bytes_reduction_percent",
            "total_node_count_reduction_percent",
        )
    )
    p150_material = all(
        max(cell["structural_delta"].values()) >= 5.0
        for cell in cells
        if cell["parameter_count"] == 150
    )
    nominated = all_correct and no_regression and p150_material
    finished = time.perf_counter()
    artifact = {
        "schema": SCHEMA,
        "state": "passed" if all_correct else "failed",
        "decision": (
            "nominate_for_bounded_xla_compile"
            if nominated
            else "reject_output_seed_candidate_and_test_broadcast_gradients"
        ),
        "question": (
            "Does differentiating reduce_sum(value) reduce true-batched autodiff "
            "graph burden without changing Kalman QR value/score semantics?"
        ),
        "evidence_roles": {
            "correctness": "promotion_veto",
            "graphdef_size": "nomination_only",
            "trace_time": "explanatory_only",
        },
        "thresholds": {
            "p150_minimum_reduction_percent": 5.0,
            "maximum_per_cell_regression_percent": 1.0,
            "rtol": RTOL,
            "atol": ATOL,
        },
        "checks": {
            "all_correctness_checks_passed": all_correct,
            "no_graph_metric_regressed_over_one_percent": no_regression,
            "both_p150_cells_met_material_reduction": p150_material,
            "candidate_nominated": nominated,
        },
        "cells": cells,
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
                    "TF_CPP_MIN_LOG_LEVEL",
                    "PYTHONDONTWRITEBYTECODE",
                )
            },
            "device_status": "CPU-only diagnostic; GPU intentionally hidden",
            "jit_compile": False,
            "dtype": "float32",
            "dimension": DIMENSION,
            "timesteps": TIMESTEPS,
            "parameter_counts": list(PARAMETER_COUNTS),
            "batch_sizes": list(BATCH_SIZES),
            "random_seeds": "N/A; deterministic fixture and parameter cloud",
            "data_version": "synthetic nested Kalman QR fixture in benchmark source",
            "wall_time_seconds": finished - started,
            "output_artifact": str(output.relative_to(REPO_ROOT)),
            "plan_file": str(PLAN_PATH.relative_to(REPO_ROOT)),
            "result_file": str(RESULT_PATH.relative_to(REPO_ROOT)),
            "source_sha256": {
                str(SCRIPT_PATH.relative_to(REPO_ROOT)): _sha256(SCRIPT_PATH),
                str(BENCHMARK_PATH.relative_to(REPO_ROOT)): _sha256(BENCHMARK_PATH),
                str(PLAN_PATH.relative_to(REPO_ROOT)): _sha256(PLAN_PATH),
            },
        },
        "nonclaims": [
            "no measured compile-memory repair",
            "no XLA or GPU viability conclusion",
            "no runtime ranking",
            "no production/default readiness",
            "no HMC, posterior, or scientific-validity claim",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(_strict_json(artifact, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(_strict_json({"state": artifact["state"], "decision": artifact["decision"], "checks": artifact["checks"]}, indent=2))
    return 0 if all_correct else 1


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--parameter-count", type=int, choices=PARAMETER_COUNTS)
    parser.add_argument("--batch-size", type=int, choices=BATCH_SIZES)
    args = parser.parse_args(argv)
    if args.child and None in (args.method, args.parameter_count, args.batch_size):
        parser.error("--child requires --method, --parameter-count, and --batch-size")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.child:
        return _child(args.method, args.parameter_count, args.batch_size)
    return _parent(args.output.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
