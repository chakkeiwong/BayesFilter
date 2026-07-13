#!/usr/bin/env python
"""Compare true-batched and TensorFlow-mapped row VJPs for Kalman QR."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import diagnose_kalman_qr_output_seed_counterfactual_2026_07_13 as common


REPO_ROOT = common.REPO_ROOT
SCRIPT_PATH = Path(__file__).resolve()
BENCHMARK_PATH = REPO_ROOT / "scripts/benchmark_kalman_qr_parameter_count_scaling.py"
PLAN_PATH = REPO_ROOT / (
    "docs/plans/bayesfilter-kalman-qr-batched-xla-lean-repair-plan-2026-07-13.md"
)
RESULT_PATH = REPO_ROOT / (
    "docs/plans/bayesfilter-kalman-qr-batched-xla-lean-repair-result-2026-07-13.md"
)
DEFAULT_OUTPUT = REPO_ROOT / (
    "docs/benchmarks/kalman_qr_mapped_row_vjp_counterfactual_2026-07-13.json"
)
SCHEMA = "bayesfilter.kalman_qr.mapped_row_vjp_counterfactual.v1"
METHODS = ("true_batched_vjp", "mapped_row_vjp")


def build_mapped_row_autodiff_fn(
    benchmark: Any,
    fixture: Any,
    *,
    batch_size: int,
    jit_compile: bool,
) -> Any:
    """Diagnostic-only TensorFlow map of singleton-batch VJPs."""

    import tensorflow as tf
    from bayesfilter.linear.kalman_qr_tf import (
        tf_qr_sqrt_kalman_log_likelihood_batched_static_while_loop,
    )

    @tf.function(
        jit_compile=jit_compile,
        reduce_retracing=True,
        input_signature=[
            tf.TensorSpec(
                [batch_size, fixture.parameter_count],
                fixture.dtype,
                name="parameters_batch",
            )
        ],
    )
    def mapped_row_autodiff_score(parameters_batch: Any) -> tuple[Any, Any]:
        params = tf.convert_to_tensor(parameters_batch, dtype=fixture.dtype)

        def row_value_score(row_params: Any) -> tuple[Any, Any]:
            with tf.GradientTape() as tape:
                tape.watch(row_params)
                tensors = benchmark._batched_model_tensors(
                    fixture, row_params[tf.newaxis, :]
                )
                row_values = (
                    tf_qr_sqrt_kalman_log_likelihood_batched_static_while_loop.python_function(
                        observations=fixture.observations,
                        transition_offset=tensors[0],
                        transition_matrix=tensors[1],
                        transition_covariance=tensors[2],
                        observation_offset=tensors[3],
                        observation_matrix=tensors[4],
                        observation_covariance=tensors[5],
                        initial_state_mean=tensors[6],
                        initial_state_covariance=tensors[7],
                        jitter=tf.constant(1.0e-9, dtype=fixture.dtype),
                        jitter_updates_filtered_covariance=True,
                    )
                )
                row_value = row_values[0]
            row_score = tape.gradient(row_value, row_params)
            if row_score is None:
                raise RuntimeError("mapped-row QR likelihood gradient is disconnected")
            return row_value, row_score

        return tf.map_fn(
            row_value_score,
            params,
            fn_output_signature=(
                tf.TensorSpec([], fixture.dtype),
                tf.TensorSpec([fixture.parameter_count], fixture.dtype),
            ),
            parallel_iterations=1,
        )

    return mapped_row_autodiff_score


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
        common.DIMENSION,
        parameter_count,
        common.TIMESTEPS,
        dtype=tf.float32,
    )
    params = benchmark._make_parameter_batch(fixture, batch_size)
    if method == "true_batched_vjp":
        selected = benchmark.build_batch_native_autodiff_fn(
            fixture, batch_size=batch_size, jit_compile=False
        )
    else:
        selected = build_mapped_row_autodiff_fn(
            benchmark, fixture, batch_size=batch_size, jit_compile=False
        )
    trace_started = time.perf_counter()
    concrete = selected.get_concrete_function()
    trace_seconds = time.perf_counter() - trace_started
    graph_def = concrete.graph.as_graph_def(add_shapes=True)
    value, score = selected(params)
    output = common._tensor_output(value, score)

    perturbed_output = None
    perturbed_params = None
    if batch_size == 4:
        axis = tf.cast(tf.range(parameter_count) + 1, tf.float32)
        perturbation = tf.constant(1.0e-3, tf.float32) * tf.math.sin(
            tf.constant(0.17, tf.float32) * axis
        )
        perturbed_params = tf.tensor_scatter_nd_add(params, [[2]], [perturbation])
        perturbed_value, perturbed_score = selected(perturbed_params)
        perturbed_output = common._tensor_output(perturbed_value, perturbed_score)

    analytical = benchmark.build_batch_native_analytic_fn(
        fixture,
        batch_size=batch_size,
        jit_compile=False,
    )
    analytical_value, analytical_score = analytical(params)
    analytical_output = common._tensor_output(analytical_value, analytical_score)
    analytical_perturbed_output = None
    if perturbed_params is not None:
        analytical_value, analytical_score = analytical(perturbed_params)
        analytical_perturbed_output = common._tensor_output(
            analytical_value, analytical_score
        )

    graph = common._graph_record(graph_def)
    payload = {
        "method": method,
        "parameter_count": parameter_count,
        "batch_size": batch_size,
        "dtype": "float32",
        "dimension": common.DIMENSION,
        "timesteps": common.TIMESTEPS,
        "tensorflow_version": tf.__version__,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "jit_compile": False,
        "trace_seconds": trace_seconds,
        "graphdef": graph,
        "output": output,
        "perturbed_output": perturbed_output,
        "analytical_output": analytical_output,
        "analytical_perturbed_output": analytical_perturbed_output,
    }
    print(common._strict_json(payload))
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
    if completed.returncode != 0:
        raise RuntimeError(
            f"child failed for method={method}, P={parameter_count}, B={batch_size}: "
            + completed.stderr[-4000:]
        )
    row = json.loads(completed.stdout)
    row["child_command"] = command
    row["child_wall_seconds"] = time.perf_counter() - started
    row["stderr_tail"] = completed.stderr[-1000:]
    return row


def _strip_outputs(child: dict[str, Any]) -> dict[str, Any]:
    output_names = (
        "output",
        "perturbed_output",
        "analytical_output",
        "analytical_perturbed_output",
    )
    result = {key: value for key, value in child.items() if key not in output_names}
    for name in output_names:
        if child[name] is not None:
            result[f"{name}_digest"] = common._output_digest(child[name])
    return result


def _metadata_passed(output: dict[str, Any], batch_size: int, parameter_count: int) -> bool:
    return (
        output["value_shape"] == [batch_size]
        and output["score_shape"] == [batch_size, parameter_count]
        and output["value_dtype"] == "float32"
        and output["score_dtype"] == "float32"
        and output["all_finite"]
    )


def _cell_record(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    baseline_output = baseline["output"]
    candidate_output = candidate["output"]
    analytical_output = candidate["analytical_output"]
    checks: dict[str, Any] = {
        "baseline_metadata": _metadata_passed(
            baseline_output, baseline["batch_size"], baseline["parameter_count"]
        ),
        "candidate_metadata": _metadata_passed(
            candidate_output, candidate["batch_size"], candidate["parameter_count"]
        ),
        "value_candidate_vs_baseline": common._comparison(
            candidate_output["value"], baseline_output["value"]
        ),
        "score_candidate_vs_baseline": common._comparison(
            candidate_output["score"], baseline_output["score"]
        ),
        "value_candidate_vs_analytical": common._comparison(
            candidate_output["value"], analytical_output["value"]
        ),
        "score_candidate_vs_analytical": common._comparison(
            candidate_output["score"], analytical_output["score"]
        ),
    }
    if candidate["batch_size"] == 4:
        baseline_perturbed = baseline["perturbed_output"]
        candidate_perturbed = candidate["perturbed_output"]
        analytical_perturbed = candidate["analytical_perturbed_output"]
        unaffected = (0, 1, 3)
        checks.update(
            {
                "perturbed_value_candidate_vs_baseline": common._comparison(
                    candidate_perturbed["value"], baseline_perturbed["value"]
                ),
                "perturbed_score_candidate_vs_baseline": common._comparison(
                    candidate_perturbed["score"], baseline_perturbed["score"]
                ),
                "perturbed_value_candidate_vs_analytical": common._comparison(
                    candidate_perturbed["value"], analytical_perturbed["value"]
                ),
                "perturbed_score_candidate_vs_analytical": common._comparison(
                    candidate_perturbed["score"], analytical_perturbed["score"]
                ),
                "unaffected_value_row_independence": common._comparison(
                    [candidate_perturbed["value"][index] for index in unaffected],
                    [candidate_output["value"][index] for index in unaffected],
                ),
                "unaffected_score_row_independence": common._comparison(
                    [candidate_perturbed["score"][index] for index in unaffected],
                    [candidate_output["score"][index] for index in unaffected],
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
    structural = {
        "serialized_bytes_reduction_percent": common._reduction_percent(
            baseline["graphdef"]["serialized_bytes"],
            candidate["graphdef"]["serialized_bytes"],
        ),
        "total_node_count_reduction_percent": common._reduction_percent(
            baseline["graphdef"]["total_node_count"],
            candidate["graphdef"]["total_node_count"],
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


def _parent(output: Path) -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("parent requires CUDA_VISIBLE_DEVICES=-1")
    started = time.perf_counter()
    children: dict[tuple[int, int, str], dict[str, Any]] = {}
    for parameter_count in common.PARAMETER_COUNTS:
        for batch_size in common.BATCH_SIZES:
            for method in METHODS:
                children[(parameter_count, batch_size, method)] = _run_child(
                    method, parameter_count, batch_size
                )
    cells = [
        _cell_record(
            children[(parameter_count, batch_size, "true_batched_vjp")],
            children[(parameter_count, batch_size, "mapped_row_vjp")],
        )
        for parameter_count in common.PARAMETER_COUNTS
        for batch_size in common.BATCH_SIZES
    ]
    all_correct = all(cell["correctness_passed"] for cell in cells)
    no_regression = all(
        value >= -1.0
        for cell in cells
        for value in cell["structural_delta"].values()
    )
    p150_material = all(
        max(cell["structural_delta"].values()) >= 5.0
        for cell in cells
        if cell["parameter_count"] == 150
    )
    nominated = all_correct and no_regression and p150_material
    artifact = {
        "schema": SCHEMA,
        "state": "passed" if all_correct else "failed",
        "decision": (
            "nominate_mapped_row_vjp_for_bounded_xla_compile"
            if nominated
            else "reject_mapped_row_vjp_candidate"
        ),
        "question": (
            "Does a TensorFlow-native mapped singleton-row VJP reduce graph "
            "burden while preserving true-batched Kalman QR semantics?"
        ),
        "checks": {
            "all_correctness_checks_passed": all_correct,
            "no_graph_metric_regressed_over_one_percent": no_regression,
            "both_p150_cells_met_material_reduction": p150_material,
            "candidate_nominated": nominated,
        },
        "cells": cells,
        "thresholds": {
            "p150_minimum_reduction_percent": 5.0,
            "maximum_per_cell_regression_percent": 1.0,
            "rtol": common.RTOL,
            "atol": common.ATOL,
        },
        "evidence_roles": {
            "correctness": "promotion_veto",
            "graphdef_size": "nomination_only",
            "trace_time": "explanatory_only",
        },
        "run_manifest": {
            "git_commit": common._git_commit(),
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
            "dimension": common.DIMENSION,
            "timesteps": common.TIMESTEPS,
            "parameter_counts": list(common.PARAMETER_COUNTS),
            "batch_sizes": list(common.BATCH_SIZES),
            "random_seeds": "N/A; deterministic fixture and parameter cloud",
            "data_version": "synthetic nested Kalman QR fixture in benchmark source",
            "wall_time_seconds": time.perf_counter() - started,
            "output_artifact": str(output.relative_to(REPO_ROOT)),
            "plan_file": str(PLAN_PATH.relative_to(REPO_ROOT)),
            "result_file": str(RESULT_PATH.relative_to(REPO_ROOT)),
            "source_sha256": {
                str(SCRIPT_PATH.relative_to(REPO_ROOT)): common._sha256(SCRIPT_PATH),
                str(BENCHMARK_PATH.relative_to(REPO_ROOT)): common._sha256(
                    BENCHMARK_PATH
                ),
                str(PLAN_PATH.relative_to(REPO_ROOT)): common._sha256(PLAN_PATH),
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
    temporary.write_text(common._strict_json(artifact, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(
        common._strict_json(
            {
                "state": artifact["state"],
                "decision": artifact["decision"],
                "checks": artifact["checks"],
            },
            indent=2,
        )
    )
    return 0 if all_correct else 1


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument(
        "--parameter-count", type=int, choices=common.PARAMETER_COUNTS
    )
    parser.add_argument("--batch-size", type=int, choices=common.BATCH_SIZES)
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
