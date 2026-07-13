#!/usr/bin/env python
"""Run isolated CPU/XLA compiles for the Kalman model-construction candidate."""

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
TRACE_ARTIFACT = REPO_ROOT / (
    "docs/benchmarks/kalman_qr_model_construction_counterfactual_2026-07-13.json"
)
DEFAULT_OUTPUT = REPO_ROOT / (
    "docs/benchmarks/kalman_qr_model_construction_cpu_xla_2026-07-13.json"
)
WORK_ROOT = Path("/tmp/kalman_qr_model_construction_cpu_xla")
SCHEMA = "bayesfilter.kalman_qr.model_construction_cpu_xla.v1"
METHODS = ("full_helper", "value_only_explicit")
DIMENSION = 10
TIMESTEPS = 8
PARAMETER_COUNT = 150
BATCH_SIZE = 4
METHOD_TIMEOUT_SECONDS = 300
KILL_GRACE_SECONDS = 10


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        common._strict_json(value, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _progress(path: Path, method: str, stage: str, started: float) -> None:
    _write_json(
        path,
        {
            "method": method,
            "stage": stage,
            "elapsed_seconds": time.perf_counter() - started,
        },
    )


def _child(method: str, output: Path, progress: Path) -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("CPU/XLA child requires CUDA_VISIBLE_DEVICES=-1")
    started = time.perf_counter()
    _progress(progress, method, "before_tensorflow_import", started)

    import tensorflow as tf

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from scripts import benchmark_kalman_qr_parameter_count_scaling as benchmark

    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    _progress(progress, method, "fixture", started)
    fixture = benchmark.make_fixture(
        DIMENSION,
        PARAMETER_COUNT,
        TIMESTEPS,
        dtype=tf.float32,
    )
    params = benchmark._make_parameter_batch(fixture, BATCH_SIZE)
    builder = (
        benchmark.build_batch_native_autodiff_fn
        if method == "full_helper"
        else benchmark.build_batch_native_autodiff_value_only_explicit_fn
    )
    _progress(progress, method, "builder", started)
    selected = builder(fixture, batch_size=BATCH_SIZE, jit_compile=True)
    _progress(progress, method, "trace", started)
    trace_started = time.perf_counter()
    concrete = selected.get_concrete_function()
    trace_seconds = time.perf_counter() - trace_started
    graph_def = concrete.graph.as_graph_def(add_shapes=True)

    _progress(progress, method, "first_xla_call", started)
    first_started = time.perf_counter()
    first_value, first_score = selected(params)
    first_output = common._tensor_output(first_value, first_score)
    first_seconds = time.perf_counter() - first_started

    _progress(progress, method, "warm_xla_call", started)
    warm_started = time.perf_counter()
    warm_value, warm_score = selected(tf.identity(params))
    warm_output = common._tensor_output(warm_value, warm_score)
    warm_seconds = time.perf_counter() - warm_started

    record = {
        "state": "passed",
        "method": method,
        "stage": "complete",
        "dimension": DIMENSION,
        "timesteps": TIMESTEPS,
        "parameter_count": PARAMETER_COUNT,
        "batch_size": BATCH_SIZE,
        "dtype": "float32",
        "jit_compile": True,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "tensorflow_version": tf.__version__,
        "graphdef": common._graph_record(graph_def),
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
    _write_json(output, record)
    _progress(progress, method, "complete", started)
    print(common._strict_json({"state": "passed", "method": method}))
    return 0


def _parse_time(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    values: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        name, raw = line.split("=", 1)
        if name in {"max_rss_kb", "exit_status"}:
            values[name] = int(raw)
        else:
            values[name] = float(raw)
    return values


def _run_method(method: str, repetition: int) -> dict[str, Any]:
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    stem = f"r{repetition}-{method}"
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
        "--child-output",
        str(child_output),
        "--progress",
        str(progress),
    ]
    command = [
        "/usr/bin/time",
        "-f",
        "max_rss_kb=%M\nuser_seconds=%U\nsystem_seconds=%S\nelapsed_seconds=%e\nexit_status=%x",
        "-o",
        str(time_output),
        "timeout",
        "--signal=TERM",
        f"--kill-after={KILL_GRACE_SECONDS}s",
        f"{METHOD_TIMEOUT_SECONDS}s",
        *child_command,
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
        timeout=METHOD_TIMEOUT_SECONDS + KILL_GRACE_SECONDS + 20,
    )
    wall_seconds = time.perf_counter() - started
    child = (
        json.loads(child_output.read_text(encoding="utf-8"))
        if child_output.is_file()
        else None
    )
    progress_record = (
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
        "repetition": repetition,
        "returncode": completed.returncode,
        "command": command,
        "wall_seconds": wall_seconds,
        "resource_usage": _parse_time(time_output),
        "last_progress": progress_record,
        "child": child,
        "stdout_tail": completed.stdout[-1000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def _output_checks(records: list[dict[str, Any]]) -> dict[str, Any]:
    passed_records = [record for record in records if record["state"] == "passed"]
    metadata_passed = bool(passed_records) and all(
        record["child"]["first_output"]["all_finite"]
        and record["child"]["warm_output"]["all_finite"]
        and record["child"]["first_output"]["value_shape"] == [BATCH_SIZE]
        and record["child"]["first_output"]["score_shape"]
        == [BATCH_SIZE, PARAMETER_COUNT]
        and record["child"]["first_output"]["value_dtype"] == "float32"
        and record["child"]["first_output"]["score_dtype"] == "float32"
        and record["child"]["concrete_function_count"] == 1
        for record in passed_records
    )
    warm_parity = bool(passed_records) and all(
        common._comparison(
            record["child"]["warm_output"]["value"],
            record["child"]["first_output"]["value"],
        )["passed"]
        and common._comparison(
            record["child"]["warm_output"]["score"],
            record["child"]["first_output"]["score"],
        )["passed"]
        for record in passed_records
    )
    by_method = {record["method"]: record for record in passed_records}
    baseline = by_method.get("full_helper")
    candidate = by_method.get("value_only_explicit")
    pair_parity = False
    if baseline is not None and candidate is not None:
        pair_parity = common._comparison(
            candidate["child"]["first_output"]["value"],
            baseline["child"]["first_output"]["value"],
        )["passed"] and common._comparison(
            candidate["child"]["first_output"]["score"],
            baseline["child"]["first_output"]["score"],
        )["passed"]
    return {
        "all_methods_completed": len(passed_records) == len(records) == 2,
        "finite_dtype_shape_and_single_trace": metadata_passed,
        "first_warm_parity": warm_parity,
        "baseline_candidate_parity": pair_parity,
    }


def _parent(output: Path, repetition: int) -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("CPU/XLA parent requires CUDA_VISIBLE_DEVICES=-1")
    started = time.perf_counter()
    records = [_run_method(method, repetition) for method in METHODS]
    checks = _output_checks(records)
    valid_pair = all(checks.values())
    rss_reduction = None
    if valid_pair:
        by_method = {record["method"]: record for record in records}
        baseline_rss = by_method["full_helper"]["resource_usage"]["max_rss_kb"]
        candidate_rss = by_method["value_only_explicit"]["resource_usage"]["max_rss_kb"]
        rss_reduction = common._reduction_percent(baseline_rss, candidate_rss)
    artifact = {
        "schema": SCHEMA,
        "state": "passed" if valid_pair else "failed",
        "decision": (
            "repeat_pair_for_peak_rss_replication"
            if valid_pair and rss_reduction is not None and rss_reduction > 0.0
            else "do_not_claim_memory_reduction"
        ),
        "question": (
            "Do baseline and explicit value-only batched autodiff compile under "
            "bounded CPU/XLA, preserve parity, and does the candidate nominate "
            "a repeatable peak-RSS reduction?"
        ),
        "cell": {
            "dimension": DIMENSION,
            "timesteps": TIMESTEPS,
            "parameter_count": PARAMETER_COUNT,
            "batch_size": BATCH_SIZE,
            "dtype": "float32",
        },
        "checks": checks,
        "peak_rss_reduction_percent": rss_reduction,
        "records": records,
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
            "device_status": "CPU-only XLA gate; GPU intentionally hidden",
            "jit_compile": True,
            "random_seeds": "N/A; deterministic fixture and parameter cloud",
            "data_version": "synthetic nested Kalman QR fixture in benchmark source",
            "wall_time_seconds": time.perf_counter() - started,
            "output_artifact": str(output.relative_to(REPO_ROOT)),
            "plan_file": str(PLAN_PATH.relative_to(REPO_ROOT)),
            "result_file": str(RESULT_PATH.relative_to(REPO_ROOT)),
            "trace_artifact": str(TRACE_ARTIFACT.relative_to(REPO_ROOT)),
            "source_sha256": {
                str(SCRIPT_PATH.relative_to(REPO_ROOT)): common._sha256(SCRIPT_PATH),
                str(BENCHMARK_PATH.relative_to(REPO_ROOT)): common._sha256(
                    BENCHMARK_PATH
                ),
                str(PLAN_PATH.relative_to(REPO_ROOT)): common._sha256(PLAN_PATH),
                str(TRACE_ARTIFACT.relative_to(REPO_ROOT)): common._sha256(
                    TRACE_ARTIFACT
                ),
            },
        },
        "evidence_roles": {
            "compile_completion_and_output_validity": "continuation_gate",
            "single_pair_peak_rss": "nomination_only",
            "first_and_warm_call_seconds": "explanatory_only",
        },
        "nonclaims": [
            "no memory-reduction claim without replication",
            "no historical T=120 B=16 repair claim",
            "no GPU viability conclusion",
            "no runtime ranking or speed superiority",
            "no production/default readiness",
            "no HMC, posterior, or scientific-validity claim",
        ],
    }
    _write_json(output, artifact)
    print(
        common._strict_json(
            {
                "state": artifact["state"],
                "decision": artifact["decision"],
                "checks": checks,
                "peak_rss_reduction_percent": rss_reduction,
            },
            indent=2,
        )
    )
    return 0 if valid_pair else 1


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repetition", type=int, default=1)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--child-output", type=Path)
    parser.add_argument("--progress", type=Path)
    args = parser.parse_args(argv)
    if args.child and None in (args.method, args.child_output, args.progress):
        parser.error("--child requires --method, --child-output, and --progress")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.child:
        return _child(args.method, args.child_output, args.progress)
    return _parent(args.output.resolve(), args.repetition)


if __name__ == "__main__":
    raise SystemExit(main())
