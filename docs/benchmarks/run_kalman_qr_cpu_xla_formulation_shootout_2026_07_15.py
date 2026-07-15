#!/usr/bin/env python
"""Compare CPU/XLA formulations for 16 independent analytical Kalman scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import signal
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = Path(__file__).resolve()
PYTHON = Path("/home/ubuntu/anaconda3/envs/tfgpu/bin/python")
HWLOC_BIND = Path("/home/ubuntu/anaconda3/envs/tfgpu/bin/hwloc-bind")
PLAN = "docs/plans/bayesfilter-kalman-qr-cpu-xla-formulation-shootout-plan-2026-07-15.md"
RESULT = "docs/plans/bayesfilter-kalman-qr-cpu-xla-formulation-shootout-result-2026-07-15.md"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "docs/benchmarks/kalman_qr_cpu_xla_formulation_shootout_2026-07-15"
SCHEMA = "bayesfilter.kalman_qr.cpu_xla_formulation_shootout.v1"
RECORD_SCHEMA = "bayesfilter.kalman_qr.cpu_xla_formulation_record.v1"
SOURCE_PATHS = (
    "bayesfilter/linear/kalman_qr_tf.py",
    "bayesfilter/linear/kalman_qr_derivatives_tf.py",
    "bayesfilter/linear/qr_factor_tf.py",
    "scripts/benchmark_kalman_qr_parameter_count_scaling.py",
    "scripts/kalman_qr_benchmark_contract.py",
    "docs/benchmarks/run_kalman_qr_cpu_xla_formulation_shootout_2026_07_15.py",
    PLAN,
)
FORMULATIONS = (
    "native_batch",
    "vectorized_strict",
    "vectorized_fallback",
    "map_sequential",
    "map_parallel_16",
    "static_unrolled",
    "sequential_b1_calls",
)
SINGLE_PROCESS_CANDIDATES = frozenset(FORMULATIONS[1:-1])
PHYSICAL_CPU_POOL = tuple(range(16, 32))
EXCLUDED_SMT_SIBLINGS = tuple(range(144, 160))
FLOAT32_TOLERANCE = 2.0e-4
WARM_ROUNDS = 2
MEASURED_ROUNDS = 5
RANDOM_SEED = 20260715
MAX_RSS_BYTES = 16 * 1024**3
MAX_PRELAUNCH_BUSY = 0.10
MAX_PRELAUNCH_LOAD = 16.0
RESOURCE_WAIT_SECONDS = 7200
RESOURCE_POLL_SECONDS = 30
CHILD_TIMEOUT_SECONDS = 900


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_manifest() -> dict[str, Any]:
    files = []
    for relative in SOURCE_PATHS:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"missing source path: {relative}")
        files.append({"path": relative, "sha256": sha256_file(path)})
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"files": files, "fingerprint": hashlib.sha256(encoded).hexdigest()}


def repo_relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def cpu_list(batch_size: int) -> tuple[int, ...]:
    if batch_size < 1 or batch_size > len(PHYSICAL_CPU_POOL):
        raise ValueError(f"unsupported batch size for CPU pool: {batch_size}")
    return PHYSICAL_CPU_POOL[:batch_size]


def topology_contract() -> dict[str, Any]:
    completed = subprocess.run(
        ["lscpu", "-e=CPU,NODE,SOCKET,CORE,ONLINE"],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    rows = []
    for line in completed.stdout.splitlines()[1:]:
        fields = line.split()
        if len(fields) == 5:
            rows.append(
                {
                    "cpu": int(fields[0]),
                    "node": int(fields[1]),
                    "socket": int(fields[2]),
                    "core": int(fields[3]),
                    "online": fields[4].lower() == "yes",
                }
            )
    by_cpu = {row["cpu"]: row for row in rows}
    selected = [by_cpu.get(cpu) for cpu in PHYSICAL_CPU_POOL]
    siblings = [by_cpu.get(cpu) for cpu in EXCLUDED_SMT_SIBLINGS]
    valid = (
        completed.returncode == 0
        and all(row is not None for row in selected + siblings)
        and all(row["online"] and row["node"] == 0 and row["socket"] == 0 for row in selected if row)
        and len({row["core"] for row in selected if row}) == 16
        and all(
            sibling["node"] == 0
            and sibling["socket"] == 0
            and sibling["core"] == primary["core"]
            for primary, sibling in zip(selected, siblings, strict=True)
            if primary is not None and sibling is not None
        )
    )
    return {"valid": valid, "selected": selected, "excluded_smt_siblings": siblings}


def cpu_stat_snapshot(cpus: Sequence[int]) -> dict[int, tuple[int, int]]:
    wanted = set(cpus)
    result: dict[int, tuple[int, int]] = {}
    with Path("/proc/stat").open("r", encoding="utf-8") as handle:
        for line in handle:
            match = re.match(r"^cpu(\d+)\s+(.+)$", line)
            if match is None:
                continue
            cpu = int(match.group(1))
            if cpu not in wanted:
                continue
            values = [int(value) for value in match.group(2).split()]
            total = sum(values)
            idle = values[3] + (values[4] if len(values) > 4 else 0)
            result[cpu] = (total, idle)
    if set(result) != wanted:
        raise RuntimeError("incomplete /proc/stat CPU snapshot")
    return result


def cpu_busy_fractions(
    before: Mapping[int, tuple[int, int]],
    after: Mapping[int, tuple[int, int]],
) -> dict[int, float] | None:
    if set(before) != set(after) or not before:
        return None
    result = {}
    for cpu in before:
        total_delta = after[cpu][0] - before[cpu][0]
        idle_delta = after[cpu][1] - before[cpu][1]
        if total_delta <= 0:
            return None
        result[cpu] = max(0.0, min(1.0, (total_delta - idle_delta) / total_delta))
    return result


def contamination_seconds(
    before: Mapping[int, tuple[int, int]],
    after: Mapping[int, tuple[int, int]],
    owned_cpu_seconds: float,
    *,
    clock_ticks: int | None = None,
) -> float | None:
    if set(before) != set(after) or not before:
        return None
    ticks = clock_ticks or int(os.sysconf(os.sysconf_names["SC_CLK_TCK"]))
    busy_ticks = 0
    for cpu in before:
        total_delta = after[cpu][0] - before[cpu][0]
        idle_delta = after[cpu][1] - before[cpu][1]
        if total_delta <= 0:
            return None
        busy_ticks += total_delta - idle_delta
    return max(0.0, busy_ticks / ticks - owned_cpu_seconds)


def wait_for_idle_cpus(
    cpus: Sequence[int],
    *,
    wait_seconds: int = RESOURCE_WAIT_SECONDS,
    poll_seconds: int = RESOURCE_POLL_SECONDS,
    sample_seconds: float = 2.0,
) -> dict[str, Any]:
    started = time.monotonic()
    last: dict[str, Any] | None = None
    while True:
        before = cpu_stat_snapshot(cpus)
        time.sleep(sample_seconds)
        after = cpu_stat_snapshot(cpus)
        busy = cpu_busy_fractions(before, after)
        load = list(os.getloadavg())
        last = {
            "observed_utc": utc_now(),
            "busy_fractions": None if busy is None else {str(k): v for k, v in busy.items()},
            "load_average": load,
        }
        if busy is not None and max(busy.values()) < MAX_PRELAUNCH_BUSY and load[0] <= MAX_PRELAUNCH_LOAD:
            return last
        if time.monotonic() - started >= wait_seconds:
            raise TimeoutError(f"target CPUs did not become idle: {last}")
        time.sleep(poll_seconds)


def rss_bytes(pid: int | None = None) -> int:
    target = pid or os.getpid()
    try:
        for line in Path(f"/proc/{target}/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except OSError:
        return 0
    return 0


def task_affinities(pid: int | None = None) -> dict[str, list[int]]:
    target = pid or os.getpid()
    result = {}
    for path in Path(f"/proc/{target}/task").glob("*/status"):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        cpu_line = next((line for line in lines if line.startswith("Cpus_allowed_list:")), None)
        if cpu_line is None:
            continue
        values = []
        for part in cpu_line.split(":", 1)[1].strip().split(","):
            if "-" in part:
                first, last = (int(value) for value in part.split("-", 1))
                values.extend(range(first, last + 1))
            elif part:
                values.append(int(part))
        result[path.parent.name] = sorted(values)
    return result


def anonymous_numa_pages() -> dict[str, Any]:
    counts: dict[int, int] = {}
    total = 0
    try:
        lines = Path("/proc/self/numa_maps").read_text(encoding="utf-8").splitlines()
    except OSError:
        return {"valid": False, "reason": "numa_maps_unreadable"}
    for line in lines:
        if " anon=" not in f" {line}":
            continue
        for node, count in re.findall(r"\bN(\d+)=(\d+)\b", line):
            counts[int(node)] = counts.get(int(node), 0) + int(count)
            total += int(count)
    fraction = counts.get(0, 0) / total if total else 0.0
    return {
        "valid": total > 0 and fraction >= 0.95,
        "pages_by_node": {str(key): value for key, value in sorted(counts.items())},
        "node0_fraction": fraction,
    }


def parity_summary(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    try:
        candidate_values = candidate["value"]
        baseline_values = baseline["value"]
        candidate_scores = candidate["score"]
        baseline_scores = baseline["score"]
    except KeyError:
        return {"passed": False, "reason": "missing_output"}
    shape = (
        len(candidate_values) == len(baseline_values)
        and len(candidate_scores) == len(baseline_scores)
        and all(len(left) == len(right) for left, right in zip(candidate_scores, baseline_scores, strict=False))
    )
    finite = shape and all(
        math.isfinite(float(value))
        for value in [*candidate_values, *baseline_values]
        + [item for row in candidate_scores for item in row]
        + [item for row in baseline_scores for item in row]
    )
    if not shape or not finite:
        return {"passed": False, "shape": shape, "finite": finite}
    value_residual = max(
        (abs(float(left) - float(right)) for left, right in zip(candidate_values, baseline_values, strict=True)),
        default=0.0,
    )
    score_residual = max(
        (
            abs(float(left) - float(right))
            for left_row, right_row in zip(candidate_scores, baseline_scores, strict=True)
            for left, right in zip(left_row, right_row, strict=True)
        ),
        default=0.0,
    )
    value_allowed = max(
        (FLOAT32_TOLERANCE + FLOAT32_TOLERANCE * abs(float(value)) for value in baseline_values),
        default=FLOAT32_TOLERANCE,
    )
    score_allowed = max(
        (
            FLOAT32_TOLERANCE + FLOAT32_TOLERANCE * abs(float(value))
            for row in baseline_scores
            for value in row
        ),
        default=FLOAT32_TOLERANCE,
    )
    return {
        "passed": value_residual <= value_allowed and score_residual <= score_allowed,
        "shape": shape,
        "finite": finite,
        "max_value_residual": value_residual,
        "max_score_residual": score_residual,
        "max_allowed_value_residual": value_allowed,
        "max_allowed_score_residual": score_allowed,
    }


def hlo_census(text: str) -> dict[str, Any]:
    tokens = {
        "while": r"\bwhile\(",
        "custom_call": r"\bcustom-call\(",
        "qr_target": r'custom_call_target="Qr"',
        "triangular_solve": r"\btriangular-solve\(",
        "dot": r"\bdot\(",
        "fusion": r"\bfusion\(",
        "map": r"\bmap\(",
    }
    return {name: len(re.findall(pattern, text)) for name, pattern in tokens.items()}


def paired_statistics(
    candidate: Sequence[float],
    baseline: Sequence[float],
    *,
    seed: int = RANDOM_SEED,
    resamples: int = 10_000,
) -> dict[str, Any]:
    if len(candidate) != len(baseline) or not candidate or any(value <= 0 for value in (*candidate, *baseline)):
        raise ValueError("paired positive equal-length timings required")
    logs = [math.log(left / right) for left, right in zip(candidate, baseline, strict=True)]
    rng = random.Random(seed)
    boot = sorted(
        statistics.fmean(logs[rng.randrange(len(logs))] for _ in logs)
        for _ in range(resamples)
    )
    positives = sum(value > 0 for value in logs if value != 0.0)
    n = sum(value != 0.0 for value in logs)
    if n:
        tail = sum(math.comb(n, index) for index in range(min(positives, n - positives) + 1)) / (2**n)
        sign_p = min(1.0, 2.0 * tail)
    else:
        sign_p = 1.0
    lower_index = min(resamples - 1, int(0.025 * resamples))
    upper_index = min(resamples - 1, int(0.975 * resamples))
    return {
        "ratios": [math.exp(value) for value in logs],
        "geometric_mean_ratio": math.exp(statistics.fmean(logs)),
        "bootstrap_95_interval": [
            math.exp(boot[lower_index]),
            math.exp(boot[upper_index]),
        ],
        "sign_test_two_sided_p": sign_p,
        "paired_block_count": len(logs),
        "seed": seed,
        "bootstrap_resamples": resamples,
    }


def nomination_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_name = {str(record.get("formulation")): record for record in records}
    baseline = by_name.get("native_batch")
    if baseline is None or baseline.get("status") != "passed":
        return {"status": "invalid_native_baseline", "candidate": None, "eligible": []}
    baseline_median = float(baseline["timing"]["measured_median_seconds"])
    eligible = []
    for name in SINGLE_PROCESS_CANDIDATES:
        record = by_name.get(name)
        if record is None or record.get("status") != "passed" or record.get("parity", {}).get("passed") is not True:
            continue
        ratio = float(record["timing"]["measured_median_seconds"]) / baseline_median
        if ratio <= 0.80:
            eligible.append({"formulation": name, "ratio": ratio})
    eligible.sort(key=lambda row: row["ratio"])
    return {
        "status": "candidate_nominated" if eligible else "no_candidate_nominated",
        "baseline_median_seconds": baseline_median,
        "eligible": eligible,
        "candidate": eligible[0] if eligible else None,
    }


def _scalar_score_body(tf: Any, fixture: Any, parameters: Any) -> tuple[Any, Any]:
    from bayesfilter.linear.kalman_qr_derivatives_tf import tf_qr_sqrt_kalman_score
    from scripts.benchmark_kalman_qr_parameter_count_scaling import _model_tensors

    tensors = _model_tensors(fixture, parameters)
    return tf_qr_sqrt_kalman_score.python_function(
        observations=fixture.observations,
        transition_offset=tensors[0],
        transition_matrix=tensors[1],
        transition_covariance=tensors[2],
        observation_offset=tensors[3],
        observation_matrix=tensors[4],
        observation_covariance=tensors[5],
        initial_state_mean=tensors[6],
        initial_state_covariance=tensors[7],
        d_initial_state_mean=tensors[8],
        d_initial_state_covariance=tensors[9],
        d_transition_offset=tensors[10],
        d_transition_matrix=tensors[11],
        d_transition_covariance=tensors[12],
        d_observation_offset=tensors[13],
        d_observation_matrix=tensors[14],
        d_observation_covariance=tensors[15],
        jitter=tf.constant(1.0e-9, dtype=fixture.dtype),
        jitter_updates_filtered_covariance=True,
    )


def _build_formulation(tf: Any, fixture: Any, formulation: str, batch_size: int) -> tuple[Any, Any]:
    from scripts.benchmark_kalman_qr_parameter_count_scaling import (
        build_batch_native_analytic_fn,
        build_scalar_analytic_row_loop_fn,
    )

    if formulation == "native_batch":
        fn = build_batch_native_analytic_fn(fixture, batch_size=batch_size, jit_compile=True)
        return fn, fn
    if formulation == "static_unrolled":
        fn = build_scalar_analytic_row_loop_fn(fixture, batch_size=batch_size, jit_compile=True)
        return fn, fn
    if formulation == "sequential_b1_calls":
        fn = build_batch_native_analytic_fn(fixture, batch_size=1, jit_compile=True)
        return fn, fn

    output_signature = (
        tf.TensorSpec([], fixture.dtype),
        tf.TensorSpec([fixture.parameter_count], fixture.dtype),
    )

    @tf.function(
        jit_compile=True,
        reduce_retracing=True,
        input_signature=[tf.TensorSpec([batch_size, fixture.parameter_count], fixture.dtype)],
    )
    def mapped(parameters_batch: Any) -> tuple[Any, Any]:
        def body(row: Any) -> tuple[Any, Any]:
            return _scalar_score_body(tf, fixture, row)

        if formulation == "vectorized_strict":
            return tf.vectorized_map(body, parameters_batch, fallback_to_while_loop=False, warn=True)
        if formulation == "vectorized_fallback":
            return tf.vectorized_map(body, parameters_batch, fallback_to_while_loop=True, warn=True)
        if formulation == "map_sequential":
            return tf.map_fn(body, parameters_batch, fn_output_signature=output_signature, parallel_iterations=1)
        if formulation == "map_parallel_16":
            return tf.map_fn(body, parameters_batch, fn_output_signature=output_signature, parallel_iterations=16)
        raise ValueError(f"unknown formulation: {formulation}")

    return mapped, mapped


def _materialize(outputs: tuple[Any, Any]) -> dict[str, Any]:
    value, score = outputs
    return {
        "value": [[float(item) for item in row] if isinstance(row, list) else float(row) for row in value.numpy().tolist()]
        if value.shape.rank > 1
        else [float(item) for item in value.numpy().tolist()],
        "score": [[float(item) for item in row] for row in score.numpy().tolist()],
        "value_shape": value.shape.as_list(),
        "score_shape": score.shape.as_list(),
        "value_dtype": value.dtype.name,
        "score_dtype": score.dtype.name,
    }


def _worker(args: argparse.Namespace) -> int:
    record_path = args.record_path.resolve()
    stage = "initialization"
    base = {
        "schema": RECORD_SCHEMA,
        "formulation": args.formulation,
        "dimension": args.dimension,
        "parameter_count": args.parameter_count,
        "timesteps": args.timesteps,
        "batch_size": args.batch_size,
        "row_ids": list(range(args.batch_size)),
        "started_utc": utc_now(),
        "source_manifest": source_manifest(),
    }
    try:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
        os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
        import tensorflow as tf

        tf.config.threading.set_intra_op_parallelism_threads(args.intra)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.benchmark_kalman_qr_parameter_count_scaling import (
            _make_parameter_cloud,
            _synchronize_outputs,
            make_fixture,
        )

        stage = "fixture"
        fixture = make_fixture(args.dimension, args.parameter_count, args.timesteps, dtype=tf.float32)
        cloud = _make_parameter_cloud(fixture)
        parameters = cloud[: args.batch_size]
        fn, compiler_fn = _build_formulation(tf, fixture, args.formulation, args.batch_size)
        if args.formulation == "sequential_b1_calls":
            concrete = fn.get_concrete_function(parameters[:1])

            def invoke() -> tuple[Any, Any]:
                outputs = [concrete(parameters[index : index + 1]) for index in range(args.batch_size)]
                return (
                    tf.concat([value for value, _ in outputs], axis=0),
                    tf.concat([score for _, score in outputs], axis=0),
                )

            compiler_inputs = (parameters[:1],)
        else:
            concrete = fn.get_concrete_function(parameters)

            def invoke() -> tuple[Any, Any]:
                return concrete(parameters)

            compiler_inputs = (parameters,)

        graph_def = concrete.graph.as_graph_def(add_shapes=True)
        graph_metrics = {
            "nodes": len(graph_def.node),
            "serialized_bytes": len(graph_def.SerializeToString(deterministic=True)),
        }

        measurement_repetitions = 2048 if args.dimension == 2 and args.timesteps == 4 else 1

        def timed_invoke(repetitions: int = 1) -> tuple[tuple[Any, Any], dict[str, Any]]:
            before_stat = cpu_stat_snapshot(cpu_list(args.batch_size))
            before_cpu = time.process_time_ns()
            before_wall = time.perf_counter_ns()
            outputs = None
            for _ in range(repetitions):
                outputs = invoke()
            if outputs is None:
                raise RuntimeError("timed invocation requires at least one repetition")
            synchronization = _synchronize_outputs(outputs)
            after_wall = time.perf_counter_ns()
            after_cpu = time.process_time_ns()
            after_stat = cpu_stat_snapshot(cpu_list(args.batch_size))
            window_wall_seconds = (after_wall - before_wall) / 1.0e9
            cpu_seconds = (after_cpu - before_cpu) / 1.0e9
            contamination = contamination_seconds(before_stat, after_stat, cpu_seconds)
            threshold = max(0.25, 0.02 * args.batch_size * window_wall_seconds)
            return outputs, {
                "repetitions": repetitions,
                "wall_seconds": window_wall_seconds / repetitions,
                "measurement_window_seconds": window_wall_seconds,
                "process_cpu_seconds": cpu_seconds,
                "average_cores": cpu_seconds / window_wall_seconds if window_wall_seconds else 0.0,
                "unattributed_target_cpu_seconds": contamination,
                "contamination_threshold_seconds": threshold,
                "contaminated": contamination is None or contamination > threshold,
                "synchronization": synchronization[0],
                "rss_bytes": rss_bytes(),
            }

        stage = "first_executable_call"
        outputs, first = timed_invoke()
        stage = "warmup"
        warmups = []
        for _ in range(WARM_ROUNDS):
            outputs, timing = timed_invoke()
            warmups.append(timing)
        stage = "measurement"
        measurements = []
        for _ in range(MEASURED_ROUNDS):
            outputs, timing = timed_invoke(measurement_repetitions)
            measurements.append(timing)
        output = _materialize(outputs)

        stage = "compiler_ir"
        hlo_error = None
        hlo_text = ""
        try:
            hlo_text = str(compiler_fn.experimental_get_compiler_ir(*compiler_inputs)(stage="optimized_hlo"))
        except Exception as exc:
            hlo_error = {"type": type(exc).__name__, "message": str(exc)}
        hlo_path = record_path.with_name("optimized_hlo.txt")
        if hlo_text:
            hlo_path.write_text(hlo_text, encoding="utf-8")
        hlo_metrics = {
            "available": bool(hlo_text),
            "bytes": len(hlo_text.encode("utf-8")),
            "sha256": hashlib.sha256(hlo_text.encode("utf-8")).hexdigest() if hlo_text else None,
            "census": hlo_census(hlo_text),
            "path": repo_relative(hlo_path) if hlo_text else None,
            "error": hlo_error,
        }
        affinity = sorted(os.sched_getaffinity(0))
        tasks = task_affinities()
        numa = anonymous_numa_pages()
        all_rounds = [first, *warmups, *measurements]
        expected_cpus = list(cpu_list(args.batch_size))
        placement_passed = (
            affinity == expected_cpus
            and bool(tasks)
            and all(value == expected_cpus for value in tasks.values())
            and numa.get("valid") is True
        )
        resource_passed = (
            max(row["rss_bytes"] for row in all_rounds) < MAX_RSS_BYTES
            and not any(row["contaminated"] for row in measurements)
        )
        source_after = source_manifest()
        source_stable = source_after["fingerprint"] == base["source_manifest"]["fingerprint"]
        status = "passed" if placement_passed and resource_passed and source_stable else "failed_gate"
        record = {
            **base,
            "status": status,
            "finished_utc": utc_now(),
            "tensorflow_version": tf.__version__,
            "jit_compile": True,
            "dtype": "float32",
            "intra_op_threads": args.intra,
            "inter_op_threads": 1,
            "vectorized_fallback_allowed": args.formulation == "vectorized_fallback",
            "measurement_repetitions": measurement_repetitions,
            "output": output,
            "timing": {
                "first_executable_call": first,
                "warmups": warmups,
                "measurements": measurements,
                "measured_median_seconds": statistics.median(row["wall_seconds"] for row in measurements),
                "measured_median_average_cores": statistics.median(row["average_cores"] for row in measurements),
            },
            "graphdef": graph_metrics,
            "optimized_hlo": hlo_metrics,
            "placement": {
                "passed": placement_passed,
                "process_affinity": affinity,
                "task_affinities": tasks,
                "numa": numa,
            },
            "resource": {
                "passed": resource_passed,
                "max_rss_bytes": max(row["rss_bytes"] for row in all_rounds),
            },
            "source_stable": source_stable,
        }
        write_json(record_path, record)
        return 0 if status == "passed" else 2
    except BaseException as exc:
        record = {
            **base,
            "status": "failed",
            "finished_utc": utc_now(),
            "failure_stage": stage,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback_tail": traceback.format_exc().splitlines()[-30:],
            },
        }
        write_json(record_path, record)
        return 1


def worker_command(
    formulation: str,
    *,
    dimension: int,
    parameter_count: int,
    timesteps: int,
    batch_size: int,
    record_path: Path,
) -> list[str]:
    cpus = cpu_list(batch_size)
    return [
        str(HWLOC_BIND),
        "--membind",
        "node:0",
        "--",
        "taskset",
        "-c",
        ",".join(str(cpu) for cpu in cpus),
        str(PYTHON),
        str(SCRIPT_PATH),
        "--worker",
        "--formulation",
        formulation,
        "--dimension",
        str(dimension),
        "--parameter-count",
        str(parameter_count),
        "--timesteps",
        str(timesteps),
        "--batch-size",
        str(batch_size),
        "--intra",
        str(batch_size),
        "--record-path",
        str(record_path),
    ]


def run_worker(
    formulation: str,
    *,
    dimension: int,
    parameter_count: int,
    timesteps: int,
    batch_size: int,
    record_dir: Path,
) -> dict[str, Any]:
    record_path = record_dir / "record.json"
    record_dir.mkdir(parents=True, exist_ok=True)
    admission = wait_for_idle_cpus(cpu_list(batch_size))
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "-1",
            "TF_FORCE_GPU_ALLOW_GROWTH": "true",
            "OMP_NUM_THREADS": str(batch_size),
            "TF_NUM_INTRAOP_THREADS": str(batch_size),
            "TF_NUM_INTEROP_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    stdout_path = record_dir / "stdout.log"
    stderr_path = record_dir / "stderr.log"
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            worker_command(
                formulation,
                dimension=dimension,
                parameter_count=parameter_count,
                timesteps=timesteps,
                batch_size=batch_size,
                record_path=record_path,
            ),
            cwd=REPO_ROOT,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=CHILD_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
            payload = {
                "schema": RECORD_SCHEMA,
                "status": "timeout",
                "formulation": formulation,
                "failure_stage": "child_wall_timeout",
                "timeout_seconds": CHILD_TIMEOUT_SECONDS,
                "source_manifest": source_manifest(),
            }
            write_json(record_path, payload)
            return payload
    payload = read_json(record_path) or {
        "schema": RECORD_SCHEMA,
        "status": "missing_record",
        "formulation": formulation,
    }
    payload["child_returncode"] = returncode
    payload["resource_admission"] = admission
    write_json(record_path, payload)
    return payload


def _load_or_initialize(output_root: Path) -> dict[str, Any]:
    manifest = source_manifest()
    path = output_root / "status.json"
    existing = read_json(path)
    if existing is None:
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=False,
                capture_output=True,
                text=True,
                timeout=15,
            ).stdout.strip()
        except Exception:
            commit = "unknown"
        return {
            "schema": SCHEMA,
            "status": "initialized",
            "created_utc": utc_now(),
            "git_commit": commit,
            "source_manifest": manifest,
            "topology": topology_contract(),
            "plan": PLAN,
            "result": RESULT,
            "phases": {},
        }
    if existing.get("schema") != SCHEMA or existing.get("source_manifest", {}).get("fingerprint") != manifest["fingerprint"]:
        raise RuntimeError("existing status source or schema drift")
    return existing


def phase_specs(phase: str) -> tuple[int, int, int, int]:
    if phase == "smoke":
        return 2, 3, 4, 4
    if phase in {"canary", "confirm"}:
        return 10, 50, 120, 16
    if phase == "transfer":
        return 30, 50, 120, 16
    raise ValueError(f"unknown phase: {phase}")


def _attach_parity(records: Sequence[dict[str, Any]]) -> None:
    baseline = next((row for row in records if row.get("formulation") == "native_batch"), None)
    if baseline is None or baseline.get("status") != "passed":
        return
    baseline_output = baseline["output"]
    for record in records:
        if record.get("status") == "passed":
            record["parity"] = parity_summary(record["output"], baseline_output)
            if record["parity"]["passed"] is not True:
                record["status"] = "failed_parity"
            path = Path(record["record_path"])
            stored = read_json(path)
            if stored is not None:
                stored["parity"] = record["parity"]
                if record["status"] == "failed_parity":
                    stored["status"] = "failed_parity"
                write_json(path, stored)


def run_screening_phase(phase: str, output_root: Path) -> dict[str, Any]:
    status = _load_or_initialize(output_root)
    if status["topology"].get("valid") is not True:
        raise RuntimeError("CPU topology contract failed")
    if status["phases"].get(phase, {}).get("status") == "complete":
        return status["phases"][phase]
    dimension, parameter_count, timesteps, batch_size = phase_specs(phase)
    order = list(FORMULATIONS)
    random.Random(RANDOM_SEED + (0 if phase == "smoke" else 1)).shuffle(order)
    records = []
    for index, formulation in enumerate(order):
        record_dir = output_root / phase / f"{index:02d}-{formulation}"
        record = run_worker(
            formulation,
            dimension=dimension,
            parameter_count=parameter_count,
            timesteps=timesteps,
            batch_size=batch_size,
            record_dir=record_dir,
        )
        record["record_path"] = repo_relative(record_dir / "record.json")
        records.append(record)
        if formulation == "native_batch" and record.get("status") != "passed":
            break
        if source_manifest()["fingerprint"] != status["source_manifest"]["fingerprint"]:
            raise RuntimeError("source drift during phase")
    _attach_parity(records)
    baseline = next((row for row in records if row.get("formulation") == "native_batch"), None)
    native_valid = baseline is not None and baseline.get("status") == "passed" and baseline.get("parity", {}).get("passed") is True
    result: dict[str, Any] = {
        "status": "complete" if native_valid else "invalid_native_baseline",
        "finished_utc": utc_now(),
        "workload": [dimension, parameter_count, timesteps, batch_size],
        "order": order,
        "records": records,
    }
    if phase == "canary" and native_valid:
        result["nomination"] = nomination_summary(records)
    status = _load_or_initialize(output_root)
    status["phases"][phase] = result
    status["status"] = "running" if result["status"] == "complete" else "invalid"
    if phase == "canary" and result.get("nomination", {}).get("candidate") is None:
        status["status"] = "complete"
        status["decision"] = "NO_SINGLE_PROCESS_FORMULATION_REPAIR_NOMINATED"
    status["updated_utc"] = utc_now()
    write_json(output_root / "status.json", status)
    return result


def run_confirmation(output_root: Path) -> dict[str, Any]:
    status = _load_or_initialize(output_root)
    nomination = status.get("phases", {}).get("canary", {}).get("nomination", {})
    candidate_row = nomination.get("candidate")
    if candidate_row is None:
        raise RuntimeError("confirmation requires a nominated canary candidate")
    if status["phases"].get("confirm", {}).get("status") == "complete":
        return status["phases"]["confirm"]
    candidate = str(candidate_row["formulation"])
    dimension, parameter_count, timesteps, batch_size = phase_specs("confirm")
    blocks = []
    for block_index in range(8):
        order = ["native_batch", candidate] if block_index % 2 == 0 else [candidate, "native_batch"]
        records = []
        for arm_index, formulation in enumerate(order):
            record_dir = output_root / "confirm" / f"block-{block_index:02d}" / f"{arm_index:02d}-{formulation}"
            record = run_worker(
                formulation,
                dimension=dimension,
                parameter_count=parameter_count,
                timesteps=timesteps,
                batch_size=batch_size,
                record_dir=record_dir,
            )
            record["record_path"] = repo_relative(record_dir / "record.json")
            records.append(record)
        _attach_parity(records)
        block_status = "passed" if all(row.get("status") == "passed" for row in records) else "failed"
        blocks.append({"block": block_index, "order": order, "status": block_status, "records": records})
        if block_status != "passed":
            break
    if len(blocks) == 8 and all(block["status"] == "passed" for block in blocks):
        candidate_times = []
        native_times = []
        for block in blocks:
            by_name = {row["formulation"]: row for row in block["records"]}
            candidate_times.append(float(by_name[candidate]["timing"]["measured_median_seconds"]))
            native_times.append(float(by_name["native_batch"]["timing"]["measured_median_seconds"]))
        statistics_result = paired_statistics(candidate_times, native_times)
        passed = statistics_result["bootstrap_95_interval"][1] < 0.90
        decision = "SINGLE_PROCESS_FORMULATION_REPAIR_CONFIRMED" if passed else "SINGLE_PROCESS_FORMULATION_REPAIR_NOT_CONFIRMED"
        phase_status = {
            "status": "complete",
            "finished_utc": utc_now(),
            "candidate": candidate,
            "blocks": blocks,
            "paired_statistics_candidate_over_native": statistics_result,
            "decision": decision,
        }
    else:
        phase_status = {
            "status": "failed",
            "finished_utc": utc_now(),
            "candidate": candidate,
            "blocks": blocks,
            "decision": "INVALID_CONFIRMATION",
        }
    status = _load_or_initialize(output_root)
    status["phases"]["confirm"] = phase_status
    status["decision"] = phase_status["decision"]
    status["status"] = "running" if phase_status["decision"] == "SINGLE_PROCESS_FORMULATION_REPAIR_CONFIRMED" else "complete"
    status["updated_utc"] = utc_now()
    write_json(output_root / "status.json", status)
    return phase_status


def run_transfer(output_root: Path) -> dict[str, Any]:
    status = _load_or_initialize(output_root)
    confirmation = status.get("phases", {}).get("confirm", {})
    if confirmation.get("decision") != "SINGLE_PROCESS_FORMULATION_REPAIR_CONFIRMED":
        raise RuntimeError("transfer requires a confirmed formulation repair")
    candidate = str(confirmation["candidate"])
    dimension, parameter_count, timesteps, batch_size = phase_specs("transfer")
    records = []
    for index, formulation in enumerate(("native_batch", candidate)):
        record_dir = output_root / "transfer" / f"{index:02d}-{formulation}"
        record = run_worker(
            formulation,
            dimension=dimension,
            parameter_count=parameter_count,
            timesteps=timesteps,
            batch_size=batch_size,
            record_dir=record_dir,
        )
        record["record_path"] = repo_relative(record_dir / "record.json")
        records.append(record)
    _attach_parity(records)
    passed = all(row.get("status") == "passed" for row in records)
    result = {
        "status": "complete" if passed else "failed",
        "finished_utc": utc_now(),
        "workload": [dimension, parameter_count, timesteps, batch_size],
        "candidate": candidate,
        "records": records,
    }
    if passed:
        by_name = {row["formulation"]: row for row in records}
        result["descriptive_candidate_over_native_ratio"] = (
            by_name[candidate]["timing"]["measured_median_seconds"]
            / by_name["native_batch"]["timing"]["measured_median_seconds"]
        )
    status = _load_or_initialize(output_root)
    status["phases"]["transfer"] = result
    status["status"] = "complete"
    status["updated_utc"] = utc_now()
    write_json(output_root / "status.json", status)
    return result


def self_check() -> int:
    checks = {
        "formulations_unique": len(FORMULATIONS) == len(set(FORMULATIONS)),
        "candidate_boundary": "native_batch" not in SINGLE_PROCESS_CANDIDATES and "sequential_b1_calls" not in SINGLE_PROCESS_CANDIDATES,
        "cpu_pool": cpu_list(16) == PHYSICAL_CPU_POOL,
        "topology": topology_contract()["valid"],
        "source_manifest": len(source_manifest()["fingerprint"]) == 64,
    }
    print(json.dumps(checks, indent=2, sort_keys=True))
    return 0 if all(checks.values()) else 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "canary", "confirm", "transfer"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--formulation", choices=FORMULATIONS)
    parser.add_argument("--dimension", type=int)
    parser.add_argument("--parameter-count", type=int)
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--intra", type=int)
    parser.add_argument("--record-path", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_check:
        return self_check()
    if args.worker:
        required = (args.formulation, args.dimension, args.parameter_count, args.timesteps, args.batch_size, args.intra, args.record_path)
        if any(value is None for value in required):
            raise SystemExit("worker arguments are incomplete")
        return _worker(args)
    if args.phase is None:
        raise SystemExit("--phase or --self-check is required")
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.phase in {"smoke", "canary"}:
        result = run_screening_phase(args.phase, args.output_root)
    elif args.phase == "confirm":
        result = run_confirmation(args.output_root)
    else:
        result = run_transfer(args.output_root)
    print(json.dumps({"phase": args.phase, "status": result.get("status"), "decision": result.get("decision")}, indent=2))
    return 0 if result.get("status") == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
