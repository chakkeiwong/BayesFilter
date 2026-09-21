"""Sparse complete-controller memory/capacity diagnostics, never runtime code."""

import dataclasses
import json
import re
import time
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.inference.quadratic_rounds_tf import (
    make_quadratic_controller,
)
from tests.test_filter_repair_quadratic_batch_memory import _memory
from tests.test_filter_repair_quadratic_batches import _equal_records
from tests.test_filter_repair_quadratic_rounds import materialize, reference
from tests.test_filter_repair_uniform_rounds import fixture


def memory(gpu):
    result = _memory(gpu)
    result["smaps_rollup"] = {
        row.split()[0].rstrip(":"): int(row.split()[1]) * 1024
        for row in Path("/proc/self/smaps_rollup").read_text().splitlines()
        if row.startswith(("Rss:", "Pss:", "Private_Dirty:", "Anonymous:"))}
    return result


@pytest.mark.parametrize("rounds", [1, 4, 8])
def test_complete_controller_capacity_and_warm_allocations(rounds, request):
    directory = Path(request.config.getoption("xmlpath")).parent

    def record_progress(stage, **observed):
        # Preserve completed measurements even if the external deadline fires.
        with (directory / "uniform-round-growth-progress.jsonl").open("a") as handle:
            handle.write(json.dumps({"stage": stage, **observed}, allow_nan=False) + "\n")

    gpu = bool(tf.config.list_logical_devices("GPU"))
    callback, config, args = fixture(5, "nonquadratic")
    config = dataclasses.replace(config, max_fit_rounds=rounds)
    changed = (args[0] + .01, args[1] * 1.1, args[2] + 1, *args[3:])
    stages = {"before": memory(gpu)}
    started = time.perf_counter()
    kernel = make_quadratic_controller(callback, 5, config)
    build_seconds = time.perf_counter() - started
    stages["built"] = memory(gpu)
    record_progress("built", seconds=build_seconds, memory=stages["built"])
    started = time.perf_counter()
    kernel.get_concrete_function()
    trace_seconds = time.perf_counter() - started
    stages["traced"] = memory(gpu)
    record_progress("traced", seconds=trace_seconds, memory=stages["traced"])
    started = time.perf_counter()
    first = materialize(kernel(*args), config, 5, int(args[2]))
    cold_seconds = time.perf_counter() - started
    stages["first_call"] = memory(gpu)
    record_progress("first_call", seconds=cold_seconds, memory=stages["first_call"])
    second = materialize(kernel(*changed), config, 5, int(changed[2]))

    def execute(index):
        inputs, expected = (args, first) if index % 2 == 0 else (changed, second)
        assert materialize(kernel(*inputs), config, 5, int(inputs[2])) == expected

    for index in range(20):
        execute(index)
    if gpu:
        tf.config.experimental.reset_memory_stats("GPU:0")
    observations = [{"additional_calls": 0, "memory": memory(gpu)}]
    record_progress("warm", **observations[-1])
    for block in range(3):
        for index in range(1000):
            execute(index)
        observations.append({"additional_calls": (block + 1) * 1000, "memory": memory(gpu)})
        record_progress("warm", **observations[-1])
    assert kernel.experimental_get_tracing_count() == 1
    graph = kernel.get_concrete_function().graph.as_graph_def()
    nodes = len(graph.node) + sum(len(fn.node_def) for fn in graph.library.function)
    hlo = kernel.experimental_get_compiler_ir(*args)(stage="hlo")
    assert hlo == kernel.experimental_get_compiler_ir(*changed)(stage="hlo")
    operands = len(re.findall(r"\bparameter\((\d+)\)", hlo[hlo.rfind("\nENTRY "):]))
    assert operands == len(args) + len(kernel.get_concrete_function().captured_inputs)
    expected, _ = reference(callback, config, args)
    changed_expected, _ = reference(callback, config, changed)
    report = {"role": "complete_controller_capacity_and_allocation_diagnostic", "round_capacity": rounds,
        "gpu": gpu, "dimension": 5, "build_seconds": build_seconds, "trace_seconds": trace_seconds,
        "cold_seconds": cold_seconds, "stages": stages, "observations": observations,
        "graph_nodes": nodes, "hlo_bytes": len(hlo.encode()), "runtime_operands": operands,
        "result": first, "changed_result": second, "original": expected, "original_changed": changed_expected,
        "nonclaims": ["No runtime cleanup, exact RSS peak, executable eviction or general leak-freedom claim.",
            "Round-capacity comparison changes only the declared cap; it is not a speed ranking."]}
    with (directory / "uniform-round-growth.json").open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    _equal_records(first, expected)
    _equal_records(second, changed_expected)
