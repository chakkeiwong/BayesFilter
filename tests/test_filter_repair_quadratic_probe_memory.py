"""Fresh-process diagnostic costs of complete paired preparation/evaluation."""

import hashlib
import json
import re
import time
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.inference.quadratic_probe_evaluation_tf import (
    make_paired_probe_evaluator,
)
from tests.test_filter_repair_quadratic_batch_memory import _memory
from tests.test_filter_repair_quadratic_batches import _target
from tests.test_filter_repair_quadratic_numerics import original, serializable
from tests.test_filter_repair_quadratic_probes import arguments, original_probes


@pytest.mark.parametrize("dimension", [3, 5])
@pytest.mark.parametrize("arm", ["before", "graph", "xla"])
def test_paired_probe_costs(arm, dimension, request):
    gpu = bool(tf.config.list_logical_devices("GPU"))
    args = arguments(dimension)
    input_hashes = [hashlib.sha256(tf.io.serialize_tensor(item).numpy()).hexdigest() for item in args]
    checkpoint, _ = original()
    stages = {"before": _memory(gpu)}
    started = time.perf_counter()
    callback = _target(4, dimension)
    kernel = (original_probes(callback, dimension, 4) if arm == "before" else
        make_paired_probe_evaluator(callback, dimension, 4, jit_compile=arm == "xla"))
    build_seconds = time.perf_counter() - started
    stages["built"] = _memory(gpu)
    trace_seconds = None
    if arm != "before":
        started = time.perf_counter()
        kernel.get_concrete_function()
        trace_seconds = time.perf_counter() - started
    stages["traced"] = _memory(gpu)
    samples, first = [], None
    for _ in range(21):
        if gpu:
            tf.config.experimental.reset_memory_stats("GPU:0")
        started = time.perf_counter()
        output = kernel(*args)
        host = {key: value.numpy() for key, value in output.items()}
        seconds = time.perf_counter() - started
        started = time.perf_counter()
        record = {key: value.tolist() for key, value in host.items()}
        copy_seconds = time.perf_counter() - started
        if first is None:
            first = record
        else:
            assert record == first
        samples.append({"seconds": seconds, "host_list_seconds": copy_seconds, "memory": _memory(gpu)})
        del output, host, record
    stages["measured"] = _memory(gpu)
    traces, nodes, hlo = None, None, ""
    changed = arguments(dimension, seed=20260910, round_index=3)
    changed = (*changed[:2], changed[2] * 1.1, changed[3] + .02,
        changed[4] - .1, changed[5] + .05, changed[6] + 7, changed[7] + 2)
    changed_record = serializable(kernel(*changed))
    if arm != "before":
        traces = kernel.experimental_get_tracing_count()
        assert traces == 1
        graph = kernel.get_concrete_function().graph.as_graph_def()
        nodes = len(graph.node) + sum(len(fn.node_def) for fn in graph.library.function)
        if arm == "graph":
            assert not any(fn.attr["_XlaMustCompile"].b for fn in graph.library.function
                           if "_XlaMustCompile" in fn.attr)
        else:
            hlo = kernel.experimental_get_compiler_ir(*args)(stage="hlo")
            assert kernel.experimental_get_compiler_ir(*changed)(stage="hlo") == hlo
            assert len(re.findall(r"\bparameter\((\d+)\)", hlo[hlo.rfind("\nENTRY "):])) == 8
    report = {
        "role": "descriptive_paired_probe_dependency_cost", "baseline": "3582b4ac",
        "arm": arm, "dimension": dimension, "gpu": gpu, "batch_size": 4,
        "jit_compile": arm == "xla", "non_jit_role": "explicit_reference_exception" if arm != "xla" else None,
        "input_sha256": input_hashes, "original_source_sha256": checkpoint.hashes(),
        "build_seconds": build_seconds, "trace_seconds": trace_seconds,
        "stages": stages, "samples": samples, "result": first, "changed_result": changed_record,
        "trace_count": traces, "graph_nodes": nodes, "hlo_bytes": len(hlo.encode()),
        "host_peak_caveat": "Stage/call RSS and reported VmHWM are observations, not exact maxima; these test workers have no driver memory sampler.",
        "timing_scope": "complete_preparation_and_evaluation_with_all_tensor_copies; list_conversion_separate",
        "nonclaims": ["Single process per arm/extent; no terminal repeated or statistical ranking.",
                      "Original design generation and batch recurrence are extracted from the original source.",
                      "Both arms include fixed-capacity histories; this is not a full public initializer cost."],
    }
    directory = Path(request.config.getoption("xmlpath")).parent
    with (directory / "quadratic-probe-memory.json").open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")


def test_paired_probe_sparse_allocation_growth(request):
    """Sparse observations exclude per-call diagnostic record accumulation."""
    from tests.test_filter_repair_quadratic_numerics import equal_fields

    gpu = bool(tf.config.list_logical_devices("GPU"))
    callback = _target(4, 5)
    kernel = make_paired_probe_evaluator(callback, 5, 4)
    args = arguments(5)
    changed = arguments(5, seed=20260910, round_index=3)
    changed = (*changed[:2], changed[2] * 1.1, changed[3] + .02,
        changed[4] - .1, changed[5] + .05, changed[6] + 7, changed[7] + 2)
    reference = original_probes(callback, 5, 4)
    for inputs in (args, changed):
        equal_fields(kernel(*inputs), reference(*inputs))
    expected = (serializable(kernel(*args)), serializable(kernel(*changed)))

    def check(index):
        inputs = args if index % 2 == 0 else changed
        assert serializable(kernel(*inputs)) == expected[index % 2]

    for index in range(20):
        check(index)
    if gpu:
        tf.config.experimental.reset_memory_stats("GPU:0")
    observations = [{"additional_calls": 0, "memory": _memory(gpu)}]
    for block in range(5):
        for index in range(1000):
            check(index)
        observations.append({"additional_calls": (block + 1) * 1000, "memory": _memory(gpu)})
    assert kernel.experimental_get_tracing_count() == 1
    assert kernel.experimental_get_compiler_ir(*args)(stage="hlo") == kernel.experimental_get_compiler_ir(*changed)(stage="hlo")
    report = {"role": "sparse_changed_input_allocation_diagnostic", "gpu": gpu,
        "dimension": 5, "batch_size": 4, "additional_calls": 5000,
        "observations": observations, "trace_count": 1, "unchanged_hlo": True,
        "result": expected[0], "changed_result": expected[1],
        "nonclaims": ["No runtime cleanup, general leak-freedom, timing or full-initializer claim."]}
    directory = Path(request.config.getoption("xmlpath")).parent
    with (directory / "quadratic-probe-growth.json").open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
