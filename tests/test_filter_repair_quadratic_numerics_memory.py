"""Fresh-process diagnostic costs of dense/paired fits and the SPD trust solver."""

import hashlib
import json
import time
from pathlib import Path

import pytest
import tensorflow as tf

from tests.test_filter_repair_quadratic_batch_memory import _memory
from tests.test_filter_repair_quadratic_numerics import (
    inputs,
    original,
    program,
    serializable,
)


@pytest.mark.parametrize("dimension", [3, 5])
@pytest.mark.parametrize("kind", ["dense", "paired", "trust"])
@pytest.mark.parametrize("arm", ["before", "graph", "xla"])
def test_numerical_dependency_costs(arm, kind, dimension, request):
    gpu = bool(tf.config.list_logical_devices("GPU"))
    arguments = inputs(kind, dimension)
    input_hashes = [hashlib.sha256(tf.io.serialize_tensor(item).numpy()).hexdigest() for item in arguments]
    # Import the same original closure in all arms, outside the measurement.
    # Its functions stay untraced except in the before arm.
    checkpoint, _ = original()
    stages = {"before": _memory(gpu)}
    started = time.perf_counter()
    kernel = program(kind, dimension, source="original" if arm == "before" else "current", jit=arm == "xla")
    build_seconds = time.perf_counter() - started
    stages["built"] = _memory(gpu)
    started = time.perf_counter()
    concrete = kernel.get_concrete_function()
    trace_seconds = time.perf_counter() - started
    stages["traced"] = _memory(gpu)
    samples, first = [], None
    for _ in range(21):
        if gpu:
            tf.config.experimental.reset_memory_stats("GPU:0")
        started = time.perf_counter()
        output = kernel(*arguments)
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
    traces = kernel.experimental_get_tracing_count()
    assert traces == 1
    graph = concrete.graph.as_graph_def()
    if arm != "xla":
        assert not any(function.attr["_XlaMustCompile"].b for function in graph.library.function
                       if "_XlaMustCompile" in function.attr)
    hlo = kernel.experimental_get_compiler_ir(*arguments)(stage="hlo") if arm == "xla" else ""
    changed = tuple(value * tf.constant(1.1, tf.float64) for value in arguments)
    changed_output = kernel(*changed)
    changed_record = serializable(changed_output)
    del changed_output
    if arm == "xla":
        assert kernel.experimental_get_compiler_ir(*changed)(stage="hlo") == hlo
    assert kernel.experimental_get_tracing_count() == 1
    report = {
        "role": "descriptive_quadratic_numerical_dependency_cost", "baseline": "3582b4ac",
        "arm": arm, "kind": kind, "dimension": dimension, "gpu": gpu,
        "jit_compile": arm == "xla", "non_jit_role": "explicit_reference_exception" if arm != "xla" else None,
        "input_sha256": input_hashes, "original_source_sha256": checkpoint.hashes(),
        "build_seconds": build_seconds, "trace_seconds": trace_seconds,
        "stages": stages, "samples": samples, "result": first, "changed_result": changed_record,
        "trace_count": traces, "graph_nodes": len(graph.node) + sum(len(fn.node_def) for fn in graph.library.function),
        "hlo_bytes": len(hlo.encode()),
        "host_peak_caveat": "Stage/call RSS and reported VmHWM are observations, not exact maxima; these test workers have no driver memory sampler.",
        "timing_scope": "complete_numerical_kernel_and_all_tensor_copies; list_conversion_separate",
        "nonclaims": ["Single process per arm/extent; no terminal repeated or statistical ranking.",
                      "Original graph is the precision authority; original trust XLA had demonstrated errors.",
                      "Enclosing fit-round controller and complete initializer costs remain open."],
    }
    directory = Path(request.config.getoption("xmlpath")).parent
    with (directory / "quadratic-numerics-memory.json").open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
