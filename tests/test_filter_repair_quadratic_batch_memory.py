"""Descriptive fresh-process costs of the ordered batch evaluator only."""

import hashlib
import inspect
import json
import time
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.inference import quadratic_batch_evaluation_tf as runtime
from tests.filter_repair_quadratic_batch_reference import original_evaluator
from tests.test_filter_repair_quadratic_batches import _target


def _memory(gpu):
    host = {row.split()[0].rstrip(":"): int(row.split()[1]) * 1024
            for row in Path("/proc/self/status").read_text().splitlines()
            if row.startswith(("VmRSS:", "VmHWM:"))}
    return {"host": host, "gpu": tf.config.experimental.get_memory_info("GPU:0") if gpu else None}


@pytest.mark.parametrize("dimension,capacity", [(3, 32), (5, 128)])
@pytest.mark.parametrize("arm", ["before", "graph", "xla"])
def test_ordered_batch_costs(dimension, capacity, arm, request):
    gpu = bool(tf.config.list_logical_devices("GPU"))
    stages = {"before": _memory(gpu)}
    batch = 4
    arguments = (
        tf.reshape(tf.range(capacity * dimension, dtype=tf.float64), [capacity, dimension]) / 100.,
        tf.constant(capacity), tf.constant(True), tf.zeros([dimension], tf.float64),
        tf.constant(-1000., tf.float64), tf.zeros([dimension], tf.float64),
        tf.constant(0, tf.int64), tf.constant(-1, tf.int64),
    )
    input_hashes = [hashlib.sha256(tf.io.serialize_tensor(item).numpy()).hexdigest() for item in arguments]
    started = time.perf_counter()
    callback = _target(batch, dimension)
    program = (original_evaluator(callback, dimension, batch) if arm == "before" else
               runtime.make_quadratic_batch_evaluator(callback, dimension, batch, capacity, jit_compile=arm == "xla"))
    build_seconds = time.perf_counter() - started
    stages["built"] = _memory(gpu)
    trace_seconds = None
    if arm != "before":
        started = time.perf_counter()
        program.get_concrete_function()
        trace_seconds = time.perf_counter() - started
    stages["traced"] = _memory(gpu)
    samples, first = [], None
    for index in range(21):
        if gpu:
            tf.config.experimental.reset_memory_stats("GPU:0")
        started = time.perf_counter()
        output = program(*arguments)
        # Complete output synchronization; no target work or serialization is
        # deferred beyond the execution measurement.
        host_output = {key: value.numpy() if tf.is_tensor(value) else value for key, value in output.items()}
        elapsed = time.perf_counter() - started
        started = time.perf_counter()
        result = {key: value.tolist() if hasattr(value, "tolist") else value for key, value in host_output.items()}
        copy_seconds = time.perf_counter() - started
        if first is None:
            first = result
        else:
            assert result == first
        samples.append({"seconds": elapsed, "host_list_seconds": copy_seconds, "memory": _memory(gpu)})
        del output, host_output, result
    stages["measured"] = _memory(gpu)
    traces, nodes, hlo = None, None, ""
    if arm != "before":
        traces = program.experimental_get_tracing_count()
        assert traces == 1
        graph = program.get_concrete_function().graph.as_graph_def()
        nodes = len(graph.node) + sum(len(fn.node_def) for fn in graph.library.function)
        if arm == "graph":
            assert not any(fn.attr["_XlaMustCompile"].b for fn in graph.library.function
                           if "_XlaMustCompile" in fn.attr)
        else:
            hlo = program.experimental_get_compiler_ir(*arguments)(stage="hlo")
    sources = {str(Path(module.__file__).relative_to(Path.cwd())):
               hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
               for module in (runtime, inspect.getmodule(original_evaluator), inspect.getmodule(_target))}
    report = {
        "role": "descriptive_quadratic_batch_dependency_cost", "baseline": "bd36a89b:evaluate",
        "arm": arm, "dimension": dimension, "capacity": capacity, "batch_size": batch,
        "jit_compile": None if arm == "before" else arm == "xla", "gpu": gpu,
        "non_jit_role": "explicit_reference_exception" if arm != "xla" else None,
        "target": "same_plain_analytical_quadratic_callback_all_arms",
        "input_sha256": input_hashes, "source_sha256": sources,
        "build_seconds": build_seconds, "trace_seconds": trace_seconds,
        "stages": stages, "samples": samples, "result": first,
        "trace_count": traces, "graph_nodes": nodes, "hlo_bytes": len(hlo.encode()),
        "host_peak_caveat": "Stage/call RSS and reported VmHWM are observations, not exact maxima. These test workers have no driver memory sampler.",
        "timing_scope": "complete_chunk_program_and_all_tensor_copies; list_conversion_separate",
        "nonclaims": ["One process per arm/extent; no terminal repeated or statistical ranking.",
                      "Enclosing fit-round controller and its complete costs remain open."],
    }
    directory = Path(request.config.getoption("xmlpath")).parent
    with (directory / "quadratic-batch-memory.json").open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
