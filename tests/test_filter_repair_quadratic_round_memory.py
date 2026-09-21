"""Fresh-process complete paired-controller cost diagnostics; not runtime code."""

import dataclasses
import hashlib
import json
import re
import time
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.inference.quadratic_rounds_tf import make_paired_quadratic_controller
from tests.test_filter_repair_quadratic_batch_memory import _memory
from tests.test_filter_repair_quadratic_batches import _equal_records
from tests.test_filter_repair_quadratic_numerics import original
from tests.test_filter_repair_quadratic_rounds import fixture, materialize, reference


@pytest.mark.parametrize("dimension", [3, 5])
@pytest.mark.parametrize("arm", ["before", "graph", "xla"])
def test_complete_paired_controller_costs(arm, dimension, request):
    gpu = bool(tf.config.list_logical_devices("GPU"))
    checkpoint, modules = original()
    callback, config, args = fixture(dimension, "nonquadratic")
    baseline = modules["trust"]
    input_hashes = [hashlib.sha256(tf.io.serialize_tensor(value).numpy()).hexdigest() for value in args]
    original_config = baseline.BatchedQuadraticCenterConfig(**{**dataclasses.asdict(config), "jit_compile_trust": False})
    stages = {"before": _memory(gpu)}
    started = time.perf_counter()
    kernel = None if arm == "before" else make_paired_quadratic_controller(callback, dimension, config, jit_compile=arm == "xla")
    build_seconds = time.perf_counter() - started
    stages["built"] = _memory(gpu)
    trace_seconds = None
    if kernel is not None:
        started = time.perf_counter()
        kernel.get_concrete_function()
        trace_seconds = time.perf_counter() - started
    stages["traced"] = _memory(gpu)

    def execute(arguments):
        if gpu:
            tf.config.experimental.reset_memory_stats("GPU:0")
        started = time.perf_counter()
        if kernel is None:
            options = dataclasses.replace(original_config, seed=int(arguments[2]))
            payload = baseline.refine_batched_quadratic_center(callback, arguments[0], arguments[1], config=options,
                _initial_evidence=(arguments[4], arguments[5]) if bool(arguments[3]) else None).payload()
            assert payload["diagnostics"].pop("full_initializer_xla") is False
            assert payload["diagnostics"].pop("jit_compile_trust") is False
            payload["diagnostics"].pop("fit_trace_count", None)
            payload["diagnostics"].pop("trust_trace_count", None)
            numerical_seconds, reporting_seconds = None, None
        else:
            computed = kernel(*arguments)
            int(computed["status"])
            numerical_seconds = time.perf_counter() - started
            report_started = time.perf_counter()
            payload = materialize(computed, config, dimension, int(arguments[2]))
            reporting_seconds = time.perf_counter() - report_started
            del computed
        elapsed = time.perf_counter() - started
        return payload, {"seconds": elapsed, "numerical_seconds": numerical_seconds,
            "reporting_seconds": reporting_seconds, "memory": _memory(gpu)}

    samples, first = [], None
    for _ in range(21):
        payload, sample = execute(args)
        if first is None:
            first = payload
        else:
            assert payload == first
        samples.append(sample)
        del payload
    stages["measured"] = _memory(gpu)
    changed = (args[0] + .01, args[1] * 1.1, args[2] + 1, *args[3:])
    changed_record, changed_cost = execute(changed)
    nodes, traces, hlo = None, None, ""
    if kernel is not None:
        traces = kernel.experimental_get_tracing_count()
        assert traces == 1
        concrete = kernel.get_concrete_function()
        graph = concrete.graph.as_graph_def()
        nodes = len(graph.node) + sum(len(fn.node_def) for fn in graph.library.function)
        if arm == "graph":
            assert not any(fn.attr["_XlaMustCompile"].b for fn in graph.library.function if "_XlaMustCompile" in fn.attr)
        else:
            hlo = kernel.experimental_get_compiler_ir(*args)(stage="hlo")
            assert kernel.experimental_get_compiler_ir(*changed)(stage="hlo") == hlo
            operands = len(re.findall(r"\bparameter\((\d+)\)", hlo[hlo.rfind("\nENTRY "):]))
            assert operands == len(args) + len(concrete.captured_inputs)
    assert first["accepted"]
    assert any("step" in row for row in first["diagnostics"]["rounds"])
    expected, _ = reference(callback, config, args)
    changed_expected, _ = reference(callback, config, changed)
    report = {"role": "descriptive_complete_paired_controller_cost", "baseline": "3582b4ac",
        "arm": arm, "dimension": dimension, "gpu": gpu, "jit_compile": arm == "xla",
        "non_jit_role": "explicit_graph_reference_exception" if arm != "xla" else None,
        "original_source_sha256": checkpoint.hashes(), "input_sha256": input_hashes,
        "build_seconds": build_seconds, "trace_seconds": trace_seconds,
        "stages": stages, "samples": samples, "result": first,
        "changed_result": changed_record, "changed_cost": changed_cost,
        "original_result": expected, "original_changed_result": changed_expected,
        "trace_count": traces, "graph_nodes": nodes, "hlo_bytes": len(hlo.encode()),
        "timing_scope": "Complete numerical refinement and complete report materialization, without initial multistart localization.",
        "host_peak_caveat": "RSS/VmHWM stage and call observations are not exact process maxima.",
        "nonclaims": ["Single process per arm/dimension; descriptive only, no statistical ranking or terminal repeats.",
            "Original trust is the graph precision authority, not pre-repair XLA cost.",
            "The candidate is not public wiring or a composed initial multistart cost."],
    }
    directory = Path(request.config.getoption("xmlpath")).parent
    with (directory / "quadratic-round-memory.json").open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    _equal_records(first, expected)
    _equal_records(changed_record, changed_expected)
