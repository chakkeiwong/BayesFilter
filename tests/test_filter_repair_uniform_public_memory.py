"""Fresh-process public refinement cost comparisons with the original source."""

import dataclasses
import hashlib
import json
import re
import time
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.inference import batched_quadratic_center as current
from bayesfilter.inference import quadratic_rounds_tf as controllers
from tests.test_filter_repair_quadratic_batch_memory import _memory
from tests.test_filter_repair_quadratic_batches import _equal_records
from tests.test_filter_repair_quadratic_numerics import original
from tests.test_filter_repair_quadratic_rounds import public_record, reference
from tests.test_filter_repair_uniform_rounds import fixture


@pytest.mark.parametrize("dimension", [3, 5])
@pytest.mark.parametrize("arm", ["before", "graph", "xla"])
def test_public_uniform_refinement_costs(arm, dimension, request):
    checkpoint, modules = original()
    gpu = bool(tf.config.list_logical_devices("GPU"))
    callback, config, args = fixture(dimension, "nonquadratic")
    baseline = modules["trust"]
    config_type = baseline.BatchedQuadraticCenterConfig if arm == "before" else current.BatchedQuadraticCenterConfig
    endpoint = baseline.refine_batched_quadratic_center if arm == "before" else current.refine_batched_quadratic_center
    controllers.clear_paired_quadratic_controller_cache()
    input_hashes = [hashlib.sha256(tf.io.serialize_tensor(value).numpy()).hexdigest() for value in args]
    stages = {"before": _memory(gpu)}

    def execute(arguments):
        options = config_type(**{**dataclasses.asdict(config), "seed": int(arguments[2]), "jit_compile_trust": arm == "xla"})
        if gpu:
            tf.config.experimental.reset_memory_stats("GPU:0")
        started = time.perf_counter()
        payload = endpoint(callback, arguments[0], arguments[1], config=options,
            _initial_evidence=(arguments[4], arguments[5]) if bool(arguments[3]) else None).payload()
        elapsed = time.perf_counter() - started
        return payload, {"seconds": elapsed, "memory": _memory(gpu)}

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
    second, second_cost = execute(changed)
    nodes, hlo, traces = None, "", None
    if arm != "before":
        options = dataclasses.replace(config, jit_compile_trust=arm == "xla")
        kernel = controllers.quadratic_controller(callback, dimension, options, jit_compile=arm == "xla")
        traces = kernel.experimental_get_tracing_count()
        assert traces == 1
        concrete = kernel.get_concrete_function()
        graph = concrete.graph.as_graph_def()
        nodes = len(graph.node) + sum(len(fn.node_def) for fn in graph.library.function)
        if arm == "xla":
            hlo = kernel.experimental_get_compiler_ir(*args)(stage="hlo")
            assert hlo == kernel.experimental_get_compiler_ir(*changed)(stage="hlo")
            arity = len(args) + len(concrete.captured_inputs)
            assert len(re.findall(r"\bparameter\((\d+)\)", hlo[hlo.rfind("\nENTRY "):])) == arity
        else:
            assert not any(fn.attr["_XlaMustCompile"].b for fn in graph.library.function if "_XlaMustCompile" in fn.attr)
    expected, original_traces = reference(callback, config, args)
    expected_changed, original_changed_traces = reference(callback, config, changed)
    public_payloads = [first, second]
    numerical = []
    for value, trace in zip(public_payloads, (original_traces, original_changed_traces), strict=True):
        # Copy the payload so the exact public schema remains in the artifact.
        value = json.loads(json.dumps(value, allow_nan=False))
        if arm == "before":
            assert value["diagnostics"].pop("full_initializer_xla") is False
            assert value["diagnostics"].pop("jit_compile_trust") is False
            for key, count in trace.items():
                assert value["diagnostics"].pop(key) == count
        else:
            value = public_record(value, jit=arm == "xla", original_traces=trace)
        numerical.append(value)
    report = {"role": "descriptive_public_uniform_refinement_cost", "baseline": "3582b4ac", "arm": arm,
        "dimension": dimension, "gpu": gpu, "jit_compile": arm == "xla",
        "non_jit_role": "explicit_graph_reference_exception" if arm != "xla" else None,
        "original_source_sha256": checkpoint.hashes(), "input_sha256": input_hashes,
        "stages": stages, "samples": samples, "public_payloads": public_payloads,
        "result": numerical[0], "changed_result": numerical[1], "changed_cost": second_cost,
        "original_result": expected, "original_changed_result": expected_changed,
        "trace_count": traces, "graph_nodes": nodes, "hlo_bytes": len(hlo.encode()),
        "timing_scope": "Actual public validation, cache lookup/factory, complete refinement and full payload; cold includes tracing/compilation.",
        "nonclaims": ["One fresh process per arm/dimension; descriptive only, no statistical ranking.",
            "Same target reuse; no arbitrary target turnover or native executable eviction claim.",
            "Multistart locator composition and actual DZ5 target remain separate."]}
    directory = Path(request.config.getoption("xmlpath")).parent
    with (directory / "uniform-public-memory.json").open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    _equal_records(numerical[0], expected)
    _equal_records(numerical[1], expected_changed)
