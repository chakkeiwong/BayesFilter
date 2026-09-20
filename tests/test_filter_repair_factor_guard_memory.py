"""Diagnostic-only fresh-process cost of the factor covariance domain guard."""

import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.inference import factor_correlation_geometry as current
from tests.test_filter_repair_factor_capacity import _memory
from tests.test_filter_repair_padded_factor import (
    _assert_runtime_parameters,
    _data,
    _public_numerics,
)

CHECKPOINT = "085baaaa"


@pytest.mark.parametrize("dimension", [3, 5])
@pytest.mark.parametrize("jit_compile", [False, True])
@pytest.mark.parametrize("arm", ["checkpoint", "candidate"])
def test_factor_guard_memory(dimension, jit_compile, arm, request):
    if arm == "checkpoint":
        source = subprocess.check_output(["git", "show",
            f"{CHECKPOINT}:bayesfilter/inference/factor_correlation_geometry.py"], text=True)
        spec = importlib.util.spec_from_loader("factor_guard_memory_reference", loader=None)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        exec(compile(source, spec.name, "exec"), module.__dict__)  # noqa: S102 - frozen diagnostic
    else:
        module = current
        source = Path(current.__file__).read_text()
    before = _memory()
    compact, _ = _data(dimension, 1, 32)
    if dimension == 3:
        scales = tf.constant([.8, 1.2, 1.4], tf.float64)
        loads = tf.constant([[.2], [-.1], [.3]], tf.float64)
    else:
        scales = tf.constant([.8, 1.1, .9, 1.2, .7], tf.float64)
        loads = tf.constant([[.3, 0.], [.12, .25], [-.2, .1], [.15, -.1], [.08, .2]], tf.float64)
    precision = tf.linalg.inv(current.factor_correlation_covariance(scales, loads))
    center, points, _, holdout, _, weights = compact
    inputs = (center, points, center[None, :] - points @ precision,
        holdout, center[None, :] - holdout @ precision, weights)
    hashes = [hashlib.sha256(tf.io.serialize_tensor(value).numpy()).hexdigest() for value in inputs]
    config = module.FactorCorrelationGeometryConfig(factor_count=1 if dimension == 3 else 2)
    program = module._make_factor_program(dimension, 2 * dimension + 1, 2 * dimension,
        config, jit_compile, module._prediction_jacobian_diagnostics)
    built = _memory()
    samples = []
    first = None
    for index in range(21):
        tf.config.experimental.reset_memory_stats("GPU:0")
        started = time.perf_counter()
        result = program(*inputs)
        public = tf.nest.map_structure(lambda value: value.numpy().tolist(), _public_numerics(result))
        elapsed = time.perf_counter() - started
        invalid = public.pop("invalid_covariance_evaluations", None)
        samples.append({"seconds": elapsed, "memory": _memory(), "invalid_evaluations": invalid})
        assert bool(public["finite"])
        assert invalid == 0 if arm == "candidate" else invalid is None
        if index == 0:
            first = public
        else:
            assert public == first
    measured = _memory()
    # Compiler inspection is deliberately after the memory/timing sequence.
    hlo = program.experimental_get_compiler_ir(*inputs, None)(stage="hlo") if jit_compile else ""
    if jit_compile and arm == "candidate":
        _assert_runtime_parameters(program, (*inputs, None), hlo)
    graph = program.get_concrete_function().graph.as_graph_def()
    nodes = len(graph.node) + sum(len(function.node_def) for function in graph.library.function)
    assert program.experimental_get_tracing_count() == 1
    report = {"role": "explanatory_covariance_guard_overhead_only", "arm": arm,
        "checkpoint": CHECKPOINT, "dimension": dimension, "factor_count": config.factor_count,
        "max_iterations": config.max_iterations, "jit_compile": jit_compile,
        "non_jit_role": None if jit_compile else "explicit_graph_reference_exception",
        "factor_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "input_sha256": hashes, "samples": samples, "result": first,
        "stages": {"before": before, "built": built, "after_measurement": measured},
        "graph_nodes": nodes, "hlo_bytes": len(hlo.encode()),
        "trace_count": program.experimental_get_tracing_count(),
        "timing_scope": "complete_numerical_call_and_public_field_materialization",
        "nonclaims": ["Single fresh process per arm; no terminal timing ranking or memory bound.",
            "Baseline isolates the domain guard, not the complete campaign repair.",
            "No optimizer, random stream, tolerance or target changes."]}
    directory = Path(request.config.getoption("xmlpath")).parent
    with (directory / "factor-guard-memory.json").open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print("FACTOR_GUARD_MEMORY " + json.dumps(report, sort_keys=True))
