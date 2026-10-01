"""Descriptive process-memory checks for compact and padded factor programs."""

import importlib.util
import inspect
import json
import subprocess
import sys
import time
from collections import namedtuple
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.inference import factor_correlation_geometry as factor
from tests.test_filter_repair_padded_factor import _data, _public_numerics


def _memory():
    host = {row.split()[0].rstrip(":"): int(row.split()[1]) * 1024
        for row in Path("/proc/self/status").read_text().splitlines()
        if row.startswith(("VmRSS:", "VmHWM:"))}
    return {"host": host, "gpu": tf.config.experimental.get_memory_info("GPU:0")}


@pytest.mark.parametrize("capacity", [0, 4, 32])
def test_factor_capacity_memory_and_records(capacity, request):
    _factor_capacity_memory(capacity, request, True)


@pytest.mark.parametrize("capacity", [0, 32])
def test_factor_capacity_graph_memory_and_records(capacity, request):
    _factor_capacity_memory(capacity, request, False)


@pytest.mark.parametrize("capacity", [0, 32])
@pytest.mark.parametrize("jit_compile", [False, True])
def test_factor_capacity_short_memory_and_records(capacity, jit_compile, request):
    _factor_capacity_memory(capacity, request, jit_compile, short=True)


def _factor_capacity_memory(capacity, request, jit_compile, short=False):
    # All arms have the same 11 active training rows. The zero-capacity label
    # selects the compact reference, not an alternative reuse rule.
    before = _memory()
    compact, padded = _data(5, 1, max(capacity, 1))
    arguments = padded if capacity else (*compact, None)
    config = factor.FactorCorrelationGeometryConfig(factor_count=2, **({"max_iterations": 4} if short else {}))
    program = factor._make_factor_program(5, 10 + capacity if capacity else 11, 10,
        config, jit_compile, factor._prediction_jacobian_diagnostics, padded_training=bool(capacity))
    built = _memory()
    samples = []
    for index in range(21):
        tf.config.experimental.reset_memory_stats("GPU:0")
        started = time.perf_counter()
        try:
            result = program(*arguments)
        except tf.errors.OpError as error:
            # Preserve graph-reference failures as failures of that numerical
            # arm, with no invented warm timing or parity evidence.
            report = {"role": "descriptive_capacity_memory_only", "capacity": capacity,
                "jit_compile": jit_compile, "max_iterations": config.max_iterations,
                "numerical_state": "failed", "exception": type(error).__name__, "message": str(error),
                "failed_call_seconds": time.perf_counter() - started,
                "stages": {"before": before, "built": built, "after_failure": _memory()},
                "completed_samples": samples, "warm_timing_available": False}
            directory = Path(request.config.getoption("xmlpath")).parent
            with (directory / "factor-capacity-memory.json").open("x") as handle:
                json.dump(report, handle, indent=2)
                handle.write("\n")
            if jit_compile or short:
                raise
            pytest.xfail("Preserved non-default graph assertion failure; see numerical_state=failed artifact")
        public = tf.nest.map_structure(lambda value: value.numpy().tolist(), _public_numerics(result))
        elapsed = time.perf_counter() - started
        samples.append({"seconds": elapsed, "memory": _memory()})
        assert public["finite"]
        if index == 0:
            first = public
        else:
            assert first == public
    measured = _memory()
    hlo = program.experimental_get_compiler_ir(*arguments)(stage="hlo") if jit_compile else ""
    graph = program.get_concrete_function().graph.as_graph_def()
    nodes = list(graph.node) + [node for function in graph.library.function for node in function.node_def]
    report = {"role": "descriptive_capacity_memory_only", "capacity": capacity, "jit_compile": jit_compile,
        "numerical_state": "passed",
        "dimension": 5, "factors": 2, "active_training_rows": 11,
        "max_iterations": config.max_iterations, "samples": samples, "result": first,
        "stages": {"before": before, "built": built, "after_measurement": measured},
        "graph_nodes": len(nodes), "hlo_bytes": len(hlo.encode()),
        "trace_count": program.experimental_get_tracing_count(),
        "nonclaims": ["No terminal repeats, general capacity bound or timing ranking.",
            "Same active data across capacities; HLO inspection is after measurement."]}
    directory = Path(request.config.getoption("xmlpath")).parent
    with (directory / "factor-capacity-memory.json").open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print("FACTOR_CAPACITY_MEMORY " + json.dumps(report, sort_keys=True))


@pytest.mark.parametrize("fixed_initial", [False, True])
def test_capacity_initial_arithmetic_localization(monkeypatch, request, fixed_initial):
    observed = namedtuple("CapacityInitialArithmetic", ("position", "objective_value",
        "objective_gradient", "converged", "failed", "weights", "offsets", "responses"))

    def observe(objective, *, initial_position, **_):
        loss = inspect.getclosurevars(objective.python_function).nonlocals["loss"]
        closure = inspect.getclosurevars(loss).nonlocals
        value, gradient = objective(initial_position)
        return observed(initial_position, value, gradient, tf.constant(False), tf.constant(False),
            closure["weights"], closure["train_z"], closure["train_response"])

    config = factor.FactorCorrelationGeometryConfig(factor_count=2)
    reports = []
    with monkeypatch.context() as context:
        context.setattr(factor.tfp.optimizer, "lbfgs_minimize", observe)
        factor._make_factor_program.cache_clear()
        try:
            for capacity in (0, 4, 32):
                compact, padded = _data(5, 1, max(capacity, 1))
                arguments = padded if capacity else (*compact, None)
                program = factor._make_factor_program(5, 10 + capacity if capacity else 11, 10,
                    config, True, factor._prediction_jacobian_diagnostics, padded_training=bool(capacity))
                state = program(*arguments)["optimizer"]
                fields = {key: getattr(state, key).numpy().tolist() for key in (
                    "position", "objective_value", "objective_gradient", "weights", "offsets", "responses")}
                if capacity == 0:
                    baseline = state
                    if fixed_initial:
                        context.setattr(factor, "_encode_state", lambda *_, position=baseline.position: position)
                        factor._make_factor_program.cache_clear()
                errors = {key: float(tf.reduce_max(tf.abs(getattr(baseline, key) -
                    (getattr(state, key)[:11] if key in ("weights", "offsets", "responses") else getattr(state, key)))))
                    for key in fields}
                reports.append({"capacity": capacity, "fields": fields, "errors": errors})
        finally:
            factor._make_factor_program.cache_clear()
    directory = Path(request.config.getoption("xmlpath")).parent
    with (directory / f"capacity-initial-arithmetic-{int(fixed_initial)}.json").open("x") as handle:
        json.dump(reports, handle, indent=2)
        handle.write("\n")
    print("CAPACITY_INITIAL_ARITHMETIC " + json.dumps([{k: v for k, v in row.items() if k != "fields"}
        for row in reports], sort_keys=True))


def test_compact_initialization_dispatch_localization(request):
    """Trial exact shape binding for COD only; runtime source is untouched."""
    source = subprocess.check_output(["git", "show", "7d08c68e:bayesfilter/inference/factor_correlation_geometry.py"], text=True)
    old_call = "            jit_compile=jit_compile,\n        )\n        dense_covariance"
    assert source.count(old_call) == 1
    source = source.replace(old_call,
        "            jit_compile=jit_compile,\n"
        "            active_training_rows=active_training_rows,\n        )\n        dense_covariance")
    old = "    root_weight = tf.sqrt(weights)[:, None]\n    raw = complete_orthogonal_lstsq(offsets * root_weight, responses * root_weight)"
    assert source.count(old) == 1
    source = source.replace(old, """    root_weight = tf.sqrt(weights)[:, None]
    if active_training_rows is None:
        raw = complete_orthogonal_lstsq(offsets * root_weight, responses * root_weight)
    else:
        dimension = int(offsets.shape[1])
        def shape_case(rows):
            def solve():
                return complete_orthogonal_lstsq(offsets[:rows] * root_weight[:rows], responses[:rows] * root_weight[:rows])
            return solve
        branches = tuple(shape_case(rows) for rows in range(2 * dimension, int(offsets.shape[0]) + 1))
        raw = tf.switch_case(active_training_rows - 2 * dimension, branches)
""")
    old = "    jit_compile: bool = True,\n) -> tf.Tensor:\n    root_weight"
    assert source.count(old) == 1
    source = source.replace(old, "    jit_compile: bool = True,\n    active_training_rows=None,\n) -> tf.Tensor:\n    root_weight")
    spec = importlib.util.spec_from_loader("compact_initializer_shape_diagnostic", loader=None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(source, spec.name, "exec"), module.__dict__)  # noqa: S102 - diagnostic-only source trial
    compact, padded = _data(5, 1, 32)
    config = factor.FactorCorrelationGeometryConfig(factor_count=2)
    observations = []
    for implementation, arguments, capacity in ((factor, (*compact, None), 11), (module, padded, 42)):
        program = implementation._make_factor_program(5, capacity, 10, config, True,
            implementation._prediction_jacobian_diagnostics, padded_training=capacity == 42)
        started = time.perf_counter()
        result = program(*arguments)
        public = tf.nest.map_structure(lambda value: value.numpy().tolist(), _public_numerics(result))
        observations.append({"public": public, "raw": result["optimizer"].position.numpy().tolist(),
            "cold_seconds": time.perf_counter() - started, "memory": _memory()})
    from tests.test_filter_repair_initializer_rounding import _record_differences

    report = {"role": "explanatory_compact_initialization_only", "observations": observations,
        "differences": _record_differences(observations[1]["public"], observations[0]["public"])}
    directory = Path(request.config.getoption("xmlpath")).parent
    with (directory / "compact-initialization-dispatch.json").open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print("COMPACT_INITIALIZATION_DISPATCH " + json.dumps({key: value for key, value in report.items()
        if key != "observations"}, sort_keys=True))
