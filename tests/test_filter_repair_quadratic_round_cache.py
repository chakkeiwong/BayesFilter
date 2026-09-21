"""Bounded paired-controller cache identity, reuse and ownership diagnostics."""

import dataclasses
import gc
import json
import re
import time
import weakref
from pathlib import Path

import tensorflow as tf

from bayesfilter.inference import quadratic_rounds_tf as runtime
from tests.test_filter_repair_quadratic_batches import _equal_records
from tests.test_filter_repair_quadratic_round_growth import memory
from tests.test_filter_repair_quadratic_rounds import fixture, materialize, reference


class EqualCallable:
    """Distinct, unhashable targets must not share a program through equality."""

    def __init__(self, callback, offset=0.):
        self.callback = callback
        self.offset = offset

    def __eq__(self, other):
        return isinstance(other, EqualCallable)

    def __call__(self, points):
        values, scores, valid = self.callback(points)
        return values + self.offset, scores, valid


def test_same_callback_reuses_runtime_inputs(request):
    runtime.clear_paired_quadratic_controller_cache()
    callback, config, args = fixture(5, "nonquadratic")
    callback = EqualCallable(callback)
    changed = (args[0] + .01, args[1] * 1.1, args[2] + 1, *args[3:])
    expected = [reference(callback, config, values)[0] for values in (args, changed)]
    gpu = bool(tf.config.list_logical_devices("GPU"))
    directory = Path(request.config.getoption("xmlpath")).parent
    observations, first = [], None
    for index in range(21):
        values = (args, changed)[index % 2]
        options = dataclasses.replace(config, seed=int(values[2]), paired_steps=list(config.paired_steps) if index % 2 else config.paired_steps)
        started = time.perf_counter()
        program = runtime.paired_quadratic_controller(callback, 5, options)
        if first is None:
            first = program
        assert program is first
        record = materialize(program(*values), options, 5, int(values[2]))
        elapsed = time.perf_counter() - started
        _equal_records(record, expected[index % 2])
        observations.append({"call": index, "seconds": elapsed, "memory": memory(gpu)})
    assert program.experimental_get_tracing_count() == 1
    hlo = program.experimental_get_compiler_ir(*args)(stage="hlo")
    assert hlo == program.experimental_get_compiler_ir(*changed)(stage="hlo")
    arity = len(args) + len(program.get_concrete_function().captured_inputs)
    assert len(re.findall(r"\bparameter\((\d+)\)", hlo[hlo.rfind("\nENTRY "):])) == arity
    with (directory / "quadratic-round-cache-reuse.json").open("x") as handle:
        json.dump({"observations": observations, "runtime_operands": arity, "trace_count": 1,
            "cache_capacity": 1, "gpu": gpu, "original_records": expected,
            "nonclaims": ["Single-target reuse only; no native executable eviction or arbitrary target-turnover memory bound."]}, handle, indent=2, allow_nan=False)
        handle.write("\n")
    runtime.clear_paired_quadratic_controller_cache()


def test_cache_identity_modes_and_resource_release(request):
    runtime.clear_paired_quadratic_controller_cache()
    counter = tf.Variable(0, dtype=tf.int64)
    callback, config, args = fixture(3, "nonquadratic", counter=counter)
    callback = EqualCallable(callback)
    first = runtime.paired_quadratic_controller(callback, 3, config)
    result = materialize(first(*args), config, 3, int(args[2]))
    assert int(counter) == result["diagnostics"]["callback_batches"]
    concrete = first.get_concrete_function()
    graph_ref, variable_ref, callback_ref = weakref.ref(concrete.graph), weakref.ref(counter), weakref.ref(callback)
    replacement_callback, _, _ = fixture(3, "nonquadratic")
    replacement_callback = EqualCallable(replacement_callback, offset=10.)
    assert callback == replacement_callback
    replacement = runtime.paired_quadratic_controller(replacement_callback, 3, config)
    assert replacement is not first
    replacement_result = materialize(replacement(*args), config, 3, int(args[2]))
    expected, _ = reference(replacement_callback, config, args)
    _equal_records(replacement_result, expected)
    assert replacement_result["center_value"] != result["center_value"]
    repeated = materialize(concrete(*args), config, 3, int(args[2]))
    _equal_records(repeated, result)
    # Changes to numerical configuration and explicit execution mode are separate.
    alternate = runtime.paired_quadratic_controller(replacement_callback, 3,
        dataclasses.replace(config, max_fit_rounds=1))
    assert alternate is not replacement
    graph = runtime.paired_quadratic_controller(replacement_callback, 3, config, jit_compile=False)
    graph_result = materialize(graph(*args), config, 3, int(args[2]))
    _equal_records(graph_result, expected)
    definition = graph.get_concrete_function().graph.as_graph_def()
    assert not any(fn.attr["_XlaMustCompile"].b for fn in definition.library.function if "_XlaMustCompile" in fn.attr)
    runtime.clear_paired_quadratic_controller_cache()
    del first, callback, counter, concrete
    gc.collect()
    released = {"graph": graph_ref() is None, "target_variable": variable_ref() is None, "callback": callback_ref() is None}
    directory = Path(request.config.getoption("xmlpath")).parent
    with (directory / "quadratic-round-cache-ownership.json").open("x") as handle:
        json.dump({"released": released, "execution_after_replacement": True, "graph_reference_has_no_xla": True,
            "distinct_equal_callbacks": True, "original_record": expected}, handle, indent=2, allow_nan=False)
        handle.write("\n")
    assert all(released.values())

def test_public_reuse_reads_changed_tensorflow_target_state():
    from bayesfilter.inference.batched_quadratic_center import (
        refine_batched_quadratic_center,
    )
    from tests.test_filter_repair_quadratic_rounds import public_record

    runtime.clear_paired_quadratic_controller_cache()
    base, config, args = fixture(3, "move")
    strength = tf.Variable(1., dtype=tf.float64)

    def callback(points):
        values, scores, valid = base(points)
        return strength * values, strength * scores, valid

    first = None
    for multiplier in (1., 2.):
        strength.assign(multiplier)
        expected, traces = reference(callback, config, args)
        result = refine_batched_quadratic_center(callback, args[0], args[1], config=config).payload()
        program = runtime.paired_quadratic_controller(callback, 3, config)
        assert program.experimental_get_tracing_count() == 1
        if first is None:
            first = program
        assert first is program
        _equal_records(public_record(result, jit=True, original_traces=traces), expected)
    runtime.clear_paired_quadratic_controller_cache()
