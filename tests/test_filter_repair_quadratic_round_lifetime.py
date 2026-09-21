"""Repeated fresh-controller construction and captured-resource diagnostics."""

import gc
import hashlib
import json
import time
import weakref
from pathlib import Path

import tensorflow as tf

from bayesfilter.inference.quadratic_rounds_tf import make_paired_quadratic_controller
from tests.test_filter_repair_quadratic_batches import _equal_records
from tests.test_filter_repair_quadratic_numerics import original
from tests.test_filter_repair_quadratic_round_growth import memory
from tests.test_filter_repair_quadratic_rounds import fixture, materialize, reference


def test_fresh_controller_lifetime(request):
    directory = Path(request.config.getoption("xmlpath")).parent
    gpu = bool(tf.config.list_logical_devices("GPU"))
    checkpoint, _ = original()
    observations, graph_refs, variable_refs, program_refs = [], [], [], []
    initial_records = None
    input_hashes = None
    before = memory(gpu)
    for iteration in range(8):
        counter = tf.Variable(0, dtype=tf.int64)
        callback, config, args = fixture(5, "nonquadratic", counter=counter)
        changed = (args[0] + .01, args[1] * 1.1, args[2] + 1, *args[3:])
        if input_hashes is None:
            input_hashes = [hashlib.sha256(tf.io.serialize_tensor(value).numpy()).hexdigest() for value in args]
        started = time.perf_counter()
        program = make_paired_quadratic_controller(callback, 5, config)
        records = [materialize(program(*values), config, 5, int(values[2])) for values in (args, changed)]
        elapsed = time.perf_counter() - started
        assert program.experimental_get_tracing_count() == 1
        assert int(counter) == sum(record["diagnostics"]["callback_batches"] for record in records)
        if initial_records is None:
            initial_records = records
        else:
            assert records == initial_records
        graph_refs.append(weakref.ref(program.get_concrete_function().graph))
        variable_refs.append(weakref.ref(counter))
        program_refs.append(weakref.ref(program))
        del program, callback, counter, args, changed, records
        observation = {"iteration": iteration + 1, "build_and_two_calls_seconds": elapsed,
            "memory": memory(gpu), "live_graphs": sum(ref() is not None for ref in graph_refs),
            "live_variables": sum(ref() is not None for ref in variable_refs),
            "live_programs": sum(ref() is not None for ref in program_refs)}
        observations.append(observation)
        with (directory / "quadratic-round-lifetime-progress.jsonl").open("a") as handle:
            handle.write(json.dumps(observation, allow_nan=False) + "\n")
    # Explanatory Python ownership check only; not a proposed runtime cleanup.
    collected = gc.collect()
    after_collection = {"collected": collected, "memory": memory(gpu),
        "live_graphs": sum(ref() is not None for ref in graph_refs),
        "live_variables": sum(ref() is not None for ref in variable_refs),
        "live_programs": sum(ref() is not None for ref in program_refs)}
    callback, config, args = fixture(5, "nonquadratic")
    changed = (args[0] + .01, args[1] * 1.1, args[2] + 1, *args[3:])
    originals = [reference(callback, config, values)[0] for values in (args, changed)]
    report = {"role": "fresh_controller_construction_and_resource_lifetime_diagnostic", "gpu": gpu,
        "before": before, "observations": observations, "after_python_collection": after_collection,
        "baseline": "3582b4ac", "original_source_sha256": checkpoint.hashes(), "input_sha256": input_hashes,
        "initial_results": initial_records, "original_results": originals,
        "nonclaims": ["Python graph/resource release is not native XLA executable eviction.",
            "Eight fresh controllers do not establish arbitrary-scope leak freedom or public cache behavior.",
            "Final explicit collection is diagnostic only; no runtime cleanup was injected."]}
    with (directory / "quadratic-round-lifetime.json").open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    for actual, expected in zip(initial_records, originals, strict=True):
        _equal_records(actual, expected)
    assert after_collection["live_graphs"] == 0
    assert after_collection["live_variables"] == 0
    assert after_collection["live_programs"] == 0
