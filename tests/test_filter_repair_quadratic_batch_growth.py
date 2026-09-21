"""Sparse observations of evaluator allocation reuse, not performance ranking."""

import gc
import json
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.inference.quadratic_batch_evaluation_tf import (
    make_quadratic_batch_evaluator,
)
from tests.filter_repair_quadratic_batch_reference import original_evaluator
from tests.test_filter_repair_quadratic_batch_memory import _memory
from tests.test_filter_repair_quadratic_batches import _target


@pytest.mark.parametrize("arm", ["before", "xla"])
def test_sparse_fixed_shape_allocation_growth(arm, request):
    _observe_growth(arm, 4, 100, request)


def test_long_sparse_xla_allocation_growth(request):
    _observe_growth("xla", 10, 1000, request)


def _observe_growth(arm, blocks, calls_per_block, request):
    dimension, capacity, batch = 5, 128, 4
    gpu = bool(tf.config.list_logical_devices("GPU"))
    callback = _target(batch, dimension)
    program = (original_evaluator(callback, dimension, batch) if arm == "before" else
               make_quadratic_batch_evaluator(callback, dimension, batch, capacity))
    arguments = (
        tf.reshape(tf.range(capacity * dimension, dtype=tf.float64), [capacity, dimension]) / 100.,
        tf.constant(capacity), tf.constant(True), tf.zeros([dimension], tf.float64),
        tf.constant(-1000., tf.float64), tf.zeros([dimension], tf.float64),
        tf.constant(0, tf.int64), tf.constant(-1, tf.int64),
    )

    def call():
        output = program(*arguments)
        result = {key: value.numpy().tolist() if tf.is_tensor(value) else value for key, value in output.items()}
        result.pop("chunks", None)
        return result

    first = call()
    for _ in range(20):
        assert call() == first
    if gpu:
        tf.config.experimental.reset_memory_stats("GPU:0")
    observations = [{"additional_calls": 0, "memory": _memory(gpu)}]
    for block in range(blocks):
        for _ in range(calls_per_block):
            assert call() == first
        observations.append({"additional_calls": (block + 1) * calls_per_block, "memory": _memory(gpu)})
    gc.collect()  # Diagnostic-only; all ordinary-execution observations precede it.
    after_collection = _memory(gpu)
    trace_count = program.experimental_get_tracing_count() if arm == "xla" else None
    if arm == "xla":
        assert trace_count == 1
    report = {"role": "sparse_fixed_shape_memory_diagnostic", "arm": arm,
              "gpu": gpu, "dimension": dimension, "capacity": capacity,
              "initial_and_warm_calls": 21, "additional_calls": blocks * calls_per_block,
              "observations": observations, "after_diagnostic_collection": after_collection,
              "trace_count": trace_count, "identical_repeated_records": True,
              "result": first, "nonclaims": ["No runtime cleanup or general leak-freedom claim.",
                                            "This is not a timing comparison."]}
    directory = Path(request.config.getoption("xmlpath")).parent
    with (directory / "quadratic-batch-growth.json").open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
