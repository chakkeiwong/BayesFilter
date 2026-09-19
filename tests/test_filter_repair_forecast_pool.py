"""Pinned scalar authorities for compiled external CPU forecast generation."""

import hashlib
import importlib.util
import subprocess
import sys

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import cpu_forecast_pool as pool
from bayesfilter.nonlinear import ssl_lstm_complexity_predictive_tf as forecast

D = tf.float64


@pytest.fixture(scope="module")
def previous():
    source = subprocess.check_output(["git", "show",
        "b7cfbb14:bayesfilter/nonlinear/ssl_lstm_complexity_predictive_tf.py"], text=True)
    spec = importlib.util.spec_from_loader("forecast_before_compiled_shards", loader=None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(source, "forecast_before_compiled_shards.py", "exec"), module.__dict__)  # noqa: S102
    return module


def _rows(count):
    rows = tf.constant([[.35, -.08, .65, .05]], D) + .001*tf.reshape(
        tf.cast(tf.range(4*count), D), [count, 4])
    seeds = tf.stack((tf.fill([count], 20260719), tf.range(70001, 70001+count)), axis=1)
    return rows, seeds


@pytest.mark.parametrize("q", [1, 2])
@pytest.mark.parametrize("count", [1, 3])
@pytest.mark.parametrize("jit", [False, True])
def test_complete_seeded_shard_preserves_scalar_values_replay_status_and_hlo(previous, q, count, jit):
    rows, seeds = _rows(count)
    reference = previous.ComplexityForecastWorker(q)
    if not jit:
        # Both source arms must use the same execution mode. In q=2 the
        # pre-existing principal-root filter's graph/XLA rounding already
        # differs beyond the before/after tolerance.
        target = forecast.complexity_posterior_target(q, jit_compile=False)
        assert target.target_signature() == reference.target.target_signature()
        original_program = previous.complexity_forecast_compiled_program(target,
            draw_count=1, replication_count=forecast.FORECAST_REPLICATION_COUNT)
        reference.program = tf.function(original_program.python_function,
            input_signature=original_program.input_signature, jit_compile=False, autograph=False)
    expected_rows = [reference.evaluate(row, seed) for row, seed in zip(rows, seeds, strict=True)]
    expected = tuple(tf.stack([row[index] for row in expected_rows]) for index in range(3))
    worker = forecast.ComplexityForecastWorker(q)
    program = worker.make_shard_program(count, jit_compile=jit)
    actual = program(rows, seeds)
    assert bool(tf.reduce_all(actual[3]))
    for result, authority in zip(actual[:3], expected, strict=True):
        tf.debugging.assert_near(result, authority, atol=1e-10, rtol=1e-10)
    replay = program(rows, seeds)
    for result, repeated in zip(actual, replay, strict=True):
        tf.debugging.assert_equal(result, repeated)
    changed = program(rows + .001, seeds + tf.constant([0, 71], tf.int32))
    assert bool(tf.reduce_any(changed[2] != actual[2]))
    assert program.experimental_get_tracing_count() == 1
    assert worker.make_shard_program(count, jit_compile=jit) is program
    graph = program.get_concrete_function().graph.as_graph_def()
    nodes = [*graph.node, *(node for function in graph.library.function for node in function.node_def)]
    assert not {node.op for node in nodes} & {"PyFunc", "EagerPyFunc", "PyFuncStateless"}
    if jit:
        assert "HloModule" in program.experimental_get_compiler_ir(rows, seeds)(stage="hlo")
    else:
        assert not any(function.attr.get("_XlaMustCompile") and function.attr["_XlaMustCompile"].b
                       for function in graph.library.function)


def test_invalid_forecast_shard_keeps_existing_worker_rejection(previous):
    rows, seeds = _rows(2)
    rows = tf.tensor_scatter_nd_update(rows, [[1, 0]], [tf.constant(float("nan"), D)])
    reference = previous.ComplexityForecastWorker(1)
    worker = forecast.ComplexityForecastWorker(1)
    with pytest.raises(previous.ComplexityPredictiveError, match="validity gate"):
        reference.evaluate(rows[1], seeds[1])
    with pytest.raises(forecast.ComplexityPredictiveError, match="validity gate"):
        worker.evaluate_batch(rows, seeds)
    assert worker.make_shard_program(2)(rows, seeds)[3].numpy().tolist() == [True, False]


@pytest.mark.parametrize("dtype", [np.float64, np.int32])
def test_forecast_raw_identity_and_serialization_preserve_contiguous_bytes(dtype):
    values = np.arange(24, dtype=dtype).reshape(6, 4)[::-2, ::2]
    if dtype == np.float64:
        values[0, 0] = -0.0
    expected = hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()
    assert pool._array_hash(values) == expected
    tensor = tf.convert_to_tensor(values)
    assert pool._array_hash(tensor) == expected
    restored = tf.io.parse_tensor(pool._serialize_tensor(tensor), tensor.dtype)
    assert pool._array_hash(restored) == expected


def test_worker_evaluates_whole_ordered_shard_once_with_serialized_results(monkeypatch):
    rows, seeds = _rows(3)
    calls = []

    def numerical(values, roots):
        calls.append((values, roots))
        result = tf.broadcast_to(values[:, 0, None, None], [3, 2, 10])
        return result, tf.ones_like(result), result + tf.cast(roots[:, 1, None, None], D)

    monkeypatch.setattr(pool, "_WORKER_FORECAST", numerical)
    monkeypatch.setattr(pool, "_WORKER_METADATA", {"pid": 10})
    result = pool._worker_eval({"rows": pool._serialize_tensor(rows), "seeds": pool._serialize_tensor(seeds),
        "worker_index": 1, "item_start": 2, "item_stop": 5, "request_id": "ordered",
        "rows_hash": "request_rows", "seeds_hash": "request_seeds"})
    assert len(calls) == 1 and calls[0][0].shape == (3, 4)
    assert result["shard_rows_hash"] == pool._array_hash(rows)
    assert result["shard_seeds_hash"] == pool._array_hash(seeds)
    assert result["item_start"] == 2 and result["item_stop"] == 5
    means = tf.io.parse_tensor(result["conditional_means"], D)
    tf.debugging.assert_equal(means[:, 0, 0], rows[:, 0])


@pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
def test_compatible_scalar_factory_uses_compiled_mapping_and_retains_order(dtype):
    def scalar(row, seed):
        mean = tf.fill([2, 10], tf.cast(row[0], dtype))
        return mean, tf.ones_like(mean), mean + tf.cast(seed[1], dtype)

    evaluate = pool._scalar_shard_evaluator(scalar)
    rows, seeds = _rows(3)
    result = evaluate(rows, seeds)
    expected = tf.nest.map_structure(lambda *values: tf.cast(tf.stack(values), D),
        *(scalar(row, seed) for row, seed in zip(rows, seeds, strict=True)))
    for actual, authority in zip(result, expected, strict=True):
        tf.debugging.assert_equal(actual, authority)
        assert actual.dtype == D


def test_shard_signature_cache_is_bounded():
    worker = forecast.ComplexityForecastWorker(1)
    for count in range(1, 19):
        worker.make_shard_program(count)
    assert len(worker._shard_programs) == 16
    assert (1, True) not in worker._shard_programs
