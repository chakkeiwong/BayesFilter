"""Fresh-process complete-owner costs for the fixed replay execution repair."""

import gc
import itertools
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from scripts.filter_repair_cost_provenance import GPUProcessMonitor
from tests.test_filter_repair_gaussian_binding_cost import memory
from tests.test_filter_repair_ssl_lstm_replay import (
    candidate,
    case,
    evaluate,
    host,
    reference,
    save,
)


def synchronized(record):
    return tf.nest.map_structure(lambda value: value.numpy(), record)


@pytest.mark.parametrize('mode,arm,horizon', itertools.product(
    ('default', 'graph', 'xla'), ('before', 'after'), (2, 8)))
def test_complete_owner_cost(mode, arm, horizon, request):
    tf.config.experimental.enable_tensor_float_32_execution(False)
    original, digest = reference()
    module = original if arm == 'before' else candidate
    config, theta, observations, manifest = case(horizon)
    assert candidate._fixed_replay_program.cache_info().currsize == 0
    options = {'jit_compile': mode != 'graph'} if arm == 'after' else {}

    def call(q, y):
        return evaluate(module, config, q, y, manifest, **options)

    owner = None
    if mode != 'default':
        owner = tf.function(call, input_signature=[tf.TensorSpec(theta.shape, theta.dtype),
            tf.TensorSpec(observations.shape, observations.dtype)],
            jit_compile=mode == 'xla', autograph=False)
    function = call if owner is None else owner
    gpu = os.environ['CUDA_VISIBLE_DEVICES'] != '-1'
    if gpu:
        tf.config.experimental.reset_memory_stats('GPU:0')
    monitor = GPUProcessMonitor(gpu)
    with monitor:
        before = memory()
        start = time.perf_counter()
        result = synchronized(function(theta, observations))
        cold = time.perf_counter()-start
        after_cold = memory()
        for _ in range(3):
            replay = synchronized(function(theta, observations))
            for a, b in zip(tf.nest.flatten(result), tf.nest.flatten(replay), strict=True):
                np.testing.assert_array_equal(a, b)
        warm = []
        for _ in range(30):
            start = time.perf_counter()
            replay = synchronized(function(theta, observations))
            warm.append(time.perf_counter()-start)
            for a, b in zip(tf.nest.flatten(result), tf.nest.flatten(replay), strict=True):
                np.testing.assert_array_equal(a, b)
        after_warm = memory()
        gc.collect()
        after_collection = memory()
    traces = owner.experimental_get_tracing_count() if owner is not None else None
    if arm == 'after':
        compiled = candidate._fixed_replay_program(config, manifest, horizon, 1e-4,
            mode != 'graph', mode == 'xla')
        assert compiled.experimental_get_tracing_count() == 1
        assert candidate._fixed_replay_program.cache_info().currsize == 1
    measured_graph = compiled if arm == 'after' and mode == 'default' else owner
    graph_nodes = None
    if measured_graph is not None:
        graph = measured_graph.get_concrete_function().graph.as_graph_def()
        graph_nodes = len(graph.node) + sum(len(function.node_def) for function in graph.library.function)
    directory = Path(request.config.getoption('xmlpath')).parent
    provenance = None
    for line in (directory/'process.log').read_text().splitlines():
        if line.startswith('{"tensorflow_version"'):
            provenance = json.loads(line)
            break
    assert provenance is not None
    save(request, 'ssl-lstm-replay-cost.json', {
        'arm': arm, 'mode': mode, 'horizon': horizon, 'baseline_sha256': digest,
        'actual_jit_compile': mode == 'xla' or (mode == 'default' and arm == 'after'),
        'rng_stream': 'original_xla_stateless_normal' if mode == 'xla' else 'original_tf_philox_normal',
        'inputs': {'theta': theta, 'observations': observations, 'manifest': manifest.as_dict()},
        'numerical_result': host(tf.nest.map_structure(tf.convert_to_tensor, result)),
        'cold_seconds': cold, 'warm_seconds': warm, 'warm_median_seconds': statistics.median(warm),
        'memory': {'before': before, 'after_cold': after_cold, 'after_warm': after_warm,
                   'after_python_collection': after_collection},
        'enclosing_trace_count': traces, 'exact_replay': True,
        'graph_node_count': graph_nodes,
        'worker_provenance': provenance, 'device_observation': monitor.payload(),
        'scope': 'Complete numerical result and parameter components synchronized to host; setup/import excluded; cold includes first tracing/compilation.',
        'nonclaims': ['One worker is not a statistical acceptance decision.',
                      'Sampled sharing cannot prove exclusivity; Python collection cannot prove native eviction.'],
    })
    assert traces in (None, 1)
