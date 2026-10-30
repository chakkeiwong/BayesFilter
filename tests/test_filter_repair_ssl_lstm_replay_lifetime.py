"""Bounded owner reuse/capacity diagnostic; no sampler or training runs."""

import dataclasses
import gc
import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf

from scripts.filter_repair_cost_provenance import GPUProcessMonitor
from tests.test_filter_repair_gaussian_binding_cost import memory
from tests.test_filter_repair_ssl_lstm_replay import (
    candidate,
    case,
    compare,
    evaluate,
    reference,
    save,
)
from tests.test_filter_repair_ssl_lstm_replay_cost import synchronized


def test_reuse_and_bounded_specialization(request):
    tf.config.experimental.enable_tensor_float_32_execution(False)
    assert candidate._fixed_replay_program.cache_info().currsize == 0
    original, digest = reference()
    config, theta, observations, manifest = case(8)
    operands = ((theta, observations), (theta + .015, observations - .013))
    gpu = os.environ['CUDA_VISIBLE_DEVICES'] != '-1'
    if gpu:
        tf.config.experimental.reset_memory_stats('GPU:0')
    monitor = GPUProcessMonitor(gpu)
    with monitor:
        before = memory()
        expected = [synchronized(evaluate(candidate, config, q, y, manifest))
                    for q, y in operands]
        after_cold = memory()
        owner = candidate._fixed_replay_program(config, manifest, 8, 1e-4, True, False)
        snapshots = {0: memory()}
        exact_replay = True
        for index in range(1, 2001):
            selected = index % 2
            q, y = operands[selected]
            actual = synchronized(evaluate(candidate, config, q, y, manifest))
            exact_replay &= all(np.array_equal(a, b) for a, b in zip(
                tf.nest.flatten(actual), tf.nest.flatten(expected[selected]), strict=True))
            if index in (10, 100, 500, 1000, 1500, 2000):
                snapshots[index] = memory()
        trace_count = owner.experimental_get_tracing_count()
        # Independent baseline runs after the primary residency/reuse samples.
        for (q, y), actual in zip(operands, expected, strict=True):
            compare(actual, evaluate(original, config, q, y, manifest))
        del owner, actual, expected
        small_config, small_theta, small_observations, small_manifest = case(2)
        capacity_before = memory()
        capacities = []
        expected_retained = None
        retained_manifest = None
        for index in range(20):
            current = dataclasses.replace(small_manifest,
                initial_seed=(20260705, 101 + index), process_seed=(20260705, 201 + index))
            result = synchronized(evaluate(candidate, small_config, small_theta,
                small_observations, current))
            if index == 19:
                expected_retained, retained_manifest = result, current
            capacities.append({'index': index, 'cache': candidate._fixed_replay_program.cache_info()._asdict(),
                               'memory': memory()})
        retained_owner = candidate._fixed_replay_program(small_config, retained_manifest, 2, 1e-4, True, False)
        retained_before = retained_owner.experimental_get_tracing_count()
        repeated = synchronized(evaluate(candidate, small_config, small_theta,
            small_observations, retained_manifest))
        retained_exact = all(np.array_equal(a, b) for a, b in zip(
            tf.nest.flatten(repeated), tf.nest.flatten(expected_retained), strict=True))
        retained_after = retained_owner.experimental_get_tracing_count()
        del retained_owner, repeated, result, expected_retained
        before_clear = memory()
        candidate._fixed_replay_program.cache_clear()
        gc.collect()
        after_clear = memory()
    late_growth = snapshots[2000]['VmRSS'] - snapshots[1000]['VmRSS']
    capacity_growth = max(row['memory']['VmRSS'] for row in capacities) - capacity_before['VmRSS']
    directory = Path(request.config.getoption('xmlpath')).parent
    provenance = next(json.loads(line) for line in (directory/'process.log').read_text().splitlines()
                      if line.startswith('{"tensorflow_version"'))
    record = {'schema': 'filter_repair.ssl_lstm_replay_lifetime.v1',
        'baseline_sha256': digest, 'calls': 2000, 'exact_replay': exact_replay,
        'trace_count': trace_count, 'memory': {'before': before, 'after_cold': after_cold,
            'snapshots': snapshots, 'capacity_before': capacity_before,
            'before_cache_clear': before_clear, 'after_cache_clear': after_clear},
        'late_1000_call_rss_growth_bytes': late_growth,
        'capacity_growth_bytes': capacity_growth, 'capacity': capacities,
        'retained_owner': {'exact_replay': retained_exact, 'traces_before': retained_before,
            'traces_after': retained_after}, 'cache_after_clear': candidate._fixed_replay_program.cache_info()._asdict(),
        'device_observation': monitor.payload(), 'worker_provenance': provenance,
        'nonclaims': ['Twenty specializations and one fixed workload do not prove universal capacity.',
            'Cache clear and Python collection do not guarantee native-memory eviction.',
            'No sampler, training, posterior, canonical-algorithm or speed-ranking claim.']}
    save(request, 'ssl-lstm-replay-lifetime.json', record)
    assert exact_replay and retained_exact
    assert trace_count == retained_before == retained_after == 1
    assert late_growth <= 16 * 1024**2
    assert capacity_growth <= 2 * 1024**3
    assert all(row['cache']['currsize'] <= 16 for row in capacities)
    assert capacities[-1]['cache']['currsize'] == 16
    assert candidate._fixed_replay_program.cache_info().currsize == 0
    if gpu:
        assert snapshots[1000]['allocator'] == snapshots[2000]['allocator']
        assert max(row['memory']['allocator']['peak'] for row in capacities) <= 256 * 1024**2
