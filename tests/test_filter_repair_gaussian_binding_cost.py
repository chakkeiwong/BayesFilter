"""Descriptive matched fresh-process Gaussian endpoint cost diagnostics."""

import gc
import os
import statistics
import time
from pathlib import Path

import pytest
import tensorflow as tf

from tests.test_filter_repair_gaussian_binding import (
    BASELINE,
    endpoint_case,
    hashes,
    numerical_record,
    original_adapter,
    runtime_setup,
    save,
)


def memory():
    values = {}
    for line in Path('/proc/self/status').read_text().splitlines():
        key, _, rest = line.partition(':')
        if key in ('VmRSS', 'VmHWM'):
            values[key] = int(rest.split()[0])*1024
    for line in Path('/proc/self/smaps_rollup').read_text().splitlines():
        if line.startswith('Rss:'):
            values['smaps_rollup_Rss'] = int(line.split()[1])*1024
    if os.environ['CUDA_VISIBLE_DEVICES'] != '-1':
        values['allocator'] = tf.config.experimental.get_memory_info('GPU:0')
    return values


@pytest.mark.parametrize('arm', ['before', 'after'])
def test_fresh_gaussian_binding_cost(arm, request, monkeypatch):
    from bayesfilter.score_study import adapters
    reference, reference_hash = original_adapter()
    module = reference if arm == 'before' else adapters
    monkeypatch.setattr(module, 'configure_runtime', runtime_setup)
    row, context = endpoint_case('ukf')
    device = context['study']['settings']['device']
    if device == 'GPU':
        tf.config.experimental.reset_memory_stats('GPU:0')
    before = memory()
    started = time.perf_counter()
    result = module.evaluate_gaussian(row, context)
    cold_seconds = time.perf_counter()-started
    after_cold = memory()
    for _ in range(2):
        assert numerical_record(module.evaluate_gaussian(row, context)) == numerical_record(result)
    warm = []
    for _ in range(30):
        started = time.perf_counter()
        replay = module.evaluate_gaussian(row, context)
        warm.append(time.perf_counter()-started)
        assert numerical_record(replay) == numerical_record(result)
    after_warm = memory()
    gc.collect()
    after_collection = memory()
    save(request, 'gaussian-binding-cost.json', {
        'schema': 'filter_gaussian_binding_cost.v1', 'arm': arm, 'device': device,
        'baseline': BASELINE, 'baseline_adapter_sha256': reference_hash,
        'source_sha256': hashes(), 'cpu_reference_exception': device == 'CPU',
        'scope': 'complete ordinary UKF endpoint including data/preparation/oracle/report formatting',
        'row': row, 'settings': context['study']['settings'], 'seed': context['study']['seed'],
        'environment': {key: os.environ.get(key) for key in ('XLA_FLAGS', 'TF_XLA_FLAGS',
            'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS', 'OPENBLAS_NUM_THREADS')},
        'cpu_affinity': sorted(os.sched_getaffinity(0)),
        'cold_seconds': cold_seconds, 'warm_seconds': warm,
        'warm_median_seconds': statistics.median(warm),
        'memory': {'before': before, 'after_cold': after_cold, 'after_warm': after_warm,
                   'after_python_collection': after_collection},
        'numerical_result': numerical_record(result), 'exact_replay': True,
        'runtime': result['runtime'],
        'nonclaims': ['One fresh process per arm; no statistical ranking or terminal cost acceptance.',
                      'Cached executables retained; Python collection does not establish native eviction.']})
