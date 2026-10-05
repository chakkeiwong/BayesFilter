"""Descriptive fresh-process owner costs; the Python arm is a reference only."""

import gc
import hashlib
import os
import resource
import statistics
import time
from pathlib import Path

import pytest
import tensorflow as tf

from tests.test_filter_repair_score_directions import (
    DTYPE,
    FIXTURE,
    FIXTURE_PATH,
    ROOT,
    _factory,
    _operands,
    _save,
)

DEPENDENCIES = (
    'bayesfilter/score_study/direction_assembly_tf.py',
    'bayesfilter/score_study/canonical_adapter_tf.py',
    'bayesfilter/score_study/control_diagnostics_tf.py',
    'bayesfilter/score_study/kdm_adapter_tf.py',
    'bayesfilter/score_study/gaussian_tf.py',
    'bayesfilter/highdim/ledh_canonical_score_tf.py',
    'bayesfilter/highdim/ledh_unified_reset_tf.py',
    'bayesfilter/highdim/ledh_unified_correction_tf.py',
    'bayesfilter/highdim/ledh_younis_kdm_resampling_tf.py',
)


def _sync(values):
    return tf.nest.map_structure(lambda tensor: tensor.numpy(), values)


def _rss():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):
            return int(line.split()[1])*1024
    raise RuntimeError('RSS unavailable')


@pytest.mark.parametrize('case,arm', [
    ('ledh_diagnostics', 'python_reference'), ('ledh_diagnostics', 'enclosing'),
    ('resampling_kdm', 'enclosing'), ('resampling_kdm', 'python_reference')])
def test_fresh_direction_cost(case, arm, request):
    theta, directions = tf.constant(FIXTURE['theta'], DTYPE), tf.eye(6, dtype=DTYPE)
    operands = _operands(case)
    before = _rss()
    started = time.perf_counter()
    owner = _factory(case, all_directions=arm == 'enclosing')
    setup_seconds = time.perf_counter()-started
    after_setup = _rss()

    def invoke():
        if arm == 'python_reference':
            rows = tuple(owner(theta, direction, *operands) for direction in tf.unstack(directions))
            # The prior endpoint stacks values/scores only. Do not add timed
            # stacking of every auxiliary field to penalize the baseline.
            return (tf.stack([row[0] for row in rows]), tf.stack([row[1] for row in rows]),
                    tuple(row[2:] for row in rows))
        result = owner(theta, directions, *operands)
        return result['outputs']

    started = time.perf_counter()
    result = _sync(invoke())
    cold_call_seconds = time.perf_counter()-started
    after_cold = _rss()
    for _ in range(3):
        _sync(invoke())
    warm_seconds = []
    for _ in range(30):
        started = time.perf_counter()
        _sync(invoke())
        warm_seconds.append(time.perf_counter()-started)
    after_warm = _rss()
    assert _factory(case, all_directions=arm == 'enclosing') is owner
    assert owner.experimental_get_tracing_count() == 1
    gc.collect()
    after_collection = _rss()
    primary_peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    if arm == 'enclosing':
        flags = owner(theta, directions, *operands)
        assert bool(flags['valid']) and bool(flags['value_invariant'])
    else:
        auxiliary = _sync(tf.nest.map_structure(lambda *values: tf.stack(values), *result[2]))
        result = (result[0], result[1], *auxiliary)
    for value in tf.nest.flatten(result):
        if tf.as_dtype(value.dtype).is_floating:
            assert bool(tf.reduce_all(tf.math.is_finite(value)))
    assert bool(tf.reduce_all(tf.abs(result[0]-result[0][0]) < 1e-9))
    if case == 'resampling_kdm':
        assert bool(tf.reduce_all(result[2]))
    report = {'case': case, 'arm': arm, 'scope': 'CPU reference, one fresh process per arm, descriptive only',
        'fixture_sha256': hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest(),
        'cpu_affinity': sorted(os.sched_getaffinity(0)),
        'extra_environment': {key: os.environ.get(key) for key in
            ('XLA_FLAGS', 'TF_XLA_FLAGS', 'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS', 'OPENBLAS_NUM_THREADS')},
        'source_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in DEPENDENCIES},
        'setup_seconds': setup_seconds, 'cold_call_seconds': cold_call_seconds,
        'cold_total_seconds': setup_seconds+cold_call_seconds,
        'warm_seconds': warm_seconds, 'warm_median_seconds': statistics.median(warm_seconds),
        'rss': {'before': before, 'after_setup': after_setup, 'after_cold': after_cold,
                'after_warm': after_warm, 'after_python_collection': after_collection},
        'primary_peak_rss_bytes': primary_peak_rss,
        'allocator': {'current_bytes': None, 'peak_bytes': None, 'reason': 'CPU reference; GPU intentionally hidden'},
        'factory_reused_same_owner': True, 'trace_count': 1,
        'flattened_output': [{'dtype': value.dtype.name, 'shape': list(value.shape), 'value': value.tolist()}
                             for value in tf.nest.flatten(result)],
        'nonclaims': ['No compiler eviction, GPU cost/capacity or statistically supported performance ranking.']}
    _save(request, 'score-direction-cost', report)
