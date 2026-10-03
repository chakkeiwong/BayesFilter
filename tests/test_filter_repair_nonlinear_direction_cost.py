"""Fresh-process nonlinear owner costs and independent OS memory counters."""

import gc
import hashlib
import json
import os
import platform
import resource
import statistics
import time
from pathlib import Path

import pytest
import tensorflow as tf

from tests.test_filter_repair_nonlinear_directions import FIXTURE, _factory, _operands
from tests.test_filter_repair_nonlinear_scope import (
    _nomination,
    _partition,
    _scope_fixture,
)
from tests.test_filter_repair_score_directions import DTYPE, _save


def _memory():
    began = time.perf_counter()
    status = {}
    for line in Path('/proc/self/status').read_text().splitlines():
        key, _, value = line.partition(':')
        if key in ('VmRSS', 'VmHWM', 'RssAnon', 'RssFile', 'RssShmem'):
            status[key] = int(value.split()[0])*1024
    smaps = {}
    for line in Path('/proc/self/smaps_rollup').read_text().splitlines():
        key, _, value = line.partition(':')
        if key in ('Rss', 'Pss', 'Anonymous', 'Shared_Clean', 'Private_Clean', 'Private_Dirty'):
            smaps[key] = int(value.split()[0])*1024
    return {'status': status, 'smaps_rollup': smaps,
        'rusage_maxrss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'read_order': 'status, smaps_rollup, rusage; not atomic',
        'read_wall_seconds': time.perf_counter()-began}


def _sync(values):
    return tf.nest.map_structure(lambda tensor: tensor.numpy(), values)


@pytest.mark.parametrize('case,arm', [(c, a) for c in ('ukf', 'ledh_diagnostics')
                                   for a in ('python_reference', 'enclosing')])
def test_nonlinear_owner_cost(case, arm, request):
    if case == 'ukf':
        fixture, nomination = FIXTURE, None
    else:
        nomination, count = _nomination('ledh')
        fixture = _scope_fixture(_partition('untouched')[0]['providers']['ledh'][0], count)
    theta, directions = tf.constant(fixture['theta'], DTYPE), tf.eye(6, dtype=DTYPE)
    operands = _operands(case, fixture=fixture)
    memory = {'before': _memory()}
    started = time.perf_counter()
    owner = _factory(case, all_directions=arm == 'enclosing', fixture=fixture)
    setup = time.perf_counter()-started
    memory['after_setup'] = _memory()

    def invoke():
        if arm == 'enclosing':
            return owner(theta, directions, *operands)['outputs']
        rows = tuple(owner(theta, direction, *operands) for direction in tf.unstack(directions))
        return tf.stack([x[0] for x in rows]), tf.stack([x[1] for x in rows]), tuple(x[2:] for x in rows)

    started = time.perf_counter()
    result = _sync(invoke())
    cold_call = time.perf_counter()-started
    memory['after_cold'] = _memory()
    for _ in range(3):
        _sync(invoke())
    warm = []
    for _ in range(30):
        started = time.perf_counter()
        _sync(invoke())
        warm.append(time.perf_counter()-started)
    memory['after_warm'] = _memory()
    gc.collect()
    memory['after_collection'] = _memory()
    assert _factory(case, all_directions=arm == 'enclosing', fixture=fixture) is owner
    assert owner.experimental_get_tracing_count() == 1
    if arm == 'python_reference':
        result = (result[0], result[1], *_sync(tf.nest.map_structure(lambda *x: tf.stack(x), *result[2])))
    else:
        flags = owner(theta, directions, *operands)
        assert bool(flags['valid']) and bool(flags['value_invariant'])
    assert all(bool(tf.reduce_all(tf.math.is_finite(x))) for x in tf.nest.flatten(result)
               if tf.as_dtype(x.dtype).is_floating)
    _save(request, 'nonlinear-direction-cost', {
        'case': case, 'arm': arm, 'nomination_run': nomination,
        'fixture_sha256': hashlib.sha256(json.dumps(fixture, sort_keys=True).encode()).hexdigest(),
        'setup_seconds': setup, 'cold_call_seconds': cold_call, 'cold_total_seconds': setup+cold_call,
        'warm_seconds': warm, 'warm_median_seconds': statistics.median(warm),
        'memory': memory, 'cpu_affinity': sorted(os.sched_getaffinity(0)),
        'process_id': os.getpid(), 'platform': platform.platform(), 'kernel_release': platform.release(),
        'extra_environment': {key: os.environ.get(key) for key in
            ('XLA_FLAGS', 'TF_XLA_FLAGS', 'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS', 'OPENBLAS_NUM_THREADS')},
        'factory_reused_same_owner': True, 'trace_count': 1,
        'flattened_output': [{'dtype': value.dtype.name, 'shape': list(value.shape), 'value': value.tolist()}
                             for value in tf.nest.flatten(result)],
        'scope': 'single-process CPU-reference cost/accounting screen; no ranking, peak guarantee or compiler eviction'})
