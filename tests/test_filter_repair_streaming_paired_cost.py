"""Fresh-process reference timing; no scientific or GPU-default claims."""

import hashlib
import json
import os
import resource
import statistics
import time
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

import bayesfilter.highdim.ledh_canonical_filter_tf as current
from tests.test_filter_repair_ledh_seeded_cost import (
    _callbacks_with_frozen_constants,
    _rss,
    _sync,
)
from tests.test_filter_repair_ledh_streaming import (
    BUFFERED_COMMIT,
    _graph_evidence,
    _numeric,
    _records_compare,
    authorities,
    buffered_authority,
)

__all__ = ['authorities', 'buffered_authority']


def _sample():
    try:
        allocator = tf.config.experimental.get_memory_info('CPU:0')
    except ValueError:
        allocator = None
    return {'rss_bytes': _rss(), 'allocator': allocator,
        'process_high_water_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        'load_average': list(os.getloadavg())}


@pytest.mark.parametrize('pair,horizon,arm', [
    pytest.param(pair, horizon, arm, id=f'{pair}-{horizon}-{arm}')
    for pair in range(5) for horizon in (32, 128) for arm in ('buffered', 'streaming')])
def test_matched_streaming_cost(pair, horizon, arm, authorities, buffered_authority, request):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    _, fixture, _ = authorities
    model = fixture._lgssm_model(13, horizon=horizon)
    callbacks = _callbacks_with_frozen_constants(fixture._callbacks_for_lgssm(model))
    observations = tf.constant(model['observations'], tf.float64)
    spec = tf.TensorSpec(observations.shape, tf.float64)
    args = (observations, tf.constant([123], tf.uint32), tf.constant([17], tf.uint32))
    controls = {'flow_substeps': 3, 'sinkhorn_steps': 2, 'balance_steps': 2,
        'dual_cap_enabled': True, 'trust_region_enabled': True}
    directory = Path(request.config.getoption('xmlpath')).parent
    module = buffered_authority[0] if arm == 'buffered' else current
    before = _sample()
    start = time.perf_counter()
    owner = module.make_seeded_canonical_value_program(callbacks, spec, particle_count=64, **controls)
    result = owner(*args)
    _sync(result)
    cold_seconds = time.perf_counter() - start
    after_cold = _sample()
    for _ in range(3):
        _sync(owner(*args))
    times = []
    for _ in range(30):
        start = time.perf_counter()
        replay = owner(*args)
        _sync(replay)
        times.append(time.perf_counter() - start)
    after_warm = _sample()
    # Everything below happens after primary timing and memory samples.
    record = {'schema': 'filter_repair_streaming_paired_cost.v1',
        'arm': arm, 'pair': pair, 'shape': {'T': horizon, 'N': 64, 'd': 2},
        'seeds': {'fixture': 13, 'process': 123, 'resample': 17},
        'input_sha256': [hashlib.sha256(tf.io.serialize_tensor(x).numpy()).hexdigest() for x in args],
        'controls': controls, 'buffered_commit': BUFFERED_COMMIT,
        'buffered_source_sha256': {p: hashlib.sha256(s.encode()).hexdigest() for p, s in buffered_authority[1].items()},
        'cold_seconds': cold_seconds, 'warm_seconds': times, 'warm_median_seconds': statistics.median(times),
        'conditioning_calls': 3, 'timed_calls': 30,
        'before': before, 'after_cold': after_cold, 'after_warm': after_warm,
        'memory_sample_excludes_hlo_export_and_comparison_owner': True,
        'device': result['value'].device, 'affinity': sorted(os.sched_getaffinity(0)),
        'threads': {'intra': os.environ['TF_NUM_INTRAOP_THREADS'], 'inter': os.environ['TF_NUM_INTEROP_THREADS']},
        'fresh_process_per_pair_horizon_arm': True, 'record': _numeric(result),
        'nonclaims': ['CPU reference only; not GPU-default performance.',
            'No canonical LEDH, HMC, leak-freedom or universal capacity claim.']}
    path = directory / 'streaming-paired-cost.json'
    path.write_text(json.dumps(record, indent=2) + '\n')
    assert 'CPU:0' in record['device'] and bool(result['program_valid'])
    assert record['record'] == _numeric(replay)
    assert len(times) == 30 and all(np.isfinite(x) and x > 0 for x in times)
    record['graph'] = _graph_evidence(owner, args, directory, 'paired', horizon, 64)
    assert record['graph']['trace_count'] == 1
    assert bool(record['graph']['process_buffer_shape_nodes']) == (arm == 'buffered')
    assert (record['graph']['hlo']['optimized_hlo']['process_shape_occurrences'] > 0) == (arm == 'buffered')
    other = current if arm == 'buffered' else buffered_authority[0]
    comparison_owner = other.make_seeded_canonical_value_program(callbacks, spec, particle_count=64, **controls)
    comparison = comparison_owner(*args)
    record['comparison'] = (_records_compare(comparison, result) if arm == 'buffered'
        else _records_compare(result, comparison))
    buffered = record['record'] if arm == 'buffered' else _numeric(comparison)
    streamed = _numeric(comparison) if arm == 'buffered' else record['record']
    extra_fields = {'process_draws_consumed', 'final_philox_state', 'resampling_draws_consumed'}
    assert set(streamed) == set(buffered) | extra_fields
    record['shared_record'] = buffered
    record['streaming_rng_diagnostics'] = {key: streamed[key] for key in sorted(extra_fields)}
    assert streamed['process_draws_consumed'] == horizon
    assert streamed['resampling_draws_consumed'] == 0
    record['exact_complete_record_agreement'] = buffered == {key: streamed[key] for key in buffered}
    path.write_text(json.dumps(record, indent=2) + '\n')
    assert record['exact_complete_record_agreement']
