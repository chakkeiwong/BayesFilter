"""Fresh-process GPU streaming costs; independent baseline comparisons after measurement."""

import gc
import hashlib
import json
import os
import statistics
import time
import weakref
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

import bayesfilter.highdim.ledh_canonical_filter_tf as current
from scripts.filter_repair_cost_provenance import GPUProcessMonitor
from tests.test_filter_repair_ledh_seeded_cost import (
    _callbacks_with_frozen_constants,
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
from tests.test_filter_repair_resource_owners import memory as resource_memory

__all__ = ['authorities', 'buffered_authority']


def _sample():
    sample = resource_memory()
    return {**sample, 'rss_bytes': sample['VmRSS'], 'process_high_water_bytes': sample['VmHWM']}


@pytest.mark.parametrize('pair,horizon,arm', [
    pytest.param(pair, horizon, arm, id=f'{pair}-{horizon}-{arm}')
    for pair in range(3) for horizon in (32, 128) for arm in ('buffered', 'streaming')])
def test_matched_streaming_cost(pair, horizon, arm, authorities, buffered_authority, request):
    assert os.environ.get('CUDA_VISIBLE_DEVICES', '').startswith('GPU-')
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
    tf.constant(0.).numpy()
    tf.config.experimental.reset_memory_stats('GPU:0')
    with GPUProcessMonitor(True) as monitor:
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
        'device': result['value'].device, 'device_observation': monitor.payload(), 'affinity': sorted(os.sched_getaffinity(0)),
        'threads': {'intra': os.environ['TF_NUM_INTRAOP_THREADS'], 'inter': os.environ['TF_NUM_INTEROP_THREADS']},
        'fresh_process_per_pair_horizon_arm': True, 'record': _numeric(result),
        'nonclaims': ['Three paired processes support only the declared fixture and hardware.',
            'No canonical LEDH, HMC, leak-freedom or universal capacity claim.']}
    path = directory / 'streaming-paired-cost.json'
    path.write_text(json.dumps(record, indent=2) + '\n')
    assert 'GPU:0' in record['device'] and bool(result['program_valid'])
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


@pytest.mark.parametrize('horizon', (32, 128))
def test_streaming_reuse(horizon, authorities, buffered_authority, request):
    _, fixture, _ = authorities
    model = fixture._lgssm_model(13, horizon=horizon)
    callbacks = _callbacks_with_frozen_constants(fixture._callbacks_for_lgssm(model))
    observations = tf.constant(model['observations'], tf.float64)
    spec = tf.TensorSpec(observations.shape, tf.float64)
    variants = [(observations+shift, tf.constant([seed], tf.uint32), tf.constant([seed-106], tf.uint32))
                for seed, shift in ((123, 0.), (124, .1))]
    controls = {'flow_substeps': 3, 'sinkhorn_steps': 2, 'balance_steps': 2,
                'dual_cap_enabled': True, 'trust_region_enabled': True}
    gpu = os.environ['CUDA_VISIBLE_DEVICES'] != '-1'
    if gpu:
        tf.config.experimental.reset_memory_stats('GPU:0')
    with GPUProcessMonitor(gpu) as monitor:
        before = _sample()
        owner = current.make_seeded_canonical_value_program(callbacks, spec, particle_count=64, **controls)
        expected = [_numeric(owner(*args)) for args in variants]
        assert expected[0] != expected[1]
        assert expected[0]['program_valid']
        # The changed T128 case is rejected identically by the buffered
        # reference (05242). Use valid work for lifetime measurement and
        # retain that changed-input refusal for the later reference check.
        valid_variants = [i for i, row in enumerate(expected) if row['program_valid']]
        snapshots = {0: _sample()}
        exact = True
        for index in range(1, 129):
            variant = valid_variants[index % len(valid_variants)]
            exact &= _numeric(owner(*variants[variant])) == expected[variant]
            if index in (16, 32, 64, 96, 128):
                snapshots[index] = _sample()
        traces = owner.experimental_get_tracing_count()
        # Additional configurations reuse the callback and valid selected seeds.
        # Compile no more than four particle-count specializations and stop
        # before the2GiB incremental bound. These are capacity, not timing arms.
        capacity_before = _sample()
        capacities = []
        candidate = None
        for count in (8, 16, 32, 64):
            candidate = current.make_seeded_canonical_value_program(callbacks, spec,
                particle_count=count, **controls)
            result = _numeric(candidate(*variants[0]))
            snapshot = _sample()
            capacities.append({'particles': count, 'program_valid': result['program_valid'],
                               'trace_count': candidate.experimental_get_tracing_count(), 'memory': snapshot})
            assert snapshot['VmRSS']-capacity_before['VmRSS'] <= 2*1024**3
            if gpu:
                assert snapshot['allocator']['peak'] <= 2*1024**3
        reference = weakref.ref(owner)
        del owner, candidate
        gc.collect()
        after_release = _sample()
    original = buffered_authority[0].make_seeded_canonical_value_program(callbacks, spec,
        particle_count=64, **controls)
    differences = []
    for args, actual in zip(variants, expected, strict=True):
        prior = _numeric(original(*args))
        for key, value in prior.items():
            np.testing.assert_array_equal(actual[key], value, err_msg=key)
        differences.append(0.)
    record = {'schema': 'filter_repair.streaming_reuse.v1', 'horizon': horizon, 'particles': 64,
        'calls': 128, 'device': 'GPU' if gpu else 'CPU', 'same_input_exact_replay': exact,
        'changed_inputs': expected, 'independent_reference_errors': differences,
        'measured_valid_variants': valid_variants,
        'trace_count': traces, 'owner_collected': reference() is None,
        'memory': {'before': before, 'snapshots': snapshots, 'capacity_before': capacity_before,
                   'after_release': after_release}, 'capacities': capacities,
        'late_64_call_rss_growth_bytes': snapshots[128]['VmRSS']-snapshots[64]['VmRSS'],
        'device_observation': monitor.payload(),
        'nonclaims': ['Finite call/configuration range; no universal capacity or native eviction.',
                     'Invalid capacity inputs cannot support a scientific capacity claim.']}
    directory = Path(request.config.getoption('xmlpath')).parent
    (directory/'streaming-reuse.json').write_text(json.dumps(record, indent=2)+'\n')
    assert exact and traces == 1 and reference() is None
    assert record['late_64_call_rss_growth_bytes'] <= 16*1024**2
    if gpu:
        assert snapshots[64]['allocator']['current'] == snapshots[128]['allocator']['current']
    assert not monitor.errors
    assert all(process['pid'] == os.getpid() for sample in monitor.samples for process in sample['processes'])


def test_changed_long_horizon_status(authorities, buffered_authority, request):
    _, fixture, _ = authorities
    model = fixture._lgssm_model(13, horizon=128)
    callbacks = _callbacks_with_frozen_constants(fixture._callbacks_for_lgssm(model))
    observations = tf.constant(model['observations'], tf.float64)+.1
    spec = tf.TensorSpec(observations.shape, tf.float64)
    controls = {'flow_substeps': 3, 'sinkhorn_steps': 2, 'balance_steps': 2,
                'dual_cap_enabled': True, 'trust_region_enabled': True}
    args = (observations, tf.constant([124], tf.uint32), tf.constant([18], tf.uint32))
    results = {}
    for label, module in (('buffered', buffered_authority[0]), ('streaming', current)):
        owner = module.make_seeded_canonical_value_program(callbacks, spec, particle_count=64, **controls)
        results[label] = owner(*args)
    report = {'schema': 'filter_repair.streaming_changed_status.v1',
        'results': {key: _numeric(value) for key, value in results.items()},
        'comparison': _records_compare(results['streaming'], results['buffered'])}
    directory = Path(request.config.getoption('xmlpath')).parent
    (directory/'streaming-changed-status.json').write_text(json.dumps(report, indent=2)+'\n')
    for key, prior in results['buffered'].items():
        np.testing.assert_array_equal(results['streaming'][key].numpy(), prior.numpy(), err_msg=key)
    assert not bool(results['buffered']['program_valid'])
