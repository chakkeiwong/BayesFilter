"""Fresh-process cost screen of the actual fixed-fitted adapter callable."""

import gc
import hashlib
import os
import platform
import statistics
import time

import pytest
import tensorflow as tf

from tests.test_filter_repair_fitted_apf_fixed import (
    FIXTURE,
    FIXTURES,
    _host,
    _reference_module,
    _save,
    _seed,
)
from tests.test_filter_repair_nonlinear_direction_cost import _memory


@pytest.mark.parametrize('model,arm', [(model, arm)
    for model in ('gaussian', 'nonlinear_scalar') for arm in ('original', 'enclosing')])
def test_fitted_apf_adapter_cost(model, arm, request, monkeypatch):
    from bayesfilter.score_study import fitted_twist_adapter, fitted_twist_tf
    original_tf = _reference_module('fitted_twist_tf')
    original_adapter = _reference_module('fitted_twist_adapter')
    row = {'model': model, 'fit_theta': FIXTURE['theta'],
           **{key: FIXTURE[key] for key in ('fit_iterations', 'fit_initial_variance', 'fit_floor_ratio')}}
    settings = {key: FIXTURE[key] for key in ('dimension', 'observation_dimension', 'particles', 'horizon')}
    settings.update(dtype='float64', jit_compile=True)
    if model == 'nonlinear_scalar':
        settings.update(FIXTURE['nonlinear_curves'])
    theta = tf.constant(FIXTURE['theta'], tf.float64)
    data = tf.constant(FIXTURE['observations'], tf.float64)
    seed = _seed(model)
    memory = {'before': _memory()}
    began = time.perf_counter()
    if arm == 'original':
        monkeypatch.setattr(fitted_twist_tf, 'make_fitted_twist_kernel', original_tf.make_fitted_twist_kernel)
        monkeypatch.setattr(fitted_twist_tf, 'make_recursive_fit_kernel', original_tf.make_recursive_fit_kernel)
        execute = original_adapter.execute_fitted_twist
    else:
        execute = fitted_twist_adapter.execute_fitted_twist
    setup = time.perf_counter() - began
    memory['after_setup'] = _memory()

    def invoke():
        owner, final, diagnostics, calls = execute(row, settings, theta, data, seed)
        return owner, _host(final), diagnostics, calls

    began = time.perf_counter()
    owner, final, diagnostics, calls = invoke()
    cold = time.perf_counter() - began
    memory['after_cold'] = _memory()
    for _ in range(3):
        other, output, fields, count = invoke()
        assert other is owner and output == final and fields == diagnostics and count == calls
    times = []
    for _ in range(30):
        began = time.perf_counter()
        other, output, fields, count = invoke()
        times.append(time.perf_counter() - began)
        assert other is owner and output == final and fields == diagnostics and count == calls
    memory['after_warm'] = _memory()
    gc.collect()
    memory['after_collection'] = _memory()
    assert owner.experimental_get_tracing_count() == 1
    assert all(bool(tf.reduce_all(tf.math.is_finite(tf.convert_to_tensor(x)))) for x in final)
    _save(request, 'fitted-apf-cost', {
        'schema': 'filter_repair_fitted_apf_cost.v1', 'model': model, 'arm': arm,
        'fixture_sha256': hashlib.sha256((FIXTURES/'fixture.json').read_bytes()).hexdigest(),
        'reference_sources': FIXTURE['baseline_sha256'],
        'setup_seconds': setup, 'cold_call_seconds': cold, 'cold_total_seconds': setup+cold,
        'warm_seconds': times, 'warm_median_seconds': statistics.median(times),
        'memory': memory, 'cpu_affinity': sorted(os.sched_getaffinity(0)),
        'process_id': os.getpid(), 'platform': platform.platform(), 'kernel_release': platform.release(),
        'extra_environment': {key: os.environ.get(key) for key in
            ('XLA_FLAGS', 'TF_XLA_FLAGS', 'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS', 'OPENBLAS_NUM_THREADS')},
        'factory_reused_same_owner': True, 'trace_count': 1, 'subkernel_calls': calls,
        'final': final, 'diagnostics': diagnostics,
        'scope': 'Actual fixed-fit adapter, CPU reference accounting; no ranking, capacity or eviction claim.'})
