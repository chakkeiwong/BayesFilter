"""Fresh-process cost and bounded reuse diagnostics for remaining score owners."""

import copy
import csv
import json
import os
import subprocess
from pathlib import Path

import pytest
import tensorflow as tf

from scripts.filter_repair_cost_provenance import GPUProcessMonitor
from tests.test_filter_repair_gaussian_binding import compare, numerical_record
from tests.test_filter_repair_gaussian_binding_cost import memory as basic_memory

CASES = ('fitted_gaussian', 'fitted_nonlinear_scalar', 'gaussian_inputs')


def memory():
    result = basic_memory()
    result['process_gpu_reservation_bytes'] = None
    if os.environ['CUDA_VISIBLE_DEVICES'] != '-1':
        output = subprocess.check_output(['nvidia-smi',
            '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader,nounits'], text=True)
        entries = [int(used) * 2**20 for uuid, pid, used in
            csv.reader(output.splitlines(), skipinitialspace=True)
            if uuid == os.environ['CUDA_VISIBLE_DEVICES'] and int(pid) == os.getpid()]
        assert len(entries) == 1
        result['process_gpu_reservation_bytes'] = entries[0]
    return result


def save(request, filename, record):
    path = Path(request.config.getoption('xmlpath')).parent/filename
    path.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')


@pytest.mark.parametrize('case', CASES)
@pytest.mark.parametrize('pair', range(3))
@pytest.mark.parametrize('arm', ('before', 'after'))
def test_complete_owner_cost(case, pair, arm, request, monkeypatch):
    from tests import test_filter_repair_fitted_apf_cost as fitted
    from tests import test_filter_repair_score_inputs_cost as inputs

    placement = tf.constant(0.).device
    assert 'GPU:0' in placement
    tf.config.experimental.reset_memory_stats('GPU:0')
    with GPUProcessMonitor(True) as monitor:
        if case.startswith('fitted_'):
            monkeypatch.setattr(fitted, '_memory', memory)
            fitted.test_fitted_apf_adapter_cost(case.removeprefix('fitted_'),
                'original' if arm == 'before' else 'enclosing', request, monkeypatch)
            filename = 'fitted-apf-cost.json'
        else:
            monkeypatch.setattr(inputs, 'memory', memory)
            inputs.test_endpoint_cost(arm, request, monkeypatch)
            filename = 'score-inputs-cost.json'
    directory = Path(request.config.getoption('xmlpath')).parent
    record = json.loads((directory/filename).read_text())
    record.update(schema='filter_repair.resource_owner_cost.v1', case=case, pair=pair,
        arm=arm, device='GPU', placement=placement, device_observation=monitor.payload(),
        scope='Complete callable, unchanged frozen fixture; fresh process per pair and arm.',
        nonclaims=['Pair-specific timing is not a population estimate.',
                   'No capacity, native eviction, posterior or HMC conclusion.'])
    save(request, 'resource-owner-cost.json', record)
    assert not monitor.errors
    assert all(process['pid'] == os.getpid() for sample in monitor.samples for process in sample['processes'])


def fitted_invocation(case):
    from bayesfilter.score_study.fitted_twist_adapter import execute_fitted_twist
    from bayesfilter.score_study.fitted_twist_execution_tf import (
        make_fitted_twist_execution,
    )
    from tests.test_filter_repair_fitted_apf_fixed import FIXTURE, _host, _seed

    model = case.removeprefix('fitted_')
    row = {'model': model, 'fit_theta': FIXTURE['theta'],
           **{key: FIXTURE[key] for key in ('fit_iterations', 'fit_initial_variance', 'fit_floor_ratio')}}
    settings = {key: FIXTURE[key] for key in ('dimension', 'observation_dimension', 'particles', 'horizon')}
    settings.update(dtype='float64', jit_compile=True)
    if model == 'nonlinear_scalar':
        settings.update(FIXTURE['nonlinear_curves'])
    theta = tf.constant(FIXTURE['theta'], tf.float64)
    data = tf.constant(FIXTURE['observations'], tf.float64)
    owners = {}

    def invoke(variant):
        current_row = dict(row, fit_theta=[x-variant*.001 for x in row['fit_theta']])
        owner, final, diagnostics, calls = execute_fitted_twist(current_row, settings,
            theta+variant*.002, data+variant*.003, _seed(model, variant))
        owners[id(owner)] = owner
        assert all(('GPU:0' if os.environ['CUDA_VISIBLE_DEVICES'] != '-1' else 'CPU:0')
                   in value.device for value in tf.nest.flatten(final))
        return {'final': _host(final), 'diagnostics': diagnostics, 'subkernel_calls': calls}

    def reference(variant, actual, monkeypatch):
        from bayesfilter.score_study import fitted_twist_tf
        from tests.test_filter_repair_fitted_apf_fixed import _reference_module

        original_tf = _reference_module('fitted_twist_tf')
        original_adapter = _reference_module('fitted_twist_adapter')
        with monkeypatch.context() as patch:
            patch.setattr(fitted_twist_tf, 'make_fitted_twist_kernel', original_tf.make_fitted_twist_kernel)
            patch.setattr(fitted_twist_tf, 'make_recursive_fit_kernel', original_tf.make_recursive_fit_kernel)
            _, final, diagnostics, calls = original_adapter.execute_fitted_twist(
                dict(row, fit_theta=[x-variant*.001 for x in row['fit_theta']]), settings,
                theta+variant*.002, data+variant*.003, _seed(model, variant))
        expected = {'final': _host(final), 'diagnostics': diagnostics, 'subkernel_calls': calls}
        left, right = copy.deepcopy(actual), copy.deepcopy(expected)
        left['diagnostics'].pop('fit_execution')
        left['diagnostics'].pop('fit_enclosing_calls')
        for value in (left, right):
            value['diagnostics'].pop('fit_digest')
        return compare(left, right, 2e-10)

    return invoke, owners, make_fitted_twist_execution, reference


def input_invocation(monkeypatch):
    from bayesfilter.score_study import adapters, input_execution_tf
    from tests.test_filter_repair_score_inputs import case, reference, runtime_setup

    monkeypatch.setattr(adapters, 'configure_runtime', runtime_setup)
    original_factory = input_execution_tf.make_score_inputs
    owners = {}

    def tracked(*args, **kwargs):
        owner = original_factory(*args, **kwargs)
        owners[id(owner)] = owner
        return owner

    monkeypatch.setattr(input_execution_tf, 'make_score_inputs', tracked)
    cases = [case('gaussian', 'twist', variant)[:2] for variant in (0, 1)]
    cases[1][1]['study']['settings']['theta'] = [x+.002 for x in cases[1][1]['study']['settings']['theta']]

    def invoke(variant):
        return numerical_record(adapters.evaluate_gaussian(*cases[variant]))

    def check(variant, actual, patch):
        original, _ = reference('adapters')
        patch.setattr(original, 'configure_runtime', runtime_setup)
        return compare(actual, numerical_record(original.evaluate_gaussian(*cases[variant])), 1e-10)

    return invoke, owners, original_factory, check


@pytest.mark.parametrize('case', CASES)
def test_fixed_owner_reuse(case, request, monkeypatch):
    gpu = os.environ['CUDA_VISIBLE_DEVICES'] != '-1'
    placement = tf.constant(0.).device
    assert ('GPU:0' if gpu else 'CPU:0') in placement
    if gpu:
        tf.config.experimental.reset_memory_stats('GPU:0')
    invoke, owners, factory, reference = (input_invocation(monkeypatch) if case == 'gaussian_inputs'
                                         else fitted_invocation(case))
    with GPUProcessMonitor(gpu) as monitor:
        before = memory()
        expected = [invoke(variant) for variant in (0, 1)]
        assert expected[0] != expected[1]
        assert len(owners) == 1
        cache_before = factory.cache_info()._asdict()
        snapshots = {0: memory()}
        exact = True
        for index in range(1, 1025):
            variant = index % 2
            exact &= invoke(variant) == expected[variant]
            if index in (16, 128, 256, 512, 768, 1024):
                snapshots[index] = memory()
        cache_after = factory.cache_info()._asdict()
        traces = [owner.experimental_get_tracing_count() for owner in owners.values()]
        # Compile the independent authority only after all primary measurements.
        errors = [reference(variant, expected[variant], monkeypatch) for variant in (0, 1)]
        late_growth = snapshots[1024]['VmRSS']-snapshots[512]['VmRSS']
    record = {'schema': 'filter_repair.resource_owner_lifetime.v1', 'case': case,
        'calls': 1024, 'device': 'GPU' if gpu else 'CPU', 'placement': placement,
        'same_input_exact_replay': exact, 'changed_seed_and_theta_witness': expected,
        'independent_reference_errors': errors, 'owner_count': len(owners), 'trace_counts': traces,
        'cache_before': cache_before, 'cache_after': cache_after,
        'memory': {'before': before, 'snapshots': snapshots},
        'late_512_call_rss_growth_bytes': late_growth, 'device_observation': monitor.payload(),
        'measurement_excludes_comparison_compilation': True,
        'nonclaims': ['One fixed shape and two changing inputs do not prove universal capacity.',
                     'Cache reuse does not imply native executable eviction.']}
    save(request, 'resource-owner-lifetime.json', record)
    assert exact and len(owners) == 1 and traces == [1]
    assert cache_before['currsize'] == cache_after['currsize']
    assert late_growth <= 16*1024**2
    assert not monitor.errors
    assert all(process['pid'] == os.getpid() for sample in monitor.samples for process in sample['processes'])
    if gpu:
        assert snapshots[1024]['allocator']['current'] == snapshots[512]['allocator']['current']
        assert max(row['allocator']['peak'] for row in snapshots.values()) <= 2*1024**3
