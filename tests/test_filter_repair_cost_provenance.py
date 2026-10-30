"""Diagnostic cost evidence must reject device identity and sharing mismatches."""

import copy
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import filter_repair_cost_provenance as monitor_module
from scripts.filter_repair_cost_provenance import (
    SCHEMA,
    GPUProcessMonitor,
    require_same_physical_gpu,
    validate_cost_device,
)


def gpu_records(index=2):
    uuid = f'GPU-device-{index}'
    preflight = {'selected_uuid': uuid, 'performance_preflight_uncontended': True,
        'desktop_fallback': False, 'utilization_percent': 0,
        'devices': [{'uuid': uuid, 'desktop': False, 'compute_processes': []}]}
    run = {'state': 'passed', 'device': 'GPU', 'gpu_uuid': uuid,
        'environment': {'CUDA_VISIBLE_DEVICES': uuid, 'TF_FORCE_GPU_ALLOW_GROWTH': 'true',
            'BAYESFILTER_TEST_DEVICE_SCOPE': 'visible'},
        'gpu_preflight': [copy.deepcopy(preflight), copy.deepcopy(preflight)],
        'gpu_performance_preflight_uncontended': True}
    provenance = {'cuda_visible_devices': uuid,
        'bayesfilter_test_device_scope': 'visible',
        'trust_basis': 'owner_designated_managed_session_visible_gpu_trusted',
        'gpu_memory_policy': {'mode': 'memory_growth', 'all_physical_devices_memory_growth': True,
            'configured_before_logical_device_initialization': True,
            'physical_devices': [{'memory_growth': True}]}}
    observation = {'schema': SCHEMA, 'enabled': True, 'uuid': uuid, 'pid': 123, 'errors': [],
        'samples': [{'monotonic_seconds': float(i), 'processes': [{'uuid': uuid, 'pid': 123, 'name': 'python'}]}
                    for i in (1, 2)]}
    return run, provenance, observation


@pytest.mark.parametrize('index', [0, 1, 2, 3])
def test_cost_identity_uses_uuid_with_no_historical_index_requirement(index):
    assert validate_cost_device(*gpu_records(index)) == f'GPU-device-{index}'


@pytest.mark.parametrize('fault', ['worker_visibility', 'observation_uuid', 'preflight_uuid',
    'shared_preflight', 'hidden_preflight_process', 'shared_during_run', 'wrong_sample_uuid',
    'missing_samples', 'failed_monitor', 'growth', 'multiple_visible_devices', 'desktop', 'pytest_scope'])
def test_invalid_gpu_cost_evidence_is_rejected(fault):
    run, provenance, observation = gpu_records()
    if fault == 'worker_visibility':
        provenance['cuda_visible_devices'] = 'GPU-device-3'
    elif fault == 'observation_uuid':
        observation['uuid'] = 'GPU-device-3'
    elif fault == 'preflight_uuid':
        run['gpu_preflight'][0]['selected_uuid'] = 'GPU-device-3'
    elif fault == 'shared_preflight':
        run['gpu_performance_preflight_uncontended'] = False
    elif fault == 'hidden_preflight_process':
        run['gpu_preflight'][1]['devices'][0]['compute_processes'] = ['another_python']
    elif fault == 'shared_during_run':
        observation['samples'][1]['processes'].append({'uuid': run['gpu_uuid'], 'pid': 456})
    elif fault == 'wrong_sample_uuid':
        observation['samples'][1]['processes'][0]['uuid'] = 'GPU-device-3'
    elif fault == 'missing_samples':
        observation['samples'] = observation['samples'][:1]
    elif fault == 'failed_monitor':
        observation['errors'] = ['query failed']
    elif fault == 'growth':
        provenance['gpu_memory_policy']['configured_before_logical_device_initialization'] = False
    elif fault == 'multiple_visible_devices':
        provenance['gpu_memory_policy']['physical_devices'] *= 2
    elif fault == 'desktop':
        run['gpu_preflight'][1]['devices'][0]['desktop'] = True
    elif fault == 'pytest_scope':
        provenance['bayesfilter_test_device_scope'] = 'cpu'
    with pytest.raises(ValueError):
        validate_cost_device(run, provenance, observation)


def test_mixed_physical_gpu_pairs_are_rejected_after_individual_validation():
    first, second = (validate_cost_device(*gpu_records(index)) for index in (2, 3))
    require_same_physical_gpu([None, first, first])
    with pytest.raises(ValueError, match='Mixed physical GPU'):
        require_same_physical_gpu([first, second])


def test_explicit_cpu_cost_reference_needs_gpu_hiding():
    run, provenance, observation = gpu_records()
    run['device'] = 'CPU'
    run['environment']['CUDA_VISIBLE_DEVICES'] = '-1'
    provenance['cuda_visible_devices'] = '-1'
    provenance['trust_basis'] = 'explicit_cpu_reference'
    provenance['gpu_memory_policy']['physical_devices'] = []
    observation.update(enabled=False, uuid=None, samples=[])
    assert validate_cost_device(run, provenance, observation) is None
    run['environment']['CUDA_VISIBLE_DEVICES'] = 'GPU-device-2'
    provenance['cuda_visible_devices'] = 'GPU-device-2'
    with pytest.raises(ValueError, match='explicitly hide'):
        validate_cost_device(run, provenance, observation)


def test_monitor_records_actual_pid_and_sharing_without_changing_workers(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-device-2')
    monitor = GPUProcessMonitor(True)
    monkeypatch.setattr(monitor_module.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout=(
        f'GPU-device-2, {monitor.pid}, own_python\nGPU-device-2, 456, other_python\n'
        'GPU-device-3, 789, unrelated_gpu\n')))
    with monitor:
        pass
    observation = monitor.payload()
    assert len(observation['samples']) >= 2 and not observation['errors']
    assert [row['pid'] for row in observation['samples'][0]['processes']] == [monitor.pid, 456]
    run, provenance, _ = gpu_records()
    with pytest.raises(ValueError, match='Shared or mismatched GPU during'):
        validate_cost_device(run, provenance, observation)


def test_monitor_failure_is_recorded_and_cannot_validate_costs(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-device-2')

    def fail(*args, **kwargs):
        raise OSError('synthetic device query failure')

    monkeypatch.setattr(monitor_module.subprocess, 'run', fail)
    with GPUProcessMonitor(True) as monitor:
        pass
    run, provenance, _ = gpu_records()
    assert monitor.payload()['errors']
    with pytest.raises(ValueError, match='failed cost-device'):
        validate_cost_device(run, provenance, monitor.payload())


@pytest.mark.parametrize('visible,scope', [('-1', 'cpu'), ('GPU-device-2', 'visible')])
def test_campaign_visibility_survives_actual_pytest_conftest(monkeypatch, visible, scope):
    from scripts.filter_repair_test_worker import configure_test_device_scope

    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', visible)
    monkeypatch.delenv('BAYESFILTER_TEST_DEVICE_SCOPE', raising=False)
    assert configure_test_device_scope() == (visible, scope)
    runpy.run_path(str(Path(__file__).with_name('conftest.py')))
    assert monitor_module.os.environ['CUDA_VISIBLE_DEVICES'] == visible
    assert monitor_module.os.environ['BAYESFILTER_TEST_DEVICE_SCOPE'] == scope


def test_contradictory_pytest_scope_fails_before_tensorflow(monkeypatch):
    from scripts.filter_repair_test_worker import configure_test_device_scope

    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-device-2')
    monkeypatch.setenv('BAYESFILTER_TEST_DEVICE_SCOPE', 'cpu')
    with pytest.raises(ValueError, match='contradicts'):
        configure_test_device_scope()
