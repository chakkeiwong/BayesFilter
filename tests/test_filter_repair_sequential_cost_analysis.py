"""Diagnostic analyzer vetoes for complete public sequential cost evidence."""

import copy
import json

import pytest

from tests.test_filter_repair_campaign import load
from tests.test_filter_repair_cost_provenance import gpu_records


def matrix(root):
    number = 0
    for device in ('CPU', 'GPU'):
        for dimension in (3, 5):
            for arm in ('prior', 'graph', 'xla'):
                for repeat in (0, 1, 2):
                    number += 1
                    directory = root / f'run-{number:05d}'
                    directory.mkdir()
                    run, provenance, observation = gpu_records()
                    provenance = {'tensorflow_version': 'diagnostic-fixture', 'tf32_enabled': True, **provenance}
                    run.update(source_sha256={'bayesfilter/fixture.py': 'unchanged'},
                        key=['test', 'fixture', 'after', 'covariance', 'on', 1, repeat, device])
                    if device == 'CPU':
                        run['device'] = 'CPU'
                        run['environment'].update(CUDA_VISIBLE_DEVICES='-1', BAYESFILTER_TEST_DEVICE_SCOPE='cpu')
                        provenance.update(cuda_visible_devices='-1', bayesfilter_test_device_scope='cpu',
                            trust_basis='explicit_cpu_reference')
                        provenance['gpu_memory_policy']['physical_devices'] = []
                        observation.update(enabled=False, uuid=None, samples=[])
                    memory = {'rollup': {'Rss': 1024 ** 3}, 'gpu': {'peak': 1024}}
                    record = {'accepted': True, 'status': 'usable', 'precision': [[1.]], 'count': 2}
                    report = {'schema': 'filter_sequential_public_cost.v1', 'dimension': dimension,
                        'arm': arm, 'numerical_authority': '3582b4ac', 'mechanism_baseline': '48acf5e96',
                        'gpu_process_observation': observation, 'gpu': device == 'GPU', 'jit_compile': arm == 'xla',
                        'samples': [{'seconds': .1, 'memory': memory}] * 20,
                        'trace_count': 1, 'result': record, 'original_result': record,
                        'changed_result': record, 'original_changed_result': record,
                        'prior_source_sha256': {'prior': 'same'}, 'original_source_sha256': {'original': 'same'},
                        'input_sha256': [f'inputs-{dimension}'], 'changed_input_sha256': [f'changed-{dimension}'],
                        'config': {'dimension': dimension}, 'stages': {'cold': memory, 'warm': memory},
                        'cold': {'seconds': 2., 'memory': memory}, 'changed_cost': {'seconds': .2, 'memory': memory},
                        'build_seconds': .1, 'trace_seconds': .2, 'graph_nodes': 10, 'graph_bytes': 100, 'hlo_bytes': 100}
                    for name, value in (('run.json', run), ('sequential-public-memory.json', report)):
                        (directory / name).write_text(json.dumps(value))
                    (directory / 'process.log').write_text(json.dumps(provenance) + '\n')
                    (directory / 'junit.xml').write_text('<testsuites><testsuite><testcase/></testsuite></testsuites>')
    return number


def test_complete_cost_matrix_accepts_separate_cpu_and_gpu_environments(tmp_path):
    last = matrix(tmp_path)
    report = load('analyze_filter_repair_sequential_public_costs').analyze(tmp_path, 1, last, ['CPU', 'GPU'])
    assert len(report['comparisons']) == 4
    assert all(not row['investigation_triggers'] for row in report['comparisons'])
    assert all(len(row['arms']['xla']['repeats']) == 3 for row in report['comparisons'])


@pytest.mark.parametrize('fault', ['source', 'environment', 'input', 'original_record', 'discrete_decision',
    'repeat', 'baseline', 'samples', 'sharing', 'test_failure'])
def test_invalid_complete_cost_cohort_fails_closed(tmp_path, fault):
    last = matrix(tmp_path)
    directory = tmp_path / 'run-00036'
    run = json.loads((directory / 'run.json').read_text())
    report = json.loads((directory / 'sequential-public-memory.json').read_text())
    if fault == 'source':
        run['source_sha256']['bayesfilter/fixture.py'] = 'changed'
    elif fault == 'environment':
        run['environment']['OMP_NUM_THREADS'] = '999'
    elif fault == 'input':
        report['input_sha256'] = ['different']
    elif fault == 'original_record':
        report['original_result']['precision'][0][0] = 2.
    elif fault == 'discrete_decision':
        report['result']['count'] = 3
    elif fault == 'repeat':
        run['key'][6] = 1
    elif fault == 'baseline':
        report['mechanism_baseline'] = 'retired-intermediate'
    elif fault == 'samples':
        report['samples'] = report['samples'][:-1]
    elif fault == 'sharing':
        process = copy.deepcopy(report['gpu_process_observation']['samples'][0]['processes'][0])
        process['pid'] += 1
        report['gpu_process_observation']['samples'][0]['processes'].append(process)
    else:
        (directory / 'junit.xml').write_text('<testsuites><testsuite><testcase><failure/></testcase></testsuite></testsuites>')
    (directory / 'run.json').write_text(json.dumps(run))
    (directory / 'sequential-public-memory.json').write_text(json.dumps(report))
    with pytest.raises((ValueError, AssertionError)):
        load('analyze_filter_repair_sequential_public_costs').analyze(tmp_path, 1, last, ['CPU', 'GPU'])


def test_incomplete_cost_cohort_cannot_pass(tmp_path):
    last = matrix(tmp_path)
    with pytest.raises(AssertionError, match='Three fresh processes'):
        load('analyze_filter_repair_sequential_public_costs').analyze(tmp_path, 1, last - 1, ['CPU', 'GPU'])


def test_every_gpu_sequential_cost_requires_unshared_preflight(tmp_path, monkeypatch):
    from types import SimpleNamespace

    driver = load('run_filter_repair_campaign')
    monkeypatch.setattr(driver, 'OUTPUT', tmp_path)
    for group in driver.TEST_BATCHES['sequential_public_cost_gpu']:
        args = SimpleNamespace(action='test', device='GPU', group=group,
            gpu_uuid='GPU-device-2', gpu_preflight=[{'performance_preflight_uncontended': False}] * 2)
        with pytest.raises(RuntimeError, match='declined before launch'):
            driver.require_unshared_cost_preflight(args)
    assert len(list(tmp_path.glob('cost-preflight-declined-*.json'))) == 6
