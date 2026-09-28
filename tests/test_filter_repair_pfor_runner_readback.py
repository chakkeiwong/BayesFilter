"""Source-bound readback of runner pfor repairs and preserved compiler failures."""

import hashlib
import json
from pathlib import Path

from scripts.enforce_filter_gradient_policy import inspect_source, verify
from tests.test_filter_repair_ledh_seeded_readback import _load, _provenance

ROOT = Path(__file__).resolve().parents[1]
RAW = Path('/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917')


def test_runner_dispositions_and_p91_evidence(request):
    catalog = json.loads((ROOT / 'docs/plans/filter_gradient_pfor_runner_dispositions_20260929.json').read_text())
    reports = []
    for number, count in ((4720, 23), (4724, 3), (4725, 3)):
        run = _load(number, 'run.json')
        assert run['state'] == 'passed'
        assert run['test_evidence'] == {'passed': True, 'tests': count, 'failure': 0, 'error': 0, 'skipped': 0}
        provenance = _provenance(number)
        assert provenance['tf32_enabled']
        policy = provenance['gpu_memory_policy']
        assert policy['configured_before_logical_device_initialization']
        assert policy['all_physical_devices_memory_growth']
        assert provenance['cuda_visible_devices'] == run['environment']['CUDA_VISIBLE_DEVICES']
        if number == 4725:
            assert provenance['trust_basis'] == 'owner_designated_managed_session_visible_gpu_trusted'
            assert len(policy['physical_devices']) == 1
        else:
            assert provenance['cuda_visible_devices'] == '-1'
        paths = [row['path'] for row in catalog['rows'] if row['path'].startswith('scripts/p91')]
        if number == 4720:
            paths = [row['path'] for row in catalog['rows'] if not row['path'].startswith('scripts/p91')]
        else:
            paths += ['bayesfilter/highdim/models.py', 'bayesfilter/testing/p91_complete_data_diagnostics_tf.py']
        for relative in paths:
            assert run['source_sha256'][relative] == hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), relative
        reports.append({'run': number, 'tests': count, 'device': run['device'],
            'verified_source_scope': paths})
    errors = []
    for number in (4724, 4725):
        for arm in ('jit_check', 'benchmark', 'mapped_target'):
            name = 'pfor-p91-' + arm
            result = _load(number, name + '.json')
            assert result['seed'] == 81129 and len(result['records']) == 2
            assert result['graph']['trace_count'] == 1 and result['graph']['jit_compile']
            assert result['graph']['no_pfor_or_host_callback']
            assert hashlib.sha256((RAW / f'run-{number:05d}' / (name + '.hlo.txt')).read_bytes()).hexdigest() == result['graph']['hlo_sha256']
            for row in result['records']:
                errors.extend(row['fd_errors'])
                assert all(('GPU:0' if number == 4725 else 'CPU:0') in device for device in row['devices'])
    assert max(errors) < 2e-6
    for number in (4721, 4722):
        run = _load(number, 'run.json')
        assert run['state'] == 'failed' and run['test_evidence']['failure'] == 2
        assert run['test_evidence']['tests'] == 3
        assert 'Cannot find body function' in (RAW / f'run-{number:05d}' / 'process.log').read_text()
    traced = _load(4723, 'pfor-p91-graph-localization.json')
    assert traced['function_count'] == 10 and len(traced['loops']) == 3
    assert all(row['all_defined_before_xla'] for row in traced['loops'])
    for row in catalog['rows']:
        source = (ROOT / row['path']).read_text()
        assert hashlib.sha256(source.encode()).hexdigest() == row['current_source_sha256']
        assert not [finding for finding in inspect_source(row['path'], source) if 'pfor' in finding['kind']]
    policy = verify()
    assert policy['passed'] and policy['guarded_source_count'] >= 300
    assert policy['exact_exceptions_used'] == 1436
    directory = Path(request.config.getoption('xmlpath')).parent
    (directory / 'pfor-runner-readback.json').write_text(json.dumps({'reports': reports,
        'maximum_finite_difference_error': max(errors), 'sites_disposed': 28,
        'paths_disposed': 17, 'preserved_failures': [4721, 4722],
        'graph_localization': traced, 'policy': policy,
        'whole_master_closed': False}, indent=2) + '\n')
