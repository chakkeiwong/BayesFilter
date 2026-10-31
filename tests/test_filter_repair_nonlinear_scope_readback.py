"""Audit frozen nominations and untouched CPU/GPU results independently."""

import hashlib
import json
from pathlib import Path

import numpy as np

from tests.test_filter_repair_ledh_seeded_readback import _load, _provenance
from tests.test_filter_repair_nonlinear_scope import (
    PROVIDERS,
    RAW,
    ROOT,
    _latest,
    _nomination,
    _partition,
)


def test_nonlinear_calibrated_scope_readback(request):
    reports = []
    for provider, diagnostics in ((p, d) for p in PROVIDERS for d in (True, False)):
        group_prefix = '' if diagnostics else 'plain_'
        suffix = '' if diagnostics else '-plain'
        number, count = _nomination(provider)
        selection = _load(number, 'nonlinear-scope-calibration.json')
        assert len(selection['calibration']) == len(selection['validation']) == 3
        assert all(row['passed'] and row['trace_count'] == 1
                   for row in (*selection['calibration'], *selection['validation']))
        devices = {}
        for device in ('CPU', 'GPU'):
            run_number, run = _latest(f'nonlinear_scope_untouched_{group_prefix}{provider}_{device.lower()}')
            assert run['state'] == 'passed' and run['device'] == device
            assert run['test_evidence'] == {'passed': True, 'tests': 1, 'failure': 0, 'error': 0, 'skipped': 0}
            result = _load(run_number, 'nonlinear-scope-untouched'+suffix+'.json')
            assert result.get('diagnostics_enabled', True) is diagnostics
            assert result['nomination_run'] == number and result['count'] == count
            assert result['untouched_sha256'] == _partition('untouched')[1]
            assert len(result['cases']) == 2
            provenance = _provenance(run_number)
            memory = provenance['gpu_memory_policy']
            assert memory['configured_before_logical_device_initialization']
            assert memory['all_physical_devices_memory_growth']
            if device == 'CPU':
                assert provenance['cuda_visible_devices'] == '-1'
            else:
                assert provenance['cuda_visible_devices'].startswith('GPU-')
                assert provenance['trust_basis'] == 'owner_designated_managed_session_visible_gpu_trusted'
                assert all(row['memory_growth'] for row in memory['physical_devices'])
            for name, digest in run['source_sha256'].items():
                if name.startswith('bayesfilter/'):
                    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, (run_number, name)
            for case in result['cases']:
                assert case['baseline_error'] <= 1e-9 and case['changed_error'] <= 1e-9
                assert case['five_point_steps'] == [2e-4, 1e-4]
                assert max(case['five_point_errors']) <= 2e-6
                assert case['graph']['trace_count'] == 1 and case['graph']['jit_compile']
                hlo = RAW/f'run-{run_number:05d}'/f'nonlinear-scope{suffix}-{provider}-{case["case"]}.hlo.txt'
                assert hashlib.sha256(hlo.read_bytes()).hexdigest() == case['graph']['hlo_sha256']
                endpoint = case['endpoint']
                assert endpoint['numerical_validity'] == 'pass' and endpoint['inference_status'] == 'mechanics_only'
                assert endpoint['runtime']['kernel_calls'] == endpoint['runtime']['traces'] == 1
                assert endpoint['diagnostics']['controls']['reset_sinkhorn_steps'] == count
                assert endpoint['diagnostics']['controls']['reset_balance_steps'] == count
                assert ('control_diagnostics' in endpoint['diagnostics']) is diagnostics
            devices[device] = {'run': run_number, 'cases': result['cases']}
        differences = []
        for cpu, gpu in zip(devices['CPU']['cases'], devices['GPU']['cases'], strict=True):
            np.testing.assert_allclose(cpu['endpoint']['value'], gpu['endpoint']['value'], atol=1e-9, rtol=1e-9)
            np.testing.assert_allclose(cpu['endpoint']['score'], gpu['endpoint']['score'], atol=1e-9, rtol=1e-9)
            differences.append(float(np.max(np.abs(np.asarray(cpu['endpoint']['score'])-gpu['endpoint']['score']))))
        reports.append({'provider': provider, 'diagnostics_enabled': diagnostics,
                        'count': count, 'nomination_run': number, 'devices': devices,
                        'cpu_gpu_score_max_errors': differences})
    for provider in PROVIDERS:
        diagnostic, plain = [row for row in reports if row['provider'] == provider]
        for device in ('CPU', 'GPU'):
            for a, b in zip(diagnostic['devices'][device]['cases'], plain['devices'][device]['cases'], strict=True):
                np.testing.assert_allclose(a['endpoint']['value'], b['endpoint']['value'], atol=1e-9, rtol=1e-9)
                np.testing.assert_allclose(a['endpoint']['score'], b['endpoint']['score'], atol=1e-9, rtol=1e-9)
    prior = _load(4805, 'nonlinear-directions-readback.json')
    assert len([row for row in prior['cases'] if row.get('qualification') == 'refusal_only']) == 6
    report = {'schema': 'filter_repair_nonlinear_calibrated_scope_summary.v1', 'providers': reports,
        'prior_failed_scope_preserved': True, 'performance_accepted': False,
        'nonclaims': ['Test-scope calibration only; no canonical tuning artifact, production default or scientific admission.']}
    (Path(request.config.getoption('xmlpath')).parent/'nonlinear-scope-readback.json').write_text(
        json.dumps(report, indent=2)+'\n')
