"""Source-bound nonlinear qualification without loading unrelated history."""

import hashlib
import json
from pathlib import Path

import numpy as np

from tests.test_filter_repair_ledh_seeded_readback import _load, _provenance

ROOT = Path(__file__).resolve().parents[1]
RAW = Path('/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917')
CASES = ('ekf', 'ukf', 'ledh', 'ledh_diagnostics', 'sgqf', 'sgqf_diagnostics',
         'kdm_covariance', 'kdm_covariance_diagnostics')


def _latest(group):
    for path in sorted(RAW.glob('run-*/run.json'), reverse=True):
        number = int(path.parent.name[4:])
        if number < 4783:
            break
        row = json.loads(path.read_text())
        if row['key'][1] == group:
            return number, row
    raise AssertionError(group)


def test_nonlinear_direction_qualification_readback(request):
    fixture = ROOT/'tests/fixtures/filter_repair_nonlinear_directions_20260929.json'
    fixture_hash = hashlib.sha256(fixture.read_bytes()).hexdigest()
    reports = []
    for case in CASES:
        devices = {}
        for device in ('CPU', 'GPU'):
            number, run = _latest(f'nonlinear_directions_{case}_{device.lower()}')
            assert run['state'] == 'passed' and run['device'] == device
            assert run['test_evidence'] == {'passed': True, 'tests': 1, 'failure': 0, 'error': 0, 'skipped': 0}
            provenance = _provenance(number)
            memory = provenance['gpu_memory_policy']
            assert memory['configured_before_logical_device_initialization']
            assert memory['all_physical_devices_memory_growth']
            if device == 'CPU':
                assert provenance['cuda_visible_devices'] == '-1'
            else:
                assert provenance['cuda_visible_devices'].startswith('GPU-')
                assert provenance['trust_basis'] == 'owner_designated_managed_session_visible_gpu_trusted'
                assert all(row['memory_growth'] for row in memory['physical_devices'])
            checked = 0
            for name, digest in run['source_sha256'].items():
                if name.startswith('bayesfilter/'):
                    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, (number, name)
                    checked += 1
            result = _load(number, f'nonlinear-directions-{case}.json')
            assert result['fixture_sha256'] == fixture_hash
            graph = result['graph']
            assert graph['trace_count'] == 1 and graph['jit_compile'] and graph['no_pfor_or_host_callback']
            hlo = RAW/f'run-{number:05d}'/f'nonlinear-directions-{case}.hlo.txt'
            assert hashlib.sha256(hlo.read_bytes()).hexdigest() == graph['hlo_sha256']
            endpoint = result['endpoint']
            if endpoint['numerical_validity'] == 'rejected':
                assert case not in ('ekf', 'ukf')
                assert result['qualification'] == 'original_and_enclosing_refusal_only_healthy_T2_gate_open'
                assert result['host_original_rejection_checked']
                assert result['scalar_values'] == [float('-inf')]*6 and result['scalar_scores'] == [0.]*6
                assert result.get('finite_auxiliary_max_absolute_error', 0.) <= 1e-9
                assert endpoint['exception'] == 'nonlinear score validity veto'
                assert endpoint['actual_factory_owner_verified'] and endpoint['trace_count'] == 1
                devices[device] = {'run': number, 'endpoint': endpoint,
                    'source_files_checked': checked, 'qualification': 'refusal_only'}
                continue
            assert result['baseline_max_absolute_error'] <= 1e-9
            assert result['changed_max_absolute_error'] <= 1e-9
            assert result['five_point_steps'] == [2e-4, 1e-4]
            assert max(result['five_point_max_absolute_errors']) <= 2e-6
            assert result['host_nonfinite_rejection_checked']
            assert endpoint['runtime']['kernel_calls'] == 1 and endpoint['runtime']['traces'] == 1
            assert endpoint['numerical_validity'] == 'pass' and endpoint['inference_status'] == 'mechanics_only'
            diag = endpoint['diagnostics']
            assert 'CPU' in diag['reference_device']
            assert diag['reference_mesh_relative_error'] <= 1e-7
            assert diag['reference_domain_relative_error'] <= 1e-7
            assert diag['reference_max_tail_mass'] <= 1e-9
            devices[device] = {'run': number, 'endpoint': endpoint, 'source_files_checked': checked,
                'max_parity_error': max(result['baseline_max_absolute_error'], result['changed_max_absolute_error']),
                'max_fd_error': max(result['five_point_max_absolute_errors'])}
        a, b = devices['CPU']['endpoint'], devices['GPU']['endpoint']
        assert a['numerical_validity'] == b['numerical_validity']
        if a['numerical_validity'] == 'rejected':
            provider = case.removesuffix('_diagnostics')
            group = ('nonlinear_directions_invalid_fixture_cpu' if provider == 'ledh'
                     else f'nonlinear_directions_invalid_{provider}_cpu')
            diagnostic_number, diagnostic_run = _latest(group)
            assert diagnostic_run['state'] == 'passed'
            diagnostic = _load(diagnostic_number, f'nonlinear-{provider}-invalid-fixture.json')
            assert not diagnostic['enclosing_valid']
            tv = diagnostic['scalar_diagnostics']['diagnostic_reset_column_tv']
            assert np.max(tv) > 1e-4
            reports.append({'case': case, 'devices': devices, 'qualification': 'refusal_only',
                'diagnostic_run': diagnostic_number, 'maximum_reset_column_tv': float(np.max(tv)),
                'healthy_T2_derivative_and_cost_gate_open': True})
            continue
        np.testing.assert_allclose(a['value'], b['value'], atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(a['score'], b['score'], atol=1e-9, rtol=1e-9)
        reports.append({'case': case, 'devices': devices,
            'cpu_gpu_max_score_difference': float(np.max(np.abs(np.asarray(a['score'])-b['score'])))})
    report = {'schema': 'filter_repair_nonlinear_direction_qualification.v1', 'cases': reports,
        'fixture_sha256': fixture_hash, 'costs_accepted': False,
        'nonclaims': ['No full canonical LEDH, exact nonlinear likelihood, scientific or HMC admission.']}
    (Path(request.config.getoption('xmlpath')).parent/'nonlinear-directions-readback.json').write_text(
        json.dumps(report, indent=2)+'\n')
