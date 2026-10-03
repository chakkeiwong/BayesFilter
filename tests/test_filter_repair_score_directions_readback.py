"""Read back the fresh CPU/GPU consumer qualification with source provenance."""

import hashlib
import json
from pathlib import Path

import numpy as np

from tests.test_filter_repair_ledh_seeded_readback import _load, _provenance

ROOT = Path(__file__).resolve().parents[1]
RAW = Path('/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917')
CASES = ('ledh', 'ledh_diagnostics', 'sgqf', 'kdm_covariance', 'integrated_kdm', 'resampling_kdm')


def _latest(group):
    # This unit starts at4756. Do not load thousands of unrelated full source
    # manifests just to find its most recent record, or retain them in memory.
    for path in sorted(RAW.glob('run-*/run.json'), reverse=True):
        number = int(path.parent.name[4:])
        if number < 4756:
            break
        row = json.loads(path.read_text())
        if row['key'][1] == group:
            return number, row
    raise AssertionError(group)



def test_score_directions_qualification_readback(request):
    fixture = ROOT / 'tests/fixtures/filter_repair_score_directions_20260929.json'
    fixture_hash = hashlib.sha256(fixture.read_bytes()).hexdigest()
    reports = []
    for case in CASES:
        devices = {}
        for device in ('CPU', 'GPU'):
            number, run = _latest(f'score_directions_{case}_{device.lower()}')
            assert run['state'] == 'passed' and run['device'] == device
            assert run['test_evidence'] == {'passed': True, 'tests': 1, 'failure': 0, 'error': 0, 'skipped': 0}
            provenance = _provenance(number)
            policy = provenance['gpu_memory_policy']
            assert policy['configured_before_logical_device_initialization']
            assert policy['all_physical_devices_memory_growth']
            if device == 'CPU':
                assert provenance['cuda_visible_devices'] == '-1'
            else:
                assert provenance['cuda_visible_devices'].startswith('GPU-')
                assert provenance['trust_basis'] == 'owner_designated_managed_session_visible_gpu_trusted'
                assert all(row['memory_growth'] for row in policy['physical_devices'])
            checked = 0
            for name, digest in run['source_sha256'].items():
                if not name.startswith('bayesfilter/'):
                    continue
                # This later shape annotation is reached only by the mixture
                # branch. Its own qualification runs bind its corrected source.
                if case != 'kdm_covariance' and name == 'bayesfilter/score_study/mixture_covariance_tf.py':
                    continue
                assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, (number, name)
                checked += 1
            result = _load(number, f'directions-{case}.json')
            assert result['fixture_sha256'] == fixture_hash
            assert result['baseline_max_absolute_error'] <= 1e-9
            assert result['changed_max_absolute_error'] <= 1e-9
            assert result['five_point_steps'] == [2e-4, 1e-4]
            assert max(result['five_point_max_absolute_errors']) <= 2e-6
            assert result['nonfinite_reset_rejected']
            assert result['host_nonfinite_rejection_checked']
            assert result['graph']['trace_count'] == 1 and result['graph']['jit_compile']
            hlo = RAW / f'run-{number:05d}' / f'directions-{case}.hlo.txt'
            assert hashlib.sha256(hlo.read_bytes()).hexdigest() == result['graph']['hlo_sha256']
            endpoint = result['endpoint']
            assert endpoint['runtime']['kernel_calls'] == 1 and endpoint['runtime']['traces'] == 1
            assert endpoint['diagnostics']['direction_execution'] == 'enclosing_tensorflow_loop'
            assert endpoint['numerical_validity'] == 'pass' and endpoint['inference_status'] == 'mechanics_only'
            devices[device] = {'run': number, 'source_files_checked': checked, 'endpoint': endpoint,
                'max_parity_error': max(result['baseline_max_absolute_error'], result['changed_max_absolute_error']),
                'max_fd_error': max(result['five_point_max_absolute_errors'])}
        a, b = devices['CPU']['endpoint'], devices['GPU']['endpoint']
        np.testing.assert_allclose(a['value'], b['value'], atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(a['score'], b['score'], atol=1e-9, rtol=1e-9)
        reports.append({'case': case, 'devices': devices,
            'cpu_gpu_max_score_difference': float(np.max(np.abs(np.asarray(a['score'])-b['score'])))})
    for device, count in (('cpu', 3), ('gpu', 2)):
        number, run = _latest('score_directions_generic_'+device)
        assert run['state'] == 'passed' and run['test_evidence']['tests'] == count
    for number in (4757, 4761, 4762):
        assert _load(number, 'run.json')['state'] == 'failed'
    report = {'schema': 'filter_repair_score_direction_qualification.v1',
        'fixture_sha256': fixture_hash, 'cases': reports, 'failed_attempts_preserved': [4757, 4761, 4762],
        'costs_accepted': False, 'nonclaims': ['No full canonical LEDH, scientific, HMC or performance admission.']}
    (Path(request.config.getoption('xmlpath')).parent / 'score-directions-readback.json').write_text(
        json.dumps(report, indent=2)+'\n')
