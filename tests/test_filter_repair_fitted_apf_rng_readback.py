"""Independently recompute whether compilation preserves fitted-APF inputs."""

import hashlib
import json
from pathlib import Path

import numpy as np

from tests.test_filter_repair_ledh_seeded_readback import _load, _provenance
from tests.test_filter_repair_nonlinear_scope import RAW, ROOT, _latest


def test_fitted_apf_rng_readback(request):
    reports = []
    for device in ('CPU', 'GPU'):
        number, run = _latest(f'fitted_apf_rng_{device.lower()}')
        assert run['state'] == 'passed' and run['device'] == device
        assert run['test_evidence'] == {'passed': True, 'tests': 1, 'failure': 0, 'error': 0, 'skipped': 0}
        report = _load(number, 'fitted-apf-rng.json')
        assert report['trace_count'] == 1 and report['jit_compile']
        assert [row['seed'] for row in report['rows']] == [[9296027, 1], [9296027, 2]]
        hlo = RAW/f'run-{number:05d}'/'fitted-apf-rng.hlo.txt'
        assert hashlib.sha256(hlo.read_bytes()).hexdigest() == report['hlo_sha256']
        name = 'tests/test_filter_repair_fitted_apf_rng.py'
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == run['source_sha256'][name]
        provenance = _provenance(number)
        assert provenance['gpu_memory_policy']['configured_before_logical_device_initialization']
        if device == 'GPU':
            assert provenance['trust_basis'] == 'owner_designated_managed_session_visible_gpu_trusted'
            assert all(row['memory_growth'] for row in provenance['gpu_memory_policy']['physical_devices'])
        else:
            assert provenance['cuda_visible_devices'] == '-1'
        fields = []
        for row in report['rows']:
            assert set(row['fields']) == {'initial', 'process', 'ancestors', 'mixture'}
            for name, field in row['fields'].items():
                left, right = np.asarray(field['reference'], dtype=np.float64), np.asarray(field['compiled'], dtype=np.float64)
                assert list(left.shape) == list(right.shape) == field['shape']
                assert hashlib.sha256(left.tobytes()).hexdigest() == field['reference_sha256']
                assert hashlib.sha256(right.tobytes()).hexdigest() == field['compiled_sha256']
                error = float(np.max(np.abs(left-right)))
                exact = bool(np.array_equal(left, right))
                compatible = exact if name in ('ancestors', 'mixture') else error <= 1e-12
                assert error == field['maximum_absolute_difference']
                assert exact is field['exact'] and compatible is field['compatible']
                assert device in field['reference_device'] and device in field['compiled_device']
                fields.append({'seed': row['seed'], 'field': name, 'exact': exact,
                               'maximum_absolute_difference': error, 'compatible': compatible})
        assert all(field['compatible'] for field in fields) is report['simple_compilation_preserves_inputs']
        reports.append({'run': number, 'device': device, 'fields': fields,
                        'simple_compilation_preserves_inputs': report['simple_compilation_preserves_inputs']})
    result = {'schema': 'filter_repair_fitted_apf_rng_summary.v1', 'devices': reports,
              'runtime_repaired': False, 'seed_migration_authorized': False,
              'nonclaims': ['An artifact test pass is not seeded-input compatibility or filter admission.']}
    (Path(request.config.getoption('xmlpath')).parent/'fitted-apf-rng-readback.json').write_text(
        json.dumps(result, indent=2)+'\n')
