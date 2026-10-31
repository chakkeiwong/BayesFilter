"""Recompute CV-interval containment and resolved/ambiguous decision gates."""

import decimal
import hashlib
import json
from pathlib import Path

from tests.test_filter_repair_iapf_resolution import _decimal_cv
from tests.test_filter_repair_ledh_seeded_readback import _load, _provenance
from tests.test_filter_repair_nonlinear_scope import RAW, ROOT, _latest


def test_iapf_resolution_readback(request):
    devices = []
    for device in ('cpu', 'gpu'):
        number, run = _latest(f'iapf_resolution_{device}')
        assert run['state'] == 'passed' and run['device'] == device.upper()
        assert run['test_evidence'] == {'passed': True, 'tests': 1, 'failure': 0, 'error': 0, 'skipped': 0}
        result = _load(number, 'iapf-resolution.json')
        assert result['complete'] and not result['runtime_migrated']
        assert len(result['rows']) == 36 and len(result['invalid']) == 7
        unresolved = []
        widths = []
        for row in result['rows']:
            candidate, original = row['candidate'], row['reference']
            assert candidate['next_particles'] == original['next_particles']
            if original['complete_window']:
                exact = _decimal_cv(row['logs'], row['k'])
                assert str(exact) == row['decimal_cv']
                lower, upper = candidate['cv_lower'], candidate['cv_upper']
                assert decimal.Decimal.from_float(lower) <= exact <= decimal.Decimal.from_float(upper)
                assert lower <= original['cv'] <= upper and lower <= candidate['cv'] <= upper
                widths.append(upper-lower)
            if row['expect_unresolved']:
                assert candidate['action'] == -2 and not candidate['decision_resolved']
                assert candidate['cv_lower'] <= row['tau'] <= candidate['cv_upper']
                unresolved.append(row['label'])
            else:
                assert candidate['decision_resolved']
                assert {0: 'fit', 1: 'final', 2: 'capacity_veto'}[candidate['action']] == original['action']
        assert len(unresolved) == 3
        for candidate in result['invalid']:
            assert not candidate['valid'] and not candidate['decision_resolved'] and candidate['action'] == -1
        for k, graph in result['graphs'].items():
            assert graph['trace_count'] == 1 and graph['jit_compile'] and not graph['host_callbacks']
            hlo = RAW/f'run-{number:05d}'/f'iapf-resolution-k{k}.hlo.txt'
            assert hashlib.sha256(hlo.read_bytes()).hexdigest() == graph['hlo_sha256']
        for name in ('bayesfilter/score_study/iapf_controller_tf.py', 'tests/test_filter_repair_iapf_resolution.py'):
            assert run['source_sha256'][name] == hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        provenance = _provenance(number)
        assert provenance['gpu_memory_policy']['configured_before_logical_device_initialization']
        if device == 'gpu':
            assert provenance['trust_basis'] == 'owner_designated_managed_session_visible_gpu_trusted'
        else:
            assert provenance['cuda_visible_devices'] == '-1'
        devices.append({'device': device, 'run': number, 'resolved_cases': 33,
                        'unresolved_cases': unresolved, 'invalid_cases': 7, 'maximum_interval_width': max(widths)})
    report = {'schema': 'filter_repair_iapf_resolution_readback.v1', 'devices': devices,
              'runtime_migrated': False, 'scope': 'Checked decision mechanics; full adaptive endpoint remains open.'}
    (Path(request.config.getoption('xmlpath')).parent/'iapf-resolution-readback.json').write_text(
        json.dumps(report, indent=2)+'\n')
