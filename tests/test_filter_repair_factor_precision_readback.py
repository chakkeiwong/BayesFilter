"""Qualification of final symmetric factor precision against preserved fits."""

import hashlib
import json
from pathlib import Path

import numpy as np

from tests.test_filter_repair_dz5_exact_fit_readback import matrix_record
from tests.test_filter_repair_dz5_initializer_fit_localization import differences
from tests.test_filter_repair_geometry_control import save

RAW = Path('/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917')


def test_projection_repairs_only_final_precision_and_reports(request):
    records, checks, failures = {}, {}, []
    source_runs = {'GPU': (4607, 4613), 'CPU': (4609, 4614)}
    for device, (before_number, after_number) in source_runs.items():
        before_path = RAW / f'run-{before_number:05d}/dz5-exact-saved-fit.json'
        after_path = RAW / f'run-{after_number:05d}/dz5-exact-saved-fit.json'
        before, after = (json.loads(path.read_text()) for path in (before_path, after_path))
        assert before['device'] == after['device'] == device
        assert before['operand_sha256'] == after['operand_sha256']
        records[device] = after
        if after['fit_error_code'] or not after['fit_numerically_usable']:
            failures.append({'device': device, 'failure': 'fit_remains_rejected', 'code': after['fit_error_code']})
        left, right = before['raw']['fit']['fits'], after['raw']['fit']['fits']
        fit_checks = []
        for family in range(3):
            for replicate in range(2):
                a, b = np.asarray(left['precision'][family][replicate]), np.asarray(right['precision'][family][replicate])
                expected = .5 * (a + a.T) if family else a
                # Full original tolerance, without a new numerical waiver.
                precision_ok = bool(np.allclose(b, expected, atol=1e-10, rtol=1e-10))
                covariance_ok = bool(np.allclose(right['covariance'][family][replicate],
                    left['covariance'][family][replicate], atol=1e-10, rtol=1e-10))
                discrete_same = all(left[key][family][replicate] == right[key][family][replicate]
                    for key in ('status', 'flags', 'anchors', 'invalid_covariance_evaluations'))
                metrics_a, metrics_b = np.asarray(left['factor_metrics'][family][replicate]), np.asarray(right['factor_metrics'][family][replicate])
                optimizer_same = np.array_equal(metrics_a[6:10], metrics_b[6:10])
                loss_same = bool(np.isclose(metrics_a[10], metrics_b[10], atol=1e-10, rtol=1e-10))
                symmetry = bool(np.array_equal(b, b.T)) if family else True
                row = {'family': ('dense', 'factor_1', 'factor_2')[family], 'replicate': replicate,
                    'precision_matches_before_projection': precision_ok,
                    'covariance_preserved': covariance_ok, 'fit_decisions_anchors_preserved': discrete_same,
                    'optimizer_counters_preserved': bool(optimizer_same), 'optimizer_loss_preserved': loss_same,
                    'factor_precision_exactly_symmetric': symmetry,
                    'maximum_precision_change': float(np.max(np.abs(a - b))),
                    'matrix_diagnostics': matrix_record(b, right['covariance'][family][replicate])}
                fit_checks.append(row)
                if not all((precision_ok, covariance_ok, discrete_same, optimizer_same, loss_same, symmetry)):
                    failures.append({'device': device, **row})
        checks[device] = {'source_runs': [before_number, after_number],
            'source_hashes': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (before_path, after_path)},
            'before_error_code': before['fit_error_code'], 'after_error_code': after['fit_error_code'],
            'selected_family': after.get('result', {}).get('selected_family'), 'fits': fit_checks,
            'complete_record_differences': differences(after.get('result'), before.get('result'))}
    result = {'schema': 'filter_factor_precision_symmetry_qualification.v1',
        'checks': checks, 'failures': failures,
        'remaining_CPU_GPU_record_differences': differences(records['GPU'].get('result'), records['CPU'].get('result')),
        'nonclaims': ['This final representation repair does not establish cross-backend optimizer equivalence, actual-consumer source admission or convergence.']}
    save(request, 'factor-precision-symmetry-qualification.json', result)
    assert not failures, failures
