"""Independent diagnostic conditioning/symmetry analysis of exact-input fits."""

import hashlib
import json
from pathlib import Path

import numpy as np

from tests.test_filter_repair_dz5_initializer_fit_localization import differences
from tests.test_filter_repair_geometry_control import save

RAW = Path('/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917')


def matrix_record(precision, covariance):
    precision, covariance = np.asarray(precision, dtype=float), np.asarray(covariance, dtype=float)
    assert precision.shape == covariance.shape == (23, 23)
    if not (np.all(np.isfinite(precision)) and np.all(np.isfinite(covariance))):
        return {'finite': False, 'spectral_diagnostics_available': False}
    skew = np.abs(precision - precision.T)
    bound = 1e-12 + 1e-10 * np.abs(precision.T)
    row, col = np.unravel_index(np.argmax(skew / bound), skew.shape)
    psym, csym = (precision + precision.T) / 2, (covariance + covariance.T) / 2
    values, cov_values = np.linalg.eigvalsh(psym), np.linalg.eigvalsh(csym)
    norm = np.linalg.norm(precision, ord=2)
    cnorm = np.linalg.norm(covariance, ord=2)
    inverse = np.linalg.inv(csym) if np.all(cov_values > 0) else None
    return {'finite': bool(np.all(np.isfinite(precision)) & np.all(np.isfinite(covariance))),
        'maximum_skew': float(np.max(skew)), 'symmetry_bound_ratio': float(np.max(skew / bound)),
        'violates_runtime_symmetry': bool(np.any(skew > bound)),
        'worst_pair': {'row': int(row), 'column': int(col), 'value': float(precision[row, col]),
            'transpose_value': float(precision[col, row]), 'allowed_error': float(bound[row, col])},
        'normwise_relative_skew': float(np.linalg.norm(precision - precision.T, ord=2) / norm) if norm else 0.,
        'precision_eigenvalues': values.tolist(), 'covariance_eigenvalues': cov_values.tolist(),
        'condition_number': float(values[-1] / values[0]) if values[0] > 0 else None,
        'covariance_condition_number': float(cov_values[-1] / cov_values[0]) if cov_values[0] > 0 else None,
        'inverse_residual_2norm': float(np.linalg.norm(covariance @ precision - np.eye(len(precision)), ord=2)),
        'inverse_backward_error': float(np.linalg.norm(covariance @ precision - np.eye(len(precision)), ord=2)
            / (cnorm * norm + 1.)),
        'independent_inverse_max_abs_error': float(np.max(np.abs(precision - inverse))) if inverse is not None else None}


def test_exact_fit_before_after_localization(request):
    runs = {'before_gpu': 4606, 'after_gpu': 4607, 'before_cpu': 4608, 'after_cpu': 4609}
    records, analysis = {}, {}
    for label, number in runs.items():
        directory = RAW / f'run-{number:05d}'
        assert json.loads((directory / 'run.json').read_text())['state'] == 'passed'
        record = json.loads((directory / 'dz5-exact-saved-fit.json').read_text())
        assert record['arm'] == label.split('_')[0] and record['device'].lower() == label.split('_')[1]
        assert record['trace_count'] == 1
        records[label] = record
        fit = record['raw']['fit']
        fields = fit['fits']
        matrices = []
        for family, (precisions, covariances) in enumerate(zip(fields['precision'], fields['covariance'], strict=True)):
            for replicate, (precision, covariance) in enumerate(zip(precisions, covariances, strict=True)):
                matrices.append({'family': ('dense', 'factor_1', 'factor_2')[family],
                    'replicate': replicate, 'status': fields['status'][family][replicate],
                    'flags': fields['flags'][family][replicate], 'anchors': fields['anchors'][family][replicate],
                    'factor_metrics': fields['factor_metrics'][family][replicate],
                    'invalid_covariance_evaluations': fields['invalid_covariance_evaluations'][family][replicate],
                    **matrix_record(precision, covariance)})
        analysis[label] = {'fit_error_code': record['fit_error_code'],
            'fit_numerically_usable': record['fit_numerically_usable'],
            'one_stability': fit['one_stability'], 'family_stability': fit['selection']['stability'],
            'matrices': matrices, 'host_peak_rss_bytes': record['host_peak_rss_bytes'],
            'allocator': record['allocator'], 'elapsed_seconds': record['elapsed_seconds']}
    assert all(record['operand_sha256'] == records['before_gpu']['operand_sha256'] for record in records.values())
    comparisons = {}
    for device in ('cpu', 'gpu'):
        before, after = records['before_' + device], records['after_' + device]
        comparisons[device] = {'same_error_code': before['fit_error_code'] == after['fit_error_code'],
            'raw_exactly_equal': before['raw'] == after['raw'],
            'raw_differences': differences(after['raw'], before['raw']),
            'public_differences': differences(after.get('result'), before.get('result'))}
    save(request, 'dz5-exact-fit-localization.json', {
        'schema': 'filter_dz5_exact_fit_conditioning_localization.v1', 'source_runs': runs,
        'role': 'independent_saved_matrix_diagnostics_not_symmetry_waiver',
        'identical_operand_sha256': records['before_gpu']['operand_sha256'],
        'analysis': analysis, 'before_after_comparisons': comparisons,
        'source_hashes': {str(RAW / f'run-{number:05d}/dz5-exact-saved-fit.json'):
            hashlib.sha256((RAW / f'run-{number:05d}/dz5-exact-saved-fit.json').read_bytes()).hexdigest()
            for number in runs.values()},
        'nonclaims': ['Good normwise residual cannot silently replace a failed symmetry gate; invalid fits remain invalid.']})
