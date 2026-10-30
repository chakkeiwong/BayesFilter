"""Adverse checks for the diagnostic SVD-consumer cost analyzer."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from tests.test_filter_repair_campaign import load


def evidence():
    precision = np.diag([1.3, 1.7, .8])
    offsets = np.pad(np.diag([3., 1.5, .7]), [(0, 4), (0, 0)])

    def values(points, wanted):
        return {'condition': float(np.linalg.cond(points)),
            'dense': {'design_condition': float(np.linalg.cond(points)),
                'design_rank': 3, 'raw_precision': wanted.tolist()},
            'block_report': [6., 3.], 'block_status': 0, 'block_precision': wanted.tolist(),
            'sequential': {'rank': 6, 'status': 1, 'projected_precision_z': wanted.tolist()},
            'quadratic': {'score_design_rank': 3, 'design_resolved': True, 'precision': wanted.tolist()}}

    names = ('cod_condition', 'dense_condition', 'dense_rank', 'dense_precision',
        'block_ranks', 'block_status', 'block_precision', 'sequential_rank',
        'sequential_status', 'sequential_precision', 'quadratic_rank',
        'quadratic_resolved', 'quadratic_precision')
    checks = dict.fromkeys(names, True)
    rows = []
    for name, spectrum in (('separated', [3., 1.5, .7]), ('near_tied', [1.+1e-8, 1., 1.-1e-8])):
        for magnitude in (1., 1e-4, 1e-10, 1e4):
            points = np.pad(np.diag(spectrum), [(0, 4), (0, 0)]) * magnitude
            rows.append({'spectrum': name, 'magnitude': magnitude, 'offsets': points.tolist(),
                'actual': values(points, precision), 'checks': checks.copy()})
    initial = values(offsets, precision)
    return {'schema': 'filter_remaining_svd_cost.v1', 'baseline': '9696666e4',
        'dimension': 3, 'arm': 'after_xla', 'jit_compile': True,
        'initial': initial, 'returned_values': copy.deepcopy(initial),
        'changed_values': values(offsets * 1.1, precision * 1.05),
        'expected_precision': precision.tolist(), 'changed_precision': (precision * 1.05).tolist(),
        'initial_offsets': offsets.tolist(), 'changed_offsets': (offsets * 1.1).tolist(),
        'initial_checks': checks.copy(), 'changed_checks': checks.copy(), 'scale_records': rows,
        'numerical_passed': True, 'replay_failures': [], 'in_run_unshared': True,
        'program': {'trace_count': 1, 'host_callbacks': [], 'hlo_unchanged': True},
        'released': dict.fromkeys(('program', 'graph', 'owner', 'callback', 'resource'), True),
        'primitive': {'trace_count': 1, 'capture_count': 0, 'graph_released': False},
        'construction_trace_seconds': 1., 'samples': [{'seconds': .1} for _ in range(20)],
        'cold': {'seconds': 2.}, 'changed': {'seconds': .1}, 'returned': {'seconds': .1}}


@pytest.mark.parametrize('fault', ['precision', 'rank', 'status', 'condition', 'hidden_nan',
    'shape', 'spectrum', 'scale_omission', 'scale_duplicate', 'expected', 'aggregate',
    'returned', 'callback', 'resource', 'graph', 'primitive_capture', 'trace', 'hlo',
    'sharing', 'host_callback', 'sample_count', 'negative_time', 'baseline'])
def test_cost_analysis_rejects_corrupt_or_incomplete_evidence(fault):
    analyzer = load('analyze_filter_repair_remaining_svd_costs')
    result = evidence()
    assert analyzer.validate_execution(result)
    row = result['scale_records'][-1]
    if fault == 'precision':
        row['actual']['dense']['raw_precision'][0][0] += .01
    elif fault == 'rank':
        row['actual']['block_report'][0] = 5
    elif fault == 'status':
        row['actual']['sequential']['status'] = 2
    elif fault == 'condition':
        row['actual']['condition'] *= 1.01
    elif fault == 'hidden_nan':
        row['actual']['quadratic']['precision'][0][0] = float('nan')
    elif fault == 'shape':
        row['actual']['dense']['raw_precision'].pop()
    elif fault == 'spectrum':
        row['offsets'][0][0] *= 1.1
    elif fault == 'scale_omission':
        result['scale_records'].pop()
    elif fault == 'scale_duplicate':
        result['scale_records'][-1] = result['scale_records'][0]
    elif fault == 'expected':
        result['expected_precision'][0][0] += .01
    elif fault == 'aggregate':
        result['numerical_passed'] = False
    elif fault == 'returned':
        result['returned_values']['condition'] += .01
    elif fault in ('callback', 'resource', 'graph'):
        result['released'][fault] = False
    elif fault == 'primitive_capture':
        result['primitive']['capture_count'] = 1
    elif fault == 'trace':
        result['program']['trace_count'] = 2
    elif fault == 'hlo':
        result['program']['hlo_unchanged'] = False
    elif fault == 'sharing':
        result['in_run_unshared'] = False
    elif fault == 'host_callback':
        result['program']['host_callbacks'] = ['PyFunc']
    elif fault == 'sample_count':
        result['samples'].pop()
    elif fault == 'negative_time':
        result['samples'][0]['seconds'] = -1.
    else:
        result['baseline'] = 'unbound'
    with pytest.raises(AssertionError):
        analyzer.validate_execution(result)


def test_invalid_numerics_are_preserved_without_a_speed_ratio():
    analyzer = load('analyze_filter_repair_remaining_svd_costs')
    result = evidence()
    result['arm'] = 'prior_xla'
    result['scale_records'][-1]['actual']['block_status'] = 2
    result['scale_records'][-1]['checks']['block_status'] = False
    result['numerical_passed'] = False
    assert not analyzer.validate_execution(result)
    assert not analyzer.comparison_metrics({'numerical_passed': False}, {'numerical_passed': True})['eligible']


def test_remaining_cost_groups_reject_shared_gpu_preflight(tmp_path, monkeypatch):
    runner = load('run_filter_repair_campaign')
    monkeypatch.setattr(runner, 'OUTPUT', tmp_path)
    for group in runner.TEST_BATCHES['remaining_svd_cost_gpu']:
        args = SimpleNamespace(action='test', device='GPU', group=group,
            gpu_uuid='GPU-device-2', gpu_preflight=[{'performance_preflight_uncontended': False}] * 2)
        with pytest.raises(RuntimeError, match='declined before launch'):
            runner.require_unshared_cost_preflight(args)
