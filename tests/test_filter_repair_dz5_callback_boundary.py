"""Independent readback of the diagnostic CDF callback-boundary experiment."""

import hashlib
import json
from pathlib import Path

import numpy as np

from tests.test_filter_repair_dz5_initializer_fit_localization import differences
from tests.test_filter_repair_geometry_control import save

RAW = Path('/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917')


def load(number, arm, barriers):
    directory = RAW / f'run-{number:05d}'
    report = json.loads((directory / 'dz5-snapshot-import.json').read_text())
    assert json.loads((directory / 'run.json').read_text())['state'] == 'passed'
    assert report['passed'] and report['arm'] == arm
    assert report['trace_count'] == 1 and not report['host_callbacks']
    assert report.get('callback_optimization_barriers', 'none') == barriers
    path = directory / 'locator-callbacks.npz'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == report['callbacks_sha256']
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    count = report['callback_rows']
    assert arrays['positions'].shape == arrays['scores'].shape == (count, 23)
    assert arrays['values'].shape == arrays['valid'].shape == (count,)
    assert np.all(arrays['valid'])
    return report, arrays


def compare(left, right, left_record, right_record):
    common = min(len(left['positions']), len(right['positions']))
    same_positions = np.all(left['positions'][:common].view(np.uint64)
        == right['positions'][:common].view(np.uint64), axis=1)
    changed = np.flatnonzero(~same_positions)
    prefix = int(changed[0]) if len(changed) else common
    outputs = {}
    for key in ('values', 'scores'):
        first, second = left[key][:prefix], right[key][:prefix]
        delta = np.abs(first - second)
        bound = 1e-8 + 1e-7 * np.abs(second)
        outputs[key] = {'identical_position_prefix_rows': prefix,
            'max_absolute_error': float(np.max(delta)) if prefix else None,
            'max_error_over_target_bound': float(np.max(delta / bound)) if prefix else None,
            'within_target_bound': bool(np.all(delta <= bound)) if prefix else None,
            'within_strict_record_bound': bool(np.all(delta <= 1e-10 + 1e-10 * np.abs(second))) if prefix else None}
    return {'callback_arrays_bitwise_equal': {key: left[key].shape == right[key].shape
            and left[key].tobytes() == right[key].tobytes() for key in left},
        'first_exact_position_difference': int(changed[0]) if len(changed) else None,
        'equal_position_output_checks': outputs,
        'complete_records_exactly_equal': left_record == right_record,
        'complete_record_differences': differences(left_record, right_record)}


def test_callback_boundary_diagnosis(request):
    source_runs = {'original': 4584, 'candidate': 4585,
        'original_barrier': 4596, 'candidate_barrier': 4597}
    reports, arrays = {}, {}
    for arm, number in source_runs.items():
        reports[arm], arrays[arm] = load(number, arm,
            'inputs_and_outputs' if arm.endswith('_barrier') else 'none')
    assert len({report['snapshot_manifest_sha256'] for report in reports.values()}) == 1
    assert all(report['settings'] == reports['original']['settings'] for report in reports.values())
    comparisons = {}
    for left, right in (('original_barrier', 'candidate_barrier'),
                        ('original_barrier', 'original'), ('candidate_barrier', 'candidate')):
        comparisons[left + '_vs_' + right] = compare(arrays[left], arrays[right],
            reports[left]['locator_result'], reports[right]['locator_result'])
    matched = comparisons['original_barrier_vs_candidate_barrier']
    result = {'schema': 'filter_dz5_callback_boundary_localization.v1',
        'role': 'harness_only_optimization_barriers_not_runtime_promotion',
        'source_runs': source_runs,
        'callback_counts': {key: row['callback_rows'] for key, row in reports.items()},
        'all_callbacks_valid': True,
        'controller_trajectories_match_bitwise_with_barriers': all(matched['callback_arrays_bitwise_equal'].values())
            and matched['complete_records_exactly_equal'],
        'comparisons': comparisons,
        'descriptive_costs': {key: {field: row.get(field) for field in
            ('locator_seconds', 'host_peak_rss_kib', 'graph_bytes')} for key, row in reports.items()},
        'source_hashes': {str(RAW / f'run-{number:05d}/dz5-snapshot-import.json'):
            hashlib.sha256((RAW / f'run-{number:05d}/dz5-snapshot-import.json').read_bytes()).hexdigest()
            for number in source_runs.values()},
        'nonclaims': ['Matching barriers cannot waive original mismatches or qualify convergence, performance, current-source consumers, HMC or training.']}
    save(request, 'dz5-callback-boundary-localization.json', result)
    assert all(check['within_target_bound'] is True for row in comparisons.values()
        for check in row['equal_position_output_checks'].values())
