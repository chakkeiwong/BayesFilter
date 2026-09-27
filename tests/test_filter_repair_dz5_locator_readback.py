"""Independent saved-array comparison of the original/current CDF trajectories."""

import hashlib
import json
from pathlib import Path

import numpy as np

from tests.test_filter_repair_dz5_initializer_fit_localization import differences
from tests.test_filter_repair_geometry_control import save

RAW = Path('/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917')


def test_first_callback_divergence_is_preserved(request):
    records, callbacks, manifests = {}, {}, {}
    for arm, number in (('original', 4584), ('candidate', 4585)):
        directory = RAW / f'run-{number:05d}'
        run = json.loads((directory / 'run.json').read_text())
        report = json.loads((directory / 'dz5-snapshot-import.json').read_text())
        assert run['state'] == 'passed' and report['passed'] and report['arm'] == arm
        assert report['trace_count'] == 1 and not report['host_callbacks']
        path = directory / 'locator-callbacks.npz'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == report['callbacks_sha256']
        with np.load(path, allow_pickle=False) as archive:
            callbacks[arm] = {key: archive[key].copy() for key in archive.files}
        count = report['callback_rows']
        assert callbacks[arm]['positions'].shape == callbacks[arm]['scores'].shape == (count, 23)
        assert callbacks[arm]['values'].shape == callbacks[arm]['valid'].shape == (count,)
        assert count == report['locator_result']['physical_target_rows']
        records[arm] = report['locator_result']
        manifests[arm] = report
    assert manifests['original']['settings'] == manifests['candidate']['settings']
    assert manifests['original']['snapshot_manifest_sha256'] == manifests['candidate']['snapshot_manifest_sha256']
    assert manifests['original']['start_replay']['point'] == manifests['candidate']['start_replay']['point']
    common = min(row['callback_rows'] for row in manifests.values())
    first_bit, first_bound = {}, {}
    for key in callbacks['original']:
        left, right = callbacks['original'][key][:common], callbacks['candidate'][key][:common]
        axis = tuple(range(1, left.ndim))
        # Compare representations, including signed zeros and NaN payloads.
        equal = np.all(left.view(np.uint8).reshape(left.shape + (left.itemsize,))
            == right.view(np.uint8).reshape(right.shape + (right.itemsize,)), axis=-1)
        same = np.all(equal, axis=axis) if axis else equal
        changed = np.flatnonzero(~same)
        first_bit[key] = int(changed[0]) if len(changed) else None
        close = np.isclose(left, right, atol=1e-10, rtol=1e-10, equal_nan=True)
        within = np.all(close, axis=axis) if axis else close
        changed = np.flatnonzero(~within)
        first_bound[key] = int(changed[0]) if len(changed) else None
    first_position = first_bit['positions']
    # After positions differ, equal callback indices no longer denote equal inputs.
    prefix = common if first_position is None else first_position
    equal_input_outputs = {}
    for key in ('values', 'scores', 'valid'):
        left, right = callbacks['original'][key][:prefix], callbacks['candidate'][key][:prefix]
        equal_input_outputs[key] = {'rows': prefix, 'exact': bool(np.array_equal(left, right, equal_nan=True)),
            'within_target_bounds': bool(np.allclose(left, right, atol=1e-8, rtol=1e-7, equal_nan=True)),
            'max_absolute_finite_error': float(np.max(np.abs(left.astype(float) - right.astype(float)))) if prefix else None}
    indices = sorted({0, *[index for index in (*first_bit.values(), *first_bound.values()) if index is not None]})
    evidence = []
    for index in indices:
        row = {'callback_index': index}
        for arm, arrays in callbacks.items():
            row[arm] = {key: values[index].tolist() for key, values in arrays.items()}
        evidence.append(row)
    result = {'schema': 'filter_dz5_locator_first_divergence.v1', 'source_runs': [4584, 4585],
        'role': 'trajectory_localization_not_numerical_admission',
        'record_differences': differences(records['candidate'], records['original']),
        'callback_counts': {key: report['callback_rows'] for key, report in manifests.items()},
        'first_bit_difference': first_bit, 'first_strict_bound_difference': first_bound,
        'identical_input_prefix_rows': prefix, 'identical_input_output_comparisons': equal_input_outputs,
        'first_difference_rows': evidence,
        'start_replays': {key: report['start_replay'] for key, report in manifests.items()},
        'nonclaims': ['A trajectory difference alone does not identify a wrong value/gradient or qualify full equivalence.']}
    save(request, 'dz5-locator-first-divergence.json', result)
    assert evidence and prefix >= 1


def test_original_operand_binding_localization(request):
    traces, reports = {}, {}
    for arm, number in (('original', 4584), ('candidate', 4585), ('original_operands', 4594)):
        directory = RAW / f'run-{number:05d}'
        report = json.loads((directory / 'dz5-snapshot-import.json').read_text())
        assert json.loads((directory / 'run.json').read_text())['state'] == 'passed'
        assert report['passed'] and report['arm'] == arm
        assert report['trace_count'] == 1 and not report['host_callbacks']
        path = directory / 'locator-callbacks.npz'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == report['callbacks_sha256']
        with np.load(path, allow_pickle=False) as archive:
            traces[arm] = {key: archive[key].copy() for key in archive.files}
        reports[arm] = report
    diagnostic = reports['original_operands']
    assert diagnostic['original_operand_binding_diagnostic']
    assert diagnostic['original_locator_sha256'] == reports['original']['original_locator_sha256']
    assert diagnostic['executed_original_source_sha256'] == hashlib.sha256(
        (RAW / 'run-04594/original-batched-local-center.py').read_bytes()).hexdigest()
    comparisons = {}
    for arm in ('original', 'candidate'):
        assert reports[arm]['settings'] == diagnostic['settings']
        assert reports[arm]['snapshot_manifest_sha256'] == diagnostic['snapshot_manifest_sha256']
        exact = {key: traces[arm][key].shape == traces['original_operands'][key].shape
            and traces[arm][key].tobytes() == traces['original_operands'][key].tobytes()
            for key in traces[arm]}
        comparisons[arm] = {'callback_arrays_bitwise_equal': exact,
            'complete_records_exactly_equal': reports[arm]['locator_result'] == diagnostic['locator_result'],
            'complete_record_differences': differences(diagnostic['locator_result'], reports[arm]['locator_result'])}
    fixed = json.loads((RAW / 'run-04593/dz5-locator-fixed-outputs.json').read_text())
    assert not fixed['record_differences']
    assert fixed['first_exact_position_difference_between_controllers'] is None
    with np.load(RAW / 'run-04593/fixed-output-locator-positions.npz', allow_pickle=False) as archive:
        assert all(archive[arm].tobytes() == traces['original']['positions'].tobytes()
            for arm in ('original', 'candidate'))
    result = {'schema': 'filter_dz5_original_operand_binding_localization.v1',
        'role': 'compilation_context_diagnostic_not_replacement_baseline',
        'source_runs': [4584, 4585, 4593, 4594],
        'callback_counts': {key: row['callback_rows'] for key, row in reports.items()},
        'comparisons': comparisons,
        'candidate_reproduced_exactly': all(comparisons['candidate']['callback_arrays_bitwise_equal'].values())
            and comparisons['candidate']['complete_records_exactly_equal'],
        'source_hashes': {str(RAW / f'run-{number:05d}/dz5-snapshot-import.json'):
            hashlib.sha256((RAW / f'run-{number:05d}/dz5-snapshot-import.json').read_bytes()).hexdigest()
            for number in (4584, 4585, 4594)},
        'nonclaims': ['Unmodified original mismatches remain; no optimizer convergence, current-source consumer or scientific admission.']}
    save(request, 'dz5-locator-operand-binding-localization.json', result)
