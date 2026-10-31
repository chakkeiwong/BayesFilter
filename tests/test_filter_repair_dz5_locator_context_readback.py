"""Independent saved first-objective diagnostic readback; no admission waiver."""

import hashlib
import json

import numpy as np

from tests.test_filter_repair_dz5_initializer_consumer import RAW
from tests.test_filter_repair_geometry_control import save


def array_comparison(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    if actual.dtype.kind == 'b':
        return {'bitwise_equal': actual.tobytes() == expected.tobytes()}
    error = np.abs(actual - expected)
    spacing = np.spacing(np.maximum(np.abs(actual), np.abs(expected)))
    return {'bitwise_equal': actual.tobytes() == expected.tobytes(),
        'maximum_absolute_error': float(np.max(error)),
        'maximum_error_in_larger_operand_ULPs': float(np.max(error / spacing))}


def check_saved_contexts(request, group_prefix='dz5_locator_first_context',
        output_name='dz5-first-objective-context-readback.json', maximum_callbacks=1,
        schema='filter_dz5_first_objective_context.v1'):
    runs = {}
    for path in sorted(RAW.glob('run-*/run.json')):
        run = json.loads(path.read_text())
        for arm in ('original', 'candidate'):
            if run['key'][1] == f'{group_prefix}_{arm}_cpu' and run['state'] == 'passed':
                assert arm not in runs, 'Repeated successful context needs explicit disposition'
                runs[arm] = path.parent
    assert set(runs) == {'original', 'candidate'}
    records, arrays, controls = {}, {}, {}
    for arm, old_number in (('original', 4584), ('candidate', 4585)):
        directory = runs[arm]
        report = json.loads((directory / 'dz5-snapshot-import.json').read_text())
        historical = RAW / f'run-{old_number:05d}'
        old_report = json.loads((historical / 'dz5-snapshot-import.json').read_text())
        assert report['snapshot_manifest_sha256'] == old_report['snapshot_manifest_sha256']
        assert 1 <= report['optimizer_callback_batches'] <= maximum_callbacks
        assert report['trace_count'] == 1 and not report['host_callbacks']
        assert report['context_hlo_sha256'] == hashlib.sha256(
            (directory / 'first-objective-context.hlo.txt').read_bytes()).hexdigest()
        with np.load(directory / 'locator-callbacks.npz', allow_pickle=False) as current:
            arrays[arm] = {key: current[key][:2].copy() for key in current.files}
        with np.load(historical / 'locator-callbacks.npz', allow_pickle=False) as previous:
            reference = {key: previous[key][:2].copy() for key in previous.files}
        assert set(arrays[arm]) == set(reference) == {'positions', 'values', 'scores', 'valid'}
        positions_equal = arrays[arm]['positions'].tobytes() == reference['positions'].tobytes()
        controls[arm] = positions_equal
        records[arm] = {'run': directory.name, 'historical_run': old_number,
            'same_first_two_positions': positions_equal,
            'comparisons': {key: array_comparison(arrays[arm][key], reference[key])
                for key in reference}}
    cross = {key: array_comparison(arrays['candidate'][key], arrays['original'][key])
        for key in arrays['original']}
    reproduces = all(row['comparisons']['scores']['bitwise_equal']
        for row in records.values()) and not cross['scores']['bitwise_equal']
    result = {'schema': schema,
        'historical_controls': records, 'cross_context_comparison': cross,
        'reproduces_historical_first_score_difference': reproduces,
        'next': 'Specify bounded graph bisection' if reproduces else
            'Smaller dispatch changes context; no repair or precise compiler cause established',
        'nonclaims': ['No full optimizer, convergence, numerical waiver or runtime repair.']}
    save(request, output_name, result)
    assert all(controls.values()), 'Changed input bytes invalidate first-score localization'


def test_saved_first_objective_contexts(request):
    check_saved_contexts(request)
