"""Independent diagnostic analysis of saved SVD-consumer costs and numerics."""

import argparse
import hashlib
import json
import math
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from filter_repair_cost_provenance import (
    require_same_physical_gpu,
    validate_cost_device,
)

ARMS = ('prior_graph', 'prior_xla', 'after_graph', 'after_xla')
TOLERANCE = 1e-10


def expected_precision(dimension):
    assert dimension in (3, 5)
    return np.diag([1.3, 1.7, .8] if dimension == 3 else [1.3, 1.7, 1.1, 1.5, .8])


def check_values(actual, offsets, precision):
    """Independent exact-quadratic reference, with NumPy used only for analysis."""
    dimension = len(precision)
    offsets = np.asarray(offsets)
    assert offsets.shape == (2 * dimension + 1, dimension)
    assert np.isfinite(offsets).all()
    singular = np.linalg.svd(offsets, compute_uv=False)
    assert singular[-1] > 0
    condition = singular[0] / singular[-1]
    rank = dimension * (dimension + 1) // 2

    def near(left, right):
        left, right = np.asarray(left), np.asarray(right)
        return bool(left.shape == right.shape and np.isfinite(left).all()
            and np.allclose(left, right, atol=TOLERANCE, rtol=TOLERANCE))

    return {'cod_condition': near(actual['condition'], condition),
        'dense_condition': near(actual['dense']['design_condition'], condition),
        'dense_rank': actual['dense']['design_rank'] == dimension,
        'dense_precision': near(actual['dense']['raw_precision'], precision),
        'block_ranks': actual['block_report'][:2] == [rank, dimension],
        'block_status': actual['block_status'] == 0,
        'block_precision': near(actual['block_precision'], precision),
        'sequential_rank': actual['sequential']['rank'] == rank,
        'sequential_status': actual['sequential']['status'] == 1,
        'sequential_precision': near(actual['sequential']['projected_precision_z'], precision),
        'quadratic_rank': actual['quadratic']['score_design_rank'] == dimension,
        'quadratic_resolved': actual['quadratic']['design_resolved'] is True,
        'quadratic_precision': near(actual['quadratic']['precision'], precision)}


def numerical_status(result):
    dimension = result['dimension']
    precision = expected_precision(dimension)
    assert result['expected_precision'] == precision.tolist()
    assert result['changed_precision'] == (precision * 1.05).tolist()
    initial_offsets = np.asarray(result['initial_offsets'])
    assert np.array_equal(result['changed_offsets'], initial_offsets * 1.1)
    evaluations = [(result['initial'], initial_offsets, precision, result['initial_checks']),
        (result['changed_values'], result['changed_offsets'], precision * 1.05, result['changed_checks'])]
    required = {(spectrum, magnitude) for spectrum in ('separated', 'near_tied')
        for magnitude in (1., 1e-4, 1e-10, 1e4)}
    assert len(result['scale_records']) == len(required)
    found = set()
    for row in result['scale_records']:
        identity = (row['spectrum'], row['magnitude'])
        assert identity in required and identity not in found
        found.add(identity)
        offsets = np.asarray(row['offsets'])
        singular = np.linalg.svd(offsets / row['magnitude'], compute_uv=False)
        spectrum = ([3., 1.5, .7] if dimension == 3 else [3., 2.3, 1.5, 1.1, .7])
        if row['spectrum'] == 'near_tied':
            spectrum = 1. + np.arange(dimension // 2, -dimension // 2, -1) * 1e-8
        assert np.allclose(singular, spectrum, atol=1e-12, rtol=1e-12)
        if identity == ('separated', 1.):
            assert np.array_equal(offsets, initial_offsets)
        evaluations.append((row['actual'], offsets, precision, row['checks']))
    statuses = []
    for actual, offsets, wanted, recorded in evaluations:
        checks = check_values(actual, offsets, wanted)
        assert checks == recorded, 'Numerical flags disagree with independently checked raw values'
        statuses.append(all(checks.values()))
    status = all(statuses)
    assert result['numerical_passed'] is status, 'Aggregate numerical flag disagrees'
    return status


def validate_execution(result):
    assert result['schema'] == 'filter_remaining_svd_cost.v1'
    assert result['baseline'] == '9696666e4'
    assert result['arm'] in ARMS and result['dimension'] in (3, 5)
    assert result['jit_compile'] is result['arm'].endswith('xla')
    assert len(result['samples']) == 20 and not result['replay_failures']
    assert result['initial'] == result['returned_values'], 'Raw replay changed'
    assert result['program']['trace_count'] == 1 and not result['program']['host_callbacks']
    assert not result['jit_compile'] or result['program']['hlo_unchanged']
    assert set(result['released']) == {'program', 'graph', 'owner', 'callback', 'resource'}
    assert all(value is True for value in result['released'].values()), 'Caller ownership retained'
    assert result['primitive']['trace_count'] == 1 and result['primitive']['capture_count'] == 0
    assert result['in_run_unshared'] is True
    assert math.isfinite(result['construction_trace_seconds']) and result['construction_trace_seconds'] > 0
    for call in (result['cold'], result['changed'], result['returned'], *result['samples']):
        assert math.isfinite(call['seconds']) and call['seconds'] > 0
    return numerical_status(result)


def comparison_metrics(left, right):
    if not left['numerical_passed'] or not right['numerical_passed']:
        return {'eligible': False, 'reason': 'Independent numerical screen failed; no speed ratio.'}
    pairs = {'warm': 'median_warm_ms', 'cold': 'median_cold_seconds',
        'host_rss': 'max_observed_rss_mib', 'gpu_allocator': 'max_gpu_allocator_peak_bytes',
        'gpu_reservation': 'max_process_gpu_reservation_bytes'}
    ratios = {name: right[field] / left[field] if left[field] else None
        for name, field in pairs.items()}
    return {'eligible': True, 'ratios': ratios,
        'extra_observed_rss_mib': right['max_observed_rss_mib'] - left['max_observed_rss_mib'],
        'investigation_triggers': [name + '_above_20_percent' for name, ratio in ratios.items()
            if ratio is not None and ratio > 1.2]}


def analyze(root, first, last, devices, repeats=3):
    groups, identities, uuids, source = {}, {}, [], None
    for number in range(first, last + 1):
        directory = root / f'run-{number:05d}'
        path = directory / 'remaining-svd-cost.json'
        if not path.is_file():
            continue
        run = json.loads((directory / 'run.json').read_text())
        if run['device'] not in devices:
            continue
        result = json.loads(path.read_text())
        cases = list(ET.parse(directory / 'junit.xml').getroot().iter('testcase'))
        assert len(cases) == 1 and not any(case.findall(tag) for case in cases for tag in ('failure', 'error', 'skipped'))
        provenance, = [json.loads(line) for line in (directory / 'process.log').read_text().splitlines()
            if line.startswith('{"tensorflow_version":')]
        uuid = validate_cost_device(run, provenance, result['gpu_process_observation'])
        uuids.append(uuid)
        device, dimension, arm, repeat = run['device'], result['dimension'], result['arm'], run['key'][6]
        assert result['gpu'] is (device == 'GPU') and repeat in range(repeats)
        assert run['key'][1] == f'remaining_svd_cost_{arm}_{dimension}_{device.lower()}'
        valid = validate_execution(result)
        assert valid or arm == 'prior_xla', 'Required accurate arm failed'
        identity = (run['source_sha256'], provenance['tensorflow_version'], provenance['tf32_enabled'],
            {key: value for key, value in run['environment'].items() if key != 'CUDA_VISIBLE_DEVICES'},
            result['original_source_sha256'])
        identities.setdefault(device, identity)
        assert identity == identities[device], 'Unmatched source or environment'
        source = run['source_sha256'] if source is None else source
        assert source == run['source_sha256'], 'Unmatched sources across devices'
        rows = groups.setdefault((device, dimension, arm), {})
        assert repeat not in rows, 'Duplicate repeat; select one frozen cohort'
        snapshots = [*result['snapshots'].values(), result['cold']['memory'], result['changed']['memory'],
            result['returned']['memory'], *(sample['memory'] for sample in result['samples'])]
        warm_rss = [row['memory']['rollup']['Rss'] / 1024**2 for row in result['samples']]
        rows[repeat] = {'run': number, 'uuid': uuid, 'numerical_passed': valid,
            'result_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'manifest_sha256': hashlib.sha256((directory / 'run.json').read_bytes()).hexdigest(),
            'cold_seconds': result['construction_trace_seconds'] + result['cold']['seconds'],
            'warm_ms': statistics.median(row['seconds'] for row in result['samples']) * 1000,
            'observed_rss_mib': max(row['rollup']['Rss'] for row in snapshots) / 1024**2,
            'host_hwm_mib': max(row['status']['VmHWM'] for row in snapshots) / 1024**2,
            'max_mapping_count': max(row['mapping_count'] for row in snapshots),
            'warm_growth_mib': warm_rss[-1] - warm_rss[0],
            'gpu_allocator_peak_bytes': max(row['gpu']['peak'] for row in snapshots) if device == 'GPU' else None,
            'process_gpu_reservation_bytes': max(row['process_gpu_reservation_bytes'] for row in snapshots) if device == 'GPU' else None,
            'released': result['released'], 'primitive': result['primitive'], 'program': result['program']}
    require_same_physical_gpu(uuids)
    assert set(groups) == {(device, dimension, arm) for device in devices for dimension in (3, 5) for arm in ARMS}, 'Incomplete matrix'
    assert all(set(rows) == set(range(repeats)) for rows in groups.values()), 'Incomplete repeats'
    comparisons = []
    for device in devices:
        for dimension in (3, 5):
            arms = {}
            for arm in ARMS:
                rows = list(groups[device, dimension, arm].values())
                arms[arm] = {'repeats': rows, 'numerical_passed': all(row['numerical_passed'] for row in rows),
                    'median_cold_seconds': statistics.median(row['cold_seconds'] for row in rows),
                    'median_warm_ms': statistics.median(row['warm_ms'] for row in rows),
                    'max_observed_rss_mib': max(row['observed_rss_mib'] for row in rows),
                    'max_gpu_allocator_peak_bytes': max(row['gpu_allocator_peak_bytes'] for row in rows) if device == 'GPU' else None,
                    'max_process_gpu_reservation_bytes': max(row['process_gpu_reservation_bytes'] for row in rows) if device == 'GPU' else None,
                    'max_mapping_count': max(row['max_mapping_count'] for row in rows),
                    'max_warm_growth_mib': max(row['warm_growth_mib'] for row in rows)}
            pairs = {left + '_to_' + right: comparison_metrics(arms[left], arms[right])
                for left, right in (('prior_xla', 'after_xla'), ('prior_graph', 'after_graph'),
                    ('after_graph', 'after_xla'), ('prior_graph', 'after_xla'))}
            comparisons.append({'device': device, 'dimension': dimension, 'arms': arms, 'pairs': pairs})
    return {'schema': 'filter_remaining_svd_cost_comparison.v1', 'comparisons': comparisons,
        'source_sha256': source, 'fresh_process_repeats': repeats,
        'role': 'pilot_only' if repeats == 1 else 'descriptive_matched_cost',
        'nonclaims': ['No speed ratio for inaccurate arms or statistical superiority claim.',
            'Reusable shape-only diagnostic graphs are bounded by fresh worker lifetime, not native eviction.',
            'Full public/actual-DZ5 capacity and final source renewal remain separate.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--first-run', type=int, required=True)
    parser.add_argument('--last-run', type=int, required=True)
    parser.add_argument('--devices', nargs='+', choices=('CPU', 'GPU'), default=['CPU', 'GPU'])
    parser.add_argument('--repeats', type=int, choices=(1, 3), default=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.root, args.first_run, args.last_run, args.devices, args.repeats)
    report['analyzer_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps([{key: row[key] for key in ('device', 'dimension', 'pairs')}
        for row in report['comparisons']], indent=2))
