"""Diagnostic matched SVD costs; reject invalid numerical or device comparisons."""

import argparse
import hashlib
import json
import math
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path

from filter_repair_cost_provenance import (
    require_same_physical_gpu,
    validate_cost_device,
)

ARMS = ('prior_graph', 'prior_xla', 'after_graph', 'after_xla')


def numerical_status(result):
    """Recompute the fixed numerical screen; do not trust a caller pass flag."""
    passed = True
    assert len(result['comparisons']) == 2
    for comparison in result['comparisons']:
        actual, expected = comparison['actual'], comparison['independent_reference']

        def equal(left, right):
            if isinstance(right, list):
                return (isinstance(left, list) and len(left) == len(right)
                    and all(equal(a, b) for a, b in zip(left, right, strict=True)))
            return (isinstance(left, (float, int)) and math.isfinite(left)
                and math.isfinite(right) and abs(left - right) <= 1e-10 + 1e-10 * abs(right))

        healthy = (all(equal(actual[name], expected[name]) for name in expected)
            and actual['on_support'] == [True] and actual['rank'] == [3])
        assert healthy is comparison['passed'], 'Numerical pass flag disagrees with raw values'
        passed = passed and healthy
    assert passed is result['numerical_passed'], 'Aggregate numerical flag is inconsistent'
    return passed


def comparison_metrics(left, right):
    if not left['numerical_passed'] or not right['numerical_passed']:
        return {'eligible': False, 'reason': 'At least one arm fails independent Kalman equivalence; no speed ranking.'}
    warm = right['median_warm_ms'] / left['median_warm_ms']
    cold = right['median_cold_seconds'] / left['median_cold_seconds']
    host = right['max_observed_rss_mib'] - left['max_observed_rss_mib']
    allocator = (right['max_gpu_allocator_peak_bytes'] / left['max_gpu_allocator_peak_bytes']
        if left['max_gpu_allocator_peak_bytes'] else None)
    triggers = []
    if warm > 1.2:
        triggers.append('warm_above_20_percent')
    if cold > 2:
        triggers.append('cold_above_2x')
    if host > 256 or right['max_observed_rss_mib'] > 2 * left['max_observed_rss_mib']:
        triggers.append('host_rss_above_256MiB_or_2x')
    if allocator is not None and allocator > 2:
        triggers.append('gpu_allocator_peak_above_2x')
    return {'eligible': True, 'warm_ratio': warm, 'cold_ratio': cold,
        'extra_observed_rss_mib': host, 'gpu_allocator_ratio': allocator,
        'investigation_triggers': triggers}


def analyze(root, first, last, devices, repeats=3):
    groups, identities, uuids, source = {}, {}, [], None
    for number in range(first, last + 1):
        directory = root / f'run-{number:05d}'
        path = directory / 'srukf-svd-cost.json'
        if not path.is_file():
            continue
        run = json.loads((directory / 'run.json').read_text())
        if run['device'] not in devices:
            continue
        result = json.loads(path.read_text())
        assert result['schema'] == 'filter_srukf_svd_cost.v1' and result['baseline'] == 'f4ea46dde'
        cases = list(ET.parse(directory / 'junit.xml').getroot().iter('testcase'))
        assert len(cases) == 1 and not any(case.findall(tag) for case in cases for tag in ('failure', 'error', 'skipped'))
        provenance, = [json.loads(line) for line in (directory / 'process.log').read_text().splitlines()
            if line.startswith('{"tensorflow_version":')]
        uuid = validate_cost_device(run, provenance, result['gpu_process_observation'])
        uuids.append(uuid)
        device, horizon, arm, repeat = run['device'], result['horizon'], result['arm'], run['key'][6]
        assert arm in ARMS and horizon in (1, 3) and repeat in range(repeats)
        assert result['gpu'] is (device == 'GPU') and result['jit_compile'] is arm.endswith('xla')
        assert len(result['samples']) == 20 and result['program']['trace_count'] == 1
        assert not result['jit_compile'] or result['program']['hlo_unchanged']
        valid = numerical_status(result)
        assert valid or arm == 'prior_xla', 'A required accurate arm failed numerical qualification'
        identity = (run['source_sha256'], provenance['tensorflow_version'], provenance['tf32_enabled'],
            {key: value for key, value in run['environment'].items() if key != 'CUDA_VISIBLE_DEVICES'},
            result['original_source_sha256'])
        identities.setdefault(device, identity)
        assert identities[device] == identity, 'Unmatched source or environment'
        source = run['source_sha256'] if source is None else source
        assert source == run['source_sha256'], 'Unmatched source across devices'
        rows = groups.setdefault((device, horizon, arm), {})
        assert repeat not in rows, 'Duplicate repeat: select a single frozen cohort'
        snapshots = [*result['snapshots'].values(), result['cold']['memory'], result['changed']['memory'],
            result['returned']['memory'], *(sample['memory'] for sample in result['samples'])]
        warm_rss = [sample['memory']['rollup']['Rss'] / 1024**2 for sample in result['samples']]
        rows[repeat] = {'run': number, 'uuid': uuid, 'numerical_passed': valid,
            'result_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'manifest_sha256': hashlib.sha256((directory / 'run.json').read_bytes()).hexdigest(),
            'cold_seconds': result['construction_trace_seconds'] + result['cold']['seconds'],
            'warm_ms': statistics.median(sample['seconds'] for sample in result['samples']) * 1000,
            'observed_rss_mib': max(row['rollup']['Rss'] for row in snapshots) / 1024**2,
            'observed_pss_mib': max(row['rollup']['Pss'] for row in snapshots) / 1024**2,
            'host_hwm_mib': max(row['status']['VmHWM'] for row in snapshots) / 1024**2,
            'warm_rss_mib': warm_rss, 'warm_growth_mib': warm_rss[-1] - warm_rss[0],
            'gpu_allocator_peak_bytes': max(row['gpu']['peak'] for row in snapshots) if device == 'GPU' else None,
            'program': result['program']}
    require_same_physical_gpu(uuids)
    assert set(groups) == {(device, horizon, arm) for device in devices for horizon in (1, 3) for arm in ARMS}, 'Incomplete matrix'
    assert all(set(rows) == set(range(repeats)) for rows in groups.values()), 'Incomplete fresh-process repeats'
    comparisons = []
    for device in devices:
        for horizon in (1, 3):
            arms = {}
            for arm in ARMS:
                rows = list(groups[device, horizon, arm].values())
                arms[arm] = {'repeats': rows, 'numerical_passed': all(row['numerical_passed'] for row in rows),
                    'median_cold_seconds': statistics.median(row['cold_seconds'] for row in rows),
                    'median_warm_ms': statistics.median(row['warm_ms'] for row in rows),
                    'max_observed_rss_mib': max(row['observed_rss_mib'] for row in rows),
                    'max_gpu_allocator_peak_bytes': max(row['gpu_allocator_peak_bytes'] for row in rows) if device == 'GPU' else None,
                    'max_warm_growth_mib': max(row['warm_growth_mib'] for row in rows)}
            pairs = {}
            for left, right in (('prior_xla', 'after_xla'), ('prior_graph', 'after_graph'),
                    ('after_graph', 'after_xla'), ('prior_graph', 'after_xla')):
                pairs[left + '_to_' + right] = comparison_metrics(arms[left], arms[right])
            comparisons.append({'device': device, 'horizon': horizon, 'arms': arms, 'pairs': pairs})
    return {'schema': 'filter_srukf_svd_cost_comparison.v1', 'comparisons': comparisons,
        'source_sha256': source, 'fresh_process_repeats': repeats,
        'role': 'pilot_only' if repeats == 1 else 'descriptive_matched_cost',
        'nonclaims': ['No speed ranking for inaccurate arms or statistical performance superiority.',
            'Short repeated calls do not prove native-memory eviction or many-signature capacity.',
            'This linear SRUKF fixture does not qualify all Kalman/UKF/analytical-gradient endpoints.']}


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
    print(json.dumps([{key: row[key] for key in ('device', 'horizon', 'pairs')}
        for row in report['comparisons']], indent=2))
