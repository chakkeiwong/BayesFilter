"""Post-run descriptive analysis of E1 records; standard-library only."""

import argparse
import hashlib
import json
import math
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path


def same_records(left, right):
    if isinstance(left, dict):
        assert isinstance(right, dict) and left.keys() == right.keys()
        for key in left:
            if key != 'artifact_hash':
                same_records(left[key], right[key])
    elif isinstance(left, list):
        assert isinstance(right, list) and len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            same_records(a, b)
    elif isinstance(left, float):
        assert isinstance(right, (int, float)) and math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-10)
    else:
        assert left == right


def analyze(first, last, root, devices=('CPU', 'GPU')):
    grouped, manifests, identity = {}, [], None
    for number in range(first, last + 1):
        directory = root / f'run-{number:05d}'
        path = directory / 'geometry-active-memory.json'
        if not path.is_file():
            continue
        run = json.loads((directory / 'run.json').read_text())
        result = json.loads(path.read_text())
        if run['device'] not in devices:
            continue
        assert run['state'] == 'passed'
        suite = ET.parse(directory / 'junit.xml').getroot().find('testsuite')
        assert suite is not None and int(suite.get('tests')) == 1
        assert int(suite.get('failures')) == int(suite.get('errors')) == int(suite.get('skipped')) == 0
        assert result['role'] == 'descriptive_active_count_fit_cost'
        assert result['baseline'] == 'ca920bac5_compact_xla' and result['numerical_authority'] == '3582b4ac'
        assert len(result['samples']) == len(result['changed_samples']) == 20
        assert result['trace_count'] == 1 and result['jit_compile'] == (result['arm'] != 'graph')
        provenance = [json.loads(line) for line in (directory / 'process.log').read_text().splitlines()
            if line.startswith('{"tensorflow_version":')]
        assert len(provenance) == 1
        provenance = provenance[0]
        device = 'GPU' if result['gpu'] else 'CPU'
        assert run['device'] == device
        visible = run['environment']['CUDA_VISIBLE_DEVICES']
        assert visible == provenance['cuda_visible_devices']
        policy = provenance['gpu_memory_policy']
        assert policy['mode'] == 'memory_growth' and policy['all_physical_devices_memory_growth']
        assert policy['configured_before_logical_device_initialization']
        uuid = run.get('gpu_uuid')
        if device == 'GPU':
            assert uuid and visible == uuid
            assert provenance['trust_basis'] == 'owner_designated_managed_session_visible_gpu_trusted'
            assert run['gpu_performance_preflight_uncontended'] is True, 'Shared-device timing is not comparable'
            assert len(run['gpu_preflight']) >= 2
            assert all(row['selected_uuid'] == uuid and row['performance_preflight_uncontended'] is True
                for row in run['gpu_preflight'][-2:])
        else:
            assert visible == '-1' and provenance['trust_basis'] == 'explicit_cpu_reference'
        current = (run['source_sha256'], result['compact_input_sha256'], result['changed_compact_input_sha256'],
            provenance['tensorflow_version'], provenance['tf32_enabled'],
            result['original_source_sha256'], result['baseline_source_sha256'])
        if identity is None:
            identity = current
        assert identity == current, 'Unmatched source, inputs or TensorFlow settings'
        for actual, original in (('result', 'original_result'), ('changed_result', 'original_changed_result'),
            ('payload', 'original_payload'), ('changed_payload', 'original_changed_payload')):
            same_records(result[actual], result[original])
        snapshots = [*result['stages'].values(), result['cold']['memory'], result['changed_cost']['memory'],
            *(row['memory'] for row in result['samples']), *(row['memory'] for row in result['changed_samples'])]
        stages_mib = {name: row['rollup']['Rss'] / 1024**2 for name, row in result['stages'].items()}
        summary = {'run': number, 'uuid': uuid, 'path': str(path),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'cold_seconds': result['build_seconds'] + result['trace_seconds'] + result['cold']['seconds'],
            'warm_ms': statistics.median(row['seconds'] for row in result['samples']) * 1000,
            'native_warm_ms': statistics.median(row['native_seconds'] for row in result['samples']) * 1000,
            'report_warm_ms': statistics.median(row['report_seconds'] for row in result['samples']) * 1000,
            'changed_count_seconds': result['changed_build_seconds'] + result['changed_cost']['seconds'],
            'observed_rss_mib': max(row['status']['VmRSS'] for row in snapshots) / 1024**2,
            'stages_rss_mib': stages_mib,
            'gpu_allocator_peak_bytes': max(row['gpu']['peak'] for row in snapshots) if result['gpu'] else None,
            'warm_residency_delta_mib': stages_mib['changing_count_reuse'] - stages_mib['changed_count'],
            'graph_nodes': result['graph_nodes'], 'hlo_bytes': result['hlo_bytes'],
            'program_instances': result['program_instances']}
        arms = grouped.setdefault((device, result['capacity']), {})
        assert result['arm'] not in arms
        arms[result['arm']] = summary
        manifests.append({'run': number, 'manifest_sha256': hashlib.sha256((directory / 'run.json').read_bytes()).hexdigest(),
            'original_source_sha256': result['original_source_sha256'],
            'baseline_source_sha256': result['baseline_source_sha256']})
    assert set(grouped) == {(device, capacity) for device in devices for capacity in (16, 24)}
    uuids = {row['uuid'] for (device, _), arms in grouped.items() if device == 'GPU' for row in arms.values()}
    assert len(uuids) == (1 if 'GPU' in devices else 0), 'Mixed physical GPU UUIDs'
    comparisons = []
    for (device, capacity), arms in grouped.items():
        assert set(arms) == {'compact', 'graph', 'xla'}
        before, after = arms['compact'], arms['xla']
        warm_ratio = after['warm_ms'] / before['warm_ms']
        cold_ratio = after['cold_seconds'] / before['cold_seconds']
        extra = after['observed_rss_mib'] - before['observed_rss_mib']
        device_ratio = (after['gpu_allocator_peak_bytes'] / before['gpu_allocator_peak_bytes']) if device == 'GPU' else None
        triggers = []
        if warm_ratio > 1.2:
            triggers.append('warm_above_20_percent')
        if cold_ratio > 2:
            triggers.append('cold_above_2x')
        if extra > 256 or after['observed_rss_mib'] > 2 * before['observed_rss_mib']:
            triggers.append('host_rss_above_256MiB_or_2x')
        if device_ratio is not None and device_ratio > 2:
            triggers.append('gpu_allocator_peak_above_2x')
        comparisons.append({'device': device, 'capacity': capacity, 'arms': arms,
            'warm_ratio': warm_ratio, 'cold_ratio': cold_ratio, 'extra_rss_mib': extra,
            'gpu_allocator_ratio': device_ratio, 'investigation_triggers': triggers})
    return {'schema': 'filter_geometry_active_costs.v1', 'devices': devices, 'comparisons': comparisons,
        'manifest_provenance': manifests, 'source_sha256': identity[0],
        'nonclaims': ['One fresh process per arm/extent/device; descriptive only, no timing ranking.',
            'Forty warm calls cannot establish native-cache release or leak freedom.',
            'Graph and XLA retain their declared distinct SVD implementations; not an identical-graph ablation.',
            'This measures the fit suffix, not the public or iterative initializer.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--first-run', type=int, required=True)
    parser.add_argument('--last-run', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--devices', nargs='+', choices=('CPU', 'GPU'), default=['CPU', 'GPU'])
    args = parser.parse_args()
    result = analyze(args.first_run, args.last_run, Path(__file__).resolve().parent, args.devices)
    result['analysis_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open('x') as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write('\n')
    print(json.dumps([{key: row[key] for key in ('device', 'capacity', 'warm_ratio', 'cold_ratio',
        'extra_rss_mib', 'gpu_allocator_ratio', 'investigation_triggers')} for row in result['comparisons']], indent=2))
