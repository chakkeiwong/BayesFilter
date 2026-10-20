"""Post-run diagnostic analysis of complete public ordered-block cost comparisons."""

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


def same_records(left, right):
    if isinstance(left, dict):
        assert isinstance(right, dict) and left.keys() == right.keys()
        for key in left:
            same_records(left[key], right[key])
    elif isinstance(left, list):
        assert isinstance(right, list) and len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            same_records(a, b)
    elif isinstance(left, float):
        assert isinstance(right, (int, float)) and math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-10)
    else:
        assert left == right


def analyze(root, first, last, devices):
    groups, identities, source_identity, device_identities, uuids = {}, {}, None, {}, []
    for number in range(first, last + 1):
        directory = root / f'run-{number:05d}'
        result_path = directory / 'block-public-memory.json'
        if not result_path.is_file():
            continue
        run = json.loads((directory / 'run.json').read_text())
        if run['device'] not in devices:
            continue
        result = json.loads(result_path.read_text())
        assert result['schema'] == 'filter_block_public_cost.v1'
        assert result['numerical_authority'] == '3582b4ac' and result['mechanism_baseline'] == 'aee3ad043'
        cases = list(ET.parse(directory / 'junit.xml').getroot().iter('testcase'))
        assert len(cases) == 1 and not any(case.findall(tag) for case in cases for tag in ('failure', 'error', 'skipped'))
        provenance, = [json.loads(line) for line in (directory / 'process.log').read_text().splitlines()
                      if line.startswith('{"tensorflow_version":')]
        uuid = validate_cost_device(run, provenance, result['gpu_process_observation'])
        uuids.append(uuid)
        device, dimension, arm, repeat = run['device'], result['dimension'], result['arm'], run['key'][6]
        assert dimension in (3, 5) and arm in ('prior', 'graph', 'xla') and repeat in (0, 1, 2)
        assert result['gpu'] == (device == 'GPU') and result['jit_compile'] == (arm == 'xla')
        assert len(result['samples']) == 3
        if arm != 'prior':
            assert result['trace_count'] == 1
        same_records(result['result'], result['original_result'])
        same_records(result['changed_result'], result['original_changed_result'])
        assert result['result']['public_summary']['completed'] and result['changed_result']['public_summary']['completed']
        assert result['result']['public_summary']['accepted_block_count'] == 2 and result['changed_result']['public_summary']['accepted_block_count'] == 2
        common = (run['source_sha256'], provenance['tensorflow_version'], provenance['tf32_enabled'],
            {key: value for key, value in run['environment'].items() if key != 'CUDA_VISIBLE_DEVICES'},
            result['prior_source_sha256'], result['original_source_sha256'])
        if source_identity is None:
            source_identity = run['source_sha256']
        assert run['source_sha256'] == source_identity, 'Unmatched source'
        device_identities.setdefault(device, common)
        assert common == device_identities[device], 'Unmatched source or execution environment'
        inputs = result['input_sha256'], result['changed_input_sha256'], result['config']
        identities.setdefault(dimension, inputs)
        assert identities[dimension] == inputs, 'Unmatched numerical inputs/configuration'
        key = device, dimension, arm
        rows = groups.setdefault(key, {})
        assert repeat not in rows, 'Duplicate cost repeat; select a single reviewed run interval'
        snapshots = [*result['stages'].values(), result['cold']['memory'], result['changed_cost']['memory'],
                     *(sample['memory'] for sample in result['samples'])]
        rss = {stage: row['rollup']['Rss'] / 1024**2 for stage, row in result['stages'].items()}
        rows[repeat] = {'run': number, 'uuid': uuid, 'result_sha256': hashlib.sha256(result_path.read_bytes()).hexdigest(),
            'manifest_sha256': hashlib.sha256((directory / 'run.json').read_bytes()).hexdigest(),
            'cold_seconds': result['build_seconds'] + result['trace_seconds'] + result['cold']['seconds'],
            'warm_ms': statistics.median(sample['seconds'] for sample in result['samples']) * 1000,
            'observed_rss_mib': max(row['rollup']['Rss'] for row in snapshots) / 1024**2,
            'stages_rss_mib': rss,
            'warm_growth_mib': rss['warm'] - rss['cold'],
            'gpu_allocator_peak_bytes': max(row['gpu']['peak'] for row in snapshots) if device == 'GPU' else None,
            'graph_nodes': result['graph_nodes'], 'graph_bytes': result['graph_bytes'], 'hlo_bytes': result['hlo_bytes']}
    require_same_physical_gpu(uuids)
    assert set(groups) == {(device, dimension, arm) for device in devices for dimension in (3, 5)
                          for arm in ('prior', 'graph', 'xla')}, 'Incomplete device/extent/arm matrix'
    assert all(set(rows) == {0, 1, 2} for rows in groups.values()), 'Three fresh processes per arm are mandatory'
    comparisons = []
    for device in devices:
        for dimension in (3, 5):
            arms = {}
            for arm in ('prior', 'graph', 'xla'):
                repeats = list(groups[device, dimension, arm].values())
                arms[arm] = {'repeats': repeats,
                    'median_cold_seconds': statistics.median(row['cold_seconds'] for row in repeats),
                    'median_warm_ms': statistics.median(row['warm_ms'] for row in repeats),
                    'max_observed_rss_mib': max(row['observed_rss_mib'] for row in repeats),
                    'max_gpu_allocator_peak_bytes': max(row['gpu_allocator_peak_bytes'] for row in repeats) if device == 'GPU' else None,
                    'max_warm_growth_mib': max(row['warm_growth_mib'] for row in repeats)}
            prior, candidate = arms['prior'], arms['xla']
            warm_ratio = candidate['median_warm_ms'] / prior['median_warm_ms']
            cold_ratio = candidate['median_cold_seconds'] / prior['median_cold_seconds']
            host_delta = candidate['max_observed_rss_mib'] - prior['max_observed_rss_mib']
            device_ratio = (candidate['max_gpu_allocator_peak_bytes'] / prior['max_gpu_allocator_peak_bytes']
                            if device == 'GPU' and prior['max_gpu_allocator_peak_bytes'] else None)
            triggers = []
            if warm_ratio > 1.2:
                triggers.append('warm_above_20_percent')
            if cold_ratio > 2:
                triggers.append('cold_above_2x')
            if host_delta > 256 or candidate['max_observed_rss_mib'] > 2 * prior['max_observed_rss_mib']:
                triggers.append('host_rss_above_256MiB_or_2x')
            if device_ratio is not None and device_ratio > 2:
                triggers.append('gpu_allocator_peak_above_2x')
            comparisons.append({'device': device, 'dimension': dimension, 'arms': arms,
                'warm_ratio': warm_ratio, 'cold_ratio': cold_ratio,
                'extra_observed_rss_mib': host_delta, 'gpu_allocator_ratio': device_ratio,
                'investigation_triggers': triggers,
                'warm_growth_role': 'Report every replicate; sustained growth requires longer reuse/churn attribution.'})
    return {'schema': 'filter_block_public_cost_comparison.v1', 'comparisons': comparisons,
        'devices': devices, 'source_sha256': source_identity,
        'nonclaims': ['Three process replicates are descriptive cost/repair evidence, not a general performance ranking.',
            'Observed RSS/PSS and allocator use do not prove peak bounds or native cache eviction.',
            'Sampled device sharing can miss brief activity between observations.',
            'Graph/XLA retain distinct declared solvers; this is not an identical-graph compiler ablation.',
            'This public ordered-block endpoint cannot close other endpoints, whole-repository compliance or HMC readiness.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--first-run', type=int, required=True)
    parser.add_argument('--last-run', type=int, required=True)
    parser.add_argument('--devices', nargs='+', choices=('CPU', 'GPU'), default=['CPU', 'GPU'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.root, args.first_run, args.last_run, args.devices)
    report['analyzer_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open('x') as out:
        json.dump(report, out, indent=2, allow_nan=False)
        out.write('\n')
    print(json.dumps([{key: row[key] for key in ('device', 'dimension', 'warm_ratio', 'cold_ratio',
        'extra_observed_rss_mib', 'gpu_allocator_ratio', 'investigation_triggers')}
        for row in report['comparisons']], indent=2))
