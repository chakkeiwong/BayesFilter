"""Diagnostic-only analysis of six matched complete-geometry cost workers."""

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(root, numbers):
    rows = []
    provenance = None
    device = None
    selected_uuid = None
    inputs = {}
    cases = set()
    for number in numbers:
        directory = root / f'run-{number:05d}'
        manifest_path = directory / 'run.json'
        data_path = directory / 'geometry-full-memory.json'
        manifest, data = read(manifest_path), read(data_path)
        assert manifest['state'] == 'passed' and manifest['exit_code'] == 0
        assert manifest['test_evidence'] == dict(passed=True, tests=1, failure=0, error=0, skipped=0)
        current_provenance = (manifest['git_head'], manifest['source_sha256'])
        if provenance is None:
            provenance = current_provenance
            device = manifest['device']
        assert current_provenance == provenance, 'source changed between cost workers'
        assert manifest['device'] == device
        assert data['gpu'] == (device == 'GPU')
        assert data['numerical_authority'] == '3582b4ac'
        assert data['mechanism_baseline'] == '91928762e'
        if data['arm'] == 'prior_refined':
            assert data['prior_comparator_patch'] == 'shared refined eigensystem in trust-region and pilot-sketch kernels only'
        assert len(data['samples']) == 20
        if device == 'GPU':
            uuid = manifest['gpu_uuid']
            if selected_uuid is None:
                selected_uuid = uuid
            assert uuid == selected_uuid, 'mixed physical GPUs'
            assert manifest['environment']['CUDA_VISIBLE_DEVICES'] == uuid
            assert manifest['gpu_performance_preflight_uncontended']
            samples = manifest['gpu_preflight'][-2:]
            assert len(samples) == 2
            for sample in samples:
                assert sample['selected_uuid'] == uuid
                assert sample['performance_preflight_uncontended']
                selected = [gpu for gpu in sample['devices'] if gpu['uuid'] == uuid]
                assert len(selected) == 1 and not selected[0]['compute_processes']
        case = (data['capacity'], data['arm'])
        assert case not in cases
        cases.add(case)
        expected_inputs = inputs.setdefault(data['capacity'], data['input_sha256'])
        assert expected_inputs == data['input_sha256'], 'unmatched realized inputs'
        snapshots = list(data['stages'].values()) + [data['cold']['memory']]
        snapshots += [sample['memory'] for sample in data['samples']]
        peak_rss = max(snapshot['status']['VmHWM'] for snapshot in snapshots)
        rusage_peak = max(snapshot['ru_maxrss_bytes'] for snapshot in snapshots)
        peak_pss = max(snapshot['rollup']['Pss'] for snapshot in snapshots)
        gpu_peak = max(snapshot['gpu']['peak'] for snapshot in snapshots) if data['gpu'] else None
        warm_rss = [sample['memory']['status']['VmRSS'] for sample in data['samples']]
        rows.append({'run': number, 'capacity': data['capacity'], 'arm': data['arm'],
            'input_sha256': data['input_sha256'], 'manifest_sha256': digest(manifest_path),
            'result_sha256': digest(data_path), 'gpu_uuid': selected_uuid,
            'cold_total_seconds': data['build_seconds'] + data['trace_seconds'] + data['cold']['seconds'],
            'build_seconds': data['build_seconds'], 'trace_seconds': data['trace_seconds'],
            'cold_call_seconds': data['cold']['seconds'],
            'warm_median_seconds': statistics.median(sample['seconds'] for sample in data['samples']),
            'peak_rss_bytes': peak_rss, 'rusage_peak_bytes': rusage_peak, 'peak_pss_bytes': peak_pss,
            'rusage_proc_peak_disagree': rusage_peak != peak_rss,
            'above_prepared_rss_bytes': peak_rss - data['stages']['prepared']['status']['VmRSS'],
            'gpu_allocator_peak_bytes': gpu_peak, 'warm_rss_bytes': warm_rss,
            'warm_rss_last_minus_first_bytes': warm_rss[-1] - warm_rss[0],
            'graph_nodes': data['graph_nodes'], 'graph_bytes': data['graph_bytes'],
            'hlo_bytes': data['hlo_bytes'], 'trace_count': data['trace_count'],
            'worker_original_record_comparison_passed': True})
    assert cases == {(capacity, arm) for capacity in (24, 120) for arm in ('prior_refined', 'graph', 'xla')}
    comparisons = []
    for capacity in (24, 120):
        by_arm = {row['arm']: row for row in rows if row['capacity'] == capacity}
        current = by_arm['xla']
        for other in ('prior_refined', 'graph'):
            baseline = by_arm[other]
            cold_ratio = current['cold_total_seconds'] / baseline['cold_total_seconds']
            warm_ratio = current['warm_median_seconds'] / baseline['warm_median_seconds']
            host_extra = current['peak_rss_bytes'] - baseline['peak_rss_bytes']
            device_ratio = (current['gpu_allocator_peak_bytes'] / baseline['gpu_allocator_peak_bytes']
                if device == 'GPU' and baseline['gpu_allocator_peak_bytes'] else None)
            comparisons.append({'capacity': capacity, 'comparison': f'xla/{other}',
                'descriptive_cold_ratio': cold_ratio, 'descriptive_warm_ratio': warm_ratio,
                'extra_host_rss_bytes': host_extra, 'device_peak_ratio': device_ratio,
                'investigation_triggers': {
                    'cold_above_2x': cold_ratio > 2,
                    'single_process_warm_above_20pct_needs_repeats': warm_ratio > 1.2,
                    'host_extra_above_256mib_or_2x': host_extra > 256 * 1024**2
                        or current['peak_rss_bytes'] > 2 * baseline['peak_rss_bytes'],
                    'device_above_2x': device_ratio is not None and device_ratio > 2}})
    return {'schema': 'filter_repair.geometry_full_descriptive_costs.v1', 'device': device,
        'gpu_uuid': selected_uuid, 'rows': rows, 'comparisons': comparisons,
        'authority': 'passed worker comparisons of complete original records at unchanged 1e-10 tolerances',
        'scope': 'Complete center-to-replay program and public formatting; prepared randomness excluded equally',
        'limitations': ['One fresh process per arm and extent; no supported performance ranking or terminal acceptance',
            'Prior_refined is 91928762e plus the demonstrated trust and pilot eigensystem repairs; unmodified prior 02714 fails parity and is excluded; graph/XLA solvers differ',
            'GPU identity and uncontended preflight are checked; no continuous external contention monitoring',
            'RSS uses /proc/self/status VmHWM; resource.ru_maxrss exceeds this already at prepared in several workers and is retained separately, not used as numerical-stage memory',
            'RSS, PSS and allocator peaks are distinct; these results do not isolate native executable lifetime']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('runs', nargs=6, type=int)
    args = parser.parse_args()
    result = analyze(args.root, args.runs)
    result['analysis_command'] = sys.argv
    result['analysis_source_sha256'] = digest(Path(__file__))
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    with args.output.with_suffix('.py').open('x') as stream:
        stream.write(Path(__file__).read_text())
    print(json.dumps({'output': str(args.output), 'comparisons': result['comparisons']}, indent=2))
