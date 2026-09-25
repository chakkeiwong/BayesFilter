"""Post-run diagnostic analysis only; standard-library calculations on artifacts."""

import argparse
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=('posterior_native', 'uniform_public', 'geometry_control'), required=True)
    parser.add_argument('--repeats', type=int, choices=(1, 3), default=1)
    parser.add_argument('--first-run', type=int, required=True)
    parser.add_argument('--last-run', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path('/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917')
    filename, role = {
        'geometry_control': ('geometry-control-memory.json', 'descriptive_complete_center_proposal_cost'),
        'posterior_native': ('posterior-native-memory.json', 'descriptive_complete_native_posterior_curvature_cost'),
        'uniform_public': ('uniform-public-memory.json', 'descriptive_public_uniform_refinement_cost'),
    }[args.kind]
    grouped = {}
    source_hashes = None
    original_hashes = None
    for number in range(args.first_run, args.last_run + 1):
        directory = root / f'run-{number:05d}'
        artifact = directory / filename
        if not artifact.exists():
            continue
        run = json.loads((directory/'run.json').read_text())
        if run['state'] != 'passed':
            raise ValueError(f'Invalid numerical comparison in {directory}')
        report = json.loads(artifact.read_text())
        if source_hashes is None:
            source_hashes = run['source_sha256']
            original_hashes = report['original_source_sha256']
        assert run['source_sha256'] == source_hashes, 'Source changed within comparison'
        assert report['original_source_sha256'] == original_hashes, 'Original closure changed'
        assert report['baseline'] == '3582b4ac'
        assert report['role'] == role
        assert len(report['samples']) == 21
        assert run['device'] == ('GPU' if report['gpu'] else 'CPU')
        assert run['environment']['CUDA_VISIBLE_DEVICES'] == ('3' if report['gpu'] else '-1')
        if report['gpu']:
            assert run['environment']['TF_FORCE_GPU_ALLOW_GROWTH'] == 'true'
            assert run['gpu_preflight']
            provenance = [json.loads(line) for line in (directory/'process.log').read_text().splitlines()
                          if line.startswith('{"tensorflow_version":')]
            assert len(provenance) == 1
            policy = provenance[0]['gpu_memory_policy']
            assert policy['mode'] == 'memory_growth'
            assert policy['all_physical_devices_memory_growth']
            assert policy['configured_before_logical_device_initialization']
            assert provenance[0]['cuda_visible_devices'] == '3'
            assert provenance[0]['trust_basis'] == 'owner_designated_managed_session_visible_gpu_trusted'
        key = ('GPU3' if report['gpu'] else 'CPU', report['dimension'])
        arm = report['arm']
        assert report['jit_compile'] == (arm == 'xla')
        if arm != 'before':
            assert report['trace_count'] == 1
        memories = list(report['stages'].values()) + [row['memory'] for row in report['samples']] + [report['changed_cost']['memory']]
        peak_rss = max(value['host']['VmRSS'] for value in memories)
        peak_hwm = max(value['host']['VmHWM'] for value in memories)
        gpu_peak = max(value['gpu']['peak'] for value in memories) if report['gpu'] else None
        result = {'run': number, 'artifact': str(artifact),
            'artifact_sha256': hashlib.sha256(artifact.read_bytes()).hexdigest(),
            'warm_median_ms': 1000*statistics.median(row['seconds'] for row in report['samples'][1:]),
            'cold_seconds': report.get('build_seconds', 0) + (report.get('trace_seconds') or 0) + report['samples'][0]['seconds'],
            'observed_peak_rss': peak_rss, 'observed_hwm': peak_hwm, 'gpu_allocator_peak': gpu_peak,
            'trace_count': report['trace_count'], 'graph_nodes': report['graph_nodes'], 'hlo_bytes': report['hlo_bytes'],
            'input_sha256': report['input_sha256']}
        grouped.setdefault(key, {}).setdefault(arm, []).append(result)
    comparisons = []
    assert grouped, 'No comparison artifacts found'
    for (device, dimension), arms in grouped.items():
        assert set(arms) == {'before', 'graph', 'xla'}, (device, dimension, arms.keys())
        assert all(len(values) == args.repeats for values in arms.values()), 'Missing or extra repetitions'
        assert len({tuple(row['input_sha256']) for values in arms.values() for row in values}) == 1
        summaries = {arm:{key: statistics.median(row[key] for row in values)
            for key in ('warm_median_ms','cold_seconds','observed_peak_rss','observed_hwm')}
            for arm, values in arms.items()}
        before, xla = summaries['before'], summaries['xla']
        warm_ratio = xla['warm_median_ms'] / before['warm_median_ms']
        cold_ratio = xla['cold_seconds'] / before['cold_seconds']
        extra_rss = xla['observed_peak_rss'] - before['observed_peak_rss']
        gpu_ratio = None
        if device == 'GPU3':
            gpu_ratio = statistics.median(row['gpu_allocator_peak'] for row in arms['xla'])/statistics.median(row['gpu_allocator_peak'] for row in arms['before'])
        triggers = []
        if warm_ratio > 1.2:
            triggers.append('warm_time_above_20_percent')
        if cold_ratio > 2:
            triggers.append('cold_time_above_2x')
        if extra_rss > 256*1024**2:
            triggers.append('extra_host_rss_above_256MiB')
        if gpu_ratio is not None and gpu_ratio > 2:
            triggers.append('gpu_allocator_peak_above_2x')
        comparisons.append({'device':device,'dimension':dimension,'arms':arms,'summaries':summaries,
            'xla_before_warm_ratio':warm_ratio,'xla_before_cold_ratio':cold_ratio,
            'xla_extra_observed_rss_mib':extra_rss/1024**2,'gpu_allocator_peak_ratio':gpu_ratio,'triggers':triggers})
    output = {'schema':'filter_controller_cost_analysis.v1','kind':args.kind,'repeats':args.repeats,'created_utc':datetime.now(timezone.utc).isoformat(),
        'first_run':args.first_run,'last_run':args.last_run,'comparisons':comparisons,
        'source_sha256':source_hashes,'original_source_sha256':original_hashes,
        'nonclaims':['Descriptive fresh-process comparison only; no statistical ranking.',
            'Observed RSS/VmHWM are not exact process peaks or GPU live tensor counts.',
            'The comparison covers this endpoint and fixture only; broader and terminal qualification remain separate.']}
    with args.output.open('x') as out:
        json.dump(output,out,indent=2,allow_nan=False)
        out.write('\n')
    print(json.dumps([{k:row[k] for k in ('device','dimension','summaries','xla_extra_observed_rss_mib','triggers')} for row in comparisons],indent=2))


if __name__ == '__main__':
    main()
