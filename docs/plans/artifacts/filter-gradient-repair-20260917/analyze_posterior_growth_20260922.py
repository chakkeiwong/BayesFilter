"""Standard-library post-run analysis of diagnostic capacity measurements."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--first-run', type=int, required=True)
    parser.add_argument('--last-run', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path('/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917')
    reports, sources = [], None
    for number in range(args.first_run, args.last_run + 1):
        directory = root / f'run-{number:05d}'
        run = json.loads((directory / 'run.json').read_text())
        assert run['state'] == 'passed'
        report_path = directory / 'posterior-growth.json'
        report = json.loads(report_path.read_text())
        if sources is None:
            sources = run['source_sha256']
        assert run['source_sha256'] == sources
        assert report['trace_count'] == 1 and report['runtime_operands'] == 3
        assert report['role'] == 'bounded_native_posterior_capacity_and_memory'
        assert report['dimension'] == 5
        assert run['environment']['CUDA_VISIBLE_DEVICES'] == ('3' if report['gpu'] else '-1')
        observations = report['observations']
        assert [value['additional_calls'] for value in observations] == [0, 1000, 2000, 3000]
        rss = [value['memory']['host']['VmRSS'] for value in observations]
        gpu = [value['memory'].get('gpu') for value in observations]
        if report['gpu']:
            assert run['gpu_preflight']
            provenance, = [json.loads(line) for line in (directory / 'process.log').read_text().splitlines()
                            if line.startswith('{"tensorflow_version":')]
            assert provenance['gpu_memory_policy']['all_physical_devices_memory_growth']
        reports.append({'run': number, 'device': 'GPU3' if report['gpu'] else 'CPU',
            'replicates': report['replicate_capacity'], 'graph_nodes': report['graph_nodes'],
            'hlo_bytes': report['hlo_bytes'],
            'stages_rss_mib': {key: value['host']['VmRSS'] / 1024**2 for key, value in report['stages'].items()},
            'rss_observations': rss, 'rss_increments_kib': [(right-left)/1024 for left,right in zip(rss,rss[1:])],
            'gpu_allocator': gpu, 'artifact': str(report_path),
            'artifact_sha256': hashlib.sha256(report_path.read_bytes()).hexdigest()})
    assert len(reports) == 6
    assert {(r['device'], r['replicates']) for r in reports} == {(d,c) for d in ('CPU','GPU3') for c in (2,4,8)}
    result = {'role': 'bounded_native_posterior_capacity_memory_analysis',
        'source_sha256': sources, 'reports': reports,
        'nonclaims': ['Observed process memory, not exact process peak or device reservation.',
            'No arbitrary target turnover, native executable eviction or general leak-freedom claim.',
            'No performance ranking between changed replicate capacities.']}
    with args.output.open('x') as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write('\n')
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()

