"""Post-run diagnostic comparison of complete uniform-refinement costs."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path('/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917')


def compare(actual, expected, path='result'):
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys(), path
        for key in expected:
            compare(actual[key], expected[key], path + '.' + key)
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), path
        for i, (left, right) in enumerate(zip(actual, expected, strict=True)):
            compare(left, right, f'{path}[{i}]')
    elif isinstance(expected, float):
        assert math.isfinite(actual) and math.isfinite(expected)
        assert abs(actual - expected) <= 1e-10 + 1e-10 * abs(expected), (path, actual, expected)
    else:
        assert type(actual) is type(expected) and actual == expected, (path, actual, expected)


parser = argparse.ArgumentParser()
parser.add_argument('--first-run', type=int, required=True)
parser.add_argument('--last-run', type=int, required=True)
parser.add_argument('--repeats', type=int, choices=(1, 3), required=True)
parser.add_argument('--scope', choices=('native', 'public'), default='native')
parser.add_argument('--output', type=Path)
args = parser.parse_args()
prefix = 'uniform_round_memory_' if args.scope == 'native' else 'uniform_public_memory_'
filename = 'uniform-round-memory.json' if args.scope == 'native' else 'uniform-public-memory.json'
groups, source = {}, None
for number in range(args.first_run, args.last_run + 1):
    directory = ROOT / f'run-{number:05d}'
    manifest = json.loads((directory / 'run.json').read_text())
    if not manifest['key'][1].startswith(prefix):
        continue
    assert manifest['state'] == 'passed'
    source = manifest['source_sha256'] if source is None else source
    assert manifest['source_sha256'] == source
    row = json.loads((directory / filename).read_text())
    if manifest['device'] == 'GPU':
        assert manifest['environment']['CUDA_VISIBLE_DEVICES'] == '3'
    compare(row['result'], row['original_result'])
    compare(row['changed_result'], row['original_changed_result'])
    groups.setdefault((manifest['device'], row['dimension']), {}).setdefault(row['arm'], []).append((number, manifest['key'][6], row))
assert groups
results = []
for (device, dimension), arms in sorted(groups.items()):
    assert set(arms) == {'before', 'graph', 'xla'}
    reference = arms['before'][0][2]
    metrics = {}
    for arm, rows in arms.items():
        assert len(rows) == args.repeats and {r[1] for r in rows} == set(range(args.repeats))
        summaries = []
        for number, repeat, row in rows:
            assert row['input_sha256'] == reference['input_sha256']
            assert row['original_source_sha256'] == reference['original_source_sha256']
            compare(row['result'], reference['result'])
            compare(row['changed_result'], reference['changed_result'])
            observations = [*row['stages'].values(), *(s['memory'] for s in row['samples']), row['changed_cost']['memory']]
            summaries.append({'run': number, 'repeat': repeat,
                'cold_ms': row['samples'][0]['seconds'] * 1000,
                'cold_total_ms': 1000 * (row['samples'][0]['seconds'] + row.get('build_seconds', 0) + (row.get('trace_seconds') or 0)),
                'build_ms': 1000 * row.get('build_seconds', 0),
                'trace_ms': 1000 * (row.get('trace_seconds') or 0),
                'warm_ms': 1000 * statistics.median(s['seconds'] for s in row['samples'][1:]),
                'host_rss_peak': max(o['host']['VmRSS'] for o in observations),
                'reported_vmhwm': max(o['host']['VmHWM'] for o in observations),
                'gpu_peak': max(o['gpu']['peak'] for o in observations) if device == 'GPU' else None,
                'warm_rss_growth': row['samples'][-1]['memory']['host']['VmRSS'] - row['samples'][1]['memory']['host']['VmRSS'],
                'trace_count': row['trace_count'], 'graph_nodes': row['graph_nodes']})
        warm = [row['warm_ms'] for row in summaries]
        metrics[arm] = {'runs': summaries, 'warm_median_ms': statistics.median(warm),
            'warm_range_ms': [min(warm), max(warm)],
            'cold_total_median_ms': statistics.median(s['cold_total_ms'] for s in summaries),
            'cold_total_range_ms': [min(s['cold_total_ms'] for s in summaries), max(s['cold_total_ms'] for s in summaries)],
            'host_rss_peak_median': statistics.median(s['host_rss_peak'] for s in summaries),
            'gpu_peak_median': statistics.median(s['gpu_peak'] for s in summaries) if device == 'GPU' else None}
    triggers = []
    for baseline in ('before', 'graph'):
        old, new = metrics[baseline], metrics['xla']
        if new['cold_total_median_ms'] > 2 * old['cold_total_median_ms']:
            triggers.append('cold_total_over_2x_vs_' + baseline)
        if new['warm_median_ms'] > 1.2 * old['warm_median_ms']:
            triggers.append('warm_over_20_percent_vs_' + baseline)
        if new['host_rss_peak_median'] - old['host_rss_peak_median'] > 256 * 2**20:
            triggers.append('host_over_256MiB_vs_' + baseline)
        if device == 'GPU' and new['gpu_peak_median'] > 2 * old['gpu_peak_median']:
            triggers.append('gpu_peak_over_2x_vs_' + baseline)
    results.append({'device': device, 'dimension': dimension, 'complete_records_equal': True,
        'metrics': metrics, 'triggers': triggers})
    print(json.dumps({'device': device, 'dimension': dimension, 'warm_ms': {k: v['warm_median_ms'] for k, v in metrics.items()}, 'triggers': triggers}))
report = {'role': 'descriptive_complete_uniform_' + args.scope + '_costs',
    'range': [args.first_run, args.last_run], 'comparisons': results,
    'analysis_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'nonclaims': ['No statistical ranking.', 'No general leak freedom or exact peak attribution.',
        'No actual DZ5 or full multistart composition evidence.']}
output = args.output or ROOT / f'uniform-{args.scope}-costs-{args.last_run:05d}.json'
with output.open('x') as handle:
    json.dump(report, handle, indent=2, allow_nan=False)
    handle.write('\n')
