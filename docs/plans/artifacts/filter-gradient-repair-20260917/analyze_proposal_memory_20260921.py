"""Independent standard-library comparison of complete proposal cost arms."""
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
runs = [int(value) for value in sys.argv[2:]]
assert len(runs) == len(set(runs)) == 6


def same(actual, expected, path='result'):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        return max((same(actual[k], expected[k], path + '.' + k) for k in expected), default=0.)
    if isinstance(expected, list):
        return max((same(a, b, path + f'[{i}]') for i, (a, b) in enumerate(zip(actual, expected, strict=True))), default=0.)
    if isinstance(expected, (str, bool, int)) or expected is None:
        assert actual == expected, (path, actual, expected)
        return 0.
    assert math.isfinite(actual) and math.isfinite(expected), path
    error = abs(actual - expected)
    assert error <= 1e-10 + 1e-10 * abs(expected), (path, actual, expected)
    return error


def metrics(row):
    samples = row['samples']
    stages = [*row['stages'].values(), *(sample['memory'] for sample in samples)]
    return {'cold_seconds': samples[0]['seconds'],
        'warm_median_seconds': statistics.median(s['seconds'] for s in samples[1:]),
        'host_peak_bytes': max(stage['host']['VmHWM'] for stage in stages),
        'gpu_peak_bytes': max(s['memory']['gpu']['peak'] for s in samples),
        'warm_host_growth_bytes': samples[-1]['memory']['host']['VmRSS'] - samples[1]['memory']['host']['VmRSS'],
        'warm_gpu_current_bytes': sorted({s['memory']['gpu']['current'] for s in samples[1:]}),
        'graph_nodes': row['graph_nodes'], 'trace_count': row['trace_count']}


rows, provenance = {}, []
for run in runs:
    directory = root / f'run-{run:05d}'
    manifest = json.loads((directory / 'run.json').read_text())
    path = directory / 'proposal-memory.json'
    row = json.loads(path.read_text())
    assert manifest['state'] == 'passed' and manifest['device'] == 'GPU'
    assert len(row['samples']) == 21
    key = row['arm'], row['dimension']
    assert key not in rows
    rows[key] = row
    provenance.append({'run': run, 'arm': key[0], 'dimension': key[1],
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'gpu_index': manifest['environment']['CUDA_VISIBLE_DEVICES']})
assert len({r['gpu_index'] for r in provenance}) == 1
assert len({json.dumps(r['source_sha256'], sort_keys=True) for r in rows.values()}) == 1
assert len({json.dumps(r['baseline_source_sha256'], sort_keys=True) for r in rows.values()}) == 1
comparisons = []
for dimension in (3, 5):
    assert len({tuple(rows[arm, dimension]['input_sha256']) for arm in ('before', 'graph', 'xla')}) == 1
    for first, second in [('before', 'xla'), ('graph', 'xla')]:
        old, new = rows[first, dimension], rows[second, dimension]
        error = same(new['result'], old['result'])
        a, b = metrics(old), metrics(new)
        host_delta = b['host_peak_bytes'] - a['host_peak_bytes']
        gpu_ratio = b['gpu_peak_bytes'] / a['gpu_peak_bytes']
        cold_ratio = b['cold_seconds'] / a['cold_seconds']
        warm_ratio = b['warm_median_seconds'] / a['warm_median_seconds']
        comparisons.append({'dimension': dimension, 'before_arm': first, 'after_arm': second,
            'before': a, 'after': b, 'maximum_absolute_error': error,
            'host_peak_delta_bytes': host_delta, 'gpu_peak_ratio': gpu_ratio,
            'cold_ratio': cold_ratio, 'warm_ratio': warm_ratio,
            'triggers': {'host_over_256MiB': host_delta >= 256 * 1024**2,
                'gpu_at_least_2x': gpu_ratio >= 2., 'cold_at_least_2x': cold_ratio >= 2.,
                'warm_over_20percent': warm_ratio > 1.2}})
result = {'schema': 'filter_repair_proposal_memory_comparison.v1', 'provenance': provenance,
    'comparisons': comparisons, 'analysis_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'nonclaims': ['Single fresh process per arm; no statistically supported ranking or terminal repeats.',
        'Before uses its original mixed host/XLA-trust route; candidate graph is explicitly non-default.',
        'The enclosing outer refinement, factor escalation and broader controllers remain open.']}
path = root / f'proposal-memory-comparison-{max(runs):05d}.json'
with path.open('x') as handle:
    json.dump(result, handle, indent=2)
    handle.write('\n')
print(path)
for c in comparisons:
    print(c['dimension'], c['before_arm'], c['after_arm'], 'error', c['maximum_absolute_error'],
        'host_delta', c['host_peak_delta_bytes'], 'gpu_ratio', c['gpu_peak_ratio'],
        'cold_ratio', c['cold_ratio'], 'warm_ratio', c['warm_ratio'], 'triggers', c['triggers'])
