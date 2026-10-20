"""Independent standard-library analysis of structured checkpoint cost arms."""
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
runs = [int(value) for value in sys.argv[2:]]
assert len(runs) == 8 and len(set(runs)) == 8
rows = {}
provenance = {}


def same(actual, expected, failures, path='result'):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        return max((same(actual[key], expected[key], failures, path + '.' + key) for key in expected), default=0.)
    if isinstance(expected, list):
        assert len(actual) == len(expected), path
        return max((same(x, y, failures, path + f'[{i}]') for i, (x, y) in enumerate(zip(actual, expected, strict=True))), default=0.)
    if isinstance(expected, (str, bool, int)) or expected is None:
        if actual != expected:
            failures.append({'path': path, 'actual': actual, 'expected': expected})
        return 0.
    assert isinstance(actual, (float, int))
    assert math.isfinite(actual) and math.isfinite(expected), (path, actual, expected)
    if abs(actual - expected) > 1e-10 + 1e-10 * abs(expected):
        failures.append({'path': path, 'actual': actual, 'expected': expected,
            'absolute_error': abs(actual - expected), 'tolerance': 1e-10 + 1e-10 * abs(expected)})
    return abs(actual - expected)


def metrics(row):
    samples = row['samples']
    stages = [*row['stages'].values(), *(sample['memory'] for sample in samples)]
    return {'cold_seconds': samples[0]['seconds'],
        'warm_median_seconds': statistics.median(sample['seconds'] for sample in samples[1:]),
        'host_peak_bytes': max(stage['host']['VmHWM'] for stage in stages),
        'gpu_peak_bytes': max(sample['memory']['gpu']['peak'] for sample in samples),
        'warm_host_first_last_bytes': [samples[i]['memory']['host']['VmRSS'] for i in (1, -1)],
        'warm_gpu_current_bytes': [sample['memory']['gpu']['current'] for sample in samples[1:]],
        'trace_count': row['trace_count'], 'graph_nodes': row['graph_nodes']}


for run in runs:
    directory = root / f'run-{run:05d}'
    manifest = json.loads((directory / 'run.json').read_text())
    assert manifest['state'] == 'passed' and manifest['device'] == 'GPU'
    report = directory / 'structured-memory.json'
    row = json.loads(report.read_text())
    key = (row['arm'], row['dimension'])
    assert key not in rows
    assert len(row['samples']) == 21 and row['max_iterations'] == 200
    assert row['result']['fit']['status'] == 'usable'
    assert row['result']['fit']['reused_training_count'] == 1
    assert row['jit_compile'] == (row['arm'] != 'graph')
    rows[key] = row
    provenance[str(key)] = {'run': run, 'report_sha256': hashlib.sha256(report.read_bytes()).hexdigest(),
        'gpu_index': manifest['environment']['CUDA_VISIBLE_DEVICES']}
assert len({v['gpu_index'] for v in provenance.values()}) == 1
assert len({json.dumps(row['source_sha256'], sort_keys=True) for row in rows.values()}) == 1
assert len({json.dumps(row['baseline_source_sha256'], sort_keys=True) for row in rows.values()}) == 1
comparisons = []
for dimension in (3, 5):
    arms = {arm: rows[(arm, dimension)] for arm in ('before', 'after', 'graph', 'xla')}
    assert len({tuple(row['input_sha256']) for row in arms.values()}) == 1
    for first, second in (('before', 'after'), ('graph', 'xla'), ('after', 'xla')):
        before = json.loads(json.dumps(arms[first]['result']))
        after = json.loads(json.dumps(arms[second]['result']))
        before['fit']['diagnostics'].pop('jit_compile')
        after['fit']['diagnostics'].pop('jit_compile')
        failures = []
        error = same(after, before, failures)
        a, b = metrics(arms[first]), metrics(arms[second])
        delta = b['host_peak_bytes'] - a['host_peak_bytes']
        gpu_ratio = b['gpu_peak_bytes'] / a['gpu_peak_bytes']
        cold_ratio = b['cold_seconds'] / a['cold_seconds']
        warm_ratio = b['warm_median_seconds'] / a['warm_median_seconds']
        comparisons.append({'dimension': dimension, 'before_arm': first, 'after_arm': second,
            'parity_passed': not failures, 'parity_failures': failures,
            'maximum_absolute_error': error, 'before': a, 'after': b,
            'host_peak_delta_bytes': delta, 'gpu_peak_ratio': gpu_ratio,
            'cold_ratio': cold_ratio, 'warm_ratio': warm_ratio,
            'triggers': {'host_over_256MiB': delta >= 256 * 1024 ** 2,
                'gpu_at_least_2x': gpu_ratio >= 2., 'cold_at_least_2x': cold_ratio >= 2.,
                'warm_at_least_2x': warm_ratio >= 2.}})
result = {'schema': 'filter_repair_structured_memory_comparison.v1', 'provenance': provenance,
    'all_parity_passed': all(row['parity_passed'] for row in comparisons),
    'comparisons': comparisons, 'analysis_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'nonclaims': ['Single fresh process per arm; no statistically supported performance ranking.',
        'Checkpoint f06fd505 isolates preparation and padding; original campaign baseline/repeats remain required.',
        'Outer sequential controller, block and quadratic controllers remain separate incomplete repairs.']}
p = root / f'structured-memory-comparison-{max(runs):05d}.json'
with p.open('x') as handle:
    json.dump(result, handle, indent=2)
    handle.write('\n')
print(p)
for row in comparisons:
    print(row['dimension'], row['before_arm'], row['after_arm'], 'error', row['maximum_absolute_error'],
        'failed_fields', len(row['parity_failures']),
        'host_delta', row['host_peak_delta_bytes'], 'gpu_ratio', row['gpu_peak_ratio'],
        'cold_ratio', row['cold_ratio'], 'warm_ratio', row['warm_ratio'], 'triggers', row['triggers'])
sys.exit(0 if result['all_parity_passed'] else 1)
