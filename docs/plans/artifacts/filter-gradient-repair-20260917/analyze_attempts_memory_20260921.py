"""Independent standard-library full-record and cost comparison for attempts."""

import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
runs = [int(value) for value in sys.argv[2:]]
assert len(runs) == len(set(runs)) == 6


def compare(actual, expected, path='result', failures=None):
    failures = [] if failures is None else failures
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        return max((compare(actual[k], expected[k], path + '.' + k, failures)
            for k in expected if k != 'jit_compile'), default=0.)
    if isinstance(expected, list):
        return max((compare(a, b, path + f'[{i}]', failures)
            for i, (a, b) in enumerate(zip(actual, expected, strict=True))), default=0.)
    if isinstance(expected, (str, bool, int)) or expected is None:
        if actual != expected:
            failures.append({'path': path, 'before': expected, 'after': actual, 'kind': 'exact_field'})
        return 0.
    if actual == expected or (math.isnan(actual) and math.isnan(expected)):
        return 0.
    error = abs(actual - expected)
    if not math.isfinite(error) or error > 1e-10 + 1e-10 * abs(expected):
        failures.append({'path': path, 'before': expected, 'after': actual, 'absolute_error': error})
    return error


def metrics(row):
    samples = row['samples']
    stages = [*row['stages'].values(), *(s['memory'] for s in samples)]
    return {'cold_seconds': samples[0]['seconds'],
        'warm_median_seconds': statistics.median(s['seconds'] for s in samples[1:]),
        'host_peak_bytes': max(s['host']['VmHWM'] for s in stages),
        'attempt_host_peak_growth_bytes': max(s['host']['VmHWM'] for s in stages) - row['stages']['built']['host']['VmHWM'],
        'gpu_peak_bytes': max(s['memory']['gpu']['peak'] for s in samples),
        'warm_host_growth_bytes': samples[-1]['memory']['host']['VmRSS'] - samples[1]['memory']['host']['VmRSS'],
        'warm_gpu_current_bytes': sorted({s['memory']['gpu']['current'] for s in samples[1:]}),
        'graph_nodes': row['graph_nodes'], 'trace_count': row['trace_count']}


rows, provenance = {}, []
for run in runs:
    directory = root / f'run-{run:05d}'
    manifest = json.loads((directory / 'run.json').read_text())
    path = directory / 'attempts-memory.json'
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
    assert len({json.dumps(rows[arm, dimension]['input_sha256'], sort_keys=True)
        for arm in ('before', 'graph', 'xla')}) == 1
    for first, second in [('before', 'xla'), ('graph', 'xla')]:
        old, new = rows[first, dimension], rows[second, dimension]
        failures = []
        error = compare(new['result'], old['result'], failures=failures)
        a, b = metrics(old), metrics(new)
        host_delta = b['host_peak_bytes'] - a['host_peak_bytes']
        gpu_ratio = b['gpu_peak_bytes'] / a['gpu_peak_bytes']
        cold_ratio = b['cold_seconds'] / a['cold_seconds']
        warm_ratio = b['warm_median_seconds'] / a['warm_median_seconds']
        comparisons.append({'dimension': dimension, 'before_arm': first, 'after_arm': second,
            'before': a, 'after': b, 'maximum_absolute_error': error, 'numerical_pass': not failures,
            'failures': failures, 'host_peak_delta_bytes': host_delta, 'gpu_peak_ratio': gpu_ratio,
            'cold_ratio': cold_ratio, 'warm_ratio': warm_ratio,
            'triggers': {'host_over_256MiB': host_delta >= 256 * 1024**2,
                'gpu_at_least_2x': gpu_ratio >= 2., 'cold_at_least_2x': cold_ratio >= 2.,
                'warm_over_20percent': warm_ratio > 1.2}})
result = {'schema': 'filter_repair_attempts_memory_comparison.v1', 'provenance': provenance,
    'comparisons': comparisons, 'analysis_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'field_disposition': 'Only the recorded jit_compile mode is excluded from graph/XLA numerical comparison.',
    'nonclaims': ['One fresh process per arm; no statistically supported ranking or terminal repeated evidence.',
        'Before uses its original host controller and XLA fitter/proposal; graph is explicitly non-default.',
        'An inherited fitter discrepancy remains a failed comparison at unchanged thresholds.',
        'Outer refinement and terminal-fit lifecycle remain open.']}
path = root / f'attempts-memory-comparison-{max(runs):05d}.json'
with path.open('x') as handle:
    json.dump(result, handle, indent=2)
    handle.write('\n')
print(path)
for c in comparisons:
    print(c['dimension'], c['before_arm'], c['after_arm'], 'error', c['maximum_absolute_error'],
        'failed_fields', len(c['failures']), 'host_delta', c['host_peak_delta_bytes'],
        'gpu_ratio', c['gpu_peak_ratio'], 'cold_ratio', c['cold_ratio'],
        'warm_ratio', c['warm_ratio'], 'triggers', c['triggers'])
