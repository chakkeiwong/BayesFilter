"""Independent diagnostic full-record, memory and timing comparison."""

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
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            failures.append({'path': path, 'kind': 'record_keys',
                'before': list(expected), 'after': list(actual) if isinstance(actual, dict) else actual})
            return math.inf
        return max((compare(actual[k], expected[k], path + '.' + k, failures)
            for k in expected if k != 'jit_compile'), default=0.)
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            failures.append({'path': path, 'kind': 'record_length', 'before': expected, 'after': actual})
            return math.inf
        return max((compare(a, b, path + f'[{i}]', failures)
            for i, (a, b) in enumerate(zip(actual, expected, strict=True))), default=0.)
    if isinstance(expected, (str, bool, int)) or expected is None:
        if actual != expected:
            failures.append({'path': path, 'before': expected, 'after': actual, 'kind': 'exact_field'})
        return 0.
    if not isinstance(actual, (float, int)):
        failures.append({'path': path, 'kind': 'changed_type', 'before': expected, 'after': actual})
        return math.inf
    if actual == expected or (math.isnan(actual) and math.isnan(expected)):
        return 0.
    error = abs(actual - expected)
    if not math.isfinite(error) or error > 1e-10 + 1e-10 * abs(expected):
        failures.append({'path': path, 'before': expected, 'after': actual, 'absolute_error': error})
    return error


def metrics(row):
    samples = row['samples']
    changes = row['changed_inputs']
    stages = [*row['stages'].values(), *(s['memory'] for s in samples),
        *(c[key]['memory'] for c in changes for key in ('first_call', 'immediate_repeat'))]
    return {'cold_seconds': row['cold_total_seconds'], 'setup_seconds': row['setup_seconds'],
        'build_seconds': row['build_seconds'], 'first_call_seconds': samples[0]['seconds'],
        'warm_median_seconds': statistics.median(s['seconds'] for s in samples[1:]),
        'warm_numerical_median_seconds': (None if samples[0]['numerical_seconds'] is None else
            statistics.median(s['numerical_seconds'] for s in samples[1:])),
        'warm_reporting_median_seconds': (None if samples[0]['reporting_seconds'] is None else
            statistics.median(s['reporting_seconds'] for s in samples[1:])),
        'host_peak_bytes': max(s['host']['VmHWM'] for s in stages),
        'gpu_peak_bytes': max(s['gpu']['peak'] for s in stages),
        'warm_host_growth_bytes': samples[-1]['memory']['host']['VmRSS'] - samples[1]['memory']['host']['VmRSS'],
        'changed_input_host_growth_bytes': row['stages']['measured']['host']['VmRSS'] - row['stages']['warm_measured']['host']['VmRSS'],
        'changed_input_first_seconds': [c['first_call']['seconds'] for c in changes],
        'changed_input_repeat_seconds': [c['immediate_repeat']['seconds'] for c in changes],
        'warm_gpu_current_bytes': sorted({s['memory']['gpu']['current'] for s in samples[1:]}),
        'graph_nodes': row['graph_nodes'], 'trace_count': row['trace_count']}


rows, provenance, manifests = {}, [], []
for run in runs:
    directory = root / f'run-{run:05d}'
    manifest = json.loads((directory / 'run.json').read_text())
    path = directory / 'lifecycle-memory.json'
    row = json.loads(path.read_text())
    assert manifest['state'] == 'passed' and manifest['device'] == 'GPU'
    assert len(row['samples']) == 21 and len(row['changed_inputs']) == 2
    assert row['optimizer_max_iterations'] == 200
    key = row['arm'], row['dimension']
    assert key not in rows
    rows[key] = row
    manifests.append(manifest)
    provenance.append({'run': run, 'arm': key[0], 'dimension': key[1],
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'gpu_index': manifest['environment']['CUDA_VISIBLE_DEVICES']})
assert len({r['gpu_index'] for r in provenance}) == 1
assert len({json.dumps(r['source_sha256'], sort_keys=True) for r in manifests}) == 1
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
        for index, (a, b) in enumerate(zip(old['changed_inputs'], new['changed_inputs'], strict=True)):
            assert a['input_sha256'] == b['input_sha256']
            error = max(error, compare(b['result'], a['result'], f'changed_inputs[{index}]', failures))
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
result = {'schema': 'filter_repair_lifecycle_memory_comparison.v1', 'provenance': provenance,
    'comparisons': comparisons, 'analysis_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'field_disposition': 'Only recorded jit_compile mode fields are excluded from graph/XLA comparison.',
    'nonclaims': ['One fresh process per arm; descriptive only, no terminal repeated evidence.',
        'Diagnostic complete record materialization is not the final public integration.',
        'The unchanged final mass helper uses XLA outside the tested lifecycle boundary in every arm.',
        'All numerical discrepancies retain unchanged comparison thresholds.',
        'External watchdog and actual full-transition-block evidence remain separate gates.']}
path = root / f'lifecycle-memory-comparison-{max(runs):05d}.json'
with path.open('x') as handle:
    json.dump(result, handle, indent=2)
    handle.write('\n')
print(path)
for c in comparisons:
    print(c['dimension'], c['before_arm'], c['after_arm'], 'error', c['maximum_absolute_error'],
        'failed_fields', len(c['failures']), 'host_delta', c['host_peak_delta_bytes'],
        'gpu_ratio', c['gpu_peak_ratio'], 'cold_ratio', c['cold_ratio'],
        'warm_ratio', c['warm_ratio'], 'triggers', c['triggers'])
