"""Diagnostic comparison of conditional-fit compilation at two capacities."""

import hashlib
import json
import math
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
runs = list(map(int, sys.argv[2:]))
assert len(runs) == len(set(runs)) == 4


def compare(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        return max((compare(actual[k], expected[k]) for k in expected), default=0.)
    if isinstance(expected, list):
        return max((compare(a, b) for a, b in zip(actual, expected, strict=True)), default=0.)
    if isinstance(expected, (str, bool, int)) or expected is None:
        assert actual == expected
        return 0.
    if actual == expected or (math.isnan(actual) and math.isnan(expected)):
        return 0.
    error = abs(actual - expected)
    assert error <= 1e-10 + 1e-10 * abs(expected), (actual, expected)
    return error


rows, summaries = {}, []
for run in runs:
    directory = root / f'run-{run:05d}'
    manifest = json.loads((directory / 'run.json').read_text())
    path = directory / 'attempts-memory.json'
    row = json.loads(path.read_text())
    assert manifest['state'] == 'passed' and manifest['device'] == 'GPU'
    assert row['warm_calls'] == 3 and len(row['samples']) == 4
    key = row['arm'], row['capacity']
    assert key not in rows
    rows[key] = row
    hlo = (directory / 'attempts.hlo').read_text()
    samples = row['samples']
    peak = max(stage['host']['VmHWM'] for stage in
        [*row['stages'].values(), *(s['memory'] for s in samples)])
    summaries.append({'run': run, 'arm': key[0], 'capacity': key[1],
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'gpu_index': manifest['environment']['CUDA_VISIBLE_DEVICES'],
        'host_peak_bytes': peak, 'host_peak_over_built_bytes': peak - row['stages']['built']['host']['VmHWM'],
        'gpu_peak_bytes': max(s['memory']['gpu']['peak'] for s in samples),
        'warm_host_growth_bytes': samples[-1]['memory']['host']['VmRSS'] - samples[1]['memory']['host']['VmRSS'],
        'warm_gpu_current_bytes': sorted({s['memory']['gpu']['current'] for s in samples[1:]}),
        'graph_nodes': row['graph_nodes'] or row['baseline_second_fit_graph_nodes'],
        'hlo_bytes': len(hlo.encode()), 'hlo_instructions': len(re.findall(r'^  %', hlo, re.MULTILINE)),
        'hlo_conditional_branch_counts': sorted(len(v.split(',')) for v in
            re.findall(r'branch_computations=\{([^}]+)\}', hlo)),
        'cold_seconds': samples[0]['seconds']})
assert len({r['gpu_index'] for r in summaries}) == 1
assert len({json.dumps(r['source_sha256'], sort_keys=True) for r in rows.values()}) == 1
comparisons = []
for capacity in (4, 32):
    a, b = rows['before', capacity], rows['xla', capacity]
    assert a['input_sha256'] == b['input_sha256']
    error = compare(b['result'], a['result'])
    am = next(r for r in summaries if (r['arm'], r['capacity']) == ('before', capacity))
    bm = next(r for r in summaries if (r['arm'], r['capacity']) == ('xla', capacity))
    comparisons.append({'capacity': capacity, 'maximum_absolute_error': error,
        'host_peak_delta_bytes': bm['host_peak_bytes'] - am['host_peak_bytes'],
        'host_growth_delta_bytes': bm['host_peak_over_built_bytes'] - am['host_peak_over_built_bytes']})
result = {'schema': 'filter_repair_attempts_capacity_diagnostic.v1',
    'summaries': summaries, 'comparisons': comparisons,
    'analysis_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'nonclaims': ['One process per arm/capacity; no statistical ranking or terminal repeats.',
        'Correlated compilation size and capacity can explain observed overhead, not an arbitrary-capacity memory cap.',
        'Inherited graph/XLA numerical discrepancies and outer-controller integration remain open.']}
path = root / f'attempts-capacity-disposition-{max(runs):05d}.json'
with path.open('x') as handle:
    json.dump(result, handle, indent=2)
    handle.write('\n')
print(path)
print(json.dumps(result, indent=2))
