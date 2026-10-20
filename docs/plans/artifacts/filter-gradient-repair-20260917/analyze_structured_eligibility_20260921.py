"""Standard-library diagnostic analysis; no numerical runtime dependencies."""
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path('/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917')


def compare(actual, expected, path='result'):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        return max((compare(actual[k], expected[k], path + '.' + k) for k in expected), default=0.)
    if isinstance(expected, list):
        return max((compare(a, b, path + f'[{i}]') for i, (a, b) in enumerate(zip(actual, expected, strict=True))), default=0.)
    if isinstance(expected, (str, bool, int)) or expected is None:
        assert actual == expected, (path, actual, expected)
        return 0.
    assert math.isfinite(actual) and math.isfinite(expected), path
    error = abs(actual - expected)
    assert error <= 1e-10 + 1e-10 * abs(expected), (path, actual, expected)
    return error


reports, manifests, provenance = [], [], []
for run in (1653, 1654):
    directory = ROOT / f'run-{run:05d}'
    manifest = json.loads((directory / 'run.json').read_text())
    path = directory / 'structured-changing-eligibility.json'
    report = json.loads(path.read_text())
    assert manifest['state'] == 'passed' and manifest['device'] == 'GPU'
    assert report['schedule'] == [1, 0, 3, 8, 16, 32] * 2
    assert report['max_iterations'] == 200
    reports.append(report)
    manifests.append(manifest)
    provenance.append({'run': run, 'report_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
assert manifests[0]['environment']['CUDA_VISIBLE_DEVICES'] == manifests[1]['environment']['CUDA_VISIBLE_DEVICES']
assert reports[0]['source_sha256'] == reports[1]['source_sha256']
comparisons = []
for old, new in zip(reports[0]['samples'], reports[1]['samples'], strict=True):
    assert old['reused_rows'] == new['reused_rows']
    assert old['input_sha256'] == new['input_sha256']
    assert old['result']['fit']['status'] == new['result']['fit']['status'] == 'usable'
    comparisons.append({'reused_rows': old['reused_rows'],
        'maximum_absolute_error': compare(new['result'], old['result']),
        'before_seconds': old['seconds'], 'after_seconds': new['seconds']})
metrics = {}
for report in reports:
    samples = report['samples']
    metrics[report['arm']] = {'first_sequence_seconds': sum(s['seconds'] for s in samples[:6]),
        'second_sequence_seconds': sum(s['seconds'] for s in samples[6:]),
        'host_peak_bytes': max(s['memory']['host']['VmHWM'] for s in samples),
        'first_to_last_rss_delta_bytes': samples[-1]['memory']['host']['VmRSS'] - samples[0]['memory']['host']['VmRSS'],
        'second_sequence_rss_growth_bytes': samples[-1]['memory']['host']['VmRSS'] - samples[6]['memory']['host']['VmRSS'],
        'gpu_peak_bytes': max(s['memory']['gpu']['peak'] for s in samples)}
hlo = []
for run in (1649, 1650):
    path = ROOT / f'run-{run:05d}' / 'structured-memory-hlo.txt'
    data = path.read_text()
    branch_counts = [len(body.split(',')) for body in re.findall(r'branch_computations=\{([^}]+)\}', data)]
    hlo.append({'run': run, 'hlo_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'bytes': len(data.encode()), 'active_shape_branch_counts': branch_counts})
assert hlo[0]['active_shape_branch_counts'] == [5, 5]
assert hlo[1]['active_shape_branch_counts'] == [33, 33]
result = {'schema': 'filter_repair_structured_eligibility_cost.v1',
    'provenance': provenance, 'comparisons': comparisons, 'metrics': metrics, 'hlo': hlo,
    'analysis_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'disposition': 'Bounded upfront compact-CPQR and Jacobian-QR branches replace repeated shape compilation; default before/after parity passes.',
    'nonclaims': ['Single-process descriptive costs; three-process terminal comparisons remain.',
        'Repeated eligible rows isolate shapes and do not add target coverage.',
        'The distinct D5 graph/XLA tolerance failure remains open.',
        'No arbitrary capacity or unlimited native executable-cache claim.']}
path = ROOT / 'structured-eligibility-cost-disposition-01654.json'
with path.open('x') as handle:
    json.dump(result, handle, indent=2)
    handle.write('\n')
print(path)
print(json.dumps(metrics, indent=2))
print('maximum_error', max(c['maximum_absolute_error'] for c in comparisons))
