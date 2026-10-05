"""Post-run reference/consumer receipt; no numerical kernel or promotion decision."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COHORTS = {
    'public_cpu': range(2887, 2906),
    'public_gpu': range(2915, 2934),
    'consumers_cpu': (2886, *range(2906, 2912), 2913),
    'consumers_gpu': range(2934, 2942),
    'block_cpu': (2942,),
    'block_gpu': (2943,),
    'policy': (2914,),
}
summary, runtime_identity = {}, None
for name, numbers in COHORTS.items():
    workers = []
    for number in numbers:
        path = ROOT / f'run-{number:05d}' / 'run.json'
        row = json.loads(path.read_text())
        assert row['state'] == 'passed', (number, row['state'])
        evidence = row['test_evidence']
        assert evidence['passed'] and evidence['tests'] > 0 and not any(evidence[k] for k in ('failure', 'error', 'skipped'))
        identity = {p: value for p, value in row['source_sha256'].items() if p.startswith('bayesfilter/')}
        if runtime_identity is None:
            runtime_identity = identity
        assert identity == runtime_identity, (number, 'runtime source changed')
        if row['device'] == 'GPU':
            assert row['gpu_uuid'] == 'GPU-541e1e19-2df4-9064-4db9-9d0d2abc3eba'
            assert row['environment']['BAYESFILTER_TEST_DEVICE_SCOPE'] == 'visible'
        workers.append({'run': number, 'tests': evidence['tests'], 'elapsed_seconds': row['elapsed_seconds'],
            'device': row['device'], 'gpu_uuid': row.get('gpu_uuid'),
            'manifest_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    summary[name] = {'workers': workers, 'tests': sum(w['tests'] for w in workers)}
assert {name: row['tests'] for name, row in summary.items()} == {
    'public_cpu': 23, 'public_gpu': 23, 'consumers_cpu': 56, 'consumers_gpu': 56,
    'block_cpu': 43, 'block_gpu': 43, 'policy': 127}
report = {'schema': 'filter_sequential_public_qualification.v1', 'numerical_authority': '3582b4ac',
    'cohorts': summary, 'runtime_source_sha256': runtime_identity,
    'harness_correction': '02912 preserved; retired cfbc32d2 lifecycle alias replaced with existing3582b4ac test in02913. No runtime/fixture/tolerance change.',
    'nonclaims': ['Consumer compatibility does not compile the ordered block controller.',
        'Deferred progress requires independent external deadlines.',
        'Costs, nested ownership, F01-F20 dispositions and main integration remain separate gates.']}
with (ROOT / 'sequential-public-qualification-02943.json').open('x') as stream:
    json.dump(report, stream, indent=2)
    stream.write('\n')
print(json.dumps({name: row['tests'] for name, row in summary.items()}))
