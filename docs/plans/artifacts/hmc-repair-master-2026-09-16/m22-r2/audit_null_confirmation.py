"""Read-only terminal audit of the fixed 512-per-arm null confirmation."""
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import tarfile
import time

start = time.monotonic()
root = Path(__file__).resolve().parent
job = root / 'null-confirmation-cpu-r1/m22-merged-null-confirmation'
index = json.loads((root / 'null-confirmation-cpu-r1/run_index.json').read_text())
result_path = job / 'result.json'
result = json.loads(result_path.read_text())
rate = result['assessment']['rates']
trials = result['assessment']['trials']
assert len(trials) == 512
assert result['execution_status'] == 'complete'
assert hashlib.sha256((job / 'attempt-001-result.json').read_bytes()).hexdigest() == index['jobs']['m22-merged-null-confirmation']['result_sha256']
assert json.loads((job / 'attempt-001-result.json').read_text()) == result
source = index['source']
archive = subprocess.check_output(['git', 'archive', source['commit'], 'bayesfilter'])
with tarfile.open(fileobj=io.BytesIO(archive)) as tree:
    original = {f.name: hashlib.sha256(tree.extractfile(f).read()).hexdigest()
                for f in tree.getmembers() if f.isfile() and f.name.endswith('.py')}
assert original == source['files']
manifest = json.loads((root.parent / 'm21-r2/source-manifest-r1.json').read_text())
assert {k: v for k, v in manifest['files'].items() if k.endswith('.py')} == original
seen = set()
look_count = 0
histogram = {}
totals = {arm: {'valid': 0, 'detected': 0} for arm in ('baseline', 'noop')}
for number, trial in enumerate(trials):
    assert json.loads((job / f'power-trial-{number:04}.json').read_text()) == trial
    for arm, record in trial.items():
        assert arm in totals and record['valid']
        path = job / f'trial-{number:04}' / arm / 'invariance.json'
        experiment = json.loads(path.read_text())
        assert not experiment['reused_cumulative_samples']
        looks, observations = experiment['looks'], experiment['observations']
        assert 1 <= len(looks) <= 3 and len(looks) == len(observations)
        beta = .05 / 3
        gamma = beta ** (1/3)
        for k, (look, obs) in enumerate(zip(looks, observations)):
            assert look['sample_count'] == (16384 if k == 0 else 32768)
            key = obs['design']['seed']
            assert key not in seen
            seen.add(key)
            raw = json.loads(Path(obs['result_path']).read_text())
            assert raw['all_substep_states_and_log_ratios_finite'] is True
            assert raw['kernel_power'] == 32 and raw['rank_draws'] == 7
            assert raw['replications'] == look['sample_count']
            values = [t['p_value'] for group in ('rank_tests', 'two_sample_tests', 'analytic_tests')
                      for t in raw[group].values()]
            assert len(values) == 3 and values == look['p_values']
            q = 3 * min(values)
            decision = ('reject' if q <= beta else 'early_nonrejection' if q > beta + gamma
                        else 'continue' if k < 2 else 'look_cap_without_rejection')
            assert decision == look['decision']
            assert k == len(looks)-1 or decision == 'continue'
            beta /= gamma
            look_count += 1
        rejected = looks[-1]['decision'] == 'reject'
        assert rejected == record['detected']
        assert experiment['decision'] == looks[-1]['decision']
        totals[arm]['valid'] += 1
        totals[arm]['detected'] += int(rejected)
        histogram[len(looks)] = histogram.get(len(looks), 0) + 1

def cdf(n, k, p):
    return sum(math.comb(n, j) * p**j * (1-p)**(n-j) for j in range(k+1))

def invert_tail(k, probability):
    lo, hi = 0., 1.
    for _ in range(60):
        mid = (lo+hi)/2
        if cdf(512, k, mid) > probability:
            lo = mid
        else:
            hi = mid
    return (lo+hi)/2

for arm, counts in totals.items():
    assert counts['valid'] == 512 == rate[arm]['valid'] == rate[arm]['planned']
    assert counts['detected'] == rate[arm]['detected']
    k = counts['detected']
    interval = (invert_tail(k-1, .975), invert_tail(k, .025))
    assert all(abs(a-b) < 1e-12 for a, b in zip(interval, rate[arm]['interval']))
    assert interval[1] <= .10
summary = {'status': 'passed_declared_null_screen', 'rates': rate,
    'whole_experiments': len(trials)*2, 'independent_looks': look_count,
    'look_count_histogram': histogram, 'unique_look_seeds': len(seen),
    'source_commit': source['commit'], 'python_source_identity': source['identity'],
    'full_package_identity': manifest['source_identity'],
    'source_identity_difference': 'full package includes symmetric_sylvester_op.cc; Python-only index does not',
    'source_matches_git_archive': True, 'all_substep_health_checked': True,
    'result_sha256': hashlib.sha256(result_path.read_bytes()).hexdigest(),
    'plan_file': 'docs/plans/bayesfilter-hmc-merged-source-continuation-2026-09-22.md',
    'wall_seconds': time.monotonic()-start,
    'limits': ['one frozen Gaussian transition experiment', 'no exact .05 size proof',
               'no stopped-posterior coverage or subtle full-fit defect power',
               'does not certify later source changes']}
(root / 'null-confirmation-audit.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(summary, indent=2))
