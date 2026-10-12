"""Independent read-only reference/accounting diagnostic; no sampler authority."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

ROOT = Path('/home/ubuntu/python/BayesFilter')
sys.path.insert(0, str(ROOT))
from scripts.calibrate_hmc_stopped_mcse import METHODS, LOOKS, finite_mean_variance, summarize


def expected_batch_lrv(n, b, rho):
    a = n // b
    return b*a/(a-1)*(finite_mean_variance(b, rho, chains=1)
                       - finite_mean_variance(a*b, rho, chains=1))


def direct_batch_covariance_oracle(n, b, rho):
    a = n // b
    covariance = [[math.fsum(rho**abs(j*b+u-k*b-v)
        for u in range(b) for v in range(b))/(b*b) for k in range(a)] for j in range(a)]
    return b/(a-1)*(sum(covariance[j][j] for j in range(a))
                     - math.fsum(value for row in covariance for value in row)/a)


started = time.monotonic()
directory = Path(__file__).resolve().parent
run = directory.parent / 'mcse-calibration-01'
data = json.loads((run / 'result.json').read_text())
manifest = json.loads((run / 'manifest.json').read_text())
assert data['status'] == manifest['status'] == 'budget_exhausted'
for name, expected in data['source_sha256'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
assert hashlib.sha256((ROOT / 'scripts/calibrate_hmc_stopped_mcse.py').read_bytes()).hexdigest() == data['driver_sha256']
oracle_checks = []
for rho in (0., .9, -.9):
    for n, b in ((12, 2), (14, 3), (23, 5), (36, 6)):
        analytic = expected_batch_lrv(n, b, rho)
        direct = direct_batch_covariance_oracle(n, b, rho)
        assert math.isclose(analytic, direct, rel_tol=1e-12, abs_tol=1e-12)
        oracle_checks.append(dict(rho=rho, n=n, b=b, analytic=analytic, direct=direct))
rows = []
for index, cell in enumerate(data['cells']):
    records = cell['records']
    assert [r['replication'] for r in records] == list(range(len(records)))
    assert summarize(records, data['replications']) == cell['summary']
    interrupted = cell.get('interrupted_replicate')
    if interrupted is not None:
        assert interrupted['replication'] == len(records)
        assert interrupted['seed'] == [data['seed'], index//2, len(records)]
    for record in records:
        assert record['seed'] == [data['seed'], index//2, record['replication']]
        assert set(record['methods']) == set(METHODS)
        for method, endpoints in record['methods'].items():
            assert endpoints['fixed']['draws'] == 10000
            if endpoints['stopped'] is not None:
                assert endpoints['stopped']['passed']
                assert endpoints['stopped']['draws'] in LOOKS
        # The two batch policies coincide exactly at b=sqrt(10000)=100.
        for key in ('estimate', 'error', 'mcse', 'valid', 'interval_miss', 'passed'):
            assert record['methods']['lugsail']['fixed'][key] == record['methods']['even_lugsail']['fixed'][key]
    methods = {}
    for method in METHODS:
        entry = cell['summary']['methods'][method]
        ratios = [r['methods'][method]['fixed']['coordinate_variance_ratio'] for r in records
                  if r['methods'][method]['fixed']['coordinate_variance_ratio'] is not None]
        lengths = entry['endpoints']['stopped']['all_output_lengths']
        methods[method] = dict(summary=entry, mean_fixed_coordinate_variance_ratio=
            statistics.mean(ratios) if ratios else None, descriptive_stop_length_range_median=
            [min(lengths), statistics.median(lengths), max(lengths)] if lengths else None)
    rows.append(dict(law=cell['law'], rho=cell['rho'], planned=data['replications'],
        completed=len(records), interrupted=int(interrupted is not None),
        unstarted=data['replications']-len(records)-int(interrupted is not None), methods=methods))
expectations = []
for rho in (0., .9, -.9):
    for n in LOOKS:
        b = math.isqrt(n)
        small = b//3
        plain = expected_batch_lrv(n, b, rho)
        lugsail = 2*plain-expected_batch_lrv(n, small, rho)
        exact = finite_mean_variance(n, rho)
        expectations.append(dict(rho=rho, n=n, batch=b, small_batch=small,
            unmasked_plain_variance_ratio=plain/(4*n*exact),
            unmasked_lugsail_variance_ratio=lugsail/(4*n*exact),
            exact_coordinate_mcse_sd_ratio=math.sqrt(exact),
            exact_gaussian_square_mcse_sd_ratio=math.sqrt(finite_mean_variance(n, rho*rho))))
report = dict(status='passed_reference_and_accounting_audit', plan=data['plan'],
    result_file=str(run / 'result.json'), result_sha256=hashlib.sha256((run / 'result.json').read_bytes()).hexdigest(),
    source_hashes_checked=len(data['source_sha256']), source=data['source'], command=[sys.executable, *sys.argv],
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    gpu_intentionally_hidden=True, cpu_reference_only=True,
    completed_condition_records=sum(row['completed'] for row in rows),
    distinct_complete_latent_seed_units=len({tuple(r['seed']) for c in data['cells'] for r in c['records']}),
    rows=rows, covariance_oracle_checks=oracle_checks, finite_gaussian_expectations=expectations,
    pointwise_intervals_not_multiplicity_adjusted=True, method_ranking=None,
    estimator_default_promotion=False, nominal_stopped_coverage_established=False,
    elapsed_seconds=time.monotonic()-started, created_utc=datetime.now(timezone.utc).isoformat())
with (directory / 'result.json').open('x') as output:
    output.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
print(json.dumps({key: report[key] for key in ('status', 'source_hashes_checked',
    'completed_condition_records', 'distinct_complete_latent_seed_units', 'elapsed_seconds')}))
print(json.dumps([row for row in expectations if row['n'] == 10000], indent=2))
