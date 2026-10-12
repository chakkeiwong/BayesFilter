"""Read-only full-denominator audit of the bounded MCSE continuation."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path('/home/ubuntu/python/BayesFilter')
sys.path.insert(0, str(ROOT))
from scripts.calibrate_hmc_stopped_mcse import METHODS, LOOKS, summarize


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


started = time.monotonic()
directory = Path(__file__).resolve().parent
campaign = directory.parent
old_path = campaign / 'mcse-calibration-01/result.json'
new_path = campaign / 'mcse-calibration-02/result.json'
old, data = (json.loads(path.read_text()) for path in (old_path, new_path))
assert data['status'] == 'completed' and data['replications'] == 128
assert digest(old_path) == data['resume_from']['result_sha256']
assert data['source'] == old['source'] and data['source_sha256'] == old['source_sha256']
assert data['cumulative_elapsed_seconds'] == data['elapsed_seconds'] + old['elapsed_seconds']
assert digest(ROOT / 'scripts/calibrate_hmc_stopped_mcse.py') == data['driver_sha256']
for name, expected in data['source_sha256'].items():
    assert digest(name) == expected, name
rows = []
for index, (original, current) in enumerate(zip(old['cells'], data['cells'])):
    assert (original['law'], original['rho']) == (current['law'], current['rho'])
    assert len(current['records']) == 128
    assert current['records'][:len(original['records'])] == original['records']
    assert summarize(current['records'], 128) == current['summary']
    assert 'interrupted_replicate' not in current
    for i, record in enumerate(current['records']):
        assert record['replication'] == i and record['seed'] == [72110, index//2, i]
        assert set(record['methods']) == set(METHODS)
        for value in record['methods'].values():
            assert value['fixed']['draws'] == 10000
            if value['stopped'] is not None:
                assert value['stopped']['passed'] and value['stopped']['draws'] in LOOKS
    rows.append(dict(law=current['law'], rho=current['rho'], carried=len(original['records']),
        newly_completed=128-len(original['records']), summary=current['summary']))
assert len(rows) == 6
report = dict(status='passed_full_replication_audit', command=[sys.executable, *sys.argv],
    script_sha256=digest(__file__), result_file=str(new_path), result_sha256=digest(new_path),
    parent_file=str(old_path), parent_sha256=digest(old_path), source=data['source'],
    source_hashes_checked=len(data['source_sha256']), completed_condition_records=768,
    independent_latent_four_chain_units=384, rows=rows,
    child_worker_seconds=data['elapsed_seconds'], total_calibration_worker_seconds=data['cumulative_elapsed_seconds'],
    covariance_reference_file=str(campaign / 'mcse-analysis-01/result.json'),
    covariance_reference_sha256=digest(campaign / 'mcse-analysis-01/result.json'),
    gpu_intentionally_hidden=True, cpu_reference_only=True,
    pointwise_intervals_not_multiplicity_adjusted=True, method_ranking=None,
    estimator_default_promotion=False, nominal_stopped_coverage_established=False,
    elapsed_seconds=time.monotonic()-started, created_utc=datetime.now(timezone.utc).isoformat())
with (directory / 'result.json').open('x') as output:
    output.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
print(json.dumps({key: report[key] for key in ('status', 'source_hashes_checked',
    'completed_condition_records', 'independent_latent_four_chain_units',
    'child_worker_seconds', 'total_calibration_worker_seconds', 'elapsed_seconds')}))
