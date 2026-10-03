"""Combine unique completed M21 and M25 worker receipts without double charging."""
import datetime
import argparse
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', default='reconciliation-terminal.json')
args = parser.parse_args()
master = root.parent
base_path = master/'m21-r2/reconciliation-progress.json'
base = json.loads(base_path.read_text())
records = list(base['records'])
keys = {r['path'] for r in records}
assert len(keys) == len(records)
m25_cpu = m25_gpu = 0.
for path in sorted(root.glob('*/execution.json')):
    receipt = json.loads(path.read_text())
    relative = str(path.relative_to(master))
    assert relative not in keys
    keys.add(relative)
    cpu, gpu = receipt['cpu_worker_seconds'], receipt['gpu_worker_seconds']
    assert cpu >= 0 and gpu >= 0
    if gpu:
        assert cpu == 0 and receipt['gpu_charge_includes_host_work']
    records.append({'phase': 'M25', 'path': relative, 'cpu_seconds': cpu,
        'gpu_seconds': gpu, 'exit_code': receipt['exit_code'],
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'note': 'measured worker wall time including failed attempts; GPU includes host work'})
    m25_cpu += cpu
    m25_gpu += gpu
cpu = sum(r['cpu_seconds'] for r in records)
gpu = sum(r['gpu_seconds'] for r in records)
assert m25_cpu <= 12000 and m25_gpu <= 4200
assert cpu <= 172800 and gpu <= 86400
result = {'schema': 'bayesfilter.hmc_campaign_reconciliation.v1',
    'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'source_ledger': str(base_path), 'source_ledger_sha256': hashlib.sha256(base_path.read_bytes()).hexdigest(),
    'terminal_for_m25': True, 'records': records,
    'm25': {'cpu_worker_seconds': m25_cpu, 'gpu_worker_seconds': m25_gpu,
        'cpu_limit_seconds': 12000, 'gpu_limit_seconds': 4200},
    'budget_seconds': {'opening': {'cpu_reference': 172800, 'gpu': 86400},
        'charged': {'cpu_reference': cpu, 'gpu': gpu},
        'remaining': {'cpu_reference': 172800-cpu, 'gpu': 86400-gpu}, 'exceeded': False},
    'uncompleted_queue_reservations_seconds': base['uncompleted_queue_reservations_seconds'],
    'remaining_excludes_future_reservations': True,
    'numerical_worker_exception': 'ended after terminal M25 audit',
    'non_numerical_git_document_build_waiting_not_charged_as_numerical_workers': True}
with (root/args.output).open('x') as f:
    json.dump(result, f, indent=2)
    f.write('\n')
print(json.dumps({k: v for k, v in result.items() if k != 'records'}, indent=2))
