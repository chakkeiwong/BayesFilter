"""Meter one trusted GPU supplied-map fit on a frozen source snapshot."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--name', required=True)
parser.add_argument('--map', choices=('exact', 'partial', 'partial_half'), required=True)
parser.add_argument('--seed', type=int, required=True)
args = parser.parse_args()
base = Path(__file__).resolve().parent
root = base/args.name
root.mkdir(exist_ok=False)
probe_command = ['/home/ubuntu/.codex/bin/codex-gpu-probe', '--framework', 'nvidia', '--gpu', '1']
probe_run = subprocess.run(probe_command, capture_output=True, text=True)
probe = json.loads(probe_run.stdout)
(root/'capacity.json').write_text(json.dumps(probe, indent=2)+'\n')
if probe.get('status') != 'PASSED' or probe.get('selected_host_gpu') != 1:
    raise RuntimeError('pinned GPU 1 is not permitted and idle')
source = base/'source-r2'
source_manifest = json.loads((base/'source-manifest-r2.json').read_text())
for relative, expected in source_manifest['files'].items():
    if hashlib.sha256((source/relative).read_bytes()).hexdigest() != expected:
        raise RuntimeError('frozen source mismatch: '+relative)
driver = source/'docs/benchmarks/run_hmc_supplied_funnel_maps_2026_09_22.py'
command = [sys.executable, str(driver), '--output', str(root/'worker'), '--device', 'gpu',
    '--map', args.map, '--seed', str(args.seed), '--seconds', '590', '--search',
    'native' if args.map == 'exact' else 'intermediate_grid']
env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', TF_FORCE_GPU_ALLOW_GROWTH='true',
    BAYESFILTER_PRELOAD_CUSTOM_OP='0', PYTHONPATH=str(source), TF_NUM_INTRAOP_THREADS='1',
    TF_NUM_INTEROP_THREADS='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
manifest = {'command': command, 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'plan_file': 'docs/plans/bayesfilter-hmc-gap-closure-continuation-2026-09-22.md',
    'source_identity': source_manifest['source_identity'], 'git_commit': source_manifest['git_commit'],
    'environment': {k:env[k] for k in ('CUDA_VISIBLE_DEVICES','TF_FORCE_GPU_ALLOW_GROWTH',
        'TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS')},
    'gpu': 1, 'jit_compile': True, 'trusted_escalated_launch': True,
    'seed': args.seed, 'map': args.map, 'driver_sha256': hashlib.sha256(driver.read_bytes()).hexdigest(),
    'data_version': 'analytic scale-3 funnel with two children', 'hard_timeout_seconds': 600,
    'result_file': str(root/'execution.json')}
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
started = time.monotonic()
with (root/'worker.log').open('x') as log:
    try:
        completed = subprocess.run(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=600)
        code = completed.returncode
    except subprocess.TimeoutExpired:
        code = 124
elapsed = time.monotonic()-started
record = {'command': command, 'exit_code': code, 'gpu_worker_seconds': elapsed,
    'cpu_worker_seconds': 0, 'gpu_charge_includes_host_work': True,
    'manifest': str(root/'manifest.json')}
(root/'execution.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record), flush=True)
print((root/'worker.log').read_text()[-2500:], flush=True)
raise SystemExit(code)
