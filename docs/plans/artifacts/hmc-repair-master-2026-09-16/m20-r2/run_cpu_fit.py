"""One metered CPU reference attempt; preserves every outcome and timeout."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--map', choices=('exact', 'partial', 'partial_half'), required=True)
parser.add_argument('--seed', type=int, required=True)
parser.add_argument('--attempt', default='r1')
parser.add_argument('--search', choices=('native', 'intermediate_grid'), default='native')
args = parser.parse_args()
repo = Path(__file__).resolve().parents[5]
group = 'fits-cpu-r1' if args.search == 'native' else 'fits-cpu-grid-r1'
root = Path(__file__).resolve().parent / group / f'{args.map}-{args.seed}-{args.attempt}'
root.mkdir(parents=True, exist_ok=False)
command = [sys.executable, str(repo / 'docs/benchmarks/run_hmc_supplied_funnel_maps_2026_09_22.py'),
           '--output', str(root / 'worker'), '--device', 'cpu_reference',
           '--seed', str(args.seed), '--map', args.map, '--seconds', '350', '--search', args.search]
env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', TF_FORCE_GPU_ALLOW_GROWTH='true',
           TF_NUM_INTRAOP_THREADS='1', TF_NUM_INTEROP_THREADS='1', OMP_NUM_THREADS='1',
           OPENBLAS_NUM_THREADS='1', BAYESFILTER_PRELOAD_CUSTOM_OP='0')
start = time.monotonic()
receipt = {'command': command, 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'timeout_seconds': 360, 'gpu_intentionally_hidden': True, 'cpu_threads': 1,
           'map': args.map, 'seed': args.seed,
           'plan_file': 'docs/plans/bayesfilter-hmc-supplied-whitening-plan-2026-09-22.md'}
with (root / 'worker.log').open('w') as log:
    try:
        proc = subprocess.run(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=360)
        code = proc.returncode
    except subprocess.TimeoutExpired:
        code = 124
receipt.update(exit_code=code, wall_seconds=time.monotonic()-start)
(root / 'execution.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(receipt), flush=True)
result = root / 'worker/result.json'
if result.exists():
    body = json.loads(result.read_text())
    body['verified_count'] = len(body.pop('verified_candidate_ids'))
    print(json.dumps(body), flush=True)
else:
    print((root / 'worker.log').read_text()[-3000:], flush=True)
sys.exit(code)
