"""Meter a single bounded CPU diagnostic child and preserve every outcome."""
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
parser.add_argument('--seconds', type=float, required=True)
parser.add_argument('command', nargs=argparse.REMAINDER)
args = parser.parse_args()
root = Path(__file__).resolve().parent / args.name
root.mkdir(parents=True, exist_ok=False)
repo = Path(__file__).resolve().parents[5]
command = args.command[1:] if args.command[:1] == ['--'] else args.command
if not command or args.seconds <= 0:
    raise ValueError('positive timeout and command required')
env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', TF_FORCE_GPU_ALLOW_GROWTH='true',
           TF_NUM_INTRAOP_THREADS='1', TF_NUM_INTEROP_THREADS='1', OMP_NUM_THREADS='1',
           OPENBLAS_NUM_THREADS='1', BAYESFILTER_PRELOAD_CUSTOM_OP='0',
           PYTHONPATH=str(repo))
record = {'command': command, 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
          'source_hashes': {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (repo/'bayesfilter').rglob('*.py')},
          'environment': {k: env[k] for k in ('CUDA_VISIBLE_DEVICES', 'TF_FORCE_GPU_ALLOW_GROWTH',
              'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
          'plan_file': 'docs/plans/bayesfilter-hmc-gap-closure-continuation-2026-09-22.md',
          'result_file': str(root / 'execution.json'), 'timeout_seconds': args.seconds,
          'gpu_intentionally_hidden': True, 'data_version': 'synthetic or saved M20 evidence; see child manifest'}
(root/'manifest.json').write_text(json.dumps(record, indent=2) + '\n')
started = time.monotonic()
with (root/'worker.log').open('x') as log:
    try:
        completed = subprocess.run(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   timeout=args.seconds)
        code = completed.returncode
    except subprocess.TimeoutExpired:
        code = 124
result = {'command': command, 'exit_code': code, 'cpu_worker_seconds': time.monotonic()-started,
          'gpu_worker_seconds': 0, 'manifest': str(root/'manifest.json')}
(root/'execution.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result), flush=True)
print((root/'worker.log').read_text()[-3500:], flush=True)
raise SystemExit(code)
