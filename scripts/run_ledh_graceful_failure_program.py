"""Bounded command driver for the adopted LEDH program (standard library only).

This records actual commands and resources; it does not issue tuning or admission
artifacts. Run a planned stage with an explicit timeout and fresh label. GPU
commands require trusted execution and a selected physical-device UUID.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--phase', required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--device', choices=('cpu', 'gpu'), required=True)
    parser.add_argument('--gpu-uuid')
    parser.add_argument('--timeout', type=float, required=True)
    parser.add_argument('--case', help='Counts a substantive case attempt, not a unit-test iteration')
    parser.add_argument('--seed', action='append', default=[])
    parser.add_argument('--data-id', default='N/A: deterministic engineering fixture')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or args.timeout <= 0:
        parser.error('a command and positive timeout are required')
    if args.device == 'gpu' and not args.gpu_uuid:
        parser.error('GPU execution requires --gpu-uuid')
    campaign = args.campaign.resolve()
    ledger_path = campaign/'campaign.json'
    ledger = json.loads(ledger_path.read_text())
    if args.device == 'gpu' and args.timeout > ledger['limits']['single_gpu_job_seconds']:
        parser.error('timeout exceeds per-job GPU allocation')
    resource_name = 'gpu_device_seconds' if args.device == 'gpu' else 'cpu_worker_seconds'
    remaining = ledger['limits'][resource_name] - ledger['charged'][resource_name]
    # CPU tasks explicitly use two intra-op workers and one inter-op worker.
    conservative_cost = args.timeout * (1 if args.device == 'gpu' else 3)
    if conservative_cost > remaining:
        parser.error('requested timeout does not fit the remaining resource allocation')
    if args.case:
        attempts = sum(item['case'] == args.case for item in ledger['attempts'])
        if attempts >= ledger['limits']['substantive_attempts_per_case']:
            parser.error('case attempt ceiling exhausted')
        ledger['attempts'].append({'case':args.case, 'label':args.label, 'attempt':attempts+1})
    run_dir = campaign/args.label
    run_dir.mkdir(exist_ok=False)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid if args.device == 'gpu' else '-1',
               TF_FORCE_GPU_ALLOW_GROWTH='true', TF_NUM_INTRAOP_THREADS='2',
               TF_NUM_INTEROP_THREADS='1', OMP_NUM_THREADS='2',
               PYTHONDONTWRITEBYTECODE='1', TF_CPP_MIN_LOG_LEVEL='2')
    repo = Path.cwd().resolve()
    # Explicit repository import root also covers diagnostic scripts below docs/.
    env['PYTHONPATH'] = str(repo) + (os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    changed = set(subprocess.check_output(['git','diff','HEAD','--name-only'],text=True).splitlines())
    changed.update(subprocess.check_output(['git','ls-files','--others','--exclude-standard'],text=True).splitlines())
    # Exclude growing run outputs; preserve all executable sources and plans.
    changed = sorted(name for name in changed if name.endswith(('.py','.sh','.md')) or not name.startswith('docs/plans/artifacts/'))
    env['LEDH_PROGRAM_RESULT_DIR'] = str(run_dir)
    manifest = {'label':args.label, 'phase':args.phase,
        'command':command, 'cwd':str(repo),
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'dirty_source_sha256':{name:hashlib.sha256((repo/name).read_bytes()).hexdigest()
            for name in changed if (repo/name).is_file()},
        'environment':str(Path(sys.executable).parent.parent),
        'device':args.device, 'gpu_uuid':args.gpu_uuid,
        'gpu_intentionally_hidden':args.device == 'cpu',
        'environment_overrides':{key:env[key] for key in ('CUDA_VISIBLE_DEVICES','TF_FORCE_GPU_ALLOW_GROWTH','TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS','OMP_NUM_THREADS')},
        'seeds':args.seed, 'data_version':args.data_id,
        'plan_file':ledger['program'], 'result_file':str(run_dir/'manifest.json'),
        'log_file':str(run_dir/'command.log'), 'timeout_seconds':args.timeout,
        'started_utc':datetime.now(timezone.utc).isoformat(), 'status':'running'}
    ledger['runs'].append({'label':args.label, 'manifest':str(run_dir/'manifest.json'), 'status':'running'})
    ledger_path.write_text(json.dumps(ledger,indent=2)+'\n')
    manifest_path = run_dir/'manifest.json'
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.monotonic()
    with (run_dir/'command.log').open('x') as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            returncode = process.wait(timeout=args.timeout)
            status = 'passed' if returncode == 0 else 'failed'
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            returncode, status = 124, 'timeout'
    wall = time.monotonic() - start
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime
    gpu = wall if args.device == 'gpu' else 0.0
    manifest.update(status=status, exit_code=returncode, wall_seconds=wall,
                    cpu_worker_seconds=cpu, gpu_device_seconds=gpu,
                    log_sha256=hashlib.sha256((run_dir/'command.log').read_bytes()).hexdigest())
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    ledger['runs'][-1]['status'] = status
    ledger['charged']['cpu_worker_seconds'] += cpu
    ledger['charged']['gpu_device_seconds'] += gpu
    ledger_path.write_text(json.dumps(ledger,indent=2)+'\n')
    print(json.dumps({'status':status,'exit_code':returncode,'wall_seconds':wall,
                     'cpu_worker_seconds':cpu,'manifest':str(manifest_path),
                     'charged':ledger['charged']}), flush=True)
    return returncode


if __name__ == '__main__':
    raise SystemExit(main())
