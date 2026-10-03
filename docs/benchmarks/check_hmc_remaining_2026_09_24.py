"""Bounded command receipts and device policy for the HMC repair program."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, required=True)
    parser.add_argument('--device', choices=('cpu_reference','gpu'), default='cpu_reference')
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--role', choices=('development', 'confirmation', 'engineering'), default='development')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or args.seconds <= 0 or (args.device=='gpu' and args.gpu is None):
        parser.error('a positive timeout and command are required')
    args.output.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(args.gpu) if args.device=='gpu' else '-1', 'TF_FORCE_GPU_ALLOW_GROWTH': 'true',
           'TF_NUM_INTRAOP_THREADS': '1', 'TF_NUM_INTEROP_THREADS': '1', 'OMP_NUM_THREADS': '1',
           'OPENBLAS_NUM_THREADS': '1', 'TF_CPP_MIN_LOG_LEVEL': '2',
           'PYTHONPATH': str(args.source.resolve()), 'MPLCONFIGDIR': str(args.output.resolve() / 'mpl')}
    manifest = {'command': command, 'cwd': str(args.source.resolve()),
                'started_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                'timeout_seconds': args.seconds, 'execution_role': args.role,
                'supervisor_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'device':args.device,
                'gpu': 'trusted execution with growth before import' if args.device=='gpu' else 'intentionally hidden before framework import',
                'environment': {k: env[k] for k in ('CUDA_VISIBLE_DEVICES', 'TF_FORCE_GPU_ALLOW_GROWTH',
                    'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS', 'PYTHONPATH')},
                'plan_file': 'docs/plans/bayesfilter-hmc-remaining-gap-program-2026-09-24.md',
                'result_file': str(args.output / 'execution.json')}
    started = time.monotonic()
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    try:
        run(args, command, env, manifest, started)
    except Exception as error:
        result = {'exit_code': 1, 'timed_out': False,
                  'elapsed_seconds': time.monotonic() - started,
                  'resource': args.device, 'failure_class': 'infrastructure',
                  'error': f'{type(error).__name__}: {error}'}
        (args.output / 'execution.json').write_text(json.dumps(result, indent=2) + '\n')
        raise


def run(args, command, env, manifest, started):
    if args.device=='gpu':
        device=subprocess.check_output(['nvidia-smi','-i',str(args.gpu),
            '--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],text=True).strip()
        processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name',
                                          '--format=csv,noheader'],text=True).strip()
        manifest['gpu_preflight']={'device':device,'processes':processes,'trust_basis':'trusted_escalated_tool_execution'}
        uuid=device.split(',')[1].strip()
        if any(line.split(',')[0].strip()==uuid and line.split(',')[2].strip()
               not in {'/usr/libexec/gnome-remote-desktop-daemon','/usr/NX/bin/nxnode.bin'}
               for line in processes.splitlines()):
            raise RuntimeError('selected GPU already has a compute process')
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    timed_out = False
    with (args.output / 'worker.log').open('w') as log:
        process = subprocess.Popen(command, cwd=args.source, env=env,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=args.seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                code = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                code = process.wait()
    result = {'exit_code': code, 'timed_out': timed_out,
              'elapsed_seconds': time.monotonic() - started, 'resource': args.device,
              'ended_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'log': str(args.output / 'worker.log')}
    (args.output / 'execution.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    raise SystemExit(code if not timed_out else 124)


if __name__ == '__main__':
    main()
