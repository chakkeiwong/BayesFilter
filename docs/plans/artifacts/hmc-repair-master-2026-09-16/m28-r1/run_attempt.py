"""Bounded M28 worker with explicit device policy and preserved attempt costs."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--name', required=True)
parser.add_argument('--seconds', type=float, required=True)
parser.add_argument('--source-root', type=Path, required=True)
parser.add_argument('--device', choices=('cpu_reference', 'gpu'), default='cpu_reference')
parser.add_argument('--gpu', type=int)
parser.add_argument('command', nargs=argparse.REMAINDER)
args = parser.parse_args()
command = args.command[1:] if args.command[:1] == ['--'] else args.command
if not command or args.seconds <= 0 or (args.device == 'gpu' and args.gpu is None):
    raise ValueError('command, positive timeout and explicit GPU index for GPU work required')
root = Path(__file__).resolve().parent / args.name
root.mkdir(parents=True, exist_ok=False)
source = args.source_root.resolve()
env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu) if args.device == 'gpu' else '-1',
           TF_FORCE_GPU_ALLOW_GROWTH='true', TF_NUM_INTRAOP_THREADS='1',
           TF_NUM_INTEROP_THREADS='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
           BAYESFILTER_PRELOAD_CUSTOM_OP='0', PYTHONPATH=str(source))
snapshot = source / 'source_snapshot.json'
commit = (json.loads(snapshot.read_text())['git_commit'] if snapshot.exists() else
          subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip())
manifest = {'command': command, 'cwd': str(source), 'git_commit': commit,
    'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'source_hashes': {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in sorted((source / 'bayesfilter').rglob('*.py'))},
    'environment': {k: env[k] for k in ('CUDA_VISIBLE_DEVICES', 'TF_FORCE_GPU_ALLOW_GROWTH',
       'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
    'device_scope': args.device, 'gpu_intentionally_hidden': args.device == 'cpu_reference',
    'plan_file': 'docs/plans/bayesfilter-hmc-post-m27-next-phase-2026-09-23.md',
    'execution_note': 'docs/plans/bayesfilter-hmc-m28-execution-note-2026-09-23.md',
    'data_version': 'declared analytic laws; see resolved design and child manifest',
    'result_file': str(root / 'execution.json'), 'timeout_seconds': args.seconds}
(root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
started = time.monotonic()
failure = None
with (root / 'worker.log').open('x') as log:
    try:
        if args.device == 'gpu':
            # Shared academic host: observe device ownership before this local run.
            device = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu),
                '--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu',
                '--format=csv,noheader'], text=True).strip()
            processes = subprocess.check_output(['nvidia-smi',
                '--query-compute-apps=gpu_uuid,pid,process_name', '--format=csv,noheader'], text=True).strip()
            manifest['gpu_preflight'] = {'device': device, 'compute_processes': processes,
                'trust_basis': 'trusted_escalated_tool_execution', 'memory_growth_before_import': True}
            (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
            selected_uuid = device.split(',')[1].strip()
            display_processes = {'/usr/libexec/gnome-remote-desktop-daemon', '/usr/NX/bin/nxnode.bin'}
            if any(line.split(',')[0].strip() == selected_uuid
                   and line.split(',')[2].strip() not in display_processes
                   for line in processes.splitlines()):
                raise RuntimeError('selected GPU already has a compute process')
        process = subprocess.Popen(command, cwd=source, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        code = process.wait(timeout=args.seconds)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        code = 124
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        code, failure = 127, str(exc)
elapsed = time.monotonic() - started
result = {'command': command, 'exit_code': code, 'elapsed_seconds': elapsed,
          'cpu_worker_seconds': elapsed if args.device == 'cpu_reference' else 0,
          'gpu_worker_seconds': elapsed if args.device == 'gpu' else 0,
          'manifest': str(root / 'manifest.json'), 'launch_failure': failure}
(root / 'execution.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result), flush=True)
print((root / 'worker.log').read_text()[-1600:], flush=True)
raise SystemExit(code)
