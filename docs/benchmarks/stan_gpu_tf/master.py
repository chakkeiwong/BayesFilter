"""Execute the reviewed GPU/TF Stan implementation and evidence program."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__).resolve().parent
PLAN = ROOT/'docs/plans/bayesfilter-stan-gpu-tf-master-2026-10-28.md'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'gpu', 'both'], default='gpu')
    p.add_argument('--gpu', default='0')
    p.add_argument('--command-buffer', action='store_true',
        help='Use XLA GPU command buffers including conditional/while control flow')
    p.add_argument('--runtime-log', action='store_true', help='Record non-profiled XLA command-buffer execution')
    p.add_argument('--phase', choices=['tests', 'validate', 'benchmark', 'diagnose', 'probe', 'qualification', 'all'], default='all')
    p.add_argument('--oracle', type=Path)
    p.add_argument('--diagnostic-runs', type=Path, nargs='*', default=[])
    p.add_argument('--targets', default='gaussian,banana,funnel,regression')
    p.add_argument('--warmup', type=int, default=500)
    p.add_argument('--draws', type=int, default=500)
    p.add_argument('--depth', type=int, default=10)
    args = p.parse_args(); out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    prior = []
    for path in sorted(out.parent.glob('*/manifest.json')):
        record = json.loads(path.read_text())
        if record.get('plan') != str(PLAN): continue
        if record.get('status') == 'running':
            raise RuntimeError('another campaign phase is running: '+str(path))
        prior.extend(record.get('attempts', []))
    manifest = dict(command=sys.argv, plan=str(PLAN), python=sys.executable,
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        attempts=[], source_hashes={}, status='running', budget_seconds=2400, max_attempts=12,
        prior_worker_seconds=sum(a.get('wall_seconds', a.get('timeout_seconds', 0.)) for a in prior),
        prior_attempts=len(prior))
    manifest['started_utc'] = datetime.now(timezone.utc).isoformat()
    manifest['result_note'] = 'docs/plans/bayesfilter-stan-gpu-tf-result-2026-10-28.md'
    manifest['benchmark_seed'] = 20261028
    manifest['data_version'] = 'analytic fixtures; regression stateless seed 20261027,61'
    snapshot = out/'source'; snapshot.mkdir()
    sources = [*SOURCE.glob('*.py'), *SOURCE.glob('*.md'), PLAN,
        ROOT/'docs/benchmarks/stan_nuts.py', ROOT/'docs/benchmarks/stan_nuts_warmup.py',
        ROOT/'tests/test_stan_gpu_tf.py', ROOT/'tests/test_stan_nuts_port.py',
        ROOT/'tests/test_stan_nuts_warmup.py', ROOT/'docs/benchmarks/stan_transition_oracle.cpp',
        ROOT/'docs/benchmarks/stan_adaptation_oracle.cpp', ROOT/'docs/benchmarks/stan_service_oracle.cpp',
        ROOT/'docs/benchmarks/benchmark_nuts_xla.py', ROOT/'docs/benchmarks/benchmark_stan_transplant.py',
        ROOT/'docs/benchmarks/audit_full_stan.py', ROOT/'docs/benchmarks/stan_targets.stan',
        ROOT/'docs/benchmarks/native_sampler_bridge/validate.py', ROOT/'docs/benchmarks/native_sampler_bridge/wrapper.py',
        ROOT/'bayesfilter/inference/hmc_ess.py', ROOT/'bayesfilter/inference/hmc_posterior_diagnostics.py']
    for source in sources:
        name = str(source.relative_to(ROOT))
        manifest['source_hashes'][name] = hashlib.sha256(source.read_bytes()).hexdigest()
        dest = snapshot/name; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, dest)
        if hashlib.sha256(dest.read_bytes()).hexdigest() != manifest['source_hashes'][name]:
            raise RuntimeError('source snapshot mismatch: '+name)
    binaries = [ROOT/'docs/benchmarks/stan_targets',
                ROOT/'.localresources/native_sampler_bridge/libbf_native.so']
    manifest['reference_binary_hashes'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                           for p in binaries if p.exists()}
    def save():
        manifest['wall_seconds'] = time.monotonic()-start
        (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    def run(name, command, device, limit):
        remaining = 2400-manifest['prior_worker_seconds']-sum(a.get('wall_seconds', 0.) for a in manifest['attempts'])
        if remaining <= 0 or manifest['prior_attempts']+len(manifest['attempts']) >= 12:
            raise RuntimeError('campaign compute/attempt budget exhausted')
        limit = min(limit, remaining)
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '-1' if device == 'cpu' else args.gpu,
            'TF_FORCE_GPU_ALLOW_GROWTH': 'true', 'BAYESFILTER_PRELOAD_CUSTOM_OP': '0',
            'BAYESFILTER_STAN_AUDIT_OUTPUT': str(out/'oracle'),
            'BAYESFILTER_STAN_PORT_OUTPUT': str(out/'oracle'),
            'BAYESFILTER_REQUIRE_STAN_ORACLE': '1', 'PYTHONPATH': str(ROOT)}
        if device == 'gpu' and args.command_buffer:
            if 'xla_gpu_enable_command_buffer' in env.get('XLA_FLAGS', ''):
                raise ValueError('conflicting inherited XLA command-buffer setting')
            env['XLA_FLAGS'] = env.get('XLA_FLAGS', '')+' --xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUBLASLT,CUDNN,WHILE,CONDITIONAL'
        if args.runtime_log:
            env['TF_CPP_VMODULE'] = 'command_buffer_thunk=3'
            env['TF_CPP_MIN_LOG_LEVEL'] = '0'
        (out/'oracle').mkdir(exist_ok=True)
        log = out/(name+'.log'); begin = time.monotonic()
        entry = dict(name=name, command=command, device=device, log=str(log),
                     timeout_seconds=limit, status='running', xla_flags=env.get('XLA_FLAGS', ''))
        manifest['attempts'].append(entry); save(); print('Running '+name, flush=True)
        try:
            with log.open('w') as handle:
                proc = subprocess.Popen(command, env=env, cwd=ROOT, stdout=handle,
                    stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    rc = proc.wait(timeout=limit)
                except subprocess.TimeoutExpired:
                    import signal
                    os.killpg(proc.pid, signal.SIGTERM)
                    try: proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL); proc.wait()
                    raise
            entry['returncode'] = rc; entry['status'] = 'passed' if rc == 0 else 'failed'
            if rc: raise RuntimeError(name+' failed; see '+str(log))
        except Exception as error:
            entry['status'] = 'failed'; entry['error'] = repr(error); raise
        finally:
            entry['wall_seconds'] = time.monotonic()-begin; save()
    try:
        if args.phase == 'probe':
            run('gpu-command-buffer-probe', [sys.executable, '-m',
                'docs.benchmarks.stan_gpu_tf.command_buffer_probe'], 'gpu', 60)
        if args.phase in ('tests', 'all'):
            run('cpu-tests', [sys.executable, '-m', 'pytest', '-q',
                'tests/test_stan_nuts_port.py', 'tests/test_stan_nuts_warmup.py',
                'tests/test_stan_gpu_tf.py'], 'cpu', 600)
        oracle = args.oracle or out/'oracle/transition-calls.jsonl'
        devices = ['cpu', 'gpu'] if args.device == 'both' else [args.device]
        for device in devices:
            if args.phase == 'qualification':
                run(device+'-qualification', [sys.executable, '-m', 'docs.benchmarks.stan_gpu_tf.worker',
                    '--device', device, '--gpu', args.gpu, '--mode', 'qualification',
                    '--oracle', str(oracle), '--targets', args.targets, '--warmup', str(args.warmup),
                    '--draws', str(args.draws), '--depth', str(args.depth),
                    '--output', str(out/(device+'-qualification'))], device, 900)
            if args.phase in ('validate', 'all'):
                run(device+'-validation', [sys.executable, '-m', 'docs.benchmarks.stan_gpu_tf.worker',
                    '--device', device, '--gpu', args.gpu, '--mode', 'validate',
                    '--oracle', str(oracle), '--output', str(out/(device+'-validation'))], device, 420)
            if args.phase in ('benchmark', 'all'):
                run(device+'-benchmark', [sys.executable, '-m', 'docs.benchmarks.stan_gpu_tf.worker',
                    '--device', device, '--gpu', args.gpu, '--mode', 'benchmark',
                    '--targets', args.targets, '--warmup', str(args.warmup), '--draws', str(args.draws),
                    '--depth', str(args.depth), '--output', str(out/(device+'-benchmark'))], device, 900)
        if args.phase in ('diagnose', 'all'):
            runs = args.diagnostic_runs or [out/(device+'-benchmark') for device in devices]
            run('posterior-diagnostics', [sys.executable, '-m', 'docs.benchmarks.stan_gpu_tf.diagnose',
                '--output', str(out/'diagnostics'), *[str(path) for path in runs]], 'cpu', 180)
        manifest['status'] = 'completed'
    except Exception as error:
        manifest['status'] = 'failed'; manifest['error'] = repr(error); raise
    finally: save()
    print(json.dumps({'output': str(out), 'status': manifest['status'],
                      'worker_seconds': sum(a['wall_seconds'] for a in manifest['attempts'])}), flush=True)


if __name__ == '__main__': main()
