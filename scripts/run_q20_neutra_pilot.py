"""Bounded, versioned q20 pilot with carried-forward campaign accounting."""
import argparse
from datetime import datetime, timezone
import fcntl
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
PLAN = 'docs/plans/bayesfilter-q20-naf-forward-reverse-plan-2026-10-06.md'
ARTIFACTS = ROOT/'docs/plans/artifacts/q20-naf-forward-reverse-2026-10-06'
SHARED = ROOT/'docs/plans/artifacts/neutra-warm-start-master-2026-09-29/campaign-r1'
INITIAL = {'gpu_process_seconds':111947.28170388959,'cpu_core_seconds':104304.108478}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def worker(args):
    output = args.output
    started, cpu = time.monotonic(), time.process_time()
    manifest = dict(command=sys.argv, plan=PLAN, started_utc=datetime.now(timezone.utc).isoformat(),
        python=sys.executable, source=str(ROOT), seed=args.seed, gpu=args.gpu,
        cpu_sample_generation='explicit_CPU_two_threads', phase=args.phase,
        TF_FORCE_GPU_ALLOW_GROWTH=os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH'),
        jit_compile=True, scientific_promotion=False)
    try:
        faulthandler.dump_traceback_later(60, repeat=True)
        if os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH') != 'true':
            raise ValueError('memory growth must precede framework import')
        import tensorflow as tf
        from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
        manifest['memory_policy'] = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
        manifest['tensorflow'] = tf.__version__
        manifest['tf32'] = tf.config.experimental.tensor_float_32_execution_enabled()
        manifest['dtype'] = 'float64_diagnostic_exception'
        write(output/'manifest.json',manifest)
        from bayesfilter.testing import q20_neutra_pilot as pilot
        with tf.device('/GPU:0'):
            if args.phase == 'price':
                result = pilot.price(output)
            elif args.phase == 'reuse-smc':
                result = pilot.reuse_smc(output,args.teacher,args.data_root)
            else:
                result = pilot.fit(output,args.teacher,args.seed,extended=args.phase=='fit-extended')
        manifest['status'] = 'complete'
        manifest['result_status'] = result['status']
    except BaseException as error:
        manifest.update(status='failed',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        manifest.update(wall_seconds=time.monotonic()-started,
            cpu_core_seconds=time.process_time()-cpu)
        manifest['artifact_sha256'] = {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for p in output.iterdir() if p.is_file() and p.name not in ('manifest.json','worker.log')}
        write(output/'manifest.json',manifest)


def launch(args):
    ARTIFACTS.mkdir(parents=True,exist_ok=True)
    with (ARTIFACTS/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state_path = ARTIFACTS/'state.json'
        state = read(state_path) if state_path.exists() else dict(initial_remaining=INITIAL,attempts=[],active=None)
        if state.get('active'):
            raise ValueError('unsettled active attempt; reconcile before resuming')
        remaining = {k:INITIAL[k]-sum(r[k] for r in state['attempts']) for k in INITIAL}
        cap = {'price':900,'reuse-smc':450,'fit':4200,'fit-extended':4200}[args.phase]
        if sum(r['gpu_process_seconds'] for r in state['attempts'])+cap>7200:
            raise ValueError('initial pilot exposure cap requires a measured continuation decision')
        if remaining['gpu_process_seconds'] < cap or remaining['cpu_core_seconds'] < 2*cap:
            raise ValueError('cannot reserve bounded worker within remaining budget')
        output = ARTIFACTS/f'{args.phase}-{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")}'
        output.mkdir()
        source = output/'source'
        source.mkdir()
        shutil.copytree(ROOT/'bayesfilter',source/'bayesfilter',
            ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        (source/'scripts').mkdir()
        shutil.copy2(Path(__file__),source/'scripts'/Path(__file__).name)
        hashes = {str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source.rglob('*') if p.is_file()}
        commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        write(output/'source.json',dict(git_commit=commit,dirty_source_preserved=True,sha256=hashes))
        env = os.environ.copy()
        env.update(PYTHONPATH=str(source),TF_FORCE_GPU_ALLOW_GROWTH='true',CUDA_VISIBLE_DEVICES=args.gpu,
            XLA_PYTHON_CLIENT_PREALLOCATE='false',TF_NUM_INTRAOP_THREADS='2',TF_NUM_INTEROP_THREADS='1',
            OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
        command = [sys.executable,str(source/'scripts'/Path(__file__).name),args.phase,
            '--worker','--output',str(output),'--gpu',args.gpu,'--seed',str(args.seed),
            '--data-root',str(ROOT)]
        if args.teacher:
            command += ['--teacher',str(args.teacher.resolve())]
        started = time.monotonic()
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        row = dict(output=str(output),phase=args.phase,command=command,git_commit=commit,
            wall_limit=cap,started_utc=datetime.now(timezone.utc).isoformat())
        with (output/'worker.log').open('w') as log:
            child = subprocess.Popen(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state['active'] = {**row,'pid':child.pid}
            write(state_path,state)
            print(json.dumps(state['active']),flush=True)
            try:
                code = child.wait(timeout=cap)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL)
                    child.wait()
                code = 124
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        row.update(exit_code=code,status='complete' if code==0 else 'failed',
            wall_seconds=time.monotonic()-started,gpu_process_seconds=time.monotonic()-started,
            cpu_core_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime)
        state['attempts'].append(row)
        state['active'] = None
        state['remaining'] = {k:INITIAL[k]-sum(r[k] for r in state['attempts']) for k in INITIAL}
        write(state_path,state)
        with (SHARED/'q20-accounting.lock').open('a') as shared_lock:
            fcntl.flock(shared_lock,fcntl.LOCK_EX)
            shared = read(SHARED/'state.json')
            if not any(r.get('output')==str(output) for r in shared['attempts']):
                shared['attempts'].append({**row,'job':output.name,'device':'gpu'})
                write(SHARED/'state.json',shared)
        print(json.dumps(dict(status=row['status'],output=str(output),remaining=state['remaining'])),flush=True)
        return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('price','reuse-smc','fit','fit-extended'))
    parser.add_argument('--gpu',default='0')
    parser.add_argument('--seed',type=int,default=61007)
    parser.add_argument('--teacher',type=Path)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--data-root',type=Path,default=ROOT)
    args = parser.parse_args()
    if not args.gpu.isdecimal():
        parser.error('one physical GPU index is required')
    if args.phase in ('fit','fit-extended','reuse-smc') and args.teacher is None:
        parser.error('fit/reuse requires a teacher source path')
    if args.worker:
        worker(args)
    else:
        sys.exit(launch(args))
