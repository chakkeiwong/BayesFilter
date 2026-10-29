"""Independent-reference million-draw replication supervisor and bounded workers.

NumPy is used only inside author-reference and post-run diagnostic functions.
TF sampling uses the shared stable-signature XLA kernel. No production imports
this standalone experiment. Binary files are little-endian float64.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
PLAN = 'docs/plans/bayesfilter-walnuts-million-replication-2026-10-29.md'
ARMS = ('author-original', 'author-corrected', 'tf-cpu', 'tf-gpu')
EVALUATORS = {
    'analytic': 'normalized analytic funnel_reference.target(11)',
    'scipy': 'unmodified author targetDistr.funnel10 (scipy)',
}


def evaluator_kind(tuning):
    """Read both historical and explicit evaluator identities; never guess."""
    kind = tuning.get('evaluator_kind')
    if kind is None:
        kind = next((k for k, label in EVALUATORS.items()
                     if tuning.get('evaluator') == label), None)
    if kind not in EVALUATORS or tuning.get('evaluator') != EVALUATORS[kind]:
        raise ValueError('unknown or inconsistent tuning evaluator')
    return kind


def validate_tuning_evaluator(tuning, expected):
    if evaluator_kind(tuning) != expected:
        raise ValueError('tuning evaluator does not match requested evaluator')


def reference_evaluator(kind, author_targets):
    if kind == 'analytic':
        from .funnel_reference import target
        return target(11)
    if kind == 'scipy':
        return author_targets.funnel10
    raise ValueError('unknown density evaluator')


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def reference_tune(out, evaluator='analytic'):
    import numpy as np  # independent author reference
    from .funnel_reference import load_author, start, check_sources
    wn, ai, td, patch = load_author(False)
    model = reference_evaluator(evaluator, td)
    q = start(0, 11)
    t = time.perf_counter()
    samples, d = wn.WALNUTS(model, q, integrator=ai.adaptLeapFrogR2P,
        M=12, H0=.3, delta0=.3, numIter=1000, warmupIter=1000,
        adaptDeltaTarget=.6, recordOrbitStats=False)
    elapsed = time.perf_counter() - t
    rng = np.random.get_state()
    result = dict(dimension=11, step=float(d[-1, 15]), delta=float(d[-1, 18]),
        seconds=elapsed, seed=0, iterations=1000, starts=[start(i, 11).tolist() for i in range(4)],
        rng_after_tuning=[rng[0], rng[1].tolist(), int(rng[2]), int(rng[3]), float(rng[4])],
        evaluator=EVALUATORS[evaluator], evaluator_kind=evaluator, source=check_sources(),
        tuning='mainFunnel.py Appendix C warmup, original author kernel; evaluator recorded explicitly',
        paper_parameters_match=False)
    np.save(out / 'warmup-draws.npy', samples, allow_pickle=False)
    np.save(out / 'warmup-diagnostics.npy', d, allow_pickle=False)
    (out / 'author-original.diff').write_text(patch)
    write(out / 'tuning.json', result)
    return result


def reference_chain(job):
    import numpy as np  # independent source reference, never TF runtime
    from .funnel_reference import load_author, start
    arm, index, count, chunk, tuning, directory = job
    out = Path(directory)
    wn, ai, td, patch = load_author(arm == 'author-corrected')
    model = reference_evaluator(evaluator_kind(tuning), td)
    q = start(index, 11)
    if index == 0:
        r = tuning['rng_after_tuning']
        np.random.set_state((r[0], np.array(r[1], dtype=np.uint32), r[2], r[3], r[4]))
    paths = [out / ('chain-%d-%s.f64' % (index, name)) for name in ('draws', 'diagnostics', 'orbit-min', 'orbit-max')]
    handles = [p.open('xb') for p in paths]
    begin = time.monotonic(); sampling = 0.; done = 0
    try:
        while done < count:
            n = min(chunk, count - done)
            t = time.perf_counter()
            samples, d, lo, hi = wn.WALNUTS(model, q,
                integrator=ai.adaptLeapFrogR2P, M=12, H0=tuning['step'],
                delta0=tuning['delta'], numIter=n, warmupIter=0, recordOrbitStats=True)
            sampling += time.perf_counter() - t
            if not np.all(np.isfinite(samples)):
                raise ValueError('nonfinite retained author states')
            for f, value in zip(handles, (samples[:, 1:].T, d, lo.T, hi.T)):
                f.write(np.ascontiguousarray(value, dtype='<f8').tobytes()); f.flush()
            q = samples[:, -1].copy(); done += n
            r = np.random.get_state()
            write(out / ('chain-%d-checkpoint.json' % index), dict(completed=done, q=q.tolist(),
                rng=[r[0], r[1].tolist(), int(r[2]), int(r[3]), float(r[4])]))
            write(out / ('chain-%d-progress.json' % index), dict(pid=os.getpid(), completed=done,
                total=count, sampling_seconds=sampling, wall_seconds=time.monotonic()-begin,
                projected_total_seconds=(time.monotonic()-begin)*count/done))
    finally:
        for f in handles: f.close()
    return dict(chain=index, draws=done, seconds=time.monotonic()-begin, sampling_seconds=sampling,
        source_patch=patch, files={p.name: dict(bytes=p.stat().st_size, sha256=sha(p)) for p in paths})


def tf_sample(a, tuning):
    from .worker import setup, graph_report
    from .funnel_xla import functions
    device = a.arm[3:]
    tf, report = setup(device)
    write(a.output/'device.json',report)
    chunk = 64  # inherited measured graph; stable shape across the million draws
    if a.draws % chunk:
        raise ValueError('TF draw count must be divisible by 64')
    with tf.device('/GPU:0' if device == 'gpu' else '/CPU:0'):
        initialize, sample, cfg = functions(tf, 11, tuning['step'], tuning['delta'], chunk)
        q = tf.constant(tuning['starts'], tf.float64)
        seeds = tf.constant([[104729+i, 0] for i in range(4)], tf.int32)
        state = initialize(q, seeds); state.q.numpy()
        t = time.perf_counter(); preview = sample(state); preview.state.q.numpy()
        report['first_call_seconds'] = time.perf_counter()-t
        report['graph'] = graph_report(sample, (state,), a.output/'sample.hlo.txt')
        elapsed = 0.; done = 0; begin = time.monotonic()
        drawfile = a.output/'draws.f64'; telemetryfile = a.output/'telemetry.jsonl'
        with drawfile.open('xb') as f, telemetryfile.open('x') as telemetry:
            while done < a.draws:
                t = time.perf_counter(); result = sample(state); result.state.q.numpy()
                elapsed += time.perf_counter()-t; state = result.state
                if not bool(tf.reduce_all(state.valid).numpy()):
                    raise ValueError('invalid retained TF state/cache')
                if not bool(tf.reduce_all(tf.math.is_finite(result.draws)).numpy()):
                    raise ValueError('nonfinite retained draws')
                # Host serialization boundary only; no NumPy computation.
                f.write(result.draws.numpy().tobytes()); done += chunk
                telemetry.write(json.dumps(dict(completed=done,
                    evaluations=int(tf.reduce_sum(tf.cast(result.evaluations,tf.int64)).numpy()),
                    macros=int(tf.reduce_sum(tf.cast(result.macros,tf.int64)).numpy()),
                    failed=int(tf.reduce_sum(tf.cast(result.failed,tf.int32)).numpy()),
                    reverse_zero=int(tf.reduce_sum(tf.cast(result.reverse_failed,tf.int32)).numpy()),
                    max_depth=int(tf.reduce_max(result.depth).numpy())))+'\n')
                if done % 1024 == 0 or done == a.draws:
                    f.flush(); telemetry.flush()
                    with tf.device('/CPU:0'):
                        checkpoint = [tf.io.serialize_tensor(x).numpy().hex() for x in tf.nest.flatten(state)]
                    write(a.output/'checkpoint.json',dict(completed=done, flat_state_hex=checkpoint,
                        state_structure='api.State, nested api.Adam', tuning_sha256=sha(a.tuning)))
                    write(a.output/'progress.json',dict(pid=os.getpid(), completed=done,total=a.draws,
                        sampling_seconds=elapsed,wall_seconds=time.monotonic()-begin,
                        projected_total_seconds=(time.monotonic()-begin)*a.draws/done))
        report.update(sampling_seconds=elapsed, draws=a.draws, chunk_size=chunk,
            timing_barrier='materialize_result_state_q', files={drawfile.name:dict(
                bytes=drawfile.stat().st_size,sha256=sha(drawfile)), telemetryfile.name:dict(
                bytes=telemetryfile.stat().st_size,sha256=sha(telemetryfile))})
        if device == 'gpu': report['allocator']=tf.config.experimental.get_memory_info('GPU:0')
    return report


def worker(a):
    tuning=json.loads(a.tuning.read_text());begin=time.monotonic()
    validate_tuning_evaluator(tuning, a.evaluator)
    report=dict(arm=a.arm,dimension=11,chains=4,draws=a.draws,step=tuning['step'],delta=tuning['delta'],
        tuning_evaluator=EVALUATORS[a.evaluator],tuning_evaluator_kind=a.evaluator,
        evaluator=EVALUATORS[a.evaluator] if a.arm.startswith('author-') else
            'TF/XLA analytic funnel; state-independent normalization omitted',
        jit_compile=a.arm.startswith('tf-'), concurrent_timing=True, seeds=list(range(4)) if
        a.arm.startswith('author-') else list(range(104729,104733)))
    try:
        if a.arm.startswith('author-'):
            jobs=[(a.arm,i,a.draws,1024,tuning,str(a.output)) for i in range(4)]
            with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
                chains=list(pool.map(reference_chain,jobs))
            report.update(chains_report=chains,cpu_workers=4,evaluator=tuning['evaluator'],
                numpy_role='independent unmodified/corrected author reference',
                gpu_intentionally_hidden=True,sampling_seconds=max(c['sampling_seconds'] for c in chains))
        else:report.update(tf_sample(a,tuning))
        report['status']='completed'
    except BaseException as e:
        report.update(status='failed',error=repr(e),error_message=str(e));raise
    finally:
        report.update(wall_seconds=time.monotonic()-begin,peak_host_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write(a.output/'result.json',report)


def environment(device,gpu):
    env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu) if device=='gpu' else '-1',
        'TF_FORCE_GPU_ALLOW_GROWTH':'true','BAYESFILTER_PRELOAD_CUSTOM_OP':'0',
        'TF_NUM_INTRAOP_THREADS':'4','TF_NUM_INTEROP_THREADS':'2','OPENBLAS_NUM_THREADS':'1',
        'OMP_NUM_THREADS':'1','PYTHONUNBUFFERED':'1','MPLBACKEND':'Agg'}
    if device=='gpu':
        env['XLA_FLAGS']=env.get('XLA_FLAGS','')+' --xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUBLASLT,CUDNN,WHILE,CONDITIONAL'
    return env


def supervise(a):
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    source=out/'source';hashes={}
    files=list(Path(__file__).parent.glob('*.py'))+list(Path(__file__).parent.joinpath('author_reference').rglob('*'))
    files += [ROOT/PLAN, *[ROOT/('bayesfilter/inference/'+name) for name in
        ('hmc_posterior_diagnostics.py','hmc_diagnostic_math.py','hmc_ess.py')]]
    for file in files:
        if not file.is_file() or '__pycache__' in file.parts:continue
        relative=file.relative_to(ROOT);dest=source/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(file,dest);hashes[str(relative)]=sha(file)
    for rel in ('docs/__init__.py','docs/benchmarks/__init__.py','bayesfilter/__init__.py','bayesfilter/inference/__init__.py'):
        p=source/rel;p.parent.mkdir(parents=True,exist_ok=True);p.touch()
    manifest=dict(plan=PLAN,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        started_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),draws_per_chain=a.draws,
        source_hashes=hashes,source_snapshot=str(source),status='preflight',
        diagnostic_package_note='Minimal snapshot package initializers; numerical dependencies copied and hashed',
        python=sys.executable,command=sys.argv,arms={},per_arm_seconds=a.arm_seconds,
        overall_seconds=a.arm_seconds+3600,output=str(out),gpu=str(a.gpu),
        reference_evaluator=EVALUATORS[a.evaluator],evaluator_kind=a.evaluator,
        tf_evaluator='analytic TensorFlow formula, state-independent normalization omitted')
    write(out/'manifest.json',manifest);begin=time.monotonic()
    env=environment('cpu',a.gpu);env['PYTHONPATH']=str(source)+os.pathsep+str(ROOT)
    tuningdir=out/'tuning';tuningdir.mkdir()
    command=[sys.executable,'-m','docs.benchmarks.walnutpie_gpu_tf.replication','--mode','tune',
        '--evaluator',a.evaluator,'--output',str(tuningdir)]
    if a.tuning:
        previous=json.loads(a.tuning.read_text())
        validate_tuning_evaluator(previous, a.evaluator)
        shutil.copy2(a.tuning,tuningdir/'tuning.json')
        manifest['tuning_source']=str(a.tuning.resolve())
        manifest['tuning_sha256']=sha(a.tuning)
    else:
        with (out/'tuning.log').open('w') as log:
            try:subprocess.run(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1200)
            except BaseException as e:
                manifest.update(status='preflight_failed',error=repr(e));write(out/'manifest.json',manifest);raise
    tuning=tuningdir/'tuning.json';children={};logs={}
    manifest['gpu_inventory']=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu','--format=csv'],text=True)
    manifest['resource_note']='Concurrent requested methods; shared hardware, no isolated runtime ranking'
    try:
        # Start all four before waiting for any one. Each owns a process group.
        for arm in ARMS:
            directory=out/arm;directory.mkdir();env=environment('gpu' if arm=='tf-gpu' else 'cpu',a.gpu)
            env['PYTHONPATH']=str(source)+os.pathsep+str(ROOT)
            cmd=[sys.executable,'-m','docs.benchmarks.walnutpie_gpu_tf.replication','--mode','worker',
                '--arm',arm,'--draws',str(a.draws),'--evaluator',a.evaluator,
                '--tuning',str(tuning),'--output',str(directory)]
            logs[arm]=(directory/'worker.log').open('w')
            p=subprocess.Popen(cmd,cwd=source,env=env,stdout=logs[arm],stderr=subprocess.STDOUT,start_new_session=True)
            children[arm]=(p,time.monotonic())
            manifest['arms'][arm]=dict(pid=p.pid,command=cmd,status='running',
                environment={k:env.get(k) for k in ('CUDA_VISIBLE_DEVICES','TF_FORCE_GPU_ALLOW_GROWTH','XLA_FLAGS','TF_NUM_INTRAOP_THREADS','OPENBLAS_NUM_THREADS')})
        manifest['status']='running';write(out/'manifest.json',manifest)
        while any(p.poll() is None for p,t in children.values()):
            for arm,(p,t) in children.items():
                row=manifest['arms'][arm]
                if row['status']=='running':row['wall_seconds']=time.monotonic()-t
                if p.poll() is None and time.monotonic()-t>a.arm_seconds:
                    os.killpg(p.pid,signal.SIGTERM)
                    try:p.wait(timeout=5)
                    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                    row['status']='timeout'
                elif p.poll() is not None and row['status']=='running':
                    row.update(status='completed' if p.returncode==0 else 'failed',returncode=p.returncode)
            manifest['wall_seconds']=time.monotonic()-begin;write(out/'manifest.json',manifest)
            time.sleep(5)
        for arm,(p,t) in children.items():
            if manifest['arms'][arm]['status']=='running':
                manifest['arms'][arm].update(status='completed' if p.returncode==0 else 'failed',returncode=p.returncode)
        manifest['status']='analyzing';write(out/'manifest.json',manifest)
        env=environment('cpu',a.gpu);env['PYTHONPATH']=str(source)+os.pathsep+str(ROOT)
        with (out/'analysis.log').open('w') as log:
            result=subprocess.run([sys.executable,'-m','docs.benchmarks.walnutpie_gpu_tf.replication',
                '--mode','analyze','--output',str(out)],cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=3600)
        manifest['status']='completed' if result.returncode==0 and all(x['status']=='completed' for x in manifest['arms'].values()) else 'incomplete'
    except BaseException as e:
        manifest.update(status='failed',error=repr(e))
        for p,t in children.values():
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        raise
    finally:
        for log in logs.values():log.close()
        manifest['wall_seconds']=time.monotonic()-begin;write(out/'manifest.json',manifest)


def analyze(out):
    from .replication_analysis import analyze as run_analysis
    run_analysis(out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['supervise','worker','tune','analyze'],default='supervise')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--tuning',type=Path)
    p.add_argument('--arm',choices=ARMS);p.add_argument('--draws',type=int,default=1000000)
    p.add_argument('--gpu',default='2');p.add_argument('--arm-seconds',type=int,default=57600)
    p.add_argument('--evaluator',choices=tuple(EVALUATORS),default='analytic')
    a=p.parse_args()
    if a.draws<64 or a.draws>1000000 or a.draws%64:raise ValueError('draws must be a multiple of 64 up to one million')
    if a.arm_seconds<1 or a.arm_seconds>57600:raise ValueError('arm deadline exceeds owner-approved 16 hours')
    if a.mode=='supervise':supervise(a)
    elif a.mode=='worker':worker(a)
    elif a.mode=='tune':reference_tune(a.output,a.evaluator)
    else:analyze(a.output)

if __name__=='__main__':main()
