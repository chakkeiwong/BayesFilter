"""Fixed-identity full80 Walnutpie qualification; no posterior claim from pilot."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO/'docs/benchmarks/artifacts/full80-walnutpie-20261030'
PLAN = REPO/'docs/plans/bayesfilter-full80-walnutpie-comparison-2026-10-30.md'

def plain(x):
    if isinstance(x,dict): return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)): return [plain(v) for v in x]
    if hasattr(x,'numpy'): return plain(x.numpy().tolist())
    if isinstance(x,float) and not math.isfinite(x): return str(x)
    return x

def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(plain(value),indent=2,allow_nan=False)+'\n')
    temp.replace(path)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu',required=True)
    parser.add_argument('--wall-seconds',type=int,default=1700)
    parser.add_argument('--worker',action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    if out.parent != ROOT or not 60 <= args.wall_seconds <= 1800:
        raise ValueError('use a fresh reviewed campaign directory and bounded wall allowance')
    if not args.worker:
        ROOT.mkdir(parents=True,exist_ok=True)
        attempts = list(ROOT.glob('*/supervisor.json'))
        previous = [json.loads(p.read_text()) for p in attempts]
        if len(attempts) >= 4 or any('elapsed_seconds' not in p for p in previous):
            raise ValueError('attempt cap or unreconciled worker')
        if sum(p['elapsed_seconds'] for p in previous)+args.wall_seconds > 1800:
            raise ValueError('aggregate qualification budget exceeded')
        out.mkdir(exist_ok=False)
        command = [sys.executable,'-B',str(Path(__file__).resolve()),'--worker',
            '--output',str(out),'--gpu',args.gpu,'--wall-seconds',str(args.wall_seconds)]
        record = dict(command=command,started_utc=datetime.now(timezone.utc).isoformat(),
            wall_limit=args.wall_seconds,plan=str(PLAN))
        write(out/'supervisor.json',record)
        started = time.monotonic()
        with (out/'worker.log').open('w') as log:
            process = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},start_new_session=True)
            record['pid'] = process.pid
            write(out/'supervisor.json',record)
            try:
                record['returncode'] = process.wait(timeout=args.wall_seconds)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
                record.update(returncode=process.returncode,timeout=True)
        record['elapsed_seconds'] = time.monotonic()-started
        write(out/'supervisor.json',record)
        print(json.dumps(record),flush=True)
        return 0 if record['returncode'] == 0 else 1
    return worker(args,out)

def worker(args,out):
    started = time.monotonic()
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu,TF_FORCE_GPU_ALLOW_GROWTH='true',
        TF_NUM_INTRAOP_THREADS='4',TF_NUM_INTEROP_THREADS='1',OMP_NUM_THREADS='4',
        OPENBLAS_NUM_THREADS='4',TF_CPP_MIN_LOG_LEVEL='2')
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4])
    sys.path.insert(0,str(REPO))
    from docs.benchmarks.full80_walnutpie_support import (verify_binding,activate_pinned_runtime,
        verify_loaded_runtime,create_adapter,load_arrays,score_model,check_state,sha,EPSILON,BASELINE)
    manifest = dict(command=[sys.executable,*sys.argv],plan=str(PLAN),result=str(out/'result.json'),
        started_utc=datetime.now(timezone.utc).isoformat(),status='starting',
        gpu=args.gpu,python=sys.version,environment={k:os.environ[k] for k in
            ('CUDA_VISIBLE_DEVICES','TF_FORCE_GPU_ALLOW_GROWTH','TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS','OMP_NUM_THREADS')},
        dtype='float64',tf32=False,jit_compile=True,mass='fixed_identity',
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        baseline=str(BASELINE),training_updates=0,posterior_comparison_complete=False)
    write(out/'manifest.json',manifest)
    def progress(stage,**info):
        row = dict(stage=stage,elapsed_seconds=time.monotonic()-started,**info)
        write(out/'progress.json',row); print(json.dumps(plain(row)),flush=True)
    try:
        binding,source = verify_binding()
        manifest.update(source=source,binding=binding)
        # Freeze the standalone kernel and runner for reproducibility.
        kernel_dir = out/'full80_kernel'; kernel_dir.mkdir()
        kernel_files = ('__init__.py','core.py','api.py','initialization.py','r2p.py',
                        'mass_coordinates.py','dense_mass.py','control.py')
        for name in kernel_files:
            p = REPO/'docs/benchmarks/walnutpie_gpu_tf'/name
            if p.exists(): shutil.copy2(p,kernel_dir/name)
        (kernel_dir/'__init__.py').write_text('')
        for p in (Path(__file__),REPO/'docs/benchmarks/full80_walnutpie_support.py'):
            shutil.copy2(p,out/p.name)
        manifest['new_source_hashes'] = {str(p.relative_to(out)):sha(p)
            for p in list(kernel_dir.glob('*.py'))+list(out.glob('*.py'))}
        sys.path.insert(0,str(out))
        activate_pinned_runtime(binding)
        import tensorflow as tf
        from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
        manifest['memory_policy'] = configure_tensorflow_gpu_memory_growth(tf,require_gpu=True)
        tf.config.experimental.enable_tensor_float_32_execution(False)
        manifest['tensorflow'] = tf.__version__
        manifest['hardware'] = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv,noheader'],text=True)
        write(out/'manifest.json',manifest)
        progress('gpu_ready')
        adapter,restored = create_adapter(tf,binding)
        manifest.update(transformed_scope=adapter.scope,restored=restored,
            imported_source_hashes=verify_loaded_runtime(binding))
        write(out/'manifest.json',manifest)
        progress('target_restored')
        from full80_kernel.core import Config
        from full80_kernel.api import make_supplied_mass_functions
        cfg = Config(4,80,warmup=2,draws=2,max_depth=5,max_halvings=5,max_error=.5,
            initial_step=EPSILON,target_accept=.7,execution_layout='packed',adapt_mass=False)
        functions = make_supplied_mass_functions(tf,score_model(tf,adapter.compiled),cfg,adapt_step=True)
        manifest.update(config=asdict(cfg),start_seed=[20261021,201],
            transition_seeds=[[202610300+i,0] for i in range(4)])
        write(out/'manifest.json',manifest)
        def timed(fn,*args):
            start=time.monotonic(); value=fn(*args)
            for v in tf.nest.flatten(value):
                if tf.is_tensor(v): v.numpy()
            return value,time.monotonic()-start
        saved=load_arrays(tf,binding['confirmation'],('z','points','logdet','score','residual','log_target'))
        z=saved['z'][:4]
        (u,ld),map_seconds=timed(adapter.frozen.generate,z)
        tf.debugging.assert_near(u,saved['points'][:4],atol=2e-7,rtol=2e-7)
        tf.debugging.assert_near(ld,saved['logdet'][:4],atol=2e-7,rtol=2e-7)
        tf.debugging.assert_near(adapter.frozen.invert(u),z,atol=2e-7,rtol=2e-7)
        progress('map_replay_passed',seconds=map_seconds)
        (lp,g,status),target_seconds=timed(adapter.compiled,z)
        tf.debugging.assert_equal(status['status_code'],tf.zeros([4],tf.int32))
        tf.debugging.assert_equal(status['valid_pre_regularized_score'],tf.ones([4],tf.bool))
        tf.debugging.assert_near(lp,saved['log_target'][:4]+ld,atol=2e-6,rtol=2e-7)
        tf.debugging.assert_near(g,saved['residual'][:4]-z,atol=2e-6,rtol=2e-7)
        progress('target_replay_passed',seconds=target_seconds)
        (vl,vg),bridge_seconds=timed(functions.value_gradient,z)
        tf.debugging.assert_near(vl,lp,atol=2e-6,rtol=2e-7)
        tf.debugging.assert_near(vg,g,atol=2e-6,rtol=2e-7)
        perm=tf.constant([2,0,3,1]);pl,pg=functions.value_gradient(tf.gather(z,perm))
        tf.debugging.assert_near(pl,tf.gather(lp,perm),atol=2e-6,rtol=2e-7)
        tf.debugging.assert_near(pg,tf.gather(g,perm),atol=2e-6,rtol=2e-7)
        changed=tf.concat([z[:1]+.01,z[1:]],0);cl,cg=functions.value_gradient(changed)
        tf.debugging.assert_near(cl[1:],lp[1:],atol=2e-6,rtol=2e-7)
        tf.debugging.assert_near(cg[1:],g[1:],atol=2e-6,rtol=2e-7)
        parity=dict(passed=True,map_seconds=map_seconds,target_compile_seconds=target_seconds,
            bridge_compile_seconds=bridge_seconds,row_isolation=True,
            max_saved_value_difference=tf.reduce_max(tf.abs(lp-saved['log_target'][:4]-ld)),
            max_saved_score_difference=tf.reduce_max(tf.abs(g-saved['residual'][:4]+z)))
        write(out/'parity.json',parity);progress('bridge_replay_passed',**parity)
        starts=tf.random.stateless_normal([4,80],[20261021,201],dtype=tf.float64)
        state,init_seconds=timed(functions.initialize,starts,tf.constant(manifest['transition_seeds'],tf.int32),
            tf.ones([4,80],tf.float64),tf.fill([4],tf.constant(EPSILON,tf.float64)))
        check_state(tf,state,adapter.compiled)
        progress('initialization_passed',seconds=init_seconds,epsilon=state.epsilon)
        chunks=[]
        for phase,fn in [('warmup',functions.warmup),('warmup',functions.warmup),
                         ('sample',functions.sample),('sample',functions.sample)]:
            previous=state
            run,seconds=timed(fn,state);state=run.state
            check_state(tf,state,adapter.compiled)
            tf.debugging.assert_all_finite(run.draws,'chunk positions')
            tf.debugging.assert_all_finite(run.log_density,'chunk log densities')
            if phase=='sample':
                for field in ('epsilon','inverse_mass','minimum','adam'):
                    for before,after in zip(tf.nest.flatten(getattr(previous,field)),tf.nest.flatten(getattr(state,field))):
                        tf.debugging.assert_equal(before,after)
            index=len(chunks)
            for name,value in zip(run._fields[1:],run[1:]):
                (out/(str(index)+'-'+name+'.tensor')).write_bytes(tf.io.serialize_tensor(value).numpy())
            checkpoint=[]
            for i,value in enumerate(tf.nest.flatten(state)):
                p=out/(str(index)+'-state-'+str(i)+'.tensor');p.write_bytes(tf.io.serialize_tensor(value).numpy())
                checkpoint.append(dict(path=p.name,sha256=sha(p),shape=value.shape.as_list(),dtype=value.dtype.name))
            write(out/(str(index)+'-checkpoint.json'),checkpoint)
            row=dict(phase=phase,seconds=seconds,transitions_per_chain=2,epsilon=state.epsilon,
                inverse_mass_identity=True,depth=run.depth,evaluations=run.evaluations,macros=run.macros,
                failed=run.failed,reverse_failed=run.reverse_failed)
            chunks.append(plain(row));write(out/'chunks.json',chunks);progress('chunk_complete',**row)
        graph=functions.sample.get_concrete_function().graph.as_graph_def()
        nodes=list(graph.node)+[n for f in graph.library.function for n in f.node_def]
        forbidden=sorted({n.op for n in nodes if any(t in n.op for t in ('PyFunc','PyCall','BfNative'))})
        if forbidden:raise ValueError('host callback in compiled sampler: '+str(forbidden))
        steady=max(chunks[1]['seconds'],chunks[3]['seconds'])/2
        result=dict(passed=True,qualification_only=True,posterior_comparison_complete=False,
            identity_mass=True,initialization_seconds=init_seconds,chunks=chunks,parity=parity,
            seconds_per_batched_transition=steady,projected_3400_transition_seconds=steady*3400,
            forecast_caveat='Only four warmup and four retained pilot transitions; adaptation may change work substantially.',
            graph=dict(node_count=len(nodes),forbidden=forbidden,sample_traces=functions.sample.experimental_get_tracing_count(),
                warmup_traces=functions.warmup.experimental_get_tracing_count(),device=state.q.device),
            allocator=tf.config.experimental.get_memory_info('GPU:0'))
        if 'GPU' not in state.q.device: raise ValueError('sampler did not execute on GPU')
        write(out/'result.json',result);manifest['status']='completed';progress('qualification_complete',projection_hours=steady*3400/3600)
        return 0
    except Exception as exc:
        manifest.update(status='failed',error=repr(exc))
        write(out/'error.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        traceback.print_exc();return 1
    finally:
        manifest['elapsed_seconds']=time.monotonic()-started
        write(out/'manifest.json',manifest)

if __name__=='__main__':
    raise SystemExit(main())
