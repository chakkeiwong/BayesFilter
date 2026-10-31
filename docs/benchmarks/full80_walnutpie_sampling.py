"""Author-controlled Walnutpie comparison for the frozen MacroFinance full80.

This executable requires an explicitly authorized expanded compute allowance.
It never borrows MacroFinance's historical balance. See the October 30 plan.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
import resource
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

REPO=Path(__file__).resolve().parents[2]
ROOT=REPO/'docs/benchmarks/artifacts/full80-walnutpie-20261030'
PLAN=REPO/'docs/plans/bayesfilter-full80-walnutpie-comparison-2026-10-30.md'
# Owner, 2026-10-30: 'you have 70 hours more'; additional GPU-worker time.
AUTHORIZED_ADDITIONAL_SECONDS=70*60*60

def validate_allowance(wall_seconds,campaign_seconds):
    if not 60<=wall_seconds<=campaign_seconds<=AUTHORIZED_ADDITIONAL_SECONDS:
        raise ValueError('authorized additional campaign cap is 70 hours')

def frozen_invariants(tf,before,after):
    for name in ('epsilon','inverse_mass','minimum','adam','iteration','weight',
                 'draw_mean','score_mean','draw_m2','score_m2','macro_total','macro_count'):
        for a,b in zip(tf.nest.flatten(getattr(before,name)),tf.nest.flatten(getattr(after,name))):
            tf.debugging.assert_equal(a,b,message='frozen tuning drift: '+name)

def validate_resume(old,current):
    for name in ('policy','config','binding_sha256','target_scope','source_hashes'):
        if name not in old or old[name]!=current[name]:
            raise ValueError('resume configuration/source mismatch: '+name)
    if old.get('kind')!='full80_walnutpie_comparison':
        raise ValueError('engineering pilot cannot resume as posterior campaign')

def run_stream(tf,state,controller,*,functions,policy,adaptation_decision,
               validate,record,assess,persist,budget,forecast,timed,progress):
    """Single state/RNG stream; all frozen outputs count, including rejections."""
    while controller['phase'] in ('adapt','retained'):
        reason=budget(controller)
        if reason:
            controller['stop_reason']=reason;persist(state,controller);return state,controller
        phase=controller['phase']
        before=state
        result,seconds=timed(functions.warmup if phase=='adapt' else functions.sample,state)
        state=result.state;validate(state)
        if phase=='retained':frozen_invariants(tf,before,state)
        row=record(result,phase,seconds,controller)
        controller['history'].append(row)
        if phase=='adapt':
            controller['adapt_count']+=row['count']
            report=adaptation_decision(state)
            controller['adaptation']=report
            if report['freeze']:
                controller['phase']='retained'
                controller['adaptation_complete']=True
                controller['handoff']='same state and RNG; no averaging or extra exclusions'
        else:
            controller['retained_count']+=row['count']
        controller['serial']+=1
        controller['forecast']=forecast(controller)
        persist(state,controller)
        progress('chunk',phase=phase,adapt_count=controller['adapt_count'],
            retained_count=controller['retained_count'],seconds=seconds,epsilon=state.epsilon)
        n=controller['retained_count']
        if phase=='retained' and n>=policy['retained_min'] and n%policy['assessment_stride']==0:
            result=assess(controller)
            controller['assessments'].append(result)
            if result['passed']:
                controller['phase']='complete';controller['stop_reason']='posterior_screens_passed'
            elif n>=policy['retained_max']:
                controller['phase']='complete';controller['stop_reason']='posterior_limit_without_pass'
            persist(state,controller)
    return state,controller

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu',required=True)
    parser.add_argument('--wall-seconds',type=int,required=True)
    parser.add_argument('--campaign-seconds',type=int,required=True)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--worker',action='store_true')
    args=parser.parse_args();out=args.output.resolve()
    validate_allowance(args.wall_seconds,args.campaign_seconds)
    if out.parent!=ROOT:
        raise ValueError('fresh reviewed output directory required')
    sys.path.insert(0,str(REPO))
    from docs.benchmarks.full80_walnutpie import write
    if args.worker:return worker(args,out)
    ROOT.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(ROOT/'comparison.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    previous=[json.loads(p.read_text()) for p in ROOT.glob('*/supervisor.json')]
    previous=[p for p in previous if p.get('kind')=='full80_walnutpie_comparison']
    if len(previous)>=4 or any('elapsed_seconds' not in p for p in previous):
        raise ValueError('comparison attempt cap or unreconciled process')
    spent=sum(p['elapsed_seconds'] for p in previous)
    if spent+args.wall_seconds>args.campaign_seconds:
        raise ValueError('aggregate additional campaign budget exceeded')
    out.mkdir(exist_ok=False)
    cmd=[sys.executable,'-B',str(Path(__file__).resolve()),'--worker','--output',str(out),
        '--gpu',args.gpu,'--wall-seconds',str(args.wall_seconds),
        '--campaign-seconds',str(args.campaign_seconds)]
    if args.resume:cmd+=['--resume',str(args.resume.resolve())]
    summary=dict(kind='full80_walnutpie_comparison',command=cmd,
        started_utc=datetime.now(timezone.utc).isoformat(),prior_seconds=spent,
        wall_limit=args.wall_seconds,campaign_limit=args.campaign_seconds,plan=str(PLAN),
        authorization='Owner 2026-10-30: you have 70 hours more',
        authorized_additional_seconds=AUTHORIZED_ADDITIONAL_SECONDS)
    write(out/'supervisor.json',summary)
    started=time.monotonic()
    with (out/'worker.log').open('w') as log:
        process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        summary['pid']=process.pid;write(out/'supervisor.json',summary)
        try:summary['returncode']=process.wait(timeout=args.wall_seconds)
        except subprocess.TimeoutExpired:
            process.kill();process.wait();summary.update(returncode=process.returncode,timeout=True)
        except BaseException:
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            summary.update(returncode=process.returncode,supervisor_interrupted=True)
    summary['elapsed_seconds']=time.monotonic()-started;write(out/'supervisor.json',summary)
    manifest=out/'manifest.json'
    if manifest.exists():
        m=json.loads(manifest.read_text())
        if m.get('status')=='running':
            m.update(status='supervisor_timeout' if summary.get('timeout') else 'worker_terminated',
                     elapsed_seconds=summary['elapsed_seconds'])
            write(manifest,m)
    print(json.dumps(summary),flush=True)
    return 0 if summary['returncode']==0 else 1

def worker(args,out):
    start=time.monotonic();cpu_start=resource.getrusage(resource.RUSAGE_SELF);cancelled=[]
    signal.signal(signal.SIGTERM,lambda *x:cancelled.append('stop_requested'))
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu,TF_FORCE_GPU_ALLOW_GROWTH='true',
        TF_NUM_INTRAOP_THREADS='4',TF_NUM_INTEROP_THREADS='1',OMP_NUM_THREADS='4',
        OPENBLAS_NUM_THREADS='4',TF_CPP_MIN_LOG_LEVEL='2',PYTHONDONTWRITEBYTECODE='1')
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4])
    from docs.benchmarks.full80_walnutpie import write,plain
    from docs.benchmarks.full80_walnutpie_support import (verify_binding,activate_pinned_runtime,
        verify_loaded_runtime,create_adapter,score_model,check_state,load_arrays,sha,BASELINE)
    from docs.benchmarks import full80_walnutpie_sampling_helpers as helpers
    from docs.benchmarks.full80_walnutpie_report import assess,compare
    policy=helpers.POLICY
    m=dict(kind='full80_walnutpie_comparison',status='running',policy=policy,
        started_utc=datetime.now(timezone.utc).isoformat(),command=[sys.executable,*sys.argv],
        plan=str(PLAN),result=str(out/'result.json'),wall_limit=args.wall_seconds,
        campaign_limit=args.campaign_seconds,baseline=str(BASELINE),resume=str(args.resume) if args.resume else None,
        authorization='Owner 2026-10-30: you have 70 hours more',
        authorized_additional_seconds=AUTHORIZED_ADDITIONAL_SECONDS,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        python=sys.version,gpu=args.gpu,dtype='float64',tf32=False,jit_compile=True,
        trust_basis='trusted_escalated_execution',mass='fixed_identity',training_updates=0,
        start_seed=[20261021,201],transition_seeds=[[202610400+i,0] for i in range(4)],
        environment={k:os.environ[k] for k in ('CUDA_VISIBLE_DEVICES','TF_FORCE_GPU_ALLOW_GROWTH',
             'TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS')})
    write(out/'manifest.json',m)
    def progress(stage,**extra):
        data=plain(dict(stage=stage,elapsed_seconds=time.monotonic()-start,**extra))
        write(out/'progress.json',data);print(json.dumps(data),flush=True)
    try:
        binding,source=verify_binding();m.update(source=source,binding=binding,binding_sha256=source['binding_sha256'])
        package=out/'full80_kernel';package.mkdir();(package/'__init__.py').write_text('')
        sources={}
        for name in ('core.py','api.py','initialization.py','r2p.py','control.py'):
            p=REPO/'docs/benchmarks/walnutpie_gpu_tf'/name
            sources['walnutpie/'+name]=sha(p);shutil.copy2(p,package/name)
        for name in ('full80_walnutpie_sampling.py','full80_walnutpie_sampling_helpers.py',
                     'full80_walnutpie.py','full80_walnutpie_support.py','full80_walnutpie_report.py'):
            p=REPO/'docs/benchmarks'/name;sources[name]=sha(p);shutil.copy2(p,out/name)
        author=REPO/'.localresources/walnutpie-5710b0334ba69cf0caa35fc02a2338b61d756953/include/walnutpie'
        m['author_sources']={name:sha(author/name) for name in ('config.hpp','adapt.hpp','adaptive_walnuts.hpp','walnuts.hpp','adam.hpp')}
        shutil.copy2(PLAN,out/'plan.md');m['source_hashes']=sources
        activate_pinned_runtime(binding);sys.path.insert(0,str(out))
        import tensorflow as tf
        from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
        m['memory_policy']=configure_tensorflow_gpu_memory_growth(tf,require_gpu=True)
        tf.config.experimental.enable_tensor_float_32_execution(False)
        m['tensorflow']=tf.__version__
        m['hardware']=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv,noheader'],text=True)
        adapter,restored=create_adapter(tf,binding)
        m.update(target_scope=adapter.scope,restored=restored)
        from full80_kernel.core import Config
        from full80_kernel.api import make_supplied_mass_functions,state_spec
        from full80_kernel.control import warmup_statistics
        cfg=Config(4,80,warmup=5,draws=5,max_depth=5,max_halvings=5,
            initial_step=policy['initial_step'],target_accept=policy['target_macro_quality'],
            adapt_mass=False,execution_layout='packed')
        m['config']=asdict(cfg)
        m['imported_source_hashes']=verify_loaded_runtime(binding);write(out/'manifest.json',m)
        model=score_model(tf,adapter.compiled)
        functions=make_supplied_mass_functions(tf,model,cfg,adapt_step=True)
        @tf.function(input_signature=[tf.TensorSpec([5,4,80],tf.float64)],jit_compile=True,autograph=False)
        def physical(z):return adapter.physical(z)
        def timed(fn,*values):
            t=time.monotonic();r=fn(*values)
            for item in tf.nest.flatten(r):
                if tf.is_tensor(item):item.numpy()
            return r,time.monotonic()-t
        anchor=load_arrays(tf,binding['confirmation'],('z','log_target','logdet','residual'))
        z=anchor['z'][:4];(lp,g,status),sec=timed(adapter.compiled,z)
        tf.debugging.assert_equal(status['status_code'],tf.zeros([4],tf.int32))
        tf.debugging.assert_equal(status['valid_pre_regularized_score'],tf.ones([4],tf.bool))
        tf.debugging.assert_near(lp,anchor['log_target'][:4]+anchor['logdet'][:4],atol=2e-6,rtol=2e-7)
        tf.debugging.assert_near(g,anchor['residual'][:4]-z,atol=2e-6,rtol=2e-7)
        write(out/'parity.json',dict(passed=True,seconds=sec));progress('target_parity_passed',seconds=sec)
        ctl=dict(phase='adapt',serial=0,adapt_count=0,retained_count=0,adaptation_complete=False,
            history=[],assessments=[],physical_samples=[],sample_files=[],diagnostic_seconds=0.)
        if args.resume:
            old=json.loads((args.resume/'manifest.json').read_text());validate_resume(old,m)
            ctl=json.loads((args.resume/'controller.json').read_text())
            if ctl['phase'] not in ('adapt','retained'):raise ValueError('resume is already complete')
            state=helpers.restore_checkpoint(tf,ctl['checkpoint'],args.resume,state_spec(tf,cfg))
            # Copy exact prior evidence so each attempt is independently readable.
            for row in ctl['sample_files']+ctl['physical_samples']:
                p=(args.resume/row['path']).resolve()
                if not p.is_relative_to(args.resume.resolve()):raise ValueError('sample outside resume attempt')
                if sha(p)!=row['sha256']:raise ValueError('prior sample hash mismatch')
                shutil.copy2(p,out/p.name)
            ctl.pop('stop_reason',None)
        else:
            q=tf.random.stateless_normal([4,80],m['start_seed'],dtype=tf.float64)
            state,sec=timed(functions.initialize,q,tf.constant(m['transition_seeds'],tf.int32),
                tf.ones_like(q),tf.fill([4],tf.constant(policy['initial_step'],tf.float64)))
            progress('initialized',seconds=sec,epsilon=state.epsilon)
        check_state(tf,state,adapter.compiled)
        def save_tensor(name,value):
            p=out/name;p.write_bytes(tf.io.serialize_tensor(value).numpy())
            return dict(path=p.name,sha256=sha(p),shape=value.shape.as_list(),dtype=value.dtype.name)
        def persist(s,c):
            c['checkpoint']=helpers.checkpoint(tf,s,out,'state-'+str(c['serial']))
            write(out/'controller.json',c)
            write(out/'physical-samples.json',c['physical_samples'])
            write(out/'assessment.json',dict(adaptation_complete=c['adaptation_complete'],
                warmup_ready=c['adaptation_complete'],warmup_ready_meaning='source adaptation ended; not stationarity',
                retained_count=c['retained_count'],last=c['assessments'][-1] if c['assessments'] else None))
        persist(state,ctl)
        graphs={}
        first_calls=set()
        def record(run,phase,seconds,c):
            tf.debugging.assert_all_finite(run.draws,'all retained positions')
            tf.debugging.assert_all_finite(run.log_density,'all retained densities')
            first=phase not in first_calls;first_calls.add(phase)
            if first:
                fn=functions.warmup if phase=='adapt' else functions.sample
                graph=fn.get_concrete_function().graph.as_graph_def()
                nodes=list(graph.node)+[n for f in graph.library.function for n in f.node_def]
                forbidden=sorted({n.op for n in nodes if any(x in n.op for x in ('PyFunc','PyCall','BfNative'))})
                if forbidden or 'GPU' not in run.state.q.device:raise ValueError('whole-transition GPU XLA qualification failed')
                graphs[phase]=dict(nodes=len(nodes),forbidden=forbidden,device=run.state.q.device,traces=fn.experimental_get_tracing_count())
                write(out/'graph.json',graphs)
                m['imported_source_hashes']=verify_loaded_runtime(binding)
                write(out/'manifest.json',m)
            serial=c['serial']
            for name,value in zip(run._fields[1:],run[1:]):
                c['sample_files'].append(save_tensor(str(serial)+'-'+phase+'-'+name+'.tensor',value))
            if phase=='retained':
                values,decode_seconds=timed(physical,run.draws)
                tf.debugging.assert_all_finite(values,'physical parameter draws')
                c['physical_samples'].append(save_tensor(str(serial)+'-physical.tensor',values))
                seconds+=decode_seconds
            return plain(dict(phase=phase,count=int(run.draws.shape[0]),seconds=seconds,
                first_compiled_call=first,epsilon=run.state.epsilon,minimum=run.state.minimum,
                evaluations=run.evaluations,depth=run.depth,macros=run.macros,
                refinement_failures=run.failed,reverse_failures=run.reverse_failed,
                allocator=tf.config.experimental.get_memory_info('GPU:0'),
                host_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        def assessment(c):
            t=time.monotonic()
            with tf.device('/CPU:0'):
                values=[]
                for row in c['physical_samples']:
                    p=out/row['path']
                    if sha(p)!=row['sha256']:raise ValueError('physical sample hash mismatch')
                    values.append(tf.io.parse_tensor(p.read_bytes(),tf.float64))
                report=assess(tf,tf.concat(values,0))
            path='assessment-'+str(c['retained_count'])+'.json';write(out/path,report)
            c['diagnostic_seconds']=time.monotonic()-t
            result=plain({k:report[k] for k in ('passed','draws_per_chain','max_rhat','min_bulk_ess','min_tail_ess','max_controller_mcse_sd')})
            result['path']=path;progress('posterior_assessment',**result);return result
        def forecast(c):
            return helpers.timing_forecast(c['history'],c['phase'],c['retained_count'],
                c['adapt_count'],c['diagnostic_seconds'])
        def budget(c):
            if cancelled:return 'stop_requested'
            remaining=args.wall_seconds-(time.monotonic()-start)
            history=[r for r in c['history'] if r['phase']==c['phase'] and not r['first_compiled_call']]
            # Compilation reserve is an engineering allowance, not a cost estimate.
            next_chunk=1.25*max((r['seconds'] for r in history[-4:]),default=600.)+30
            if remaining<next_chunk:return 'insufficient_budget_for_next_chunk'
            estimate=forecast(c)
            if c['phase']=='retained' and estimate.get('sampling_measured') and estimate['next_assessment_seconds']>remaining:
                return 'insufficient_budget_for_next_posterior_assessment'
            return None
        state,ctl=run_stream(tf,state,ctl,functions=functions,policy=policy,
            adaptation_decision=lambda s:helpers.adaptation_decision(tf,s,statistics_fn=warmup_statistics),
            validate=lambda s:check_state(tf,s,adapter.compiled),record=record,assess=assessment,
            persist=persist,budget=budget,forecast=forecast,timed=timed,progress=progress)
        result=dict(status=ctl['stop_reason'],adaptation_complete=ctl['adaptation_complete'],
            adaptation_count=ctl['adapt_count'],retained_count=ctl['retained_count'],
            posterior_passed=bool(ctl['assessments'] and ctl['assessments'][-1]['passed']),
            comparison_complete=False,forecast=ctl.get('forecast'),allocator=tf.config.experimental.get_memory_info('GPU:0'))
        if result['posterior_passed']:
            from dz5_financial_contract import NAMES
            with tf.device('/CPU:0'):
                baseline=load_arrays(tf,BASELINE/'sequential-samples.npz',['private_retained_raw'])['private_retained_raw']
                reference=assess(tf,baseline)
                candidate=assess(tf,tf.concat([tf.io.parse_tensor((out/r['path']).read_bytes(),tf.float64) for r in ctl['physical_samples']],0))
                comparison=compare(tf,reference,candidate,NAMES,warmup_ready=ctl['adaptation_complete'])
            write(out/'baseline.json',reference);write(out/'candidate.json',candidate);write(out/'comparison.json',comparison)
            result.update(comparison_complete=comparison['qualified'],comparison_verdict=comparison['verdict'])
        write(out/'result.json',result);m['status']=result['status'];progress('stopped',**result)
        return 0
    except Exception as exc:
        m.update(status='failed',error=repr(exc))
        write(out/'error.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        traceback.print_exc();return 1
    finally:
        usage=resource.getrusage(resource.RUSAGE_SELF)
        m.update(elapsed_seconds=time.monotonic()-start,host_peak_rss_kib=usage.ru_maxrss,
            cpu_seconds=usage.ru_utime+usage.ru_stime-cpu_start.ru_utime-cpu_start.ru_stime)
        write(out/'manifest.json',m)

if __name__=='__main__':raise SystemExit(main())
