#!/usr/bin/env python3
"""Bounded master for the chapter's covariance-guided mixture candidate.

Commands are stable for narrow execpolicy approvals. GPU is the default;
--device cpu is a deliberately hidden-GPU diagnostic exception. No installs,
network calls or external publication. Numerical kernels are shared TF modules.
"""
from pathlib import Path
import argparse
import dataclasses
import datetime
import hashlib
import json
import math
import os
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
PLAN=ROOT/'docs/plans/ledh-covariance-proposal-implementation-20261008.md'
OUT=ROOT/'docs/plans/artifacts/ledh-covariance-implementation-20261008-01'
TEST_PATH='tests/highdim/test_covariance_proposal.py'

def clean(value):
    if hasattr(value,'numpy'): return clean(value.numpy().tolist())
    if dataclasses.is_dataclass(value): return clean(dataclasses.asdict(value))
    if isinstance(value,dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [clean(v) for v in value]
    if isinstance(value,Path): return str(value)
    if isinstance(value,float) and not math.isfinite(value): return str(value)
    return value

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(clean(value),indent=2,allow_nan=False)+'\n')

def prepare(args):
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
    os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4')
    os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
    if args.device=='cpu': os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory=configure_tensorflow_gpu_memory_growth(tf,require_gpu=args.device=='gpu')
    tf.config.experimental.enable_tensor_float_32_execution(args.dtype=='float32' and not args.no_tf32)
    return tf,memory

def spec_for(name):
    if name=='lgssm':
        from bayesfilter.highdim.sqmc_lgssm_tf import LGSSMSpec
        return LGSSMSpec('frozen_3d',3)
    if name=='ksc':
        from bayesfilter.highdim.sqmc_ksc_tf import KSCSpec
        return KSCSpec()
    from bayesfilter.highdim.sqmc_nonlinear_tf import NonlinearSQMCSpec
    return NonlinearSQMCSpec(name)

def observations(spec,h,seed,tf,dtype):
    if hasattr(spec,'observations'): return spec.observations(h,seed,dtype=dtype,jit_compile=True)
    # Existing KSC simulator is the fixed FP64 independent data authority.
    theta=spec.default_theta(tf.float64 if getattr(spec,'family','')=='ksc_mixture' else dtype)
    return tf.cast(spec.simulate(theta,h,seed,jit_compile=True),dtype)

def design(tf,n,h,d,seed,dtype):
    return (tf.random.stateless_normal([n,d],[seed,1],dtype=dtype),
            tf.random.stateless_normal([h,n,d],[seed,2],dtype=dtype),
            tf.random.stateless_uniform([h,n,2],[seed,3],dtype=dtype))

def controls(args):
    from bayesfilter.highdim.covariance_proposal_tf import ProposalControls
    return ProposalControls(flow_steps=args.flow_steps,reset_steps=args.reset_steps,
        reset_epsilon=args.reset_epsilon,correction_steps=args.correction_steps)

def provenance(args,tf,memory):
    paths=list((ROOT/'bayesfilter/highdim').glob('covariance_proposal*_tf.py'))+[Path(__file__),PLAN]+[ROOT/p for p in ('bayesfilter/highdim/ledh_canonical_models_tf.py','bayesfilter/highdim/ledh_canonical_score_stages_tf.py','bayesfilter/highdim/ledh_marginal_weights_tf.py','bayesfilter/highdim/sqmc_lgssm_tf.py','bayesfilter/highdim/sqmc_ksc_tf.py','bayesfilter/highdim/sqmc_nonlinear_tf.py','bayesfilter/testing/nonlinear_bootstrap_fisher_reference_tf.py','docs/chapters/ledh_covariance_proposal_body.tex')]
    return dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        command=sys.argv,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        python=sys.executable,tensorflow=tf.__version__,device=args.device,
        cpu_only=args.device=='cpu',gpu_intentionally_hidden=args.device=='cpu',
        memory_policy=memory,jit_compile=True,dtype=args.dtype,tf32=args.dtype=='float32' and not args.no_tf32,
        plan=str(PLAN),args=vars(args),evidence_role='candidate_diagnostic_not_admission',default_promoted=False)

def destination(args):
    p=Path(args.output) if args.output else OUT/(args.command+'-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
    p.mkdir(parents=True,exist_ok=False)
    return p

def preflight(args):
    out=destination(args); tf,memory=prepare(args)
    @tf.function(input_signature=[tf.TensorSpec([3,3],tf.float32)],jit_compile=True,autograph=False)
    def probe(x): return tf.linalg.cholesky(x)
    p=probe(tf.eye(3)); record=provenance(args,tf,memory)
    record.update(logical_devices=[str(d) for d in tf.config.list_logical_devices()],probe=clean(p),probe_device=p.device)
    write(out/'manifest.json',record);print(out)

def tests(args):
    out=destination(args)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',TF_FORCE_GPU_ALLOW_GROWTH='true',TF_NUM_INTRAOP_THREADS='4',TF_NUM_INTEROP_THREADS='2',TF_CPP_MIN_LOG_LEVEL='3')
    paths=[TEST_PATH]
    if args.regressions: paths+=['tests/highdim/test_ledh_marginal_weights.py','tests/highdim/test_sqmc_ksc.py','tests/highdim/test_sqmc_full_lgssm.py']
    start=time.monotonic()
    with (out/'pytest.log').open('w') as log:
        result=subprocess.run([sys.executable,'-m','pytest','-q',*paths],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=args.timeout)
    write(out/'result.json',dict(returncode=result.returncode,wall_seconds=time.monotonic()-start,cpu_only=True,gpu_intentionally_hidden=True,command=paths,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
    print(out); print('returncode',result.returncode);print('\n'.join((out/'pytest.log').read_text().splitlines()[-12:]));return result.returncode

def scope(args,spec):
    return dict(model=args.model,target_id=spec.target_id,n=args.particles,horizon=args.horizon,dtype=args.dtype,device=args.device,jit_compile=True,tf32=args.dtype=='float32' and not args.no_tf32,controls=clean(controls(args)))

def evaluate(args,tf,spec,program,theta,obs,seed,beta):
    fixed=design(tf,args.particles,args.horizon,spec.dimension,seed,theta.dtype)
    rows=[]; output=None
    for k in range(spec.parameter_count):
        out=program(theta,tf.one_hot(k,spec.parameter_count,dtype=theta.dtype),obs,*fixed,beta)
        if output is None: output=out
        rows.append(out['score'])
        if not bool(out['valid']): output=dict(output,valid=tf.constant(False))
        if abs(float(out['value'])-float(output['value']))>1e-5: raise RuntimeError('direction-dependent primal value')
    from bayesfilter.highdim.covariance_proposal_tf import TRACE_FIELDS
    parity=[]
    if args.check_derivatives:
        for k in range(spec.parameter_count):
            direction=tf.one_hot(k,spec.parameter_count,dtype=theta.dtype)
            values=[]
            for scale in [float(v) for v in args.fd_scales.split(',')]:
                if not math.isfinite(scale) or scale<=0: raise ValueError('finite positive FD scales required')
                step=scale*max(1.,abs(float(theta[k])))
                plus=program(theta+step*direction,tf.zeros_like(theta),obs,*fixed,beta)
                minus=program(theta-step*direction,tf.zeros_like(theta),obs,*fixed,beta)
                fd=(float(plus['value'])-float(minus['value']))/(2*step)
                values.append(dict(step=step,finite_difference=fd,error=fd-float(rows[k]),valid=bool(plus['valid']) and bool(minus['valid'])))
            parity.append(dict(coordinate=k,analytical=float(rows[k]),differences=values))
    return dict(value=output['value'],score=tf.stack(rows),valid=output['valid'],steps_completed=output['steps_completed'],trace_fields=TRACE_FIELDS,trace=output['trace'],derivative_diagnostics=parity),output

def calibrate(args):
    out=destination(args); tf,memory=prepare(args); start=time.monotonic()
    from bayesfilter.highdim.covariance_proposal_tf import make_filter,build_maps,draw_from_maps,component_densities,observation_density
    from bayesfilter.highdim.covariance_proposal_beta_tf import make_beta_solver,pilot_ratios,pilot_objective
    spec=spec_for(args.model); dt=tf.as_dtype(args.dtype); theta=spec.default_theta(dt); ctl=controls(args)
    program=make_filter(spec,args.particles,args.horizon,ctl,dt)
    equal=tf.constant([1/3]*3,dt); n=args.particles; d=spec.dimension
    times=sorted(set([0,args.horizon//2,args.horizon-1]))
    @tf.function(input_signature=[tf.TensorSpec([n,d],dt),tf.TensorSpec([getattr(spec,'observation_dimension',d)],dt),tf.TensorSpec([n,d],dt),tf.TensorSpec([n,2],dt)],jit_compile=True,autograph=False)
    def pilot(anc,y,noise,u):
        model,_=spec.model(theta,tf.zeros_like(theta))
        maps=build_maps(model,theta,anc,tf.zeros_like(anc),y,ctl)
        x,dx=draw_from_maps(maps,equal,u,noise)
        ld,_,valid=component_densities(x,dx,*maps[:4])
        return ld,observation_density(model,theta,x,dx,y)[0]+ld[:,0],valid & maps[12]
    datasets=[]; context_valid=[]
    for phase in range(2):
        obs=observations(spec,args.horizon,args.data_seed+phase,tf,dt)
        fixed=design(tf,n,args.horizon,d,args.seed+100*phase,dt)
        base=program(theta,tf.zeros_like(theta),obs,*fixed,equal)
        context_valid.append(bool(base['valid']))
        if not bool(base['valid']):
            write(out/'tuning.json',dict(scope=scope(args,spec),valid=False,reason='invalid pilot context filter',steps_completed=base['steps_completed'],context_valid=context_valid,provenance=provenance(args,tf,memory)))
            print(out);return 2
        lds=[]; las=[]
        for t in times:
            noise=tf.random.stateless_normal([n,d],[args.seed+777+phase,t],dtype=dt)
            u=tf.random.stateless_uniform([n,2],[args.seed+888+phase,t],dtype=dt)
            ld,la,ok=pilot(base['ancestor_history'][t],obs[t],noise,u)
            context_valid.append(bool(ok));lds.append(ld);las.append(la)
        datasets.append((tf.stack(lds),tf.stack(las)))
    fit=make_beta_solver(len(times),n,dt)(*datasets[0],tf.constant(args.floor,dt),tf.constant(1e-7 if dt==tf.float64 else 1e-4,dt))
    vr,vl=pilot_ratios(*datasets[1]); fval=pilot_objective(fit['beta'],vr,vl)[0]; equalval=pilot_objective(equal,vr,vl)[0]
    accepted=all(context_valid) and bool(fit['converged']) and math.isfinite(float(fval))
    record=dict(scope=scope(args,spec),beta=fit['beta'],floor=args.floor,calibration_objective=fit['objective'],gap=fit['gap'],iterations=fit['iterations'],converged=fit['converged'],validation_objective=fval,validation_equal_objective=equalval,valid=accepted,context_valid=context_valid,times=times,calibration_data_seed=args.data_seed,validation_data_seed=args.data_seed+1,seed=args.seed,wall_seconds=time.monotonic()-start,provenance=provenance(args,tf,memory),no_score_or_full_filter_optimality_claim=True)
    write(out/'tuning.json',record); print(out); print(json.dumps(clean({k:record[k] for k in ('beta','gap','validation_objective','validation_equal_objective','valid')})));return 0 if accepted else 2

def run(args):
    if args.campaign: return campaign(args)
    out=destination(args); tf,memory=prepare(args); start=time.monotonic()
    from bayesfilter.highdim.covariance_proposal_tf import make_filter
    spec=spec_for(args.model); dt=tf.as_dtype(args.dtype); theta=spec.default_theta(dt)
    beta=tf.constant([float(x) for x in args.beta.split(',')],dt)
    if args.tuning:
        tuning=json.loads(Path(args.tuning).read_text())
        if tuning['scope']!=scope(args,spec) or not tuning['valid']: raise ValueError('missing/invalid/mismatched tuning scope')
        if args.data_seed in (tuning['calibration_data_seed'],tuning['validation_data_seed']): raise ValueError('final data overlaps calibration/validation')
        beta=tf.constant(tuning['beta'],dt)
    if args.dataset:
        saved=json.loads(Path(args.dataset).read_text())
        if saved['scope']['model']!=args.model or saved['scope']['horizon']!=args.horizon: raise ValueError('dataset scope mismatch')
        obs=tf.constant(saved['observations'],dt)
        theta=tf.constant(saved['theta'],dt)
        args.data_seed=saved['data_seed']
        if args.tuning and args.data_seed in (tuning['calibration_data_seed'],tuning['validation_data_seed']): raise ValueError('saved dataset overlaps calibration/validation')
    else: obs=observations(spec,args.horizon,args.data_seed,tf,dt)
    program=make_filter(spec,args.particles,args.horizon,controls(args),dt)
    results=[]
    for seed in range(args.seed,args.seed+args.seeds):
        result,_=evaluate(args,tf,spec,program,theta,obs,seed,beta);result['seed']=seed;results.append(result)
    reference=None
    if args.reference and hasattr(spec,'reference_value_and_score'):
        value,score=spec.reference_value_and_score(tf.cast(theta,tf.float64),tf.cast(obs,tf.float64))
        reference=dict(value=value,score=score,kind='Kalman exact' if args.model=='lgssm' else 'independently refined KSC grid')
    if args.reference and args.model in ('predator_prey','sir_d18'):
        from bayesfilter.testing.nonlinear_bootstrap_fisher_reference_tf import make_bootstrap_fisher_kernel
        from docs.benchmarks.run_nonlinear_bootstrap_reference import summarize
        kernel=make_bootstrap_fisher_kernel(spec.model,spec.initial_mean(tf.float64),spec.parameter_count,spec.observation_dimension,args.horizon,args.reference_particles)
        rows=[]
        for seed in range(args.seed+10000,args.seed+10000+4):
            value,score,trace=kernel(tf.cast(theta,tf.float64),tf.cast(obs,tf.float64),tf.constant(seed))
            rows.append(dict(log_likelihood=float(value),score=clean(score),minimum_particle_ess=float(tf.reduce_min(trace[:,1])),maximum_particle_weight=float(tf.reduce_max(trace[:,2])),final_distinct_initial_ancestors=float(trace[-1,3]),seed=seed))
        summary=summarize(rows)
        reference=dict(value=summary['log_likelihood'],score=summary['score'],kind='bootstrap Fisher diagnostic, not exact oracle',particles=args.reference_particles,rows=rows,summary=summary)
    record=dict(scope=scope(args,spec),theta=theta,observations=obs,data_seed=args.data_seed,beta=beta,results=results,reference=reference,wall_seconds=time.monotonic()-start,provenance=provenance(args,tf,memory),tuning=args.tuning or None)
    write(out/'results.json',record); print(out);print(json.dumps(clean(dict(results=[{k:r[k] for k in ('seed','value','score','valid')} for r in results],reference=reference))));return 0 if all(bool(r['valid']) for r in results) else 2

def campaign(args):
    """Bounded sequential diagnostic matrix; each child preserves its own scope."""
    root=destination(args); start=time.monotonic(); jobs=[]
    base=[sys.executable,'-B',str(Path(__file__).resolve())]
    def job(name,command,model,h,n=128,extra=()):
        if len(jobs)>=40 or time.monotonic()-start>2700: return None
        output=root/name
        cmd=base+[command,'--model',model,'--horizon',str(h),'--particles',str(n),'--dtype','float64','--seeds','2','--data-seed','26100820','--output',str(output),*extra]
        row=dict(name=name,command=cmd,output=str(output),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        jobs.append(row);write(root/'jobs.json',jobs)
        tic=time.monotonic()
        with (root/(name+'.log')).open('w') as log:
            try: row['returncode']=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=240).returncode
            except subprocess.TimeoutExpired: row['returncode']='timeout'
        row['wall_seconds']=time.monotonic()-tic;write(root/'jobs.json',jobs)
        print(name,row['returncode'],round(row['wall_seconds'],1),flush=True)
        return output
    for model in ('lgssm','ksc','predator_prey','sir_d18'):
        job(model+'-t3-parity','run',model,3,64,('--reference','--check-derivatives'))
        calibration=job(model+'-t50-cal','calibrate',model,50)
        tuning=calibration/'tuning.json' if calibration else None
        extras=['--reference']
        if tuning and tuning.exists() and json.loads(tuning.read_text())['valid']:
            extras+=['--tuning',str(tuning),'--data-seed','26100830']
        job(model+'-t50-mixture','run',model,50,extra=extras)
        # Identical final data for these heuristic ablations; no oracle selection.
        final_seed='26100830' if '--tuning' in extras else '26100820'
        for name,beta in (('equal','0.3333333333333333,0.3333333333333333,0.3333333333333333'),('identity','1,0,0'),('global','0.1,0.9,0'),('local','0.1,0,0.9')):
            job(model+'-t50-'+name,'run',model,50,extra=('--beta',beta,'--data-seed',final_seed))
    for model in ('predator_prey','sir_d18'):
        for h in (10,20,40):
            job(model+'-t'+str(h)+'-equal','run',model,h,extra=('--reference',))
    write(root/'completion.json',dict(jobs=len(jobs),wall_seconds=time.monotonic()-start,default_promoted=False))
    print(root)
    return 0

def report(args):
    paths=sorted(OUT.rglob('results.json')); rows=[]
    for p in paths:
        obj=json.loads(p.read_text());rows.append(dict(path=str(p),scope=obj['scope'],beta=obj['beta'],results=[{k:r[k] for k in ('seed','value','score','valid')} for r in obj['results']],reference=obj.get('reference')))
    write(OUT/'summary.json',rows)
    table=['# Complete numerical results','', 'Invalid or incomplete runs are diagnostics; their values are not full-horizon likelihood estimates. Score coordinates follow the model adapter parameter order. All numbers are descriptive.','', '| Run | Seed | Completed / T | Valid | Log likelihood | Score vector |','|---|---:|---:|---|---:|---|']
    def number(x): return format(x,'.9g') if isinstance(x,(int,float)) else str(x)
    for p in paths:
        obj=json.loads(p.read_text());name=str(p.parent.relative_to(OUT));h=obj['scope']['horizon']
        for r in obj['results']:
            vector=', '.join(number(x) for x in r['score'])
            table.append(f"| {name} | {r['seed']} | {r.get('steps_completed','unrecorded')}/{h} | {r['valid']} | {number(r['value'])} | [{vector}] |")
        ref=obj.get('reference')
        if ref:
            vector=', '.join(number(x) for x in ref['score'])
            table.append(f"| {name}: {ref['kind']} | reference | {h}/{h} | reference | {number(ref['value'])} | [{vector}] |")
            if ref.get('summary'):
                summary=ref['summary'];vector=', '.join(number(x) for x in summary['jackknife_mcse_score'])
                table.append(f"| {name}: reference MCSE | MCSE | — | estimated | {number(summary['jackknife_mcse_log_likelihood'])} | [{vector}] |")
    (OUT/'numerical-tables.md').write_text('\n'.join(table)+'\n')
    print(OUT/'summary.json');print(OUT/'numerical-tables.md')

def audit(args):
    # This inventory accompanies, and does not replace, mathematical review/tests.
    import ast
    found={}
    for path in sorted((ROOT/'bayesfilter/highdim').glob('covariance_proposal*_tf.py')):
        tree=ast.parse(path.read_text());found[str(path.relative_to(ROOT))]=[n.name for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
        if any(isinstance(n,ast.Attribute) and n.attr in ('GradientTape','ForwardAccumulator','vectorized_map') for n in ast.walk(tree)): raise ValueError('forbidden runtime derivative engine')
    out=destination(args);write(out/'inventory.json',found); print(out)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['preflight','tests','audit','calibrate','run','report'])
    parser.add_argument('--device',choices=['gpu','cpu'],default='gpu')
    parser.add_argument('--dtype',choices=['float32','float64'],default='float32')
    parser.add_argument('--model',choices=['lgssm','ksc','predator_prey','sir_d18'],default='sir_d18')
    parser.add_argument('--particles',type=int,default=64);parser.add_argument('--horizon',type=int,default=3)
    parser.add_argument('--seed',type=int,default=261008100);parser.add_argument('--data-seed',type=int,default=26100810)
    parser.add_argument('--seeds',type=int,default=2);parser.add_argument('--output')
    parser.add_argument('--flow-steps',type=int,default=16);parser.add_argument('--reset-steps',type=int,default=40)
    parser.add_argument('--reset-epsilon',type=float,default=1.);parser.add_argument('--correction-steps',type=int,default=0)
    parser.add_argument('--floor',type=float,default=.1);parser.add_argument('--beta',default='0.3333333333333333,0.3333333333333333,0.3333333333333333')
    parser.add_argument('--tuning');parser.add_argument('--reference',action='store_true');parser.add_argument('--regressions',action='store_true')
    parser.add_argument('--timeout',type=int,default=1200)
    parser.add_argument('--reference-particles',type=int,default=8192)
    parser.add_argument('--no-tf32',action='store_true')
    parser.add_argument('--check-derivatives',action='store_true')
    parser.add_argument('--fd-scales',default='1e-4,5e-5')
    parser.add_argument('--dataset')
    parser.add_argument('--campaign',action='store_true')
    args=parser.parse_args()
    if args.particles<=18 or args.particles>3000 or args.horizon<1 or args.horizon>120 or args.seeds<1 or args.seeds>8: parser.error('outside bounded candidate scope')
    return globals()[args.command](args) or 0

if __name__=='__main__': raise SystemExit(main())
