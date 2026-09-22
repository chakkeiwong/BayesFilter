"""Diagnostic capture and fixed-cloud fit isolation; no production candidate."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from unittest.mock import patch

from diagnose_iapf_adaptive_consumer import REPO, r_literal, save, setup

ROOT = REPO / 'docs/plans/artifacts/iapf-fit-input-isolation-20260922-01'
PRIOR = REPO / 'docs/plans/artifacts/iapf-initialization-isolation-20260922-01'
GEOMETRY = REPO / 'docs/plans/artifacts/iapf-guide-geometry-20260922-01'
PLAN = 'docs/plans/iapf-fit-input-isolation-2026-09-22.md'
CASES = {(2,82,'cloud_moments','initial_peak'), (2,82,'log_quadratic','initial_peak'),
         (5,82,'log_quadratic','initial_peak'), (10,82,'log_quadratic','native')}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preserve(directory, source_root, relative):
    path = source_root / relative
    if sha(path) != read(source_root / 'manifest.json')['outputs'][relative]:
        raise RuntimeError('Changed input ' + str(path))
    destination = directory / 'inputs' / source_root.name / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)
    return read(destination)


def name(row):
    return f"d{row['d']}-s{row['seed']}-{row['initialization']}-{row['objective_scale']}"


def difference(a, b):
    if isinstance(a, dict):
        if set(a) != set(b):
            raise ValueError('Different replay keys')
        return max([difference(a[k], b[k]) for k in a] or [0.])
    if isinstance(a, list):
        if len(a) != len(b):
            raise ValueError('Different replay lengths')
        return max([difference(x,y) for x,y in zip(a,b)] or [0.])
    if isinstance(a, (bool,str)) or a is None:
        if a != b:
            raise ValueError('Different replay status')
        return 0.
    return abs(a-b)


def capture(tf, directory):
    from bayesfilter.score_study import iapf_fit_tf
    from bayesfilter.score_study.iapf_adapter import execute_iapf
    from bayesfilter.score_study.contracts import DiagnosticFailure
    original = iapf_fit_tf.make_density_recursive_fit_kernel
    prior = preserve(directory, PRIOR, 'attempt03_gpu_consumer/results.json')
    rows = [r for r in prior if (r['d'],r['seed'],r['initialization'],r['objective_scale']) in CASES]
    summary = []
    for row in rows:
        calls = []
        def factory(*args, **kwargs):
            kernel = original(*args, **kwargs)
            class Captured:
                def __call__(self, *values):
                    result = kernel(*values)
                    calls.append(dict(input={k:v.numpy().tolist() for k,v in zip(('theta','observations','clouds'),values)},
                        result={k:v.numpy().tolist() for k,v in zip(('centers','covariances','floors','valid','converged','diagnostics'),result)}))
                    return result
                def experimental_get_tracing_count(self):
                    return kernel.experimental_get_tracing_count()
            return Captured()
        d = row['d']; case_seed = row['seed']
        def seed(stream, replicate=case_seed, group='final'):
            raw=hashlib.sha256(f'{case_seed}:{d}:{d}:{group}:{replicate}:{stream}'.encode()).digest()
            return [int.from_bytes(raw[:4],'little')%(2**31-1),int.from_bytes(raw[4:8],'little')%(2**31-1)]
        status = 'complete'; error = None; value = score = None
        with patch.object(iapf_fit_tf, 'make_density_recursive_fit_kernel', factory):
            try:
                _,final,diag,_ = execute_iapf(dict(model='linear_gaussian',method='iapf',role='diagnostic',iapf=row['config']),
                    row['settings'],tf.constant(row['config']['fit_theta'],tf.float64),
                    tf.constant(row['observations']['fitted'],tf.float64),seed)
                value,score=final[:2]
            except DiagnosticFailure as exc:
                status='candidate_rejected';error=str(exc);diag=exc.diagnostics['details']
        errors = {'history':difference(diag['fit_iterations'],row['diagnostics']['fit_iterations'])}
        if status=='complete':
            value=float(value.numpy()) if hasattr(value,'numpy') else value
            score=score.numpy().tolist() if hasattr(score,'numpy') else score
            errors.update(fit=difference(diag['fit'],row['diagnostics']['fit']),
                value=difference(value,row['regimes']['fitted']['value']),score=difference(score,row['regimes']['fitted']['score']))
        passed = status==row['status'] and error==row.get('error') and max(errors.values())<=1e-9 and bool(calls)
        payload=dict(case=name(row),row=row,status=status,error=error,diagnostics=diag,value=value,score=score,
                     calls=calls,replay_errors=errors,passed=passed)
        save(directory/(name(row)+'.json'),payload)
        summary.append(dict(case=name(row),status=status,captured_calls=len(calls),replay_errors=errors,passed=passed))
        print(json.dumps(summary[-1]),flush=True)
        if not passed:raise RuntimeError('Actual consumer replay failed')
    return dict(cases=summary,passed=len(summary)==4 and all(r['passed'] for r in summary))


def isolate(tf, directory):
    from bayesfilter.score_study.iapf_fit_tf import bounded_density_fit, _density_profile
    from bayesfilter.score_study.gaussian_tf import parameterized_model, gaussian_log_density_and_tangent
    from bayesfilter.score_study.fitted_twist_tf import normalizer
    capture_dir=ROOT/'attempt01_gpu_capture'
    if not read(capture_dir/'summary.json')['passed']:raise RuntimeError('Capture not valid')
    prior=preserve(directory,PRIOR,'attempt03_gpu_consumer/results.json')
    exact=preserve(directory,GEOMETRY,'attempt01_cpu_reference/exact-guides.json')
    results=[];reference=[];traces={};max_replay=0.;max_terminal=0.;max_identity=0.
    def host(value):
        if isinstance(value,dict):return {k:host(v) for k,v in value.items()}
        return value.numpy().tolist()
    for selected in read(capture_dir/'summary.json')['cases']:
        payload=read(capture_dir/(selected['case']+'.json'));row=payload['row'];call=payload['calls'][-1]
        source=capture_dir/(selected['case']+'.json');destination=directory/'captures'/source.name
        destination.parent.mkdir(exist_ok=True);shutil.copy2(source,destination)
        d=row['d'];N=len(call['input']['clouds'][0]);T=4;dtype=tf.float64;config=row['config']
        matching=next(r for r in prior if r['d']==d and r['seed']==row['seed'] and r['status']=='complete')
        if difference(matching['observations']['fitted'],row['observations']['fitted'])!=0:raise RuntimeError('Oracle observation mismatch')
        oracle=exact[name(matching)+'/fitted']
        @tf.function(input_signature=[tf.TensorSpec([6],dtype),tf.TensorSpec([T,d],dtype),
            tf.TensorSpec([T,N,d],dtype),tf.TensorSpec([T,d],dtype),tf.TensorSpec([T,d,d],dtype),
            tf.TensorSpec([T],dtype),tf.TensorSpec([T,d],dtype),tf.TensorSpec([T,d,d],dtype)],jit_compile=True)
        def targets(theta,y,clouds,centers,covs,floors,exact_c,exact_v):
            A,_,H,_,_,_,_,_,Q,_,R,_=parameterized_model(theta,d,d)
            actual=[];true=[]
            for t in range(T):
                x=clouds[t]
                lg=gaussian_log_density_and_tangent(y[t]-tf.einsum('ij,nj->ni',H,x),
                    tf.zeros([6,N,d],dtype),R,tf.zeros([6,1,d,d],dtype))[0]
                lf=normalizer(tf.einsum('ij,nj->ni',A,x),tf.zeros([6,N,d],dtype),Q,
                    tf.zeros([6,d,d],dtype),centers[t+1],covs[t+1],floors[t+1])[0] if t+1<T else tf.zeros([N],dtype)
                actual.append(lg+lf)
                true.append(gaussian_log_density_and_tangent(x-exact_c[t],tf.zeros([6,N,d],dtype),
                    exact_v[t],tf.zeros([6,1,d,d],dtype))[0])
            return tf.stack(actual),tf.stack(true)
        inp=call['input'];out=call['result']
        tensors=[tf.constant(v,dtype) for v in [inp['theta'],inp['observations'],inp['clouds'],
            out['centers'],out['covariances'],out['floors'],oracle['centers'],oracle['covariances']]]
        actual,true=targets(*tensors)
        max_terminal=max(max_terminal,float(tf.reduce_max(tf.abs((actual[-1]-tf.reduce_max(actual[-1]))-(true[-1]-tf.reduce_max(true[-1])))).numpy()))
        controls=dict(mean_bound=config['mean_bound'],sd_lower=config['sd_lower'],sd_upper=config['sd_upper'],
            max_backtracks=config['max_backtracks'],tolerance=config['fit_tolerance'],floor_ratio=config['floor_ratio'])
        signature=[tf.TensorSpec([N,d],dtype),tf.TensorSpec([N],dtype),tf.TensorSpec([d],dtype),
            tf.TensorSpec([d,d],dtype),tf.TensorSpec([d],dtype),tf.TensorSpec([d,d],dtype)]
        def build(objective):
            scale=config['fit_objective_scale'] if objective=='density_l2' else 'native'
            @tf.function(input_signature=signature,jit_compile=True)
            def kernel(x,logb,exact_c,exact_v,saved_c,saved_v):
                mean=tf.reduce_mean(x,0);sd=tf.sqrt(tf.reduce_mean((x-mean)**2,0));z=(x-mean)/sd
                logb=logb-tf.reduce_max(logb)
                def fit(initialization,steps):
                    return bounded_density_fit(x,logb,**controls,max_steps=steps,objective=objective,
                        initialization=initialization,objective_scale=scale)
                initial=fit(config['fit_initialization'],0);final=fit(config['fit_initialization'],config['max_fit_steps'])
                cloud=fit('cloud_moments',0);qr=fit('log_quadratic',0)
                def profile(c,v):
                    params=tf.concat([(c-mean)/sd,.5*tf.math.log(tf.linalg.diag_part(v))-tf.math.log(sd)],0)
                    loss,grad,_,shape,_=_density_profile(z,logb,params)
                    _,shape_grad,_,_,_=_density_profile(z,logb,params,'relative_shape')
                    logp=-tf.reduce_sum(params[d:])-.5*tf.reduce_sum(((z-params[:d])*tf.exp(-params[d:]))**2,1)
                    energy=tf.exp(tf.reduce_logsumexp(2*logp)-tf.math.log(tf.cast(N,dtype)))
                    delta=c-exact_c
                    kl=.5*(tf.reduce_sum(tf.linalg.diag_part(exact_v)/tf.linalg.diag_part(v))+
                        tf.reduce_sum(delta**2/tf.linalg.diag_part(v))-tf.cast(d,dtype)+
                        tf.reduce_sum(tf.math.log(tf.linalg.diag_part(v)))-tf.linalg.logdet(exact_v))
                    return dict(parameters=params,loss=loss,shape=shape,energy=energy,gradient=grad,
                        shape_gradient=shape_grad,KL=kl,center=c,covariance=v)
                states={k:profile(v[0],v[1]) for k,v in [('initial',initial),('final',final),('cloud',cloud),('QR',qr)]}
                states['saved']=profile(saved_c,saved_v)
                states['exact_diagonal']=profile(exact_c,tf.linalg.diag(tf.linalg.diag_part(exact_v)))
                # Full exact Gaussian is not diagonal. Evaluate its empirical objective directly.
                stdv=exact_v/(sd[:,None]*sd[None,:]);delta=(x-exact_c)/sd
                solved=tf.transpose(tf.linalg.solve(stdv,tf.transpose(delta)))
                logp=-.5*tf.linalg.logdet(stdv)-.5*tf.reduce_sum(delta*solved,1)
                a=tf.reduce_max(logp);p=tf.exp(logp-a);b=tf.exp(logb)
                residual=p-tf.reduce_sum(p*b)/tf.reduce_sum(b*b)*b
                energy=tf.exp(2*a)*tf.reduce_mean(p*p);shape=tf.reduce_sum(residual*residual)/tf.reduce_sum(p*p)
                full=dict(loss=energy*shape,energy=energy,shape=shape,KL=tf.zeros([],dtype))
                return dict(z=z,log_target=logb,states=states,full_exact=full,
                    initial_info=initial[3],final_info=final[3],QR_info=qr[3],floor=final[2])
            return kernel
        kernels={key:build(key) for key in ('density_l2','relative_shape')}
        for t in range(T):
            for target,logs in [('actual',actual),('exact',true)]:
                for objective,kernel in kernels.items():
                    computed=host(kernel(tensors[2][t],logs[t],tensors[6][t],tensors[7][t],tensors[3][t],tensors[4][t]))
                    states=computed.pop('states');z=computed.pop('z');logb=computed.pop('log_target')
                    key=f"{name(row)}/t{t}/{target}/{objective}"
                    item=dict(case=name(row),d=d,N=N,time=t,target=target,objective=objective,
                        states=states,**computed,heuristic_dominance_verdict='fixed_cloud_only_prior_filtering_veto_unresolved')
                    if target=='actual' and objective=='density_l2':
                        replay=max(difference(states['final']['center'],out['centers'][t]),
                            difference(states['final']['covariance'],out['covariances'][t]),difference(item['floor'],out['floors'][t]))
                        max_replay=max(max_replay,replay);item['captured_fit_replay_error']=replay
                    for state,metrics in states.items():
                        max_identity=max(max_identity,abs(metrics['loss']-metrics['energy']*metrics['shape'])/(1+abs(metrics['loss'])))
                    reference.append(dict(key=key,z=z,log_target=logb,states=states))
                    results.append(item)
        traces[name(row)]=dict(targets=targets.experimental_get_tracing_count(),**{k:v.experimental_get_tracing_count() for k,v in kernels.items()})
        print(name(row)+' isolated',flush=True)
    save(directory/'results.json',results)
    (directory/'reference-input.R').write_text('input <- '+r_literal(reference)+'\n')
    checks=dict(case_count=len(results)==64,actual_fit_replay=max_replay<=1e-8,terminal_target=max_terminal<=1e-9,
        density_identity=max_identity<=1e-10,trace_counts=all(v==1 for r in traces.values() for v in r.values()),
        finite=all(math.isfinite(s['loss']) and math.isfinite(s['shape']) and math.isfinite(s['KL']) for r in results for s in r['states'].values()))
    return dict(evaluations=len(results),checks=checks,max_replay_error=max_replay,
        max_terminal_target_error=max_terminal,max_density_identity_error=max_identity,traces=traces,passed=all(checks.values()))


def verify(directory):
    source=ROOT/'attempt02_gpu_isolate/reference-input.R'
    shutil.copy2(source,directory/'input.R')
    command=['Rscript','--vanilla','docs/benchmarks/check_iapf_fit_inputs.R',str(directory/'input.R'),str(directory/'checks.csv')]
    started=time.monotonic()
    with (directory/'R.log').open('w') as log:
        completed=subprocess.run(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,timeout=240)
    save(directory/'R-run.json',dict(command=command,exit_code=completed.returncode,wall_seconds=time.monotonic()-started))
    if completed.returncode:raise RuntimeError('Independent R check failed')
    with (directory/'checks.csv').open() as stream:rows=list(csv.DictReader(stream))
    errors={k:max(float(r[k]) for r in rows) for k in ('loss_error','energy_error','shape_error','gradient_error','shape_gradient_error','identity_error')}
    return dict(rows=len(rows),max_errors=errors,passed=len(rows)==384 and max(errors.values())<=1e-8)


def worker(args):
    directory=ROOT/args.attempt;directory.mkdir(exist_ok=False)
    if args.mode=='verify':
        result=verify(directory)
        environment=dict(device='cpu',GPU='intentionally hidden',backend='base R independent reference')
    else:
        tf,environment=setup(args.device)
        result=capture(tf,directory) if args.mode=='capture' else isolate(tf,directory)
        if args.device=='gpu':environment['allocator']=tf.config.experimental.get_memory_info('GPU:0')
    save(directory/'summary.json',dict(environment=environment,**result))
    if not result['passed']:raise RuntimeError('Required checks failed')


def launch(args):
    if not args.attempt.isidentifier():raise ValueError('Unique versioned attempt required')
    manifest=ROOT/(args.attempt+'-launch.json')
    if manifest.exists():raise FileExistsError(manifest)
    budget=read(ROOT/'budget.json');resource=args.device;seconds=300
    if len(list(ROOT.glob('*-launch.json')))>=6:raise RuntimeError('Attempt cap')
    if budget['phase_'+resource+'_seconds']+seconds>(3600 if resource=='cpu' else 1800) or budget['remaining_'+resource+'_seconds']<seconds:raise RuntimeError('Budget cap')
    sources=['docs/benchmarks/diagnose_iapf_fit_inputs.py','docs/benchmarks/check_iapf_fit_inputs.R',
        'docs/benchmarks/diagnose_iapf_adaptive_consumer.py','docs/benchmarks/reference_iapf_paper.R',PLAN]
    sources += [str(p.relative_to(REPO)) for p in (REPO/'bayesfilter/score_study').glob('*.py')]
    sources += ['bayesfilter/runtime/gpu_memory_policy.py']
    hashes={}
    for path in sources:
        p=REPO/path
        if not p.exists():
            if args.mode=='capture' and path.endswith('.R') and 'check_iapf' in path:continue
            raise FileNotFoundError(p)
        target=ROOT/(args.attempt+'-source')/path;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,target);hashes[path]=sha(p)
    command=[sys.executable,str(Path(__file__).resolve()),'--worker','--attempt',args.attempt,'--mode',args.mode,'--device',args.device]
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'-1' if resource=='cpu' else 'GPU-68251639-fe82-8f81-3ccc-2953c32e805b',
         'TF_FORCE_GPU_ALLOW_GROWTH':'true','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
         'TF_NUM_INTRAOP_THREADS':'1','TF_NUM_INTEROP_THREADS':'1','BAYESFILTER_PRELOAD_CUSTOM_OP':'0'}
    record=dict(command=command,device=resource,mode=args.mode,status='running',sources=hashes,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),plan=PLAN,
        prior_manifest=sha(PRIOR/'manifest.json'),geometry_manifest=sha(GEOMETRY/'manifest.json'),
        seeds='Original phase4 observations/configuration and SHA256 seed namespace replayed')
    save(manifest,record);started=time.monotonic()
    with (ROOT/(args.attempt+'.log')).open('x') as log:
        try:code=subprocess.run(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=seconds).returncode
        except subprocess.TimeoutExpired:code=124
    elapsed=time.monotonic()-started;record.update(status='finished',exit_code=code,wall_seconds=elapsed);save(manifest,record)
    budget=read(ROOT/'budget.json');budget['phase_'+resource+'_seconds']+=elapsed;budget['remaining_'+resource+'_seconds']-=elapsed;save(ROOT/'budget.json',budget)
    print(json.dumps({k:record[k] for k in ('status','exit_code','wall_seconds')}));return code


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--attempt',required=True)
    parser.add_argument('--device',choices=['cpu','gpu'],default='cpu')
    parser.add_argument('--mode',choices=['capture','isolate','verify'],default='capture')
    parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    if args.worker:worker(args)
    else:sys.exit(launch(args))
