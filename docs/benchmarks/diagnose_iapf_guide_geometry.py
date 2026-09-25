"""Diagnostic oracle geometry; no production filter or fitting implementation."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from diagnose_iapf_adaptive_consumer import REPO, r_literal, save, setup
from diagnose_iapf_positive_floor import probability_stats, sha, log_ratios

ROOT = REPO / 'docs/plans/artifacts/iapf-guide-geometry-20260922-01'
PRIOR = REPO / 'docs/plans/artifacts/iapf-initialization-isolation-20260922-01'
FLOOR = REPO / 'docs/plans/artifacts/iapf-positive-floor-20260922-01'
PLAN = 'docs/plans/iapf-guide-geometry-diagnosis-2026-09-22.md'


def read(path):
    return json.loads(path.read_text())


def preserve(directory, relative):
    path = PRIOR / relative
    if sha(path) != read(PRIOR / 'manifest.json')['outputs'][relative]:
        raise RuntimeError('Changed prior input: ' + relative)
    target = directory / 'inputs' / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)
    return read(target)


def case_name(row):
    return f"d{row['d']}-s{row['seed']}-{row['initialization']}-{row['objective_scale']}"


def reference(tf, directory):
    from bayesfilter.score_study.gaussian_tf import parameterized_model
    rows = preserve(directory, 'attempt03_gpu_consumer/results.json')
    payload = []; kernels = {}
    for row in rows:
        if row['status'] != 'complete':
            continue
        name = case_name(row); d = row['d']
        inputs = preserve(directory, 'attempt03_gpu_consumer/' + name + '/final-inputs.json')
        if d not in kernels:
            # Factory binds dimension rather than retaining a loop variable.
            def make_kernel(d):
                @tf.function(input_signature=[tf.TensorSpec([6], tf.float64)], jit_compile=True)
                def model(theta):
                    return parameterized_model(theta, d, d)[::2]
                return model
            kernels[d] = make_kernel(d)
        A, H, m0, P0, Q, R = kernels[d](tf.constant(inputs['theta'], tf.float64))
        payload.append(dict(case=name, A=A.numpy().tolist(), H=H.numpy().tolist(),
            m0=m0.numpy().tolist(), P0=P0.numpy().tolist(), Q=Q.numpy().tolist(), R=R.numpy().tolist(),
            initial_noise=inputs['draws'][0], observations=row['observations']))
    source = directory / 'input.R'
    source.write_text('input <- ' + r_literal(payload) + '\n')
    command = ['Rscript', '--vanilla', 'docs/benchmarks/diagnose_iapf_guide_geometry.R', str(source), str(directory)]
    started = time.monotonic()
    with (directory / 'R.log').open('x') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=120, cwd=REPO)
    save(directory / 'R-run.json', dict(command=command, exit_code=result.returncode,
                                       wall_seconds=time.monotonic()-started))
    if result.returncode:
        raise RuntimeError('Independent R oracle failed')
    exact = {}
    with (directory / 'metrics.csv').open() as stream:
        for record in csv.DictReader(stream):
            key = record['case'] + '/' + record['regime']; d = int(record['d'])
            exact[key] = dict(metrics={k:float(v) for k,v in record.items() if k not in ('case','regime')},
                              centers=[[0.]*d for _ in range(4)],
                              covariances=[[[0.]*d for _ in range(d)] for _ in range(4)])
    with (directory / 'coefficients.csv').open() as stream:
        for r in csv.DictReader(stream):
            item = exact[r['case']+'/'+r['regime']]; t=int(r['time'])-1; j=int(r['row'])-1; k=int(r['column'])-1
            if k < 0:
                item['centers'][t][j] = float(r['value'])
            else:
                item['covariances'][t][j][k] = float(r['value'])
    errors = []
    for row in rows:
        if row['status'] == 'complete':
            for regime in row['observations']:
                item = exact[case_name(row)+'/'+regime]
                errors.append(abs(item['metrics']['exact_kalman']-row['regimes'][regime]['exact_value']))
    save(directory / 'exact-guides.json', exact)
    return dict(cases=len(exact), max_R_TF_Kalman_error=max(errors),
        max_message_identity_error=max(v['metrics']['identity_error'] for v in exact.values()),
        max_integration_identity_error=max(v['metrics']['integration_identity_error'] for v in exact.values()),
        positive_definite=all(v['metrics']['minimum_eigenvalue']>0 for v in exact.values()),
        passed=len(exact)==38 and max(errors)<=1e-9,
        trace_counts={str(k):v.experimental_get_tracing_count() for k,v in kernels.items()})


def geometry_kernel(tf, d):
    dtype = tf.float64
    @tf.function(input_signature=[tf.TensorSpec([4,d],dtype),tf.TensorSpec([4,d,d],dtype),
        tf.TensorSpec([4,d],dtype),tf.TensorSpec([4,d,d],dtype)],jit_compile=True)
    def kernel(center, cov, exact_center, exact_cov):
        L = tf.linalg.cholesky(cov); diff = center-exact_center
        mean = .5*tf.reduce_sum(diff*tf.linalg.cholesky_solve(L,diff[...,None])[...,0],-1)
        covariance = .5*(tf.linalg.trace(tf.linalg.cholesky_solve(L,exact_cov))-d
                         +tf.linalg.logdet(cov)-tf.linalg.logdet(exact_cov))
        floor = .5*(tf.reduce_sum(tf.math.log(tf.linalg.diag_part(exact_cov)),-1)-tf.linalg.logdet(exact_cov))
        return mean, covariance, mean+covariance, floor
    return kernel


def consumer(tf, directory):
    from bayesfilter.score_study.fitted_twist_tf import make_fitted_twist_kernel
    reference_dir = ROOT / 'attempt01_cpu_reference'
    if not read(reference_dir / 'summary.json')['passed']:
        raise RuntimeError('Required reference incomplete')
    exact = read(reference_dir / 'exact-guides.json')
    shutil.copy2(reference_dir/'exact-guides.json',directory/'exact-guides.json')
    old = preserve(directory, 'attempt03_gpu_consumer/results.json')
    floor_results = {(r['case'],r['regime'],r['arm']):r for r in read(FLOOR/'attempt03_gpu_replay/results.json')}
    kernels = {}; geometry = {}; records = []; errors = dict(baseline_replay=0., tail_replay=0.,
        oracle_initial_value=0., oracle_probability=0., oracle_postinitial_ess=0., diagonal_KL_identity=0.)
    checks = dict(finite=True, valid_cdf=True)
    for row in old:
        if row['status'] != 'complete':
            continue
        name=case_name(row);d=row['d'];n=row['diagnostics']['actual_particle_count']
        inputs=preserve(directory,'attempt03_gpu_consumer/'+name+'/final-inputs.json')
        theta=tf.constant(inputs['theta'],tf.float64)
        draws=[tf.constant(v,tf.float64) for v in inputs['draws']]
        fitted=[tf.constant(v,tf.float64) for v in inputs['coefficients']]
        if (d,n) not in kernels:
            kernels[d,n]=make_fitted_twist_kernel(d,d,n,4,include_numerical_trace=True)
        if d not in geometry:
            geometry[d]=geometry_kernel(tf,d)
        for regime,observations in row['observations'].items():
            target=exact[name+'/'+regime]; y=tf.constant(observations,tf.float64)
            full_center=tf.constant(target['centers'],tf.float64)
            full_cov=tf.constant(target['covariances'],tf.float64)
            guides={'fitted':fitted[:2], 'KL_diagonal_oracle':(full_center,tf.linalg.diag(tf.linalg.diag_part(full_cov))),
                    'full_oracle':(full_center,full_cov)}
            for guide,(center,cov) in guides.items():
                stats=geometry[d](center,cov,full_center,full_cov)
                mean,covariance,kl,kl_floor=[v.numpy().tolist() for v in stats]
                if guide=='KL_diagonal_oracle':
                    errors['diagonal_KL_identity']=max(errors['diagonal_KL_identity'],max(abs(a-b) for a,b in zip(kl,kl_floor)))
                peak=-.5*(tf.constant(d*math.log(2*math.pi),tf.float64)+tf.linalg.logdet(cov))
                floors={'baseline':fitted[2] if guide=='fitted' else peak+tf.constant(math.log(.01),tf.float64),
                        'R_tail_N_minus2':peak+tf.constant(log_ratios(d,n)['R_tail_N_minus2'],tf.float64),
                        'negligible':peak-tf.constant(1000.,tf.float64)}
                for floor,lf in floors.items():
                    out=kernels[d,n](theta,y,*draws,center,cov,lf)
                    trace=out[-1]; cdf=trace['ancestor_cdf']; probabilities=trace['gaussian_probability']
                    weights=cdf-tf.concat([tf.zeros([5,1],tf.float64),cdf[:,:-1]],1)
                    ess=1/tf.reduce_sum(weights**2,1)
                    value=float(out[0].numpy()); score=out[1].numpy().tolist()
                    if guide=='fitted' and floor!='negligible':
                        saved=floor_results[name,regime,floor]
                        error=max(abs(value-saved['value']),max(abs(a-b) for a,b in zip(score,saved['score'])))
                        key='baseline_replay' if floor=='baseline' else 'tail_replay'
                        errors[key]=max(errors[key],error)
                    if guide=='full_oracle' and floor=='negligible':
                        errors['oracle_initial_value']=max(errors['oracle_initial_value'],abs(value-target['metrics']['initial_sample_prediction']))
                        errors['oracle_probability']=max(errors['oracle_probability'],float(tf.reduce_max(tf.abs(probabilities-1)).numpy()))
                        errors['oracle_postinitial_ess']=max(errors['oracle_postinitial_ess'],float(tf.reduce_max(tf.abs(ess[1:]/n-1)).numpy()))
                    checks['finite'] &= all(bool(tf.reduce_all(tf.math.is_finite(v)).numpy()) for v in [out[0],out[1],out[2],probabilities,cdf,*stats])
                    checks['valid_cdf'] &= bool(tf.reduce_all(weights>=-1e-12).numpy()) and bool(tf.reduce_all(tf.abs(cdf[:,-1]-1)<1e-10).numpy())
                    exact_value=target['metrics']['exact_kalman']; saved=row['regimes'][regime]
                    rec=dict(case=name,d=d,N=n,seed=row['seed'],regime=regime,guide=guide,floor=floor,
                        value=value,score=score,exact_value=exact_value,log_value_error=value-exact_value,
                        oracle_initial_sample_prediction=target['metrics']['initial_sample_prediction'],
                        oracle_initial_integration_error=target['metrics']['initial_sample_prediction']-exact_value,
                        guide_KL_by_time=kl,guide_KL_mean_by_time=mean,guide_KL_covariance_by_time=covariance,
                        diagonal_family_minimum_KL_by_time=kl_floor,
                        probability_by_time=probability_stats(tf,probabilities),ess_by_time=ess.numpy().tolist(),
                        heuristic_absolute_errors=saved['heuristic_absolute_errors'],
                        heuristic_dominance_verdict='descriptive_veto' if any(abs(value-exact_value)>v for v in saved['heuristic_absolute_errors'].values()) else 'no_observed_loss_this_draw')
                    records.append(rec)
            save(directory/'results.json',records)
        print(json.dumps(dict(case=name,status='complete')),flush=True)
    tolerances={k:1e-12 if k=='oracle_probability' else 1e-9 for k in errors}
    checks.update({k:v<=tolerances[k] for k,v in errors.items()})
    traces={str(k):v.experimental_get_tracing_count() for k,v in kernels.items()}
    checks['trace_counts']=all(v==1 for v in traces.values())
    checks['case_count']=len(records)==19*2*3*3
    return dict(evaluations=len(records),checks=checks,max_errors=errors,trace_counts=traces,
                passed=all(checks.values()),statistical_ranking=None,default_changed=False)


def worker(args):
    directory=ROOT/args.attempt;directory.mkdir(exist_ok=False)
    tf,environment=setup(args.device)
    with tf.device('/CPU:0' if args.device=='cpu' else '/GPU:0'):
        result=reference(tf,directory) if args.mode=='reference' else consumer(tf,directory)
    if args.device=='gpu':environment['allocator']=tf.config.experimental.get_memory_info('GPU:0')
    save(directory/'summary.json',dict(environment=environment,**result))
    if not result['passed']:raise RuntimeError('Required diagnostic checks failed')


def launch(args):
    if not args.attempt.isidentifier():raise ValueError('Unique versioned attempt required')
    manifest=ROOT/(args.attempt+'-launch.json')
    if manifest.exists():raise FileExistsError(manifest)
    budget=read(ROOT/'budget.json');resource=args.device;seconds=300
    if len(list(ROOT.glob('*-launch.json')))>=6:raise RuntimeError('Launch cap')
    if budget['phase_'+resource+'_seconds']+seconds>(3600 if resource=='cpu' else 900) or budget['remaining_'+resource+'_seconds']<seconds:raise RuntimeError('Budget')
    sources=['docs/benchmarks/diagnose_iapf_guide_geometry.py','docs/benchmarks/diagnose_iapf_guide_geometry.R',
        'docs/benchmarks/diagnose_iapf_positive_floor.py','docs/benchmarks/diagnose_iapf_adaptive_consumer.py',
        'docs/benchmarks/reference_iapf_paper.R','bayesfilter/score_study/fitted_twist_tf.py',
        'bayesfilter/score_study/gaussian_tf.py','bayesfilter/score_study/conditional_means_tf.py',
        'bayesfilter/score_study/complete_data_score_tf.py','bayesfilter/runtime/gpu_memory_policy.py',PLAN]
    hashes={}
    for name in sources:
        p=REPO/name;target=ROOT/(args.attempt+'-source')/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,target);hashes[name]=sha(p)
    command=[sys.executable,str(Path(__file__).resolve()),'--worker','--attempt',args.attempt,'--device',args.device,'--mode',args.mode]
    record=dict(command=command,device=args.device,mode=args.mode,sources=hashes,status='running',
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),plan=PLAN,
        input_identity={'prior_manifest':sha(PRIOR/'manifest.json'),'floor_manifest':sha(FLOOR/'manifest.json')},
        seeds='Saved actual draws; no new random draws')
    if args.mode=='consumer':record['reference_sha256']=sha(ROOT/'attempt01_cpu_reference/exact-guides.json')
    save(manifest,record)
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'-1' if args.device=='cpu' else 'GPU-68251639-fe82-8f81-3ccc-2953c32e805b',
        'TF_FORCE_GPU_ALLOW_GROWTH':'true','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
        'TF_NUM_INTRAOP_THREADS':'1','TF_NUM_INTEROP_THREADS':'1','BAYESFILTER_PRELOAD_CUSTOM_OP':'0'}
    started=time.monotonic()
    with (ROOT/(args.attempt+'.log')).open('x') as log:
        try:code=subprocess.run(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=seconds).returncode
        except subprocess.TimeoutExpired:code=124
    elapsed=time.monotonic()-started;record.update(status='finished',exit_code=code,wall_seconds=elapsed);save(manifest,record)
    budget=read(ROOT/'budget.json');budget['phase_'+resource+'_seconds']+=elapsed;budget['remaining_'+resource+'_seconds']-=elapsed;save(ROOT/'budget.json',budget)
    print(json.dumps({k:record[k] for k in ['status','exit_code','wall_seconds']}))
    return code


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--attempt',required=True)
    parser.add_argument('--device',choices=['cpu','gpu'],default='cpu')
    parser.add_argument('--mode',choices=['reference','consumer'],default='reference');parser.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    if args.worker:worker(args)
    else:sys.exit(launch(args))
