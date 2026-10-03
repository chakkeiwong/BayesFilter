"""Diagnostic covariance/floor factorial on preserved data; no promotion route."""
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

from diagnose_iapf_adaptive_consumer import REPO, r_literal, save, setup
from diagnose_iapf_score_regression import materialize, read, sha

ROOT = REPO/'docs/plans/artifacts/iapf-covariance-floor-factorial-20260922-01'
PRIOR = REPO/'docs/plans/artifacts/iapf-score-regression-20260922-01'
PLAN = 'docs/plans/iapf-covariance-floor-factorial-2026-09-22.md'
GPU = 'GPU-68251639-fe82-8f81-3ccc-2953c32e805b'
ARMS = {f'{cov}_{floor}': (cov, floor) for cov in ('diagonal', 'full')
        for floor in ('positive', 'negligible')}
HEURISTICS = ['bootstrap', 'current_observation', 'full_oracle']


def verify_prior(directory):
    manifest = read(PRIOR/'manifest.json')
    checks = {p: sha(PRIOR/p) == digest for p, digest in manifest['outputs'].items()}
    if not all(checks.values()) or not read(PRIOR/'verification.json')['passed']:
        raise RuntimeError('Preserved phase12 artifact failed verification')
    save(directory/'input-verification.json', dict(manifest_sha256=sha(PRIOR/'manifest.json'), checks=checks))


def factorial(tf, directory):
    from diagnostic_iapf_score_regression_tf import make_recursive_score_fit, make_oracle
    from bayesfilter.score_study.fitted_twist_tf import make_fitted_twist_kernel
    from bayesfilter.score_study.gaussian_tf import parameterized_model
    verify_prior(directory)
    dtype = tf.float64
    cases = []; r_inputs = []; checks = {}
    for prior in read(PRIOR/'attempt04_gpu_downstream/cases.json'):
        case = prior['case']; d = prior['d']; N = prior['N']; T = prior['T']
        theta = tf.constant(prior['theta'], dtype); y = tf.constant(prior['observations'], dtype)
        regular = make_fitted_twist_kernel(d,d,N,T,include_numerical_trace=True)
        bootstrap = make_fitted_twist_kernel(d,d,N,T,constant_twist=True,include_numerical_trace=True)
        kernels = {name: make_recursive_score_fit(d,N,T,*modes) for name,modes in ARMS.items()}
        oracle = make_oracle(d,T)
        def draws(keys):
            specs = [('initial',[N,d],True),('process',[T,N,d],True),
                     ('ancestors',[T+1,N],False),('mixture',[T,N],False)]
            return [(tf.random.stateless_normal if normal else tf.random.stateless_uniform)(shape,keys[name],dtype=dtype)
                    for name,shape,normal in specs]
        placeholder = (tf.zeros([T,d],dtype),tf.eye(d,batch_shape=[T],dtype=dtype),tf.zeros([T],dtype))
        clouds = bootstrap(theta,y,*draws(prior['seeds']['pilot']),*placeholder)[2]
        exact_c, exact_v, _, _, _ = oracle(theta,y)
        coefficients = {}; fits = {}; fit_seconds = {}; accepted = {}
        for name,kernel in kernels.items():
            started = time.monotonic(); out = kernel(theta,y,clouds); out[0].numpy()
            fit_seconds[name] = time.monotonic()-started
            coefficients[name] = out[:3]; accepted[name] = bool(out[3])
            fits[name] = materialize(dict(valid=out[3],diagnostics=out[4],
                columns=['valid','cloud_margin','precision_margin','guard_tolerance','whitening_error','skew_norm','score_residual']))
        replay_error = max(float(tf.reduce_max(tf.abs(a-tf.constant(b,dtype))).numpy())
            for a,b in zip(coefficients['diagonal_positive'],prior['coefficients']['score_projection']))
        oracle_error = max(float(tf.reduce_max(tf.abs(a-b)).numpy())
            for a,b in zip(coefficients['full_negligible'][:2],(exact_c,exact_v)))
        checks[case+'/replay_coefficients'] = accepted['diagonal_positive'] and replay_error<=1e-8
        checks[case+'/oracle_coefficients'] = accepted['full_negligible'] and oracle_error<=1e-8
        records = [dict(row) for row in prior['records'] if row['method'] in HEURISTICS]
        old_values = {(r['method'],r['replicate']):r['value'] for r in prior['records']}
        initial_noises = []; predictions = []
        for replicate,keys in enumerate(prior['seeds']['final']):
            final = draws(keys); initial_noises.append(materialize(final[0]))
            predictions.append(old_values['full_oracle',replicate])
            for name in ARMS:
                if not accepted[name]:
                    records.append(dict(method=name,replicate=replicate,status='fit_rejected',value=None,squared_error=None))
                    continue
                out = regular(theta,y,*final,*coefficients[name]); value = float(out[0].numpy())
                trace = out[-1]; cdf = trace['ancestor_cdf']
                weights = cdf-tf.concat([tf.zeros([T+1,1],dtype),cdf[:,:-1]],axis=1)
                ess = 1/tf.reduce_sum(weights**2,axis=1)
                probability = trace['gaussian_probability']
                finite = math.isfinite(value) and bool(tf.reduce_all(tf.math.is_finite(out[1])))
                finite &= bool(tf.reduce_all(tf.math.is_finite(ess))) and bool(tf.reduce_all(tf.math.is_finite(probability)))
                record = dict(method=name,replicate=replicate,status='complete' if finite else 'consumer_invalid',
                    value=value if finite else None,squared_error=(value-prior['exact_value'])**2 if finite else None,
                    minimum_ess=float(tf.reduce_min(ess).numpy()),
                    gaussian_probability_min=float(tf.reduce_min(probability).numpy()),
                    gaussian_probability_mean=float(tf.reduce_mean(probability).numpy()))
                records.append(record)
                if name=='diagonal_positive':
                    error = abs(value-old_values['score_projection',replicate])
                    record['replay_error'] = error
                    checks[case+f'/replay_value_{replicate}'] = finite and error<=1e-8
                if name=='full_negligible':
                    error = abs(value-predictions[-1]); record['oracle_value_error'] = error
                    checks[case+f'/oracle_value_{replicate}'] = finite and error<=1e-8
                    checks[case+f'/oracle_ess_{replicate}'] = bool(tf.reduce_max(tf.abs(ess[1:]/N-1))<=1e-8)
        traces = {name:kernel.experimental_get_tracing_count() for name,kernel in
                  {**kernels,'filter':regular,'bootstrap':bootstrap,'oracle':oracle}.items()}
        checks[case+'/single_traces'] = all(n==1 for n in traces.values())
        result = dict(case=case,d=d,N=N,T=T,seed=prior['seed'],theta=prior['theta'],observations=prior['observations'],
            seeds=prior['seeds'],exact_value=prior['exact_value'],accepted=accepted,fits=fits,fit_seconds=fit_seconds,
            coefficients=materialize(coefficients),records=records,traces=traces,
            replay_coefficient_error=replay_error,oracle_coefficient_error=oracle_error)
        save(directory/(case+'.json'),result); cases.append(result)
        r_inputs.append(dict(case=case,d=d,y=prior['observations'],model=materialize(parameterized_model(theta,d,d)[::2]),
            clouds=materialize(clouds),coefficients=materialize(coefficients),accepted=accepted,
            exact_value=prior['exact_value'],initial_noises=initial_noises,initial_predictions=predictions))
        save(directory/'partial.json',[dict(case=c['case'],accepted=c['accepted']) for c in cases])
        print(case+' '+str(accepted),flush=True)
    save(directory/'cases.json',cases)
    (directory/'reference-input.R').write_text('input <- '+r_literal(r_inputs)+'\n')
    return dict(passed=all(checks.values()),checks=checks,cases=len(cases))


def reference(directory):
    prior = ROOT/'attempt01_gpu_factorial'
    if not read(prior/'summary.json')['passed']:
        raise RuntimeError('GPU engineering check failed')
    copied = directory/'reference-input.R'; shutil.copy2(prior/'reference-input.R',copied)
    command = ['Rscript','--vanilla','docs/benchmarks/check_iapf_covariance_floor.R',str(copied),str(directory)]
    started = time.monotonic()
    with (directory/'R.log').open('x') as log:
        code = subprocess.run(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,timeout=500).returncode
    save(directory/'R-run.json',dict(command=command,exit_code=code,wall_seconds=time.monotonic()-started))
    if code: raise RuntimeError('Independent R check failed')
    cases = read(prior/'cases.json'); per_case = []; conditional = []; heuristic = []; pairs = []
    methods = HEURISTICS+list(ARMS)
    for case in cases:
        for method in methods:
            rows = [r for r in case['records'] if r['method']==method]
            completed = len(rows)==8 and all(r['status']=='complete' for r in rows)
            per_case.append(dict(case=case['case'],d=case['d'],method=method,complete=completed,
                mse=sum(r['squared_error'] for r in rows)/8 if completed else None))
    for d in [2,5,10]:
        means = {}
        for method in methods:
            rows = [r for r in per_case if r['d']==d and r['method']==method]
            complete = len(rows)==4 and all(r['complete'] for r in rows)
            means[method] = sum(r['mse'] for r in rows)/4 if complete else None
            conditional.append(dict(d=d,method=method,data_sets=len(rows),complete=complete,mse=means[method]))
        for method in ARMS:
            equivalent = ['full_oracle'] if method=='full_negligible' else []
            losses = [h for h in HEURISTICS if h not in equivalent and means[method] is not None and means[method]>means[h]]
            heuristic.append(dict(d=d,method=method,losses=losses,
                promotion_veto=means[method] is None or bool(losses),ranking_supported=False,
                verified_finite_program_equivalence=equivalent))
    for case in cases:
        values = {r['method']:r['mse'] for r in per_case if r['case']==case['case']}
        if all(values[m] is not None for m in ARMS):
            dp,dn,fp,fn = [values[m] for m in ARMS]
            pairs.append(dict(case=case['case'],d=case['d'],
                covariance_effect_positive=dp-fp,covariance_effect_negligible=dn-fn,
                floor_effect_diagonal=dp-dn,floor_effect_full=fp-fn,
                interaction=(dp-dn)-(fp-fn)))
    for filename,value in [('per-case-mse.json',per_case),('conditional-summary.json',conditional),
                           ('heuristic-dominance.json',heuristic),('paired-effects.json',pairs)]:
        save(directory/filename,value)
    with (directory/'reference-checks.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    return dict(passed=len(rows)==48,reference_rows=len(rows),
        maximum_reference_error=max(float(r['candidate_error']) for r in rows),
        maximum_oracle_error=max(float(r['oracle_error']) for r in rows),
        maximum_initial_error=max(float(r['initial_error']) for r in rows))


def worker(args):
    directory=ROOT/args.attempt; directory.mkdir(exist_ok=False)
    if args.mode=='gpu_factorial':
        tf,environment=setup('gpu'); save(directory/'environment.json',environment)
        result=factorial(tf,directory);result['allocator']=tf.config.experimental.get_memory_info('GPU:0')
    else:
        save(directory/'environment.json',dict(cpu_only=True,gpu_intentionally_hidden=True,
            visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),python=sys.version,
            R=subprocess.check_output(['Rscript','--version'],text=True,stderr=subprocess.STDOUT).strip()))
        result=reference(directory)
    save(directory/'summary.json',result)
    if not result['passed']: raise RuntimeError('Engineering verification failed')


def launch(args):
    if not args.attempt.isidentifier(): raise ValueError('Versioned attempt required')
    manifest=ROOT/(args.attempt+'-launch.json')
    if manifest.exists(): raise FileExistsError(manifest)
    resource='gpu' if args.mode=='gpu_factorial' else 'cpu'; seconds=600
    budget=read(ROOT/'budget.json')
    if len(list(ROOT.glob('*-launch.json')))>=4 or budget['phase_'+resource+'_seconds']+seconds>3600 or budget['remaining_'+resource+'_seconds']<seconds:
        raise RuntimeError('Attempt or budget cap')
    paths=[PLAN,'docs/benchmarks/diagnose_iapf_covariance_floor.py','docs/benchmarks/check_iapf_covariance_floor.R',
        'docs/benchmarks/diagnostic_iapf_score_regression_tf.py','docs/benchmarks/diagnose_iapf_score_regression.py',
        'docs/benchmarks/diagnose_iapf_adaptive_consumer.py','docs/benchmarks/reference_iapf_paper.R',
        'bayesfilter/runtime/gpu_memory_policy.py']
    paths += [str(p.relative_to(REPO)) for p in (REPO/'bayesfilter/score_study').glob('*.py')]
    hashes={}
    for relative in paths:
        source=REPO/relative;dest=ROOT/(args.attempt+'-source')/relative
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest);hashes[relative]=sha(source)
    env={**os.environ,'CUDA_VISIBLE_DEVICES':GPU if resource=='gpu' else '-1',
        'TF_FORCE_GPU_ALLOW_GROWTH':'true','BAYESFILTER_PRELOAD_CUSTOM_OP':'0','OMP_NUM_THREADS':'1',
        'OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','TF_NUM_INTRAOP_THREADS':'1','TF_NUM_INTEROP_THREADS':'1'}
    command=[sys.executable,str(Path(__file__).resolve()),'--mode',args.mode,'--attempt',args.attempt,'--worker']
    record=dict(status='running',command=command,resource=resource,plan=PLAN,result=str(ROOT/'result.md'),
        python=sys.executable,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        environment={k:env[k] for k in ['CUDA_VISIBLE_DEVICES','TF_FORCE_GPU_ALLOW_GROWTH','BAYESFILTER_PRELOAD_CUSTOM_OP',
            'OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS']},
        data_version='phase12 preserved12cases; manifest-verified',seeds='phase12 saved per-case seed dictionaries',sources=hashes)
    save(manifest,record);started=time.monotonic()
    with (ROOT/(args.attempt+'.log')).open('x') as log:
        try: code=subprocess.run(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=seconds).returncode
        except subprocess.TimeoutExpired: code=124
    elapsed=time.monotonic()-started;record.update(status='finished',exit_code=code,wall_seconds=elapsed);save(manifest,record)
    budget['phase_'+resource+'_seconds']+=elapsed;budget['remaining_'+resource+'_seconds']-=elapsed;save(ROOT/'budget.json',budget)
    print(json.dumps(dict(exit_code=code,wall_seconds=elapsed)));return code


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['gpu_factorial','cpu_reference'],required=True)
    parser.add_argument('--attempt',required=True);parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    if args.worker: worker(args)
    else: sys.exit(launch(args))
