"""Fresh, paired one-pass diagnostic; no adaptive or paper replication claim."""
import csv
import hashlib
import math
import shutil
import time

from diagnose_iapf_score_regression import ROOT, read, sha, materialize, r_run
from diagnose_iapf_adaptive_consumer import controls, r_literal, save

N=256;T=8
METHODS=['bootstrap','current_observation','full_oracle','density_QR_peak','diagonal_QR','score_projection']
HEURISTICS=METHODS[:3]


def stream(d,seed,group,replicate,name):
    raw=hashlib.sha256(f'phase12:{d}:{seed}:{group}:{replicate}:{name}'.encode()).digest()
    return [int.from_bytes(raw[:4],'little')%(2**31-1),int.from_bytes(raw[4:8],'little')%(2**31-1)]


def downstream(tf,directory):
    if not read(ROOT/'attempt02_cpu_reference/summary.json')['passed']:
        raise RuntimeError('Independent reference gate missing')
    from diagnostic_iapf_score_regression_tf import make_recursive_score_fit,make_recursive_diagonal_qr_fit,make_oracle,gaussian_log
    from bayesfilter.score_study.iapf_adapter import FIT_DIAGNOSTIC_COLUMNS
    from bayesfilter.score_study.iapf_fit_tf import make_density_recursive_fit_kernel
    from bayesfilter.score_study.fitted_twist_tf import make_fitted_twist_kernel
    from bayesfilter.score_study.gaussian_tf import parameterized_model,make_data_kernel,make_gaussian_kernel
    dtype=tf.float64;config=controls();theta=tf.constant(config['fit_theta'],dtype)
    cases=[];r_inputs=[];checks={};all_draws=[]
    for d in [2,5,10]:
        regular=make_fitted_twist_kernel(d,d,N,T,include_numerical_trace=True)
        bootstrap=make_fitted_twist_kernel(d,d,N,T,constant_twist=True,include_numerical_trace=True)
        data=make_data_kernel(d,d,T);oracle=make_oracle(d,T);score_fit=make_recursive_score_fit(d,N,T)
        density=make_density_recursive_fit_kernel(d,d,N,T,4.,.2,4.,2000,30,1e-7,.01,
            initialization='log_quadratic',objective_scale='initial_peak')
        qr=make_recursive_diagonal_qr_fit(d,N,T)
        kalman=make_gaussian_kernel(d,d,6)
        # Bind the dimension through this factory; signatures have one fixed shape.
        def reference_factory(d,oracle,kalman):
            @tf.function(input_signature=[tf.TensorSpec([6],dtype),tf.TensorSpec([T,d],dtype),
                                          tf.TensorSpec([N,d],dtype)],jit_compile=True)
            def reference(theta,y,initial):
                model=parameterized_model(theta,d,d);A,H,m0,P0,Q,R=model[::2]
                center,cov,constants,_,_=oracle(theta,y)
                x0=m0+tf.matmul(initial,tf.linalg.cholesky(P0),transpose_b=True)
                integral=gaussian_log(center[0]-tf.matmul(x0,A,transpose_b=True),Q+cov[0])
                predicted=tf.reduce_logsumexp(integral)-tf.math.log(tf.constant(N,dtype))+tf.reduce_sum(constants)
                return kalman(y,*model)[0],predicted,A,H,m0,P0,Q,R
            return reference
        reference=reference_factory(d,oracle,kalman)
        for seed in range(92101,92105):
            case=f'd{d}-s{seed}';data_seed=stream(d,seed,'data',0,'observations')
            all_draws.append(tuple(data_seed));y=data(theta,tf.constant(data_seed,tf.int32))
            def draws(group,rep):
                specs=[('initial',[N,d],True),('process',[T,N,d],True),
                       ('ancestors',[T+1,N],False),('mixture',[T,N],False)]
                keys={name:stream(d,seed,group,rep,name) for name,_,_ in specs}
                values=[(tf.random.stateless_normal if normal else tf.random.stateless_uniform)(shape,keys[name],dtype=dtype)
                        for name,shape,normal in specs]
                all_draws.extend(tuple(s) for s in keys.values())
                return values,keys
            pilot,keys=draws('pilot',0)
            placeholder=(tf.zeros([T,d],dtype),tf.eye(d,batch_shape=[T],dtype=dtype),tf.zeros([T],dtype))
            pilot_out=bootstrap(theta,y,*pilot,*placeholder);clouds=pilot_out[2]
            c,v,constants,current_c,current_v=oracle(theta,y)
            def gaussian_floor(cov):
                return -.5*tf.constant(d*math.log(2*math.pi),dtype)-.5*tf.linalg.logdet(cov)-1000.
            coefficients={'bootstrap':placeholder,'current_observation':(current_c,tf.repeat(current_v[None],T,axis=0),
                tf.fill([T],gaussian_floor(current_v))), 'full_oracle':(c,v,gaussian_floor(v))}
            fit_records={};fit_times={};accepted={m:True for m in HEURISTICS}
            for method,kernel in [('density_QR_peak',density),('diagonal_QR',qr),('score_projection',score_fit)]:
                started=time.monotonic();out=kernel(theta,y,clouds);out[0].numpy();fit_times[method]=time.monotonic()-started
                coefficients[method]=out[:3]
                if method=='score_projection':
                    accepted[method]=bool(out[3]);fit_records[method]=materialize(dict(valid=out[3],diagnostics=out[4],
                        columns=['valid','cloud_margin','precision_margin','guard_tolerance','whitening_error','skew_norm','score_residual']))
                else:
                    accepted[method]=bool(out[3]) and (bool(out[4]) if method=='density_QR_peak' else True)
                    fit_records[method]=materialize(dict(valid=out[3],optimizer_converged=out[4],diagnostics=out[5],
                        columns=FIT_DIAGNOSTIC_COLUMNS if method=='density_QR_peak' else
                        ['initialization_valid','initialization_rank_margin','initialization_clipped','initial_shape_residual']))
            records=[];initial_noises=[];predictions=[];seed_records={'data':data_seed,'pilot':keys,'final':[]}
            exact_value=None;model=None
            for replicate in range(8):
                final,keys=draws('final',replicate);seed_records['final'].append(keys);initial_noises.append(materialize(final[0]))
                exact,predicted,*model=reference(theta,y,final[0]);exact_value=float(exact.numpy());predictions.append(float(predicted.numpy()))
                for method in METHODS:
                    if not accepted[method]:
                        records.append(dict(method=method,replicate=replicate,status='fit_rejected',value=None,squared_error=None));continue
                    kernel=bootstrap if method=='bootstrap' else regular
                    out=kernel(theta,y,*final,*coefficients[method]);value=float(out[0].numpy())
                    finite=math.isfinite(value) and bool(tf.reduce_all(tf.math.is_finite(out[1])))
                    trace=out[-1];cdf=trace['ancestor_cdf']
                    weights=cdf-tf.concat([tf.zeros([T+1,1],dtype),cdf[:,:-1]],axis=1)
                    ess=1/tf.reduce_sum(weights**2,axis=1)
                    trace_finite=bool(tf.reduce_all(tf.math.is_finite(ess))) and bool(tf.reduce_all(tf.math.is_finite(trace['gaussian_probability'])))
                    status='complete' if finite and trace_finite else 'consumer_invalid'
                    record=dict(method=method,replicate=replicate,status=status,value=value if finite else None,
                        squared_error=(value-exact_value)**2 if finite else None,
                        minimum_ess=float(tf.reduce_min(ess).numpy()) if trace_finite else None)
                    records.append(record)
                    if method=='full_oracle':
                        checks[case+f'/oracle_initial_{replicate}']=finite and abs(value-float(predicted.numpy()))<=1e-8
                        checks[case+f'/oracle_ess_{replicate}']=bool(tf.reduce_max(tf.abs(ess[1:]/N-1))<=1e-8)
            traces={name:kernel.experimental_get_tracing_count() for name,kernel in
                [('filter',regular),('bootstrap',bootstrap),('data',data),('oracle',oracle),('score_fit',score_fit),
                 ('density_fit',density),('qr_fit',qr),('reference',reference)]}
            checks[case+'/single_traces']=all(count==1 for count in traces.values())
            result=dict(case=case,d=d,seed=seed,N=N,T=T,theta=config['fit_theta'],observations=materialize(y),
                exact_value=exact_value,accepted=accepted,fits=fit_records,fit_seconds=fit_times,
                coefficients={k:materialize(v) for k,v in coefficients.items()},records=records,traces=traces,seeds=seed_records)
            save(directory/(case+'.json'),result);cases.append(result)
            r_inputs.append(dict(case=case,d=d,y=materialize(y),model=materialize(model),clouds=materialize(clouds),
                score_coefficients=materialize(coefficients['score_projection']),score_valid=accepted['score_projection'],
                exact_coefficients=materialize(coefficients['full_oracle'][:2]),one_step=materialize(coefficients['current_observation'][:2]),
                exact_value=exact_value,initial_noises=initial_noises,initial_predictions=predictions,records=records))
            save(directory/'partial.json',[dict(case=c['case'],accepted=c['accepted']) for c in cases])
            print(case+' '+str(accepted),flush=True)
    checks['disjoint_random_streams']=len(set(all_draws))==len(all_draws)
    (directory/'reference-input.R').write_text('input <- '+r_literal(r_inputs)+'\n')
    save(directory/'cases.json',cases)
    return dict(cases=len(cases),checks=checks,passed=all(checks.values()))


def results(directory):
    prior=ROOT/'attempt04_gpu_downstream'
    if not read(prior/'summary.json')['passed']: raise RuntimeError('Downstream engineering checks failed')
    copied=directory/'reference-input.R';shutil.copy2(prior/'reference-input.R',copied)
    r_run(directory,'docs/benchmarks/check_iapf_score_regression_downstream.R',copied)
    cases=read(prior/'cases.json');table=[];per_case=[]
    for case in cases:
        for method in METHODS:
            rows=[r for r in case['records'] if r['method']==method]
            complete=all(r['status']=='complete' for r in rows) and len(rows)==8
            per_case.append(dict(case=case['case'],d=case['d'],method=method,complete=complete,
                mse=sum(r['squared_error'] for r in rows)/8 if complete else None))
    verdict={}
    for d in [2,5,10]:
        means={}
        for method in METHODS:
            rows=[r for r in per_case if r['d']==d and r['method']==method];complete=all(r['complete'] for r in rows)
            mse=sum(r['mse'] for r in rows)/4 if complete else None
            table.append(dict(d=d,method=method,complete_cases=sum(r['complete'] for r in rows),mse=mse))
            means[method]=mse
        for method in METHODS[3:]:
            losses=[h for h in HEURISTICS if means[method] is not None and means[h] is not None and means[method]>means[h]]
            verdict[f'd{d}/{method}']=dict(candidate_complete=means[method] is not None,losses_to=losses,
                promotion_veto=means[method] is None or bool(losses))
    save(directory/'per-case-mse.json',per_case);save(directory/'conditional-summary.json',table)
    save(directory/'heuristic-dominance.json',verdict)
    paired=[]
    for row in per_case:
        if row['method']!='score_projection': continue
        peers={r['method']:r for r in per_case if r['case']==row['case']}
        for method in METHODS[:-1]:
            peer=peers[method]
            paired.append(dict(case=row['case'],d=row['d'],comparator=method,
                difference=row['mse']-peer['mse'] if row['complete'] and peer['complete'] else None))
    save(directory/'paired-differences.json',paired)
    return dict(passed=True,input_sha256=sha(copied),conditional=table,heuristic_dominance=verdict,
                inference='descriptive_only_four_data_sets_per_dimension_no_ranking',default_change=False)
