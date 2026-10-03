"""Fresh downstream diagnostics for optional iAPF fitting controls.

Scores differentiate a finite program with fixed fitted guides and locally fixed
discrete labels. This is not a marginal-model-score or method-ranking campaign.
"""
import hashlib
import time

from diagnose_iapf_adaptive_consumer import controls, save


def consumer(tf,directory):
    from bayesfilter.score_study.iapf_adapter import execute_iapf
    from bayesfilter.score_study.contracts import DiagnosticFailure
    from bayesfilter.score_study.gaussian_tf import make_data_kernel,make_gaussian_kernel,make_particle_kernel,parameterized_model
    from bayesfilter.score_study.fitted_twist_tf import make_fitted_twist_kernel
    dtype=tf.float64;results=[]
    for d in (2,5,10):
        for case_seed in (81,82):
            T=4;o=d;config=controls();config['max_particles']=1024
            theta=tf.constant(config['fit_theta'],dtype)
            observations={regime:make_data_kernel(d,o,T)(theta,tf.constant([948000+case_seed+(1000 if regime=='heldout' else 0),d],tf.int32))
                          for regime in ('fitted','heldout')}
            settings=dict(dimension=d,observation_dimension=o,particles=128,horizon=T,dtype='float64',jit_compile=True)
            def seed(name,replicate=case_seed,group='final'):
                raw=hashlib.sha256(f'{case_seed}:{d}:{o}:{group}:{replicate}:{name}'.encode()).digest()
                return [int.from_bytes(raw[:4],'little')%(2**31-1),int.from_bytes(raw[4:8],'little')%(2**31-1)]
            oracles={regime:make_gaussian_kernel(d,o,6)(y,*parameterized_model(theta,d,o)) for regime,y in observations.items()}
            heuristic_cache={}
            for initialization in ('cloud_moments','log_quadratic'):
                for scale in ('native','initial_peak'):
                    case_dir=directory/f'd{d}-s{case_seed}-{initialization}-{scale}';case_dir.mkdir()
                    chosen={**config,'fit_initialization':initialization,'fit_objective_scale':scale}
                    row=dict(model='linear_gaussian',method='iapf',role='diagnostic',iapf=chosen)
                    rec=dict(d=d,seed=case_seed,initialization=initialization,objective_scale=scale,config=chosen,
                        settings=settings,observations={k:v.numpy().tolist() for k,v in observations.items()},status='running')
                    start=time.monotonic()
                    try:
                        kernel,out,diag,calls=execute_iapf(row,settings,theta,observations['fitted'],seed)
                    except DiagnosticFailure as exc:
                        rec.update(status='candidate_rejected',error=str(exc),diagnostics=exc.diagnostics['details'],wall_seconds=time.monotonic()-start)
                        save(case_dir/'result.json',rec);results.append(rec)
                        save(directory/'results.json',results)
                        print(f'd{d}/s{case_seed}/{initialization}/{scale}: {rec["error"]}',flush=True)
                        continue
                    n=diag['actual_particle_count']
                    def draw(name,shape,normal):
                        return (tf.random.stateless_normal if normal else tf.random.stateless_uniform)(shape,seed('iapf_final_'+name),dtype=dtype)
                    draws=[draw('initial',[n,d],True),draw('process',[T,n,d],True),draw('ancestors',[T+1,n],False),draw('mixture',[T,n],False)]
                    coefficients=[tf.constant(diag['fit'][k],dtype) for k in ('centers','covariances','log_floors')]
                    rec.update(status='complete',diagnostics=diag,calls=calls,regimes={})
                    # Save actual final inputs, rather than assuming seeds reproduce across devices.
                    save(case_dir/'final-inputs.json',dict(theta=theta.numpy().tolist(),draws=[v.numpy().tolist() for v in draws],coefficients=[v.numpy().tolist() for v in coefficients]))
                    for regime,y in observations.items():
                        value=out if regime=='fitted' else kernel(theta,y,*draws,*coefficients)
                        key=(n,regime)
                        if key not in heuristic_cache:
                            heuristics={}
                            for label,adapted in [('bootstrap',False),('one_step_optimal',True)]:
                                b=make_particle_kernel(d,o,n,T,adapted=adapted,resampling=True)(theta,y,*draws[:2],draws[2][1:])
                                heuristics[label]=dict(value=float(b[0].numpy()),score=b[1].numpy().tolist(),minimum_ess=float(b[2].numpy()))
                            b=make_fitted_twist_kernel(d,o,n,T,constant_twist=True)(theta,y,*draws,*coefficients)
                            heuristics['constant_guide']=dict(value=float(b[0].numpy()),score=b[1].numpy().tolist())
                            heuristic_cache[key]=heuristics
                        exact=float(oracles[regime][0].numpy());actual=float(value[0].numpy())
                        entry=dict(value=actual,score=value[1].numpy().tolist(),exact_value=exact,
                            exact_score=oracles[regime][1].numpy().tolist(),log_value_error=actual-exact,
                            particles=n,heuristics=heuristic_cache[key])
                        entry['heuristic_absolute_errors']={k:abs(v['value']-exact) for k,v in entry['heuristics'].items()}
                        entry['heuristic_dominance_verdict']='descriptive_veto' if any(abs(actual-exact)>v for v in entry['heuristic_absolute_errors'].values()) else 'no_observed_loss_this_draw'
                        if initialization=='log_quadratic' and scale=='initial_peak' and case_seed==81:
                            trace=make_fitted_twist_kernel(d,o,n,T,include_numerical_trace=True)
                            nominal=trace(theta,y,*draws,*coefficients)
                            labels=nominal[-1]['ancestor_indices'];mixture=draws[3]<nominal[-1]['gaussian_probability']
                            attempted=[];selected=None
                            for step in (1e-5,1e-6,1e-7,1e-8):
                                derivatives=[];stable=True
                                for h in (step,step/2):
                                    components=[]
                                    for direction in tf.unstack(tf.eye(6,dtype=dtype)):
                                        plus=trace(theta+h*direction,y,*draws,*coefficients)
                                        minus=trace(theta-h*direction,y,*draws,*coefficients)
                                        for shifted in (plus,minus):
                                            stable=stable and bool(tf.reduce_all(shifted[-1]['ancestor_indices']==labels).numpy()) and bool(tf.reduce_all((draws[3]<shifted[-1]['gaussian_probability'])==mixture).numpy())
                                        components.append(float(((plus[0]-minus[0])/(2*h)).numpy()))
                                    derivatives.append(components)
                                attempted.append(dict(h=step,labels_stable=stable))
                                if stable:
                                    analytic=nominal[1].numpy().tolist()
                                    error=max(abs(a-b) for derivative in derivatives for a,b in zip(analytic,derivative))
                                    selected=dict(h=step,finite_differences=derivatives,analytical_score=analytic,max_error=error,pass_derivative=error<=2e-5)
                                    break
                            entry['derivative_check']=dict(attempts=attempted,selected=selected,
                                status='inconclusive_label_boundary' if selected is None else 'pass' if selected['pass_derivative'] else 'veto')
                        rec['regimes'][regime]=entry
                    rec['wall_seconds']=time.monotonic()-start
                    save(case_dir/'result.json',rec);results.append(rec);save(directory/'results.json',results)
                    print(f'd{d}/s{case_seed}/{initialization}/{scale}: complete N={n}',flush=True)
    checks=[x['derivative_check'] for r in results if r['status']=='complete' for x in r['regimes'].values() if 'derivative_check' in x]
    return dict(cases=len(results),complete=sum(r['status']=='complete' for r in results),candidate_rejected=sum(r['status']=='candidate_rejected' for r in results),
        derivative_checks=checks,statistical_ranking='not supported; conditional diagnostics only',default_readiness=False)
