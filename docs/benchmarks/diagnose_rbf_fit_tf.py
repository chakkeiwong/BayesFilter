"""Trusted GPU/XLA numerical validation of full-horizon RBF NNLS fitting."""
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,'/home/chakwong/BayesFilter')
import tensorflow as tf
from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
memory=configure_tensorflow_gpu_memory_growth(tf,require_gpu=True)
tf.config.experimental.enable_tensor_float_32_execution(False)
from bayesfilter.replication.rbf_controlled_smc_tf import RBFControlledSMC,rbf_log_policy,rbf_gaussian_mixture,multimodal_transition
spec=importlib.util.spec_from_file_location('nnls_snapshot',Path(__file__).with_name('rbf_nnls_tf.py'))
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
D=tf.float64


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fixture',type=Path,required=True);parser.add_argument('--accuracy-calibration',action='store_true');parser.add_argument('--stopping-factor',type=float,default=10.);parser.add_argument('--policy-calibration',type=Path);parser.add_argument('--fresh-seed',type=int,default=20261008);args=parser.parse_args()
    started=time.monotonic();fixture=json.loads(args.fixture.read_text());checks=[];extra={'stopping_factor':args.stopping_factor}
    def check(name,actual,expected,atol=1e-6,rtol=1e-8):
        actual=tf.cast(tf.convert_to_tensor(actual),D);expected=tf.cast(tf.convert_to_tensor(expected),D)
        error=tf.abs(actual-expected);okay=tf.math.is_finite(actual)&(error<=atol+rtol*tf.abs(expected))
        checks.append(dict(name=name,passed=bool(tf.reduce_all(okay).numpy()),
            max_absolute_error=float(tf.reduce_max(error).numpy()),atol=atol,rtol=rtol,
            failures=int(tf.reduce_sum(tf.cast(~okay,tf.int32)).numpy())))
    def flag(name,value):check(name,tf.cast(value,D),tf.ones_like(tf.cast(value,D)),0.,0.)
    def save(name,value):tf.io.write_file(str(args.output/(name+'.tensor')),tf.io.serialize_tensor(value))
    try:
        if fixture.get('stopping_factor') is not None and fixture['stopping_factor']!=args.stopping_factor:raise ValueError('Reference and candidate stopping factors differ')
        with tf.device('/GPU:0'):
            # Closed-form controls include a passive-variable deletion and duplicate columns.
            design=tf.constant([[[1,0,0],[0,1,0],[0,0,1]],[[10,1,0],[0,1,0],[0,0,1]],
                [[1,1,0],[0,0,1],[0,0,0]],[[1,0,0],[0,1,0],[0,0,1]]],D)
            target=tf.constant([[2,-1,0],[1,2,0],[2,3,0],[0,0,0]],D)
            solver=core.NonnegativeLeastSquares(rows=3,columns=3,replicas=4,capacity=3)
            controls=solver.solve(design,target)
            flag('nnls_controls_valid',controls['valid'])
            check('nnls_controls_solution',controls['coefficients'],tf.constant([[2,0,0],[0,1.5,0],[2,0,3],[0,0,0]],D),1e-12,1e-12)
            flag('passive_removal_exercised',controls['removed'][1]>0)
            flag('nnls_no_worse_than_zero',controls['rss']<=tf.reduce_sum(target**2,-1))
            limited=core.NonnegativeLeastSquares(rows=3,columns=3,replicas=4,capacity=1)
            rejected=limited.solve(tf.tile(tf.eye(3,dtype=D)[None],[4,1,1]),tf.ones([4,3],D))
            flag('capacity_exhaustion_rejected',~rejected['valid'])
            flag('primitive_single_trace',solver.solve.experimental_get_tracing_count()==1)
            if not all(c['passed'] for c in checks):raise RuntimeError('NNLS primitive validation failed')
            if fixture['scope']=='FROZEN_RBF_NNLS_FAILURE':
                if fixture['reference_coefficients'] is None:raise RuntimeError('No eligible independent NNLS reference')
                n=int(fixture['particles']);x=tf.constant(fixture['x'],D)
                G=tf.exp(-tf.constant(fixture['beta'],D)*(x[:,None]-x[None,:])**2)[None]
                target=tf.constant(fixture['target'],D)[None];reference=tf.constant(fixture['reference_coefficients'],D)[None]
                factors=[1.,0.1,0.01] if args.accuracy_calibration else [10.]
                if args.policy_calibration:
                    second=json.loads(args.policy_calibration.read_text());factors=[1.,.1,.01,.001]*2
                    x=tf.repeat(tf.constant([fixture['x'],second['x']],D),4,axis=0)
                    beta=tf.repeat(tf.constant([fixture['beta'],second['beta']],D),4,axis=0)
                    G=tf.exp(-beta[:,None,None]*(x[:,:,None]-x[:,None,:])**2)
                    target=tf.repeat(tf.constant([fixture['target'],second['target']],D),4,axis=0)
                    reference=tf.repeat(tf.constant([fixture['reference_coefficients'],second['reference_coefficients']],D),4,axis=0)
                else:
                    G=tf.repeat(G,len(factors),axis=0);target=tf.repeat(target,len(factors),axis=0);reference=tf.repeat(reference,len(factors),axis=0)
                diagnostic=core.NonnegativeLeastSquares(rows=n,columns=n,replicas=len(factors),capacity=128,stopping_factor=factors)
                begin=time.monotonic();solved=diagnostic.solve(G,target);solved['valid'].numpy()
                if args.accuracy_calibration or args.policy_calibration:
                    arm_checks=[];passing=[]
                    for j,factor in enumerate(factors):
                        mark=len(checks)
                        flag('valid',solved['valid'][j])
                        check('prediction',tf.linalg.matvec(G[j],solved['coefficients'][j]),tf.linalg.matvec(G[j],reference[j]),1e-7,1e-6)
                        check('objective',solved['rss'][j],tf.reduce_sum((tf.linalg.matvec(G[j],reference[j])-target[j])**2),1e-8,1e-6)
                        if args.policy_calibration:
                            def policy(coef):
                                top=tf.math.top_k(coef,k=16,sorted=True)
                                return (beta[j:j+1],top.values[None],tf.gather(x[j],top.indices)[None],tf.zeros([1],D))
                            own_policy=policy(solved['coefficients'][j]);ref_policy=policy(reference[j])
                            check('retained_log_policy',rbf_log_policy(x[j:j+1],*own_policy),rbf_log_policy(x[j:j+1],*ref_policy),1e-7,1e-6)
                            check('retained_gaussian_integral',rbf_gaussian_mixture(x[j:j+1],tf.constant(10.,D),*own_policy)['log_integral'],
                                  rbf_gaussian_mixture(x[j:j+1],tf.constant(10.,D),*ref_policy)['log_integral'],1e-7,1e-6)
                        entries=checks[mark:];del checks[mark:]
                        okay=all(c['passed'] for c in entries)
                        arm_checks.append(dict(stopping_factor=factor,case=j//4 if args.policy_calibration else 0,passed=okay,checks=entries))
                        if okay:passing.append(factor)
                    if args.policy_calibration:
                        passing=[factor for factor in (1.,.1,.01,.001) if all(a['passed'] for a in arm_checks if a['stopping_factor']==factor)]
                    extra['accuracy_calibration']=arm_checks
                    extra['selected_stopping_factor']=max(passing) if passing else None
                    flag('eligible_accuracy_candidate',bool(passing))
                else:
                    flag('frozen_failure_nnls_valid',solved['valid'])
                    check('frozen_failure_prediction',tf.linalg.matvec(G,solved['coefficients']),tf.linalg.matvec(G,reference),1e-7,1e-6)
                    check('frozen_failure_objective',solved['rss'],tf.reduce_sum((tf.linalg.matvec(G,reference)-target)**2,-1),1e-8,1e-6)
                flag('frozen_failure_single_trace',diagnostic.solve.experimental_get_tracing_count()==1)
                extra['frozen_nnls']={k:v.numpy().tolist() for k,v in solved.items() if k!='coefficients'}
                extra['first_call_seconds']=time.monotonic()-begin
                save('frozen_coefficients',solved['coefficients'])
            else:
                h,n,m=int(fixture['observations']),int(fixture['particles']),int(fixture['components']);b=len(fixture['cases'])
                def field(name,round_index=1):return tf.stack([tf.constant(c['rounds'][round_index][name],D) for c in fixture['cases']],axis=1)
                cloud=field('x',0);source_beta=field('beta');source_w=field('weights');source_k=field('knots');source_c=field('constant')
                source_target=tf.stack([tf.constant(c['fit_target'],D) for c in fixture['cases']],1)
                source_coef=tf.stack([tf.constant(c['full_coefficients'],D) for c in fixture['cases']],1)
                y=tf.tile(tf.constant(fixture['y'],D)[:,None],[1,b])
                engine=RBFControlledSMC(observations=h,particles=n,replicas=b,components=m,observation_variance=fixture['observation_variance'])
                replay=engine.forward(source_beta,source_w,source_k,source_c,y,field('normal'),field('component_uniform'),field('uniform'))
                flag('full_source_policy_forward_valid',replay['valid'])
                for name in ('x','log_incremental_weight','logZ','ess'):check('full_source_policy_'+name,replay[name],field(name))
                check('full_source_policy_ancestry',replay['ancestors'],field('ancestors'),0.,0.)
                if not all(c['passed'] for c in checks):raise RuntimeError('Full reference forward replay failed')
                # Independent frozen targets separate the solver from backward-recursion errors.
                indices=tf.constant([0,h//2,h-1]);frozen_x=tf.reshape(tf.gather(cloud,indices),[3*b,n])
                frozen_beta=tf.reshape(tf.gather(source_beta,indices),[3*b]);frozen_target=tf.reshape(tf.gather(source_target,indices),[3*b,n])
                frozen_coef=tf.reshape(tf.gather(source_coef,indices),[3*b,n])
                G=tf.exp(-frozen_beta[:,None,None]*(frozen_x[:,:,None]-frozen_x[:,None,:])**2)
                diagnostic=core.NonnegativeLeastSquares(rows=n,columns=n,replicas=3*b,capacity=128,stopping_factor=args.stopping_factor)
                frozen=diagnostic.solve(G,frozen_target);flag('frozen_nnls_valid',frozen['valid'])
                predicted=tf.linalg.matvec(G,frozen['coefficients']);reference=tf.linalg.matvec(G,frozen_coef)
                check('frozen_nnls_prediction',predicted,reference,1e-7,1e-6)
                check('frozen_nnls_objective',frozen['rss'],tf.reduce_sum((reference-frozen_target)**2,-1),1e-8,1e-6)
                extra['frozen_nnls']={k:v.numpy().tolist() for k,v in frozen.items() if k!='coefficients'}
                save('frozen_coefficients',frozen['coefficients'])
                if not all(c['passed'] for c in checks):raise RuntimeError('Frozen NNLS validation failed')
                fitter=core.RBFBackwardFit(observations=h,particles=n,replicas=b,components=m,
                    observation_variance=fixture['observation_variance'],bandwidth_factor=fixture['bandwidth_factor'],stopping_factor=args.stopping_factor)
                fit_started=time.monotonic();fitted=fitter.fit(cloud,y);fitted['valid'].numpy();extra['fit_first_call_seconds']=time.monotonic()-fit_started
                for name,value in fitted.items():save('fit_'+name,value)
                flag('backward_fit_valid',fitted['valid'])
                extra['fit_diagnostics']={k:fitted[k].numpy().tolist() for k in ('iterations','peak_active','removed','dual_error','complementarity','minimum_qr_margin')}
                if not bool(fitted['valid'].numpy()):raise RuntimeError('Full backward fit invalid; preserve time/support diagnostics')
                # Post-run diagnostic tensors evaluate every time and both replicas.
                flat_x=tf.reshape(cloud,[h*b,n]);flat_beta=tf.reshape(source_beta,[h*b])
                matrices=tf.exp(-flat_beta[:,None,None]*(flat_x[:,:,None]-flat_x[:,None,:])**2)
                pred=tf.linalg.matvec(matrices,tf.reshape(fitted['coefficients'],[h*b,n]))
                ref=tf.linalg.matvec(matrices,tf.reshape(source_coef,[h*b,n]));ref_target=tf.reshape(source_target,[h*b,n])
                check('all_times_nnls_prediction',pred,ref,1e-7,1e-6)
                check('all_times_nnls_objective',tf.reduce_sum((pred-ref_target)**2,-1),tf.reduce_sum((ref-ref_target)**2,-1),1e-8,1e-6)
                check('backward_targets',fitted['target'],source_target,1e-7,1e-6)
                def flat(policy):return (tf.reshape(policy['beta'],[h*b]),tf.reshape(policy['weights'],[h*b,m]),
                    tf.reshape(policy['knots'],[h*b,m]),tf.reshape(policy['constant'],[h*b]))
                source_policy=dict(beta=source_beta,weights=source_w,knots=source_k,constant=source_c)
                check('retained_log_policy',rbf_log_policy(flat_x,*flat(fitted)),rbf_log_policy(flat_x,*flat(source_policy)),1e-7,1e-6)
                fitted_integral=rbf_gaussian_mixture(flat_x,tf.constant(10.,D),*flat(fitted))
                source_integral=rbf_gaussian_mixture(flat_x,tf.constant(10.,D),*flat(source_policy))
                check('retained_gaussian_integral',fitted_integral['log_integral'],source_integral['log_integral'],1e-7,1e-6)
                output=engine.forward(*(fitted[k] for k in ('beta','weights','knots','constant')),y,field('normal'),field('component_uniform'),field('uniform'))
                flag('fitted_forward_valid',output['valid'])
                for name in ('x','log_incremental_weight','logZ','ess'):
                    check('fitted_forward_'+name,output[name],field(name));save('fitted_forward_'+name,output[name])
                check('fitted_forward_ancestry',output['ancestors'],field('ancestors'),0.,0.);save('fitted_forward_ancestry',output['ancestors'])
                fresh=engine.forward(*(fitted[k] for k in ('beta','weights','knots','constant')),y,
                    tf.random.stateless_normal([h,b,n],[args.fresh_seed,1],dtype=D),tf.random.stateless_uniform([h,b,n],[args.fresh_seed,2],dtype=D),
                    tf.random.stateless_uniform([h,b],[args.fresh_seed,3],dtype=D))
                flag('fresh_fitted_consumer_valid',fresh['valid'])
                flag('fitter_single_trace',fitter.fit.experimental_get_tracing_count()==1)
                flag('consumer_single_trace',engine.forward.experimental_get_tracing_count()==1)
                extra.update(final_logZ=output['logZ'][-1].numpy().tolist(),fresh_final_logZ=fresh['logZ'][-1].numpy().tolist(),
                    traces=dict(fit=fitter.fit.experimental_get_tracing_count(),forward=engine.forward.experimental_get_tracing_count(),nnls=diagnostic.solve.experimental_get_tracing_count()))
    except Exception as exc:
        extra['failure']=repr(exc)
    summary=dict(status='PASS' if checks and all(c['passed'] for c in checks) and 'failure' not in extra else 'FAIL',
        scope=('RBF_FROZEN_NNLS_VALIDATION' if fixture['scope']=='FROZEN_RBF_NNLS_FAILURE' else 'RBF_NNLS_FULL_HORIZON_FIT_CONSUMER'),replication='INCOMPLETE',checks=checks,**extra,
        seeds=fixture.get('seeds',[20261006,20261007]),fresh_seed=args.fresh_seed,gpu_memory_policy=memory,tensorflow=tf.__version__,device='/device:GPU:0',
        dtype='float64',jit_compile=True,tf32=False,numpy_runtime=False,pfor=False,python_time_loop=False,
        allocator=tf.config.experimental.get_memory_info('GPU:0'),wall_seconds=time.monotonic()-started,
        nonclaims=['No Table2 repetitions','Only first refinement and R=1','No ranking, general NNLS admission, HMC or default readiness'])
    (args.output/'summary.json').write_text(json.dumps(summary,allow_nan=False,indent=2)+'\n')
    print(json.dumps(dict(status=summary['status'],checks=len(checks),failure=extra.get('failure'),wall_seconds=summary['wall_seconds'])))
    if summary['status']!='PASS':raise SystemExit(1)


if __name__=='__main__':main()
