"""Focused GPU/XLA fail-closed consumer regression; no empirical promotion."""
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
import argparse,importlib.util,json,math,sys,time
from pathlib import Path
sys.path.insert(0,'/home/chakwong/BayesFilter')
import tensorflow as tf
from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
memory=configure_tensorflow_gpu_memory_growth(tf,require_gpu=True)
tf.config.experimental.enable_tensor_float_32_execution(False)
from bayesfilter.replication.lorenz_refinement_tf import LorenzRefinement
D=tf.float64


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();start=time.monotonic()
    campaign=args.root/'docs/plans/artifacts/controlled-smc-kdm-replication-20260928'
    summary=dict(status='FAIL',scope='FIRST_FAILURE_CONSUMER_REGRESSION',gpu_memory_policy=memory,
        device='GPU',dtype='float64',tf32=False,jit_compile=True,tensorflow=tf.__version__,checks=[],
        replication='INCOMPLETE',heuristic_dominance_verdict='NOT_ASSESSED_CONTROL_FLOW_ONLY')
    def clean(v):
        if isinstance(v,float) and not math.isfinite(v):return None
        if isinstance(v,list):return [clean(x) for x in v]
        if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
        return v
    def persist():
        summary['wall_seconds']=time.monotonic()-start
        (args.output/'summary.json').write_text(json.dumps(clean(summary),indent=2,allow_nan=False)+'\n')
    def check(name,good,**extra):
        summary['checks'].append(dict(name=name,passed=bool(good),**extra));persist()
        if not good:raise ValueError(name)
    def parity(name,new,old,exact=False):
        actual,expected=tf.cast(new,D),tf.cast(old,D)
        error=tf.abs(actual-expected)
        limit=tf.zeros_like(expected) if exact else 1e-7+1e-9*tf.abs(expected)
        check(name,tf.reduce_all(tf.math.is_finite(actual)&(error<=limit)).numpy(),max_absolute=float(tf.reduce_max(error).numpy()))
    def load(folder,key,dtype=D):
        return tf.io.parse_tensor(tf.io.read_file(str(folder/(key+'.tensor'))),dtype)
    try:
        summary['devices']=[x.name for x in tf.config.list_logical_devices('GPU')]
        check('trusted_gpu_visible',bool(summary['devices']))
        h,n=4,128
        controller=LorenzRefinement(forcing=2.5,first_failure_guards=True,observations=h,particles=n)
        engine=controller.engine
        y=tf.zeros([h,1,6],D)
        normal=tf.random.stateless_normal([5,h,1,n,8],[20261002,1],dtype=D)
        uniform=tf.fill([5,h,1],tf.constant(.3,D))
        policy=controller.initial_policy(y)
        spec=importlib.util.spec_from_file_location('pre_r1_kernel',campaign/'pre-r1-multivariate-20261002.py')
        old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        baseline=old.MultivariateGaussianControlledSMC(observations=h,particles=n,replicas=1,dimension=8,
            observation_dimension=6,initial_mean=[0.]*8,initial_covariance=tf.eye(8,dtype=D)*.01,
            transition_covariance=tf.eye(8,dtype=D)*.001,transition_mean=engine.f,
            observation_log_density=engine.logg,solver='centered_qr')
        before=baseline.forward(*policy,y,normal[0],uniform[0]);after=engine.forward(*policy,y,normal[0],uniform[0])
        check('healthy_forward_valid',before['valid'].numpy() and after['valid'].numpy())
        for k in ('x','ancestors','logZ','log_incremental_weight','ess','precision_margin'):
            parity('healthy_'+k,after[k],before[k],exact=k=='ancestors')
        check('healthy_no_failure',int(after['first_failure_time'].numpy())==-1 and int(after['completed_steps'].numpy())==h)
        bad_a=tf.tensor_scatter_nd_update(policy[0],[[0,0,0,0]],[-100.])
        bad=engine.forward(bad_a,*policy[1:],y,normal[0],uniform[0])
        check('invalid_gaussian_stops_before_resampling',not bad['valid'].numpy() and
            int(bad['first_failure_time'].numpy())==0 and int(bad['failure_codes'][0].numpy())==2 and
            bool(tf.reduce_all(bad['ancestors']==-1).numpy()) and int(bad['completed_steps'].numpy())==0)
        badnormal=tf.tensor_scatter_nd_update(normal[0],[[0,0,0,0]],[float('nan')])
        bad=engine.forward(*policy,y,badnormal,uniform[0])
        check('invalid_particle_stops_before_lookahead',not bad['valid'].numpy() and int(bad['failure_codes'][0].numpy())==3 and
              bool(tf.reduce_all(bad['ancestors']==-1).numpy()))
        oldfit=baseline.fit(*policy,before['x'],before['log_incremental_weight'])
        fit=engine.fit(*policy,after['x'],after['log_incremental_weight'])
        check('healthy_fit_valid',oldfit['valid'].numpy() and fit['valid'].numpy())
        for k in ('A','b','c','correction_log_integral'):
            parity('healthy_fit_'+k,fit[k],oldfit[k])
        badweights=tf.tensor_scatter_nd_update(after['log_incremental_weight'],[[h-2,0,0]],[float('nan')])
        badfit=engine.fit(*policy,after['x'],badweights)
        check('backward_stops_at_first_invalid_target',not badfit['valid'].numpy() and
            int(badfit['first_failure_time'].numpy())==h-2 and int(badfit['failure_code'].numpy())==9 and
            badfit['attempted_times'].numpy().tolist()==[False,False,True,True])
        for k in ('A','b','c'):parity('unexecuted_prefix_keeps_old_'+k,badfit[k][:h-2],policy[('A','b','c').index(k)][:h-2],exact=True)
        successful=controller.run(y,normal,uniform,tf.constant(1))
        check('actual_controller_success',successful['valid'].numpy() and int(successful['iterations'].numpy())==1)
        failure_normal=tf.tensor_scatter_nd_update(normal,[[1,0,0,0,0]],[float('nan')])
        failed=controller.run(y,failure_normal,uniform,tf.constant(1))
        check('rejected_refit_not_counted',not failed['valid'].numpy() and int(failed['iterations'].numpy())==0 and
              failed['attempted_refits'][0,0].numpy() and not bool(tf.reduce_any(failed['accepted_refits']).numpy()))
        check('consumer_failure_location',int(failed['first_failure']['refit'].numpy())==1 and
              int(failed['first_failure']['time'].numpy())==0 and int(failed['first_failure']['codes'][0].numpy())==3)
        for k in ('A','b','c','x','ancestors','logZ'):
            parity('last_valid_'+k,failed['history'][k][1],failed['history'][k][0],exact=True)
        check('stable_trace_counts',engine.forward.experimental_get_tracing_count()==1 and
            engine.fit.experimental_get_tracing_count()==1 and controller.run.experimental_get_tracing_count()==1)
        # Replay the actual failed proposal, not merely a synthetic large mean.
        saved=campaign/'attempt121_c2_lorenz_bounded_trigger_safety/covariance_floor_0.01'
        design=campaign/'attempt76_c2_lorenz_batch_validation/rejected_batch'
        a,b,c=(load(saved,k)[4,:,0:1] for k in ('A','b','c'))
        savednormal=load(design,'normal');saveduniform=load(design,'uniform');savedy=load(design,'y')
        large=LorenzRefinement(forcing=2.5,first_failure_guards=True,particles=int(savednormal.shape[3]),observations=int(savedy.shape[0]))
        replay=large.engine.forward(a,b,c,savedy[:,0:1],savednormal[4,:,0:1],saveduniform[4,:,0:1])
        check('saved_failure_localized_before_resampling',not replay['valid'].numpy() and
            int(replay['first_failure_time'].numpy())==0 and int(replay['failure_codes'][0].numpy())==4 and
            bool(tf.reduce_all(replay['ancestors']==-1).numpy()))
        summary['saved_failure']=dict(time=int(replay['first_failure_time'].numpy()),
            codes=replay['failure_codes'].numpy().tolist(),mean_maxabs=float(tf.reduce_max(tf.abs(replay['failure_mean'])).numpy()),
            completed_steps=int(replay['completed_steps'].numpy()))
        summary['status']='PASS';persist()
    except Exception as exc:
        summary['error']=repr(exc);persist()
        import traceback;traceback.print_exc()
    print(json.dumps(dict(status=summary['status'],checks=len(summary['checks']),error=summary.get('error'),saved_failure=summary.get('saved_failure'))))
    if summary['status']!='PASS':raise SystemExit(1)


if __name__=='__main__':main()
