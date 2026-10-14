"""Exact-scope GPU validation of accepted-state bookkeeping and recursion."""
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
import argparse,importlib.util,json,sys,time
from pathlib import Path
sys.path.insert(0,'/home/chakwong/BayesFilter')
import tensorflow as tf
from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
memory=configure_tensorflow_gpu_memory_growth(tf,require_gpu=True)
tf.config.experimental.enable_tensor_float_32_execution(False)
from bayesfilter.replication.lorenz_refinement_tf import LorenzRefinement
D=tf.float64


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();start=time.monotonic();campaign=args.root/'docs/plans/artifacts/controlled-smc-kdm-replication-20260928'
    summary=dict(status='FAIL',scope='EXACT_SCOPE_ACCEPTANCE_BOOKKEEPING_REPAIR',gpu_memory_policy=memory,
        device='GPU/XLA plus CPU graph diagnostic',dtype='float64',tf32=False,jit_compile=True,
        checks=[],recursion=[],replication='INCOMPLETE',fail_fast_default=False)
    def persist():
        summary['wall_seconds']=time.monotonic()-start
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    def comparison(a,b,exact=False):
        if a.shape!=b.shape:raise ValueError('Comparison shape mismatch: '+str((a.shape,b.shape)))
        a,b=tf.cast(a,D),tf.cast(b,D);diff=tf.abs(a-b)
        return dict(passed=bool(tf.reduce_all(tf.math.is_finite(a)&(diff<=(tf.zeros_like(b) if exact else 1e-7+1e-9*tf.abs(b)))).numpy()),max_absolute=float(tf.reduce_max(diff).numpy()))
    def check(name,good,**extra):
        summary['checks'].append(dict(name=name,passed=bool(good),**extra));persist()
        if not good:raise ValueError(name)
    def mod(name,file):
        spec=importlib.util.spec_from_file_location(name,file);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
    def load(folder,k):return tf.io.parse_tensor(tf.io.read_file(str(folder/(k+'.tensor'))),D)
    try:
        kernel=mod('original_kernel',campaign/'pre-r1-multivariate-20261002.py')
        original=mod('original_controller',campaign/'pre-r1-controller-20261002.py');original.MultivariateGaussianControlledSMC=kernel.MultivariateGaussianControlledSMC
        reference=campaign/'attempt69_c1_lorenz_refinement_reference';fixture=json.loads((reference/'fixture.json').read_text())
        def read(entry):return tf.reshape(tf.io.decode_raw((reference/entry['file']).read_bytes(),D),entry['shape'])
        y=tf.constant(fixture['metadata']['y'],D)[:,None,:]
        normal=read(fixture['normal'])[:,:,None,:,:];uniform=read(fixture['uniform'])[:,:,None]
        baseline=original.LorenzRefinement(forcing=2.5);controller=LorenzRefinement(forcing=2.5)
        before=baseline.run(y,normal,uniform,tf.constant(4));after=controller.run(y,normal,uniform,tf.constant(4))
        check('full_healthy_valid',before['valid'].numpy() and after['valid'].numpy())
        check('four_successful_refits',int(after['iterations'].numpy())==4 and int(after['attempted_iterations'].numpy())==4)
        for k in ('A','b','c','x','ancestors','log_incremental_weight','logZ','ess'):
            result=comparison(after['history'][k],before['history'][k],exact=k=='ancestors');check('healthy_'+k,result.pop('passed'),**result)
        # Inject one rejected forward result through the actual controller chain.
        broken=tf.tensor_scatter_nd_update(normal,[[1,0,0,0,0]],[float('nan')])
        failed=controller.run(y,broken,uniform,tf.constant(1))
        check('rejected_refit_not_success',not failed['valid'].numpy() and int(failed['iterations'].numpy())==0 and
            int(failed['attempted_iterations'].numpy())==1 and failed['rejected_refit'].numpy())
        for k in ('A','b','c','x','ancestors','logZ'):
            result=comparison(failed['history'][k][1],failed['history'][k][0],exact=True);check('rejection_preserves_'+k,result.pop('passed'),**result)
        check('one_controller_trace',controller.run.experimental_get_tracing_count()==1)
        candidate=campaign/'attempt108_c2_lorenz_bounded_integral_safety/covariance_floor_0.001'
        captured=campaign/'attempt111_c2_lorenz_bounded_crossover_capture'
        values=[load(candidate,k)[3] for k in ('A','b','c','x','log_incremental_weight')]
        check('four_replica_scope',int(values[0].shape[1])==4)
        consumer=LorenzRefinement(forcing=2.5,replicas=4,precision_policy='nearest_covariance',precision_floor=.001,
            precision_eigen_tolerance=8*2.220446049250313e-16,precision_eigen_refinement=True,precision_stable_integral=True)
        engine=consumer.engine
        with tf.device('/GPU:0'):gpu=engine.fit(*values)
        cpu_function=tf.function(engine._fit,input_signature=engine.fit.input_signature,jit_compile=False,autograph=False)
        with tf.device('/CPU:0'):cpu=cpu_function(*values)
        check('four_replica_fits_valid',gpu['valid'].numpy() and cpu['valid'].numpy())
        for lane,result in (('gpu',gpu),('cpu_graph',cpu)):
            folder=args.output/lane;folder.mkdir()
            for k,v in result.items():tf.io.write_file(str(folder/(k+'.tensor')),tf.io.serialize_tensor(v))
            summary['recursion'].append(dict(lane=lane,comparisons={k:comparison(result[k],load(captured,k)) for k in
                ('A','b','c','correction_log_integral','projection_added_A')}))
        summary['default_bookkeeping_repair_passed']=True
        summary['full_recursive_admission']=False
        summary['full_recursive_admission_reason']='Independent reference114/130 still fails unchanged tolerance; capture is localization only.'
        summary['status']='PASS';persist()
    except Exception as exc:
        summary['error']=repr(exc);persist();import traceback;traceback.print_exc()
    print(json.dumps({k:summary.get(k) for k in ('status','default_bookkeeping_repair_passed','recursion','error','wall_seconds')}))
    if summary['status']!='PASS':raise SystemExit(1)


if __name__=='__main__':main()
