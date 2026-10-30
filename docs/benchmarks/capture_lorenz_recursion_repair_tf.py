"""GPU/XLA full-scope no-fire regression and CPU-graph localization."""
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
import argparse,hashlib,importlib.util,json,sys,time
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
    args=p.parse_args();start=time.monotonic()
    campaign=args.root/'docs/plans/artifacts/controlled-smc-kdm-replication-20260928'
    summary=dict(status='FAIL',scope='FULL_SCOPE_GUARD_PARITY_AND_RECURSION_CAPTURE',
        gpu_memory_policy=memory,device='GPU/XLA plus explicitly placed CPU graph diagnostic',
        dtype='float64',tf32=False,jit_compile=True,checks=[],recursion_comparisons=[],replication='INCOMPLETE')
    def persist():
        summary['wall_seconds']=time.monotonic()-start
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    def compare(a,b,exact=False):
        a,b=tf.cast(a,D),tf.cast(b,D);diff=tf.abs(a-b)
        return dict(passed=bool(tf.reduce_all(tf.math.is_finite(a)&(diff<=(tf.zeros_like(b) if exact else 1e-7+1e-9*tf.abs(b)))).numpy()),
                    max_absolute=float(tf.reduce_max(diff).numpy()))
    def check(name,good,strict=True,**extra):
        summary['checks'].append(dict(name=name,passed=bool(good),**extra));persist()
        if not good and strict:raise ValueError(name)
    def module(name,path):
        spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
    def load(folder,key):return tf.io.parse_tensor(tf.io.read_file(str(folder/(key+'.tensor'))),D)
    try:
        oldkernel=module('pre_r1_kernel',campaign/'pre-r1-multivariate-20261002.py')
        oldcontroller=module('pre_r1_controller',campaign/'pre-r1-controller-20261002.py')
        oldcontroller.MultivariateGaussianControlledSMC=oldkernel.MultivariateGaussianControlledSMC
        reference=campaign/'attempt69_c1_lorenz_refinement_reference'
        fixture=json.loads((reference/'fixture.json').read_text())
        def read(entry):
            data=(reference/entry['file']).read_bytes()
            check('input_hash_'+entry['file'],hashlib.sha256(data).hexdigest()==entry['sha256'])
            return tf.reshape(tf.io.decode_raw(data,D),entry['shape'])
        y=tf.constant(fixture['metadata']['y'],D)[:,None,:]
        normal=read(fixture['normal'])[:,:,None,:,:];uniform=read(fixture['uniform'])[:,:,None]
        baseline=oldcontroller.LorenzRefinement(forcing=2.5)
        current=LorenzRefinement(forcing=2.5)
        before=baseline.run(y,normal,uniform,tf.constant(4))
        after=current.run(y,normal,uniform,tf.constant(4))
        check('full_healthy_controller_valid',before['valid'].numpy() and after['valid'].numpy())
        check('full_healthy_iteration_count',int(before['iterations'].numpy())==int(after['iterations'].numpy()))
        summary['healthy_iterations']=int(after['iterations'].numpy())
        for k in ('A','b','c','x','ancestors','log_incremental_weight','logZ','ess'):
            result=compare(after['history'][k],before['history'][k],exact=k=='ancestors')
            check('full_healthy_'+k,result.pop('passed'),strict=False,**result)
        summary['healthy_per_stage']=[dict(stage=i,comparisons={k:compare(after['history'][k][i],before['history'][k][i],exact=k=='ancestors')
            for k in ('A','b','c','x','ancestors','logZ')}) for i in range(5)]
        for label,result in (('healthy_before',before),('healthy_after',after)):
            folder=args.output/label;folder.mkdir()
            for k,v in result['history'].items():tf.io.write_file(str(folder/(k+'.tensor')),tf.io.serialize_tensor(v))
        stage=next((i for i in range(1,5) if any(not summary['healthy_per_stage'][i]['comparisons'][k]['passed'] for k in ('A','b','c'))),1)
        frozen=[before['history'][k][stage-1] for k in ('A','b','c','x','log_incremental_weight')]
        oldfit=baseline.engine.fit(*frozen);newfit=current.engine.fit(*frozen)
        summary['healthy_frozen_fit']=dict(stage=stage,comparisons={k:compare(newfit[k],oldfit[k]) for k in ('A','b','c','correction_log_integral')})
        frozenpolicy=[before['history'][k][stage] for k in ('A','b','c')]
        oldforward=baseline.engine.forward(*frozenpolicy,y,normal[stage],uniform[stage])
        newforward=current.engine.forward(*frozenpolicy,y,normal[stage],uniform[stage])
        summary['healthy_frozen_forward']=dict(stage=stage,comparisons={k:compare(newforward[k],oldforward[k],exact=k=='ancestors') for k in ('x','ancestors','logZ','log_incremental_weight')})
        persist()
        # Capture identical frozen inputs in both backends. No new random draw.
        candidate=campaign/'attempt108_c2_lorenz_bounded_integral_safety/covariance_floor_0.001'
        captured=campaign/'attempt111_c2_lorenz_bounded_crossover_capture'
        values=[load(candidate,k)[3,:,0:1] for k in ('A','b','c','x','log_incremental_weight')]
        engine=LorenzRefinement(forcing=2.5,precision_policy='nearest_covariance',precision_floor=.001,
            precision_eigen_tolerance=8*2.220446049250313e-16,precision_eigen_refinement=True,
            precision_stable_integral=True).engine
        with tf.device('/GPU:0'):gpu=engine.fit(*values)
        graph=tf.function(engine._fit,input_signature=engine.fit.input_signature,jit_compile=False,autograph=False)
        with tf.device('/CPU:0'):cpu=graph(*values)
        check('both_frozen_backward_valid',gpu['valid'].numpy() and cpu['valid'].numpy())
        for lane,result in (('gpu',gpu),('cpu_graph',cpu)):
            folder=args.output/lane;folder.mkdir()
            for k,v in result.items():tf.io.write_file(str(folder/(k+'.tensor')),tf.io.serialize_tensor(v))
            summary['recursion_comparisons'].append(dict(lane=lane,keys={k:compare(result[k],load(captured,k)) for k in
                ('A','b','c','correction_log_integral','projection_added_A')}))
        summary['traces']=dict(gpu=engine.fit.experimental_get_tracing_count(),cpu_graph=graph.experimental_get_tracing_count())
        check('stable_graphs',all(v==1 for v in summary['traces'].values()))
        summary['guard_admissible']=all(v['passed'] for v in summary['checks'])
        summary['status']='PASS';persist()
    except Exception as exc:
        summary['error']=repr(exc);persist();import traceback;traceback.print_exc()
    print(json.dumps({k:summary.get(k) for k in ('status','healthy_iterations','recursion_comparisons','error','wall_seconds')}))
    if summary['status']!='PASS':raise SystemExit(1)


if __name__=='__main__':main()
