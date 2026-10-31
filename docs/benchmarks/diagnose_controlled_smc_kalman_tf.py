"""GPU/XLA consumer vs independent exact scalar Gaussian reference checks."""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
os.environ['BAYESFILTER_PRELOAD_CUSTOM_OP']='0'
ROOT=Path('/home/chakwong/BayesFilter');sys.path.insert(0,str(ROOT))


def exact_references(observations,alpha,q,r,m0,v0):
    # Independent CPU diagnostic scalar recursions; no runtime implementation.
    m,v,total=m0,v0,0.
    for t,y in enumerate(observations):
        if t: m,v=alpha*m,alpha*alpha*v+q
        innovation=y-m;variance=v+r
        total-=.5*(math.log(2*math.pi*variance)+innovation*innovation/variance)
        gain=v/variance;m+=gain*innovation;v*=1-gain
    a,b,c=0.,0.,0.;policy=[]
    for y in reversed(observations):
        denominator=1+2*q*a;k=q/denominator
        a,b,c=(alpha*alpha*a/denominator+.5/r,
               alpha*b/denominator-y/r,
               c+.5*math.log(denominator)-.5*k*b*b+.5*y*y/r+.5*math.log(2*math.pi*r))
        policy.append([a,b,c])
    return total,list(reversed(policy))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--fixture')
    args=parser.parse_args();output=Path(args.output)
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory=configure_tensorflow_gpu_memory_growth(tf,require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    spec=importlib.util.spec_from_file_location('csmc_snapshot',output/'sources/controlled_smc_tf.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    values=[-.7,.1,1.4,-.4,.8,1.1,-.3,.2];h=len(values);n=64
    alpha,q,r,m0,v0=.8,.35,.4,.2,1.3
    expected_logz,expected_policy=exact_references(values,alpha,q,r,m0,v0)
    def observation(y,x):return -.5*((y-x)**2/r+math.log(2*math.pi*r))
    kernel=module.ScalarGaussianControlledSMC(observations=h,particles=n,alpha=alpha,
        process_variance=q,initial_mean=m0,initial_variance=v0,observation_log_density=observation,
        precision_policy='source_scalar_projection')
    y=tf.constant(values,tf.float64)
    def draw(stream):
        return (tf.random.stateless_normal([h,n],[20260929,stream],dtype=tf.float64),
                tf.random.stateless_uniform([h],[20260929,stream+1],dtype=tf.float64))
    bootstrap=kernel.forward(tf.zeros([h,3],tf.float64),y,*draw(71))
    fitted=kernel.fit(tf.zeros([h,3],tf.float64),bootstrap['x'],bootstrap['log_incremental_weight'])
    checks=[]
    def check(name,actual,expected):
        actual=tf.cast(actual,tf.float64);expected=tf.convert_to_tensor(expected,dtype=tf.float64)
        error=tf.abs(actual-expected);allowed=1e-8+1e-8*tf.abs(expected)
        checks.append(dict(name=name,pass_check=bool(tf.reduce_all(tf.logical_and(tf.math.is_finite(error),error<=allowed)).numpy()),
                           maximum_absolute_error=float(tf.reduce_max(error).numpy())))
    check('optimal_policy_coefficients',fitted['policy'],expected_policy)
    checks.append(dict(name='bootstrap_and_fit_valid',pass_check=bool(bootstrap['valid'].numpy()) and bool(fitted['valid'].numpy())))
    checks.append(dict(name='no_precision_projection',pass_check=not bool(tf.reduce_any(fitted['projected']).numpy())))
    evaluations=[]
    for stream in [73,75]:
        result=kernel.forward(fitted['policy'],y,*draw(stream))
        check(f'kalman_logZ_stream{stream}',result['logZ'][-1],expected_logz)
        check(f'unit_weights_stream{stream}',result['log_incremental_weight'],tf.zeros([h,n],tf.float64))
        check(f'ess_stream{stream}',result['ess'],tf.fill([h],tf.constant(n,tf.float64)))
        checks.append(dict(name=f'valid_stream{stream}',pass_check=bool(result['valid'].numpy())))
        evaluations.append(dict(stream=stream,logZ=float(result['logZ'][-1].numpy()),min_ess=float(tf.reduce_min(result['ess']).numpy())))
    traces={name:getattr(kernel,name).experimental_get_tracing_count() for name in ['forward','fit']}
    checks.append(dict(name='one_trace_per_signature',pass_check=all(value==1 for value in traces.values())))
    summary=dict(status='PASS' if all(c['pass_check'] for c in checks) else 'FAIL',
        scope='SCALAR_GAUSSIAN_EXACT_KALMAN_CONTROL',replication='INCOMPLETE',ranking='NOT_TESTED',checks=checks,
        configuration=dict(observations=values,particles=n,alpha=alpha,process_variance=q,observation_variance=r,
                           initial_mean=m0,initial_variance=v0,seed=20260929,streams=list(range(71,77))),
        kalman_logZ=expected_logz,optimal_policy=expected_policy,fitted_policy=fitted['policy'].numpy().tolist(),
        bootstrap_logZ=float(bootstrap['logZ'][-1].numpy()),evaluations=evaluations,
        minimum_gram_eigenvalue_ratio=float(tf.reduce_min(fitted['gram_eigenvalue_ratio']).numpy()),
        trace_counts=traces,jit_compile=True,dtype='float64',tf32=False,numpy_runtime=False,pfor=False,
        tensorflow=tf.__version__,device=result['logZ'].device,gpu_memory_policy=memory,
        gpu_allocator=tf.config.experimental.get_memory_info('GPU:0'),
        limitations=['constructed scalar Gaussian identity only','no posterior moment or stochastic performance claim'])
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:summary[k] for k in ['status','scope','kalman_logZ','trace_counts']}))
    return 0 if summary['status']=='PASS' else 1


if __name__=='__main__':sys.exit(main())
