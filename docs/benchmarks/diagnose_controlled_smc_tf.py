"""GPU/XLA fixed-input and analytic diagnostics; no empirical ranking."""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
os.environ['BAYESFILTER_PRELOAD_CUSTOM_OP']='0'
ROOT=Path('/home/chakwong/BayesFilter')
sys.path.insert(0,str(ROOT))
import tensorflow as tf
from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
memory=configure_tensorflow_gpu_memory_growth(tf,require_gpu=True)
tf.config.experimental.enable_tensor_float_32_execution(False)
# Execute the snapshotted numerical module, so evidence binds the callable.
spec=importlib.util.spec_from_file_location('controlled_smc_snapshot',Path(__file__).with_name('controlled_smc_tf.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--fixture',required=True)
    parser.add_argument('--parity-only',action='store_true',help='Replay saved source clouds; analytic and random controls already run separately')
    args=parser.parse_args();output=Path(args.output);fixture=json.loads(Path(args.fixture).read_text())
    h=len(fixture['y']);n=len(fixture['rounds'][0]['x'][0]);d=tf.float64
    core=module.ScalarGaussianControlledSMC(observations=h,particles=n,alpha=fixture['alpha'],
        process_variance=fixture['variance'],initial_mean=0,initial_variance=1,
        observation_log_density=module.binomial_logistic_observation(50),precision_policy='source_scalar_projection')
    rows=[]
    def check(name,actual,expected,tolerance=1e-7):
        actual=tf.cast(actual,d);expected=tf.convert_to_tensor(expected,dtype=d)
        error=tf.abs(actual-expected);scaled=error/(tolerance+tolerance*tf.abs(expected))
        good=bool(tf.reduce_all(tf.logical_and(tf.math.is_finite(error),scaled<=1)).numpy())
        rows.append(dict(name=name,pass_check=good,max_abs=float(tf.reduce_max(error).numpy()),max_tolerance_units=float(tf.reduce_max(scaled).numpy())))
    def flag(name,value): rows.append(dict(name=name,pass_check=bool(value)))
    y=tf.constant(fixture['y'],d)
    timings=[]
    for j,round_ in enumerate(fixture['rounds']):
        p=tf.constant(round_['policy'],d);x=tf.constant(round_['x'],d)
        w=tf.constant(round_['log_incremental_weight'],d)
        start=time.monotonic();result=core.cloud_weights(p,y,x);result['logZ'].numpy()
        timings.append(dict(kernel='cloud_weights',round=j,seconds=time.monotonic()-start))
        check(f'round{j}:incremental',result['log_incremental_weight'],w)
        check(f'round{j}:logZ',result['logZ'],round_['logZ'])
        check(f'round{j}:ESS',result['ess'],round_['ess'])
        flag(f'round{j}:valid',result['valid'].numpy())
        if j+1<len(fixture['rounds']):
            start=time.monotonic();fit=core.fit(p,x,w);fit['policy'].numpy()
            timings.append(dict(kernel='fit',round=j,seconds=time.monotonic()-start))
            check(f'round{j}:backward_policy',fit['policy'],fixture['rounds'][j+1]['policy'])
            flag(f'round{j}:fit_valid',fit['valid'].numpy())
            rows.append(dict(name=f'round{j}:fit_diagnostics',pass_check=True,
                minimum_gram_eigenvalue_ratio=float(tf.reduce_min(fit['gram_eigenvalue_ratio']).numpy()),
                projection_count=int(tf.reduce_sum(tf.cast(fit['projected'],tf.int32)).numpy()),
                maximum_fit_residual=float(tf.reduce_max(fit['max_residual']).numpy())))
        # Recover draws and one uniform per time from the saved source trajectory.
        # This replays the actual forward endpoint including native resampling.
        ancestors=tf.cast(tf.constant(round_['ancestors'],d),tf.int32)
        previous=tf.gather(x[:-1],ancestors[:-1],batch_dims=1)
        base_mean=tf.concat([tf.zeros([1,n],d),fixture['alpha']*previous],axis=0)
        variance=tf.concat([tf.ones([1,1],d),tf.fill([h-1,1],tf.constant(fixture['variance'],d))],axis=0)
        k=1/(1/variance+2*p[:,0,None]);mean=k*(base_mean/variance-p[:,1,None])
        normal=(x-mean)/tf.sqrt(k)
        cdf=tf.cumsum(tf.nn.softmax(w,axis=1),axis=1)
        upper=tf.gather(cdf,ancestors,batch_dims=1)*n-tf.cast(tf.range(n)[None,:],d)
        cdf_before=tf.concat([tf.zeros([h,1],d),cdf[:,:-1]],axis=1)
        lower=tf.gather(cdf_before,ancestors,batch_dims=1)*n-tf.cast(tf.range(n)[None,:],d)
        lo=tf.maximum(0.,tf.reduce_max(lower,axis=1));hi=tf.minimum(1.,tf.reduce_min(upper,axis=1))
        flag(f'round{j}:recovered_uniform_interval',tf.reduce_all(hi>lo).numpy())
        uniform=.5*(lo+hi)
        start=time.monotonic();replayed=core.forward(p,y,normal,uniform);replayed['logZ'].numpy()
        timings.append(dict(kernel='forward',round=j,seconds=time.monotonic()-start))
        check(f'round{j}:forward_cloud',replayed['x'],x)
        check(f'round{j}:forward_logZ',replayed['logZ'],round_['logZ'])
        flag(f'round{j}:forward_ancestry',tf.reduce_all(replayed['ancestors']==ancestors).numpy())
        flag(f'round{j}:forward_valid',replayed['valid'].numpy())
    adaptive=[]
    if not args.parity_only:
        # Independent numerical integration of a Gaussian times a quadratic twist.
        def primitive(mean,variance,coeff): return module.gaussian_twist(mean,variance,coeff)
        primitive=tf.function(primitive,input_signature=[tf.TensorSpec([],d),tf.TensorSpec([],d),tf.TensorSpec([3],d)],jit_compile=True)
        for index,(mean,var,a,b,c) in enumerate([(0.,1.,0.,0.,0.),(.7,1.3,.2,-.4,.5),(-1.,.8,-.2,.3,-.7)]):
            m,k,log_integral,valid=primitive(tf.constant(mean,d),tf.constant(var,d),tf.constant([a,b,c],d))
            grid=tf.linspace(m-12*tf.sqrt(k),m+12*tf.sqrt(k),20001)
            density=tf.exp(-.5*(grid-mean)**2/var-.5*math.log(2*math.pi*var)-a*grid*grid-b*grid-c)
            integral=(grid[1]-grid[0])*(tf.reduce_sum(density)-.5*(density[0]+density[-1]))
            check(f'gaussian_integral_{index}',tf.math.log(integral),log_integral,1e-9)
            check(f'proposal_normalization_{index}',integral/tf.exp(log_integral),1.,1e-9)
            flag(f'gaussian_valid_{index}',valid.numpy())
        bad=primitive(tf.constant(0.,d),tf.constant(1.,d),tf.constant([-.5,0.,0.],d))
        flag('zero_precision_rejected',not bool(bad[3].numpy()) and not bool(tf.math.is_finite(bad[1]).numpy()))
        deficient=core.fit(tf.zeros([h,3],d),tf.ones([h,n],d),tf.zeros([h,n],d))
        flag('rank_deficiency_rejected',not bool(deficient['valid'].numpy()))
        check('systematic_fixture',module.systematic_ancestors(tf.constant([.1,.2,.3,.4],d),tf.constant(.5,d)),[1,2,3,3])
        # Fresh random streams exercise fitting then the actual adaptive consumer.
        noise=tf.random.stateless_normal([h,n],seed=[20260929,1],dtype=d)
        uniform=tf.random.stateless_uniform([h],seed=[20260929,2],dtype=d)
        policy=tf.zeros([h,3],d);adaptive=[]
        for iteration in range(3):
            result=core.forward(policy,y,noise,uniform)
            adaptive.append(float(result['logZ'][-1].numpy()));flag(f'adaptive{iteration}:valid',result['valid'].numpy())
            if iteration<2: policy=core.fit(policy,result['x'],result['log_incremental_weight'])['policy']
    traces={name:getattr(core,name).experimental_get_tracing_count() for name in ['forward','cloud_weights','fit']}
    flag('stable_signatures',all(v==1 for v in traces.values()))
    summary=dict(status='PASS' if all(row['pass_check'] for row in rows) else 'FAIL',
        scope='SCALAR_NEUROSCIENCE_FIXED_INPUT_GPU_XLA_PARITY',replication='INCOMPLETE',ranking='NOT_TESTED',
        checks=rows,timings=timings,traces=traces,adaptive_logZ_descriptive=adaptive,parity_only=args.parity_only,
        gpu_memory_policy=memory,gpu_allocator=tf.config.experimental.get_memory_info('GPU:0'),
        tensorflow=tf.__version__,device=result['logZ'].device,dtype='float64',tf32=False,jit_compile=True,
        seed=20260929,particles=n,observations=h,pfor=False,numpy_runtime=False,
        implementation_scope='scalar affine Gaussian dynamics; multidimensional route remains open')
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:summary[k] for k in ['status','scope','traces','adaptive_logZ_descriptive','gpu_allocator']}))
    return 0 if summary['status']=='PASS' else 1

if __name__=='__main__':sys.exit(main())
