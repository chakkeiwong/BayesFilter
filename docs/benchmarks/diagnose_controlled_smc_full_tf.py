"""Full-horizon scalar controlled-SMC GPU/XLA execution diagnostic."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
os.environ['BAYESFILTER_PRELOAD_CUSTOM_OP']='0'
ROOT=Path('/home/chakwong/BayesFilter');sys.path.insert(0,str(ROOT))
import tensorflow as tf
from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
memory=configure_tensorflow_gpu_memory_growth(tf,require_gpu=True)
tf.config.experimental.enable_tensor_float_32_execution(False)
spec=importlib.util.spec_from_file_location('controlled_smc_snapshot',Path(__file__).with_name('controlled_smc_tf.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--fixture')
    args=parser.parse_args();output=Path(args.output)
    # An isolated diagnostic process converts the existing MAT reference data.
    # The actual TF consumer receives ordinary JSON; no NumPy runtime path.
    data=ROOT/'.localresources/code/heng-controlled-smc/StateSpaceModels/NeuroScience/Data_NeuroScience.mat'
    command=[sys.executable,'-c','from scipy.io import loadmat; import json,sys; print(json.dumps(loadmat(sys.argv[1])["y"].ravel().tolist()))',str(data)]
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='-1'
    y_values=json.loads(subprocess.check_output(command,text=True,env=env))
    (output/'observations.json').write_text(json.dumps(y_values)+'\n')
    h=len(y_values);n=1024;d=tf.float64
    core=module.ScalarGaussianControlledSMC(observations=h,particles=n,alpha=.99,process_variance=.11,
        initial_mean=0,initial_variance=1,observation_log_density=module.binomial_logistic_observation(50),
        precision_policy='source_scalar_projection')
    y=tf.constant(y_values,d);policy=tf.zeros([h,3],d);rows=[];healthy=True
    for iteration in range(3):
        noise=tf.random.stateless_normal([h,n],seed=[20260929,iteration*2+31],dtype=d)
        uniform=tf.random.stateless_uniform([h],seed=[20260929,iteration*2+32],dtype=d)
        start=time.monotonic();result=core.forward(policy,y,noise,uniform)
        logz=float(result['logZ'][-1].numpy());elapsed=time.monotonic()-start
        healthy=healthy and bool(result['valid'].numpy())
        row=dict(iteration=iteration,logZ=logz,minimum_ess=float(tf.reduce_min(result['ess']).numpy()),forward_seconds=elapsed)
        if iteration<2:
            start=time.monotonic();fit=core.fit(policy,result['x'],result['log_incremental_weight'])
            policy=fit['policy'];policy.numpy();row['fit_seconds']=time.monotonic()-start
            healthy=healthy and bool(fit['valid'].numpy())
            row['minimum_gram_eigenvalue_ratio']=float(tf.reduce_min(fit['gram_eigenvalue_ratio']).numpy())
            row['projection_count']=int(tf.reduce_sum(tf.cast(fit['projected'],tf.int32)).numpy())
        rows.append(row);print(json.dumps(row),flush=True)
        if not healthy: break
    (output/'final-policy.json').write_text(json.dumps(policy.numpy().tolist(),allow_nan=False)+'\n')
    traces={name:getattr(core,name).experimental_get_tracing_count() for name in ['forward','fit']}
    healthy=healthy and all(v==1 for v in traces.values())
    summary=dict(status='PASS' if healthy else 'FAIL',scope='FULL_HORIZON_SCALAR_GPU_XLA_EXECUTION_ONLY',
        replication='INCOMPLETE',ranking='NOT_TESTED',rows=rows,observations=h,particles=n,iterations=2,
        seed=20260929,stream_policy='fresh stateless streams for each refinement; not author-RNG parity',
        gpu_memory_policy=memory,gpu_allocator=tf.config.experimental.get_memory_info('GPU:0'),
        host_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,traces=traces,
        tensorflow=tf.__version__,dtype='float64',tf32=False,jit_compile=True,device=result['logZ'].device,
        pfor=False,numpy_runtime=False)
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    return 0 if healthy else 1

if __name__=='__main__':sys.exit(main())
