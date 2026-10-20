"""Actual-source KDM/IWSG endpoint vs batched GPU/XLA primitive diagnostics."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true';os.environ['BAYESFILTER_PRELOAD_CUSTOM_OP']='0'
ROOT=Path('/home/chakwong/BayesFilter');sys.path.insert(0,str(ROOT))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--fixture')
    args=parser.parse_args();output=Path(args.output)
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='-1'
    # Match compiled dependencies in the pinned CPU reference environment.
    reference_lib='/home/chakwong/anaconda3/envs/tf-gpu2/lib'
    env['LD_LIBRARY_PATH']=reference_lib+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    command=[str(ROOT/'.localresources/envs/mdpf-reference-20260924/bin/python'),str(output/'sources/reference_kdm_iwsg.py'),'--output',str(output/'author-fixture.json')]
    subprocess.run(command,env=env,check=True)
    reference=json.loads((output/'author-fixture.json').read_text())
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory=configure_tensorflow_gpu_memory_growth(tf,require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    spec=importlib.util.spec_from_file_location('kdm_snapshot',output/'sources/kdm_iwsg_tf.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    kernel=module.GaussianMixtureIWSG(batch_size=2,particles=3,samples=3,dimensions=1,
        density_weight_epsilon=1e-8,injection_epsilon=1e-8)
    inputs=[tf.constant(reference[name],tf.float64) for name in ['locations','weights','bandwidth','samples']]
    result=kernel.score_check(*inputs);checks=[]
    def check(name,actual,expected):
        actual=tf.cast(actual,tf.float64);expected=tf.convert_to_tensor(expected,dtype=tf.float64)
        error=tf.abs(actual-expected);scaled=error/(1e-7+1e-7*tf.abs(expected))
        checks.append(dict(name=name,pass_check=bool(tf.reduce_all(tf.logical_and(tf.math.is_finite(error),scaled<=1)).numpy()),
                           maximum_absolute_error=float(tf.reduce_max(error).numpy())))
    for name in ['log_density','iwsg','raw_loss','normalized_loss']:
        check(name,result[name],reference[name])
    for name in ['raw_gradient','normalized_gradient']:
        for i,label in enumerate(['locations','weights','bandwidth']):check(name+':'+label,result[name][i],reference[name][i])
    checks.append(dict(name='sample_gradient_stopped',pass_check=bool(result['sample_gradient_stopped'].numpy()) and reference['sample_gradient_stopped']))
    checks.append(dict(name='valid',pass_check=bool(result['valid'].numpy())))
    # Independent numerical density normalization, batch native.
    dense=module.GaussianMixtureIWSG(batch_size=2,particles=3,samples=20001,dimensions=1,
        density_weight_epsilon=1e-8,injection_epsilon=1e-8)
    grid=tf.linspace(tf.constant(-14.,tf.float64),tf.constant(14.,tf.float64),20001)
    samples=tf.broadcast_to(grid[None,:,None],[2,20001,1]);prob=tf.exp(dense.evaluate(*inputs[:3],samples)['log_density'])
    integral=(grid[1]-grid[0])*(tf.reduce_sum(prob,axis=1)-.5*(prob[:,0]+prob[:,-1]))
    check('density_normalization',integral,tf.ones([2],tf.float64))
    invalid=kernel.evaluate(inputs[0],inputs[1],-inputs[2],inputs[3])
    checks.append(dict(name='negative_bandwidth_rejected',pass_check=not bool(invalid['valid'].numpy())))
    # Repeated call verifies stable signature; no further stochastic comparison.
    kernel.score_check(*inputs)
    traces=kernel.score_check.experimental_get_tracing_count()
    checks.append(dict(name='one_trace',pass_check=traces==1))
    summary=dict(status='PASS' if all(c['pass_check'] for c in checks) else 'FAIL',
        scope='BATCHED_GAUSSIAN_KDM_IWSG_SOURCE_ENDPOINT_PRIMITIVE_PARITY',replication='INCOMPLETE',ranking='NOT_TESTED',
        checks=checks,batch_size=2,scalar_sample_loop=False,pfor=False,numpy_runtime=False,
        dtype='float64',jit_compile=True,tf32=False,trace_count=traces,tensorflow=tf.__version__,
        device=result['log_density'].device,gpu_memory_policy=memory,
        gpu_allocator=tf.config.experimental.get_memory_info('GPU:0'),source_command=command,
        source_environment={k:env[k] for k in ['CUDA_VISIBLE_DEVICES','LD_LIBRARY_PATH']},
        limitations=['Gaussian primitive only','Appendix B coefficients and initialization unrecovered',
                      'full learning, circular bearings, normalized density baselines and independent training repetitions open'])
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:summary[k] for k in ['status','scope','trace_count']}))
    return 0 if summary['status']=='PASS' else 1

if __name__=='__main__':sys.exit(main())
