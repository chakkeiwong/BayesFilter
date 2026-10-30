"""GPU/XLA analytic and corrected-source replay checks for the RBF consumer."""
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
sys.path.insert(0,'/home/chakwong/BayesFilter')
import tensorflow as tf
from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
memory=configure_tensorflow_gpu_memory_growth(tf,require_gpu=True)
tf.config.experimental.enable_tensor_float_32_execution(False)
spec=importlib.util.spec_from_file_location('rbf_snapshot',Path(__file__).with_name('rbf_controlled_smc_tf.py'))
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
D=tf.float64


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fixture',type=Path,required=True);args=parser.parse_args()
    started=time.monotonic();fixture=json.loads(args.fixture.read_text());checks=[]
    def check(name,actual,expected,atol=1e-8,rtol=1e-9):
        actual=tf.convert_to_tensor(actual,D);expected=tf.convert_to_tensor(expected,D)
        error=tf.abs(actual-expected)
        passed=bool(tf.reduce_all(tf.math.is_finite(actual)&(error<=atol+rtol*tf.abs(expected))).numpy())
        checks.append(dict(name=name,passed=passed,max_absolute_error=float(tf.reduce_max(error).numpy()),atol=atol,rtol=rtol))
    def flag(name,value):check(name,float(bool(value)),1.,0.,0.)
    try:
        mixture=tf.function(core.rbf_gaussian_mixture,input_signature=[tf.TensorSpec([2,3],D),tf.TensorSpec([],D),
            tf.TensorSpec([2],D),tf.TensorSpec([2,2],D),tf.TensorSpec([2,2],D),tf.TensorSpec([2],D)],jit_compile=True,autograph=False)
        mean=tf.constant([[-3.,0.,3.],[-1.,.5,2.]],D);q=tf.constant(2.,D)
        beta=tf.constant([.4,.7],D);weights=tf.constant([[.2,.8],[1.,0.]],D)
        knots=tf.constant([[-2.,3.],[-1.,1.]],D);constant=tf.constant([.1,-.2],D)
        m=mixture(mean,q,beta,weights,knots,constant);flag('mixture_valid',m['valid'].numpy())
        # Independently complete the square in precision form, as in supplement9.1.
        precision=1/q+2*beta;variance=1/precision
        expected_mean=(mean[:,:,None]/q+2*beta[:,None,None]*knots[:,None,:])*variance[:,None,None]
        logmass=tf.math.log(weights)[:,None,:]-beta[:,None,None]*knots[:,None,:]**2+.5*precision[:,None,None]*expected_mean**2
        expected_integral=(constant[:,None]+.5*tf.math.log(variance/q)[:,None]
            -.5*mean**2/q+tf.reduce_logsumexp(logmass,-1))
        check('component_means',m['mean'],expected_mean,1e-12,1e-12)
        check('component_variances',m['variance'],variance,1e-12,1e-12)
        check('mixture_probabilities',m['probabilities'],tf.nn.softmax(logmass,-1),1e-12,1e-12)
        check('gaussian_integral',m['log_integral'],expected_integral,1e-12,1e-12)
        # Joint density identity verifies policy/normalizer and normalized mixture together.
        x=tf.constant([[-1.,2.,4.],[.1,-2.,3.]],D)
        logproposal=tf.reduce_logsumexp(tf.math.log(m['probabilities'])-.5*(tf.math.log(2*math.pi*m['variance'])[:,None,None]
            +(x[:,:,None]-m['mean'])**2/m['variance'][:,None,None]),-1)
        logbase=-.5*(tf.math.log(2*math.pi*q)+(x-mean)**2/q)
        check('density_ratio',logproposal,logbase+core.rbf_log_policy(x,beta,weights,knots,constant)-m['log_integral'],1e-12,1e-12)
        for name,b,w in [('negative_beta',-beta,weights),('negative_weight',beta,-weights),('zero_mass',beta,tf.zeros_like(weights))]:
            invalid=mixture(mean,q,b,w,knots,constant)
            flag(name+'_rejected',not bool(invalid['valid'].numpy()))
        flag('mixture_single_trace',mixture.experimental_get_tracing_count()==1)
        h,n,k=(int(fixture[key]) for key in ('observations','particles','components'))
        engine=core.RBFControlledSMC(observations=h,particles=n,replicas=2,components=k,observation_variance=fixture['observation_variance'])
        # Different source rounds occupy the leading replica dimension of one endpoint.
        def field(name):return tf.stack([tf.constant(r[name],D) for r in fixture['rounds']],axis=1)
        y=tf.tile(tf.constant(fixture['y'],D)[:,None],[1,2])
        output=engine.forward(*(field(k) for k in ('beta','weights','knots','constant')),y,
                              field('normal'),field('component_uniform'),field('uniform'))
        flag('corrected_source_consumer_valid',output['valid'].numpy())
        for name in ('x','log_incremental_weight','logZ','ess'):check('corrected_source_'+name,output[name],field(name))
        check('corrected_source_ancestry',tf.cast(output['ancestors'],D),field('ancestors'),0.,0.)
        # Independent fresh component/normal/resampling streams exercise the repaired path.
        fresh=engine.forward(*(field(k) for k in ('beta','weights','knots','constant')),y,
            tf.random.stateless_normal([h,2,n],[20261005,1],dtype=D),
            tf.random.stateless_uniform([h,2,n],[20261005,2],dtype=D),
            tf.random.stateless_uniform([h,2],[20261005,3],dtype=D))
        flag('fresh_consumer_valid',fresh['valid'].numpy());flag('forward_single_trace',engine.forward.experimental_get_tracing_count()==1)
        for name in ('x','log_incremental_weight','logZ','ess','ancestors'):
            tf.io.write_file(str(args.output/(name+'.tensor')),tf.io.serialize_tensor(output[name]))
        summary=dict(status='PASS' if all(c['passed'] for c in checks) else 'FAIL',scope='RBF_CORRECTED_FORWARD_CONSUMER',replication='INCOMPLETE',checks=checks,
            source_target='ONE_EXPLICIT_ANCESTOR_PROBABILITY_INDEX_REPAIR',fit_implemented=False,
            final_logZ=output['logZ'][-1].numpy().tolist(),fresh_final_logZ=fresh['logZ'][-1].numpy().tolist(),
            seed=20261005,gpu_memory_policy=memory,tensorflow=tf.__version__,device=tf.config.list_logical_devices('GPU')[0].name,
            dtype='float64',jit_compile=True,tf32=False,numpy_runtime=False,pfor=False,python_time_loop=False,
            traces={'forward':engine.forward.experimental_get_tracing_count(),'mixture':mixture.experimental_get_tracing_count()},
            allocator=tf.config.experimental.get_memory_info('GPU:0'),wall_seconds=time.monotonic()-started,
            nonclaims=['No NNLS fitting port','No Table2 replication','No stochastic ranking or general default'])
    except Exception as exc:
        summary=dict(status='FAIL',error=repr(exc),checks=checks,gpu_memory_policy=memory,wall_seconds=time.monotonic()-started)
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n');raise
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:summary[k] for k in ('status','scope','final_logZ','wall_seconds')}))


if __name__=='__main__':main()
