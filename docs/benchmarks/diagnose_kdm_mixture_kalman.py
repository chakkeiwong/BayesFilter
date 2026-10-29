"""Independent branching and gradient checks for the Appendix B exact reference."""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import sys
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
os.environ['BAYESFILTER_PRELOAD_CUSTOM_OP']='0'
ROOT=Path('/home/chakwong/BayesFilter');sys.path.insert(0,str(ROOT))
import tensorflow as tf
from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
memory=configure_tensorflow_gpu_memory_growth(tf,require_gpu=True)
tf.config.experimental.enable_tensor_float_32_execution(False)
spec=importlib.util.spec_from_file_location('mixture_reference',Path(__file__).with_name('mixture_kalman_reference_tf.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def logsumexp(values):
    m=max(values)
    return m+math.log(math.fsum(math.exp(v-m) for v in values))


def branching_reference(theta,actions,observations,states):
    """Python diagnostic only: grow branches and normalize at each time."""
    a,b,c1,c2,o1,o2,v=theta
    w1=1/(1+math.exp(v));components=((c1,o1,w1),(c2,o2,1-w1))
    result={k:[] for k in ('means','variances','logweights','logZ','terminal_logdensity')}
    for action,observation,state in zip(actions,observations,states):
        branches=[(0,.1,1.2,0.)];evidence=0.
        for t,(u,y) in enumerate(zip(action,observation)):
            expanded=[]
            for path,m,p,logweight in branches:
                m=a*m+b*u;p=a*a*p+.3**2
                for z,(c,offset,w) in enumerate(components):
                    residual=y-c*m-offset;s=c*c*p+.4**2
                    expanded.append((path+(z<<t),m+p*c*residual/s,p*.4**2/s,
                        logweight+math.log(w)-.5*(math.log(2*math.pi*s)+residual**2/s)))
            increment=logsumexp([row[3] for row in expanded]);evidence+=increment
            branches=[(path,m,p,w-increment) for path,m,p,w in expanded]
        branches.sort(key=lambda row:row[0])
        for key,index in (('means',1),('variances',2),('logweights',3)):
            result[key].append([row[index] for row in branches])
        result['logZ'].append(evidence)
        result['terminal_logdensity'].append(logsumexp([
            w-.5*(math.log(2*math.pi*p)+(state-m)**2/p) for _,m,p,w in branches]))
    result['loss']=-math.fsum(result['terminal_logdensity'])/len(states)
    return result


def kalman_reference(theta,actions,observations,states):
    a,b,c,_,offset,_,_=theta;evidence=[];densities=[]
    for action,observation,state in zip(actions,observations,states):
        m,p,z=.1,1.2,0.
        for u,y in zip(action,observation):
            m=a*m+b*u;p=a*a*p+.3**2;s=c*c*p+.4**2;residual=y-c*m-offset
            z-=.5*(math.log(2*math.pi*s)+residual**2/s)
            m+=p*c*residual/s;p*=.4**2/s
        evidence.append(z);densities.append(-.5*(math.log(2*math.pi*p)+(state-m)**2/p))
    return evidence,densities


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    output=Path(parser.parse_args().output)
    theta=[.8,.4,1.,-1.2,.2,-.3,.4]
    actions=[[.2,-.1,.4,0.,-.2],[-.5,.3,.1,-.2,.6],[.1,.2,-.3,.4,-.5],[0.,0.,.5,-.5,.2]]
    observations=[[.2,-.6,.8,-.3,.1],[1.1,-1.2,.3,.2,-.7],[-.2,.5,-.8,.9,-.4],[.7,-.5,.2,-.1,.6]]
    states=[.3,-.4,.1,.7]
    fixture=dict(theta=theta,actions=actions,observations=observations,terminal_states=states,
        initial_mean=.1,initial_variance=1.2,sigma=.3,gamma=.4,
        provenance='CONSTRUCTED_REFERENCE_CONTROLS_NOT_AUTHOR_EXPERIMENT_SETTINGS')
    (output/'fixture.json').write_text(json.dumps(fixture,indent=2)+'\n')
    core=module.BimodalLinearMixtureKalmanReference(batch=4,horizon=5,sigma=.3,gamma=.4,initial_mean=.1,initial_variance=1.2)
    args_tf=[tf.constant(v,tf.float64) for v in (theta,actions,observations,states)]
    checks=[];device=None
    def check(name,got,want,atol=1e-9,rtol=1e-9):
        got=tf.convert_to_tensor(got,tf.float64);want=tf.convert_to_tensor(want,tf.float64)
        error=tf.abs(got-want)
        passed=bool(tf.reduce_all(tf.logical_and(tf.math.is_finite(got),error<=atol+rtol*tf.abs(want))).numpy())
        checks.append(dict(name=name,passed=passed,max_abs=float(tf.reduce_max(error).numpy())))
        if not passed:raise RuntimeError('Reference check failed: '+name)
    try:
        result=core.evaluate(*args_tf);device=result['logZ'].device
        if not bool(result['valid'].numpy()):raise RuntimeError('Invalid exact mixture result')
        reference=branching_reference(theta,actions,observations,states)
        for key in ('means','variances','logweights','logZ','terminal_logdensity','loss'):
            check('branching_reference.'+key,result[key],reference[key])
        check('normalized_weights',tf.reduce_sum(tf.exp(result['logweights']),axis=1),[1.]*4)
        gradient=core.value_and_gradient(*args_tf)
        if not bool(gradient['valid'].numpy()):raise RuntimeError('Invalid exact-reference gradient')
        delta=1e-5;finite_difference=[]
        for index in range(7):
            plus,minus=list(theta),list(theta);plus[index]+=delta;minus[index]-=delta
            finite_difference.append((branching_reference(plus,actions,observations,states)['loss']-
                branching_reference(minus,actions,observations,states)['loss'])/(2*delta))
        check('total_parameter_gradient',gradient['gradient'],finite_difference,2e-6,2e-6)
        collapsed=list(theta);collapsed[3]=collapsed[2];collapsed[5]=collapsed[4]
        reduced=core.evaluate(tf.constant(collapsed,tf.float64),*args_tf[1:])
        kalman_z,kalman_density=kalman_reference(collapsed,actions,observations,states)
        check('collapsed.logZ',reduced['logZ'],kalman_z)
        check('collapsed.terminal_density',reduced['terminal_logdensity'],kalman_density)
        try:
            module.BimodalLinearMixtureKalmanReference(batch=4,horizon=5,sigma=.3,gamma=0.,initial_mean=.1,initial_variance=1.2)
        except ValueError:checks.append(dict(name='reject_zero_observation_variance',passed=True,max_abs=0.))
        else:raise RuntimeError('Accepted invalid observation variance')
        bad=core.evaluate(tf.constant([float('nan')]+theta[1:],tf.float64),*args_tf[1:])
        if bool(bad['valid'].numpy()):raise RuntimeError('Accepted nonfinite theta')
        checks.append(dict(name='reject_nonfinite_theta',passed=True,max_abs=0.))
        traces={name:getattr(core,name).experimental_get_tracing_count() for name in ('evaluate','value_and_gradient')}
        if any(v!=1 for v in traces.values()):raise RuntimeError('Unexpected retracing')
        status,error='PASS',None
    except Exception as exc:status,error='FAIL',str(exc)
    traces={name:getattr(core,name).experimental_get_tracing_count() for name in ('evaluate','value_and_gradient')}
    (output/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    summary=dict(status=status,scope='APPENDIX_B_EQUATIONS_EXACT_REFERENCE_AND_GRADIENT_CONTROLS_ONLY',
        replication='INCOMPLETE',source_settings='UNRECOVERED',error=error,checks=len(checks),
        passed_checks=sum(c['passed'] for c in checks),traces=traces,gpu_memory_policy=memory,
        gpu_allocator=tf.config.experimental.get_memory_info('GPU:0'),
        host_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        tensorflow=tf.__version__,dtype='float64',tf32=False,jit_compile=True,device=device,
        seed='N/A deterministic constructed controls',pfor=False,numpy_runtime=False)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return 0 if status=='PASS' else 1


if __name__=='__main__':sys.exit(main())
