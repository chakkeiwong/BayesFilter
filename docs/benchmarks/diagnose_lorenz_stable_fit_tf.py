"""GPU/XLA endpoint validation against an independent full-rank SVD reference."""
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path('/home/chakwong/BayesFilter')))
import tensorflow as tf
from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
memory=configure_tensorflow_gpu_memory_growth(tf,require_gpu=True)
tf.config.experimental.enable_tensor_float_32_execution(False)
D=tf.float64
spec=importlib.util.spec_from_file_location('multivariate_snapshot',Path(__file__).with_name('multivariate_controlled_smc_tf.py'))
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--fixture',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    args=parser.parse_args();started=time.monotonic();checks=[]
    fixture=json.loads(args.fixture.read_text());reference=json.loads(args.reference.read_text())
    def check(name,actual,expected,atol=1e-7,rtol=1e-9):
        a=tf.convert_to_tensor(actual,D);e=tf.convert_to_tensor(expected,D)
        delta=tf.abs(a-e);passed=bool(tf.reduce_all(tf.math.is_finite(a)&(delta<=atol+rtol*tf.abs(e))).numpy())
        item=dict(name=name,passed=passed,max_absolute_error=float(tf.reduce_max(delta).numpy()),atol=atol,rtol=rtol)
        checks.append(item)
        if not passed:
            scaled=delta/(atol+rtol*tf.abs(e)) if atol or rtol else delta
            index=int(tf.argmax(tf.reshape(scaled,[-1])).numpy())
            item.update(max_scaled_error=float(tf.reduce_max(scaled).numpy()),flat_index=index,
                        actual_at_worst=float(tf.reshape(a,[-1])[index].numpy()),
                        expected_at_worst=float(tf.reshape(e,[-1])[index].numpy()))
    def flag(name,condition):
        check(name,float(bool(condition)),1.,0.,0.)
        if not condition:raise ValueError('Invalid consumer: '+name)
    def stack_source(stage,key):return tf.stack([tf.constant(r['rounds'][stage][key],D) for r in fixture['replicates']],axis=1)
    def stack_reference(group,key):return tf.stack([tf.constant(r[group][key],D) for r in reference['replicates']],axis=1)
    engine=core.MultivariateGaussianControlledSMC(observations=101,particles=512,replicas=2,dimension=8,
        observation_dimension=6,initial_mean=[0.]*8,initial_covariance=fixture['initial_covariance'],
        transition_covariance=fixture['transition_covariance'],
        transition_mean=lambda x:core.lorenz_rk4(x,tf.constant(fixture['forcing'],D)),
        observation_log_density=lambda y,x:-.5/fixture['observation_variance']*tf.reduce_sum((y-x[...,:6])**2,-1),
        solver='centered_qr',jit_compile=True)
    y=tf.tile(tf.constant(fixture['y'],D)[:,None,:],[1,2,1])
    try:
        # Exact Gaussian ratio identity in well-conditioned, batched coordinates.
        ratio=tf.function(core.centered_correction_integral,input_signature=[
            tf.TensorSpec([2,3,2],D),tf.TensorSpec([2,2],D),tf.TensorSpec([],D),
            tf.TensorSpec([2,2,2],D),tf.TensorSpec([2,2],D),tf.TensorSpec([2],D),
            tf.TensorSpec([2,2],D),tf.TensorSpec([2,2],D),
            tf.TensorSpec([2,2,2],D),tf.TensorSpec([2,2],D),tf.TensorSpec([2],D)],
            jit_compile=True,autograph=False)
        mean=tf.constant([[[.1,.3],[-.2,.6],[.8,-.4]],[[.3,-.2],[.4,.9],[-.5,.3]]],D)
        precision=tf.constant([[2.,.1],[.1,1.]],D);ld=-tf.linalg.slogdet(precision)[1]
        olda=tf.constant([[[.2,.01],[.01,.3]],[[.1,-.03],[-.03,.2]]],D)
        oldb=tf.constant([[.1,-.2],[.3,.1]],D);oldc=tf.constant([.1,.4],D)
        mu=tf.constant([[1.2,-.4],[-.3,.7]],D);scale=tf.constant([[.5,2.],[2.,.3]],D)
        za=tf.constant([[[.02,.003],[.003,.04]],[[.03,-.001],[-.001,.01]]],D)
        zb=tf.constant([[.1,-.02],[-.03,.2]],D);zc=tf.constant([.2,-.1],D)
        da=za/(scale[:,:,None]*scale[:,None,:]);linear=zb/scale
        db=linear-2*tf.einsum('bi,bij->bj',mu,da)
        dc=zc-tf.reduce_sum(linear*mu,-1)+tf.einsum('bi,bij,bj->b',mu,da,mu)
        direct=core.multivariate_twist(mean,precision,ld,olda+da,oldb+db,oldc+dc)[2]-core.multivariate_twist(mean,precision,ld,olda,oldb,oldc)[2]
        value,valid=ratio(mean,precision,ld,olda,oldb,oldc,mu,scale,za,zb,zc)
        flag('gaussian_ratio_valid',valid.numpy());check('gaussian_ratio_identity',value,direct,1e-12,1e-12)
        _,invalid=ratio(mean,precision,ld,olda,oldb,oldc,mu,scale,za-1000*tf.eye(2,batch_shape=[2],dtype=D),zb,zc)
        flag('negative_ratio_precision_rejected',not bool(invalid.numpy()))
        flag('ratio_single_trace',ratio.experimental_get_tracing_count()==1)
        healthy_x=tf.random.stateless_normal([2,512,8],[20260929,91],dtype=D)
        healthy_a=.1*tf.eye(8,batch_shape=[2],dtype=D)
        healthy_b=tf.reshape(tf.linspace(tf.constant(-.2,D),tf.constant(.3,D),16),[2,8])
        healthy_c=tf.constant([.7,-.3],D)
        healthy=engine.regress(healthy_x,core.quadratic(healthy_x,healthy_a,healthy_b,healthy_c))
        flag('healthy_polynomial_valid',healthy['valid'].numpy())
        for k,expected in [('A',healthy_a),('b',healthy_b),('c',healthy_c)]:
            check('healthy_polynomial_'+k,healthy[k],expected,1e-10,1e-10)
        source_replay=engine.forward(*(stack_source(1,k) for k in ('A','b','c')),y,stack_source(1,'normal'),stack_source(1,'uniform'))
        flag('unchanged_source_forward_valid',source_replay['valid'].numpy())
        for k in ('x','log_incremental_weight','logZ','ess'):
            check('unchanged_source_forward_'+k,source_replay[k],stack_source(1,k))
        check('unchanged_source_ancestry',tf.cast(source_replay['ancestors'],D),stack_source(1,'ancestors'),0.,0.)
        policy=[stack_source(0,k) for k in ('A','b','c')]
        fitted=engine.fit(*policy,stack_source(0,'x'),stack_source(0,'log_incremental_weight'))
        flag('fit_valid',fitted['valid'].numpy())
        for k in ('A','b','c'):
            check('independent_svd_'+k,fitted[k],stack_reference('policy',k))
            tf.io.write_file(str(args.output/('policy_'+k+'.tensor')),tf.io.serialize_tensor(fitted[k]))
        replay=engine.forward(*(fitted[k] for k in ('A','b','c')),y,stack_source(1,'normal'),stack_source(1,'uniform'))
        flag('replay_valid',replay['valid'].numpy())
        for k in ('x','log_incremental_weight','logZ','ess'):
            check('independent_forward_'+k,replay[k],stack_reference('forward',k))
        check('independent_forward_ancestry',tf.cast(replay['ancestors'],D),stack_reference('forward','ancestors'),0.,0.)
        tf.io.write_file(str(args.output/'replay_logZ.tensor'),tf.io.serialize_tensor(replay['logZ']))
        seeds=[20261001,20261002]
        normal=tf.stack([tf.random.stateless_normal([101,512,8],[s,11],dtype=D) for s in seeds],axis=1)
        uniform=tf.stack([tf.random.stateless_uniform([101],[s,12],dtype=D) for s in seeds],axis=1)
        fresh=engine.forward(*(fitted[k] for k in ('A','b','c')),y,normal,uniform)
        flag('fresh_forward_valid',fresh['valid'].numpy())
        tf.io.write_file(str(args.output/'fresh_logZ.tensor'),tf.io.serialize_tensor(fresh['logZ']))
        traces={name:fn.experimental_get_tracing_count() for name,fn in [('fit',engine.fit),('forward',engine.forward),('regress',engine.regress)]}
        for name,count in traces.items():flag(name+'_single_trace',count==1)
        summary=dict(status='PASS' if all(c['passed'] for c in checks) else 'FAIL',scope='LORENZ_EQUIVALENT_SOLVER_REPAIR',replication='INCOMPLETE',
            solver_verdict='CENTERED_QR_VALIDATED_FOR_D8_H101_N512_CONSUMER' if all(c['passed'] for c in checks) else 'NOT_PROMOTED_PARITY_FAILURE',checks=checks,
            independent_reference_solver=reference['solver'],source_normal_equations='NUMERICALLY_UNRESOLVED_DIAGNOSTIC_ONLY',
            recursion='CENTERED_GAUSSIAN_CORRECTION_RATIO',
            minimum_precision_margin=float(tf.reduce_min(fitted['precision_margin']).numpy()),
            minimum_gram_eigenvalue_ratio=float(tf.reduce_min(fitted['gram_eigenvalue_ratio']).numpy()),
            fresh_forward_logZ=fresh['logZ'][-1].numpy().tolist(),fresh_seeds=seeds,
            traced_signatures=traces,gpu_memory_policy=memory,jit_compile=True,tf32=False,dtype='float64',
            tensorflow=tf.__version__,device=tf.config.list_logical_devices('GPU')[0].name,
            cpu_only=False,numpy_runtime=False,pfor=False,python_time_loop=False,
            allocator=tf.config.experimental.get_memory_info('GPU:0'),wall_seconds=time.monotonic()-started,
            nonclaims=['Not Figure6/Table1 replication','No stochastic ranking','No general default or HMC claim'])
    except Exception as exc:
        summary=dict(status='FAIL',error=repr(exc),checks=checks,gpu_memory_policy=memory,wall_seconds=time.monotonic()-started)
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
        raise
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:summary[k] for k in ('status','solver_verdict','fresh_forward_logZ','wall_seconds')}))


if __name__=='__main__':main()
