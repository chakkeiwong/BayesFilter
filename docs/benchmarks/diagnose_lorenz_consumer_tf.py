"""Bounded diagnostic of full-covariance source replay and quadratic fitting."""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
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
D=tf.float64
spec=importlib.util.spec_from_file_location('multivariate_snapshot',Path(__file__).with_name('multivariate_controlled_smc_tf.py'))
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)


def scalar(x):return float(x.numpy())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fixture',type=Path,required=True)
    args=parser.parse_args();started=time.monotonic()
    fixture=json.loads(args.fixture.read_text());checks=[];fitting=[];traces={}
    def compare(name,actual,expected,atol=1e-7,rtol=1e-9,required=True):
        actual=tf.cast(tf.convert_to_tensor(actual),D)
        expected=tf.cast(tf.convert_to_tensor(expected),D)
        delta=tf.abs(actual-expected);bound=atol+rtol*tf.abs(expected)
        passed=bool(tf.reduce_all(tf.math.is_finite(actual)&(delta<=bound)).numpy())
        item=dict(name=name,passed=passed,required=required,
                  max_absolute_error=scalar(tf.reduce_max(delta)),atol=atol,rtol=rtol)
        checks.append(item)
        if required and not passed:raise RuntimeError(name+' failed: '+str(item))
        return passed
    def flag(name,condition):compare(name,float(bool(condition)),1.,0.,0.)
    def observe_fit(name,result):
        return dict(name=name,valid=bool(result['valid'].numpy()),
                    minimum_precision_margin=scalar(tf.reduce_min(result['precision_margin'])),
                    minimum_gram_eigenvalue_ratio=scalar(tf.reduce_min(result['gram_eigenvalue_ratio'])),
                    residual_sum_squares=scalar(tf.reduce_sum(result['residual_sum_squares'])))
    def save_tensor(name,value):
        tf.io.write_file(str(args.output/(name+'.tensor')),tf.io.serialize_tensor(value))
    def engine(solver,normalized=False,replicas=2):
        constant=-.5*6*math.log(2*math.pi*1e-4) if normalized else 0.
        return core.MultivariateGaussianControlledSMC(observations=101,particles=512,
            replicas=replicas,dimension=8,observation_dimension=6,initial_mean=[0.]*8,
            initial_covariance=fixture['initial_covariance'],transition_covariance=fixture['transition_covariance'],
            transition_mean=lambda x:core.lorenz_rk4(x,tf.constant(fixture['forcing'],D)),
            observation_log_density=lambda y,x:-.5/1e-4*tf.reduce_sum((y-x[...,:6])**2,-1)+constant,
            solver=solver,jit_compile=True)
    try:
        # Independent Gaussian conditioning identity with a full observation covariance.
        base=tf.constant([[1.,.2],[.2,2.]],D);obs=tf.constant([[.7,-.1],[-.1,.5]],D)
        mean=tf.constant([[[.5,-.7]]],D);measurement=tf.constant([.1,.8],D)
        op=tf.linalg.inv(obs);a=.5*op[None,:,:];b=-tf.linalg.matvec(op,measurement)[None,:]
        constant=(.5*tf.tensordot(measurement,tf.linalg.matvec(op,measurement),1)
                  +.5*(2*math.log(2*math.pi)+tf.linalg.slogdet(obs)[1]))[None]
        twist=tf.function(core.multivariate_twist,input_signature=[
            tf.TensorSpec([1,1,2],D),tf.TensorSpec([2,2],D),tf.TensorSpec([],D),
            tf.TensorSpec([1,2,2],D),tf.TensorSpec([1,2],D),tf.TensorSpec([1],D)],jit_compile=True,autograph=False)
        cm,cov,li,margin,valid=twist(mean,tf.linalg.inv(base),tf.linalg.slogdet(base)[1],a,b,constant)
        total=base+obs;delta=measurement-mean[0,0]
        exact_mean=mean[0,0]+tf.linalg.matvec(base,tf.linalg.solve(total,delta[:,None])[:,0])
        exact_cov=base-tf.linalg.matmul(base,tf.linalg.solve(total,base))
        exact_log=-.5*(2*math.log(2*math.pi)+tf.linalg.slogdet(total)[1]+tf.tensordot(delta,tf.linalg.solve(total,delta[:,None])[:,0],1))
        compare('full_gaussian_mean',cm[0,0],exact_mean,1e-12,1e-12)
        compare('full_gaussian_covariance',cov[0],exact_cov,1e-12,1e-12)
        compare('full_gaussian_integral',li[0,0],exact_log,1e-12,1e-12)
        flag('full_gaussian_valid',tf.reduce_all(valid).numpy())
        checked=tf.function(core.covariance_from_precision,input_signature=[tf.TensorSpec([2,2],D)],jit_compile=True,autograph=False)
        flag('negative_precision_rejected',not bool(checked(tf.constant([[1.,0.],[0.,-1.]],D))[3].numpy()))
        flag('asymmetric_precision_rejected',not bool(checked(tf.constant([[1.,.1],[0.,1.]],D))[3].numpy()))
        flag('singular_precision_rejected',not bool(checked(tf.constant([[1.,0.],[0.,0.]],D))[3].numpy()))
        raw=engine('source_normal_equations');qr=engine('centered_qr')
        healthy=tf.random.stateless_normal([2,512,8],[20260929,81],dtype=D)
        true_a=tf.eye(8,batch_shape=[2],dtype=D)*.2+tf.ones([2,8,8],D)*.01
        true_b=tf.reshape(tf.range(16,dtype=D),[2,8])/100
        true_c=tf.constant([.3,-.7],D)
        truth=core.quadratic(healthy,true_a,true_b,true_c)
        for name,eng in [('normal_equations',raw),('centered_qr',qr)]:
            fitted=eng.regress(healthy,truth);flag('healthy_'+name+'_valid',fitted['valid'].numpy())
            for key,target in [('A',true_a),('b',true_b),('c',true_c)]:
                compare('healthy_'+name+'_'+key,fitted[key],target,1e-10,1e-10)
        # Recover exact author source consumers from their own stored clouds first.
        source_rounds=[]
        for stage in range(2):
            def stacked(key):return tf.stack([tf.constant(rep['rounds'][stage][key],D) for rep in fixture['replicates']],axis=1)
            source_rounds.append({key:stacked(key) for key in ('A','b','c','x','log_incremental_weight','logZ','ess','normal','uniform','transition_means','ancestors')})
        y=tf.tile(tf.constant(fixture['y'],D)[:,None,:],[1,2,1])
        rk4=tf.function(lambda x:core.lorenz_rk4(x,tf.constant(fixture['forcing'],D)),
            input_signature=[tf.TensorSpec([101,2,512,8],D)],jit_compile=True,autograph=False)
        for stage,source in enumerate(source_rounds):
            tag='stage'+str(stage)
            compare(tag+'_RK4',rk4(source['x']),source['transition_means'])
            policy=[source[k] for k in ('A','b','c')]
            weighted=raw.cloud_weights(*policy,y,source['x'])
            flag(tag+'_cloud_valid',weighted['valid'].numpy())
            for key in ('log_incremental_weight','logZ','ess'):
                compare(tag+'_cloud_'+key,weighted[key],source[key])
            replay=raw.forward(*policy,y,source['normal'],source['uniform'])
            flag(tag+'_replay_valid',replay['valid'].numpy())
            for key in ('x','log_incremental_weight','logZ','ess'):
                compare(tag+'_replay_'+key,replay[key],source[key])
            compare(tag+'_ancestry',replay['ancestors'],source['ancestors'],0.,0.)
            save_tensor(tag+'_replay_logZ',replay['logZ'])
        normalized=engine('source_normal_equations',normalized=True)
        source=source_rounds[0];policy=[source[k] for k in ('A','b','c')]
        normalized_result=normalized.cloud_weights(*policy,y,source['x'])
        constant=-.5*6*math.log(2*math.pi*1e-4)
        compare('normalized_prefix_identity',normalized_result['logZ'],source['logZ']+tf.range(1,102,dtype=D)[:,None]*constant)
        compare('normalized_weight_identity',normalized_result['log_incremental_weight'],source['log_incremental_weight']+constant)
        # Diagnose, without selecting on likelihood, both equivalent LS solvers.
        fitted_raw=raw.fit(*policy,source['x'],source['log_incremental_weight'])
        fitted_qr=qr.fit(*policy,source['x'],source['log_incremental_weight'])
        for name,result in [('source_normal_equations',fitted_raw),('centered_qr',fitted_qr)]:
            fitting.append(observe_fit(name,result))
            for key,value in result.items():save_tensor(name+'_'+key,value)
        next_policy=source_rounds[1]
        source_fit_agreement=all([compare('source_fit_'+key,fitted_raw[key],next_policy[key],required=False) for key in ('A','b','c')])
        # Exact common targets use the author's next policy, avoiding recursive
        # differences between solvers as a confounder of residual comparison.
        x=source['x'];means=raw.f(x[:-1])
        next_new=core.multivariate_twist(means,raw.p,raw.ldq,next_policy['A'][1:],next_policy['b'][1:],next_policy['c'][1:])[2]
        next_old=core.multivariate_twist(means,raw.p,raw.ldq,source['A'][1:],source['b'][1:],source['c'][1:])[2]
        correction=tf.concat([next_new-next_old,tf.zeros([1,2,512],D)],0)
        target=-source['log_incremental_weight']-correction
        # One batch contains every time/seed cloud; no per-row Python target calls.
        common_raw=engine('source_normal_equations',replicas=202)
        common_qr=engine('centered_qr',replicas=202)
        xx=tf.reshape(x,[202,512,8]);yy=tf.reshape(target,[202,512])
        rr=common_raw.regress(xx,yy);qq=common_qr.regress(xx,yy)
        delta=qq['residual_sum_squares']-rr['residual_sum_squares']
        tolerance=1e-8*tf.maximum(rr['residual_sum_squares'],tf.constant(1e-12,D))
        nonharm=bool(tf.reduce_all(delta<=tolerance).numpy())
        fitted_difference=tf.reduce_max(tf.abs(core.quadratic(xx,rr['A'],rr['b'],rr['c'])-core.quadratic(xx,qq['A'],qq['b'],qq['c'])))
        common=dict(valid_raw=bool(rr['valid'].numpy()),valid_qr=bool(qq['valid'].numpy()),
            residual_nonincrease=nonharm,max_residual_increase=scalar(tf.reduce_max(delta)),
            maximum_fitted_value_difference=scalar(fitted_difference),
            minimum_raw_gram_eigenvalue_ratio=scalar(tf.reduce_min(rr['gram_eigenvalue_ratio'])),
            minimum_qr_gram_eigenvalue_ratio=scalar(tf.reduce_min(qq['gram_eigenvalue_ratio'])),
            residual_sum_squares_raw=scalar(tf.reduce_sum(rr['residual_sum_squares'])),
            residual_sum_squares_qr=scalar(tf.reduce_sum(qq['residual_sum_squares'])))
        for name,result in [('common_target_normal',rr),('common_target_qr',qq)]:
            for key,value in result.items():save_tensor(name+'_'+key,value)
        for name,fn in [('gaussian',twist),('precision_guard',checked),('rk4',rk4),('forward',raw.forward),
            ('weights',raw.cloud_weights),('normalized_weights',normalized.cloud_weights),
            ('raw_fit',raw.fit),('qr_fit',qr.fit),('raw_regress',raw.regress),('qr_regress',qr.regress),
            ('common_raw',common_raw.regress),('common_qr',common_qr.regress)]:
            traces[name]=fn.experimental_get_tracing_count();flag(name+'_one_trace',traces[name]==1)
        fit_status='SOURCE_FIT_PARITY_PASS' if source_fit_agreement and fitting[0]['valid'] else 'SOURCE_FIT_NUMERICALLY_UNRESOLVED'
        summary=dict(status='PASS',scope='MULTIVARIATE_SOURCE_FORWARD_AND_FITTING_DIAGNOSTIC',
            forward_parity='PASS',fit_status=fit_status,source_fit_agreement=source_fit_agreement,
            solver_promotion='NOT_PROMOTED',replication='INCOMPLETE',checks=checks,
            required_checks=sum(c['required'] for c in checks),fitting=fitting,common_target=common,
            traced_signatures=traces,gpu_memory_policy=memory,
            dtype='float64',jit_compile=True,tf32=False,device=tf.config.list_logical_devices('GPU')[0].name,
            python_time_loop=False,pfor=False,numpy_runtime=False,
            cpu_only=False,source_fixture=str(args.fixture),seeds=[20260929,20260930],
            tensorflow=tf.__version__,wall_seconds=time.monotonic()-started,
            allocator=tf.config.experimental.get_memory_info('GPU:0'),host_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            nonclaims=['Not Figure6 or Table1 replication','No empirical ranking','No default promotion','No HMC claim'])
    except Exception as exc:
        summary=dict(status='FAIL',error=repr(exc),checks=checks,fitting=fitting,
                     gpu_memory_policy=memory,wall_seconds=time.monotonic()-started)
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
        raise
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:summary[k] for k in ('status','required_checks','fit_status','common_target','wall_seconds')}))


if __name__=='__main__':main()
