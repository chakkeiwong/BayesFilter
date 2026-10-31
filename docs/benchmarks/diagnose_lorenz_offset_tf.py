"""Diagnostic GPU/XLA calls to the actual regression and forward endpoints.

Policy swaps isolate the effect of source solver branches. They are diagnostic
interventions, not a new controlled-SMC algorithm or a qualified runtime.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','output','reference','capture','conditional'):
        parser.add_argument('--'+name,required=True,type=Path)
    args=parser.parse_args();start=time.monotonic()
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
    sys.path.insert(0,str(args.root))
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    # Preserve the GPU model used in the earlier comparison while another
    # workload may be using the other physical card.
    physical=tf.config.list_physical_devices('GPU')
    details=[dict(name=d.name,**tf.config.experimental.get_device_details(d)) for d in physical]
    selected=[d for d,info in zip(physical,details) if '4080 SUPER' in info.get('device_name','')]
    assert len(selected)==1,details
    tf.config.set_visible_devices(selected,'GPU')
    memory=configure_tensorflow_gpu_memory_growth(tf,require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    tf.config.set_soft_device_placement(False)
    assert tf.config.list_logical_devices('GPU')
    dtype=tf.float64
    implementation=args.output/'sources/bayesfilter/replication/multivariate_controlled_smc_tf.py'
    old=args.root/'docs/plans/artifacts/controlled-smc-kdm-replication-20260928/attempt154_p0_perturb_tf/sources/bayesfilter/replication/multivariate_controlled_smc_tf.py'
    assert sha(implementation)==sha(old),'Numerical core changed since perturbation'
    for rel in ('bayesfilter/replication/controlled_smc_tf.py','bayesfilter/runtime/gpu_memory_policy.py'):
        assert sha(args.root/rel)==sha(args.output/'sources'/rel)
    spec=importlib.util.spec_from_file_location('frozen_offset_core',implementation)
    core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
    reference=json.loads((args.reference/'fixture.json').read_text())
    capture=json.loads((args.capture/'fixture.json').read_text())
    conditional=json.loads((args.conditional/'manifest.json').read_text())
    assert reference['source_coefficients_exact']==101
    hashes={}

    def load(directory,index,key):
        item=index['arrays'][key];path=directory/item['file']
        assert sha(path)==item['sha256'],str(path)
        hashes[str(path)]=item['sha256']
        with tf.device('/CPU:0'):
            return tf.reshape(tf.io.decode_raw(tf.io.read_file(str(path)),dtype,little_endian=True),item['shape'])

    def saved(name):
        path=args.conditional/(name+'.tensor')
        assert sha(path)==conditional['output_sha256'][path.name]
        hashes[str(path)]=sha(path)
        with tf.device('/CPU:0'):
            return tf.io.parse_tensor(tf.io.read_file(str(path)),dtype)

    def save(name,value):
        with tf.device('/CPU:0'):
            tf.io.write_file(str(args.output/(name+'.tensor')),tf.io.serialize_tensor(value))

    source_x=load(args.reference,reference,'x')
    target=load(args.reference,reference,'target')
    beta=load(args.reference,reference,'beta')
    reference_prediction=load(args.reference,reference,'qr_prediction')
    source_svd=load(args.reference,reference,'svd_mask')>0
    base=[load(args.capture,capture,'case3_stage0_'+k)[:,None] for k in ('A','b','c')]
    source_policy=[load(args.capture,capture,'case3_stage1_'+k)[:,None] for k in ('A','b','c')]
    qr_policy=[saved('case3_stage1_centered_qr_fit_'+k) for k in ('A','b','c')]
    inputs=[load(args.capture,capture,'case3_'+k)[:,None] if k=='y' else
            load(args.capture,capture,'case3_stage1_'+k)[:,None] for k in ('y','normal','uniform')]
    source_logz=load(args.capture,capture,'case3_stage1_logZ')
    expected_qr_logz=saved('case3_stage1_centered_qr_forward_logZ')[-1,0]
    source_ancestors=tf.cast(load(args.capture,capture,'case3_stage1_ancestors'),tf.int32)
    assert bool(tf.reduce_all(source_x==load(args.capture,capture,'case3_stage0_x')).numpy())
    covariance=lambda v:[[v if i==j else 0. for j in range(8)] for i in range(8)]
    engines={}
    rows=[];fits={};forward_rows=[]
    with tf.device('/GPU:0'):
        for solver in ('source_normal_equations','centered_qr'):
            engine=core.MultivariateGaussianControlledSMC(observations=101,particles=512,replicas=1,
                dimension=8,observation_dimension=6,initial_mean=[0.]*8,
                initial_covariance=covariance(.01),transition_covariance=covariance(.001),
                transition_mean=lambda x:core.lorenz_rk4(x,tf.constant(4.8801,dtype)),
                observation_log_density=lambda y,x:-.5/1e-4*tf.reduce_sum((y-x[...,:6])**2,-1),
                solver=solver,jit_compile=True,precision_policy='none',first_failure_guards=False)
            engines[solver]=engine;values={k:[] for k in ('A','b','c')}
            for t in range(101):
                result=engine.regress(source_x[t:t+1],target[t:t+1])
                assert 'GPU:0' in result['A'].device
                assert bool(result['valid'].numpy()),(solver,t)
                for k in values:values[k].append(result[k])
                prediction=core.quadratic(source_x[t:t+1],result['A'],result['b'],result['c'])[0]
                source_prediction=tf.linalg.matvec(engine._design(source_x[t]),beta[t])
                delta=prediction-source_prediction
                rows.append(dict(solver=solver,time=t,source_svd=bool(source_svd[t].numpy()),
                    prediction_difference_max=float(tf.reduce_max(tf.abs(delta)).numpy()),
                    prediction_difference_rms=float(tf.sqrt(tf.reduce_mean(delta**2)).numpy()),
                    independent_qr_prediction_difference_max=float(tf.reduce_max(tf.abs(prediction-reference_prediction[t])).numpy()),
                    residual_rms=float(tf.sqrt(tf.reduce_mean((prediction-target[t])**2)).numpy())))
            fits[solver]=[tf.stack(values[k]) for k in ('A','b','c')]
            for k,value in zip(('A','b','c'),fits[solver]):save(solver+'_direct_'+k,value)
            print(json.dumps(dict(progress='direct_regressions_complete',solver=solver,seconds=time.monotonic()-start)),flush=True)
        engine=engines['source_normal_equations']

        def swap(original,replacement,mask):
            return [tf.where(tf.reshape(mask,[101]+[1]*(len(a.shape)-1)),b,a) for a,b in zip(original,replacement)]

        at28=tf.range(101)==28
        candidates=dict(source_policy=source_policy,conditional_qr=qr_policy,
            source_with_qr_at_svd_times=swap(source_policy,qr_policy,source_svd),
            qr_with_source_at_svd_times=swap(qr_policy,source_policy,source_svd),
            source_with_qr_at_time28=swap(source_policy,qr_policy,at28),
            qr_with_source_at_time28=swap(qr_policy,source_policy,at28))
        for name,policy in candidates.items():
            result=engine.forward(*policy,*inputs)
            valid=bool(result['valid'].numpy())
            assert valid,name
            delta=float((result['logZ'][-1,0]-source_logz[-1]).numpy())
            row=dict(name=name,valid=valid,final_logZ=float(result['logZ'][-1,0].numpy()),
                delta_logZ=delta,likelihood_percent=100*math.expm1(delta),
                ancestry_mismatches=int(tf.reduce_sum(tf.cast(result['ancestors'][:,0]!=source_ancestors,tf.int32)).numpy()),
                minimum_ess=float(tf.reduce_min(result['ess']).numpy()))
            forward_rows.append(row)
            for k in ('logZ','ess','ancestors','log_incremental_weight'):save(name+'_'+k,result[k])
            print(json.dumps(dict(progress='forward_complete',**row)),flush=True)
        assert abs(forward_rows[0]['delta_logZ'])<1e-6,'Author-policy control failed'
        assert forward_rows[0]['ancestry_mismatches']==0,'Author ancestry control failed'
        assert abs(forward_rows[1]['final_logZ']-float(expected_qr_logz.numpy()))<1e-6,'Saved QR control failed'
    qr_error=max(r['independent_qr_prediction_difference_max'] for r in rows if r['solver']=='centered_qr')
    trace_counts={s:dict(regress=e.regress.experimental_get_tracing_count(),forward=e.forward.experimental_get_tracing_count()) for s,e in engines.items()}
    summary=dict(status='PASS',scope='DIRECT_SOURCE_TARGET_REGRESSION_AND_POLICY_SWAP_DIAGNOSTIC',replication='INCOMPLETE',
        scientific_verdict='DIRECT_SOLVER_DIFFERENCE_CONFIRMED' if qr_error<=1e-7 else 'DIRECT_QR_REFERENCE_MISMATCH',
        diagnostic_only=True,author_method_repair_in_scope=False,source_coefficients_exact=101,
        direct_fit_rows=rows,forward_rows=forward_rows,independent_qr_prediction_max_error=qr_error,
        input_sha256=hashes,seed=20260930,case=3,forcing=4.8801,
        gpu_memory_policy=memory,gpu_devices=details,selected_physical_device=selected[0].name,
        tensorflow=tf.__version__,device='GPU:0',jit_compile=True,tf32=False,dtype='float64',trace_counts=trace_counts,
        allocator=tf.config.experimental.get_memory_info('GPU:0'),algorithm_seconds=time.monotonic()-start,
        controls_passed=True,statistical_ranking='not tested',default_readiness='not established')
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:summary[k] for k in ('status','scientific_verdict','independent_qr_prediction_max_error','algorithm_seconds','trace_counts')}),flush=True)


if __name__=='__main__':main()
