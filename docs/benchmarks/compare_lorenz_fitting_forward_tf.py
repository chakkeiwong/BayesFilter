"""Diagnostic: existing TF fits followed by the real source-matched forward consumer.

No source solver, policy or numerical protection changes. Host case loops,
reductions and serialization are diagnostic orchestration only.
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


def solver_verdict(rows):
    if not rows:
        return 'NO_COMPARISON'
    if any(not r['fit_valid'] or r.get('forward_valid') is not True for r in rows):
        return 'ADDITIONAL_INVALID_RESULT_ON_SOURCE_VALID_INPUT'
    if any(not r['forward_screen_passed'] or not r['policy_screen_passed'] for r in rows):
        return 'FITTING_DOWNSTREAM_MISMATCH'
    return 'SCOPED_FITTING_FORWARD_SCREEN_PASSED'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','output','reference'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args(); start=time.monotonic()
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
    os.environ['BAYESFILTER_PRELOAD_CUSTOM_OP']='0'
    sys.path.insert(0,str(args.root))
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory=configure_tensorflow_gpu_memory_growth(tf,require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    tf.config.set_soft_device_placement(False)
    sources=args.output/'sources'
    for relative in ('bayesfilter/replication/controlled_smc_tf.py','bayesfilter/runtime/gpu_memory_policy.py'):
        assert sha(args.root/relative)==sha(sources/relative),'Dependency differs from snapshot'
    implementation=sources/'bayesfilter/replication/multivariate_controlled_smc_tf.py'
    spec=importlib.util.spec_from_file_location('frozen_fit_core',implementation)
    core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
    ref=json.loads((args.reference/'fixture.json').read_text())
    assert ref['schema']=='lorenz_forward_reference_v1'
    assert ref['source_results_exactly_unchanged'] and ref['source_rng_exactly_unchanged']
    assert len(ref['cases'])==6 and sum(c['stages']-1 for c in ref['cases'])==16
    dtype=tf.float64
    summary=dict(status='FAIL',scope='MATCHED_AUTHOR_FIT_THEN_FORWARD',replication='INCOMPLETE',
        tensorflow=tf.__version__,jit_compile=True,tf32=False,dtype='float64',
        gpu_memory_policy=memory,source_controls=[],fits=[],trace_counts={},
        input_fixture_sha256=sha(args.reference/'fixture.json'),implementation_sha256=sha(implementation),
        endpoints=['MultivariateGaussianControlledSMC.fit -> _fit -> _regress',
                   'MultivariateGaussianControlledSMC.forward -> _forward -> _cloud_weights'],
        nonclaims=['P1 non-inferiority','solver superiority','autonomous adaptive trajectory parity',
                   'MATLAB runtime equivalence','E0 robustness','full replication'])

    def scalar(value):
        value=float(value.numpy())
        return value if math.isfinite(value) else None

    def load(name):
        item=ref['arrays'][name];path=args.reference/item['file']
        assert sha(path)==item['sha256'],'Corrupt reference '+name
        with tf.device('/CPU:0'):
            value=tf.reshape(tf.io.decode_raw(path.read_bytes(),dtype,little_endian=True),item['shape'])
        return tf.identity(value)

    def compare(actual,expected):
        assert actual.shape==expected.shape,(actual.shape,expected.shape)
        delta=tf.abs(actual-expected)
        scaled=delta/(tf.constant(1e-7,dtype)+tf.constant(1e-9,dtype)*tf.abs(expected))
        axes=list(range(1,actual.shape.rank))
        by_time=tf.reduce_max(scaled,axis=axes) if axes else scaled
        times=by_time.numpy().tolist()
        bad=[i for i,v in enumerate(times) if not math.isfinite(v) or v>1]
        return dict(passed=not bad,max_absolute_error=scalar(tf.reduce_max(delta)),
                    max_scaled_error=scalar(tf.reduce_max(scaled)),
                    first_differing_time=min(bad) if bad else None,atol=1e-7,rtol=1e-9)

    def forward_comparison(result,values):
        comparisons={k:compare(result[k][:,0],values[v]) for k,v in
                     (('x','x'),('log_incremental_weight','lw'),('logZ','logZ'),('ess','ess'))}
        mismatch=tf.reduce_any(result['ancestors'][:,0]!=tf.cast(values['ancestors'],tf.int32),axis=-1).numpy().tolist()
        valid=bool(result['valid'].numpy())
        return dict(forward_valid=valid,comparisons=comparisons,ancestry_exact=not any(mismatch),
                    first_ancestry_difference=next((i for i,v in enumerate(mismatch) if v),None),
                    final_logZ_delta=scalar(result['logZ'][-1,0]-values['logZ'][-1]),
                    forward_screen_passed=valid and not any(mismatch) and all(c['passed'] for c in comparisons.values()))

    def save(prefix,result,keys=None):
        with tf.device('/CPU:0'):
            for key,value in result.items():
                if keys is None or key in keys:
                    tf.io.write_file(str(args.output/(prefix+key+'.tensor')),tf.io.serialize_tensor(value))

    def engine(forcing,solver):
        covariance=lambda value:[[value if i==j else 0. for j in range(8)] for i in range(8)]
        return core.MultivariateGaussianControlledSMC(observations=101,particles=512,replicas=1,
            dimension=8,observation_dimension=6,initial_mean=[0.]*8,
            initial_covariance=covariance(.01),transition_covariance=covariance(.001),
            transition_mean=lambda x:core.lorenz_rk4(x,forcing.read_value()),
            observation_log_density=lambda y,x:-.5/1e-4*tf.reduce_sum((y-x[...,:6])**2,-1),
            solver=solver,jit_compile=True,precision_policy='none',first_failure_guards=False)

    try:
        with tf.device('/GPU:0'):
            solvers=('source_normal_equations','centered_qr')
            forcing_variable=tf.Variable(2.5,dtype=dtype,trainable=False)
            engines={s:engine(forcing_variable,s) for s in solvers}
            summary["graph_instance_count"]=len(engines)
            summary["forcing_policy"]="shared variable assigned once per case before graph calls"
            # This diagnostic consumer evaluates policies; the fit/forward work uses shared endpoints.
            evaluate=tf.function(core.quadratic,input_signature=[
                tf.TensorSpec([101,1,512,8],dtype),tf.TensorSpec([101,1,8,8],dtype),
                tf.TensorSpec([101,1,8],dtype),tf.TensorSpec([101,1],dtype)],jit_compile=True,autograph=False)
            for case in ref['cases']:
                case_id=case['case'];forcing=case['forcing'];y=load(f'case{case_id}_y')[:,None]
                forcing_variable.assign(tf.constant(forcing,dtype))
                for stage in range(case['stages']-1):
                    prefix=f'case{case_id}_stage{stage}_'
                    next_prefix=f'case{case_id}_stage{stage+1}_'
                    old={k:load(prefix+k)[:,None] for k in ('A','b','c','x','lw')}
                    values={k:load(next_prefix+k) for k in ('A','b','c','x','lw','logZ','ess','ancestors','normal','uniform')}
                    reference_policy=[values[k][:,None] for k in ('A','b','c')]
                    control=engines[solvers[0]].forward(*reference_policy,y,values['normal'][:,None],values['uniform'][:,None])
                    control_row=dict(case=case_id,stage=stage+1,**forward_comparison(control,values))
                    summary['source_controls'].append(control_row)
                    assert control_row['forward_screen_passed'],'Source policy forward control failed'
                    for solver in solvers:
                        current=engines[solver]
                        fit=current.fit(*[old[k] for k in ('A','b','c','x','lw')])
                        assert 'GPU:0' in fit['A'].device,'Fitting not on GPU'
                        row=dict(case=case_id,stage=stage+1,forcing=forcing,seed=case['seed'],solver=solver,
                                 fit_valid=bool(fit['valid'].numpy()),device=fit['A'].device,
                                 minimum_precision_margin=scalar(tf.reduce_min(fit['precision_margin'])),
                                 residual_sum_squares=scalar(tf.reduce_sum(fit['residual_sum_squares'])),
                                 coefficient_comparisons={k:compare(fit[k],reference_policy[i]) for i,k in enumerate(('A','b','c'))})
                        candidate_policy=[fit[k] for k in ('A','b','c')]
                        row['policy_comparisons']={}
                        for label,cloud in (('source_fit_cloud',old['x']),('next_source_cloud',values['x'][:,None])):
                            row['policy_comparisons'][label]=compare(evaluate(cloud,*candidate_policy),evaluate(cloud,*reference_policy))
                        row['policy_screen_passed']=all(v['passed'] for v in row['policy_comparisons'].values())
                        save(next_prefix+solver+'_fit_',fit)
                        if row['fit_valid']:
                            result=current.forward(*candidate_policy,y,values['normal'][:,None],values['uniform'][:,None])
                            assert 'GPU:0' in result['x'].device,'Forward not on GPU'
                            row.update(forward_comparison(result,values))
                            save(next_prefix+solver+'_forward_',result,('x','logZ','ess','ancestors','valid','log_incremental_weight'))
                        else:
                            row.update(forward_valid=None,forward_screen_passed=False,
                                       forward_disposition='Not executed: fitted Gaussian policy rejected')
                        summary['fits'].append(row)
                        (args.output/'progress.json').write_text(json.dumps(dict(fits=len(summary['fits']),last=row),indent=2,allow_nan=False)+'\n')
            for solver,current in engines.items():
                for label in ('fit','forward'):
                    count=getattr(current,label).experimental_get_tracing_count()
                    summary['trace_counts'][f'{solver}_{label}']=count
                    # A candidate rejected for every fit has no numerical forward calls.
                    assert count<=1,'Unexpected retracing'
                assert current.fit.experimental_get_tracing_count()==1,'Fitting consumer never called'
            summary['trace_counts']['policy_evaluation']=evaluate.experimental_get_tracing_count()
            assert summary['trace_counts']['policy_evaluation']==1
            assert len(summary['fits'])==32 and len(summary['source_controls'])==16
            summary['solver_verdicts']={s:solver_verdict([r for r in summary['fits'] if r['solver']==s]) for s in solvers}
            summary['fitting_forward_screen_passed']=all(v=='SCOPED_FITTING_FORWARD_SCREEN_PASSED' for v in summary['solver_verdicts'].values())
            summary['comparison_verdict']='FITTING_COMPARISON_ACCOUNTED_FOR'
            summary['gpu_allocator_bytes']=tf.config.experimental.get_memory_info('GPU:0')
            summary['status']='PASS'
    except Exception as exc:
        summary['error']=repr(exc)
        raise
    finally:
        summary['algorithm_seconds']=time.monotonic()-start
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
        (args.output/'result.md').write_text('# Fitting followed by the forward consumer\n\n'
            'Execution: '+summary['status']+'. Replication remains incomplete.\n\n'
            +json.dumps(summary.get('solver_verdicts',{}),indent=2)+'\n\n'
            'Policy and consumer outputs are saved for localization. See the reviewed plan for diagnostic roles.\n')
    print(json.dumps(dict(status=summary['status'],solver_verdicts=summary['solver_verdicts'],seconds=summary['algorithm_seconds'])))


if __name__=='__main__':
    main()
