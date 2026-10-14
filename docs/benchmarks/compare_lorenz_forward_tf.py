"""Diagnostic GPU/XLA replay at the existing forward consumer, using author draws.

No fitting or numerical protection is added. Host loops enumerate fixed cases;
TensorFlow reporting reductions outside the kernel are diagnostic only.
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


def comparison_verdict(stages, failure):
    if not stages or any(not s['valid'] for s in stages):
        return 'EXTRA_PORT_INVALID_RESULT'
    if failure.get('precision_rejected') is not True:
        return 'SOURCE_FAILURE_NOT_REJECTED'
    if failure.get('forward_rejected') is not True:
        return 'FAILED_POLICY_FORWARD_NOT_CONFIRMED'
    if not all(s['value_screen_passed'] and s['ancestry_exact'] for s in stages):
        return 'FORWARD_MISMATCH_REQUIRES_LOCALIZATION'
    return 'FIXED_POLICY_FORWARD_SCREEN_PASSED'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--failure',type=Path,required=True)
    args=parser.parse_args();start=time.monotonic()
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
        assert sha(args.root/relative)==sha(sources/relative),'Live dependency differs from snapshot'
    spec=importlib.util.spec_from_file_location('p0_frozen_core',sources/'bayesfilter/replication/multivariate_controlled_smc_tf.py')
    core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
    ref=json.loads((args.reference/'fixture.json').read_text())
    failed=json.loads((args.failure/'failed-policy.json').read_text())
    assert ref['schema']=='lorenz_forward_reference_v1' and failed['schema']=='lorenz_failed_policy_v1'
    assert ref['source_results_exactly_unchanged'] and ref['source_rng_exactly_unchanged']
    assert failed['source_failure_reproduced']
    assert len(ref['cases'])==6 and sum(c['stages'] for c in ref['cases'])==22
    dtype=tf.float64
    summary=dict(status='FAIL',scope='MATCHED_AUTHOR_FIXED_POLICY_FORWARD',replication='INCOMPLETE',
        tensorflow=tf.__version__,jit_compile=True,tf32=False,dtype='float64',gpu_memory_policy=memory,
        source_fixture_sha256=sha(args.reference/'fixture.json'),failure_fixture_sha256=sha(args.failure/'failed-policy.json'),
        implementation_sha256=sha(sources/'bayesfilter/replication/multivariate_controlled_smc_tf.py'),
        call_chain='MultivariateGaussianControlledSMC.forward -> _forward -> multivariate_twist / systematic_ancestors -> _cloud_weights',
        source_runtime_covariance_rule=ref['sampler_covariance_rule'],port_covariance_rule='Unmodified existing Cholesky sampler',
        author_failure_is_replication_veto=False,author_method_repair_in_scope=False,
        stages=[],failure={},trace_counts={},
        nonclaims=['fitting parity','P1 no-worse stochastic qualification','full Figure6 replication',
                   'MATLAB equivalence','matched failing random trajectory','whole replication completion'])
    def load(index,root,name):
        item=index['arrays'][name];path=root/item['file']
        assert sha(path)==item['sha256'],'Corrupt reference array '+name
        with tf.device('/CPU:0'):
            value=tf.reshape(tf.io.decode_raw(path.read_bytes(),dtype,little_endian=True),item['shape'])
        return tf.identity(value)
    def compare(actual,expected):
        assert actual.shape==expected.shape,(actual.shape,expected.shape)
        delta=tf.abs(actual-expected)
        scaled=delta/(tf.constant(1e-7,dtype)+tf.constant(1e-9,dtype)*tf.abs(expected))
        finite=bool(tf.reduce_all(tf.math.is_finite(actual)).numpy())
        axes=list(range(1,actual.shape.rank))
        per_time=tf.reduce_max(scaled,axis=axes) if axes else scaled
        times=per_time.numpy().tolist()
        bad=[i for i,v in enumerate(times) if not math.isfinite(v) or v>1]
        maximum=float(tf.reduce_max(delta).numpy())
        return dict(passed=finite and not bad,finite=finite,first_differing_time=min(bad) if bad else None,
                    max_absolute_error=maximum if math.isfinite(maximum) else None,atol=1e-7,rtol=1e-9)
    def engine(forcing):
        covariance=lambda value:[[value if i==j else 0. for j in range(8)] for i in range(8)]
        return core.MultivariateGaussianControlledSMC(observations=101,particles=512,replicas=1,
            dimension=8,observation_dimension=6,initial_mean=[0.]*8,
            initial_covariance=covariance(.01),transition_covariance=covariance(.001),
            transition_mean=lambda x:core.lorenz_rk4(x,tf.constant(forcing,dtype)),
            observation_log_density=lambda y,x:-.5/1e-4*tf.reduce_sum((y-x[...,:6])**2,-1),
            solver='source_normal_equations',jit_compile=True,precision_policy='none')
    try:
        with tf.device('/GPU:0'):
            engines={f:engine(f) for f in (2.5,4.8801,8.5)}
            for case in ref['cases']:
                y=load(ref,args.reference,f"case{case['case']}_y")[:,None,:]
                endpoint=engines[case['forcing']].forward
                for stage in range(case['stages']):
                    prefix=f"case{case['case']}_stage{stage}_"
                    values={k:load(ref,args.reference,prefix+k) for k in
                            ('A','b','c','x','ancestors','lw','logZ','ess','normal','uniform')}
                    result=endpoint(values['A'][:,None],values['b'][:,None],values['c'][:,None],y,
                                    values['normal'][:,None],values['uniform'][:,None])
                    assert 'GPU:0' in result['x'].device,'Forward computation is not on GPU'
                    row=dict(case=case['case'],stage=stage,forcing=case['forcing'],seed=case['seed'],
                             valid=bool(result['valid'].numpy()),device=result['x'].device,comparisons={})
                    for key,refkey in (('x','x'),('log_incremental_weight','lw'),('logZ','logZ'),('ess','ess')):
                        row['comparisons'][key]=compare(result[key][:,0],values[refkey])
                    ancestral=result['ancestors'][:,0]!=tf.cast(values['ancestors'],tf.int32)
                    mismatch=tf.reduce_any(ancestral,axis=-1).numpy().tolist()
                    row['ancestry_exact']=not any(mismatch)
                    row['first_ancestry_difference']=next((i for i,v in enumerate(mismatch) if v),None)
                    row['value_screen_passed']=all(c['passed'] for c in row['comparisons'].values())
                    summary['stages'].append(row)
                    with tf.device('/CPU:0'):
                        for key in ('x','ancestors','log_incremental_weight','logZ','ess','valid'):
                            tf.io.write_file(str(args.output/(prefix+key+'.tensor')),tf.io.serialize_tensor(result[key]))
                    (args.output/'progress.json').write_text(json.dumps(dict(stages=len(summary['stages']),last_stage=row),indent=2)+'\n')
            covariance_endpoint=tf.function(core.covariance_from_precision,
                input_signature=[tf.TensorSpec([8,8],dtype)],jit_compile=True,autograph=False)
            cov=covariance_endpoint(load(failed,args.failure,'precision'))
            failure=dict(precision_rejected=not bool(cov[3].numpy()),forward_rejected=None,
                         source_time_index=failed['time_index'],policy_real_finite=failed['policy_real_finite'],
                         role='Same-policy domain rejection, not matched failing trajectory')
            if failed['policy_real_finite']:
                endpoint=engine(7.884615384615385).forward
                result=endpoint(*[load(failed,args.failure,k)[:,None] for k in ('A','b','c')],
                                load(ref,args.reference,'case0_y')[:,None,:],
                                tf.zeros([101,1,512,8],dtype),tf.fill([101,1],tf.constant(.5,dtype)))
                failure['forward_rejected']=not bool(result['valid'].numpy())
                summary['trace_counts']['failed_policy_forward']=endpoint.experimental_get_tracing_count()
            else:
                failure['limitation']='Original complex/nonfinite policy preserved; no real-valued substitution allowed'
            summary['failure']=failure
            summary['trace_counts'].update({str(f):e.forward.experimental_get_tracing_count() for f,e in engines.items()})
            summary['trace_counts']['covariance']=covariance_endpoint.experimental_get_tracing_count()
            assert all(c==1 for c in summary['trace_counts'].values()),'Unexpected retracing'
            summary['gpu_allocator_bytes']=tf.config.experimental.get_memory_info('GPU:0')
        summary['comparison_verdict']=comparison_verdict(summary['stages'],summary['failure'])
        summary['forward_screen_passed']=summary['comparison_verdict']=='FIXED_POLICY_FORWARD_SCREEN_PASSED'
        summary['status']='PASS'  # Completed diagnostic; comparison verdict is separate.
    except Exception as exc:
        summary['error']=repr(exc)
        raise
    finally:
        summary['algorithm_seconds']=time.monotonic()-start
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
        verdict=summary.get('comparison_verdict','HARNESS_INCOMPLETE')
        (args.output/'result.md').write_text(f'''# Matched author forward-consumer result

Comparison verdict: {verdict}. Diagnostic execution: {summary['status']}.
See summary.json for every stage and saved tensors for actual consumer outputs.

| Decision | Primary criterion | Veto | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Fixed-policy forward comparison | {verdict} | Additional invalid cases and failed-policy acceptance recorded separately | Octave covariance perturbation and resampling sensitivity | Localize the first consumer discrepancy, or proceed to fitting parity if this screen passes | Full replication or P1 qualification |

| Inference status | Assessment |
|---|---|
| Hard veto screen | Per-stage validity and source-failure rejection in summary.json |
| Statistically supported ranking | None; deterministic common-draw comparison |
| Descriptive differences | All numerical errors and timings |
| Default readiness | Not established |
| Next evidence needed | Fitting parity and separately calibrated repeated-run no-worse evidence |

The strongest alternative explanation for a pathwise mismatch is the known
runtime sampling difference amplified through resampling. Inspect its first
occurrence before treating a later discrepancy as an independent cause.
Passing this finite set does not establish complete replication. Author
numerical failures remain accepted reference outcomes; no author repair ran.
''')
    print(json.dumps({k:summary[k] for k in ('status','comparison_verdict','forward_screen_passed','algorithm_seconds')}))


if __name__=='__main__':main()
