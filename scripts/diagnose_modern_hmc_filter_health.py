"""Actual-consumer diagnostic composition, never a production admission route.

Uses the existing MacroFinance chart and analytic QR filter. Explicit CPU/GPU
checks are diagnostic only. No external consumer file is modified.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

BASE = Path(__file__).resolve().parents[1]
CONSUMER = Path(os.environ.get('BAYESFILTER_DIAGNOSTIC_CONSUMER_ROOT', '/home/ubuntu/python/MacroFinance')).resolve()
PLAN = 'docs/plans/bayesfilter-modern-hmc-filter-health-continuation-2026-10-24.md'
GPU_PLAN = 'docs/plans/bayesfilter-modern-hmc-shared-gpu-continuation-2026-10-24.md'


def write(path, payload):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False))
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--compile-chain', action='store_true')
    parser.add_argument('--device', choices=('CPU','GPU'), default='CPU')
    args = parser.parse_args()
    plan = GPU_PLAN if args.device == 'GPU' else PLAN
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    report = dict(status='running', checks={}, plan=plan, cpu_diagnostic=args.device=='CPU',
                  posterior_qualified=False, gpu_qualified=False, consumer_installed=False)
    paths = list((BASE/'bayesfilter').rglob('*.py'))
    paths += [CONSUMER/n for n in ('one_country_affine_macro_phase9_hmc.py',
        'one_country_affine_macro_governance.py','one_country_affine_macro_transition_2x2.py',
        'one_country_affine_macro_prior_measurement_contract.py')]
    paths += [args.inputs.resolve(), Path(__file__).resolve()]
    hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    manifest = dict(command=[sys.executable,*sys.argv], source_sha256=hashes,
        runtime_authority=str(BASE), git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=BASE,text=True).strip(),
        started_utc=datetime.now(timezone.utc).isoformat(), plan=plan, device=args.device,
        environment={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','TF_FORCE_GPU_ALLOW_GROWTH',
            'TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS','OMP_NUM_THREADS')},
        seeds=[[20261024,701],[20261024,702]], compile_requested=args.compile,
        data='saved P18 nine-coordinate 48-period synthetic fixture')
    write(args.output/'manifest.json',manifest)
    def mark(stage):
        write(args.output/'progress.json',dict(stage=stage,elapsed_seconds=time.monotonic()-start))
    try:
        if os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH') != 'true':
            raise ValueError('memory-growth setting required before import')
        if args.device == 'CPU' and os.environ.get('CUDA_VISIBLE_DEVICES') != '-1':
            raise ValueError('CPU hiding required before import')
        sys.path.insert(0,str(CONSUMER));sys.path.insert(0,str(BASE))
        if args.device == 'GPU':
            from bayesfilter.runtime.gpu_provenance import query_nvidia_smi_gpus, selected_nvidia_gpu
            manifest['physical_gpu'] = selected_nvidia_gpu(query_nvidia_smi_gpus())
            manifest['gpu_trust_basis'] = 'trusted_escalated_gpu_execution'
            write(args.output/'manifest.json',manifest)
        import tensorflow as tf
        from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
        memory = configure_tensorflow_gpu_memory_growth(tf,require_gpu=args.device=='GPU')
        tf.config.experimental.enable_tensor_float_32_execution(False)
        import tensorflow_probability as tfp
        manifest.update(tensorflow=tf.__version__, tensorflow_probability=tfp.__version__,
            tf32_enabled=tf.config.experimental.tensor_float_32_execution_enabled(),
            dtype='float64', memory_policy=memory, compile_chain_requested=args.compile_chain)
        write(args.output/'manifest.json',manifest)
        import one_country_affine_macro_phase9_hmc as consumer
        from bayesfilter.linear.kalman_tf import tf_kalman_log_likelihood
        from bayesfilter.linear.kalman_qr_derivatives_tf import tf_qr_sqrt_kalman_score_factors_batched_static as kernel
        from bayesfilter.ssm import build_ssm_posterior_adapter
        inputs = json.loads(args.inputs.read_text())
        original = consumer._build_generic_target_with_runtime(inputs['observations'],
            dataset_id=inputs['design']['dataset_id'],
            runtime=(tf,tf_kalman_log_likelihood,kernel,memory),
            diagnostic_admission={'role':'source_bound_reference_diagnostic_only','plan':plan})
        observations=tf.constant(inputs['observations'],tf.float64)
        accepted=set(inspect.signature(kernel.python_function).parameters) - {'observations','jitter','jitter_updates_filtered_covariance','return_health'}
        def measured(theta):
            payload=consumer._tf_chart_batch_with_derivatives(theta,tf,include_support_checks=True)
            value,score,health=kernel(observations=observations,jitter=tf.constant(0.,tf.float64),
                                     return_health=True,**{k:v for k,v in payload.items() if k in accepted})
            health=dict(health)
            support=payload['support_valid']
            health['valid_pre_regularized_score'] &= support
            health['status_code']=tf.where((health['status_code']==0)&~support,4,health['status_code'])
            health['chart_support_valid']=support
            health['chart_lyapunov_relative_residual']=payload['lyapunov_relative_residual']
            return value,score,health
        status_graph=tf.function(lambda q:measured(q)[2],
            input_signature=[tf.TensorSpec([4,9],tf.float64)],autograph=False,jit_compile=False)
        diagnostic=build_ssm_posterior_adapter(contract=original.contract,
            prior_log_prob_and_grad=original.prior_log_prob_and_grad,
            filter_log_likelihood_and_grad=original.filter_log_likelihood_and_grad,
            target_status_telemetry=status_graph,
            target_status_manifest=dict(schema='macrofinance-measured-filter-health-diagnostic-v1',
                consumer_source_sha256=hashes[str(CONSUMER/'one_country_affine_macro_phase9_hmc.py')],
                filter_source_sha256=hashes[str(BASE/'bayesfilter/linear/kalman_qr_derivatives_tf.py')],
                scope='diagnostic composition, not installed consumer or posterior evidence'),
            dtype=tf.float64,target_scope=original.target_scope,evidence_path=plan)
        center=tf.constant(inputs['initial_position'],tf.float64)
        factor=tf.linalg.cholesky(tf.constant(inputs['initial_covariance'],tf.float64))
        offsets=tf.repeat(tf.constant([[-.15],[-.05],[.05],[.15]],tf.float64),9,axis=1)
        points=center+offsets@tf.transpose(factor)
        mark('value_score_health_parity')
        a=original.log_prob_and_grad(points);b=diagnostic.log_prob_and_grad(points)
        for x,y in zip(a,b):tf.debugging.assert_equal(x,y)
        measured_value,measured_score,health=measured(points)
        prior=original.prior_log_prob_and_grad(points)
        tf.debugging.assert_near(a[0],measured_value+prior[0],atol=1e-10,rtol=1e-10)
        tf.debugging.assert_near(a[1],measured_score+prior[1],atol=1e-10,rtol=1e-10)
        tf.debugging.assert_equal(health['status_code'],tf.zeros([4],tf.int32))
        report['checks']['same_value_score']=True
        report['health']={k:v.numpy().tolist() for k,v in health.items()}
        report['memory_policy']=memory
        report['consumer_capability']=original.value_score_capability().__dict__
        from bayesfilter.inference.hmc_modern_kernels import ReusableModernHMCRunner
        from bayesfilter.inference.hmc_transition_spec import HMCTransitionProfile
        mark('status_traced_randomized_transitions')
        # Tiny fixed mechanics run; no covariance adaptation or posterior claim.
        runner=ReusableModernHMCRunner(diagnostic,points,profile=HMCTransitionProfile('hmc','uniform_duration'),
            num_results=2,use_xla=False,target_status_trace_policy='per_chain_step')
        result=runner.run(current_state=points,seed=(20261024,701),step_size=.001,num_leapfrog_steps=3)
        from bayesfilter.inference.hmc_verification import target_status_telemetry_has_failure
        for key in ('target_status_telemetry','proposed_target_status_telemetry'):
            if target_status_telemetry_has_failure(result.trace[key],expected_shape=(2,4)):
                raise ValueError('measured transition status failure: '+key)
        report['checks']['accepted_and_proposed_status']=True
        report['transition_metadata']=result.metadata
        write(args.output/'transitions.json',dict(samples=result.samples.numpy().tolist(),
            trace=tf.nest.map_structure(lambda x:tf.convert_to_tensor(x).numpy().tolist(),result.trace)))
        if args.compile:
            mark('compile_exact_value_score_and_health')
            combined=tf.function(lambda q:(diagnostic.log_prob_and_grad(q),diagnostic.target_status_telemetry(q)),
                input_signature=[tf.TensorSpec([4,9],tf.float64)],jit_compile=True,autograph=False)
            with tf.device('/'+args.device+':0'):
                compiled=combined(tf.identity(points))
            for x,y in zip(a,compiled[0]):tf.debugging.assert_near(x,y,atol=1e-9,rtol=1e-9)
            tf.debugging.assert_equal(compiled[1]['status_code'],tf.zeros([4],tf.int32))
            report['checks'][args.device.lower()+'_xla_complete_target']=True
            if args.device == 'GPU':
                if 'device:GPU:' not in compiled[0][0].device:
                    raise ValueError('complete target did not execute on GPU')
                with tf.device('/CPU:0'):
                    cpu_combined=tf.function(lambda q:(diagnostic.log_prob_and_grad(q),
                        diagnostic.target_status_telemetry(q)), input_signature=[tf.TensorSpec([4,9],tf.float64)],
                        jit_compile=True,autograph=False)
                    cpu_target=cpu_combined(tf.identity(points))
                for x,y in zip(cpu_target[0],compiled[0]):
                    tf.debugging.assert_near(x,y,atol=1e-9,rtol=1e-9)
                report['checks']['cpu_gpu_target_parity']=True
        if args.compile_chain:
            mark('compile_shared_chain_body_diagnostic')
            # Diagnostic of the exact shared numerical body, not a public
            # admitted runner: keep both consumer capability flags unchanged.
            signature=[tf.TensorSpec([4,9],tf.float64),tf.TensorSpec([2],tf.int32),
                tf.TensorSpec([],tf.float64),tf.TensorSpec([],tf.int32),tf.TensorSpec([],tf.int32)]
            compiled_chain=tf.function(runner._chain,input_signature=signature,
                jit_compile=True,autograph=False)
            call=(points,tf.constant([20261024,702],tf.int32),tf.constant(.001,tf.float64),
                  tf.constant(3,tf.int32),tf.constant(0,tf.int32))
            with tf.device('/'+args.device+':0'):
                device_call=tuple(tf.identity(x) for x in call)
                samples,trace=compiled_chain(*device_call)
                replay=compiled_chain(*device_call)
            for x,y in zip(tf.nest.flatten((samples,trace)),tf.nest.flatten(replay)):
                tf.debugging.assert_equal(x,y)
            tf.debugging.assert_all_finite(samples,'compiled diagnostic states')
            for key in ('target_status_telemetry','proposed_target_status_telemetry'):
                if target_status_telemetry_has_failure(trace[key],expected_shape=(2,4)):
                    raise ValueError('compiled diagnostic status failure')
            if original.value_score_capability().xla_hmc_ready:
                raise ValueError('diagnostic must not change consumer admission')
            report['checks'][args.device.lower()+'_xla_shared_chain_body']=True
            report['checks'][args.device.lower()+'_xla_deterministic_replay']=True
            if args.device == 'GPU':
                if 'device:GPU:' not in samples.device:
                    raise ValueError('shared chain did not execute on GPU')
                with tf.device('/CPU:0'):
                    cpu_chain=tf.function(runner._chain,input_signature=signature,
                        jit_compile=True,autograph=False)
                    cpu_samples,cpu_trace=cpu_chain(*(tf.identity(x) for x in call))
                tf.debugging.assert_near(cpu_samples,samples,atol=1e-9,rtol=1e-9)
                for key in ('target_log_prob','log_acceptance_correction','log_accept_ratio','initial_momentum'):
                    tf.debugging.assert_near(cpu_trace[key],trace[key],atol=1e-9,rtol=1e-9)
                for key in ('trajectory_steps','is_accepted'):
                    tf.debugging.assert_equal(cpu_trace[key],trace[key])
                for key in ('target_status_telemetry','proposed_target_status_telemetry'):
                    if target_status_telemetry_has_failure(cpu_trace[key],expected_shape=(2,4)):
                        raise ValueError('CPU comparator status failure')
                report['checks']['cpu_gpu_shared_chain_parity']=True
                report['chain_output_device']=samples.device
                report['gpu_allocator_bytes']=tf.config.experimental.get_memory_info('GPU:0')
                report['gpu_numerical_diagnostic_passed']=True
            report['compiled_chain_scope']='internal numerical diagnostic, not public execution admission'
            report['compiled_chain_row_evaluations']=int(tf.reduce_sum(trace['value_score_calls_per_chain']))*2
            write(args.output/'compiled-transitions.json',dict(samples=samples.numpy().tolist(),
                trace=tf.nest.map_structure(lambda x:tf.convert_to_tensor(x).numpy().tolist(),trace)))
        report['source_unchanged']=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
        if not report['source_unchanged']:raise ValueError('source drift during diagnostic')
        report['status']='completed'
    except Exception as exc:
        report.update(status='error',error=repr(exc),traceback=traceback.format_exc())
    report.update(elapsed_seconds=time.monotonic()-start,finished_utc=datetime.now(timezone.utc).isoformat())
    write(args.output/'result.json',report)
    print(json.dumps({k:report[k] for k in ('status','checks','elapsed_seconds','error') if k in report}))
    return 0 if report['status']=='completed' else 1


if __name__=='__main__':
    raise SystemExit(main())
